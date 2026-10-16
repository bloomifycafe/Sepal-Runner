"""R0 baseline of the oracle (the model as sepal.serve.decision_api serves it) and of llama.cpp prefill, on the shared workloads.

    python -m sepal.runner.r0_bench gpu    --shapes t1k_q1 ... [--batch-tokens 16384 32768 65536] [--passes 2]    (hold the GPU lock, GPU otherwise idle)
    python -m sepal.runner.r0_bench cpu    --shapes t1k_q1 ... [--threads 12] [--budget-seconds 300]
    python -m sepal.runner.r0_bench llama  --device gpu|cpu [--threads 12] [--prompts 1024 4096 16384]
    python -m sepal.runner.r0_bench table                                                              (markdown table from the JSON files)

Run in the Unsloth environment (runs/pilot-unsloth/venv) with PYTHONPATH set to the repo root. Results: runs/runner-bench/r0/*.json.
`gpu` times (a) batch 1: `decision_api.answer` per request, one at a time, exactly as the server answers (encode + forward + softmax),
and (b) saturated: `UnslothScorer.answers` with its padded length-sorted batches at each `--batch-tokens` (tokenising threads running
ahead), the whole workload at once. Every number is wall time of this machine (RTX 3060, Ryzen 5 7600X).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

RUNS = Path(os.environ.get("SEPAL_RUNS", "E:/MYCOMPANY/BlumSOC/Sepal/runs"))
WORKLOADS = RUNS / "runner-bench" / "workloads"
OUT = RUNS / "runner-bench" / "r0"
LLAMA = Path("E:/MYCOMPANY/BlumSOC/bench/llama.cpp-b11514/bin/llama-bench.exe")
GGUF = Path("E:/MYCOMPANY/BlumSOC/bench/models/qwen35-0.8b-base-f16.gguf")
GRID = [f"t{t}_q{q}" for t in ("1k", "4k", "16k") for q in (1, 8, 32)]
ALL_SHAPES = GRID + ["shared_t4k_q8"]


def load_requests(shape: str) -> list[dict]:
    with open(WORKLOADS / f"{shape}.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def percentile(values: list[float], p: float) -> float:
    xs = sorted(values)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * p / 100
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def environment(extra: dict | None = None) -> dict:
    import importlib.metadata as md

    env = {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "machine": "AMD Ryzen 5 7600X (6C/12T, Zen 4), RTX 3060 12 GB, Windows 11", "python": platform.python_version()}
    for name in ("torch", "transformers", "unsloth", "peft", "triton", "triton-windows"):
        try:
            env[name] = md.version(name)
        except md.PackageNotFoundError:
            pass
    try:
        env["nvidia_smi"] = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                                           capture_output=True, text=True, timeout=20).stdout.strip()
        env["compute_processes_at_start"] = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
                                                           capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as error:      # noqa: BLE001 - informational only
        env["nvidia_smi"] = f"unavailable: {error}"
    env.update(extra or {})
    return env


def save(name: str, payload: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / (name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, OUT / name)


def summarise(latencies: list[float], tokens: list[int]) -> dict:
    total = sum(latencies)
    return {"requests": len(latencies), "total_seconds": round(total, 3), "requests_per_s": round(len(latencies) / total, 4),
            "tokens_per_s": round(sum(tokens) / total, 1), "latency_p50_s": round(percentile(latencies, 50), 4),
            "latency_p95_s": round(percentile(latencies, 95), 4), "latency_max_s": round(max(latencies), 4), "tokens_total": sum(tokens)}


# ---- GPU -------------------------------------------------------------------------------------------------------------------

def run_gpu(args) -> None:
    import torch

    from sepal.runner import oracle
    from sepal.serve.decision_api import answer, to_record

    t0 = time.perf_counter()
    served = oracle.ServedOracle(batch_tokens=max(args.batch_tokens))
    load_seconds = time.perf_counter() - t0
    scorer, api_raw = served.scorer, served.api_raw
    env = environment({"model_load_seconds": round(load_seconds, 2), "model_load_note": "load_scorer(calibrated=True) wall time after `import torch`, Unsloth already imported by sepal.train.unsloth_arm lazily inside it"})
    for shape in args.shapes:
        requests = load_requests(shape)
        result = {"shape": shape, "oracle": "as served: load_scorer(calibrated=True), bf16 autocast", "environment": env, "batch1": {}, "saturated": {}}
        for r in requests[:2]:                                   # warm-up: kernels, allocator, tokenizer
            answer(scorer, r["body"], "sepal")
        passes = []
        for p in range(args.passes):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            lat, toks = [], []
            for r in requests:
                a = time.perf_counter()
                out = answer(scorer, r["body"], "sepal")
                torch.cuda.synchronize()
                lat.append(time.perf_counter() - a)
                toks.append(out["usage"]["input_tokens"])
            passes.append({**summarise(lat, toks), "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
                           "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3)})
        best = max(passes, key=lambda x: x["requests_per_s"])
        result["batch1"] = {"best_pass": best, "passes": passes}
        records = [{**to_record(r["body"]), "_meta": {"id": r["id"]}} for r in requests]
        scorer.api = api_raw                                     # the padded-batch path (NoTruncateApi has no logits_many)
        for bt in args.batch_tokens:
            scorer.batch_tokens = bt
            passes = []
            for p in range(args.passes):
                api_raw.oom_cap = None
                batch_times: list[float] = []
                original = api_raw.logits_many

                def timed(encoded, _orig=original, _times=batch_times):
                    a = time.perf_counter()
                    out = _orig(encoded)
                    _times.append(time.perf_counter() - a)
                    return out

                api_raw.logits_many = timed
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                a = time.perf_counter()
                try:
                    _, _, tokens = scorer.answers(records)
                    torch.cuda.synchronize()
                    wall = time.perf_counter() - a
                    passes.append({"requests": len(records), "total_seconds": round(wall, 3), "requests_per_s": round(len(records) / wall, 4),
                                   "tokens_per_s": round(tokens / wall, 1), "tokens_total": tokens, "batches": len(batch_times),
                                   "batch_seconds_p50": round(percentile(batch_times, 50), 4), "batch_seconds_p95": round(percentile(batch_times, 95), 4),
                                   "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
                                   "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3), "oom_cap_tokens": api_raw.oom_cap})
                except torch.cuda.OutOfMemoryError as error:
                    torch.cuda.empty_cache()
                    passes.append({"error": f"out of memory: {str(error)[:120]}"})
                finally:
                    api_raw.logits_many = original
            ok = [x for x in passes if "error" not in x]
            result["saturated"][str(bt)] = {"best_pass": max(ok, key=lambda x: x["requests_per_s"]) if ok else None, "passes": passes}
            print(shape, "saturated", bt, result["saturated"][str(bt)]["best_pass"], flush=True)
        scorer.api = __import__("sepal.serve.decision_api", fromlist=["NoTruncateApi"]).NoTruncateApi(api_raw, served.max_len)
        save(f"oracle_gpu_{shape}.json", result)
        print(shape, "batch1", result["batch1"]["best_pass"], flush=True)


# ---- CPU -------------------------------------------------------------------------------------------------------------------

def run_cpu(args) -> None:
    from sepal.runner import oracle

    oracle.hide_gpu()
    import torch

    torch.set_num_threads(args.threads)
    t0 = time.perf_counter()
    o = oracle.Fp32Oracle(device="cpu")
    load_seconds = time.perf_counter() - t0
    env = environment({"model_load_seconds": round(load_seconds, 2), "torch_threads": torch.get_num_threads(),
                       "note": "Fp32Oracle: fp32 weights, PEFT LoRA unmerged, no autocast, reference PyTorch gated delta rule (no fla/Triton on CPU)"})
    from sepal.serve.decision_api import to_record
    from sepal.train import unsloth_convert as conv

    warm = load_requests("t1k_q1")[0]
    record = to_record(warm["body"])
    o.forward(o.encode(conv.state_text(record["state"]), conv.request_questions(record)))
    for shape in args.shapes:
        requests = load_requests(shape)
        lat, toks, enc_seconds = [], [], []
        started = time.perf_counter()
        for r in requests:
            record = to_record(r["body"])
            a = time.perf_counter()
            enc = o.encode(conv.state_text(record["state"]), conv.request_questions(record))
            enc_seconds.append(time.perf_counter() - a)
            a = time.perf_counter()
            o.forward(enc)
            lat.append(time.perf_counter() - a)
            toks.append(len(enc.input_ids))
            print(shape, len(lat), f"{lat[-1]:.2f} s", flush=True)
            if time.perf_counter() - started > args.budget_seconds:
                break
        res = {"shape": shape, "oracle": "fp32 CPU", "environment": env, "requests_of_workload": len(requests), **summarise(lat, toks),
               "encode_seconds_mean": round(sum(enc_seconds) / len(enc_seconds), 4), "stopped_by_budget": len(lat) < len(requests),
               "budget_seconds": args.budget_seconds}
        save(f"oracle_cpu_{shape}.json", res)
        print(shape, res, flush=True)


# ---- llama.cpp -------------------------------------------------------------------------------------------------------------

def run_llama(args) -> None:
    cmd = [str(LLAMA), "-m", str(GGUF), "-p", ",".join(str(p) for p in args.prompts), "-n", "0", "-r", str(args.reps), "-o", "json",
           "-ngl", "99" if args.device == "gpu" else "0", "-t", str(args.threads)]
    started = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    rows = json.loads(proc.stdout) if proc.stdout.strip().startswith("[") else []
    out = {"device": args.device, "command": cmd, "environment": environment({"llama_cpp": "b11514 CUDA 12.4 Windows build", "gguf": GGUF.name,
           "note": "trunk only (Qwen3.5-0.8B-Base f16 GGUF): no LoRA, no Clef head, no per-position head read; prefill pp tokens, tg 0"}),
           "wall_seconds": round(time.time() - started, 1), "returncode": proc.returncode, "stderr_tail": proc.stderr[-400:],
           "rows": [{"n_prompt": r.get("n_prompt"), "n_gen": r.get("n_gen"), "best_tokens_per_s": max(r.get("samples_ts", [r.get("avg_ts", 0)])),
                     "avg_tokens_per_s": r.get("avg_ts"), "stddev_ts": r.get("stddev_ts"), "samples_ts": r.get("samples_ts"),
                     "n_gpu_layers": r.get("n_gpu_layers"), "n_threads": r.get("n_threads"), "flash_attn": r.get("flash_attn"), "backends": r.get("backends")} for r in rows]}
    save(f"llama_{args.device}.json", out)
    print(json.dumps(out["rows"], indent=1))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    g = sub.add_parser("gpu")
    g.add_argument("--shapes", nargs="+", default=ALL_SHAPES)
    g.add_argument("--batch-tokens", nargs="+", type=int, default=[16384, 32768, 65536])
    g.add_argument("--passes", type=int, default=2)
    c = sub.add_parser("cpu")
    c.add_argument("--shapes", nargs="+", default=["t1k_q1", "t1k_q8", "t1k_q32"])
    c.add_argument("--threads", type=int, default=12)
    c.add_argument("--budget-seconds", type=float, default=300.0)
    l = sub.add_parser("llama")
    l.add_argument("--device", choices=["gpu", "cpu"], required=True)
    l.add_argument("--threads", type=int, default=12)
    l.add_argument("--reps", type=int, default=3)
    l.add_argument("--prompts", nargs="+", type=int, default=[1024, 4096, 16384])
    args = parser.parse_args(argv)
    {"gpu": run_gpu, "cpu": run_cpu, "llama": run_llama}[args.command](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
