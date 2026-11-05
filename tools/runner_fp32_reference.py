"""fp32 parity references: the trained model in plain PyTorch fp32 (no autocast), for the records a parity run scored.

The parity gates of docs/RUNNER.md section 14 are measured against the Unsloth bf16 rows (`eval/rows-<panel>.json`). That reference carries its
own bf16 error (on LONG-LOG about 0.05 in p at the worst questions, see docs/PARITY-LONGLOG.md), so a runner can be judged better against an fp32
reference. Three steps, in three environments:

  1. export   (Sepal checkout, PYTHONPATH = Sepal;Sepal-Runner)  records of a seeded sample -> JSON with token ids, option ids and the bf16 rows.
         python tools/runner_fp32_reference.py export --run runs/box-results/full15-4b/F15-s0 --panel LONG-LOG --n 20 --out sample.json
  2. score    (any box with torch + transformers + peft; CPU is enough; `--clef` is Unsloth's vendored joint_schema_model.py)
         python tools/runner_fp32_reference.py score --sample sample.json --final FINAL --base BASE --clef joint_schema_model.py \\
                --only 4,5,10 --out fp32.json
  3. compare  (anywhere)  runner logits (sepal-score output, same order as the `--input` workload, ids = sample indices) against the fp32 rows and the bf16 rows.
         python tools/runner_fp32_reference.py compare --sample sample.json --fp32 fp32.json --runner run_cpu-f32.jsonl [--runner2 run_cpu-bf16.jsonl]

Only our own panels (TOOL, LONG-LOG) may be exported to a rented box. Everything is reported as max |dp| at temperature 1, max |dlogit| and flips.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import sys
from pathlib import Path


def softmax(z):
    top = max(z)
    e = [math.exp(v - top) for v in z]
    s = sum(e)
    return [x / s for x in e]


def export(a) -> int:
    from sepal.train import readout, roundspec
    from sepal.train import unsloth_convert as conv
    from sepal_runner.encode import Encoder

    if a.panel.startswith("SEC"):
        raise SystemExit("SOCDecide panels stay on this PC")
    run = Path(a.run)
    max_len = int(json.loads((run / "final" / "sepal.json").read_text(encoding="utf-8"))["max_len_score"])
    spec = roundspec.load(a.spec, ".")
    records = readout.panel_records(spec, a.panel)
    oracle = {r["qid"]: r for r in json.loads((run / "eval" / f"rows-{a.panel}.json").read_text(encoding="utf-8"))}
    records = [r for r in records if all(f"{r['_meta']['id']}::{q}" in oracle for q in r["questions"])]
    sample = random.Random(a.seed).sample(records, min(a.n, len(records)))
    enc = Encoder.from_file(run / "final" / "tokenizer.json")
    out = []
    for r in sample:
        questions, state = conv.request_questions(r), conv.state_text(r["state"])
        e = enc.encode({"state": state, "questions": questions}, max_len)
        ordered = []
        for q, eq in zip(r["questions"], e.questions):
            kind = r["questions"][q]["type"]
            keys = conv.to_our_keys(kind, {o: float(i) for i, o in enumerate(eq.option_ids)})      # our key -> option index
            logits = oracle[f"{r['_meta']['id']}::{q}"]["logits"]
            ordered.append([logits[k] for k, _ in sorted(keys.items(), key=lambda kv: kv[1])])
        out.append({"id": r["_meta"]["id"], "state": state, "questions": questions, "max_len": max_len, "input_ids": list(e.input_ids),
                    "qspans": [list(q.question_span) for q in e.questions], "ospans": [[list(s) for s in q.option_spans] for q in e.questions],
                    "oids": [list(q.option_ids) for q in e.questions], "bf16_rows": ordered})
    Path(a.out).write_text(json.dumps(out), encoding="utf-8")
    print(f"{len(out)} records, tokens {[len(o['input_ids']) for o in out]}")
    return 0


def score(a) -> int:
    from sepal_runner import oracle

    oracle.hide_gpu()
    spec = importlib.util.spec_from_file_location("clef_vendor", a.clef)
    module = importlib.util.module_from_spec(spec)
    sys.modules["clef_vendor"] = module
    spec.loader.exec_module(module)
    import torch

    torch.set_num_threads(a.threads)
    sample = json.loads(Path(a.sample).read_text(encoding="utf-8"))
    only = [int(x) for x in a.only.split(",")] if a.only else range(len(sample))
    o = oracle.Fp32Oracle(final_dir=Path(a.final), base_dir=Path(a.base), device="cpu")
    result = {}
    for i in only:
        r = sample[i]
        enc = o.encode(r["state"], r["questions"], r["max_len"])
        same = list(enc.input_ids) == r["input_ids"]
        spans = all(list(q.question_span) == s and [list(x) for x in q.option_spans] == t for q, s, t in zip(enc.questions, r["qspans"], r["ospans"]))
        flat = o.forward(enc)["logits"].tolist()
        rows, k = [], 0
        for q in enc.questions:
            rows.append(flat[k:k + len(q.option_ids)])
            k += len(q.option_ids)
        result[str(i)] = {"id": r["id"], "tokens": len(enc.input_ids), "ids_same_as_runner_encoder": same, "spans_same": spans, "logits": rows}
        print(i, r["id"], len(enc.input_ids), "ids_same", same, "spans_same", spans, flush=True)
        Path(a.out).write_text(json.dumps(result), encoding="utf-8")
    return 0


def distance(x, y):
    dp = dz = 0.0
    flips = n = 0
    for u, v in zip(x, y):
        pu, pv = softmax(u), softmax(v)
        dp = max(dp, max(abs(p - q) for p, q in zip(pu, pv)))
        dz = max(dz, max(abs(p - q) for p, q in zip(u, v)))
        flips += pu.index(max(pu)) != pv.index(max(pv))
        n += 1
    return dp, dz, flips, n


def compare(a) -> int:
    sample = json.loads(Path(a.sample).read_text(encoding="utf-8"))
    fp32 = json.loads(Path(a.fp32).read_text(encoding="utf-8"))
    runs = {}
    for label, path in (("runner", a.runner), ("runner2", a.runner2)):
        if path:
            runs[label] = {int(json.loads(line)["id"]): json.loads(line)["logits"] for line in open(path, encoding="utf-8")}
    header = ["record", "tokens", "bf16~fp32"] + [f"{k}~fp32" for k in runs] + [f"{k}~bf16" for k in runs]
    print(" | ".join(header), "   (max dp / max dlogit / flips)")
    totals: dict[str, list[float]] = {}
    for key, row in sorted(fp32.items(), key=lambda kv: int(kv[0])):
        i = int(key)
        cols = [("bf16~fp32", distance(sample[i]["bf16_rows"], row["logits"]))]
        cols += [(f"{k}~fp32", distance(v[i], row["logits"])) for k, v in runs.items() if i in v]
        cols += [(f"{k}~bf16", distance(v[i], sample[i]["bf16_rows"])) for k, v in runs.items() if i in v]
        for name, d in cols:
            t = totals.setdefault(name, [0.0, 0.0, 0, 0])
            t[0], t[1], t[2], t[3] = max(t[0], d[0]), max(t[1], d[1]), t[2] + d[2], t[3] + d[3]
        print(f"{row['id']} | {row['tokens']} | " + " | ".join("%.3g/%.3g/%d" % d[:3] for _, d in cols))
    print("overall:", {k: {"max_dp": round(v[0], 4), "max_dz": round(v[1], 3), "flips": f"{v[2]}/{v[3]}"} for k, v in totals.items()})
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--run", required=True)
    e.add_argument("--panel", required=True)
    e.add_argument("--spec", default="experiments/rounds/full15-4b.yaml")
    e.add_argument("--n", type=int, default=20)
    e.add_argument("--seed", default="runner-rescore")
    e.add_argument("--out", required=True)
    s = sub.add_parser("score")
    s.add_argument("--sample", required=True)
    s.add_argument("--final", required=True)
    s.add_argument("--base", required=True)
    s.add_argument("--clef", required=True)
    s.add_argument("--only")
    s.add_argument("--threads", type=int, default=int(os.environ.get("NT", "8")))
    s.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--sample", required=True)
    c.add_argument("--fp32", required=True)
    c.add_argument("--runner", required=True)
    c.add_argument("--runner2")
    a = p.parse_args(argv)
    return {"export": export, "score": score, "compare": compare}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
