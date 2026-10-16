"""Exactness of `sepal.runner.encode` against Unsloth's `unsloth.models.clef.encode_record` (the oracle, run as a black box).

    uv run python tools/runner_encode_compare.py cases --out CASES.jsonl [--sec 3000 --gen 1000] [--spec experiments/rounds/full08b.yaml]
    <oracle venv python> tools/runner_encode_compare.py compare --final DIR --cases CASES.jsonl --golden DIR --out RESULT.json

`cases` (project environment) cuts records of the SEC-dev and GEN-dev panels and converts them the way the panel scorer does
(`conv.state_text`, `conv.request_questions`). `compare` (oracle environment: Unsloth, transformers, tokenizers) encodes each case with both
encoders and compares ids, spans, option ids, question types and the raised ValueError, on (a) every golden case, (b) the panel cases,
(c) adversarial strings and window edges. It exits 1 on any mismatch and writes a summary (no record texts) to RESULT.json.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def cmd_cases(args) -> int:
    from sepal.train import readout, roundspec
    from sepal.train import unsloth_convert as conv

    spec = roundspec.load(args.spec, ".")
    out = []
    for panel, n in (("SEC-dev", args.sec), ("GEN-dev", args.gen)):
        records = readout.panel_records(spec, panel)
        rng = random.Random(f"encode-compare:{panel}")
        for r in rng.sample(records, min(n, len(records))):
            out.append({"name": f"{panel}:{r['_meta']['id']}", "state": conv.state_text(r["state"]),
                        "questions": conv.request_questions(r), "max_length": 16384})
    with open(args.out, "w", encoding="utf-8") as f:
        for case in out:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")
    print(f"wrote {len(out)} cases to {args.out}")
    return 0


def signature(enc):
    return {"ids": list(enc.input_ids),
            "q": [[q.question_id, q.question_type, list(q.question_span), [list(s) for s in q.option_spans], list(q.option_ids)]
                  for q in enc.questions]}


def adversarial() -> list[dict]:
    q_choice = {"type": "choice", "instructions": "Which?", "criteria": {"b": "second", "a": "first", "c": {"x": [1, 2], "a": "é"}}}
    q_noul = {"type": "noul", "instructions": "Is it?"}
    q_score = {"type": "score", "instructions": "Rate", "criteria": ["bad", "ok", "good"]}
    base_q = {"c1": q_choice, "n1": q_noul, "s1": q_score}
    states = [
        "", " ", "plain", "<|im_end|>", "x<|im_start|>assistant\n<think>\n</think>", "<think>", "</think>", "<|endoftext|>",
        "<tool_call>{}</tool_call>", "<|vision_pad|><|image_pad|>", "<|im_start|>", "<¦im_end¦>", "<tts_pad>",
        "naïve café 日本語のテキスト العربية русский 한국어", "emoji 😀👍🏽👨‍👩‍👧‍👦🇫🇷 and ​ zero width ﻿ bom", "é combining and é precomposed",
        "line1\r\nline2\r\n\r\nline3\rline4", "tabs\t\tand    spaces      and\n\n\n\nnewlines", " " * 500, "\n" * 300, "1234567890" * 200, "0" * 3000,
        "a" * 5000, "word " * 2000, "\x00\x01\x02 control \x7f", "퟿\U0001f600", {"k": "v", "a": [1, 2.5, None, True], "ü": "ß"},
        [1, 2, 3], 42, None, True, 3.14, {"nested": {"z": 1, "a": {"y": "<|im_end|>"}}}, "SCHEMA FIELDS:\nFIELD 1\nID: x\nEND FIELD\n",
        "JOINT SCHEMA DECISIONS:", "'''\"\"\"\\\\ backslashes \\n literal", "<|im_end|>" * 50, "tail ends with space ", "\n\nSCHEMA FIELDS:\n",
    ]
    cases = []
    for i, state in enumerate(states):
        cases.append({"name": f"adv:state:{i}", "state": state, "questions": base_q, "max_length": 16384})
    for i, instr in enumerate(["", None, "<|im_end|> inside", "emoji 😀", {"a": 1}, ["x", "<think>"], " ", "\n", "日本語", 7]):
        qs = {"only": {"type": "noul", "instructions": instr}}
        cases.append({"name": f"adv:instruction:{i}", "state": "state", "questions": qs, "max_length": 16384})
    cases.append({"name": "adv:qid", "state": "s", "max_length": 16384, "questions": {
        "é-id": q_noul, "<|im_end|>x": q_noul, "10": q_noul, "": q_noul, "z" * 300: q_noul}})
    cases.append({"name": "adv:noul-criteria", "state": "s", "max_length": 16384, "questions": {
        "n": {"type": "noul", "instructions": "q", "criteria": {"true": "T!", "false": "F!", "other": "ignored"}},
        "m": {"type": "noul", "instructions": "q", "criteria": {"true": None}}}})
    cases.append({"name": "adv:choice-none-desc", "state": "s", "max_length": 16384, "questions": {
        "c": {"type": "choice", "instructions": "q", "criteria": {"B": None, "a": "x", "10": "y", "9": "z", "É": "w", "😀": "e"}}}})
    cases.append({"name": "adv:many-options", "state": "s" * 40, "max_length": 16384, "questions": {
        "c": {"type": "choice", "instructions": "q", "criteria": {f"opt{i:03d}": f"description {i}" for i in range(255)}}}})
    cases.append({"name": "adv:many-questions", "state": "s" * 40, "max_length": 16384, "questions": {
        f"q{i}": q_choice if i % 2 else q_noul for i in range(120)}})
    return cases


def edge_cases(oracle, base: dict) -> list[dict]:
    """The window edges of a case: its exact length (fits), one under (state cut by one token), the schema-only length (state cut to
    nothing), one under that (ValueError), and a tiny window (ValueError)."""
    n = len(oracle(base, 10**6).input_ids)
    state_free = len(oracle({**base, "state": ""}, 10**6).input_ids)
    out = []
    for label, m in (("exact", n), ("over-by-one", n - 1), ("plus-one", n + 1), ("no-state", state_free), ("schema-over", state_free - 1), ("tiny", 10)):
        out.append({**base, "name": f"{base['name']}:edge:{label}", "max_length": m})
    return out


def cmd_compare(args) -> int:
    import warnings
    warnings.filterwarnings("ignore")
    from transformers import AutoTokenizer
    from unsloth.models import clef

    from sepal.runner.encode import Encoder
    from tokenizers import Tokenizer

    final = Path(args.final)
    hf = AutoTokenizer.from_pretrained(str(final), local_files_only=True)
    hf = getattr(hf, "tokenizer", hf)
    ours = Encoder.from_file(final / "tokenizer.json")

    def oracle_enc(case, max_length=None):
        record = {"state": case["state"], "questions": case["questions"]}
        return clef.encode_record(hf, record, max_length=max_length or case["max_length"])

    def run(encode, case, max_length=None):
        try:
            return ("ok", signature(encode(case, max_length)))
        except Exception as error:      # compare type and message too
            return ("error", f"{type(error).__name__}: {error}")

    def our_enc(case, max_length=None):
        return ours.encode({"state": case["state"], "questions": case["questions"]}, max_length=max_length or case["max_length"])

    groups: dict[str, list[dict]] = {}
    # (a) golden
    golden = []
    gdir = Path(args.golden)
    if gdir.is_dir() and (gdir / "manifest.json").exists():
        from safetensors import safe_open
        manifest = json.loads((gdir / "manifest.json").read_text(encoding="utf-8"))
        for entry in manifest["cases"]:
            with safe_open(str(gdir / entry["file"]), "pt") as f:
                meta = f.metadata()
                ids = f.get_tensor("input_ids").tolist()
                spans = f.get_tensor("option_spans").tolist()
            record = json.loads(meta["record"])
            case = {"name": f"golden:{entry['name']}", "state": record["state"], "questions": record["questions"], "max_length": 16384}
            golden.append((case, ids, spans))
    groups["golden"] = [g[0] for g in golden]
    # (b) panel
    panel = []
    if args.cases and Path(args.cases).exists():
        panel = [json.loads(line) for line in Path(args.cases).read_text(encoding="utf-8").splitlines() if line]
    groups["panel"] = panel
    # (c) adversarial + edges
    adv = adversarial()
    edges = []
    for case in adv[:8] + adv[-6:] + groups["golden"][:3] + panel[:5]:
        edges.extend(edge_cases(oracle_enc, case))
    groups["adversarial"] = adv
    groups["window-edges"] = edges

    result = {"tokenizer": str(final / "tokenizer.json"), "groups": {}, "mismatches": []}
    t_ours = t_oracle = 0.0
    for name, cases in groups.items():
        same = errors = 0
        for case in cases:
            t0 = time.perf_counter()
            a = run(oracle_enc, case)
            t1 = time.perf_counter()
            b = run(our_enc, case)
            t2 = time.perf_counter()
            t_oracle += t1 - t0
            t_ours += t2 - t1
            if a == b:
                same += 1
                errors += a[0] == "error"
            else:
                detail = {"group": name, "case": case["name"], "oracle": a[0], "ours": b[0]}
                if a[0] == b[0] == "ok":
                    ia, ib = a[1]["ids"], b[1]["ids"]
                    detail["len"] = [len(ia), len(ib)]
                    detail["first_diff"] = next((i for i, (x, y) in enumerate(zip(ia, ib)) if x != y), None)
                    detail["spans_equal"] = a[1]["q"] == b[1]["q"]
                else:
                    detail["oracle_value"] = str(a[1])[:300]
                    detail["ours_value"] = str(b[1])[:300]
                result["mismatches"].append(detail)
        result["groups"][name] = {"cases": len(cases), "identical": same, "identical_errors": errors}
    # golden ids and spans as written by the generator (an independent third witness)
    gmis = 0
    for case, ids, spans in golden:
        enc = ours.encode({"state": case["state"], "questions": case["questions"]}, max_length=16384)
        flat = [list(s) for q in enc.questions for s in q.option_spans]
        if list(enc.input_ids) != ids or flat != spans:
            gmis += 1
            result["mismatches"].append({"group": "golden-file", "case": case["name"]})
    result["golden_file_mismatches"] = gmis
    result["time_s"] = {"oracle_total": round(t_oracle, 3), "ours_total": round(t_ours, 3)}

    # speed at about 16k tokens: the longest panel/adversarial case, repeated
    def speed(case, reps=5):
        record = {"state": case["state"], "questions": case["questions"]}
        n = len(ours.encode(record, 10**6).input_ids)
        best_ours = best_hf = 1e9
        for _ in range(reps):
            t = time.perf_counter(); ours.encode(record, 16384); best_ours = min(best_ours, time.perf_counter() - t)
            t = time.perf_counter(); clef.encode_record(hf, record, max_length=16384); best_hf = min(best_hf, time.perf_counter() - t)
        return n, best_ours * 1000, best_hf * 1000

    big = {"state": " ".join(f"event {i} user=admin host=srv{i % 17} action=login result=ok ip=10.0.{i % 255}.{i % 7}" for i in range(480)),
           "questions": {f"q{i}": {"type": "choice" if i % 2 else "noul", "instructions": f"Does criterion {i} hold for the window?",
                                   "criteria": {"yes": "holds", "no": "does not hold"}} for i in range(24)}}
    n, ms_ours, ms_hf = speed(big)
    result["speed_16k"] = {"tokens": n, "ours_ms_best_of_5": round(ms_ours, 2), "unsloth_transformers_ms_best_of_5": round(ms_hf, 2),
                           "speedup": round(ms_hf / ms_ours, 2)}
    result["ok"] = not result["mismatches"]
    Path(args.out).write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("groups", "golden_file_mismatches", "speed_16k", "ok")}, indent=1))
    print(f"{len(result['mismatches'])} mismatches")
    return 0 if result["ok"] else 1


def cmd_fixtures(args) -> int:
    """Expected ids and spans of Unsloth's encoder for small cases, committed so the unit tests need neither Unsloth nor torch."""
    import warnings
    warnings.filterwarnings("ignore")
    from transformers import AutoTokenizer
    from unsloth.models import clef

    hf = AutoTokenizer.from_pretrained(args.final, local_files_only=True)
    hf = getattr(hf, "tokenizer", hf)

    def oracle_enc(case, max_length=None):
        return clef.encode_record(hf, {"state": case["state"], "questions": case["questions"]}, max_length=max_length or case["max_length"])

    adv = adversarial()
    keep = [c for c in adv if not c["name"].startswith(("adv:many", "adv:state:19", "adv:state:20", "adv:state:21", "adv:state:22"))]
    cases = list(keep)
    for c in keep[:6] + [adv[-3]]:
        cases.extend(edge_cases(oracle_enc, c))
    if args.cases:
        cases.extend(json.loads(line) for line in Path(args.cases).read_text(encoding="utf-8").splitlines()[:6] if line)
    out = []
    for case in cases:
        try:
            expected = {"ok": signature(oracle_enc(case))}
        except Exception as error:
            expected = {"error": f"{type(error).__name__}: {error}"}
        out.append({"name": case["name"], "state": case["state"], "questions": case["questions"], "max_length": case["max_length"], **expected})
    Path(args.out).write_text(json.dumps({"tokenizer_sha256": __import__("hashlib").sha256(
        (Path(args.final) / "tokenizer.json").read_bytes()).hexdigest(), "cases": out}, ensure_ascii=False, separators=(",", ":")) + chr(10),
        encoding="utf-8")
    print(f"wrote {len(out)} fixture cases to {args.out}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cases")
    c.add_argument("--out", required=True)
    c.add_argument("--spec", default="experiments/rounds/full08b.yaml")
    c.add_argument("--sec", type=int, default=3000)
    c.add_argument("--gen", type=int, default=1000)
    k = sub.add_parser("compare")
    k.add_argument("--final", required=True)
    k.add_argument("--cases")
    k.add_argument("--golden", default="runs/runner-golden/full08b")
    k.add_argument("--out", required=True)
    f = sub.add_parser("fixtures")
    f.add_argument("--final", required=True)
    f.add_argument("--cases")
    f.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    return {"cases": cmd_cases, "compare": cmd_compare, "fixtures": cmd_fixtures}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
