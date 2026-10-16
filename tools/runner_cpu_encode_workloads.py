"""Encode the runner benchmark workloads (runs/runner-bench/workloads/*.jsonl) into the token-level JSONL that
runner/tools (sepal-score, sepal-cpu-bench) read.

    uv run python tools/runner_cpu_encode_workloads.py --final <final dir> --out runs/runner-bench/cpu/encoded [--shapes t1k_q8,...]

Output line: {"id", "input_ids", "question_types", "instruction_spans", "option_offsets", "option_spans"}.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sepal.runner.encode import Encoder  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--final", required=True)
    p.add_argument("--workloads", default=str(ROOT / "runs/runner-bench/workloads"))
    p.add_argument("--out", required=True)
    p.add_argument("--shapes", default="")
    p.add_argument("--max-length", type=int, default=65536)
    a = p.parse_args()
    enc = Encoder.from_file(Path(a.final) / "tokenizer.json")
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((Path(a.workloads) / "manifest.json").read_text(encoding="utf-8"))
    shapes = a.shapes.split(",") if a.shapes else list(manifest["shapes"])
    for shape in shapes:
        src = Path(a.workloads) / manifest["shapes"][shape]["file"]
        n = tokens = 0
        with open(src, encoding="utf-8") as f, open(out_dir / f"{shape}.jsonl", "w", encoding="utf-8") as g:
            for line in f:
                row = json.loads(line)
                body = row["body"]
                e = enc.encode({"state": body["state"], "questions": body["questions"], "id": row["id"]}, max_length=a.max_length)
                offsets = [0]
                for q in e.questions:
                    offsets.append(offsets[-1] + len(q.option_spans))
                g.write(json.dumps({
                    "id": row["id"], "input_ids": list(e.input_ids), "question_types": [q.question_type for q in e.questions],
                    "instruction_spans": [list(q.question_span) for q in e.questions], "option_offsets": offsets,
                    "option_spans": [list(s) for q in e.questions for s in q.option_spans]}) + "\n")
                n += 1
                tokens += len(e.input_ids)
        print(f"{shape}: {n} requests, {tokens / n:.0f} tokens mean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
