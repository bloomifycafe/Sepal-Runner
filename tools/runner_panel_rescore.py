"""Re-score a slice of a panel (default SEC-dev) through the runner (PanelScorer over RunnerApi) and compare with the oracle's rows of the same run.

    uv run python tools/runner_panel_rescore.py --run runs/box-results/full08b/U2-s0 --base DIR --lib PATH --profile gpu-f16 --n 400 --out result.json

The oracle rows are `<run>/eval/rows-<panel>.json` (Unsloth's bf16 autocast scoring of the whole panel at temperature 1, raw logits included), so the
comparison needs no second oracle run: the same records, matched by question id. Reports accuracy (micro, and macro over tasks) of both, the
max and p99 |dp| and the flips (argmax changes) at temperature 1 and at the calibrated temperatures of `final/sepal.json`, and the logit differences.
The GPU lock (runs/gpu.lock.d) is the caller's business: hold it when the profile runs on the GPU.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path


def softmax(z: dict[str, float], t: float) -> dict[str, float]:
    top = max(z.values())
    e = {k: math.exp((v - top) / t) for k, v in z.items()}
    s = sum(e.values())
    return {k: v / s for k, v in e.items()}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--lib")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--panel", default="SEC-dev")
    parser.add_argument("--spec", default="experiments/rounds/full08b.yaml")
    parser.add_argument("--n", type=int, default=400, help="records (seeded sample of the panel)")
    parser.add_argument("--seed", default="runner-rescore")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out")
    args = parser.parse_args(argv)

    from sepal.eval.panel import rows_for
    from sepal.runner.engine import open_runner_api
    from sepal.train import readout, roundspec
    from sepal.train.unsloth_arm import MAX_SCORE_LEN, PanelScorer, UnslothScorer, fitted_temperatures

    run = Path(args.run)
    spec = roundspec.load(args.spec, ".")
    records = readout.panel_records(spec, args.panel)
    oracle = {r["qid"]: r for r in json.loads((run / "eval" / f"rows-{args.panel}.json").read_text(encoding="utf-8"))}
    records = [r for r in records if any(f"{r['_meta']['id']}::{q}" in oracle for q in r["questions"])]
    sample = random.Random(args.seed).sample(records, min(args.n, len(records)))
    print(f"{len(sample)} records of {args.panel}", flush=True)

    api, engine = open_runner_api(run / "final", args.base, args.lib, args.profile)
    scorer = UnslothScorer(api, MAX_SCORE_LEN, name="runner", temperatures=None, workers=args.workers)
    panel = PanelScorer(scorer)
    started = time.perf_counter()
    scored = panel.score_records(sample)
    elapsed = time.perf_counter() - started
    rows = rows_for(sample, scored.probabilities, logits=scored.logits)
    scorer.close()
    engine.close()

    temps = fitted_temperatures(str(run))
    by_task: dict[str, list[list[int]]] = defaultdict(list)
    dp, dp_cal, flips, flips_cal, dz, missing = [], [], 0, 0, [], 0
    for row in rows:
        ref = oracle.get(row.qid)
        if ref is None or row.logits is None:
            missing += 1
            continue
        kind = row.type
        k = len(ref["logits"])
        t = temps.temperature_for(kind, k)
        for scale, bucket in ((1.0, dp), (t, dp_cal)):
            a, b = softmax(row.logits, scale), softmax(ref["logits"], scale)
            bucket.extend(abs(a[x] - b[x]) for x in a)
            if max(a, key=a.get) != max(b, key=b.get):
                if scale == 1.0:
                    flips += 1
                else:
                    flips_cal += 1
        dz.extend(abs(row.logits[x] - ref["logits"][x]) for x in row.logits)
        mine, theirs = max(row.probabilities, key=row.probabilities.get), max(ref["probabilities"], key=ref["probabilities"].get)
        by_task[row.task].append([int(mine == row.label), int(theirs == row.label)])

    def pct(v, q):
        v = sorted(v)
        return v[min(len(v) - 1, int(q * (len(v) - 1)))] if v else float("nan")

    def acc(pairs, i):
        return sum(p[i] for p in pairs) / len(pairs) if pairs else float("nan")

    allp = [p for v in by_task.values() for p in v]
    result = {"panel": args.panel, "records": len(sample), "questions": len(allp), "unmatched": missing, "profile": args.profile,
              "score_seconds": round(elapsed, 2), "records_per_s": round(len(sample) / elapsed, 2), "tokens": scored.tokens,
              "accuracy_micro": {"runner": acc(allp, 0), "oracle": acc(allp, 1)},
              "accuracy_macro_over_tasks": {"runner": sum(acc(v, 0) for v in by_task.values()) / len(by_task),
                                            "oracle": sum(acc(v, 1) for v in by_task.values()) / len(by_task)},
              "per_task": {t: {"n": len(v), "runner": acc(v, 0), "oracle": acc(v, 1)} for t, v in sorted(by_task.items())},
              "dp_t1": {"max": max(dp), "p99": pct(dp, 0.99), "mean": sum(dp) / len(dp)}, "flips_t1": flips,
              "dp_calibrated": {"max": max(dp_cal), "p99": pct(dp_cal, 0.99), "mean": sum(dp_cal) / len(dp_cal)}, "flips_calibrated": flips_cal,
              "abs_logit_diff": {"max": max(dz), "p99": pct(dz, 0.99), "mean": sum(dz) / len(dz)},
              "oracle": "Unsloth bf16 autocast rows of the same run (eval/rows-%s.json)" % args.panel}
    print(json.dumps({k: v for k, v in result.items() if k != "per_task"}, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
