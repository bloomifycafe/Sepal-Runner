"""End-to-end parity of the Python path (encoder -> ctypes -> engine -> UnslothScorer temperatures/softmax) against the golden set.

    python tools/runner_golden_parity.py --golden DIR --final DIR --base DIR --lib PATH --profile cpu-ref [--cases a,b] [--max-tokens 2000] [--out R.json]

For each golden case: the encoder's input_ids / spans must equal the golden's exactly; the engine's logits are compared with `logits_fp32`; the
probabilities (softmax per question at temperature 1 and at the calibrated temperatures of final/sepal.json, applied by the same
`UnslothScorer._finish` the server uses) are compared with the softmax of `logits_fp32`. Reports max |dlogit|, max |dp|, flips (argmax changes).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def softmax(z, t):
    z = np.asarray(z, dtype=np.float64) / t
    e = np.exp(z - z.max())
    return e / e.sum()


def main(argv=None) -> int:
    from sepal_runner.engine import open_runner_api
    from sepal_runner.shared.scoring import UnslothScorer
    from sepal_runner.shared.temperatures import fitted_temperatures
    from sepal_runner.shared.decision import NoTruncateApi

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--golden", required=True)
    p.add_argument("--final", required=True)
    p.add_argument("--base", required=True)
    p.add_argument("--lib")
    p.add_argument("--profile", required=True)
    p.add_argument("--cases", help="comma list of case names (default: all within --max-tokens)")
    p.add_argument("--max-tokens", type=int, default=2000)
    p.add_argument("--out")
    args = p.parse_args(argv)

    manifest = json.loads((Path(args.golden) / "manifest.json").read_text(encoding="utf-8"))
    wanted = set(args.cases.split(",")) if args.cases else None
    api, engine = open_runner_api(args.final, args.base, args.lib, args.profile, max_tokens=max(args.max_tokens, 64))
    temps = fitted_temperatures(str(Path(args.final).parent))
    scorers = {"T1": UnslothScorer(NoTruncateApi(api, 1 << 20), 1 << 20, name="t1", temperatures=None, workers=0),
               "cal": UnslothScorer(NoTruncateApi(api, 1 << 20), 1 << 20, name="cal", temperatures=temps, workers=0)}
    rows, worst = [], {"ids": 0, "dlogit": 0.0, "dp_T1": 0.0, "dp_cal": 0.0, "flips_T1": 0, "flips_cal": 0, "questions": 0}
    for case in manifest["cases"]:
        if (wanted and case["name"] not in wanted) or (not wanted and case["tokens"] > args.max_tokens):
            continue
        with safe_open(str(Path(args.golden) / case["file"]), "numpy") as f:
            meta = f.metadata()
            ids, spans, offsets = f.get_tensor("input_ids"), f.get_tensor("instruction_spans"), f.get_tensor("option_offsets")
            ospans, gold = f.get_tensor("option_spans"), f.get_tensor("logits_fp32").astype(np.float64)
        record = json.loads(meta["record"])
        encoded = api.encode(record["state"], record["questions"], 1 << 20)
        ids_ok = list(encoded.input_ids) == ids.tolist()
        spans_ok = ([list(q.question_span) for q in encoded.questions] == spans.tolist()
                    and [list(s) for q in encoded.questions for s in q.option_spans] == ospans.tolist())
        started = time.perf_counter()
        logits = api.logits(encoded)
        seconds = time.perf_counter() - started
        flat = np.concatenate([np.asarray(q, dtype=np.float64) for q in logits])
        dlogit = float(np.abs(flat - gold).max())
        row = {"case": case["name"], "tokens": case["tokens"], "ids_equal": ids_ok, "spans_equal": spans_ok, "max_dlogit": dlogit, "engine_s": round(seconds, 2)}
        qids = json.loads(meta["question_ids"])
        for tag, scorer in scorers.items():
            probs, _, _ = scorer._finish(record, encoded, logits)
            dp, flips = 0.0, 0
            for qi, qid in enumerate(qids):
                lo, hi = int(offsets[qi]), int(offsets[qi + 1])
                kind = record["questions"][qid]["type"]
                t = 1.0 if tag == "T1" or temps is None else temps.temperature_for(kind, hi - lo)
                want = softmax(gold[lo:hi], t)
                got = np.array(list(probs[qid].values()))
                dp = max(dp, float(np.abs(got - want).max()))
                flips += int(got.argmax() != want.argmax())
            row[f"max_dp_{tag}"], row[f"flips_{tag}"] = dp, flips
            worst[f"dp_{tag if tag == 'cal' else 'T1'}"] = max(worst[f"dp_{tag if tag == 'cal' else 'T1'}"], dp)
            worst[f"flips_{tag if tag == 'cal' else 'T1'}"] += flips
        worst["ids"] += int(not (ids_ok and spans_ok))
        worst["dlogit"] = max(worst["dlogit"], dlogit)
        worst["questions"] += len(qids)
        rows.append(row)
        print(json.dumps(row), flush=True)
    engine.close()
    summary = {"profile": args.profile, "cases": len(rows), **worst}
    print(json.dumps(summary))
    if args.out:
        Path(args.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=1) + "\n", encoding="utf-8")
    return 0 if worst["ids"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
