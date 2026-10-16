"""Re-score a slice of a panel THROUGH THE HTTP SERVER and compare with the oracle's rows (accuracy, probabilities, flips).

    uv run python tools/runner_server_rescore.py prepare --run runs/box-results/full08b/U2-s0 --n 300 --out slice.json     (needs the panel data)
    python tools/runner_server_rescore.py compare --slice slice.json --url http://127.0.0.1:8000 [--clients 8] [--out result.json]

`prepare` samples records of the panel (seeded), writes the /v1/systemone bodies and, per question, the oracle's raw logits, label, task, and its
probabilities at temperature 1 and at the calibrated temperatures. `compare` posts every body to the server (several clients at once, so the
server batches), then compares the served answer (calibrated probabilities; for noul the "yes" probability) with the oracle's calibrated ones.
`compare` needs only the standard library, so it can run on a box without the panel data.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import random
import time
import urllib.request
from collections import defaultdict
from pathlib import Path


def softmax(z: dict, t: float) -> dict:
    top = max(z.values())
    e = {k: math.exp((v - top) / t) for k, v in z.items()}
    s = sum(e.values())
    return {k: v / s for k, v in e.items()}


def prepare(args) -> int:
    from sepal.train import readout, roundspec
    from sepal.train import unsloth_convert as conv
    from sepal.train.unsloth_arm import fitted_temperatures

    run = Path(args.run)
    spec = roundspec.load(args.spec, ".")
    oracle = {r["qid"]: r for r in json.loads((run / "eval" / f"rows-{args.panel}.json").read_text(encoding="utf-8"))}
    records = [r for r in readout.panel_records(spec, args.panel) if any(f"{r['_meta']['id']}::{q}" in oracle for q in r["questions"])]
    sample = random.Random(args.seed).sample(records, min(args.n, len(records)))
    temps = fitted_temperatures(str(run))
    items = []
    for r in sample:
        qs = conv.request_questions(r)
        refs = {}
        for q in qs:
            row = oracle.get(f"{r['_meta']['id']}::{q}")
            if row is None:
                continue
            k = len(row["logits"])
            refs[q] = {"task": row["task"], "type": row["type"], "label": row["label"], "logits": row["logits"],
                       "p1": softmax(row["logits"], 1.0), "pcal": softmax(row["logits"], temps.temperature_for(row["type"], k))}
        items.append({"id": r["_meta"]["id"], "body": {"state": conv.state_text(r["state"]), "questions": qs}, "refs": refs})
    Path(args.out).write_text(json.dumps({"panel": args.panel, "items": items}), encoding="utf-8")
    print(f"{len(items)} records, {sum(len(i['refs']) for i in items)} questions -> {args.out}")
    return 0


def post(url: str, body: dict, key: str | None):
    headers = {"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})}
    request = urllib.request.Request(url.rstrip("/") + "/v1/systemone", data=json.dumps(body).encode("utf-8"), headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=1800) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def compare(args) -> int:
    data = json.loads(Path(args.slice).read_text(encoding="utf-8"))
    items = data["items"]
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(args.clients) as pool:
        replies = list(pool.map(lambda it: post(args.url, it["body"], args.key), items))
    elapsed = time.perf_counter() - started
    dp, flips, errors, by_task, dz_tokens = [], 0, defaultdict(int), defaultdict(list), 0
    for item, (status, reply) in zip(items, replies):
        if status != 200:
            errors[status] += 1
            continue
        dz_tokens += reply["usage"]["input_tokens"]
        for q, ref in item["refs"].items():
            ans = reply["answers"][q]
            if ans["type"] == "noul":
                got = {"yes": ans["noul"], "no": 1 - ans["noul"]}
            else:
                got = ans["probabilities"]
            want = ref["pcal"]
            dp.extend(abs(got[k] - want[k]) for k in want)
            mine, theirs = max(got, key=got.get), max(want, key=want.get)
            flips += int(mine != theirs)
            by_task[ref["task"]].append([int(mine == ref["label"]), int(theirs == ref["label"])])
    allp = [p for v in by_task.values() for p in v]

    def acc(pairs, i):
        return sum(p[i] for p in pairs) / len(pairs) if pairs else float("nan")

    dp.sort()
    result = {"records": len(items), "questions": len(allp), "errors": dict(errors), "seconds": round(elapsed, 2),
              "records_per_s": round(len(items) / elapsed, 2), "tokens": dz_tokens, "tokens_per_s": round(dz_tokens / elapsed),
              "accuracy_micro": {"server": acc(allp, 0), "oracle": acc(allp, 1)},
              "accuracy_macro_over_tasks": {"server": sum(acc(v, 0) for v in by_task.values()) / max(1, len(by_task)),
                                            "oracle": sum(acc(v, 1) for v in by_task.values()) / max(1, len(by_task))},
              "dp_calibrated": {"max": dp[-1] if dp else None, "p99": dp[int(0.99 * (len(dp) - 1))] if dp else None},
              "flips": flips, "flip_rate": flips / max(1, len(allp)),
              "per_task": {t: {"n": len(v), "server": acc(v, 0), "oracle": acc(v, 1)} for t, v in sorted(by_task.items())}}
    print(json.dumps(result, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--run", required=True)
    p.add_argument("--panel", default="SEC-dev")
    p.add_argument("--spec", default="experiments/rounds/full08b.yaml")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--seed", default="runner-server-rescore")
    p.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--slice", required=True)
    c.add_argument("--url", required=True)
    c.add_argument("--key")
    c.add_argument("--clients", type=int, default=8)
    c.add_argument("--out")
    args = parser.parse_args(argv)
    return prepare(args) if args.cmd == "prepare" else compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
