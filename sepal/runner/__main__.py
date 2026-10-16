"""Batch scoring without a server.

    python -m sepal.runner score --in requests.jsonl --out answers.jsonl --final DIR --base DIR --lib PATH --profile gpu-f16

`requests.jsonl` holds one `/v1/systemone` body per line; `answers.jsonl` gets one line per request, in order: the answer JSON the Decision
API returns, or `{"error": {"status": 422, "detail": "..."}}` for a refused or failed request. The requests go through the same service as
the HTTP server (tokenising pool, batching scheduler, result cache), so a big file scores in large engine calls.
"""
from __future__ import annotations

import argparse
import json
import sys
import time


def score(args) -> int:
    from sepal.serve import runner_api

    service, engine = runner_api.load_service(args.final, args.base, args.lib, args.profile, args.name, **runner_api.service_options(args))
    try:
        with open(args.input, encoding="utf-8") as f:
            bodies = [json.loads(line) for line in f if line.strip()]
        started = time.perf_counter()
        results = service.outcomes(bodies)
        elapsed = time.perf_counter() - started
        with open(args.output, "w", encoding="utf-8", newline="\n") as f:
            for status, payload in results:
                f.write(json.dumps(payload if status == 200 else {"error": {"status": status, **payload}}, ensure_ascii=False) + "\n")
        failed = sum(1 for status, _ in results if status != 200)
        print(f"{len(results)} requests in {elapsed:.2f} s ({len(results) / max(elapsed, 1e-9):.1f}/s), {failed} refused or failed; "
              f"{json.dumps(service.snapshot())}", file=sys.stderr)
        return 0
    finally:
        service.close()
        engine.close()


def main(argv: list[str] | None = None) -> int:
    from sepal.serve import runner_api

    parser = argparse.ArgumentParser(prog="python -m sepal.runner", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("score", help="score a JSONL file of request bodies")
    p.add_argument("--in", dest="input", required=True)
    p.add_argument("--out", dest="output", required=True)
    runner_api.add_arguments(p)
    args = parser.parse_args(argv)
    return score(args)


if __name__ == "__main__":
    raise SystemExit(main())
