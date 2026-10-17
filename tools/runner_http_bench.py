"""HTTP load generator for a `/v1/systemone` server (sepal_runner.server or decision_api): concurrent keep-alive clients over request
bodies, requests per second and latency percentiles as the client sees them, so the Python front end's overhead is visible next to the
engine's own numbers.

    uv run python tools/runner_http_bench.py --url http://127.0.0.1:8000 --bodies runs/runner-bench/workloads/X.jsonl \
        --concurrency 16 [--requests N] [--bust-cache] [--out result.json]

`--bodies` is a .jsonl of bodies (or of {"body": ...} / {"request": ...} lines) or a .json list / {"bodies"|"requests": [...]}. Bodies are sent
in a loop until `--requests` were answered. `--bust-cache` appends a unique string to every state so the result cache never hits
(a real engine measurement); without it a repeated body is a cache hit, which is what the cache is for.
"""
from __future__ import annotations

import argparse
import http.client
import itertools
import json
import statistics
import threading
import time
from pathlib import Path
from urllib.parse import urlparse


def load_bodies(path: str) -> list[dict]:
    text = Path(path).read_text(encoding="utf-8")
    if path.endswith(".jsonl"):
        items = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        data = json.loads(text)
        items = data if isinstance(data, list) else data.get("bodies") or data.get("requests") or []
    return [i.get("body") or i.get("request") if isinstance(i, dict) and ("body" in i or "request" in i) else i for i in items]


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))] if ordered else 0.0


def run(url: str, bodies: list[dict], concurrency: int, requests: int, bust: bool, timeout: float = 600.0) -> dict:
    target = urlparse(url)
    counter = itertools.count()
    lock = threading.Lock()
    latencies: list[float] = []
    statuses: dict[int, int] = {}
    tokens = [0]
    errors: list[str] = []

    def worker():
        conn = http.client.HTTPConnection(target.hostname, target.port, timeout=timeout)
        while True:
            i = next(counter)
            if i >= requests:
                break
            body = bodies[i % len(bodies)]
            if bust and isinstance(body, dict):
                body = {**body, "state": (body["state"] if isinstance(body.get("state"), str) else json.dumps(body.get("state"))) + f"\nnonce {i}"}
            data = json.dumps(body).encode("utf-8")
            started = time.perf_counter()
            try:
                conn.request("POST", "/v1/systemone", data, {"Content-Type": "application/json"})
                response = conn.getresponse()
                payload = json.loads(response.read())
                status = response.status
            except Exception as error:      # noqa: BLE001
                with lock:
                    errors.append(f"{type(error).__name__}: {error}")
                conn.close()
                conn = http.client.HTTPConnection(target.hostname, target.port, timeout=timeout)
                continue
            elapsed = time.perf_counter() - started
            with lock:
                latencies.append(elapsed * 1000)
                statuses[status] = statuses.get(status, 0) + 1
                if status == 200:
                    tokens[0] += payload.get("usage", {}).get("input_tokens", 0)
        conn.close()

    started = time.perf_counter()
    threads = [threading.Thread(target=worker) for _ in range(concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - started
    answered = len(latencies)
    return {"url": url, "concurrency": concurrency, "requests_sent": requests, "answered": answered, "statuses": statuses,
            "client_errors": len(errors), "first_errors": errors[:3], "wall_s": round(wall, 3),
            "requests_per_s": round(answered / wall, 2) if wall else 0, "input_tokens": tokens[0],
            "input_tokens_per_s": round(tokens[0] / wall, 1) if wall else 0, "bust_cache": bust,
            "latency_ms": {"mean": round(statistics.fmean(latencies), 1) if latencies else 0, "p50": round(percentile(latencies, 0.5), 1),
                           "p95": round(percentile(latencies, 0.95), 1), "p99": round(percentile(latencies, 0.99), 1),
                           "max": round(max(latencies), 1) if latencies else 0}}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--bodies", required=True)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--requests", type=int, default=0, help="requests to send (default: one pass over the bodies)")
    parser.add_argument("--limit", type=int, default=0, help="use only the first N bodies")
    parser.add_argument("--bust-cache", action="store_true")
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    bodies = load_bodies(args.bodies)
    if args.limit:
        bodies = bodies[:args.limit]
    result = run(args.url, bodies, args.concurrency, args.requests or len(bodies), args.bust_cache)
    result["bodies"] = len(bodies)
    result["bodies_file"] = args.bodies
    print(json.dumps(result, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    return 0 if not result["client_errors"] and set(result["statuses"]) <= {200, 422} else 1


if __name__ == "__main__":
    raise SystemExit(main())
