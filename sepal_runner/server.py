"""`POST /v1/systemone` served by the Sepal runner (C++ engine behind the C ABI, Python front end).

    python -m sepal_runner.server --final DIR --base DIR --lib PATH --profile gpu-f16 [--host 127.0.0.1] [--port 8000] [--name sepal]

The same wire format, refusals and answer JSON as Sepal's `sepal.serve.decision_api` (it uses the copies in `sepal_runner.shared` of
`to_record`, `Refused`, `NoTruncateApi`, `UnslothScorer` for the temperatures and the softmax, and `build_answer`), but the model is the runner: a request is
tokenised on a thread pool (`sepal_runner.encode`, the fast `tokenizers` library), the encoded records of concurrent requests are
coalesced into one `sepal_score` call (a token budget and a short maximum wait), and identical requests are answered once: a result
cache keyed by the canonical request, and in-flight coalescing of identical requests that arrive while the first is being scored.

Endpoints: `POST /v1/systemone`; `POST /v1/systemone/batch` with `{"requests": [body, ...]}` answering
`{"results": [{"status": 200, "body": answer} | {"status": 422, "body": {"detail": ...}}, ...]}`; `GET /health`; `GET /stats`.
Binds to 127.0.0.1; a non-loopback host is refused unless an API key is set (`--api-key` / $SEPAL_API_KEY; Bearer or X-API-Key header).
A full server (`--max-pending` requests in flight) answers 429, a request waiting longer than `--request-timeout` 504; `/stats` (JSON) and `/metrics` (Prometheus text) report counters and latency percentiles.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import hashlib
import hmac
import os
import ipaddress
import json
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from sepal_runner.shared.decision import MAX_OPTIONS, NoTruncateApi, Refused, build_answer, to_record  # noqa: F401
from sepal_runner.shared.scoring import MAX_SCORE_LEN, ContextOverflow, UnslothScorer

TOKEN_BUDGET = 65536        # tokens of one engine call at most (a single longer request goes alone)
MAX_BATCH_REQUESTS = 64
MAX_WAIT_S = 0.002          # how long a batch waits for company once the engine is idle
MAX_BATCH_BODIES = 4096     # bodies of one /v1/systemone/batch call
MAX_PENDING = 512           # requests in flight (queued, tokenising or scoring) before new ones get 429
REQUEST_TIMEOUT_S = 600.0   # a request waits at most this long for its answer (504)


class BatchScheduler:
    """Coalesces `submit(encoded)` calls from many threads into `backend.logits_many(batch)` calls on one worker thread.

    While the engine is busy requests pile up and the next call takes all that fit the token budget at once; when the engine is idle
    the first request waits at most `max_wait_s` for more. A batch that fails is retried one request at a time, so a single bad request
    cannot fail its neighbours."""

    def __init__(self, backend: Any, token_budget: int = TOKEN_BUDGET, max_requests: int = MAX_BATCH_REQUESTS, max_wait_s: float = MAX_WAIT_S):
        self.backend = backend
        self.token_budget, self.max_requests, self.max_wait_s = int(token_budget), int(max_requests), float(max_wait_s)
        self._queue: queue.Queue = queue.Queue()
        self._carry: tuple | None = None
        self._stop = False
        self.batches = 0
        self.requests = 0
        self.tokens = 0
        self.engine_seconds = 0.0
        self.sizes: collections.Counter = collections.Counter()
        self._thread = threading.Thread(target=self._run, name="sepal-scheduler", daemon=True)
        self._thread.start()

    def submit(self, encoded: Any) -> concurrent.futures.Future:
        future: concurrent.futures.Future = concurrent.futures.Future()
        if self._stop:
            future.set_exception(RuntimeError("scheduler is closed"))
        else:
            self._queue.put((encoded, future, len(encoded.input_ids)))
        return future

    def logits_many(self, records) -> list:
        """Blocking, for `RunnerApi` users (the CLI and tests): submit all, wait for all."""
        return [f.result() for f in [self.submit(r) for r in records]]

    def _gather(self) -> list[tuple]:
        first = self._carry or self._queue.get()
        self._carry = None
        if first is None:
            return []
        batch, tokens = [first], first[2]
        deadline = time.monotonic() + self.max_wait_s
        while len(batch) < self.max_requests:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    item = self._queue.get(timeout=remaining)
                except queue.Empty:
                    break
            if item is None:
                self._stop = True
                break
            if tokens + item[2] > self.token_budget:
                self._carry = item
                break
            batch.append(item)
            tokens += item[2]
        return batch

    def _run(self) -> None:
        while not self._stop:
            batch = self._gather()
            if not batch:
                continue
            started = time.perf_counter()
            try:
                results = self.backend.logits_many([b[0] for b in batch])
                for (_, future, _), rows in zip(batch, results):
                    future.set_result(rows)
            except Exception as error:                 # noqa: BLE001
                if len(batch) == 1:
                    batch[0][1].set_exception(error)
                else:
                    for item in batch:
                        try:
                            item[1].set_result(self.backend.logits_many([item[0]])[0])
                        except Exception as single:    # noqa: BLE001
                            item[1].set_exception(single)
            self.engine_seconds += time.perf_counter() - started
            self.batches += 1
            self.requests += len(batch)
            self.tokens += sum(b[2] for b in batch)
            self.sizes[len(batch)] += 1
        for item in [self._carry] if self._carry else []:
            item[1].set_exception(RuntimeError("scheduler is closed"))

    def close(self) -> None:
        self._stop = True
        self._queue.put(None)
        self._thread.join(timeout=5)
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if item is not None:
                item[1].set_exception(RuntimeError("scheduler is closed"))


def canonical_key(body: Any) -> str:
    """The request as the model sees it: its state and its questions in order (the order is part of the prompt). `model` and every
    other field do not change the answer, so they are not in the key."""
    if isinstance(body, dict):
        subject = {"state": body.get("state", ""), "questions": body.get("questions")}
    else:
        subject = body
    return hashlib.sha256(json.dumps(subject, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


class RunnerService:
    """Requests in, `decision_api` answers out, through the thread pool, the scheduler and the caches. `submit(body)` returns a Future
    whose result is the answer dict, or whose exception is `Refused` (400/422) or the failure that makes a 500. Answers are shared
    between callers of an identical request: treat them as read-only."""

    def __init__(self, scorer: UnslothScorer, scheduler: BatchScheduler, name: str, workers: int = 4, cache_size: int = 4096,
                 max_pending: int = MAX_PENDING, request_timeout_s: float = REQUEST_TIMEOUT_S):
        self.scorer, self.scheduler, self.name = scorer, scheduler, name
        self.max_pending, self.request_timeout_s = max(1, int(max_pending)), float(request_timeout_s)
        self._latencies: collections.deque = collections.deque(maxlen=2000)
        self.pool = concurrent.futures.ThreadPoolExecutor(max(1, workers), thread_name_prefix="sepal-tokenise")
        self.cache_size = max(0, int(cache_size))
        self._cache: collections.OrderedDict[str, dict] = collections.OrderedDict()
        self._inflight: dict[str, concurrent.futures.Future] = {}
        self._lock = threading.Lock()
        self.stats = collections.Counter()
        self.started = time.time()

    def submit(self, body: Any) -> concurrent.futures.Future:
        key = canonical_key(body)
        with self._lock:
            self.stats["requests"] += 1
            if key in self._cache:
                self._cache.move_to_end(key)
                self.stats["cache_hits"] += 1
                done: concurrent.futures.Future = concurrent.futures.Future()
                done.set_result(self._cache[key])
                return done
            if key in self._inflight:
                self.stats["coalesced"] += 1
                return self._inflight[key]
            if len(self._inflight) >= self.max_pending:
                self.stats["rejected_429"] += 1
                rejected: concurrent.futures.Future = concurrent.futures.Future()
                rejected.set_exception(Refused(429, f"server busy: {self.max_pending} requests in flight, retry later"))
                return rejected
            future: concurrent.futures.Future = concurrent.futures.Future()
            self._inflight[key] = future
        self.pool.submit(self._start, body, key, future)
        return future

    # -- the pipeline: tokenise (pool) -> score (scheduler) -> probabilities and answer (scheduler thread) --

    def _finish(self, key: str, future: concurrent.futures.Future, payload: dict | None, error: BaseException | None) -> None:
        with self._lock:
            self._inflight.pop(key, None)
            if error is None and self.cache_size:
                self._cache[key] = payload
                while len(self._cache) > self.cache_size:
                    self._cache.popitem(last=False)
            self.stats["errors" if error is not None and not isinstance(error, Refused) else "refused" if error else "scored"] += 1
        if error is None:
            future.set_result(payload)
        else:
            future.set_exception(error)

    def _start(self, body: Any, key: str, future: concurrent.futures.Future) -> None:
        try:
            record = to_record(body)
            encoded = self.scorer._encode(record)
            if isinstance(encoded, ContextOverflow):
                raise Refused(422, str(encoded)) from encoded
            scored = self.scheduler.submit(encoded)
        except BaseException as error:                 # noqa: BLE001
            return self._finish(key, future, None, error)

        def done(f: concurrent.futures.Future) -> None:
            try:
                probabilities, _, tokens = self.scorer._finish(record, encoded, f.result())
                payload = build_answer(record, probabilities, tokens, self.name)
            except BaseException as error:             # noqa: BLE001
                return self._finish(key, future, None, error)
            self._finish(key, future, payload, None)

        scored.add_done_callback(done)

    def answer(self, body: Any) -> dict:
        return self.submit(body).result()

    def _wait(self, future: concurrent.futures.Future, started: float) -> tuple[int, dict]:
        try:
            result = 200, future.result(timeout=max(0.0, self.request_timeout_s - (time.monotonic() - started)))
        except Refused as refusal:
            result = refusal.status, {"detail": refusal.message}
        except concurrent.futures.TimeoutError:
            self.stats["timeouts"] += 1
            result = 504, {"detail": f"no answer within {self.request_timeout_s:g} s"}
        except Exception as error:                     # noqa: BLE001  (a real failure: the client records an error and may retry)
            result = 500, {"detail": f"{type(error).__name__}: {error}"}
        with self._lock:
            self._latencies.append(time.monotonic() - started)
            self.stats[f"status_{result[0]}"] += 1
        return result

    def outcome(self, body: Any) -> tuple[int, dict]:
        """(HTTP status, JSON body) exactly as `decision_api`'s handler answers; 429 when the server is full, 504 on timeout."""
        started = time.monotonic()
        return self._wait(self.submit(body), started)

    def outcomes(self, bodies: list) -> list[tuple[int, dict]]:
        started = time.monotonic()
        futures = [self.submit(b) for b in bodies]
        return [self._wait(f, started) for f in futures]

    def snapshot(self) -> dict:
        s = self.scheduler
        with self._lock:
            lat = sorted(self._latencies)
        pct = (lambda q: round(lat[min(len(lat) - 1, int(q * len(lat)))] * 1000, 1)) if lat else (lambda q: None)
        return {"pending": len(self._inflight), "max_pending": self.max_pending, "queue_depth": s._queue.qsize(),
                "latency_ms_p50": pct(0.5), "latency_ms_p95": pct(0.95), "latency_ms_p99": pct(0.99), "model": self.name, "uptime_s": round(time.time() - self.started, 1), **self.stats,
                "cache_entries": len(self._cache), "batches": s.batches, "batched_requests": s.requests, "batched_tokens": s.tokens,
                "mean_batch_requests": round(s.requests / s.batches, 2) if s.batches else 0, "engine_seconds": round(s.engine_seconds, 3),
                "batch_sizes": dict(sorted(s.sizes.items()))}

    def metrics_text(self) -> str:
        """Prometheus text exposition of `snapshot()`'s numbers."""
        lines = []
        for key, value in self.snapshot().items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                lines.append(f"sepal_runner_{key} {value}")
        return "\n".join(lines) + "\n"

    def close(self) -> None:
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.scheduler.close()


def make_handler(service: RunnerService, name: str, api_key: str | None = None):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send(self, status: int, payload: dict) -> None:
            data = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _authorised(self) -> bool:
            if not api_key:
                return True
            given = self.headers.get("X-API-Key") or ""
            bearer = self.headers.get("Authorization") or ""
            if bearer.startswith("Bearer "):
                given = bearer[7:]
            return hmac.compare_digest(given.encode("utf-8"), api_key.encode("utf-8"))

        def do_GET(self):
            if self.path == "/health":
                return self._send(200, {"status": "ok", "model": name})
            if not self._authorised():
                return self._send(401, {"detail": "missing or wrong API key"})
            if self.path == "/stats":
                return self._send(200, service.snapshot())
            if self.path == "/metrics":
                data = service.metrics_text().encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                return self.wfile.write(data)
            self._send(404, {"detail": "not found"})

        def do_POST(self):
            raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if not self._authorised():
                return self._send(401, {"detail": "missing or wrong API key"})
            if self.path not in ("/v1/systemone", "/v1/systemone/batch"):
                return self._send(404, {"detail": "not found"})
            try:
                body = json.loads(raw)
                if self.path == "/v1/systemone":
                    status, payload = service.outcome(body)
                    return self._send(status, payload)
                bodies = body["requests"]
                if not isinstance(bodies, list) or len(bodies) > MAX_BATCH_BODIES:
                    raise Refused(422, f"requests must be a list of at most {MAX_BATCH_BODIES} bodies")
                results = [{"status": s, "body": b} for s, b in service.outcomes(bodies)]
                self._send(200, {"results": results})
            except Refused as refusal:
                self._send(refusal.status, {"detail": refusal.message})
            except Exception as error:                 # noqa: BLE001
                self._send(500, {"detail": f"{type(error).__name__}: {error}"})

        def log_message(self, *args):
            pass

    return Handler


class Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 256


def is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def serve(service: RunnerService, name: str, host: str, port: int, api_key: str | None = None) -> Server:
    """Binds and returns the server (the caller runs `serve_forever`); refuses a non-loopback host without an API key."""
    if not is_loopback(host) and not api_key:
        raise SystemExit(f"refusing to bind {host}: a non-loopback address needs an API key (--api-key or $SEPAL_API_KEY)")
    return Server((host, port), make_handler(service, name, api_key))


def build_service(api: Any, temperatures: Any, name: str, *, backend: Any = None, workers: int = 4, cache_size: int = 4096,
                  token_budget: int = TOKEN_BUDGET, max_requests: int = MAX_BATCH_REQUESTS, max_wait_s: float = MAX_WAIT_S,
                  limit: int = MAX_SCORE_LEN, max_pending: int = MAX_PENDING, request_timeout_s: float = REQUEST_TIMEOUT_S) -> RunnerService:
    """A service over a `RunnerApi` (its `backend` is replaced by a scheduler over it). `temperatures` as `UnslothScorer` takes them."""
    scheduler = BatchScheduler(backend if backend is not None else api.backend, token_budget, max_requests, max_wait_s)
    scored = type(api)(api.encoder, scheduler, api.hidden_size)      # a RunnerApi whose logits go through the scheduler
    scorer = UnslothScorer(NoTruncateApi(scored, limit), limit, name=name, temperatures=temperatures, workers=0)
    return RunnerService(scorer, scheduler, name, workers=workers, cache_size=cache_size, max_pending=max_pending, request_timeout_s=request_timeout_s)


def load_service(final: str, base: str, lib: str | None, profile: str, name: str, *, calibrated: bool = True, **options) -> tuple[RunnerService, Any]:
    from pathlib import Path

    from sepal_runner.engine import open_runner_api
    from sepal_runner.shared.temperatures import fitted_temperatures

    api, engine = open_runner_api(final, base, lib, profile, max_tokens=MAX_SCORE_LEN)
    temperatures = fitted_temperatures(str(Path(final).parent)) if calibrated else None
    return build_service(api, temperatures, name, **options), engine


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--final", required=True, help="the checkpoint's final/ directory (tokenizer.json, sepal.json, adapter, head)")
    parser.add_argument("--base", required=True, help="the base model directory")
    parser.add_argument("--lib", default=None, help="sepal_runner_*.dll/.so (default: $SEPAL_RUNNER_LIB); 'mock': a fake engine, to measure the Python front end alone")
    parser.add_argument("--profile", required=True, help="precision profile, e.g. gpu-f16, gpu-f32, cpu-f32, cpu-bf16")
    parser.add_argument("--name", default="sepal")
    parser.add_argument("--workers", type=int, default=4, help="tokenising threads")
    parser.add_argument("--cache-size", type=int, default=4096, help="result cache entries (0: off)")
    parser.add_argument("--max-pending", type=int, default=MAX_PENDING, help="requests in flight before new ones get 429")
    parser.add_argument("--request-timeout", type=float, default=REQUEST_TIMEOUT_S, help="seconds before a request is answered 504")
    parser.add_argument("--token-budget", type=int, default=TOKEN_BUDGET)
    parser.add_argument("--max-batch-requests", type=int, default=MAX_BATCH_REQUESTS)
    parser.add_argument("--max-wait-ms", type=float, default=MAX_WAIT_S * 1000)
    parser.add_argument("--temperature-1", action="store_true", help="serve raw (temperature 1) probabilities instead of the calibrated ones")


def service_options(args) -> dict:
    return {"workers": args.workers, "cache_size": args.cache_size, "max_pending": args.max_pending, "request_timeout_s": args.request_timeout, "token_budget": args.token_budget,
            "max_requests": args.max_batch_requests, "max_wait_s": args.max_wait_ms / 1000, "calibrated": not args.temperature_1}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_arguments(parser)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api-key", default=os.environ.get("SEPAL_API_KEY"), help="required for a non-loopback --host; sent as 'Authorization: Bearer KEY' or 'X-API-Key: KEY' (/health is open)")
    args = parser.parse_args(argv)
    if not is_loopback(args.host) and not args.api_key:
        raise SystemExit(f"refusing to bind {args.host}: a non-loopback address needs an API key (--api-key or $SEPAL_API_KEY)")
    service, engine = load_service(args.final, args.base, args.lib, args.profile, args.name, **service_options(args))
    server = serve(service, args.name, args.host, args.port, args.api_key)
    print(f"serving {args.name} ({args.profile}) on http://{args.host}:{args.port}/v1/systemone", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        service.close()
        engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
