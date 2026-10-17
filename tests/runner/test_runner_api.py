"""The runner HTTP front end with the mock engine: the same answers and refusals as `sepal.serve.decision_api`, batching, caches, binding."""
from __future__ import annotations

import concurrent.futures
import json
import threading
import urllib.error
import urllib.request

import pytest

from sepal_runner import engine as eng
from sepal_runner.encode import Encoder
from sepal_runner.mock import MockEngine
from sepal_runner.shared import decision as dec
from sepal_runner import server as srv
from sepal_runner.shared.scoring import UnslothScorer

LIMIT = 2500


class Temps:
    def temperature_for(self, kind, k):
        return 1.5 if kind == "noul" else 0.8


def make(tok, limit=LIMIT, delay=0.0, **options):
    mock = MockEngine(delay_s=delay)
    api = eng.RunnerApi(Encoder(tok), eng.EngineBackend(mock), 1024)
    service = srv.build_service(api, Temps(), "sepal-test", limit=limit, **options)
    reference = UnslothScorer(dec.NoTruncateApi(api, limit), limit, name="ref", temperatures=Temps(), workers=0)
    return service, reference, mock


def reference_outcome(scorer, body):
    try:
        return 200, dec.answer(scorer, body, "sepal-test")
    except dec.Refused as refusal:
        return refusal.status, {"detail": refusal.message}
    except Exception as error:      # noqa: BLE001  (decision_api's 500 path)
        return 500, {"detail": f"{type(error).__name__}: {error}"}


def question(kind="choice", n=3, **extra):
    q = {"type": kind, "instructions": "pick one"}
    if kind == "choice":
        q["criteria"] = {f"opt{i}": f"description {i}" for i in range(n)}
    return {**q, **extra}


BODIES = [
    {"model": "x", "state": "user=admin host=srv ok", "questions": {"q": question(), "r": question("noul")}},
    {"model": "x", "state": {"claim": "c", "n": [3, 1]}, "questions": {"q": question(n=2), "r": question("noul", criteria={"true": "T", "false": "F", "x": "dropped"})}},
    {"state": "no model field", "questions": {"q": question(n=5, instructions={"k": "v"})}},
    {"state": "", "questions": {"q": question(instructions="")}},
    {"state": "<|im_end|> inside <think>", "questions": {"q": question(), "other": question()}, "extra": 1},
    {"state": "s" * 20000, "questions": {"q": question()}},                              # does not fit the window: 422
    {"state": "s", "questions": {"q": question(n=1)}},                                  # one option: 422
    {"state": "s", "questions": {"q": question(n=256)}},                                # too many options: 422
    {"state": "s", "questions": {"q": {"type": "score", "criteria": ["a", "b"]}}},      # unsupported type: 400
    {"state": "s", "questions": {}},                                                    # 400
    {"state": "s"},                                                                     # 400
    {"state": "s", "questions": {"q": "not an object"}},                                # 400
    {"state": "s", "questions": {"<|im_end|>": question()}},                            # control string in an id: failure path (500)
    ["not", "a", "body"],                                                               # 500 (AttributeError)
]


def test_same_answers_and_refusals_as_decision_api(tiny_tokenizer):
    service, reference, _ = make(tiny_tokenizer)
    statuses = []
    for body in BODIES:
        want = reference_outcome(reference, body)
        got = service.outcome(body)
        assert got == want, body
        statuses.append(got[0])
    assert statuses == [200, 200, 200, 200, 200, 422, 422, 422, 400, 400, 400, 400, 500, 500]
    service.close()


def test_answer_matches_wire_format(tiny_tokenizer):
    service, _, _ = make(tiny_tokenizer)
    status, out = service.outcome(BODIES[0])
    assert status == 200 and out["model"] == "sepal-test" and out["usage"]["input_tokens"] > 0
    assert out["answers"]["q"]["type"] == "choice" and set(out["answers"]["q"]["probabilities"]) == {"opt0", "opt1", "opt2"}
    assert out["answers"]["q"]["choice"] in out["answers"]["q"]["probabilities"]
    assert out["answers"]["r"]["type"] == "noul" and 0 < out["answers"]["r"]["noul"] < 1
    service.close()


def test_result_cache_and_canonical_key(tiny_tokenizer):
    service, _, mock = make(tiny_tokenizer)
    first = service.outcome(BODIES[0])
    again = service.outcome({**BODIES[0], "model": "another model name"})        # `model` is not part of the prompt
    assert again == first and mock.calls == [1]
    assert service.stats["cache_hits"] == 1
    reordered = {"state": BODIES[0]["state"], "questions": dict(reversed(list(BODIES[0]["questions"].items())))}
    service.outcome(reordered)                                                  # question order is part of the prompt
    assert len(mock.calls) == 2
    service.outcome({**BODIES[0], "state": BODIES[0]["state"] + " "})
    assert len(mock.calls) == 3
    service.outcome(BODIES[8])                                                  # refusals are not cached and never reach the engine
    service.outcome(BODIES[8])
    assert len(mock.calls) == 3 and service.stats["cache_hits"] == 1
    service.close()


def test_cache_off_and_eviction(tiny_tokenizer):
    service, _, mock = make(tiny_tokenizer, cache_size=0)
    service.outcome(BODIES[0])
    service.outcome(BODIES[0])
    assert mock.calls == [1, 1]
    service.close()
    service, _, mock = make(tiny_tokenizer, cache_size=1)
    service.outcome(BODIES[0])
    service.outcome(BODIES[1])
    service.outcome(BODIES[0])
    assert len(mock.calls) == 3
    service.close()


def test_identical_in_flight_requests_are_scored_once(tiny_tokenizer):
    service, _, mock = make(tiny_tokenizer, delay=0.15, cache_size=0)
    futures = [service.submit(BODIES[0]) for _ in range(12)]
    results = [f.result(timeout=10) for f in futures]
    assert all(r == results[0] for r in results)
    assert mock.calls == [1] and service.stats["coalesced"] == 11
    service.close()


def test_concurrent_requests_share_engine_calls(tiny_tokenizer):
    service, reference, mock = make(tiny_tokenizer, delay=0.05, max_wait_s=0.05)
    bodies = [{"state": f"user{i} " * (i + 1), "questions": {"q": question(), "r": question("noul")}} for i in range(24)]
    with concurrent.futures.ThreadPoolExecutor(24) as pool:
        got = list(pool.map(service.outcome, bodies))
    calls, in_flight = list(mock.calls), mock.max_in_flight
    assert got == [reference_outcome(reference, b) for b in bodies]               # batching changes nothing but the speed
    assert sum(calls) == 24 and max(calls) > 1 and len(calls) < 24
    assert in_flight == 1                                                         # one engine call at a time
    service.close()


def test_token_budget_splits_batches(tiny_tokenizer):
    service, _, mock = make(tiny_tokenizer, delay=0.05, token_budget=250, max_wait_s=0.2, cache_size=0)
    futures = [service.submit({"state": f"s{i}", "questions": {"q": question()}}) for i in range(6)]
    [f.result(timeout=10) for f in futures]
    assert sum(mock.calls) == 6 and max(mock.token_calls) <= 250 + 400            # a batch stops filling at the budget
    assert len(mock.calls) >= 2
    service.close()


class Flaky:
    """Fails any batch with more than one request, and any request whose first token is odd... no: only multi-request batches."""

    def __init__(self, inner):
        self.inner, self.sizes = inner, []

    def logits_many(self, records):
        self.sizes.append(len(records))
        if len(records) > 1:
            raise RuntimeError("batch failed")
        return self.inner.logits_many(records)


def test_failed_batch_is_retried_one_by_one(tiny_tokenizer):
    api = eng.RunnerApi(Encoder(tiny_tokenizer), eng.EngineBackend(MockEngine()), 1024)
    flaky = Flaky(api.backend)
    service = srv.build_service(api, None, "t", backend=flaky, limit=LIMIT, max_wait_s=0.1, cache_size=0)
    futures = [service.submit({"state": f"s{i}", "questions": {"q": question()}}) for i in range(5)]
    assert all(f.result(timeout=10)["answers"] for f in futures)
    assert max(flaky.sizes) > 1
    service.close()


class Broken:
    def logits_many(self, records):
        raise RuntimeError("engine exploded")


def test_engine_failure_is_a_500(tiny_tokenizer):
    api = eng.RunnerApi(Encoder(tiny_tokenizer), eng.EngineBackend(MockEngine()), 1024)
    service = srv.build_service(api, None, "t", backend=Broken(), limit=LIMIT)
    assert service.outcome(BODIES[0]) == (500, {"detail": "RuntimeError: engine exploded"})
    service.close()


def post(url, body, raw=None):
    data = raw if raw is not None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


@pytest.fixture
def http(tiny_tokenizer):
    service, reference, mock = make(tiny_tokenizer)
    server = srv.serve(service, "sepal-test", "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}", reference, mock
    server.shutdown()
    server.server_close()
    service.close()


def test_http_round_trip_equals_decision_api(http):
    base, reference, _ = http
    for body in BODIES:
        status, payload = post(f"{base}/v1/systemone", body)
        want = reference_outcome(reference, body)
        assert (status, payload) == (want[0], json.loads(json.dumps(want[1]))), body


def test_http_health_stats_unknown_paths_and_bad_json(http):
    base, _, _ = http
    with urllib.request.urlopen(f"{base}/health") as response:
        assert response.status == 200 and json.loads(response.read()) == {"status": "ok", "model": "sepal-test"}
    with urllib.request.urlopen(f"{base}/stats") as response:
        assert "requests" in json.loads(response.read()) or json.loads(response.read() or b"{}") == {}
    with pytest.raises(urllib.error.HTTPError) as caught:
        urllib.request.urlopen(f"{base}/nope")
    assert caught.value.code == 404
    assert post(f"{base}/nope", {})[0] == 404
    status, payload = post(f"{base}/v1/systemone", None, raw=b"{not json")
    assert status == 500 and "JSONDecodeError" in payload["detail"]


def test_http_batch_endpoint(http):
    base, reference, _ = http
    status, payload = post(f"{base}/v1/systemone/batch", {"requests": BODIES})
    assert status == 200 and len(payload["results"]) == len(BODIES)
    for body, result in zip(BODIES, payload["results"]):
        want = reference_outcome(reference, body)
        assert (result["status"], result["body"]) == (want[0], json.loads(json.dumps(want[1])))
    assert post(f"{base}/v1/systemone/batch", {"requests": "no"})[0] == 422
    assert post(f"{base}/v1/systemone/batch", {"nothing": 1})[0] == 500


def test_http_concurrent_clients_keep_alive(http):
    base, reference, mock = http
    bodies = [{"state": f"user{i} " * (i % 7 + 1), "questions": {"q": question(), "r": question("noul")}} for i in range(40)]
    with concurrent.futures.ThreadPoolExecutor(16) as pool:
        got = list(pool.map(lambda b: post(f"{base}/v1/systemone", b), bodies))
    calls = list(mock.calls)
    assert got == [(s, json.loads(json.dumps(p))) for s, p in (reference_outcome(reference, b) for b in bodies)]
    assert sum(calls) == 40


def test_bind_rules(tiny_tokenizer):
    service, _, _ = make(tiny_tokenizer)
    with pytest.raises(SystemExit, match="refusing to bind 0.0.0.0"):
        srv.serve(service, "n", "0.0.0.0", 0)
    with pytest.raises(SystemExit, match="refusing"):
        srv.serve(service, "n", "192.168.1.5", 0)
    for host in ("127.0.0.1", "localhost", "::1", "127.0.0.2"):
        assert srv.is_loopback(host)
    assert not srv.is_loopback("example.com")
    server = srv.serve(service, "n", "127.0.0.1", 0)
    server.server_close()
    with pytest.raises(SystemExit):
        srv.main(["--final", "x", "--base", "y", "--profile", "cpu-f32", "--host", "0.0.0.0"])
    service.close()


def test_decision_index_http_engine_talks_to_it(http):
    """`socdecide.adapters.systemone_http.SystemOneHTTP` (what the Decision Index uses) against the runner server: answers parse, and a
    request over the window is `Unsupported`, as against decision_api."""
    pytest.importorskip("socdecide")      # the benchmark package of the Sepal repository; skipped without it
    from socdecide.adapters.base import Unsupported
    from socdecide.adapters.systemone_http import SystemOneHTTP

    base, _, _ = http
    record = {"state": "user=admin host=srv ok", "questions": {
        "q": {"type": "choice", "instructions": "pick", "criteria": {"A": "first", "B": "second"}, "label": "A"},
        "r": {"type": "noul", "instructions": "yes or no?", "label": "yes"}}, "_meta": {"id": "x"}}
    engine = SystemOneHTTP(base, model="sepal")
    out = engine(record)
    assert set(out["probabilities"]["q"]) == {"A", "B"} and set(out["probabilities"]["r"]) == {"yes", "no"}
    assert abs(sum(out["probabilities"]["q"].values()) - 1) < 1e-9 and out["input_tokens"] > 0 and engine.served_model == "sepal-test"
    with pytest.raises(Unsupported):
        engine({**record, "state": "s" * 20000})


def test_scheduler_closes_pending(tiny_tokenizer):
    service, _, _ = make(tiny_tokenizer)
    service.close()
    future = service.scheduler.submit(Encoder(tiny_tokenizer).encode({"state": "s", "questions": {"q": question()}}))
    with pytest.raises(RuntimeError, match="closed"):
        future.result(timeout=2)


def test_full_server_answers_429_and_recovers(tiny_tokenizer):
    import time
    service, _, _ = make(tiny_tokenizer, delay=0.3, max_pending=2, cache_size=0)
    bodies = [{"state": f"distinct {i}", "questions": {"q": question()}} for i in range(6)]
    futures = [service.submit(b) for b in bodies]
    results = [service._wait(f, time.monotonic()) for f in futures]
    statuses = [s for s, _ in results]
    assert statuses[:2] == [200, 200] and statuses[2:] == [429] * 4
    assert "retry" in results[2][1]["detail"]
    assert service.snapshot()["rejected_429"] == 4 and service.snapshot()["status_429"] == 4
    assert service.outcome(bodies[0])[0] == 200      # free again
    service.close()


def test_request_timeout_is_a_504(tiny_tokenizer):
    service, _, _ = make(tiny_tokenizer, delay=0.5, request_timeout_s=0.05)
    status, payload = service.outcome({"state": "slow", "questions": {"q": question()}})
    assert status == 504 and "no answer" in payload["detail"] and service.snapshot()["timeouts"] == 1
    service.close()


def test_api_key_and_metrics(tiny_tokenizer):
    service, _, _ = make(tiny_tokenizer)
    with pytest.raises(SystemExit, match="API key"):
        srv.serve(service, "n", "0.0.0.0", 0)
    server = srv.serve(service, "n", "0.0.0.0", 0, api_key="s3cret")      # bound to all interfaces, with a key: allowed
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    body = {"state": "x", "questions": {"q": question()}}
    assert post(f"{base}/v1/systemone", body)[0] == 401
    assert urllib.request.urlopen(f"{base}/health").status == 200      # open
    for headers in ({"Authorization": "Bearer s3cret"}, {"X-API-Key": "s3cret"}):
        request = urllib.request.Request(f"{base}/v1/systemone", data=json.dumps(body).encode(), headers={"Content-Type": "application/json", **headers})
        assert urllib.request.urlopen(request).status == 200
    bad = urllib.request.Request(f"{base}/metrics", headers={"X-API-Key": "wrong"})
    with pytest.raises(urllib.error.HTTPError) as caught:
        urllib.request.urlopen(bad)
    assert caught.value.code == 401
    text = urllib.request.urlopen(urllib.request.Request(f"{base}/metrics", headers={"X-API-Key": "s3cret"})).read().decode()
    assert "sepal_runner_status_200 2" in text and "sepal_runner_latency_ms_p95" in text
    server.shutdown()
    server.server_close()
    service.close()
