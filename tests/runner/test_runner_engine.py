"""The C ABI binding: flat-array layout, the ctypes call path through a fake library, errors, and `RunnerApi` under `UnslothScorer`."""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from sepal.runner import engine as eng
from sepal.runner.encode import Encoder
from sepal.runner.mock import FakeLib, MockEngine
from sepal.train.unsloth_arm import UnslothScorer

QUESTIONS = {
    "c": {"type": "choice", "instructions": "Which?", "criteria": {"b": "second", "a": "first"}},
    "n": {"type": "noul", "instructions": "Is it?"},
    "s": {"type": "score", "instructions": "Rate", "criteria": ["bad", "ok", "good"]},
}


def encoded(tok, state="user=admin ok", questions=None):
    return Encoder(tok).encode({"state": state, "questions": questions or QUESTIONS})


def test_pack_layout(tiny_tokenizer):
    a, b = encoded(tiny_tokenizer, "one"), encoded(tiny_tokenizer, "two two two", {"n": QUESTIONS["n"]})
    batch = eng.pack([a, b])
    assert batch.n_requests == 2
    assert batch.token_ids.dtype == np.uint32 and batch.token_offsets.dtype == np.uint64
    assert batch.question_types.dtype == np.uint32 and batch.option_spans.dtype == np.uint32
    assert list(batch.token_offsets) == [0, len(a.input_ids), len(a.input_ids) + len(b.input_ids)]
    assert list(batch.question_offsets) == [0, 3, 4]
    assert list(batch.question_types) == [1, 0, 2, 0]
    assert list(batch.option_offsets) == [0, 2, 4, 7, 9]
    assert batch.n_options == 9 and batch.n_questions == 4
    assert list(batch.token_ids[:len(a.input_ids)]) == list(a.input_ids)
    assert list(batch.instruction_spans[:2]) == list(a.questions[0].question_span)
    assert list(batch.option_spans[:4]) == [*a.questions[0].option_spans[0], *a.questions[0].option_spans[1]]
    eng.validate(batch, 16384)


def test_validate_rejects_bad_batches(tiny_tokenizer):
    batch = eng.pack([encoded(tiny_tokenizer)])
    with pytest.raises(ValueError, match="engine takes"):
        eng.validate(batch, 10)
    batch.option_spans[1] = 10**6
    with pytest.raises(ValueError, match="past its"):
        eng.validate(batch, None)
    batch.option_spans[0], batch.option_spans[1] = 5, 4
    with pytest.raises(ValueError, match="begins after"):
        eng.validate(batch, None)


def test_unpack_splits_by_request_and_question(tiny_tokenizer):
    batch = eng.pack([encoded(tiny_tokenizer), encoded(tiny_tokenizer, "x", {"n": QUESTIONS["n"]})])
    rows = eng.unpack(batch, np.arange(9, dtype=np.float32))
    assert rows == [[[0.0, 1.0], [2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0]]]


def test_cengine_through_a_fake_library_matches_the_mock_directly(tiny_tokenizer):
    lib = FakeLib(MockEngine(max_tokens=2000, hidden_size=1024))
    with eng.CEngine({"final_dir": "f", "base_dir": "b", "profile": "mock"}, lib=lib) as engine:
        assert lib.opened["profile"] == "mock" and engine.max_tokens == 2000 and engine.hidden_size == 1024
        batch = eng.pack([encoded(tiny_tokenizer, "alpha"), encoded(tiny_tokenizer, "beta gamma delta")])
        got = engine.score(batch)
        assert got.dtype == np.float32 and np.array_equal(got, MockEngine.logits(batch))
        assert engine.last_times["tokens"] == batch.n_tokens
    assert lib.closed == 1


def test_cengine_errors():
    with pytest.raises(eng.EngineError, match="open failed on purpose"):
        eng.CEngine({}, lib=FakeLib(fail_open=True))
    with pytest.raises(eng.EngineError, match="C API 7"):
        eng.CEngine({}, lib=FakeLib(version=7))
    with pytest.raises(eng.EngineError, match="no runner library"):
        eng.CEngine({}, lib=None)
    with pytest.raises(eng.EngineError, match="not found"):
        eng.CEngine({}, lib="nope/sepal_runner.dll")


def test_a_request_over_the_engine_limit_is_refused_before_the_call(tiny_tokenizer):
    lib = FakeLib(MockEngine(max_tokens=30))
    engine = eng.CEngine({}, lib=lib)
    with pytest.raises(ValueError, match="engine takes"):
        engine.score(eng.pack([encoded(tiny_tokenizer)]))
    assert lib.engine.calls == []


def test_batch_neighbours_do_not_change_a_result(tiny_tokenizer):
    mock = MockEngine()
    backend = eng.EngineBackend(mock)
    records = [encoded(tiny_tokenizer, s) for s in ("a", "b b", "user=admin " * 5)]
    together = backend.logits_many(records)
    alone = [backend.logits_many([r])[0] for r in records]
    assert together == alone


def test_runner_api_interface_under_unsloth_scorer(tiny_tokenizer):
    api = eng.RunnerApi(Encoder(tiny_tokenizer), eng.EngineBackend(MockEngine()), 1024)
    assert api.training is False and api.hidden_size == 1024
    scorer = UnslothScorer(api, 16384, workers=0)
    record = {"state": "user=admin ok", "questions": QUESTIONS, "_meta": {"id": "r1"}}
    probabilities, logits, tokens = scorer.score(record)
    assert tokens == len(api.encode("user=admin ok", QUESTIONS, 16384).input_ids)
    for qid, row in probabilities.items():
        assert math.isclose(sum(row.values()), 1.0)
    assert set(probabilities["n"]) == {"yes", "no"}            # Unsloth's true/false come back as our keys
    top = max(logits["c"], key=logits["c"].get)
    assert max(probabilities["c"], key=probabilities["c"].get) == top
    many_p, many_z, _ = scorer.answers([record, {**record, "_meta": {"id": "r2"}, "state": "other"}])
    assert many_p["r1"] == probabilities and many_z["r1"] == logits       # a batch changes nothing but the speed
    assert many_p["r2"] != probabilities
    scorer.close()
