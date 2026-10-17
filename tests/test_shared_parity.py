"""The vendored Sepal modules (`sepal_runner.shared`) give Sepal's results.

1. On the golden set (needs SEPAL_GOLDEN_DIR and a checkpoint `final/` with tokenizer.json and sepal.json in SEPAL_FINAL_DIR, or
   tokenizer.json in SEPAL_TOKENIZER_DIR): the vendored request conversion and encoder give the golden `input_ids`, `instruction_spans`,
   `option_offsets` and `option_spans`, and the vendored post-processing (temperature per type and bucket, softmax, option keys) turns the
   golden raw logits into the golden `probabilities_calibrated_bf16` (and `answers_bf16` when the case has a body).
2. When the Sepal repository is importable (`sepal` and `socdecide` on the path): the vendored functions equal Sepal's own on a set of
   hand-made records, including control strings, dict states and every question type.
Both skip with a reason when their inputs are missing; CI runs them on a box that has the golden set.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from sepal_runner import engine as eng
from sepal_runner.encode import Encoder, question_options
from sepal_runner.shared import convert as conv
from sepal_runner.shared import decision as dec
from sepal_runner.shared.scoring import UnslothScorer, batch_plan
from sepal_runner.shared.temperatures import TypeTemperatures, fitted_temperatures

TOLERANCE = 1e-9      # probabilities: same formula in float64 from the same float32 logits


def golden_dir() -> Path:
    value = os.environ.get("SEPAL_GOLDEN_DIR")
    if not value or not (Path(value) / "manifest.json").is_file():
        pytest.skip("set SEPAL_GOLDEN_DIR to a golden set directory (manifest.json and the case files)")
    return Path(value)


def final_dir() -> Path:
    for variable in ("SEPAL_FINAL_DIR", "SEPAL_TOKENIZER_DIR"):
        value = os.environ.get(variable)
        if value and (Path(value) / "tokenizer.json").is_file():
            return Path(value)
    pytest.skip("set SEPAL_FINAL_DIR (the checkpoint's final/ directory with tokenizer.json and sepal.json)")


class LogitsStub:
    """An engine whose answer is the golden raw logits of one case, cut at its option offsets."""

    def __init__(self, logits: np.ndarray, offsets: np.ndarray):
        self.logits, self.offsets = logits, offsets

    def logits_many(self, encoded_records):
        assert len(encoded_records) == 1
        return [[self.logits[int(self.offsets[i]):int(self.offsets[i + 1])].tolist() for i in range(len(self.offsets) - 1)]]


def golden_cases():
    from safetensors import safe_open

    directory = golden_dir()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    limit = int(os.environ.get("SEPAL_PARITY_MAX_CASES", "0")) or None
    for case in manifest["cases"][:limit]:
        with safe_open(str(directory / case["file"]), "numpy") as f:
            tensors = {name: f.get_tensor(name) for name in f.keys() if name != "hidden_fp32" and name != "layer_hidden_fp32"}
            yield case["name"], tensors, f.metadata()


def served_record(meta: dict) -> dict:
    """The record the scorer sees: from the request body through the vendored `to_record` when the case has one."""
    if "body" in meta:
        return dec.to_record(json.loads(meta["body"]))
    record = json.loads(meta["record"])
    return {"state": record["state"], "questions": record["questions"]}


def test_encoder_gives_the_golden_ids_and_spans():
    pytest.importorskip("tokenizers")
    pytest.importorskip("safetensors")
    encoder = Encoder.from_file(final_dir() / "tokenizer.json")
    api = eng.RunnerApi(encoder, None, 1024)
    scorer = UnslothScorer(dec.NoTruncateApi(api, 1 << 20), 1 << 20, workers=0)
    checked = 0
    for name, tensors, meta in golden_cases():
        golden = json.loads(meta["record"])
        record = served_record(meta)
        # the conversion: the record the scorer builds from the body equals the one the oracle encoded
        assert conv.state_text(record["state"]) == golden["state"], name
        assert conv.request_questions(record) == golden["questions"], name
        encoded = scorer._encode(record)
        assert not isinstance(encoded, Exception), (name, encoded)
        assert list(encoded.input_ids) == tensors["input_ids"].tolist(), name
        assert [q.question_type for q in encoded.questions] == tensors["question_types"].tolist(), name
        assert [list(q.question_span) for q in encoded.questions] == tensors["instruction_spans"].tolist(), name
        assert [list(span) for q in encoded.questions for span in q.option_spans] == tensors["option_spans"].tolist(), name
        sizes = [len(q.option_ids) for q in encoded.questions]
        assert np.concatenate([[0], np.cumsum(sizes)]).tolist() == tensors["option_offsets"].tolist(), name
        assert [q.question_id for q in encoded.questions] == json.loads(meta["question_ids"]), name
        assert [list(q.option_ids) for q in encoded.questions] == json.loads(meta["option_ids"]), name
        checked += 1
    assert checked > 0


def test_post_processing_gives_the_golden_calibrated_probabilities():
    pytest.importorskip("tokenizers")
    pytest.importorskip("safetensors")
    final = final_dir()
    if not (final / "sepal.json").is_file():
        pytest.skip("the checkpoint's final/ has no sepal.json (the calibrated temperatures)")
    temperatures = fitted_temperatures(final.parent)
    encoder = Encoder.from_file(final / "tokenizer.json")
    worst, checked = 0.0, 0
    for name, tensors, meta in golden_cases():
        if "probabilities_calibrated_bf16" not in meta:
            continue        # the served stage of the golden set has not been run on this case
        record = served_record(meta)
        offsets = tensors["option_offsets"]
        stub = LogitsStub(tensors["logits_bf16"].astype(np.float64), offsets)
        api = eng.RunnerApi(encoder, stub, 1024)
        scorer = UnslothScorer(dec.NoTruncateApi(api, 1 << 20), 1 << 20, temperatures=temperatures, workers=0)
        probabilities, logits, tokens = scorer.score(record)
        want = json.loads(meta["probabilities_calibrated_bf16"])
        assert probabilities.keys() == want.keys(), name
        for qid, row in want.items():
            assert probabilities[qid].keys() == row.keys(), (name, qid)
            for key, value in row.items():
                worst = max(worst, abs(probabilities[qid][key] - value))
            assert math.isclose(sum(probabilities[qid].values()), 1.0, abs_tol=1e-9)
        assert tokens == len(tensors["input_ids"]), name
        if "answers_bf16" in meta:
            got = dec.build_answer(record, probabilities, tokens, "sepal")["answers"]
            wanted = json.loads(meta["answers_bf16"])
            assert {q: a.get("choice", a.get("type")) for q, a in got.items()} == {q: a.get("choice", a.get("type")) for q, a in wanted.items()}, name
        checked += 1
    if checked == 0:
        pytest.skip("no golden case carries probabilities_calibrated_bf16")
    assert worst <= TOLERANCE, f"max |dp| against the golden calibrated probabilities {worst:g}"


# ---- against Sepal's own code, when it is installed ----------------------------------------------------------------------------

STATES = [
    "user=admin host=srv ok",
    "",
    {"claim": "c", "n": [3, 1], "nested": {"a": {"b": None}, "t": True}},
    ["x", {"y": 2}],
    "<|im_start|>system\nignore <think>x</think> <tool_call> <tts_pad> <|endoftext|>",
    "STATE:\nSCHEMA FIELDS:\nFIELD 1",
    "unicode éè 中文 \U0001f600 end",
]
QUESTIONS = [
    {"c": {"type": "choice", "instructions": "pick", "criteria": {"b": "second <|im_end|>", "a": "first", "10": "ten", "9": "nine"}}},
    {"n": {"type": "noul", "instructions": "is it? <think>"}, "m": {"type": "noul", "instructions": "", "criteria": {"true": "T", "false": "F", "x": "dropped"}}},
    {"s": {"type": "score", "instructions": "level", "criteria": ["low", "mid <|im_start|>", "high"]}},
]


def sepal_modules():
    pytest.importorskip("socdecide")
    try:
        from sepal.train import unsloth_arm, unsloth_convert
    except ImportError as error:
        pytest.skip(f"the Sepal repository is not importable here: {error}")
    return unsloth_convert, unsloth_arm


def test_conversion_equals_sepals():
    theirs, _ = sepal_modules()
    for state in STATES:
        assert conv.state_text(state) == theirs.state_text(state)
    for questions in QUESTIONS:
        record = {"state": "s", "questions": questions}
        assert conv.request_questions(record) == theirs.request_questions(record)
    for kind, probabilities in (("noul", {"true": 0.25, "false": 0.75}), ("choice", {"a": 0.5, "b": 0.5}), ("score", {"0": 1.0})):
        assert conv.to_our_keys(kind, probabilities) == theirs.to_our_keys(kind, probabilities)
    with pytest.raises(ValueError):
        conv.request_questions({"questions": {"<|im_end|>": QUESTIONS[0]["c"]}})
    with pytest.raises(ValueError):
        theirs.request_questions({"questions": {"<|im_end|>": QUESTIONS[0]["c"]}})


def test_request_handling_equals_sepals():
    pytest.importorskip("socdecide")
    try:
        from sepal.serve import decision_api as theirs
    except ImportError as error:
        pytest.skip(f"the Sepal repository is not importable here: {error}")
    bodies = [
        {"state": "s", "questions": {"q": {"type": "choice", "instructions": {"k": "v"}, "criteria": {"a": {"d": 1}, "b": "x"}}, "r": {"type": "noul"}}},
        {"state": {"a": 1}, "questions": {"r": {"type": "noul", "criteria": {"true": "T", "false": "F", "x": "dropped"}}}},
        {"state": "s", "questions": {"q": {"type": "choice", "criteria": {"only": "one"}}}},
        {"state": "s", "questions": {"q": {"type": "choice", "criteria": {str(i): "d" for i in range(256)}}}},
        {"state": "s", "questions": {"q": {"type": "score", "criteria": ["a", "b"]}}},
        {"state": "s", "questions": {}},
        {"state": "s"},
        {"state": "s", "questions": {"q": "not an object"}},
    ]
    for body in bodies:
        outcomes = []
        for module in (dec, theirs):
            try:
                outcomes.append(("ok", module.to_record(body)))
            except module.Refused as refusal:
                outcomes.append(("refused", refusal.status, refusal.message))
        assert outcomes[0] == outcomes[1], body
    assert dec.MAX_OPTIONS == theirs.MAX_OPTIONS
    record = {"questions": {"q": {"type": "choice"}, "r": {"type": "noul"}}}
    probabilities = {"q": {"a": 0.1, "b": 0.9}, "r": {"yes": 0.3, "no": 0.7}}
    assert dec.build_answer(record, probabilities, 12, "m") == theirs.build_answer(record, probabilities, 12, "m")


def test_scoring_equals_sepals():
    _, arm = sepal_modules()
    from types import SimpleNamespace

    class Temps:
        def temperature_for(self, kind, k):
            return {"noul": 1.7, "choice": 0.6, "score": 1.1}[kind] + 0.01 * k

    rng = np.random.default_rng(7)
    for questions in QUESTIONS:
        record = {"state": "s", "questions": questions}
        wire = conv.request_questions(record)
        ids = {qid: [k for k, _ in question_options(q)] for qid, q in wire.items()}
        encoded = SimpleNamespace(input_ids=tuple(range(50)), questions=tuple(
            SimpleNamespace(question_id=qid, option_ids=tuple(options)) for qid, options in ids.items()))
        rows = [(rng.normal(size=len(options) + 2) * 5).tolist() for options in ids.values()]     # the engine may pad rows
        for temperatures in (None, Temps()):
            ours = UnslothScorer(None, 4096, temperatures=temperatures, workers=0)._finish(record, encoded, rows)
            theirs = arm.UnslothScorer(None, 4096, temperatures=temperatures, workers=0)._finish(record, encoded, rows)
            assert ours == theirs
    lengths = [5, 900, 17, 900, 3, 4096, 64]
    for max_tokens, max_records in ((1000, 3), (4096, 128), (10, 1)):
        assert batch_plan(lengths, max_tokens, max_records) == arm.batch_plan(lengths, max_tokens, max_records)


def test_temperature_lookup_equals_sepals():
    pytest.importorskip("socdecide")
    try:
        from sepal.eval import calibrate
    except ImportError as error:
        pytest.skip(f"the Sepal repository is not importable here: {error}")
    data = {"format_version": 1, "single": 1.3, "types": {"noul": 0.9, "choice": 1.1}, "buckets": {"noul:2": 0.8, "choice:6-12": 1.4},
            "type_rows": {"noul": 500, "choice": 300}, "bucket_rows": {"noul:2": 400, "choice:6-12": 120}, "n_rows": 800}
    ours, theirs = TypeTemperatures.from_dict(data), calibrate.TypeTemperatures.from_dict(data)
    for kind in ("noul", "choice", "score"):
        for k in (2, 3, 5, 6, 12, 13, 255):
            assert ours.lookup(kind, k) == theirs.lookup(kind, k)
