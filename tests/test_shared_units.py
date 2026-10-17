"""The vendored modules on their own: no Sepal, no data, no tokenizer download."""
from __future__ import annotations

import json
import math

import pytest

from sepal_runner.shared import convert as conv
from sepal_runner.shared import decision as dec
from sepal_runner.shared.scoring import UnslothScorer, batch_plan
from sepal_runner.shared.temperatures import TypeTemperatures, bucket_key, fitted_temperatures, k_bucket, read_temperatures


def test_control_strings_are_defanged_and_the_rest_is_untouched():
    assert conv.neutralise("a <|im_end|> b <think>c</think> <tts_pad>") == "a <¦im_end¦> b <¦think¦>c<¦/think¦> <¦tts_pad¦>"
    assert conv.neutralise("STATE: plain <b>text</b>") == "STATE: plain <b>text</b>"
    assert conv.neutralise(conv.neutralise("<|x|>")) == conv.neutralise("<|x|>")
    with pytest.raises(ValueError):
        conv.check_identifier("<|im_start|>")


def test_state_text_of_json_values():
    assert conv.state_text("as is") == "as is"
    assert conv.state_text({"a": {"b": 1}, "c": [True, "x"]}) == "a.b: 1\nc.0: True\nc.1: x"


def test_to_record_and_refusals():
    body = {"state": "s", "questions": {"q": {"type": "choice", "instructions": {"k": 1}, "criteria": {"a": "x", "b": {"d": 2}}},
                                        "r": {"type": "noul", "criteria": {"true": "T", "false": "F", "other": "dropped"}}}}
    record = dec.to_record(body)
    assert record["questions"]["q"] == {"type": "choice", "instructions": '{"k":1}', "criteria": {"a": "x", "b": '{"d":2}'}}
    assert record["questions"]["r"]["criteria"] == {"true": "T", "false": "F"}
    for bad, status in (({"questions": {}}, 400), ({"state": "s", "questions": {"q": {"type": "score"}}}, 400),
                        ({"state": "s", "questions": {"q": {"type": "choice", "criteria": {"a": "x"}}}}, 422),
                        ({"state": "s", "questions": {"q": {"type": "choice", "criteria": {str(i): "d" for i in range(256)}}}}, 422)):
        with pytest.raises(dec.Refused) as caught:
            dec.to_record(bad)
        assert caught.value.status == status


def test_build_answer_shape():
    record = {"questions": {"q": {"type": "choice"}, "r": {"type": "noul"}}}
    answer = dec.build_answer(record, {"q": {"a": 0.2, "b": 0.8}, "r": {"yes": 0.4, "no": 0.6}}, 9, "m")
    assert answer == {"model": "m", "answers": {"q": {"type": "choice", "choice": "b", "probabilities": {"a": 0.2, "b": 0.8}},
                                                "r": {"type": "noul", "noul": 0.4}}, "usage": {"input_tokens": 9}}


class Api:
    """encode: ids 0..n-1 and the given options; logits: fixed rows."""

    def __init__(self, options, rows):
        self.options, self.rows = options, rows

    def encode(self, state, questions, max_length):
        from types import SimpleNamespace

        return SimpleNamespace(input_ids=tuple(range(10)), questions=tuple(
            SimpleNamespace(question_id=qid, option_ids=tuple(self.options[qid])) for qid in questions))

    def logits(self, encoded):
        return self.rows


def test_temperature_then_softmax_by_hand():
    temperatures = TypeTemperatures.from_dict({"format_version": 1, "single": 1.0, "types": {"noul": 2.0}, "buckets": {"choice:3-5": 0.5},
                                               "type_rows": {}, "bucket_rows": {}, "n_rows": 0})
    api = Api({"n": ["true", "false"], "c": ["a", "b", "c"]}, [[2.0, 0.0], [1.0, 0.0, -1.0]])
    record = {"state": "s", "questions": {"n": {"type": "noul", "instructions": "i"}, "c": {"type": "choice", "instructions": "i", "criteria": {"a": "", "b": "", "c": ""}}}}
    scorer = UnslothScorer(api, 100, temperatures=temperatures, workers=0)
    probabilities, logits, tokens = scorer.score(record)
    assert tokens == 10 and logits["n"] == {"yes": 2.0, "no": 0.0}
    yes = math.exp(1.0) / (math.exp(1.0) + 1.0)                   # logits / 2.0
    assert probabilities["n"]["yes"] == pytest.approx(yes, abs=1e-15)
    z = [math.exp(2.0), math.exp(0.0), math.exp(-2.0)]            # logits / 0.5
    assert probabilities["c"]["a"] == pytest.approx(z[0] / sum(z), abs=1e-15)
    assert UnslothScorer(api, 100, workers=0).score(record)[0]["n"]["yes"] == pytest.approx(math.exp(2.0) / (math.exp(2.0) + 1.0), abs=1e-15)


def test_buckets_and_lookup():
    assert [k_bucket(k) for k in (2, 3, 5, 6, 12, 13, 255)] == ["2", "3-5", "3-5", "6-12", "6-12", "13+", "13+"]
    assert bucket_key("choice", 7) == "choice:6-12"
    with pytest.raises(ValueError):
        k_bucket(1)
    t = TypeTemperatures(single=1.5, types={"noul": 0.9}, buckets={"noul:2": 0.8}, type_rows={}, bucket_rows={}, n_rows=0)
    assert t.lookup("noul", 2) == (0.8, "bucket") and t.lookup("noul", 4) == (0.9, "type") and t.lookup("choice", 3) == (1.5, "single")


def test_reading_temperatures_from_sepal_json(tmp_path):
    final = tmp_path / "run" / "final"
    final.mkdir(parents=True)
    assert read_temperatures(final) is None
    (final / "sepal.json").write_text(json.dumps({"temperature": 1.25}), encoding="utf-8")
    assert fitted_temperatures(final.parent).single == 1.25
    table = {"format_version": 1, "single": 1.1, "types": {"choice": 0.7}, "buckets": {}, "type_rows": {}, "bucket_rows": {}, "n_rows": 3}
    (final / "sepal.json").write_text(json.dumps({"temperatures": table}), encoding="utf-8")
    assert fitted_temperatures(final.parent).temperature_for("choice", 4) == 0.7
    table["format_version"] = 2
    (final / "sepal.json").write_text(json.dumps({"temperatures": table}), encoding="utf-8")
    with pytest.raises(ValueError):
        read_temperatures(final)


def test_batch_plan_groups_longest_first():
    assert batch_plan([10, 100, 50, 100], 200, 8) == [[1, 3], [2, 0]]
    assert batch_plan([300, 5], 100, 8) == [[0], [1]]
