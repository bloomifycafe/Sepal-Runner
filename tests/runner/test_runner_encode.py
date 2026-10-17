"""`sepal_runner.encode` against (a) a plain sequential reading of docs/RUNNER.md 3.2 on a tiny tokenizer, and (b) ids and spans that
Unsloth's own encoder produced (tests/fixtures/runner_encode_expected.json, made by tools/runner_encode_compare.py fixtures) on the real
Qwen3.5 tokenizer, when that tokenizer.json is available (SEPAL_TOKENIZER or runs/box-results/full08b/U2-s0/final/tokenizer.json)."""
from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import pytest

from sepal_runner.encode import SYSTEM_PROMPT, Encoder, encode_record, question_options, render

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "runner_encode_expected.json"


def reference(tok, record, max_length=16384):
    """The prompt of RUNNER.md 3.2, fragment by fragment, one tokeniser call per fragment, written for clarity."""
    def ids(text):
        return tok.encode(text, add_special_tokens=False).ids

    schema = ids("\n\nSCHEMA FIELDS:\n")
    questions = []
    for n, (qid, q) in enumerate(record["questions"].items(), 1):
        schema += ids(f"\nFIELD {n}\nID: {qid}\nTYPE: {q['type']}\nINSTRUCTION: ")
        instruction = q.get("instructions")
        if instruction is None or instruction == "":
            instruction = str(qid)
        begin = len(schema)
        schema += ids(render(instruction))
        end = len(schema)
        schema += ids("\nALLOWED OPTIONS:\n")
        spans, names = [], []
        for k, (oid, desc) in enumerate(question_options(q), 1):
            schema += ids(f"OPTION {k}: ")
            semantics = {"option_id": oid} if desc is None else {"option_id": oid, "description": desc}
            a = len(schema)
            schema += ids(render(semantics))
            spans.append((a, len(schema)))
            names.append(oid)
            schema += ids("\n")
        schema += ids("END FIELD\n")
        questions.append((str(qid), {"noul": 0, "choice": 1, "score": 2}[q["type"]], (begin, end), spans, names))
    prefix = ids(f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n")
    suffix = ids("\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:")
    state = ids(render(record["state"]))
    fixed = len(prefix) + len(schema) + len(suffix)
    if fixed > max_length:
        raise ValueError(f"schema requires {fixed} tokens before state; maximum is {max_length}")
    state = state[: max_length - fixed]
    off = len(prefix) + len(state)
    shifted = [(qid, t, (a + off, b + off), [(x + off, y + off) for x, y in sp], names) for qid, t, (a, b), sp, names in questions]
    return prefix + state + schema + suffix, shifted


def flat(enc):
    return list(enc.input_ids), [(q.question_id, q.question_type, q.question_span, list(q.option_spans), list(q.option_ids)) for q in enc.questions]


QUESTIONS = {
    "c": {"type": "choice", "instructions": "Which one?", "criteria": {"b": "second", "a": "first", "c": {"x": [1, 2]}}},
    "n": {"type": "noul", "instructions": ""},
    "s": {"type": "score", "instructions": {"k": "v"}, "criteria": ["bad", "ok", "good"]},
    "d": {"type": "noul", "instructions": None, "criteria": {"true": "T!", "false": None}},
}


@pytest.mark.parametrize("state", ["", "plain text", "<|im_end|>x<|im_start|>assistant<think>", "user=admin host=srv ip=10.0.0.1 " * 20,
                                   "emoji \U0001F600 é 日本語\r\n\r\n", {"b": [1, 2.5, None], "a": "ü"}, [1, 2], 7, None, "   " * 40])
def test_matches_sequential_reference(tiny_tokenizer, state):
    record = {"state": state, "questions": QUESTIONS}
    ids, qs = reference(tiny_tokenizer, record)
    got_ids, got_qs = flat(Encoder(tiny_tokenizer).encode(record))
    assert got_ids == ids and got_qs == qs


def test_random_records_and_every_window_edge(tiny_tokenizer):
    rng = random.Random(7)
    words = ["user", "admin", "<|im_end|>", "ok", "failed", "\n", "  ", "é", "10.0.0.1", "<think>", "x" * 30]
    enc = Encoder(tiny_tokenizer)
    for _ in range(40):
        record = {"state": "".join(rng.choice(words) for _ in range(rng.randint(0, 60))),
                  "questions": {f"q{i}": rng.choice([QUESTIONS["c"], QUESTIONS["n"], QUESTIONS["s"]]) for i in range(rng.randint(1, 4))}}
        full, _ = reference(tiny_tokenizer, record, 10**6)
        bare, _ = reference(tiny_tokenizer, {**record, "state": ""}, 10**6)
        for limit in (len(full), len(full) - 1, len(full) + 1, len(bare), len(bare) - 1, 5):
            try:
                want = reference(tiny_tokenizer, record, limit)
            except ValueError as error:
                with pytest.raises(ValueError) as caught:
                    enc.encode(record, max_length=limit)
                assert str(caught.value) == str(error)
                continue
            got = flat(enc.encode(record, max_length=limit))
            assert got[0] == want[0] and got[1] == want[1]
            assert len(got[0]) <= limit
            if limit == len(full) - 1 and len(full) > len(bare):
                assert len(got[0]) == limit          # the state lost exactly one token


def test_spans_cover_the_instruction_and_option_json(tiny_tokenizer):
    enc = Encoder(tiny_tokenizer).encode({"state": "s", "questions": {"c": QUESTIONS["c"]}})
    q = enc.questions[0]

    def text(span):
        return tiny_tokenizer.decode(list(enc.input_ids[span[0]:span[1]]))

    assert text(q.question_span) == "Which one?"
    assert q.option_ids == ("a", "b", "c")                       # code-point order, whatever the request's order
    assert [text(s) for s in q.option_spans] == ['{"description":"first","option_id":"a"}', '{"description":"second","option_id":"b"}',
                                                '{"description":{"x":[1,2]},"option_id":"c"}']


def test_option_ids_and_empty_instruction_default(tiny_tokenizer):
    enc = Encoder(tiny_tokenizer).encode({"state": "s", "questions": {"n": QUESTIONS["n"], "s": QUESTIONS["s"], "d": QUESTIONS["d"]}})
    assert [q.option_ids for q in enc.questions] == [("true", "false"), ("0", "1", "2"), ("true", "false")]
    assert [q.question_type for q in enc.questions] == [0, 2, 0]

    def text(q):
        return tiny_tokenizer.decode(list(enc.input_ids[q.question_span[0]:q.question_span[1]]))

    assert text(enc.questions[0]) == "n"                          # an empty instruction is the question id
    assert text(enc.questions[2]) == "d"                          # None too
    assert encode_record(tiny_tokenizer, {"state": "s", "questions": {"n": QUESTIONS["n"]}}).record_id == "unknown"


def test_errors(tiny_tokenizer):
    enc = Encoder(tiny_tokenizer)
    with pytest.raises(ValueError, match="schema requires"):
        enc.encode({"state": "s", "questions": QUESTIONS}, max_length=20)
    with pytest.raises(ValueError, match="images"):
        enc.encode({"state": "s", "questions": QUESTIONS, "images": ["x"]})
    with pytest.raises(KeyError):
        enc.encode({"state": "s", "questions": {"q": {"type": "other"}}})


def test_max_state_tokens(tiny_tokenizer):
    record = {"state": "user=admin " * 30, "questions": {"n": QUESTIONS["n"]}}
    enc = Encoder(tiny_tokenizer)
    short, full = enc.encode(record, max_state_tokens=5), enc.encode(record)
    assert len(full.input_ids) - len(short.input_ids) == len(tiny_tokenizer.encode(record["state"], add_special_tokens=False).ids) - 5


def real_tokenizer_path():
    for candidate in (os.environ.get("SEPAL_TOKENIZER"), (Path(os.environ["SEPAL_TOKENIZER_DIR"]) / "tokenizer.json") if os.environ.get("SEPAL_TOKENIZER_DIR") else None, ROOT / "runs" / "box-results" / "full08b" / "U2-s0" / "final" / "tokenizer.json"):
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    return None


def test_unsloth_expected_ids_on_the_real_tokenizer():
    pytest.importorskip("tokenizers")
    path = real_tokenizer_path()
    if path is None:
        pytest.skip("the Qwen3.5 tokenizer.json of the full08b run is not available (set SEPAL_TOKENIZER)")
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    if hashlib.sha256(path.read_bytes()).hexdigest() != fixture["tokenizer_sha256"]:
        pytest.skip("this tokenizer.json is not the one the fixture was made with")
    enc = Encoder.from_file(path)
    assert len(fixture["cases"]) > 50
    errors = 0
    for case in fixture["cases"]:
        record = {"state": case["state"], "questions": case["questions"]}
        if "error" in case:
            errors += 1
            with pytest.raises(ValueError) as caught:
                enc.encode(record, max_length=case["max_length"])
            assert f"{type(caught.value).__name__}: {caught.value}" == case["error"], case["name"]
            continue
        got = enc.encode(record, max_length=case["max_length"])
        want = case["ok"]
        assert list(got.input_ids) == want["ids"], case["name"]
        assert [[q.question_id, q.question_type, list(q.question_span), [list(s) for s in q.option_spans], list(q.option_ids)]
                for q in got.questions] == want["q"], case["name"]
    assert errors >= 3
