"""Our own implementation of Clef's `encode_record` (Apache-2.0 source: clef-flash `joint_schema_model.py`, the file Unsloth vendors
byte for byte) on the fast HF `tokenizers` library. The runner's front end uses this instead of Unsloth's encoder, which needs torch and
the slow `transformers` tokenizer wrapper.

Same contract as the oracle (docs/RUNNER.md 3.2): the prompt is the concatenation of the ids of separately tokenised fragments (prefix,
state, schema head, per question a header / the instruction / `ALLOWED OPTIONS` / per option `OPTION k: `, the option JSON, newline, then
`END FIELD`, and the suffix), special tokens written in a text are matched as special tokens (nothing is neutralised here: the caller does
that, `sepal.train.unsloth_convert.state_text`), the state is cut at its end to what the window leaves, the questions and options are never
cut, and `ValueError` is raised when prefix, schema and suffix alone exceed the window. Spans are half-open and absolute in the record.

All fragments of a record go through ONE `encode_batch` call, which `tokenizers` runs on its own Rust threads without the GIL, so a state of
16k tokens and a hundred small schema fragments tokenise in parallel; `Encoder.encode` is safe to call from many Python threads.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SYSTEM_PROMPT = (
    "Read the complete state and schema. Decide every field jointly. Each answer "
    "must be exactly one of that field's allowed options."
)
QUESTION_TYPES = {"noul": 0, "choice": 1, "score": 2}
DEFAULT_MAX_LENGTH = 16384
SCHEMA_HEAD = "\n\nSCHEMA FIELDS:\n"
PREFIX = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n"
SUFFIX = "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:"


def render(value: Any) -> str:
    """A string as it is; anything else as compact JSON with sorted keys and no ASCII escaping."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def question_options(question: dict[str, Any]) -> list[tuple[str, Any]]:
    """(option id, description) in prompt order: noul `true`, `false`; choice by key in code-point order; score `0..n-1` in list order."""
    question_type = str(question["type"])
    if question_type == "noul":
        criteria = {
            "true": "The proposition is true or the answer is yes.",
            "false": "The proposition is false or the answer is no.",
        }
        criteria.update(question.get("criteria") or {})
        return [(key, criteria[key]) for key in ("true", "false")]
    if question_type == "choice":
        return sorted((str(key), value) for key, value in question["criteria"].items())
    return [(str(index), value) for index, value in enumerate(question["criteria"])]


@dataclass(frozen=True)
class EncodedQuestion:
    question_id: str
    question_type: int
    question_span: tuple[int, int]
    option_spans: tuple[tuple[int, int], ...]
    option_ids: tuple[str, ...]


@dataclass(frozen=True)
class EncodedRecord:
    input_ids: tuple[int, ...]
    questions: tuple[EncodedQuestion, ...]
    record_id: str
    media: dict[str, Any] | None = None


class Encoder:
    """Clef's `encode_record` over a `tokenizers.Tokenizer` (`final/tokenizer.json`)."""

    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer

    @classmethod
    def from_file(cls, path: str | Path) -> "Encoder":
        from tokenizers import Tokenizer

        return cls(Tokenizer.from_file(str(path)))

    def tokens(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def _batch(self, texts: list[str]) -> list[list[int]]:
        return [e.ids for e in self.tokenizer.encode_batch(texts, add_special_tokens=False)]

    def encode(self, record: dict[str, Any], max_length: int = DEFAULT_MAX_LENGTH,
               max_state_tokens: int | None = None) -> EncodedRecord:
        """`record` = {"state": ..., "questions": {qid: {type, instructions, criteria}}, optional "id"}. Media is not supported (Sepal sends none)."""
        if record.get("images") or record.get("videos"):
            raise ValueError("records with images or videos are not supported by the runner")
        # Every fragment, in order. A slot is (kind, index into `texts`); the layout is rebuilt after one batched tokenisation.
        texts: list[str] = [PREFIX, render(record["state"]), SCHEMA_HEAD]
        plan: list[tuple[str, Any]] = []         # per question: (question_id, type, [fragment indices ...])
        for question_index, (question_id, question) in enumerate(record["questions"].items()):
            instructions = question.get("instructions")
            if instructions is None or instructions == "":
                instructions = str(question_id)
            header = f"\nFIELD {question_index + 1}\nID: {question_id}\nTYPE: {question['type']}\nINSTRUCTION: "
            at_header = len(texts)
            texts.extend((header, render(instructions), "\nALLOWED OPTIONS:\n"))
            options: list[tuple[str, int, int]] = []      # (option id, index of the label, index of the json)
            for option_index, (option_id, description) in enumerate(question_options(question)):
                semantics = {"option_id": option_id}
                if description is not None:
                    semantics["description"] = description
                options.append((option_id, len(texts), len(texts) + 1))
                texts.extend((f"OPTION {option_index + 1}: ", render(semantics), "\n"))
            at_end = len(texts)
            texts.append("END FIELD\n")
            plan.append((str(question_id), QUESTION_TYPES[str(question["type"])], at_header, options, at_end))
        texts.append(SUFFIX)
        ids = self._batch(texts)

        prefix_ids, state_ids, schema_ids = ids[0], ids[1], list(ids[2])
        suffix_ids = ids[-1]
        questions: list[EncodedQuestion] = []
        for question_id, question_type, at_header, options, at_end in plan:
            schema_ids.extend(ids[at_header])
            question_start = len(schema_ids)
            schema_ids.extend(ids[at_header + 1])
            question_end = len(schema_ids)
            schema_ids.extend(ids[at_header + 2])
            spans: list[tuple[int, int]] = []
            option_ids: list[str] = []
            for option_id, label_at, json_at in options:
                schema_ids.extend(ids[label_at])
                start = len(schema_ids)
                schema_ids.extend(ids[json_at])
                spans.append((start, len(schema_ids)))
                option_ids.append(option_id)
                schema_ids.extend(ids[json_at + 1])
            schema_ids.extend(ids[at_end])
            questions.append(EncodedQuestion(question_id, question_type, (question_start, question_end), tuple(spans), tuple(option_ids)))

        if max_state_tokens is not None:
            state_ids = state_ids[:max_state_tokens]
        fixed_length = len(prefix_ids) + len(schema_ids) + len(suffix_ids)
        if fixed_length > max_length:
            raise ValueError(f"schema requires {fixed_length} tokens before state; maximum is {max_length}")
        state_ids = state_ids[: max_length - fixed_length]
        schema_offset = len(prefix_ids) + len(state_ids)
        shifted = tuple(
            EncodedQuestion(
                q.question_id, q.question_type,
                (q.question_span[0] + schema_offset, q.question_span[1] + schema_offset),
                tuple((s + schema_offset, e + schema_offset) for s, e in q.option_spans),
                q.option_ids,
            )
            for q in questions
        )
        input_ids = tuple(prefix_ids + state_ids + schema_ids + suffix_ids)
        if not input_ids or not shifted:
            raise ValueError("record produced no model input or questions")
        return EncodedRecord(input_ids=input_ids, questions=shifted, record_id=str(record.get("id", "unknown")))


def encode_record(tokenizer: Any, record: dict[str, Any], max_length: int = DEFAULT_MAX_LENGTH,
                  max_state_tokens: int | None = None) -> EncodedRecord:
    """Function form: `tokenizer` is a `tokenizers.Tokenizer` or an `Encoder`."""
    encoder = tokenizer if isinstance(tokenizer, Encoder) else Encoder(tokenizer)
    return encoder.encode(record, max_length, max_state_tokens)
