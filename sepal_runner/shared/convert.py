"""Record conversion: the state text, the questions without labels, option keys back from the model's option ids.

Copied from Sepal (Apache-2.0, https://github.com/bloomifycafe/Sepal), commit f846e8d:
  sepal/train/unsloth_convert.py  (neutralise, check_identifier, state_text, convert_question, request_questions, to_our_keys)
  socdecide/schema.py             (state_text, here `schema_state_text`)
No Unsloth code is used or copied. Keep the behaviour identical: `tests/test_shared_parity.py` compares it with Sepal's.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

NOUL_TO_UNSLOTH = {"yes": "true", "no": "false"}
NOUL_FROM_UNSLOTH = {v: k for k, v in NOUL_TO_UNSLOTH.items()}

# The Qwen3.5 tokenizer turns every one of its added tokens written out in a text into that token's id, so a caller's text could close the
# user turn and open an assistant turn inside the state. `neutralise` rewrites any such string as `<¦name¦>`, which the tokenizer reads
# as ordinary characters. It covers the ChatML family `<|word|>` and the added tokens without bars. It does not touch look-alikes of the
# prompt's plain-text section markers (`STATE:`, `SCHEMA FIELDS:`, ...).
_CONTROL = re.compile(r"<\|[A-Za-z0-9_]+\|>|</?(?:think|tool_call|tool_response)>|<tts_[a-z_]+>")


def _defang(match: re.Match) -> str:
    return "<¦" + match.group(0)[1:-1].strip("|") + "¦>"


def neutralise(text: str) -> str:
    """`text` with every tokenizer control string rewritten so it cannot become a control token. Idempotent."""
    return _CONTROL.sub(_defang, text) if isinstance(text, str) else text


def check_identifier(name: str) -> str:
    """Question ids and option keys cannot be rewritten (answers are mapped back by them), so one that holds a control string is refused."""
    if _CONTROL.search(str(name)):
        raise ValueError(f"identifier {name!r} holds a tokenizer control string")
    return str(name)


def schema_state_text(state: Any) -> str:
    """A state as text. Strings pass through; JSON values become `key: value` lines, nested keys joined by '.'."""
    if isinstance(state, str):
        return state
    lines: list[str] = []

    def walk(prefix: str, value: Any):
        if isinstance(value, dict):
            for k, v in value.items():
                walk(f"{prefix}.{k}" if prefix else str(k), v)
        elif isinstance(value, list):
            for i, v in enumerate(value):
                walk(f"{prefix}.{i}" if prefix else str(i), v)
        else:
            lines.append(f"{prefix}: {value}" if prefix else str(value))

    walk("", state)
    return "\n".join(lines)


def state_text(state) -> str:
    """The state text the model reads, in training and in scoring: the state as text with control strings neutralised."""
    return neutralise(schema_state_text(state))


def convert_question(question: dict) -> dict:
    """The question as the model reads it: type, instructions, criteria (control strings neutralised)."""
    kind = question["type"]
    out: dict = {"type": kind, "instructions": neutralise(question.get("instructions") or "")}
    criteria = question.get("criteria")
    if kind == "choice":
        out["criteria"] = {check_identifier(key): neutralise(value) for key, value in criteria.items()}
    elif kind == "score":
        out["criteria"] = [neutralise(level) for level in criteria]
    elif kind == "noul" and criteria:
        out["criteria"] = {k: neutralise(v) for k, v in criteria.items() if k in ("true", "false")}
    return out


def request_questions(record: dict) -> dict:
    """The questions of a record without any label."""
    return {check_identifier(qid): convert_question(q) for qid, q in record["questions"].items()}


def to_our_keys(kind: str, probabilities: Mapping[str, float]) -> dict[str, float]:
    """{option id: p} from the model, as {our option key: p}."""
    if kind == "noul":
        return {NOUL_FROM_UNSLOTH[k]: float(v) for k, v in probabilities.items()}
    return {str(k): float(v) for k, v in probabilities.items()}
