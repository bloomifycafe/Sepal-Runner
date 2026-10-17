"""The `POST /v1/systemone` wire format: request to record, refusals, record and probabilities to answer.

Copied from Sepal (Apache-2.0, https://github.com/bloomifycafe/Sepal), commit f846e8d, sepal/serve/decision_api.py (MAX_OPTIONS, KINDS,
Refused, NoTruncateApi, text, to_record, answer, build_answer). The single-request HTTP handler of that file is not copied; the runner has
its own server (`sepal_runner.server`). Keep the behaviour identical: `tests/test_shared_parity.py` compares it with Sepal's.

Request  `{"model": str, "state": str | json, "questions": {qid: {"type": "choice" | "noul", "instructions": str | json,
          "criteria": {option key: description}}}}`
Response `{"model": name, "answers": {qid: {"type": "choice", "choice": key, "probabilities": {key: p}}
          | {"type": "noul", "noul": p_true}}, "usage": {"input_tokens": n}}`
"""
from __future__ import annotations

import json
from typing import Any

from sepal_runner.shared.scoring import ContextOverflow, UnslothScorer

MAX_OPTIONS = 255
KINDS = ("choice", "noul")


class Refused(Exception):
    """A request the server declines on purpose: `status` (422 capacity, 400 malformed) and the message the client sees."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


class NoTruncateApi:
    """Wraps an encoder that cuts a state that does not fit: ask for one token more than the window and refuse when the record
    needs more than the window."""

    def __init__(self, api: Any, limit: int):
        self.api, self.limit = api, limit

    def encode(self, state, questions, max_length):
        try:
            encoded = self.api.encode(state, questions, self.limit + 1)
        except ValueError as error:      # the schema alone does not fit
            raise ValueError(f"maximum context length is {self.limit} tokens: {error}") from error
        if len(encoded.input_ids) > self.limit:
            raise ValueError(f"maximum context length is {self.limit} tokens; the request needs more")
        return encoded

    def logits(self, encoded):
        return self.api.logits(encoded)


def text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def to_record(body: dict) -> dict:
    """The request as one corpus-style record (strings for instructions and descriptions); `Refused` if malformed or over a limit."""
    questions = body.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise Refused(400, "questions must be a non-empty object")
    out: dict[str, dict] = {}
    for qid, q in questions.items():
        kind = q.get("type") if isinstance(q, dict) else None
        if kind not in KINDS:
            raise Refused(400, f"question {qid!r}: unsupported type {kind!r} (this server answers choice and noul)")
        item: dict = {"type": kind, "instructions": text(q.get("instructions") or "")}
        criteria = q.get("criteria")
        if kind == "choice":
            if not isinstance(criteria, dict) or len(criteria) < 2:
                raise Refused(422, f"question {qid!r}: a choice needs at least two options")
            if len(criteria) > MAX_OPTIONS:
                raise Refused(422, f"question {qid!r}: {len(criteria)} options per choice; the limit is {MAX_OPTIONS}")
            item["criteria"] = {str(k): text(v) for k, v in criteria.items()}
        elif isinstance(criteria, dict):
            item["criteria"] = {str(k): text(v) for k, v in criteria.items() if k in ("true", "false")}
        out[str(qid)] = item
    return {"state": body.get("state", ""), "questions": out}


def answer(scorer: UnslothScorer, body: dict, name: str) -> dict:
    """The Decision API response for a request body; `Refused` for malformed or over-capacity requests."""
    record = to_record(body)
    try:
        probabilities, _, tokens = scorer.score(record)
    except ContextOverflow as error:
        raise Refused(422, str(error)) from error
    return build_answer(record, probabilities, tokens, name)


def build_answer(record: dict, probabilities: dict, tokens: int, name: str) -> dict:
    """The response body from a scored record."""
    answers: dict[str, dict] = {}
    for qid, q in record["questions"].items():
        p = probabilities[qid]
        if q["type"] == "noul":
            answers[qid] = {"type": "noul", "noul": p["yes"]}
        else:
            answers[qid] = {"type": "choice", "choice": max(p, key=p.get), "probabilities": p}
    return {"model": name, "answers": answers, "usage": {"input_tokens": tokens}}
