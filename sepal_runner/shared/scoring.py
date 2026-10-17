"""From raw option logits to calibrated probabilities: temperature per question type and option-count bucket, then softmax.

Copied from Sepal (Apache-2.0, https://github.com/bloomifycafe/Sepal), commit f846e8d:
  sepal/train/unsloth_arm.py   (UnslothScorer: _encode, _finish, score, prefetch, answers, close; batch_plan; MAX_SCORE_LEN)
  sepal/eval/predictor.py      (ContextOverflow)
Nothing of Unsloth is imported, called or copied: the engine returns the raw head logits, and this module only applies the
temperature and the softmax in float64 and maps the option ids back to our option keys. The class keeps Sepal's name `UnslothScorer`
so code written against Sepal reads the same; `Scorer` is an alias.
"""
from __future__ import annotations

import concurrent.futures
import math
from collections.abc import Sequence
from typing import Any

from sepal_runner.shared import convert as conv

MAX_SCORE_LEN = 16384           # the serving window in tokens
SCORE_BATCH_TOKENS_1024 = 65536  # padded tokens of one scoring batch at most
SCORE_BATCH_RECORDS = 128        # records of one scoring batch at most
ENCODE_WORKERS = 4               # threads that tokenise the next chunk while the engine scores this one


class ContextOverflow(ValueError):
    """A request does not fit the context. `tokens` is the offending count and `limit` what it may be."""

    def __init__(self, message: str, tokens: int, limit: int):
        super().__init__(message)
        self.tokens, self.limit = tokens, limit


def batch_plan(lengths: Sequence[int], max_tokens: int, max_records: int) -> list[list[int]]:
    """Indices of `lengths` grouped into scoring batches: longest first, so a batch holds neighbours of one length and its padding
    is small. A batch holds `max_records` records at most and pads to its longest member: `len(batch) * longest <= max_tokens`,
    except a single record longer than that, which goes alone."""
    order = sorted(range(len(lengths)), key=lambda i: (-lengths[i], i))
    batches: list[list[int]] = []
    current: list[int] = []
    for i in order:
        if current and (len(current) + 1) * lengths[current[0]] > max_tokens or len(current) >= max_records:
            batches.append(current)
            current = []
        current.append(i)
    if current:
        batches.append(current)
    return batches


class UnslothScorer:
    """Scores records with a decision model behind `api` (`encode`, `logits`, optionally `logits_many`): probabilities and logits in our
    option keys. `temperatures` is `shared.temperatures.TypeTemperatures` or None (temperature 1)."""

    def __init__(self, api: Any, max_length: int = MAX_SCORE_LEN, name: str = "U1", temperatures=None,
                 batch_tokens: int = SCORE_BATCH_TOKENS_1024, batch_records: int = SCORE_BATCH_RECORDS, workers: int = ENCODE_WORKERS):
        self.api, self.max_length, self.name = api, max_length, name
        self.temperatures = temperatures
        self.batch_tokens, self.batch_records, self.workers = max(1, int(batch_tokens)), max(1, int(batch_records)), max(0, int(workers))
        self.truncated = 0
        self.unanswered: list[tuple[str, str]] = []
        self._pool: concurrent.futures.ThreadPoolExecutor | None = None
        self._encoded: dict[int, concurrent.futures.Future] = {}

    # -- one record --

    def _encode(self, record: dict):
        """The encoded record, or a ContextOverflow (returned, not raised, so a worker thread hands it back as a value)."""
        questions = conv.request_questions(record)
        state = conv.state_text(record["state"])         # the same text training reads (control strings neutralised)
        try:
            return self.api.encode(state, questions, self.max_length)
        except ValueError as error:
            return ContextOverflow(str(error), 0, self.max_length)

    def _finish(self, record: dict, encoded, rows) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]], int]:
        if len(encoded.input_ids) >= self.max_length:
            self.truncated += 1
        probabilities: dict[str, dict[str, float]] = {}
        logits: dict[str, dict[str, float]] = {}
        for question, row in zip(encoded.questions, rows):
            kind = record["questions"][question.question_id]["type"]
            ids = list(question.option_ids)
            z = [float(x) for x in row[:len(ids)]]
            scale = self.temperatures.temperature_for(kind, len(ids)) if self.temperatures is not None else 1.0
            top = max(z)
            expo = [math.exp((x - top) / scale) for x in z]
            total = sum(expo)
            probabilities[question.question_id] = conv.to_our_keys(kind, {i: e / total for i, e in zip(ids, expo)})
            logits[question.question_id] = conv.to_our_keys(kind, dict(zip(ids, z)))
        return probabilities, logits, len(encoded.input_ids)

    def score(self, record: dict) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]], int]:
        """(probabilities, logits, input tokens), each {qid: {our option key: value}}. The logits are raw. The probabilities are at
        temperature 1, or at the temperature of each question's type and option-count bucket when `temperatures` is set."""
        encoded = self._encode(record)
        if isinstance(encoded, ContextOverflow):
            raise encoded
        return self._finish(record, encoded, self.api.logits(encoded))

    # -- many records --

    def _executor(self) -> concurrent.futures.ThreadPoolExecutor | None:
        if self.workers and self._pool is None:
            self._pool = concurrent.futures.ThreadPoolExecutor(self.workers, thread_name_prefix="sepal-encode")
        return self._pool

    def prefetch(self, records: Sequence[dict]) -> None:
        """Start tokenising `records` on the worker threads; `answers` for the same records picks the results up. Optional."""
        pool = self._executor()
        if pool is not None:
            for record in records:
                self._encoded.setdefault(id(record), pool.submit(self._encode, record))

    def _encode_all(self, records: Sequence[dict]) -> list:
        self.prefetch(records)
        out = []
        for record in records:
            future = self._encoded.pop(id(record), None)
            out.append(future.result() if future is not None else self._encode(record))
        return out

    def answers(self, records: Sequence[dict], log=None) -> tuple[dict, dict, int]:
        """(probabilities, logits, tokens) keyed by `_meta.id`. A record that does not fit has no entry and is listed in `unanswered`."""
        encoded = self._encode_all(records)
        results: dict[int, tuple] = {}
        ready: list[int] = []
        for index, (record, enc) in enumerate(zip(records, encoded)):
            if isinstance(enc, ContextOverflow):
                self.unanswered.append((record["_meta"]["id"], str(enc)))
            else:
                ready.append(index)
        many = getattr(self.api, "logits_many", None)
        if many is None:
            batches = [[i] for i in ready]
        else:
            batches = [[ready[j] for j in b] for b in batch_plan([len(encoded[i].input_ids) for i in ready], self.batch_tokens, self.batch_records)]
        done = 0
        for batch in batches:
            rows_of = many([encoded[i] for i in batch]) if many is not None else [self.api.logits(encoded[batch[0]])]
            for i, rows in zip(batch, rows_of):
                results[i] = (rows,)
            done += len(batch)
            if log and done // 100 != (done - len(batch)) // 100:
                log(f"  {self.name}: {done}/{len(records)} records")
        probabilities, logits, tokens = {}, {}, 0
        for index in ready:
            record = records[index]
            rid = record["_meta"]["id"]
            p, z, n = self._finish(record, encoded[index], results[index][0])
            probabilities[rid], logits[rid], tokens = p, z, tokens + n
        return probabilities, logits, tokens

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool, self._encoded = None, {}


Scorer = UnslothScorer
