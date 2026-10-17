"""A deterministic stand-in for the engine, for tests and for load-testing the Python front end without a DLL.

`MockEngine.score(batch)` returns logits that depend only on the request's own tokens and spans (never on its batch neighbours), like the
real engine. `FakeLib` has the `sepal_*` functions of the C ABI with ctypes-pointer arguments, so `CEngine(options, lib=FakeLib())` runs
the real marshalling code (argument order, dtypes, offsets) without a shared library.
"""
from __future__ import annotations

import ctypes
import json
import threading
import time

import numpy as np

from sepal_runner.engine import Batch


class MockEngine:
    def __init__(self, delay_s: float = 0.0, per_token_s: float = 0.0, max_tokens: int = 16384, hidden_size: int = 1024):
        self.delay_s, self.per_token_s = delay_s, per_token_s
        self.max_tokens, self.hidden_size = max_tokens, hidden_size
        self.calls: list[int] = []          # n_requests of every score call
        self.token_calls: list[int] = []
        self.in_flight = 0
        self.max_in_flight = 0
        self._lock = threading.Lock()

    def score(self, batch: Batch) -> np.ndarray:
        with self._lock:
            self.calls.append(batch.n_requests)
            self.token_calls.append(batch.n_tokens)
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            if self.delay_s or self.per_token_s:
                time.sleep(self.delay_s + self.per_token_s * batch.n_tokens)
            return self.logits(batch)
        finally:
            with self._lock:
                self.in_flight -= 1

    @staticmethod
    def logits(batch: Batch) -> np.ndarray:
        out = np.empty(batch.n_options, dtype=np.float32)
        qo, oo = batch.question_offsets, batch.option_offsets
        for r in range(batch.n_requests):
            ids = batch.token_ids[int(batch.token_offsets[r]):int(batch.token_offsets[r + 1])].astype(np.int64)
            context = int(ids.sum()) % 997          # the whole request (state included) changes every logit, like attention would
            for q in range(int(qo[r]), int(qo[r + 1])):
                kind = int(batch.question_types[q])
                for o in range(int(oo[q]), int(oo[q + 1])):
                    begin, end = int(batch.option_spans[2 * o]), int(batch.option_spans[2 * o + 1])
                    h = (int(ids[begin:end].sum()) * 2654435761 + context * 31 + kind * 7919 + (o - int(oo[q])) * 104729) % 4001
                    out[o] = h / 1000.0 - 2.0
        return out


def _array(pointer, kind, count: int) -> np.ndarray:
    return np.ctypeslib.as_array(ctypes.cast(pointer, ctypes.POINTER(kind)), shape=(count,)) if count else np.zeros(0)


class FakeLib:
    """`sepal_*` as Python callables taking what ctypes would pass (pointers); backed by a `MockEngine`."""

    def __init__(self, engine: MockEngine | None = None, fail_open: bool = False, version: int = 1):
        self.engine, self.fail_open, self.version = engine or MockEngine(), fail_open, version
        self.opened = None
        self.closed = 0
        self.error = b""

    def sepal_api_version(self):
        return self.version

    def sepal_last_error(self):
        return self.error

    def sepal_open(self, options_json, out):
        if self.fail_open:
            self.error = b"open failed on purpose"
            return 1
        self.opened = json.loads(options_json.decode("utf-8"))
        out._obj.value = 1                  # `out` is ctypes.byref(c_void_p)
        return 0

    def sepal_close(self, handle):
        self.closed += 1

    def sepal_info(self, handle, buffer, capacity, needed):
        text = json.dumps({"profile": (self.opened or {}).get("profile", "mock"), "max_tokens": self.engine.max_tokens,
                           "hidden_size": self.engine.hidden_size, "backend": "mock"}).encode("utf-8")
        needed._obj.value = len(text) + 1
        if capacity >= len(text) + 1:
            ctypes.memmove(buffer, text + b"\0", len(text) + 1)
        return 0

    def sepal_score(self, handle, n, token_ids, token_offsets, question_offsets, question_types, instruction_spans, option_offsets,
                    option_spans, logits_out, times_out):
        u32, u64 = ctypes.c_uint32, ctypes.c_uint64
        n = int(n)
        tok_off = _array(token_offsets, u64, n + 1)
        q_off = _array(question_offsets, u64, n + 1)
        nq = int(q_off[-1])
        o_off = _array(option_offsets, u64, nq + 1)
        no = int(o_off[-1])
        batch = Batch(n, _array(token_ids, u32, int(tok_off[-1])), tok_off, q_off, _array(question_types, u32, nq),
                      _array(instruction_spans, u32, 2 * nq), o_off, _array(option_spans, u32, 2 * no))
        out = self.engine.score(batch)
        ctypes.memmove(logits_out, out.ctypes.data, out.nbytes)
        if times_out:
            for i, value in enumerate((0.0, 0.0, 1.0, float(batch.n_tokens), 0.0)):
                times_out[i] = value
        return 0
