"""Python side of the runner's C ABI (`runner/include/sepal/c_api.h`): flat-array marshalling, a ctypes binding, and `RunnerApi`, a
drop-in for `sepal.train.unsloth_arm._RealApi` (encode, logits, logits_many, hidden_size, training), so `UnslothScorer`, `PanelScorer`
and `sepal.serve.decision_api.answer` run unchanged with the runner as the backend. The engine returns raw logits; temperatures and the
softmax stay in `UnslothScorer` (`calibrated=True` reads `final/sepal.json`).

    CEngine      a loaded library (explicit path or $SEPAL_RUNNER_LIB; nothing is searched for or downloaded). ctypes releases the GIL for
                 the length of `sepal_score`, so tokenising and HTTP threads run while the engine works.
    pack         records -> the flat arrays of `sepal_score`; `unpack` -> per record, per question, the raw logits
    RunnerApi    the `_RealApi` interface over any backend with `logits_many`
"""
from __future__ import annotations

import ctypes
import json
import os
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sepal_runner.encode import DEFAULT_MAX_LENGTH, Encoder

API_VERSION = 1
LIB_ENV = "SEPAL_RUNNER_LIB"
TIMES = ("trunk_ms", "head_ms", "total_ms", "tokens", "tokens_reused")


class EngineError(RuntimeError):
    """`sepal_*` returned non-zero; the message is the engine's `sepal_last_error()`."""


@dataclass
class Batch:
    """The arguments of one `sepal_score` call, as contiguous numpy arrays of the dtypes the header names."""

    n_requests: int
    token_ids: np.ndarray            # u32 [total tokens]
    token_offsets: np.ndarray        # u64 [n+1]
    question_offsets: np.ndarray     # u64 [n+1]
    question_types: np.ndarray       # u32 [n_questions]
    instruction_spans: np.ndarray    # u32 [2 n_questions]
    option_offsets: np.ndarray       # u64 [n_questions+1]
    option_spans: np.ndarray         # u32 [2 n_options]

    @property
    def n_questions(self) -> int:
        return len(self.question_types)

    @property
    def n_options(self) -> int:
        return int(self.option_offsets[-1])

    @property
    def n_tokens(self) -> int:
        return len(self.token_ids)


def pack(records: Sequence[Any]) -> Batch:
    """Flat arrays for encoded records (`input_ids`, `questions` with `question_type`, `question_span`, `option_spans`). Spans are
    relative to a request's own first token, which is what the encoder produces."""
    tok_off, q_off, o_off = [0], [0], [0]
    types: list[int] = []
    ispans: list[int] = []
    ospans: list[int] = []
    id_parts = []
    for record in records:
        id_parts.append(np.asarray(record.input_ids, dtype=np.uint32))
        tok_off.append(tok_off[-1] + len(record.input_ids))
        for q in record.questions:
            types.append(q.question_type)
            ispans.extend(q.question_span)
            for span in q.option_spans:
                ospans.extend(span)
            o_off.append(o_off[-1] + len(q.option_spans))
        q_off.append(q_off[-1] + len(record.questions))
    return Batch(
        n_requests=len(records),
        token_ids=np.concatenate(id_parts) if id_parts else np.zeros(0, np.uint32),
        token_offsets=np.asarray(tok_off, np.uint64), question_offsets=np.asarray(q_off, np.uint64),
        question_types=np.asarray(types, np.uint32), instruction_spans=np.asarray(ispans, np.uint32),
        option_offsets=np.asarray(o_off, np.uint64), option_spans=np.asarray(ospans, np.uint32))


def unpack(batch: Batch, logits: np.ndarray) -> list[list[list[float]]]:
    """Per request, per question, the raw option logits as Python floats (`UnslothScorer._finish` takes exactly this)."""
    values = np.asarray(logits, dtype=np.float32).astype(np.float64).tolist()
    out = []
    qo, oo = batch.question_offsets, batch.option_offsets
    for r in range(batch.n_requests):
        out.append([values[int(oo[q]):int(oo[q + 1])] for q in range(int(qo[r]), int(qo[r + 1]))])
    return out


def validate(batch: Batch, max_tokens: int | None = None) -> None:
    """Cheap structural checks before the library sees the arrays (a bad span would be an out-of-range read in C)."""
    n = batch.n_requests
    if len(batch.token_offsets) != n + 1 or len(batch.question_offsets) != n + 1:
        raise ValueError("offset arrays must have n_requests + 1 entries")
    spans = np.concatenate([batch.instruction_spans, batch.option_spans]).reshape(-1, 2).astype(np.int64)
    if len(spans) and ((spans[:, 0] > spans[:, 1]).any()):
        raise ValueError("a span begins after it ends")
    for r in range(n):
        length = int(batch.token_offsets[r + 1] - batch.token_offsets[r])
        if max_tokens is not None and length > max_tokens:
            raise ValueError(f"request {r} has {length} tokens; the engine takes {max_tokens}")
        q0, q1 = int(batch.question_offsets[r]), int(batch.question_offsets[r + 1])
        if q0 == q1:
            raise ValueError(f"request {r} has no questions")
        o0, o1 = int(batch.option_offsets[q0]), int(batch.option_offsets[q1])
        mine = np.concatenate([batch.instruction_spans[2 * q0:2 * q1], batch.option_spans[2 * o0:2 * o1]])
        if len(mine) and int(mine.max()) > length:
            raise ValueError(f"request {r}: a span ends past its {length} tokens")


def _library_path(lib: str | os.PathLike | None) -> Path:
    path = lib or os.environ.get(LIB_ENV)
    if not path:
        raise EngineError(f"no runner library: pass the path of sepal_runner_*.dll/.so or set ${LIB_ENV}")
    path = Path(path)
    if not path.is_file():
        raise EngineError(f"runner library not found: {path}")
    return path


class CEngine:
    """The C ABI through ctypes. `lib` is a path (default `$SEPAL_RUNNER_LIB`) or an object with the `sepal_*` functions (tests)."""

    def __init__(self, options: dict, lib: Any = None):
        if lib is None or isinstance(lib, (str, os.PathLike)):
            path = _library_path(lib)
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(str(path.parent.resolve()))     # the CUDA runtime DLLs sit next to the library
            lib = ctypes.CDLL(str(path))
            self._declare(lib)
        self.lib = lib
        self._handle = ctypes.c_void_p()
        version = self.lib.sepal_api_version()
        if version != API_VERSION:
            raise EngineError(f"library speaks C API {version}; this binding speaks {API_VERSION}")
        status = self.lib.sepal_open(json.dumps(options).encode("utf-8"), ctypes.byref(self._handle))
        if status != 0:
            raise EngineError(self._error("sepal_open"))
        self.options = dict(options)
        self.info = self._info()
        self.max_tokens = int(self.info.get("max_tokens") or options.get("max_tokens") or DEFAULT_MAX_LENGTH)
        self.hidden_size = int(self.info.get("hidden_size") or 1024)
        self.last_times: dict[str, float] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _declare(lib: Any) -> None:
        p = ctypes.POINTER
        lib.sepal_api_version.restype = ctypes.c_int32
        lib.sepal_last_error.restype = ctypes.c_char_p
        lib.sepal_open.argtypes = [ctypes.c_char_p, p(ctypes.c_void_p)]
        lib.sepal_open.restype = ctypes.c_int32
        lib.sepal_close.argtypes = [ctypes.c_void_p]
        lib.sepal_close.restype = None
        lib.sepal_info.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint64, p(ctypes.c_uint64)]
        lib.sepal_info.restype = ctypes.c_int32
        lib.sepal_score.argtypes = [ctypes.c_void_p, ctypes.c_uint64, p(ctypes.c_uint32), p(ctypes.c_uint64), p(ctypes.c_uint64),
                                    p(ctypes.c_uint32), p(ctypes.c_uint32), p(ctypes.c_uint64), p(ctypes.c_uint32),
                                    p(ctypes.c_float), p(ctypes.c_double)]
        lib.sepal_score.restype = ctypes.c_int32

    def _error(self, what: str) -> str:
        message = self.lib.sepal_last_error()
        return f"{what} failed: {message.decode('utf-8', 'replace') if message else 'no message'}"

    def _info(self) -> dict:
        needed = ctypes.c_uint64(0)
        buffer = ctypes.create_string_buffer(4096)
        if self.lib.sepal_info(self._handle, buffer, 4096, ctypes.byref(needed)) != 0:
            raise EngineError(self._error("sepal_info"))
        if needed.value > 4096:
            buffer = ctypes.create_string_buffer(int(needed.value))
            if self.lib.sepal_info(self._handle, buffer, needed.value, ctypes.byref(needed)) != 0:
                raise EngineError(self._error("sepal_info"))
        return json.loads(buffer.value.decode("utf-8"))

    def score(self, batch: Batch) -> np.ndarray:
        """Raw logits float32 [n_options] for the batch (one `sepal_score`; the GIL is released during the call)."""
        validate(batch, self.max_tokens)
        out = np.empty(batch.n_options, dtype=np.float32)
        times = np.zeros(5, dtype=np.float64)

        def ptr(array: np.ndarray, kind):
            return array.ctypes.data_as(ctypes.POINTER(kind))

        status = self.lib.sepal_score(
            self._handle, batch.n_requests, ptr(batch.token_ids, ctypes.c_uint32), ptr(batch.token_offsets, ctypes.c_uint64),
            ptr(batch.question_offsets, ctypes.c_uint64), ptr(batch.question_types, ctypes.c_uint32),
            ptr(batch.instruction_spans, ctypes.c_uint32), ptr(batch.option_offsets, ctypes.c_uint64),
            ptr(batch.option_spans, ctypes.c_uint32), ptr(out, ctypes.c_float), ptr(times, ctypes.c_double))
        if status != 0:
            raise EngineError(self._error("sepal_score"))
        self.last_times = dict(zip(TIMES, times.tolist()))
        return out

    def close(self) -> None:
        with self._lock:
            if self._handle:
                self.lib.sepal_close(self._handle)
                self._handle = ctypes.c_void_p()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class EngineBackend:
    """`logits_many` straight on an engine (serialised: one scorer thread at a time)."""

    def __init__(self, engine: Any):
        self.engine, self._lock = engine, threading.Lock()

    def logits_many(self, records: Sequence[Any]) -> list[list[list[float]]]:
        batch = pack(records)
        with self._lock:
            logits = self.engine.score(batch)
        return unpack(batch, logits)


class RunnerApi:
    """The interface `UnslothScorer` takes (`sepal.train.unsloth_arm.DecisionApi` + `logits_many`), served by the runner."""

    training = False

    def __init__(self, encoder: Encoder, backend: Any, hidden_size: int = 1024):
        self.encoder, self.backend, self.hidden_size = encoder, backend, hidden_size
        self.tokenizer = encoder.tokenizer

    def encode(self, state, questions, max_length):
        return self.encoder.encode({"state": state, "questions": questions}, max_length=max_length)

    def logits(self, encoded):
        return self.logits_many([encoded])[0]

    def logits_many(self, encoded_records):
        return self.backend.logits_many(list(encoded_records))


def open_runner_api(final: str | os.PathLike, base: str | os.PathLike, lib: str | os.PathLike | None, profile: str,
                    extra: dict | None = None, max_tokens: int = DEFAULT_MAX_LENGTH) -> tuple[RunnerApi, CEngine]:
    """Open the library on a checkpoint and tokenise with that checkpoint's `tokenizer.json`."""
    options = {"final_dir": str(final), "base_dir": str(base), "profile": profile, "max_tokens": max_tokens, **(extra or {})}
    if lib == "mock":                    # the Python front end alone over a fake engine: for load tests and dry runs
        from sepal_runner.mock import FakeLib, MockEngine

        lib = FakeLib(MockEngine(max_tokens=max_tokens))
    engine = CEngine(options, lib)
    encoder = Encoder.from_file(Path(final) / "tokenizer.json")
    return RunnerApi(encoder, EngineBackend(engine), engine.hidden_size), engine
