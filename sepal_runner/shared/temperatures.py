"""The calibrated temperatures of a checkpoint (`final/sepal.json`): per question type and option-count bucket.

Copied from Sepal (Apache-2.0), commit f846e8d: sepal/eval/calibrate.py (TypeTemperatures, k_bucket, bucket_key; that file is adapted from
Kev, Apache-2.0, see NOTICE), sepal/eval/checkpoint.py (read_temperatures) and sepal/train/unsloth_arm.py (fitted_temperatures). Only
reading and lookup are here, not the fitting.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

META_FILE = "sepal.json"
FORMAT_VERSION = 1
K_BUCKETS = ((2, 2, "2"), (3, 5, "3-5"), (6, 12, "6-12"), (13, 10**9, "13+"))


def k_bucket(k: int) -> str:
    """The option-count bucket of a question with `k` options ("2", "3-5", "6-12", "13+")."""
    for low, high, name in K_BUCKETS:
        if low <= k <= high:
            return name
    raise ValueError(f"a question needs at least 2 options, got {k}")


def bucket_key(kind: str, k: int) -> str:
    """`"noul:2"`, `"score:3-5"`, `"choice:6-12"`: the key of a (type, option-count bucket)."""
    return f"{kind}:{k_bucket(k)}"


@dataclass(frozen=True)
class TypeTemperatures:
    """`single` for everything, `types` per question type, `buckets` per (type, option-count bucket). A question takes its bucket's
    temperature, else its type's, else the single one."""

    single: float
    types: Mapping[str, float]
    buckets: Mapping[str, float]
    type_rows: Mapping[str, int]
    bucket_rows: Mapping[str, int]
    n_rows: int
    shrink: float = 0.0

    def lookup(self, kind: str, k: int) -> tuple[float, str]:
        """(temperature, level) for a question of type `kind` with `k` options; level is `bucket`, `type` or `single`."""
        key = bucket_key(kind, k)
        if key in self.buckets:
            return self.buckets[key], "bucket"
        if kind in self.types:
            return self.types[kind], "type"
        return self.single, "single"

    def temperature_for(self, kind: str, k: int) -> float:
        return self.lookup(kind, k)[0]

    @classmethod
    def from_dict(cls, data: Mapping) -> "TypeTemperatures":
        if data.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"temperatures: format_version {data.get('format_version')!r} is not {FORMAT_VERSION}")
        return cls(single=float(data["single"]), types={k: float(v) for k, v in data["types"].items()},
                   buckets={k: float(v) for k, v in data["buckets"].items()}, type_rows=dict(data["type_rows"]),
                   bucket_rows=dict(data["bucket_rows"]), n_rows=int(data["n_rows"]), shrink=float(data.get("shrink", 0.0)))


def read_temperatures(checkpoint: str | Path) -> TypeTemperatures | None:
    """The per-type temperatures recorded in the checkpoint's sepal.json, or None when it has none."""
    path = Path(checkpoint) / META_FILE
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8")).get("temperatures")
    return None if data is None else TypeTemperatures.from_dict(data)


def fitted_temperatures(run: str | Path) -> TypeTemperatures:
    """The temperatures a finished run recorded in `<run>/final/sepal.json`: the per-type table, else its single temperature for every question."""
    final = Path(run) / "final" if (Path(run) / "final" / META_FILE).is_file() else Path(run)   # a run directory, or the checkpoint itself
    table = read_temperatures(final)
    if table is not None:
        return table
    meta = json.loads((final / META_FILE).read_text(encoding="utf-8"))
    return TypeTemperatures(single=float(meta.get("temperature", 1.0)), types={}, buckets={}, type_rows={}, bucket_rows={}, n_rows=0)
