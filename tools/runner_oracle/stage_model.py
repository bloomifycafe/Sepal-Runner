"""Copy the trained model's final/ dir to a staging dir with the box base path rewritten to the local base snapshot.

    python tools/runner_oracle/stage_model.py [--final DIR] [--out DIR]

The adapter and the unsloth decision config name a box path (/workspace/...) for the base model; the oracle needs the local one.
Nothing else is changed (weights, head, sepal.json are byte copies). The staged dir is the `final/` of a run dir (`<out>/final`).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = Path(os.environ.get("SEPAL_RUNS", "runs"))
FINAL = RUNS / "box-results" / "full08b" / "U2-s0" / "final"
BASE = Path(os.environ.get("SEPAL_BASE", r"C:\Users\FSNaix\.cache\huggingface\hub\models--Qwen--Qwen3.5-0.8B-Base\snapshots\dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"))
OUT = RUNS / "runner-golden" / "_stage"


def stage(final: Path = FINAL, base: Path = BASE, out: Path = OUT) -> Path:
    dest = out / "final"
    dest.mkdir(parents=True, exist_ok=True)
    for f in final.iterdir():
        target = dest / f.name
        if not target.exists() or target.stat().st_size != f.stat().st_size:
            shutil.copy2(f, target)
    for name, key in (("adapter_config.json", "base_model_name_or_path"), ("unsloth_decision_config.json", "base_model")):
        data = json.loads((final / name).read_text(encoding="utf-8"))
        data[key] = str(base)
        (dest / name).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--final", default=str(FINAL))
    p.add_argument("--out", default=str(OUT))
    a = p.parse_args()
    print(stage(Path(a.final), BASE, Path(a.out)))
