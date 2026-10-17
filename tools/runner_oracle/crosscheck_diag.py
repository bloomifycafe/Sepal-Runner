"""Where do Unsloth's fp32 path and the vendored fp32 oracle differ? Run on a GPU box in the Unsloth env (PYTHONPATH = repo root, SEPAL_RUNS/SEPAL_BASE set).

    python tools/runner_oracle/crosscheck_diag.py case1 case2 ...

For each golden case: CPU golden (file) vs Fp32Oracle on CUDA (same code, other device) vs UnslothFp32 (FastDecisionModel, fp32, no autocast), hidden
and logits, plus the hidden scale (max |h|, rms) so differences can be read relative to it, plus the vendored head applied to Unsloth's hidden.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

from sepal_runner import oracle, oracle_golden as g


def stats(a, b, name):
    d = np.abs(a - b)
    return f"{name}: max {d.max():.3e} mean {d.mean():.3e} rel(max/scale) {d.max() / (np.abs(b).max() + 1e-12):.2e}"


def nofla(names):
    """Fp32Oracle on CUDA with `fla` blocked (transformers' reference PyTorch gated delta rule, no Triton tf32 dots) and TF32 off everywhere."""
    import torch

    sys.modules["fla"] = None
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    o = oracle.Fp32Oracle(device="cuda")
    for name in names:
        tensors, meta = g.read_case(g.GOLDEN_DIR / f"{name}.safetensors")
        record = json.loads(meta["record"])
        enc = o.encode(record["state"], record["questions"], max_length=65536)
        a = o.forward(enc)
        print(name, "cuda fp32, fla blocked, tf32 off vs cpu golden:", stats(a["hidden"], tensors["hidden_fp32"], "h"), stats(a["logits"], tensors["logits_fp32"], "l"), flush=True)


def main(names):
    import torch

    results = {}
    o = oracle.Fp32Oracle(device="cuda")
    u = oracle.UnslothFp32()
    for name in names:
        tensors, meta = g.read_case(g.GOLDEN_DIR / f"{name}.safetensors")
        record = json.loads(meta["record"])
        enc = o.encode(record["state"], record["questions"], max_length=65536)
        assert list(enc.input_ids) == tensors["input_ids"].tolist()
        gold_h, gold_l = tensors["hidden_fp32"], tensors["logits_fp32"]
        a = o.forward(enc)
        b = u.forward(enc)
        # vendored head over Unsloth's hidden states: separates the backbone difference from the head difference
        ids = torch.tensor([list(enc.input_ids)], device="cuda")
        hid = torch.from_numpy(b["hidden"]).cuda()[None]
        with torch.no_grad():
            per_q = o.head(hid, ids, torch.ones_like(ids), [enc], o.embedding.detach())[0]
        l_vh = torch.cat([z.float().reshape(-1) for z in per_q]).cpu().numpy()
        # Unsloth's head over the oracle's hidden states
        rows = {
            "hidden |h| max": float(np.abs(gold_h).max()), "hidden rms": float(np.sqrt((gold_h ** 2).mean())),
            "cuda-oracle vs cpu-golden hidden": stats(a["hidden"], gold_h, "h"), "cuda-oracle vs cpu-golden logits": stats(a["logits"], gold_l, "l"),
            "unsloth vs cpu-golden hidden": stats(b["hidden"], gold_h, "h"), "unsloth vs cpu-golden logits": stats(b["logits"], gold_l, "l"),
            "unsloth vs cuda-oracle hidden": stats(b["hidden"], a["hidden"], "h"), "unsloth vs cuda-oracle logits": stats(b["logits"], a["logits"], "l"),
            "vendored head on unsloth hidden vs unsloth logits": stats(l_vh, b["logits"], "l"),
            "vendored head on unsloth hidden vs cpu-golden logits": stats(l_vh, gold_l, "l"),
        }
        results[name] = rows
        print(name, json.dumps(rows, indent=1), flush=True)
    (g.GOLDEN_DIR / "_crosscheck_diag.json").write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    (nofla if os.environ.get("DIAG_NOFLA") else main)(sys.argv[1:] or ["choice_2", "sec_det1", "wl_t1k_q8"])
