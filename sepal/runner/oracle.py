"""The oracle of the fast runner: the trained Sepal model (Qwen3.5-0.8B-Base + LoRA + Clef head) evaluated in plain PyTorch.

Two ways to run it, both from the Unsloth environment (`runs/pilot-unsloth/venv`, Unsloth 2026.10.4 / transformers 5.17.0):

* `Fp32Oracle` (CPU or GPU, no Unsloth import): fp32 weights, the LoRA applied as a PEFT adapter, NO autocast, the head in fp32 (its
  bf16 weights upcast). It uses Unsloth's vendored Clef files (`unsloth/_vendor/clef/joint_schema_model.py`, the Apache-2.0 reference
  encoder and head) loaded by path, because `import unsloth` refuses to start without a GPU. This is the golden `fp32` variant.
* `ServedOracle` (GPU): the model exactly as `sepal.serve.decision_api` serves it: `sepal.train.unsloth_arm.load_scorer(calibrated=True)`,
  its own autocast, the encoder `_RealApi.encode` (Unsloth's `clef.encode_record`), `NoTruncateApi` on top. This is the `bf16` variant.

`conv.state_text` / `conv.request_questions` (sepal.train.unsloth_convert) turn a corpus record or a `/v1/systemone` body into
what the encoder takes, exactly as `UnslothScorer._encode` does.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

RUNS = Path(os.environ.get("SEPAL_RUNS", r"E:\MYCOMPANY\BlumSOC\Sepal\runs"))      # on a rented box: a private runs dir (SEPAL_RUNS)
FINAL = RUNS / "box-results" / "full08b" / "U2-s0" / "final"
BASE = Path(os.environ.get("SEPAL_BASE", r"C:\Users\FSNaix\.cache\huggingface\hub\models--Qwen--Qwen3.5-0.8B-Base\snapshots\dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"))
STAGE = RUNS / "runner-golden" / "_stage"   # tools/runner_oracle/stage_model.py: final/ with the local base path
MAX_SCORE_LEN = 16384


def hide_gpu() -> None:
    """Call before importing torch/transformers to run on the CPU only. Also blocks `fla` (flash-linear-attention): its Triton kernels
    cannot run without a GPU, and transformers then uses its reference PyTorch gated delta rule."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    sys.modules["fla"] = None


def load_vendored_clef():
    """Unsloth's vendored Clef reference (encoder, head), imported by file path so that `import unsloth` is not needed."""
    if "clef_vendor" in sys.modules:
        return sys.modules["clef_vendor"]
    spec = importlib.util.find_spec("unsloth")
    if spec is None or not spec.origin:
        raise RuntimeError("unsloth is not installed in this environment")
    path = Path(spec.origin).parent / "_vendor" / "clef" / "joint_schema_model.py"
    mod_spec = importlib.util.spec_from_file_location("clef_vendor", path)
    module = importlib.util.module_from_spec(mod_spec)
    sys.modules["clef_vendor"] = module
    mod_spec.loader.exec_module(module)
    return module


def record_for(state_text: str, questions: dict) -> dict:
    return {"state": state_text, "questions": questions}


class Fp32Oracle:
    """fp32, LoRA applied (unmerged PEFT adapter), no autocast, fp32 head. Runs on `device` ("cpu" or "cuda")."""

    def __init__(self, final_dir: Path = STAGE / "final", base_dir: Path = BASE, device: str = "cpu"):
        import torch
        import peft.tuners.lora.torchao as peft_torchao
        from peft import PeftModel
        from safetensors.torch import load_file
        from transformers import AutoModelForImageTextToText, AutoTokenizer

        peft_torchao.is_torchao_available = lambda: False        # the venv's torchao 0.13 is older than PEFT wants; no quantised layers here
        self.torch, self.device = torch, torch.device(device)
        self.clef = load_vendored_clef()
        self.tokenizer = AutoTokenizer.from_pretrained(str(final_dir), local_files_only=True)
        backbone = AutoModelForImageTextToText.from_pretrained(str(base_dir), dtype=torch.float32, local_files_only=True)
        self.model = PeftModel.from_pretrained(backbone, str(final_dir), is_trainable=False)
        self.model.eval().to(self.device)
        head_config = json.loads((Path(final_dir) / "joint_head_config.json").read_text(encoding="utf-8"))
        self.head = self.clef.JointSchemaHead(**head_config)
        self.head.load_state_dict(load_file(str(Path(final_dir) / "joint_head.safetensors")), strict=True)   # bf16 -> fp32 copy
        self.head.float().eval().to(self.device)
        base = self.model.get_base_model()
        self.text_model = base.model.language_model
        self.embedding = base.get_output_embeddings().weight
        self.layers = list(self.text_model.layers)

    def encode(self, state: str, questions: dict, max_length: int = MAX_SCORE_LEN):
        return self.clef.encode_record(self.tokenizer, record_for(state, questions), max_length=max_length)

    def forward(self, encoded, layers: bool = False) -> dict:
        """{"hidden": [L,1024] fp32 numpy (after the final norm), "logits": [P] fp32 numpy (questions in order, options in the
        encoder's order), "layers": [n_layers, L, 1024] numpy or None (output of every decoder layer, before the final norm)}."""
        torch = self.torch
        captured: list = []
        hooks = []
        if layers:
            for layer in self.layers:
                hooks.append(layer.register_forward_hook(
                    lambda module, args, out: captured.append((out[0] if isinstance(out, tuple) else out).detach().float().cpu())))
        try:
            ids = torch.tensor([list(encoded.input_ids)], dtype=torch.long, device=self.device)
            mask = torch.ones_like(ids)
            with torch.no_grad():
                # One unpadded sequence: no attention mask, so SDPA takes its causal (flash-style) path instead of materialising an
                # L x L mask (8.6 GB of scores at 16k tokens on the CPU). Same maths as an all-ones mask.
                hidden = self.text_model(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state
                per_question = self.head(hidden, ids, mask, [encoded], self.embedding.detach())[0]
        finally:
            for h in hooks:
                h.remove()
        flat = torch.cat([z.float().reshape(-1) for z in per_question]).cpu().numpy()
        out = {"hidden": hidden[0].float().cpu().numpy(), "logits": flat, "layers": None}
        if layers:
            out["layers"] = torch.stack([c[0] for c in captured]).numpy()
        return out


class ServedOracle:
    """The oracle as decision_api serves it (GPU). `scorer.score(record)` gives the calibrated probabilities; `raw_logits` the raw rows."""

    def __init__(self, run_dir: Path = STAGE, max_len: int = MAX_SCORE_LEN, calibrated: bool = True, batch_tokens: int = 0,
                 batch_records: int = 128, workers: int = 4):
        from sepal.serve.decision_api import NoTruncateApi
        from sepal.train.unsloth_arm import load_scorer

        self.scorer = load_scorer(str(run_dir), max_len, calibrated=calibrated, batch_tokens=batch_tokens,
                                  batch_records=batch_records, workers=workers)
        self.api_raw = self.scorer.api
        self.scorer.api = NoTruncateApi(self.api_raw, max_len)
        self.max_len = max_len

    def encode(self, state: str, questions: dict):
        return self.scorer.api.encode(state, questions, self.max_len)

    def raw_logits(self, encoded) -> list[float]:
        """Flat raw logits of one record, as the served path computes them (autocast bf16, one record at a time)."""
        rows = self.api_raw.logits(encoded)
        return [float(x) for row, q in zip(rows, encoded.questions) for x in row[:len(q.option_ids)]]


class UnslothFp32:
    """Unsloth's own `FastDecisionModel` on the GPU with fp32 weights and NO autocast (the cross-check of `Fp32Oracle`): Unsloth's fused
    Clef head (`clef.batched_logits`) over fp32 hidden states. `forward` returns the same dict as `Fp32Oracle.forward` (no layers)."""

    def __init__(self, run_dir: Path = STAGE, max_len: int = MAX_SCORE_LEN):
        import torch
        from unsloth import FastDecisionModel

        from sepal.train.unsloth_arm import _scrub_env

        _scrub_env()
        self.torch = torch
        self.model, self.tokenizer = FastDecisionModel.from_pretrained(str(Path(run_dir) / "final"), max_seq_length=max_len,
                                                                      dtype=torch.float32, local_files_only=True)
        FastDecisionModel.for_inference(self.model)
        self.tokenizer = getattr(self.tokenizer, "tokenizer", self.tokenizer)
        self.device = next(self.model.parameters()).device
        self.dtypes = sorted({str(p.dtype) for p in self.model.parameters()})

    def forward(self, encoded) -> dict:
        import numpy as np

        torch = self.torch
        backbone = self.model._backbone()
        text_model = getattr(backbone, "model", None) or backbone.base_model
        text_model = getattr(text_model, "language_model", text_model)
        box: list = []
        hook = text_model.register_forward_hook(lambda module, args, out: box.append(out.last_hidden_state.detach().float().cpu()))
        try:
            ids = torch.tensor([list(encoded.input_ids)], dtype=torch.long, device=self.device)
            mask = torch.ones_like(ids)
            with torch.no_grad():                       # no autocast: fp32 throughout
                logits, _ = self.model(input_ids=ids, attention_mask=mask, records=[encoded])
        finally:
            hook.remove()
        flat = []
        for row, q in zip(logits.float().cpu().tolist(), encoded.questions):
            flat.extend(row[:len(q.option_ids)])
        return {"hidden": box[0][0].numpy(), "logits": np.asarray(flat, dtype=np.float32), "layers": None}


def flatten_options(encoded) -> tuple[list[str], list[list[str]], list[int], list[tuple[int, int]]]:
    """(question ids, option ids per question, option_offsets [Q+1], option_spans [P] absolute half-open)."""
    qids, options, offsets, spans = [], [], [0], []
    for q in encoded.questions:
        qids.append(q.question_id)
        options.append(list(q.option_ids))
        spans.extend((int(s), int(e)) for s, e in q.option_spans)
        offsets.append(len(spans))
    return qids, options, offsets, spans
