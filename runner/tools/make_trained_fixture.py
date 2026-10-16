"""Write the tiny "trained model" fixture that the `trained` and `clef` C++ suites read.

    <oracle venv>/Scripts/python.exe runner/tools/make_trained_fixture.py   (needs torch, numpy, safetensors)

Output, in runner/tests/fixtures/tiny_trained/ (about 100 kB, committed). The base is the existing tiny_checkpoint.
  adapter_config.json, adapter_model.safetensors   a LoRA r=4, alpha=8 (scale 2) on a few modules, float32, PEFT names
  joint_head_config.json, joint_head.safetensors   a small Clef head (hidden 24, width 16, 2 heads), float32, random
  sepal.json                                       placeholder run metadata
  expected_merged.safetensors                      NumPy merge W' = fp32(W) + fp32(scale) * fp32(B @ A with a float64
                                                   product) for every adapted module ("f32.<name>"), and the full
                                                   float64 merge ("f64.<name>")
  head_case.safetensors                            one request in the golden-set layout (input_ids, question_types,
                                                   instruction_spans, option_offsets, option_spans, hidden_fp32) and
                                                   logits_fp32 from the Clef JointSchemaHead run by PyTorch in float64
                                                   on the fixture's head and the base embedding

The expected head logits need the Clef head class; by default the copy that Unsloth vendors in the oracle environment
is loaded from its file and run (read and executed as the oracle; nothing of it is copied here).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "runner" / "tests" / "fixtures" / "tiny_checkpoint"
OUT = REPO / "runner" / "tests" / "fixtures" / "tiny_trained"
DEFAULT_CLEF = Path("E:/MYCOMPANY/BlumSOC/Sepal/runs/pilot-unsloth/venv/Lib/site-packages/unsloth/_vendor/clef/joint_schema_model.py")
PREFIX = "model.language_model."
RANK, ALPHA = 4, 8.0
MODULES = [
    "layers.0.mlp.down_proj",
    "layers.0.linear_attn.in_proj_a",
    "layers.1.linear_attn.in_proj_qkv",
    "layers.3.self_attn.q_proj",
    "layers.3.self_attn.o_proj",
    "layers.7.mlp.gate_proj",
]
HEAD = dict(hidden_size=24, width=16, routing_layers=1, layers=2, heads=2, feedforward=32)


def read_safetensors(path: Path) -> dict[str, np.ndarray]:
    raw = path.read_bytes()
    n = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8 : 8 + n])
    out = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        a, b = meta["data_offsets"]
        data = raw[8 + n + a : 8 + n + b]
        if meta["dtype"] == "BF16":
            bits = np.frombuffer(data, dtype=np.uint16).astype(np.uint32) << 16
            out[name] = bits.view(np.float32).reshape(meta["shape"])
        elif meta["dtype"] == "F32":
            out[name] = np.frombuffer(data, dtype=np.float32).reshape(meta["shape"])
    return out


def base_tensors() -> dict[str, np.ndarray]:
    tensors: dict[str, np.ndarray] = {}
    for shard in sorted(BASE.glob("model-*.safetensors")):
        tensors.update(read_safetensors(shard))
    return {k[len(PREFIX) :]: v for k, v in tensors.items() if k.startswith(PREFIX)}


def load_clef(path: Path):
    spec = importlib.util.spec_from_file_location("clef_oracle", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["clef_oracle"] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clef-module", type=Path, default=DEFAULT_CLEF)
    args = parser.parse_args()
    rng = np.random.default_rng(20261016)
    OUT.mkdir(parents=True, exist_ok=True)
    base = base_tensors()

    # --- adapter and the independent NumPy merge ---
    adapter: dict[str, np.ndarray] = {}
    merged: dict[str, np.ndarray] = {}
    scale = ALPHA / RANK
    for module in MODULES:
        w = base[module + ".weight"]
        out_dim, in_dim = w.shape
        a = (rng.standard_normal((RANK, in_dim)) * 0.2).astype(np.float32)
        b = (rng.standard_normal((out_dim, RANK)) * 0.2).astype(np.float32)
        key = f"base_model.model.{PREFIX}{module}"
        adapter[key + ".lora_A.weight"] = a
        adapter[key + ".lora_B.weight"] = b
        product = b.astype(np.float64) @ a.astype(np.float64)
        merged["f32." + module + ".weight"] = (w + np.float32(scale) * product.astype(np.float32)).astype(np.float32)
        merged["f64." + module + ".weight"] = w.astype(np.float64) + scale * product
    save_file(adapter, str(OUT / "adapter_model.safetensors"), metadata={"format": "pt"})
    save_file(merged, str(OUT / "expected_merged.safetensors"))
    (OUT / "adapter_config.json").write_text(
        json.dumps(
            {
                "alpha_pattern": {}, "bias": "none", "fan_in_fan_out": False, "lora_alpha": ALPHA, "lora_bias": False,
                "lora_dropout": 0.0, "modules_to_save": None, "peft_type": "LORA", "r": RANK, "rank_pattern": {},
                "use_dora": False, "use_rslora": False,
                "base_model_name_or_path": "/a/box/path/that/is/ignored",
            },
            indent=2,
        )
    )

    # --- head ---
    clef = load_clef(args.clef_module)
    torch.manual_seed(7)
    head = clef.JointSchemaHead(**HEAD)
    with torch.no_grad():
        for name, p in head.named_parameters():
            if p.ndim == 0:
                p.fill_({"prior_logit_scale": 1.5, "joint_logit_scale": 2.0, "residual_gate": 0.4}[name])
            elif "norm" in name and name.endswith(".weight"):
                p.copy_(1.0 + 0.1 * torch.randn_like(p))
            elif name.endswith(".bias"):
                p.copy_(0.1 * torch.randn_like(p))
            else:
                p.copy_(torch.randn_like(p) / np.sqrt(p.shape[-1]))
    state = {k: v.detach().to(torch.float32).numpy().copy() for k, v in head.state_dict().items()}
    save_file(state, str(OUT / "joint_head.safetensors"))
    (OUT / "joint_head_config.json").write_text(json.dumps(HEAD, indent=2))
    (OUT / "sepal.json").write_text(json.dumps({"note": "fixture"}))

    # --- a request, and the oracle's logits in float64 ---
    length, vocab, hidden_size = 20, 64, HEAD["hidden_size"]
    input_ids = rng.integers(0, vocab, size=length).astype(np.int32)
    hidden = (rng.standard_normal((length, hidden_size)) * 1.5 + 0.3).astype(np.float32)
    instruction_spans = np.array([[3, 6], [9, 10], [11, 14]], dtype=np.int32)
    option_spans = np.array([[6, 8], [8, 9], [10, 11], [12, 14], [14, 17], [17, 18], [18, 20]], dtype=np.int32)
    option_offsets = np.array([0, 2, 4, 7], dtype=np.int32)
    question_types = np.array([0, 1, 2], dtype=np.int32)
    records = [
        clef.EncodedRecord(
            input_ids=tuple(int(t) for t in input_ids),
            questions=tuple(
                clef.EncodedQuestion(
                    question_id=f"q{q}",
                    question_type=int(question_types[q]),
                    question_span=tuple(int(x) for x in instruction_spans[q]),
                    option_spans=tuple(
                        tuple(int(x) for x in option_spans[k]) for k in range(option_offsets[q], option_offsets[q + 1])
                    ),
                    option_ids=tuple(str(k) for k in range(option_offsets[q], option_offsets[q + 1])),
                )
                for q in range(3)
            ),
            record_id="r0",
        )
    ]
    head64 = head.double().eval()
    with torch.no_grad():
        result = head64(
            torch.from_numpy(hidden.astype(np.float64))[None],
            torch.from_numpy(input_ids.astype(np.int64))[None],
            torch.ones(1, length, dtype=torch.long),
            records,
            torch.from_numpy(base["embed_tokens.weight"].astype(np.float64)),
        )
    logits = np.concatenate([t.numpy() for t in result[0]]).astype(np.float32)
    save_file(
        {
            "input_ids": input_ids, "question_types": question_types, "instruction_spans": instruction_spans,
            "option_offsets": option_offsets, "option_spans": option_spans, "hidden_fp32": hidden, "logits_fp32": logits,
        },
        str(OUT / "head_case.safetensors"),
    )
    print("wrote", OUT, "logits", logits)


if __name__ == "__main__":
    main()
