"""Write the tiny checkpoint that the loader tests read, and the answers the NumPy reference engine gives for it.

    uv run --project <main checkout> --group verify python runner/tools/make_tiny_checkpoint.py

Output, in runner/tests/fixtures/tiny_checkpoint/ (a few hundred kilobytes, committed):

  config.json                    the geometry of `PRESETS["tiny"]` in the layout of a transformers 5 Qwen3.5 config
                                 (a `text_config` member, `rope_parameters`, `layer_types`)
  model-0000{1,2}-of-00002.safetensors, model.safetensors.index.json
                                 the text model split over two shards under the prefix `model.language_model.`, in
                                 bfloat16, plus a vision-tower tensor, a vocabulary projection and a
                                 multi-token-prediction tensor that the engine must list and ignore
  readout.safetensors, decision_config.json
                                 a decision readout [6, hidden] in bfloat16 and a temperature
  expected.json                  token ids, the final hidden state at the last token (float64, from the NumPy
                                 reference engine on the bfloat16-rounded weights), and the probabilities over 3
                                 and 6 options

The safetensors files are written here by hand (the format is a length, a JSON header and the bytes), so this
script needs only NumPy and the reference engine.
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from sepal_runner.refengine import PRESETS, Row, label_row_probabilities, run_rows  # noqa: E402
from sepal_runner.refengine import upstream  # noqa: E402
from sepal_runner.refengine.weights import _random_tensor, body_shapes, embedding_shapes  # noqa: E402

OUT = REPO / "runner" / "tests" / "fixtures" / "tiny_checkpoint"
PREFIX = "model.language_model."
SEED = 20261009
TEMPERATURE = 1.25
READOUT_ROWS = 6
TOKENS = [3, 17, 5, 60, 41, 2, 9, 33, 28, 12, 7, 50]


def to_bfloat16_bits(values: np.ndarray) -> np.ndarray:
    """Round float32 values to the nearest bfloat16 (ties to even), as the 16 stored bits."""
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32).astype(np.uint64)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def from_bfloat16_bits(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32).astype(np.float64)


def write_safetensors(path: Path, tensors: dict[str, np.ndarray], metadata: dict[str, str] | None = None) -> None:
    """tensors hold bfloat16 bit patterns (uint16) or float32 arrays. No gaps; the wider element types come first,
    as the reference writer orders them, so that every tensor starts aligned for its element type."""
    header: dict = {}
    if metadata:
        header["__metadata__"] = metadata
    offset = 0
    blobs = []
    for name, array in sorted(tensors.items(), key=lambda item: -item[1].dtype.itemsize):
        dtype = "BF16" if array.dtype == np.uint16 else "F32"
        data = np.ascontiguousarray(array).tobytes()
        header[name] = {"dtype": dtype, "shape": list(array.shape), "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
        blobs.append(data)
    text = json.dumps(header, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 8)  # the header is padded so that the data starts 8-byte aligned
    path.write_bytes(struct.pack("<Q", len(text)) + text + b"".join(blobs))


def main() -> None:
    config = PRESETS["tiny"]
    rng = np.random.default_rng(SEED)
    shapes = embedding_shapes(config) | body_shapes(config)
    stored = {name: to_bfloat16_bits(_random_tensor(name, shape, rng)) for name, shape in shapes.items()}

    # The engine's weights are the stored values widened: what the loader must reproduce.
    checkpoint = {PREFIX + name: from_bfloat16_bits(bits) for name, bits in stored.items()}
    weights = upstream.engine_weights(config, checkpoint)

    readout_bits = to_bfloat16_bits(rng.standard_normal((READOUT_ROWS, config.hidden_size)) / np.sqrt(config.hidden_size))
    readout = from_bfloat16_bits(readout_bits)

    result = run_rows(config, weights, [Row(tuple(TOKENS), 0, None, (len(TOKENS) - 1,))])[0]
    hidden = np.asarray(result.hidden)[0]
    expected = {
        "token_ids": TOKENS,
        "temperature": TEMPERATURE,
        "hidden": hidden.tolist(),
        "probabilities_3": label_row_probabilities(hidden, readout[:3], TEMPERATURE).tolist(),
        "probabilities_6": label_row_probabilities(hidden, readout[:6], TEMPERATURE).tolist(),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    names = list(stored)
    half = len(names) // 2
    shards = {"model-00001-of-00002.safetensors": names[:half], "model-00002-of-00002.safetensors": names[half:]}
    extra = {  # tensors a text prefill engine never loads; they ride in the second shard
        "model.visual.patch_embed.proj.weight": np.arange(8, dtype=np.float32).reshape(2, 4),
        "lm_head.weight": to_bfloat16_bits(np.ones((4, 3), dtype=np.float32)),
        "mtp.fc.weight": to_bfloat16_bits(np.ones((3, 3), dtype=np.float32)),
    }
    weight_map = {}
    for file_name, members in shards.items():
        tensors = {PREFIX + name: stored[name] for name in members}
        if file_name.startswith("model-00002-of"):
            tensors |= extra
        write_safetensors(OUT / file_name, tensors, {"format": "pt"})
        weight_map |= {name: file_name for name in tensors}
    (OUT / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map}, indent=1), encoding="utf-8")

    write_safetensors(OUT / "readout.safetensors", {"weight": readout_bits})
    (OUT / "decision_config.json").write_text(
        json.dumps({"format_version": 1, "temperature": TEMPERATURE, "max_options": READOUT_ROWS - 1}, indent=1), encoding="utf-8")

    kinds = ["full_attention" if kind == "attention" else "linear_attention" for kind in config.layer_kinds]
    text_config = {
        "model_type": "qwen3_5_text", "attention_bias": False, "attn_output_gate": True, "hidden_act": "silu",
        "full_attention_interval": config.full_attention_interval, "layer_types": kinds,
        "num_hidden_layers": config.num_layers, "hidden_size": config.hidden_size,
        "intermediate_size": config.intermediate_size, "num_attention_heads": config.attention_heads,
        "num_key_value_heads": config.attention_kv_heads, "head_dim": config.attention_head_dim,
        "linear_num_key_heads": config.deltanet_key_heads, "linear_num_value_heads": config.deltanet_value_heads,
        "linear_key_head_dim": config.deltanet_key_head_dim, "linear_value_head_dim": config.deltanet_value_head_dim,
        "linear_conv_kernel_dim": config.conv_kernel, "vocab_size": config.vocab_size,
        "rms_norm_eps": config.rms_norm_eps, "mlp_only_layers": [], "tie_word_embeddings": True,
        "rope_parameters": {"rope_type": "default", "rope_theta": config.rope_theta,
                            "partial_rotary_factor": config.partial_rotary_factor, "mrope_interleaved": True,
                            "mrope_section": [11, 11, 10]},
    }
    (OUT / "config.json").write_text(
        json.dumps({"architectures": ["Qwen3_5Model"], "model_type": "qwen3_5", "tie_word_embeddings": True,
                    "text_config": text_config, "vision_config": {"model_type": "qwen3_5_vision", "depth": 1}}, indent=1),
        encoding="utf-8")
    (OUT / "expected.json").write_text(json.dumps(expected, indent=1), encoding="utf-8")
    total = sum(path.stat().st_size for path in OUT.iterdir())
    print(f"wrote {len(list(OUT.iterdir()))} files, {total} bytes, to {OUT}")


if __name__ == "__main__":
    main()
