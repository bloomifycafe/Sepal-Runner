"""Compare the reference engine with the real Qwen3.5 0.8B checkpoint run by transformers.

    uv run --group verify python tools/verify_refengine.py [--json out.json] [--max-tokens 96] [--no-gpu]
    uv run --group verify python tools/verify_refengine.py --tiny [--json out.json]

Downloads Qwen/Qwen3.5-0.8B-Base (Apache-2.0, about 1.75 GB) into the default Hugging Face cache on first use, so it
is a tool and not a test. It needs roughly 20 GB of free memory: the engine holds the text model in float64.

It reports, and prints as JSON:
  * the geometry in the checkpoint's config and tensor shapes against ``PRESETS["0.8B"]``;
  * which tensors the text model has (bias terms, the vocabulary projection, how norm weights are stored);
  * final hidden states of a few short token sequences from the engine (NumPy float64) against the upstream model
    in float32 on the CPU, in float64 with its float32 casts promoted, and in float32 on the GPU when there is one.
The measured numbers of the recorded run are in docs/REFENGINE-VERIFICATION.md.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sepal_runner.refengine import PRESETS, Row, run_rows  # noqa: E402
from sepal_runner.refengine import attention, executor, upstream  # noqa: E402

REPO = "Qwen/Qwen3.5-0.8B-Base"
TEXTS = (
    "The quick brown fox jumps over the lazy dog.",
    "alert tcp $EXTERNAL_NET any -> $HOME_NET 445 (msg:\"SMB exploit attempt\"; sid:1000001; rev:1;)",
    "2026-10-09T08:14:22Z host=web-03 user=svc_backup action=login result=failure src=203.0.113.7 count=41",
    # Longer than 64 tokens, so upstream's chunked delta rule carries a state from one chunk into the next.
    "An analyst reviewing the overnight queue found forty-one failed logins for a service account from a single "
    "external address, followed by one success and, within two minutes, a scheduled task created on the backup "
    "server. The account had not been used interactively for ninety days. The address belongs to a hosting provider "
    "that the organisation has no relationship with. Decide whether this should be escalated, and say which single "
    "observation carries the most weight.",
)


def tensor_report(shapes: dict[str, tuple[int, ...]], prefix: str) -> dict:
    text = {name[len(prefix):]: shape for name, shape in shapes.items() if name.startswith(prefix)}
    outside = sorted({name.split(".")[0] if not name.startswith("model.") else ".".join(name.split(".")[:2])
                      for name in shapes if not name.startswith(prefix)})
    return {
        "text_model_prefix": prefix,
        "text_model_tensors": len(text),
        "tensor_groups_outside_text_model": outside,
        "lm_head_tensors_in_file": sorted(name for name in shapes if "lm_head" in name),
        "text_model_bias_tensors": sorted(name for name in text if name.endswith(".bias")),
        "conv1d_tensors_layer0": sorted(name for name in text if name.startswith("layers.0.linear_attn.conv1d")),
        "shapes": {name: list(text[name]) for name in (
            "embed_tokens.weight", "layers.0.linear_attn.in_proj_qkv.weight", "layers.0.linear_attn.in_proj_z.weight",
            "layers.0.linear_attn.in_proj_a.weight", "layers.0.linear_attn.in_proj_b.weight",
            "layers.0.linear_attn.conv1d.weight", "layers.0.linear_attn.A_log", "layers.0.linear_attn.dt_bias",
            "layers.0.linear_attn.norm.weight", "layers.0.linear_attn.out_proj.weight", "layers.0.mlp.gate_proj.weight",
            "layers.3.self_attn.q_proj.weight", "layers.3.self_attn.k_proj.weight", "layers.3.self_attn.v_proj.weight",
            "layers.3.self_attn.o_proj.weight", "layers.3.self_attn.q_norm.weight", "norm.weight")},
        "layers_with_linear_attn": sum(f"layers.{i}.linear_attn.A_log" in text for i in range(64)),
        "layers_with_self_attn": sum(f"layers.{i}.self_attn.q_proj.weight" in text for i in range(64)),
    }


def norm_weight_report(tensors: dict) -> dict:
    """Mean and range of each family of norm weights: a zero-centred family (scale = 1 + w) sits near 0, a plain one
    (scale = w) near 1."""
    families = {
        "input_layernorm.weight": [], "post_attention_layernorm.weight": [], "self_attn.q_norm.weight": [],
        "self_attn.k_norm.weight": [], "linear_attn.norm.weight": [], "norm.weight (final)": []}
    for name, tensor in tensors.items():
        values = tensor.double().numpy().ravel()
        if name == "norm.weight":
            families["norm.weight (final)"].append(values)
        else:
            for family in families:
                if name.endswith("." + family):
                    families[family].append(values)
    report = {}
    for family, parts in families.items():
        values = np.concatenate(parts)
        report[family] = {"tensors": len(parts), "mean": float(values.mean()), "min": float(values.min()), "max": float(values.max())}
    return report


def compare(reference: np.ndarray, other: np.ndarray, embedding: np.ndarray) -> dict:
    """reference, other [T, hidden]. The vocabulary projection is tied to the embedding, so the top token of a
    position is the argmax of hidden @ embedding^T."""
    difference = np.abs(reference - other)
    cosine = (reference * other).sum(-1) / (np.linalg.norm(reference, axis=-1) * np.linalg.norm(other, axis=-1))
    top_reference, top_other = (reference @ embedding.T).argmax(-1), (other @ embedding.T).argmax(-1)
    return {
        "max_abs_diff": float(difference.max()), "mean_abs_diff": float(difference.mean()),
        "max_abs_hidden": float(np.abs(other).max()), "rms_hidden": float(np.sqrt((other * other).mean())),
        "min_cosine": float(cosine.min()),
        "top_token_agreement": f"{int((top_reference == top_other).sum())}/{len(top_reference)}",
        "top_dimension_agreement": f"{int((np.abs(reference).argmax(-1) == np.abs(other).argmax(-1)).sum())}/{len(reference)}",
    }


def tiny_report() -> dict:
    """The measurements behind the tiny-model tests, as numbers instead of pass or fail. Nothing is downloaded."""
    import torch
    import transformers

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests" / "refengine"))
    from test_refengine_upstream import ALTERNATIVES, LENGTHS

    config = PRESETS["tiny"]
    rng = np.random.default_rng(3)
    report: dict = {"transformers": transformers.__version__, "torch": torch.__version__,
                    "execution_path": upstream.execution_path(), "geometry": dict(vars(config)), "parity": [],
                    "continuation": [], "alternative_readings": {}}

    def engine_hidden(weights, ids, parent=None, start=0):
        return run_rows(config, weights, [Row(ids, start, parent, tuple(range(len(ids))))])[0]

    def promoted(model, ids, cache=None, use_cache=False):
        with upstream.float64_arithmetic(model):
            return upstream.hidden_states(model, ids, cache, use_cache)

    for implementation in ("sdpa", "eager"):
        model = upstream.random_text_model(config, seed=1)
        model.config._attn_implementation = implementation
        weights = upstream.engine_weights(config, model.state_dict())
        for length in LENGTHS:
            ids = tuple(int(t) for t in rng.integers(0, config.vocab_size, length))
            ours = engine_hidden(weights, ids).hidden
            report["parity"].append({
                "attention": implementation, "tokens": length, "max_abs_hidden": float(np.abs(ours).max()),
                "max_abs_diff_float64_promoted": float(np.abs(ours - promoted(model, ids)[0]).max()),
                "max_abs_diff_upstream_unmodified": float(np.abs(ours - upstream.hidden_states(model, ids)[0]).max())})
        ids = tuple(int(t) for t in rng.integers(0, config.vocab_size, 90))
        whole = promoted(model, ids)[0]
        for split in (1, 2, 3, 37, 64, 89):
            with upstream.float64_arithmetic(model):
                prefix, cache = upstream.hidden_states(model, ids[:split], use_cache=True)
                from_cache = upstream.snapshot_from_cache(config, cache)
                suffix, _ = upstream.hidden_states(model, ids[split:], cache)
            own = engine_hidden(weights, ids[:split]).snapshot
            state_diff = {"recurrent": 0.0, "conv_tail": 0.0, "keys": 0.0, "values": 0.0}
            for theirs, ours in zip(from_cache.layers, own.layers, strict=True):
                for field in vars(theirs):
                    state_diff[field] = max(state_diff[field], float(np.abs(getattr(theirs, field) - getattr(ours, field)).max()))
            continued = engine_hidden(weights, ids[split:], from_cache, split).hidden
            report["continuation"].append({
                "attention": implementation, "prefix_tokens": split, "suffix_tokens": len(ids) - split,
                "upstream_two_passes_vs_one": float(np.abs(np.concatenate([prefix, suffix]) - whole).max()),
                "snapshot_vs_cache": state_diff,
                "engine_continuing_upstream_cache_vs_upstream_one_pass": float(np.abs(continued - whole[split:]).max())})
        if implementation == "sdpa":
            report["cache_inventory"] = dict(zip(("deltanet layer 0", "attention layer 3"),
                                                 (upstream.cache_state_names(cache)[0], upstream.cache_state_names(cache)[3])))
            ids = tuple(int(t) for t in np.random.default_rng(5).integers(0, config.vocab_size, 40))
            expected = promoted(model, ids)[0]
            report["alternative_readings"]["(none: the engine as it is)"] = float(np.abs(engine_hidden(weights, ids).hidden - expected).max())
            for name, (module, alternative) in ALTERNATIVES.items():
                function = name.split(".")[1]
                original = getattr(module, function)
                setattr(module, function, alternative)
                try:
                    report["alternative_readings"][name] = float(np.abs(engine_hidden(weights, ids).hidden - expected).max())
                finally:
                    setattr(module, function, original)
    report["summary"] = {
        "max_abs_diff_float64_promoted": max(row["max_abs_diff_float64_promoted"] for row in report["parity"]),
        "max_abs_diff_upstream_unmodified": max(row["max_abs_diff_upstream_unmodified"] for row in report["parity"]),
        "max_snapshot_vs_cache": max(max(row["snapshot_vs_cache"].values()) for row in report["continuation"]),
        "max_upstream_two_passes_vs_one": max(row["upstream_two_passes_vs_one"] for row in report["continuation"]),
        "smallest_alternative_reading_diff": min(v for k, v in report["alternative_readings"].items() if not k.startswith("("))}
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", type=Path, help="also write the report here")
    parser.add_argument("--max-tokens", type=int, default=96, help="truncate each sequence to this many tokens")
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--tiny", action="store_true", help="measure the tiny random model instead; no download")
    args = parser.parse_args()
    if args.tiny:
        text = json.dumps(tiny_report(), indent=2)
        print(text)
        if args.json:
            args.json.write_text(text + "\n", encoding="utf-8")
        return

    run_started = time.perf_counter()
    import safetensors
    import torch
    import transformers
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration, Qwen3_5TextModel

    report: dict = {"repo": REPO, "transformers": transformers.__version__, "torch": torch.__version__,
                    "safetensors": safetensors.__version__, "execution_path": upstream.execution_path()}
    path = Path(snapshot_download(REPO, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"]))
    report["revision"] = path.name

    # Geometry: config and tensor shapes.
    full_config = Qwen3_5Config.from_pretrained(path)
    text_config = full_config.text_config
    config = upstream.model_config(text_config)
    preset = PRESETS["0.8B"]
    report["geometry"] = {
        "matches_preset_0.8B": config == preset,
        "differences": {field: [getattr(config, field), getattr(preset, field)] for field in vars(preset) if getattr(config, field) != getattr(preset, field)},
        "layer_kinds": {kind: config.layer_kinds.count(kind) for kind in ("deltanet", "attention")},
        "config": {k: v for k, v in vars(config).items()},
        "tie_word_embeddings": {"top_level": full_config.tie_word_embeddings, "text_config": text_config.tie_word_embeddings},
        "stored_dtype": json.loads((path / "config.json").read_text())["text_config"].get("dtype"),
    }

    stored: dict = {}
    for file in sorted(path.glob("*.safetensors")):
        stored |= load_file(file)
    prefix = next(p for p in upstream.TEXT_MODEL_PREFIXES if p + "embed_tokens.weight" in stored)
    report["tensors"] = tensor_report({name: tuple(t.shape) for name, t in stored.items()}, prefix)
    by_dtype: dict[str, set[str]] = {}
    for name, tensor in stored.items():
        if name.startswith(prefix):
            by_dtype.setdefault(str(tensor.dtype), set()).add(".".join(name.split(".")[-2:]))
    report["tensors"]["stored_dtypes"] = {dtype: sorted(names) for dtype, names in by_dtype.items()}
    text_tensors = {name: tensor.float() for name, tensor in upstream.text_model_arrays(config, stored).items()}
    report["tensors"]["text_model_tensors_not_used_by_engine"] = sorted(
        name[len(prefix):] for name in stored if name.startswith(prefix) and name[len(prefix):] not in text_tensors)
    report["norm_weights"] = norm_weight_report({n: t for n, t in text_tensors.items() if n.endswith("norm.weight")})
    del stored

    # The upstream text model, loaded strictly from the stored tensors, and the official loader as a cross-check that
    # it places the same tensors under the same names and ties the vocabulary projection.
    model = Qwen3_5TextModel(text_config).eval()
    model.load_state_dict(text_tensors, strict=True)
    official = Qwen3_5ForConditionalGeneration.from_pretrained(path, dtype=torch.float32).eval()
    official_text = next(m for m in official.modules() if isinstance(m, Qwen3_5TextModel))
    official_state = official_text.state_dict()
    report["official_loader"] = {
        "same_tensor_names": sorted(official_state) == sorted(text_tensors),
        "bit_identical_tensors": all(torch.equal(official_state[name], text_tensors[name]) for name in text_tensors),
        "lm_head_is_embedding": official.lm_head.weight is official_text.embed_tokens.weight,
        "conv1d_bias_is_none": all(layer.linear_attn.conv1d.bias is None for layer in official_text.layers if hasattr(layer, "linear_attn")),
    }
    del official, official_text, official_state

    tokenizer = AutoTokenizer.from_pretrained(path)
    sequences = [tuple(tokenizer(text, add_special_tokens=False)["input_ids"][:args.max_tokens]) for text in TEXTS]

    started = time.perf_counter()
    weights = upstream.engine_weights(config, text_tensors)
    report["engine_weight_load_seconds"] = round(time.perf_counter() - started, 1)
    engine, engine_seconds = [], []
    for ids in sequences:
        started = time.perf_counter()
        engine.append(run_rows(config, weights, [Row(ids, 0, None, tuple(range(len(ids))))])[0].hidden)
        engine_seconds.append(round(time.perf_counter() - started, 2))
    embedding = np.array(weights["embed_tokens.weight"])

    # The counterfactual for how norm weights are stored: the same pass with scale = w instead of 1 + w.
    def plain_rms_norm(x, weight, eps):
        return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * weight

    zero_centred = executor.rms_norm
    executor.rms_norm = attention.rms_norm = plain_rms_norm
    try:
        first = sequences[0]
        plain_scale_hidden = run_rows(config, weights, [Row(first, 0, None, tuple(range(len(first))))])[0].hidden
    finally:
        executor.rms_norm = attention.rms_norm = zero_centred
    del weights

    def run(label: str, runner) -> None:
        for index, ids in enumerate(sequences):
            started = time.perf_counter()
            hidden = runner(ids)
            seconds = round(time.perf_counter() - started, 2)
            report["sequences"][index]["upstream"][label] = compare(engine[index], hidden, embedding) | {"seconds": seconds}

    report["sequences"] = [{"text": text, "tokens": len(ids), "token_ids": list(ids), "engine_seconds": seconds, "upstream": {}}
                           for text, ids, seconds in zip(TEXTS, sequences, engine_seconds)]
    run("float32 cpu", lambda ids: upstream.hidden_states(model, ids)[0])
    report["norm_weights"]["counterfactual_scale_w_instead_of_1_plus_w"] = compare(
        plain_scale_hidden, upstream.hidden_states(model, sequences[0])[0], embedding) | {"sequence": 0, "against": "float32 cpu"}

    # float64 before anything narrower: the stored values are exact in float32 and float64, but not in float16.
    model.double()

    def promoted(ids):
        with upstream.float64_arithmetic(model):
            return upstream.hidden_states(model, ids)[0]

    run("float64 cpu, float32 casts promoted", promoted)
    run("float64 cpu, upstream unmodified", lambda ids: upstream.hidden_states(model, ids)[0])
    model.float()
    if torch.cuda.is_available() and not args.no_gpu:
        model.cuda()
        report["gpu"] = torch.cuda.get_device_name(0)
        run("float32 gpu", lambda ids: upstream.hidden_states(model, ids)[0])
        model.half()     # lossy and in place: nothing may use this model's weights afterwards
        run("float16 gpu", lambda ids: upstream.hidden_states(model, ids)[0])
    del model

    report["summary"] = {
        label: {"max_abs_diff": max(s["upstream"][label]["max_abs_diff"] for s in report["sequences"]),
                "min_cosine": min(s["upstream"][label]["min_cosine"] for s in report["sequences"]),
                "top_token_agreement": [s["upstream"][label]["top_token_agreement"] for s in report["sequences"]]}
        for label in report["sequences"][0]["upstream"]}
    report["total_seconds"] = round(time.perf_counter() - run_started, 1)
    text = json.dumps(report, indent=2)
    print(text)
    if args.json:
        args.json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
