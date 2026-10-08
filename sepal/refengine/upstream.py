"""The bridge to the upstream model: the transformers Qwen3.5 text model as the thing the reference engine is checked
against. Nothing else in the package imports this module, and it imports PyTorch and transformers only when called
(dependency group ``verify``).

Line numbers cited in this package refer to ``transformers/models/qwen3_5/modeling_qwen3_5.py`` of transformers
5.19.0, the version the check was made against (docs/REFENGINE-VERIFICATION.md).

Two things about the upstream code decide how a comparison has to be made:

* It computes parts of the pass in float32 whatever the dtype of the weights: both norms, the RoPE table, the whole
  delta rule (so the cached recurrent state is float32) and, on the eager path, the attention softmax. A model
  converted with ``.double()`` is therefore still float32-accurate. ``float64_arithmetic`` runs the same upstream
  code with every one of those casts promoted, which is what makes agreement to 1e-9 a meaningful test.
* Without the optional packages ``flash-linear-attention`` and ``causal-conv1d`` (and without ``kernels``) it falls
  back to its own PyTorch functions, which are the ones read and cited here. ``execution_path`` reports which ran.
"""
from __future__ import annotations

import importlib.util
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

import numpy as np

from .config import ModelConfig
from .snapshot import AttentionState, DeltaNetState, LayerState, Snapshot
from .weights import Weights, body_shapes, embedding_shapes, head_shapes

TEXT_MODEL_PREFIXES = ("model.language_model.", "language_model.", "model.", "")


def execution_path() -> dict[str, str]:
    """Which implementation upstream uses here for each kernelised function: an optional package or its own PyTorch."""
    fast = {"gated delta rule": "fla", "causal conv": "causal_conv1d", "hub kernels": "kernels"}
    return {what: (package if importlib.util.find_spec(package) else "pure PyTorch fallback") for what, package in fast.items()}


def text_config(config: ModelConfig, **overrides: Any):
    """The upstream text config with this geometry."""
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    return Qwen3_5TextConfig(
        vocab_size=config.vocab_size, hidden_size=config.hidden_size, intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_layers, num_attention_heads=config.attention_heads,
        num_key_value_heads=config.attention_kv_heads, head_dim=config.attention_head_dim,
        linear_conv_kernel_dim=config.conv_kernel, linear_key_head_dim=config.deltanet_key_head_dim,
        linear_value_head_dim=config.deltanet_value_head_dim, linear_num_key_heads=config.deltanet_key_heads,
        linear_num_value_heads=config.deltanet_value_heads, rms_norm_eps=config.rms_norm_eps,
        layer_types=["full_attention" if kind == "attention" else "linear_attention" for kind in config.layer_kinds],
        rope_parameters={"rope_type": "default", "rope_theta": config.rope_theta,
                         "partial_rotary_factor": config.partial_rotary_factor, "mrope_section": [11, 11, 10]},
        **overrides)


def model_config(upstream_text_config: Any, pointer_dim: int = 256) -> ModelConfig:
    """The engine geometry an upstream text config describes. The pointer head is not part of the upstream model."""
    c = upstream_text_config
    kinds = list(c.layer_types)
    interval = kinds.index("full_attention") + 1
    config = ModelConfig(
        num_layers=c.num_hidden_layers, full_attention_interval=interval, hidden_size=c.hidden_size,
        intermediate_size=c.intermediate_size, attention_heads=c.num_attention_heads,
        attention_kv_heads=c.num_key_value_heads, attention_head_dim=c.head_dim,
        deltanet_key_heads=c.linear_num_key_heads, deltanet_value_heads=c.linear_num_value_heads,
        deltanet_key_head_dim=c.linear_key_head_dim, deltanet_value_head_dim=c.linear_value_head_dim,
        conv_kernel=c.linear_conv_kernel_dim, vocab_size=c.vocab_size, rms_norm_eps=c.rms_norm_eps,
        rope_theta=c.rope_parameters["rope_theta"], partial_rotary_factor=c.rope_parameters["partial_rotary_factor"],
        pointer_dim=pointer_dim, tied_lm_head=bool(c.tie_word_embeddings))
    expected = ["full_attention" if kind == "attention" else "linear_attention" for kind in config.layer_kinds]
    if kinds != expected:
        raise ValueError("the upstream layer_types are not a fixed full-attention interval")
    return config


def random_text_model(config: ModelConfig, seed: int, **overrides: Any):
    """A float64 upstream text model of this geometry in eval mode, with every parameter random.

    Upstream initialises the zero-centred norm weights to 0 and the gated norm weight to 1; they are perturbed here so
    that ``w`` and ``1 + w`` cannot be confused, and every other tensor is redrawn at a scale that keeps activations
    of order one through the stack."""
    import torch
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

    model = Qwen3_5TextModel(text_config(config, **overrides)).double().eval()
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            noise = torch.randn(parameter.shape, generator=generator, dtype=torch.float64)
            if name.endswith(("norm.weight", "layernorm.weight")):
                parameter.add_(0.1 * noise)
            elif name.endswith(("A_log", "dt_bias")):
                parameter.copy_(0.5 * noise)
            elif name == "embed_tokens.weight":
                parameter.copy_(noise)
            else:
                parameter.copy_(noise / parameter.shape[-1] ** 0.5)
    return model


def text_model_arrays(config: ModelConfig, tensors: Mapping[str, Any]) -> dict[str, Any]:
    """The text model's tensors out of a checkpoint or a state dict, under the engine's names.

    The prefix in front of ``embed_tokens.weight`` is found and removed; tensors outside the text model (vision tower,
    vocabulary projection, multi-token-prediction head) are left out. Nothing is renamed beyond the prefix."""
    wanted = embedding_shapes(config) | body_shapes(config)
    for prefix in TEXT_MODEL_PREFIXES:
        if prefix + "embed_tokens.weight" in tensors:
            break
    else:
        raise ValueError("no embed_tokens.weight under a known text-model prefix")
    missing = sorted(name for name in wanted if prefix + name not in tensors)
    if missing:
        raise ValueError(f"the checkpoint lacks {len(missing)} text-model tensors, e.g. {missing[0]}")
    return {name: tensors[prefix + name] for name in wanted}


def engine_weights(config: ModelConfig, tensors: Mapping[str, Any]) -> Weights:
    """Engine weights from upstream tensors (NumPy arrays or CPU torch tensors). The pointer head, which upstream
    does not have, is zero."""
    arrays = {name: np.asarray(tensor, dtype=np.float64) if not hasattr(tensor, "detach") else tensor.detach().double().cpu().numpy()
              for name, tensor in text_model_arrays(config, tensors).items()}
    arrays |= {name: np.zeros(shape) for name, shape in head_shapes(config).items()}
    return Weights.from_arrays(config, arrays)


@contextmanager
def float64_arithmetic(model: Any) -> Iterator[None]:
    """Run the upstream code with its float32 casts promoted to float64, and its RoPE frequencies recomputed in
    float64. The code that runs is upstream's own, line for line; only ``torch.float32``, ``torch.float`` and
    ``Tensor.float()`` mean float64 while this is active. Not thread-safe: it patches the module and the class."""
    import torch
    from transformers.models.qwen3_5 import modeling_qwen3_5 as upstream

    class Promoted:
        float32 = torch.float64
        float = torch.float64

        def __getattr__(self, name: str) -> Any:
            return getattr(torch, name)

    own_float = torch.Tensor.__dict__.get("float")
    saved_torch, saved_frequencies = upstream.torch, model.rotary_emb.inv_freq
    upstream.torch = Promoted()
    torch.Tensor.float = torch.Tensor.double
    try:
        model.rotary_emb.inv_freq = model.rotary_emb.compute_default_rope_parameters(model.config)[0]
        yield
    finally:
        model.rotary_emb.inv_freq = saved_frequencies
        upstream.torch = saved_torch
        if own_float is None:
            del torch.Tensor.float
        else:
            torch.Tensor.float = own_float


def hidden_states(model: Any, token_ids: Sequence[int], cache: Any = None, use_cache: bool = False):
    """Upstream final hidden states [T, hidden] for one sequence as a float64 array, and the cache it returns.

    With ``cache`` the tokens continue it (positions follow from its length, as upstream derives them)."""
    import torch

    device = next(model.parameters()).device
    with torch.no_grad():
        output = model(input_ids=torch.tensor([list(token_ids)], device=device), past_key_values=cache,
                       use_cache=use_cache or cache is not None)
    return output.last_hidden_state[0].double().cpu().numpy(), output.past_key_values


def snapshot_from_cache(config: ModelConfig, cache: Any) -> Snapshot:
    """What the upstream cache of one sequence holds, laid out as an engine snapshot.

    Upstream keeps the last K conv inputs per DeltaNet layer as [1, conv_dim, K]; only the last K-1 are ever read
    again, and those are the engine's conv tail. Its recurrent state is [1, value heads, key dim, value dim]; its
    attention keys and values are [1, kv heads, tokens, head dim], the keys after their norm and RoPE."""
    layers: list[LayerState] = []
    for kind, layer in zip(config.layer_kinds, cache.layers, strict=True):
        if kind == "deltanet":
            conv = layer.conv_states[0][0].double().cpu().numpy()
            layers.append(DeltaNetState(recurrent=layer.recurrent_states[0][0].double().cpu().numpy(),
                                        conv_tail=conv[:, conv.shape[1] - (config.conv_kernel - 1):].T))
        else:
            layers.append(AttentionState(keys=layer.keys[0].double().cpu().numpy().transpose(1, 0, 2),
                                         values=layer.values[0].double().cpu().numpy().transpose(1, 0, 2)))
    return Snapshot(n_tokens=cache.get_seq_length(), layers=tuple(layers))


def cache_state_names(cache: Any) -> list[dict[str, tuple[int, ...]]]:
    """Every tensor each cache layer carries, by attribute name and shape: the inventory a snapshot is checked
    against."""
    import torch

    inventory = []
    for layer in cache.layers:
        found: dict[str, tuple[int, ...]] = {}
        for name, value in vars(layer).items():
            if isinstance(value, torch.Tensor):
                found[name] = tuple(value.shape)
            elif isinstance(value, dict):
                found |= {f"{name}[{key}]": tuple(item.shape) for key, item in value.items() if isinstance(item, torch.Tensor)}
        inventory.append(found)
    return inventory
