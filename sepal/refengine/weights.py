"""Weights: the tensor names and shapes a geometry implies, a read-only container, a loader and a test initialiser.

Names follow the transformers Qwen3.5 text model (``layers.<i>.linear_attn.in_proj_qkv.weight`` and so on), so a real
checkpoint's tensors can be passed to ``Weights.from_arrays`` unchanged. Projections are stored one tensor per
projection, as checkpoints hold them; the layer functions concatenate them where the model computes one product.
Three groups: the embedding table, the body (everything a pass multiplies by: the "non-embedding parameters" of
docs/RUNNER.md section 3) and the pointer head, which belongs to an adapter and not to the base model.

The vocabulary projection is never part of the weights: a prefill-only engine does not have one. Where it is a
separate matrix (9B) only its size is reported, by ``parameter_counts``.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from math import prod

import numpy as np
from numpy.typing import ArrayLike

from .arrays import Array, frozen
from .config import ModelConfig

Shapes = dict[str, tuple[int, ...]]

EMBEDDING = "embed_tokens.weight"
FINAL_NORM = "norm.weight"
HEAD_PREFIX = "head"


def embedding_shapes(config: ModelConfig) -> Shapes:
    return {EMBEDDING: (config.vocab_size, config.hidden_size)}


def deltanet_shapes(config: ModelConfig) -> Shapes:
    """One Gated DeltaNet mixer. ``conv1d.weight`` keeps the checkpoint layout [channels, 1, taps]."""
    d, hv = config.hidden_size, config.deltanet_value_heads
    return {
        "in_proj_qkv.weight": (config.conv_dim, d),
        "in_proj_z.weight": (config.deltanet_value_dim, d),
        "in_proj_b.weight": (hv, d),
        "in_proj_a.weight": (hv, d),
        "conv1d.weight": (config.conv_dim, 1, config.conv_kernel),
        "A_log": (hv,),
        "dt_bias": (hv,),
        "norm.weight": (config.deltanet_value_head_dim,),
        "out_proj.weight": (d, config.deltanet_value_dim),
    }


def attention_shapes(config: ModelConfig) -> Shapes:
    """One full-attention mixer. ``q_proj`` is twice the attention width: the output gate is packed with the query."""
    d = config.hidden_size
    return {
        "q_proj.weight": (2 * config.attention_width, d),
        "k_proj.weight": (config.attention_kv_width, d),
        "v_proj.weight": (config.attention_kv_width, d),
        "q_norm.weight": (config.attention_head_dim,),
        "k_norm.weight": (config.attention_head_dim,),
        "o_proj.weight": (d, config.attention_width),
    }


def mlp_shapes(config: ModelConfig) -> Shapes:
    d, m = config.hidden_size, config.intermediate_size
    return {"gate_proj.weight": (m, d), "up_proj.weight": (m, d), "down_proj.weight": (d, m)}


def body_shapes(config: ModelConfig) -> Shapes:
    """Every tensor of the decoder stack and the final norm: what "non-embedding parameters" counts."""
    shapes: Shapes = {}
    for i, kind in enumerate(config.layer_kinds):
        mixer, mixer_shapes = ("linear_attn", deltanet_shapes) if kind == "deltanet" else ("self_attn", attention_shapes)
        shapes |= {f"layers.{i}.{mixer}.{name}": shape for name, shape in mixer_shapes(config).items()}
        shapes |= {f"layers.{i}.mlp.{name}": shape for name, shape in mlp_shapes(config).items()}
        shapes[f"layers.{i}.input_layernorm.weight"] = (config.hidden_size,)
        shapes[f"layers.{i}.post_attention_layernorm.weight"] = (config.hidden_size,)
    shapes[FINAL_NORM] = (config.hidden_size,)
    return shapes


def head_shapes(config: ModelConfig) -> Shapes:
    """The pointer head: two linear maps (with bias) from the hidden size to the pointer dim."""
    d, p = config.hidden_size, config.pointer_dim
    return {f"{HEAD_PREFIX}.{name}": shape
            for name, shape in {"q.weight": (p, d), "q.bias": (p,), "k.weight": (p, d), "k.bias": (p,)}.items()}


def weight_shapes(config: ModelConfig) -> Shapes:
    """Every tensor ``Weights`` holds for this geometry."""
    return embedding_shapes(config) | body_shapes(config) | head_shapes(config)


def _count(shapes: Shapes) -> int:
    return sum(prod(shape) for shape in shapes.values())


@dataclass(frozen=True)
class ParameterCounts:
    """Parameter counts of a geometry, from shapes alone (nothing is allocated)."""

    embedding: int
    non_embedding: int
    lm_head: int          # 0 when the vocabulary projection is tied to the embedding; never loaded either way
    pointer_head: int

    @property
    def base_model(self) -> int:
        """What the published base checkpoint holds: embedding, body and an untied vocabulary projection."""
        return self.embedding + self.non_embedding + self.lm_head


def parameter_counts(config: ModelConfig) -> ParameterCounts:
    return ParameterCounts(
        embedding=_count(embedding_shapes(config)),
        non_embedding=_count(body_shapes(config)),
        lm_head=0 if config.tied_lm_head else config.vocab_size * config.hidden_size,
        pointer_head=_count(head_shapes(config)))


class Weights(Mapping[str, Array]):
    """Read-only float64 tensors keyed by name. ``scope`` gives one module's tensors under their short names."""

    def __init__(self, tensors: Mapping[str, Array]) -> None:
        self._tensors = dict(tensors)

    @classmethod
    def from_arrays(cls, config: ModelConfig, arrays: Mapping[str, ArrayLike]) -> Weights:
        """The loader interface: any mapping of name to array (a checkpoint, a test fixture) becomes the model's
        weights in float64. Strict, because a missing or mis-shaped tensor must not load silently."""
        expected = weight_shapes(config)
        missing, unexpected = sorted(expected.keys() - arrays.keys()), sorted(arrays.keys() - expected.keys())
        if missing or unexpected:
            raise ValueError(f"weights do not match the geometry: missing {missing[:3]} ({len(missing)} in all), "
                             f"unexpected {unexpected[:3]} ({len(unexpected)} in all)")
        tensors = {name: frozen(arrays[name]) for name in expected}
        wrong = sorted(name for name, shape in expected.items() if tensors[name].shape != shape)
        if wrong:
            name = wrong[0]
            raise ValueError(f"{len(wrong)} tensors have another shape, e.g. {name}: {tensors[name].shape}, expected {expected[name]}")
        return cls(tensors)

    def scope(self, prefix: str) -> Weights:
        """The tensors under ``prefix.``, with the prefix removed from their names."""
        start = prefix + "."
        return Weights({name[len(start):]: tensor for name, tensor in self._tensors.items() if name.startswith(start)})

    def __getitem__(self, name: str) -> Array:
        return self._tensors[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._tensors)

    def __len__(self) -> int:
        return len(self._tensors)


def _random_tensor(name: str, shape: tuple[int, ...], rng: np.random.Generator) -> Array:
    """Values for one tensor of a test model, by its role. Projections and conv kernels are scaled by fan-in so
    activations keep their size through the stack; the zero-centred norm weights are small but not zero, so ``1 + w``
    and ``w`` differ; the gated norm's weight is near one, as the real model initialises it."""
    values = rng.standard_normal(shape)
    if name == EMBEDDING:
        return values
    if name.endswith(("A_log", "dt_bias")):
        return 0.5 * values
    if name.endswith("linear_attn.norm.weight"):
        return 1.0 + 0.1 * values
    if name.endswith(("norm.weight", "layernorm.weight", ".bias")):
        return 0.1 * values
    return values / np.sqrt(shape[-1])


def random_weights(config: ModelConfig, seed: int) -> Weights:
    """Seeded random weights for tests. Only for small geometries: this allocates every tensor."""
    rng = np.random.default_rng(seed)
    return Weights.from_arrays(config, {name: _random_tensor(name, shape, rng) for name, shape in weight_shapes(config).items()})
