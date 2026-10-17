"""Model geometry: every size the forward pass needs, read from one object and never derived from another size.

The four real Qwen3.5 geometries are named presets (docs/RUNNER.md section 3). ``tiny`` is a test geometry whose
dimensions are all different from each other where the real ones coincide (key and value head dims, key and value
head counts, attention width and hidden size), so a transposed or mis-sized tensor fails loudly instead of passing by
accident.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

LayerKind = Literal["deltanet", "attention"]


@dataclass(frozen=True)
class ModelConfig:
    """The geometry of one hybrid model. All counts are positive integers."""

    num_layers: int
    full_attention_interval: int
    hidden_size: int
    intermediate_size: int
    attention_heads: int
    attention_kv_heads: int
    attention_head_dim: int
    deltanet_key_heads: int
    deltanet_value_heads: int
    deltanet_key_head_dim: int
    deltanet_value_head_dim: int
    conv_kernel: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    partial_rotary_factor: float
    pointer_dim: int
    tied_lm_head: bool = True

    def __post_init__(self) -> None:
        if self.attention_heads % self.attention_kv_heads:
            raise ValueError("attention_heads must be a multiple of attention_kv_heads (grouped-query attention)")
        if self.deltanet_value_heads % self.deltanet_key_heads:
            raise ValueError("deltanet_value_heads must be a multiple of deltanet_key_heads (key heads are shared)")
        if self.rotary_dim % 2 or not 0 < self.rotary_dim <= self.attention_head_dim:
            raise ValueError("partial_rotary_factor must select an even, non-empty part of the attention head dim")
        if self.conv_kernel < 1 or self.num_layers < 1 or self.full_attention_interval < 1:
            raise ValueError("conv_kernel, num_layers and full_attention_interval must be at least 1")

    def layer_kind(self, layer: int) -> LayerKind:
        """Layer ``i`` is full attention iff ``(i + 1) % interval == 0``; every other layer is Gated DeltaNet."""
        return "attention" if (layer + 1) % self.full_attention_interval == 0 else "deltanet"

    @property
    def layer_kinds(self) -> tuple[LayerKind, ...]:
        return tuple(self.layer_kind(i) for i in range(self.num_layers))

    @property
    def attention_width(self) -> int:
        """Heads times head dim. NOT the hidden size in general (0.8B: 2,048 against 1,024)."""
        return self.attention_heads * self.attention_head_dim

    @property
    def attention_kv_width(self) -> int:
        return self.attention_kv_heads * self.attention_head_dim

    @property
    def rotary_dim(self) -> int:
        """How many of the head dims RoPE rotates (64 of 256 at every real size)."""
        return int(self.attention_head_dim * self.partial_rotary_factor)

    @property
    def deltanet_key_dim(self) -> int:
        return self.deltanet_key_heads * self.deltanet_key_head_dim

    @property
    def deltanet_value_dim(self) -> int:
        return self.deltanet_value_heads * self.deltanet_value_head_dim

    @property
    def conv_dim(self) -> int:
        """Channels of the depthwise convolution: q and k (key dim each) and v (value dim), concatenated."""
        return 2 * self.deltanet_key_dim + self.deltanet_value_dim


def _qwen35(layers: int, hidden: int, mlp: int, heads: int, kv_heads: int, value_heads: int, tied: bool) -> ModelConfig:
    """A real Qwen3.5 geometry: the arguments are what differs between sizes, the rest is common to all four."""
    return ModelConfig(
        num_layers=layers, full_attention_interval=4, hidden_size=hidden, intermediate_size=mlp,
        attention_heads=heads, attention_kv_heads=kv_heads, attention_head_dim=256,
        deltanet_key_heads=16, deltanet_value_heads=value_heads, deltanet_key_head_dim=128, deltanet_value_head_dim=128,
        conv_kernel=4, vocab_size=248_320, rms_norm_eps=1e-6, rope_theta=1e7, partial_rotary_factor=0.25,
        pointer_dim=256, tied_lm_head=tied)


PRESETS: dict[str, ModelConfig] = {
    "0.8B": _qwen35(24, 1024, 3584, 8, 2, 16, tied=True),
    "2B": _qwen35(24, 2048, 6144, 8, 2, 16, tied=True),
    "4B": _qwen35(32, 2560, 9216, 16, 4, 32, tied=True),
    "9B": _qwen35(32, 4096, 12288, 16, 4, 32, tied=False),
    "tiny": ModelConfig(
        num_layers=8, full_attention_interval=4, hidden_size=24, intermediate_size=40,
        attention_heads=4, attention_kv_heads=2, attention_head_dim=16,
        deltanet_key_heads=2, deltanet_value_heads=4, deltanet_key_head_dim=6, deltanet_value_head_dim=5,
        conv_kernel=4, vocab_size=64, rms_norm_eps=1e-6, rope_theta=1e7, partial_rotary_factor=0.25,
        pointer_dim=12),
}
REAL_PRESETS: tuple[str, ...] = ("0.8B", "2B", "4B", "9B")
