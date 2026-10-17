"""Snapshots: what a run of tokens leaves behind for the rows that continue it.

Per Gated DeltaNet layer that is the recurrent state and the conv tail (the last K-1 conv inputs); per attention
layer the keys (after their norm and RoPE) and the values of every token so far. A snapshot is a value: its arrays
are read-only copies, so running any number of rows from it cannot change it, and branches never see each other.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .arrays import Array, frozen
from .config import ModelConfig


@dataclass(frozen=True, eq=False)
class DeltaNetState:
    """recurrent: [value heads, key head dim, value head dim]. conv_tail: [K-1, conv_dim], oldest input first."""

    recurrent: Array
    conv_tail: Array

    def __post_init__(self) -> None:
        object.__setattr__(self, "recurrent", frozen(self.recurrent))
        object.__setattr__(self, "conv_tail", frozen(self.conv_tail))


@dataclass(frozen=True, eq=False)
class AttentionState:
    """keys and values: [tokens so far, kv heads, head dim], in token order."""

    keys: Array
    values: Array

    def __post_init__(self) -> None:
        object.__setattr__(self, "keys", frozen(self.keys))
        object.__setattr__(self, "values", frozen(self.values))


LayerState = DeltaNetState | AttentionState


@dataclass(frozen=True, eq=False)
class Snapshot:
    """The state after ``n_tokens`` tokens: one layer state per decoder layer, in layer order."""

    n_tokens: int
    layers: tuple[LayerState, ...]


def empty_layer_state(config: ModelConfig, layer: int) -> LayerState:
    """What a layer holds before any token: a zero recurrent state and conv tail, or no keys and values."""
    if config.layer_kind(layer) == "deltanet":
        return DeltaNetState(
            recurrent=np.zeros((config.deltanet_value_heads, config.deltanet_key_head_dim, config.deltanet_value_head_dim)),
            conv_tail=np.zeros((config.conv_kernel - 1, config.conv_dim)))
    empty = np.zeros((0, config.attention_kv_heads, config.attention_head_dim))
    return AttentionState(keys=empty, values=empty)


def empty_snapshot(config: ModelConfig) -> Snapshot:
    """The snapshot a row without a parent continues: nothing has been read yet."""
    return Snapshot(n_tokens=0, layers=tuple(empty_layer_state(config, i) for i in range(config.num_layers)))
