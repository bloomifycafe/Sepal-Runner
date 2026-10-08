# Adapted from jaredpalmer/kev@5e42a7a kev/fused_qwen35.py (Apache-2.0)
"""The full-attention mixer: gated grouped-query attention with per-head norms and partial RoPE.

The projection, the packing of the output gate with the query and the order of operations are Kev's
``attention_forward``. The rotary embedding and the attention product are library calls there, so they are written
out here as their own functions. Each was since checked against the upstream transformers model and found to be
what it computes (docs/REFENGINE-VERIFICATION.md; the upstream lines are cited next to
``sepal.refengine.UNVERIFIED``).
"""
from __future__ import annotations

from math import sqrt

import numpy as np

from .arrays import Array
from .config import ModelConfig
from .primitives import project, rms_norm, sigmoid, softmax
from .snapshot import AttentionState
from .weights import Weights


def unpack_query_and_gate(q_gate: Array, config: ModelConfig) -> tuple[Array, Array]:
    """The ``q_proj`` output [T, 2 * heads * head_dim] as the query and its output gate, each [T, heads, head_dim].

    The packing is per head: each head's 2 * head_dim outputs are its query, then its gate."""
    per_head = q_gate.reshape(q_gate.shape[0], config.attention_heads, 2 * config.attention_head_dim)
    return per_head[..., :config.attention_head_dim], per_head[..., config.attention_head_dim:]


def rope_angles(positions: Array, config: ModelConfig) -> Array:
    """[T, rotary_dim / 2] rotation angles: position times ``theta ** (-2i / rotary_dim)``.

    One position per token: for text the model's three position axes carry the same value, so its sectioned
    multi-axis RoPE reduces to this."""
    rotary = config.rotary_dim
    inverse_frequency = config.rope_theta ** (-np.arange(0, rotary, 2, dtype=np.float64) / rotary)
    return positions[:, None] * inverse_frequency[None, :]


def apply_rope(x: Array, angles: Array) -> Array:
    """Rotate the FIRST rotary_dim dims of every head; the rest pass through. x [T, heads, head_dim].

    Dim i is paired with dim i + rotary_dim / 2 (the two halves of the rotated part, not neighbouring dims):
    ``(a, b) -> (a cos - b sin, b cos + a sin)``."""
    half = angles.shape[1]
    cos, sin = np.cos(angles)[:, None, :], np.sin(angles)[:, None, :]
    first, second, rest = x[..., :half], x[..., half:2 * half], x[..., 2 * half:]
    return np.concatenate([first * cos - second * sin, second * cos + first * sin, rest], axis=-1)


def attention_scale(config: ModelConfig) -> float:
    """What the query-key dot product is multiplied by before the softmax: ``1 / sqrt(head_dim)``."""
    return 1.0 / sqrt(config.attention_head_dim)


def grouped_query_attention(q: Array, keys: Array, values: Array, context: int, scale: float) -> Array:
    """Causal grouped-query attention of a run of tokens over its context and itself.

    q [T, heads, head_dim]; keys, values [context + T, kv heads, head_dim], the first ``context`` rows from the
    parent and the rest this run's own. Token t attends to every context row and to own rows 0..t. Query head h
    reads kv head ``h // (heads / kv heads)``. Returns [T, heads, head_dim]."""
    tokens, heads, _ = q.shape
    group = heads // keys.shape[1]
    visible = np.arange(keys.shape[0])[None, :] <= context + np.arange(tokens)[:, None]
    output = np.empty(q.shape)
    for head in range(heads):
        scores = scale * (q[:, head] @ keys[:, head // group].T)
        output[:, head] = softmax(np.where(visible, scores, -np.inf)) @ values[:, head // group]
    return output


def attention_mixer(x: Array, positions: Array, weights: Weights, state: AttentionState, config: ModelConfig) -> tuple[Array, AttentionState]:
    """The mixer on the normed layer input x [T, hidden] at ``positions`` [T], continuing ``state``.

    Returns the mixer output [T, hidden] and the state with this run's keys and values appended. ``state`` is only
    read."""
    tokens, eps = x.shape[0], config.rms_norm_eps
    kv_shape = (tokens, config.attention_kv_heads, config.attention_head_dim)
    q_gate, k, v = project(x, [weights[f"{name}_proj.weight"] for name in ("q", "k", "v")])
    q, gate = unpack_query_and_gate(q_gate, config)
    angles = rope_angles(positions, config)
    q = apply_rope(rms_norm(q, weights["q_norm.weight"], eps), angles)
    k = apply_rope(rms_norm(k.reshape(kv_shape), weights["k_norm.weight"], eps), angles)
    keys, values = np.concatenate([state.keys, k]), np.concatenate([state.values, v.reshape(kv_shape)])
    attended = grouped_query_attention(q, keys, values, state.keys.shape[0], attention_scale(config))
    gated = (attended * sigmoid(gate)).reshape(tokens, config.attention_width)
    return gated @ weights["o_proj.weight"].T, AttentionState(keys=keys, values=values)
