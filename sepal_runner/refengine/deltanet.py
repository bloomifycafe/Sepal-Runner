# Adapted from jaredpalmer/kev@5e42a7a kev/fused_qwen35.py (Apache-2.0)
"""The Gated DeltaNet mixer, token by token.

The projection, the split sizes and the order of operations are Kev's ``deltanet_forward``. Kev hands the
convolution, the recurrence and the gated norm to library kernels whose bodies are not in the reference files, so
each of those is written out here as its own small function. Each was since checked against the upstream
transformers model and found to be what it computes (docs/REFENGINE-VERIFICATION.md; the upstream lines are cited
next to ``sepal_runner.refengine.UNVERIFIED``).
"""
from __future__ import annotations

from math import sqrt

import numpy as np

from .arrays import Array
from .config import ModelConfig
from .primitives import project, sigmoid, silu, softplus
from .snapshot import DeltaNetState
from .weights import Weights

L2_EPS = 1e-6


def causal_conv(inputs: Array, kernel: Array, tail: Array) -> tuple[Array, Array]:
    """Depthwise causal convolution continued from ``tail``, without activation.

    inputs [T, C]; kernel [C, K]; tail [K-1, C], the K-1 inputs before this run, oldest first. Output t is
    ``sum_j kernel[:, j] * window[t + j]`` over the window of the K inputs ending at t, so the LAST tap multiplies
    the current token (the cross-correlation a left-padded Conv1d computes). Returns the outputs [T, C] and the new
    tail: the last K-1 inputs of tail-then-inputs."""
    taps = kernel.shape[1]
    window = np.concatenate([tail, inputs], axis=0)
    outputs = sum(kernel[:, j] * window[j:j + inputs.shape[0]] for j in range(taps))
    return outputs, window[window.shape[0] - (taps - 1):]


def l2_normalise(x: Array) -> Array:
    """``x / sqrt(sum(x^2) + eps)`` over the last axis. A zero vector stays zero."""
    return x / np.sqrt(np.sum(x * x, axis=-1, keepdims=True) + L2_EPS)


def share_key_heads(per_key_head: Array, config: ModelConfig) -> Array:
    """[T, key heads, dk] -> [T, value heads, dk]: value head h reads key head ``h // (value heads / key heads)``."""
    return np.repeat(per_key_head, config.deltanet_value_heads // config.deltanet_key_heads, axis=1)


def split_heads(conv_output: Array, config: ModelConfig) -> tuple[Array, Array, Array]:
    """The activated conv output [T, conv_dim] as q, k [T, key heads, dk] and v [T, value heads, dv]."""
    key_dim, tokens = config.deltanet_key_dim, conv_output.shape[0]
    q, k, v = np.split(conv_output, [key_dim, 2 * key_dim], axis=-1)
    key_shape = (tokens, config.deltanet_key_heads, config.deltanet_key_head_dim)
    return q.reshape(key_shape), k.reshape(key_shape), v.reshape(tokens, config.deltanet_value_heads, config.deltanet_value_head_dim)


def decay_gate(a: Array, a_log: Array, dt_bias: Array) -> Array:
    """The per-head decay in (0, 1]: ``exp(g)`` with ``g = -exp(A_log) * softplus(a + dt_bias)``."""
    return np.exp(-np.exp(a_log) * softplus(a + dt_bias))


def delta_rule_update(state: Array, k: Array, v: Array, decay: Array, beta: Array) -> Array:
    """One token of the gated delta rule, all value heads at once.

    state [H, dk, dv]; k [H, dk]; v [H, dv]; decay, beta [H]. The old state is decayed FIRST; the memory then
    replaces a fraction beta of what it returns for key k by v:
    ``S' = decay * S;  S_new = S' + k (x) beta * (v - k . S')``."""
    decayed = decay[:, None, None] * state
    recalled = np.einsum("hk,hkv->hv", k, decayed)
    correction = beta[:, None] * (v - recalled)
    return decayed + k[:, :, None] * correction[:, None, :]


def delta_rule_output_scale(config: ModelConfig) -> float:
    """What the read ``q . S`` is multiplied by: ``1 / sqrt(key head dim)``."""
    return 1.0 / sqrt(config.deltanet_key_head_dim)


def delta_rule_scan(q: Array, k: Array, v: Array, decay: Array, beta: Array, state: Array, scale: float) -> tuple[Array, Array]:
    """The recurrence over a run of tokens, strictly one token after another.

    q, k [T, H, dk]; v [T, H, dv]; decay, beta [T, H]; state [H, dk, dv]. Each token updates the state, then reads
    it with its own query (so a token sees its own write). Returns the reads [T, H, dv] and the final state."""
    outputs = np.empty(v.shape)
    for t in range(q.shape[0]):
        state = delta_rule_update(state, k[t], v[t], decay[t], beta[t])
        outputs[t] = scale * np.einsum("hk,hkv->hv", q[t], state)
    return outputs, state


def gated_norm_scale(weight: Array) -> Array:
    """The gated norm's learned scale: the stored weight itself, NOT ``1 + weight`` like the other norms."""
    return weight


def gated_norm_gate(z: Array) -> Array:
    """The gate the normed output is multiplied by: ``silu(z)`` (Kev names the activation "swish")."""
    return silu(z)


def gated_rms_norm(x: Array, z: Array, weight: Array, eps: float) -> Array:
    """RMSNorm of each value head's read, scaled by the norm weight and gated by z. x, z [T, H, dv]."""
    normed = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return normed * gated_norm_scale(weight) * gated_norm_gate(z)


def deltanet_mixer(x: Array, weights: Weights, state: DeltaNetState, config: ModelConfig) -> tuple[Array, DeltaNetState]:
    """The mixer on the normed layer input x [T, hidden], continuing ``state``.

    Returns the mixer output [T, hidden] and the state after the last token. ``state`` is only read."""
    mixed, z, b, a = project(x, [weights[f"in_proj_{name}.weight"] for name in ("qkv", "z", "b", "a")])
    conv_output, conv_tail = causal_conv(mixed, weights["conv1d.weight"][:, 0, :], state.conv_tail)
    q, k, v = split_heads(silu(conv_output), config)
    q, k = share_key_heads(l2_normalise(q), config), share_key_heads(l2_normalise(k), config)
    reads, recurrent = delta_rule_scan(q, k, v, decay_gate(a, weights["A_log"], weights["dt_bias"]), sigmoid(b),
                                       state.recurrent, delta_rule_output_scale(config))
    normed = gated_rms_norm(reads, z.reshape(reads.shape), weights["norm.weight"], config.rms_norm_eps)
    output = normed.reshape(x.shape[0], config.deltanet_value_dim) @ weights["out_proj.weight"].T
    return output, DeltaNetState(recurrent=recurrent, conv_tail=conv_tail)
