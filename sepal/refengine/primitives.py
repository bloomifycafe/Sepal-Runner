"""Elementwise functions, norms and the projection helper shared by every layer. All float64, all pure."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .arrays import Array


def sigmoid(x: Array) -> Array:
    """1 / (1 + exp(-x)), written through logaddexp so a large negative input does not overflow."""
    return np.exp(-np.logaddexp(0.0, -x))


def silu(x: Array) -> Array:
    """x * sigmoid(x), also called swish."""
    return x * sigmoid(x)


def softplus(x: Array) -> Array:
    """log(1 + exp(x))."""
    return np.logaddexp(0.0, x)


def softmax(logits: Array) -> Array:
    """Softmax over the last axis. Entries of -inf get probability exactly 0."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    weights = np.exp(shifted)
    return weights / weights.sum(axis=-1, keepdims=True)


def rms_norm(x: Array, weight: Array, eps: float) -> Array:
    """Zero-centred RMSNorm over the last axis: ``x / sqrt(mean(x^2) + eps) * (1 + weight)``.

    The stored weight is the offset from one, so a zero weight is the identity scale. A zero vector stays zero."""
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * (1.0 + weight)


def project(x: Array, weights: Sequence[Array]) -> list[Array]:
    """Several bias-free projections of the same input as ONE product with their weights concatenated, then split.

    x is [T, in]; each weight is [out_i, in]; the results are [T, out_i]. The model computes these as one GEMM
    (in-projections of a mixer, gate and up of the MLP), and so does every backend this is the oracle for."""
    product = x @ np.concatenate(weights, axis=0).T
    return np.split(product, np.cumsum([w.shape[0] for w in weights])[:-1], axis=-1)


def swiglu_mlp(x: Array, gate_weight: Array, up_weight: Array, down_weight: Array) -> Array:
    """Bias-free SwiGLU: ``down(silu(gate(x)) * up(x))``."""
    gate, up = project(x, [gate_weight, up_weight])
    return (silu(gate) * up) @ down_weight.T
