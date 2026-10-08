# Adapted from jaredpalmer/kev@5e42a7a kev/model.py (Apache-2.0)
"""The two readouts: from final hidden states at the readout positions to a probability per option.

Every question is read out on its own, from its own hidden states: nothing here mixes questions, so a question's
probabilities cannot depend on which other questions share its request. The pointer head is Kev's ``PointerHead``.
"""
from __future__ import annotations

from collections.abc import Sequence
from math import sqrt

from .arrays import Array
from .config import ModelConfig
from .primitives import softmax
from .weights import EMBEDDING, HEAD_PREFIX, Weights


def check_temperature(temperature: float) -> None:
    if not temperature > 0:
        raise ValueError("temperature must be positive")


def pointer_logits(config: ModelConfig, weights: Weights, h_decide: Array, h_options: Array, temperature: float = 1.0) -> Array:
    """One question's logits [K] from the hidden state at its decide token [hidden] and at the end of each of its
    K options [K, hidden].

    Two linear maps into the pointer dim, one for the decide vector and one for the option vectors; the logit of
    an option is their dot product divided by ``sqrt(pointer_dim)`` and by the calibration temperature."""
    check_temperature(temperature)
    head = weights.scope(HEAD_PREFIX)
    decide = head["q.weight"] @ h_decide + head["q.bias"]
    options = h_options @ head["k.weight"].T + head["k.bias"]
    return options @ decide / sqrt(config.pointer_dim) / temperature


def pointer_probabilities(config: ModelConfig, weights: Weights, h_decide: Array, h_options: Array, temperature: float = 1.0) -> Array:
    """One question's probability per option [K]: the softmax of its pointer logits."""
    return softmax(pointer_logits(config, weights, h_decide, h_options, temperature))


def label_rows(config: ModelConfig, weights: Weights, label_token_ids: Sequence[int]) -> Array:
    """The embedding rows [K, hidden] of a question's valid label tokens (the vocabulary projection's rows where
    it is tied to the embedding). Labels must be distinct, or one answer would be counted twice."""
    if len(set(label_token_ids)) != len(label_token_ids) or not label_token_ids:
        raise ValueError("a question needs at least one label token, each listed once")
    if not all(0 <= token < config.vocab_size for token in label_token_ids):
        raise ValueError("a label token id is outside the vocabulary")
    return weights[EMBEDDING][list(label_token_ids)]


def label_row_logits(h_slot: Array, rows: Array, temperature: float = 1.0) -> Array:
    """One question's logits [K]: the hidden state at its answer slot [hidden] times its label rows [K, hidden],
    divided by the calibration temperature. Only these K rows of the vocabulary are ever read."""
    check_temperature(temperature)
    return rows @ h_slot / temperature


def label_row_probabilities(h_slot: Array, rows: Array, temperature: float = 1.0) -> Array:
    """One question's probability per label [K]: the softmax over its valid labels only."""
    return softmax(label_row_logits(h_slot, rows, temperature))
