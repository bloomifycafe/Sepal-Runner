# Adapted from jaredpalmer/kev@5e42a7a kev/fused_qwen35.py (Apache-2.0)
"""The row executor: the engine's request semantics, one row at a time.

A request is a set of rows. A row is a run of tokens that continues a parent: a stored snapshot, another row of the
same request, or nothing. It reads its parent's snapshot (the DeltaNet states, the attention keys and values) and
then its own tokens causally, and leaves a snapshot of its own. Rows that share a parent never see each other.

The optimised engine interleaves all rows layer by layer (docs/RUNNER.md section 6). This reference instead finishes
one row before starting the next, parents first, and every row goes through the same function whether its parent
was computed a moment ago or loaded from a store. A row's result is therefore a function of its own chain and of
nothing else, bit for bit: that is the definition the engine's batching, caching and scheduling must preserve.
The decoder layer is Kev's ``decoder_forward``.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from .arrays import Array, frozen
from .attention import attention_mixer
from .config import ModelConfig
from .deltanet import deltanet_mixer
from .primitives import rms_norm, swiglu_mlp
from .snapshot import AttentionState, DeltaNetState, LayerState, Snapshot, empty_snapshot
from .weights import EMBEDDING, FINAL_NORM, Weights


@dataclass(frozen=True, eq=False)
class Row:
    """One run of tokens.

    token_ids: at least one token. start_position: the position of the first token (a branch restarts where its
    state ended). parent: the snapshot or the row this one continues; None starts from nothing. readout_offsets:
    offsets into this row's own tokens whose final hidden states the caller wants, in the order wanted.
    Rows are compared by identity: two rows with the same content are still two rows."""

    token_ids: tuple[int, ...]
    start_position: int
    parent: Snapshot | Row | None = None
    readout_offsets: tuple[int, ...] = ()


@dataclass(frozen=True, eq=False)
class RowResult:
    """hidden: [len(readout_offsets), hidden], after the final norm. snapshot: the state after the row's last token."""

    hidden: Array
    snapshot: Snapshot = field(repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "hidden", frozen(self.hidden))


def decoder_layer(x: Array, positions: Array, weights: Weights, state: LayerState, config: ModelConfig) -> tuple[Array, LayerState]:
    """One decoder layer on x [T, hidden], continuing ``state``; ``weights`` is the layer's own scope.

    ``h = norm(x); r = x + mixer(h); y = r + mlp(norm(r))``, the mixer chosen by the kind of state the layer keeps."""
    h = rms_norm(x, weights["input_layernorm.weight"], config.rms_norm_eps)
    if isinstance(state, DeltaNetState):
        mixed, new_state = deltanet_mixer(h, weights.scope("linear_attn"), state, config)
    else:
        mixed, new_state = attention_mixer(h, positions, weights.scope("self_attn"), state, config)
    residual = x + mixed
    n = rms_norm(residual, weights["post_attention_layernorm.weight"], config.rms_norm_eps)
    mlp = weights.scope("mlp")
    return residual + swiglu_mlp(n, mlp["gate_proj.weight"], mlp["up_proj.weight"], mlp["down_proj.weight"]), new_state


def continue_snapshot(config: ModelConfig, weights: Weights, parent: Snapshot, token_ids: Sequence[int], start_position: int) -> tuple[Array, Snapshot]:
    """Run tokens through the whole stack from ``parent``.

    Returns the last layer's output [T, hidden] BEFORE the final norm, and the snapshot after the last token."""
    positions = start_position + np.arange(len(token_ids), dtype=np.float64)
    x = weights[EMBEDDING][list(token_ids)]
    states: list[LayerState] = []
    for i, state in enumerate(parent.layers):
        x, new_state = decoder_layer(x, positions, weights.scope(f"layers.{i}"), state, config)
        states.append(new_state)
    return x, Snapshot(n_tokens=parent.n_tokens + len(token_ids), layers=tuple(states))


def check_snapshot(config: ModelConfig, snapshot: Snapshot) -> None:
    """Refuse a snapshot of another geometry: its layer kinds must be this model's, in order."""
    expected = tuple(DeltaNetState if kind == "deltanet" else AttentionState for kind in config.layer_kinds)
    if tuple(type(state) for state in snapshot.layers) != expected:
        raise ValueError("the parent snapshot was not made by a model of this geometry")


def check_row(config: ModelConfig, row: Row) -> None:
    """Admission for one row: reject before any compute what could only fail or silently misread later."""
    if not row.token_ids:
        raise ValueError("a row needs at least one token")
    if not all(0 <= token < config.vocab_size for token in row.token_ids):
        raise ValueError("a token id is outside the vocabulary")
    if row.start_position < 0:
        raise ValueError("start_position must not be negative")
    if not all(0 <= offset < len(row.token_ids) for offset in row.readout_offsets):
        raise ValueError("a readout offset is outside the row")
    if isinstance(row.parent, Snapshot):
        check_snapshot(config, row.parent)


def execution_order(rows: Sequence[Row]) -> list[Row]:
    """The rows with every parent row before its children, otherwise in request order.

    A parent row must be part of the request: the request is the unit that is admitted, scheduled and answered."""
    requested = {id(row) for row in rows}
    ordered: list[Row] = []
    placed: set[int] = set()

    def place(row: Row) -> None:
        if id(row) in placed:
            return
        if isinstance(row.parent, Row):
            if id(row.parent) not in requested:
                raise ValueError("a row's parent row is not part of the request")
            place(row.parent)
        placed.add(id(row))
        ordered.append(row)

    for row in rows:
        place(row)
    return ordered


def run_rows(config: ModelConfig, weights: Weights, rows: Sequence[Row]) -> list[RowResult]:
    """Run a request. Returns one result per row, in request order.

    Parents run before children. A child continues its parent's snapshot, which is the same value whether the
    parent ran in this call or its snapshot was kept from an earlier one, so both give identical bits."""
    for row in rows:
        check_row(config, row)
    results: dict[int, RowResult] = {}
    for row in execution_order(rows):
        if row.parent is None:
            parent = empty_snapshot(config)
        elif isinstance(row.parent, Snapshot):
            parent = row.parent
        else:
            parent = results[id(row.parent)].snapshot
        last_layer, snapshot = continue_snapshot(config, weights, parent, row.token_ids, row.start_position)
        hidden = rms_norm(last_layer[list(row.readout_offsets)], weights[FINAL_NORM], config.rms_norm_eps)
        results[id(row)] = RowResult(hidden=hidden, snapshot=snapshot)
    return [results[id(row)] for row in rows]
