"""Helpers shared by the reference-engine tests."""
import numpy as np

from sepal.refengine import AttentionState, DeltaNetState, ModelConfig, Row, Snapshot, Weights, run_rows

EXACT = 1e-10   # two orderings of the same float64 computation agree to this; the C++ fp64 path is held to it too


def token_ids(config: ModelConfig, seed: int, count: int) -> tuple[int, ...]:
    """``count`` seeded random token ids."""
    return tuple(int(t) for t in np.random.default_rng(seed).integers(0, config.vocab_size, count))


def snapshot_arrays(snapshot: Snapshot) -> list[np.ndarray]:
    """Every array a snapshot holds, in a fixed order."""
    arrays = []
    for state in snapshot.layers:
        if isinstance(state, DeltaNetState):
            arrays += [state.recurrent, state.conv_tail]
        else:
            assert isinstance(state, AttentionState)
            arrays += [state.keys, state.values]
    return arrays


def assert_snapshots_close(a: Snapshot, b: Snapshot, atol: float = EXACT) -> None:
    assert a.n_tokens == b.n_tokens
    for x, y in zip(snapshot_arrays(a), snapshot_arrays(b), strict=True):
        np.testing.assert_allclose(x, y, rtol=0, atol=atol)


def all_offsets(ids: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(range(len(ids)))


def hidden_of(config: ModelConfig, weights: Weights, ids: tuple[int, ...], parent: Snapshot | None = None, start: int = 0) -> np.ndarray:
    """The final hidden state of every token of one row run alone."""
    return run_rows(config, weights, [Row(ids, start, parent, all_offsets(ids))])[0].hidden


def snapshot_of(config: ModelConfig, weights: Weights, ids: tuple[int, ...]) -> Snapshot:
    """The snapshot after one root row."""
    return run_rows(config, weights, [Row(ids, 0)])[0].snapshot
