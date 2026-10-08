"""The request invariants every backend is held to: continuation, chaining, batch invariance, isolation,
snapshot immutability and hit = miss."""
import dataclasses

import numpy as np
import pytest

from refengine_support import EXACT, all_offsets, assert_snapshots_close, hidden_of, snapshot_arrays, snapshot_of, token_ids
from sepal.refengine import Row, empty_snapshot, random_weights, run_rows

LENGTH = 10


@pytest.fixture(scope="module")
def sequence(config):
    return token_ids(config, seed=1, count=LENGTH)


@pytest.fixture(scope="module")
def one_pass(config, weights, sequence):
    return run_rows(config, weights, [Row(sequence, 0, None, all_offsets(sequence))])[0]


def branches_of(config, count=4):
    """Ragged question branches: different lengths, different tokens."""
    return [token_ids(config, seed=100 + i, count=2 + 2 * i) for i in range(count)]


def test_hidden_states_are_finite_and_token_dependent(config, one_pass):
    assert one_pass.hidden.shape == (LENGTH, config.hidden_size)
    assert np.isfinite(one_pass.hidden).all()
    assert np.abs(one_pass.hidden[0] - one_pass.hidden[-1]).max() > 1e-3


@pytest.mark.parametrize("split", range(1, LENGTH))
def test_continuing_a_snapshot_equals_one_pass(config, weights, sequence, one_pass, split):
    head = run_rows(config, weights, [Row(sequence[:split], 0, None, all_offsets(sequence[:split]))])[0]
    tail = run_rows(config, weights, [Row(sequence[split:], split, head.snapshot, all_offsets(sequence[split:]))])[0]
    np.testing.assert_allclose(np.concatenate([head.hidden, tail.hidden]), one_pass.hidden, rtol=0, atol=EXACT)
    assert_snapshots_close(tail.snapshot, one_pass.snapshot)


@pytest.mark.parametrize("cuts", [(3, 7), (1, 2, 3), (2, 4, 6, 8), tuple(range(1, LENGTH))])
def test_chained_rows_equal_one_row(config, weights, sequence, one_pass, cuts):
    rows, parent = [], None
    for start, end in zip((0, *cuts), (*cuts, LENGTH)):
        parent = Row(sequence[start:end], start, parent, tuple(range(end - start)))
        rows.append(parent)
    results = run_rows(config, weights, rows)
    np.testing.assert_allclose(np.concatenate([r.hidden for r in results]), one_pass.hidden, rtol=0, atol=EXACT)
    assert_snapshots_close(results[-1].snapshot, one_pass.snapshot)
    assert [r.snapshot.n_tokens for r in results] == list((*cuts, LENGTH))


def test_branches_after_a_state_equal_state_plus_branch_rows(config, weights, sequence):
    """The request algorithm: the state once and each branch from its snapshot, against one causal row per
    question with the state repeated."""
    state = Row(sequence, 0)
    branches = branches_of(config)
    results = run_rows(config, weights, [state] + [Row(b, LENGTH, state, all_offsets(b)) for b in branches])
    for branch, result in zip(branches, results[1:]):
        np.testing.assert_allclose(result.hidden, hidden_of(config, weights, sequence + branch)[LENGTH:], rtol=0, atol=EXACT)
        assert result.snapshot.n_tokens == LENGTH + len(branch)


def test_batch_invariance_is_bit_exact(config, weights, sequence):
    """Many branches in one request give each branch the bits it gets alone, in any request order."""
    snapshot = snapshot_of(config, weights, sequence)
    branches = branches_of(config)
    rows = [Row(b, LENGTH, snapshot, all_offsets(b)) for b in branches]
    together = run_rows(config, weights, rows)
    backwards = run_rows(config, weights, rows[::-1])[::-1]
    for branch, result, mirrored in zip(branches, together, backwards):
        alone = hidden_of(config, weights, branch, parent=snapshot, start=LENGTH)
        assert np.array_equal(result.hidden, alone) and np.array_equal(mirrored.hidden, alone)


def test_branches_are_isolated(config, weights, sequence):
    """Changing one branch's tokens (or its length) leaves every other branch's bits alone."""
    state = Row(sequence, 0)
    branches = branches_of(config)
    changed = [branches[0], token_ids(config, seed=999, count=7), *branches[2:]]
    before = run_rows(config, weights, [state] + [Row(b, LENGTH, state, all_offsets(b)) for b in branches])
    after = run_rows(config, weights, [state] + [Row(b, LENGTH, state, all_offsets(b)) for b in changed])
    for i in (0, 2, 3):
        assert np.array_equal(before[i + 1].hidden, after[i + 1].hidden)
    assert not np.array_equal(before[2].hidden[:1], after[2].hidden[:1])


def test_a_branch_reads_its_state(config, weights, sequence):
    """The isolation above is not indifference: a branch's result changes with one early state token, and with
    where the branch is placed."""
    branch = branches_of(config)[1]
    other_state = ((sequence[0] + 1) % config.vocab_size, *sequence[1:])
    base = hidden_of(config, weights, branch, snapshot_of(config, weights, sequence), LENGTH)
    assert np.abs(base - hidden_of(config, weights, branch, snapshot_of(config, weights, other_state), LENGTH)).max() > 1e-4
    assert np.abs(base - hidden_of(config, weights, branch, snapshot_of(config, weights, sequence), LENGTH + 5)).max() > 1e-6


def test_snapshot_is_not_mutated_by_its_children(config, weights, sequence):
    snapshot = snapshot_of(config, weights, sequence)
    before = [a.copy() for a in snapshot_arrays(snapshot)]
    branches = branches_of(config)
    first = run_rows(config, weights, [Row(b, LENGTH, snapshot, all_offsets(b)) for b in branches])
    second = run_rows(config, weights, [Row(b, LENGTH, snapshot, all_offsets(b)) for b in branches])
    assert all(np.array_equal(a, b) for a, b in zip(snapshot_arrays(snapshot), before, strict=True))
    assert all(np.array_equal(a.hidden, b.hidden) for a, b in zip(first, second))
    assert snapshot.n_tokens == LENGTH and snapshot.layers[3].keys.shape[0] == LENGTH
    for array in snapshot_arrays(snapshot) + [first[0].hidden]:
        with pytest.raises(ValueError):
            array[...] = 0.0


def test_hit_equals_miss_bit_for_bit(config, weights, sequence):
    """Miss: the state and its branches in one request. Hit: the branches from the snapshot an earlier request
    left. Identical bits, for the hidden states and for the snapshots the branches leave."""
    branches = branches_of(config)
    state = Row(sequence, 0)
    miss = run_rows(config, weights, [state] + [Row(b, LENGTH, state, all_offsets(b)) for b in branches])[1:]
    stored = snapshot_of(config, weights, sequence)
    hit = run_rows(config, weights, [Row(b, LENGTH, stored, all_offsets(b)) for b in branches])
    for cold, warm in zip(miss, hit):
        assert np.array_equal(cold.hidden, warm.hidden)
        assert all(np.array_equal(a, b) for a, b in zip(snapshot_arrays(cold.snapshot), snapshot_arrays(warm.snapshot), strict=True))


def test_hit_equals_miss_through_a_prefix_chain(config, weights, sequence):
    """Prefix row -> tail row -> branch, with nothing, the prefix, or the whole state already stored."""
    branch = branches_of(config)[2]
    prefix = Row(sequence[:6], 0)
    tail = Row(sequence[6:], 6, prefix)
    cold = run_rows(config, weights, [prefix, tail, Row(branch, LENGTH, tail, all_offsets(branch))])
    stored_prefix = run_rows(config, weights, [Row(sequence[:6], 0)])[0].snapshot
    warm_tail = Row(sequence[6:], 6, stored_prefix)
    warm = run_rows(config, weights, [warm_tail, Row(branch, LENGTH, warm_tail, all_offsets(branch))])
    hot = run_rows(config, weights, [Row(branch, LENGTH, warm[0].snapshot, all_offsets(branch))])
    assert np.array_equal(cold[2].hidden, warm[1].hidden) and np.array_equal(cold[2].hidden, hot[0].hidden)


def test_parents_run_first_whatever_the_request_order(config, weights, sequence):
    state = Row(sequence[:5], 0)
    middle = Row(sequence[5:8], 5, state, (2,))
    leaf = Row(sequence[8:], 8, middle, (0, 1))
    forward = run_rows(config, weights, [state, middle, leaf])
    shuffled = run_rows(config, weights, [leaf, state, middle])
    assert np.array_equal(shuffled[0].hidden, forward[2].hidden) and np.array_equal(shuffled[2].hidden, forward[1].hidden)


def test_readouts_follow_the_requested_offsets(config, weights, sequence, one_pass):
    picked = run_rows(config, weights, [Row(sequence, 0, None, (7, 0, 7))])[0].hidden
    assert np.array_equal(picked, one_pass.hidden[[7, 0, 7]])
    assert run_rows(config, weights, [Row(sequence, 0)])[0].hidden.shape == (0, config.hidden_size)


def test_an_explicit_empty_snapshot_is_no_parent(config, weights, sequence, one_pass):
    explicit = run_rows(config, weights, [Row(sequence, 0, empty_snapshot(config), all_offsets(sequence))])[0]
    assert np.array_equal(explicit.hidden, one_pass.hidden)


def test_shifting_every_position_changes_nothing(config, weights, sequence, one_pass):
    """Positions enter through RoPE only, and RoPE is relative."""
    shifted = run_rows(config, weights, [Row(sequence, 4000, None, all_offsets(sequence))])[0]
    np.testing.assert_allclose(shifted.hidden, one_pass.hidden, rtol=0, atol=1e-8)


def test_results_depend_on_the_weights(config, sequence, one_pass):
    other = hidden_of(config, random_weights(config, seed=1), sequence)
    assert np.abs(other - one_pass.hidden).max() > 1e-3


def test_admission_rejects_bad_rows(config, weights, sequence):
    outsider = Row(sequence[:3], 0)
    for bad in (Row((), 0), Row((config.vocab_size,), 0), Row((-1,), 0), Row(sequence, -1), Row(sequence, 0, None, (LENGTH,)),
                Row(sequence[3:], 3, outsider)):
        with pytest.raises(ValueError):
            run_rows(config, weights, [bad])


def test_a_snapshot_of_another_geometry_is_refused(config, weights, sequence):
    foreign = empty_snapshot(dataclasses.replace(config, full_attention_interval=2))
    with pytest.raises(ValueError, match="geometry"):
        run_rows(config, weights, [Row(sequence, 0, foreign)])
