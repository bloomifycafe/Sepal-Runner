"""The pointer head and the label-row readout, alone and at the end of a whole request."""
from math import sqrt

import numpy as np
import pytest

from refengine_support import token_ids
from sepal.refengine import Row, label_row_probabilities, label_rows, pointer_probabilities, run_rows
from sepal.refengine.readout import label_row_logits, pointer_logits

STATE_LENGTH = 9
# (token seed, branch length, offsets of the option ends); the decide token is the last of its branch
POINTER_QUESTIONS = [(50, 8, (2, 4, 6)), (51, 5, (1, 3)), (52, 14, (2, 3, 5, 8, 10, 12))]
# (token seed, branch length, label token ids); the answer slot is the last token of its branch
LABEL_QUESTIONS = [(70, 4, (3, 17)), (71, 6, (5, 6, 7, 40, 41)), (72, 3, (0, 63, 9))]


def pointer_request(config, weights, questions, temperature=1.0):
    """A whole request with the pointer readout: one state row, one branch per question."""
    state = Row(token_ids(config, seed=5, count=STATE_LENGTH), 0)
    rows = [Row(token_ids(config, seed, length), STATE_LENGTH, state, (length - 1, *options)) for seed, length, options in questions]
    return [pointer_probabilities(config, weights, result.hidden[0], result.hidden[1:], temperature)
            for result in run_rows(config, weights, [state, *rows])[1:]]


def label_request(config, weights, questions, temperature=1.0):
    """A whole request with the label-row readout: one state row, one answer-slot branch per question."""
    state = Row(token_ids(config, seed=5, count=STATE_LENGTH), 0)
    rows = [Row(token_ids(config, seed, length), STATE_LENGTH, state, (length - 1,)) for seed, length, _ in questions]
    return [label_row_probabilities(result.hidden[0], label_rows(config, weights, labels), temperature)
            for result, (_, _, labels) in zip(run_rows(config, weights, [state, *rows])[1:], questions)]


@pytest.mark.parametrize("request_fn, questions", [(pointer_request, POINTER_QUESTIONS), (label_request, LABEL_QUESTIONS)])
def test_readout_is_a_distribution_per_question(config, weights, request_fn, questions):
    answers = request_fn(config, weights, questions)
    for probabilities, (_, _, options) in zip(answers, questions):
        assert probabilities.shape == (len(options),)
        assert (probabilities > 0).all() and probabilities.sum() == pytest.approx(1.0, abs=1e-12)
        assert probabilities.max() - probabilities.min() > 1e-6   # not the uniform answer of a dead readout


@pytest.mark.parametrize("request_fn, questions", [(pointer_request, POINTER_QUESTIONS), (label_request, LABEL_QUESTIONS)])
def test_readout_ignores_unrelated_questions(config, weights, request_fn, questions):
    """A question's answer is the same bits alone, with one more question after it, or with others before it."""
    everything = request_fn(config, weights, questions)
    for count in (1, 2):
        for partial, full in zip(request_fn(config, weights, questions[:count]), everything):
            assert np.array_equal(partial, full)
    assert np.array_equal(request_fn(config, weights, questions[::-1])[::-1][0], everything[0])


def test_pointer_logit_formula(config, weights):
    rng = np.random.default_rng(3)
    h_decide, h_options = rng.standard_normal(config.hidden_size), rng.standard_normal((5, config.hidden_size))
    decide = weights["head.q.weight"] @ h_decide + weights["head.q.bias"]
    expected = [(weights["head.k.weight"] @ h + weights["head.k.bias"]) @ decide / sqrt(config.pointer_dim) / 2.5 for h in h_options]
    np.testing.assert_allclose(pointer_logits(config, weights, h_decide, h_options, 2.5), expected, atol=1e-13)


def test_label_row_logit_formula(config, weights):
    h_slot = np.random.default_rng(4).standard_normal(config.hidden_size)
    labels = (9, 2, 30)
    rows = label_rows(config, weights, labels)
    assert np.array_equal(rows, weights["embed_tokens.weight"][list(labels)])
    np.testing.assert_allclose(label_row_logits(h_slot, rows, 0.5), [weights["embed_tokens.weight"][t] @ h_slot / 0.5 for t in labels], atol=1e-13)


def test_temperature_flattens_without_reordering(config, weights):
    sharp, flat = pointer_request(config, weights, POINTER_QUESTIONS, 1.0), pointer_request(config, weights, POINTER_QUESTIONS, 3.0)
    for a, b in zip(sharp, flat):
        assert np.array_equal(np.argsort(a), np.argsort(b)) and b.max() < a.max()
        np.testing.assert_allclose(np.log(b) - np.log(b[0]), (np.log(a) - np.log(a[0])) / 3.0, atol=1e-12)


def test_readouts_refuse_bad_input(config, weights):
    h = np.zeros(config.hidden_size)
    for temperature in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError):
            pointer_logits(config, weights, h, h[None], temperature)
        with pytest.raises(ValueError):
            label_row_logits(h, h[None], temperature)
    for labels in ((), (4, 4), (config.vocab_size,)):
        with pytest.raises(ValueError):
            label_rows(config, weights, labels)
