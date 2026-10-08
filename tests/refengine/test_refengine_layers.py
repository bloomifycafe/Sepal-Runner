"""Each layer function against an independent transcription of its formula, and the layer-level invariants."""
import numpy as np
import pytest

from refengine_support import EXACT
from sepal.refengine.attention import apply_rope, attention_mixer, grouped_query_attention, rope_angles, unpack_query_and_gate
from sepal.refengine.deltanet import causal_conv, delta_rule_scan, deltanet_mixer, l2_normalise, share_key_heads
from sepal.refengine.primitives import project, rms_norm, sigmoid, silu, softmax, softplus, swiglu_mlp
from sepal.refengine.snapshot import empty_layer_state

RNG_SEED = 7


@pytest.fixture
def rng():
    return np.random.default_rng(RNG_SEED)


def test_elementwise_functions_are_stable_and_exact():
    x = np.array([-800.0, -2.0, 0.0, 3.0, 800.0])
    np.testing.assert_allclose(sigmoid(x), [0.0, 1 / (1 + np.exp(2.0)), 0.5, 1 / (1 + np.exp(-3.0)), 1.0], atol=1e-15)
    np.testing.assert_allclose(softplus(x), [0.0, np.log1p(np.exp(-2.0)), np.log(2.0), np.log1p(np.exp(3.0)), 800.0], atol=1e-13)
    np.testing.assert_allclose(silu(x), x * sigmoid(x))
    assert softmax(np.array([0.0, -np.inf, 0.0])).tolist() == [0.5, 0.0, 0.5]


def test_rms_norm_is_zero_centred(rng):
    x, w = rng.standard_normal((5, 8)), 0.1 * rng.standard_normal(8)
    unit = rms_norm(x, np.zeros(8), eps=0.0)
    np.testing.assert_allclose(np.sqrt(np.mean(unit**2, axis=-1)), 1.0)       # a zero weight is the identity scale
    np.testing.assert_allclose(rms_norm(x, w, 1e-6), x / np.sqrt((x**2).mean(-1, keepdims=True) + 1e-6) * (1 + w))
    assert not rms_norm(np.zeros((2, 8)), w, 1e-6).any()


def test_project_is_the_separate_projections(rng):
    x, a, b = rng.standard_normal((6, 5)), rng.standard_normal((3, 5)), rng.standard_normal((4, 5))
    pa, pb = project(x, [a, b])
    np.testing.assert_allclose(pa, x @ a.T, atol=1e-14)
    np.testing.assert_allclose(pb, x @ b.T, atol=1e-14)


def test_swiglu(rng):
    x, gate, up, down = rng.standard_normal((4, 5)), rng.standard_normal((7, 5)), rng.standard_normal((7, 5)), rng.standard_normal((5, 7))
    g = x @ gate.T
    np.testing.assert_allclose(swiglu_mlp(x, gate, up, down), (g / (1 + np.exp(-g)) * (x @ up.T)) @ down.T, atol=1e-13)


def test_causal_conv_matches_a_loop_and_continues_from_its_tail(rng):
    channels, taps, tokens = 3, 4, 9
    inputs, kernel, tail = rng.standard_normal((tokens, channels)), rng.standard_normal((channels, taps)), rng.standard_normal((taps - 1, channels))
    outputs, new_tail = causal_conv(inputs, kernel, tail)
    history = np.concatenate([tail, inputs])
    for t in range(tokens):
        for c in range(channels):   # the window of `taps` inputs ending at token t, last tap on token t itself
            assert outputs[t, c] == pytest.approx(sum(kernel[c, j] * history[t + j, c] for j in range(taps)), abs=1e-13)
    assert np.array_equal(new_tail, inputs[-(taps - 1):])
    for split in range(1, tokens):
        head, mid_tail = causal_conv(inputs[:split], kernel, tail)
        rest, end_tail = causal_conv(inputs[split:], kernel, mid_tail)
        np.testing.assert_allclose(np.concatenate([head, rest]), outputs, atol=1e-13)
        assert np.array_equal(end_tail, new_tail)   # a run shorter than the tail keeps part of the old tail


def test_l2_normalise(rng):
    x = rng.standard_normal((4, 2, 6))
    np.testing.assert_allclose(np.linalg.norm(l2_normalise(x), axis=-1), 1.0, atol=1e-6)
    assert not l2_normalise(np.zeros((1, 2, 6))).any()


def test_key_heads_are_shared_by_consecutive_value_heads(config, rng):
    k = rng.standard_normal((3, config.deltanet_key_heads, config.deltanet_key_head_dim))
    shared = share_key_heads(k, config)
    ratio = config.deltanet_value_heads // config.deltanet_key_heads
    assert shared.shape[1] == config.deltanet_value_heads
    assert all(np.array_equal(shared[:, h], k[:, h // ratio]) for h in range(config.deltanet_value_heads))


def test_delta_rule_scan_is_the_standard_gated_delta_rule(rng):
    """Against the matrix form of docs/RUNNER.md section 3, with S as [value dim, key dim]:
    S_t = decay_t * S_{t-1} (I - beta_t k_t k_t^T) + beta_t v_t k_t^T;  o_t = S_t q_t * scale."""
    tokens, heads, dk, dv, scale = 6, 3, 4, 5, 0.37
    q, k, v = rng.standard_normal((tokens, heads, dk)), rng.standard_normal((tokens, heads, dk)), rng.standard_normal((tokens, heads, dv))
    decay, beta = rng.uniform(0.2, 1.0, (tokens, heads)), rng.uniform(0.0, 1.0, (tokens, heads))
    start = rng.standard_normal((heads, dk, dv))
    outputs, final = delta_rule_scan(q, k, v, decay, beta, start, scale)
    for h in range(heads):
        s = start[h].T.copy()
        for t in range(tokens):
            s = decay[t, h] * s @ (np.eye(dk) - beta[t, h] * np.outer(k[t, h], k[t, h])) + beta[t, h] * np.outer(v[t, h], k[t, h])
            np.testing.assert_allclose(outputs[t, h], scale * (s @ q[t, h]), atol=1e-12)
        np.testing.assert_allclose(final[h], s.T, atol=1e-12)


def test_delta_rule_stores_and_recalls_a_value():
    """With beta = 1, no decay and a unit key, the memory returns exactly the value just written under that key."""
    k = np.array([[[0.6, 0.8]]])
    v = np.array([[[1.0, -2.0, 3.0]]])
    outputs, _ = delta_rule_scan(k, k, v, np.ones((1, 1)), np.ones((1, 1)), np.full((1, 2, 3), 0.5), scale=1.0)
    np.testing.assert_allclose(outputs, v, atol=1e-15)


def deltanet_inputs(config, weights, rng, tokens):
    return rng.standard_normal((tokens, config.hidden_size)), weights.scope("layers.0.linear_attn"), empty_layer_state(config, 0)


def test_deltanet_mixer_continues_from_its_state_at_every_split(config, weights, rng):
    x, scope, empty = deltanet_inputs(config, weights, rng, 9)
    whole, whole_state = deltanet_mixer(x, scope, empty, config)
    assert whole.shape == x.shape and np.abs(whole).max() > 1e-3
    for split in range(1, len(x)):
        head, mid = deltanet_mixer(x[:split], scope, empty, config)
        rest, end = deltanet_mixer(x[split:], scope, mid, config)
        np.testing.assert_allclose(np.concatenate([head, rest]), whole, rtol=0, atol=EXACT)
        np.testing.assert_allclose(end.recurrent, whole_state.recurrent, rtol=0, atol=EXACT)
        np.testing.assert_allclose(end.conv_tail, whole_state.conv_tail, rtol=0, atol=EXACT)


def test_deltanet_zero_input_left_padding_is_exact(config, weights, rng):
    """Zero-input positions before the first real token leave the recurrent state, the conv tail and the output at
    zero, so a left-padded run ends in the state of the unpadded one."""
    x, scope, empty = deltanet_inputs(config, weights, rng, 6)
    plain, plain_state = deltanet_mixer(x, scope, empty, config)
    for pad in (1, 2, 5):
        padding = np.zeros((pad, config.hidden_size))
        pad_out, pad_state = deltanet_mixer(padding, scope, empty, config)
        assert not pad_out.any() and not pad_state.recurrent.any() and not pad_state.conv_tail.any()
        padded, padded_state = deltanet_mixer(np.concatenate([padding, x]), scope, empty, config)
        assert not padded[:pad].any()
        np.testing.assert_allclose(padded[pad:], plain, rtol=0, atol=EXACT)
        np.testing.assert_allclose(padded_state.recurrent, plain_state.recurrent, rtol=0, atol=EXACT)
        np.testing.assert_allclose(padded_state.conv_tail, plain_state.conv_tail, rtol=0, atol=EXACT)


def test_deltanet_state_has_the_snapshot_shapes_and_is_read_only(config, weights, rng):
    x, scope, empty = deltanet_inputs(config, weights, rng, 2)   # shorter than the conv tail
    _, state = deltanet_mixer(x, scope, empty, config)
    assert state.recurrent.shape == (config.deltanet_value_heads, config.deltanet_key_head_dim, config.deltanet_value_head_dim)
    assert state.conv_tail.shape == (config.conv_kernel - 1, config.conv_dim)
    assert not state.conv_tail[0].any() and state.conv_tail[1:].any()   # one slot of the old (zero) tail is kept
    with pytest.raises(ValueError):
        state.recurrent[0, 0, 0] = 1.0


def test_query_and_gate_are_packed_per_head(config):
    width = 2 * config.attention_width
    q, gate = unpack_query_and_gate(np.arange(2.0 * width).reshape(2, width), config)
    head_dim = config.attention_head_dim
    assert q.shape == gate.shape == (2, config.attention_heads, head_dim)
    assert q[0, 1].tolist() == list(range(2 * head_dim, 3 * head_dim))
    assert gate[0, 1].tolist() == list(range(3 * head_dim, 4 * head_dim))


def test_rope_rotates_only_the_rotary_part_and_is_relative(config, rng):
    rotary, tokens = config.rotary_dim, 5
    x, y = rng.standard_normal((2, tokens, 3, config.attention_head_dim))
    positions = np.arange(tokens, dtype=np.float64) + 11
    rx, ry = apply_rope(x, rope_angles(positions, config)), apply_rope(y, rope_angles(positions, config))
    assert rope_angles(positions, config).shape == (tokens, rotary // 2)
    assert np.array_equal(rx[..., rotary:], x[..., rotary:])
    assert not np.allclose(rx[..., :rotary], x[..., :rotary])
    np.testing.assert_allclose(np.linalg.norm(rx, axis=-1), np.linalg.norm(x, axis=-1))
    assert np.array_equal(apply_rope(x, rope_angles(np.zeros(tokens), config)), x)
    # the dot product of two rotated vectors depends on the difference of their positions only
    shifted = rope_angles(positions + 1000, config)
    np.testing.assert_allclose(np.einsum("thd,shd->tsh", rx, ry),
                               np.einsum("thd,shd->tsh", apply_rope(x, shifted), apply_rope(y, shifted)), atol=1e-9)


def test_rope_frequencies(config):
    angles = rope_angles(np.array([3.0]), config)
    rotary = config.rotary_dim
    assert angles[0].tolist() == pytest.approx([3.0 * config.rope_theta ** (-2 * i / rotary) for i in range(rotary // 2)])


def test_grouped_query_attention_matches_a_per_token_loop(rng):
    tokens, context, heads, kv_heads, dim, scale = 4, 3, 4, 2, 6, 0.41
    q = rng.standard_normal((tokens, heads, dim))
    keys, values = rng.standard_normal((2, context + tokens, kv_heads, dim))
    output = grouped_query_attention(q, keys, values, context, scale)
    for t in range(tokens):
        seen = context + t + 1   # the whole context and own tokens up to and including t
        for h in range(heads):
            g = h // (heads // kv_heads)
            scores = np.array([scale * q[t, h] @ keys[j, g] for j in range(seen)])
            probabilities = np.exp(scores) / np.exp(scores).sum()
            np.testing.assert_allclose(output[t, h], probabilities @ values[:seen, g], atol=1e-13)


def attention_inputs(config, weights, rng, tokens):
    return rng.standard_normal((tokens, config.hidden_size)), weights.scope("layers.3.self_attn"), empty_layer_state(config, 3)


def test_attention_mixer_continues_from_its_keys_and_values_at_every_split(config, weights, rng):
    x, scope, empty = attention_inputs(config, weights, rng, 8)
    positions = np.arange(len(x), dtype=np.float64)
    whole, whole_state = attention_mixer(x, positions, scope, empty, config)
    assert whole_state.keys.shape == whole_state.values.shape == (len(x), config.attention_kv_heads, config.attention_head_dim)
    for split in range(1, len(x)):
        head, mid = attention_mixer(x[:split], positions[:split], scope, empty, config)
        rest, end = attention_mixer(x[split:], positions[split:], scope, mid, config)
        np.testing.assert_allclose(np.concatenate([head, rest]), whole, rtol=0, atol=EXACT)
        np.testing.assert_allclose(end.keys, whole_state.keys, rtol=0, atol=EXACT)
        np.testing.assert_allclose(end.values, whole_state.values, rtol=0, atol=EXACT)
        assert mid.keys.shape[0] == split   # the parent's state was not extended by the child


def test_attention_mixer_is_causal_and_position_dependent(config, weights, rng):
    x, scope, empty = attention_inputs(config, weights, rng, 6)
    positions = np.arange(len(x), dtype=np.float64)
    base, _ = attention_mixer(x, positions, scope, empty, config)
    changed = x.copy()
    changed[4] += 1.0
    after, _ = attention_mixer(changed, positions, scope, empty, config)
    np.testing.assert_allclose(after[:4], base[:4], rtol=0, atol=EXACT)
    assert np.abs(after[4:] - base[4:]).max() > 1e-3
    stretched, _ = attention_mixer(x, 3 * positions, scope, empty, config)   # another relative layout: another answer
    assert np.abs(stretched - base).max() > 1e-6
