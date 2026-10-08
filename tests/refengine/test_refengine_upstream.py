"""Parity with the upstream model: a small randomly initialised transformers Qwen3.5 text model against the engine.

Skipped without PyTorch and transformers (dependency group ``verify``). Nothing is downloaded; the comparison with
the real checkpoint is tools/verify_refengine.py. What these tests establish is recorded in
docs/REFENGINE-VERIFICATION.md.
"""
import numpy as np
import pytest
from refengine_support import all_offsets, snapshot_arrays, token_ids

from sepal.refengine import PRESETS, UNVERIFIED, Row, attention, deltanet, run_rows
from sepal.refengine.primitives import sigmoid, softmax

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
from sepal.refengine import upstream  # noqa: E402

PARITY = 1e-9            # float64 against float64: the two computations differ only in the order of operations
FLOAT32_NOISE = 5e-3     # the unmodified upstream model computes its norms, RoPE and delta rule in float32
LENGTHS = (1, 3, 5, 64, 65, 150)   # shorter than the conv kernel; one, just over one and several delta-rule chunks of 64


@pytest.fixture(scope="module", params=["sdpa", "eager"])
def model(request, config):
    model = upstream.random_text_model(config, seed=1)
    model.config._attn_implementation = request.param
    return model


@pytest.fixture(scope="module")
def engine_weights(model, config):
    return upstream.engine_weights(config, model.state_dict())


def engine_hidden(config, weights, ids):
    return run_rows(config, weights, [Row(ids, 0, None, all_offsets(ids))])[0].hidden


def promoted_hidden(model, ids, cache=None, use_cache=False):
    with upstream.float64_arithmetic(model):
        return upstream.hidden_states(model, ids, cache, use_cache)


def test_nothing_is_left_unverified():
    assert UNVERIFIED == ()


def test_tiny_geometry_keeps_the_structure_of_the_real_configs(config):
    real = PRESETS["0.8B"]
    assert (config.full_attention_interval, config.conv_kernel, config.partial_rotary_factor, config.rope_theta) == \
        (real.full_attention_interval, real.conv_kernel, real.partial_rotary_factor, real.rope_theta)
    assert config.layer_kinds.count("attention") >= 1 and config.attention_heads > config.attention_kv_heads > 1
    assert config.deltanet_value_heads > config.deltanet_key_heads > 1     # key-head sharing is exercised


def test_geometry_round_trips_through_the_upstream_config():
    for name in ("0.8B", "2B", "4B"):
        assert upstream.model_config(upstream.text_config(PRESETS[name], tie_word_embeddings=True)) == PRESETS[name]


def test_upstream_runs_its_pure_pytorch_path_here():
    """The numbers below are for whichever path this reports; with the optional kernels installed they would be
    measurements of those kernels instead."""
    assert set(upstream.execution_path().values()) <= {"pure PyTorch fallback", "fla", "causal_conv1d", "kernels"}


@pytest.mark.parametrize("length", LENGTHS)
def test_final_hidden_states_match_upstream_in_float64(model, engine_weights, config, length):
    ids = token_ids(config, seed=length, count=length)
    expected, _ = promoted_hidden(model, ids)
    np.testing.assert_allclose(engine_hidden(config, engine_weights, ids), expected, rtol=0, atol=PARITY)


def test_unmodified_upstream_differs_only_by_its_float32_arithmetic(model, engine_weights, config):
    """Without the promotion the same model is float32-accurate, although its weights are float64."""
    ids = token_ids(config, seed=7, count=150)
    unmodified, _ = upstream.hidden_states(model, ids)
    difference = np.abs(engine_hidden(config, engine_weights, ids) - unmodified).max()
    assert PARITY < difference < FLOAT32_NOISE


def test_float64_promotion_restores_upstream(model):
    import torch
    from transformers.models.qwen3_5 import modeling_qwen3_5

    before = model.rotary_emb.inv_freq
    with upstream.float64_arithmetic(model):
        assert model.rotary_emb.inv_freq.dtype == torch.float64 and torch.ones(1).float().dtype == torch.float64
    assert modeling_qwen3_5.torch is torch and model.rotary_emb.inv_freq is before
    assert torch.ones(1, dtype=torch.float64).float().dtype == torch.float32


# Each reading that used to be listed in UNVERIFIED, replaced by the alternative it was listed against. Parity must
# break for every one of them, or the parity test would not have been able to tell the two readings apart.

def reversed_taps(inputs, kernel, tail, original=deltanet.causal_conv):
    return original(inputs, kernel[:, ::-1], tail)


def clamped_l2(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), deltanet.L2_EPS)


def tiled_key_heads(per_key_head, config):
    return np.tile(per_key_head, (1, config.deltanet_value_heads // config.deltanet_key_heads, 1))


def write_then_decay(state, k, v, decay, beta):
    correction = beta[:, None] * (v - np.einsum("hk,hkv->hv", k, state))
    return decay[:, None, None] * (state + k[:, :, None] * correction[:, None, :])


def read_before_update(q, k, v, decay, beta, state, scale):
    outputs = np.empty(v.shape)
    for t in range(q.shape[0]):
        outputs[t] = scale * np.einsum("hk,hkv->hv", q[t], state)
        state = deltanet.delta_rule_update(state, k[t], v[t], decay[t], beta[t])
    return outputs, state


def angles_over_head_dim(positions, config):
    frequency = config.rope_theta ** (-np.arange(0, config.rotary_dim, 2, dtype=np.float64) / config.attention_head_dim)
    return positions[:, None] * frequency[None, :]


def interleaved_rope(x, angles):
    rotary = 2 * angles.shape[1]
    cos, sin = np.cos(angles)[:, None, :], np.sin(angles)[:, None, :]
    even, odd = x[..., 0:rotary:2], x[..., 1:rotary:2]
    rotated = np.stack([even * cos - odd * sin, odd * cos + even * sin], axis=-1).reshape(*x.shape[:-1], rotary)
    return np.concatenate([rotated, x[..., rotary:]], axis=-1)


def tiled_query_attention(q, keys, values, context, scale):
    tokens, heads, _ = q.shape
    visible = np.arange(keys.shape[0])[None, :] <= context + np.arange(tokens)[:, None]
    output = np.empty(q.shape)
    for head in range(heads):
        scores = scale * (q[:, head] @ keys[:, head % keys.shape[1]].T)
        output[:, head] = softmax(np.where(visible, scores, -np.inf)) @ values[:, head % keys.shape[1]]
    return output


ALTERNATIVES = {
    "deltanet.causal_conv": (deltanet, reversed_taps),
    "deltanet.l2_normalise": (deltanet, clamped_l2),
    "deltanet.share_key_heads": (deltanet, tiled_key_heads),
    "deltanet.delta_rule_update": (deltanet, write_then_decay),
    "deltanet.delta_rule_scan": (deltanet, read_before_update),
    "deltanet.delta_rule_output_scale": (deltanet, lambda config: 1.0),
    "deltanet.gated_norm_scale": (deltanet, lambda weight: 1.0 + weight),
    "deltanet.gated_norm_gate": (deltanet, sigmoid),
    "attention.rope_angles": (attention, angles_over_head_dim),
    "attention.apply_rope": (attention, interleaved_rope),
    "attention.attention_scale": (attention, lambda config: 1.0 / config.attention_head_dim),
    "attention.grouped_query_attention": (attention, tiled_query_attention),
}


@pytest.mark.parametrize("name", ALTERNATIVES)
def test_the_other_reading_of_each_settled_detail_breaks_parity(model, engine_weights, config, monkeypatch, name):
    ids = token_ids(config, seed=5, count=40)
    expected, _ = promoted_hidden(model, ids)
    module, alternative = ALTERNATIVES[name]
    monkeypatch.setattr(module, name.split(".")[1], alternative)
    assert np.abs(engine_hidden(config, engine_weights, ids) - expected).max() > 1000 * PARITY


# Snapshot semantics: the upstream cache against the engine's snapshot.

@pytest.mark.parametrize("split", [1, 2, 3, 37, 64, 89])
def test_upstream_cache_continuation_and_engine_snapshot_agree(model, engine_weights, config, split):
    """Upstream: prefix, keep the cache, suffix == one pass. The cache holds what the engine's snapshot holds, and
    the engine continues the upstream cache to the same hidden states. A suffix of one token takes upstream's
    token-by-token delta rule and conv update, a longer one its chunked rule: split 89 of 90 covers the former."""
    ids = token_ids(config, seed=9, count=90)
    whole, _ = promoted_hidden(model, ids)
    with upstream.float64_arithmetic(model):
        prefix, cache = upstream.hidden_states(model, ids[:split], use_cache=True)
        from_cache = upstream.snapshot_from_cache(config, cache)      # copied before the suffix updates it in place
        suffix, _ = upstream.hidden_states(model, ids[split:], cache)
    np.testing.assert_allclose(np.concatenate([prefix, suffix]), whole, rtol=0, atol=PARITY)

    own = run_rows(config, engine_weights, [Row(ids[:split], 0)])[0].snapshot
    assert from_cache.n_tokens == own.n_tokens == split
    for theirs, ours in zip(snapshot_arrays(from_cache), snapshot_arrays(own), strict=True):
        np.testing.assert_allclose(ours, theirs, rtol=0, atol=PARITY)

    continued = run_rows(config, engine_weights, [Row(ids[split:], split, from_cache, all_offsets(ids[split:]))])[0].hidden
    np.testing.assert_allclose(continued, whole[split:], rtol=0, atol=PARITY)


def test_the_upstream_cache_carries_no_state_the_snapshot_lacks(model, config):
    """Per DeltaNet layer one conv state and one recurrent state, per attention layer keys and values, nothing else.
    The conv state has K columns where the snapshot has K-1: the oldest is never read again (checked by zeroing)."""
    ids = token_ids(config, seed=13, count=20)
    _, cache = promoted_hidden(model, ids[:11], use_cache=True)
    expected = {
        "deltanet": {"conv_states[0]": (1, config.conv_dim, config.conv_kernel),
                     "recurrent_states[0]": (1, config.deltanet_value_heads, config.deltanet_key_head_dim, config.deltanet_value_head_dim)},
        "attention": {"keys": (1, config.attention_kv_heads, 11, config.attention_head_dim),
                      "values": (1, config.attention_kv_heads, 11, config.attention_head_dim)},
    }
    assert upstream.cache_state_names(cache) == [expected[kind] for kind in config.layer_kinds]

    for suffix in (ids[11:], ids[11:12]):     # the multi-token and the single-token cached paths
        outputs = []
        for zero_oldest in (False, True):
            _, cache = promoted_hidden(model, ids[:11], use_cache=True)
            if zero_oldest:
                for kind, layer in zip(config.layer_kinds, cache.layers, strict=True):
                    if kind == "deltanet":
                        layer.conv_states[0][..., 0] = 0.0
            outputs.append(promoted_hidden(model, suffix, cache)[0])
        np.testing.assert_array_equal(outputs[0], outputs[1])


def test_short_prefix_leaves_a_zero_padded_conv_state(model, config):
    """After fewer tokens than the kernel, upstream's conv state is left-padded with zeros, like the engine's tail."""
    ids = token_ids(config, seed=17, count=2)
    _, cache = promoted_hidden(model, ids, use_cache=True)
    conv = cache.layers[0].conv_states[0][0].numpy()
    assert not conv[:, :config.conv_kernel - 2].any() and conv[:, config.conv_kernel - 2:].all()
