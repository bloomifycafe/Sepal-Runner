"""Geometry, parameter counts and the weight container."""
import dataclasses
import importlib

import numpy as np
import pytest

from sepal_runner.refengine import PRESETS, REAL_PRESETS, UNVERIFIED, Weights, parameter_counts, random_weights, weight_shapes

# docs/RUNNER.md section 3: non-embedding parameters and the layer split, read from the base model files
PUBLISHED = {"0.8B": (498e6, 18, 6), "2B": (1.37e9, 18, 6), "4B": (3.57e9, 24, 8), "9B": (6.92e9, 24, 8)}


@pytest.mark.parametrize("name", REAL_PRESETS)
def test_real_preset_parameter_count_matches_published(name):
    config = PRESETS[name]
    counts = parameter_counts(config)
    published, _, _ = PUBLISHED[name]
    print(f"{name}: non-embedding {counts.non_embedding:,}, embedding {counts.embedding:,}, lm head {counts.lm_head:,}, "
          f"base model {counts.base_model:,}, pointer head {counts.pointer_head:,}")
    assert abs(counts.non_embedding - published) / published < 0.01
    assert counts.embedding == 248_320 * config.hidden_size
    assert counts.pointer_head == 2 * (config.hidden_size * 256 + 256)


@pytest.mark.parametrize("name", REAL_PRESETS)
def test_real_preset_layer_split(name):
    kinds = PRESETS[name].layer_kinds
    _, deltanet, attention = PUBLISHED[name]
    assert (kinds.count("deltanet"), kinds.count("attention")) == (deltanet, attention)
    assert all((kind == "attention") == ((i + 1) % 4 == 0) for i, kind in enumerate(kinds))
    assert kinds[-1] == "attention"


def test_only_9b_has_a_separate_lm_head():
    assert [parameter_counts(PRESETS[name]).lm_head for name in REAL_PRESETS] == [0, 0, 0, 248_320 * 4096]


def test_real_geometry_traps():
    """Attention width differs from the hidden size at 0.8B and 4B, and q_proj is twice the attention width."""
    for name, width in (("0.8B", 2048), ("2B", 2048), ("4B", 4096), ("9B", 4096)):
        config = PRESETS[name]
        shapes = weight_shapes(config)
        assert config.attention_width == width and config.rotary_dim == 64
        assert shapes["layers.3.self_attn.q_proj.weight"] == (2 * width, config.hidden_size)
        assert shapes["layers.0.linear_attn.in_proj_qkv.weight"] == (config.conv_dim, config.hidden_size)
    assert PRESETS["0.8B"].attention_width != PRESETS["0.8B"].hidden_size
    assert PRESETS["4B"].attention_width != PRESETS["4B"].hidden_size
    assert (PRESETS["0.8B"].conv_dim, PRESETS["4B"].conv_dim) == (6144, 8192)


def test_tiny_preset_keeps_coinciding_sizes_apart(config):
    assert config.deltanet_key_head_dim != config.deltanet_value_head_dim
    assert config.deltanet_key_heads != config.deltanet_value_heads
    assert config.attention_width != config.hidden_size
    assert config.layer_kinds.count("attention") == 2


@pytest.mark.parametrize("change", [{"attention_kv_heads": 3}, {"deltanet_key_heads": 3}, {"partial_rotary_factor": 0.0},
                                    {"partial_rotary_factor": 0.2}, {"conv_kernel": 0}])
def test_inconsistent_geometry_is_refused(config, change):
    with pytest.raises(ValueError):
        dataclasses.replace(config, **change)


def test_random_weights_are_seeded(config, weights):
    again, other = random_weights(config, seed=0), random_weights(config, seed=1)
    assert set(weights) == set(weight_shapes(config))
    assert all(np.array_equal(weights[name], again[name]) for name in weights)
    assert any(not np.array_equal(weights[name], other[name]) for name in weights)


def test_loader_converts_copies_and_freezes(config, weights):
    source = {name: np.array(tensor, dtype=np.float32) for name, tensor in weights.items()}
    loaded = Weights.from_arrays(config, source)
    name = "layers.0.mlp.gate_proj.weight"
    kept = loaded[name].copy()
    source[name][:] = 0.0
    assert loaded[name].dtype == np.float64 and np.array_equal(loaded[name], kept)
    with pytest.raises(ValueError):
        loaded[name][0, 0] = 1.0


def test_loader_is_strict(config, weights):
    arrays = dict(weights)
    name = "layers.3.self_attn.q_proj.weight"
    with pytest.raises(ValueError, match="missing"):
        Weights.from_arrays(config, {k: v for k, v in arrays.items() if k != name})
    with pytest.raises(ValueError, match="unexpected"):
        Weights.from_arrays(config, arrays | {"lm_head.weight": np.zeros((config.vocab_size, config.hidden_size))})
    with pytest.raises(ValueError, match="another shape"):
        Weights.from_arrays(config, arrays | {name: arrays[name][: config.attention_width]})


def test_unverified_entries_name_real_functions():
    assert len({name for name, _ in UNVERIFIED}) == len(UNVERIFIED)
    for name, description in UNVERIFIED:
        module, function = name.split(".")
        assert callable(getattr(importlib.import_module(f"sepal_runner.refengine.{module}"), function))
        assert description
