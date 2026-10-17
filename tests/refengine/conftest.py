"""Shared fixtures: the tiny geometry and one set of seeded random weights for it."""
import pytest

from sepal_runner.refengine import PRESETS, ModelConfig, Weights, random_weights


@pytest.fixture(scope="session")
def config() -> ModelConfig:
    return PRESETS["tiny"]


@pytest.fixture(scope="session")
def weights(config: ModelConfig) -> Weights:
    return random_weights(config, seed=0)
