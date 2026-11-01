"""`sepal serve` planning: which profile and limits a machine gets, and when a model is refused. No weights, no GPU needed."""
from types import SimpleNamespace

import pytest

from sepal_runner import cli

GIB = 1 << 30
FOUR_B = cli.Shape(trunk_bytes=int(7.14e9), hidden=2560, kv_bytes_per_token=32768, slot_bytes=int(50.3 * (1 << 20)))
SMALL = cli.Shape(trunk_bytes=int(1.0e9), hidden=1024, kv_bytes_per_token=12288, slot_bytes=int(20 * (1 << 20)))


def args(**kw):
    base = dict(profile="auto", allow_q8w=False, lib="x_cpu.dll")
    base.update(kw)
    return SimpleNamespace(**base)


@pytest.fixture(autouse=True)
def libraries(monkeypatch):
    monkeypatch.setattr(cli, "find_library", lambda kind, explicit: f"/lib/{kind}")


def hw(ram=32, gpu=None):
    return cli.Hardware(6, 12, ram * GIB, ram * GIB, *( (gpu[0], gpu[1] * GIB, gpu[2] * GIB) if gpu else (None, 0, 0)))


def test_cpu_only_gets_cpu_bf16():
    plan = cli.make_plan(FOUR_B, hw(32), 24576, args())
    assert plan.profile == "cpu-bf16" and plan.window == 24576 and plan.engine_options["threads"] == 6


def test_12gb_card_gets_gpu_f16_with_a_window_that_fits():
    plan = cli.make_plan(FOUR_B, hw(32, ("RTX 3060", 12, 11.2)), 24576, args())
    assert plan.profile == "gpu-f16" and plan.window <= 24576 and plan.engine_options["trunk_budget_mib"] > 8000


def test_8gb_card_falls_back_to_cpu_and_never_picks_q8w_unasked():
    plan = cli.make_plan(FOUR_B, hw(32, ("RTX 3070", 8, 7.5)), 24576, args())
    assert plan.profile == "cpu-bf16" and any("GPU not used" in n for n in plan.notes)


def test_q8w_needs_the_flag_and_prints_a_warning():
    with pytest.raises(cli.Refusal):
        cli.make_plan(FOUR_B, hw(32, ("RTX 3070", 8, 7.5)), 24576, args(profile="gpu-q8w"))
    plan = cli.make_plan(FOUR_B, hw(32, ("RTX 3070", 8, 7.5)), 24576, args(profile="gpu-q8w", allow_q8w=True))
    assert plan.profile == "gpu-q8w" and any("WARNING" in n for n in plan.notes)


def test_cpu_q8_is_never_served():
    with pytest.raises(cli.Refusal):
        cli.make_plan(FOUR_B, hw(32), 24576, args(profile="cpu-q8"))


def test_model_that_does_not_fit_is_refused_with_what_would_fit():
    with pytest.raises(cli.Refusal) as e:
        cli.make_plan(FOUR_B, hw(8), 24576, args())
    assert "does not fit" in str(e.value) and "0.8B" in str(e.value)


def test_small_model_fits_8gb():
    assert cli.make_plan(SMALL, hw(8), 16384, args()).profile == "cpu-bf16"
