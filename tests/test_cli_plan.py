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


def hw(ram=32, gpu=None, cc=None):
    """A machine with `ram` GiB and optionally a card (name, total GiB, free GiB) of compute capability `cc` (None: the driver did not say)."""
    return cli.Hardware(6, 12, ram * GIB, ram * GIB, *( (gpu[0], gpu[1] * GIB, gpu[2] * GIB) if gpu else (None, 0, 0)), compute_cap=cc)


GTX_1080_TI = ("GeForce GTX 1080 Ti", 11, 10.8)
GTX_1080 = ("GeForce GTX 1080", 8, 7.5)
GTX_1070 = ("GeForce GTX 1070", 8, 7.5)
GTX_1060 = ("GeForce GTX 1060 6GB", 6, 5.6)
TESLA_P40 = ("Tesla P40", 24, 23.5)
RTX_3060 = ("RTX 3060", 12, 11.2)
RTX_3070 = ("RTX 3070", 8, 7.5)


def test_cpu_only_gets_cpu_bf16():
    plan = cli.make_plan(FOUR_B, hw(32), 24576, args())
    # The engine counts its threads (physical cores of the allowed CPUs, capped by a cgroup quota): the plan does not guess them.
    assert plan.profile == "cpu-bf16" and plan.window == 24576 and "threads" not in plan.engine_options


def test_cpu_threads_cpus_and_kernel_set_pass_through():
    plan = cli.make_plan(FOUR_B, hw(32), 24576, args(threads=12, cpus="8-19", cpu_isa="avx2-tuned"))
    assert plan.engine_options["threads"] == 12 and plan.engine_options["cpus"] == "8-19" and plan.engine_options["cpu_isa"] == "avx2-tuned"
    assert "cpu_isa" not in cli.make_plan(FOUR_B, hw(32), 24576, args(cpu_isa="auto")).engine_options


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


def test_cpu_q8_needs_the_flag_and_fits_where_bf16_does_not():
    with pytest.raises(cli.Refusal):
        cli.make_plan(FOUR_B, hw(32), 24576, args(profile="cpu-q8"))
    plan = cli.make_plan(FOUR_B, hw(9), 24576, args(allow_q8w=True))     # 8 GiB free: bf16 needs about 9 GiB, int8 about 6.5
    assert plan.profile == "cpu-q8" and any("WARNING" in n for n in plan.notes)


def test_cpu_plan_turns_on_the_prefix_cache_when_ram_is_spare():
    assert cli.make_plan(FOUR_B, hw(32), 24576, args()).engine_options["prefix_cache_mib"] > 0
    assert "prefix_cache_mib" not in cli.make_plan(SMALL, hw(3), 16384, args()).engine_options


def test_model_that_does_not_fit_is_refused_with_what_would_fit():
    with pytest.raises(cli.Refusal) as e:
        cli.make_plan(FOUR_B, hw(8), 24576, args())
    assert "does not fit" in str(e.value) and "0.8B" in str(e.value)


def test_small_model_fits_8gb():
    assert cli.make_plan(SMALL, hw(8), 16384, args()).profile == "cpu-bf16"


# ---------------------------------------------------------------- Pascal: gpu-bf16 and gpu-w8g (compute capability below 8.0)

def test_1080_ti_gets_gpu_bf16():
    plan = cli.make_plan(FOUR_B, hw(32, GTX_1080_TI, cc=6.1), 24576, args())
    assert plan.profile == "gpu-bf16" and 4096 <= plan.window <= 24576
    assert plan.engine_options["max_call_tokens"] == 2048 and plan.engine_options["trunk_budget_mib"] > 7000
    assert not any("WARNING" in n or "int8" in n for n in plan.notes)          # bfloat16 weights, nothing to warn about


@pytest.mark.parametrize("card", [GTX_1080, GTX_1070])
def test_8gb_pascal_cards_get_gpu_w8g_with_a_note_that_it_stores_int8(card):
    plan = cli.make_plan(FOUR_B, hw(32, card, cc=6.1), 24576, args())
    assert plan.profile == "gpu-w8g" and plan.window >= 4096 and plan.engine_options["max_call_tokens"] == 2048
    assert any("int8 weights" in n and "128" in n for n in plan.notes)
    assert plan.engine_options["trunk_budget_mib"] < 7500                       # inside the 7.5 GiB the card has free


def test_p40_gets_gpu_bf16_with_the_whole_window():
    plan = cli.make_plan(FOUR_B, hw(32, TESLA_P40, cc=6.1), 24576, args())
    assert plan.profile == "gpu-bf16" and plan.window == 24576 and plan.engine_options["max_call_tokens"] == 2048


def test_small_model_on_a_pascal_card_gets_gpu_bf16():
    assert cli.make_plan(SMALL, hw(8, GTX_1070, cc=6.1), 16384, args()).profile == "gpu-bf16"


def test_pascal_card_too_small_for_both_falls_back_to_cpu_and_names_both():
    plan = cli.make_plan(FOUR_B, hw(32, GTX_1060, cc=6.1), 24576, args())
    note = next(n for n in plan.notes if "GPU not used" in n)
    assert plan.profile == "cpu-bf16" and "gpu-bf16 needs" in note and "gpu-w8g needs" in note


@pytest.mark.parametrize("profile", ["gpu-f16", "gpu-q8w"])
def test_binary16_profiles_are_refused_on_a_pascal_card_naming_the_ones_that_run(profile):
    for allow in (False, True):
        with pytest.raises(cli.Refusal) as e:
            cli.make_plan(FOUR_B, hw(32, GTX_1080_TI, cc=6.1), 24576, args(profile=profile, allow_q8w=allow))
        message = str(e.value)
        assert "compute capability 8.0" in message and "6.1" in message and "gpu-bf16" in message and "gpu-w8g" in message


def test_explicit_gpu_bf16_and_gpu_w8g_work_on_any_cuda_card():
    for cc in (6.1, 8.6, None):
        assert cli.make_plan(FOUR_B, hw(32, TESLA_P40, cc=cc), 24576, args(profile="gpu-bf16")).profile == "gpu-bf16"
        assert cli.make_plan(FOUR_B, hw(32, GTX_1080, cc=cc), 24576, args(profile="gpu-w8g")).profile == "gpu-w8g"
    plan = cli.make_plan(FOUR_B, hw(32, RTX_3060, cc=8.6), 24576, args(profile="gpu-w8g"))      # not behind --allow-q8
    assert plan.profile == "gpu-w8g" and any("int8 weights" in n for n in plan.notes)
    assert not any("WARNING" in n for n in plan.notes)


def test_explicit_gpu_bf16_that_does_not_fit_points_at_gpu_w8g():
    with pytest.raises(cli.Refusal) as e:
        cli.make_plan(FOUR_B, hw(32, GTX_1080, cc=6.1), 24576, args(profile="gpu-bf16"))
    assert "does not fit" in str(e.value) and "gpu-w8g" in str(e.value)


def test_explicit_gpu_profiles_need_a_cuda_card():
    for profile in cli.GPU_PROFILES:
        with pytest.raises(cli.Refusal):
            cli.make_plan(FOUR_B, hw(32), 24576, args(profile=profile, allow_q8w=True))


# ---------------------------------------------------------------- compute capability 8.0 and later, or unknown: nothing changes

def test_ampere_12gb_card_still_gets_gpu_f16_and_the_same_plan_as_before():
    plan = cli.make_plan(FOUR_B, hw(32, RTX_3060, cc=8.6), 24576, args())
    assert plan.profile == "gpu-f16" and plan.window <= 24576 and plan.engine_options["trunk_budget_mib"] > 8000
    assert plan == cli.make_plan(FOUR_B, hw(32, RTX_3060, cc=None), 24576, args())     # an unknown capability is a modern card
    assert plan == cli.make_plan(FOUR_B, hw(32, RTX_3060, cc=12.0), 24576, args())


def test_ampere_8gb_card_still_falls_back_to_cpu_and_q8w_needs_the_flag():
    plan = cli.make_plan(FOUR_B, hw(32, RTX_3070, cc=8.6), 24576, args())
    assert plan.profile == "cpu-bf16" and any("GPU not used" in n for n in plan.notes)
    q8 = cli.make_plan(FOUR_B, hw(32, RTX_3070, cc=8.6), 24576, args(allow_q8w=True))
    assert q8.profile == "gpu-q8w" and any("WARNING" in n for n in q8.notes)


def test_unknown_capability_on_a_12gb_card_is_taken_as_modern():
    assert cli.make_plan(FOUR_B, hw(32, GTX_1080_TI, cc=None), 24576, args()).profile == "gpu-f16"


def test_existing_profiles_keep_their_device_figures():
    assert cli.gpu_need(FOUR_B, "gpu-f16", 24576, 4096) == (10383903821, 1932735283)
    assert cli.gpu_need(FOUR_B, "gpu-q8w", 16384, 2048) == (6135083473, 1932735283)


def test_gpu_bf16_and_gpu_w8g_device_figures():
    bf16, outside = cli.gpu_need(FOUR_B, "gpu-bf16", 24576, 2048)
    w8g, _ = cli.gpu_need(FOUR_B, "gpu-w8g", 24576, 2048)
    # weights 6.65 GiB (bf16) / 3.69e9 bytes (int8 + scales), buffers 174 KB a call token, float32 K/V 64 KiB a token, 4 slots of 50.3 MiB.
    extra = 2048 * 174 * 1024 + int(1.15 * 24576 * 65536) + 4 * FOUR_B.slot_bytes + (64 << 20)
    assert abs(bf16 - (7.14e9 + extra)) < 1e6 and abs(w8g - (3.69e9 + extra)) < 5e7
    assert outside == cli.gpu_need(FOUR_B, "gpu-f16", 24576, 2048)[1]
    # float32 keys and values cost twice the binary16 bytes per token of context
    assert cli.gpu_need(FOUR_B, "gpu-bf16", 24576, 2048)[0] - cli.gpu_need(FOUR_B, "gpu-bf16", 12288, 2048)[0] == pytest.approx(1.15 * 12288 * 65536, abs=1)


# ---------------------------------------------------------------- nvidia-smi

class Run:
    def __init__(self, stdout, returncode=0):
        self.stdout, self.returncode = stdout, returncode


def fake_smi(monkeypatch, *replies):
    """nvidia-smi is present; each call answers with the next reply (a Run), and the asked-for fields are recorded."""
    asked = []
    queue = list(replies)

    def run(command, **kw):
        asked.append(command[1])
        return queue.pop(0)

    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(cli.subprocess, "run", run)
    return asked


def test_parse_gpu_line_with_and_without_compute_cap():
    assert cli.parse_gpu_line("NVIDIA GeForce GTX 1080 Ti, 11264, 10800, 6.1") == ("NVIDIA GeForce GTX 1080 Ti", 11264, 10800, 6.1)
    assert cli.parse_gpu_line("NVIDIA GeForce RTX 3060, 12288, 11500, 8.6")[3] == 8.6
    assert cli.parse_gpu_line("NVIDIA B200, 183359, 180000, 10.0")[3] == 10.0
    assert cli.parse_gpu_line("NVIDIA GeForce GTX 1080 Ti, 11264, 10800") == ("NVIDIA GeForce GTX 1080 Ti", 11264, 10800, None)
    assert cli.parse_gpu_line("Tesla P40, 24576, 24400, [N/A]")[3] is None
    assert cli.parse_gpu_line("Tesla P40, 24576, 24400, ")[3] is None
    with pytest.raises(ValueError):
        cli.parse_gpu_line('Field "compute_cap" is not a valid field to query.')


def test_detect_hardware_reads_the_compute_capability(monkeypatch):
    asked = fake_smi(monkeypatch, Run("NVIDIA GeForce GTX 1080 Ti, 11264, 10800, 6.1\n"))
    found = cli.detect_hardware()
    assert (found.gpu_name, found.vram_total, found.vram_free, found.compute_cap) == ("NVIDIA GeForce GTX 1080 Ti", 11264 * (1 << 20), 10800 * (1 << 20), 6.1)
    assert asked == ["--query-gpu=name,memory.total,memory.free,compute_cap"] and "compute capability 6.1" in found.describe()


def test_detect_hardware_with_a_driver_that_lacks_the_compute_cap_field(monkeypatch):
    # An old nvidia-smi rejects the field (error text, status 2); the second query without it answers and the capability stays unknown.
    asked = fake_smi(monkeypatch, Run('Field "compute_cap" is not a valid field to query.\n', 2), Run("NVIDIA GeForce GTX 1080 Ti, 11264, 10800\n"))
    found = cli.detect_hardware()
    assert found.gpu_name == "NVIDIA GeForce GTX 1080 Ti" and found.vram_total == 11264 * (1 << 20) and found.compute_cap is None
    assert asked[1] == "--query-gpu=name,memory.total,memory.free" and "compute capability" not in found.describe()
    assert cli.make_plan(FOUR_B, found, 24576, args()).profile == "gpu-f16"       # unknown: treated as a modern card


def test_detect_hardware_when_the_error_text_comes_with_status_zero(monkeypatch):
    fake_smi(monkeypatch, Run('Field "compute_cap" is not a valid field to query.\n'), Run("Tesla P40, 24576, 24400\n"))
    found = cli.detect_hardware()
    assert found.gpu_name == "Tesla P40" and found.compute_cap is None


def test_detect_hardware_without_a_working_nvidia_smi(monkeypatch):
    fake_smi(monkeypatch, Run("", 9), Run("", 9))
    assert cli.detect_hardware().gpu_name is None
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    assert cli.detect_hardware().gpu_name is None
