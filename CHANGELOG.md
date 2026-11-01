# Changelog

## Unreleased

### 2026-11-01: 4B on small machines

Measured on the owner's PC (Ryzen 5 7600X 6 cores / 12 threads, 64 GB, RTX 3060 12 GB, Windows 11, MSVC 19.44, CUDA 13.1 from the pinned wheels) with
Sepal-4B (F15-s0, LoRA r64 + Clef head on Qwen3.5-4B 851bf6e). Limits are simulated with CPU affinity and a Job Object memory cap; memory speed and
clocks are those of this faster PC. Results: Sepal `runs/box-results/full15-4b/runner/`.

- Loading no longer holds the whole model in float32. `TrainedModel::merged_scope` / `merged_tensor` merge the LoRA one layer (CPU) or one tensor
  (GPU) at a time; `CpuTrunk` and `CudaEngine` take a fetch callback beside the old whole-model constructors (which are unchanged). Peak committed
  memory at load, cpu-bf16 4B: 23.2 GiB -> 7.8 GiB (a 16 GB machine could not load it before); gpu 4B: 25.7 GiB -> 11.8 GiB. Same weights, same speed.
- New profile `gpu-q8w` (int8 weights, binary16 activations, all four multiply groups; the engine option was there, the profile was not). 4B weights
  4.6 GiB on the device against 8.0 GiB for `gpu-f16`.
- The GPU engine gives a spent chain snapshot's state slot back as soon as the next segment has consumed it, so a long request cut into small calls
  no longer needs one slot per segment. Measured: gpu-f16 on a 12 GB card with `max_call_tokens` 2048 stopped on the first long TOOL request with "0 of 8 state slots free"; after the change the 150-record TOOL sample scores to the end.
- Test `trained.streamed_merge_equals_whole_merge` (bit-equal to the whole merge).

### 2026-10-17: split from Sepal

- The runner left the Sepal repository (main at f846e8d) with the git history of its files: `runner/` (C++ engine, CPU and CUDA), the Python
  front end (`sepal/runner/`, `sepal/serve/runner_api.py`, `sepal/refengine/`), `tools/runner_*`, `tools/verify_refengine.py`,
  `tools/export_real_golden.py`, `tests/runner/`, the runner tests, `reports/runner/`, `docs/RUNNER.md`, `docs/REFENGINE-VERIFICATION.md`.
- Python package renamed `sepal_runner` (`sepal.runner` -> `sepal_runner`, `sepal.serve.runner_api` -> `sepal_runner.server`,
  `sepal.refengine` -> `sepal_runner.refengine`). The C++ tree and its CMake files are unchanged.
- New `sepal_runner/shared/`: copies of the Sepal modules the front end imported (record conversion, request handling, temperature and softmax
  post-processing, temperature lookup), so the server runs without Sepal, PyTorch or Unsloth. `tests/test_shared_parity.py` checks them against
  the golden set and, when Sepal is importable, against Sepal's own functions.
- Added Apache-2.0 `LICENSE`, `NOTICE`, `pyproject.toml`, a CI workflow, this changelog.
- First measurements here (2026-10-17, Norway box: RTX 3060 Ti, CUDA 12.8, g++ 11.4, CMake 4.4; box shared with other jobs): `runner/` ctest 30/30 (4 skipped
  without a real model), with the full08b checkpoint and a 20-case golden subset `trained_real` and `clef_golden` pass; `runner/cuda` build and ctest 89/89
  (7 golden tests pass); `sepal-score` cpu-f32 / gpu-f32 on 3 golden cases max |dlogit| 1.2e-5 / 1.1e-5; `tools/runner_golden_parity.py` same 3 cases
  max |dp| 1.1e-6 (cpu-f32) and 1.3e-6 (gpu-f32), 0 flips; `pytest` 131 passed, 7 skipped (torch and Sepal not installed).
### Open

- The answer JSON carries the served name in `model` but no separate profile field; a field that labels each answer with its profile and mode
  (fast or strict) would need agreement with the Decision API clients.
- The oracle tools (`sepal_runner/oracle*.py`, `r0_bench.py`, `workloads.py`, `tools/runner_panel_rescore.py`, `tools/runner_server_rescore.py`,
  `tools/runner_encode_compare.py`) still import Sepal's training-side code and need a Sepal checkout.
