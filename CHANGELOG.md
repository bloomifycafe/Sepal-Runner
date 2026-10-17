# Changelog

## Unreleased

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
