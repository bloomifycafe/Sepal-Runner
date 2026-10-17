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

### Open

- The answer JSON carries the served name in `model` but no separate profile field; a field that labels each answer with its profile and mode
  (fast or strict) would need agreement with the Decision API clients.
- The oracle tools (`sepal_runner/oracle*.py`, `r0_bench.py`, `workloads.py`, `tools/runner_panel_rescore.py`, `tools/runner_server_rescore.py`,
  `tools/runner_encode_compare.py`) still import Sepal's training-side code and need a Sepal checkout.
