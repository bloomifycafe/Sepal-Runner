# Sepal-Runner

The fast inference runner for the [Sepal](https://github.com/bloomifycafe/Sepal) decision models. It has a C++20 engine (CPU and CUDA) behind a
small C ABI, and a Python front end that tokenises requests, batches them into engine calls and serves the Decision API,
`POST /v1/systemone`. The engine scores the trained model (Qwen3.5 with the LoRA merged in, and Clef's `JointSchemaHead` reading the final hidden
state of every position) in one forward pass per request, however many yes/no or multiple-choice questions the request carries.

This repository was split from Sepal on 2026-10-17, keeping the git history of the files that moved. Sepal trains and evaluates the model and
holds the benchmark. This repository serves it. The design document is [docs/RUNNER.md](docs/RUNNER.md). Links in it to `reports/`,
`MODEL.md` and similar paths point into the Sepal repository.

Licence: Apache-2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)). Status: engine and front end are built and tested; nothing here trains a model.

## Quickstart

```
pip install -e ".[serve]"                 # numpy, tokenizers, huggingface_hub, psutil
sepal serve BlossomsAI/Sepal-4B           # or a local checkpoint directory; add --dry-run to see the plan only
python examples/client.py                 # one alert, two questions
```

`sepal serve` fetches the adapter and the pinned base on first use (Hub downloads are anonymous unless `HF_TOKEN` is set; the Sepal repositories are
gated, so accept their terms on the Hub first, or point it at a directory you already have), looks at the CPU, RAM, GPU and free VRAM, and picks:

- `gpu-f16` on a CUDA card with 12 GB and up, with `max_call_tokens`, `trunk_budget_mib`, `prefix_cache` and the context window computed from the free
  VRAM; otherwise `cpu-bf16` with one thread per physical core of the CPUs it may use (capped by a container's CPU quota) and a
  prefix-state cache sized from the spare RAM. Next to another engine on the same server, give it its own cores with `--cpus 8-19`
  (one thread per physical core, pinned; see [runner/README.md](runner/README.md#running-next-to-another-engine-for-example-strata-on-the-same-server));
  `--cpu-isa` forces a kernel set (`avx2-tuned` is chosen automatically on AVX2 CPUs without AVX-512, and a set the CPU lacks is refused).
- `gpu-q8w` and `cpu-q8` (int8) only with `--allow-q8`, with a printed warning: on this model they exceed the q8 parity gate (see below).
- A model that fits neither is refused before anything is loaded, with the numbers and what would fit.

It prints the machine, profile, window and expected memory, then serves on `127.0.0.1:8000` (`--host` elsewhere needs `--api-key` or
`$SEPAL_API_KEY`): `POST /v1/systemone` (the TypeSafe body unchanged), `POST /v1/systemone/batch`, `GET /v1/models`, `/healthz`, `/readyz`, `/metrics`.
A request longer than the window gets `422 {"detail": "maximum context length is N tokens; the request needs more"}`.
The runner libraries come from `runner/build/release` and `runner/cuda/build/release` (see Build), or `$SEPAL_RUNNER_LIB_DIR` / `--lib`.

### What to expect (Sepal-4B)

Measured on a Ryzen 5 7600X (6 cores / 12 threads) and an RTX 3060 12 GB with the limits below **simulated** (CPU affinity and a memory cap on a
faster machine: memory speed and clocks are not those of a real cheap machine). Batch 1, one request per call. Short alert: about 900 tokens, 1 question.
Long log: about 15.9k tokens, 32 questions. Source: `runs/box-results/full15-4b/runner/SUMMARY.md` in the Sepal repository.

| Tier | Profile | Short alert p50 / p95 | Long log p50 | Committed memory |
|---|---|---|---|---|
| CPU 4 cores / 16 GB | cpu-bf16 | 8.3 / 8.6 s | 167 s | 8.2 GiB (9.3 on a long log) |
| CPU 6 cores | cpu-bf16 | 5.7 / 6.1 s | 112 s | same |
| CPU 2 cores / 8 GB | cpu-bf16 | does not load (needs 7.8 GiB) | | |
| GPU 12 GB + 4 threads | gpu-f16 | 0.23 / 0.24 s (4.2 alerts/s at p95 <= 1 s) | 4.8 s | VRAM +11.1 GB |
| GPU 8 GB | gpu-q8w (`--allow-q8`) | 0.36 s | not for states over about 7k tokens | VRAM +7.7 GB |

Parity against the PyTorch reference rows (SEC-dev, TOOL, LONG-LOG samples; gates in docs/RUNNER.md section 14): `cpu-bf16` and `gpu-f16` pass;
`cpu-q8` and `gpu-q8w` fail the max |dp| gate on near-tie questions (0.12 to 0.14 against 0.05).

## What is in it

| Path | What |
|---|---|
| `runner/` | The C++ engine. `runner/src/cpu`: the CPU backend (`cpu-f32`, `cpu-bf16`, opt-in `cpu-q8`); `runner/cuda`: the CUDA backend (`gpu-f32`, `gpu-f16` and their `-strict` forms); reference path, C ABI (`runner/include/sepal/c_api.h`), tools `sepal-score`, `sepal-info`, `sepal-*-bench`. See [runner/README.md](runner/README.md) and [runner/cuda/README.md](runner/cuda/README.md) |
| `sepal_runner/` | The Python front end: `encode.py` (Clef's encoder on the `tokenizers` library), `engine.py` (ctypes binding, batching arrays), `server.py` (HTTP server, thread pool, batching scheduler, result cache), `mock.py` (a fake engine for tests and load tests), `__main__.py` (batch scoring of a JSONL file) |
| `sepal_runner/shared/` | The small Sepal modules the front end needs, copied in: record conversion, request handling, temperature and softmax post-processing, temperature lookup. `tests/test_shared_parity.py` keeps them equal to Sepal's |
| `sepal_runner/refengine/` | The NumPy float64 reference engine the C++ reference path is ported from (used for the tiny golden fixtures) |
| `sepal_runner/oracle*.py`, `r0_bench.py`, `workloads.py`, `tools/runner_*`, `tools/runner_oracle/` | The oracle side: golden-vector writers, benchmarks, parity and rescoring tools. These run Sepal's training-side code and Unsloth as a black box and need a Sepal checkout on `PYTHONPATH` (`tools/runner_golden_parity.py` and the serving path do not) |
| `tests/` | Python tests (front end, vendored-module parity, reference engine, oracle helpers) |
| `reports/runner/` | Committed measurement records |

## Build

Requirements: CMake 3.21 or newer, Ninja, a C++20 compiler (MSVC 19.3x+, GCC 11+, Clang 14+). Nothing is downloaded; warnings are errors.

Linux:

```sh
cd runner
cmake --preset release && cmake --build --preset release && ctest --preset release
```

Windows (PowerShell; Visual Studio 2022 with the C++ workload):

```powershell
cmd /c '"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cd runner && cmake --preset release && cmake --build --preset release && ctest --preset release'
```

CUDA backend (optional; `runner/cuda`, needs a CUDA toolkit with cuBLAS, 12.x or 13.x; see [runner/cuda/README.md](runner/cuda/README.md)):

```sh
cd runner/cuda
cmake --preset release && cmake --build --preset release && ctest --preset release
```

The engine libraries are `sepal_runner_cpu` and `sepal_runner_cuda` (`.so` / `.dll`), which the Python front end loads with `--lib PATH` or
`$SEPAL_RUNNER_LIB`.

## Python front end

```sh
pip install -e ".[dev]"      # numpy, tokenizers; pytest and safetensors for the tests
pytest -q
```

Serve a checkpoint (`final/` holds `tokenizer.json`, `sepal.json` with the calibrated temperatures, the adapter and the head):

```sh
python -m sepal_runner.server --final RUN/final --base BASE_MODEL_DIR --lib build/libsepal_runner_cuda.so --profile gpu-f16 --name sepal-gpu-f16 --port 8000
python -m sepal_runner score --in requests.jsonl --out answers.jsonl --final RUN/final --base BASE_MODEL_DIR --lib LIB --profile cpu-bf16
```

## Profiles and modes

A profile is the precision of the engine; it is also the identity of an answer, because two profiles give different last digits.

| Profile | Trunk precision | Notes |
|---|---|---|
| `cpu-f32` | float32 | gate: max abs dp vs the fp32 oracle <= 1e-4, 0 flips |
| `cpu-bf16` | bfloat16 weights | gate: max abs dp <= 0.03, p99 <= 0.01, flips <= 0.5% |
| `gpu-f32` | float32 | same gate as `cpu-f32` |
| `gpu-f16` | float16 | same gate as `cpu-bf16` |
| `gpu-f32-strict`, `gpu-f16-strict` | as above, batch-invariant | the hidden states of a request are bit-identical alone, among other requests and at any place of a batch; 8 to 22% slower |
| `cpu-q8` (opt-in) | int8 weights x int8 activations | fails its gate: off |

Fast mode is the default. `cpu-*`, `gpu-f32` and `gpu-f16` answers may differ in the last bits depending on which requests share an
engine call, and the GPU DeltaNet layers may use the chunked kernels. Ask for a `-strict` profile when a request must give the same bits
every time. Name the server after its profile (`--name sepal-gpu-f16`): the `model` field of every answer then says which mode produced
it, and `GET /stats` and `/health` report the same name. (The answer JSON has no separate profile field; adding one is open, see
[CHANGELOG.md](CHANGELOG.md).)

## API

`POST /v1/systemone` takes `{"model", "state": str | json, "questions": {qid: {"type": "choice" | "noul", "instructions", "criteria": {option key: description}}}}`
and answers `{"model", "answers": {qid: {"type": "choice", "choice", "probabilities"} | {"type": "noul", "noul": p_yes}}, "usage": {"input_tokens"}}`,
the wire format of the Decision Index. Nothing is truncated: a request that does not fit the 16,384-token window, or a choice with fewer than 2 or
more than 255 options, is refused with HTTP 422. Also `POST /v1/systemone/batch` (`{"requests": [...]}`), `GET /health`, `GET /stats`, `GET /metrics`
(Prometheus text). The server binds to 127.0.0.1; any other host needs an API key (`--api-key` or `$SEPAL_API_KEY`). A full server answers 429, a
request that waits too long 504. Identical requests are answered once (result cache and in-flight coalescing). The probabilities are the engine's
raw option logits at the calibrated temperature of the question's type and option-count bucket (`final/sepal.json`), then a softmax, computed in
Python (`sepal_runner/shared/scoring.py`).

## Measured against the oracle

The oracle is the fp32 forward of the trained model (Unsloth 2026.10.4, transformers 5.17, torch 2.8) on the golden set: 30 cases from 185 to 31,850
tokens, 165 questions. Every number below was measured on the machine named. The full tables, shapes and caveats are in `runner/README.md` and
`runner/cuda/README.md`, and `docs/RUNNER.md` explains the markers (measured / derived / read from source / unverified).

| What | Result | Machine and date |
|---|---|---|
| `gpu-f32` vs fp32 oracle | max abs dp 7e-6 (T=1 and calibrated), 0 flips (gate 1e-4) | RTX 3090, CUDA 12.8, 2026-10-16/17 |
| `gpu-f32-strict` | max abs dp 5.8e-6 | same |
| `gpu-f16` | max abs dp 7.9e-3 (scan) / 6.8e-3 (chunked DeltaNet) at batch 1, 1.3e-2 at batch 20, p99 5.7e-3, 0 flips (gate 0.03 / 0.01) | same |
| `gpu-f16-strict` | max abs dp 2.3e-3 | same |
| `cpu-f32` | max abs dp 5.7e-6, 0 flips | Xeon E5-2683 v4 (AVX2), 2026-10-16 |
| `cpu-bf16` | max abs dp 6.8e-3 (T=1), 4.5e-3 (calibrated), p99 3.4e-3, 0 flips | same |
| `cpu-q8` vs its bf16 twin | max abs dp 0.25, 1 flip in 43 questions: fails (gate 0.05 / 1%) | same |
| int8 on the GPU (W8A16 / W8A8) | max abs dp 0.073 / 0.13: does not pass; off | RTX 3090 |
| CPU speed, `cpu-bf16`, 6 threads | 192-238 tokens/s at 1k-4k tokens, 170 tokens/s at 16k; about 5-6x llama.cpp b11514 prefill (36.5 tokens/s) | Ryzen 5 5600X (AVX2), shared box, lower bounds, 2 runs differ by up to 10% |
| GPU speed, `gpu-f16` default, 0.8B | 21,461 tokens/s prefill at 1 row x 1000 (llama.cpp b11514: 10,653); 24.4 alerts/s at state + 8 questions, 1 alert per call (llama-server: 1.7) | RTX 3060 12 GB, driver 591.86, CUDA 13.1, best of two runs of 7 calls (`runner/cuda/README.md`, Speed) |

Raw lines and per-case rows are under `reports/runner/` and, for the benchmark runs, in the READMEs. The numbers above were produced in the Sepal
repository before the split; the first measurements in this repository (the build and test run that closed the split) are in
[CHANGELOG.md](CHANGELOG.md).

## Tests

```sh
pytest -q                                         # Python: front end with the mock engine, vendored-module parity, reference engine
cd runner && ctest --preset release               # C++
```

Environment variables the tests use when set (each test skips with a reason when its input is missing):

| Variable | Used for |
|---|---|
| `SEPAL_GOLDEN_DIR` | a golden set directory (`manifest.json` and the case `.safetensors` files): vendored encoder and post-processing parity |
| `SEPAL_FINAL_DIR` | the checkpoint's `final/` (tokenizer.json, sepal.json) for the same tests; `SEPAL_TOKENIZER_DIR` (a directory with tokenizer.json) or `SEPAL_TOKENIZER` (the file) serve the encoder-only tests |
| `SEPAL_TRAINED_FINAL`, `SEPAL_TRAINED_BASE`, `SEPAL_GOLDEN_DIR` | the C++ golden tests (`ctest -R golden`) |

To check the vendored modules against Sepal itself, put a Sepal checkout on `PYTHONPATH`; those tests are skipped otherwise.

## Provenance and the Sepal side

The shared modules were copied from Sepal at f846e8d (see their first lines). If Sepal changes the prompt, the conversion or the calibration, change
the copies here and keep the parity tests green: they compare against the golden set that Sepal's oracle writes.
