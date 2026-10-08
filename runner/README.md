# Sepal runner

The C++20 inference engine for the Sepal decision models. The design is [../docs/RUNNER.md](../docs/RUNNER.md).
This directory holds the first slice of stage R1: a tested CPU foundation with no third-party dependencies.

## What is here

| Part | Files | What it does |
|---|---|---|
| Geometry | `include/sepal/config.hpp`, `src/core/config.cpp` | The model geometry as data, the four presets (0.8B, 2B, 4B, 9B), validation, and derived quantities: parameter counts, K/V bytes per state token, recurrent-state bytes, FLOPs per token |
| Element types | `include/sepal/dtype.hpp` | `f32`, `f16`, `bf16`, `i8`; exact decoding of the 16-bit float formats |
| Tensor | `include/sepal/tensor.hpp` | An owning float32 buffer with a shape, and non-owning matrix views |
| GEMM provider | `include/sepal/gemm.hpp`, `src/provider/gemm_ref.cpp` | The provider interface (`gemm_f`) and the reference provider that later providers are tested against |
| Operations | `include/sepal/ops.hpp`, `src/model/ops.cpp` | RMSNorm applied as `1 + w`, SiLU, the bias-free SwiGLU MLP, softmax, depthwise causal convolution with a carried tail |
| Planner | `include/sepal/planner.hpp`, `src/plan/planner.cpp` | A memory plan (weights, working buffers, snapshot arena), or a refusal that names the shortfall and the knob to reduce |
| Tool | `tools/sepal_info.cpp` | `sepal-info`: a preset's derived quantities and a memory plan for a budget |
| Tests | `tests/` | A small self-registering harness (`tests/test.hpp`) and one suite per part |

No size is hard-wired outside the preset table in `src/core/config.cpp`.

## Not here yet

The DeltaNet recurrence, attention (RoPE, grouped-query attention, K/V cache), the weight loader (GGUF), the
tokenizer and encoder, the FP64 reference path, the pass executor, snapshots and caches, the C ABI and the
server come next. Of the provider interface, only `gemm_f` exists; `gemm_qd8` and `gemm_dq` come with the
quantised profiles.

## Build and test

Requirements: CMake 3.21 or newer, Ninja, and a C++20 compiler (MSVC 19.3x or newer, GCC 11 or newer, Clang 14 or
newer). Nothing is downloaded. Warnings are errors. Build output goes to `runner/build/`, which is not committed.

There are two presets: `release` (Ninja, Release) and `asan` (Ninja, RelWithDebInfo with AddressSanitizer).

### Windows

Visual Studio 2022 with the C++ workload provides the compiler, CMake and Ninja, but not on the default `PATH`.
Either open "x64 Native Tools Command Prompt for VS 2022", or enter the environment in the same command. From
PowerShell, in the repository root:

```powershell
cmd /c '"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cd runner && cmake --preset release && cmake --build --preset release && ctest --preset release'
cmd /c '"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cd runner && cmake --preset asan && cmake --build --preset asan && ctest --preset asan'
```

The AddressSanitizer build needs the MSVC AddressSanitizer component and runs from the same environment, which
puts the sanitizer runtime DLL on the `PATH`.

### Linux

```sh
cd runner
cmake --preset release && cmake --build --preset release && ctest --preset release
cmake --preset asan && cmake --build --preset asan && ctest --preset asan
```

To choose a compiler, set `CXX` before the first configure, for example `CXX=clang++ cmake --preset release`.

### Without presets

```sh
cmake -S runner -B runner/build/manual -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build runner/build/manual
ctest --test-dir runner/build/manual --output-on-failure
```

Options: `-DSEPAL_ASAN=ON` enables AddressSanitizer; `-DSEPAL_BUILD_TESTS=OFF` skips the tests.

### Status of the two platforms

This slice was built and tested with MSVC 19.44 on Windows 11, in both presets. It is written to the C++20
standard with compiler extensions off and no platform headers, but it has not yet been compiled with GCC or
Clang; the first Linux build may meet warnings that MSVC does not raise.

## Running one suite

`sepal_tests` runs everything; `sepal_tests <suite>` runs one of `config`, `dtype`, `tensor`, `gemm`, `ops`,
`planner`, and fails if the name matches no test case.

## sepal-info

```
sepal-info <preset> [--weights w16|w8|gpu-f16|gpu-q8w|cpu-q8] [--budget-gib 12] [--reserve-gib 0]
           [--tokens 4096] [--rows 32] [--snapshot-tokens 2000] [--snapshots 1] [--embedding-on-device]
```

It exits with 0 when a plan is granted, 2 when the planner refuses and 1 on a usage error. Example:
`build/release/sepal-info 9b --weights w16 --budget-gib 12` refuses and names the weight precision as the knob.

## Conventions the code fixes

- **GEMM.** `Y[t, n] = beta * Y[t, n] + sum over k of X[t, k] * W[n, k]`, with `W` stored one output feature per
  row. Output row `t` depends on input row `t` and the weights only. The reference provider accumulates each
  output in float64 over `k = 0 .. K-1` and rounds to float32 once.
- **RMSNorm.** `y = x / sqrt(mean(x^2) + eps) * (1 + w)`: the stored weight is centred on zero.
- **Convolution.** Filter `[C, K]`; tap `K-1` multiplies the current input and tap 0 the oldest, which is the
  layout of a PyTorch `Conv1d` weight with left padding `K-1`. The tail is the last `K-1` inputs, oldest first.
  The convolution is bias-free. Both points are to be confirmed against the base model files when the loader
  exists.
- **Floating point.** No fast-math and no fused multiply-add contraction (`/fp:precise`, `-ffp-contract=off`).

## Numbers this code derives

Computed by `count_params` from the geometry alone, and checked in the tests against the figures in
`docs/RUNNER.md` section 3. Nothing here was measured.

| | 0.8B | 2B | 4B | 9B |
|---|---|---|---|---|
| Non-embedding parameters | 498,113,344 | 1,373,265,728 | 3,570,052,096 | 6,919,565,824 |
| Trunk weights at 16 bits | 1.00 GB | 2.75 GB | 7.14 GB | 13.84 GB |
| Trunk weights at 8 bits | 0.50 GB | 1.37 GB | 3.57 GB | 6.92 GB |
| K/V per state token, fp16 | 12,288 B | 12,288 B | 32,768 B | 32,768 B |
| Recurrent state, fp32 | 18 MiB | 18 MiB | 48 MiB | 48 MiB |

Two things to know when reading planner output:

- The 8-bit figures above take every trunk weight at one byte (profile `w8`), as the design's table does. The
  design's own `gpu-q8w` and `cpu-q8` profiles keep DeltaNet `out_proj` and `in_proj_a / b` at fp16 and the small
  tensors at fp32, so their trunk is larger: 7.33 GB on 9B, not 6.92 GB.
- The working-buffer figure is an estimate built from the geometry (the formula is documented at `working_bytes`
  in `include/sepal/planner.hpp`). Stage R3 replaces it with the executor's real allocations.

## Provenance

The refusing-planner pattern follows Strata's `include/strata/plan/plan.hpp` (MIT). No code was copied or
adapted from Strata in this slice.
