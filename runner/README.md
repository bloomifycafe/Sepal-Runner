# Sepal runner

The C++20 inference engine for the Sepal decision models. The design is [../docs/RUNNER.md](../docs/RUNNER.md).
This directory holds the first two slices of stage R1, with no third-party dependencies: a tested CPU foundation,
and on it a complete forward pass (the reference path) that is compared with the NumPy oracle `sepal/refengine`.

## What is here

| Part | Files | What it does |
|---|---|---|
| Geometry | `include/sepal/config.hpp`, `src/core/config.cpp` | The model geometry as data, the four presets (0.8B, 2B, 4B, 9B) and the oracle's `tiny` test geometry, validation, and derived quantities: parameter counts, K/V bytes per state token, recurrent-state bytes, FLOPs per token |
| Element types | `include/sepal/dtype.hpp` | `f32`, `f16`, `bf16`, `i8`; exact decoding of the 16-bit float formats |
| Tensor | `include/sepal/tensor.hpp` | An owning buffer with a shape and non-owning matrix views, as templates over the element type: `Tensor` / `MatView` are float32, `TensorD` / `MatViewD` float64 |
| GEMM provider | `include/sepal/gemm.hpp`, `src/provider/gemm_ref.cpp` | The provider interface (`gemm_f`, and `gemm_f64` for the float64 reference path) and the reference provider that later providers are tested against |
| Operations | `include/sepal/ops.hpp`, `src/model/ops.cpp` | The first slice's float32 kernels: RMSNorm applied as `1 + w`, SiLU, the bias-free SwiGLU MLP, softmax, depthwise causal convolution with a carried tail |
| Planner | `include/sepal/planner.hpp`, `src/plan/planner.cpp` | A memory plan (weights, working buffers, snapshot arena), or a refusal that names the shortfall and the knob to reduce |
| Weights | `include/sepal/weights.hpp`, `src/core/weights.cpp` | The tensor names and shapes a geometry implies, a read-only container keyed by tensor name with `scope("layers.3")`, a strict loader from any name-to-array map, and seeded random weights for benchmarks |
| Golden file | `include/sepal/golden.hpp`, `src/io/golden.cpp`, `include/sepal/json.hpp`, `src/io/json.cpp` | The reader of the golden-vector format and the small JSON reader its header needs |
| Safetensors | `include/sepal/safetensors.hpp`, `src/io/safetensors.cpp`, `include/sepal/mapped_file.hpp`, `src/io/mapped_file.cpp` | A zero-copy safetensors reader (read-only mapping on Windows and POSIX; every offset, size, alignment and overlap checked) and `Checkpoint`, one or many files with or without an index |
| Loader | `include/sepal/loader.hpp`, `src/model/loader.cpp` | `config.json` to a `ModelConfig` (compared with the closest preset), checkpoint names to engine names, the ignored tensors listed, `Weights<float/double>` for the reference path and a raw zero-copy view in the stored precision; `DecisionReadout` for the label-row readout. See "Real checkpoints" |
| Reference path | `include/sepal/ref/*.hpp` | The whole forward pass and the request semantics, ported function for function from `sepal/refengine`: see "The reference path" below |
| Tools | `tools/sepal_info.cpp`, `tools/sepal_bench.cpp`, `tools/sepal_run.cpp`, `tools/export_golden.py`, `tools/make_tiny_checkpoint.py` | `sepal-info`: a preset's derived quantities and a memory plan for a budget. `sepal-bench`: tokens per second of the reference path. `sepal-run`: one pass of a real checkpoint. `export_golden.py`: writes the golden file from the oracle. `make_tiny_checkpoint.py`: writes the loader's test checkpoint |
| Tests | `tests/` | A small self-registering harness (`tests/test.hpp`) and one suite per part; `tests/golden/tiny.sgv` is the golden file |

No size is hard-wired outside the preset table in `src/core/config.cpp`.

## Not here yet

The GGUF loader, the tokenizer and encoder, the optimised layer-interleaved pass executor, the snapshot
store and caches, the C ABI and the server come next. Of the provider interface, `gemm_f` and `gemm_f64` exist;
`gemm_qd8` and `gemm_dq` come with the quantised profiles. The numbers in "The reference path" and below come from
the `tiny` test geometry or from random weights; the comparison with transformers on a real checkpoint is under
"Real checkpoints".

## The reference path

`include/sepal/ref/` is a readable port of the oracle, one header per oracle module and one function per oracle
function, with the same names, so a disagreement can be found by comparing function with function:

| Header | Oracle module | Holds |
|---|---|---|
| `ref/primitives.hpp` | `primitives.py` | `sigmoid`, `silu`, `softplus`, softmax, `rms_norm`, `project`, `swiglu_mlp` |
| `ref/deltanet.hpp` | `deltanet.py` | `causal_conv`, `l2_normalise`, `share_key_heads`, `split_heads`, `decay_gate`, `delta_rule_update`, `delta_rule_scan`, `gated_rms_norm`, `deltanet_mixer` |
| `ref/attention.hpp` | `attention.py` | `unpack_query_and_gate`, `rope_angles`, `apply_rope`, `grouped_query_attention`, `attention_mixer` |
| `ref/snapshot.hpp` | `snapshot.py` | `DeltaNetState`, `AttentionState`, `Snapshot`, `empty_snapshot` |
| `ref/executor.hpp` | `executor.py` | `Row`, `RowResult`, `decoder_layer`, `continue_snapshot`, `check_row`, `execution_order`, `run_rows` |
| `ref/readout.hpp` | `readout.py` | `pointer_logits`, `pointer_probabilities`, `label_rows`, `label_row_logits`, `label_row_probabilities` |

Everything in `sepal::ref` is a template over the element type. With `double` it is the FP64 reference forward
pass (oracle O2 of the design); with `float` the same code is the unoptimised float32 path. It is single-threaded
and nothing in it is tuned.

Where the port differs from the oracle in form, and why:

- **Matrix multiplies** all go through the GEMM provider: the in-projections, the out-projections, the MLP, the
  attention scores (Q K^T per head) and the weighted sum of values, the pointer head and the label rows. The
  oracle's `project` concatenates the weights of several projections and computes one product; the port computes
  one product per weight, which under the provider contract (an output depends on its input row and its weight
  row only) gives the same values without copying weights. Packing them into one GEMM is for the optimised path.
- **Snapshots** are immutable values behind `std::shared_ptr<const ...>`. The oracle concatenates a parent's keys
  and values with a row's own into a new array per row; the port's `AttentionState` instead holds a link to its
  parent's state and only its own tokens, so children share a parent's keys and values by reference and none copies
  them. A DeltaNet child starts its own recurrence from a copy of the parent's recurrent state, which it must
  (the recurrence rewrites it); the parent's is never written.
- **A parent row** is named by its index in the request (`ParentRow{index}`), where the oracle holds the Python
  object. A chain of parents that returns to itself can therefore be written, and is refused.
- **The kernels of `ops.hpp`** (float32 only, with a combined gate-and-up weight) are not used by the reference
  path, which has its own templated `rms_norm`, `silu`, `swiglu_mlp`, softmax and `causal_conv` so that one text
  serves both element types. The two sets compute the same functions and should be merged when the optimised
  executor is written.
- **RoPE angles** are float64 in both element types, so a large position is not rounded before its cosine.
- **The delta rule** updates the row's own working state in place, one token after another, where the oracle
  returns a new array per token.

### Oracle details ported as unverified

The oracle lists, in `sepal.refengine.UNVERIFIED`, the details the Kev reference files do not settle. Each is
ported as it stands and lives in exactly ONE function, marked `UNVERIFIED` in its comment with the oracle's name
for it. When one changes in the oracle, change that one function and regenerate the golden file.

| Oracle entry | C++ function (`sepal::ref`) | What is assumed |
|---|---|---|
| `deltanet.causal_conv` | `causal_conv` | The last kernel tap multiplies the current token (cross-correlation); the tail is oldest first; no bias |
| `deltanet.l2_normalise` | `l2_normalise` | q and k are divided by `sqrt(sum(x^2) + 1e-6)`: eps inside the root, not `max(norm, eps)` |
| `deltanet.share_key_heads` | `share_key_heads` | Value head h reads key head `h / (value heads / key heads)`: repeat-interleave, not tiling |
| `deltanet.delta_rule_update` | `delta_rule_update` | The old state is decayed before the update, the write itself is not decayed; state axes are [key dim, value dim] |
| `deltanet.delta_rule_scan` | `delta_rule_scan` | A token's output reads the state after its own update, not before |
| `deltanet.delta_rule_output_scale` | `delta_rule_output_scale` | The read is scaled by `1 / sqrt(key head dim)` |
| `deltanet.gated_norm_scale` | `gated_norm_scale` | The gated norm multiplies by its stored weight `w`, not by `1 + w` like the other norms |
| `deltanet.gated_norm_gate` | `gated_norm_gate` | The gated norm's gate is `silu(z)`, not `sigmoid(z)` |
| `attention.rope_angles` | `rope_angles` | One position axis with frequencies `theta ** (-2i / rotary_dim)` |
| `attention.apply_rope` | `apply_rope` | The first `rotary_dim` dims of a head rotate, dim i paired with dim `i + rotary_dim / 2` (half-split) |
| `attention.attention_scale` | `attention_scale` | Scores are scaled by `1 / sqrt(head_dim)` |
| `attention.grouped_query_attention` | `grouped_query_attention` | Query head h reads kv head `h / (heads / kv heads)`: repeat-interleave, not tiling |

The golden file records the list as it stood when the file was written (header key `unverified`), and a test
checks that it has twelve entries, so a regenerated file with a changed list is noticed.

## Measured against the oracle

Measured on the `tiny` geometry (8 layers, hidden 24) with the golden file's seeded random weights, MSVC 19.44,
Release. Nothing here was measured on real weights or a real geometry. Largest absolute difference from the
NumPy float64 oracle; the float64 checks are asserted at 1e-9.

| Check | float64 | float32 |
|---|---|---|
| Per-layer hidden states, 13 tokens: worst of the 8 layer outputs (and layer 0 alone) | 5.6e-14 (1.7e-14) | 4.0e-5 (6.0e-6) |
| Final hidden states at the readout positions, 5 rows with parents | 4.0e-14 | 5.8e-6 |
| Snapshot contents after the state row, all layers | 2.0e-14 | 7.7e-6 |
| One pass over state + branch + leaf (31 tokens) | 8.2e-15 | not compared |
| Branches continued from the oracle's snapshot, loaded as a stored snapshot | 8.9e-15 | not compared |
| Each DeltaNet mixer alone, from the oracle's input and state (output, state, tail) | 2.0e-15 | not compared |
| Every intermediate inside one DeltaNet mixer (15 arrays) | 2.8e-15 | not compared |
| Each attention mixer alone (output, keys, values) | 8.9e-16 | not compared |
| Every intermediate inside one attention mixer (10 arrays) | 8.9e-16 | not compared |
| Each decoder layer alone, all 8 | 2.6e-15 | not compared |
| Both readouts from the oracle's hidden states (logits and probabilities) | 3.6e-15 | not compared |
| Both readouts end to end | 9.2e-14 | 8.3e-6 |

The layer outputs are residual-stream values before the final norm and grow through the stack, which is why the
per-layer figures exceed the final-hidden-state figures. The float32 suite asserts only a sanity limit of 1e-2,
some 250 times the largest value above: no float32 accuracy bound is claimed from a 24-wide test geometry.

Request invariants, compared C++ against C++ and found EXACTLY equal (difference 0):

- one row over state + branch + leaf against the same tokens as three chained rows: hidden states in float64
  and in float32, and the snapshot afterwards in float64;
- each branch run alone from the state's snapshot against the same branch inside the request, and the request
  with its rows reversed (float64);
- a snapshot's contents before and after four children ran from it twice, and the two runs of the children
  against each other (float64).

## sepal-bench

```
sepal-bench [--preset tiny|0.8b|2b|4b|9b]... [--state 512] [--branches 8] [--branch-tokens 50] [--vocab 4096] [--seed 1]
```

One state row and `--branches` rows that continue it, float32, seeded random weights, one thread. It measures the
reference path: scalar kernels and the scalar reference GEMM, deliberately unoptimised. The figure is a baseline
to improve on, not a statement about the engine. For a real geometry only `--vocab` embedding rows are allocated
(the embedding is looked up, never multiplied, so its size does not change the timing).

Measured on an AMD Ryzen 5 7600X, Windows 11, MSVC 19.44 Release, one thread, 512-token state and 8 branches of 50
tokens (912 tokens):

| Geometry | State row | Branches | Total |
|---|---|---|---|
| `tiny` (50,182 non-embedding parameters) | 10,028 tokens/s | 8,999 tokens/s | 9,549 tokens/s (0.096 s) |
| 0.8B (498,113,344 non-embedding parameters) | 3.35 tokens/s | 3.29 tokens/s | 3.32 tokens/s (274.8 s) |

One run each. The `tiny` run lasts a tenth of a second and an earlier run of it gave 8,019 tokens/s, so read
that row to one significant figure; a 16-token run of 0.8B gave 3.40 tokens/s.

## Golden vectors

`tools/export_golden.py` imports `sepal.refengine`, builds the `tiny` preset with seeded random weights, runs a
fixed set of requests through the oracle and writes `tests/golden/tiny.sgv` (671 kB, committed). The C++ tests take
the weights, the requests and the expected results from that file only. To regenerate it, from the repository root:

```
uv run python runner/tools/export_golden.py
```

Format `SEPALGV1`; every integer and float is little-endian:

```
8 bytes   magic, the ASCII text "SEPALGV1"
u32       H, the length of the header in bytes
H bytes   the header: one UTF-8 JSON object
u32       N, the number of arrays
N times:  u32 name length, the name (UTF-8), u32 rank R, R x u64 shape,
          prod(shape) x f64 values in row-major order
```

The header holds the geometry (`config`), the seed, the oracle's `UNVERIFIED` list and `cases`: every test request
(token ids, start positions, parents, readout offsets) and the name of each array that holds an expected result.
Array names are paths: `weights/<tensor name>`, then one prefix per case.

| Case | Contents |
|---|---|
| `ladder` | One 13-token sequence from nothing: the embedding, the output of every decoder layer for every token, and the hidden states after the final norm |
| `request` | Five rows: a 20-token state row, three branches of it, and a branch of a branch; a child is listed before its parent on purpose. The final hidden states at each row's readout offsets |
| `snapshot` | Every layer's state after the state row: recurrent state and conv tail per DeltaNet layer, keys and values per attention layer |
| `steps` | One branch continued from that snapshot, layer by layer: each layer's input, normed input, mixer output, output and state afterwards, so each layer can be tested alone |
| `deltanet_detail`, `attention_detail` | Every intermediate value inside one DeltaNet mixer and one attention mixer of `steps` |
| `one_pass` | The state row, a branch and its child as ONE row, read out at the same tokens |
| `pointer`, `label` | Questions for the two readouts: logits and probabilities, at several temperatures |

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

Everything here was built and tested with MSVC 19.44 on Windows 11, in both presets. It is written to the C++20
standard with compiler extensions off and no platform headers, but it has not yet been compiled with GCC or
Clang; the first Linux build may meet warnings that MSVC does not raise.

## Real checkpoints

`LoadedModel::open(dir)` reads `dir/config.json`, maps the safetensors files (`model.safetensors`, or the shards
named by `model.safetensors.index.json`, or `model-0000k-of-0000N.safetensors` alone), finds the text model under
its prefix (`model.language_model.`, `language_model.`, `model.` or none) and checks every tensor against the
geometry with the strict check of `weights.hpp`. Anything the engine does not load is listed in `ignored()` with a
reason: `visual.*` / `model.visual.*` (vision tower), `lm_head.*` (vocabulary projection), `mtp.*` / `model.mtp.*`
(multi-token-prediction head). Any other tensor, a missing one, one of another shape, or a config the engine does not
compute (no output gate, attention bias, an activation other than SiLU, layer types that are not a fixed
full-attention interval) refuses the checkpoint with a message that names it. The pointer head belongs to an adapter
and is not in a checkpoint; `weights<T>()` fills it with zeros, as the Python `engine_weights` does.

`raw()` gives every loaded tensor by engine name as (pointer into the mapping, F32/F16/BF16, shape): a GPU engine
uploads from it without widening on the host. `DecisionReadout::open_directory(dir)` reads `readout.safetensors`
and the temperature of `decision_config.json`; `probabilities(hidden, N)` is the softmax over the first N rows of
`hidden . row / temperature`, in float64.

```
sepal-run <checkpoint dir> <token file> [--readout <dir>] [--options N] [--precision f32|f64] [--show N]
```

The token file is token ids separated by spaces, commas or newlines. `sepal-run` prints the final hidden state's
first values and, with `--readout`, the probabilities.

**Against transformers.** `tools/export_real_golden.py` runs `mstrasser/jeff-base` in transformers on the CPU in
float32 and writes `runs/real-golden/case_{i,ii,iii,iv}.safetensors` (git-ignored; the file documents every
tensor). `tests/test_real_model.cpp` compares the CPU reference path with them and is skipped, with a message,
unless both variables are set:

```
uv run --project <main checkout> --group verify python tools/export_real_golden.py
SEPAL_REAL_MODEL=<snapshot dir> SEPAL_REAL_GOLDEN=runs/real-golden build/release/sepal_tests real
```

It takes about five minutes on one thread (16 + 64 + 300 + 300 + 179 tokens in float32, 16 in float64 as well).
Measured on 2026-10-09: largest difference over every layer and token 1.7e-05 (16 tokens) and 1.8e-05 (64 tokens) in
float32 against transformers' float32; 3.9e-14 for the float64 pass against transformers with its casts promoted to
float64; final hidden state 7.6e-06 to 3.9e-05 and probabilities at most 1.1e-06 on the longer cases, with the three
branches of one 300-token state matching the three full prompts.

## Running one suite

`sepal_tests` runs everything; `sepal_tests <suite>` runs one of `config`, `dtype`, `tensor`, `gemm`, `ops`,
`planner`, `golden`, `deltanet`, `attention`, `executor`, `readout`, `float32`, `safetensors`, `loader`, `real`, and fails if the name matches no
test case. The last five print the largest difference from the oracle that each comparison saw; CTest shows a
test's output only when it fails, so run the suite directly to read them, for example
`build/release/sepal_tests executor`.

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
  `gemm_f64` is the same sum with float64 inputs and outputs.
- **RMSNorm.** `y = x / sqrt(mean(x^2) + eps) * (1 + w)`: the stored weight is centred on zero. The DeltaNet
  gated norm is the exception the oracle assumes: it scales by `w` itself.
- **Convolution.** Filter `[C, K]`; tap `K-1` multiplies the current input and tap 0 the oldest, which is the
  layout of a PyTorch `Conv1d` weight with left padding `K-1`. The tail is the last `K-1` inputs, oldest first.
  The convolution is bias-free. Both points are among the oracle's unverified details (`deltanet.causal_conv`).
- **Pointer head.** Two linear maps WITH bias, `q` and `k`, from the hidden size to the pointer dim, as the oracle
  and the reference model define them: `2 * (P * H + P)` parameters, which is what `count_params` reports.
- **Float32 in the reference path.** Each kernel computes in float32, except the reference GEMM, which
  accumulates in float64 and rounds once, and the RoPE angles, which are float64.
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

## The CPU engine (rt-cpu)

`src/cpu/` is the fast CPU backend: own thread pool, runtime ISA dispatch (AVX-512 VNNI/BF16 or AVX2+FMA), packed
GEMM (k-blocked, 96 x 64 tasks, fp32 chain per output so the result does not depend on threads or batch), chunked
Gated DeltaNet, flash-style GQA attention, chunked-prefill trunk, and the Clef head. `sepal_runner_cpu` registers
`cpu-f32`, `cpu-bf16` and `cpu-q8` next to the reference engines; extra open options: `chunk_tokens`, `step_tokens`,
`delta_chunk`, `group_tokens`, and `q8` (a string with any of `mlp`, `attn`, `dn_in`). On AVX2, `cpu-bf16` keeps bf16
weights (half the memory) and widens them to fp32 per cache block, so it runs the fp32 kernel with float activations.
`cpu-q8` is opt-in only: it fails its gate (below).

Tools: `sepal-cpu-eval` (parity against the golden set, per profile, T = 1 and calibrated, q8 against its bf16 twin),
`sepal-cpu-bench` (tokens/s, requests/s, per-call latency, peak RSS on an encoded workload; `--phases` splits the trunk
time), `tools/runner_cpu_encode_workloads.py` (encodes `runs/runner-bench/workloads` for it).

Parity, all 30 golden cases up to 31.9k tokens, 165 questions, AVX2 (Xeon E5-2683 v4, measured 2026-10-16):
cpu-f32 max |dp| 5.7e-6 (gate 1e-4), 0 flips; cpu-bf16 max |dp| 6.8e-3 at T = 1, 4.5e-3 calibrated, p99 3.4e-3
(gate 0.03 / 0.01), 0 flips. AVX-512 (Ryzen 7600X, 20 short cases): f32 2.4e-6; bf16 (bf16 activations too) 1.3e-2.
cpu-q8 (int8 x int8, MLP + attention + DeltaNet input) vs its bf16 twin: max |dp| 0.25, 1 flip in 43 questions: fails
(gate 0.05 / 1%); each weight group alone is still above 0.03. On AVX2 it is also slower than fp32 and stores 16-bit
pairs, so there is no gain. Keep it off.

The default thread count is the physical cores (`default_thread_count()`): on the 5600X 12 threads (SMT) ran 20-25%
slower than 6 (bf16, t1k_q8: 3 / 6 / 9 / 12 threads = 143 / 233 / 181 / 181 tok/s).

Speed, Ryzen 5 5600X (AVX2, vast.ai box shared with a neighbour that kept 2-3 threads busy, so these are lower
bounds; batch 1; two runs of each differ by up to 10%): with the default 6 threads cpu-f32 195-203 tok/s on 913-token
requests with 1 question (0.23 requests/s), 191-198 tok/s at 1.3k tokens with 8 questions (0.15 requests/s), 179-204
tok/s at 3.5k tokens with 32 questions, 168 tok/s at 4k; cpu-bf16 192-195 / 229-236 / 215-223 tok/s for the same
shapes (0.18 requests/s at 1.3k x 8 questions), 238 tok/s at 4k, 170 tok/s at 16k (93 s per request). With 12 threads:
f32 166 tok/s (1k), 131 at 16k, 105 at 32k (attention is quadratic); bf16 179. Peak RSS 6.5 GB (f32) / 5.6 GB (bf16),
mostly the transient fp32 merge at load. llama.cpp b11514 on the same box (12 threads), base f16/bf16/f32 GGUF prefill
pp512: 36.5 / 36.4 / 36.7 tok/s, q8_0 48.9 tok/s: cpu-bf16 is about 5-6x faster. Raw lines: runs/runner-bench/cpu/.

## Provenance

The refusing-planner pattern follows Strata's `include/strata/plan/plan.hpp` (MIT). No code was copied or
adapted from Strata in this slice.
