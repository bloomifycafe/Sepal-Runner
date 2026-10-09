# Sepal runner: CUDA matrix-multiply provider

The first slice of stage R2 of [../../docs/RUNNER.md](../../docs/RUNNER.md): a CUDA toolchain that needs no system
installer, a cuBLAS provider behind the engine's `GemmProvider` interface, its tests against the reference
provider, and a benchmark. It is a CMake project of its own. It builds the engine in `runner/` as a subdirectory
and changes nothing outside `runner/cuda/`.

Everything measured below was measured on one machine on 2026-10-09: Windows 11, NVIDIA GeForce RTX 3060 12 GB
(compute capability 8.6), driver 591.86, MSVC 19.44, CMake 3.31, Ninja.

> **Update.** The sections up to "Attention kernels" describe the first slice, the cuBLAS provider, and are kept as
> measured. Every item of the "not on the GPU" column below is now on the GPU and assembled into an engine that runs
> a real model end to end: see [The GPU engine](#the-gpu-engine) at the end of this file.

## What is and is not on the GPU yet

| On the GPU | Not on the GPU |
|---|---|
| Every matrix multiply: `Y = beta * Y + X W^T`, fp16 or fp32 operands, fp32 output | The DeltaNet recurrence, the causal convolution, the gated norm |
| Weights: uploaded once, kept on the device | Attention: RoPE, the K/V cache, scores, softmax |
| Element-type conversion of operands (fp32, fp16, bf16), by four small kernels compiled with `nvcc` | RMSNorm, SiLU, the SwiGLU product, the residual additions |
| A device arena: one allocation at start, none afterwards | The embedding lookup and the readout |
| | Activations between two multiplies: today each call uploads X and downloads Y |

So the engine cannot run a model on the GPU yet. The benchmark measures the matrix multiplies alone, which the
design takes to be about 80% of a pass (docs/RUNNER.md section 12, a figure measured elsewhere on another model).

## The toolchain

The CUDA Toolkit is not installed on this machine. NVIDIA publishes its components as Python wheels, and on
Windows the CUDA 13 wheels contain the whole compiler. `setup_toolchain.ps1` installs the versions pinned in
`toolchain-requirements.txt` into `runner/cuda/.toolchain/` (git-ignored, about 0.8 GB) with `uv`. They unpack into
one toolkit tree, `.toolchain/Lib/site-packages/nvidia/cu13/`.

| Package | Version | What it holds on Windows (by inspection) |
|---|---|---|
| `nvidia-cuda-nvcc` | 13.1.115 | `bin/nvcc.exe`, `cudafe++.exe`, `ptxas.exe`, `nvlink.exe`, `fatbinary.exe`, `bin2c.exe` |
| `nvidia-nvvm` | 13.1.115 | `nvvm/bin/cicc.exe`, `nvvm64_40_0.dll`, `nvvm/libdevice` |
| `nvidia-cuda-crt` | 13.1.115 | `include/crt`: the device-side runtime headers |
| `nvidia-cuda-runtime` | 13.1.80 | `cuda_runtime.h` and the other runtime headers; `lib/x64/cudart.lib`, `cudart_static.lib`, `cudadevrt.lib`, `cuda.lib`; `cudart64_13.dll` |
| `nvidia-cuda-cccl` | 13.1.115 | `include/cccl`: Thrust, CUB, libcu++ (not used yet) |
| `nvidia-cuda-nvrtc` | 13.1.115 | `nvrtc.h`, `nvrtc64_130_0.dll` (not used yet; no import library) |
| `nvidia-cublas` | 13.2.1.1 | `cublas_v2.h`, `cublas64_13.dll`, `cublasLt64_13.dll`; **no import library** |

This is CUDA 13.1 Update 1, the release the installed driver reports as its newest supported version
(`cudaDriverGetVersion` returns 13010). Three things found by inspection, not assumed:

- **The CUDA 12 wheels cannot compile CUDA C++ on Windows.** `nvidia-cuda-nvcc-cu12` (12.1.105 and 12.9.86 were
  unpacked) holds `ptxas.exe`, `nvvm64_40_0.dll` and `libdevice` only: no `nvcc.exe`, `cudafe++.exe` or `cicc.exe`.
  The unsuffixed CUDA 13 packages hold all of them, so full compilation with `nvcc` works: outcome (a).
- **No wheel ships `cublas.lib`.** The provider's import library is built at build time from
  `cublas64_13.def`, which lists the nine cuBLAS functions it calls, with MSVC's `lib /def`. A new cuBLAS call
  needs a new line in that file.
- **The executables link the CUDA runtime statically** (`cudart_static.lib`, CMake's default), so at run time they
  need only `cublas64_13.dll` and `cublasLt64_13.dll`, found through the `PATH`.

Newer CUDA 13 wheels exist (13.4.92 at the time of writing). They were not tried: a newer runtime than the
driver's is outside what this machine was tested with.

## Set up, build, test, run

From PowerShell in the repository root. Requirements: `uv`, Visual Studio 2022 with the C++ workload (it brings
CMake and Ninja), an NVIDIA GPU of the RTX 30 series or newer with a driver that supports CUDA 13.1.

```powershell
# 1. The toolchain (once; -Force deletes and recreates it)
pwsh runner/cuda/setup_toolchain.ps1

# 2. Configure, build, test
cmd /c '"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cd runner\cuda && cmake --preset release && cmake --build --preset release && ctest --preset release'

# 3. Run the tools directly: cuBLAS must be on the PATH (CTest sets this up by itself)
$env:PATH = "$PWD\runner\cuda\.toolchain\Lib\site-packages\nvidia\cu13\bin\x86_64;$env:PATH"
runner\cuda\build\release\sepal_cuda_tests.exe            # or one suite: contract, parity_small, parity_real, rows
runner\cuda\build\release\sepal-cuda-bench.exe
runner\cuda\build\release\sepal-cuda-bench.exe --presets 0.8b --tokens 1024 --precisions f16 --reps 9
```

CMake options: `-DSEPAL_CUDA_ROOT=<toolkit root>` uses another CUDA 13 toolkit tree;
`-DCMAKE_CUDA_ARCHITECTURES="75;86"` adds Turing (the default is 86, Ampere and newer). The build takes the engine
from `runner/` with `add_subdirectory(.. sepal_core_build)` and switches the engine's own tests off; those belong
to the parent project. Warnings are errors for the C++ (`/W4 /WX`) and for the CUDA source (`-Werror=all-warnings`
and `/W4 /WX` for the host pass).

The CMake file has a branch for Linux. It has never been run.

## The provider

`include/sepal/cuda/cuda_gemm.hpp`, `src/cuda_gemm.cpp`, `src/convert_kernels.cu`.

- **Interface.** `sepal::cuda::CudaGemm` implements `sepal::GemmProvider::gemm_f`, with the same argument checks
  and messages as the reference provider.
- **Arena.** One `cudaMalloc` when the provider is created. Weights, device matrices and the cuBLAS workspace are
  taken from its bottom and stay; what a call borrows is taken from the rest and returned when the call ends,
  including when it throws. A call that does not fit throws and names the need, the free space and the option to
  raise.
- **Weights.** `upload_weights` copies a matrix to the device once and returns an id; `register_weights` also
  makes `gemm_f(x, w, ...)` recognise that host matrix (by pointer, element type and shape) and use the device
  copy. An unregistered `w` is uploaded for that call only.
- **Precision.** One path per provider. `F32`: fp32 operands (fp16 and bf16 inputs widened exactly),
  `cublasSgemm`. `F16`: fp16 operands (fp32 and bf16 inputs rounded to nearest on the device), `cublasGemmEx` with
  fp32 accumulation and fp32 output. cuBLAS accepts fp16 operands with an fp32 result directly, so no fp16 result
  is ever rounded.
- **Stream.** Every device call runs on a non-blocking stream the provider owns.
- **Errors.** `sepal::cuda::CudaError` carries the name of the failing call, the code and the library's text,
  for example `cudaSetDevice failed: invalid device ordinal (code 101)`.
- **Device-resident calls.** `upload_matrix`, `alloc_output`, `gemm_device`, `synchronize`, `download`: multiplies
  whose operands are already on the device and that do not wait. The benchmark uses them; the executor will.

## Parity with the reference provider

`sepal_cuda_tests parity_real`: random matrices at the six projection shapes of the 0.8B preset (hidden 1,024,
MLP 3,584) and of the 4B preset (hidden 2,560, MLP 9,216), 8 rows each, against the reference provider, which
accumulates in float64. Outputs have rms 0.33 and largest magnitude 1.8. `max_rel` is the largest elementwise
relative difference over the elements at least as large as the rms. Worst case over the twelve shapes:

| Path | Operands given to both providers | max abs difference | max relative difference |
|---|---|---|---|
| fp32 | fp32 | 5.960e-07 | 1.155e-06 |
| fp16 | fp16 | 1.216e-05 | 1.921e-05 |
| fp16 | fp32, which the CUDA provider rounds to fp16 (the reference keeps fp32) | 5.339e-04 | 1.088e-03 |

The second row is the arithmetic of the fp16 path alone; the third adds the cost of storing the operands in fp16.

## Row independence: the finding the design needs

The interface asks that an output row depend on its input row and the weights and on nothing else, and that a
row have the same bits alone or in any batch. `sepal_cuda_tests rows` asks four questions at the twelve real
shapes, with batch sizes T = 1, 2, 3, 7, 16, 64, 256 and 1,024.

| Question | fp32 | fp16 |
|---|---|---|
| Does a row depend on the **values** of the other rows in the batch? | No: bit-identical, 12 of 12 shapes | No: bit-identical, 12 of 12 |
| Does a row depend on its **position** in a batch of a given size? | No: bit-identical, 12 of 12 | No: bit-identical, 12 of 12 |
| Is a row alone bit-identical to the same row in a batch? | **No**, 0 of 12 shapes; 92-97% of elements differ | **No**, 0 of 12; up to 99.6% of elements differ |
| Do all batch sizes give one bit pattern? | **No**, 0 of 12 | **No**, 0 of 12 |
| Largest difference between batch sizes | 4.530e-06 (1.4e-05 of the output rms) | 1.252e-05 (3.8e-05 of the output rms) |

cuBLAS picks its kernel from the shape of the call, so a row's bits are a function of the row, the weights and the
batch size T. The groups of batch sizes that agree differ from shape to shape and follow no simple rule: for the
4B MLP up-projection in fp32 they are {1} {2,3,7} {16} {64,1024} {256}; for the 0.8B DeltaNet in-projection in
fp16 they are {1,2,3,7,16,256,1024} {64}. cuBLAS's pedantic mode (`CudaGemmOptions::pedantic`) changes none of the
four answers.

What this means for docs/RUNNER.md: with cuBLAS, "a cache hit and a cache miss compute identical bits" (section 6,
idea 3) holds only if every multiply of a given layer is called with the same T whatever the batch contains. That
is true by construction if T is always padded to one fixed size, and false if T is bucketed to several sizes or
left as it comes. The alternatives are a kernel of our own with a fixed summation order, which `nvcc` now makes
possible, or relaxing that gate to a tolerance of about 1e-4 of the output rms per multiply. This slice does not
choose.

## Benchmark

`sepal-cuda-bench`, pasted from its output. **These are the matrix multiplies only, so they are an upper bound on
engine throughput, not the engine's throughput.** Random weights and activations; fp32 outputs; 2 warm-up passes,
then the median of 5 timed passes, each ending with a wait for the device. "all GEMMs" is every projection of every
layer of a preset (the concatenated mixer in-projection, the mixer out-projection, the MLP's gate and up as one
multiply, the MLP's down), each with its own weights: 96 multiplies for 0.8B and 2B, 128 for 4B.

```
                   |  MLP layer   MLP layer         |  MLP layer   MLP layer |  all GEMMs   all GEMMs        
model prec  tokens |  device ms    tokens/s TFLOP/s |    host ms    tokens/s |  device ms    tokens/s TFLOP/s
-----------------------------------------------------------------------------------------------------------
0.8b  fp16      64 |      0.122      523732   11.53 |      0.569      112478 |       5.33       12011   11.95
0.8b  fp16     256 |      0.267      960600   21.15 |      1.577      162292 |      13.26       19306   19.21
0.8b  fp16    1024 |      0.976     1049503   23.11 |      5.369      190742 |      44.14       23201   23.09
0.8b  fp16    4096 |      3.694     1108855   24.42 |     20.816      196771 |     173.11       23661   23.55
2b    fp16      64 |      0.318      201195   15.19 |      1.001       63930 |      12.85        4979   13.67
2b    fp16     256 |      0.898      284951   21.51 |      2.991       85584 |      34.70        7377   20.25
2b    fp16    1024 |      3.156      324502   24.50 |     10.868       94217 |     118.49        8642   23.73
2b    fp16    4096 |     13.103      312593   23.60 |     42.811       95677 |     482.03        8497   23.33
4b    fp16      64 |      0.575      111324   15.76 |      1.480       43258 |      28.36        2257   16.11
4b    fp16     256 |      1.630      157094   22.24 |      4.537       56424 |      84.45        3032   21.64
4b    fp16    1024 |      5.737      178478   25.26 |     17.172       59633 |     296.10        3458   24.69
4b    fp16    4096 |     23.478      174464   24.70 |     66.933       61196 |    1216.16        3368   24.04
0.8b  fp32      64 |      0.259      247582    5.45 |      0.716       89348 |      12.99        4927    4.90
0.8b  fp32     256 |      0.782      327491    7.21 |      2.066      123929 |      37.92        6750    6.72
0.8b  fp32    1024 |      2.986      342934    7.55 |      7.562      135409 |     143.54        7134    7.10
0.8b  fp32    4096 |     12.508      327476    7.21 |     28.080      145868 |     567.26        7221    7.19
2b    fp32      64 |      1.121       57117    4.31 |      1.826       35044 |      41.50        1542    4.23
2b    fp32     256 |      2.540      100803    7.61 |      4.712       54331 |     101.30        2527    6.94
2b    fp32    1024 |     10.872       94183    7.11 |     17.727       57766 |     391.73        2614    7.18
2b    fp32    4096 |     42.692       95942    7.24 |     69.572       58874 |    1554.50        2635    7.23
```

- "MLP layer device": the two multiplies of one MLP with operands on the device. "MLP layer host": the same two
  through `gemm_f`, which uploads X and downloads Y on every call; in fp16 that path takes 2.6 to 5.6 times as
  long as the device path, which is why activations must stay on the device.
- **4B in fp32 was not measured**: its weights are 13.30 GiB and 10.98 GiB of the card was free. The tool said so
  and skipped it.
- **Peak device memory** (the arena is allocated once; "took" is the fall in the driver's free-memory figure from
  just before the arena to its lowest point, so it includes cuBLAS but not the CUDA context):

  | Case | Weights | Arena | Arena used at peak | Provider with cuBLAS took |
  |---|---|---|---|---|
  | 0.8B fp16 | 0.93 GiB | 1.342 GiB | 1.267 GiB | 1.355 GiB |
  | 2B fp16 | 2.56 GiB | 3.182 GiB | 3.073 GiB | 3.193 GiB |
  | 4B fp16 | 6.65 GiB | 7.590 GiB | 7.421 GiB | 7.602 GiB |
  | 0.8B fp32 | 1.85 GiB | 2.347 GiB | 2.237 GiB | 2.357 GiB |
  | 2B fp32 | 5.11 GiB | 5.849 GiB | 5.676 GiB | 5.859 GiB |

  The arenas are sized for the largest case measured, 4,096 tokens, and hold the benchmark's activation buffers.
- The fp16 path runs at 23 to 25 TFLOP/s from 1,024 tokens up, and the fp32 path at about 7.2: fp16 is 3.3 times
  faster on this card. At 64 tokens both paths lose a third to a half of that.
- What the tool does not measure: the fp32-to-fp16 conversion of activations between multiplies that an fp16
  engine needs, and everything in the right-hand column of the first table.

### Against the design's derived figures

docs/RUNNER.md section 12 derives its GPU figures from other people's measurements. Its "8 GB GPU" row is the one
anchored on a llama.cpp report for an RTX 3060; it gives alerts per second without a cache for a 1,000-token
alert, which is the same number in thousands of tokens per second. The measured column is "all GEMMs" in fp16 at 1,024 tokens. The
last column multiplies it by 0.8, the design's own assumed GEMM share of a pass; that factor is the design's
assumption, not a measurement of ours.

| Model | Design, derived (tokens/s) | Measured, GEMM only | Measured x 0.8 | Reading |
|---|---|---|---|---|
| 0.8B | 10,000 to 20,000 | 23,201 | 18,561 | Consistent: inside the range, near its top |
| 2B | 4,000 to 8,000 | 8,642 | 6,914 | Consistent: inside the range |
| 4B | 3,200 to 4,300 | 3,458 | 2,766 | Optimistic: the upper bound alone is 20% under the top of the range, and with the 0.8 factor it is 14% under the bottom |

Against the stage exit criteria (section 16), which are full-engine targets, so the GEMM-only figure can only
show whether they are reachable:

- R2, "at least 10,000 tokens/s on 0.8B": the multiplies alone run at 23,201, 2.3 times the target. Reachable.
- R2, "at least 3,500 tokens/s on 4B": the multiplies alone run at 3,458 (1,024 tokens) and 3,368 (4,096 tokens),
  1% and 4% **under** the target before anything else in the pass is counted. A pass that runs every multiply
  through fp16 cuBLAS on this card cannot meet it; with the 0.8 factor the expectation is about 2,770 tokens/s,
  21% under.
- R3, "at least 4 alerts/s on 4B without a cache": one 1,024-token pass of multiplies takes 296 ms, so about 3.4
  alerts per second is the ceiling for such a pass.

The 4B targets need either a faster multiply than cuBLAS fp16 gives here (the int8 paths of R4 are the design's
candidate, unmeasured) or less work per alert (last-layer pruning, the prefix cache).

## What should move to the GPU next

1. Keep activations on the device between multiplies: an executor that owns device buffers and calls
   `gemm_device`, with the residual stream in fp32 and an fp16 copy made by the conversion kernel.
2. The elementwise and per-row operations, as `nvcc`-compiled kernels: RMSNorm, SiLU and the SwiGLU product,
   residual additions. They are small but sit between every pair of multiplies.
3. The DeltaNet scan: convolution, the delta-rule recurrence and the gated norm. It is the only sequential part
   and the one the design adapts from Strata.
4. Attention: per-head norm, RoPE, the K/V cache and the tiled causal kernel.
5. Decide the row-independence question above before the executor fixes how T is chosen.

## Files

| File | What it is |
|---|---|
| `toolchain-requirements.txt`, `setup_toolchain.ps1` | The pinned wheels and the script that recreates `.toolchain/` |
| `CMakeLists.txt`, `CMakePresets.json`, `cublas64_13.def`, `cublasLt64_13.def` | The build; the import-library definitions of the cuBLAS and cuBLASLt entry points used |
| `include/sepal/cuda/cuda_gemm.hpp`, `src/cuda_gemm.cpp` | The provider |
| `src/convert_kernels.hpp`, `src/convert_kernels.cu` | The element-type conversion kernels |
| `src/gemm_kernels.hpp`, `src/gemm_kernels.cu` | The batch-invariant tensor-core multiply (binary16 or float32 accumulators) and the 2-D widening used by binary16-accumulating cuBLAS multiplies |
| `tests/test_cuda_gemm.cpp` | Suites `contract`, `parity_small`, `parity_real`, `rows` |
| `tests/test_gemm_kernels.cpp` | Suites `gemmk_convert`, `gemmk_accuracy`, `gemmk_invariance`, `gemmk_edge` (CTest `cuda_gemmk_*`) |
| `src/gemm_s8_kernels.hpp`, `src/gemm_s8_kernels.cu`, `src/gemm_ptx.cuh` | The int8 multiplies (int8 activations on `mma.m16n8k32`; binary16 activations with int8 weights), the row and weight quantisers, and the PTX helpers they share with `gemm_kernels.cu` |
| `tests/test_gemm_s8_kernels.cpp` | Suites `s8_quantize`, `s8_gemm`, `s8_invariance`, `s8_edge` (CTest `cuda_s8_*`) |
| `tests/test_cuda_engine_multiply.cpp` | Suites `engine_multiply` (the multiply options on a tiny geometry) and `engine_int8` (the int8 options) |
| `tests/test_cuda_engine_real_int8.cpp` | Suites `real_int8` and `real_int8_2b`: accuracy of every int8 variant on the real checkpoints |
| `tools/sepal_cuda_bench.cpp` | `sepal-cuda-bench` |
| `tools/gemm_probe.cpp` | `sepal-cuda-gemm-probe`: cuBLAS, cuBLASLt (every candidate algorithm) and the own kernel at the engine's shapes, with their error |
| `tools/engine_ab.cpp` | `sepal-cuda-engine-ab`: engine option sets compared inside one process |
| `tools/int8_probe.cpp` | `sepal-cuda-int8-probe`: int8 against binary16 multiplies (cuBLAS, cuBLASLt, the own kernels) at the engine's shapes |

## Attention kernels

`src/kernels_attention.hpp`, `src/kernels_attention.cu`: the full-attention mixer's work between the q/k/v
multiply and the `o_proj` multiply. Library `sepal_cuda_attention`, tests `sepal_cuda_kernel_attention_tests`
(CTest `cuda_attention_*`), benchmark `sepal-cuda-attention-bench`. The contract is in the header; in short:

- Two launches per layer over many rows: `launch_attention_prepare` (per-head RMSNorm of q and k, partial RoPE,
  the row's keys and values written to its own segment) and `launch_attention` (causal grouped-query attention,
  times sigmoid(gate), written in the `o_proj` operand type). Launch prepare for all rows, then attention for all
  rows: no waves are needed, because a branch needs its parent's keys and values, not its output.
- K/V layout: a segment is `[tokens, kv_heads, head_dim]` for keys and again for values, float32 or binary16.
  A row lists its segments (parent chain, oldest first, its own last) in a device table; segments are read in
  place, never copied. Three arrays describe a launch: `RowSpan`, `AttentionRowDesc`, `KvSegment`.
- Paths: `Reference` (a plain kernel, either type, any head_dim up to 256) and `Tiled` (mma.sync tensor cores,
  binary16 only, head_dim 16/64/128/256, group size 1 to 16). Both reduce in an order that depends only on the
  row's own chain, so a token's bits do not depend on other rows, their number, or the row's place in the list.
  RoPE angles are float64 products with float64 sine and cosine, as in the reference, and match it bit for bit
  at every position from 0 to 75,775.
- The cuBLAS import library needs no new function: nothing here calls cuBLAS.

Time per layer in ms (minimum of 30 synchronised repetitions on the shared RTX 3060; 0.8B and 2B have the same
attention geometry, 8 heads over 2 KV heads of 256, and measured alike), `sepal-cuda-attention-bench`:

| Request | f32 plain | f16 plain | f16 tiled (prepare + attention) |
|---|---|---|---|
| (a) 16 rows x 640 | 43.0 | 39.8 | 2.88 (0.68 + 2.19) |
| (b) 1 row x 4096 | 135.3 | 107.3 | 3.73 (0.28 + 3.45) |
| (c) 8 states x 640 + 64 branches x 27 | 47.3 | 39.2 | 2.34 (0.46 + 1.87) |

## DeltaNet kernels

`src/kernels_deltanet.hpp`, `src/kernels_deltanet.cu`: the Gated DeltaNet mixer's work between the four input
multiplies and the `out_proj` multiply. Library `sepal_cuda_deltanet_kernels`, tests `sepal_cuda_kernel_deltanet_tests`
(CTest `cuda_deltanet_*`), benchmark `sepal-cuda-deltanet-bench`. The contract (rows, slots, waves, taps for tests) is
in the header and unchanged by the rewrite described here; the engine calls `launch_deltanet_wave` as before.

**What one launch does.** One block per (row, value head). The block holds two kinds of warps that work at the same time:

- *scan warps* (128 threads for dv = 128) keep the head's 128 x 128 state in registers. A thread owns a tile of four
  adjacent columns and a quarter of the keys; the four lanes of a column quad meet only to complete a dot product (two
  shuffles). The recurrence is all they do.
- *input warps* (128 threads) run one chunk of 16 tokens ahead of the scan: the causal convolution and SiLU of the
  q, k, v channels, the per-token decay and beta, the L2 normalisation of q and k. Their global loads are issued a step
  ahead; the scan finds each chunk finished in one of two shared-memory stage buffers. The gated RMSNorm of the scan's
  reads, written in the `out_proj` operand type, is done by the same warps, one chunk behind the scan.

Named barriers (full/empty pairs for the stage buffers and for the read buffers) pass the buffers back and forth, so
the global-memory latency and the arithmetic of the convolution, the norms and the SiLU overlap the recurrence instead
of stopping it. The last token of a chunk takes the first token of the next chunk in the same pass over the registers.

**Why this shape (measured, RTX 3060).** Phase by phase, the earlier one-column-per-thread kernel spent 4.3 ms of its
7.0 ms (16 rows x 640) in the staging phases (convolution, normalisations, gated norm), which ran while the recurrence
waited, and 3.5 ms in the recurrence; the recurrence alone was limited by shared-memory reads of the k and q vectors
(a broadcast 128-bit read costs the SM about 4 cycles whatever the lanes read), not by the multiply-adds. Giving every
thread four columns divides those reads by four, and running the staging in its own warps hides it. A microbenchmark of
the recurrence alone, cycles per (token, head) per SM: 926 with one column per thread, 704 with four columns, 595 with
four columns and no per-element decay multiply.

**The decay without a multiply per state element.** The state is kept as `tile = S / g` with a scalar `g` (the product
of the decays since the last fold). A token's decay multiplies `g`; `recalled = g (k.tile)`, the correction is
`beta (v - g (k.tile))`, `tile += k (correction / g)`, `read = scale * g (q.tile)`. The first token of every chunk (and
any later token that would take `g` below 2^-40, flagged per chunk) folds the accumulated decay into the tile with one
multiply per element; at the end of a row `tile * g` is stored. That is three operations per state element per token
instead of four. The scalars come from the input warps as a short sequential walk over the chunk, a function of the
row's own tokens.

**What stays and what changes in the numbers.** Unchanged: a token's bits depend only on its own row and parent chain
(not on the other rows, their number, the wave, or the row's place), one fixed reduction order per output element, no
float atomics, `--fmad=false`, every multiply-add an explicit round-to-nearest intrinsic; the divisions are correctly
rounded (the reciprocal refined by two fused steps, the fast path of `__fdiv_rn` without its branch). Changed: the
results differ from the previous kernel in the last bits (the reduction tree and the scaled state are different), and a
row cut in two and continued from the first part's stored state now agrees with the uncut row to rounding
(1e-6 of the rms of the output) instead of bit for bit, because the decay scale restarts at the cut. Cuts happen
at the same places in every call the engine makes, and the engine's strict-mode suites (a branch from a snapshot against
the same branch in one call, row permutations, rows among other rows) are bit-identical. `lanes_per_column` is accepted
and ignored (one thread mapping); more than 4 conv taps use a run-time tap count.

Accuracy against the float64 reference (`growth` suite, 0.8B geometry, one row, nonzero start; max |difference| over the
rms of the reference): after 4096 tokens the output is 9.1e-6 off (before: 9.8e-6) and the state 4.0e-6 (before 3.9e-6);
after 1024 tokens 5.7e-6 (before 6.8e-6) and 4.4e-6 (4.5e-6). The float32 CPU reference is 1.3e-5 off at 4096 tokens.

Time per layer in ms, `sepal-cuda-deltanet-bench --presets 0.8b` (minimum of 30 repetitions, the earlier kernel and the
new one run alternately in one session on the otherwise idle card; the 1 x 600 and 1 x 2048 rows from a paired run of
both kernels in one process):

| Request | before | after | speed-up |
|---|---|---|---|
| (a) 16 rows x 640 | 7.05 | 4.27 | 1.65x |
| (b) 1 row x 4096 | 5.72 | 2.66 | 2.15x |
| (c) 8 states x 640 + 64 branches x 27 | 5.49 | 3.86 | 1.42x |
| (d) 1 row x 600 | 0.85 | 0.43 | 1.99x |
| (e) 1 row x 2048 | 2.85 | 1.35 | 2.11x |

DeltaNet time per engine call in ms (`sepal-cuda-engine-bench --sections profile`, float16, natural batching, 18
DeltaNet layers, median of 3 runs of each build run alternately; the profile mode synchronises after each phase):

| Call | before | after | speed-up | wall before -> after |
|---|---|---|---|---|
| 1 row x 64 | 1.77 | 1.18 | 1.5x | 10.4 -> 9.6 |
| 1 row x 600 | 15.27 | 7.50 | 2.04x | 49.5 -> 41.2 |
| 1 row x 2048 | 51.67 | 24.23 | 2.13x | 151.1 -> 124.1 |
| 16 rows x 640 | 126.14 | 75.17 | 1.68x | 577.9 -> 529.3 |
| 1 alert per call | 19.94 | 10.88 | 1.8x | 64.4 -> 55.8 |
| 16 alerts per call | 171.46 | 117.76 | 1.46x | 780.7 -> 731.1 |
| 16 alerts, states cached | 43.23 | 39.47 | 1.10x | 207.0 -> 205.3 |

The long single rows improved by a factor of two; the batched calls by less. In a 16 x 640 call the recurrence is no
longer what limits a block: a chunk of 16 tokens takes the scan about 16.6k cycles when it never waits and the input
warps about 18k for their convolution, normalisation and output work, so the two roughly balance and the card is run
at one block per SM (255 registers per thread, 66 KB of shared memory). The short branch rows of the alert workload
(26 tokens) pay a pipeline fill (the first chunk's convolution before the recurrence can start) that is about half of
their block time. Ideas not done: a cheaper convolution/SiLU stage (it is the largest part of the input warps' time),
a separate pair of warps for the output stage (needs fewer registers in the input warps to fit), a smaller first chunk,
and a chunked formulation of the recurrence on tensor cores (not attempted: in float32 it does the same multiply-adds
and a half-precision form would not meet the accuracy bars).

Tests added: strong decays (down to exactly zero) at every position of a chunk against the float64 reference, and the
same row among other rows bit for bit; conv tap counts 1, 2, 3, 5 and 8; chunk-boundary lengths at the real size; the
continuation test now checks agreement to rounding. The tests were checked by breaking the kernel on purpose (dropping
the fold of the decay scale, and dropping the final multiply by `g`): 10 and 14 of the 20 cases failed.

## The GPU engine

`include/sepal/cuda/cuda_engine.hpp`, `src/cuda_engine.cpp`, `src/engine_kernels.{hpp,cu}`: `CudaEngine` runs a Qwen3.5
hybrid model (Gated DeltaNet and full-attention layers) on the GPU, from a `LoadedModel` (a checkpoint, bf16/f16/f32
converted on the device) or from a config and `Weights<float>`. Library `sepal_cuda_engine`; tests
`sepal_cuda_engine_tests` (CTest `cuda_engine_*`); benchmark `sepal-cuda-engine-bench`; workload generator
`tools/make_alert_workload.py` at the repository root.

```cpp
sepal::cuda::CudaEngine engine(sepal::LoadedModel::open(dir), {.precision = Precision::F16, .gemm_rows = 0});
std::vector<sepal::cuda::EngineRow> rows(9);          // a state and eight questions about it
rows[0].token_ids = state;                             // rows[0].keep_snapshot = true would store the state
for (int k = 0; k < 8; ++k) {
    rows[k + 1].token_ids = suffix[k];
    rows[k + 1].parent = sepal::cuda::ParentRow{0};    // or a DeviceSnapshot from an earlier call
    rows[k + 1].readout_offsets = {suffix[k].size() - 1};
}
auto results = engine.run(rows);                       // results[k + 1].hidden: final-normed hidden state, float32, host
```

- **One call, one flat token list.** All rows go through the stack together, layer by layer; every multiply runs once
  over all tokens; DeltaNet rows whose parent is a row of the same call run in a later wave; attention reads parents'
  keys and values in place. Semantics are those of `sepal::ref::run_rows` (a row continues a snapshot, another row of
  the call, or nothing; rows sharing a parent never see each other; positions are explicit or follow the parent).
- **Device snapshots** (`DeviceSnapshot`): immutable, reference counted; own a DeltaNet state slot and the row's keys
  and values (one heap block, all attention layers) and keep the parent's keys and values alive. Memory is taken at
  creation out of `memory_budget_bytes` (weights, call buffers for `max_call_tokens`, a slot pool, a key/value heap; without a budget, pools for
  `pool_snapshots` snapshots, see "Snapshot density and tiers");
  `run` allocates nothing on the device, throws `BudgetExceeded` before doing any work when a call does not fit, and the
  engine stays usable.
- **Multiply batching.** `gemm_rows = 0` is natural (one multiply over all tokens). `gemm_rows = T` is strict: every
  multiply is cut into multiplies of exactly T rows, the last zero-padded, so a row's bits do not depend on the rest of
  the call. All of the engine's own kernels are batch-invariant. With `multiply_path = Invariant` the multiplies are too
  (the engine's own kernel, see "Multiply throughput" below), and strict mode then pads nothing.
- **Residual ladder**: `EngineRow::capture_layers` returns every layer's output for that row.
- **Tuning that is on by default**: the last layer's queries, attention, `o_proj` and MLP are computed for the tokens
  that are read out only (`prune_last_layer`; its keys and values are still stored for every token; bit-identical to
  the full pass in strict mode, which needed a fix, see the adversarial review below); on the float16 path the MLP's gate and up multiply writes binary16 (`fp16_gate_up`).
  Measured on the same session, both off then on: 16 rows x 640 16.3k -> 17.2k tokens/s, 4800 tokens 12.7k -> 13.3k, 16
  alerts per call 18.9 -> 19.8 alerts/s, 16 alerts with the states cached 72 -> 75 alerts/s.

### Correctness (RTX 3060, the numbers `sepal_cuda_engine_tests` prints)

| Test | Result |
|---|---|
| float32 engine vs CPU reference executor, tiny geometry, 400 random calls (2,529 rows, 75,491 tokens, 737 snapshots kept, 328 rows continuing stored snapshots) | largest hidden-state difference 1.83e-4; 3 of 400 calls over 1e-4, all explained: the float32 reference is itself 1.7e-4 from float64 on the same calls (engine 1.4e-4) because the random weights amplify float32 rounding noise through the DeltaNet layers (`diag` suite; without stored parents, 1,000 calls: engine 4.7e-4 and reference 4.3e-4 from float64) |
| float32 engine vs transformers float32, jeff-base, cases i-iv | every layer of cases i and ii within 1.05e-5; final hidden within 2.6e-5; option probabilities within 1.6e-6; case iv as one call and as a stored snapshot continued by a later call agree to 1e-5 |
| float16 engine vs float32 golden, jeff-base, 6 prompts | |Δhidden| max 6.9e-3, mean 9.7e-4; |Δp| max 7.4e-4, mean 1.6e-4; no answer flip. transformers' own float16: 1.95e-2 / 1.6e-3 and 5.0e-4 / 2.5e-4 |
| Qwen3.5-2B-Base, float32 engine vs CPU reference (51 tokens, state with two branches) | 5.2e-5; float16 vs float32 engine: max 1.5e-2, mean 1.5e-3 |
| strict mode (tiny, float32 and float16, gemm_rows 8/16/24/32/64; real model, 128 and 1024) | a row alone vs among other rows at random places: bit-identical in every trial; branch from a stored snapshot vs the same branch run with its state: bit-identical (118 of 118 tiny, 18 of 18 real, plus three-deep chains); permuting the rows of a call: 420 of 420 rows and 420 of 420 left-behind snapshots tiny, 111 of 111 rows real; repeated runs identical in both modes |
| natural mode drift | real model, 36 branch rows alone vs in calls of 4 to 16 rows: every row differs in bits; float16 max |Δp| 7.3e-4; float32 max |Δp| 1.6e-6 |
| multiply options (`gemmk_*`, `engine_multiply`, `real_f16`, `real_strict`, `real_2b`) | the own kernel against float64: float32 accumulation 2e-7 to 6e-6 relative rms, binary16 accumulation with a flush every 64 / 128 / 256 values of k 4.4e-4 / 6.1e-4 / 8.4e-4; a row's bits identical across six tile shapes, batches of 1 to 213 rows and every place in a batch; in the engine, with either accumulation, natural and strict (`gemm_rows` 0, 16, 48, 128) on the tiny and the real geometries: every row bit-identical alone, among other rows and continued from a stored snapshot; cuBLAS with binary16 accumulation in strict mode is bit-identical only since the last layer runs the same two multiplies pruned or not (before: 0 of 18 snapshot continuations identical); accuracy against the golden in "Multiply throughput" below |
| memory | 3,000 calls creating and releasing snapshots: device memory before and after identical, pools back to zero use, heap fully merged; budget overrun fails with `BudgetExceeded`, nothing is leaked, the next call gives the same bits; snapshots outlive their parents' handles and the engine |
| refusals | bad token ids, empty rows, readouts out of range, cyclic or dangling parents, empty handle, snapshot of another engine or geometry, too many rows/tokens: `std::invalid_argument`, nothing left allocated |

The real-model suites (`real_f32`, `real_f16`, `real_strict`, `real_natural`, `real_2b`, `real_alerts`, `real_cpu`, `real_long`,
`real_soak`, `real_diag`) print `SKIPPED: real-model ...` unless `SEPAL_REAL_MODEL` (checkpoint directory), `SEPAL_REAL_GOLDEN`
(`tools/export_real_golden.py` output) and, for `real_2b`, `SEPAL_REAL_MODEL_2B` are set (`real_alerts`, `real_cpu`, `real_long`,
`real_soak` need `SEPAL_ALERTS` = `runs/alert_workload.json` and `SEPAL_ALERT_GOLDEN` instead of `SEPAL_REAL_GOLDEN`); CTest
shows them as Skipped (`SKIP_REGULAR_EXPRESSION`), not Passed. `SEPAL_GPU_LOCK` (a directory) makes `real_cpu` take a lock
around its use of the device; `SEPAL_CPU_CACHE` (a directory) keeps the CPU reference's answers between runs.
Long runs: `sepal_cuda_engine_tests soak` (`SEPAL_SOAK_MINUTES`, `SEPAL_SOAK_SEED`) is not in CTest.

### Adversarial review (a second reviewer who did not write the engine)

Tests: `test_cuda_engine_attack.cpp` (suites `attack_shapes`, `attack_invariance`, `attack_lifetime`),
`test_cuda_engine_attack_memory.cpp` (`attack_memory`, `attack_deepchain`, `attack_determinism`, `attack_concurrency`),
`test_cuda_engine_attack_geometry.cpp` (`attack_geometry`), `test_cuda_engine_soak.cpp` (`soak_short` in CTest; `soak` and
`real_soak`), `test_cuda_engine_real_alerts.cpp` (`real_alerts`, `real_cpu`, `real_long`, `real_diag`); helpers in
`tests/attack_util.hpp`. `tools/export_alert_golden.py` writes the float32 transformers hidden state of all 512 prompts of the
alert workload (`runs/alert-golden/alert_golden.safetensors`), which `real_alerts` compares the engine with.

What was found and fixed:

1. **Strict mode was not batch-invariant with `prune_last_layer` on.** In the pruned last layer the multiply of the queries
   and gates and the one of the keys and values were narrower than the full pass's single `q|gate|k|v` multiply, and a
   multiply's bits can depend on its width. Whether a call is pruned depends on the other rows of the call (the number of
   read-out rows against the number of tokens), so a row's bits depended on its neighbours. It showed on the real model in float32
   (a state in the call versus the same state stored: 0 of 3 branches identical for states of 64 tokens and more at
   `gemm_rows` 128; with pruning off 3 of 3) and on a 256-wide geometry in float32 and float16 (16 of 110 rows identical).
   Strict mode now multiplies the full width for every token in that layer and gathers the read-out rows afterwards (one more
   call buffer, `2 * attn_width` floats per read-out row: 16 MiB at 0.8B), so pruning changes no bit; natural mode keeps the
   narrow multiplies.
2. A failed `CudaEngine` construction (budget too small, weights of another geometry, budget above the card) leaked the cuBLAS
   handle, the stream and the events: 5.6 MiB of device memory per failure. They are released now.
3. Float16 with an odd hidden size threw from `gather_rows` in the middle of a call when pruning was on; pruning is switched off
   for such a geometry.
4. Threads: `run` is serialised by a mutex (documented in the header); `run` now sets its device, and `memory()` and
   `last_profile()` no longer race with other calls.

Not changed, but true and worth knowing:

* Attention locates each key by scanning the segment list (`locate_key`), so one query over a chain of S snapshot links costs
  O(S^2): the attention phase of a one-token call takes 3.8 ms at 250 links, 15 ms at 500, 33 ms at 750 and 60 ms at 1000.
  `KvBlock` destruction recurses along the chain; chains deeper than 400 were not released in a test (building one is too slow).
* `run` refuses a start position whose `start + chain tokens` reaches 2^32 in `check_attention_rows`, after the call's memory
  was taken (it is given back), not at admission.
* More than `max_call_tokens` read-outs of a single row (repeated offsets) are refused even though they are not extra tokens.

Measured on the 64-alert workload (jeff-base, 512 prompts, transformers float32 as the reference; `real_alerts`):

| engine and way of running | |Δp| median / p99 / max | max abs Δhidden | answer flips |
|---|---|---|---|
| float32 natural, alone / 16 per call / reversed / states cached / questions split / one question per call | 5e-7 / 3.5e-6 / 4.2e-6 (alone); worst of the six 5.6e-6 | 4.4e-5 (worst of the six 5.6e-5) | 0 |
| float32 strict 128, all six ways | 4.9e-7 / 2.8e-6 / 4.9e-6; the six ways are bit-identical | 3.6e-5 | 0 |
| float16 natural, six ways | 2.0e-4 / 9.3e-4 / 1.7e-3 | 1.1e-2 (1.5e-2 worst way) | 0 against float32; 1 between two float16 batch compositions (alert 11 q2, p = 0.49975 against 0.50055) |
| float16 strict 128, six ways | 1.9e-4 / 8.9e-4 / 1.2e-3; the six ways are bit-identical | 1.4e-2 | 0 |

No row is an outlier (the largest |Δp| is under 2.2 times the 99th percentile in every row of the table), and the largest
differences sit at question 1 of a few alerts in every configuration: those are the prompts nearest the decision boundary. The
CPU float32 reference, on alerts 0 and 63 (1,723 tokens, 528 s, 0.31 s per token): 1.7e-6 from transformers in probability,
3.1e-5 in hidden state; the float32 engine is 2.1e-6 / 3.3e-5 from it and the float16 engine 9.6e-4 / 7.8e-3.
Soaks (seeded; the seed is printed first and `SEPAL_SOAK_SEED` replays the requests). Tiny geometry, six engines on the same
random requests (parent rows in any order, stored snapshots, explicit positions, readouts, captures, kept and released
snapshots, small-pool engines that now and then refuse a call): 22 minutes, seed 20261010, 20,195 calls, 205,388 rows,
3,484,404 tokens, 71,513 snapshots kept: 0 failures; the float32 engine is within 3.7e-3 of the CPU reference (rows of one to a
few tokens through random weights amplify rounding; the engine and the reference are equally far from float64 there) and the
two strict engines (small pools against roomy pools, other rows spliced in, last-layer pruning off) are bit-identical on all
120,444 compared rows. A first run (seed 20261009, 21,129 calls) agreed on every strict row (126,082 of 126,082). On the real model
(`real_soak`, requests cut from the workload's own text, seed 778, 150 s per phase): 3,204 calls, 10,315 rows, 699,275 tokens
with the float16 strict engines bit-identical on 7,704 of 7,704 read-out rows (1,345 calls refused by the small-pool engine);
then 759 calls, 7,516 rows, 470,635 tokens float32 against float16 natural, 5,622 read-outs, largest hidden difference 8.6e-2
and 2 answer flips, both at p = 0.5 within 4e-4.
Rows of 8,192 and 16,384 tokens, whole against pieces of 1,024 to 4,096 tokens continuing snapshots, twelve read-outs each:
float32 within 6.7e-5, float16 within 1.5e-2 (|Δp| 6.6e-4); float16 against float32 at 8,192 tokens within 1.4e-3 relative L2.

### Speed (RTX 3060 12 GB, driver 591.86, CUDA 13.1, cuBLAS 13.2.1, float16; best of 7 runs; llama.cpp b11514 for comparison)

`sepal-cuda-engine-bench --model <dir> --workload runs/alert_workload.json [--gemm-rows 1024] [--json out.json]`.

| Prefill, tokens/s | 0.8B natural | 0.8B strict 1024 | llama.cpp 0.8B | 2B natural | 2B strict 1024 | llama.cpp 2B |
|---|---|---|---|---|---|---|
| 1 row x 64 | 8,610 | 1,352 | 5,204 | 4,020 | 506 | 3,281 |
| 1 row x 600 | 12,906 | 9,328 | 9,422 | 6,179 | 4,176 | 6,342 |
| 1 row x 1000 | 13,522 | 12,886 | 10,653 | 6,287 | 6,329 | |
| 1 row x 2048 | 13,613 | 13,166 | 11,056 | 6,859 | 6,483 | 6,888 |
| 1 row x 4800 | 13,432 | 12,511 | 10,018 | 6,666 | 6,167 | |
| 16 rows x 640 | 17,571 | 16,725 | | 7,923 | 7,317 | |

| Alerts per second (state + 8 questions) | 0.8B natural | 0.8B strict 1024 | llama-server 0.8B | 2B natural | 2B strict 1024 | llama-server 2B |
|---|---|---|---|---|---|---|
| 1 alert per call | 16.1 | 14.0 | 1.7 (2.9 best) | 7.6 | 6.5 | 1.1 (2.5 best) |
| 4 alerts per call | 18.4 | 15.9 | | 8.4 | 7.1 | |
| 16 alerts per call | 20.2 | 18.6 | | 9.2 | 8.2 | |
| state cached, 1 alert per call | 56.3 | 19.7 | | 25.1 | 7.8 | |
| state cached, 16 alerts per call | 77.1 | 65.5 | | 34.4 | 28.8 | |

Where a call's time goes (synchronised profiling mode, so each total is longer than the same call unprofiled), ms:

| Call (0.8B, natural) | multiplies | DeltaNet | attention | elementwise | host + launch | wall |
|---|---|---|---|---|---|---|
| 1 row x 600 | 28.4 | 15.5 | 1.2 | 2.6 | 2.1 | 49.8 |
| 16 rows x 640 | 407 | 128 | 15.6 | 37 | 2.7 | 591 |
| 16 alerts per call | 546 | 175 | 24.8 | 49.8 | 3.2 | 799 |

The multiplies run at 24 to 25 TFLOP/s, which is the RTX 3060's tensor rate with binary16 operands and float32 accumulation (28 SMs x
512 FLOPs per clock x about 1.9 GHz = 27.8 TFLOP/s; cuBLAS reaches 96% of it): on the 2B model they are 81% of a call, so raw tokens per
second there (83% of a 16 x 640 call) cannot move far from llama.cpp's **while the accumulators are float32**. That rate is not the
card's tensor peak: the same tensor cores run twice as fast with binary16 accumulators, which is what the next section measures and
the new `EngineOptions` use. What is left is the DeltaNet scan and the memory-bound elementwise kernels. (The table above and the throughput tables are
from before the DeltaNet kernel was rewritten; its effect on the profile is in the "DeltaNet kernels" section: 1 x 600
15.3 -> 7.5 ms, 16 x 640 126 -> 75 ms, 16 alerts 171 -> 118 ms of DeltaNet time per call.)

Peak device memory (0.8B and 2B, float16, the bench's default budget of 85% of the free memory): the engine reserves
9,557 MiB at creation (0.8B: weights 1,436, call buffers 962, 148 state slots of 19.3 MiB 2,851, key/value heap 4,307;
2B: weights 3,590, call buffers 1,460, 93 slots 1,792, heap 2,714); in use at the peak of the workload (16 alerts per
call): 2,869 MiB for the 0.8B and 5,521 MiB for the 2B. The process took 9.7 GiB of the 12 GiB card (the desktop holds
the rest).

### Multiply throughput: what the card can do, and what the engine does with it

`sepal-cuda-gemm-probe` (`tools/gemm_probe.cpp`, target `sepal-cuda-gemm-probe`) times one multiply at a time the way the engine
issues it (`Y^T = op_T(W [N,K]) X^T`, binary16 operands, 32 MB cuBLAS workspace) and measures its error against a float64 product.
Every figure is the shortest of several batches of back-to-back calls (the card is shared with the desktop and with other jobs,
so one slow batch must not count). `cublasGemmEx` is called with `CUBLAS_GEMM_DEFAULT`; cuBLASLt is asked for 24 algorithms and every
one is timed. Same card and software as above (cuBLAS 13.2.1, driver 591.86, SM clock about 1.9 GHz under load).

**1. What cuBLAS delivers against what the card can do.** The previous section's number is right and its reading is not. 28 SMs x
512 FLOPs per clock x 1.94 GHz = 27.8 TFLOP/s is the tensor-core rate with binary16 operands and **float32 accumulation**, and cuBLAS gets
96% of it on large shapes (26.6 to 26.8). It is not the card's tensor peak: GeForce cards run the same tensor cores at **twice** that
rate when the accumulators are binary16 (`CUBLAS_COMPUTE_16F`), and cuBLAS gets 44 to 49 TFLOP/s there (88% of 55.7).

| 8192 x 8192 x 4096, TFLOP/s | cuBLAS `GemmEx` | cuBLASLt, best of its candidates |
|---|---|---|
| binary16 operands, float32 accumulation, float32 output (the engine today) | 26.6 | 26.5 |
| the same with binary16 output | 26.8 | 26.6 |
| the same with the weights stored [K][N] instead of [N][K] | 26.6 | 26.4 |
| `CUBLAS_COMPUTE_32F_FAST_16F` | 26.5 | |
| binary16 accumulation, binary16 output (`CUBLAS_COMPUTE_16F`) | 43.8 | 47.8; 48.9 with the weights [K][N] |
| TF32 (float32 operands) | 13.3 | |
| float32 (`cublasSgemm`) | 7.7 | |

At the engine's own shapes, in TFLOP/s: the first column is today's call (float32 accumulation, float32 output, weights [N][K]); the
others are the variants discussed below (`sepal-cuda-gemm-probe shapes --model 0.8b|2b --lt 1`):

*Qwen3.5-0.8B shapes, TFLOP/s:*

| shape (N x K) | tokens | cuBLAS (engine today) | W stored [K][N] | batched 64 rows | cuBLASLt, best of 24 | binary16 accumulate | binary16 acc., Lt best, [K][N] | own kernel, float32 acc. | own kernel, binary16 acc., flush 128 |
|---|---|---|---|---|---|---|---|---|---|
| mlp_gate_up (7168 x 1024) | 64 | 13.2 | 12.3 |  | 15.6 | 17.4 | 18.0 | 12.6 | 12.7 |
| mlp_gate_up (7168 x 1024) | 256 | 22.8 | 23.1 | 15.6 | 22.9 | 34.0 | 40.1 | 21.3 | 33.4 |
| mlp_gate_up (7168 x 1024) | 1024 | 24.0 | 24.1 | 14.5 | 25.8 | 44.0 | 44.4 | 22.5 | 35.6 |
| mlp_gate_up (7168 x 1024) | 4096 | 24.4 | 25.1 | 14.2 | 26.1 | 45.2 | 45.5 | 22.9 | 36.4 |
| mlp_down (1024 x 3584) | 64 | 11.9 | 11.9 |  | 12.0 | 16.1 | 16.9 | 8.5 | 9.2 |
| mlp_down (1024 x 3584) | 256 | 19.6 | 19.2 | 12.8 | 19.6 | 30.9 | 33.7 | 12.5 | 14.1 |
| mlp_down (1024 x 3584) | 1024 | 21.4 | 21.3 | 20.1 | 23.5 | 39.9 | 41.8 | 16.7 | 26.7 |
| mlp_down (1024 x 3584) | 4096 | 23.8 | 23.8 | 23.7 | 25.3 | 43.2 | 45.0 | 21.1 | 35.7 |
| deltanet_in (8224 x 1024) | 64 | 10.3 | 12.4 |  | 12.5 | 13.3 | 15.5 | 9.8 | 10.9 |
| deltanet_in (8224 x 1024) | 256 | 17.9 | 18.0 | 14.9 | 18.0 | 32.3 | 37.4 | 19.9 | 30.0 |
| deltanet_in (8224 x 1024) | 1024 | 23.7 | 23.7 | 14.3 | 25.0 | 34.6 | 41.8 | 21.8 | 34.6 |
| deltanet_in (8224 x 1024) | 4096 | 24.2 | 25.7 | 14.3 | 25.8 | 35.2 | 42.8 | 22.4 | 36.0 |
| deltanet_out (1024 x 2048) | 64 | 11.6 | 12.0 |  | 12.0 | 14.5 | 15.1 | 8.1 | 8.7 |
| deltanet_out (1024 x 2048) | 256 | 17.6 | 17.6 | 12.8 | 17.6 | 27.8 | 28.5 | 12.9 | 15.2 |
| deltanet_out (1024 x 2048) | 1024 | 22.5 | 22.7 | 20.6 | 22.5 | 38.2 | 39.0 | 16.5 | 26.3 |
| deltanet_out (1024 x 2048) | 4096 | 23.0 | 23.1 | 24.2 | 24.5 | 41.0 | 43.6 | 21.0 | 35.0 |
| attn_in (5120 x 1024) | 64 | 13.4 | 15.2 |  | 13.4 | 16.8 | 17.3 | 11.7 | 12.4 |
| attn_in (5120 x 1024) | 256 | 23.1 | 23.6 | 19.8 | 23.2 | 32.1 | 38.6 | 19.8 | 29.1 |
| attn_in (5120 x 1024) | 1024 | 23.2 | 23.4 | 15.6 | 24.3 | 42.5 | 42.8 | 21.3 | 33.8 |
| attn_in (5120 x 1024) | 4096 | 23.9 | 26.1 | 14.4 | 26.0 | 44.9 | 44.9 | 22.6 | 35.9 |
| attn_out (1024 x 2048) | 64 | 11.6 | 11.9 |  | 12.0 | 14.5 | 15.1 | 8.1 | 8.7 |
| attn_out (1024 x 2048) | 256 | 17.6 | 17.7 | 12.8 | 17.5 | 27.8 | 28.6 | 12.8 | 15.2 |
| attn_out (1024 x 2048) | 1024 | 22.5 | 22.8 | 20.5 | 22.4 | 38.3 | 39.3 | 16.5 | 26.1 |
| attn_out (1024 x 2048) | 4096 | 23.1 | 23.1 | 24.2 | 24.5 | 41.1 | 43.6 | 21.0 | 35.2 |

*Qwen3.5-2B shapes, TFLOP/s:*

| shape (N x K) | tokens | cuBLAS (engine today) | W stored [K][N] | batched 64 rows | cuBLASLt, best of 24 | binary16 accumulate | binary16 acc., Lt best, [K][N] | own kernel, float32 acc. | own kernel, binary16 acc., flush 128 |
|---|---|---|---|---|---|---|---|---|---|
| mlp_gate_up (12288 x 2048) | 64 | 18.1 | 10.7 |  | 18.1 | 17.9 | 20.1 | 12.5 | 12.8 |
| mlp_gate_up (12288 x 2048) | 256 | 21.7 | 25.1 | 14.5 | 21.6 | 39.8 | 41.9 | 22.2 | 34.4 |
| mlp_gate_up (12288 x 2048) | 1024 | 25.5 | 25.5 | 14.3 | 26.1 | 43.0 | 46.8 | 22.8 | 37.8 |
| mlp_gate_up (12288 x 2048) | 4096 | 24.4 | 26.1 | 14.1 | 26.6 | 42.3 | 46.9 | 23.4 | 35.5 |
| mlp_down (2048 x 6144) | 64 | 12.4 | 12.7 |  | 14.6 | 17.7 | 19.8 | 6.7 | 7.9 |
| mlp_down (2048 x 6144) | 256 | 21.9 | 22.1 | 13.3 | 21.8 | 33.8 | 37.2 | 14.8 | 19.9 |
| mlp_down (2048 x 6144) | 1024 | 23.2 | 23.6 | 18.3 | 24.7 | 43.2 | 45.1 | 20.8 | 34.3 |
| mlp_down (2048 x 6144) | 4096 | 25.0 | 25.2 | 15.4 | 26.4 | 40.5 | 47.1 | 22.4 | 35.7 |
| deltanet_in (8224 x 2048) | 64 | 12.3 | 15.5 |  | 13.7 | 14.3 | 17.4 | 9.2 | 10.8 |
| deltanet_in (8224 x 2048) | 256 | 19.2 | 19.2 | 14.6 | 19.9 | 37.2 | 38.6 | 20.5 | 32.4 |
| deltanet_in (8224 x 2048) | 1024 | 25.0 | 24.9 | 14.7 | 25.7 | 41.5 | 43.1 | 22.3 | 36.8 |
| deltanet_in (8224 x 2048) | 4096 | 23.6 | 25.0 | 14.5 | 26.2 | 42.1 | 45.6 | 22.8 | 36.0 |
| deltanet_out (2048 x 2048) | 64 | 10.8 | 10.8 |  | 13.2 | 16.7 | 17.5 | 6.4 | 7.7 |
| deltanet_out (2048 x 2048) | 256 | 16.9 | 17.9 | 13.4 | 17.7 | 30.6 | 33.7 | 14.5 | 19.1 |
| deltanet_out (2048 x 2048) | 1024 | 20.2 | 20.3 | 18.0 | 24.0 | 39.3 | 42.8 | 20.4 | 32.0 |
| deltanet_out (2048 x 2048) | 4096 | 22.9 | 23.4 | 23.0 | 25.6 | 42.7 | 43.8 | 22.2 | 36.7 |
| attn_in (5120 x 2048) | 64 | 16.0 | 16.9 |  | 16.1 | 18.3 | 19.2 | 11.7 | 12.4 |
| attn_in (5120 x 2048) | 256 | 24.5 | 24.6 | 20.7 | 24.5 | 36.9 | 41.9 | 20.4 | 31.4 |
| attn_in (5120 x 2048) | 1024 | 24.4 | 24.5 | 16.5 | 25.2 | 42.5 | 44.9 | 21.7 | 36.3 |
| attn_in (5120 x 2048) | 4096 | 24.0 | 25.6 | 15.0 | 26.4 | 42.9 | 47.1 | 23.0 | 37.2 |
| attn_out (2048 x 2048) | 64 | 10.8 | 10.8 |  | 13.2 | 16.7 | 17.5 | 6.4 | 7.7 |
| attn_out (2048 x 2048) | 256 | 16.9 | 17.8 | 13.4 | 17.7 | 30.6 | 33.6 | 14.5 | 19.1 |
| attn_out (2048 x 2048) | 1024 | 20.2 | 20.3 | 18.0 | 23.9 | 39.2 | 42.4 | 20.4 | 32.2 |
| attn_out (2048 x 2048) | 4096 | 22.8 | 23.4 | 23.1 | 25.6 | 42.7 | 43.9 | 22.2 | 36.5 |


- **Shape and orientation.** From 1,024 tokens up cuBLAS runs the engine's wide shapes (N of 5,120 and more) at 23 to 25.5 TFLOP/s
  (86 to 96% of the 26.6 ceiling) and the narrow output projections (N of 1,024 or 2,048) at 20 to 23; at 256 tokens it is at 17
  to 25, at 64 tokens at 11 to 18. Storing the weights [K][N] instead of [N][K] changes nothing with float32 accumulation (within 3%;
  up to +8% in a few cells, and +20% at 64 tokens for two of the shapes). Cutting the rows into 64-row batched multiplies is
  slower (14 to 24). There is no tall-skinny penalty beyond the bandwidth limit below; `lda`/`ldb` are the natural ones and every
  operand pointer is 256-byte aligned with every leading dimension a multiple of 8, so alignment is not a factor.

**2. Other entry points.**

- *cuBLASLt with the heuristic search* finds an algorithm 3 to 9% faster than the one `cublasGemmEx` picks (which is Lt's first
  candidate, identical timings) from 1,024 tokens up, 18% at 64 tokens for the 7,168-wide gate_up, and 19% for the 2B model's 2,048 x
  2,048 output projections at 1,024 tokens. It is not used: the winner has to be found by timing, per shape and per process, and a
  different winner gives different bits.
- *Compute types.* binary16 output instead of float32 saves 0 to 10% of the multiply and costs the traffic of widening it again; the
  engine already does it where the consumer takes binary16 (the MLP's gate and up). `FAST_16F` is the same speed as `COMPUTE_32F` (the operands are
  already binary16). TF32 and float32 are 2 and 3.5 times slower. `CUBLAS_COMPUTE_16F` is 1.8 times faster and is the one lever
  that moves the multiplies, at an accuracy cost (below). int8 is measured in "int8 multiplies" at the end of this file.
- *Strided batched over row blocks* (`cublasGemmStridedBatchedEx`, 64 rows per batch entry): slower than one call (above).
- *Pedantic mode* changes no result (`CudaGemmOptions::pedantic`, README above).

Accuracy of each variant at one multiply (relative rms error against float64, X ~ N(0,1), W ~ N(0,1/K); the same figures for K = 1,024
and K = 3,584 unless two are given):

| Variant | Error |
|---|---|
| float32 accumulation, float32 output (today) | 1.7e-6 / 1.5e-6 |
| float32 accumulation, binary16 output | 2.1e-4 |
| `CUBLAS_COMPUTE_16F`, binary16 output | 1.7e-3 / 3.1e-3 (grows with the square root of K) |
| own kernel, binary16 accumulators flushed into float32 every 64 / 128 / 256 / 512 values of k | 4.4e-4 / 6.1e-4 / 8.4e-4 / 1.2e-3 |
| own kernel, binary16 accumulators, never flushed (the same arithmetic as `COMPUTE_16F`) | 1.7e-3 / 3.1e-3 |

**Engine accuracy** (`real_f16`, jeff-base, six prompts against the float32 golden; gates: |Δp| <= 0.03, no answer flip, today 7.4e-4;
every row has 0 flips). "all four" is `gate_up`, `down`, `mixer_in` and `mixer_out` together.

| Multiplies | max/mean \|Δp\| | max/mean \|Δhidden\| |
|---|---|---|
| cuBLAS, float32 accumulation (default) | 7.4e-4 / 1.6e-4 | 6.9e-3 / 9.7e-4 |
| own kernel, float32 accumulation | 2.9e-4 / 9.6e-5 | 1.2e-2 / 9.4e-4 |
| own kernel, binary16 accumulation, flush 64, all four | 4.4e-4 / 2.2e-4 | 1.6e-2 / 1.8e-3 |
| own kernel, binary16 accumulation, flush 128, all four | 1.2e-3 / 4.7e-4 | 2.2e-2 / 2.6e-3 |
| own kernel, binary16 accumulation, flush 256, all four | 2.0e-3 / 6.4e-4 | 3.1e-2 / 3.6e-3 |
| own kernel, binary16 accumulation, never flushed, all four | 2.6e-3 / 1.0e-3 | 1.9e-1 / 1.1e-2 |
| cuBLAS `COMPUTE_16F`, all four | 4.8e-3 / 1.4e-3 | 1.3e-1 / 8.1e-3 |
| cuBLAS `COMPUTE_16F`, `gate_up` only / `down` only / `mixer_in` only / `mixer_out` only | 1.3e-3 / 1.0e-3 / 1.9e-3 / 9.5e-4 | 4.9e-2 / 1.3e-1 / 4.0e-2 / 2.9e-2 |
| transformers' own float16, for scale | 5.0e-4 / 2.5e-4 | 2.0e-2 / 1.6e-3 |

On Qwen3.5-2B (51 tokens, mean \|Δhidden\| against the float32 engine): cuBLAS float32 1.8e-3, cuBLAS `COMPUTE_16F` 1.1e-2, own kernel float32 2.0e-3,
own kernel binary16 flush 64 / 128 / 256: 3.8e-3 / 4.4e-3 / 6.3e-3.

**3. Removing work, and why small calls are slow.**

- *Unused multiplies.* None left on the dense path: the last layer's queries, `o_proj` and MLP run for the read-out tokens only
  (`prune_last_layer`); the keys and values of every token are needed by later rows and snapshots; the read-out norm and head run
  on the read-out rows. Projections that share an input are already one multiply (q/k/v, the DeltaNet's q/k/v/z/b/a, gate and up).
  The multiplies of one layer form a chain (input projection, scan or attention, output projection, MLP), so there is nothing to
  run on a second stream, except the pruned last layer's two multiplies, which are small.
- *64-token calls.* The profile's 5.8 ms of multiplies for 64 tokens is inflated by the synchronisation around every phase; the 96
  multiplies unprofiled take 5.3 ms of the call's 7.5 ms. The floor is memory: the layers' weights are 0.99 GB in binary16 and the card
  reads at 360 GB/s at best, 2.8 ms; cuBLAS runs these skinny multiplies at 190 GB/s (the 7,168 x 1,024 gate/up weight, 14.7 MB, in 0.075 ms), 52% of the
  limit, and our own kernel, tried with six tile shapes, is at 0.070 to 0.074 ms with four of them: neither gets close to the bandwidth with so few
  rows. cuBLASLt's best algorithm is 13% faster at that size (not used, see above). Launch and setup overhead is not the cause: the
  probe's back-to-back calls include the launches, and the device time of the 96 multiplies matches the call.
- *Strict mode's padding.* `gemm_rows = 1024` pads a 64-token call to 1,024 rows, which is why it ran at 1,352 tokens/s against 8,610 natural.
  A fixed ladder of block sizes does not repair it with cuBLAS: a row's bits depend on the block size it was multiplied in, so the
  ladder would have to be chosen from the row's own length and the rows grouped by rung before every multiply, one cuBLAS call per
  group and layer, which is many small launches for a call with many rows. The engine's own kernel makes the question disappear
  (below): its result for a row does not depend on the batch at all, so strict mode pads nothing.
  One more thing found while checking this: `CUBLAS_COMPUTE_16F` kernels round differently from the float32-accumulate kernels
  that cuBLAS picks for the other shapes, so the last layer's keys and values must come from the same multiply whether or not the
  layer is pruned; in strict mode the engine now issues the unpruned last layer as the two multiplies the pruned one uses (queries
  and gates; keys and values).

**4. A hand-written tensor-core kernel is plausible, and it is in the engine.** `src/gemm_kernels.cu`: `mma.sync.m16n8k16`
tiles of 128x128 (and five smaller shapes for few rows or narrow outputs), `cp.async` pipeline, XOR-swizzled shared memory, `ldmatrix`; about 300
lines. First version, tuned only by tile shapes and stage counts (columns "own kernel" in the tables above): 21 to 23 TFLOP/s with float32
accumulators on the wide shapes from 1,024 tokens (cuBLAS 24 to 25; 16 to 21 on the narrow ones), 33 to 37 with binary16 accumulators flushed
every 128 (cuBLAS `COMPUTE_16F` 40 to 45); at 64 tokens it is 20 to 40% below cuBLAS on the narrow shapes. An ablation says why it stops there: a
variant that only loads the tiles, with no multiply, already takes longer than the multiplies alone would, so the kernel is bound by the
global-to-shared traffic of its 128x128 tile. A larger tile would move less data per FLOP but does not fit the register budget
next to the float32 flush accumulators, so closing the rest of the gap is real work, not a parameter.
What the kernel has that cuBLAS cannot offer: **a row's bits depend on that row, the weights, K and the flush interval only**
(tests: across six tile shapes, batch sizes 1 to 213 and every place in a batch, bit for bit), the **flush** that brings binary16
accumulation back to the accuracy of a binary16 output, and **no padding in strict mode**.

#### What is implemented (`EngineOptions`)

| Field | Default | What it does |
|---|---|---|
| `multiply_path` | `Cublas` | `Invariant` runs every float16-path multiply on `src/gemm_kernels.cu`: bits independent of the batch (natural mode is invariant too), strict mode (`gemm_rows` > 0) pads nothing. A multiply whose shape the kernel does not take (K or a row stride not a multiple of 8, odd N) falls back to cuBLAS and strict mode pads as before |
| `fp16_accumulate.{gate_up, down, mixer_in, mixer_out}` | all `false` | which multiplies accumulate in binary16: with `Cublas`, `CUBLAS_COMPUTE_16F` and a binary16 result widened by a small kernel; with `Invariant`, the own kernel with the flush |
| `fp16_accumulate.flush_k` | 128 | `Invariant` only: binary16 values summed before they are added to the float32 sum (a multiple of 32; 0 never flushes) |

`sepal-cuda-engine-bench` takes `--multiply-path cublas|invariant`, `--fp16-accumulate all|none|gate_up,down,...` and `--fp16-flush-k N`;
`sepal-cuda-engine-ab` runs several option sets in one process, round after round, and reports the best of each (a shared GPU
needs that). Nothing is on by default: binary16 accumulation changes the numbers, and `Invariant` changes the bits and is
slower than cuBLAS with float32 accumulators.

#### Engine speed, before and after (same session, `sepal-cuda-engine-ab`, best of 2 rounds x 5 reps, tokens/s)

*Qwen3.5-0.8B, natural mode (`gemm_rows` 0), prefill, best of 3 rotated rounds of 7 timed calls each:*

| tokens/s | cuBLAS (today) | cuBLAS binary16 acc., MLP only | cuBLAS binary16 acc., all four | own kernel, float32 acc. | own, binary16 acc., flush 64 | own, binary16 acc., flush 128 |
|---|---|---|---|---|---|---|
| 1 row x 64 | 8,605 | 9,140 (+6%) | 8,856 (+3%) | 7,196 (-16%) | 7,535 (-12%) | 7,586 (-12%) |
| 1 row x 128 | 11,389 | 12,847 (+13%) | 12,461 (+9%) | 8,548 (-25%) | 9,616 (-16%) | 9,702 (-15%) |
| 1 row x 300 | 11,723 | 14,464 (+23%) | 12,956 (+11%) | 10,851 (-7%) | 13,388 (+14%) | 13,603 (+16%) |
| 1 row x 600 | 13,099 | 16,547 (+26%) | 14,809 (+13%) | 12,250 (-6%) | 14,847 (+13%) | 15,120 (+15%) |
| 1 row x 1000 | 13,672 | 16,606 (+21%) | 15,711 (+15%) | 12,452 (-9%) | 15,758 (+15%) | 16,128 (+18%) |
| 1 row x 2048 | 13,895 | 17,211 (+24%) | 15,755 (+13%) | 13,124 (-6%) | 16,427 (+18%) | 16,805 (+21%) |
| 1 row x 4800 | 13,892 | 16,488 (+19%) | 15,393 (+11%) | 12,932 (-7%) | 16,006 (+15%) | 16,373 (+18%) |
| 16 rows x 640 | 17,895 | 22,692 (+27%) | 20,816 (+16%) | 16,819 (-6%) | 21,731 (+21%) | 22,162 (+24%) |

*Qwen3.5-2B, the same:*

| tokens/s | cuBLAS (today) | cuBLAS binary16 acc., MLP only | cuBLAS binary16 acc., all four | own kernel, float32 acc. | own, binary16 acc., flush 64 | own, binary16 acc., flush 128 |
|---|---|---|---|---|---|---|
| 1 row x 64 | 4,326 | 4,742 (+10%) | 4,503 (+4%) | 3,013 (-30%) | 3,268 (-24%) | 3,296 (-24%) |
| 1 row x 128 | 5,831 | 6,907 (+18%) | 6,366 (+9%) | 4,912 (-16%) | 5,492 (-6%) | 5,562 (-5%) |
| 1 row x 300 | 5,840 | 8,083 (+38%) | 6,856 (+17%) | 5,164 (-12%) | 6,943 (+19%) | 7,128 (+22%) |
| 1 row x 600 | 6,650 | 9,012 (+36%) | 8,148 (+23%) | 5,987 (-10%) | 8,150 (+23%) | 8,326 (+25%) |
| 1 row x 1000 | 6,667 | 9,539 (+43%) | 8,584 (+29%) | 6,271 (-6%) | 8,687 (+30%) | 8,923 (+34%) |
| 1 row x 2048 | 7,062 | 9,676 (+37%) | 8,756 (+24%) | 6,475 (-8%) | 8,977 (+27%) | 9,171 (+30%) |
| 1 row x 4800 | 6,792 | 9,369 (+38%) | 8,364 (+23%) | 6,471 (-5%) | 8,652 (+27%) | 8,803 (+30%) |
| 16 rows x 640 | 8,107 | 11,098 (+37%) | 9,927 (+22%) | 7,351 (-9%) | 9,636 (+19%) | 9,828 (+21%) |

llama.cpp b11514 on the same card, for scale (earlier table): 0.8B 5,204 / 9,422 / 11,056 / 10,018 at 64 / 600 / 2,048 / 4,800 tokens; 2B 3,281 / 6,342 / 6,888 at 64 / 600 / 2,048.

*Strict mode (`gemm_rows` 1024, bit-identical whatever shares the call): cuBLAS pads every multiply to a multiple of 1,024 rows; the own kernel pads nothing. Tokens per second, alerts per second for the last two rows; best of 2 rounds x 5 calls in one process.*

*Qwen3.5-0.8B:*

| strict `gemm_rows` 1024 | cuBLAS padded | own kernel, float32 acc. | own, binary16 acc., flush 64 |
|---|---|---|---|
| 1 row x 64 | 1,343 | 7,161 (+433%) | 7,541 (+461%) |
| 1 row x 128 | 2,540 | 8,535 (+236%) | 9,617 (+279%) |
| 1 row x 600 | 9,190 | 12,130 (+32%) | 14,621 (+59%) |
| 1 row x 2048 | 13,084 | 12,666 (-3%) | 16,114 (+23%) |
| 16 rows x 640 | 16,576 | 16,250 (-2%) | 21,080 (+27%) |
| 1 alert(s) per call [alerts/s] | 14.0 | 14.7 (+5%) | 19.2 (+37%) |
| 16 alert(s) per call [alerts/s] | 18.6 | 18.6 (+0%) | 23.8 (+28%) |

*Qwen3.5-2B:*

| strict `gemm_rows` 1024 | cuBLAS padded | own kernel, float32 acc. | own, binary16 acc., flush 64 |
|---|---|---|---|
| 1 row x 64 | 524 | 3,016 (+476%) | 3,263 (+523%) |
| 1 row x 128 | 1,029 | 4,915 (+377%) | 5,483 (+433%) |
| 1 row x 600 | 4,274 | 5,992 (+40%) | 8,142 (+91%) |
| 1 row x 2048 | 6,686 | 6,444 (-4%) | 8,959 (+34%) |
| 16 rows x 640 | 7,510 | 7,335 (-2%) | 9,686 (+29%) |
| 1 alert(s) per call [alerts/s] | 6.7 | 7.4 (+10%) | 10.4 (+55%) |
| 16 alert(s) per call [alerts/s] | 8.4 | 8.5 (+1%) | 11.0 (+31%) |


Reading: **binary16 accumulation is the one thing that moves the multiplies**, and it moves them 1.2 to 1.4 times. With cuBLAS
(`cublas16`) the engine is 24% (0.8B) and 37% (2B) faster at 2,048 tokens for a 6.5x larger |Δp| (4.8e-3, still 6x inside the gate); with
the own kernel and flush 64 it is 18% and 27% faster at the accuracy of today's engine (|Δp| 4.4e-4 against 7.4e-4), and bit-identical
whatever shares the call. On the 2B model that is 9.0k tokens/s at 2,048 tokens against llama.cpp's 6.9k: 1.3x; with `cublas16`, 9.7k: 1.4x.
The own kernel with float32 accumulators costs 5 to 8% at 2,048 tokens and 15 to 30% below 300 tokens, and buys batch invariance in natural
mode and strict mode without padding (the strict rows above).
In strict mode the own kernel removes the padding: a 64-token call goes from 1,343 to 7,161 tokens/s on the 0.8B (5.3x; natural cuBLAS
is 8,500) and from 524 to 3,016 on the 2B (5.8x), a single alert from 14.0 to 14.7 alerts/s (float32 accumulators) or 19.2 (binary16, flush 64)
on the 0.8B and from 6.7 to 7.4 or 10.4 on the 2B, while a row's bits still do not depend on the call. The 2B strict rows were measured
with the other jobs on the card at their noisiest; read them to within 10%.

## Snapshot density and tiers

What a snapshot costs, how to make it cheaper, and how to move it off the device. API: `include/sepal/cuda/cuda_engine.hpp`
(`EngineOptions::half_state`, `pool_snapshots`; `CudaEngine::to_host`, `to_device`, `save`, `load_host`, `load`, `state_saturations`;
`HostSnapshot`, `SnapshotFileError`). Files: `src/snapshot_file.{hpp,cpp}` (format, hash, reader). Tests: `test_cuda_engine_snapshots.cpp`
(`snap_half`, `snap_tiers`, `snap_fuzz`, `snap_default`, `snap_default_big`), `test_cuda_engine_real_snapshots.cpp` (`real_half_state`,
`real_tiers`), the kernel tests `cuda_deltanet_storage`.

### Bytes per snapshot

A 645-token alert state (alert 0 of the workload), measured by `real_tiers` from `EngineMemory::slot_bytes` and
`DeviceSnapshot::own_bytes()`. The 0.8B and the 2B have the same DeltaNet and attention geometry (18 DeltaNet layers of
16 x 128 x 128 state, 3 x 6,144 conv tail; 6 attention layers with 2 key/value heads of 256), so the snapshot is the same size.

| | float32 state (default) | binary16 state (`half_state`) |
|---|---|---|
| slot (recurrent state + conv tail, 18 layers) | 19.27 MiB | 9.63 MiB |
| keys and values of 645 tokens (binary16, 6 layers) | 7.56 MiB | 7.56 MiB |
| **one 645-token alert** | **26.82 MiB** | **17.19 MiB (-36%)** |
| alerts resident on a 12 GB card, 0.8B (weights 1,436 MiB, call buffers of 4,096 tokens 275 MiB, 600 MiB left to the driver) | 371 | **580 (+56%)** |
| the same, 2B (weights 3,590 MiB, call buffers 404 MiB) | 286 | **447 (+56%)** |

**Slot right-sizing: nothing to fix.** The pool bytes per slot equal the slot (19.27 MiB; the 256-byte alignment of each layer's
region is below 0.01%), the slot holds exactly the conv tail and the recurrent state, and the keys and values of a snapshot are
padded to 256 bytes per layer and kind (1.5 KiB per snapshot). The earlier "19.3 MiB slot" is the state itself. The levers
are the element size (this option) and what is kept at all, not slack.

### binary16 state (`EngineOptions::half_state`, off by default)

Both arrays of a slot (the recurrent state and the conv tail) are binary16; the DeltaNet kernels widen what they load and round to
nearest even what they store, all arithmetic stays float32 (`kernels_deltanet.hpp`, `DeltaNetStates::half_storage`).

**Determinism.** A row's state reaches a child row (in the same call, or from a stored snapshot) through the slot, and only
through the slot, so both paths read the same rounded values: with the option on, a branch continued from a stored snapshot and
the same branch run in the call of its state agree bit for bit in strict mode (`snap_half`: 44 continuations, float32 and
float16 paths, `gemm_rows` 8 and 24, plus a four-deep chain in one call against four calls). The row's own outputs (including a
row that has no stored parent) never see the rounding and are bit-identical to the float32-state engine's. The kernel test
(`cuda_deltanet_storage`) checks the stores are exactly the float32 results rounded to nearest even, and that a row run from
binary16 slots gives the bits that float32 slots holding the widened values give.

**Range.** A value beyond +-65504 is clamped (never infinity) and counted in a device counter (`CudaEngine::state_saturations()`).
Measured on the real model (float32 snapshots of alert states 0-11, case iv, an 8,000-token row and the end of a 30-link
chain): the largest |recurrent state| is 14.1 (per layer 0.85 to 14.1) and the largest |conv tail| 34.6, four orders of magnitude
inside the range; the binary16 runs counted 0 saturations in all the runs below. Of the nonzero state values 9.95% are below 2^-14
(binary16 subnormals, fewer mantissa bits) and 0.007% below 2^-25 (round to zero); their absolute error is at most 3e-8.

**Accuracy cost** (`real_half_state`, jeff-base, float16 engine in natural mode; every pair is the same engine with float32 state
against binary16 state, so the difference is the state rounding alone):

| Prompts | |Δp| median / p99 / max | |Δh|inf median / max | relative L2 median / max | flips |
|---|---|---|---|---|
| the 6 prompts of `real_f16`: cases i-iii (no stored parent) | bit-identical | | | 0 |
| case iv, 3 branches from a stored state | max 5.4e-4 | max 5.2e-3 | | 0 |
| 512 prompts of the alert workload from stored states | 2.4e-4 / 1.0e-3 / 1.3e-3 | 5.8e-3 / 1.3e-2 | 5.6e-4 / 9.5e-4 | 0 |
| 8,000-token row, 8 questions after it | 4.2e-4 / - / 6.3e-4 | 6.0e-3 / 1.1e-2 | 6.1e-4 / 7.7e-4 | 0 |
| chain of 1, 2, 5, 10, 20, 30 links of 64 tokens, 8 questions each | max 8.3e-4 (any depth) | max 1.5e-2 | 5.6e-4 to 7.1e-4 | 0 |

For scale, the float16 engine with float32 state against the transformers float32 golden on the same 512 prompts: |Δp| median 1.9e-4,
p99 7.5e-4, max 1.1e-3, relative L2 4.7e-4 median; with binary16 state: 2.0e-4, 8.8e-4, 1.7e-3, 4.8e-4. The binary16 state adds about as
much as the float16 weights and activations already do, it does not compound: the 30-link chain is 7.1e-4 against 6.2e-4 after one link (the
state forgets as fast as the rounding accumulates). No answer flips in 512 prompts, and none in the chain.

**Speed.** The DeltaNet kernels load and store the state half as wide; `sepal-cuda-engine-bench --half-state 1` against `0`, two alternating
rounds of 5 repetitions, the 64-alert workload on the 0.8B (float16, natural): the 16-alert state-cached call 80.6 and 79.3 alerts/s with float32 state
against 78.7 and 78.6 with binary16 state (-1.5%); 1 alert per call 57.4 and 56.3 against 55.8 and 57.4; the state passes themselves
(16 alerts per call, no cache) 21.9 and 21.8 against 21.6 and 21.5 (-1%). The difference is inside the run-to-run spread (about 2%): the state is not what
limits the scan, and the scattered 2-byte loads and the conversions cost about what the halved traffic saves.

### Host and file tiers

```cpp
HostSnapshot host = engine.to_host(snapshot, HostMemory::Pinned);   // copy device -> host (the whole chain's keys and values)
DeviceSnapshot back = engine.to_device(host);                       // host -> device: a new slot and heap blocks; BudgetExceeded if full
engine.save(snapshot, "alert-17.snap");                             // device (or HostSnapshot) -> file, atomic rename
DeviceSnapshot loaded = engine.load("alert-17.snap");               // file -> validated -> device, usable as a parent
HostSnapshot h = engine.load_host("alert-17.snap");                 // file -> host only
```

Every step is a byte copy of what the engine stores, so a promoted snapshot continues with the very bits the original does (tested
for float32 and binary16 state, both precisions, chains of 1 to 5 snapshots with gaps in the positions). A chain (a snapshot whose
keys and values continue its parent's) is saved with all of its ancestors' keys and values, block by block, with each block's start
position, and comes back as the same chain (the attention kernel sees the same segments, hence the same bits); a chain saved
and loaded is not shared with the other snapshots that had the same ancestors (the copy is self-contained). `save(load(save(x)))`
is the same file as `save(x)`: padding is not stored. Handles outlive things as before: a loaded snapshot keeps the pools alive
after its engine is gone, a `HostSnapshot` is independent of the engine and the device.

**File format** (`src/snapshot_file.hpp` has the byte table): header of 168 bytes (magic `SEPALSNP`, version 1, fingerprints of the
geometry, of the weights as they sit on the device (a hash kernel over the weight allocation, padding zeroed, a few ms, cached) and of
the compute options that change bits (gemm rows, multiply path, binary16 accumulation, gate/up precision, last-layer pruning), precision,
state format, layer counts, element counts, bytes per key/value token, token count, end position, block count, sizes, a hash of the header);
a block table of 24 bytes per block; the payload (state, then each block's keys and values per attention layer, unpadded); a trailer of the hash
of table and payload and an end marker. The hash is our own 64-bit multiply-rotate hash: it finds truncation and corruption, it does not
authenticate.

**Untrusted input.** The loader checks, in this order and before allocating anything from a header value: the file is a regular file and
at least header + trailer long; magic; the header hash; version and header size; reserved fields zero; every identity field against the
reading engine's own (a refusal names the field: model geometry, weights, precision, state format (`half_state`), layer counts, element
counts, key/value size, compute options); block count in [1, 65536]; state size equals the engine's own figure; the file size equals
exactly header + table + payload + trailer (all products and sums overflow-checked), so the only allocation, the payload, is bounded by the
size of the file on disk; per block: tokens > 0, positions below 2^32, key/value bytes equal to what the tokens need; the blocks add up
to the header's token and byte counts and the last block ends at the end position; then the payload is read in 4 MiB pieces and hashed,
and the payload hash and end marker verified. Only then does `to_device` touch the device. Nothing is allocated on the device for a
refused file; a `to_device` that finds the pools full throws `BudgetExceeded` and leaves them as they were.

`snap_fuzz` on a two-block file of 8,888 bytes: every one of 1,952 truncations (all lengths below 400 and above n-400, every 7th between)
and one trailing byte; every one of the 8,888 single-bit flips; 300 random multi-byte smashes; 293 headers and block tables with
extreme values (0, 1, 2^31-1, 2^32-1, 2^63, 2^64-1 and more) in every field with the hashes made valid again; 200 random files, half of them
with the right magic: all refused with `SnapshotFileError`, none crashed, none threw `bad_alloc`. (One entry is deliberately accepted:
the start position of the first block is free, a gap in positions is legal.)

**Times** (`real_tiers`: RTX 3060, a 645-token alert state, best of 7 after a warm-up, float16 engine; the file tier is read through the
operating system's cache: a cold read depends on the disk and is not measured here; the host figures include the allocation of the host
buffer, which for pinned memory is most of the difference to pageable at this size):

| ms | recompute the state | to_host pageable | to_host pinned | to_device | save (device -> file) | load_host (file -> host) | load (file -> device) |
|---|---|---|---|---|---|---|---|
| 0.8B, float32 state (26.8 MiB) | 44.9 | 5.5 | 4.7 | 3.1 | 15.7 | 18.9 | 18.8 |
| 0.8B, binary16 state (17.2 MiB) | 45.9 | 3.8 | 3.1 | 2.5 | 10.2 | 13.5 | 12.6 |
| 2B, float32 state | 102.7 | 5.5 | 4.6 | 3.1 | 15.2 | 18.2 | 19.0 |
| 2B, binary16 state | 102.2 | 3.7 | 3.1 | 2.5 | 10.9 | 13.5 | 13.3 |

So the host tier pays at once (a demote plus a promote is 6 to 8 ms against 45 ms of recompute on the 0.8B and 100 ms on the 2B),
and a file in the operating system's cache pays too (a load of 13 to 19 ms is 2.4 to 3.5 times faster than the 0.8B recompute and 5 to
8 times than the 2B's; saving costs 10 to 16 ms once). The warm load is bound by the host (reading, hashing and copying 17 to 27 MiB, about
1.3 to 1.5 GB/s), so a cold file pays as long as the disk delivers the file in less than recompute minus upload: on the 0.8B the disk must
read about 0.4 GB/s with binary16 state (0.7 GB/s with float32), on the 2B about 0.2 GB/s (0.3). A SATA SSD is marginal for the 0.8B, an NVMe
drive is far inside. The state is bytes, not compute, so the cost does not grow with the model: the 2B's tier times equal the 0.8B's,
while its recompute is 2.3 times longer. No automatic policy is implemented: the calls are explicit.

### The default reservation

Before, an engine created without a budget took 85% of the free device memory (about 9.5 GB on the 12 GB card, mostly empty pools: the
16-alert workload peaks at 2.9 GB in total). Now `memory_budget_bytes = 0` means: the weights, the call buffers, and pools for
`pool_snapshots` (default 64) snapshots of `pool_snapshot_tokens` (default 1,024) tokens, within 85% of the free memory.
On the 0.8B with the bench's settings (call buffers for 16,384 tokens) that is 4,399 MiB with float32 state (weights 1,436, call buffers 962,
64 slots of 19.3 MiB 1,233, key/value heap 768) and 3,783 MiB with binary16 state, against about 9.5 GB; the call buffers are now the largest
item after the weights and `max_call_tokens` is their lever (275 MiB at 4,096 tokens). The 16-alert cached bench runs unchanged in it.
`pool_snapshots = 0` restores the old behaviour; a given budget or `state_slots` works as before. `BudgetExceeded` is thrown by `run`
exactly as before when a call needs more than is free; what changes is the size of "free": a workload that keeps more than 64 snapshots
at the default has to ask (`pool_snapshots`, `state_slots` or a budget), with the message that says so.

Not done: pools that grow on demand. The slot pool is addressed by the DeltaNet kernels as one base pointer plus a slot index, so growing
needs either an indirection table in the kernels or CUDA virtual memory management (`cuMemAddressReserve`/`cuMemMap`: reserve the
address range of the budget, commit physical pages as slots and heap blocks are used). The latter keeps the kernels unchanged, but a
committed page is not given back when the last snapshot goes (or needs `cuMemUnmap` bookkeeping), which would change the
"device memory after equals before" guarantee the memory tests rely on, and it needs the driver API (`cuda.lib`) in the build. It is
the right next step if the default is still too large for a small card.

### Checked by breaking it on purpose

Three deliberate bugs, one build: the binary16 store rounding toward zero instead of to nearest (the kernel test `half_slots_are_the_float_result_rounded_at_the_store`
fails 6 checks; the engine-level hit = miss tests cannot see it, both sides round the same way); the payload hash not enforced on load
(`snap_fuzz.every_flipped_byte_is_refused` fails 8,961 of 9,174 checks); the values of attention layer 0 not uploaded by `to_device` (4 of the 6
`snap_tiers` cases fail, 35 checks). The tests also caught a real bug of the first version of this change: zeroing the weight allocation with
`cudaMemset`, which runs on the legacy stream and does not wait for, nor is waited for by, the engine's non-blocking stream, raced with the weight
upload and zeroed part of the vocabulary and wide-geometry weights (`attack_geometry`: 2 failures); it is now `cudaMemsetAsync` on the engine's stream.

### What the two reference projects do

*HyperQwen* (Apache-2.0, vLLM-based, 64-layer 27B model with 48 DeltaNet layers, one RTX 3090): the "16-bit recurrent state" is only
the stock vLLM flag `--mamba-ssm-cache-dtype float16` (`docs/optimizations.md` item 3, `batch/start_qwen.sh`): fp16, not bf16, "10 mantissa
bits"; no scaling, clamping or range analysis, and the widening/rounding happens inside vLLM's kernels, not in their repository.
Reported: with `--max-num-seqs 64`, 37 requests actually ran with the float32 state (about 150 MB per request) and all 64 with fp16;
perplexity 8.045 against 8.044 on 33k tokens; GSM8K 95.5%; 516 to 707 tokens/s at 64 concurrent (also with an int8 lm_head step). The measured
quantity is perplexity over a decode, which never hands a rounded state to a prefill; our measurement is the hidden-state and probability
difference of a continuation. Their KVarN backend (a port of Huawei CSL's, `kvarn/`, `docs/long-context.md`) stores keys at 4 bits and values
at 2 bits per element (Hadamard rotation, 128-token tiles with iterative variance normalisation, fp16 scales, the first 128 tokens kept
fp16), about 840 bytes per token and layer against 2,048 for fp8; perplexity 8.223 against 8.236, needle-in-a-haystack correct to 240k tokens,
100k-context decoding 1.22 times faster, but 2.13 times slower with 3-token speculation at 112k. It is a lossy cache by their own account.
We did not port it: our K/V is binary16 (2,048 bytes per token and layer for keys and values together) and is 44% of an alert snapshot
(7.6 of 17.2 MiB) once the state is halved; int8 with per-token-per-head scales (not implemented) would save 3.8 MiB (22% of the snapshot),
a 4-bit key / 2-bit value scheme about 6 MiB (35%) before scales, at an accuracy cost nobody has measured on this model.

*Strata* (MIT): `docs/DETAILS.md` has the measurements. Session files (`conversation_file.hpp`): little-endian, a 64-byte header (magic
`STRSESS\x01`, version, model and config fingerprints, payload length, a header hash), a payload with every array count-prefixed, and a
trailer hash and end marker; read fail-closed in a fixed order (regular file with one hard link and no symlink, size bounds, magic, header hash,
version, fingerprints before the payload is parsed, payload length against file size, RAM preflight, every count against the remaining bytes
and the runtime's limits before allocation, payload hash), all before any device write; written to a hidden temporary file and renamed. The
hash is "xxHash64-style", it detects accidents and does not authenticate. The file holds the recurrent state and the key/value of the whole
conversation (about 1.2 GB for 63k tokens), only the deepest checkpoint. Tiers: up to 6 RAM checkpoints of about 118 MB, taken at each new
turn and every 16K prompt tokens, LRU with the root pinned; optional host "parking"; an explicit disk save/restore without automatic spilling.
Measured on an RTX 4070 Ti 12 GB, NVMe: a 63,025-token conversation, file 1.20 GB, save 0.65 s, restore 1.05 s in a new process, next
32-token turn 1.14 s against 25.5 s cold. Their `MESSAGE_BOUNDARY_CACHE` (an extra checkpoint at the previous turn boundary, opt-in, for
prefixes of 8,192 tokens or more) is reported at 17.2 s to 11.5 s for the first edit of 100 records, in a comparison that also changes the
chunk size and so is not an isolated gain; `KV_PREFETCH` (uploading the next layer's key/value during prefill) measured 5.3% at 32K and 4.1% at 64K,
from timings the document itself calls historical; `VRAM_ELASTIC` is about resizing the expert cache and has nothing to do with conversation
state. Our format shares their order of checks and the fail-closed stance; it differs in verifying the identity from the engine's own
geometry and a hash of the device weights (theirs hashes the first and last MiB of the model files), and in refusing a file whose size does not
equal what the header derives, which bounds every allocation.

## int8 multiplies: what the card does, what they cost in accuracy, and whether to have them

Everything in this section was measured on the RTX 3060 12 GB of the sections above (driver 591.86, CUDA 13.1, cuBLAS 13.2.1), on jeff-base (the
Qwen3.5-0.8B fine-tune) and Qwen3.5-2B-Base, float16 path. Code: `src/gemm_s8_kernels.{hpp,cu}`, `EngineOptions::int8_multiply`; off by default.

### Conclusion

* **The int8 tensor cores are there.** `mma.sync` int8 (`m16n8k32`) in our own kernel runs at 64 to 72 TOP/s at 4,096 tokens (49 to 70 at 1,024), 1.5 to 1.8 times
  cuBLAS's binary16-accumulate rate (39 to 45) and 2.6 to 3.1 times its float32-accumulate rate (21 to 25), which is what the engine runs today; cuBLAS's own int8
  reaches 83 to 92 TOP/s on big shapes (the card's peak is 111). At the engine's shapes our kernel is 1.4 to 3.0 times today's multiply from 256 tokens up and 1.4 to 2.1 times at 64
  tokens. In the engine, with all four multiply groups on int8 (W8A8): **1.62x (0.8B) and 1.94x (2B) the default engine at 2,048 tokens** (27.2k and 15.3k tokens/s),
  1.29x and 1.38x the cuBLAS binary16-accumulate engine, and **2.2x llama.cpp's 6.9k on the 2B**.
* **But per-row int8 activations do not meet the design's accuracy gate on this model.** The gate (`q8` row of docs/RUNNER.md: largest |dp| <= 0.05 and at most 1% answer flips
  against the float32 golden) is failed by every W8A8 variant, every single group alone included: 8 to 10 flips of 512 (1.6 to 2.0%) for gate/up, down or the input projections alone,
  26 for the output projections alone, 44 (8.6%, largest |dp| 0.15) for all four. Emulating finer activation scales (below) shows that activation outliers, not the weights, cause it;
  the fix would be another scheme (scales per group along k, smoothing), not a tuning of this one.
* **int8 weights with binary16 activations (W8A16, the design's `gpu-q8w`) meet the gate** (all four groups: 4 flips of 512 = 0.78%, three of them within 0.012 of p = 0.5, largest |dp| 0.039;
  without the output projections 5 flips, 0.034), **cut the weight memory by a third** (0.8B 1,436 -> 964 MiB, 2B 3,590 -> 2,284 MiB; the embedding stays binary16) and are **not faster**: -4% (0.8B)
  and -5% (2B) at 2,048 tokens against the default engine, -8% and -18% at 64 tokens. Worth having for memory, not for speed.
* **Small calls** do not gain what halved weight traffic suggests: W8A8 is +22% at 64 tokens on both models (0.8B 9.3k -> 11.4k tokens/s), because those multiplies are latency-bound at that size, not
  bandwidth-bound (our kernel reaches 200 GB/s of int8 weights on the 7,168 x 1,024 gate/up at 64 tokens, cuBLAS int8 216).
* **Recommendation.** Keep `Int8Multiply`, off by default. Use W8A16 where weight memory decides (a model that would not otherwise fit). Do not use W8A8 as it is for the alert read-out: it buys
  1.29 to 1.38x over cuBLAS binary16 accumulation and 1.6 to 1.9x over the default, at a largest |dp| of 0.15 and 44 flips of 512 against 2.0e-3 and 1 for the own kernel with binary16 accumulation
  (flush 64). The owner's 2x llama.cpp on the 2B (13.8k tokens/s at 2,048 tokens) is reached by W8A8 only (15.3k); with an option that passes the gate the 2B is at 1.45x (own kernel, binary16 accumulation,
  flush 64, 10.0k) to 1.6x (cuBLAS binary16 accumulation, 11.1k).

### What HyperQwen and Strata do (read from `ref/`, nothing built or run; HyperQwen `3acb93f`, Apache-2.0; Strata `fb58e0d`, MIT)

No code was taken from either repository, so `NOTICE` is unchanged. This is what their documents, patches and sources say.

| | HyperQwen (vLLM patches for a 27B hybrid; RTX 3090 at 250 W) | Strata (GGUF-format prefill engine; RTX 5070) |
|---|---|---|
| int8 GEMM source | **none in the repo**: stock vLLM's W4A8-INT8 Marlin kernel (`VLLM_MARLIN_INPUT_DTYPE=int8`); the patches fix a sign bug in its group scales, select layers by regex (`VLLM_MARLIN_INT8_INCLUDE_RE`), relax asserts, and optionally route through an external tuned Marlin build that is not in the repo | dense layers: **no quantised GEMM**; `dequant_f16` of the whole weight into a scratch buffer, then cuBLAS `GemmEx` (fp16, fp32 accumulate) in `src/prefill/gemm.cu`. The one int8 tensor-core kernel (`mma.m16n8k32.s32.s8.s8.s32`, `src/prefill/moe_fused.cu`) is for the 2-bit routed MoE experts |
| weights | int4 g128 symmetric (AutoRound, offline), fp16 scales; `in_proj_a / b`, MTP head, vision tower stay bf16; lm_head and embeddings int8 or int4 g128 | GGUF block formats (Q2_0 ... Q8_0, K-quants), fp16 scale per block; nothing per output channel |
| activations | dynamic, per token (row), absmax/127 (`marlin_quant_input`, in vLLM; where it is fused is not in the repo). Batch default: MLP only; single-user default: MLP + attention + DeltaNet projections | MoE path: quantised once per token, absmax/127, one float32 scale per 32 values, fused SwiGLU epilogue that re-quantises the intermediate |
| what they measured | prefill at 1k tokens 1,437 -> 1,638 tok/s with int8 MLP (+14%), 1,845 with all linears (+28%); +19% at 51k; batch-64 decode +24% (MLP) to +45% (all). Perplexity over 33k tokens: gate/up +0.9%, whole MLP +2.2%, all linears +3.7%; GSM8K 95.5 -> 95.0 (MLP). Per-GEMM error with int8 activations 0.9 to 1.05% against 0.26% for W4A16 on random weights. Sensitivity: one docstring line (GDN `in_proj` in early layers and `down_proj` in the last layers are worst); no per-layer numbers in the repo | prompt rate +16 to 22% from the fused int8 Q2_0 MoE path (4K: 1,294 -> 1,570 tok/s); output KL 0.009 against 0.012 for the older kernels; no dense-GEMM TFLOP/s |
| int8-QK attention | `prefill-attn-int8.patch`, Triton, SageAttention-style (K mean-smoothing, int8 K per 64-key tile, int8 Q per row, int32 `tl.dot`, bf16 P.V), head dim 256, 16 full-attention layers; 1.27 to 1.35x over FA2 at 4k to 51k context, +2.7% / +5.3% end to end at 16k / 51k | -- |

What we now do that they do: per-row dynamic int8 activations with float32 scales; int8 tensor-core multiplies with exact int32 sums; int8 selectable per group of multiplies (their regex, our
`Int8Multiply` flags); the DeltaNet's `in_proj_a / b` kept at 16 bits (HyperQwen does too). What we do differently: weights are int8 per output channel (they are int4 per 128 along k), quantised when the engine is created
from the checkpoint's values (they load a pre-quantised checkpoint), and the kernel is batch-invariant by construction. What we do not do: int4 weights, int8 attention or K/V, fusing the activation quantiser into the producing
kernel, a tuned tile table, an fp16 recurrent state, smoothing. Their ranking of sensitivity is not what we see here: with per-row activations the output projections are the worst group by flips (26 of 512 alone), then down (10), and
gate/up and the input projections the best (8 and 9), and nothing is cheap: HyperQwen's +1.5% perplexity step from the MLP to all linears is, here, the step from 14 to 44 flips.

### The implementation

* `launch_gemm_s8`: the structure of `gemm_kernels.cu` (cp.async pipeline, XOR-swizzled shared memory, ldmatrix) with a 64-byte k tile (two `m16n8k32` steps; a 64-byte row of int8 is what a 32-half row was, so the swizzle and the ldmatrix addressing are
  the binary16 kernel's), int32 accumulators and `(float(sum) * sx[row]) * sw[col]` in the epilogue; six tile shapes. The integer sum is exact, so a row's bits depend on its own int8 row, the weights and K only.
* `launch_gemm_w8a16`: the binary16 kernel with the weights as int8 in shared memory. They are stored in an order (`S8WeightLayout::Fp16Pairs`: positions 4q ... 4q+3 of every 16 hold k = 2q, 2q+1, 2q+8, 2q+9) that lets one `ldmatrix` give a thread
  exactly the four bytes of its two B registers, which are widened to binary16 exactly (`prmt` with the exponent of 1024, subtract 1152) and multiplied with float32 accumulators; the channel scale is applied at the end.
* `launch_quantize_rows_s8`: scale = max |x| / 127, `round-half-even(x * (127 / max))` clamped to +-127, a zero row gets scale 0. 0.04 ms for 2,048 x 2,048 (4% of its multiply), 0.16 ms for 4,096 x 3,584, the down projection's input, which is
  36% of that multiply (0.44 ms): it is a separate pass, not fused with SwiGLU or the norm, and fusing it would recover most of that.
* `launch_quantize_weights_s8`: the same rule per output channel. The engine does it at creation: a stacked matrix (gate|up, q|k|v, ...) is uploaded as binary16 into a scratch buffer and quantised on the device; the binary16 copy of a selected group is not kept.
* `EngineOptions::int8_multiply = {gate_up, down, mixer_in, mixer_out, int8_activations}`. A selected group overrides `fp16_accumulate` for that group. The DeltaNet's `in_proj_b / in_proj_a` rows are stored and multiplied in binary16 (own kernel,
  float32 accumulation) whatever is selected. The float32 path, a K that is not a multiple of 16 and widths the kernels do not take are refused at creation. Strict mode pads nothing for int8 groups; with every group on int8 (or the own kernel for
  the rest) natural mode is batch-invariant as well. Tools: `sepal-cuda-engine-bench --int8 w8a8:all|w8a16:noout|...`, `sepal-cuda-engine-ab --options w8a8-all,w8a16-mlp,...` (groups: `gate_up`, `down`, `mixer_in`, `mixer_out`, `mlp`, `noout`, `all`).

### 1. What the card does (`sepal-cuda-int8-probe`, minimum of three batches, 32 MB workspace, errors against float64 in the tool's output)

Large shapes (the own binary16-accumulate figure of this run is low: the card was shared with other jobs; the shape tables below have it):

| T x N x K | cuBLAS f32 acc | cuBLAS f16 acc | own W8A8 | own W8A16 | cuBLAS int8 (int32 out) | cuBLASLt int8, best of its 3 candidates |
|---|---|---|---|---|---|---|
| 8192 x 8192 x 4096 | 25.4 | 42.1 | 64.7 | 25.3 | 82.8 | 85.6 |
| 4096 x 4096 x 4096 | 25.0 | 39.9 | 66.8 | 25.0 | 88.6 | 85.2 |
| 8192 x 4096 x 8192 | 25.6 | 38.4 | 56.1 | 24.8 | 92.2 | 91.9 |
| 2048 x 8192 x 2048 | 23.9 | 40.6 | 66.8 | 24.7 | 77.9 | 78.1 |

The int8 tensor rate is twice the binary16-accumulate one (2 x 55.7 = 111 TOP/s: 28 SMs at 1.94 GHz): cuBLAS reaches 75 to 83% of it, our kernel 51 to 65% (the binary16 kernel's ablation found the same load-bound limit; no tuning of the int8 kernel beyond the
tile shapes was done). cuBLASLt's heuristic offers nothing above `GemmEx`'s choice (its first candidate). cuBLAS int8 writes int32: the float32 dequantisation is another pass, not in its figure. At 64 tokens it is 1% to 2.1 times above our kernel (the 2B's deep
down projection, K = 6,144: 34.5 against 16.5); at 4,096 tokens on the deep shapes (K 3,584 to 6,144) 16 to 34% above (76 to 89 against 65 to 68). W8A16 is at the float32-accumulate rate (25 TFLOP/s): the arithmetic is the binary16 multiply's, with the conversion in front.
The quantisation error of one multiply (Gaussian operands, K 1,024 to 6,144): 1.1 to 1.3e-2 relative rms for W8A8, 0.8 to 0.9e-2 for W8A16 (weights alone), against 1.7e-6 for today's multiply.

Engine shapes, TFLOP/s (TOP/s for int8), the 2B probe run shared with other jobs (a few cuBLAS cells are low, e.g. deltanet_in 1,024 tokens):

*Qwen3.5-0.8B shapes, TFLOP/s (TOP/s for int8):*

| shape (N x K) | tokens | cuBLAS, f32 acc (today) | cuBLAS, f16 acc | own, f16 acc, flush 128 | own W8A8 | own W8A8 + row quantisation | own W8A16 | cuBLAS int8, int32 out |
|---|---|---|---|---|---|---|---|---|
| mlp_gate_up (7168 x 1024) | 64 | 12.9 | 17.3 | 12.6 | 26.0 | 24.1 | 15.6 | 27.6 |
| mlp_gate_up (7168 x 1024) | 256 | 22.1 | 33.7 | 33.4 | 53.3 | 49.4 | 21.6 | 59.2 |
| mlp_gate_up (7168 x 1024) | 1024 | 21.9 | 43.4 | 34.8 | 62.7 | 60.1 | 24.0 | 64.8 |
| mlp_gate_up (7168 x 1024) | 4096 | 23.2 | 45.4 | 35.0 | 66.2 | 64.1 | 23.3 | 64.7 |
| mlp_down (1024 x 3584) | 64 | 11.9 | 15.9 | 9.2 | 17.9 | 14.9 | 8.6 | 22.1 |
| mlp_down (1024 x 3584) | 256 | 19.5 | 31.1 | 15.7 | 27.2 | 23.5 | 13.8 | 41.2 |
| mlp_down (1024 x 3584) | 1024 | 23.8 | 40.0 | 27.1 | 52.0 | 41.0 | 17.7 | 61.1 |
| mlp_down (1024 x 3584) | 4096 | 24.0 | 43.2 | 36.4 | 67.9 | 50.5 | 22.7 | 83.7 |
| deltanet_in (8224 x 1024) | 64 | 10.3 | 13.4 | 10.9 | 21.4 | 20.1 | 12.8 | 27.8 |
| deltanet_in (8224 x 1024) | 256 | 18.0 | 32.4 | 30.2 | 51.5 | 48.3 | 21.0 | 50.0 |
| deltanet_in (8224 x 1024) | 1024 | 22.8 | 28.6 | 35.0 | 62.7 | 60.4 | 22.7 | 61.0 |
| deltanet_in (8224 x 1024) | 4096 | 24.4 | 34.9 | 34.6 | 65.2 | 63.4 | 22.4 | 63.8 |
| deltanet_out (1024 x 2048) | 64 | 11.5 | 14.4 | 8.8 | 16.0 | 12.9 | 7.4 | 16.1 |
| deltanet_out (1024 x 2048) | 256 | 17.7 | 28.1 | 15.3 | 25.1 | 21.5 | 13.3 | 38.4 |
| deltanet_out (1024 x 2048) | 1024 | 22.5 | 38.4 | 26.4 | 49.2 | 39.3 | 17.4 | 58.0 |
| deltanet_out (1024 x 2048) | 4096 | 21.6 | 41.5 | 32.5 | 65.2 | 47.6 | 19.2 | 75.7 |
| attn_in (5120 x 1024) | 64 | 13.3 | 16.8 | 12.3 | 22.0 | 20.1 | 14.4 | 26.1 |
| attn_in (5120 x 1024) | 256 | 23.2 | 31.8 | 29.4 | 44.3 | 40.7 | 20.8 | 42.9 |
| attn_in (5120 x 1024) | 1024 | 22.4 | 42.7 | 34.4 | 60.5 | 57.1 | 22.7 | 61.5 |
| attn_in (5120 x 1024) | 4096 | 21.8 | 45.0 | 34.8 | 65.4 | 61.8 | 22.5 | 64.5 |
| attn_out (1024 x 2048) | 64 | 11.6 | 14.6 | 8.8 | 16.1 | 12.8 | 8.2 | 19.7 |
| attn_out (1024 x 2048) | 256 | 17.7 | 27.9 | 15.3 | 25.5 | 21.8 | 13.3 | 38.1 |
| attn_out (1024 x 2048) | 1024 | 22.0 | 38.2 | 26.4 | 49.0 | 39.3 | 17.4 | 57.5 |
| attn_out (1024 x 2048) | 4096 | 21.5 | 41.6 | 35.9 | 64.1 | 50.8 | 20.8 | 74.9 |

*Qwen3.5-2B shapes, TFLOP/s (TOP/s for int8):*

| shape (N x K) | tokens | cuBLAS, f32 acc (today) | cuBLAS, f16 acc | own, f16 acc, flush 128 | own W8A8 | own W8A8 + row quantisation | own W8A16 | cuBLAS int8, int32 out |
|---|---|---|---|---|---|---|---|---|
| mlp_gate_up (12288 x 2048) | 64 | 17.8 | 17.9 | 12.8 | 24.7 | 23.9 | 11.4 | 34.3 |
| mlp_gate_up (12288 x 2048) | 256 | 18.1 | 39.5 | 34.3 | 62.2 | 59.8 | 22.5 | 66.5 |
| mlp_gate_up (12288 x 2048) | 1024 | 21.9 | 36.8 | 37.1 | 70.4 | 68.5 | 24.0 | 76.8 |
| mlp_gate_up (12288 x 2048) | 4096 | 23.9 | 40.5 | 33.6 | 72.0 | 69.9 | 25.1 | 75.5 |
| mlp_down (2048 x 6144) | 64 | 12.4 | 17.7 | 8.0 | 16.5 | 15.4 | 12.8 | 34.5 |
| mlp_down (2048 x 6144) | 256 | 22.0 | 34.3 | 20.1 | 39.7 | 36.0 | 18.0 | 54.8 |
| mlp_down (2048 x 6144) | 1024 | 23.3 | 43.3 | 34.5 | 66.2 | 49.2 | 19.3 | 64.2 |
| mlp_down (2048 x 6144) | 4096 | 19.0 | 38.9 | 33.8 | 65.9 | 52.0 | 23.1 | 88.6 |
| deltanet_in (8224 x 2048) | 64 | 11.9 | 14.1 | 11.0 | 21.4 | 20.5 | 10.1 | 28.8 |
| deltanet_in (8224 x 2048) | 256 | 11.4 | 23.2 | 18.9 | 35.4 | 32.7 | 11.1 | 35.6 |
| deltanet_in (8224 x 2048) | 1024 | 11.7 | 26.7 | 27.2 | 50.3 | 48.5 | 16.1 | 55.9 |
| deltanet_in (8224 x 2048) | 4096 | 19.3 | 40.5 | 35.6 | 67.4 | 64.9 | 23.3 | 78.3 |
| deltanet_out (2048 x 2048) | 64 | 10.8 | 16.7 | 7.7 | 14.5 | 13.2 | 12.5 | 27.8 |
| deltanet_out (2048 x 2048) | 256 | 16.9 | 30.9 | 19.4 | 35.6 | 31.6 | 17.4 | 47.7 |
| deltanet_out (2048 x 2048) | 1024 | 20.4 | 39.7 | 33.1 | 60.8 | 53.2 | 21.1 | 63.6 |
| deltanet_out (2048 x 2048) | 4096 | 22.6 | 43.1 | 37.3 | 68.9 | 59.8 | 23.9 | 75.5 |
| attn_in (5120 x 2048) | 64 | 16.1 | 18.6 | 12.7 | 22.9 | 21.4 | 12.4 | 31.9 |
| attn_in (5120 x 2048) | 256 | 24.8 | 37.2 | 31.7 | 56.1 | 51.2 | 15.7 | 42.7 |
| attn_in (5120 x 2048) | 1024 | 15.6 | 29.2 | 32.6 | 59.3 | 55.7 | 19.9 | 68.0 |
| attn_in (5120 x 2048) | 4096 | 20.4 | 39.5 | 38.1 | 71.2 | 66.8 | 24.4 | 80.1 |
| attn_out (2048 x 2048) | 64 | 10.8 | 16.7 | 7.7 | 14.6 | 13.2 | 12.4 | 27.8 |
| attn_out (2048 x 2048) | 256 | 17.0 | 30.8 | 19.4 | 35.5 | 31.5 | 17.4 | 47.6 |
| attn_out (2048 x 2048) | 1024 | 20.4 | 39.7 | 33.1 | 61.3 | 53.4 | 21.8 | 63.6 |
| attn_out (2048 x 2048) | 4096 | 23.2 | 43.1 | 37.3 | 68.6 | 60.0 | 24.1 | 75.9 |


### 2. Accuracy on the real model (`sepal_cuda_engine_tests real_int8`, `real_int8_2b`)

Reference: transformers float32 hidden states (the `real_f16` goldens for the six prompts, `runs/alert-golden` for the 512 prompts of the 64-alert workload). "dp" is the probability of the first option for the 512 prompts and the
largest over all options for the six; "flips" are answer flips (argmax); "near 0.5" counts the flips whose golden probability is within 0.01 of one half. Gate: largest |dp| <= 0.05 and flips <= 1% (5 of 512). 16 alerts per call; the
six-prompt rows are the prompts of `real_f16`. `base` is the default engine (cuBLAS, float32 accumulation), `acc16` the own kernel with binary16 accumulation flushed every 64, for scale. All other rows have the cuBLAS path for the multiplies not on int8.

The 512 prompts:

| variant | dp median | dp p99 | dp max | rel L2 of hidden, median | flips | near 0.5 | gate |
|---|---|---|---|---|---|---|---|
| base | 2.0e-4 | 1.0e-3 | 1.9e-3 | 4.7e-4 | 0 | 0 | pass |
| acc16 (flush 64, all four) | 3.3e-4 | 1.5e-3 | 2.0e-3 | 9.4e-4 | 1 | 1 | pass |
| **W8A16, all four** | 6.4e-3 | 2.8e-2 | 3.9e-2 | 2.6e-2 | 4 | 3 | **pass** |
| **W8A16, all but the output projections** | 7.1e-3 | 2.3e-2 | 3.4e-2 | 2.2e-2 | 5 | 3 | **pass** |
| W8A8, gate/up only | 9.6e-3 | 5.4e-2 | 7.3e-2 | 3.3e-2 | 8 | 5 | fail |
| W8A8, down only | 1.5e-2 | 5.5e-2 | 6.2e-2 | 7.7e-2 | 10 | 4 | fail |
| W8A8, input projections only | 7.3e-3 | 4.5e-2 | 6.4e-2 | 2.7e-2 | 9 | 7 | fail |
| W8A8, output projections only | 1.9e-2 | 9.1e-2 | 1.1e-1 | 5.4e-2 | 26 | 7 | fail |
| W8A8, gate/up + input projections | 1.1e-2 | 5.1e-2 | 1.0e-1 | 4.3e-2 | 13 | 5 | fail |
| W8A8, MLP (gate/up + down) | 1.7e-2 | 7.4e-2 | 1.3e-1 | 8.3e-2 | 14 | 3 | fail |
| W8A8, all but the output projections | 1.8e-2 | 8.7e-2 | 1.3e-1 | 8.7e-2 | 15 | 5 | fail |
| W8A8, all four | 3.1e-2 | 1.3e-1 | 1.5e-1 | 1.1e-1 | 44 | 7 | fail |

The six prompts of `real_f16` (no flips in any variant; largest |dp| over all options): base 5.7e-4, acc16 6.7e-4, W8A16 all four 9.1e-3, W8A16 without output projections 1.2e-2, W8A8 gate/up 1.6e-2, down 2.8e-2, input projections 1.3e-2,
output projections 3.3e-2, gate/up + input projections 2.1e-2, MLP 2.2e-2, all but output projections 4.1e-2, **all four 5.1e-2 (fails the gate on six prompts)**; relative L2 of the hidden state 4.2e-4 (base), 2.2e-2 (W8A16 all four), 8.0e-2 (W8A8 all four).

How close to 0.5 the flips are (distance of the float32 golden's probability from 0.5): W8A16 all four 0.001, 0.005, 0.008, 0.012; W8A16 without output projections 0.001, 0.005, 0.008, 0.011, 0.012; acc16 0.001. W8A8 flips reach much further:
gate/up up to 0.051, down 0.024, input projections 0.030, output projections 0.044, MLP 0.097, all four 0.116 (44 flips, 7 of them within 0.01 of 0.5 and 22 further than 0.03). The model is a base fine-tune whose probabilities
sit near 0.5 on some prompts; the W8A16 flips are all in that zone, the W8A8 ones are not.

Row independence on the real model, 512 prompts, one alert per call / 16 per call / states cached and the questions in a later call: bit-identical results in all three ways for the two profiles with every group on int8 (W8A8 all four, W8A16 all four).
W8A16 without the output projections has cuBLAS multiplies left in, and the three ways differ a little (largest |dp| against the golden 0.0331, 0.0335, 0.0325).

**Where the W8A8 error comes from.** Activations were quantised and dequantised in binary16 in groups along k and fed to the W8A16 multiply (a diagnostic, not kept in the code), so the weights' error is the W8A16 row's and only the activation scale changes:

| activation scale | flips of 512 | largest dp | six prompts, largest dp |
|---|---|---|---|
| none (W8A16 all four) | 4 | 3.9e-2 | 9.1e-3 |
| per 32 values of k | 5 | 4.0e-2 | 2.0e-2 |
| per 128 | 6 | 6.4e-2 | 2.9e-2 |
| per 512 | 23 | 1.1e-1 | 2.8e-2 |
| per row (what W8A8 does; the real W8A8 kernel: 44 and 1.5e-1) | 42 | 2.6e-1 | 3.7e-2 |
| per row, scale cut to 85% of the maximum (clipping) | 30 | 2.0e-1 | 9.7e-2 |
| per row, cut to 70% | 35 | 4.4e-1 | 1.2e-1 |

Clipping the outliers makes it worse, shrinking the group makes it better: a few large channels carry information and set the row's scale. Scales per 128 get to the weights' error only in flips; per 32 pass the gate but have no tensor-core epilogue
that is cheap (the scales would have to be applied inside the k loop). Smoothing the large channels into the weights (folded into the RMSNorm weight for the input projections and gate/up, into the up projection's rows for down) is the usual remedy and needs activation statistics; it
was not tried.

Qwen3.5-2B-Base has no golden for the alert workload; the float32 engine is the reference, 64 prompts (8 alerts), mean relative L2 of the final hidden state: base 5.6e-4, acc16 1.0e-3, **W8A16 all four 3.4e-2**, W8A8 gate/up 5.1e-2, input projections 4.3e-2, output projections 5.6e-2,
down 6.8e-2, MLP 8.5e-2, all but output projections 9.7e-2, all four 1.1e-1 (the same order as on the 0.8B).

### 3. Speed (`sepal-cuda-engine-ab`, same process, rotating through the option sets, best of 2 or 3 rounds of 4 or 5 timed calls; tokens/s, alerts/s for the last rows)

*Qwen3.5-0.8B, natural mode:*

| | cuBLAS (today) | cuBLAS f16 acc, all four | own, f16 acc, flush 64 | W8A16 all four | W8A8 MLP | W8A8 all but output | W8A8 all four |
|---|---|---|---|---|---|---|---|
| 1 row x 64 | 9,341 | 9,974 (+7%) | 6,702 (-28%) | 8,585 (-8%) | 10,220 (+9%) | 11,383 (+22%) | 11,442 (+22%) |
| 1 row x 600 | 15,757 | 19,795 (+26%) | 17,905 (+14%) | 15,164 (-4%) | 18,826 (+19%) | 22,859 (+45%) | 24,813 (+57%) |
| 1 row x 2048 | 16,757 | 21,025 (+25%) | 20,218 (+21%) | 16,074 (-4%) | 20,726 (+24%) | 25,598 (+53%) | 27,188 (+62%) |
| 1 row x 4800 | 16,741 | 18,528 (+11%) | 19,377 (+16%) | 15,944 (-5%) | 20,274 (+21%) | 24,559 (+47%) | 25,944 (+55%) |
| 16 rows x 640 | 19,355 | 23,639 (+22%) | 23,154 (+20%) | 18,914 (-2%) | 24,563 (+27%) | 30,752 (+59%) | 32,882 (+70%) |
| 1 alert per call [alerts/s] | 18.9 | 23.9 (+26%) | 23.3 (+23%) | 18.3 (-3%) | 23.2 (+23%) | 28.9 (+53%) | 30.3 (+60%) |
| 16 alerts per call | 21.7 | 26.6 (+23%) | 26.3 (+21%) | 20.9 (-4%) | 27.1 (+25%) | 34.1 (+57%) | 36.5 (+68%) |
| weights, MiB | 1,436 | 1,436 | 1,436 | 964 | not printed | not printed | 964 |

*Qwen3.5-2B, natural mode (a separate session, 3 rounds; llama.cpp b11514 for scale: 3,281 / 6,342 / 6,888 tokens/s at 64 / 600 / 2,048):*

| | cuBLAS (today) | cuBLAS f16 acc, all four | own, f16 acc, flush 64 | W8A16 all four | W8A8 all four |
|---|---|---|---|---|---|
| 1 row x 64 | 4,525 | 4,967 (+10%) | 3,374 (-25%) | 3,716 (-18%) | 5,532 (+22%) |
| 1 row x 600 | 7,357 | 10,177 (+38%) | 9,013 (+23%) | 6,790 (-8%) | 13,921 (+89%) |
| 1 row x 2048 | 7,872 | 11,066 (+41%) | 10,015 (+27%) | 7,461 (-5%) | 15,263 (+94%) |
| 16 rows x 640 | 8,459 | 11,700 (+38%) | 10,103 (+19%) | 8,042 (-5%) | 16,595 (+96%) |
| 1 alert per call [alerts/s] | 8.5 | 12.3 (+45%) | 11.3 (+33%) | 8.4 (-1%) | 17.2 (+102%) |
| 16 alerts per call | 9.6 | 13.2 (+38%) | 11.3 (+18%) | 9.2 (-4%) | 18.7 (+95%) |
| weights, MiB | 3,590 | 3,590 | 3,590 | 2,285 | 2,285 |


Alerts per second with the states cached (`sepal-cuda-engine-bench --sections alerts,cached`, best of 5):

| | 0.8B cuBLAS | cuBLAS f16 acc | own f16 acc | W8A16 | W8A8 | 2B cuBLAS | cuBLAS f16 acc | own f16 acc | W8A16 | W8A8 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 alert per call | 58.1 | 74.6 | 65.0 | 52.9 | 84.8 | 26.0 | 39.6 | 31.8 | 25.4 | 50.0 |
| 16 alerts per call | 79.8 | 101.3 | 98.1 | 76.3 | 130.0 | 35.3 | 49.5 | 45.5 | 34.5 | 70.4 |
| same, not cached: 1 / 16 alerts per call | 19.2 / 21.9 | 24.3 / 28.1 | 24.0 / 27.0 | 18.4 / 21.2 | 31.3 / 37.2 | 8.4 / 9.5 | 12.4 / 13.2 | 11.4 / 11.3 | 8.4 / 9.2 | 17.3 / 18.7 |

Strict mode (`gemm_rows` 1024; the int8 groups pad nothing, the cuBLAS option pads every multiply to 1,024 rows), 0.8B:

| | cuBLAS padded | own, f16 acc, flush 64 | W8A8 all four | W8A16 all four |
|---|---|---|---|---|
| 1 row x 64 | 1,399 | 8,121 | 11,423 | 8,576 |
| 1 row x 600 | 10,851 | 18,431 | 24,811 | 15,459 |
| 1 row x 2048 | 16,497 | 20,913 | 27,657 | 16,412 |
| 16 rows x 640 | 18,401 | 24,008 | 33,608 | 18,988 |
| 1 alert per call | 16.4 | 23.7 | 31.2 | 18.5 |
| 16 alerts per call | 20.1 | 26.7 | 37.3 | 21.5 |

Why not more: the multiplies are 60 to 75% of a 0.8B call. With the multiply 2.6 times faster than today's the call can be at most 1.6 to 1.9 times faster, and the row quantisation (one pass per int8 multiply: 5 to 16% of it at the gate/up shapes, 16% at 64 tokens, and 36% at the down projection's), the unchanged DeltaNet, attention and elementwise kernels, and the 64-token calls' latency take the rest. The speed of the W8A8 kernel itself (51 to 65% of the card's int8 peak, cuBLAS int8 75 to 83%) is the remaining headroom in the multiplies.

### 4. Memory

Weights (MiB, `EngineMemory::weights_bytes`): 0.8B 1,436 (binary16) -> 964 (all four groups on int8); 2B 3,590 -> 2,284. 2B by group alone: gate/up 3,015, down 3,302, input projections 3,243, output projections 3,494, MLP 2,727, all but output 2,380.
What remains is the binary16 embedding (485 MiB at 0.8B, 970 MiB at 2B; it is also the tied head) and the float32 scales. The call buffers grow by the int8 activation buffer and the scales for W8A8 (0.8B: 962 -> 1,018 MiB at 16,384 tokens). By arithmetic and
not run (no 9B checkpoint here): a 9B at hidden size 4,096 has about 13.8 GB of binary16 layer weights and a 2.0 GB embedding; the layers on int8 are about 7 GB, so with the embedding 9 GB, leaving about 3 GB of a 12 GiB card (before the display's share) for call buffers and snapshots,
where binary16 (15.9 GB) cannot be loaded at all. The embedding could be stored int8 per row as well (the design's `cpu-q8` profile does); not done.

### 5. Tests

* `sepal_cuda_kernel_s8_tests` (CTest `cuda_s8_*`): `s8_quantize` (host oracle, bit for bit: bytes and scales, both weight layouts, K from 16 to 6,144; zero rows, a 65,504 outlier, subnormals, +-127), `s8_gemm` (the exact integer product of the
  device's own int8 operands scaled in float32, bit for bit, five shapes; W8A16 against float64 of the dequantised weights, 1e-5; quantisation error against the unquantised product; an outlier-channel case), `s8_invariance` (six tile shapes, batches of 1 to 213, every place: bit-identical, both kernels),
  `s8_edge` (eight tiny and ragged shapes with every tile shape, zero rows, 14 refusals: K, odd N, misaligned or null pointers, strides, unknown configuration). 465 checks.
* `sepal_cuda_engine_tests engine_int8` (CTest `cuda_engine_engine_int8`): defaults and refusals (float32 path, H = 24), closeness to the float32 engine for seven option sets and memory below binary16's, bit-identity alone / among other rows / from a stored snapshot for four option sets in natural and strict (`gemm_rows` 16, 48)
  modes with cuBLAS or the own kernel for the rest, strict pruned against unpruned last layer and repeated runs. 1,188 checks.
* `real_int8`, `real_int8_2b`: the tables above (they assert finite results, the default engine's gate, and bit-identity of the all-int8 profiles across batching ways). The planted-bug check: dropping the sign flip in the W8A16 weight widening (`v ^= 0x80808080` -> `0x00808080`) fails `s8_gemm` in every shape
  (`err < 1e-5` and `quantisation < 1.2e-2`); reverted.

The whole CTest suite with all the real-model environment variables set: 67 of 67 pass (23 minutes, `real_cpu` 520 s of it), none skipped; the 60 tests that existed before still pass, strict-mode bit-identity suites included.

### Not done

* Tuning the W8A8 kernel (51 to 65% of the int8 peak; cuBLAS int8 is 10 to 35% faster on the deep shapes), fusing the row quantisation into SwiGLU and the norms, and split-k for small calls (the 64-token multiplies stay latency-bound).
* A cheaper way to W8A8 that passes the gate: smoothing (needs calibration), scales per group along k in the epilogue (per 128 still fails on |dp|, per 32 is expensive), outlier channels kept in binary16. None tried beyond the emulation above.
* int8 embedding, int8 attention or K/V, int4 weights, an int8 route for the CPU reference.
* No 9B run; llama.cpp was not re-measured (its figures are from the earlier section).
