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
| `CMakeLists.txt`, `CMakePresets.json`, `cublas64_13.def` | The build; the cuBLAS import-library definition |
| `include/sepal/cuda/cuda_gemm.hpp`, `src/cuda_gemm.cpp` | The provider |
| `src/convert_kernels.hpp`, `src/convert_kernels.cu` | The element-type conversion kernels |
| `tests/test_cuda_gemm.cpp` | Suites `contract`, `parity_small`, `parity_real`, `rows` |
| `tools/sepal_cuda_bench.cpp` | `sepal-cuda-bench` |

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
  creation out of `memory_budget_bytes` (weights, call buffers for `max_call_tokens`, a slot pool, a key/value heap);
  `run` allocates nothing on the device, throws `BudgetExceeded` before doing any work when a call does not fit, and the
  engine stays usable.
- **Multiply batching.** `gemm_rows = 0` is natural (one multiply over all tokens). `gemm_rows = T` is strict: every
  multiply is cut into multiplies of exactly T rows, the last zero-padded, so a row's bits do not depend on the rest of
  the call. All of the engine's own kernels are batch-invariant.
- **Residual ladder**: `EngineRow::capture_layers` returns every layer's output for that row.
- **Tuning that is on by default**: the last layer's queries, attention, `o_proj` and MLP are computed for the tokens
  that are read out only (`prune_last_layer`; its keys and values are still stored for every token; bit-identical to
  the full pass in strict mode); on the float16 path the MLP's gate and up multiply writes binary16 (`fp16_gate_up`).
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
| memory | 3,000 calls creating and releasing snapshots: device memory before and after identical, pools back to zero use, heap fully merged; budget overrun fails with `BudgetExceeded`, nothing is leaked, the next call gives the same bits; snapshots outlive their parents' handles and the engine |
| refusals | bad token ids, empty rows, readouts out of range, cyclic or dangling parents, empty handle, snapshot of another engine or geometry, too many rows/tokens: `std::invalid_argument`, nothing left allocated |

The real-model suites (`real_f32`, `real_f16`, `real_strict`, `real_natural`, `real_2b`) print SKIPPED unless
`SEPAL_REAL_MODEL` (checkpoint directory), `SEPAL_REAL_GOLDEN` (`tools/export_real_golden.py` output) and, for `real_2b`,
`SEPAL_REAL_MODEL_2B` are set.

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

The multiplies run at 24 to 25 TFLOP/s, which is the RTX 3060's float16-with-float32-accumulation tensor peak (28 SMs x
512 FLOPs per clock x about 1.78 GHz = 25.5 TFLOP/s): on the 2B model they are 81% of a call, so raw tokens per second
there (83% of a 16 x 640 call) cannot move far from llama.cpp's. What is left is the DeltaNet scan and the
memory-bound elementwise kernels. (The table above and the throughput tables are from before the DeltaNet kernel was
rewritten; its effect on the profile is in the "DeltaNet kernels" section: 1 x 600 15.3 -> 7.5 ms, 16 x 640 126 -> 75 ms,
16 alerts 171 -> 118 ms of DeltaNet time per call.)

Peak device memory (0.8B and 2B, float16, the bench's default budget of 85% of the free memory): the engine reserves
9,557 MiB at creation (0.8B: weights 1,436, call buffers 962, 148 state slots of 19.3 MiB 2,851, key/value heap 4,307;
2B: weights 3,590, call buffers 1,460, 93 slots 1,792, heap 2,714); in use at the peak of the workload (16 alerts per
call): 2,869 MiB for the 0.8B and 5,521 MiB for the 2B. The process took 9.7 GiB of the 12 GiB card (the desktop holds
the rest).
