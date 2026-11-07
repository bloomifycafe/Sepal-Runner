# Changelog

## Unreleased

### 2026-11-07: Pascal GPUs (GTX 10xx, Tesla P40): float32 SIMT profiles gpu-bf16 and gpu-w8g, Lean GpuTrack

Measured on the RTX 3060 and on a GTX 1080 Ti (vast.ai, CUDA 12.6); raw files in the Sepal repository, `reports/runner-gpu-pascal/`.
Details: `runner/cuda/README.md`, "Pascal".

- GTX 1080 Ti: gpu-bf16 900 tok/s at 1k tokens (p50 1.07 s), 578 at 16k, 672 on TOOL, 580 on LONG-LOG (24.6k-token records fit at a 9.2 GB
  trunk budget); gpu-w8g at an 8 GB card's budget 967 / 578 / 743. Same logits as the 3060 to 9.5e-7 (|dp| <= 1.7e-7, no flips). The SIMT
  multiply at 86-90% of cuBLAS 12 SGEMM (bf16), 101-109% (w8g). The float32 attention is 45% of a 16k request there: next to speed up.
- The DeltaNet scan cannot launch on Pascal at head dim 128 (66 KB of shared memory, 48 KB per block): gpu-bf16 / gpu-w8g run the float32
  chunked kernels on every card, the scan suites are skipped on such a card.

- New profiles `gpu-bf16` (bfloat16 weights) and `gpu-w8g` (MLP weights int8 with a float32 scale per 128 input values, the rest
  bfloat16): float32 arithmetic on the ordinary CUDA cores, no tensor cores, compute capability 6.1 and up. `sepal serve` picks them on
  cards below 8.0 (bf16 when it fits, else w8g); `gpu-f16` / `gpu-q8w` are refused there. Embedding kept in host memory (1.27 GB).
- Kernels: a float32 SIMT multiply reading bf16 or int8-group weights (fused SwiGLU and residual add; 99-101% of cuBLAS SGEMM on the
  3060; bit-exact against a host emulation of its schedule), float32 flash attention (`AttentionPath::TiledF32`), a float32 chunked (WY)
  Gated DeltaNet (the Pascal profiles' DeltaNet; 0.6-0.8x the scan's speed on the 3060, 3% of a call). All batch-invariant.
- Speed (3060, one request per call): gpu-bf16 917 tok/s at 1k tokens, 752 at 16k, 818 on TOOL; gpu-w8g the same within 2%; gpu-f16
  4,180 / 3,300 / 3,670; cpu-bf16 (6 cores) 167 / 141 / 151.
- Parity: gpu-bf16 within the gpu-f16 gate on SEC-dev n300 (0.023 / p99 0.0087 / 0 flips against the PyTorch rows) and TOOL n60 (0.0096),
  and on LONG-LOG n12 0.032 / 0.011 / 0 against cpu-f32 (the fp32 forward), where cpu-bf16 is 0.064 / 1 flip. gpu-w8g: no flip in 712
  questions, accuracy unchanged, max |dp| against gpu-bf16 0.070 (SEC-dev), 0.053 (LONG-LOG), 0.009 (TOOL): over the q8 row's 0.05 on
  single near-tie questions. int8 in the DeltaNet and attention projections moved LONG-LOG to 0.10-0.14 (1 flip), hence MLP-only.
- Tried and not kept: TiledF32 attention with 32 query rows per block and keys staged 64 dimensions at a time (8k request 1.74 s of
  attention) and with one row per thread over 64 dimensions (1.83 s); kept: two rows per thread over 32 dimensions (1.35 s). gpu-w8g with the design's int8 column (3.93 GB, LONG-LOG 0.143 / 1 flip against fp32), with
  everything int8 (SEC-dev 0.040 / p99 0.0125 against the rows), with attention also bfloat16 (SEC-dev 0.051).
- Build: tensor-core code compiled only into sm_80+ images (`SEPAL_TENSOR_CORES`); preset `pascal` (Linux, CUDA 12, sm_61); the whole CUDA
  project builds with CUDA 12.6 for sm_61;86 (CI job `cuda-compile`). `gpu_parity.cpp` builds under MSVC /WX again.
- Lean: `proofs/GpuTrack` (core Lean, no sorry): the chunked gated delta rule equals the token recurrence over any commutative ring; the
  SIMT multiply's schedule gives the in-order fma fold for any tiling and thread assignment (with widening, dequantisation and the packed
  SwiGLU); group-int8 factoring. CI job `lean-proofs`.


### 2026-11-06: exact track on Broadwell (GEMM tile writes, flash attention, Lean proofs)

Measured on an E5-2673 v4 box (vast.ai, cores 0-11 of a shared host; every change run alternately against the one before;
logs: Sepal `reports/runner-broadwell/exact-track/`).

- GEMM, same bits. The tuned AVX2 GEMM kept all finished tiles of a task (up to 126 x 64 floats, rows ldy apart, mostly 4K-aliased
  lines that miss and need ownership) and wrote them after its last k slice, serialised after the compute. It now writes each tile
  right after its last micro-kernel call, with the tile's y lines prefetched for writing (`SEPAL_GEMM_SCHEDULE=fused-prefetch`,
  default; `tile` the old schedule). Same kc, same fp32 FMA chain per output: byte-for-byte test over the six 4B products and odd
  edges, plus a check against a scalar `std::fma` chain. Layer mix +2.1 to +2.4% at 12 cores, +2.7 to +3.2% at 10;
  dn_in / attn_qkv +6 to +7%, dn_out / attn_o +4%, down +2%, gate_up unchanged (its SwiGLU output is half the size; it is now the
  slowest product per FLOP). 16k log at 12 cores: matrix products 157 -> 146 s, +1.4% end to end; short alerts +2%.
- Last-layer row pruning (keep only the final row in the last layer) is not done: the Clef head reads every final row (memory
  projection and cross-attention over all positions, span means), so it would change the logits.
- Attention: new `flash` kernel (default; `SEPAL_ATTENTION=tiled` keeps the old one). S, P and the output accumulator stay in the
  micro-kernel tile layout (no copies), the micro kernel accumulates P V itself, K/V panels are the outer loop, the last key tile
  is cut to the visible keys, and the softmax loops have no 64-bit division (one per 8 scores cost more than the exp: the softmax
  was 25% of the attention time, now 12%). Same online softmax over the same absolute 128-key tiles, so rows stay independent of
  batch, chunking and threads (new test). sepal-cpu-attn-bench, 16k prefill: +15 to +18% (12 and 10 cores), 1k +17 to +26%;
  in the 4B 16k log attention 27.8 -> 23.9 s, +1.7 to +1.8% end to end. Parity against the tiled kernel (the current
  runner output, itself bit-identical to the 0.4 box rows): LONG-LOG 20 records max |dp| 3.6e-5, p99 2.6e-5, 0/262 flips; TOOL 40
  records (the 40 shortest of TOOL_n60) 1.9e-6, 0/67; the 6 LONG-LOG records with fp32 references 1.9e-5 vs tiled, and vs fp32
  max |dp| 0.0317, 0/89 flips (unchanged).
- Lean 4 proofs in `proofs/ExactTrack` (core Lean, no Mathlib, no sorry, `lake build` clean): GEMM bit-identity under any task /
  tile / thread partition with fixed kc slices; last-row pruning of a causal layer stack (no runtime user, see above); online
  softmax equals softmax over an abstract exp field. fp32 error bound of online vs two-pass softmax in `proofs/LEAN-NOTES.md`
  section 10.4 (prose, not machine-checked).
- Tools: `sepal-cpu-gemm-bench --schedule a,b --measure-peak 1`; new `sepal-cpu-attn-bench`.
- Tried and not kept (same box, numbers in the logs): `fused` without the prefetch, layer mix +0.4% at 12 cores (fused-prefetch
  +3.5%); other task blockings with fused-prefetch (cols 128/192/256, rows 84/60/48, row-major order), no change beyond the host's
  +-5% noise at one core, so the cache-derived blocking stays; the first flash version (tile layout, in-kernel accumulation, panel-outer
  loops, but with a 64-bit division per 8 softmax scores) was only +1 to +3% at 16k, profiling with rdtsc showed the softmax at 25% of
  the attention time, removing the division gave the +15 to +18%. Profile of the 16k log after both changes (12 cores): matrix products
  146 s of 178 s, attention 24 s, DeltaNet 5.3 s, other 3.0 s; nothing outside the GEMM is large enough for a bigger exact win.

### 2026-11-05: LONG-LOG parity traced to the reference's bf16 error

The LONG-LOG failure of the `cpu-bf16` gate (every CPU path, `cpu-f32` too) comes from the reference rows, not from the runner. An fp32 PyTorch
forward (no autocast, same adapter, head, tokens and 24,576 window) on 6 of the 20 sample records, 89 questions: `cpu-f32` matches it to 2.9e-5
in p (gate 1e-4), `cpu-bf16` to 0.032 (p99 0.016), the bf16 rows only to 0.048 (p99 0.026); the one flip in the sample is the bf16 row's. Nothing in
the runner changed and the gates are unchanged. New `tools/runner_fp32_reference.py` (export a sample, score it with the fp32 oracle on a torch box,
compare runner logits with both references) and `docs/PARITY-LONGLOG.md`.

### 2026-11-05: tuned AVX2 kernel set (x86-64-v3), thread placement, Broadwell

Measured on a Xeon E5-2698 v4 (Broadwell-EP, AVX2 + FMA, no AVX-512) in a vast.ai container with a 9.6-CPU quota; results and
logs: Sepal `runs/box-results/broadwell/`. 4B `cpu-bf16` on 10 cores: short alerts 71.1 -> 86.0 tok/s (12.9 -> 10.7 s each),
16k-token logs 62.2 -> 73.6 tok/s, TOOL requests 66.2 -> 79.1; 1 to 9 cores +17 to +24%; the container's default thread count
43.7 -> 90.5. Parity of `avx2-tuned` against the PyTorch rows: SEC-dev 300 max |dp| 0.023, p99 0.009, 0 flips (passes);
TOOL 60 max 0.0096, p99 0.0003, 0 flips (passes);
LONG-LOG 20 max 0.049, p99 0.030, 1 flip in 262, the same as the original AVX2 code to 1.4e-5 (this panel was already over
the gate with the original code and with AVX-512, 0.052 / 0.049 on 12 records).

- New CPU kernel set `avx2-tuned`, chosen automatically on AVX2 CPUs without AVX-512 after a cpuid check of the whole x86-64-v3
  level (`SEPAL_CPU_ISA`, open option `cpu_isa`, `sepal serve --cpu-isa`; a set the CPU lacks is refused with the missing features
  named). 6x16 micro-kernels with all twelve accumulators in registers, the k loop unrolled four times in GCC/Clang assembly
  (intrinsics under MSVC), bf16 weights widened inside the kernel (`vpunpck*wd` on port 5) instead of a separate pass, blocking
  derived from the cache sizes read with cpuid. Bit-identical to the generic `avx2` set (test
  `cpu_gemm.tuned_avx2_is_bit_identical_to_generic_avx2`).
- Thread count: physical cores of the allowed CPUs capped by a cgroup CPU quota (a 10-CPU container on an 80-thread host ran 40
  spinning threads before); threads pinned one per physical core, one socket first, when the pool owns its CPUs; `cpus` option,
  `SEPAL_CPU_SET` and `sepal serve --cpus` to keep the runner off another engine's cores; `SEPAL_CPU_PIN=auto|on|off`.
  `sepal serve` no longer passes psutil's host core count; it prints the kernel set and thread layout (`sepal_info` field `runtime`).
- Buffers of 8 MiB and more advised for transparent huge pages on Linux (`SEPAL_HUGEPAGES=0` turns it off).
- DeltaNet: its micro-GEMM keeps the accumulators in named registers (GCC kept the `__m256 acc[R][NV]` array in memory and
  stored all of it on every k step), row blocks of 6; chunk preparation as small register-tiled products (q.k and k.k, the
  blocked forward substitution, the intra-chunk output); the units buffer is kept between layers; subnormal floats are flushed
  to zero inside the scan's tasks (MXCSR FTZ/DAZ, restored after each task): fast-forgetting heads produce decays below
  1.2e-38, and every operation on them took a microcode assist. 4B, 10 Broadwell cores, three ~860-token requests: DeltaNet
  2.90 s -> 1.08 s. Results change only through the one-chain sums of q.k and k.k and the flushed subnormals (parity below).
  A one-task-per-head variant (units kept in L2) was faster on synthetic data but 2x slower on the real model (a few heads
  dominate) and is not kept.
- `AlignedBuffer` move now carries the capacity (a moved-from buffer could be resized into a null pointer).
- `sepal-cpu-gemm-bench --shapes 4b|0.8b --isa avx2-tuned --peak-core GFLOPS`; `sepal-cpu-bench --cpus --cpu-isa --prefix-cache-mib`.

### 2026-11-01: `sepal serve`, int8 blocks, CPU prefix states

- `sepal serve MODEL` (`sepal_runner/cli.py`): fetch, hardware plan, serve; `/v1/models`, `/healthz`, `/readyz`; `examples/client.py`; `docker/Dockerfile.cpu`
  (built and run here under Docker Desktop: the engine compiles with GCC on Linux, the container answers with an API key) and `docker/Dockerfile.cuda`
  (written, not built or run).
- cpu-q8 AVX-512 kernel: scales per block of 128 values for weights and activations, in a separate non-inlined block function (MSVC spilled otherwise).
  4B, SEC-dev against the PyTorch reference: max |dp| 0.49 -> 0.14 over 300 records (p99 0.034, 1 flip of 432); TOOL (60 records) 0.017; LONG-LOG
  (12 records) 0.23, 2 flips of 162. Still over the 0.05 gate, so opt-in. 1.55x the speed of cpu-bf16 at 4 and 6 cores; committed memory 5.7 GiB for 4B (bf16 8.2),
  so 4B loads under an 8 GB cap.
- CPU prefix states (`prefix_cache_mib`): a request that shares its leading tokens with an earlier one resumes from the state after them.
  Same 4k log with 8 question sets, cpu-bf16, 6 threads: 27.5 s -> 6.9 s per request from the third on.

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
