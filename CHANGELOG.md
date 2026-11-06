# Changelog

## Unreleased

### 2026-11-06: one-level Strassen / Winograd GEMM (prototype, opt in, off by default)

`SEPAL_CPU_GEMM=strassen1|winograd1` (or the `cpu_gemm` open option) runs the tuned AVX2 float GEMMs (`F32_Y3`, `BF16_Y3`) as one level of a
7-product 2x2 algorithm in the Strassen Reloaded form: the weight-side sums are precomputed at load in float32 (exact unless two bf16 exponents
differ by more than 15; 5 n k extra bytes for Strassen, 4 n k for Winograd, i.e. 2.5x and 2x the bf16 weight), the activation-side sums are
formed once per call from the packed blocks, single quadrants are read in place, and each product tile is added with its sign into its output
quadrants after its last k slice. The micro-kernels are unchanged. Applies to products of at least `SEPAL_FAST_GEMM_MIN_ROWS` rows (512) with an
even k and an even panel count; everything else stays classical. It gives up two determinism rules: an output is no longer one chain in increasing
k, and a row's result depends on the row it is paired with (t and t + rows/2), so on the batch; thread counts still change no bit. Max error
against float64 on small and odd shapes: 2.4e-7 (Strassen) and 4.0e-7 (Winograd) of max |x||w|, against 1.7e-7 classical. PC (Ryzen 7600X,
MSVC intrinsics, avx2-tuned, rows 1024) layer mix: 1 thread +3.7% / +6.7%, 6 threads +5.6% / +7.5%. Broadwell measurement pending.
`sepal-cpu-gemm-bench --gemm classical,strassen1,winograd1 --measure-peak 1` compares the modes interleaved.

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
