# Sepal runner

Design document for the inference runner that serves the Sepal decision models. Status: design only; the repository
holds no runner code yet. **Nothing in this document was measured by this project.** Every number carries a marker:

- **measured elsewhere**: published or committed by another project; the source is named.
- **derived**: computed from a named anchor by scaling or arithmetic. A target that R0 or R2 replaces, not a result.
- **unverified**: an assumption, or a reported fact that nobody on this project has checked.

Reference code is cited as `repo@commit:path:lines`; short names and commits are listed in
[REFERENCES.md](REFERENCES.md). Related documents: [../README.md](../README.md), [MODEL.md](MODEL.md),
[DATA.md](DATA.md), [BENCHMARK.md](BENCHMARK.md), [ROADMAP.md](ROADMAP.md).

## 1. What this document decides

| Decision | Choice | Status |
|---|---|---|
| Substrate | Our own C++20 engine. Every matrix multiply goes through a swappable provider (reference, oneDNN, KleidiAI, cuBLAS, hipBLAS) | Decided |
| Role of llama.cpp / ggml | Oracle, throughput baseline and kernel donor. Not the engine | Decided |
| What can still flip | The weak-machine tiers move onto a pinned llama.cpp fork only if the R0 settling experiment meets every threshold in section 5 | Open until the end of R0 |
| First publishable number | Stage R2: the `gpu-f16` profile on one RTX 3060, passing its accuracy gate | Planned |
| Build first | Stage R0 (section 17): reference runner, golden set, settling experiment, geometry confirmation, substrate memo | Next |

The engine exists for one operation that neither llama.cpp nor the reference server provides: many ragged question
branches, from many requests, continuing one immutable snapshot of a state, in a single pass over the weights. The
row DAG, the snapshot trie and the token-budget scheduler all follow from making that operation cheap and exact.

## 2. Requirements and workload

A decision model reads a `state` and typed questions and returns a probability per option in one forward pass. The
caller is a SOC platform that will call the runner later. Out of scope: generation, sampling, streaming, training.

| Property | Value | Marker | What it forces |
|---|---|---|---|
| Request shape | One state plus 3-30 short question branches | unverified (design assumption) | Many ragged rows per state |
| State length | 200-2,000 tokens typical; limit 65,536 | typical range unverified; limit from `kev@5e42a7a:kev/model.py:14-28` | Short and long lanes; tiled attention |
| Row limit | State plus one branch <= 73,728 tokens | same lines | Admission before any compute |
| Options | `noul` 2, `choice` <= 255, `score` 2-10 levels | measured elsewhere (TypeSafe API; `kev@5e42a7a:kev/api.py:14`) | Readout width up to 255 |
| Question packs | A fixed, registered set of questions reused across requests | unverified (design assumption) | Tokenise once; the pack is cacheable |
| Shared prefixes | Alerts from one rule share leading state segments (render order in [DATA.md](DATA.md)) | unverified until a trace exists | Snapshot trie |
| Duplicate storms | Identical requests arrive in bursts | unverified until a trace exists | Result cache; in-flight coalescing |
| Latency | Batching is welcome; p95 <= 1 s is the benchmark's throughput criterion | from [BENCHMARK.md](BENCHMARK.md) | Token-budget batches with deadlines |
| Answers | Reproducible and calibrated; each answer names its precision profile | requirement | Determinism rules; per-profile temperature |
| Model sizes | 0.8B / 2B / 4B / 9B from one binary | requirement | Geometry read from the artefact |
| Hardware | A 4-core CPU without a GPU, up to several 24 GB GPUs | requirement | Precision profiles; scale-out |

## 3. What the model computes

This is the engine's contract. It was checked against Kev's reference code; the two transcriptions to port from are
`kev@5e42a7a:kev/fused_qwen35.py:62-118` (PyTorch) and `kev@5e42a7a:kev/mlx_model.py:128-229` (MLX). The geometry
was read from the base model files (measured elsewhere: each `config.json` at `huggingface.co/Qwen/Qwen3.5-<size>-Base`,
parameter counts summed from the safetensors headers). R0 confirms it by loading the files.

| | 0.8B | 2B | 4B | 9B |
|---|---|---|---|---|
| Layers (DeltaNet + full attention) | 24 = 18 + 6 | 24 = 18 + 6 | 32 = 24 + 8 | 32 = 24 + 8 |
| `hidden_size` / MLP `intermediate_size` | 1,024 / 3,584 | 2,048 / 6,144 | 2,560 / 9,216 | 4,096 / 12,288 |
| Attention heads / KV heads / `head_dim` | 8 / 2 / 256 | 8 / 2 / 256 | 16 / 4 / 256 | 16 / 4 / 256 |
| DeltaNet key / value heads; key / value head dim; conv taps | 16 / 16; 128 / 128; 4 | 16 / 16; 128 / 128; 4 | 16 / 32; 128 / 128; 4 | 16 / 32; 128 / 128; 4 |
| LM head tied to the embedding | yes | yes | yes | no (separate 1.02B-parameter head) |
| Non-embedding parameters | 498M | 1.37B | 3.57B | 6.92B |
| Body weights on device, fp16 / int8 (derived: 2 or 1 byte per parameter) | 1.0 / 0.5 GB | 2.7 / 1.4 GB | 7.1 / 3.6 GB | 13.8 / 6.9 GB |

Layer `i` is full attention iff `(i + 1) % 4 == 0`; all others are Gated DeltaNet. Common to all sizes: vocabulary
248,320; `rms_norm_eps` 1e-6; `rope_theta` 10,000,000 with `partial_rotary_factor` 0.25, so only 64 of the 256 head
dims rotate; `mrope_section` [11, 11, 10]; context 262,144. Weights are stored bf16 except DeltaNet `A_log` and the
gated-norm weight (fp32), and the config sets `mamba_ssm_dtype` float32. Two traps for the loader: `head_dim x heads`
differs from `hidden_size` at 0.8B and 4B, and `q_proj` is twice the attention width because the output gate is
packed with the query. Every size is read from the config, never derived from another.

**Decoder layer** (`kev@5e42a7a:kev/fused_qwen35.py:103-118`): `h = rmsnorm(x; 1 + w_in)`; `r = x + mixer(h)`;
`n = rmsnorm(r; 1 + w_post)`; `y = r + down(silu(gate(n)) * up(n))`. Gate and up are one GEMM.

**DeltaNet mixer** on the normed input (`kev@5e42a7a:kev/fused_qwen35.py:62-85`):

1. One bias-free GEMM with the in-projections concatenated; the output splits into `mixed [conv_dim]`,
   `z [value_dim]`, `b [Hv]`, `a [Hv]`.
2. Depthwise causal convolution over `mixed` (K = 4 taps), then SiLU, continued from the parent's tail: its last
   K-1 conv inputs.
3. Split into `q`, `k`, `v`; L2-normalise `q` and `k` per head; key heads are shared by their value heads.
4. The gated delta rule per value head (standard form, as inferred from the reference), with `beta_t = sigmoid(b_t)`
   and `g_t = -exp(A_log) * softplus(a_t + dt_bias)`: `S_t = exp(g_t) * S_{t-1} (I - beta_t k_t k_t^T) + beta_t v_t k_t^T`,
   `o_t = S_t q_t / sqrt(dk)`. The `1/sqrt(dk)` scale is as Strata's kernel applies it; to confirm against the
   transformers Qwen3.5 implementation when the FP64 reference is transcribed in R1.
5. Gated RMSNorm of `o` with gate `z`, then `out_proj`.

A token-sequential transcription of step 4 is `Strata@fb58e0d:src/prefill/kernels.cu:394-448`. It serves another
model of the family and fuses a sigmoid gate into its output norm (`Strata@fb58e0d:include/strata/prefill/kernels.hpp:59-63`);
the Kev call names `swish` (`kev@5e42a7a:kev/fused_qwen35.py:84`). The gate is a kernel parameter, tested in section 14.

**Attention mixer** (`kev@5e42a7a:kev/fused_qwen35.py:88-100`): one GEMM for `q` (with its output gate), `k`, `v`;
per-head RMSNorm of `q` and `k` with weight `1 + w`; RoPE on `q` and `k`; K (post-norm, post-RoPE) and V appended to
the cache; causal grouped-query attention; output multiplied by `sigmoid(gate)`; `o_proj`.

**Snapshot.** What a state leaves for the rows that continue it: per DeltaNet layer the recurrent state
`S [Hv, dk, dv]` and the conv tail `[K-1, conv_dim]`; per attention layer K and V for every state token. Derived
from the table: K/V at fp16 is 12,288 bytes per state token on 0.8B / 2B and 32,768 on 4B / 9B; the recurrent state
of all layers at fp32 is 18 MiB on 0.8B / 2B and 48 MiB on 4B / 9B. Two properties keep snapshots simple:

- A row that continues a snapshot reads it and never advances it (`kev@5e42a7a:kev/fused_qwen35.py:62-65`):
  branches never continue from each other, so snapshots can be immutable.
- A zero input leaves the DeltaNet state unchanged, so left-padding is exact (`kev@5e42a7a:kev/shared_prefix.py:14-15`).

**Request algorithm** (Kev's `encode`, `rows_of`, `PointerHead`: `kev@5e42a7a:kev/model.py:87-127`,
`kev@5e42a7a:kev/model.py:188-202`, `kev@5e42a7a:kev/model.py:205-223`). The delimiters are five reused Qwen special
tokens (`kev@5e42a7a:kev/model.py:9-11`); their ids, read from the base model files, are `<state>` 248060, `<q>`
248061, `<opt>` 248049, `</opt>` 248050 and `<decide>` 248062.

```
decide(state_segments, pack, adapter, profile):
  S = [<state>] + user_tokens(seg_1) + ... + user_tokens(seg_n)          # positions 0 .. Ls-1
  for each question j in pack:
    B_j = [<q>] + user_tokens(instr_j)
          + for each option o: [<opt>] + user_tokens(o) + [</opt>]
          + [<decide>]                                                   # positions Ls, Ls+1, ...
  admit: Ls <= 65,536 and Ls + len(B_j) <= 73,728 for every j, else reject
  snap = pass(S)                                    # state pass: publishes a snapshot, no readout
  h_j  = pass(B_j, parent = snap)   for all j       # every branch restarts at position Ls
  H    = final_norm(h_j)            at readout positions only
  z_jk = dot(k(H[</opt>_k]), q(H[<decide>])) / sqrt(256) / T             # pointer head, fp32
  p_j  = softmax over k of z_jk
```

`user_tokens` rewrites `<|name|>` so caller text cannot produce a delimiter token (`kev@5e42a7a:kev/model.py:78-81`).
Sepal tokenises state segments independently, which Kev does not; training must do the same ([DATA.md](DATA.md)).

**Never loaded.** A prefill-only engine has no vocabulary projection, sampler or decode-step path. Kev loads the
backbone without the LM head (`kev@5e42a7a:kev/model.py:251-257`). At 0.8B / 2B / 4B the head is tied to the
embedding, which stays in host memory as a lookup table; at 9B it is a separate matrix (about 2.0 GB at bf16,
derived) that is never read. The label-row readout of section 6 needs at most 255 rows, stored in the adapter.

## 4. Why the reference runtime is not enough

Kev's PyTorch server is correct and is our oracle. These are its costs, all measured elsewhere by Kev. One more is
structural: a cache miss runs a state pass and then a branch pass, so it reads every weight twice and pays the
bandwidth floor of section 8 twice.

| Loss | Number | Source |
|---|---|---|
| Kernel launches | About 2,000 per pass | `kev@5e42a7a:kev/cuda_graphs.py:3-8` |
| Short requests are launch-bound | 110-145 ms eager vs 13-22 ms with CUDA graphs and fused kernels (Kev-4B, H100, new short state) | `kev@5e42a7a:runs/serving-4b-h100/report.json`, `kev@5e42a7a:runs/fused-4b-h100/report.json` |
| Batching stops above 1,024 state tokens | Flat 10-12 requests/s for 2,200-token states from 1 to 64 clients (Kev-4B, H100) | `kev@5e42a7a:runs/grouping-4b-h100/report.json`, `kev@5e42a7a:kev/cuda_graphs.py:46-48` |
| State K/V replicated per row | 24.7 GB in one pass vs 8.9 GB one row at a time: 64 questions on a 4.8k-token state (Kev-4B, MLX) | `kev@5e42a7a:kev/model.py:41-47` |
| Resident memory | 14.3 GB for 8.4 GB of 4B weights | `kev@5e42a7a:docs/model-cards/kev-4b.md:258` |
| Prefix cache | Exact match on the whole state only; 4 states by default | `kev@5e42a7a:kev/serve.py:27-61` |
| Precision | Float only (fp32, bf16, fp16); no integer or compressed weights | `kev@5e42a7a:kev/checkpoint.py:134-141` |
| Platform | The fast kernels are Triton on CUDA, so Linux-only in practice (our inference; the cited lines do not say it); Apple Silicon takes a separate MLX path | `kev@5e42a7a:kev/fused_qwen35.py:1-20`, `kev@5e42a7a:kev/mlx_model.py:128-131` |

## 5. Substrate decision

**The argument.** The defining operation is ragged branches from many requests continuing one immutable snapshot.
llama.cpp is built around sequences that each own their memory and advance together. As reported upstream
(unverified by this project; R0 pins a commit and checks each item):

- it runs Qwen3.5: PR #19468 adds `ggml_gated_delta_net` on CPU, CUDA (token-sequential), Vulkan, SYCL and WebGPU;
- it exposes per-token hidden states after the final norm, so it can serve as an oracle;
- it re-gathers recurrent state on every micro-batch and requires equal new-token counts per sequence;
- its server cannot reuse cache on hybrid models (issue #24714, marked not planned).

The last two items are limits on exactly the operation we need. Lifting them inside llama.cpp means owning a fork of
its batch and memory layers.

**Settling experiment (R0).** 0.8B, a pinned llama.cpp build, a 1,000-token state, the laptop CPU and the RTX 3060.

| | Measurement | Flip threshold |
|---|---|---|
| (a) | max \|dp\| at readout positions vs transformers fp32 | <= 1e-3 |
| (b) | Branch tokens/s for 16 ragged branches on one state, divided by flat prefill tokens/s | >= 0.7 on both machines |
| (c) | Peak memory with 16 branches, divided by peak with one branch | recorded; no threshold |
| (d) | Restore time for a second request with the same state, divided by the state-pass time | <= 0.1 |
| (e) | Tokens/s for 8 states x 8 branches in one batch, divided by flat prefill | >= 0.7 on both machines |

The weak-machine tiers flip onto a pinned llama.cpp fork only if (a), (b), (d) and (e) all hold. Otherwise the
decision in section 1 stands. **Reverse check:** if our CPU path runs below 0.8x llama.cpp's flat prefill on the same
machine, we replace the GEMM provider, not the engine.

| Source | Components |
|---|---|
| Ours | Pass executor, snapshot store, caches, scheduler, planner, pack format, CPU DeltaNet / conv / attention kernels, FP64 reference path, C ABI, server, router |
| Strata (MIT) | GGUF reader and dequantisation (`Strata@fb58e0d:include/strata/artifact/gguf_reader.hpp`, `Strata@fb58e0d:include/strata/artifact/dequant.hpp`, `Strata@fb58e0d:src/kernels/cuda/dequant_bf16.cu`); GEMM wrapper (`Strata@fb58e0d:src/prefill/gemm.cu`); DeltaNet `gdn_gates` / `gdn_conv` / `gdn_recurrence` (`Strata@fb58e0d:src/prefill/kernels.cu:1903-2004`); graph capture, device arena, HIP shim, cache retention policy, state file, planner pattern (`Strata@fb58e0d:include/strata/core/graph.hpp`, `Strata@fb58e0d:include/strata/core/device.hpp`, `Strata@fb58e0d:include/strata/platform/hip_compat/strata_hip.h`, `Strata@fb58e0d:include/strata/program/conv_cache.hpp`, `Strata@fb58e0d:include/strata/core/conversation_file.hpp`, `Strata@fb58e0d:include/strata/plan/plan.hpp`); the design of the four-tier harness (its per-kernel parity programs, for example `Strata@fb58e0d:src/prefill/gdn_rec_parity.cu`) |
| Kev (Apache-2.0) | `encode` / `user_tokens` / `admit` / `rows_of`, the `kev/api.py` mapping, `PointerHead`, the merge-in-fp32-round-once rule (`kev@5e42a7a:kev/checkpoint.py:93-96`), the kernel-environment block (`kev@5e42a7a:kev/predictors.py:67-90`) |
| ggml (MIT) | A second transcription of the delta rule; tokenizer cross-check |
| Third party | oneDNN (x86), KleidiAI (ARM), cuBLAS, hipBLAS |

Strata's DeltaNet kernels must be re-parameterised: `Strata@fb58e0d:src/prefill/kernels.cu:22-23` hard-codes
`N = 2560 ... S = 128, HK = 16, HV = 48, C = 10240`, and the recurrence writes the state back in place
(`Strata@fb58e0d:src/prefill/kernels.cu:404-447`). We need geometry as arguments, a const parent state, a separate
output state and a row dimension.

**What Strata is not.** It has no CPU path for the trunk. Its attention kernel is sparse top-k, not dense causal
attention. Its batch slots carry one token per row (`Strata@fb58e0d:docs/BATCHING.md:1-5`), with no shared-prefix
batching. It has no backend abstraction layer. About 10% of it is relevant to a dense prefill engine (unverified).

## 6. Architecture

```
HTTP | C ABI -> Encoder -> ResultCache -> Coalescer -> Scheduler <- Planner
                                                          | Batch (row DAG, flat token list)
   SnapshotTrie {device | host | disk} <-> PassExecutor -> Readout + Calibration
                                           |- GemmProvider {ref, onednn, kleidiai, cublas, hipblas}
                                           '- SeqKernels {deltanet, conv, attention}
```

Three ideas carry the design:

1. **Row DAG, layer-interleaved.** A request becomes rows: a prefix row, the tail of the state, then branches. A
   row's parent is a cached snapshot or another row in the same pass. Consecutive uncached segments of one request
   are computed as one row, and the scan publishes a snapshot at each declared boundary inside it. Per layer, every
   GEMM runs once over the flat token list, so a cache miss reads the weights once, not once for the state and again
   for the branches. Only the delta-rule recurrence is sequential along the parent chain, and three waves per layer
   always suffice (`wave_begin[4]`): a prefix row shared by several requests in the batch; each request's own tail
   row, continuing from it or from a cached snapshot; the branches. Convolution and attention need no waves: their
   inputs for all rows exist as soon as the layer's GEMM has run. K/V is never replicated.
2. **Ragged rows packed with no per-row padding.** Tokens of all rows sit back to back in one list; a row is an
   offset and a length. Only the total `T` is bucketed, so captured GPU graphs are keyed by one number.
3. **Decomposition depends on the request, never on cache state.** Snapshots are published only at the
   caller-declared segment ends. A token's result depends on the tokens before it in its own chain and on nothing
   else in the batch, and must not change with whether that context was computed in this pass or restored from a
   snapshot. So a cache hit and a cache miss compute identical bits (tier 3 in section 14).

```
struct Snapshot {                // immutable after publish, refcounted
  Key key;                       // H(parent.key, segment ids) + base_sha + adapter + profile
  u32 n_tokens;
  KvRun kv[n_attn];              // refs to f16|i8 pages (K post-norm/RoPE, V), shared with ancestors
  StateBlob dn[n_dn];            // S[Hv,dk,dv] f32|f16 + conv tail[K-1,conv_dim] f32
  Tier tier; u64 bytes, last_use; bool pinned; };
struct Row   { u32 tok_off, len, pos0; i32 parent_row; Snapshot* parent_snap;
               u32 readout_off, n_readout; u16 adapter; bool publish; };
struct Batch { u32 T; i32* ids; Row* rows; u32 wave_begin[4]; };
struct Pack  { id, hash; branch token ids; readout offsets; };   // tokenised once
struct Profile { id; dtype per tensor class; quant recipe; deltanet variant; kv_fmt; state_fmt; };
```

**GEMM provider.** The contract that makes batching invisible: output row `t` never depends on other rows. Strata
already wraps this shape, fp32 out with optional dequantise-to-scratch (`Strata@fb58e0d:include/strata/prefill/gemm.hpp:28-48`).

```
gemm_f  (X[T,K] f16|bf16|f32, W[N,K], Y[T,N] f32, beta)
gemm_qd8(X f32 -> per-row dynamic s8, Wq s8 + scale[N], Y f32)     // exact int32 accumulate
gemm_dq (X f16, Wq s8|GGUF blocks -> f16 scratch, Y f32)           // compressed weights on device
caps()  -> dtypes, ISA, version, integer_exact
```

**Sequence kernel.**
`deltanet_scan(rows, mixed, a, b, z, const state_in[parent], const tail_in[parent], y, opt state_out[row], variant)`
runs convolution, recurrence and gated norm for the rows of one wave, parallel over (row, value head, column), and
writes `state_out` only at the declared boundaries of rows that publish. The default variant is sequential: token by
token inside one launch, state in registers, as Strata's default (`Strata@fb58e0d:src/prefill/kernels.cu:394-448`).
The chunked variant (32-token chunks, WY form) is opt-in per device: on the recurrence alone it is 1.8x on an RTX
5090 but 0.73x on an RTX 3060 (measured elsewhere, `Strata@fb58e0d:docs/DETAILS.md:1431-1436`). The variants give
different bits, so the variant is part of the profile and of the kernel-environment hash.

**Attention.** A row's context is a list of at most four segments: ancestor pages from the trie, an in-pass prefix
row, an in-pass tail row, and its own tokens. Scores against shared context are one unmasked GEMM per parent group
(all branches of one request share the same K); the row's own suffix is a small causal tile; the two merge by online
softmax. Long state rows use a tiled kernel with memory linear in length: untiled, 16 x 32k^2 x 4 bytes = 69 GB per
layer on 4B at 32k tokens (derived by Kev, `kev@5e42a7a:kev/predictors.py:38-48`).

- **Last-layer pruning.** The last layer is always full attention (layer rule, section 3). State rows compute only
  `k` and `v` there. Branch rows compute attention, MLP and final norm only at readout positions.
- **Loader and artefacts.** GGUF v3 container; geometry comes from `sepal.*` keys, never hard-wired or derived. The
  embedding table stays in host memory; a pass uploads only the rows it needs. Artefacts: `base-<size>-<profile>`;
  `adapter-<name>` (LoRA, head, calibration table keyed by profile); optional `merged-...`. The release also ships a
  human-readable `sepal.json` (layout, readout, label-token table, temperature and thresholds per profile; see
  [MODEL.md](MODEL.md)); the packaging tool writes the same metadata into the adapter artefact, which the runner reads.

**Readouts and layouts.** [MODEL.md](MODEL.md) tests two readouts and three layouts. One row DAG serves them all.

| Readout | Gather | Logits |
|---|---|---|
| Pointer head | `H` at each `</opt>` and at `<decide>` | `z_j = k(h_opt_j) . q(h_decide) / sqrt(256) / T`, fp32 |
| Label-row slots | `H` at each answer slot | `h` times the LM-head rows of the question's label tokens (<= 255 rows), divided by `T`, fp32 (label-token pattern: `AnyJev@d458d13:anyjev/readout.py:24-48`) |

| Layout | Rows | What is cacheable |
|---|---|---|
| Isolated branches after the state | prefix row -> tail row -> one row per question | State prefixes |
| All questions in one sequence | prefix row -> tail row -> one row holding every question | State prefixes |
| Pack-first | pack row (pinned root) -> state rows -> short answer-slot rows | The pack, then state prefixes under it |

## 7. Caches

| Level (cheapest first) | Key | A hit costs | Notes |
|---|---|---|---|
| Result cache | Hash of (raw state bytes, pack, adapter, profile, calibration) | One hash; no tokenising | About 200 bytes per entry (derived); bounded by bytes |
| In-flight coalescing | Same key | Nothing; the request attaches to the running one | Removes duplicate storms before the first answer exists |
| Snapshot trie | `H(parent.key, segment ids) + base_sha + adapter + profile` | The uncached suffix and the branches | Nodes only at caller-declared segment ends |

**Why nodes only at declared boundaries.** K/V pages are shared between a node and its ancestors, but the DeltaNet
state is a full copy per node. Derived size of one node for a 600-token state (anchor: section 3):

- **4B:** 600 x 32,768 bytes = 19.7 MB of K/V, plus 48 MiB (50.3 MB) of fp32 recurrent state, plus 2.4 MB of conv
  tails (24 layers x 3 taps x 8,192 channels x 4 bytes): about 72 MB in all. That falls to 47 MB with fp16 state,
  and to 37 MB with int8 K/V as well.
- **0.8B / 2B:** 7.4 MB of K/V, plus 18 MiB (18.9 MB) of state, plus 1.3 MB of conv tails (18 layers x 3 taps x
  6,144 channels x 4 bytes): about 28 MB.

The state part dominates below about 1,500 tokens at every size. A node per token is not affordable, and an
undeclared boundary would break idea 3 of section 6.

**Tiers and eviction.** Tiers are device, host and disk. Roots marked `cache: pin` are never evicted; other nodes are
least-recently-used by bytes and are demoted one tier at a time before being dropped. Strata uses the same retention
rule for its checkpoint chain (`Strata@fb58e0d:include/strata/program/conv_cache.hpp:1-40`). Snapshots are
refcounted, so eviction never frees pages under a running pass. They are keyed by adapter: a LoRA changes every
layer, so one adapter's prefix is useless to another. Branch rows never publish; a pack-first pack row does.

## 8. Scheduler and memory planner

- **Lanes.** Short and long (state over 4k tokens). A long state cannot delay short requests.
- **Token-budget batches.** A batch is filled by tokens, not by request count, up to the planner's limit.
- **Bandwidth floor.** A pass reads every weight once, so it costs at least (weight bytes / memory bandwidth). With
  FLOPs per token about 2 x parameters, the break-even is about (peak FLOP/s / bandwidth in bytes/s) x (bytes per
  weight / 2) tokens, independent of model size. Derived: roughly 100 tokens on an RTX 3060, 200-400 on an H100,
  about 3 on a CPU; Kev states 200-400 for H100-class cards at bf16 (`kev@5e42a7a:kev/cuda_graphs.py:54-55`). So a
  CPU is always compute-bound and runs work as it arrives, and a GPU pass is not dispatched below the floor: the
  scheduler holds a sub-floor batch until more tokens arrive or the earliest deadline runs out of slack.
- **Deadlines, fairness, backpressure.** Earliest deadline first (`deadline_ms`, `priority`) with per-tenant
  fairness, so one caller's storm cannot starve another. The queue is bounded; when it is full the server answers 429.

**Planner.** From geometry, profile and device memory it computes resident weights, scratch per pass as a function of
`T`, the snapshot budget per tier, the largest batch and the longest state the device admits. It refuses instead of
overcommitting, following `Strata@fb58e0d:include/strata/plan/plan.hpp:1-12`. A load that does not fit fails at start
and names the profile that would fit; a request that cannot fit is rejected at admission. This matters most on
Windows, where an oversubscribed GPU allocation does not fail but pages to system memory (measured elsewhere,
`Strata@fb58e0d:src/program/generate.cpp:4309`). Target (derived, R3): 64 questions on a 4.8k state in weights + 1.5 GB.

## 9. Request lifecycle

1. Parse the request; resolve model, adapter, pack and profile.
2. Look up the result cache on raw bytes. A hit answers immediately.
3. Coalesce: if an identical request is in flight, attach to it.
4. Tokenise each state segment; run admission (state <= 65,536 tokens, state + branch <= 73,728).
5. Walk the snapshot trie along the declared segments; build rows for what is missing, with their parents.
6. Enqueue in the short or long lane with a deadline; the scheduler forms a token-budget batch under the planner.
7. One fused, layer-interleaved pass; publishing rows add a snapshot to the trie at each declared boundary.
8. Gather readout positions; apply the head in fp32, then temperature and softmax.
9. Fill the result cache, answer every attached request, record metrics.

## 10. API

**HTTP.** `POST /v1/systemone` accepts the TypeSafe body unchanged (`state`, `model`, `questions`; mapping as in
`kev@5e42a7a:kev/api.py:102-160`). The extension fields are optional:

| Field | Meaning |
|---|---|
| `state` as `[{text, cache: pin \| lru \| none, key}]` | Declares segments and their cache policy; a plain string is one uncached segment |
| `pack: "<id>"` | Uses a registered pack in place of inline `questions` |
| `deadline_ms`, `priority` | Scheduling hints |
| `model: "sepal-4b@<adapter>"` | Selects an adapter on a loaded base |
| Response block `sepal` | `{profile, kernel_env, calibration, prefix_tokens_reused, result_cache}` |

Other endpoints: `POST /v1/systemone/batch` (many requests); `PUT /v1/packs/{id}` (register a pack, tokenised once);
`PUT | DELETE /v1/adapters/{name}` (load or unload, no restart); `GET /v1/models`, `/healthz`, `/readyz`, `/metrics`.
Status: 422 admission failure (count and limit named), 429 backpressure, 401 bad key, 404 unknown pack or adapter.

Example request and response. All probability and token values are **illustrative**, not model output.

```
POST /v1/systemone
{ "model": "sepal-4b@soc-v1",
  "state": [ {"text": "rule: ...\nasset: ...", "cache": "pin", "key": "rule-1"},
             {"text": "event: ...\nraw: ...", "cache": "none"} ],
  "pack": "triage-v1", "deadline_ms": 800 }

{ "answers": {
    "verdict":  {"type": "choice", "choice": "benign", "confidence": 0.715,
                 "probabilities": {"false_positive": 0.12, "benign": 0.81, "malicious": 0.07}},
    "escalate": {"type": "noul", "noul": 0.09} },
  "usage": {"input_tokens": 652},
  "sepal": {"profile": "gpu-f16", "kernel_env": "<hash>", "calibration": "<id>",
            "prefix_tokens_reused": 412, "result_cache": "miss"} }
```

- **C ABI.** `sepal_open`, `sepal_pack_register`, `sepal_adapter_load`, `sepal_submit`, `sepal_decide_batch`, plus
  the close, poll and free calls fixed in R1. The server is a client of this ABI.
- **Python.** A wheel: `eng = sepal.Engine("sepal-2b"); pack = eng.pack(questions); eng.decide(state, pack)`.
- **Adapter selection and hot-swap.** LoRA is applied unmerged and selected per row, so rows of different adapters
  share one batch: the base GEMM runs once and each adapter adds its low-rank term on its own rows. Loading does not
  stop the server. An unloaded adapter finishes its in-flight rows; its snapshots and results are then evicted.
- **Auth and bind defaults.** Bind 127.0.0.1. A non-loopback bind refuses to start without an API key.

## 11. Precision profiles

| Class | `cpu-q8` | `gpu-q8w` (<= 8 GB) | `gpu-f16` (>= 12 GB) |
|---|---|---|---|
| Embedding | int8 per row, host | host, fp16 upload | same |
| MLP gate / up / down | int8 per-channel + dynamic int8 activations | int8 stored -> fp16 compute | fp16 |
| Attention q/k/v/o, DeltaNet `in_proj_qkv / z` | int8 + dynamic int8, pending ablation | int8 -> fp16 | fp16 |
| DeltaNet `out_proj` (`ssm_out`), `in_proj_a / b` | fp16 stored, fp32 compute | fp16 | fp16 |
| `A_log`, `dt_bias`, norms, conv | fp32 | fp32 | fp32 |
| Recurrent state | fp32 | fp16 after experiment | fp32, fp16 opt-in |
| K/V | fp16 | fp16; int8 opt-in | fp16 |
| Head | fp32 | fp32 | fp32 |

**Evidence** (all measured elsewhere, on larger siblings; none on our sizes):

- No 4-bit profile. Reported on a larger sibling: KLD at the 99.9th percentile about 0.10 at Q8 and 0.41 at Q4;
  `ssm_out`, `attn_*`, the gates and beta / alpha are the sensitive tensors; Unsloth advises against 4-bit QLoRA on
  Qwen3.5 (Unsloth pages, see [REFERENCES.md](REFERENCES.md)).
- fp16 recurrent state on a 27B with 4-bit weights left perplexity unchanged (8.045 -> 8.044) and let 64 requests stay
  resident where 37 did before (`HyperQwen@3acb93f:docs/quality.md:30-38`, `HyperQwen@3acb93f:batch/README.md:36-37`).
- int8 activations on the same 27B cost +0.9% perplexity on gate / up only, +2.2% on the whole MLP and +3.7% on all
  linear layers (same table). Hence `cpu-q8` quantises MLP activations first and leaves the rest to an ablation.

**Coupling to training.** Adapters are served unmerged on quantised profiles. Training the adapter against the exact
dequantised base the runner computes is the intended mitigation for those profiles, adopted only if the go / no-go
experiment in [MODEL.md](MODEL.md) passes. Until then quantised profiles serve the standard adapter with per-profile
calibration and must pass the gates of section 14. On float profiles a merge follows Kev's rule: add the delta in
fp32 and round once (`kev@5e42a7a:kev/checkpoint.py:93-96`).

**Calibration.** One temperature is fitted per (adapter, profile, kernel set) and stored in the adapter's calibration
table; thresholds are refitted with it ([MODEL.md](MODEL.md)). Every answer carries the profile id.

## 12. Throughput model

Every figure in this section is **derived** unless marked otherwise. None is measured by this project.

- **FLOPs per token** are 2 x non-embedding parameters: 1.00 G at 0.8B, 2.75 G at 2B, 7.14 G at 4B and 13.84 G at 9B
  (derived; anchor: the parameter counts read from the base model files, section 3).
- **GEMM share** is taken as about 80% of pass time. Anchor, measured elsewhere: 79% of a 4k-token prefill on a 27B
  (`HyperQwen@3acb93f:docs/optimizations.md:125`).
- **Reference alert.** A 600-token state of which 400 tokens are a shared prefix, with 8 questions of 50 tokens:
  1,000 tokens without a cache, 600 with the prefix cache. Branches are 40-67% of the tokens.

Alerts per second, no cache / prefix cache. The second figure is the first, before rounding, times 1,000 / 600. On a
12 GB card the 9B with 8-bit weights is derived at about 1.9-2.4 alerts/s without cache.

| Hardware | 0.8B | 2B | 4B | 9B | Marker |
|---|---|---|---|---|---|
| 4-core laptop CPU | 0.15-0.28 / 0.25-0.47 | 0.06-0.10 / 0.09-0.18 | 0.02-0.04 / 0.035-0.065 | 0.011-0.020 / 0.018-0.034 | derived; nearest anchor: LiteRT dynamic-int8 CPU prefill |
| 8-core desktop CPU | 0.4-0.7 / 0.7-1.2 | 0.15-0.26 / 0.25-0.43 | 0.06-0.10 / 0.09-0.17 | 0.029-0.051 / 0.048-0.084 | derived; same anchor |
| 8 GB GPU | 10-20 / 17-33 | 4-8 / 7-13 | 3.2-4.3 / 5.3-7.2 | tight: about 6.9 GB of 8-bit weights before any working memory; unverified whether it fits | derived; nearest anchor: llama.cpp CUDA prefill on an RTX 3060 |
| 24 GB GPU | 30-100 / 50-167 | 12-40 / 20-67 | 11-29 / 18-48 | 5.6-15 / 9.4-25 | derived; anchor: llama.cpp CUDA prefill on RTX 3090 and 4090, scaled by FLOPs per token |
| 4 x 24 GB | 114-380 / 190-630 | 46-152 / 76-253 | 42-110 / 70-184 | 21-57 / 36-95 | derived; the 24 GB row times about 3.8 |

**Anchors**, all measured elsewhere (third-party reports; sources in [REFERENCES.md](REFERENCES.md)). The llama.cpp
figures were read by a research pass on 2026-10-08 and have not been re-read since; re-read each source before quoting.

- LiteRT model page, dynamic int8: 0.8B prefill at 70.5 tokens/s on a Raspberry Pi 5 and 666 on an M4 Max CPU.
- llama.cpp CUDA prompt-processing scoreboard (discussion #15013; Llama-2-7B Q4_0, 512-token prompt): 2,408 / 5,560 /
  14,771 tokens/s on RTX 3060 / 3090 / 4090.
- llama.cpp Vulkan scoreboard (discussion #10879; same test): 4,666 and 9,452 tokens/s on RTX 3090 and 4090, which
  is 0.84x and 0.64x of the CUDA figures (derived). The design record states this as about 0.65-0.9x of CUDA. The
  two scoreboards have different submitters, so the ratio is indicative only.
- llama.cpp issue #24714 (a user's report): a 2B Qwen3.5 at about 168 tokens/s on a Radeon 780M integrated GPU
  (Vulkan).

What moves these figures:

- **Weak-machine levers:** int8 GEMM; not computing at all (result cache, coalescing, prefix snapshots); the smallest
  model that passes the size rule in [MODEL.md](MODEL.md); moving question text into the cached prefix with the
  pack-first layout, since branches are 40-67% of the tokens.
- **Scale levers:** padding-free token-budget batches; prefix-affinity routing; fp16 state for snapshot density.
  For the routing, measured elsewhere on other serving stacks: 2.0x and 2.9x against round-robin (llm-d post "sticky
  until saturated"), and 82,665 -> 158,596 tokens/s as the cache hit rate rose from 20% to 75% (SGLang v0.4 post).
- **Not budgeted: early exit.** Reported as least suited to state-space-family models such as this one (measured
  elsewhere: arXiv 2603.23701); no stage builds it.

## 13. Scale-out

- **Workers.** One process per GPU, or per NUMA node on CPU hosts. A worker owns its weights, queues and trie.
- **Router.** Stateless; bounded-load consistent hashing on (adapter, first pinned segment key), so requests that
  share a prefix reach the worker that holds it. A key stays sticky until that worker's queued tokens pass a
  threshold, then spills to the next worker on the ring.
- **Shared and not shared.** Snapshots and queues are not shared between workers. An optional shared result cache is
  keyed with the kernel-environment id, so workers with different kernels never exchange answers.
- **Deployment.** Docker or systemd by default; Kubernetes only beyond one host.
- **Failure handling.** A decision is a pure function of its request, so the router retries on another worker. A
  lost worker loses only cache; pinned roots are rebuilt from the disk tier or recomputed. `/readyz` stays false
  until weights are loaded. Targets (derived, R6): 4 GPUs >= 3.6x one; prefix hit >= 70% on a trace; recovery < 2 s.

## 14. Verification and reproducibility

**Oracles.** O1: Kev's PyTorch path in fp32 on CPU. O2: our own scalar FP64 forward. O3: llama.cpp F32 (non-gating).

| Tier | What | Pass condition |
|---|---|---|
| 1 | Per-kernel FP64 transcriptions | The kernel matches within tolerance, and every rival formula differs from the reference by more than 100x that tolerance |
| 2 | Per-layer residual ladder | Hidden state after each layer within a per-layer bound against O2 |
| 3 | Snapshot hash | A cache hit and a cache miss give the same bits |
| 4 | End-to-end gates | The table below, on the golden sets |

Rival formulas for tier 1: decay applied before vs after the update; norm weight `1 + w` vs `w`; conv tail order;
RoPE layout; the `1/sqrt(dk)` scale; the output-gate activation (section 3).

| Profile | max \|dp\| | Flips | Other |
|---|---|---|---|
| fp32 | <= 1e-4 | 0 | none |
| `gpu-f16` | <= 0.03 (p99 <= 0.01) | <= 0.5% | dECE <= 0.002 |
| `cpu-q8` vs its float twin | <= 0.05 | <= 1% | vs the fp32 model: accuracy >= -0.5 pp, dECE <= 0.005 |

The design record sets no row for `gpu-q8w`; provisionally it is held to the `cpu-q8` row until R4 registers its own.
Reference tolerances, measured elsewhere by Kev: fp32 paths agree below 1e-4 (`kev@5e42a7a:tests/test_model.py:70-87`,
`kev@5e42a7a:tests/test_model.py:115-132`); served bf16 differs from fp32 by about 0.03 with about one flip in 300
(`kev@5e42a7a:README.md:409`); one swapped conv kernel moved a bf16 27B by 0.03-0.06 (cause at
`kev@5e42a7a:runs/drift-v1/REPORT.md:14-22`, size at `kev@5e42a7a:runs/drift-v1/REPORT.md:91-92`).

- **Golden sets.** v1 (R0): edge cases (1 token, 255 options, a 65,536-token state, each limit +/- 1, delimiter
  forgeries); 500 synthetic alert-shaped requests with shared prefixes; 2,000 questions from Kev's public development
  rows; long states. v2 adds SOCDecide development rows once that benchmark's first version exists. Each set stores
  fp32 CPU reference outputs with the kernel-environment block.
- **Determinism rules.** Fixed reduction trees per row. Threads partition outputs only, so no thread count changes
  the order of a sum. No TF32 and no fast-math: Kev measured TF32 dot products in a DeltaNet kernel moving an fp32
  read by about 0.003 in p (measured elsewhere, `kev@5e42a7a:AGENTS.md:431-432`).
- **Kernel-environment block.** Runner sha, compiler, provider and version, ISA, GPU, driver, OS mode, profile and
  weight shas, hashed onto every answer and every benchmark row. Results compare bit for bit only on equal hashes.
- **Accepting a new backend.** A provider is accepted when it declares `caps()`, passes a batch-invariance test (row
  `t` gives the same bits in any batch) and passes tiers 1-4.
- **Accepting a hardware report.** It needs its kernel-environment block, tier 4 results from the same build and the
  benchmark's 200-item parity subset at max |dp| <= 0.02 ([BENCHMARK.md](BENCHMARK.md)). Throughput alone is not recorded.

## 15. Language, build, packaging

- **Language.** C++20. The kernels and every lifted component are C++.
- **Build.** CMake presets; hash-pinned dependencies; CUDA 12.9 baseline (unverified); static CPU binary, ISA dispatch.
- **Packages.** Wheels `sepal-runner` and `sepal-runner-cu12`; images `sepal:cpu` and `sepal:cuda12`.
- **Platforms.** Windows native and Linux are first-class; WSL2 for development.
- **Licences.** oneDNN and KleidiAI Apache-2.0; ggml and Strata MIT; Kev Apache-2.0; the CUDA runtime with cuBLAS
  under the NVIDIA EULA (dynamically linked, never static); hipBLAS unverified until R5. Vendored code gets a `NOTICE` entry.

```
runner/
  include/sepal/                         public C ABI
  src/{abi, core, artifact, tokenizer, model, provider, kernels/{cpu, cuda},
       snapshot, cache, sched, plan, readout, server, router}
  python/  tools/  tests/  third_party/  docs/
reference/                               R0: Kev serve fork and oracle scripts
golden/                                  golden requests and reference outputs
```

## 16. Stages R0-R6

| Stage | Scope | Exit criterion (derived thresholds are targets) | Person-weeks (estimate) |
|---|---|---|---|
| R0 | Kev serve fork under WSL2; golden set v1; settling experiment; confirm the four base configs by loading the files; GGUF key spec | Goldens committed; PyTorch baseline alerts/s on the 3060; substrate memo | 1-2 |
| R1 | GGUF loader, encoder, FP64 and fp32 paths, oneDNN provider, single-request executor, ABI + Python | fp32 gate on 0.8B; all four geometries load; >= 0.8x llama.cpp CPU flat prefill | 4-5 |
| **R2** | cuBLAS provider; Strata GEMM / DeltaNet / graph / device adapted; attention | `gpu-f16` gate; 3060 >= 3,500 tokens/s on 4B and >= 10k tokens/s on 0.8B (derived). **First publishable number on the RTX 3060** | 4 |
| R3 | Multi-request row DAG, caches, tiers, scheduler, planner, server | Hit = miss bits; 3060 4B >= 4 / >= 7 alerts/s without / with prefix (derived); 64 questions on a 4.8k state <= weights + 1.5 GB | 5 |
| R4 | Quantiser, `cpu-q8`, KleidiAI, `gpu-q8w`, fp16 state, int8 K/V, calibration tooling (needs a trained adapter) | Task gates; laptop >= 150 tokens/s on 0.8B; 4B resident <= 6.5 GB | 4-5 |
| R5 | HIP via shim + hipBLAS; Vulkan only if an iGPU measures >= 2x its CPU | Gates on one Radeon; >= 0.6x same-class CUDA | 3-4 |
| R6 | Router, supervisor, manifests | 4 GPUs >= 3.6x; prefix hit >= 70% on a trace; worker kill recovered < 2 s | 3 |

## 17. First two weeks

1. Under WSL2, serve the released Kev-0.8B through Kev's own server on the RTX 3060 and measure baseline alerts/s on
   alert-shaped requests. Kev-4B is 14.3 GB resident (section 4) and does not fit 12 GB with CUDA graphs on.
2. Build golden set v1 (section 14) from Kev's public development rows and synthetic alert-shaped requests. Store
   fp32 CPU reference outputs with the environment block.
3. Run the settling experiment of section 5 against a pinned llama.cpp build on the laptop CPU and the 3060.
4. Confirm the geometry of section 3 by loading the four base model files. Write the `sepal.*` GGUF key specification.
5. Write the substrate memo: the decision, the measured numbers, and revised throughput targets that replace the
   derived figures of section 12. These five items are the R0 checklist.

## 18. Risks and the experiments that resolve them

| Risk | Experiment | Stage | If it goes badly |
|---|---|---|---|
| Chunked DeltaNet is slower on small GPUs (measured elsewhere: 0.73x on an RTX 3060) | Time both variants per device class | R2 | Sequential stays the default; chunked is selected by `caps()` |
| fp16 recurrent state is untested at 0.8B / 2B | Tier 4 gate with fp16 state on both sizes | R4 | State stays fp32; fewer snapshots per GB |
| int8 on `ssm_out` and `attn_*` costs accuracy | Per-tensor-class ablation against the `cpu-q8` gate | R4 | Those classes stay fp16 |
| Windows launch cost (measured elsewhere: 28-31 us per launch on the legacy default stream under WDDM, 3.63 us on a real stream, 0.805 us inside a captured graph; `Strata@fb58e0d:src/program/generate.cpp:2788-2794`) | Launches per pass and time per launch on the 3060 under Windows | R2 | Own stream always; graphs per `T` bucket; fuse small kernels |
| Unmerged LoRA overhead | Unmerged vs merged tokens/s on one adapter | R3 | Offer `merged-...` artefacts for single-adapter hosts |
| The snapshot trie does not pay at real alert diversity | Replay a trace; record prefix hit rate and bytes held | R3 | Keep the result cache and coalescing; shrink the trie budget |
| Segment-wise tokenisation changes answers | Whole-state vs per-segment tokenisation on the golden set | R0 | Train on the segmented form only; reject mixed use |
| cuBLAS picks kernels by shape, breaking batch invariance | Same row in different batch sizes; compare bits | R2 | Fixed `T` buckets or an own GEMM kernel; tier 3 relaxed only with the provider flagged |
| The standard adapter does not absorb activation quantisation | Profile gates with the standard adapter; then the go / no-go experiment in [MODEL.md](MODEL.md) (adapter trained against the dequantised base) | R4 | Float activations on the affected classes |
| fp16 results drift across GPU families | Golden set on two vendors and two generations | R2, R5 | Calibration per kernel set, as section 11 already requires |
| A 65k-token state does not fit in 8 GB | Planner arithmetic, then a real run | R3 | The planner lowers the admitted state length on that device |
