# Reference engine: verification against the upstream model

> **Moved from Sepal (2026-10-17).** This document was written in the Sepal repository; links to `../reports/`, `MODEL.md`, `DATA.md`, `BENCHMARK.md`, `ROADMAP.md` and `REFERENCES.md` point into https://github.com/bloomifycafe/Sepal. Python paths read `sepal_runner/...` (was `sepal/runner`, `sepal/refengine`, `sepal/serve/runner_api.py`).

`sepal_runner/refengine/` was written from a study of another project's code, without the model's own source. Twelve
details of its arithmetic were therefore listed as unverified. This document records how they were checked against
the Qwen3.5 implementation in `transformers`, what was measured, and what is still not covered.

**Result.** All twelve readings are what the upstream model computes. No arithmetic in the engine changed.
`UNVERIFIED` is now empty.

| Comparison (engine in NumPy float64 against upstream) | Largest absolute difference | Hidden-state magnitude |
|---|---|---|
| Tiny random model, upstream in float64 | 9.84e-13 | up to 4.04 |
| Real 0.8B checkpoint, upstream in float64 | 2.24e-13 | up to 43.4 |
| Real 0.8B checkpoint, upstream in float32 on CPU | 5.01e-05 | up to 43.4 |

## What was compared, and on what

- Date 2026-10-09. Windows 11, Python 3.12, `transformers` 5.19.0, `torch` 2.8.0+cu128, `safetensors` 0.8.0,
  RTX 3060 12 GB.
- Upstream source: `transformers/models/qwen3_5/modeling_qwen3_5.py`. Every line number below is in that file at
  version 5.19.0. The cache is `transformers/cache_utils.py`.
- **Execution path.** `flash-linear-attention`, `causal-conv1d` and `kernels` are not installed, so upstream ran its
  own PyTorch functions, and logged that it did: `torch_chunk_gated_delta_rule` (L291), `torch_recurrent_gated_delta_rule`
  (L428), `causal_conv1d_fn` (L260), `causal_conv1d_update` (L240). The fast kernels were not run.
- **Checkpoint.** `Qwen/Qwen3.5-0.8B-Base`, revision `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`, Apache-2.0, one
  safetensors file of 1,746,942,600 bytes.

Every number here comes from one of two commands, which print the full report as JSON:

    uv run --group verify python tools/verify_refengine.py --tiny     # tiny random model, no download
    uv run --group verify python tools/verify_refengine.py            # real checkpoint, downloads it

The tiny-model checks also run as tests in `tests/refengine/test_refengine_upstream.py`, with a tolerance of 1e-9.
They skip when `transformers` cannot be imported.

### Upstream is float32 inside, whatever the weights are

Upstream casts to float32 in the middle of the pass regardless of the model's dtype: both norms (L841-845, L217-222),
the RoPE table (L177, L186), the decay (L609), the whole delta rule (L328-331, L448-451) and the softmax of the eager
attention path (L731). A model converted with `.double()` is therefore only float32-accurate, and its cached
recurrent state is float32.

To compare at float64, `sepal_runner.refengine.upstream.float64_arithmetic` runs the same upstream code with `torch.float32`,
`torch.float` and `Tensor.float()` meaning float64 inside that module, and with the RoPE frequencies recomputed in
float64. No upstream line is replaced. Both variants are reported below: "float64" is the promoted run, "unmodified"
is the `.double()` model as it ships.

## The twelve details

"Other reading" is the largest hidden-state difference on the tiny model (40 tokens, upstream in float64) when that
one engine function is swapped for the alternative it had been listed against. With the engine as it is, the same
measurement gives 8.00e-14, so the parity test can tell each pair of readings apart.

| Engine function | What upstream computes | Lines | Engine changed | Other reading |
|---|---|---|---|---|
| `deltanet.causal_conv` | `F.conv1d` (cross-correlation), left-padded by K-1 and cut to the sequence: the last tap multiplies the current token. `bias=False`. | 260-279, 240-256, 512-519 | no | 4.15 (taps reversed) |
| `deltanet.l2_normalise` | `x * rsqrt(sum(x^2) + 1e-6)`: eps inside the root | 284-287, 334-335, 454-455 | no | 4.04e-05 (`x / max(norm, eps)`) |
| `deltanet.share_key_heads` | `repeat_interleave(value heads // key heads)` on the head axis | 610-612 | no | 4.13 (tiling) |
| `deltanet.delta_rule_update` | decay the state, read it with k, write `k (x) beta (v - k.S)`; the write is not decayed; state is [value heads, key dim, value dim] | 470-479, 402-415 | no | 2.27 (write, then decay) |
| `deltanet.delta_rule_scan` | the output reads the state after the token's own update | 481 | no | 3.49 (read before update) |
| `deltanet.delta_rule_output_scale` | the query is multiplied by `key_head_dim ** -0.5` | 337-338, 458 | no | 0.916 (no scale) |
| `deltanet.gated_norm_scale` | the weight itself, initialised to one, not `1 + w` | 211, 221 | no | 3.53 (`1 + w`) |
| `deltanet.gated_norm_gate` | `silu(gate)` | 213, 222 | no | 3.45 (sigmoid) |
| `attention.rope_angles` | `inv_freq = 1 / theta ** (arange(0, dim, 2) / dim)`, `dim = int(head_dim * partial_rotary_factor)`. Three position axes, equal for text, so one axis suffices. | 170-177, 185-203, 1250-1259 | no | 0.856 (divide by head dim) |
| `attention.apply_rope` | the first `rotary_dim` dims rotate, paired half-split: `x * cos + cat(-x2, x1) * sin` | 656-660, 688-698, 203 | no | 1.39 (interleaved pairs) |
| `attention.attention_scale` | `head_dim ** -0.5` | 748, 727 | no | 3.28 (`1 / head_dim`) |
| `attention.grouped_query_attention` | each key-value head repeated in place for its group of query heads | 702-711, 724-725 | no | 2.90 (tiling) |

Also read and confirmed, though never listed: the output gate is packed per head as query then gate and applied as
`sigmoid` after attention (L777-780, L808); zero-centred RMSNorm `x * rsqrt(mean(x^2) + eps) * (1 + w)` (L837-845);
decay `exp(-exp(A_log) * softplus(a + dt_bias))` (L609, L473); `beta = sigmoid(b)` (L607); the layer order
`x + mixer(norm(x))`, then `+ mlp(norm(.))` (L873-902); the final norm (L1292).

## Tiny-model parity

Geometry `PRESETS["tiny"]`: 8 layers with full attention at layers 3 and 7 (interval 4), hidden 24, MLP 40, attention
heads / key-value heads / head dim 4 / 2 / 16, DeltaNet key / value heads 2 / 4 with head dims 6 / 5, conv kernel 4,
partial rotary factor 0.25, theta 1e7, vocabulary 64. Every parameter is random; norm weights are perturbed so that
`w` and `1 + w` differ. Weights go into the engine through `Weights.from_arrays` under upstream's own tensor names.
All positions of each sequence are compared. Largest absolute difference:

| Tokens | SDPA, float64 | SDPA, unmodified | Eager, float64 | Eager, unmodified |
|---|---|---|---|---|
| 1 | 2.89e-15 | 3.46e-06 | 1.39e-15 | 5.01e-07 |
| 3 | 2.11e-15 | 1.13e-06 | 3.08e-15 | 1.44e-06 |
| 5 | 2.25e-14 | 3.50e-06 | 6.36e-15 | 8.41e-06 |
| 64 | 1.06e-13 | 4.54e-05 | 2.79e-14 | 1.85e-05 |
| 65 | 9.84e-13 | 2.72e-04 | 1.53e-13 | 3.39e-05 |
| 150 | 1.04e-13 | 4.29e-05 | 3.65e-13 | 4.64e-05 |

The two attention columns are separate random models. Lengths 65 and 150 cross upstream's delta-rule chunk of 64
(L297), so its chunked algorithm and the engine's token-by-token loop agree across chunk boundaries. The unmodified
column is upstream's float32 arithmetic and nothing else: promoting the casts removes it entirely.

## Real-checkpoint parity

Four sequences tokenised with the checkpoint's tokenizer, no special tokens, every position compared. "Top token" is
the argmax of `hidden @ embedding^T` (the vocabulary projection is tied). "Cosine" is the smallest over positions of
the cosine between the two hidden vectors.

| Tokens | Upstream | Largest abs diff | Mean abs diff | 1 - cosine | Top token agrees |
|---|---|---|---|---|---|
| 10 | float64 CPU | 1.82e-13 | 1.90e-14 | 2.2e-16 | 10/10 |
| 10 | unmodified `.double()` CPU | 7.87e-06 | 1.15e-06 | 1.1e-13 | 10/10 |
| 10 | float32 CPU | 5.01e-05 | 3.41e-06 | 1.9e-12 | 10/10 |
| 10 | float32 GPU | 3.54e-05 | 3.00e-06 | 1.3e-12 | 10/10 |
| 10 | float16 GPU | 5.52e-02 | 6.45e-03 | 5.1e-06 | 10/10 |
| 36 | float64 CPU | 1.71e-13 | 1.62e-14 | 4.4e-16 | 36/36 |
| 36 | unmodified `.double()` CPU | 1.89e-05 | 1.02e-06 | 1.8e-13 | 36/36 |
| 36 | float32 CPU | 3.14e-05 | 2.76e-06 | 1.6e-12 | 36/36 |
| 36 | float32 GPU | 3.72e-05 | 3.51e-06 | 3.2e-12 | 36/36 |
| 36 | float16 GPU | 5.48e-02 | 4.85e-03 | 4.1e-06 | 36/36 |
| 53 | float64 CPU | 2.24e-13 | 1.52e-14 | 4.4e-16 | 53/53 |
| 53 | unmodified `.double()` CPU | 2.16e-05 | 1.09e-06 | 9.5e-13 | 53/53 |
| 53 | float32 CPU | 4.34e-05 | 2.43e-06 | 1.4e-12 | 53/53 |
| 53 | float32 GPU | 4.89e-05 | 3.18e-06 | 2.8e-12 | 53/53 |
| 53 | float16 GPU | 5.75e-02 | 4.28e-03 | 2.4e-06 | 53/53 |
| 85 | float64 CPU | 1.82e-13 | 1.32e-14 | 4.4e-16 | 85/85 |
| 85 | unmodified `.double()` CPU | 6.84e-06 | 8.04e-07 | 6.1e-14 | 85/85 |
| 85 | float32 CPU | 3.00e-05 | 2.17e-06 | 8.2e-13 | 85/85 |
| 85 | float32 GPU | 3.06e-05 | 3.16e-06 | 1.6e-12 | 85/85 |
| 85 | float16 GPU | 4.51e-02 | 4.29e-03 | 3.0e-06 | 85/85 |

Hidden states have root mean square 3.6 to 4.1 and largest magnitude 36.3 to 43.4. The dimension of largest magnitude
also agrees at every position in every row. The float16 row is the stored bfloat16 weights converted with `.half()`,
which is itself lossy, then run in float16; it measures upstream's half precision, not the engine.

**Timing.** The engine took 1.30 s, 2.17 s, 2.62 s and 3.69 s for 10, 36, 53 and 85 tokens, after 1.2 s to copy the
weights to float64. Upstream took 0.18 to 0.43 s in float32 on CPU and about 1 s in float64. The whole tool, with the
weights already in the cache, ran in 42.3 s. Timings are single runs from `time.perf_counter`.

### Geometry of the 0.8B checkpoint

Read from `config.json` and from the tensor shapes in the safetensors file; equal to `PRESETS["0.8B"]` field for field.

| Assumed | Found |
|---|---|
| 24 layers, 18 DeltaNet + 6 attention | 24; 18 `linear_attn`, 6 `self_attn`, full attention at every fourth layer |
| hidden 1,024, MLP 3,584 | `embed_tokens` [248320, 1024], `gate_proj` [3584, 1024] |
| attention heads / kv heads / head dim 8 / 2 / 256 | `q_proj` [4096, 1024] (8 x 256, doubled for the gate), `k_proj` and `v_proj` [512, 1024], `o_proj` [1024, 2048], `q_norm` [256] |
| DeltaNet key / value heads 16 / 16, dims 128 / 128 | `in_proj_qkv` [6144, 1024], `in_proj_z` [2048, 1024], `in_proj_a` and `in_proj_b` [16, 1024], `A_log` and `dt_bias` [16], gated norm [128], `out_proj` [1024, 2048] |
| conv kernel 4 | `conv1d.weight` [6144, 1, 4] |
| vocabulary 248,320 | 248,320 |
| tied LM head | `tie_word_embeddings` true at both config levels; no `lm_head` tensor in the file; the official loader's `lm_head.weight` is the embedding object |
| rotary | theta 1e7, partial factor 0.25, `mrope_section` [11, 11, 10]; `rms_norm_eps` 1e-6 |

- **Conv bias: none.** The only conv tensor is `conv1d.weight`, and the module is built with `bias=False` (L515).
  The text model has no bias tensor of any kind.
- **Norm weights are stored zero-centred**, except the gated norm. The code scales by `1 + w` (L844) and initialises
  `w` to zero (L933-935). Stored means: `input_layernorm` 0.411 (range -0.605 to 1.469), `post_attention_layernorm`
  0.224, `q_norm` 0.506, `k_norm` 0.504, final norm 3.309; the gated norm, which is plain, 0.941 (range 0.348 to
  1.194). The statistics alone would not decide it. Running the engine with scale `w` instead of `1 + w` on the
  10-token sequence does: largest difference 34.8, cosine 0.028, top token 0/10.
- **Stored dtype.** bfloat16, except `A_log` and the final `norm.weight`, which are float32. All are exact in float32
  and float64, so both sides compute from identical values.
- **Other tensors in the file.** 320 belong to the text model, under the prefix `model.language_model.`, and the
  engine uses all of them. The rest are a vision tower (`model.visual.*`) and a multi-token-prediction head (`mtp.*`,
  which upstream also ignores, L915).
- The text model is loaded strictly from the stored tensors. As a cross-check, `Qwen3_5ForConditionalGeneration.from_pretrained`
  gives a text model with the same tensor names and bit-identical values.

## Snapshot semantics

On the tiny model, 90 tokens split at 1, 2, 3, 37, 64 and 89, for both attention paths, upstream in float64. A suffix
of one token takes upstream's token-by-token rule and `causal_conv1d_update`; longer suffixes take the chunked rule
with an initial state.

| Check | Largest difference over all splits |
|---|---|
| Upstream, prefix then suffix with its own cache, against upstream in one pass | 2.28e-13 |
| Engine snapshot after the prefix against the upstream cache: recurrent state | 1.22e-14 |
| ... conv tail | 1.58e-14 |
| ... attention keys | 2.17e-13 |
| ... attention values | 2.11e-13 |
| Engine continuing from the upstream cache, against upstream in one pass | 1.81e-13 |

The upstream cache holds, per DeltaNet layer, `conv_states[0]` [1, conv dim, K] and `recurrent_states[0]`
[1, value heads, key dim, value dim]; per attention layer, `keys` and `values` [1, kv heads, tokens, head dim], the
keys after their norm and RoPE. Those are the only tensors (asserted by a test). Differences from the engine's snapshot:

- **The conv state has K columns; the snapshot has K-1.** The oldest column is never read: zeroing it leaves the
  continuation bit-identical on both cached paths (test). After fewer than K tokens it is left-padded with zeros,
  like the engine's tail. An exporter of upstream caches must drop the first column.
- **No state is missing from the snapshot.** Upstream derives the next position from the cached key length
  (L1251-1252); the engine is given `start_position` and records `n_tokens`, which was equal in every case.
- In an unmodified upstream model the cached recurrent state is float32 even when the weights are float64.

## What changed in the engine

- No arithmetic. `deltanet.py`, `attention.py`, `primitives.py` and `executor.py` compute what they computed before.
- `sepal_runner/refengine/__init__.py`: `UNVERIFIED` is empty; each former entry is a comment citing the upstream lines.
- New `sepal_runner/refengine/upstream.py`: builds the upstream config and model for a geometry, converts upstream tensors
  to engine weights and an upstream cache to an engine snapshot, and provides the float64 promotion. It imports
  PyTorch and transformers only when called.
- New `tests/refengine/test_refengine_upstream.py` and `tools/verify_refengine.py`.

## What remains unverified

None of the twelve. Outside them, the following was not checked:

- **The fast kernels.** `flash-linear-attention`, `causal-conv1d` and the hub kernels were not installed, so parity
  is with upstream's PyTorch fallback, which upstream describes as the reference for them.
- **Key-head sharing on a real checkpoint.** 0.8B has 16 key and 16 value heads, so sharing is the identity there.
  It is confirmed on the tiny model (2 and 4) and in the source. The 4B and 9B geometries (16 and 32) rely on that.
- **The 2B, 4B and 9B checkpoints** were not downloaded; their presets are unchecked against real configs.
- **Longer sequences.** At most 150 tokens on the tiny model and 85 on the real one.
- **Batches with padding.** Upstream zeroes padded positions before the DeltaNet mixer (L227-236, L547). The engine
  runs one unpadded row at a time and has no counterpart.
- **Non-text positions.** With images the three RoPE axes differ (L185-203); the engine has one axis.
- **The cache's `record_past` mode** (`cache_utils.py` L1028-1061), which keeps more conv history for cropping.
- **The pointer head and the readouts** are Sepal's own and have no upstream counterpart; the tool loads the head as
  zeros.
