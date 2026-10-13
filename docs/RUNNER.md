# Sepal runner

Design document for the inference runner that serves the Sepal decision models. A future runner builder starts here.

## Status and what changed on 2026-10-13

**Runner status.** Paused since 2026-10-11; nobody is building it. Two pieces of this design exist as code: the first
C++ slice of stage R1 in `runner/` (build, model geometry as data, a reference matrix-multiply provider, basic
operations, a refusing memory planner; listed in `runner/README.md`), and a NumPy float64 reference implementation of
the trunk and of the old request semantics in `sepal/refengine/`. Neither has loaded a model file. The runner has no
DeltaNet step, no attention, no weight loader, no executor and no server, and has been built with MSVC on Windows
only. Stage R0 has not started. Nothing in the runner was measured on a model.

**Model contract changed.** On 2026-10-13 the owner decided to train Sepal with Unsloth's decision trainer, after a
local pilot ([../reports/pilot-unsloth/REPORT.md](../reports/pilot-unsloth/REPORT.md)). The runner therefore has to
serve this model, and not the Kev-style pointer head or the label-row slots this document was written for:

- **Backbone:** Qwen3.5 base at 0.8B / 2B / 4B / 9B, with a LoRA adapter (rank 16, alpha 32, dropout 0) on every
  linear layer of the language model, the DeltaNet projections `in_proj_qkv / z / a / b` and `out_proj` included.
  The vision tower and the output embedding are not adapted.
- **Readout:** Cloudflare's Clef `JointSchemaHead` (Apache-2.0), a small attention network that reads the final hidden
  states of **every** position of the sequence. It is not a pointer head and not label rows.
- **One prompt per request:** a system line, `STATE:` and the state, then `SCHEMA FIELDS:` with one block per
  question, then `JOINT SCHEMA DECISIONS:`. All questions of a request share this one sequence and one forward pass.
  Section 3 gives the exact format.

**What still holds, and what is superseded.**

| Section | State |
|---|---|
| 1 Decisions | Substrate and stage decisions hold. The sentence on ragged question branches is superseded |
| 2 Requirements | Edited: request shape, question packs, limits |
| 3 What the model computes | **Rewritten**: 3.1-3.7 are the new contract (read from source); 3.8 is the trunk (DeltaNet, attention, geometry), **unaffected** |
| 4 Why the reference runtime is not enough | Edited: the reference is now the Unsloth / Transformers forward; Kev's numbers are kept as evidence about the class of cost |
| 5 Substrate decision | Holds; settling items (b) and (e) redefined |
| 6 Architecture | Edited: row DAG reduced to prefix row plus one tail row per request; readout is the Clef head; last-layer pruning is gone |
| 7 Caches | **Rewritten** for the new prompt: what can and cannot be cached |
| 8 Scheduler and planner | Holds; the planner's head size and the 64-question target are superseded |
| 9, 10 Lifecycle, API | **Rewritten** |
| 11 Precision profiles | Holds (trunk); head row and calibration edited; one new risk on fp16 |
| 12 Throughput model | **Redone**: token counts and every rate |
| 13 Scale-out | **Unaffected** |
| 14 Verification | Edited: the oracle is the Unsloth / Transformers forward of the trained model |
| 15 Language, build | Unaffected, plus a licence line |
| 16, 17 Stages, first two weeks | **Rewritten** |
| 18 Risks | Edited |
| 19 Open questions | **New** |
| 20 Superseded design | **New**: the pointer-head / label-row design and the numbers built on it, kept for the record. Do not build from it |

**Licence status is an open question** (section 19, item 1): the decision modules of Unsloth carry
`AGPL-3.0-only` headers and Clef is Apache-2.0. A runner builder may read Unsloth's code to understand the contract
but may not copy AGPL code into an Apache-2.0 runner. Implement the head from the Apache-2.0 Clef source or from the
description in section 3.

**Markers.** Every number carries one:

- **read from source**: a fact read in a file; the file and lines are cited as `repo@version:path:lines`.
- **measured (pilot)**: read from an artefact of the 2026-10-13 pilot (`runs/pilot-unsloth/`, git-ignored, in the main
  checkout) or from its report. A pilot is one run on a 0.8B model; it is evidence about format and size, not about speed
  at scale.
- **measured elsewhere**: published or committed by another project; the source is named.
- **derived**: computed from a named anchor by scaling or arithmetic. A target that R0 or R2 replaces, not a result.
- **unverified**: an assumption, or a reported fact that nobody on this project has checked.
- **computed by this project's code**: arithmetic that code in this repository performs on the geometry of section
  3.8; the computing file is named. These are sizes and counts, not measurements of a running engine.

`unsloth@2026.10.4` means the installed wheel of that version (the pilot's `runs/pilot-unsloth/venv/Lib/site-packages/`),
not a git commit. Other reference code is cited as `repo@commit:path:lines`; short names and commits are listed in
[REFERENCES.md](REFERENCES.md). Related documents: [../README.md](../README.md), [MODEL.md](MODEL.md),
[DATA.md](DATA.md), [BENCHMARK.md](BENCHMARK.md), [ROADMAP.md](ROADMAP.md).
## 1. What this document decides

| Decision | Choice | Status |
|---|---|---|
| Substrate | Our own C++20 engine. Every matrix multiply goes through a swappable provider (reference, oneDNN, KleidiAI, cuBLAS, hipBLAS) | Decided |
| Role of llama.cpp / ggml | Oracle, throughput baseline and kernel donor. Not the engine | Decided |
| What can still flip | The weak-machine tiers move onto a pinned llama.cpp fork only if the R0 settling experiment meets every threshold in section 5 | Open until the end of R0 |
| First publishable number | Stage R2: the `gpu-f16` profile on one RTX 3060, passing its accuracy gate | Planned |
| Build first | Stage R0 (section 17): oracle and golden set from the Unsloth / Transformers forward, tokeniser cross-check, settling experiment, geometry confirmation, substrate memo | Not started. A first slice of R1 was written ahead of it (section 16). Runner paused |

The engine exists for one operation that neither llama.cpp nor the reference server provides: the tails of many
requests (the rest of the state, then the question schema), each continuing an immutable snapshot of a shared state
prefix, in a single pass over the weights, with the final hidden states of every position handed to the Clef head.
**Superseded (2026-10-13):** the first version of this sentence said "many ragged question branches ... continuing
one immutable snapshot". With one prompt per request the questions are no longer separate branches (section 7). The
row DAG, the snapshot trie and the token-budget scheduler still follow from making the new operation cheap and exact.
## 2. Requirements and workload

A decision model reads a `state` and typed questions and returns a probability per option in one forward pass. The
caller is a SOC platform that will call the runner later. Out of scope: generation, sampling, streaming, training.
The request shape follows Cloudflare's `/v1/systemone` body (section 10).

| Property | Value | Marker | What it forces |
|---|---|---|---|
| Request shape | One state plus 3-30 questions, all in one sequence after the state | unverified (design assumption) | Long rows; the question block is part of every row |
| State length | 200-2,000 tokens typical | unverified | Short and long lanes; tiled attention |
| Sequence limit | **Open.** The Kev API allowed a 65,536-token state and 73,728 per row (`kev@5e42a7a:kev/model.py:14-28`). Unsloth trains at 4,096 by default (6,144 in our pilot) and serves at 16,384; the whole request, schema included, must fit | read from source: `unsloth@2026.10.4:unsloth/models/_decision_common.py:62-64`; `reports/pilot-unsloth/REPORT.md` and `sepal/train/unsloth_arm.py:33-34` for the pilot | Admission before any compute; section 19, item 4 |
| Options | `noul` 2, `choice` <= 255, `score` 2-10 levels | measured elsewhere (TypeSafe API; `kev@5e42a7a:kev/api.py:14`). The Clef code itself sets no limit (`clef-flash@fde727a:joint_schema_model.py:552-561`); every option costs tokens | Head width scales with the option count |
| Question packs | A fixed, registered set of questions reused across requests | unverified (design assumption) | The pack's token ids can be cached; its compute cannot (section 7) |
| Shared prefixes | Alerts from one rule share leading state tokens (render order in [DATA.md](DATA.md)) | unverified until a trace exists | Snapshot trie over token ids |
| Duplicate storms | Identical requests arrive in bursts | unverified until a trace exists | Result cache; in-flight coalescing |
| Latency | Batching is welcome; p95 <= 1 s is the benchmark's throughput criterion | from [BENCHMARK.md](BENCHMARK.md) | Token-budget batches with deadlines |
| Answers | Reproducible and calibrated; each answer names its precision profile | requirement | Determinism rules; per-profile temperature |
| Model sizes | 0.8B / 2B / 4B / 9B from one binary | requirement | Geometry read from the artefact |
| Hardware | A 4-core CPU without a GPU, up to several 24 GB GPUs | requirement | Precision profiles; scale-out |

## 3. What the model computes

This is the engine's contract. 3.1-3.7 describe the model as trained by Unsloth's decision trainer; 3.8 is the trunk,
which did not change. The facts in 3.1-3.7 were read on 2026-10-13 from the installed Unsloth 2026.10.4 and the Clef
module it vendors, and checked against the pilot's saved model (`runs/pilot-unsloth/U1/final`). Nothing was executed
from the third-party files; the token counts below come from running a tokeniser on the template strings (section 12).

### 3.1 What is served

A Qwen3.5 text backbone with LoRA adapters, plus the Clef head. Training loads the backbone, adds LoRA with
`finetune_vision_layers=False`, `finetune_language_layers=True` and all-linear targets
(`unsloth@2026.10.4:unsloth/models/decision.py:706-736`), and trains the head from scratch at a learning rate of 1e-4
(`unsloth@2026.10.4:unsloth/models/decision.py:34`; a plain language model gets a fresh head through
`unsloth@2026.10.4:unsloth/models/decision_from_lm.py:144-229`).

Measured (pilot, 0.8B, `runs/pilot-unsloth/U1/final`): the adapter has r = 16, alpha 32, dropout 0, `use_rslora` false,
372 tensors (186 modules, A and B each), all float32, 10,822,656 parameters. Module names present: `q_proj`, `k_proj`,
`v_proj`, `o_proj`, `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `out_proj`, `gate_proj`, `up_proj`, `down_proj`
(`reports/pilot-unsloth/REPORT.md:9`; the adapter's key names are `base_model.model.model.language_model.layers.N.<mixer>.<proj>.lora_A.weight`
and `.lora_B.weight`). The runner applies the adapter to the DeltaNet projections too, including the small
`in_proj_a / b`, so those cannot be left out of the LoRA path on any profile.

### 3.2 The prompt

One string of ChatML is built by hand; the model's chat template is not used
(`clef-flash@fde727a:joint_schema_model.py:150-157`, same bytes as `unsloth@2026.10.4:unsloth/_vendor/clef/joint_schema_model.py:150-157`).
With `S` the rendered state and `F` the system line (below), the token sequence is the concatenation of the ids of:

```
<|im_start|>system\n{F}<|im_end|>\n<|im_start|>user\nSTATE:\n          prefix
{S}                                                                    state, cut at its end if needed
\n\nSCHEMA FIELDS:\n                                                    schema head
  for each question n = 1..Q, in the order of the request:
\nFIELD {n}\nID: {id}\nTYPE: {type}\nINSTRUCTION: {instruction}\nALLOWED OPTIONS:\n
  OPTION {k}: {"description":...,"option_id":...}\n      for each option k, one line
END FIELD\n
\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:      suffix
```

`F` is "Read the complete state and schema. Decide every field jointly. Each answer must be exactly one of that field's
allowed options." (`unsloth@2026.10.4:unsloth/_vendor/clef/joint_schema_model.py:24-27`). Rules, all read from
`unsloth@2026.10.4:unsloth/_vendor/clef/joint_schema_model.py`:

- **Rendering (`:35-43`).** A string is used as it is. Anything else is compact JSON with `ensure_ascii=False`,
  separators `,` and `:` and **sorted keys**. The same function renders instructions and option semantics.
- **Instruction (`:120-123`).** `None` or an empty string is replaced by the question id.
- **Options (`:46-57`).** `choice`: `sorted((str(key), value))`, so options are in code-point order of the key, whatever
  order the caller sent. `noul`: always `true` then `false`, with default descriptions "The proposition is true or the
  answer is yes." and "The proposition is false or the answer is no.", which `criteria` may override. `score`: option ids
  are the strings `"0" ... "n-1"` in list order, the description is the list entry.
- **Option line (`:129-138`).** `OPTION k: ` then the compact JSON `{"description": d, "option_id": id}` (keys sorted, so
  `description` first; `description` is left out when it is `None`), then a newline.
- **Spans (`:119-139`).** The encoder records the token span of the instruction text alone (not the `INSTRUCTION:`
  label) and of each option's JSON alone (not `OPTION k:`).
- **Tokenisation is piecewise (`:77-78`, `:110-188`).** Each fragment above is tokenised on its own with
  `add_special_tokens=False` and the id lists are concatenated: the prefix, the state, `\n\nSCHEMA FIELDS:\n`, then for every
  question the header fragment, the instruction, `\nALLOWED OPTIONS:\n`, and for every option `OPTION k: `, the JSON, `\n`,
  then `END FIELD\n`, and finally the suffix. The ids are therefore **not** the ids of tokenising the whole string, and a
  runner must reproduce the fragment boundaries exactly. Special tokens written in the text (`<|im_start|>` and so on)
  are matched as special tokens; text from the caller is **not** neutralised, so a state containing `<|im_end|>` is
  tokenised to the real special token (section 19, item 8).
- **Truncation (`:162-170`).** Optional `max_state_tokens`, then the state is cut **at its end** to
  `max_length - (prefix + schema + suffix)` tokens. Questions and options are never cut. If prefix, schema and suffix
  alone exceed `max_length`, `encode_record` raises `ValueError` (`:166-169`). The cut is silent; Unsloth's serving call
  re-encodes with one more token to tell a cut state from one that fits exactly
  (`unsloth@2026.10.4:unsloth/models/decision.py:506-513`).
- **Lengths.** `encode_record` defaults to 16,384 (`:106`). Unsloth's own defaults: 4,096 for training
  (`CLEF_MAX_LEN`), 16,384 for `predict` and serving (`CLEF_SERVE_MAX_LEN`)
  (`unsloth@2026.10.4:unsloth/models/_decision_common.py:62-64`, `unsloth@2026.10.4:unsloth/models/decision.py:1704-1725`). The
  pilot trained at 6,144 with states over 4,096 tokens skipped (`sepal/train/unsloth_arm.py:33`, `:157-166`) and scored
  at 16,384 (`sepal/train/unsloth_arm.py:34`).
- **Media.** Images and videos add vision placeholders after `STATE:` (`:28-29`, `:158-161`). Sepal sends none; the
  runner needs no vision encoder.

Measured (pilot): the saved tokenizer has `<|im_start|>` 248045, `<|vision_pad|>` 248055 and `<|endoftext|>` 248044
(`runs/pilot-unsloth/U1/final/tokenizer.json`). The pilot's padding token became `<|vision_pad|>` (`reports/pilot-unsloth/REPORT.md:53`);
it only matters for padded batches, and a ragged runner has none.

How our corpus reaches this format is in `sepal/train/unsloth_convert.py`: the state is `schema.state_text(record["state"])`
(`:60`), and our `noul` option keys `yes / no` become `true / false` (`:20-21`, `:38-40`). The runner's API must accept the
Unsloth names; any mapping back to ours belongs to the caller.

### 3.3 Which hidden states the head reads

- **Layer.** The last one only. `ClefModel.forward` calls the text model with `use_cache=False` and takes
  `last_hidden_state` (`unsloth@2026.10.4:unsloth/_vendor/clef/joint_schema_model.py:468-491`; Unsloth's trainer wrapper does the
  same, `unsloth@2026.10.4:unsloth/models/decision.py:292-342`). The Transformers Qwen3.5 text model applies its final
  RMSNorm before returning it (`transformers@5.17.0:src/transformers/models/qwen3_5/modeling_qwen3_5.py:1302`, the pilot's
  installed copy). No vocabulary projection is computed.
- **Positions.** Every non-padding position. The head's first operation is a LayerNorm over the hidden size
  (`:353`), then:
  - **memory**: every position, projected from the hidden size to the head width (`:357`);
  - **question vector**: mean of the normalised hidden states over the question's instruction span (`:359-364`);
  - **option vector**: mean over the option's JSON span (`:373-378`);
  - **lexical vector**: mean of the **output-embedding rows** of the option's token ids (`:379-383`, matrix passed at
    `:490`). At 0.8B / 2B / 4B the output embedding is tied to the input embedding; at 9B it is a separate matrix
    (section 3.8). The rows are the frozen base rows: LoRA does not touch the embedding or the output head
    (`unsloth@2026.10.4:unsloth/models/decision.py:706-736`);
  - **global vector**: the normalised hidden state of the **last** position, the `:` of `JOINT SCHEMA DECISIONS:` (`:358`).
- **Consequence for a runner.** The final hidden state of every token is an input, so no layer can be skipped for any
  token, including the state tokens and the final layer's MLP. "Last-layer pruning" of the old design is gone
  (section 20).

### 3.4 The head

`JointSchemaHead(hidden_size, width, routing_layers, layers, heads, feedforward, dropout)`
(`unsloth@2026.10.4:unsloth/_vendor/clef/joint_schema_model.py:281-459`). Unsloth's subclass has the same parameters and
state dict and a faster batched forward (`unsloth@2026.10.4:unsloth/models/clef.py:479-`); the reference per-record forward
is also kept there (`forward_per_record`). Structure, for one request with `Q` questions and `P` options in all:

1. `h = LayerNorm(hidden)`; `memory = W_mem h` for all `L` positions (bias-free).
2. Per option: `query = W_ctx mean(h[option span]) + W_lex mean(E[option token ids]) + W_oq q_vec[question]`
   (three bias-free projections; `q_vec` is the question vector).
3. **Evidence routing**, 2 layers: every option query cross-attends to `memory` (pre-norm multi-head attention, then a
   GELU feed-forward). There is no self-attention among options here (`:242-278`, `:399-403`).
4. Per question: `field = W_q q_vec + LayerNorm(sum_k softmax_k(option_k . W_q q_vec / sqrt(width)) option_k) + W_g global + type_embedding[type]`.
5. **Field layers**, 4 pre-norm `TransformerDecoderLayer`s: the `Q` field vectors **self-attend to each other without a
   mask** and cross-attend to `memory` (`:313-326`, `:421-424`). This is where answers to different questions interact.
6. Logit of option `k`: `prior + sigmoid(g) * (s_joint * cos(field, option_k) + MLP([field, option_k, field*option_k, |field-option_k|]))`,
   with `prior = s_prior * cos(lexical_k, normalise(q_vec + global))`; `g`, `s_prior`, `s_joint` are learned scalars and both
   scales are `exp(clamp(., max=log 100))` (`:426-457`).

**Sizes.** `default_head_config` picks width 1,024 if the backbone hidden size is at least 3,072 and 512 otherwise, `heads =
width / 64`, `feedforward = 4 * width`, 2 routing and 4 decoder layers (`unsloth@2026.10.4:unsloth/models/decision_from_lm.py:29-42`).
Parameter counts below are computed by a short script written for this document from the module definitions (not
by Unsloth's code); the 0.8B count equals the pilot's file and the 9B count equals the figure REFERENCES.md derived for
Clef-flash, which uses the same head configuration.

| | 0.8B | 2B | 4B | 9B |
|---|---|---|---|---|
| `hidden_size` (section 3.8) | 1,024 | 2,048 | 2,560 | 4,096 |
| Width / heads / feed-forward | 512 / 8 / 2,048 | 512 / 8 / 2,048 | 512 / 8 / 2,048 | 1,024 / 16 / 4,096 |
| Head parameters | 27,324,932 (**measured (pilot)**: sum of the 122 tensors in `joint_head.safetensors`) | 30,472,708 (derived) | 32,046,596 (derived) | 121,762,820 (derived) |
| Head at bf16 | 54.7 MB (pilot file: 54,661,902 bytes, scalars at fp32) | 61 MB (derived) | 64 MB (derived) | 243.5 MB (derived) |

0.8B / 2B / 4B rows follow from the default rule and the hidden sizes; whether the owner trains 2B / 4B / 9B with the
default is not decided in this repository (not verified). Always read the head config from the artefact.

**Cost of the head.** Per token of the sequence, the memory projection and the key/value projections of the memory in the
six cross-attention layers cost `2*h*w + 6*2*2*w*w` FLOPs: 7.3 MFLOP at 0.8B and 33.6 MFLOP at 9B (derived). That is 0.7%
and 0.24% of the trunk's FLOPs per token (section 12). Attention scores and the option/field work are not counted.
The head is a per-request cost on top of the trunk and does not batch with other requests' rows except by padding or by a
layout like `build_layout` (`unsloth@2026.10.4:unsloth/models/clef.py:281-340`).

### 3.5 From logits to probabilities

Each question has its own logit vector, one entry per option in the encoded order. Softmax is over that question's own
options only, in fp32 (`unsloth@2026.10.4:unsloth/_vendor/clef/joint_schema_model.py:568`).

| Type | Options (ids) | Answer (`systemone_answer`, `:523-543`) |
|---|---|---|
| `noul` | `true`, `false` | `noul = p(true)` |
| `choice` | sorted keys | `choice` = argmax, `confidence` = its probability, plus the probabilities |
| `score` | `"0" ... "n-1"` | `score = sum_i i * p_i`, `confidence = max p_i`, plus a legend and the probabilities |

The vendored function rounds to 4 decimals; a runner should return unrounded values and round only for display.

**Where a temperature applies.** The vendored Cloudflare code holds none. Unsloth's serving path divides the logits by a
temperature before the softmax (`unsloth@2026.10.4:unsloth/models/decision.py:453-503`, `:483-498`):
`T = head_temperature * T_bucket`, where `T_bucket` is the entry of `temperature_by_options` for the key `<type>:<size>`
(`size` one of `2`, `3-5`, `6-10`, `11+`) if present, else the per-type entry of `temperature`; per-type and bucket values are clamped to
[0.5, 5.0]. Source: `unsloth@2026.10.4:unsloth/models/_decision_common.py:402-417` and
`studio/backend/vendor/laya/common.py:219-240` of the installed package. **Trap:** the per-type list is indexed
in Unsloth's type order `("choice", "score", "noul")` (`_decision_common.py:57`), not Clef's `noul=0, choice=1, score=2`
(`joint_schema_model.py:32`). A `head_temperature` that Unsloth fitted is folded into the saved head weights at save time
(`log`-scale shift of the two logit scales and division of the last scorer layer, `decision.py:778-795`), recorded as
`folded_temperature`, and then does not appear again at serving (`decision.py:811-835`); if the fold would exceed the clamp
it stays as `head_temperature` in the config. The pilot's config holds `[1.0, 1.0, 1.0]` (**measured (pilot)**): no
temperature was fitted. Our own calibration ([MODEL.md](MODEL.md)) is a separate step; the runner stores its temperature per (adapter, profile, kernel set) as before (section 11).

### 3.6 Artefact files and formats

Measured (pilot): `model.save_pretrained(dir)` on a LoRA model writes the adapter and the head, not merged weights
(`unsloth@2026.10.4:unsloth/models/decision.py:906-931`, `sepal/train/unsloth_arm.py:202`). Folder
`runs/pilot-unsloth/U1/final`:

| File | Bytes | Format |
|---|---|---|
| `adapter_model.safetensors` | 43,346,432 | LoRA A/B, float32, 372 tensors, 10.82M parameters |
| `adapter_config.json` | 1,900 | PEFT config; `base_model_name_or_path` is a **local snapshot path** in the pilot, and `target_modules` is a regex over language-layer linears |
| `joint_head.safetensors` | 54,661,902 | 122 tensors; weights bf16, the three scalars (`prior_logit_scale`, `joint_logit_scale`, `residual_gate`) fp32 |
| `joint_head_config.json` | 123 | `{"hidden_size":1024,"width":512,"routing_layers":2,"layers":4,"heads":8,"feedforward":2048}` |
| `unsloth_decision_config.json` | 298 | `layout: "clef"`, `max_len` (6144), `temperature` [1,1,1], `base_model` (local path), `load_in_4bit` false, `fine_tuned` true |
| `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, `processor_config.json`, `README.md` | | Tokenizer and processor files, copied for reference |

Head and config file names are `unsloth@2026.10.4:unsloth/models/clef.py:36`. Writing is in `decision.py:838-851`.

**Merged 16-bit model.** `save_pretrained_merged(dir, tokenizer, save_method="merged_16bit")` merges the LoRA into the
backbone and writes it as a Hugging Face folder (`config.json`, `model*.safetensors` and an index), the processor files, the head
files, the decision config, and, for a `Qwen3_5ForConditionalGeneration` backbone, Cloudflare's `joint_schema_model.py` and
`LICENSE` so that Cloudflare's own loader works on the folder
(`unsloth@2026.10.4:unsloth/models/decision.py:854-903`, `:1319-1331`; Cloudflare's loader is `joint_schema_model.py:494-520`,
which reads `joint_head_config.json` and `joint_head.safetensors`). `save_pretrained` on a full fine-tune does the same. A merged folder
was **not** produced in the pilot and nothing here was run on one (not verified). How Unsloth rounds the merge is not
verified here; Kev's rule (merge the delta in fp32, round once) is the runner's own for float profiles.

### 3.7 Never loaded

A prefill-only engine has no sampler or decode-step path and never computes the vocabulary projection. It still needs
**rows of the output-embedding matrix**: the lexical vector and the prior of every option use them (section 3.3). At
0.8B / 2B / 4B these rows are the embedding table, kept in host memory as a lookup table. At 9B the output head is a separate
matrix (about 2.0 GB at bf16, derived); the old claim that it is never read is superseded, but only the rows of tokens that
occur in option spans are read, so a gather from host memory per request is enough and the matrix need not sit on the device.

### 3.8 Trunk: geometry, DeltaNet, attention (unaffected)

This part is unchanged by the 2026-10-13 decision. It was checked against Kev's reference code; the two transcriptions to
port from are `kev@5e42a7a:kev/fused_qwen35.py:62-118` (PyTorch) and `kev@5e42a7a:kev/mlx_model.py:128-229` (MLX). The geometry
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
| Body weights on device with every weight at one width, fp16 / int8 (derived: 2 or 1 byte per parameter; the planner's uniform profiles `w16` / `w8`) | 1.0 / 0.5 GB | 2.7 / 1.4 GB | 7.1 / 3.6 GB | 13.8 / 6.9 GB |
| Body weights under the quantised profiles `gpu-q8w` and `cpu-q8` of section 11 (computed by this project's code: `runner/src/plan/planner.cpp`) | 0.54 GB | 1.45 GB | 3.83 GB | 7.33 GB |

The int8 figure in the first of these two rows is not the size of any profile this design serves. The quantised
profiles keep the DeltaNet out-projection and the two small in-projections `in_proj_a / b` at 16 bits and the norms,
convolution and gate parameters at 32 bits (section 11), so their body is larger: 7.33 GB at 9B against 6.92 GB.
The `gpu-f16` profile is the fp16 figure. GB here is 10^9 bytes.

The parameter counts are reproduced by `count_params` in `runner/src/core/config.cpp` from the geometry alone
(498,113,344; 1,373,265,728; 3,570,052,096; 6,919,565,824), and its tests hold them to this table. That agreement
checks the arithmetic, not the geometry: both still rest on the published configs until R0 loads the files.

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

**Request-level semantics that survive.** A row that continues a snapshot reads it and never advances it, so snapshots are
immutable. What changed is who continues a snapshot: no longer one branch per question, but the tail of a request (section 7).

## 4. Why the reference runtime is not enough

The reference runtime for the Sepal model is the Unsloth / Transformers forward: the whole sequence of every request in one
pass with `use_cache=False`, so there is no reuse between requests (`unsloth@2026.10.4:unsloth/_vendor/clef/joint_schema_model.py:478-484`),
and the pilot ran it on Windows with the reference causal-convolution path and eager RMSNorm because `torch.compile` and
`causal_conv1d` were unavailable there (`reports/pilot-unsloth/REPORT.md:47`; **measured (pilot)**: 111 s to score the
900 records of the fixed subset on an RTX 3060, whether model loading is included is not stated). It is our oracle
(section 14). The costs below were measured by Kev on **Kev's** PyTorch server (a different readout); they describe the
class of cost of a PyTorch eager server and are not measurements of Unsloth's path. The structural cost "a cache miss runs
a state pass and then a branch pass, so it reads every weight twice" does not apply any more: a miss is one pass over state
and schema together.

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

**The argument.** The defining operation is request tails from many requests continuing immutable snapshots, with the final hidden
states of every position returned for the head (**revised 2026-10-13**; it was "ragged branches ... continuing one
immutable snapshot"). llama.cpp is built around sequences that each own their memory and advance together. It would
serve only the trunk: the Clef head is not part of it and would run outside. As reported upstream
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
| (b) | Tokens/s for 16 request tails (ragged schemas, 500-1,500 tokens each) continuing one shared state snapshot, divided by flat prefill tokens/s (redefined 2026-10-13; was 16 ragged question branches) | >= 0.7 on both machines |
| (c) | Peak memory with 16 tails, divided by peak with one tail | recorded; no threshold |
| (d) | Restore time for a second request with the same state, divided by the state-pass time | <= 0.1 |
| (e) | Tokens/s for 8 states x 2 tails in one batch, divided by flat prefill | >= 0.7 on both machines |

The weak-machine tiers flip onto a pinned llama.cpp fork only if (a), (b), (d) and (e) all hold. Otherwise the
decision in section 1 stands. **Reverse check:** if our CPU path runs below 0.8x llama.cpp's flat prefill on the same
machine, we replace the GEMM provider, not the engine.

| Source | Components |
|---|---|
| Ours | Pass executor, snapshot store, caches, scheduler, planner, pack format, CPU DeltaNet / conv / attention kernels, FP64 reference path, C ABI, server, router |
| Strata (MIT) | GGUF reader and dequantisation (`Strata@fb58e0d:include/strata/artifact/gguf_reader.hpp`, `Strata@fb58e0d:include/strata/artifact/dequant.hpp`, `Strata@fb58e0d:src/kernels/cuda/dequant_bf16.cu`); GEMM wrapper (`Strata@fb58e0d:src/prefill/gemm.cu`); DeltaNet `gdn_gates` / `gdn_conv` / `gdn_recurrence` (`Strata@fb58e0d:src/prefill/kernels.cu:1903-2004`); graph capture, device arena, HIP shim, cache retention policy, state file, planner pattern (`Strata@fb58e0d:include/strata/core/graph.hpp`, `Strata@fb58e0d:include/strata/core/device.hpp`, `Strata@fb58e0d:include/strata/platform/hip_compat/strata_hip.h`, `Strata@fb58e0d:include/strata/program/conv_cache.hpp`, `Strata@fb58e0d:include/strata/core/conversation_file.hpp`, `Strata@fb58e0d:include/strata/plan/plan.hpp`); the design of the four-tier harness (its per-kernel parity programs, for example `Strata@fb58e0d:src/prefill/gdn_rec_parity.cu`) |
| Kev (Apache-2.0) | The `kev/api.py` mapping, the merge-in-fp32-round-once rule (`kev@5e42a7a:kev/checkpoint.py:93-96`), the kernel-environment block (`kev@5e42a7a:kev/predictors.py:67-90`). Not any more: `encode` / `user_tokens` / `admit` / `rows_of` and `PointerHead` (section 20) |
| Clef (Apache-2.0) | The record encoder and the head, as the specification to implement (section 3); the files `joint_schema_model.py` and `LICENSE` in the Clef Hub repositories or in the vendored copy under `unsloth/_vendor/clef/` |
| ggml (MIT) | A second transcription of the delta rule; tokenizer cross-check |
| Third party | oneDNN (x86), KleidiAI (ARM), cuBLAS, hipBLAS |

Strata's DeltaNet kernels must be re-parameterised: `Strata@fb58e0d:src/prefill/kernels.cu:22-23` hard-codes
`N = 2560 ... S = 128, HK = 16, HV = 48, C = 10240`, and the recurrence writes the state back in place
(`Strata@fb58e0d:src/prefill/kernels.cu:404-447`). We need geometry as arguments, a const parent state, a separate
output state and a row dimension. (Unaffected by the 2026-10-13 decision.)

**What Strata is not.** It has no CPU path for the trunk. Its attention kernel is sparse top-k, not dense causal
attention. Its batch slots carry one token per row (`Strata@fb58e0d:docs/BATCHING.md:1-5`), with no shared-prefix
batching. It has no backend abstraction layer. About 10% of it is relevant to a dense prefill engine (unverified).

## 6. Architecture

```
HTTP | C ABI -> Encoder -> ResultCache -> Coalescer -> Scheduler <- Planner
                                                          | Batch (row DAG, flat token list)
   SnapshotTrie {device | host | disk} <-> PassExecutor -> Head stage (Clef) + Calibration
                                           |- GemmProvider {ref, onednn, kleidiai, cublas, hipblas}
                                           '- SeqKernels {deltanet, conv, attention}
```

Three ideas carry the design (revised 2026-10-13):

1. **Row DAG, layer-interleaved, now two levels deep.** A request becomes at most two rows: a **prefix row** (system
   prefix plus a state prefix, shared by requests that start with the same tokens) and a **tail row** (the rest of the
   state, the schema, the suffix). The tail's parent is a cached snapshot or the prefix row in the same pass. Per layer,
   every GEMM runs once over the flat token list, so a cache miss reads the weights once. Only the delta-rule recurrence
   is sequential along the parent chain, and two waves per layer suffice: prefix rows, then tail rows. Convolution and
   attention need no waves. K/V is never replicated. There are no per-question branch rows (section 7).
2. **Ragged rows packed with no per-row padding.** Tokens of all rows sit back to back in one list; a row is an
   offset and a length. Only the total `T` is bucketed, so captured GPU graphs are keyed by one number.
3. **Decomposition depends on the request, never on cache state.** Snapshots are published only at positions the
   request declares (or at the end of the state). A token's result depends on the tokens before it in its own chain and
   on nothing else in the batch, and must not change with whether that context was computed in this pass or restored
   from a snapshot. So a cache hit and a cache miss compute identical bits (tier 3 in section 14).

```
struct Snapshot {                // immutable after publish, refcounted
  Key key;                       // H(parent.key, token ids) + base_sha + adapter + profile
  u32 n_tokens;                  // system prefix + state tokens covered
  KvRun kv[n_attn];              // refs to f16|i8 pages (K post-norm/RoPE, V), shared with ancestors
  StateBlob dn[n_dn];            // S[Hv,dk,dv] f32|f16 + conv tail[K-1,conv_dim] f32
  Mem mem;                       // NEW: head memory for the covered tokens, [n_tokens, width] (section 7)
  Tier tier; u64 bytes, last_use; bool pinned; };
struct Row   { u32 tok_off, len, pos0; i32 parent_row; Snapshot* parent_snap;
               bool publish; u16 adapter; };            // the tail row also carries the SchemaSpans of its request
struct SchemaSpans { per question: question_span, option_spans[], option_ids[], type; };   // offsets relative to the schema start
struct Batch { u32 T; i32* ids; Row* rows; u32 wave_begin[3]; };
struct Pack  { id, hash; schema token ids; SchemaSpans; };     // tokenised once; the compute is not reusable (section 7)
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

**Attention.** A row's context is a list of at most three segments: ancestor pages from the trie, an in-pass prefix row,
and its own tokens. Scores against shared context are one unmasked GEMM per parent group; the row's own suffix is a causal
tile; the two merge by online softmax. Long rows use a tiled kernel with memory linear in length: untiled, 16 x 32k^2 x 4
bytes = 69 GB per layer on 4B at 32k tokens (derived by Kev, `kev@5e42a7a:kev/predictors.py:38-48`).

- **Every layer, every token.** The head reads the final hidden state of every position (section 3.3), so no layer is
  pruned for any token. The old "last-layer pruning" (state rows computing only `k` and `v` in the last attention layer,
  branch rows computing attention and MLP only at readout positions) is superseded and gone.
- **Head memory output.** The pass must hand over, for each request, `LayerNorm`-ready hidden states of all positions (or
  the head's memory projection of them). At 0.8B that is 2 KB per token at bf16 (derived), 3.3 MB for the 1,605-token
  reference alert of section 12.
- **Loader and artefacts.** GGUF v3 container; geometry comes from `sepal.*` keys, never hard-wired or derived. The
  embedding table stays in host memory; a pass uploads only the rows it needs, and the head gathers the rows of option
  tokens from it (section 3.7). Artefacts: `base-<size>-<profile>`; `adapter-<name>` (LoRA, **Clef head weights and its
  config**, calibration table keyed by profile); optional `merged-...`. The release also ships a human-readable `sepal.json`
  (prompt version, system line, readout, temperature and thresholds per profile; see [MODEL.md](MODEL.md)); the packaging
  tool writes the same metadata into the adapter artefact, which the runner reads. Until a packaging tool exists, the
  runner reads Unsloth's folder directly: `adapter_model.safetensors`, `joint_head.safetensors`, `joint_head_config.json`,
  `unsloth_decision_config.json` (section 3.6).

**Readout.** One readout: the Clef head of section 3.4, run in fp32 after the trunk pass. Requests of one batch can be
padded to a common `[Q, O]` layout as Unsloth does (`unsloth@2026.10.4:unsloth/models/clef.py:281-340`, `:404-`); the padded
logits are masked to `-1e4` past each question's options. The head is the only place where questions meet, and its
output for a request depends on that request's own hidden states only. The old readouts and layouts (pointer head,
label-row slots; isolated branches, one sequence, pack-first) are in section 20.

## 7. Caches

**What the new prompt allows.** The sequence is `[system prefix][state][schema][suffix]`. The causal backbone makes the hidden
state of a token depend only on the tokens before it. Two consequences decide everything below.

1. **What is cacheable: the system prefix plus a prefix of the state tokens.** The first 36 tokens (system line, `STATE:`;
   count computed by this project by tokenising the template string with the pilot's tokenizer, not through Unsloth's
   encoder) are identical for every request. The state is tokenised on its own (section 3.2), so its token ids depend on
   the state text alone and not on the schema. Position `p` of the first `36 + k` tokens therefore has the same K/V, the
   same DeltaNet state and the same final hidden state in every request whose rendered state begins with the same first `k`
   tokens. A snapshot at position `36 + k` is exactly the state the oracle would have computed, **provided that the oracle
   would have read at least `k` state tokens**. The oracle keeps `cut = min(len(state_ids), max_state_tokens,
   max_length - fixed_length)` state tokens, where `fixed_length` includes the schema, so `cut` depends on the schema's
   length when the state is long. A snapshot at `k > cut` must not be used; a snapshot at `k <= cut` may.
2. **What is not cacheable: anything after the state's last token.** The schema block follows the state, so its K/V and
   states depend on the whole state, and the suffix and the head's inputs depend on all of it.

So:

| Level | Key | A hit saves | Notes |
|---|---|---|---|
| Result cache | Hash of (raw state bytes, ordered questions with ids, types, instructions, criteria, adapter, profile, calibration, `max_length`) | Everything | Question order is part of the key: it changes the prompt |
| In-flight coalescing | Same key | Everything | Removes duplicate storms |
| State-prefix snapshot | `H(parent.key, token ids)` + `base_sha` + adapter + profile | The trunk pass over the system prefix and the first `k` state tokens, and the head's memory for them | **Valid only when the token prefix is identical.** Nodes at declared boundaries and at the end of the state |
| Per-question branch snapshot | none | not valid | Question `j` attends to questions `1..j-1` in the backbone and the model was trained on the concatenation (item 3 below) |
| Cross-request question-pack compute | none | not valid | The schema follows the state, so nothing computed for it can be shared across states. Only its **token ids and spans** are reusable (a registered pack, tokenised once) |

**Three details a builder must get right.**

1. **The head needs more than the trunk's state.** The head reads the final hidden state of **every** prefix token as
   memory (section 3.3). A snapshot must carry, for the tokens it covers, either the final-layer hidden states or the
   head's `memory_projection(LayerNorm(h))`. Storing the projection is enough for state tokens, because no state token
   belongs to a question or option span and the global vector is the last token of the sequence: **1,024 bytes per token**
   at width 512 and bf16 (0.8B / 2B / 4B), 2,048 at width 1,024 (9B) (derived). The per-layer key/value projections of that
   memory depend only on each token's own memory vector and may be cached too or recomputed (about 0.7% of trunk FLOPs at 0.8B, derived).
2. **A snapshot is a position, and a recurrent state cannot be cut.** K/V can be used up to any `k`, but a DeltaNet state
   exists only at the positions where it was saved. If `cut` falls before the last snapshot, use the nearest earlier one (or
   none) and recompute from there. The state's rendered text has to tokenise to the same ids as the cached prefix, so key the
   trie by token ids, never by text.
3. **Per-question branches are not a valid shortcut.** Running each question as a separate branch after the state snapshot
   would change the hidden states the head reads, and the model was trained on one concatenated sequence. Adding, removing
   or reordering a question changes every later question's tokens and, through the unmasked field self-attention, every
   field. This is what makes a result depend on the other questions (section 19, item 3). The old benefit of branching,
   K/V not replicated for 64 questions, is gone with it; a request is one row.

**Why nodes only at declared boundaries.** K/V pages are shared between a node and its ancestors, but the DeltaNet state is a
full copy per node. Derived size of one node for a 600-token state (anchor: section 3.8):

- **4B:** 600 x 32,768 bytes = 19.7 MB of K/V, plus 48 MiB (50.3 MB) of fp32 recurrent state, plus 2.4 MB of conv tails
  (24 layers x 3 taps x 8,192 channels x 4 bytes), plus 0.6 MB of head memory (600 x 1,024 bytes): about 73 MB in all. That
  falls to 48 MB with fp16 state, and to 38 MB with int8 K/V as well.
- **0.8B / 2B:** 7.4 MB of K/V, plus 18 MiB (18.9 MB) of state, plus 1.3 MB of conv tails (18 layers x 3 taps x 6,144
  channels x 4 bytes), plus 0.6 MB of head memory: about 28 MB.

The state part dominates below about 1,500 tokens at every size. A node per token is not affordable, and an undeclared
boundary would break idea 3 of section 6. Boundaries are chosen on token ids (for example at a newline that the trie has seen),
and the declared-segment extension of section 10 only suggests them.

**Tiers and eviction.** Tiers are device, host and disk. Roots marked `cache: pin` are never evicted; other nodes are
least-recently-used by bytes and are demoted one tier at a time before being dropped. Strata uses the same retention rule
for its checkpoint chain (`Strata@fb58e0d:include/strata/program/conv_cache.hpp:1-40`). Snapshots are refcounted, so eviction
never frees pages under a running pass. They are keyed by adapter: a LoRA changes every layer, so one adapter's prefix is
useless to another. Tail rows never publish.

**Bit-exactness against the oracle.** Unsloth runs the whole sequence in one pass. A prefix-plus-tail split changes where the
DeltaNet kernels start a chunk, so the runner will not match the oracle bit for bit, only within the tolerances of section 14.
That is also true of a miss against a hit once the kernel decomposition differs; idea 3 of section 6 keeps those two equal.

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
`Strata@fb58e0d:src/program/generate.cpp:4309`). Target (derived, R3): **superseded** (was 64 questions on a 4.8k state in weights + 1.5 GB, built on branch rows); the new R3 target is in section 16.

A first version of the planner exists (`runner/include/sepal/planner.hpp`, `runner/src/plan/planner.cpp`, tool
`sepal-info`). It computes weights per profile, a working-buffer figure, and a snapshot arena from what remains, and
on a shortfall it returns a refusal that names the missing bytes and the knob to reduce (weight precision, model
size, token budget or snapshot budget), never a smaller plan. Its weight and snapshot figures are exact arithmetic
on the geometry. Its working-buffer figure is an estimate from a formula documented in the header, not an
allocation count; R3 replaces it with the executor's real allocations. Its head size counts the two weight matrices
without their bias vectors (512 values, 2 kB at fp32), which changes no figure quoted in this document. **That is the
old pointer head.** The Clef head is 27.3M parameters (54.7 MB at bf16) at 0.8B and 121.8M (243.5 MB) at 9B (section 3.4), and the
planner must also size the head memory per snapshot token (section 7) and the per-request head workspace. Until it is
updated its figures for the head are wrong by that amount; the trunk figures are unaffected.

## 9. Request lifecycle

1. Parse the request; resolve model, adapter and profile. Validate each question (`type` in `noul / choice / score`;
   `choice` and `score` need non-empty `criteria`) as `systemone()` does (`unsloth@2026.10.4:unsloth/_vendor/clef/joint_schema_model.py:552-561`).
2. Look up the result cache on raw bytes. A hit answers immediately.
3. Coalesce: if an identical request is in flight, attach to it.
4. Render and tokenise: state with the rendering of section 3.2, schema fragment by fragment, in request order. Admission:
   system prefix + schema + suffix must fit `max_length` or the request is rejected (the encoder raises); cut the state at
   its end to what is left and remember whether it was cut.
5. Walk the snapshot trie along the state token ids up to `cut`; build a prefix row for what is shared and missing, and the tail
   row (remaining state, schema, suffix).
6. Enqueue in the short or long lane with a deadline; the scheduler forms a token-budget batch under the planner.
7. One fused, layer-interleaved pass; a row marked `publish` adds a snapshot, with its head memory, at each declared boundary.
8. Run the head in fp32 on each request's hidden states (cached memory for the prefix, fresh for the tail), divide logits by
   the temperature of section 3.5, softmax per question.
9. Fill the result cache, answer every attached request, record metrics.

## 10. API

**HTTP.** `POST /v1/systemone` accepts the TypeSafe body unchanged: `model`, `state` (a string or any JSON value), `questions`
(a dict of id to `{type, instructions, criteria}`), as Cloudflare's `systemone()` does
(`unsloth@2026.10.4:unsloth/_vendor/clef/joint_schema_model.py:547-576`; `kev@5e42a7a:kev/api.py:102-160` for the older mapping).
The response has `answers` per question (section 3.5) and `usage.input_tokens` (the length of the whole sequence, schema
included). Question order in the request is the order in the prompt. The extension fields are optional:

| Field | Meaning |
|---|---|
| `state_segments`: `[{text, cache: pin \| lru \| none, key}]` | Replaces `state`. The texts are concatenated with no separator, so the rendered state is the same text the model was trained on. **Not** a list in `state`: a list `state` is a JSON value and is rendered as compact JSON. Segment ends are hints for snapshot boundaries, which are placed on token ids |
| `pack: "<id>"` | Uses a registered pack (pre-tokenised schema) in place of inline `questions`. Saves tokenisation only (section 7) |
| `max_length` | Request-level cap on the whole sequence, up to the server limit (open question, section 19 item 4) |
| `deadline_ms`, `priority` | Scheduling hints |
| `model: "sepal-4b@<adapter>"` | Selects an adapter on a loaded base |
| Response block `sepal` | `{profile, kernel_env, calibration, prefix_tokens_reused, result_cache, truncated}`; `truncated` is true when the state was cut |

Other endpoints: `POST /v1/systemone/batch` (many requests); `PUT /v1/packs/{id}`; `PUT | DELETE /v1/adapters/{name}`
(load or unload, no restart); `GET /v1/models`, `/healthz`, `/readyz`, `/metrics`. Status: 422 admission failure (the
schema alone does not fit, count and limit named), 429 backpressure, 401 bad key, 404 unknown pack or adapter.

`noul` options are `true` and `false`; our corpus names them `yes` and `no` and `sepal/train/unsloth_convert.py` converts on
the way in. A `noul` answer is `p(true)`.

Example request and response. All probability and token values are **illustrative**, not model output.

```
POST /v1/systemone
{ "model": "sepal-4b@soc-v1",
  "state_segments": [ {"text": "rule: ...\nasset: ...\n", "cache": "pin", "key": "rule-1"},
                      {"text": "event: ...\nraw: ...", "cache": "none"} ],
  "questions": { "verdict":  {"type": "choice", "instructions": "Is this alert benign?",
                              "criteria": {"false_positive": "...", "benign": "...", "malicious": "..."}},
                 "escalate": {"type": "noul", "instructions": "Escalate to a human?"} },
  "deadline_ms": 800 }

{ "answers": {
    "verdict":  {"type": "choice", "choice": "benign", "confidence": 0.715,
                 "probabilities": {"false_positive": 0.12, "benign": 0.81, "malicious": 0.07}},
    "escalate": {"type": "noul", "noul": 0.09} },
  "usage": {"input_tokens": 1103, "output_tokens": 0},
  "sepal": {"profile": "gpu-f16", "kernel_env": "<hash>", "calibration": "<id>",
            "prefix_tokens_reused": 412, "result_cache": "miss", "truncated": false} }
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
| Recurrent state | fp32 | fp16 after experiment (measured on the GPU engine: `EngineOptions::half_state`, `runner/cuda/README.md` "Snapshot density and tiers": no flips in 512 prompts, |Δp| at most 1.3e-3 against float32 state) | fp32, fp16 opt-in |
| K/V | fp16 | fp16; int8 opt-in | fp16 |
| Clef head | fp32 (stored bf16, scalars fp32; upcasting is exact) | fp32 | fp32 |

**Accumulation on `gpu-f16`.** Operands are fp16 everywhere in that column; what the tensor cores add the products into is a
separate choice, and the GPU engine's default makes it fp16 for the four groups of matrix multiplies (MLP gate/up, MLP down,
the input projections, the output projections), which runs the multiplies at twice the fp32-accumulate rate on a GeForce card.
In natural batching that is cuBLAS `CUBLAS_COMPUTE_16F`; in strict batching (the mode whose cache hit equals its miss bit for
bit) it is the engine's own kernel, which adds fp16 partial sums into fp32 every 64 values of the inner dimension and is
batch-invariant by construction. Measured against the fp32 transformers golden on the 512-prompt alert workload
(`runner/cuda/README.md`, "The default multiply configuration"): natural max |dp| 8.4e-3 (p99 6.6e-3, 2 flips of 512, both within 0.01
of p = 0.5), strict max |dp| 2.0e-3 (p99 1.5e-3, 1 flip), against 1.9e-3 and 0 flips with fp32 accumulation; both inside the
`gpu-f16` row of the gate table in section 14. The fp32 accumulation stays selectable (`multiply_path = Cublas`, no `fp16_accumulate`).

**Weight size per profile.** Computed by this project's code: `trunk_weight_bytes` in `runner/src/plan/planner.cpp`,
from the geometry of section 3.8 and the table above, printed by `sepal-info`. Body weights only, in GB of 10^9 bytes;
the embedding table is separate and stays in host memory on the GPU profiles. Per-channel quantisation scales are
not counted (below 0.1% of an 8-bit matrix, per the planner's header).

| Profile | 0.8B | 2B | 4B | 9B |
|---|---|---|---|---|
| `w16`: every weight at 16 bits (arithmetic only, not a served profile) | 1.00 | 2.75 | 7.14 | 13.84 |
| `gpu-f16` | 1.00 | 2.75 | 7.14 | 13.84 |
| `gpu-q8w`, `cpu-q8` | 0.54 | 1.45 | 3.83 | 7.33 |
| `w8`: every weight at 8 bits (arithmetic only, not a served profile) | 0.50 | 1.37 | 3.57 | 6.92 |

The gap between the last two rows is the DeltaNet `out_proj` and `in_proj_a / b` kept at 16 bits and the small
tensors kept at 32. Nothing in the design shrinks a quantised 9B to 6.92 GB: that figure describes only the
all-8-bit arithmetic. The size can grow instead: if the per-tensor ablation of section 18 goes badly, the attention
projections and the DeltaNet `in_proj_qkv / z`, which the table quantises "pending ablation", also stay at 16 bits.

**What the planner says about the 9B on an 8 GB card** (computed by this project's code, same file; the card is
taken as 8 GiB = 8.59 GB, the embedding is on the host, and the working-buffer part is the planner's estimate, so
these are plans, not runs):

| Request | Reserve held back for driver and display | `gpu-q8w` (7.34 GB with the head) | `w8` (6.93 GB with the head) |
|---|---|---|---|
| Pass of 4,096 tokens in 32 rows, one 2,000-token snapshot | none | granted; room for 1 snapshot; largest pass 4,318 tokens | granted; room for 4 snapshots; largest pass 5,995 tokens |
| Same | 0.5 GiB | refused: pass must be 2,134 tokens or fewer | refused: no room for the snapshot |
| Pass of 1,000 tokens in 9 rows, one 600-token snapshot (the reference alert of section 12) | 0.5 GiB | granted; room for 6 snapshots | granted; room for 11 snapshots |
| Same | 1 GiB | refused: pass must be 341 tokens or fewer | granted; room for 4 snapshots |

So the weights of a quantised 9B do fit in 8 GiB, and what remains is 1.25 GB for everything else. Whether that is
usable depends on how much of the card the driver, the display and the CUDA context take, which nobody on this
project has measured: the 9B on an 8 GB card stays **unverified**, and the first real load in R4 settles it. On a
12 GiB card the same default request is granted under `gpu-q8w` with room for 37 snapshots (same code).

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
table; thresholds are refitted with it ([MODEL.md](MODEL.md)). Every answer carries the profile id. The temperature
divides the logits before the per-question softmax (section 3.5). Unsloth's own per-type and per-option-count temperatures
and the order trap of section 3.5 apply if the runner reads `unsloth_decision_config.json`.

**New risk: fp16 and the gated delta net.** Unsloth reports that Qwen3.5's gated delta net produces NaN in pure float16 and
puts such a load on a float32-activation path (`unsloth@2026.10.4:unsloth/models/decision.py:644-650`,
`unsloth@2026.10.4:unsloth/models/decision_from_lm.py:107`; reported, not checked here). The `gpu-f16` profile keeps the
recurrent state and the small tensors in fp32 and was measured on the earlier GPU engine to stay within the gate on goldens of
the old format (above), but it has not been run on the trained Sepal model; the Tier 4 gate on the new goldens decides.

**Training precision.** The pilot trained and scored in bf16 (`reports/pilot-unsloth/REPORT.md:9`). Unsloth also offers a
dynamic 4-bit base (`load_in_4bit`) whose dequantised weights differ from ours; an adapter trained on that base is served on
the same base only (`unsloth@2026.10.4:unsloth/models/decision.py:594-703`, `config["load_in_4bit"]`). The coupling to
training above is unchanged.

## 12. Throughput model

Redone on 2026-10-13 for the one-prompt layout. Every figure in this section is **derived** unless marked otherwise. None
is measured by this project.

- **FLOPs per token** are 2 x non-embedding parameters: 1.00 G at 0.8B, 2.75 G at 2B, 7.14 G at 4B and 13.84 G at 9B
  (derived; anchor: the parameter counts read from the base model files, section 3.8). The head adds 0.7% (0.8B) to 0.24% (9B)
  of that per token (section 3.4); it is left out of the rates below.
- **GEMM share** is taken as about 80% of pass time. Anchor, measured elsewhere: 79% of a 4k-token prefill on a 27B
  (`HyperQwen@3acb93f:docs/optimizations.md:125`).
- **Token accounting.** The sequence is `fixed + state + questions`. The fixed part is the sum of the three template
  pieces, computed by this project by tokenising the template strings with the pilot tokenizer (not through Unsloth's
  encoder): system prefix 36, `SCHEMA FIELDS` head 7, suffix 18 = **61 tokens**. A question block is 26 tokens of scaffold
  (`FIELD n`, `ID:`, `TYPE:`, `INSTRUCTION:`, `ALLOWED OPTIONS:`, `END FIELD`, with a one-token id), plus the id's own tokens,
  the instruction, and per option 13 tokens of scaffold (`OPTION k: `, the JSON keys and quotes, the newline) plus the
  option's own id and description. The JSON per option is why a question costs far more than in the old format.
- **Reference alert (redefined).** A 600-token state, of which the first 400 tokens are shared with other alerts, and 8
  `choice` questions with 3 options each. Each question carries 50 tokens of own content (instruction, option ids and
  descriptions) and a 3-token id: both are assumptions carried from the old reference alert (unverified). One question
  block is then `26 + 3 + 3 x 13 + 50 = 118` tokens (derived).

| Quantity | Old (branches) | New (one prompt) | Marker |
|---|---|---|---|
| Tokens, no cache | 1,000 = 600 state + 8 x 50 | **1,605** = 61 fixed + 600 state + 8 x 118 | derived |
| Share of the tokens that are questions | 40% | 59% (944 of 1,605) | derived |
| Cacheable prefix | 400 (shared state head) | 436 (36 system prefix + 400) | derived |
| Tokens computed with a prefix hit | 600 | **1,169** = 1,605 - 436 | derived |

The question block is 59% of the work and none of it is cacheable across requests (section 7). The old reference alert's
cached branch text, "moving question text into the cached prefix with the pack-first layout", is no longer available with
this prompt. For scale: the pilot's training records average about 740 tokens each, 5,923,716 tokens over 2 epochs of
roughly 4,000 records (derived from `reports/pilot-unsloth/REPORT.md:41-47`, the record count is the nominal slice size),
with about 1.9 questions per record (7,634 questions, `REPORT.md:53`). Real alerts are shorter than the reference alert;
the table below is for the reference alert only.

Alerts per second, no cache / prefix cache. The first figure of each pair is the old no-cache rate (derived, kept in section 20.4)
times 1,000/1,605 = 0.623. The second is the old no-cache rate times 1,000/1,169 = 0.855, which is the old prefix-cache rate times 0.513. The old 9B-on-12-GB statement stands: the rate is a FLOPs-per-token scaling and does not depend on
the weight size; the weights it would run on are the 7.33 GB of `gpu-q8w` (section 11), which the planner grants on 12 GiB.

| Hardware | 0.8B | 2B | 4B | 9B | Marker |
|---|---|---|---|---|---|
| 4-core laptop CPU | 0.094-0.17 / 0.13-0.24 | 0.037-0.062 / 0.051-0.086 | 0.012-0.025 / 0.017-0.034 | 0.0069-0.013 / 0.0094-0.017 | derived; nearest anchor: LiteRT dynamic-int8 CPU prefill |
| 8-core desktop CPU | 0.25-0.44 / 0.34-0.60 | 0.094-0.16 / 0.13-0.22 | 0.037-0.062 / 0.051-0.086 | 0.018-0.032 / 0.025-0.044 | derived; same anchor |
| 8 GB GPU | 6.2-12.5 / 8.6-17 | 2.5-5.0 / 3.4-6.8 | 2.0-2.7 / 2.7-3.7 | tight: 7.33 GB of weights under `gpu-q8w` before any working memory (6.92 GB only if every weight were 8-bit; section 11); unverified whether it fits, so no rate is given | derived; nearest anchor: llama.cpp CUDA prefill on an RTX 3060. The two weight sizes are computed by `runner/src/plan/planner.cpp` |
| 24 GB GPU | 19-62 / 26-86 | 7.5-25 / 10-34 | 6.9-18 / 9.4-25 | 3.5-9.4 / 4.8-13 | derived; anchor: llama.cpp CUDA prefill on RTX 3090 and 4090, scaled by FLOPs per token |
| 4 x 24 GB | 71-240 / 98-330 | 29-95 / 39-130 | 26-69 / 36-94 | 13-36 / 18-49 | derived; the 24 GB row times about 3.8 |

**One measured anchor on the new model (pilot).** Unsloth's forward scored the fixed 900-record subset of the pilot in 111 s on
an RTX 3060 under native Windows, with the reference causal convolution and eager RMSNorm (`reports/pilot-unsloth/REPORT.md:41-47`):
about 8.1 records/s (derived from that figure). The records are the pilot's short records, not the reference alert, and
whether the 111 s includes loading is not stated. R0 re-measures this as the baseline (section 17).

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
  model that passes the size rule in [MODEL.md](MODEL.md). The pack-first lever of the old document is gone.
- **Scale levers:** padding-free token-budget batches; prefix-affinity routing on the state's leading tokens; fp16 state for
  snapshot density. For the routing, measured elsewhere on other serving stacks: 2.0x and 2.9x against round-robin (llm-d
  post "sticky until saturated"), and 82,665 -> 158,596 tokens/s as the cache hit rate rose from 20% to 75% (SGLang v0.4 post).
- **Schema tokens are the new target.** With 59% of the reference alert in the question block, shorter instructions and
  fewer or shorter option descriptions cut cost directly. That is a training and data choice ([MODEL.md](MODEL.md)); the
  runner cannot change the prompt without changing the answers.
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

**Oracles (revised 2026-10-13).** O1: the forward of the **trained model** in fp32 on CPU, in two forms: O1a, Cloudflare's own
`joint_schema_model.py` (`load_release_model`, `unsloth@2026.10.4:unsloth/_vendor/clef/joint_schema_model.py:494-520`, Apache-2.0)
on a merged 16-bit folder produced by `save_pretrained_merged` (section 3.6); O1b, Unsloth's own forward and encoder
(`FastDecisionModel.predict`, `unsloth@2026.10.4:unsloth/models/decision.py:1704-1726`; AGPL-3.0-only, read and run as a black box, never copied). O1a
is the licence-clean oracle; O1b is checked against it once, because Unsloth's head forward is a re-implementation with
the same state dict. O2: our own scalar FP64 forward. O3: llama.cpp F32 (non-gating, trunk only). Kev's PyTorch path is no
longer an oracle for this model.

**State of O2.** It exists as `sepal/refengine/`: a NumPy float64 forward pass driven by a geometry object, with
the snapshot contents of section 3.8, a row executor in which a row's result depends on its own chain only, and the two
**old** readouts (pointer head, label rows). **It has no Clef head and no one-sequence encoder**: the trunk part is
unaffected, the readout and the request layout must be added from section 3 (R1 work, section 16). Its tests run on random weights and small test geometries and check its own invariants (a row continued
from a stored snapshot equals the same row computed in one go; branches do not see each other; left-padding with
zero input is exact). It has not been compared with O1 or with any real checkpoint, so it is not yet an oracle for
the model, only a definition of the request semantics. Twelve arithmetic details that Kev's reference hands to
library kernels were written out from their description and are listed in `sepal/refengine/__init__.py`
(`UNVERIFIED`) until a real model confirms them:

- DeltaNet: the convolution's tap order and absence of bias; the L2 normalisation's epsilon placement; how value
  heads share key heads; the operand order of the delta-rule update; whether a token's output reads the state after
  its own update; the `1/sqrt(dk)` output scale; whether the gated norm multiplies by `w` or by `1 + w`; the gated
  norm's gate activation (SiLU, not sigmoid).
- Attention: the RoPE frequencies for text positions; the half-split pairing of rotated dimensions; the
  `1/sqrt(head_dim)` score scale; how query heads share key-value heads.

These overlap the rival formulas of tier 1 below, and R1 settles each against O1.

| Tier | What | Pass condition |
|---|---|---|
| 1 | Per-kernel FP64 transcriptions | The kernel matches within tolerance, and every rival formula differs from the reference by more than 100x that tolerance |
| 2 | Per-layer residual ladder | Hidden state after each layer within a per-layer bound against O2 |
| 3 | Snapshot hash | A cache hit and a cache miss give the same bits |
| 4 | End-to-end gates | The table below, on the golden sets |

Rival formulas for tier 1: decay applied before vs after the update; norm weight `1 + w` vs `w`; conv tail order;
RoPE layout; the `1/sqrt(dk)` scale; the output-gate activation (section 3.8).

| Profile | max \|dp\| | Flips | Other |
|---|---|---|---|
| fp32 | <= 1e-4 | 0 | none |
| `gpu-f16` | <= 0.03 (p99 <= 0.01) | <= 0.5% | dECE <= 0.002 |
| `cpu-q8` vs its float twin | <= 0.05 | <= 1% | vs the fp32 model: accuracy >= -0.5 pp, dECE <= 0.005 |

These gates are design targets set before the 2026-10-13 decision; the metric (probability difference at the same question) is unchanged by it. The design record sets no row for `gpu-q8w`; provisionally it is held to the `cpu-q8` row until R4 registers its own.
Reference tolerances, measured elsewhere by Kev: fp32 paths agree below 1e-4 (`kev@5e42a7a:tests/test_model.py:70-87`,
`kev@5e42a7a:tests/test_model.py:115-132`); served bf16 differs from fp32 by about 0.03 with about one flip in 300
(`kev@5e42a7a:README.md:409`); one swapped conv kernel moved a bf16 27B by 0.03-0.06 (cause at
`kev@5e42a7a:runs/drift-v1/REPORT.md:14-22`, size at `kev@5e42a7a:runs/drift-v1/REPORT.md:91-92`).

- **Golden sets.** v1 (R0): edge cases (one question with two options, 255 options, a schema that exactly fits and one token over,
  a state cut by 1 token and by many, each limit +/- 1, `noul` with custom criteria, `score` with 10 levels, option
  order permuted in the request, delimiter forgeries in the state, non-ASCII); 500 synthetic alert-shaped requests with shared
  prefixes; SOCDecide development rows converted by `sepal/train/unsloth_convert.py` (Kev's public development rows are no
  longer used); long states. Each set stores fp32 CPU reference outputs from O1a with the kernel-environment block. **A
  tokeniser golden** is part of v1: the token ids and spans of Unsloth's `encode_record` for every request, which the runner's
  encoder must reproduce exactly (section 17).
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
- **Licences.** oneDNN and KleidiAI Apache-2.0; ggml and Strata MIT; Kev Apache-2.0; Clef Apache-2.0 (the head and encoder are implemented from it); **Unsloth's decision modules AGPL-3.0-only: read, never copied (section 19, item 1)**; the CUDA runtime with cuBLAS
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

The tree above is the plan. What exists today under `runner/`: `include/sepal/{config, dtype, tensor, gemm, ops,
planner}.hpp` (C++ headers; the C ABI is not written), `src/{core, model, plan, provider}`, `tools/sepal_info.cpp`,
`tests/` and CMake presets `release` and `asan`, with no third-party dependency. `reference/` and `golden/` do not
exist. The slice was built and tested with MSVC 19.44 on Windows 11 (as `runner/README.md` records; the Release
configuration was rebuilt and its 8 test entries passed on 2026-10-09). It has not been compiled with GCC or Clang,
so Linux, which this section names as first-class, is untested. The FP64 reference path is in Python
(`sepal/refengine/`), not in this tree.

## 16. Stages R0-R6

Revised 2026-10-13. Scope changes are marked; exit criteria that are tokens-per-second thresholds are unchanged and still
derived targets.

| Stage | Scope | Exit criterion (derived thresholds are targets) | Person-weeks (estimate) |
|---|---|---|---|
| R0 | O1a oracle on the pilot artefact and golden set v1 with the tokeniser golden; Unsloth baseline records/s and alerts/s on the 3060; settling experiment (redefined); confirm the four base configs by loading the files; GGUF key spec including the head | Goldens committed; baseline measured; substrate memo; the Unsloth artefact loads in a Transformers process with Cloudflare's loader | 1-2 |
| R1 | GGUF loader, **Sepal encoder reproducing Unsloth's token ids**, FP64 and fp32 paths including the **Clef head**, oneDNN provider, single-request executor (one sequence), ABI + Python | fp32 gate on 0.8B against O1a; tokeniser golden matches on every request; all four geometries load; >= 0.8x llama.cpp CPU flat prefill | 4-5 |
| **R2** | cuBLAS provider; Strata GEMM / DeltaNet / graph / device adapted; attention; head on GPU | `gpu-f16` gate; 3060 >= 3,500 tokens/s on 4B and >= 10k tokens/s on 0.8B (derived). **First publishable number on the RTX 3060** | 4 |
| R3 | Prefix row + tail row, state-prefix snapshots with head memory, caches, tiers, scheduler, planner (Clef head sizes), server | Hit = miss bits; 3060 4B >= 2.5 / >= 3.6 alerts/s without / with prefix on the redefined reference alert (derived: the old 4 / 7 scaled by 0.623 and 0.513, section 12) | 5 |
| R4 | Quantiser, `cpu-q8`, KleidiAI, `gpu-q8w`, fp16 state, int8 K/V, calibration tooling (needs a trained adapter; the 0.8B pilot adapter exists but is not a round result) | Task gates; laptop >= 150 tokens/s on 0.8B; 4B resident <= 6.5 GB | 4-5 |
| R5 | HIP via shim + hipBLAS; Vulkan only if an iGPU measures >= 2x its CPU | Gates on one Radeon; >= 0.6x same-class CUDA | 3-4 |
| R6 | Router, supervisor, manifests | 4 GPUs >= 3.6x; prefix hit >= 70% on a trace; worker kill recovered < 2 s | 3 |

The old R3 target "64 questions on a 4.8k state <= weights + 1.5 GB" is dropped: with questions in one sequence the working
memory grows with the total sequence, which the planner bounds by `max_length`.

**Status on 2026-10-13.** No stage has met its exit criterion and the estimates are unchanged; the runner is paused.

- **R0: not started.** No baseline, golden set, settling experiment or substrate memo exists.
- **R1: started, on the old contract.** Written and unit-tested: the build, the geometry presets with validation and derived
  sizes, the element types, the reference provider for `gemm_f`, RMSNorm, SiLU, the SwiGLU MLP, softmax, the causal convolution
  with a carried tail, and the planner. Not written: the GGUF loader, the encoder, the DeltaNet recurrence, attention, the fp32
  executor, the oneDNN provider, the C ABI, the Python wheel, and now the Clef head. All of these except the head are
  independent of the decision. The planner's pointer-head sizing is superseded (section 8).
- **Outside the stage list.** The FP64 reference of R1's scope was written in Python (`sepal/refengine/`) with the old readouts, and the
  token layouts of the old design as `sepal/model/layouts.py` (removed 2026-10-14; it is in git history); neither is the new contract. The C++ encoder must instead
  reproduce `encode_record` token for token (section 17).

## 17. First two weeks

For a future builder; nothing here has been started.

1. **Secure an artefact.** `runs/pilot-unsloth/U1/final` is git-ignored and exists on one machine. Copy it to a stable place,
   record the sha256 of each file (section 3.6), and note the base snapshot hash `dc7cdfe2` from `unsloth_decision_config.json`.
   Rewrite the local path in `adapter_config.json` before sharing.
2. **Build the oracle O1a.** In a Transformers process, produce a merged 16-bit folder (`save_pretrained_merged`) from the adapter,
   or merge by hand with Kev's fp32 rule, and run Cloudflare's `joint_schema_model.py` on it. Check it against Unsloth's own
   `predict` on a handful of requests (they should agree to rounding).
3. **Build golden set v1 and the tokeniser golden** (section 14): requests, `encode_record` ids and spans, and fp32 outputs, with
   the environment block.
4. **Measure the baseline.** Unsloth's forward on the RTX 3060 (pilot: 111 s for 900 records, section 12): records/s,
   tokens/s and peak memory on the reference alert and on the real record-length mix.
5. **Run the settling experiment** of section 5, as redefined, against a pinned llama.cpp build on the laptop CPU and the 3060.
6. **Write the runner's encoder** and compare its ids to the tokeniser golden on at least 10,000 real requests before any other
   code depends on it. The piecewise tokenisation (section 3.2) is the main trap.
7. **Confirm the geometry** of section 3.8 by loading the four base model files; write the `sepal.*` GGUF key specification,
   including the head tensors and their config.
8. **Write the substrate memo**: the decision, the measured numbers, and revised throughput targets that replace the derived
   figures of section 12. Settle or record the open questions of section 19.

## 18. Risks and the experiments that resolve them

| Risk | Experiment | Stage | If it goes badly |
|---|---|---|---|
| Chunked DeltaNet is slower on small GPUs (measured elsewhere: 0.73x on an RTX 3060) | Time both variants per device class | R2 | Sequential stays the default; chunked is selected by `caps()` |
| fp16 recurrent state is untested at 0.8B / 2B | Tier 4 gate with fp16 state on both sizes | R4 | State stays fp32; fewer snapshots per GB |
| int8 on `ssm_out` and `attn_*` costs accuracy | Per-tensor-class ablation against the `cpu-q8` gate | R4 | Those classes stay fp16 |
| Windows launch cost (measured elsewhere: 28-31 us per launch on the legacy default stream under WDDM, 3.63 us on a real stream, 0.805 us inside a captured graph; `Strata@fb58e0d:src/program/generate.cpp:2788-2794`) | Launches per pass and time per launch on the 3060 under Windows | R2 | Own stream always; graphs per `T` bucket; fuse small kernels |
| Unmerged LoRA overhead | Unmerged vs merged tokens/s on one adapter | R3 | Offer `merged-...` artefacts for single-adapter hosts |
| The snapshot trie does not pay at real alert diversity | Replay a trace; record prefix hit rate and bytes held | R3 | Keep the result cache and coalescing; shrink the trie budget |
| The runner's tokeniser differs from `encode_record` (fragment boundaries, special tokens, JSON rendering) | Tokeniser golden on 10,000 real requests (section 17) | R0, R1 | Fix the encoder; never compensate downstream. (Replaces the old risk on segment-wise tokenisation.) |
| cuBLAS picks kernels by shape, breaking batch invariance | Same row in different batch sizes; compare bits | R2 | Fixed `T` buckets or an own GEMM kernel; tier 3 relaxed only with the provider flagged |
| The standard adapter does not absorb activation quantisation | Profile gates with the standard adapter; then the go / no-go experiment in [MODEL.md](MODEL.md) (adapter trained against the dequantised base) | R4 | Float activations on the affected classes |
| fp16 results drift across GPU families | Golden set on two vendors and two generations | R2, R5 | Calibration per kernel set, as section 11 already requires |
| A long sequence does not fit in 8 GB | Planner arithmetic, then a real run | R3 | The planner lowers the admitted sequence length on that device |
| The head's memory and cross-attention over every position dominate for long sequences | Time the head stage at 4k and 16k tokens | R2 | Cache the per-layer memory K/V with the snapshot; run the head on the device |
| fp16 activation overflow in the DeltaNet (Unsloth reports NaN in pure fp16) | Tier 4 gate on the trained model at `gpu-f16`; fp32 activations as fallback | R2 | Float32 path for the DeltaNet block on that profile |
| Questions interact (order and set change other answers) | Golden pairs: same request with a question added, removed or reordered; measure probability change | R0 | Document the interaction as part of the contract; fix order in the API; see section 19 |

## 19. Open questions for the builder

Each is open; none is decided by this document.

1. **Licence of the code the runner may use (release question, left for later by the owner).** Facts, read on 2026-10-13:
   - Cloudflare's `joint_schema_model.py` (the encoder, the head, `systemone`) is Apache-2.0: the Hub repositories
     `Cloudflare/clef` and `Cloudflare/clef-flash` say Apache-2.0 and carry a `LICENSE` file (REFERENCES.md section 2.6; the Clef page
     was re-read on 2026-10-13); Unsloth vendors it **byte for byte** with that `LICENSE` and a manifest naming the upstream
     revision `6a3ac5fc030a3f33f7fa860a1271886b64a6606d` and the sha256 of both files
     (`unsloth@2026.10.4:unsloth/_vendor/clef/clef_manifest.json`, `README.md`, `LICENSE`). The vendored `.py` itself has no
     SPDX header; its licence is the `LICENSE` beside it.
   - Unsloth's own decision modules carry `# SPDX-License-Identifier: AGPL-3.0-only` and "Copyright 2023-present Daniel
     Han-Chen & the Unsloth team": `unsloth/models/decision.py:1`, `decision_from_lm.py:1`, `_decision_common.py:1`,
     `_decision_fast.py:1`, `clef.py:1`. `clef.py:4-7` adds that portions are derived from Cloudflare's file (Apache-2.0)
     and sit under the AGPL header of that module.
   - The package metadata of Unsloth 2026.10.4 declares `License-Expression: Apache-2.0` and ships both `LICENSE` (Apache-2.0)
     and `COPYING` (GNU AGPL v3) (`unsloth_2026.10.4.dist-info`). The upstream README says the core package is Apache-2.0 and
     optional components such as Studio are AGPL-3.0 (read on 2026-10-13 from github.com/unslothai/unsloth). The two readings
     do not agree on the decision modules: the file headers say AGPL, the package says Apache. Which governs is not settled here.
   - Consequence now: an Apache-2.0 runner **must not copy** Unsloth's AGPL files (including the Triton/PyTorch fast head
     forward in `clef.py` and the temperature and calibration code). Implement the head and encoder from Cloudflare's
     Apache-2.0 source or from section 3. Using Unsloth to train and as an oracle process does not copy its code into the
     runner; the legal reading for the trained weights and for any hosted use is not decided (needs a lawyer; deferred until before release).
2. **Option-order sensitivity.** `choice` options are sorted by key in the prompt, so the order in the request never reaches the
   model. Whether the model is sensitive to the *key names* and to their sorted position is untested; our own training
   shuffled option order and gated on the flip rate ([MODEL.md](MODEL.md)), which Unsloth's trainer does not do
   (`reports/pilot-unsloth/REPORT.md:53`).
3. **Question isolation is lost.** Questions share one sequence; later questions see earlier ones in the backbone and fields
   attend to each other in the head (section 3.4, step 5). Adding, removing or reordering a question can change other answers. The
   old design's isolated branches would have measured and bounded this; arm A3's gate was meant to. Nothing measures it now.
   Training ran with `permute_fields` off by default (`unsloth@2026.10.4:unsloth/models/decision.py:233`, `:1093-1094`), so the
   pilot always saw the dataset's field order (not verified that the pilot's call left the default): order sensitivity is plausible.
   The API fixes the order as given by the caller.
4. **Truncation policy against the 65,536-token API limit.** Unsloth cuts the end of the state silently to fit `max_length`
   and trains at 4,096-6,144, serves at 16,384. The Kev-derived API admitted 65,536 state tokens. Which limit the runner
   admits, whether it rejects or cuts, and whether cuts are reported (`truncated`, section 10) is open. The model has
   never seen a state past its training length; its quality there is unverified.
5. **Head training path.** Cloudflare trained the head together with the adapter (unverified, REFERENCES.md); the pilot trained it
   from scratch with the LoRA. The 2B / 4B / 9B head sizes in section 3.4 are the default rule, not a decision.
6. **Merged versus unmerged serving.** The pilot saved an adapter, not a merged model. A merged folder has not been produced
   and loaded with Cloudflare's loader (section 3.6). Merging bf16 weights costs a rounding that the unmerged path does not.
7. **Temperature.** None is fitted yet. Unsloth fits per-type and per-option-count temperatures on a held-out set and clamps them
   to [0.5, 5.0]; ours is fitted per (adapter, profile, kernel set). The two must not be applied twice, and the type-order
   trap of section 3.5 applies.
8. **Delimiter forgery.** Caller text is tokenised as it comes, so `<|im_end|>` or `<|im_start|>` in a state becomes the real
   special token, in training as in serving. Neutralising it (as Kev's `user_tokens` did) changes tokens relative to training.
   Whether to neutralise in the runner and in the data together is a security decision for a SOC product.
9. **fp16 and the DeltaNet** (section 11): whether `gpu-f16` is safe on the trained model.
10. **Hidden-state cost across the boundary.** The head needs all positions; whether it runs inside the runner (planned) or
    outside is a build decision that affects the llama.cpp fork option.
11. **Where Unsloth's code is wrong or unstable.** It was read at a single version and a pilot used one machine; the
    format could change between releases. Pin the version and re-read `encode_record` on every upgrade (section 17, item 6).

## 20. Superseded design (before 2026-10-13)

Kept for the record. Do not build from this section: the model it describes is not the model that will be trained. Where a
trunk fact appears it is repeated, unchanged, in section 3.8. Citations in this section keep their original form.

### 20.1 Request algorithm and pointer head

Kev's `encode`, `rows_of`, `PointerHead` (`kev@5e42a7a:kev/model.py:87-127`, `kev@5e42a7a:kev/model.py:188-202`,
`kev@5e42a7a:kev/model.py:205-223`). The delimiters were five reused Qwen special tokens (`kev@5e42a7a:kev/model.py:9-11`);
their ids, read from the base model files, are `<state>` 248060, `<q>` 248061, `<opt>` 248049, `</opt>` 248050 and
`<decide>` 248062.

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

`k` and `q` are the pointer head's two projections. Each is a plain linear layer with a bias term:
`k(h) = W_k h + b_k` and `q(h) = W_q h + b_q`, with `W` of shape `[256, hidden_size]` and `b` of length 256
([MODEL.md](MODEL.md), section 3). Unlike the trunk's projections they are not bias-free, and the reference
implementation defines them so (`sepal/refengine/readout.py`, `sepal/refengine/weights.py`). The head therefore has
`2 x (256 x hidden_size + 256)` parameters.

`user_tokens` rewrites `<|name|>` so caller text cannot produce a delimiter token (`kev@5e42a7a:kev/model.py:78-81`).
Sepal tokenises state segments independently, which Kev does not; training must do the same ([DATA.md](DATA.md)).

Never loaded (old text): "A prefill-only engine has no vocabulary projection ... at 9B it is a separate matrix (about 2.0 GB
at bf16, derived) that is never read. The label-row readout of section 6 needs at most 255 rows, stored in the adapter."
The last sentence of the second part is superseded by section 3.7.

### 20.2 Old readouts and layouts

[MODEL.md](MODEL.md) tested two readouts and three layouts. One row DAG served them all.

| Readout | Gather | Logits |
|---|---|---|
| Pointer head | `H` at each `</opt>` and at `<decide>` | `z_j = k(h_opt_j) . q(h_decide) / sqrt(256) / T`, fp32; `k` and `q` are linear layers with bias (section 3) |
| Label-row slots | `H` at each answer slot | `h` times the LM-head rows of the question's label tokens (<= 255 rows), divided by `T`, fp32 (label-token pattern: `AnyJev@d458d13:anyjev/readout.py:24-48`) |

| Layout | Rows | What is cacheable |
|---|---|---|
| Isolated branches after the state | prefix row -> tail row -> one row per question | State prefixes |
| All questions in one sequence | prefix row -> tail row -> one row holding every question | State prefixes |
| Pack-first | pack row (pinned root) -> state rows -> short answer-slot rows | The pack, then state prefixes under it |

The old idea 1 of section 6 read: "A request becomes rows: a prefix row, the tail of the state, then branches ... three
waves per layer always suffice (`wave_begin[4]`): a prefix row shared by several requests in the batch; each request's own tail
row, continuing from it or from a cached snapshot; the branches." Last-layer pruning: "State rows compute only `k` and `v` in
the last layer. Branch rows compute attention, MLP and final norm only at readout positions."

### 20.3 Old caches

| Level (cheapest first) | Key | A hit costs | Notes |
|---|---|---|---|
| Result cache | Hash of (raw state bytes, pack, adapter, profile, calibration) | One hash; no tokenising | About 200 bytes per entry (derived); bounded by bytes |
| In-flight coalescing | Same key | Nothing; the request attaches to the running one | Removes duplicate storms before the first answer exists |
| Snapshot trie | `H(parent.key, segment ids) + base_sha + adapter + profile` | The uncached suffix and the branches | Nodes only at caller-declared segment ends |

The old table also listed a pack-first pack row as a pinned root, and "Branch rows never publish; a pack-first pack row does".
Old snapshot sizes for a 600-token state (4B: about 72 MB; 0.8B / 2B: about 28 MB) are kept in section 7, with the head
memory added.

### 20.4 Old throughput table

Reference alert: a 600-token state of which 400 tokens are a shared prefix, with 8 questions of 50 tokens: 1,000 tokens without a
cache, 600 with the prefix cache. Branches were 40-67% of the tokens. Alerts per second, no cache / prefix cache (derived):

| Hardware | 0.8B | 2B | 4B | 9B |
|---|---|---|---|---|
| 4-core laptop CPU | 0.15-0.28 / 0.25-0.47 | 0.06-0.10 / 0.09-0.18 | 0.02-0.04 / 0.035-0.065 | 0.011-0.020 / 0.018-0.034 |
| 8-core desktop CPU | 0.4-0.7 / 0.7-1.2 | 0.15-0.26 / 0.25-0.43 | 0.06-0.10 / 0.09-0.17 | 0.029-0.051 / 0.048-0.084 |
| 8 GB GPU | 10-20 / 17-33 | 4-8 / 7-13 | 3.2-4.3 / 5.3-7.2 | not given |
| 24 GB GPU | 30-100 / 50-167 | 12-40 / 20-67 | 11-29 / 18-48 | 5.6-15 / 9.4-25 |
| 4 x 24 GB | 114-380 / 190-630 | 46-152 / 76-253 | 42-110 / 70-184 | 21-57 / 36-95 |

### 20.5 Old stages and checklist

R0 was "Kev serve fork under WSL2; golden set v1; settling experiment; confirm the four base configs by loading the files; GGUF
key spec". R1 was "GGUF loader, encoder, FP64 and fp32 paths, oneDNN provider, single-request executor, ABI + Python". R3 included
"64 questions on a 4.8k state <= weights + 1.5 GB". The old first-two-weeks checklist served released Kev-0.8B through Kev's server for a
baseline, built golden set v1 from Kev's public development rows (2,000 questions) and synthetic requests, ran the settling
experiment, confirmed the geometry and wrote the substrate memo.

### 20.6 Old API extension

`state` as `[{text, cache, key}]` (a list of segments; superseded because a list is also a valid JSON `state`), `pack`, `deadline_ms`,
`priority`, `model: "sepal-4b@<adapter>"`, response block `sepal {profile, kernel_env, calibration, prefix_tokens_reused, result_cache}`;
the example response used a pack named `triage-v1` and a 652-token usage figure.
