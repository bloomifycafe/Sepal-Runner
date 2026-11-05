# LONG-LOG parity: the gate is failed by the reference's bf16 error, not by the runner

Date 2026-11-05. Model: Sepal-4B (F15-s0: LoRA r64 + Clef head on Qwen3.5-4B). Panel: LONG-LOG (our own), the 20-record seeded sample
(`runner-rescore`) that failed the `cpu-bf16` gate (max |dp| 0.049, p99 0.030, 1 flip in 262 questions) on every CPU path, `cpu-f32` included.

## Question: which side is closer to the truth?

The reference rows (`eval/rows-LONG-LOG.json`) are Unsloth's forward under bf16 autocast, so they carry bf16 rounding of their own. The truth
was rebuilt as a PyTorch fp32 forward without autocast (`sepal_runner/oracle.py` `Fp32Oracle`: transformers 5.19, PEFT adapter unmerged,
Unsloth's vendored Clef head in fp32, CPU, SDPA causal attention, transformers' reference gated delta rule), on the same tokens: the vendored
encoder reproduces the runner encoder's token ids and spans exactly on all 6 records re-encoded (`ids_same`, `spans_same`).
Max |dp| at temperature 1 / max |dlogit| / flips, for the 5 worst records of the sample by |dp| that fit the box's 27 GB (the two
24,576-token records, ll-x-32k-d4-00003 and ll-x-32k-d2-00028, are the 2nd and 3rd worst; the torch fp32 forward on them was killed by
the box's memory, so they are not covered) plus one passing record:

| record (tokens) | bf16 reference vs fp32 | runner cpu-bf16 vs fp32 | runner cpu-f32 vs fp32 | runner cpu-bf16 vs bf16 reference (the gate) |
|---|---|---|---|---|
| ll-o-4k-d4-00017 (6,582) | 0.0480 / 0.180 / 1 flip | 0.0317 / 0.133 / 0 | 1.7e-5 / 1.0e-4 / 0 | 0.0488 / 0.307 / 1 |
| ll-o-16k-d2-00017 (18,646) | 0.0263 / 0.173 / 0 | 0.0068 / 0.057 / 0 | 2.9e-5 / 2.5e-4 / 0 | 0.0251 / 0.152 / 0 |
| ll-i-2k-d2-00058 (5,207) | 0.0253 / 0.129 / 0 | 0.0074 / 0.032 / 0 | 1.1e-5 / 5.3e-5 / 0 | 0.0187 / 0.160 / 0 |
| ll-i-2k-d2-00006 (4,901) | 0.0141 / 0.105 / 0 | 0.0079 / 0.029 / 0 | 2.8e-5 / 9.4e-5 / 0 | 0.0220 / 0.133 / 0 |
| ll-i-4k-d4-00040 (5,366) | 0.0153 / 0.063 / 0 | 0.0034 / 0.019 / 0 | 6.3e-6 / 2.9e-5 / 0 | 0.0181 / 0.081 / 0 |
| ll-i-2k-d2-00009 (4,132) | 0.0063 / 0.193 / 0 | 0.0029 / 0.018 / 0 | 1.6e-6 / 5.7e-5 / 0 | 0.0065 / 0.199 / 0 |

Over the 231 option probabilities (89 questions) of these records (a sample chosen for being the worst, so these figures are pessimistic): bf16 reference vs fp32 max 0.0480, p99 0.0263; runner cpu-bf16 vs fp32 max 0.0317, p99 0.0163.

- The runner's `cpu-f32` equals the fp32 PyTorch oracle to 2.9e-5 in p (gate 1e-4) and 2.5e-4 in logits, with no flip. The runner is correct.
- The runner's `cpu-bf16` is closer to the truth than the reference is in all 6 records, by a factor of 1.5 to 4.5 in max |dp|.
- The one flip (record ll-o-4k-d4-00017, question q8, near-tie between `no` and `insufficient`) is the reference's: against fp32 the reference
  picks the other option, and so does the runner's cpu-f32; the runner scored 9 of 17 right against the reference's 8 of 17.
- The 30 golden cases (up to 31.9k tokens, RUNNER.md section 14) already showed the same on a fully independent check: cpu-f32 5.7e-6, cpu-bf16 6.8e-3.

The `cpu-bf16` gate compares two bf16 computations, each about 0.03 to 0.05 away from the truth on the worst questions, so the difference
exceeds what either one has alone. LONG-LOG fails it where SEC-dev and TOOL do not because its questions are many near-tie yes/no and
choice fields over long evidence, where the probability is most sensitive to a logit error (a logit error of 0.18 moves p by 0.048 at a
near-tie, by under 0.001 at a confident answer).

## Where the error comes from

Not the runner's long-sequence arithmetic. Its fp32 path matches the fp32 oracle equally well at 4k and 18.6k tokens (1.7e-5 and 2.9e-5), so there
is no drift with length in DeltaNet state accumulation, attention softmax, RoPE (positions and frequencies are computed in double) or the conv tail.
The reference's error is not monotone in length either (4.1k: 0.006, 4.9k: 0.014, 5.2k: 0.025, 6.6k: 0.048, 18.6k: 0.026): it depends on how close
to a tie the question is. The input is identical: same token ids, same spans, same 24,576 window (`max_len_score`), same fragments.

## What was changed

Nothing in the runner. Added `tools/runner_fp32_reference.py` (export a seeded sample with token ids and the bf16 rows; score it with the fp32
oracle on any torch box; compare runner logits with both). Gates are unchanged. A parity run that must pass the gates against a bf16 reference
should also be judged against an fp32 reference; for LONG-LOG the fp32 comparison is the one to read (`cpu-bf16` vs fp32: max 0.0317, p99 0.0163 on
the 6 records, `cpu-f32` vs fp32 below the 1e-4 gate).

Raw data: Sepal `runs/box-results/broadwell/fp32ref/`.
