"""The reference engine: an exact, slow, readable NumPy float64 forward pass of the Sepal decision model.

This is the oracle every optimised backend is tested against (O2 in docs/RUNNER.md section 14). It defines the
model's layers, what a snapshot holds, how a request's rows continue snapshots and each other, and the two readouts.
Everything is driven by a ``ModelConfig``; no size is hard-wired. Clarity beats speed throughout.
"""
from .config import PRESETS, REAL_PRESETS, ModelConfig
from .executor import Row, RowResult, run_rows
from .readout import label_row_probabilities, label_rows, pointer_probabilities
from .snapshot import AttentionState, DeltaNetState, Snapshot, empty_snapshot
from .weights import ParameterCounts, Weights, parameter_counts, random_weights, weight_shapes

# Details of the arithmetic that are assumed and not checked against the upstream model, as (module.function, what
# is assumed). Empty: the twelve that the Kev reference files left open (Kev calls library kernels there) were each
# read in the upstream source and confirmed numerically, and none needed a change. Evidence, method and measured
# differences: docs/REFENGINE-VERIFICATION.md; the tests are tests/refengine/test_refengine_upstream.py.
# Lines are of transformers 5.19.0, transformers/models/qwen3_5/modeling_qwen3_5.py.
UNVERIFIED: tuple[tuple[str, str], ...] = (
    # deltanet.causal_conv: F.conv1d (cross-correlation) left-padded by K-1 and cut to the sequence, so the last tap
    #   multiplies the current token; conv1d has bias=False. L260-279 (cached: L240-256), L512-519.
    # deltanet.l2_normalise: x * rsqrt(sum(x^2) + 1e-6), eps inside the root. L284-287, called at L334-335, L454-455.
    # deltanet.share_key_heads: repeat_interleave(value heads // key heads) over the head axis. L610-612.
    # deltanet.delta_rule_update: the state is decayed, then read with k and written; the write is not decayed; the
    #   state is [value heads, key dim, value dim]. L470-479 (token by token), L402-415 (chunked, same result).
    # deltanet.delta_rule_scan: a token's output reads the state after its own update. L481.
    # deltanet.delta_rule_output_scale: the query is scaled by key head dim ** -0.5. L337-338, L458.
    # deltanet.gated_norm_scale: the gated norm multiplies by its weight itself, initialised to one. L211, L221.
    # deltanet.gated_norm_gate: the gate is silu(z). L213, L222.
    # attention.rope_angles: inv_freq = 1 / theta ** (arange(0, dim, 2) / dim), dim = int(head_dim * partial rotary
    #   factor); the three position axes carry the same value for text, so the sectioned table reduces to one axis.
    #   L170-177, L185-203, L1250-1259.
    # attention.apply_rope: the first rotary_dim dims rotate, paired half-split: x * cos + cat(-x2, x1) * sin with
    #   cos and sin each cat(f, f). L656-660, L688-698, L203.
    # attention.attention_scale: head_dim ** -0.5. L748, L727.
    # attention.grouped_query_attention: each kv head is repeated in place for its group of query heads
    #   (repeat_interleave). L702-711, L724-725.
)

__all__ = [
    "PRESETS", "REAL_PRESETS", "UNVERIFIED", "AttentionState", "DeltaNetState", "ModelConfig", "ParameterCounts", "Row",
    "RowResult", "Snapshot", "Weights", "empty_snapshot", "label_row_probabilities", "label_rows", "parameter_counts",
    "pointer_probabilities", "random_weights", "run_rows", "weight_shapes",
]
