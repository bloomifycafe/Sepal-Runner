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

# Details the Kev reference files do not settle, because Kev calls a library kernel there (flash-linear-attention,
# transformers) whose body is not among them. Each is the reading the reference supports best, isolated in the one
# function named, as (module.function, what was assumed). To flip one, change that function only; the request
# invariants (continuation, isolation, hit = miss) hold for either reading. Check each against the upstream Qwen3.5
# model code.
UNVERIFIED: tuple[tuple[str, str], ...] = (
    ("deltanet.causal_conv", "tap order: the last kernel tap multiplies the current token (cross-correlation), the tail is oldest first, no bias"),
    ("deltanet.l2_normalise", "q and k are divided by sqrt(sum(x^2) + 1e-6): eps inside the root, not max(norm, eps)"),
    ("deltanet.share_key_heads", "value head h reads key head h // (value heads / key heads): repeat-interleave, not tiling"),
    ("deltanet.delta_rule_update", "operand order: the old state is decayed before the update, the write itself is not decayed; state axes are [key dim, value dim]"),
    ("deltanet.delta_rule_scan", "a token's output reads the state after its own update, not before"),
    ("deltanet.delta_rule_output_scale", "the read is scaled by 1 / sqrt(key head dim)"),
    ("deltanet.gated_norm_scale", "the gated norm multiplies by its stored weight w, not by 1 + w like the other norms"),
    ("deltanet.gated_norm_gate", "the gated norm's gate activation is silu(z) ('swish' in the Kev call), not sigmoid(z)"),
    ("attention.rope_angles", "one position axis with frequencies theta ** (-2i / rotary_dim): the sectioned three-axis RoPE reduces to this for text"),
    ("attention.apply_rope", "the first rotary_dim dims of a head rotate, dim i paired with dim i + rotary_dim / 2 (half-split, not interleaved)"),
    ("attention.attention_scale", "scores are scaled by 1 / sqrt(head_dim)"),
    ("attention.grouped_query_attention", "query head h reads kv head h // (heads / kv heads): repeat-interleave, not tiling"),
)

__all__ = [
    "PRESETS", "REAL_PRESETS", "UNVERIFIED", "AttentionState", "DeltaNetState", "ModelConfig", "ParameterCounts", "Row",
    "RowResult", "Snapshot", "Weights", "empty_snapshot", "label_row_probabilities", "label_rows", "parameter_counts",
    "pointer_probabilities", "random_weights", "run_rows", "weight_shapes",
]
