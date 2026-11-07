// The memory planner (docs/RUNNER.md section 8). It refuses instead of overcommitting: a plan is an arithmetic
// result of the geometry, the precision profile, the token budget and the bytes available, and when the parts
// do not fit the caller gets a refusal that names the shortfall and the knob to reduce, never a smaller plan
// it did not ask for. The refusing-planner pattern follows Strata's include/strata/plan/plan.hpp; no code is
// taken from it.
//
// The working-buffer figure is an ESTIMATE derived from the geometry (see working_bytes). Nothing here was
// measured; stage R3 replaces it with the executor's real allocations.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "sepal/config.hpp"
#include "sepal/dtype.hpp"

namespace sepal {

/// Storage precision per tensor class (docs/RUNNER.md section 11).
struct PrecisionProfile {
    std::string id;
    DType embedding = DType::F16;       // host memory unless PlanRequest::embedding_on_device
    DType mlp = DType::F16;             // gate, up, down
    DType mixer_proj = DType::F16;      // attention q/k/v/o; DeltaNet in_proj_qkv and in_proj_z
    DType dn_sensitive = DType::F16;    // DeltaNet out_proj, in_proj_a, in_proj_b (out_proj only unless dn_out_proj is set)
    DType small = DType::F32;           // norms, conv, A_log, dt_bias
    DType recurrent_state = DType::F32; // S in a stored snapshot
    DType conv_tail = DType::F32;       // conv tail in a stored snapshot
    DType kv = DType::F16;              // attention K and V
    DType head = DType::F32;            // pointer head
    /// When set, the DeltaNet out_proj has this type and dn_sensitive covers in_proj_a and in_proj_b alone
    /// (gpu-w8g: out_proj is int8 with the other mixer projections, the two small gate projections stay bfloat16).
    std::optional<DType> dn_out_proj;
    /// 0: no scales are counted (per-channel scales are below 0.1% of an 8-bit matrix). N > 0: every int8 matrix carries
    /// one float32 scale per row and group of N input values, and trunk_weight_bytes counts them (3.1% at N = 128).
    std::uint32_t scale_group = 0;
};

/// "w16" and "w8": every trunk weight at 2 or 1 bytes, the arithmetic behind the design's weight figures.
/// "gpu-f16", "gpu-q8w", "cpu-q8": the profiles of docs/RUNNER.md section 11, where some classes stay wider.
/// "gpu-bf16", "gpu-w8g": the float32 arithmetic profiles of the CUDA library for cards without tensor cores
/// (Pascal, compute capability 6.x, and later). gpu-bf16 stores the matrices as bfloat16; gpu-w8g stores the MLP and
/// the mixer input and output projections as int8 with a float32 scale per 128 input values (DeltaNet in_proj_a /
/// in_proj_b stay bfloat16). Both keep the embedding table in host memory and hold K/V and the recurrent state in
/// float32.
const std::vector<PrecisionProfile>& precision_profiles();
std::optional<PrecisionProfile> find_profile(std::string_view id);

/// Bytes of the trunk weights (every non-embedding parameter) under a profile.
/// Per-channel quantisation scales are not counted; they are below 0.1% of an 8-bit matrix. Per-group scales
/// (PrecisionProfile::scale_group > 0) are counted, rows x ceil(input values / group) float32 values per int8 matrix.
std::uint64_t trunk_weight_bytes(const ModelConfig& cfg, const PrecisionProfile& profile);

/// Bytes one stored snapshot of `state_tokens` tokens occupies: K/V for every token, plus the recurrent state
/// and the conv tails of every DeltaNet layer.
std::uint64_t snapshot_bytes(const ModelConfig& cfg, const PrecisionProfile& profile, std::uint64_t state_tokens);

/// Estimated working memory of one layer-interleaved pass over `tokens` tokens in `rows` rows. Monotone in both.
///
///   activations   tokens x 4 bytes x (2 H + widest GEMM output + I + widest mixer output)
///                 residual stream, normed input, the widest of {gate+up, DeltaNet in-proj, attention in-proj},
///                 the MLP product, and the mixer output before its out-projection
///   in-pass K/V   tokens x kv_bytes_per_token at the profile's K/V precision
///   score tile    attn_heads x min(attention_tile, tokens) x tokens x 4 bytes
///   row state     rows x one DeltaNet layer of float32 state and conv tail (a pass holds one layer at a time;
///                 the all-layer copy exists only in published snapshots, which live in the snapshot arena)
std::uint64_t working_bytes(const ModelConfig& cfg, const PrecisionProfile& profile, std::uint64_t tokens,
                            std::uint64_t rows, std::uint64_t attention_tile);

struct PlanRequest {
    ModelConfig config;
    PrecisionProfile profile;
    std::uint64_t available_bytes = 0;      // device memory the runner may use
    std::uint64_t reserve_bytes = 0;        // held back for the driver, the OS or other processes
    std::uint64_t token_budget = 0;         // largest total token count T of one pass
    std::uint64_t max_rows = 32;            // largest number of rows in one pass
    std::uint64_t attention_tile = 256;     // query tokens scored against the context at once
    std::uint64_t min_snapshots = 1;        // the snapshot arena must hold at least this many snapshots ...
    std::uint64_t min_snapshot_tokens = 0;  // ... of this many state tokens each
    bool embedding_on_device = false;       // true when the embedding table shares the budget (CPU hosts)
};

struct Plan {
    std::uint64_t available_bytes = 0;
    std::uint64_t reserve_bytes = 0;
    std::uint64_t weights_bytes = 0;        // trunk weights plus the pointer head
    std::uint64_t embedding_bytes = 0;      // counted against the budget only if embedding_on_device
    std::uint64_t embedding_host_bytes = 0; // otherwise it is reported here
    std::uint64_t working_bytes = 0;        // at token_budget
    std::uint64_t snapshot_arena_bytes = 0; // everything that remains
    std::uint64_t token_budget = 0;
    std::uint64_t max_token_budget = 0;     // largest token budget this request would still be granted
    std::uint64_t snapshot_bytes_each = 0;  // one snapshot of min_snapshot_tokens
    std::uint64_t snapshots_held = 0;       // snapshot_arena_bytes / snapshot_bytes_each

    std::uint64_t committed_bytes() const {
        return reserve_bytes + weights_bytes + embedding_bytes + working_bytes + snapshot_arena_bytes;
    }
};

enum class Knob {
    WeightPrecision, // the weights alone do not fit, and 8-bit weights would
    ModelSize,       // the weights do not fit at any supported precision
    TokenBudget,     // weights fit; the working buffers at this token budget do not
    SnapshotBudget,  // weights and working buffers fit; the minimum snapshot arena does not
};

std::string_view name_of(Knob knob);

struct Refusal {
    Knob knob = Knob::ModelSize;
    std::uint64_t required_bytes = 0;  // what the refused part needs in total, reserve included
    std::uint64_t available_bytes = 0;
    std::uint64_t shortfall_bytes = 0; // required_bytes - available_bytes
    /// TokenBudget: the largest token budget that fits (never 0). SnapshotBudget: the largest
    /// min_snapshot_tokens that fits at the requested min_snapshots (may be 0). Otherwise 0.
    std::uint64_t suggested_value = 0;
    std::string message; // one sentence: what does not fit, by how much, and what to reduce
};

using PlanResult = std::variant<Plan, Refusal>;

/// Throws std::invalid_argument for a malformed request (invalid geometry, zero token budget, zero rows,
/// zero attention tile, or a reserve that is not below the available bytes). Everything else is a Plan or a
/// Refusal.
PlanResult make_plan(const PlanRequest& request);

/// "7.14 GB (6.65 GiB)". Decimal first, because that is how the design documents quote sizes.
std::string format_bytes(std::uint64_t bytes);

} // namespace sepal
