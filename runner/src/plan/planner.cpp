#include "sepal/planner.hpp"

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <utility>

namespace sepal {

namespace {

PrecisionProfile uniform_profile(std::string id, DType weights) {
    PrecisionProfile p;
    p.id = std::move(id);
    p.embedding = weights;
    p.mlp = weights;
    p.mixer_proj = weights;
    p.dn_sensitive = weights;
    p.small = weights;
    return p; // state, conv tail, K/V and head keep their defaults
}

// Token budgets above this are not searched; it is far beyond any batch the design considers.
constexpr std::uint64_t kMaxSearchTokens = std::uint64_t{1} << 26;

// Largest t in [0, kMaxSearchTokens] with fits(t), given fits is monotone (true, then false) and fits(0) holds.
template <typename Fits>
std::uint64_t largest_fitting(Fits fits) {
    std::uint64_t lo = 0;
    std::uint64_t hi = kMaxSearchTokens + 1;
    while (hi - lo > 1) {
        const std::uint64_t mid = lo + (hi - lo) / 2;
        if (fits(mid)) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return lo;
}

} // namespace

const std::vector<PrecisionProfile>& precision_profiles() {
    static const std::vector<PrecisionProfile> all = [] {
        std::vector<PrecisionProfile> v;
        v.push_back(uniform_profile("w16", DType::F16));
        v.push_back(uniform_profile("w8", DType::I8));

        PrecisionProfile f16; // defaults are the gpu-f16 column
        f16.id = "gpu-f16";
        v.push_back(f16);

        PrecisionProfile q8w = f16;
        q8w.id = "gpu-q8w";
        q8w.mlp = DType::I8;
        q8w.mixer_proj = DType::I8;
        v.push_back(q8w);

        PrecisionProfile cpu = q8w;
        cpu.id = "cpu-q8";
        cpu.embedding = DType::I8;
        v.push_back(cpu);
        return v;
    }();
    return all;
}

std::optional<PrecisionProfile> find_profile(std::string_view id) {
    for (const PrecisionProfile& p : precision_profiles()) {
        if (p.id == id) return p;
    }
    return std::nullopt;
}

std::uint64_t trunk_weight_bytes(const ModelConfig& cfg, const PrecisionProfile& profile) {
    const ParamCounts n = count_params(cfg);
    return n.mlp * bytes_of(profile.mlp) + n.mixer_proj * bytes_of(profile.mixer_proj) +
           n.dn_sensitive * bytes_of(profile.dn_sensitive) + n.small * bytes_of(profile.small);
}

std::uint64_t snapshot_bytes(const ModelConfig& cfg, const PrecisionProfile& profile, std::uint64_t state_tokens) {
    return state_tokens * kv_bytes_per_token(cfg, profile.kv) + recurrent_state_bytes(cfg, profile.recurrent_state) +
           conv_tail_bytes(cfg, profile.conv_tail);
}

std::uint64_t working_bytes(const ModelConfig& cfg, const PrecisionProfile& profile, std::uint64_t tokens,
                            std::uint64_t rows, std::uint64_t attention_tile) {
    require_valid(cfg);
    const std::uint64_t f32 = bytes_of(DType::F32);

    const std::uint64_t widest_gemm =
        std::max({2 * cfg.intermediate_size, cfg.dn_in_proj_width(), cfg.attn_in_proj_width()});
    const std::uint64_t widest_mixer_out = std::max(cfg.dn_value_dim(), cfg.attn_width());
    const std::uint64_t floats_per_token =
        2 * cfg.hidden_size + widest_gemm + cfg.intermediate_size + widest_mixer_out;

    const std::uint64_t activations = tokens * floats_per_token * f32;
    const std::uint64_t pass_kv = tokens * kv_bytes_per_token(cfg, profile.kv);
    const std::uint64_t score_tile = cfg.attn_heads * std::min(attention_tile, tokens) * tokens * f32;
    const std::uint64_t one_layer_state =
        (cfg.dn_value_heads * cfg.dn_key_head_dim * cfg.dn_value_head_dim + (cfg.conv_kernel - 1) * cfg.dn_conv_dim()) *
        f32;
    return activations + pass_kv + score_tile + rows * one_layer_state;
}

std::string_view name_of(Knob knob) {
    switch (knob) {
    case Knob::WeightPrecision: return "weight precision";
    case Knob::ModelSize: return "model size";
    case Knob::TokenBudget: return "token budget";
    case Knob::SnapshotBudget: return "snapshot budget";
    }
    return "?";
}

std::string format_bytes(std::uint64_t bytes) {
    char buffer[64];
    const double b = static_cast<double>(bytes);
    std::snprintf(buffer, sizeof buffer, "%.2f GB (%.2f GiB)", b / 1e9, b / (1024.0 * 1024.0 * 1024.0));
    return buffer;
}

PlanResult make_plan(const PlanRequest& request) {
    const ModelConfig& cfg = request.config;
    const PrecisionProfile& profile = request.profile;
    require_valid(cfg);
    if (request.token_budget == 0) throw std::invalid_argument("make_plan: token_budget must be greater than 0");
    if (request.max_rows == 0) throw std::invalid_argument("make_plan: max_rows must be greater than 0");
    if (request.attention_tile == 0) throw std::invalid_argument("make_plan: attention_tile must be greater than 0");
    if (request.reserve_bytes >= request.available_bytes) {
        throw std::invalid_argument("make_plan: reserve_bytes must be below available_bytes");
    }

    const ParamCounts params = count_params(cfg);
    const std::uint64_t usable = request.available_bytes - request.reserve_bytes;
    const std::uint64_t head_bytes = params.pointer_head * bytes_of(profile.head);
    const std::uint64_t weights = trunk_weight_bytes(cfg, profile) + head_bytes;
    const std::uint64_t embedding_all = params.embedding * bytes_of(profile.embedding);
    const std::uint64_t embedding = request.embedding_on_device ? embedding_all : 0;
    const std::uint64_t fixed = weights + embedding;
    const std::uint64_t snapshot_each = snapshot_bytes(cfg, profile, request.min_snapshot_tokens);
    const std::uint64_t snapshot_min = request.min_snapshots * snapshot_each;

    const auto working = [&](std::uint64_t tokens) {
        return working_bytes(cfg, profile, tokens, request.max_rows, request.attention_tile);
    };
    const auto refuse = [&](Knob knob, std::uint64_t required_usable, std::uint64_t suggested, std::string message) {
        Refusal r;
        r.knob = knob;
        r.required_bytes = request.reserve_bytes + required_usable;
        r.available_bytes = request.available_bytes;
        r.shortfall_bytes = r.required_bytes - r.available_bytes;
        r.suggested_value = suggested;
        r.message = std::move(message);
        return PlanResult{r};
    };
    // The weights, with or without one token of working memory, do not fit: say which weight knob helps.
    const auto refuse_weights = [&](std::uint64_t required_usable, const std::string& what) {
        const std::uint64_t weights_8bit = params.non_embedding() * bytes_of(DType::I8) + head_bytes + embedding;
        const std::uint64_t shortfall = required_usable - usable;
        std::string message = what + " need " + format_bytes(required_usable) + " but " + format_bytes(usable) +
                              " are usable, short by " + format_bytes(shortfall) + "; ";
        if (weights_8bit < fixed && weights_8bit + (required_usable - fixed) <= usable) {
            message += "reduce the weight precision: with 8-bit trunk weights the same part needs " +
                       format_bytes(weights_8bit + (required_usable - fixed));
            return refuse(Knob::WeightPrecision, required_usable, 0, std::move(message));
        }
        message += "no weight precision fits, so reduce the model size (choose a smaller preset)";
        return refuse(Knob::ModelSize, required_usable, 0, std::move(message));
    };

    // 1. Weights.
    if (fixed > usable) {
        return refuse_weights(fixed, request.embedding_on_device ? "the weights and the embedding table"
                                                                 : "the weights");
    }

    // 2. Working buffers at the requested token budget.
    const std::uint64_t max_tokens_with_arena =
        largest_fitting([&](std::uint64_t t) { return fixed + working(t) + snapshot_min <= usable; });
    const std::uint64_t working_now = working(request.token_budget);
    if (fixed + working_now > usable) {
        const std::uint64_t max_tokens =
            largest_fitting([&](std::uint64_t t) { return fixed + working(t) <= usable; });
        if (max_tokens == 0) {
            return refuse_weights(fixed + working(1), "the weights and the working buffers of a one-token pass");
        }
        const std::uint64_t suggested = max_tokens_with_arena != 0 ? max_tokens_with_arena : max_tokens;
        const std::uint64_t required = fixed + working_now;
        return refuse(Knob::TokenBudget, required, suggested,
                      "the weights and the working buffers for a token budget of " +
                          std::to_string(request.token_budget) + " need " + format_bytes(required) + " but " +
                          format_bytes(usable) + " are usable, short by " + format_bytes(required - usable) +
                          "; reduce the token budget to " + std::to_string(suggested) + " or less");
    }

    // 3. The minimum snapshot arena.
    const std::uint64_t remaining = usable - fixed - working_now;
    if (snapshot_min > remaining) {
        const std::uint64_t per_snapshot = remaining / request.min_snapshots;
        const std::uint64_t state_part = snapshot_bytes(cfg, profile, 0);
        const std::uint64_t fitting_tokens =
            per_snapshot > state_part ? (per_snapshot - state_part) / kv_bytes_per_token(cfg, profile.kv) : 0;
        const std::uint64_t required = fixed + working_now + snapshot_min;
        return refuse(Knob::SnapshotBudget, required, fitting_tokens,
                      "after the weights and the working buffers, " + format_bytes(remaining) +
                          " remain but " + std::to_string(request.min_snapshots) + " snapshot(s) of " +
                          std::to_string(request.min_snapshot_tokens) + " state tokens need " +
                          format_bytes(snapshot_min) + ", short by " + format_bytes(snapshot_min - remaining) +
                          "; reduce the snapshot budget (at most " + std::to_string(fitting_tokens) +
                          " state tokens per snapshot fit), or the token budget");
    }

    Plan plan;
    plan.available_bytes = request.available_bytes;
    plan.reserve_bytes = request.reserve_bytes;
    plan.weights_bytes = weights;
    plan.embedding_bytes = embedding;
    plan.embedding_host_bytes = request.embedding_on_device ? 0 : embedding_all;
    plan.working_bytes = working_now;
    plan.snapshot_arena_bytes = remaining;
    plan.token_budget = request.token_budget;
    plan.max_token_budget = max_tokens_with_arena;
    plan.snapshot_bytes_each = snapshot_each;
    plan.snapshots_held = remaining / snapshot_each;
    return PlanResult{plan};
}

} // namespace sepal
