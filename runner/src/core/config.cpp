#include "sepal/config.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace sepal {

namespace {

ModelConfig make_preset(std::string name, std::uint64_t layers, std::uint64_t hidden, std::uint64_t mlp,
                        std::uint64_t heads, std::uint64_t kv_heads, std::uint64_t dn_key_heads,
                        std::uint64_t dn_value_heads, bool lm_head_tied) {
    ModelConfig c;
    c.name = std::move(name);
    c.num_layers = layers;
    c.full_attention_interval = 4;
    c.hidden_size = hidden;
    c.intermediate_size = mlp;
    c.attn_heads = heads;
    c.attn_kv_heads = kv_heads;
    c.attn_head_dim = 256;
    c.dn_key_heads = dn_key_heads;
    c.dn_value_heads = dn_value_heads;
    c.dn_key_head_dim = 128;
    c.dn_value_head_dim = 128;
    c.conv_kernel = 4;
    c.vocab_size = 248320;
    c.rms_norm_eps = 1e-6;
    c.rope_theta = 1e7;
    c.partial_rotary_factor = 0.25;
    c.pointer_dim = 256;
    c.lm_head_tied = lm_head_tied;
    return c;
}

// head_dim * partial_rotary_factor, or 0 when that is not a whole number.
std::uint64_t rotary_dims_or_zero(const ModelConfig& cfg) {
    const double dims = static_cast<double>(cfg.attn_head_dim) * cfg.partial_rotary_factor;
    const double rounded = std::round(dims);
    if (!(rounded >= 1.0) || std::fabs(dims - rounded) > 1e-9) return 0;
    return static_cast<std::uint64_t>(rounded);
}

} // namespace

std::uint64_t ModelConfig::attn_rotary_dims() const { return rotary_dims_or_zero(*this); }

std::vector<std::string> validate(const ModelConfig& cfg) {
    std::vector<std::string> errors;
    const auto positive = [&errors](std::uint64_t value, const char* field) {
        if (value == 0) errors.push_back(std::string(field) + " must be greater than 0");
        return value != 0;
    };

    const bool layers_ok = positive(cfg.num_layers, "num_layers");
    const bool interval_ok = positive(cfg.full_attention_interval, "full_attention_interval");
    positive(cfg.hidden_size, "hidden_size");
    positive(cfg.intermediate_size, "intermediate_size");
    const bool heads_ok = positive(cfg.attn_heads, "attn_heads");
    const bool kv_heads_ok = positive(cfg.attn_kv_heads, "attn_kv_heads");
    const bool head_dim_ok = positive(cfg.attn_head_dim, "attn_head_dim");
    const bool key_heads_ok = positive(cfg.dn_key_heads, "dn_key_heads");
    const bool value_heads_ok = positive(cfg.dn_value_heads, "dn_value_heads");
    positive(cfg.dn_key_head_dim, "dn_key_head_dim");
    positive(cfg.dn_value_head_dim, "dn_value_head_dim");
    positive(cfg.conv_kernel, "conv_kernel");
    positive(cfg.vocab_size, "vocab_size");
    positive(cfg.pointer_dim, "pointer_dim");

    if (layers_ok && interval_ok && cfg.num_layers % cfg.full_attention_interval != 0) {
        errors.push_back("num_layers (" + std::to_string(cfg.num_layers) +
                         ") must be a multiple of full_attention_interval (" +
                         std::to_string(cfg.full_attention_interval) +
                         "): the last layer has to be a full-attention layer");
    }
    if (heads_ok && kv_heads_ok && cfg.attn_heads % cfg.attn_kv_heads != 0) {
        errors.push_back("attn_heads (" + std::to_string(cfg.attn_heads) + ") must be a multiple of attn_kv_heads (" +
                         std::to_string(cfg.attn_kv_heads) + ") for grouped-query attention");
    }
    if (key_heads_ok && value_heads_ok && cfg.dn_value_heads % cfg.dn_key_heads != 0) {
        errors.push_back("dn_value_heads (" + std::to_string(cfg.dn_value_heads) +
                         ") must be a multiple of dn_key_heads (" + std::to_string(cfg.dn_key_heads) +
                         "): each key head is shared by a whole number of value heads");
    }
    if (!(cfg.rms_norm_eps > 0.0) || !std::isfinite(cfg.rms_norm_eps)) {
        errors.push_back("rms_norm_eps must be a finite number greater than 0");
    }
    if (!(cfg.rope_theta > 1.0) || !std::isfinite(cfg.rope_theta)) {
        errors.push_back("rope_theta must be a finite number greater than 1");
    }
    const bool factor_ok = cfg.partial_rotary_factor > 0.0 && cfg.partial_rotary_factor <= 1.0;
    if (!factor_ok) {
        errors.push_back("partial_rotary_factor must be in (0, 1]");
    }
    if (factor_ok && head_dim_ok) {
        const std::uint64_t rotary = rotary_dims_or_zero(cfg);
        if (rotary == 0 || rotary % 2 != 0) {
            errors.push_back("attn_head_dim x partial_rotary_factor must be a positive even whole number: "
                             "RoPE rotates dimensions in pairs");
        }
    }
    return errors;
}

void require_valid(const ModelConfig& cfg) {
    const std::vector<std::string> errors = validate(cfg);
    if (errors.empty()) return;
    std::string message = "invalid model configuration '" + cfg.name + "':";
    for (const std::string& e : errors) message += "\n  - " + e;
    throw std::invalid_argument(message);
}

const std::vector<ModelConfig>& presets() {
    // Read from each base model's config.json (docs/RUNNER.md section 3); R0 confirms them by loading the files.
    static const std::vector<ModelConfig> all = {
        make_preset("0.8b", 24, 1024, 3584, 8, 2, 16, 16, true),
        make_preset("2b", 24, 2048, 6144, 8, 2, 16, 16, true),
        make_preset("4b", 32, 2560, 9216, 16, 4, 16, 32, true),
        make_preset("9b", 32, 4096, 12288, 16, 4, 16, 32, false),
    };
    return all;
}

std::optional<ModelConfig> find_preset(std::string_view name) {
    for (const ModelConfig& c : presets()) {
        if (c.name == name) return c;
    }
    return std::nullopt;
}

ParamCounts count_params(const ModelConfig& cfg) {
    require_valid(cfg);
    const std::uint64_t h = cfg.hidden_size;
    const std::uint64_t n_attn = cfg.num_attention_layers();
    const std::uint64_t n_dn = cfg.num_deltanet_layers();

    ParamCounts p;
    p.embedding = cfg.vocab_size * h;
    p.separate_lm_head = cfg.lm_head_tied ? 0 : cfg.vocab_size * h;
    p.pointer_head = 2 * h * cfg.pointer_dim;

    // Every layer: two RMSNorm weights, and the bias-free MLP (gate and up: 2 x [I, H]; down: [H, I]).
    p.mlp = cfg.num_layers * 3 * h * cfg.intermediate_size;
    p.small = cfg.num_layers * 2 * h;

    // Full attention: q_proj carries the output gate, so it is twice the attention width.
    const std::uint64_t attn_proj = h * cfg.attn_in_proj_width() // q (+ gate), k, v
                                    + cfg.attn_width() * h;      // o_proj
    p.mixer_proj += n_attn * attn_proj;
    p.small += n_attn * 2 * cfg.attn_head_dim; // per-head RMSNorm weights of q and k

    // Gated DeltaNet.
    p.mixer_proj += n_dn * (h * cfg.dn_conv_dim() + h * cfg.dn_value_dim()); // in_proj_qkv, in_proj_z
    p.dn_sensitive += n_dn * (cfg.dn_value_dim() * h                         // out_proj
                              + 2 * h * cfg.dn_value_heads);                 // in_proj_b, in_proj_a
    p.small += n_dn * (cfg.dn_conv_dim() * cfg.conv_kernel // depthwise conv, bias-free
                       + 2 * cfg.dn_value_heads            // A_log, dt_bias
                       + cfg.dn_value_head_dim);           // gated output norm weight

    p.small += h; // final norm
    return p;
}

std::uint64_t kv_bytes_per_token(const ModelConfig& cfg, DType precision) {
    require_valid(cfg);
    return cfg.num_attention_layers() * 2 * cfg.attn_kv_width() * bytes_of(precision);
}

std::uint64_t recurrent_state_bytes(const ModelConfig& cfg, DType precision) {
    require_valid(cfg);
    return cfg.num_deltanet_layers() * cfg.dn_value_heads * cfg.dn_key_head_dim * cfg.dn_value_head_dim *
           bytes_of(precision);
}

std::uint64_t conv_tail_bytes(const ModelConfig& cfg, DType precision) {
    require_valid(cfg);
    return cfg.num_deltanet_layers() * (cfg.conv_kernel - 1) * cfg.dn_conv_dim() * bytes_of(precision);
}

std::uint64_t flops_per_token(const ModelConfig& cfg) { return 2 * count_params(cfg).non_embedding(); }

} // namespace sepal
