// Model geometry as data. Nothing in the engine may hard-wire a size: every dimension is read from a
// ModelConfig, and each size is stated on its own, never derived from another size (docs/RUNNER.md section 3).
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "sepal/dtype.hpp"

namespace sepal {

struct ModelConfig {
    std::string name;

    // Trunk
    std::uint64_t num_layers = 0;
    std::uint64_t full_attention_interval = 0; // layer i is full attention iff (i + 1) % interval == 0
    std::uint64_t hidden_size = 0;
    std::uint64_t intermediate_size = 0; // SwiGLU MLP width

    // Full-attention layers. attn_heads * attn_head_dim need not equal hidden_size (it does not at 0.8B and 4B).
    std::uint64_t attn_heads = 0;
    std::uint64_t attn_kv_heads = 0;
    std::uint64_t attn_head_dim = 0;

    // Gated DeltaNet layers. Key heads are shared by their value heads.
    std::uint64_t dn_key_heads = 0;
    std::uint64_t dn_value_heads = 0;
    std::uint64_t dn_key_head_dim = 0;
    std::uint64_t dn_value_head_dim = 0;
    std::uint64_t conv_kernel = 0; // taps of the depthwise causal convolution

    std::uint64_t vocab_size = 0;
    double rms_norm_eps = 0.0;
    double rope_theta = 0.0;
    double partial_rotary_factor = 0.0;

    std::uint64_t pointer_dim = 0; // width of the pointer head's key and query projections
    bool lm_head_tied = true;      // false: a separate vocabulary matrix exists in the base files (never loaded)

    // ---- derived layer structure ----
    bool is_full_attention(std::uint64_t layer) const { return (layer + 1) % full_attention_interval == 0; }
    std::uint64_t num_attention_layers() const { return num_layers / full_attention_interval; }
    std::uint64_t num_deltanet_layers() const { return num_layers - num_attention_layers(); }

    // ---- derived widths ----
    std::uint64_t attn_width() const { return attn_heads * attn_head_dim; }       // q (without gate) and o_proj input
    std::uint64_t attn_kv_width() const { return attn_kv_heads * attn_head_dim; } // one of K or V, per token
    std::uint64_t attn_rotary_dims() const;                                       // head dims that RoPE rotates
    std::uint64_t dn_key_dim() const { return dn_key_heads * dn_key_head_dim; }
    std::uint64_t dn_value_dim() const { return dn_value_heads * dn_value_head_dim; }
    std::uint64_t dn_conv_dim() const { return 2 * dn_key_dim() + dn_value_dim(); } // q, k, v channels
    /// Output width of the concatenated DeltaNet in-projection: mixed, z, b, a.
    std::uint64_t dn_in_proj_width() const { return dn_conv_dim() + dn_value_dim() + 2 * dn_value_heads; }
    /// Output width of the concatenated attention in-projection: q with its output gate, k, v.
    std::uint64_t attn_in_proj_width() const { return 2 * attn_width() + 2 * attn_kv_width(); }
};

/// Parameter counts grouped by the tensor classes of the precision profiles (docs/RUNNER.md section 11).
struct ParamCounts {
    std::uint64_t embedding = 0;       // vocab x hidden; stays in host memory
    std::uint64_t mlp = 0;             // gate, up, down
    std::uint64_t mixer_proj = 0;      // attention q/k/v/o; DeltaNet in_proj_qkv and in_proj_z
    std::uint64_t dn_sensitive = 0;    // DeltaNet out_proj, in_proj_a, in_proj_b
    std::uint64_t small = 0;           // norms, conv, A_log, dt_bias
    std::uint64_t pointer_head = 0;    // key and query projections, a weight and a bias each (adapter, not base)
    std::uint64_t separate_lm_head = 0; // vocab x hidden when the head is not tied; never loaded

    /// The trunk: everything a pass reads on the device. Excludes the embedding, the pointer head and the LM head.
    std::uint64_t non_embedding() const { return mlp + mixer_proj + dn_sensitive + small; }
};

/// Empty when the configuration is valid; otherwise one message per violated rule, each naming the field.
std::vector<std::string> validate(const ModelConfig& cfg);

/// Throws std::invalid_argument listing every violation.
void require_valid(const ModelConfig& cfg);

/// The four Qwen3.5 base geometries: "0.8b", "2b", "4b", "9b".
const std::vector<ModelConfig>& presets();
std::optional<ModelConfig> find_preset(std::string_view name);

/// The test geometry of the reference engine (`sepal.refengine.PRESETS["tiny"]`): sizes that coincide in the
/// real models differ here, so a transposed or mis-sized tensor fails loudly. Not a model anyone trained.
ModelConfig tiny_test_config();

// ---- derived quantities (all require a valid configuration) ----

ParamCounts count_params(const ModelConfig& cfg);

/// Bytes of attention K and V appended per state token, over all full-attention layers.
std::uint64_t kv_bytes_per_token(const ModelConfig& cfg, DType precision);

/// Bytes of the DeltaNet recurrent state S[Hv, dk, dv], over all DeltaNet layers.
std::uint64_t recurrent_state_bytes(const ModelConfig& cfg, DType precision);

/// Bytes of the conv tails [K-1, conv_dim], over all DeltaNet layers.
std::uint64_t conv_tail_bytes(const ModelConfig& cfg, DType precision);

/// 2 x non-embedding parameters: one multiply and one add per weight (docs/RUNNER.md section 12).
std::uint64_t flops_per_token(const ModelConfig& cfg);

} // namespace sepal
