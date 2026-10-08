#include <cstdint>
#include <stdexcept>
#include <string>

#include "sepal/config.hpp"
#include "test.hpp"

using namespace sepal;

namespace {

ModelConfig preset(const char* name) {
    const auto p = find_preset(name);
    if (!p) throw std::runtime_error(std::string("missing preset ") + name);
    return *p;
}

// A small geometry with no relation to the presets, so the formulas are checked by hand and not against themselves.
ModelConfig tiny() {
    ModelConfig c;
    c.name = "tiny";
    c.num_layers = 4;
    c.full_attention_interval = 4;
    c.hidden_size = 8;
    c.intermediate_size = 16;
    c.attn_heads = 2;
    c.attn_kv_heads = 1;
    c.attn_head_dim = 4;
    c.dn_key_heads = 1;
    c.dn_value_heads = 2;
    c.dn_key_head_dim = 2;
    c.dn_value_head_dim = 3;
    c.conv_kernel = 4;
    c.vocab_size = 10;
    c.rms_norm_eps = 1e-6;
    c.rope_theta = 1e7;
    c.partial_rotary_factor = 0.5;
    c.pointer_dim = 4;
    return c;
}

bool mentions(const std::vector<std::string>& errors, const char* field) {
    for (const std::string& e : errors) {
        if (e.find(field) != std::string::npos) return true;
    }
    return false;
}

std::uint64_t rounded(std::uint64_t value, std::uint64_t unit) { return (value + unit / 2) / unit; }

} // namespace

SEPAL_TEST(config, presets_hold_the_design_table) {
    CHECK_EQ(presets().size(), 4u);
    struct Row {
        const char* name;
        std::uint64_t layers, hidden, mlp, heads, kv_heads, dn_key_heads, dn_value_heads, dn_layers, attn_layers;
    };
    const Row rows[] = {
        {"0.8b", 24, 1024, 3584, 8, 2, 16, 16, 18, 6},
        {"2b", 24, 2048, 6144, 8, 2, 16, 16, 18, 6},
        {"4b", 32, 2560, 9216, 16, 4, 16, 32, 24, 8},
        {"9b", 32, 4096, 12288, 16, 4, 16, 32, 24, 8},
    };
    for (const Row& r : rows) {
        const ModelConfig c = preset(r.name);
        CHECK(validate(c).empty());
        CHECK_EQ(c.num_layers, r.layers);
        CHECK_EQ(c.hidden_size, r.hidden);
        CHECK_EQ(c.intermediate_size, r.mlp);
        CHECK_EQ(c.attn_heads, r.heads);
        CHECK_EQ(c.attn_kv_heads, r.kv_heads);
        CHECK_EQ(c.attn_head_dim, 256u);
        CHECK_EQ(c.dn_key_heads, r.dn_key_heads);
        CHECK_EQ(c.dn_value_heads, r.dn_value_heads);
        CHECK_EQ(c.dn_key_head_dim, 128u);
        CHECK_EQ(c.dn_value_head_dim, 128u);
        CHECK_EQ(c.conv_kernel, 4u);
        CHECK_EQ(c.vocab_size, 248320u);
        CHECK_EQ(c.rms_norm_eps, 1e-6);
        CHECK_EQ(c.rope_theta, 1e7);
        CHECK_EQ(c.partial_rotary_factor, 0.25);
        CHECK_EQ(c.full_attention_interval, 4u);
        CHECK_EQ(c.num_deltanet_layers(), r.dn_layers);
        CHECK_EQ(c.num_attention_layers(), r.attn_layers);
        CHECK_EQ(c.attn_rotary_dims(), 64u); // 64 of the 256 head dims rotate
    }
    CHECK(find_preset("4b").has_value());
    CHECK(!find_preset("27b").has_value());
}

SEPAL_TEST(config, layer_rule) {
    const ModelConfig c = preset("0.8b");
    std::uint64_t attention = 0;
    for (std::uint64_t i = 0; i < c.num_layers; ++i) {
        CHECK_EQ(c.is_full_attention(i), (i + 1) % 4 == 0);
        if (c.is_full_attention(i)) ++attention;
    }
    CHECK_EQ(attention, c.num_attention_layers());
    CHECK(!c.is_full_attention(0));
    CHECK(c.is_full_attention(3));
    CHECK(c.is_full_attention(c.num_layers - 1)); // the last layer is always full attention
}

SEPAL_TEST(config, loader_traps_are_not_assumed_away) {
    // head_dim x heads differs from hidden_size at 0.8B and 4B, so the attention width is its own quantity.
    CHECK(preset("0.8b").attn_width() != preset("0.8b").hidden_size);
    CHECK(preset("4b").attn_width() != preset("4b").hidden_size);
    CHECK_EQ(preset("2b").attn_width(), preset("2b").hidden_size);
    // q_proj carries the output gate: 2 x attention width, plus k and v.
    CHECK_EQ(preset("0.8b").attn_in_proj_width(), 2u * 2048u + 2u * 512u);
    CHECK_EQ(preset("0.8b").dn_conv_dim(), 6144u);
    CHECK_EQ(preset("4b").dn_conv_dim(), 8192u);
    CHECK_EQ(preset("4b").dn_in_proj_width(), 8192u + 4096u + 64u);
}

SEPAL_TEST(config, param_count_by_hand_on_a_tiny_geometry) {
    const ModelConfig c = tiny();
    CHECK(validate(c).empty());
    CHECK_EQ(c.num_deltanet_layers(), 3u);
    CHECK_EQ(c.num_attention_layers(), 1u);
    const ParamCounts p = count_params(c);
    CHECK_EQ(p.mlp, 1536u);         // 4 layers x 3 x 8 x 16
    CHECK_EQ(p.mixer_proj, 640u);   // attention 8 x 24 + 8 x 8 = 256; DeltaNet 3 x (8 x 10 + 8 x 6) = 384
    CHECK_EQ(p.dn_sensitive, 240u); // 3 x (6 x 8 + 2 x 8 x 2)
    CHECK_EQ(p.small, 221u);        // layer norms 64 + q/k norms 8 + 3 x (40 + 4 + 3) + final norm 8
    CHECK_EQ(p.non_embedding(), 2637u);
    CHECK_EQ(p.embedding, 80u);
    CHECK_EQ(p.pointer_head, 64u);
    CHECK_EQ(p.separate_lm_head, 0u);
    CHECK_EQ(flops_per_token(c), 5274u);
    CHECK_EQ(kv_bytes_per_token(c, DType::F16), 16u);     // 1 layer x (K + V) x 4 x 2 bytes
    CHECK_EQ(recurrent_state_bytes(c, DType::F32), 144u); // 3 layers x 2 x 2 x 3 x 4 bytes
    CHECK_EQ(conv_tail_bytes(c, DType::F32), 360u);       // 3 layers x 3 x 10 x 4 bytes
}

SEPAL_TEST(config, param_counts_match_the_design) {
    // 0.8B summed by hand: 24 x (11,010,048 MLP + 2,048 norms) + 6 x 7,340,544 + 18 x 10,543,264 + 1,024.
    CHECK_EQ(count_params(preset("0.8b")).non_embedding(), 498113344u);

    // docs/RUNNER.md section 3: 498M, 1.37B, 3.57B, 6.92B (three significant figures).
    CHECK_EQ(rounded(count_params(preset("0.8b")).non_embedding(), 1000000), 498u);
    CHECK_EQ(rounded(count_params(preset("2b")).non_embedding(), 10000000), 137u);
    CHECK_EQ(rounded(count_params(preset("4b")).non_embedding(), 10000000), 357u);
    CHECK_EQ(rounded(count_params(preset("9b")).non_embedding(), 10000000), 692u);

    // Section 12: FLOPs per token 1.00 G, 2.75 G, 7.14 G, 13.84 G.
    CHECK_EQ(rounded(flops_per_token(preset("0.8b")), 10000000), 100u);
    CHECK_EQ(rounded(flops_per_token(preset("2b")), 10000000), 275u);
    CHECK_EQ(rounded(flops_per_token(preset("4b")), 10000000), 714u);
    CHECK_EQ(rounded(flops_per_token(preset("9b")), 10000000), 1384u);

    // Only 9B has a separate LM head, of about 1.02B parameters; it is counted apart and never loaded.
    CHECK_EQ(count_params(preset("0.8b")).separate_lm_head, 0u);
    CHECK_EQ(count_params(preset("4b")).separate_lm_head, 0u);
    CHECK_EQ(count_params(preset("9b")).separate_lm_head, 248320ull * 4096ull);
    CHECK_EQ(rounded(count_params(preset("9b")).separate_lm_head, 10000000), 102u);
}

SEPAL_TEST(config, snapshot_quantities_match_the_design) {
    // Section 3: K/V at fp16 is 12,288 bytes per state token on 0.8B / 2B and 32,768 on 4B / 9B.
    CHECK_EQ(kv_bytes_per_token(preset("0.8b"), DType::F16), 12288u);
    CHECK_EQ(kv_bytes_per_token(preset("2b"), DType::F16), 12288u);
    CHECK_EQ(kv_bytes_per_token(preset("4b"), DType::F16), 32768u);
    CHECK_EQ(kv_bytes_per_token(preset("9b"), DType::F16), 32768u);
    CHECK_EQ(kv_bytes_per_token(preset("4b"), DType::I8), 16384u);

    // Recurrent state of all layers at fp32: 18 MiB on 0.8B / 2B and 48 MiB on 4B / 9B.
    const std::uint64_t mib = 1024ull * 1024ull;
    CHECK_EQ(recurrent_state_bytes(preset("0.8b"), DType::F32), 18u * mib);
    CHECK_EQ(recurrent_state_bytes(preset("2b"), DType::F32), 18u * mib);
    CHECK_EQ(recurrent_state_bytes(preset("4b"), DType::F32), 48u * mib);
    CHECK_EQ(recurrent_state_bytes(preset("9b"), DType::F32), 48u * mib);
    CHECK_EQ(recurrent_state_bytes(preset("4b"), DType::F16), 24u * mib);

    // Section 7: conv tails are 18 x 3 x 6,144 x 4 and 24 x 3 x 8,192 x 4 bytes.
    CHECK_EQ(conv_tail_bytes(preset("0.8b"), DType::F32), 1327104u);
    CHECK_EQ(conv_tail_bytes(preset("4b"), DType::F32), 2359296u);
}

SEPAL_TEST(config, validation_names_the_field) {
    const ModelConfig good = preset("4b");

    ModelConfig c = good;
    c.hidden_size = 0;
    CHECK(mentions(validate(c), "hidden_size"));

    c = good;
    c.num_layers = 30; // the last layer would be a DeltaNet layer
    CHECK(mentions(validate(c), "num_layers"));
    CHECK(mentions(validate(c), "full_attention_interval"));

    c = good;
    c.attn_kv_heads = 3;
    CHECK(mentions(validate(c), "attn_kv_heads"));

    c = good;
    c.dn_value_heads = 24; // not a multiple of 16 key heads
    CHECK(mentions(validate(c), "dn_value_heads"));

    c = good;
    c.rms_norm_eps = 0.0;
    CHECK(mentions(validate(c), "rms_norm_eps"));

    c = good;
    c.rope_theta = -1.0;
    CHECK(mentions(validate(c), "rope_theta"));

    c = good;
    c.partial_rotary_factor = 1.5;
    CHECK(mentions(validate(c), "partial_rotary_factor"));

    c = good;
    c.partial_rotary_factor = 0.3; // 76.8 dims: not a whole number
    CHECK(mentions(validate(c), "partial_rotary_factor"));

    c = good;
    c.conv_kernel = 0;
    c.vocab_size = 0;
    c.pointer_dim = 0;
    const auto errors = validate(c); // every violation is reported, not only the first
    CHECK_EQ(errors.size(), 3u);
    CHECK(mentions(errors, "conv_kernel"));
    CHECK(mentions(errors, "vocab_size"));
    CHECK(mentions(errors, "pointer_dim"));

    // A default-constructed configuration is invalid and must not divide by zero while saying so.
    CHECK(!validate(ModelConfig{}).empty());
}

SEPAL_TEST(config, derived_quantities_reject_invalid_geometry) {
    ModelConfig c = preset("2b");
    c.attn_kv_heads = 0;
    CHECK_THROWS(require_valid(c), std::invalid_argument);
    CHECK_THROWS(count_params(c), std::invalid_argument);
    CHECK_THROWS(kv_bytes_per_token(c, DType::F16), std::invalid_argument);
    CHECK_THROWS(flops_per_token(c), std::invalid_argument);
    try {
        require_valid(c);
    } catch (const std::invalid_argument& e) {
        CHECK(std::string(e.what()).find("attn_kv_heads") != std::string::npos);
    }
}
