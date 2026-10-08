// The JSON reader, the golden-file loader and the weights container.

#include <fstream>
#include <functional>
#include <iterator>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "golden_fixture.hpp"
#include "sepal/json.hpp"

using namespace sepal;
using namespace sepal::test;

namespace {

std::vector<unsigned char> golden_bytes() {
    std::ifstream stream(std::string(SEPAL_GOLDEN_DIR) + "/tiny.sgv", std::ios::binary);
    const std::vector<char> raw((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
    return {raw.begin(), raw.end()};
}

} // namespace

// ---- JSON ----

SEPAL_TEST(golden, json_values) {
    const json::Value v = json::parse(R"( {"a": [1, 2.5, -3e2], "b": {"c": null, "d": true, "e": false}, "s": "x\né\"y"} )");
    CHECK_EQ(v.at("a").items().size(), 3u);
    CHECK_EQ(v.at("a").at(0).as_uint(), 1u);
    CHECK_EQ(v.at("a").at(1).as_number(), 2.5);
    CHECK_EQ(v.at("a").at(2).as_number(), -300.0);
    CHECK(v.at("b").at("c").is_null());
    CHECK(v.at("b").at("d").as_bool());
    CHECK(!v.at("b").at("e").as_bool());
    CHECK_EQ(v.at("s").as_string(), std::string("x\n\xC3\xA9\"y"));
    CHECK(v.find("missing") == nullptr);
    CHECK_EQ(v.keys().size(), 3u);
    CHECK_EQ(json::parse("[]").items().size(), 0u);
    CHECK_EQ(json::parse(" 7 ").as_uint(), 7u);
}

SEPAL_TEST(golden, json_refuses_malformed_text_and_wrong_kinds) {
    CHECK_THROWS(json::parse(""), std::runtime_error);
    CHECK_THROWS(json::parse("{\"a\": 1,}"), std::runtime_error);
    CHECK_THROWS(json::parse("[1 2]"), std::runtime_error);
    CHECK_THROWS(json::parse("\"open"), std::runtime_error);
    CHECK_THROWS(json::parse("1 2"), std::runtime_error);
    CHECK_THROWS(json::parse("nul"), std::runtime_error);
    CHECK_THROWS(json::parse("-"), std::runtime_error);
    const json::Value v = json::parse(R"({"n": 1.5, "s": "x"})");
    CHECK_THROWS(v.at("n").as_uint(), std::runtime_error);   // not whole
    CHECK_THROWS(v.at("s").as_number(), std::runtime_error); // not a number
    CHECK_THROWS(v.at("absent"), std::runtime_error);
    CHECK_THROWS(v.at("n").items(), std::runtime_error);
    CHECK_THROWS(json::parse("-1").as_uint(), std::runtime_error);
}

// ---- The golden file ----

SEPAL_TEST(golden, header_describes_the_tiny_geometry) {
    const GoldenFile& file = golden();
    CHECK_EQ(file.header.at("oracle").as_string(), std::string("sepal.refengine"));
    const ModelConfig got = file.config();
    const ModelConfig want = tiny_test_config();
    CHECK_EQ(got.name, want.name);
    CHECK_EQ(got.num_layers, want.num_layers);
    CHECK_EQ(got.full_attention_interval, want.full_attention_interval);
    CHECK_EQ(got.hidden_size, want.hidden_size);
    CHECK_EQ(got.intermediate_size, want.intermediate_size);
    CHECK_EQ(got.attn_heads, want.attn_heads);
    CHECK_EQ(got.attn_kv_heads, want.attn_kv_heads);
    CHECK_EQ(got.attn_head_dim, want.attn_head_dim);
    CHECK_EQ(got.dn_key_heads, want.dn_key_heads);
    CHECK_EQ(got.dn_value_heads, want.dn_value_heads);
    CHECK_EQ(got.dn_key_head_dim, want.dn_key_head_dim);
    CHECK_EQ(got.dn_value_head_dim, want.dn_value_head_dim);
    CHECK_EQ(got.conv_kernel, want.conv_kernel);
    CHECK_EQ(got.vocab_size, want.vocab_size);
    CHECK_EQ(got.rms_norm_eps, want.rms_norm_eps);
    CHECK_EQ(got.rope_theta, want.rope_theta);
    CHECK_EQ(got.partial_rotary_factor, want.partial_rotary_factor);
    CHECK_EQ(got.pointer_dim, want.pointer_dim);
    CHECK_EQ(got.lm_head_tied, want.lm_head_tied);
    CHECK_EQ(got.attn_rotary_dims(), 4u);
}

SEPAL_TEST(golden, every_named_array_exists_with_its_shape) {
    const GoldenFile& file = golden();
    const TensorD& ladder = named(golden_case("ladder"), "layer_outputs");
    const std::size_t tokens = golden_case("ladder").at("token_ids").items().size();
    CHECK_EQ(ladder.shape().size(), 3u);
    CHECK_EQ(ladder.shape().at(0), 8u);
    CHECK_EQ(ladder.shape().at(1), tokens);
    CHECK_EQ(ladder.shape().at(2), 24u);
    CHECK_THROWS(file.array("no/such/array"), std::out_of_range);
    // The oracle's list of unverified details travels with the file.
    CHECK_EQ(file.header.at("unverified").items().size(), 12u);
}

SEPAL_TEST(golden, loader_refuses_damaged_files) {
    const std::vector<unsigned char> bytes = golden_bytes();
    CHECK(!bytes.empty());
    CHECK_EQ(parse_golden(bytes).arrays.size(), golden().arrays.size());

    std::vector<unsigned char> truncated(bytes.begin(), bytes.end() - 5);
    CHECK_THROWS(parse_golden(truncated), std::runtime_error);
    std::vector<unsigned char> extended = bytes;
    extended.push_back(0);
    CHECK_THROWS(parse_golden(extended), std::runtime_error);
    std::vector<unsigned char> wrong_magic = bytes;
    wrong_magic[0] = 'X';
    CHECK_THROWS(parse_golden(wrong_magic), std::runtime_error);
    const std::vector<unsigned char> only_magic(bytes.begin(), bytes.begin() + 8);
    CHECK_THROWS(parse_golden(only_magic), std::runtime_error);
    // A header length far beyond the file must be refused before anything is allocated for it.
    std::vector<unsigned char> huge_header = bytes;
    huge_header[8] = huge_header[9] = huge_header[10] = huge_header[11] = 0xFF;
    CHECK_THROWS(parse_golden(huge_header), std::runtime_error);
    CHECK_THROWS(load_golden(std::string(SEPAL_GOLDEN_DIR) + "/absent.sgv"), std::runtime_error);
}

// ---- Weights ----

SEPAL_TEST(golden, weight_shapes_agree_with_the_parameter_counts) {
    // Two independent derivations of the same numbers: the tensor list and the closed-form counts of config.cpp.
    std::vector<ModelConfig> configs = presets();
    configs.push_back(tiny_test_config());
    for (const ModelConfig& c : configs) {
        const ParamCounts p = count_params(c);
        CHECK_EQ(count_elements(embedding_shapes(c)), p.embedding);
        CHECK_EQ(count_elements(body_shapes(c)), p.non_embedding());
        CHECK_EQ(count_elements(head_shapes(c)), p.pointer_head);
    }
    // The pointer head has biases: two weights [P, H] and two biases [P].
    const ModelConfig c = tiny_test_config();
    CHECK_EQ(count_elements(head_shapes(c)), 2u * (12u * 24u + 12u));
}

SEPAL_TEST(golden, weights_load_strictly_and_scope_by_prefix) {
    const Model<double>& m = model<double>();
    CHECK_EQ(m.weights.size(), weight_shapes(m.config).size());
    CHECK_EQ(m.weights.size(), golden().header.at("weights").items().size());
    const Weights<double> layer = m.weights.scope("layers.0");
    CHECK_EQ(&layer.scope("linear_attn")["A_log"], &m.weights["layers.0.linear_attn.A_log"]);
    CHECK(layer.contains("input_layernorm.weight"));
    CHECK(!layer.contains("self_attn.q_proj.weight")); // layer 0 is DeltaNet
    CHECK_THROWS(layer["no_such.weight"], std::out_of_range);
    CHECK_EQ(m.weights["head.q.bias"].size(), 12u);

    std::map<std::string, TensorD, std::less<>> arrays = golden().weights();
    arrays.erase("norm.weight");
    CHECK_THROWS(Weights<double>::from_arrays(m.config, arrays), std::invalid_argument);
    arrays = golden().weights();
    arrays.emplace("layers.0.extra", TensorD({1}));
    CHECK_THROWS(Weights<double>::from_arrays(m.config, arrays), std::invalid_argument);
    arrays = golden().weights();
    arrays["head.q.bias"] = TensorD({13});
    CHECK_THROWS(Weights<double>::from_arrays(m.config, arrays), std::invalid_argument);
}

SEPAL_TEST(golden, random_weights_are_seeded_and_follow_the_roles) {
    const ModelConfig c = tiny_test_config();
    const Weights<float> a = random_weights<float>(c, 7);
    const Weights<float> b = random_weights<float>(c, 7);
    const Weights<float> other = random_weights<float>(c, 8);
    CHECK_EQ(a.size(), weight_shapes(c).size());
    bool same = true;
    bool differs = false;
    for (const auto& [name, shape] : weight_shapes(c)) {
        for (std::size_t i = 0; i < a[name].size(); ++i) {
            same = same && a[name][i] == b[name][i];
            differs = differs || a[name][i] != other[name][i];
        }
    }
    CHECK(same);
    CHECK(differs);
    // The gated norm's weight is near one; the zero-centred norm weights are near zero.
    for (const float w : a["layers.0.linear_attn.norm.weight"].span()) CHECK(w > 0.8f && w < 1.2f);
    for (const float w : a["layers.0.input_layernorm.weight"].span()) CHECK(w > -0.2f && w < 0.2f);
}
