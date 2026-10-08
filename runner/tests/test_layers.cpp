// The two mixers and the decoder layer, each on its own against the oracle's intermediate values.
//
// Every test here feeds a layer the ORACLE's input and the ORACLE's parent state, so a layer is judged alone:
// an error in one layer cannot hide in, or be blamed on, another.

#include <cmath>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "golden_fixture.hpp"
#include "sepal/ref/attention.hpp"
#include "sepal/ref/deltanet.hpp"
#include "sepal/ref/executor.hpp"

using namespace sepal;
using namespace sepal::test;
using namespace sepal::ref;

namespace {

const double tol = oracle_tolerance;

std::vector<double> step_positions() {
    const json::Value& steps = golden_case("steps");
    const double start = steps.at("start_position").as_number();
    std::vector<double> positions(steps.at("token_ids").items().size());
    for (std::size_t t = 0; t < positions.size(); ++t) positions[t] = start + static_cast<double>(t);
    return positions;
}

const Snapshot<double>& parent_snapshot() {
    static const SnapshotPtr<double> snapshot = snapshot_from_golden<double>(golden_case("snapshot"));
    return *snapshot;
}

Weights<double> layer_weights(std::size_t layer) { return model<double>().weights.scope("layers." + std::to_string(layer)); }

} // namespace

// ---- Gated DeltaNet ----

SEPAL_TEST(deltanet, every_intermediate_of_one_mixer_matches_the_oracle) {
    const Model<double>& m = model<double>();
    const json::Value& detail = golden_case("deltanet_detail");
    const std::size_t layer = static_cast<std::size_t>(detail.at("layer").as_uint());
    const Weights<double> w = layer_weights(layer).scope("linear_attn");
    const DeltaNetState<double>& state = *std::get<DeltaNetStatePtr<double>>(parent_snapshot().layers.at(layer));
    const TensorD& x = named(golden_case("steps").at("layers").at(layer), "normed");
    const ModelConfig& c = m.config;
    Worst worst("[f64] deltanet mixer intermediates, layer " + std::to_string(layer));

    // The single in-projection, as four products with the four weights it is made of.
    const TensorD mixed = project(*m.gemm, x.mat(), w["in_proj_qkv.weight"]);
    const TensorD z = project(*m.gemm, x.mat(), w["in_proj_z.weight"]);
    TensorD b = project(*m.gemm, x.mat(), w["in_proj_b.weight"]);
    const TensorD a = project(*m.gemm, x.mat(), w["in_proj_a.weight"]);
    EXPECT_CLOSE(worst, "in_proj_qkv", mixed, named(detail, "mixed"), tol);
    EXPECT_CLOSE(worst, "in_proj_z", z, named(detail, "z"), tol);
    EXPECT_CLOSE(worst, "in_proj_b", b, named(detail, "b"), tol);
    EXPECT_CLOSE(worst, "in_proj_a", a, named(detail, "a"), tol);

    // The causal conv, continued from the parent's tail.
    ConvResult<double> conv = causal_conv<double>(mixed.mat(), w["conv1d.weight"].as_mat(dim(c.conv_kernel)), state.conv_tail.mat());
    EXPECT_CLOSE(worst, "conv output", conv.outputs, named(detail, "conv_output"), tol);
    EXPECT_CLOSE(worst, "conv tail", conv.tail, named(detail, "conv_tail"), tol);

    // SiLU, the split into heads, L2-normalised q and k, key heads shared by value heads.
    for (double& value : conv.outputs.span()) value = silu(value);
    SplitHeads<double> split = split_heads<double>(conv.outputs.mat(), c);
    l2_normalise(split.q.as_mat(dim(c.dn_key_head_dim)));
    l2_normalise(split.k.as_mat(dim(c.dn_key_head_dim)));
    const TensorD q = share_key_heads(split.q, c);
    const TensorD k = share_key_heads(split.k, c);
    CHECK(q.shape() == named(detail, "q").shape());
    CHECK(split.v.shape() == named(detail, "v").shape());
    EXPECT_CLOSE(worst, "q", q, named(detail, "q"), tol);
    EXPECT_CLOSE(worst, "k", k, named(detail, "k"), tol);
    EXPECT_CLOSE(worst, "v", split.v, named(detail, "v"), tol);

    // The two gates.
    const TensorD decay = decay_gate<double>(a.mat(), w["A_log"].span(), w["dt_bias"].span());
    for (double& value : b.span()) value = sigmoid(value);
    EXPECT_CLOSE(worst, "decay gate", decay, named(detail, "decay"), tol);
    EXPECT_CLOSE(worst, "beta gate", b, named(detail, "beta"), tol);

    // The delta-rule recurrence, token by token from the parent's recurrent state.
    const ScanResult<double> scan = delta_rule_scan(q, k, split.v, decay, b, state.recurrent, delta_rule_output_scale(c));
    CHECK(scan.reads.shape() == named(detail, "reads").shape());
    EXPECT_CLOSE(worst, "reads", scan.reads, named(detail, "reads"), tol);
    EXPECT_CLOSE(worst, "recurrent state", scan.state, named(detail, "recurrent"), tol);

    // The gated norm and the out projection.
    const std::size_t dv = dim(c.dn_value_head_dim);
    const TensorD normed = gated_rms_norm<double>(scan.reads.as_mat(dv), z.as_mat(dv), w["norm.weight"].span(), c.rms_norm_eps);
    EXPECT_CLOSE(worst, "gated norm", normed, named(detail, "normed"), tol);
    const TensorD output = project<double>(*m.gemm, normed.as_mat(dim(c.dn_value_dim())), w["out_proj.weight"]);
    EXPECT_CLOSE(worst, "out projection", output, named(detail, "output"), tol);
}

SEPAL_TEST(deltanet, each_layer_alone_matches_the_oracle) {
    const Model<double>& m = model<double>();
    const std::vector<json::Value>& layers = golden_case("steps").at("layers").items();
    Worst worst("[f64] deltanet mixer alone, every DeltaNet layer (output, recurrent state, conv tail)");
    std::size_t tested = 0;
    for (std::size_t i = 0; i < layers.size(); ++i) {
        if (m.config.is_full_attention(i)) continue;
        const DeltaNetState<double>& state = *std::get<DeltaNetStatePtr<double>>(parent_snapshot().layers.at(i));
        const DeltaNetResult<double> got =
            deltanet_mixer<double>(*m.gemm, named(layers[i], "normed").mat(), layer_weights(i).scope("linear_attn"), state, m.config);
        const std::string name = "layer " + std::to_string(i);
        EXPECT_CLOSE(worst, name + " mixer output", got.output, named(layers[i], "mixer_output"), tol);
        EXPECT_CLOSE(worst, name + " recurrent state", got.state->recurrent, named(layers[i].at("state"), "recurrent"), tol);
        EXPECT_CLOSE(worst, name + " conv tail", got.state->conv_tail, named(layers[i].at("state"), "conv_tail"), tol);
        ++tested;
    }
    CHECK_EQ(tested, 6u);
}

SEPAL_TEST(deltanet, a_run_in_two_pieces_equals_the_run_whole) {
    // Bit for bit: the conv tail and the recurrent state carry everything a later token needs.
    const Model<double>& m = model<double>();
    const std::size_t layer = 0;
    const TensorD& x = named(golden_case("steps").at("layers").at(layer), "normed");
    const DeltaNetStatePtr<double>& start = std::get<DeltaNetStatePtr<double>>(parent_snapshot().layers.at(layer));
    const Weights<double> w = layer_weights(layer).scope("linear_attn");
    const DeltaNetResult<double> whole = deltanet_mixer<double>(*m.gemm, x.mat(), w, *start, m.config);
    for (const std::size_t cut : {std::size_t{1}, std::size_t{2}, std::size_t{5}, x.mat().rows - 1}) {
        const DeltaNetResult<double> head = deltanet_mixer<double>(*m.gemm, x.mat().slice_rows(0, cut), w, *start, m.config);
        const DeltaNetResult<double> tail =
            deltanet_mixer<double>(*m.gemm, x.mat().slice_rows(cut, x.mat().rows - cut), w, *head.state, m.config);
        bool same = true;
        for (std::size_t i = 0; i < head.output.size(); ++i) same = same && head.output[i] == whole.output[i];
        for (std::size_t i = 0; i < tail.output.size(); ++i) same = same && tail.output[i] == whole.output[head.output.size() + i];
        for (std::size_t i = 0; i < whole.state->recurrent.size(); ++i) same = same && tail.state->recurrent[i] == whole.state->recurrent[i];
        for (std::size_t i = 0; i < whole.state->conv_tail.size(); ++i) same = same && tail.state->conv_tail[i] == whole.state->conv_tail[i];
        CHECK(same);
    }
}

SEPAL_TEST(deltanet, the_mixer_does_not_write_its_parent_state) {
    const Model<double>& m = model<double>();
    const std::size_t layer = 1;
    const TensorD& x = named(golden_case("steps").at("layers").at(layer), "normed");
    const DeltaNetStatePtr<double>& start = std::get<DeltaNetStatePtr<double>>(parent_snapshot().layers.at(layer));
    const TensorD recurrent_before = start->recurrent;
    const TensorD tail_before = start->conv_tail;
    const DeltaNetResult<double> got = deltanet_mixer<double>(*m.gemm, x.mat(), layer_weights(layer).scope("linear_attn"), *start, m.config);
    CHECK(got.state != start);
    CHECK(got.state->recurrent.data() != start->recurrent.data());
    CHECK_EQ(max_abs_diff<double>(start->recurrent.span(), recurrent_before.span()), 0.0);
    CHECK_EQ(max_abs_diff<double>(start->conv_tail.span(), tail_before.span()), 0.0);
    // The state did move: the test would be vacuous if the layer left it as it was.
    CHECK(max_abs_diff<double>(got.state->recurrent.span(), recurrent_before.span()) > 1e-3);
}

// ---- Full attention ----

SEPAL_TEST(attention, every_intermediate_of_one_mixer_matches_the_oracle) {
    const Model<double>& m = model<double>();
    const json::Value& detail = golden_case("attention_detail");
    const std::size_t layer = static_cast<std::size_t>(detail.at("layer").as_uint());
    const Weights<double> w = layer_weights(layer).scope("self_attn");
    const AttentionStatePtr<double>& state = std::get<AttentionStatePtr<double>>(parent_snapshot().layers.at(layer));
    const TensorD& x = named(golden_case("steps").at("layers").at(layer), "normed");
    const ModelConfig& c = m.config;
    const std::size_t head_dim = dim(c.attn_head_dim);
    const std::size_t tokens = x.mat().rows;
    const std::vector<double> positions = step_positions();
    Worst worst("[f64] attention mixer intermediates, layer " + std::to_string(layer));

    // The query with its packed output gate, the keys and the values.
    const TensorD q_gate = project(*m.gemm, x.mat(), w["q_proj.weight"]);
    EXPECT_CLOSE(worst, "q_proj (query and gate)", q_gate, named(detail, "q_gate"), tol);
    const QueryAndGate<double> unpacked = unpack_query_and_gate<double>(q_gate.mat(), c);
    EXPECT_CLOSE(worst, "output gate", unpacked.gate, named(detail, "gate"), tol);

    // Per-head norms, then RoPE on the rotated part of the head dims.
    const TensorD angles = rope_angles(positions, c);
    EXPECT_CLOSE(worst, "RoPE angles", angles, named(detail, "angles"), tol);
    TensorD q = rms_norm<double>(unpacked.query.as_mat(head_dim), w["q_norm.weight"].span(), c.rms_norm_eps);
    q.reshape({tokens, dim(c.attn_heads), head_dim});
    apply_rope(q, angles);
    EXPECT_CLOSE(worst, "q after norm and RoPE", q, named(detail, "q"), tol);
    TensorD k = rms_norm<double>(project(*m.gemm, x.mat(), w["k_proj.weight"]).as_mat(head_dim), w["k_norm.weight"].span(), c.rms_norm_eps);
    k.reshape({tokens, dim(c.attn_kv_heads), head_dim});
    apply_rope(k, angles);
    EXPECT_CLOSE(worst, "k after norm and RoPE", k, named(detail, "k"), tol);
    TensorD v = project(*m.gemm, x.mat(), w["v_proj.weight"]);
    v.reshape({tokens, dim(c.attn_kv_heads), head_dim});

    // The parent's keys and values followed by the run's own.
    auto joined = std::make_shared<AttentionState<double>>();
    joined->parent = state;
    joined->keys = k;
    joined->values = v;
    joined->tokens = state->tokens + tokens;
    EXPECT_CLOSE(worst, "keys, parent then own", gather_attention(*joined, &AttentionState<double>::keys, c), named(detail, "keys"), tol);
    EXPECT_CLOSE(worst, "values, parent then own", gather_attention(*joined, &AttentionState<double>::values, c), named(detail, "values"), tol);

    // Grouped-query attention, the sigmoid output gate, the out projection.
    TensorD attended = grouped_query_attention(*m.gemm, q, *joined, state->tokens, attention_scale(c));
    EXPECT_CLOSE(worst, "attended", attended, named(detail, "attended"), tol);
    for (std::size_t i = 0; i < attended.size(); ++i) attended[i] *= sigmoid(unpacked.gate[i]);
    EXPECT_CLOSE(worst, "gated", attended, named(detail, "gated"), tol);
    const TensorD output = project<double>(*m.gemm, attended.as_mat(dim(c.attn_width())), w["o_proj.weight"]);
    EXPECT_CLOSE(worst, "out projection", output, named(detail, "output"), tol);
}

SEPAL_TEST(attention, each_layer_alone_matches_the_oracle) {
    const Model<double>& m = model<double>();
    const std::vector<json::Value>& layers = golden_case("steps").at("layers").items();
    const std::vector<double> positions = step_positions();
    Worst worst("[f64] attention mixer alone, every attention layer (output, keys, values)");
    std::size_t tested = 0;
    for (std::size_t i = 0; i < layers.size(); ++i) {
        if (!m.config.is_full_attention(i)) continue;
        const AttentionStatePtr<double>& state = std::get<AttentionStatePtr<double>>(parent_snapshot().layers.at(i));
        const AttentionResult<double> got = attention_mixer<double>(*m.gemm, named(layers[i], "normed").mat(), positions,
                                                                    layer_weights(i).scope("self_attn"), state, m.config);
        const std::string name = "layer " + std::to_string(i);
        EXPECT_CLOSE(worst, name + " mixer output", got.output, named(layers[i], "mixer_output"), tol);
        EXPECT_CLOSE(worst, name + " keys", gather_attention(*got.state, &AttentionState<double>::keys, m.config),
                     named(layers[i].at("state"), "keys"), tol);
        EXPECT_CLOSE(worst, name + " values", gather_attention(*got.state, &AttentionState<double>::values, m.config),
                     named(layers[i].at("state"), "values"), tol);
        // The new state shares the parent's keys and values: it holds a link and only its own tokens.
        CHECK(got.state->parent == state);
        CHECK_EQ(got.state->own_tokens(), positions.size());
        CHECK_EQ(got.state->keys.shape().at(0), positions.size());
        ++tested;
    }
    CHECK_EQ(tested, 2u);
}

SEPAL_TEST(attention, rope_rotates_only_the_first_rotary_dims) {
    const ModelConfig c = tiny_test_config(); // head dim 16, 4 rotated
    TensorD x({2, 1, 16});
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = 1.0 + static_cast<double>(i);
    const TensorD before = x;
    const std::vector<double> positions = {0.0, 3.0};
    const TensorD angles = rope_angles(positions, c);
    CHECK_EQ(angles.shape().at(1), 2u);
    CHECK_EQ(angles[0], 0.0);
    CHECK_EQ(angles[2], 3.0);                       // frequency of pair 0 is theta^0 = 1
    CHECK_NEAR(angles[3], 3.0 / std::sqrt(1e7), 1e-15); // pair 1: theta^(-2/4)
    apply_rope(x, angles);
    for (std::size_t i = 0; i < 16; ++i) CHECK_EQ(x[i], before[i]); // position 0 is the identity
    for (std::size_t i = 4; i < 16; ++i) CHECK_EQ(x[16 + i], before[16 + i]);
    // Token 1, pair 0: dims 0 and 2 (half-split pairing), angle 3.
    CHECK_NEAR(x[16 + 0], before[16 + 0] * std::cos(3.0) - before[16 + 2] * std::sin(3.0), 1e-12);
    CHECK_NEAR(x[16 + 2], before[16 + 2] * std::cos(3.0) + before[16 + 0] * std::sin(3.0), 1e-12);
}

SEPAL_TEST(attention, a_token_never_attends_to_a_later_token) {
    // Changing the last token of a run must leave every earlier output of the mixer bit-identical.
    const Model<double>& m = model<double>();
    const std::size_t layer = 3;
    const AttentionStatePtr<double>& state = std::get<AttentionStatePtr<double>>(parent_snapshot().layers.at(layer));
    TensorD x = named(golden_case("steps").at("layers").at(layer), "normed");
    const std::vector<double> positions = step_positions();
    const Weights<double> w = layer_weights(layer).scope("self_attn");
    const AttentionResult<double> before = attention_mixer<double>(*m.gemm, x.mat(), positions, w, state, m.config);
    const std::size_t hidden = x.mat().cols;
    const std::size_t last = x.mat().rows - 1;
    for (std::size_t i = 0; i < hidden; ++i) x[last * hidden + i] += 1.0;
    const AttentionResult<double> after = attention_mixer<double>(*m.gemm, x.mat(), positions, w, state, m.config);
    bool earlier_same = true;
    for (std::size_t i = 0; i < last * hidden; ++i) earlier_same = earlier_same && before.output[i] == after.output[i];
    CHECK(earlier_same);
    bool last_differs = false;
    for (std::size_t i = last * hidden; i < before.output.size(); ++i) last_differs = last_differs || before.output[i] != after.output[i];
    CHECK(last_differs);
}

// ---- The decoder layer ----

SEPAL_TEST(executor, each_decoder_layer_alone_matches_the_oracle) {
    const Model<double>& m = model<double>();
    const std::vector<json::Value>& layers = golden_case("steps").at("layers").items();
    const std::vector<double> positions = step_positions();
    Worst worst("[f64] decoder layer alone, all 8 layers (normed input, output)");
    for (std::size_t i = 0; i < layers.size(); ++i) {
        const TensorD& x = named(layers[i], "input");
        const std::string name = "layer " + std::to_string(i);
        const TensorD normed = rms_norm<double>(x.mat(), layer_weights(i)["input_layernorm.weight"].span(), m.config.rms_norm_eps);
        EXPECT_CLOSE(worst, name + " normed input", normed, named(layers[i], "normed"), tol);
        const LayerResult<double> got = decoder_layer<double>(*m.gemm, x.mat(), positions, layer_weights(i), parent_snapshot().layers.at(i), m.config);
        EXPECT_CLOSE(worst, name + " output", got.output, named(layers[i], "output"), tol);
    }
}
