#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "sepal/ops.hpp"
#include "test.hpp"

using namespace sepal;

namespace {

void fill(Tensor& t, std::uint32_t seed) {
    std::uint32_t s = seed;
    for (float& v : t.span()) {
        s = s * 1664525u + 1013904223u;
        v = static_cast<float>(static_cast<double>(s >> 8) / 8388608.0 - 1.0);
    }
}

bool same_bits(const Tensor& a, const Tensor& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) return false;
    }
    return true;
}

} // namespace

// ---- RMSNorm ----

SEPAL_TEST(ops, rmsnorm_hand_computed) {
    // x = [3, 4]: mean square 12.5, rms 3.5355339. Weights [0, 1] mean scales [1, 2].
    const Tensor x({1, 2}, {3, 4});
    const std::vector<float> w = {0.0f, 1.0f};
    Tensor y({1, 2});
    rmsnorm(x.mat(), w, 0.0, y.mat());
    CHECK_NEAR(y[0], 3.0 / std::sqrt(12.5), 1e-7);
    CHECK_NEAR(y[1], 8.0 / std::sqrt(12.5), 1.5e-7); // half a float32 ulp at 2.26
    CHECK_NEAR(y[0], 0.848528137, 1e-7);
    CHECK_NEAR(y[1], 2.262741700, 1.5e-7);
}

SEPAL_TEST(ops, rmsnorm_weight_is_one_plus_w) {
    // The rival formula (scale = w) would return zeros for a zero weight. The model's formula returns x / rms.
    const Tensor x({1, 4}, {1, -1, 1, -1}); // rms is exactly 1
    const std::vector<float> zero_w(4, 0.0f);
    Tensor y({1, 4});
    rmsnorm(x.mat(), zero_w, 0.0, y.mat());
    CHECK_EQ(y[0], 1.0f);
    CHECK_EQ(y[1], -1.0f);

    const std::vector<float> w = {0.5f, -0.25f, -1.0f, 1.0f};
    rmsnorm(x.mat(), w, 0.0, y.mat());
    CHECK_EQ(y[0], 1.5f);
    CHECK_EQ(y[1], -0.75f);
    CHECK_EQ(y[2], 0.0f);
    CHECK_EQ(y[3], -2.0f);
}

SEPAL_TEST(ops, rmsnorm_eps_rows_and_aliasing) {
    // eps sits inside the square root: x = [1], eps = 3 gives 1 / sqrt(1 + 3).
    const Tensor one({1, 1}, {1});
    const std::vector<float> w1 = {0.0f};
    Tensor y1({1, 1});
    rmsnorm(one.mat(), w1, 3.0, y1.mat());
    CHECK_EQ(y1[0], 0.5f);

    // A zero row stays zero and finite.
    const Tensor zeros({1, 2});
    const std::vector<float> w2 = {0.0f, 0.0f};
    Tensor y2({1, 2}, 7.0f);
    rmsnorm(zeros.mat(), w2, 1e-6, y2.mat());
    CHECK_EQ(y2[0], 0.0f);
    CHECK_EQ(y2[1], 0.0f);

    // Each row is normalised on its own, and y may be x.
    Tensor rows({2, 2}, {3, 4, 30, 40});
    rmsnorm(rows.mat(), w2, 0.0, rows.mat());
    CHECK_NEAR(rows[0], 0.848528137, 1e-7);
    CHECK_NEAR(rows[3], 1.131370850, 1e-7);
    CHECK_NEAR(rows[2], rows[0], 1e-7); // scale invariance: [3, 4] and [30, 40] normalise to the same row

    Tensor bad({1, 3});
    CHECK_THROWS(rmsnorm(rows.mat(), w2, 0.0, bad.mat()), std::invalid_argument);
    CHECK_THROWS(rmsnorm(rows.mat(), w1, 0.0, rows.mat()), std::invalid_argument);
}

// ---- SiLU ----

SEPAL_TEST(ops, silu_hand_computed) {
    CHECK_EQ(silu(0.0f), 0.0f);
    CHECK_NEAR(silu(1.0f), 0.7310585786, 1e-7);  // 1 / (1 + e^-1)
    CHECK_NEAR(silu(-1.0f), -0.2689414214, 1e-7); // -1 / (1 + e)
    CHECK_NEAR(silu(std::log(3.0f)), 0.75 * std::log(3.0), 1e-6); // sigmoid(ln 3) = 3/4
    CHECK_NEAR(silu(20.0f), 20.0, 1e-6);
    CHECK_NEAR(silu(-20.0f), 0.0, 1e-6);
    CHECK_EQ(silu(-1000.0f), 0.0f); // e^1000 overflows to infinity; the result must be 0, not NaN
    CHECK_EQ(silu(1000.0f), 1000.0f);

    std::vector<float> v = {0.0f, 1.0f, -1.0f};
    silu_inplace(v);
    CHECK_EQ(v[0], 0.0f);
    CHECK_EQ(v[1], silu(1.0f));
    CHECK_EQ(v[2], silu(-1.0f));
}

// ---- SwiGLU MLP ----

SEPAL_TEST(ops, swiglu_mlp_hand_computed) {
    // H = 2, I = 2. gate = identity, up = diag(2, 3), down = [[1, 1], [1, -1]], x = [1, -1].
    //   gate(x) = [1, -1]     up(x) = [2, -3]
    //   silu(gate) * up = [0.7310586 * 2, -0.2689414 * -3] = [1.4621172, 0.8068243]
    //   down = [1.4621172 + 0.8068243, 1.4621172 - 0.8068243] = [2.2689414, 0.6552929]
    const auto gemm = make_reference_gemm();
    const Tensor x({1, 2}, {1, -1});
    const Tensor w_gate_up({4, 2}, {1, 0, 0, 1, /* up */ 2, 0, 0, 3});
    const Tensor w_down({2, 2}, {1, 1, 1, -1});

    Tensor y({1, 2});
    swiglu_mlp(*gemm, x.mat(), w_gate_up.mat(), w_down.mat(), y.mat(), 0.0f);
    CHECK_NEAR(y[0], 2.2689414214, 1e-6);
    CHECK_NEAR(y[1], 0.6552928931, 1e-6);

    // beta == 1 adds onto the residual stream already in y.
    Tensor r({1, 2}, {10, 20});
    swiglu_mlp(*gemm, x.mat(), w_gate_up.mat(), w_down.mat(), r.mat(), 1.0f);
    CHECK_NEAR(r[0], 12.2689414214, 1e-5);
    CHECK_NEAR(r[1], 20.6552928931, 1e-5);
}

SEPAL_TEST(ops, swiglu_mlp_rows_are_independent_and_shapes_checked) {
    const auto gemm = make_reference_gemm();
    const std::size_t hidden = 5, inter = 7, rows = 4;
    Tensor x({rows, hidden});
    Tensor w_gate_up({2 * inter, hidden});
    Tensor w_down({hidden, inter});
    fill(x, 11);
    fill(w_gate_up, 12);
    fill(w_down, 13);

    Tensor batch({rows, hidden});
    swiglu_mlp(*gemm, x.mat(), w_gate_up.mat(), w_down.mat(), batch.mat(), 0.0f);
    for (std::size_t t = 0; t < rows; ++t) {
        Tensor alone({1, hidden});
        swiglu_mlp(*gemm, ConstMatView(x.mat()).slice_rows(t, 1), w_gate_up.mat(), w_down.mat(), alone.mat(), 0.0f);
        for (std::size_t i = 0; i < hidden; ++i) CHECK_EQ(alone[i], batch.mat()(t, i));
    }

    Tensor odd({2 * inter + 1, hidden});
    Tensor wrong_down({hidden, inter + 1});
    CHECK_THROWS(swiglu_mlp(*gemm, x.mat(), odd.mat(), w_down.mat(), batch.mat(), 0.0f), std::invalid_argument);
    CHECK_THROWS(swiglu_mlp(*gemm, x.mat(), w_gate_up.mat(), wrong_down.mat(), batch.mat(), 0.0f),
                 std::invalid_argument);
}

// ---- softmax ----

SEPAL_TEST(ops, softmax_hand_computed) {
    // exp of [0, ln 2, ln 3] is [1, 2, 3], so the probabilities are [1/6, 1/3, 1/2].
    const std::vector<float> logits = {0.0f, std::log(2.0f), std::log(3.0f)};
    std::vector<float> p(3);
    softmax(logits, p);
    CHECK_NEAR(p[0], 1.0 / 6.0, 1e-7);
    CHECK_NEAR(p[1], 1.0 / 3.0, 1e-7);
    CHECK_NEAR(p[2], 0.5, 1e-7);
    CHECK_NEAR(p[0] + p[1] + p[2], 1.0, 1e-6);
}

SEPAL_TEST(ops, softmax_edge_cases) {
    // Logits that overflow exp() on their own still give a clean answer.
    std::vector<float> big = {1000.0f, 1000.0f};
    softmax(big, big); // in place
    CHECK_EQ(big[0], 0.5f);
    CHECK_EQ(big[1], 0.5f);

    const std::vector<float> spread = {-1000.0f, 0.0f, 1000.0f};
    std::vector<float> p(3);
    softmax(spread, p);
    CHECK_EQ(p[0], 0.0f);
    CHECK_EQ(p[1], 0.0f);
    CHECK_EQ(p[2], 1.0f);

    const std::vector<float> single = {-3.0f};
    std::vector<float> one(1);
    softmax(single, one);
    CHECK_EQ(one[0], 1.0f);

    // Shift invariance, and the widest readout the design allows: 255 options.
    const std::vector<float> wide(255, 4.0f);
    std::vector<float> wide_p(255);
    softmax(wide, wide_p);
    CHECK_NEAR(wide_p[0], 1.0 / 255.0, 1e-9);
    CHECK_EQ(wide_p[0], wide_p[254]);

    const std::vector<float> none;
    std::vector<float> none_p;
    CHECK_THROWS(softmax(none, none_p), std::invalid_argument);
    CHECK_THROWS(softmax(single, p), std::invalid_argument);
}

// ---- depthwise causal convolution ----

SEPAL_TEST(ops, conv_hand_computed) {
    // C = 2, K = 3. Channel 0 filter [1, 2, 3]; channel 1 filter [0.5, 0, -1]. Tap K-1 multiplies the newest input.
    const Tensor w({2, 3}, {1, 2, 3, 0.5f, 0, -1});
    const Tensor tail({2, 2}, {10, 1, 20, 2}); // the two inputs before x, oldest first
    const Tensor x({2, 2}, {30, 3, 40, 4});
    Tensor y({2, 2});
    Tensor tail_out({2, 2});
    causal_conv1d_depthwise(x.mat(), w.mat(), tail.mat(), y.mat(), tail_out.mat());
    CHECK_EQ(y.mat()(0, 0), 140.0f); // 1*10 + 2*20 + 3*30
    CHECK_EQ(y.mat()(1, 0), 200.0f); // 1*20 + 2*30 + 3*40
    CHECK_EQ(y.mat()(0, 1), -2.5f);  // 0.5*1 + 0*2 - 1*3
    CHECK_EQ(y.mat()(1, 1), -3.0f);  // 0.5*2 + 0*3 - 1*4
    CHECK_EQ(tail_out[0], 30.0f);
    CHECK_EQ(tail_out[1], 3.0f);
    CHECK_EQ(tail_out[2], 40.0f);
    CHECK_EQ(tail_out[3], 4.0f);
}

SEPAL_TEST(ops, conv_is_causal_from_a_zero_tail) {
    const Tensor w({2, 3}, {1, 2, 3, 0.5f, 0, -1});
    const Tensor zero_tail({2, 2});
    const Tensor x({2, 2}, {30, 3, 40, 4});
    Tensor y({2, 2});
    causal_conv1d_depthwise(x.mat(), w.mat(), zero_tail.mat(), y.mat(), MatView{});
    CHECK_EQ(y.mat()(0, 0), 90.0f);  // only the newest tap sees data: 3*30
    CHECK_EQ(y.mat()(1, 0), 180.0f); // 2*30 + 3*40
    CHECK_EQ(y.mat()(0, 1), -3.0f);
    CHECK_EQ(y.mat()(1, 1), -4.0f);

    // Changing a later input never changes an earlier output.
    const Tensor x2({2, 2}, {30, 3, -999, 999});
    Tensor y2({2, 2});
    causal_conv1d_depthwise(x2.mat(), w.mat(), zero_tail.mat(), y2.mat(), MatView{});
    CHECK_EQ(y2.mat()(0, 0), y.mat()(0, 0));
    CHECK_EQ(y2.mat()(0, 1), y.mat()(0, 1));
}

SEPAL_TEST(ops, conv_tail_when_the_piece_is_shorter_than_the_tail) {
    // One new input with K = 3: the new tail keeps the newer old row and appends the input.
    const Tensor w({2, 3}, {1, 2, 3, 0.5f, 0, -1});
    Tensor tail({2, 2}, {10, 1, 20, 2});
    const Tensor x({1, 2}, {30, 3});
    Tensor y({1, 2});
    causal_conv1d_depthwise(x.mat(), w.mat(), tail.mat(), y.mat(), tail.mat()); // tail updated in place
    CHECK_EQ(y[0], 140.0f);
    CHECK_EQ(tail[0], 20.0f);
    CHECK_EQ(tail[1], 2.0f);
    CHECK_EQ(tail[2], 30.0f);
    CHECK_EQ(tail[3], 3.0f);

    // No new input: nothing is produced and the tail is unchanged.
    const Tensor none({0, 2});
    Tensor no_y({0, 2});
    causal_conv1d_depthwise(none.mat(), w.mat(), tail.mat(), no_y.mat(), tail.mat());
    CHECK_EQ(tail[0], 20.0f);
    CHECK_EQ(tail[3], 3.0f);
}

SEPAL_TEST(ops, conv_in_pieces_equals_conv_whole) {
    // The model's shape: K = 4 taps. Every split point, including pieces shorter than the tail.
    const std::size_t channels = 3, taps = 4, steps = 11;
    Tensor w({channels, taps});
    Tensor x({steps, channels});
    Tensor tail0({taps - 1, channels});
    fill(w, 21);
    fill(x, 22);
    fill(tail0, 23); // a non-zero parent tail, as a branch continuing a state would have

    Tensor whole({steps, channels});
    Tensor whole_tail({taps - 1, channels});
    causal_conv1d_depthwise(x.mat(), w.mat(), tail0.mat(), whole.mat(), whole_tail.mat());

    for (std::size_t split = 0; split <= steps; ++split) {
        Tensor y({steps, channels});
        Tensor carried({taps - 1, channels});
        Tensor final_tail({taps - 1, channels});
        const ConstMatView xv = x.mat();
        causal_conv1d_depthwise(xv.slice_rows(0, split), w.mat(), tail0.mat(), y.mat().slice_rows(0, split),
                                carried.mat());
        causal_conv1d_depthwise(xv.slice_rows(split, steps - split), w.mat(), carried.mat(),
                                y.mat().slice_rows(split, steps - split), final_tail.mat());
        CHECK(same_bits(y, whole));
        CHECK(same_bits(final_tail, whole_tail));
    }

    // One token at a time, carrying the tail in place.
    Tensor y({steps, channels});
    Tensor tail = tail0;
    for (std::size_t t = 0; t < steps; ++t) {
        causal_conv1d_depthwise(ConstMatView(x.mat()).slice_rows(t, 1), w.mat(), tail.mat(),
                                y.mat().slice_rows(t, 1), tail.mat());
    }
    CHECK(same_bits(y, whole));
    CHECK(same_bits(tail, whole_tail));
}

SEPAL_TEST(ops, conv_single_tap_and_shape_errors) {
    // K = 1 has no history: the tail has no rows.
    const Tensor w1({2, 1}, {2, -3});
    const Tensor x({2, 2}, {1, 1, 5, 5});
    Tensor y({2, 2});
    causal_conv1d_depthwise(x.mat(), w1.mat(), ConstMatView(nullptr, 0, 2), y.mat(), MatView{});
    CHECK_EQ(y[0], 2.0f);
    CHECK_EQ(y[1], -3.0f);
    CHECK_EQ(y[2], 10.0f);
    CHECK_EQ(y[3], -15.0f);

    const Tensor w({2, 3});
    const Tensor tail({2, 2});
    const Tensor short_tail({1, 2});
    const Tensor wide_x({2, 3});
    Tensor wrong_y({1, 2});
    Tensor wrong_tail_out({1, 2});
    CHECK_THROWS(causal_conv1d_depthwise(x.mat(), w.mat(), short_tail.mat(), y.mat(), MatView{}),
                 std::invalid_argument);
    CHECK_THROWS(causal_conv1d_depthwise(wide_x.mat(), w.mat(), tail.mat(), y.mat(), MatView{}),
                 std::invalid_argument);
    CHECK_THROWS(causal_conv1d_depthwise(x.mat(), w.mat(), tail.mat(), wrong_y.mat(), MatView{}),
                 std::invalid_argument);
    CHECK_THROWS(causal_conv1d_depthwise(x.mat(), w.mat(), tail.mat(), y.mat(), wrong_tail_out.mat()),
                 std::invalid_argument);
}
