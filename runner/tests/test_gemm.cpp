#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "sepal/gemm.hpp"
#include "test.hpp"

using namespace sepal;

namespace {

// Deterministic values in [-1, 1) with varied magnitudes; no <random>, so every platform sees the same inputs.
void fill(Tensor& t, std::uint32_t seed) {
    std::uint32_t s = seed;
    for (float& v : t.span()) {
        s = s * 1664525u + 1013904223u;
        v = static_cast<float>(static_cast<double>(s >> 8) / 8388608.0 - 1.0);
    }
}

} // namespace

SEPAL_TEST(gemm, caps_describe_the_reference) {
    const auto gemm = make_reference_gemm();
    const GemmCaps caps = gemm->caps();
    CHECK_EQ(caps.name, "reference");
    CHECK_EQ(caps.isa, "scalar");
    CHECK(caps.fixed_summation_order);
    CHECK(!caps.integer_exact);
    CHECK_EQ(caps.float_dtypes.size(), 3u);
}

SEPAL_TEST(gemm, hand_computed) {
    const auto gemm = make_reference_gemm();
    const Tensor x({2, 3}, {1, 2, 3, 4, 5, 6});
    const Tensor w({2, 3}, {1, 0, -1, 2, 1, 0}); // two output features, one per row
    Tensor y({2, 2});
    gemm->gemm_f(x.mat(), w.mat(), y.mat(), 0.0f);
    CHECK_EQ(y[0], -2.0f); // 1 - 3
    CHECK_EQ(y[1], 4.0f);  // 2 + 2
    CHECK_EQ(y[2], -2.0f); // 4 - 6
    CHECK_EQ(y[3], 13.0f); // 8 + 5
}

SEPAL_TEST(gemm, beta_accumulates_into_y) {
    const auto gemm = make_reference_gemm();
    const Tensor x({2, 3}, {1, 2, 3, 4, 5, 6});
    const Tensor w({2, 3}, {1, 0, -1, 2, 1, 0});
    Tensor y({2, 2}, {10, 20, 30, 40});
    gemm->gemm_f(x.mat(), w.mat(), y.mat(), 0.5f);
    CHECK_EQ(y[0], 3.0f);
    CHECK_EQ(y[1], 14.0f);
    CHECK_EQ(y[2], 13.0f);
    CHECK_EQ(y[3], 33.0f);

    // With beta == 0 the previous contents of Y are not read at all.
    Tensor dirty({2, 2}, std::numeric_limits<float>::quiet_NaN());
    gemm->gemm_f(x.mat(), w.mat(), dirty.mat(), 0.0f);
    CHECK_EQ(dirty[0], -2.0f);
    CHECK_EQ(dirty[3], 13.0f);
}

SEPAL_TEST(gemm, accumulates_in_float64_in_index_order) {
    // In float32 accumulation 1e8 + 1 rounds back to 1e8 and the sum is 0; the documented float64 sum is 1.
    const auto gemm = make_reference_gemm();
    const Tensor x({1, 3}, {1e8f, 1.0f, -1e8f});
    const Tensor w({1, 3}, {1, 1, 1});
    Tensor y({1, 1});
    gemm->gemm_f(x.mat(), w.mat(), y.mat(), 0.0f);
    CHECK_EQ(y[0], 1.0f);
}

SEPAL_TEST(gemm, a_row_never_depends_on_other_rows) {
    const auto gemm = make_reference_gemm();
    const std::size_t rows = 6, width = 37, features = 5;
    Tensor x({rows, width});
    Tensor w({features, width});
    fill(x, 1);
    fill(w, 2);

    Tensor batch({rows, features});
    gemm->gemm_f(x.mat(), w.mat(), batch.mat(), 0.0f);

    // Each row alone gives the same bits as inside the batch.
    for (std::size_t t = 0; t < rows; ++t) {
        Tensor alone({1, features});
        gemm->gemm_f(x.mat().slice_rows(t, 1), w.mat(), alone.mat(), 0.0f);
        for (std::size_t n = 0; n < features; ++n) CHECK_EQ(alone[n], batch.mat()(t, n));
    }

    // The same rows in reverse order, in the company of rows of huge magnitude.
    Tensor shuffled({rows + 2, width});
    for (std::size_t t = 0; t < rows; ++t) {
        for (std::size_t k = 0; k < width; ++k) shuffled.mat()(rows - t, k) = x.mat()(t, k);
    }
    for (std::size_t k = 0; k < width; ++k) {
        shuffled.mat()(0, k) = 1e30f;
        shuffled.mat()(rows + 1, k) = -1e30f;
    }
    Tensor out({rows + 2, features});
    gemm->gemm_f(shuffled.mat(), w.mat(), out.mat(), 0.0f);
    for (std::size_t t = 0; t < rows; ++t) {
        for (std::size_t n = 0; n < features; ++n) CHECK_EQ(out.mat()(rows - t, n), batch.mat()(t, n));
    }
}

SEPAL_TEST(gemm, sixteen_bit_inputs_are_decoded_exactly) {
    const auto gemm = make_reference_gemm();
    const std::vector<std::uint16_t> x_f16 = {0x3C00, 0xC000, 0x3800};  // 1, -2, 0.5 as binary16
    const std::vector<std::uint16_t> w_bf16 = {0x3F80, 0x4000, 0xBF80}; // 1, 2, -1 as bfloat16
    Tensor y({1, 1});
    gemm->gemm_f(MatRef(x_f16.data(), DType::F16, 1, 3), MatRef(w_bf16.data(), DType::BF16, 1, 3), y.mat(), 0.0f);
    CHECK_EQ(y[0], -3.5f); // 1 - 4 - 0.5

    // The same weights given as float32 produce the same bits.
    const Tensor w({1, 3}, {1, 2, -1});
    Tensor y32({1, 1});
    gemm->gemm_f(MatRef(x_f16.data(), DType::F16, 1, 3), w.mat(), y32.mat(), 0.0f);
    CHECK_EQ(y32[0], y[0]);
}

SEPAL_TEST(gemm, rejects_bad_shapes_and_types) {
    const auto gemm = make_reference_gemm();
    const Tensor x({2, 3});
    const Tensor w({4, 3});
    const Tensor w_wrong_width({4, 5});
    Tensor y({2, 4});
    Tensor y_wrong({2, 3});
    CHECK_THROWS(gemm->gemm_f(x.mat(), w_wrong_width.mat(), y.mat(), 0.0f), std::invalid_argument);
    CHECK_THROWS(gemm->gemm_f(x.mat(), w.mat(), y_wrong.mat(), 0.0f), std::invalid_argument);

    const std::vector<std::int8_t> q(12, 1);
    CHECK_THROWS(gemm->gemm_f(x.mat(), MatRef(q.data(), DType::I8, 4, 3), y.mat(), 0.0f), std::invalid_argument);

    // An empty batch is valid and writes nothing.
    const Tensor no_rows({0, 3});
    Tensor no_out({0, 4});
    gemm->gemm_f(no_rows.mat(), w.mat(), no_out.mat(), 0.0f);
    CHECK_EQ(no_out.size(), 0u);
}

// ---- gemm_f64: the float64 reference path ----

SEPAL_TEST(gemm, f64_hand_computed_and_beta) {
    const auto gemm = make_reference_gemm();
    const TensorD x({2, 3}, {1, 2, 3, 4, 5, 6});
    const TensorD w({2, 3}, {1, 0, -1, 2, 1, 0});
    TensorD y({2, 2}, {10, 20, 30, 40});
    gemm->gemm_f64(x.mat(), w.mat(), y.mat(), 0.0); // beta 0: y is written without being read
    CHECK_EQ(y[0], -2.0);
    CHECK_EQ(y[1], 4.0);
    CHECK_EQ(y[2], -2.0);
    CHECK_EQ(y[3], 13.0);
    gemm->gemm_f64(x.mat(), w.mat(), y.mat(), 0.5);
    CHECK_EQ(y[0], -3.0);
    CHECK_EQ(y[3], 19.5);
}

SEPAL_TEST(gemm, f64_keeps_what_float32_rounds_away) {
    // 1 + 2^-40 is a float64 value that float32 cannot hold: the float64 product must keep it.
    const auto gemm = make_reference_gemm();
    const double tiny = 1.0 / 1099511627776.0;
    const TensorD x({1, 2}, {1.0 + tiny, 1.0});
    const TensorD w({1, 2}, {1.0, -1.0});
    TensorD y({1, 1});
    gemm->gemm_f64(x.mat(), w.mat(), y.mat(), 0.0);
    CHECK_EQ(y[0], tiny);
}

SEPAL_TEST(gemm, f64_row_is_independent_of_its_batch) {
    const auto gemm = make_reference_gemm();
    Tensor seed_x({5, 7});
    Tensor seed_w({4, 7});
    fill(seed_x, 11);
    fill(seed_w, 12);
    TensorD x_values({5, 7});
    TensorD w_values({4, 7});
    for (std::size_t i = 0; i < x_values.size(); ++i) x_values[i] = static_cast<double>(seed_x[i]) / 3.0;
    for (std::size_t i = 0; i < w_values.size(); ++i) w_values[i] = static_cast<double>(seed_w[i]) / 7.0;
    const TensorD& x = x_values;
    const TensorD& w = w_values;
    TensorD batch({5, 4});
    gemm->gemm_f64(x.mat(), w.mat(), batch.mat(), 0.0);
    for (std::size_t t = 0; t < 5; ++t) {
        TensorD alone({1, 4});
        gemm->gemm_f64(x.mat().slice_rows(t, 1), w.mat(), alone.mat(), 0.0);
        for (std::size_t n = 0; n < 4; ++n) CHECK_EQ(alone[n], batch[t * 4 + n]);
    }
    // The entry point that code written for both precisions uses reaches the same kernel.
    TensorD through_helper({5, 4});
    sepal::gemm(*gemm, x.mat(), w.mat(), through_helper.mat());
    for (std::size_t i = 0; i < batch.size(); ++i) CHECK_EQ(through_helper[i], batch[i]);
}

SEPAL_TEST(gemm, f64_shape_errors_throw) {
    const auto gemm = make_reference_gemm();
    const TensorD x({2, 3});
    const TensorD w_bad({2, 4});
    const TensorD w({2, 3});
    TensorD y({2, 2});
    TensorD y_bad({3, 2});
    CHECK_THROWS(gemm->gemm_f64(x.mat(), w_bad.mat(), y.mat(), 0.0), std::invalid_argument);
    CHECK_THROWS(gemm->gemm_f64(x.mat(), w.mat(), y_bad.mat(), 0.0), std::invalid_argument);
    CHECK_THROWS(gemm->gemm_f64(ConstMatViewD(nullptr, 2, 3), w.mat(), y.mat(), 0.0), std::invalid_argument);
}
