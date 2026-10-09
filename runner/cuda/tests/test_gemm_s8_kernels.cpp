// Tests of the int8 multiply kernels (src/gemm_s8_kernels.*): per-row and per-channel quantisation, the int8-activation
// multiply (exact int32 sums) and the binary16-activation multiply with int8 weights. Needs a CUDA device.
//
//   sepal_cuda_kernel_s8_tests            every suite
//   sepal_cuda_kernel_s8_tests <suite>    one of: s8_quantize, s8_gemm, s8_invariance, s8_edge
//
// What is compared with what:
//   - quantisation: a host oracle that does the same float32 steps; the bytes and the scales must be identical.
//   - the int8-activation multiply: the exact integer product of the device's own int8 operands, then the same two float32
//     multiplications, bit for bit; and, for the whole quantised multiply, the float64 product of the unquantised operands
//     (the figure printed is the relative rms error, the cost of quantisation).
//   - the binary16-activation multiply: float64 product of the binary16 activations and the dequantised weights.
//   - invariance: bit for bit across tile shapes, batch sizes and places in the batch.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "device.hpp"
#include "gemm_s8_kernels.hpp"
#include "test.hpp"

namespace {

using sepal::cuda::DeviceBuffer;
using sepal::cuda::S8WeightLayout;

struct Stream {
    cudaStream_t handle = nullptr;
    Stream() { sepal::cuda::check(cudaStreamCreate(&handle), "cudaStreamCreate"); }
    ~Stream() { cudaStreamDestroy(handle); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
};

cudaStream_t stream() {
    static Stream s;
    return s.handle;
}

void sync() { sepal::cuda::check(cudaStreamSynchronize(stream()), "cudaStreamSynchronize"); }

std::vector<__half> random_half(std::mt19937& rng, std::size_t count, float scale) {
    std::normal_distribution<float> nd(0.0f, scale);
    std::vector<__half> v(count);
    for (auto& e : v) e = __float2half(nd(rng));
    return v;
}

// The host's version of both quantisers: scale = max|x| / 127, q = round-half-even(x * (127 / max|x|)) clamped to +-127.
void quantize_row(const float* x, std::size_t k, std::int8_t* q, float* scale) {
    float amax = 0.0f;
    for (std::size_t i = 0; i < k; ++i) amax = std::max(amax, std::fabs(x[i]));
    const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
    *scale = amax / 127.0f;
    for (std::size_t i = 0; i < k; ++i) {
        const float v = std::nearbyint(x[i] * inv);
        q[i] = static_cast<std::int8_t>(std::min(127.0f, std::max(-127.0f, v)));
    }
}

struct Problem {
    std::size_t t, n, k;
    std::vector<__half> x, w;
    std::vector<float> xf, wf;
    DeviceBuffer<__half> dx, dw;
    DeviceBuffer<std::int8_t> xq, wq, wqp;
    DeviceBuffer<float> sx, sw, swp;
    Problem(std::size_t t_, std::size_t n_, std::size_t k_, std::uint32_t seed, float x_scale = 1.0f) : t(t_), n(n_), k(k_) {
        std::mt19937 rng(seed);
        x = random_half(rng, t * k, x_scale);
        w = random_half(rng, n * k, 1.0f / std::sqrt(static_cast<float>(k)));
        xf.resize(x.size());
        wf.resize(w.size());
        for (std::size_t i = 0; i < x.size(); ++i) xf[i] = __half2float(x[i]);
        for (std::size_t i = 0; i < w.size(); ++i) wf[i] = __half2float(w[i]);
        dx = DeviceBuffer<__half>(x);
        dw = DeviceBuffer<__half>(w);
        xq = DeviceBuffer<std::int8_t>(t * k);
        wq = DeviceBuffer<std::int8_t>(n * k);
        wqp = DeviceBuffer<std::int8_t>(n * k);
        sx = DeviceBuffer<float>(t);
        sw = DeviceBuffer<float>(n);
        swp = DeviceBuffer<float>(n);
        sepal::cuda::launch_quantize_rows_s8(dx.data(), k, xq.data(), k, sx.data(), t, k, stream());
        sepal::cuda::launch_quantize_weights_s8(dw.data(), k, wq.data(), sw.data(), n, k, S8WeightLayout::Plain, stream());
        sepal::cuda::launch_quantize_weights_s8(dw.data(), k, wqp.data(), swp.data(), n, k, S8WeightLayout::Fp16Pairs, stream());
        sync();
    }
    // Rows [row0, row0 + rows) of the int8-activation multiply, float32 result.
    std::vector<float> run_s8(std::size_t row0, std::size_t rows, int config) const {
        DeviceBuffer<float> y(rows * n);
        sepal::cuda::launch_gemm_s8(xq.data() + row0 * k, k, sx.data() + row0, wq.data(), k, sw.data(), y.data(), n, false, rows, n, k, config, stream());
        sync();
        return y.download();
    }
    std::vector<float> run_w8a16(std::size_t row0, std::size_t rows, int config) const {
        DeviceBuffer<float> y(rows * n);
        sepal::cuda::launch_gemm_w8a16(dx.data() + row0 * k, k, wqp.data(), k, swp.data(), y.data(), n, false, rows, n, k, config, stream());
        sync();
        return y.download();
    }
};

double relative_rms_error(const std::vector<float>& got, const std::vector<double>& want) {
    double err = 0, ref = 0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double d = static_cast<double>(got[i]) - want[i];
        err += d * d;
        ref += want[i] * want[i];
    }
    return std::sqrt(err / std::max(ref, 1e-300));
}

bool same_bits(const float* a, const float* b, std::size_t count) { return std::memcmp(a, b, count * sizeof(float)) == 0; }

// Float64 product of the unquantised (binary16) operands.
std::vector<double> exact_product(const Problem& p) {
    std::vector<double> y(p.t * p.n);
    for (std::size_t r = 0; r < p.t; ++r)
        for (std::size_t c = 0; c < p.n; ++c) {
            double acc = 0;
            for (std::size_t i = 0; i < p.k; ++i) acc += static_cast<double>(p.xf[r * p.k + i]) * p.wf[c * p.k + i];
            y[r * p.n + c] = acc;
        }
    return y;
}

} // namespace

SEPAL_TEST(s8_quantize, rows_and_weights_match_the_host_oracle_bit_for_bit) {
    for (const std::size_t k : {16u, 48u, 1024u, 1040u, 3584u, 6144u}) {
        const Problem p(37, 21, k, 5, 3.0f);
        // Activations.
        std::vector<std::int8_t> xq = p.xq.download();
        std::vector<float> sx = p.sx.download();
        bool exact = true;
        for (std::size_t r = 0; r < p.t; ++r) {
            std::vector<std::int8_t> want(k);
            float scale = 0;
            quantize_row(&p.xf[r * k], k, want.data(), &scale);
            exact = exact && std::memcmp(want.data(), &xq[r * k], k) == 0 && std::memcmp(&scale, &sx[r], sizeof(float)) == 0;
        }
        CHECK(exact);
        // Weights, both layouts: the plain one as the oracle has it, the other the same bytes permuted within every group of 16.
        std::vector<std::int8_t> wq = p.wq.download(), wqp = p.wqp.download();
        std::vector<float> sw = p.sw.download(), swp = p.swp.download();
        bool w_exact = true, perm_exact = true;
        for (std::size_t r = 0; r < p.n; ++r) {
            std::vector<std::int8_t> want(k);
            float scale = 0;
            quantize_row(&p.wf[r * k], k, want.data(), &scale);
            w_exact = w_exact && std::memcmp(want.data(), &wq[r * k], k) == 0 && std::memcmp(&scale, &sw[r], sizeof(float)) == 0 &&
                      std::memcmp(&scale, &swp[r], sizeof(float)) == 0;
            for (std::size_t g = 0; g < k; g += 16) {
                for (int q = 0; q < 4; ++q) {
                    const int src[4] = {2 * q, 2 * q + 1, 2 * q + 8, 2 * q + 9};
                    for (int e = 0; e < 4; ++e) perm_exact = perm_exact && wqp[r * k + g + static_cast<std::size_t>(4 * q + e)] == want[g + static_cast<std::size_t>(src[e])];
                }
            }
        }
        CHECK(w_exact);
        CHECK(perm_exact);
    }
}

SEPAL_TEST(s8_quantize, zero_rows_outliers_and_the_extremes_of_binary16) {
    const std::size_t k = 64;
    std::vector<__half> x(4 * k, __float2half(0.0f));
    for (std::size_t i = 0; i < k; ++i) x[k + i] = __float2half(i == 5 ? 65504.0f : 0.01f * static_cast<float>(i));   // a huge outlier among small values
    for (std::size_t i = 0; i < k; ++i) x[2 * k + i] = __float2half(i % 2 == 0 ? -1.0f : 1.0f);                          // exactly +-127
    for (std::size_t i = 0; i < k; ++i) x[3 * k + i] = __float2half(i == 0 ? 6.0e-8f : 0.0f);                            // a subnormal
    DeviceBuffer<__half> dx(x);
    DeviceBuffer<std::int8_t> q(4 * k);
    DeviceBuffer<float> s(4);
    sepal::cuda::launch_quantize_rows_s8(dx.data(), k, q.data(), k, s.data(), 4, k, stream());
    sync();
    const auto qh = q.download();
    const auto sh = s.download();
    CHECK_EQ(sh[0], 0.0f);
    for (std::size_t i = 0; i < k; ++i) CHECK_EQ(static_cast<int>(qh[i]), 0);
    CHECK(std::fabs(sh[1] - 65504.0f / 127.0f) <= 1e-3f);
    CHECK_EQ(static_cast<int>(qh[k + 5]), 127);
    CHECK_EQ(static_cast<int>(qh[k + 6]), 0); // below half a step of the outlier's
    for (std::size_t i = 0; i < k; ++i) CHECK_EQ(static_cast<int>(qh[2 * k + i]), i % 2 == 0 ? -127 : 127);
    CHECK_EQ(static_cast<int>(qh[3 * k]), 127);
    for (const float v : sh) CHECK(std::isfinite(v));
}

SEPAL_TEST(s8_gemm, the_int8_multiply_is_the_exact_integer_product_scaled_in_float32) {
    struct Case {
        std::size_t t, n, k;
    };
    for (const Case c : {Case{100, 200, 1024}, Case{37, 66, 3584}, Case{5, 40, 6144}, Case{64, 96, 1040}, Case{300, 130, 2048}}) {
        const Problem p(c.t, c.n, c.k, 11);
        const std::vector<float> got = p.run_s8(0, c.t, -1);
        const auto xq = p.xq.download(), wq = p.wq.download();
        const auto sx = p.sx.download(), sw = p.sw.download();
        std::size_t differing = 0;
        for (std::size_t r = 0; r < c.t; ++r)
            for (std::size_t col = 0; col < c.n; ++col) {
                std::int64_t acc = 0;
                for (std::size_t i = 0; i < c.k; ++i) acc += static_cast<std::int64_t>(xq[r * c.k + i]) * wq[col * c.k + i];
                const float want = static_cast<float>(static_cast<std::int32_t>(acc)) * sx[r] * sw[col];
                if (std::memcmp(&want, &got[r * c.n + col], sizeof(float)) != 0) ++differing;
            }
        CHECK_EQ(differing, std::size_t{0});
        const double err = relative_rms_error(got, exact_product(p));
        std::printf("  W8A8 T=%zu N=%zu K=%zu: relative rms error against the unquantised product %.3e\n", c.t, c.n, c.k, err);
        CHECK(err < 2.0e-2);
        // The binary16 result is the float32 one rounded.
        DeviceBuffer<__half> yh(c.t * c.n);
        sepal::cuda::launch_gemm_s8(p.xq.data(), c.k, p.sx.data(), p.wq.data(), c.k, p.sw.data(), yh.data(), c.n, true, c.t, c.n, c.k, -1, stream());
        sync();
        const auto half = yh.download();
        std::size_t half_differing = 0;
        for (std::size_t i = 0; i < got.size(); ++i) half_differing += __half2float(half[i]) != __half2float(__float2half(got[i])) ? 1 : 0;
        CHECK_EQ(half_differing, std::size_t{0});
    }
}

SEPAL_TEST(s8_gemm, the_w8a16_multiply_against_float64_of_the_dequantised_weights) {
    struct Case {
        std::size_t t, n, k;
    };
    for (const Case c : {Case{100, 200, 1024}, Case{37, 66, 3584}, Case{5, 40, 6144}, Case{64, 96, 1040}, Case{300, 130, 2048}}) {
        const Problem p(c.t, c.n, c.k, 12);
        const std::vector<float> got = p.run_w8a16(0, c.t, -1);
        const auto wq = p.wq.download();
        const auto sw = p.sw.download();
        std::vector<double> want(c.t * c.n);
        for (std::size_t r = 0; r < c.t; ++r)
            for (std::size_t col = 0; col < c.n; ++col) {
                double acc = 0;
                for (std::size_t i = 0; i < c.k; ++i) acc += static_cast<double>(p.xf[r * c.k + i]) * wq[col * c.k + i];
                want[r * c.n + col] = acc * sw[col];
            }
        const double err = relative_rms_error(got, want);
        const double quantisation = relative_rms_error(got, exact_product(p));
        std::printf("  W8A16 T=%zu N=%zu K=%zu: against the dequantised weights %.3e, against the unquantised product %.3e\n", c.t, c.n, c.k, err, quantisation);
        CHECK(err < 1.0e-5);
        CHECK(quantisation < 1.2e-2);
    }
}

SEPAL_TEST(s8_gemm, outliers_in_the_activations_cost_accuracy_in_the_int8_multiply_only) {
    // One huge channel per row: a per-row scale spends the whole int8 range on it. W8A16 does not mind.
    const std::size_t t = 32, n = 64, k = 512;
    Problem p(t, n, k, 3);
    std::vector<__half> x = p.x;
    for (std::size_t r = 0; r < t; ++r) x[r * k + 7] = __float2half(300.0f);
    DeviceBuffer<__half> dx(x);
    DeviceBuffer<std::int8_t> xq(t * k);
    DeviceBuffer<float> sx(t);
    sepal::cuda::launch_quantize_rows_s8(dx.data(), k, xq.data(), k, sx.data(), t, k, stream());
    DeviceBuffer<float> y8(t * n), y16(t * n);
    sepal::cuda::launch_gemm_s8(xq.data(), k, sx.data(), p.wq.data(), k, p.sw.data(), y8.data(), n, false, t, n, k, -1, stream());
    sepal::cuda::launch_gemm_w8a16(dx.data(), k, p.wqp.data(), k, p.swp.data(), y16.data(), n, false, t, n, k, -1, stream());
    sync();
    std::vector<double> want(t * n);
    for (std::size_t r = 0; r < t; ++r)
        for (std::size_t c = 0; c < n; ++c) {
            double acc = 0;
            for (std::size_t i = 0; i < k; ++i) acc += static_cast<double>(__half2float(x[r * k + i])) * p.wf[c * k + i];
            want[r * n + c] = acc;
        }
    const double e8 = relative_rms_error(y8.download(), want), e16 = relative_rms_error(y16.download(), want);
    std::printf("  with a 300-sigma channel: W8A8 %.3e, W8A16 %.3e\n", e8, e16);
    CHECK(e16 < 1.2e-2);
    CHECK(std::isfinite(e8));
}

SEPAL_TEST(s8_invariance, a_rows_bits_do_not_depend_on_the_tile_shape_the_batch_or_the_place) {
    const std::size_t t = 213, n = 258, k = 1056;
    const Problem p(t, n, k, 77);
    const Problem tail(97, 66, 1040, 78);
    for (const bool w8a16 : {false, true}) {
        for (const Problem* q : {&p, &tail}) {
            const auto run = [&](std::size_t row0, std::size_t rows, int config) { return w8a16 ? q->run_w8a16(row0, rows, config) : q->run_s8(row0, rows, config); };
            const std::vector<float> all = run(0, q->t, -1);
            for (int config = 0; config <= 5; ++config) {
                const std::vector<float> other = run(0, q->t, config);
                CHECK(same_bits(all.data(), other.data(), all.size()));
            }
            std::mt19937 rng(3);
            for (int trial = 0; trial < 12; ++trial) {
                const std::size_t row = rng() % q->t;
                const std::size_t rows = 1 + rng() % std::min<std::size_t>(40, q->t - row);
                const std::vector<float> part = run(row, rows, -1);
                CHECK(same_bits(all.data() + row * q->n, part.data(), rows * q->n));
                const std::vector<float> alone = run(row, 1, -1);
                CHECK(same_bits(all.data() + row * q->n, alone.data(), q->n));
            }
        }
    }
}

SEPAL_TEST(s8_edge, tiny_shapes_zero_rows_and_refused_arguments) {
    struct S {
        std::size_t t, n, k;
    };
    for (const S s : {S{1, 2, 16}, S{1, 130, 64}, S{129, 2, 96}, S{127, 258, 32}, S{2, 6, 160}, S{3, 4, 16}, S{17, 10, 48}, S{33, 34, 1040}}) {
        const Problem p(s.t, s.n, s.k, 21);
        const std::vector<double> want = exact_product(p);
        for (int config = -1; config <= 5; ++config) {
            CHECK(relative_rms_error(p.run_s8(0, s.t, config), want) <= 3.0e-2);
            CHECK(relative_rms_error(p.run_w8a16(0, s.t, config), want) <= 3.0e-2);
        }
    }
    // A row of zeros gives zeros, whatever its neighbours are.
    {
        Problem p(8, 32, 64, 4);
        std::vector<__half> x = p.x;
        for (std::size_t i = 0; i < 64; ++i) x[3 * 64 + i] = __float2half(0.0f);
        DeviceBuffer<__half> dx(x);
        DeviceBuffer<std::int8_t> xq(8 * 64);
        DeviceBuffer<float> sx(8), y(8 * 32);
        sepal::cuda::launch_quantize_rows_s8(dx.data(), 64, xq.data(), 64, sx.data(), 8, 64, stream());
        sepal::cuda::launch_gemm_s8(xq.data(), 64, sx.data(), p.wq.data(), 64, p.sw.data(), y.data(), 32, false, 8, 32, 64, -1, stream());
        sync();
        const auto yh = y.download();
        for (std::size_t c = 0; c < 32; ++c) CHECK_EQ(yh[3 * 32 + c], 0.0f);
        bool nonzero = false;
        for (std::size_t c = 0; c < 32; ++c) nonzero = nonzero || yh[32 + c] != 0.0f;
        CHECK(nonzero);
    }
    const Problem p(4, 8, 64, 1);
    DeviceBuffer<float> y(32);
    DeviceBuffer<__half> yh(32);
    const auto refused = [&](auto&& call) {
        try {
            call();
        } catch (const std::invalid_argument&) {
            return true;
        }
        return false;
    };
    using namespace sepal::cuda;
    CHECK(refused([&] { launch_gemm_s8(p.xq.data(), 64, p.sx.data(), p.wq.data(), 64, p.sw.data(), y.data(), 8, false, 4, 8, 72, -1, stream()); }));  // K not a multiple of 16
    CHECK(refused([&] { launch_gemm_s8(p.xq.data(), 64, p.sx.data(), p.wq.data(), 64, p.sw.data(), y.data(), 7, false, 4, 7, 64, -1, stream()); }));  // odd N
    CHECK(refused([&] { launch_gemm_s8(p.xq.data() + 1, 64, p.sx.data(), p.wq.data(), 64, p.sw.data(), y.data(), 8, false, 4, 8, 64, -1, stream()); })); // misaligned X
    CHECK(refused([&] { launch_gemm_s8(p.xq.data(), 64, nullptr, p.wq.data(), 64, p.sw.data(), y.data(), 8, false, 4, 8, 64, -1, stream()); }));
    CHECK(refused([&] { launch_gemm_s8(p.xq.data(), 56, p.sx.data(), p.wq.data(), 64, p.sw.data(), y.data(), 8, false, 4, 8, 64, -1, stream()); })); // ld_x not a multiple of 16
    CHECK(refused([&] { launch_gemm_s8(p.xq.data(), 64, p.sx.data(), p.wq.data(), 64, p.sw.data(), y.data(), 8, false, 4, 8, 64, 9, stream()); }));  // unknown configuration
    CHECK(refused([&] { launch_gemm_w8a16(p.dx.data(), 64, p.wqp.data(), 64, p.swp.data(), y.data(), 8, false, 4, 8, 72, -1, stream()); }));
    CHECK(refused([&] { launch_gemm_w8a16(p.dx.data(), 64, p.wqp.data(), 64, p.swp.data(), y.data(), 7, false, 4, 7, 64, -1, stream()); }));
    CHECK(refused([&] { launch_gemm_w8a16(p.dx.data() + 1, 64, p.wqp.data(), 64, p.swp.data(), y.data(), 8, false, 4, 8, 64, -1, stream()); }));
    CHECK(refused([&] { launch_gemm_w8a16(p.dx.data(), 64, p.wqp.data(), 64, nullptr, y.data(), 8, false, 4, 8, 64, -1, stream()); }));
    CHECK(refused([&] { launch_quantize_rows_s8(p.dx.data(), 64, p.xq.data(), 64, p.sx.data(), 4, 72, stream()); }));
    CHECK(refused([&] { launch_quantize_rows_s8(p.dx.data(), 64, p.xq.data() + 1, 64, p.sx.data(), 4, 64, stream()); }));
    CHECK(refused([&] { launch_quantize_weights_s8(p.dw.data(), 64, p.wq.data(), p.sw.data(), 8, 56, S8WeightLayout::Plain, stream()); }));
    CHECK(refused([&] { launch_quantize_weights_s8(nullptr, 64, p.wq.data(), p.sw.data(), 8, 64, S8WeightLayout::Plain, stream()); }));
    CHECK(gemm_s8_supports(8, 64, 64, 64, 8));
    CHECK(!gemm_s8_supports(8, 72, 72, 72, 8));
    CHECK(gemm_w8a16_supports(8, 64, 72, 64, 8));
    CHECK(!gemm_w8a16_supports(8, 56, 56, 56, 8));
    // Zero rows or columns: nothing to do.
    launch_gemm_s8(p.xq.data(), 64, p.sx.data(), p.wq.data(), 64, p.sw.data(), y.data(), 8, false, 0, 8, 64, -1, stream());
    launch_gemm_w8a16(p.dx.data(), 64, p.wqp.data(), 64, p.swp.data(), y.data(), 8, false, 0, 8, 64, -1, stream());
    launch_quantize_rows_s8(p.dx.data(), 64, p.xq.data(), 64, p.sx.data(), 0, 64, stream());
    sync();
}

// ---- smoothing and calibration ----

namespace {

std::vector<float> random_factors(std::mt19937& rng, std::size_t k) {
    std::uniform_real_distribution<float> ud(0.1f, 10.0f);
    std::vector<float> v(k);
    for (float& e : v) e = ud(rng);
    return v;
}

bool refused_call(const std::function<void()>& call) {
    try {
        call();
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

} // namespace

SEPAL_TEST(s8_smoothing, the_factors_enter_both_quantisers_as_a_float32_product_and_ones_change_nothing) {
    std::mt19937 rng(8);
    for (const std::size_t k : {16u, 48u, 1024u, 3584u}) {
        const Problem p(29, 19, k, 6, 3.0f);
        const std::vector<float> s = random_factors(rng, k);
        const std::vector<float> ones(k, 1.0f);
        const DeviceBuffer<float> ds(s), d1(ones);
        // Activations: x * s[k] in float32, then the same rule as without.
        DeviceBuffer<std::int8_t> xq(p.t * k);
        DeviceBuffer<float> sx(p.t);
        sepal::cuda::launch_quantize_rows_s8(p.dx.data(), k, xq.data(), k, sx.data(), p.t, k, stream(), ds.data());
        sync();
        const auto got = xq.download();
        const auto got_scale = sx.download();
        bool exact = true;
        for (std::size_t r = 0; r < p.t; ++r) {
            std::vector<float> scaled(k);
            for (std::size_t i = 0; i < k; ++i) scaled[i] = p.xf[r * k + i] * s[i];
            std::vector<std::int8_t> want(k);
            float scale = 0;
            quantize_row(scaled.data(), k, want.data(), &scale);
            exact = exact && std::memcmp(want.data(), &got[r * k], k) == 0 && std::memcmp(&scale, &got_scale[r], sizeof(float)) == 0;
        }
        CHECK(exact);
        // Weights, both layouts.
        for (const S8WeightLayout layout : {S8WeightLayout::Plain, S8WeightLayout::Fp16Pairs}) {
            DeviceBuffer<std::int8_t> wq(p.n * k);
            DeviceBuffer<float> sw(p.n);
            sepal::cuda::launch_quantize_weights_s8(p.dw.data(), k, wq.data(), sw.data(), p.n, k, layout, stream(), ds.data());
            sync();
            const auto wh = wq.download();
            const auto swh = sw.download();
            bool w_exact = true;
            for (std::size_t r = 0; r < p.n; ++r) {
                std::vector<float> scaled(k);
                for (std::size_t i = 0; i < k; ++i) scaled[i] = p.wf[r * k + i] * s[i];
                std::vector<std::int8_t> want(k);
                float scale = 0;
                quantize_row(scaled.data(), k, want.data(), &scale);
                w_exact = w_exact && std::memcmp(&scale, &swh[r], sizeof(float)) == 0;
                for (std::size_t g = 0; g < k; g += 16) {
                    for (std::size_t e = 0; e < 16; ++e) {
                        const std::size_t pos = layout == S8WeightLayout::Plain ? e : 4 * ((e & 7) >> 1) + (e & 1) + 2 * (e >> 3);
                        w_exact = w_exact && wh[r * k + g + pos] == want[g + e];
                    }
                }
            }
            CHECK(w_exact);
        }
        // Factors of one change nothing, bit for bit.
        DeviceBuffer<std::int8_t> xq1(p.t * k), wq1(p.n * k);
        DeviceBuffer<float> sx1(p.t), sw1(p.n);
        sepal::cuda::launch_quantize_rows_s8(p.dx.data(), k, xq1.data(), k, sx1.data(), p.t, k, stream(), d1.data());
        sepal::cuda::launch_quantize_weights_s8(p.dw.data(), k, wq1.data(), sw1.data(), p.n, k, S8WeightLayout::Plain, stream(), d1.data());
        sync();
        CHECK(xq1.download() == p.xq.download());
        CHECK(wq1.download() == p.wq.download());
        CHECK(same_bits(sx1.download().data(), p.sx.download().data(), p.t));
        CHECK(same_bits(sw1.download().data(), p.sw.download().data(), p.n));
    }
}

SEPAL_TEST(s8_smoothing, dividing_the_large_channels_out_of_the_activations_lowers_the_error_of_the_int8_multiply) {
    // Four channels 200 times larger than the rest in every row: a row's scale is set by them. Factors s = sqrt(max|x| / max|w|) per channel
    // (the SmoothQuant rule, alpha 1/2) move most of that into the weights.
    const std::size_t t = 64, n = 96, k = 1024;
    Problem p(t, n, k, 9);
    std::vector<__half> x = p.x;
    std::mt19937 rng(10);
    for (std::size_t r = 0; r < t; ++r)
        for (const std::size_t c : {17u, 300u, 301u, 900u}) x[r * k + c] = __float2half(200.0f * (0.5f + 0.001f * static_cast<float>(rng() % 1000)));
    DeviceBuffer<__half> dx(x);
    std::vector<float> xf(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) xf[i] = __half2float(x[i]);
    std::vector<float> s(k, 1.0f);
    for (std::size_t c = 0; c < k; ++c) {
        float xm = 0, wm = 0;
        for (std::size_t r = 0; r < t; ++r) xm = std::max(xm, std::fabs(xf[r * k + c]));
        for (std::size_t r = 0; r < n; ++r) wm = std::max(wm, std::fabs(p.wf[r * k + c]));
        s[c] = std::sqrt(xm / wm);
    }
    std::vector<double> want(t * n);
    for (std::size_t r = 0; r < t; ++r)
        for (std::size_t c = 0; c < n; ++c) {
            double acc = 0;
            for (std::size_t i = 0; i < k; ++i) acc += static_cast<double>(xf[r * k + i]) * p.wf[c * k + i];
            want[r * n + c] = acc;
        }
    std::vector<float> inv(k);
    for (std::size_t c = 0; c < k; ++c) inv[c] = 1.0f / s[c];
    const DeviceBuffer<float> ds(s), dinv(inv);
    double err[2];
    for (int smoothed = 0; smoothed < 2; ++smoothed) {
        DeviceBuffer<std::int8_t> xq(t * k), wq(n * k);
        DeviceBuffer<float> sx(t), sw(n), y(t * n);
        sepal::cuda::launch_quantize_rows_s8(dx.data(), k, xq.data(), k, sx.data(), t, k, stream(), smoothed ? dinv.data() : nullptr);
        sepal::cuda::launch_quantize_weights_s8(p.dw.data(), k, wq.data(), sw.data(), n, k, S8WeightLayout::Plain, stream(), smoothed ? ds.data() : nullptr);
        sepal::cuda::launch_gemm_s8(xq.data(), k, sx.data(), wq.data(), k, sw.data(), y.data(), n, false, t, n, k, -1, stream());
        sync();
        err[smoothed] = relative_rms_error(y.download(), want);
    }
    std::printf("  with four 200-sigma channels: W8A8 %.3e, smoothed %.3e\n", err[0], err[1]);
    CHECK(err[1] < err[0] / 3.0);
    CHECK(err[1] < 2.0e-2);
}

SEPAL_TEST(s8_smoothing, channel_statistics_and_column_maxima_match_the_host) {
    const std::size_t rows = 77, cols = 200, ld = 208;
    std::mt19937 rng(12);
    std::vector<__half> x(rows * ld, __float2half(0.0f));
    std::normal_distribution<float> nd(0.0f, 2.0f);
    for (std::size_t r = 0; r < rows; ++r)
        for (std::size_t c = 0; c < cols; ++c) x[r * ld + c] = __float2half(nd(rng));
    DeviceBuffer<__half> dx(x);
    DeviceBuffer<float> amax(cols), cmax(cols);
    DeviceBuffer<double> sums(2 * cols);
    amax.zero();
    sums.zero();
    // Two calls (a split of the rows) add up to one pass over all of them.
    sepal::cuda::launch_channel_stats(dx.data(), ld, 30, cols, amax.data(), sums.data(), stream());
    sepal::cuda::launch_channel_stats(dx.data() + 30 * ld, ld, rows - 30, cols, amax.data(), sums.data(), stream());
    sepal::cuda::launch_column_absmax(dx.data(), ld, cmax.data(), rows, cols, stream());
    sync();
    const auto a = amax.download(), m = cmax.download();
    const auto s = sums.download();
    bool ok = true;
    for (std::size_t c = 0; c < cols; ++c) {
        float want_max = 0;
        double want_abs = 0, want_sq = 0;
        for (std::size_t r = 0; r < rows; ++r) {
            const float v = __half2float(x[r * ld + c]);
            want_max = std::max(want_max, std::fabs(v));
            want_abs += std::fabs(static_cast<double>(v));
            want_sq += static_cast<double>(v) * v;
        }
        ok = ok && a[c] == want_max && m[c] == want_max && std::fabs(s[c] - want_abs) <= 1e-9 * want_abs && std::fabs(s[cols + c] - want_sq) <= 1e-9 * want_sq;
    }
    CHECK(ok);
    // The same calls again give the same bits.
    DeviceBuffer<float> amax2(cols);
    DeviceBuffer<double> sums2(2 * cols);
    amax2.zero();
    sums2.zero();
    sepal::cuda::launch_channel_stats(dx.data(), ld, 30, cols, amax2.data(), sums2.data(), stream());
    sepal::cuda::launch_channel_stats(dx.data() + 30 * ld, ld, rows - 30, cols, amax2.data(), sums2.data(), stream());
    sync();
    CHECK(sums2.download() == sums.download());
    // Refusals.
    CHECK(refused_call([&] { sepal::cuda::launch_channel_stats(nullptr, ld, rows, cols, amax.data(), sums.data(), stream()); }));
    CHECK(refused_call([&] { sepal::cuda::launch_channel_stats(dx.data(), 100, rows, cols, amax.data(), sums.data(), stream()); }));
    CHECK(refused_call([&] { sepal::cuda::launch_column_absmax(dx.data(), 100, cmax.data(), rows, cols, stream()); }));
    // A misaligned factor vector is refused.
    const Problem q(4, 8, 64, 1);
    const DeviceBuffer<float> f(std::vector<float>(80, 1.0f));
    CHECK(refused_call([&] { sepal::cuda::launch_quantize_rows_s8(q.dx.data(), 64, q.xq.data(), 64, q.sx.data(), 4, 64, stream(), f.data() + 1); }));
    CHECK(refused_call([&] { sepal::cuda::launch_quantize_weights_s8(q.dw.data(), 64, q.wq.data(), q.sw.data(), 8, 64, S8WeightLayout::Plain, stream(), f.data() + 1); }));
}
