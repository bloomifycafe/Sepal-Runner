// Tests of the multiply kernels (src/gemm_kernels.*): the 2-D binary16 -> float32 widening and the batch-invariant
// tensor-core multiply (float32 accumulators, or binary16 accumulators with a float32 flush). Needs a CUDA device.
//
//   sepal_cuda_kernel_gemm_tests              every suite
//   sepal_cuda_kernel_gemm_tests <suite>      one of: gemmk_convert, gemmk_accuracy, gemmk_invariance, gemmk_edge
//
// What is compared with what:
//   - accuracy: the float64 product of the (binary16) operands; the figure is the rms of the error over the rms of the
//     result, printed by every case. Operands: X ~ N(0, 1), W ~ N(0, 1/K), so results have rms about 1.
//   - invariance: bit for bit, across tile shapes, batch sizes and places in the batch.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "device.hpp"
#include "gemm_kernels.hpp"
#include "test.hpp"

namespace {

using sepal::cuda::DeviceBuffer;
using sepal::cuda::GemmAccumulate;
using sepal::cuda::launch_gemm_nt;
using sepal::cuda::launch_half_to_float_2d;

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

// Y[t, n] in float64 of binary16 operands.
std::vector<double> reference(const std::vector<__half>& x, const std::vector<__half>& w, std::size_t t, std::size_t n, std::size_t k) {
    std::vector<float> xf(x.size()), wf(w.size());
    for (std::size_t i = 0; i < x.size(); ++i) xf[i] = __half2float(x[i]);
    for (std::size_t i = 0; i < w.size(); ++i) wf[i] = __half2float(w[i]);
    std::vector<double> y(t * n);
    for (std::size_t r = 0; r < t; ++r) {
        for (std::size_t c = 0; c < n; ++c) {
            double acc = 0;
            for (std::size_t i = 0; i < k; ++i) acc += static_cast<double>(xf[r * k + i]) * wf[c * k + i];
            y[r * n + c] = acc;
        }
    }
    return y;
}

double relative_rms_error(const std::vector<float>& got, const std::vector<double>& want) {
    double err = 0, ref = 0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double d = static_cast<double>(got[i]) - want[i];
        err += d * d;
        ref += want[i] * want[i];
    }
    return std::sqrt(err / std::max(ref, 1e-300));
}

struct Problem {
    std::size_t t, n, k;
    std::vector<__half> x, w;
    DeviceBuffer<__half> dx, dw;
    Problem(std::size_t t_, std::size_t n_, std::size_t k_, std::uint32_t seed) : t(t_), n(n_), k(k_) {
        std::mt19937 rng(seed);
        x = random_half(rng, t * k, 1.0f);
        w = random_half(rng, n * k, 1.0f / std::sqrt(static_cast<float>(k)));
        dx = DeviceBuffer<__half>(x);
        dw = DeviceBuffer<__half>(w);
    }
    // Runs the kernel on rows [row0, row0 + rows) of X; float32 result.
    std::vector<float> run(std::size_t row0, std::size_t rows, GemmAccumulate mode, unsigned flush_k, int config) const {
        DeviceBuffer<float> y(rows * n);
        launch_gemm_nt(dx.data() + row0 * k, k, dw.data(), k, y.data(), n, false, rows, n, k, mode, flush_k, config, stream());
        sync();
        return y.download();
    }
    std::vector<__half> run_half(GemmAccumulate mode, unsigned flush_k) const {
        DeviceBuffer<__half> y(t * n);
        launch_gemm_nt(dx.data(), k, dw.data(), k, y.data(), n, true, t, n, k, mode, flush_k, -1, stream());
        sync();
        return y.download();
    }
};

bool same_bits(const float* a, const float* b, std::size_t count) { return std::memcmp(a, b, count * sizeof(float)) == 0; }

} // namespace

SEPAL_TEST(gemmk_convert, widening_with_strides_matches_the_host) {
    std::mt19937 rng(5);
    struct Shape {
        std::size_t rows, cols, ld_src, ld_dst;
    };
    // The first two take the vectorised path, the others the scalar one (odd widths, strides, unaligned pointers).
    for (const Shape s : {Shape{5, 24, 32, 28}, Shape{300, 128, 128, 128}, Shape{7, 13, 13, 15}, Shape{3, 8, 9, 8}, Shape{1, 1, 1, 1}}) {
        const std::vector<__half> src = random_half(rng, s.rows * s.ld_src + 1, 3.0f);
        DeviceBuffer<__half> dsrc(src);
        DeviceBuffer<float> ddst(s.rows * s.ld_dst + 1);
        ddst.zero();
        // Start one element in so that the unaligned cases really are unaligned.
        const std::size_t shift = s.cols % 8 == 0 ? 0 : 1;
        launch_half_to_float_2d(dsrc.data() + shift, s.ld_src, ddst.data() + shift, s.ld_dst, s.rows, s.cols, stream());
        sync();
        const std::vector<float> got = ddst.download();
        bool exact = true, untouched = true;
        for (std::size_t r = 0; r < s.rows; ++r) {
            for (std::size_t c = 0; c < s.ld_dst; ++c) {
                const float have = got[shift + r * s.ld_dst + c];
                if (c < s.cols) {
                    exact = exact && have == __half2float(src[shift + r * s.ld_src + c]);
                } else {
                    untouched = untouched && have == 0.0f;
                }
            }
        }
        CHECK(exact);
        CHECK(untouched);
    }
}

SEPAL_TEST(gemmk_accuracy, both_accumulation_modes_against_float64) {
    struct Case {
        std::size_t t, n, k;
        GemmAccumulate mode;
        unsigned flush;
        double bound; // relative rms error allowed
    };
    // float32 accumulation is limited by the binary16 rounding of nothing at all here (the operands are exact), so it is
    // a few 1e-6 (the tensor cores truncate while accumulating). The error of a binary16 chain grows with the square root of its length: about 2e-3 for K = 1024 summed in
    // binary16 alone, about 4e-4 with a flush every 64.
    const GemmAccumulate f32 = GemmAccumulate::Float32, f16 = GemmAccumulate::Float16Flush;
    for (const Case c : {Case{100, 200, 1024, f32, 0, 1e-5}, Case{37, 66, 3584, f32, 0, 1e-5}, Case{100, 200, 1024, f16, 64, 1e-3},
                         Case{37, 66, 3584, f16, 128, 1.5e-3}, Case{300, 130, 2048, f16, 256, 2.5e-3}, Case{64, 96, 1024, f16, 0, 4e-3},
                         Case{5, 40, 6144, f16, 128, 1.8e-3}, Case{9, 24, 24, f16, 128, 1e-3}, Case{50, 8, 104, f32, 0, 1e-5}}) {
        const Problem p(c.t, c.n, c.k, 11);
        const std::vector<double> want = reference(p.x, p.w, c.t, c.n, c.k);
        const std::vector<float> got = p.run(0, c.t, c.mode, c.flush, -1);
        const double err = relative_rms_error(got, want);
        std::printf("  T=%zu N=%zu K=%zu %s flush %u: relative rms error %.3e (allowed %.1e)\n", c.t, c.n, c.k,
                    c.mode == f32 ? "float32" : "binary16", c.flush, err, c.bound);
        CHECK(err <= c.bound);
        for (const float v : got) CHECK(std::isfinite(v));
        // The binary16 output is the float32 output rounded to binary16.
        const std::vector<__half> half = p.run_half(c.mode, c.flush);
        std::size_t differing = 0;
        for (std::size_t i = 0; i < got.size(); ++i) differing += __half2float(half[i]) != __half2float(__float2half(got[i])) ? 1 : 0;
        CHECK_EQ(differing, std::size_t{0});
    }
}

SEPAL_TEST(gemmk_invariance, a_rows_bits_do_not_depend_on_the_tile_shape_the_batch_or_the_place) {
    // K and N chosen so that several tile shapes are exercised, with ragged edges; the second problem has K = 1048, not a multiple of 32.
    const std::size_t t = 213, n = 258, k = 1056;
    const Problem p(t, n, k, 77);
    const Problem tail(97, 66, 1048, 78);
    for (const GemmAccumulate mode : {GemmAccumulate::Float32, GemmAccumulate::Float16Flush}) {
        for (const unsigned flush : {0u, 64u, 128u}) {
            if (mode == GemmAccumulate::Float32 && flush != 0) continue;
            for (const Problem* q : {&p, &tail}) {
                const std::vector<float> all = q->run(0, q->t, mode, flush, -1);
                // Every tile shape gives the same bits.
                for (int config = 0; config <= 5; ++config) {
                    const std::vector<float> other = q->run(0, q->t, mode, flush, config);
                    CHECK(same_bits(all.data(), other.data(), all.size()));
                }
                // A row alone, in a short batch, and at another place in a different batch: the same bits.
                std::mt19937 rng(3 + flush);
                for (int trial = 0; trial < 12; ++trial) {
                    const std::size_t row = rng() % q->t;
                    const std::size_t rows = 1 + rng() % std::min<std::size_t>(40, q->t - row);
                    const std::vector<float> part = q->run(row, rows, mode, flush, -1);
                    CHECK(same_bits(all.data() + row * q->n, part.data(), rows * q->n));
                    const std::vector<float> alone = q->run(row, 1, mode, flush, -1);
                    CHECK(same_bits(all.data() + row * q->n, alone.data(), q->n));
                }
            }
        }
    }
}

SEPAL_TEST(gemmk_edge, tiny_shapes_and_refused_arguments) {
    // Smallest shapes: one row, two columns, one tile of k; and sizes that are not multiples of any tile.
    struct S {
        std::size_t t, n, k;
    };
    for (const S s : {S{1, 2, 32}, S{1, 130, 64}, S{129, 2, 96}, S{127, 258, 32}, S{2, 6, 160}, S{3, 4, 8}, S{17, 10, 40}}) {
        const Problem p(s.t, s.n, s.k, 21);
        const std::vector<double> want = reference(p.x, p.w, s.t, s.n, s.k);
        for (int config = -1; config <= 5; ++config) {
            CHECK(relative_rms_error(p.run(0, s.t, GemmAccumulate::Float16Flush, 64, config), want) <= 1.5e-3);
            CHECK(relative_rms_error(p.run(0, s.t, GemmAccumulate::Float32, 0, config), want) <= 1e-5);
        }
    }
    const Problem p(4, 8, 64, 1);
    DeviceBuffer<float> y(32);
    const GemmAccumulate f16 = GemmAccumulate::Float16Flush;
    const auto refused = [&](auto&& call) {
        try {
            call();
        } catch (const std::invalid_argument&) {
            return true;
        }
        return false;
    };
    CHECK(refused([&] { launch_gemm_nt(p.dx.data(), 64, p.dw.data(), 64, y.data(), 8, false, 4, 8, 12, f16, 64, -1, stream()); }));  // K not a multiple of 8
    CHECK(refused([&] { launch_gemm_nt(p.dx.data(), 64, p.dw.data(), 64, y.data(), 8, false, 4, 7, 64, f16, 64, -1, stream()); }));  // odd N
    CHECK(refused([&] { launch_gemm_nt(p.dx.data(), 64, p.dw.data(), 64, y.data(), 8, false, 4, 8, 64, f16, 48, -1, stream()); }));  // flush not a multiple of 32
    CHECK(refused([&] { launch_gemm_nt(p.dx.data() + 1, 64, p.dw.data(), 64, y.data(), 8, false, 4, 8, 64, f16, 64, -1, stream()); })); // misaligned X
    CHECK(refused([&] { launch_gemm_nt(nullptr, 64, p.dw.data(), 64, y.data(), 8, false, 4, 8, 64, f16, 64, -1, stream()); }));
    CHECK(refused([&] { launch_gemm_nt(p.dx.data(), 60, p.dw.data(), 64, y.data(), 8, false, 4, 8, 64, f16, 64, -1, stream()); })); // ld_x not a multiple of 8
    CHECK(refused([&] { launch_gemm_nt(p.dx.data(), 64, p.dw.data(), 64, y.data(), 8, false, 4, 8, 64, f16, 64, 9, stream()); }));  // unknown configuration
    CHECK(sepal::cuda::gemm_nt_supports(8, 64, 64, 64, 8));
    CHECK(!sepal::cuda::gemm_nt_supports(8, 12, 12, 12, 8));
    // Zero rows or columns: nothing to do.
    launch_gemm_nt(p.dx.data(), 64, p.dw.data(), 64, y.data(), 8, false, 0, 8, 64, f16, 64, -1, stream());
    sync();
}
