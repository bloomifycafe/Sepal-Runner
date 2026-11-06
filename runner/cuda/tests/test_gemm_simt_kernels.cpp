// Tests of the float32 SIMT multiply of the Pascal profiles (src/gemm_simt_kernels.*). Needs a CUDA device (any
// architecture from sm_61 up: the kernel uses no tensor cores).
//
//   sepal_cuda_kernel_simt_tests              every suite
//   sepal_cuda_kernel_simt_tests <suite>      one of: simt_exact, simt_invariance, simt_epilogue, simt_quant, simt_edge
//
// What is compared with what:
//   - exact: the kernel against a host emulation of its schedule (acc = fmaf(x, w, acc) for k = 0 .. K-1 from 0, with the
//     widened or dequantised weight), bit for bit; and the rms error against the float64 product, printed.
//   - invariance: bit for bit across the three tile shapes, batch sizes and a row's place in the batch.
//   - epilogue: Add in place and the fused SwiGLU against Store followed by launch_swiglu, bit for bit.
//   - quant: the group quantiser and the bfloat16 rounding against host emulations, bit for bit.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>

#include "device.hpp"
#include "gemm_simt_kernels.hpp"
#include "kernels_elementwise.hpp"
#include "test.hpp"

namespace {

using sepal::cuda::DeviceBuffer;
using sepal::cuda::SimtEpilogue;
using sepal::cuda::SimtMatrix;
using sepal::cuda::SimtWeight;

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

std::vector<float> random_float(std::mt19937& rng, std::size_t count, float scale) {
    std::normal_distribution<float> nd(0.0f, scale);
    std::vector<float> v(count);
    for (auto& e : v) e = nd(rng);
    return v;
}

std::uint16_t host_bf16(float f) {
    std::uint32_t bits;
    std::memcpy(&bits, &f, 4);
    if ((bits & 0x7F800000u) == 0x7F800000u && (bits & 0x007FFFFFu) != 0) return static_cast<std::uint16_t>((bits >> 16) | 0x40u);
    return static_cast<std::uint16_t>((bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16);
}
float bf16_value(std::uint16_t b) {
    const std::uint32_t bits = static_cast<std::uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

// The host's copy of the quantiser (same formulas, IEEE float arithmetic).
void host_quantize(const std::vector<float>& w, std::size_t rows, std::size_t k, std::vector<std::int8_t>& q, std::vector<float>& s) {
    const std::size_t groups = k / 128;
    q.assign(rows * k, 0);
    s.assign(rows * groups, 0.0f);
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t g = 0; g < groups; ++g) {
            float m = 0.0f;
            for (std::size_t i = 0; i < 128; ++i) m = std::max(m, std::fabs(w[r * k + g * 128 + i]));
            const float scale = m / 127.0f;
            s[r * groups + g] = scale;
            for (std::size_t i = 0; i < 128; ++i) {
                int v = 0;
                if (scale > 0.0f) v = static_cast<int>(std::min(std::max(std::nearbyint(w[r * k + g * 128 + i] / scale), -127.0f), 127.0f));
                q[r * k + g * 128 + i] = static_cast<std::int8_t>(v);
            }
        }
    }
}

// A multiply problem: X [T, K], the weights in one stored type, the dequantised float32 weights the kernel multiplies.
struct Problem {
    std::size_t t, n, k;
    SimtWeight type;
    std::vector<float> x, wd; // wd: the widened / dequantised weights, [N, K]
    DeviceBuffer<float> dx;
    DeviceBuffer<float> dw32;
    DeviceBuffer<std::uint16_t> dw16;
    DeviceBuffer<std::int8_t> dw8;
    DeviceBuffer<float> ds;
    Problem(std::size_t t_, std::size_t n_, std::size_t k_, SimtWeight type_, std::uint32_t seed) : t(t_), n(n_), k(k_), type(type_) {
        std::mt19937 rng(seed);
        x = random_float(rng, t * k, 1.0f);
        const std::vector<float> w = random_float(rng, n * k, 1.0f / std::sqrt(static_cast<float>(k)));
        dx = DeviceBuffer<float>(x);
        wd.resize(n * k);
        if (type == SimtWeight::F32) {
            wd = w;
            dw32 = DeviceBuffer<float>(w);
        } else if (type == SimtWeight::BF16) {
            std::vector<std::uint16_t> b(n * k);
            for (std::size_t i = 0; i < b.size(); ++i) {
                b[i] = host_bf16(w[i]);
                wd[i] = bf16_value(b[i]);
            }
            dw16 = DeviceBuffer<std::uint16_t>(b);
        } else {
            std::vector<std::int8_t> q;
            std::vector<float> s;
            host_quantize(w, n, k, q, s);
            for (std::size_t r = 0; r < n; ++r)
                for (std::size_t i = 0; i < k; ++i) wd[r * k + i] = static_cast<float>(q[r * k + i]) * s[r * (k / 128) + i / 128];
            dw8 = DeviceBuffer<std::int8_t>(q);
            ds = DeviceBuffer<float>(s);
        }
    }
    SimtMatrix matrix() const {
        SimtMatrix m;
        m.type = type;
        m.ld_w = k;
        if (type == SimtWeight::F32) m.w = dw32.data();
        else if (type == SimtWeight::BF16) m.w = dw16.data();
        else {
            m.w = dw8.data();
            m.scales = ds.data();
            m.ld_s = k / 128;
        }
        return m;
    }
    std::vector<float> run(std::size_t row0, std::size_t rows, int config) const {
        DeviceBuffer<float> y(rows * n);
        sepal::cuda::launch_gemm_simt(dx.data() + row0 * k, k, matrix(), y.data(), n, rows, n, k, SimtEpilogue::Store, config,
                                      stream());
        sync();
        return y.download();
    }
    // The schedule on the host: one fmaf chain per element, k increasing.
    float emulate(std::size_t r, std::size_t c) const {
        float acc = 0.0f;
        for (std::size_t i = 0; i < k; ++i) acc = std::fmaf(x[r * k + i], wd[c * k + i], acc);
        return acc;
    }
};

bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

const char* name_of(SimtWeight t) { return t == SimtWeight::F32 ? "f32" : t == SimtWeight::BF16 ? "bf16" : "int8g"; }

} // namespace

SEPAL_TEST(simt_exact, matches_the_host_schedule_bit_for_bit) {
    struct Shape {
        std::size_t t, n, k;
    };
    for (const SimtWeight type : {SimtWeight::F32, SimtWeight::BF16, SimtWeight::Int8Group}) {
        for (const Shape s : {Shape{37, 200, 256}, Shape{130, 128, 512}, Shape{600, 96, 384}, Shape{1, 260, 2560}}) {
            const Problem p(s.t, s.n, s.k, type, 11);
            const std::vector<float> got = p.run(0, s.t, -1);
            std::size_t mismatches = 0;
            double err = 0, ref = 0;
            for (std::size_t r = 0; r < s.t; ++r) {
                for (std::size_t c = 0; c < s.n; ++c) {
                    const float want = p.emulate(r, c);
                    const float have = got[r * s.n + c];
                    if (std::memcmp(&want, &have, 4) != 0) ++mismatches;
                    double exact = 0;
                    for (std::size_t i = 0; i < s.k; ++i) exact += static_cast<double>(p.x[r * s.k + i]) * p.wd[c * s.k + i];
                    err += (have - exact) * (have - exact);
                    ref += exact * exact;
                }
            }
            std::printf("  %-5s [%zu x %zu] x [%zu]: %zu of %zu differ from the host schedule; rms error vs float64 %.2e\n", name_of(type), s.t,
                        s.k, s.n, mismatches, s.t * s.n, std::sqrt(err / std::max(ref, 1e-300)));
            CHECK_EQ(mismatches, std::size_t{0});
            CHECK(std::sqrt(err / std::max(ref, 1e-300)) < 1e-5);
        }
    }
}

SEPAL_TEST(simt_invariance, tile_shape_batch_and_place_do_not_change_bits) {
    for (const SimtWeight type : {SimtWeight::F32, SimtWeight::BF16, SimtWeight::Int8Group}) {
        const Problem p(300, 300, 256, type, 23);
        const std::vector<float> base = p.run(0, 300, 0);
        CHECK(same_bits(base, p.run(0, 300, 1)));
        CHECK(same_bits(base, p.run(0, 300, 2)));
        CHECK(same_bits(base, p.run(0, 300, -1)));
        // A slice of the batch alone: the same bits as inside the batch.
        for (const std::size_t row0 : {std::size_t{0}, std::size_t{1}, std::size_t{129}, std::size_t{299}}) {
            for (const int config : {0, 1, 2}) {
                const std::size_t rows = std::min<std::size_t>(17, 300 - row0);
                const std::vector<float> part = p.run(row0, rows, config);
                CHECK(std::memcmp(part.data(), base.data() + row0 * 300, rows * 300 * sizeof(float)) == 0);
            }
        }
        // Run twice: the same bits.
        CHECK(same_bits(base, p.run(0, 300, 0)));
    }
}

SEPAL_TEST(simt_epilogue, add_in_place_and_fused_swiglu_match_the_unfused_path) {
    std::mt19937 rng(31);
    for (const SimtWeight type : {SimtWeight::BF16, SimtWeight::Int8Group}) {
        // Add, in place on the residual.
        {
            const Problem p(70, 136, 256, type, 41);
            const std::vector<float> plain = p.run(0, 70, -1);
            const std::vector<float> res = random_float(rng, 70 * 136, 2.0f);
            DeviceBuffer<float> dr(res);
            sepal::cuda::launch_gemm_simt(p.dx.data(), p.k, p.matrix(), dr.data(), 136, 70, 136, 256, SimtEpilogue::Add, -1, stream());
            sync();
            const std::vector<float> got = dr.download();
            bool exact = true;
            for (std::size_t i = 0; i < got.size(); ++i) exact = exact && got[i] == res[i] + plain[i];
            CHECK(exact);
        }
        // SwiGLU: gate/up rows packed in blocks of 64 against Store + launch_swiglu on the original order.
        {
            const std::size_t inter = 192, t = 75, k = 256;
            const Problem p(t, 2 * inter, k, type, 43); // rows 0..I-1 gate, I..2I-1 up
            const std::vector<float> unfused_mm = p.run(0, t, -1);
            DeviceBuffer<float> dmm(unfused_mm);
            DeviceBuffer<float> want(t * inter);
            sepal::cuda::launch_swiglu(dmm.data(), 2 * inter, dmm.data() + inter, 2 * inter, t, inter, want.data(), sepal::cuda::Elem::F32, inter, stream());
            // The packed weights: the same stored rows, reordered.
            SimtMatrix m = p.matrix();
            DeviceBuffer<std::uint16_t> p16;
            DeviceBuffer<std::int8_t> p8;
            DeviceBuffer<float> ps;
            const std::size_t rows = 2 * inter;
            if (type == SimtWeight::BF16) {
                const std::vector<std::uint16_t> src = p.dw16.download();
                std::vector<std::uint16_t> dst(src.size());
                for (std::size_t r = 0; r < rows; ++r)
                    std::memcpy(&dst[r * k], &src[sepal::cuda::pack_gate_up_source(r, inter) * k], k * 2);
                p16 = DeviceBuffer<std::uint16_t>(dst);
                m.w = p16.data();
            } else {
                const std::vector<std::int8_t> src = p.dw8.download();
                const std::vector<float> s = p.ds.download();
                std::vector<std::int8_t> dst(src.size());
                std::vector<float> ds(s.size());
                for (std::size_t r = 0; r < rows; ++r) {
                    const std::size_t from = sepal::cuda::pack_gate_up_source(r, inter);
                    std::memcpy(&dst[r * k], &src[from * k], k);
                    std::memcpy(&ds[r * (k / 128)], &s[from * (k / 128)], (k / 128) * 4);
                }
                p8 = DeviceBuffer<std::int8_t>(dst);
                ps = DeviceBuffer<float>(ds);
                m.w = p8.data();
                m.scales = ps.data();
            }
            DeviceBuffer<float> got(t * inter);
            for (const int config : {0, 1, 2}) {
                sepal::cuda::launch_gemm_simt(p.dx.data(), k, m, got.data(), inter, t, rows, k, SimtEpilogue::SwiGLU, config, stream());
                sync();
                CHECK(same_bits(got.download(), want.download()));
            }
        }
    }
}

SEPAL_TEST(simt_quant, quantiser_and_bf16_rounding_match_the_host) {
    std::mt19937 rng(53);
    const std::size_t rows = 37, k = 640;
    std::vector<float> w = random_float(rng, rows * k, 0.05f);
    for (std::size_t i = 0; i < 128; ++i) w[5 * k + 256 + i] = 0.0f; // an all-zero group
    w[7] = 3.0f;                                                         // an outlier
    DeviceBuffer<float> dw(w);
    DeviceBuffer<std::int8_t> dq(rows * k);
    DeviceBuffer<float> ds(rows * (k / 128));
    sepal::cuda::launch_quantize_group_s8(dw.data(), k, dq.data(), k, ds.data(), k / 128, rows, k, stream());
    sync();
    std::vector<std::int8_t> q;
    std::vector<float> s;
    host_quantize(w, rows, k, q, s);
    CHECK(dq.download() == q);
    CHECK(ds.download() == s);
    // Error of the dequantised weight: at most half a step of its group.
    double worst = 0;
    for (std::size_t r = 0; r < rows; ++r)
        for (std::size_t i = 0; i < k; ++i) {
            const double step = s[r * (k / 128) + i / 128];
            const double e = std::fabs(static_cast<double>(q[r * k + i]) * step - w[r * k + i]);
            if (step > 0) worst = std::max(worst, e / step);
        }
    std::printf("  worst |w - s q| / s = %.4f\n", worst);
    CHECK(worst <= 0.5 + 1e-6);

    std::vector<float> f = random_float(rng, 4097, 10.0f);
    const float inf = std::numeric_limits<float>::infinity();
    const float specials[] = {0.0f, -0.0f, inf, -inf, 3.3895314e38f, 1e-40f, 1.00390625f, 1.01171875f};
    std::memcpy(f.data(), specials, sizeof specials);
    DeviceBuffer<float> df(f);
    DeviceBuffer<std::uint16_t> db(f.size());
    sepal::cuda::launch_float_to_bf16(df.data(), db.data(), f.size(), stream());
    sync();
    const std::vector<std::uint16_t> got = db.download();
    bool exact = true;
    for (std::size_t i = 0; i < f.size(); ++i) exact = exact && got[i] == host_bf16(f[i]);
    CHECK(exact);
}

SEPAL_TEST(simt_edge, shapes_and_refusals) {
    // T = 1, N not a multiple of 128, K = 16: still the host schedule.
    for (const SimtWeight type : {SimtWeight::F32, SimtWeight::BF16}) {
        // N = 5 with an output stride of 8: the scalar tail of the epilogue; the padding columns stay untouched.
        const Problem p(3, 5, 16, type, 61);
        DeviceBuffer<float> y(3 * 8);
        y.zero();
        sepal::cuda::launch_gemm_simt(p.dx.data(), 16, p.matrix(), y.data(), 8, 3, 5, 16, SimtEpilogue::Store, -1, stream());
        sync();
        const std::vector<float> got = y.download();
        bool exact = true;
        for (std::size_t r = 0; r < 3; ++r)
            for (std::size_t c = 0; c < 8; ++c) exact = exact && got[r * 8 + c] == (c < 5 ? p.emulate(r, c) : 0.0f);
        CHECK(exact);
    }
    using sepal::cuda::gemm_simt_supports;
    CHECK(!gemm_simt_supports(SimtWeight::Int8Group, 128, 256 + 16, 272, 272, 128, SimtEpilogue::Store)); // K not a multiple of 128
    CHECK(!gemm_simt_supports(SimtWeight::BF16, 128, 24, 24, 24, 128, SimtEpilogue::Store));              // K not a multiple of 16
    CHECK(!gemm_simt_supports(SimtWeight::BF16, 192, 256, 256, 256, 96, SimtEpilogue::SwiGLU));           // N not a multiple of 128
    CHECK(gemm_simt_supports(SimtWeight::BF16, 256, 256, 256, 256, 128, SimtEpilogue::SwiGLU));
    const Problem p(4, 128, 256, SimtWeight::BF16, 67);
    DeviceBuffer<float> y(4 * 128);
    CHECK_THROWS(sepal::cuda::launch_gemm_simt(p.dx.data(), 256, p.matrix(), y.data(), 128, 4, 128, 256, SimtEpilogue::Store, 7, stream()),
                 std::invalid_argument);
    // T = 0 is a no-op.
    sepal::cuda::launch_gemm_simt(p.dx.data(), 256, p.matrix(), y.data(), 128, 0, 128, 256, SimtEpilogue::Store, -1, stream());
    sync();
}
