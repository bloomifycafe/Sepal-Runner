// Tests of the CUDA provider against the reference provider. Needs a CUDA device.
//
//   sepal_cuda_tests              every suite
//   sepal_cuda_tests <suite>      one of: contract, parity_small, parity_real, rows
//
// parity_real and rows print the measured differences; those lines are the record the README quotes.

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include "sepal/config.hpp"
#include "sepal/cuda/cuda_gemm.hpp"
#include "sepal/gemm.hpp"

namespace {

using sepal::DType;
using sepal::MatRef;
using sepal::MatView;
using sepal::cuda::CudaError;
using sepal::cuda::CudaGemm;
using sepal::cuda::CudaGemmOptions;
using sepal::cuda::Precision;

// ---- a minimal harness ----

int g_checks = 0;
int g_failures = 0;

void report(bool ok, const std::string& text, int line) {
    ++g_checks;
    if (ok) return;
    ++g_failures;
    std::cerr << "test_cuda_gemm.cpp:" << line << ": FAILED: " << text << "\n";
}

#define CHECK(cond) report(static_cast<bool>(cond), #cond, __LINE__)
#define CHECK_MSG(cond, text) report(static_cast<bool>(cond), (text), __LINE__)

template <typename Exception, typename Fn>
bool throws(Fn&& fn) {
    try {
        fn();
    } catch (const Exception&) {
        return true;
    } catch (...) {
    }
    return false;
}

// ---- data ----

class Rng {
public:
    explicit Rng(std::uint64_t seed) : state_(seed) {}
    std::uint64_t next() { // splitmix64
        std::uint64_t z = (state_ += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    float uniform(float lo, float hi) { // 24 random bits
        const float u = static_cast<float>(next() >> 40) * (1.0f / 16777216.0f);
        return lo + (hi - lo) * u;
    }

private:
    std::uint64_t state_;
};

std::vector<float> random_f32(std::size_t count, float scale, std::uint64_t seed) {
    Rng rng(seed);
    std::vector<float> v(count);
    for (float& f : v) f = rng.uniform(-scale, scale);
    return v;
}

// float32 to IEEE binary16 bits, round to nearest with ties to even.
std::uint16_t float_to_half(float f) {
    const std::uint32_t x = std::bit_cast<std::uint32_t>(f);
    const std::uint16_t sign = static_cast<std::uint16_t>((x >> 16) & 0x8000u);
    const std::uint32_t exp = (x >> 23) & 0xFFu;
    std::uint32_t man = x & 0x7FFFFFu;
    if (exp == 255) return static_cast<std::uint16_t>(sign | 0x7C00u | (man != 0 ? 0x200u : 0u));
    const int e = static_cast<int>(exp) - 127 + 15;
    if (e >= 31) return static_cast<std::uint16_t>(sign | 0x7C00u);
    if (e <= 0) {
        if (e < -10) return sign;
        man |= 0x800000u;
        const unsigned shift = static_cast<unsigned>(14 - e);
        std::uint32_t half_man = man >> shift;
        const std::uint32_t rest = man & ((1u << shift) - 1u);
        const std::uint32_t halfway = 1u << (shift - 1u);
        if (rest > halfway || (rest == halfway && (half_man & 1u) != 0)) ++half_man;
        return static_cast<std::uint16_t>(sign | half_man);
    }
    std::uint32_t out = (static_cast<std::uint32_t>(e) << 10) | (man >> 13);
    const std::uint32_t rest = man & 0x1FFFu;
    if (rest > 0x1000u || (rest == 0x1000u && (out & 1u) != 0)) ++out; // may carry into the exponent: still right
    return static_cast<std::uint16_t>(sign | out);
}

std::vector<std::uint16_t> to_half(const std::vector<float>& v) {
    std::vector<std::uint16_t> h(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) h[i] = float_to_half(v[i]);
    return h;
}

std::vector<std::uint16_t> to_bf16_truncated(const std::vector<float>& v) {
    std::vector<std::uint16_t> b(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) {
        b[i] = static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(v[i]) >> 16);
    }
    return b;
}

// ---- comparison ----

struct Diff {
    double max_abs = 0.0;
    double max_rel = 0.0; // elementwise |a - b| / |b|, over the elements with |b| >= rms(b)
    double ref_rms = 0.0;
    double ref_max = 0.0;
    std::size_t differing = 0; // elements whose bits differ
    std::size_t count = 0;
};

Diff compare(const std::vector<float>& got, const std::vector<float>& ref) {
    Diff d;
    d.count = ref.size();
    double sum_sq = 0.0;
    for (const float r : ref) {
        sum_sq += static_cast<double>(r) * static_cast<double>(r);
        d.ref_max = std::max(d.ref_max, std::fabs(static_cast<double>(r)));
    }
    d.ref_rms = ref.empty() ? 0.0 : std::sqrt(sum_sq / static_cast<double>(ref.size()));
    for (std::size_t i = 0; i < ref.size(); ++i) {
        if (std::bit_cast<std::uint32_t>(got[i]) != std::bit_cast<std::uint32_t>(ref[i])) ++d.differing;
        const double a = std::fabs(static_cast<double>(got[i]) - static_cast<double>(ref[i]));
        if (!(a <= d.max_abs)) d.max_abs = a; // also catches NaN
        const double r = std::fabs(static_cast<double>(ref[i]));
        if (r >= d.ref_rms && r > 0.0) d.max_rel = std::max(d.max_rel, a / r);
    }
    return d;
}

CudaGemm& provider(Precision p) {
    // One provider per precision for the whole run: 2 GiB each is enough for the largest case here.
    static CudaGemm f16(CudaGemmOptions{0, Precision::F16, std::size_t{2} << 30, std::size_t{32} << 20});
    static CudaGemm f32(CudaGemmOptions{0, Precision::F32, std::size_t{2} << 30, std::size_t{32} << 20});
    return p == Precision::F16 ? f16 : f32;
}

const char* name_of_precision(Precision p) { return p == Precision::F16 ? "fp16" : "fp32"; }

std::vector<float> multiply(const sepal::GemmProvider& g, MatRef x, MatRef w, float beta = 0.0f,
                            const std::vector<float>* y0 = nullptr) {
    std::vector<float> y(x.rows * w.rows, 0.0f);
    if (y0 != nullptr) y = *y0;
    g.gemm_f(x, w, MatView(y.data(), x.rows, w.rows), beta);
    return y;
}

// ---- suite: contract ----

void contract_caps_and_device() {
    const CudaGemm& g = provider(Precision::F16);
    const sepal::GemmCaps caps = g.caps();
    CHECK(caps.name == "cublas");
    CHECK(caps.float_dtypes.size() == 3);
    CHECK(!caps.fixed_summation_order);
    const sepal::cuda::DeviceInfo info = g.device_info();
    CHECK(!info.name.empty());
    CHECK(info.total_bytes > 0);
    std::printf("  device: %s, sm_%d%d, %.2f GiB, CUDA runtime %d, driver supports CUDA %d, cuBLAS %d\n",
                info.name.c_str(), info.compute_major, info.compute_minor,
                static_cast<double>(info.total_bytes) / 1073741824.0, info.runtime_version, info.driver_version,
                info.cublas_version);
}

void contract_rejects_bad_arguments() {
    const CudaGemm& g = provider(Precision::F32);
    const std::vector<float> x(6, 1.0f);
    const std::vector<float> w(8, 1.0f);
    std::vector<float> y(4, 0.0f);
    const std::vector<std::int8_t> bytes(8, 1);
    CHECK(throws<std::invalid_argument>([&] { // X has 3 columns, W has 4
        g.gemm_f(MatRef(x.data(), DType::F32, 2, 3), MatRef(w.data(), DType::F32, 2, 4), MatView(y.data(), 2, 2), 0.0f);
    }));
    CHECK(throws<std::invalid_argument>([&] { // Y has the wrong shape
        g.gemm_f(MatRef(x.data(), DType::F32, 2, 3), MatRef(w.data(), DType::F32, 2, 3), MatView(y.data(), 1, 2), 0.0f);
    }));
    CHECK(throws<std::invalid_argument>([&] { // int8 is not a float type
        g.gemm_f(MatRef(x.data(), DType::F32, 2, 3), MatRef(bytes.data(), DType::I8, 2, 3), MatView(y.data(), 2, 2),
                 0.0f);
    }));
    CHECK(throws<std::invalid_argument>([&] {
        g.gemm_f(MatRef(nullptr, DType::F32, 2, 3), MatRef(w.data(), DType::F32, 2, 3), MatView(y.data(), 2, 2), 0.0f);
    }));
    CHECK(throws<std::invalid_argument>([&] { g.gemm_f(MatRef(x.data(), DType::F32, 2, 3), sepal::cuda::WeightId{},
                                                        MatView(y.data(), 2, 2), 0.0f); }));
    // An empty batch is valid and writes nothing.
    g.gemm_f(MatRef(x.data(), DType::F32, 0, 3), MatRef(w.data(), DType::F32, 2, 3), MatView(y.data(), 0, 2), 0.0f);
    CHECK(y[0] == 0.0f);
}

void contract_errors_name_the_call() {
    bool named = false;
    try {
        const CudaGemm bad(CudaGemmOptions{4096, Precision::F32, std::size_t{1} << 20, 0});
    } catch (const CudaError& e) {
        named = e.call() == "cudaSetDevice" && std::string(e.what()).find("cudaSetDevice failed") == 0;
        std::printf("  a device that does not exist: %s\n", e.what());
    }
    CHECK(named);

    // An arena too small for the call: the error names the arena, the need and the knob.
    CudaGemm tiny(CudaGemmOptions{0, Precision::F32, std::size_t{1} << 20, 0});
    const std::vector<float> x(1024 * 512, 0.5f);
    const std::vector<float> w(64 * 512, 0.5f);
    std::vector<float> y(1024 * 64);
    bool arena_named = false;
    try {
        tiny.gemm_f(MatRef(x.data(), DType::F32, 1024, 512), MatRef(w.data(), DType::F32, 64, 512),
                    MatView(y.data(), 1024, 64), 0.0f);
    } catch (const CudaError& e) {
        arena_named = e.call() == "arena" && std::string(e.what()).find("arena_bytes") != std::string::npos;
        std::printf("  an arena that is too small: %s\n", e.what());
    }
    CHECK(arena_named);
    // The failed call released what it had borrowed: a call that fits still works.
    std::vector<float> y_small(4 * 64);
    tiny.gemm_f(MatRef(x.data(), DType::F32, 4, 512), MatRef(w.data(), DType::F32, 64, 512),
                MatView(y_small.data(), 4, 64), 0.0f);
    CHECK(std::fabs(y_small[0] - 128.0f) < 1e-3f);
}

void contract_beta_and_weight_paths() {
    const std::size_t t = 5, n = 7, k = 33;
    const std::vector<float> x = random_f32(t * k, 1.0f, 1);
    const std::vector<float> w = random_f32(n * k, 0.2f, 2);
    const std::vector<float> y0 = random_f32(t * n, 1.0f, 3);
    const MatRef xr(x.data(), DType::F32, t, k);
    const MatRef wr(w.data(), DType::F32, n, k);
    const auto reference = sepal::make_reference_gemm();

    for (const Precision p : {Precision::F32, Precision::F16}) {
        CudaGemm& g = provider(p);
        const double tol = p == Precision::F32 ? 1e-5 : 5e-3;
        for (const float beta : {0.0f, 1.0f, -0.5f}) {
            const Diff d = compare(multiply(g, xr, wr, beta, &y0), multiply(*reference, xr, wr, beta, &y0));
            CHECK_MSG(d.max_abs < tol, std::string("beta path, ") + name_of_precision(p));
        }
        // With beta == 0, Y is written without being read: NaN in Y must not leak into the result.
        const std::vector<float> nan_y(t * n, std::nanf(""));
        const std::vector<float> clean = multiply(g, xr, wr, 0.0f, &nan_y);
        CHECK(std::none_of(clean.begin(), clean.end(), [](float f) { return std::isnan(f); }));

        // The three ways to name W give the same bits: travelling with the call, registered, and by id.
        const std::vector<float> travelling = multiply(g, xr, wr);
        const sepal::cuda::WeightId id = g.register_weights(wr);
        CHECK(g.register_weights(wr).index == id.index);
        const std::vector<float> registered = multiply(g, xr, wr);
        std::vector<float> by_id(t * n);
        g.gemm_f(xr, id, MatView(by_id.data(), t, n), 0.0f);
        CHECK(compare(registered, travelling).differing == 0);
        CHECK(compare(by_id, travelling).differing == 0);

        // Device-resident operands give the same bits as the host round trip.
        const sepal::cuda::DeviceMatrix xd = g.upload_matrix(xr);
        const sepal::cuda::DeviceMatrix yd = g.alloc_output(t, n);
        g.gemm_device(xd, id, yd, 0.0f);
        std::vector<float> resident(t * n);
        g.download(yd, MatView(resident.data(), t, n));
        CHECK(compare(resident, travelling).differing == 0);
    }
}

void contract_sixteen_bit_inputs() {
    // On the fp32 path, binary16 and bfloat16 inputs are widened exactly, as the reference decodes them: the two
    // providers see the same operands, so only the summation differs.
    const std::size_t t = 6, n = 9, k = 200;
    const std::vector<float> x = random_f32(t * k, 1.0f, 11);
    const std::vector<float> w = random_f32(n * k, 0.1f, 12);
    const auto reference = sepal::make_reference_gemm();
    const std::vector<std::uint16_t> xh = to_half(x), wh = to_half(w);
    const std::vector<std::uint16_t> xb = to_bf16_truncated(x), wb = to_bf16_truncated(w);

    CudaGemm& g32 = provider(Precision::F32);
    const MatRef x16(xh.data(), DType::F16, t, k), w16(wh.data(), DType::F16, n, k);
    const MatRef xbf(xb.data(), DType::BF16, t, k), wbf(wb.data(), DType::BF16, n, k);
    CHECK(compare(multiply(g32, x16, w16), multiply(*reference, x16, w16)).max_abs < 1e-5);
    CHECK(compare(multiply(g32, xbf, wbf), multiply(*reference, xbf, wbf)).max_abs < 1e-5);
    CHECK(compare(multiply(g32, x16, wbf), multiply(*reference, x16, wbf)).max_abs < 1e-5);

    // On the fp16 path binary16 operands are used as they are; bfloat16 is rounded to binary16 (lossy).
    CudaGemm& g16 = provider(Precision::F16);
    CHECK(compare(multiply(g16, x16, w16), multiply(*reference, x16, w16)).max_abs < 1e-5);
    CHECK(compare(multiply(g16, xbf, wbf), multiply(*reference, xbf, wbf)).max_abs < 5e-3);
}

// ---- suite: parity_small ----

void parity_small_random_shapes() {
    const auto reference = sepal::make_reference_gemm();
    Rng rng(2024);
    double worst32 = 0.0, worst16 = 0.0;
    for (int trial = 0; trial < 40; ++trial) {
        const std::size_t t = 1 + rng.next() % 40;
        const std::size_t n = 1 + rng.next() % 300;
        const std::size_t k = 1 + rng.next() % 300;
        const std::vector<float> x = random_f32(t * k, 1.0f, 100 + static_cast<std::uint64_t>(trial));
        const std::vector<float> w = random_f32(n * k, 1.0f / std::sqrt(static_cast<float>(k)),
                                                200 + static_cast<std::uint64_t>(trial));
        const MatRef xr(x.data(), DType::F32, t, k), wr(w.data(), DType::F32, n, k);
        worst32 = std::max(worst32, compare(multiply(provider(Precision::F32), xr, wr), multiply(*reference, xr, wr))
                                        .max_abs);
        const std::vector<std::uint16_t> xh = to_half(x), wh = to_half(w);
        const MatRef x16(xh.data(), DType::F16, t, k), w16(wh.data(), DType::F16, n, k);
        worst16 = std::max(worst16, compare(multiply(provider(Precision::F16), x16, w16),
                                            multiply(*reference, x16, w16))
                                        .max_abs);
    }
    std::printf("  40 random shapes up to [40, 300] x [300, 300]: worst |cuda - reference| fp32 %.3g, fp16 %.3g\n",
                worst32, worst16);
    CHECK(worst32 < 1e-5);
    CHECK(worst16 < 1e-5);
}

// ---- the real layer shapes ----

struct Projection {
    std::string name;
    std::size_t n = 0; // output features: rows of W
    std::size_t k = 0; // input features: columns of W
};

// Every distinct GEMM of one layer of each kind, as docs/RUNNER.md section 3 lists them.
std::vector<Projection> projections(const sepal::ModelConfig& c) {
    const std::size_t h = c.hidden_size, i = c.intermediate_size;
    return {
        {"mlp gate+up", 2 * i, h},
        {"mlp down", h, i},
        {"deltanet in_proj", c.dn_in_proj_width(), h},
        {"deltanet out_proj", h, c.dn_value_dim()},
        {"attention in_proj", c.attn_in_proj_width(), h},
        {"attention o_proj", h, c.attn_width()},
    };
}

void print_diff_row(const char* preset, const Projection& p, std::size_t t, const char* path, const Diff& d) {
    std::printf("  %-4s %-18s X[%zu,%zu] W[%zu,%zu]  %-26s max_abs %.3e  max_rel %.3e  (ref rms %.3f, max %.3f)\n",
                preset, p.name.c_str(), t, p.k, p.n, p.k, path, d.max_abs, d.max_rel, d.ref_rms, d.ref_max);
}

void parity_real_layer_shapes() {
    const auto reference = sepal::make_reference_gemm();
    const std::size_t t = 8;
    Diff worst32, worst16, worst16q;
    std::printf("  parity against the reference provider, %zu rows per case; max_rel is over |ref| >= rms(ref)\n", t);
    for (const char* preset : {"0.8b", "4b"}) {
        const sepal::ModelConfig cfg = *sepal::find_preset(preset);
        std::uint64_t seed = 7000;
        for (const Projection& p : projections(cfg)) {
            const std::vector<float> x = random_f32(t * p.k, 1.0f, ++seed);
            const std::vector<float> w = random_f32(p.n * p.k, 1.0f / std::sqrt(static_cast<float>(p.k)), ++seed);
            const MatRef xr(x.data(), DType::F32, t, p.k), wr(w.data(), DType::F32, p.n, p.k);
            const std::vector<float> ref32 = multiply(*reference, xr, wr);

            // fp32 path, fp32 operands.
            const Diff d32 = compare(multiply(provider(Precision::F32), xr, wr), ref32);
            print_diff_row(preset, p, t, "fp32", d32);

            // fp16 path, binary16 operands given to both providers: the arithmetic alone.
            const std::vector<std::uint16_t> xh = to_half(x), wh = to_half(w);
            const MatRef x16(xh.data(), DType::F16, t, p.k), w16(wh.data(), DType::F16, p.n, p.k);
            const Diff d16 = compare(multiply(provider(Precision::F16), x16, w16), multiply(*reference, x16, w16));
            print_diff_row(preset, p, t, "fp16 (fp16 operands)", d16);

            // fp16 path, fp32 operands rounded on the device, against the fp32 reference: what fp16 storage costs.
            const Diff d16q = compare(multiply(provider(Precision::F16), xr, wr), ref32);
            print_diff_row(preset, p, t, "fp16 (fp32 rounded to fp16)", d16q);

            const auto keep_worst = [](Diff& worst, const Diff& d) {
                worst.max_abs = std::max(worst.max_abs, d.max_abs);
                worst.max_rel = std::max(worst.max_rel, d.max_rel);
            };
            keep_worst(worst32, d32);
            keep_worst(worst16, d16);
            keep_worst(worst16q, d16q);
        }
    }
    std::printf("  WORST fp32:                        max_abs %.3e  max_rel %.3e\n", worst32.max_abs, worst32.max_rel);
    std::printf("  WORST fp16 (fp16 operands):        max_abs %.3e  max_rel %.3e\n", worst16.max_abs, worst16.max_rel);
    std::printf("  WORST fp16 (fp32 rounded to fp16): max_abs %.3e  max_rel %.3e\n", worst16q.max_abs,
                worst16q.max_rel);
    CHECK(worst32.max_abs < 1e-4 && worst32.max_rel < 1e-4);
    CHECK(worst16.max_abs < 1e-4 && worst16.max_rel < 1e-4);
    CHECK(worst16q.max_abs < 2e-2 && worst16q.max_rel < 2e-2);
}

// ---- suite: rows ----
//
// The interface contract: a row's result depends on that row and on the weights, and on nothing else; the
// reference provider gives the same bits for a row alone and inside any batch. This suite measures how far
// cuBLAS keeps that, in four separate questions per layer shape:
//
//   alone      is a row computed alone (T = 1) bit-identical to the same row inside a batch of T rows?
//   classes    which batch sizes give each other's bits for the rows they share?
//   position   at one batch size, are a row's bits the same wherever the row sits in the batch?
//   batchmates at one batch size and position, are a row's bits unaffected by the other rows' values?
//
// Only `batchmates` is required to hold for the test to pass (anything else there would be a wrong result, not a
// rounding difference). The other three are printed as findings and bounded to rounding size.

struct BitDiff {
    std::size_t elements = 0;
    std::size_t differing = 0;
    std::size_t rows = 0;
    std::size_t rows_differing = 0;
    double max_abs = 0.0;

    void add_row(const float* a, const float* b, std::size_t n) {
        ++rows;
        bool differs = false;
        for (std::size_t c = 0; c < n; ++c) {
            ++elements;
            if (std::bit_cast<std::uint32_t>(a[c]) == std::bit_cast<std::uint32_t>(b[c])) continue;
            differs = true;
            ++differing;
            const double d = std::fabs(static_cast<double>(a[c]) - static_cast<double>(b[c]));
            if (!(d <= max_abs)) max_abs = d;
        }
        if (differs) ++rows_differing;
    }
    void merge(const BitDiff& o) {
        elements += o.elements;
        differing += o.differing;
        rows += o.rows;
        rows_differing += o.rows_differing;
        max_abs = std::max(max_abs, o.max_abs);
    }
    bool identical() const { return differing == 0; }
};

struct RowsSummary {
    int cases = 0;
    int alone_identical = 0;
    int single_class = 0;
    int position_identical = 0;
    int batchmates_identical = 0;
    double max_abs = 0.0;
    double max_abs_over_rms = 0.0;
};

RowsSummary rows_run(Precision prec, bool pedantic, bool print_cases) {
    const std::vector<std::size_t> batches = {2, 3, 7, 16, 64, 256, 1024};
    const std::vector<std::size_t> fixed = {64, 1024}; // batch sizes for the position and batchmates questions
    const std::size_t t_max = batches.back();
    RowsSummary summary;
    // A provider of its own, so that this suite's weights do not accumulate in the shared ones.
    CudaGemm g(CudaGemmOptions{0, prec, std::size_t{3} << 30, std::size_t{32} << 20, pedantic});
    for (const char* preset : {"0.8b", "4b"}) {
        const sepal::ModelConfig cfg = *sepal::find_preset(preset);
        std::uint64_t seed = 9000;
        for (const Projection& p : projections(cfg)) {
            const std::vector<float> x = random_f32(t_max * p.k, 1.0f, ++seed);
            const std::vector<float> w = random_f32(p.n * p.k, 1.0f / std::sqrt(static_cast<float>(p.k)), ++seed);
            const std::vector<float> other = random_f32(t_max * p.k, 1.0f, ++seed);
            const sepal::cuda::WeightId id = g.upload_weights(MatRef(w.data(), DType::F32, p.n, p.k));
            const auto run = [&](const std::vector<float>& in, std::size_t t) {
                std::vector<float> y(t * p.n);
                g.gemm_f(MatRef(in.data(), DType::F32, t, p.k), id, MatView(y.data(), t, p.n), 0.0f);
                return y;
            };

            // alone: every row by itself, against every batch.
            std::vector<float> alone(t_max * p.n);
            for (std::size_t r = 0; r < t_max; ++r) {
                g.gemm_f(MatRef(x.data() + r * p.k, DType::F32, 1, p.k), id, MatView(alone.data() + r * p.n, 1, p.n),
                         0.0f);
            }
            double sum_sq = 0.0;
            for (const float f : alone) sum_sq += static_cast<double>(f) * static_cast<double>(f);
            const double rms = std::sqrt(sum_sq / static_cast<double>(alone.size()));

            std::vector<std::vector<float>> by_batch;
            BitDiff alone_diff;
            std::string alone_differs_at;
            for (const std::size_t t : batches) {
                by_batch.push_back(run(x, t));
                BitDiff d;
                for (std::size_t r = 0; r < t; ++r) {
                    d.add_row(by_batch.back().data() + r * p.n, alone.data() + r * p.n, p.n);
                }
                if (!d.identical()) alone_differs_at += " " + std::to_string(t);
                alone_diff.merge(d);
            }

            // classes: batch sizes (1 included) whose shared rows have the same bits.
            std::vector<std::size_t> sizes = {1};
            sizes.insert(sizes.end(), batches.begin(), batches.end());
            const auto rows_of = [&](std::size_t index) -> const float* {
                return index == 0 ? alone.data() : by_batch[index - 1].data();
            };
            std::vector<std::vector<std::size_t>> classes; // indices into sizes
            for (std::size_t i = 0; i < sizes.size(); ++i) {
                bool placed = false;
                for (std::vector<std::size_t>& cls : classes) {
                    const std::size_t rep = cls.front();
                    const std::size_t shared = std::min(sizes[i], sizes[rep]);
                    if (std::memcmp(rows_of(i), rows_of(rep), shared * p.n * sizeof(float)) == 0) {
                        cls.push_back(i);
                        placed = true;
                        break;
                    }
                }
                if (!placed) classes.push_back({i});
            }
            std::string class_text;
            for (const std::vector<std::size_t>& cls : classes) {
                class_text += "{";
                for (std::size_t j = 0; j < cls.size(); ++j) {
                    class_text += (j != 0 ? "," : "") + std::to_string(sizes[cls[j]]);
                }
                class_text += "} ";
            }

            // position: the same rows, rotated by five places inside the batch.
            // batchmates: one row kept in its place, every other row replaced.
            BitDiff position_diff, mates_diff;
            for (const std::size_t t : fixed) {
                const std::vector<float>& base =
                    by_batch[static_cast<std::size_t>(std::find(batches.begin(), batches.end(), t) - batches.begin())];
                std::vector<float> rotated(t * p.k);
                for (std::size_t r = 0; r < t; ++r) {
                    std::memcpy(rotated.data() + ((r + 5) % t) * p.k, x.data() + r * p.k, p.k * sizeof(float));
                }
                const std::vector<float> y_rot = run(rotated, t);
                for (std::size_t r = 0; r < t; ++r) {
                    position_diff.add_row(y_rot.data() + ((r + 5) % t) * p.n, base.data() + r * p.n, p.n);
                }
                for (const std::size_t keep : {std::size_t{0}, std::size_t{17}, t - 1}) {
                    std::vector<float> mixed(other.begin(), other.begin() + static_cast<std::ptrdiff_t>(t * p.k));
                    std::memcpy(mixed.data() + keep * p.k, x.data() + keep * p.k, p.k * sizeof(float));
                    const std::vector<float> y_mixed = run(mixed, t);
                    mates_diff.add_row(y_mixed.data() + keep * p.n, base.data() + keep * p.n, p.n);
                }
            }

            const double worst = std::max(alone_diff.max_abs, position_diff.max_abs);
            ++summary.cases;
            summary.alone_identical += alone_diff.identical() ? 1 : 0;
            summary.single_class += classes.size() == 1 ? 1 : 0;
            summary.position_identical += position_diff.identical() ? 1 : 0;
            summary.batchmates_identical += mates_diff.identical() ? 1 : 0;
            summary.max_abs = std::max(summary.max_abs, worst);
            summary.max_abs_over_rms = std::max(summary.max_abs_over_rms, worst / rms);

            if (print_cases) {
                std::printf("  %s %-4s %-18s W[%zu,%zu]\n", name_of_precision(prec), preset, p.name.c_str(), p.n, p.k);
                if (alone_diff.identical()) {
                    std::printf("      alone:      bit-identical in all %zu rows\n", alone_diff.rows);
                } else {
                    std::printf("      alone:      NOT bit-identical: %zu of %zu rows, %.2f%% of elements, max_abs "
                                "%.3e (%.1e of the output rms %.3f); at T =%s\n",
                                alone_diff.rows_differing, alone_diff.rows,
                                100.0 * static_cast<double>(alone_diff.differing) /
                                    static_cast<double>(alone_diff.elements),
                                alone_diff.max_abs, alone_diff.max_abs / rms, rms, alone_differs_at.c_str());
                }
                std::printf("      classes:    %s\n", class_text.c_str());
                if (position_diff.identical()) {
                    std::printf("      position:   bit-identical after rotating the batch (T = 64, 1024)\n");
                } else {
                    std::printf("      position:   NOT bit-identical after rotating the batch: %zu of %zu rows, "
                                "max_abs %.3e\n",
                                position_diff.rows_differing, position_diff.rows, position_diff.max_abs);
                }
                std::printf("      batchmates: %s\n", mates_diff.identical()
                                                          ? "bit-identical when every other row is replaced"
                                                          : "DIFFERENT when other rows are replaced");
            }
            CHECK_MSG(mates_diff.identical(), "a row must not depend on the values of other rows, " + p.name);
            CHECK_MSG(worst < 1e-4, "batch-shape differences must stay at rounding size, " + p.name);
        }
    }
    return summary;
}

void print_rows_summary(const char* label, const RowsSummary& s) {
    std::printf("  ROW INDEPENDENCE %-14s alone = in-batch bits in %d of %d shapes; one bit pattern for all batch "
                "sizes in %d of %d; position-invariant in %d of %d; independent of batchmates' values in %d of %d; "
                "largest difference %.3e (%.1e of the output rms)\n",
                label, s.alone_identical, s.cases, s.single_class, s.cases, s.position_identical, s.cases,
                s.batchmates_identical, s.cases, s.max_abs, s.max_abs_over_rms);
}

void rows_real_layer_shapes() {
    std::printf("  batch sizes compared: T = 1 (alone), 2, 3, 7, 16, 64, 256, 1024; in 'classes', batch sizes inside\n"
                "  one pair of braces give identical bits for the rows they share\n");
    const RowsSummary f32 = rows_run(Precision::F32, false, true);
    const RowsSummary f16 = rows_run(Precision::F16, false, true);
    print_rows_summary("fp32:", f32);
    print_rows_summary("fp16:", f16);
}

void rows_pedantic_mode() {
    // The same questions with cuBLAS in its pedantic mode: does it restore batch-independent bits?
    print_rows_summary("fp32 pedantic:", rows_run(Precision::F32, true, false));
    print_rows_summary("fp16 pedantic:", rows_run(Precision::F16, true, false));
}

struct Case {
    const char* suite;
    const char* name;
    void (*fn)();
};

const Case kCases[] = {
    {"contract", "caps_and_device", contract_caps_and_device},
    {"contract", "rejects_bad_arguments", contract_rejects_bad_arguments},
    {"contract", "errors_name_the_call", contract_errors_name_the_call},
    {"contract", "beta_and_weight_paths", contract_beta_and_weight_paths},
    {"contract", "sixteen_bit_inputs", contract_sixteen_bit_inputs},
    {"parity_small", "random_shapes", parity_small_random_shapes},
    {"parity_real", "layer_shapes", parity_real_layer_shapes},
    {"rows", "real_layer_shapes", rows_real_layer_shapes},
    {"rows", "pedantic_mode", rows_pedantic_mode},
};

} // namespace

int main(int argc, char** argv) {
    const char* filter = argc > 1 ? argv[1] : nullptr;
    int cases_run = 0;
    int cases_failed = 0;
    for (const Case& c : kCases) {
        if (filter != nullptr && std::strcmp(filter, c.suite) != 0) continue;
        ++cases_run;
        const int failures_before = g_failures;
        std::printf("[run ] %s.%s\n", c.suite, c.name);
        std::fflush(stdout);
        try {
            c.fn();
        } catch (const std::exception& e) {
            report(false, std::string("unexpected exception: ") + e.what(), 0);
        }
        const bool failed = g_failures != failures_before;
        if (failed) ++cases_failed;
        std::printf("%s %s.%s\n", failed ? "[FAIL]" : "[ ok ]", c.suite, c.name);
        std::fflush(stdout);
    }
    if (cases_run == 0) {
        std::fprintf(stderr, "no test case matches suite '%s'\n", filter != nullptr ? filter : "");
        return 1;
    }
    std::printf("%d case(s), %d check(s), %d failed check(s) in %d case(s)\n", cases_run, g_checks, g_failures,
                cases_failed);
    return g_failures == 0 ? 0 : 1;
}
