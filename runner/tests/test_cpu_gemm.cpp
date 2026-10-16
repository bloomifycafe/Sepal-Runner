// The packed CPU GEMM against the reference provider, on both instruction-set families the machine supports,
// for odd shapes, and the determinism rules: thread counts and batch composition change no bit.

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "sepal/cpu/gemm.hpp"
#include "sepal/cpu/isa.hpp"
#include "sepal/cpu/provider.hpp"
#include "sepal/cpu/thread_pool.hpp"
#include "test.hpp"

namespace {

using namespace sepal;
using namespace sepal::cpu;

struct Rng {
    std::uint64_t s;
    explicit Rng(std::uint64_t seed) : s(seed) {}
    float next() { // uniform in [-1, 1)
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        return static_cast<float>(static_cast<std::int32_t>(s >> 33)) / 1073741824.0f - 1.0f;
    }
};

std::vector<float> random_matrix(std::size_t rows, std::size_t cols, std::uint64_t seed, float scale = 1.0f) {
    Rng rng(seed);
    std::vector<float> m(rows * cols);
    for (float& v : m) v = rng.next() * scale;
    return m;
}

std::vector<Isa> isas() {
    std::vector<Isa> out = {Isa::Avx2};
    if (isa_supported(Isa::Avx512)) out.push_back(Isa::Avx512);
    return out;
}

float round_bf16(float x) { return bf16_to_float(float_to_bf16(x)); }

// Reference product with float64 accumulation, and the sum of |x w| for the error bound.
void reference(const std::vector<float>& x, const std::vector<float>& w, std::size_t rows, std::size_t n, std::size_t k,
               std::vector<double>& y, std::vector<double>& bound) {
    y.assign(rows * n, 0.0);
    bound.assign(rows * n, 0.0);
    for (std::size_t t = 0; t < rows; ++t) {
        for (std::size_t j = 0; j < n; ++j) {
            double acc = 0.0, abs_acc = 0.0;
            for (std::size_t i = 0; i < k; ++i) {
                const double p = static_cast<double>(x[t * k + i]) * static_cast<double>(w[j * k + i]);
                acc += p;
                abs_acc += std::fabs(p);
            }
            y[t * n + j] = acc;
            bound[t * n + j] = abs_acc;
        }
    }
}

std::vector<float> run_gemm(const std::vector<float>& x, const std::vector<float>& w, std::size_t rows, std::size_t n,
                            std::size_t k, WeightKind kind, Isa isa, unsigned threads) {
    ThreadPool pool(threads);
    const PackedWeight pw = pack_weight(w.data(), k, n, k, kind, isa, pool);
    PackedActs pa;
    pack_acts(pa, x.data(), k, rows, k, pw.variant(), pool);
    std::vector<float> y(rows * n, -7.0f);
    gemm_packed(pw, pa, y.data(), n, false, pool);
    return y;
}

struct Shape {
    std::size_t rows, n, k;
};
const Shape shapes[] = {{1, 1, 1}, {5, 7, 3}, {13, 37, 70}, {29, 100, 129}, {96, 64, 256}, {50, 33, 1024}};

} // namespace

SEPAL_TEST(cpu_gemm, features_and_pool) {
    CHECK(isa_supported(Isa::Avx2));
    CHECK(!features().brand.empty());
    for (const unsigned threads : {1u, 2u, 5u, 12u}) {
        ThreadPool pool(threads);
        CHECK_EQ(pool.size(), threads);
        std::vector<std::atomic<int>> hits(1000);
        std::atomic<long> sum{0};
        pool.parallel_for(1000, [&](std::size_t i, unsigned worker) {
            hits[i].fetch_add(1);
            sum.fetch_add(static_cast<long>(i));
            if (worker >= threads) throw std::logic_error("worker id out of range");
        });
        bool all_once = true;
        for (auto& h : hits) all_once = all_once && h.load() == 1;
        CHECK(all_once);
        CHECK_EQ(sum.load(), 499500L);
        // A nested call runs serially inside the task.
        std::atomic<int> inner{0};
        pool.parallel_for(8, [&](std::size_t, unsigned) { pool.parallel_for(8, [&](std::size_t, unsigned) { inner.fetch_add(1); }); });
        CHECK_EQ(inner.load(), 64);
        CHECK_THROWS(pool.parallel_for(50, [&](std::size_t i, unsigned) { if (i == 17) throw std::runtime_error("boom"); }), std::runtime_error);
        // The pool still works after an exception.
        std::atomic<int> after{0};
        pool.parallel_for(10, [&](std::size_t, unsigned) { after.fetch_add(1); });
        CHECK_EQ(after.load(), 10);
    }
}

SEPAL_TEST(cpu_gemm, f32_matches_reference) {
    for (const Isa isa : isas()) {
        for (const Shape s : shapes) {
            const auto x = random_matrix(s.rows, s.k, 1);
            const auto w = random_matrix(s.n, s.k, 2);
            std::vector<double> ref, bound;
            reference(x, w, s.rows, s.n, s.k, ref, bound);
            const auto y = run_gemm(x, w, s.rows, s.n, s.k, WeightKind::F32, isa, 3);
            double worst = 0;
            for (std::size_t i = 0; i < y.size(); ++i) worst = std::max(worst, std::fabs(y[i] - ref[i]) / (bound[i] + 1e-30));
            CHECK(worst < 2e-6);
        }
    }
}

SEPAL_TEST(cpu_gemm, bf16_matches_reference_on_rounded_inputs) {
    for (const Isa isa : isas()) {
        for (const Shape s : shapes) {
            auto x = random_matrix(s.rows, s.k, 3);
            auto w = random_matrix(s.n, s.k, 4);
            const auto y = run_gemm(x, w, s.rows, s.n, s.k, WeightKind::BF16, isa, 2);
            for (float& v : x) v = round_bf16(v);
            for (float& v : w) v = round_bf16(v);
            std::vector<double> ref, bound;
            reference(x, w, s.rows, s.n, s.k, ref, bound);
            double worst = 0;
            for (std::size_t i = 0; i < y.size(); ++i) worst = std::max(worst, std::fabs(y[i] - ref[i]) / (bound[i] + 1e-30));
            CHECK(worst < 2e-6);
        }
    }
}

SEPAL_TEST(cpu_gemm, i8_matches_integer_emulation) {
    for (const Isa isa : isas()) {
        for (const Shape s : shapes) {
            const auto x = random_matrix(s.rows, s.k, 5, 3.0f);
            const auto w = random_matrix(s.n, s.k, 6, 0.5f);
            const auto y = run_gemm(x, w, s.rows, s.n, s.k, WeightKind::I8, isa, 4);
            // Emulate: symmetric per-feature weight scale, per-row activation scale, exact integer sums.
            std::vector<float> ws(s.n);
            std::vector<std::int8_t> wq(s.n * s.k);
            for (std::size_t j = 0; j < s.n; ++j) {
                float amax = 0;
                for (std::size_t i = 0; i < s.k; ++i) amax = std::max(amax, std::fabs(w[j * s.k + i]));
                ws[j] = amax > 0 ? amax / 127.0f : 1.0f;
                for (std::size_t i = 0; i < s.k; ++i)
                    wq[j * s.k + i] = static_cast<std::int8_t>(std::clamp(std::nearbyint(w[j * s.k + i] * (1.0f / ws[j])), -127.0f, 127.0f));
            }
            double worst = 0;
            for (std::size_t t = 0; t < s.rows; ++t) {
                float amax = 0;
                for (std::size_t i = 0; i < s.k; ++i) amax = std::max(amax, std::fabs(x[t * s.k + i]));
                const float xs = amax > 0 ? amax / 127.0f : 1.0f;
                std::vector<std::int8_t> xq(s.k);
                for (std::size_t i = 0; i < s.k; ++i)
                    xq[i] = static_cast<std::int8_t>(std::clamp(std::nearbyint(x[t * s.k + i] * (1.0f / xs)), -127.0f, 127.0f));
                for (std::size_t j = 0; j < s.n; ++j) {
                    std::int64_t acc = 0;
                    for (std::size_t i = 0; i < s.k; ++i) acc += static_cast<std::int64_t>(xq[i]) * wq[j * s.k + i];
                    const double expect = static_cast<double>(static_cast<float>(acc) * ws[j] * xs);
                    worst = std::max(worst, std::fabs(y[t * s.n + j] - expect) / (std::fabs(expect) + 1e-3));
                }
            }
            CHECK(worst < 1e-5);
        }
    }
}

SEPAL_TEST(cpu_gemm, i8_is_close_to_float_product) {
    const Shape s{40, 96, 512};
    const auto x = random_matrix(s.rows, s.k, 7);
    const auto w = random_matrix(s.n, s.k, 8);
    std::vector<double> ref, bound;
    reference(x, w, s.rows, s.n, s.k, ref, bound);
    for (const Isa isa : isas()) {
        const auto y = run_gemm(x, w, s.rows, s.n, s.k, WeightKind::I8, isa, 2);
        double err = 0, norm = 0;
        for (std::size_t i = 0; i < y.size(); ++i) {
            err += (y[i] - ref[i]) * (y[i] - ref[i]);
            norm += ref[i] * ref[i];
        }
        CHECK(std::sqrt(err / norm) < 0.02);
    }
}

SEPAL_TEST(cpu_gemm, thread_count_and_batch_do_not_change_a_bit) {
    const Shape s{70, 130, 300};
    const auto x = random_matrix(s.rows, s.k, 9);
    const auto w = random_matrix(s.n, s.k, 10);
    for (const Isa isa : isas()) {
        for (const WeightKind kind : {WeightKind::F32, WeightKind::BF16, WeightKind::I8}) {
            const auto one = run_gemm(x, w, s.rows, s.n, s.k, kind, isa, 1);
            for (const unsigned threads : {2u, 5u, 12u}) {
                const auto many = run_gemm(x, w, s.rows, s.n, s.k, kind, isa, threads);
                CHECK(std::memcmp(one.data(), many.data(), one.size() * sizeof(float)) == 0);
            }
            // Rows computed alone (batch of one, and a different position in a smaller batch) match the full batch.
            for (const std::size_t row : {std::size_t{0}, std::size_t{13}, std::size_t{69}}) {
                std::vector<float> single(x.begin() + static_cast<std::ptrdiff_t>(row * s.k), x.begin() + static_cast<std::ptrdiff_t>((row + 1) * s.k));
                const auto alone = run_gemm(single, w, 1, s.n, s.k, kind, isa, 3);
                CHECK(std::memcmp(alone.data(), one.data() + row * s.n, s.n * sizeof(float)) == 0);
            }
        }
    }
}

SEPAL_TEST(cpu_gemm, accumulate_adds_to_y) {
    const Shape s{19, 45, 80};
    const auto x = random_matrix(s.rows, s.k, 11);
    const auto w = random_matrix(s.n, s.k, 12);
    for (const Isa isa : isas()) {
        ThreadPool pool(3);
        const PackedWeight pw = pack_weight(w.data(), s.k, s.n, s.k, WeightKind::F32, isa, pool);
        PackedActs pa;
        pack_acts(pa, x.data(), s.k, s.rows, s.k, pw.variant(), pool);
        std::vector<float> base = random_matrix(s.rows, s.n, 13);
        std::vector<float> y = base, plain(s.rows * s.n);
        gemm_packed(pw, pa, y.data(), s.n, true, pool);
        gemm_packed(pw, pa, plain.data(), s.n, false, pool);
        bool same = true;
        for (std::size_t i = 0; i < y.size(); ++i) same = same && y[i] == base[i] + plain[i];
        CHECK(same);
    }
}

SEPAL_TEST(cpu_gemm, swiglu_pair) {
    const Shape s{23, 70, 96};
    const auto x = random_matrix(s.rows, s.k, 14);
    const auto gate = random_matrix(s.n, s.k, 15);
    const auto up = random_matrix(s.n, s.k, 16);
    std::vector<double> g, u, bound;
    reference(x, gate, s.rows, s.n, s.k, g, bound);
    reference(x, up, s.rows, s.n, s.k, u, bound);
    for (const Isa isa : isas()) {
        ThreadPool pool(4);
        const PackedWeight pw = pack_weight_swiglu(gate.data(), up.data(), s.k, s.n, s.k, WeightKind::F32, isa, pool);
        PackedActs pa;
        pack_acts(pa, x.data(), s.k, s.rows, s.k, pw.variant(), pool);
        std::vector<float> y(s.rows * s.n);
        gemm_packed(pw, pa, y.data(), s.n, false, pool);
        double worst = 0;
        for (std::size_t i = 0; i < y.size(); ++i) {
            const double expect = g[i] / (1.0 + std::exp(-g[i])) * u[i];
            worst = std::max(worst, std::fabs(y[i] - expect) / (1.0 + std::fabs(expect)));
        }
        CHECK(worst < 1e-5);
    }
}

SEPAL_TEST(cpu_gemm, provider_interface) {
    const Shape s{17, 40, 50};
    const auto x = random_matrix(s.rows, s.k, 17);
    const auto w = random_matrix(s.n, s.k, 18);
    const auto reference_provider = make_reference_gemm();
    std::vector<float> expect(s.rows * s.n), got(s.rows * s.n);
    reference_provider->gemm_f(ConstMatView(x.data(), s.rows, s.k), ConstMatView(w.data(), s.n, s.k), MatView(expect.data(), s.rows, s.n), 0.0f);
    const auto fast = make_cpu_gemm(WeightKind::F32, best_isa(), 2);
    fast->gemm_f(ConstMatView(x.data(), s.rows, s.k), ConstMatView(w.data(), s.n, s.k), MatView(got.data(), s.rows, s.n), 0.0f);
    double worst = 0;
    for (std::size_t i = 0; i < got.size(); ++i) worst = std::max(worst, static_cast<double>(std::fabs(got[i] - expect[i])));
    CHECK(worst < 1e-5);
    CHECK_THROWS(fast->gemm_f(ConstMatView(x.data(), s.rows, s.k), ConstMatView(w.data(), s.n, s.k - 1), MatView(got.data(), s.rows, s.n), 0.0f), std::invalid_argument);
}
