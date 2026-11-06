// The packed CPU GEMM against the reference provider, on both instruction-set families the machine supports,
// for odd shapes, and the determinism rules: thread counts and batch composition change no bit.

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
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
    if (isa_supported(Isa::Avx2Tuned)) out.push_back(Isa::Avx2Tuned);
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

SEPAL_TEST(cpu_gemm, default_thread_count_is_the_physical_cores) {
    const unsigned n = default_thread_count();
    CHECK(n >= 1);
    CHECK(n <= std::max(1u, std::thread::hardware_concurrency()));
    ThreadPool pool(0);
    CHECK_EQ(pool.size(), n);
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
            // The AVX2 variants keep float activations (only the weights are bf16); AVX-512 BF16 rounds both.
            if (pick_variant(WeightKind::BF16, isa) == Variant::BF16_Z)
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
        // AVX-512 VNNI: one scale per (feature or row, block of kBlockValues values); AVX2: one per feature or row.
        const bool blocked = pick_variant(WeightKind::I8, isa) == Variant::I8_Z;
        for (const Shape s : shapes) {
            const std::size_t block = blocked ? kBlockValues : s.k;
            const std::size_t blocks = (s.k + block - 1) / block;
            const auto x = random_matrix(s.rows, s.k, 5, 3.0f);
            const auto w = random_matrix(s.n, s.k, 6, 0.5f);
            const auto y = run_gemm(x, w, s.rows, s.n, s.k, WeightKind::I8, isa, 4);
            // Emulate: symmetric scales per block, exact integer sums per block, the float sum over blocks in increasing k.
            std::vector<float> ws(s.n * blocks);
            std::vector<std::int8_t> wq(s.n * s.k);
            for (std::size_t j = 0; j < s.n; ++j) {
                for (std::size_t b = 0; b < blocks; ++b) {
                    const std::size_t lo = b * block, hi = std::min(s.k, lo + block);
                    float amax = 0;
                    for (std::size_t i = lo; i < hi; ++i) amax = std::max(amax, std::fabs(w[j * s.k + i]));
                    ws[j * blocks + b] = amax > 0 ? amax / 127.0f : 1.0f;
                    for (std::size_t i = lo; i < hi; ++i)
                        wq[j * s.k + i] = static_cast<std::int8_t>(std::clamp(std::nearbyint(w[j * s.k + i] * (1.0f / ws[j * blocks + b])), -127.0f, 127.0f));
                }
            }
            double worst = 0;
            for (std::size_t t = 0; t < s.rows; ++t) {
                std::vector<float> xs(blocks);
                std::vector<std::int8_t> xq(s.k);
                for (std::size_t b = 0; b < blocks; ++b) {
                    const std::size_t lo = b * block, hi = std::min(s.k, lo + block);
                    float amax = 0;
                    for (std::size_t i = lo; i < hi; ++i) amax = std::max(amax, std::fabs(x[t * s.k + i]));
                    xs[b] = amax > 0 ? amax / 127.0f : 1.0f;
                    for (std::size_t i = lo; i < hi; ++i)
                        xq[i] = static_cast<std::int8_t>(std::clamp(std::nearbyint(x[t * s.k + i] * (1.0f / xs[b])), -127.0f, 127.0f));
                }
                for (std::size_t j = 0; j < s.n; ++j) {
                    float expect = 0;
                    for (std::size_t b = 0; b < blocks; ++b) {
                        std::int64_t acc = 0;
                        for (std::size_t i = b * block; i < std::min(s.k, (b + 1) * block); ++i) acc += static_cast<std::int64_t>(xq[i]) * wq[j * s.k + i];
                        const float scale = ws[j * blocks + b] * xs[b];
                        expect = b == 0 ? static_cast<float>(acc) * scale : std::fma(static_cast<float>(acc), scale, expect);   // the kernel's float chain over blocks
                    }
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

SEPAL_TEST(cpu_gemm, tuned_avx2_is_bit_identical_to_generic_avx2) {
    if (!isa_supported(Isa::Avx2Tuned)) return;
    // Same products, same order of the sums: the tuned kernels (in-kernel bf16 widening, 4x unrolled assembly, cache-derived
    // blocking) must give every bit of the generic AVX2 path, for both float kinds, odd shapes, k blocking and thread counts.
    const Shape extra[] = {{7, 300, 1000}, {97, 129, 2600}, {200, 48, 513}};
    for (const WeightKind kind : {WeightKind::F32, WeightKind::BF16}) {
        for (int set = 0; set < 2; ++set) {
            for (std::size_t i = 0; i < (set == 0 ? std::size(shapes) : std::size(extra)); ++i) {
                const Shape s = set == 0 ? shapes[i] : extra[i];
                const auto x = random_matrix(s.rows, s.k, 31);
                const auto w = random_matrix(s.n, s.k, 32);
                const auto generic = run_gemm(x, w, s.rows, s.n, s.k, kind, Isa::Avx2, 3);
                for (const unsigned threads : {1u, 4u}) {
                    const auto tuned = run_gemm(x, w, s.rows, s.n, s.k, kind, Isa::Avx2Tuned, threads);
                    CHECK(std::memcmp(generic.data(), tuned.data(), generic.size() * sizeof(float)) == 0);
                }
            }
        }
    }
    // SwiGLU pairs and accumulation as well.
    const std::size_t rows = 37, n = 70, k = 777;
    const auto x = random_matrix(rows, k, 41);
    const auto g = random_matrix(n, k, 42), u = random_matrix(n, k, 43);
    for (const WeightKind kind : {WeightKind::F32, WeightKind::BF16}) {
        std::vector<float> out[2];
        int i = 0;
        for (const Isa isa : {Isa::Avx2, Isa::Avx2Tuned}) {
            ThreadPool pool(2);
            const PackedWeight pw = pack_weight_swiglu(g.data(), u.data(), k, n, k, kind, isa, pool);
            PackedActs pa;
            pack_acts(pa, x.data(), k, rows, k, pw.variant(), pool);
            out[i].assign(rows * n, 0.0f);
            gemm_packed(pw, pa, out[i].data(), n, false, pool);
            const PackedWeight plain = pack_weight(g.data(), k, n, k, kind, isa, pool);
            PackedActs pb;
            pack_acts(pb, x.data(), k, rows, k, plain.variant(), pool);
            gemm_packed(plain, pb, out[i].data(), n, true, pool);
            ++i;
        }
        CHECK(std::memcmp(out[0].data(), out[1].data(), out[0].size() * sizeof(float)) == 0);
    }
}

// One product under one output schedule; y starts as `y0` (row stride ldy >= n) so accumulation and untouched padding are checked.
std::vector<float> run_schedule(GemmSchedule sched, const PackedWeight& pw, const PackedActs& pa, const std::vector<float>& y0,
                                std::size_t ldy, bool accumulate, ThreadPool& pool) {
    set_gemm_schedule(sched);
    std::vector<float> y = y0;
    gemm_packed(pw, pa, y.data(), ldy, accumulate, pool);
    set_gemm_schedule(GemmSchedule::Default);
    return y;
}

SEPAL_TEST(cpu_gemm, tuned_schedules_are_bit_identical) {
    if (!isa_supported(Isa::Avx2Tuned)) return;
    // The output schedules only move the moment a finished tile is written: every output byte, the padding between rows
    // and every accumulated value must be those of the Tile schedule (the 0.4 one), for the six Qwen3.5-4B products and odd
    // edges (partial row blocks, partial panels, k not a multiple of the slice or of the 4x unroll), plain, accumulating and
    // SwiGLU, on 1 and 5 threads. Plain products are also checked against the scalar model the Lean proof is about
    // (proofs/ExactTrack/GemmPartition.lean): one fp32 FMA chain per element, in increasing k, starting at +0.
    struct Case { std::size_t rows, n, k; };
    const Case model4b[] = {{13, 9216, 2560}, {13, 2560, 9216}, {13, 12288, 2560}, {7, 2560, 4096}, {1, 10240, 2560}, {7, 2560, 4096}};
    const bool swiglu4b[] = {true, false, false, false, false, false};
    const Case odd[] = {{1, 1, 1}, {5, 7, 3}, {13, 37, 70}, {29, 100, 129}, {97, 129, 2600}, {200, 48, 513}, {127, 33, 257}, {6, 16, 256}};
    const GemmSchedule others[] = {GemmSchedule::Fused, GemmSchedule::FusedPrefetch};
    CHECK(gemm_schedule() != GemmSchedule::Default);
    CHECK(parse_gemm_schedule("fused-prefetch") == GemmSchedule::FusedPrefetch);
    CHECK_THROWS(parse_gemm_schedule("strassen"), std::invalid_argument);
    std::size_t checked = 0;
    for (const unsigned threads : {1u, 5u}) {
        ThreadPool pool(threads);
        for (const WeightKind kind : {WeightKind::F32, WeightKind::BF16}) {
            for (int set = 0; set < 2; ++set) {
                const std::size_t count = set == 0 ? std::size(model4b) : std::size(odd);
                for (std::size_t c = 0; c < count; ++c) {
                    const Case s = set == 0 ? model4b[c] : odd[c];
                    if (set == 0 && (threads != 5 || kind != WeightKind::BF16)) continue; // the large shapes once, as the runner runs them
                    const bool swiglu = set == 0 ? swiglu4b[c] : c % 3 == 1;
                    const auto x = random_matrix(s.rows, s.k, 51 + c);
                    const auto w = random_matrix(swiglu ? 2 * s.n : s.n, s.k, 61 + c, 0.25f);
                    const PackedWeight pw = swiglu ? pack_weight_swiglu(w.data(), w.data() + s.n * s.k, s.k, s.n, s.k, kind, Isa::Avx2Tuned, pool)
                                                   : pack_weight(w.data(), s.k, s.n, s.k, kind, Isa::Avx2Tuned, pool);
                    PackedActs pa;
                    pack_acts(pa, x.data(), s.k, s.rows, s.k, pw.variant(), pool);
                    const std::size_t ldy = s.n + 3; // odd row stride: rows start at every alignment
                    const auto y0 = random_matrix(s.rows, ldy, 71 + c);
                    for (const bool accumulate : {false, true}) {
                        if (accumulate && swiglu) continue;
                        const auto tile = run_schedule(GemmSchedule::Tile, pw, pa, y0, ldy, accumulate, pool);
                        for (const GemmSchedule other : others) {
                            const auto got = run_schedule(other, pw, pa, y0, ldy, accumulate, pool);
                            CHECK(std::memcmp(tile.data(), got.data(), tile.size() * sizeof(float)) == 0);
                            ++checked;
                        }
                        if (set == 1 && !swiglu && !accumulate) {
                            bool model_ok = true;
                            for (std::size_t t = 0; t < s.rows; ++t)
                                for (std::size_t j = 0; j < s.n; ++j) {
                                    float acc = 0.0f;
                                    for (std::size_t i = 0; i < s.k; ++i) {
                                        const float wv = kind == WeightKind::BF16 ? round_bf16(w[j * s.k + i]) : w[j * s.k + i];
                                        acc = std::fma(x[t * s.k + i], wv, acc);
                                    }
                                    model_ok = model_ok && std::memcmp(&acc, &tile[t * ldy + j], sizeof(float)) == 0;
                                    for (std::size_t j2 = s.n; j2 < ldy && j == 0; ++j2) model_ok = model_ok && tile[t * ldy + j2] == y0[t * ldy + j2];
                                }
                            CHECK(model_ok);
                        }
                    }
                }
            }
        }
    }
    CHECK(checked > 40);
}

SEPAL_TEST(cpu_gemm, kernel_sets_names_features_and_refusal) {
    CHECK(parse_isa("avx2") == Isa::Avx2);
    CHECK(parse_isa("avx2-tuned") == Isa::Avx2Tuned);
    CHECK(parse_isa("x86-64-v3") == Isa::Avx2Tuned);
    CHECK(parse_isa("avx512") == Isa::Avx512);
    CHECK_THROWS(parse_isa("sse2"), std::invalid_argument);
    CHECK(std::string(name_of(Isa::Avx2Tuned)) == "avx2-tuned");
    CHECK(missing_features(Isa::Avx2).empty());
    // A set the CPU lacks is refused with the missing features named, never run.
    for (const Isa isa : {Isa::Avx2Tuned, Isa::Avx512}) {
        if (isa_supported(isa)) {
            set_isa_override(isa);
            CHECK(best_isa() == isa);
            clear_isa_override();
            continue;
        }
        bool named = false;
        try {
            require_isa(isa);
        } catch (const std::runtime_error& e) {
            named = std::string(e.what()).find(missing_features(isa)) != std::string::npos;
        }
        CHECK(named);
        CHECK_THROWS(set_isa_override(isa), std::runtime_error);
    }
    if (isa_supported(Isa::Avx2Tuned) && !isa_supported(Isa::Avx512)) CHECK(best_isa() == Isa::Avx2Tuned || best_isa() == Isa::Avx2);
    // The tuned blocking fits the caches it was derived from.
    const GemmBlocking b = gemm_blocking(Variant::BF16_Y3);
    CHECK(b.kc >= 64 && b.kc <= 1024 && b.rows % 6 == 0 && b.cols % 16 == 0);
    if (cache_sizes().l1d != 0) CHECK(b.kc * (16 * 2 + 6 * 4) <= cache_sizes().l1d);
}

SEPAL_TEST(cpu_gemm, cpu_lists_and_pinned_pools) {
    const std::vector<CpuSlot> all = cpu_slots();
    CHECK_THROWS(cpu_slots(std::string("x-3")), std::invalid_argument);
    CHECK_THROWS(cpu_slots(std::string("5-2")), std::invalid_argument);
    if (all.empty()) return;  // no topology here (not Linux)
    CHECK(physical_cores(all) >= 1);
    // First threads of every core come first, then hyper-threads.
    bool seen_sibling = false, order_ok = true;
    for (const CpuSlot& s : all) {
        if (!s.first_thread) seen_sibling = true;
        else if (seen_sibling) order_ok = false;
    }
    CHECK(order_ok);
    const std::string one = std::to_string(all[0].cpu);
    const std::vector<CpuSlot> single = cpu_slots(one);
    CHECK_EQ(single.size(), std::size_t{1});
    CHECK_THROWS(cpu_slots(std::string("100000")), std::invalid_argument);
    // A pool on an explicit list runs every task exactly once and reports its layout.
    const std::string two = all.size() >= 2 ? one + "," + std::to_string(all[1].cpu) : one;
    ThreadPool pool(0, two);
    CHECK(pool.size() >= 1);
    std::vector<std::atomic<int>> hits(500);
    pool.parallel_for(500, [&](std::size_t i, unsigned) { hits[i].fetch_add(1); });
    bool all_once = true;
    for (auto& h : hits) all_once = all_once && h.load() == 1;
    CHECK(all_once);
    CHECK(!pool.layout().empty());
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
