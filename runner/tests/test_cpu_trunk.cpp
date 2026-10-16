// The CPU trunk pieces against the reference path: the Gated DeltaNet recurrence (token-sequential and chunked),
// flash-style attention, and the whole trunk layer by layer in float32 and float64, plus determinism.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "sepal/config.hpp"
#include "sepal/cpu/attention.hpp"
#include "sepal/cpu/deltanet.hpp"
#include "sepal/cpu/elementwise.hpp"
#include "sepal/cpu/isa.hpp"
#include "sepal/cpu/trunk.hpp"
#include "sepal/ref/deltanet.hpp"
#include "sepal/ref/executor.hpp"
#include "sepal/weights.hpp"
#include "test.hpp"

namespace {

using namespace sepal;
using namespace sepal::cpu;

struct Rng {
    std::uint64_t s;
    explicit Rng(std::uint64_t seed) : s(seed * 2654435761u + 12345u) {}
    double next() { // uniform in [-1, 1)
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        return static_cast<double>(static_cast<std::int32_t>(s >> 33)) / 1073741824.0 - 1.0;
    }
};

// ---- DeltaNet ---------------------------------------------------------------------------------------------------

struct DeltaCase {
    std::size_t tokens, key_heads, value_heads, dk, dv;
    std::vector<float> qk_in;  // [T, 2 * key_heads * dk]: q then k, normalised
    std::vector<float> v, g, beta, state0;
};

DeltaCase make_delta_case(std::size_t tokens, std::size_t hk, std::size_t hv, std::size_t dk, std::size_t dv, std::uint64_t seed) {
    DeltaCase c{tokens, hk, hv, dk, dv, {}, {}, {}, {}, {}};
    Rng rng(seed);
    const std::size_t key_dim = hk * dk;
    c.qk_in.resize(tokens * 2 * key_dim);
    for (std::size_t t = 0; t < tokens; ++t) {
        for (int part = 0; part < 2; ++part) {
            for (std::size_t h = 0; h < hk; ++h) {
                float* x = c.qk_in.data() + t * 2 * key_dim + static_cast<std::size_t>(part) * key_dim + h * dk;
                double sum = 0;
                for (std::size_t i = 0; i < dk; ++i) {
                    x[i] = static_cast<float>(rng.next());
                    sum += static_cast<double>(x[i]) * x[i];
                }
                for (std::size_t i = 0; i < dk; ++i) x[i] = static_cast<float>(x[i] / std::sqrt(sum + 1e-6));
            }
        }
    }
    c.v.resize(tokens * hv * dv);
    for (float& x : c.v) x = static_cast<float>(rng.next());
    c.g.resize(tokens * hv);
    c.beta.resize(tokens * hv);
    for (std::size_t i = 0; i < c.g.size(); ++i) {
        c.g[i] = static_cast<float>(-0.02 - 1.5 * std::fabs(rng.next()) * std::fabs(rng.next()) * 4);  // decay between ~1 and ~0.002
        c.beta[i] = static_cast<float>(0.5 + 0.45 * rng.next());
    }
    c.state0.resize(hv * dk * dv);
    for (float& x : c.state0) x = static_cast<float>(0.3 * rng.next());
    return c;
}

struct DeltaResult {
    std::vector<float> out, state;
};

DeltaResult run_cpu_delta(const DeltaCase& c, std::size_t chunk, unsigned threads, std::size_t split_at = 0) {
    ThreadPool pool(threads);
    const std::size_t key_dim = c.key_heads * c.dk, ld_in = 2 * key_dim + c.value_heads * c.dv;
    // Row layout of the conv output: q | k | v.
    std::vector<float> rows(c.tokens * ld_in);
    for (std::size_t t = 0; t < c.tokens; ++t) {
        std::memcpy(rows.data() + t * ld_in, c.qk_in.data() + t * 2 * key_dim, 2 * key_dim * sizeof(float));
        std::memcpy(rows.data() + t * ld_in + 2 * key_dim, c.v.data() + t * c.value_heads * c.dv, c.value_heads * c.dv * sizeof(float));
    }
    DeltaResult r;
    r.out.assign(c.tokens * c.value_heads * c.dv, -1.0f);
    r.state = c.state0;
    const DeltaGeometry geo{c.key_heads, c.value_heads, c.dk, c.dv};
    const float scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(c.dk)));
    const std::size_t ld_out = c.value_heads * c.dv;
    auto segment = [&](std::size_t t0, std::size_t n) {
        DeltaSegment s;
        s.rows = n;
        s.q = rows.data() + t0 * ld_in;
        s.k = s.q + key_dim;
        s.v = s.q + 2 * key_dim;
        s.g = c.g.data() + t0 * c.value_heads;
        s.beta = c.beta.data() + t0 * c.value_heads;
        s.state = r.state.data();
        s.out = r.out.data() + t0 * ld_out;
        return s;
    };
    const std::size_t cut = split_at == 0 || split_at >= c.tokens ? c.tokens : split_at;
    for (std::size_t t0 = 0; t0 < c.tokens; t0 += cut) {
        const DeltaSegment s = segment(t0, std::min(cut, c.tokens - t0));
        if (chunk == 0) delta_scan_sequential(geo, {&s, 1}, ld_in, ld_out, scale, pool);
        else delta_scan_chunked(geo, {&s, 1}, ld_in, ld_out, scale, chunk, pool);
    }
    return r;
}

DeltaResult run_ref_delta(const DeltaCase& c) {
    const std::size_t hv = c.value_heads, dk = c.dk, dv = c.dv, hk = c.key_heads;
    const std::size_t key_dim = hk * dk;
    TensorD q({c.tokens, hv, dk}), k({c.tokens, hv, dk}), v({c.tokens, hv, dv}), decay({c.tokens, hv}), beta({c.tokens, hv});
    TensorD state({hv, dk, dv});
    for (std::size_t t = 0; t < c.tokens; ++t) {
        for (std::size_t h = 0; h < hv; ++h) {
            const std::size_t kh = h / (hv / hk);
            for (std::size_t i = 0; i < dk; ++i) {
                q[(t * hv + h) * dk + i] = c.qk_in[t * 2 * key_dim + kh * dk + i];
                k[(t * hv + h) * dk + i] = c.qk_in[t * 2 * key_dim + key_dim + kh * dk + i];
            }
            for (std::size_t j = 0; j < dv; ++j) v[(t * hv + h) * dv + j] = c.v[(t * hv + h) * dv + j];
            decay[t * hv + h] = std::exp(static_cast<double>(c.g[t * hv + h]));
            beta[t * hv + h] = c.beta[t * hv + h];
        }
    }
    for (std::size_t i = 0; i < state.size(); ++i) state[i] = c.state0[i];
    const auto scan = ref::delta_rule_scan<double>(q, k, v, decay, beta, state, 1.0 / std::sqrt(static_cast<double>(dk)));
    DeltaResult r;
    for (std::size_t i = 0; i < scan.reads.size(); ++i) r.out.push_back(static_cast<float>(scan.reads[i]));
    for (std::size_t i = 0; i < scan.state.size(); ++i) r.state.push_back(static_cast<float>(scan.state[i]));
    return r;
}

double max_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double worst = 0;
    for (std::size_t i = 0; i < a.size(); ++i) worst = std::max(worst, static_cast<double>(std::fabs(a[i] - b[i])));
    return worst;
}

bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

} // namespace

SEPAL_TEST(cpu_deltanet, sequential_and_chunked_match_the_reference_recurrence) {
    struct Geometry {
        std::size_t hk, hv, dk, dv;
    };
    for (const Geometry g : {Geometry{2, 4, 16, 32}, Geometry{1, 2, 24, 24}, Geometry{2, 2, 128, 128}}) {
        for (const std::size_t tokens : {std::size_t{1}, std::size_t{7}, std::size_t{64}, std::size_t{65}, std::size_t{200}}) {
            const DeltaCase c = make_delta_case(tokens, g.hk, g.hv, g.dk, g.dv, 100 + tokens);
            const DeltaResult ref = run_ref_delta(c);
            const DeltaResult seq = run_cpu_delta(c, 0, 3);
            CHECK(max_diff(seq.out, ref.out) < 2e-5);
            CHECK(max_diff(seq.state, ref.state) < 2e-5);
            for (const std::size_t chunk : {std::size_t{16}, std::size_t{64}}) {
                const DeltaResult ch = run_cpu_delta(c, chunk, 4);
                CHECK(max_diff(ch.out, ref.out) < 5e-5);
                CHECK(max_diff(ch.state, ref.state) < 5e-5);
            }
        }
    }
}

SEPAL_TEST(cpu_deltanet, carrying_state_across_calls_equals_one_call) {
    const DeltaCase c = make_delta_case(300, 2, 4, 16, 32, 7);
    const DeltaResult whole = run_ref_delta(c);
    const DeltaResult pieces = run_cpu_delta(c, 64, 3, 128);
    CHECK(max_diff(pieces.out, whole.out) < 5e-5);
    CHECK(max_diff(pieces.state, whole.state) < 5e-5);
}

SEPAL_TEST(cpu_deltanet, thread_count_changes_no_bit) {
    const DeltaCase c = make_delta_case(333, 2, 4, 16, 32, 9);
    for (const std::size_t chunk : {std::size_t{0}, std::size_t{64}}) {
        const DeltaResult one = run_cpu_delta(c, chunk, 1);
        for (const unsigned threads : {2u, 5u, 12u}) {
            const DeltaResult many = run_cpu_delta(c, chunk, threads);
            CHECK(same_bits(one.out, many.out));
            CHECK(same_bits(one.state, many.state));
        }
    }
}

SEPAL_TEST(cpu_deltanet, long_sequence_with_strong_decay_stays_finite) {
    DeltaCase c = make_delta_case(1500, 2, 2, 16, 16, 11);
    for (float& g : c.g) g *= 30.0f; // per-token log decay far below the float exp range over a chunk
    const DeltaResult ref = run_ref_delta(c);
    const DeltaResult ch = run_cpu_delta(c, 64, 3);
    bool finite = true;
    for (const float x : ch.out) finite = finite && std::isfinite(x);
    CHECK(finite);
    CHECK(max_diff(ch.out, ref.out) < 1e-4);
}

// ---- attention --------------------------------------------------------------------------------------------------

SEPAL_TEST(cpu_attention, matches_a_double_precision_softmax) {
    struct Shape {
        std::size_t heads, kv, hd;
    };
    for (const Isa isa : {Isa::Avx2, Isa::Avx512}) {
        if (!isa_supported(isa)) continue;
        for (const Shape sh : {Shape{4, 2, 32}, Shape{8, 2, 64}, Shape{2, 2, 32}}) {
            const std::size_t total = 330, start = 100;
            const std::size_t rows = total - start;
            Rng rng(5);
            std::vector<float> keys(total * sh.kv * sh.hd), vals(total * sh.kv * sh.hd);
            for (float& x : keys) x = static_cast<float>(rng.next());
            for (float& x : vals) x = static_cast<float>(rng.next());
            KvCache cache(sh.kv, sh.hd, total, isa);
            for (std::size_t p = 0; p < total; ++p)
                for (std::size_t g = 0; g < sh.kv; ++g) cache.append(p, g, keys.data() + (p * sh.kv + g) * sh.hd, vals.data() + (p * sh.kv + g) * sh.hd);
            std::vector<float> q(rows * sh.heads * sh.hd), gate(rows * sh.heads * 2 * sh.hd);
            for (float& x : q) x = static_cast<float>(rng.next() * 2);
            for (float& x : gate) x = static_cast<float>(rng.next() * 2);
            std::vector<float> out(rows * sh.heads * sh.hd, 0.0f);
            AttentionSegment seg;
            seg.cache = &cache;
            seg.start = start;
            seg.rows = rows;
            seg.q = q.data();
            seg.gate = gate.data() + sh.hd; // gate follows the query inside each head's pair
            seg.out = out.data();
            ThreadPool pool(4);
            attention_forward({sh.heads, sh.kv, sh.hd}, isa, {&seg, 1}, sh.heads * sh.hd, sh.heads * 2 * sh.hd, 2 * sh.hd, sh.heads * sh.hd, pool);
            double worst = 0;
            const std::size_t group = sh.heads / sh.kv;
            for (std::size_t t = 0; t < rows; ++t) {
                for (std::size_t h = 0; h < sh.heads; ++h) {
                    const std::size_t g = h / group;
                    const std::size_t visible = start + t + 1;
                    std::vector<double> s(visible);
                    double peak = -1e300;
                    for (std::size_t j = 0; j < visible; ++j) {
                        double dot = 0;
                        for (std::size_t d = 0; d < sh.hd; ++d) dot += static_cast<double>(q[(t * sh.heads + h) * sh.hd + d]) * keys[(j * sh.kv + g) * sh.hd + d];
                        s[j] = dot;
                        peak = std::max(peak, dot);
                    }
                    double sum = 0;
                    for (double& x : s) {
                        x = std::exp(x - peak);
                        sum += x;
                    }
                    for (std::size_t d = 0; d < sh.hd; ++d) {
                        double o = 0;
                        for (std::size_t j = 0; j < visible; ++j) o += s[j] * vals[(j * sh.kv + g) * sh.hd + d];
                        const double gt = gate[t * sh.heads * 2 * sh.hd + h * 2 * sh.hd + sh.hd + d];
                        const double expect = o / sum / (1.0 + std::exp(-gt));
                        worst = std::max(worst, std::fabs(expect - out[(t * sh.heads + h) * sh.hd + d]));
                    }
                }
            }
            CHECK(worst < 2e-5);
        }
    }
}

// ---- the whole trunk --------------------------------------------------------------------------------------------

namespace {

ModelConfig small_config() {
    ModelConfig c;
    c.name = "cpu-test";
    c.num_layers = 4;
    c.full_attention_interval = 4;
    c.hidden_size = 64;
    c.intermediate_size = 96;
    c.attn_heads = 4;
    c.attn_kv_heads = 2;
    c.attn_head_dim = 32;
    c.dn_key_heads = 2;
    c.dn_value_heads = 4;
    c.dn_key_head_dim = 16;
    c.dn_value_head_dim = 16;
    c.conv_kernel = 4;
    c.vocab_size = 64;
    c.rms_norm_eps = 1e-6;
    c.rope_theta = 1e7;
    c.partial_rotary_factor = 0.25;
    c.pointer_dim = 12;
    c.lm_head_tied = true;
    return c;
}

std::vector<std::uint32_t> tokens_of(std::size_t n, std::uint64_t seed, std::uint32_t vocab) {
    Rng rng(seed);
    std::vector<std::uint32_t> t(n);
    for (auto& x : t) x = static_cast<std::uint32_t>((rng.next() + 1.0) * 0.5 * (vocab - 1));
    return t;
}

CpuTrunk::EmbeddingFn embedding_of(const Weights<float>& w, std::size_t hidden) {
    const BasicTensor<float>* table = &w["embed_tokens.weight"];
    return [table, hidden](std::uint32_t token, float* out) { std::memcpy(out, table->data() + token * hidden, hidden * sizeof(float)); };
}

} // namespace

SEPAL_TEST(cpu_trunk, float32_matches_the_reference_layer_by_layer) {
    const ModelConfig cfg = small_config();
    const Weights<float> wf = random_weights<float>(cfg, 3);
    const Weights<double> wd = random_weights<double>(cfg, 3);
    const auto provider = make_reference_gemm();
    for (const std::size_t n : {std::size_t{1}, std::size_t{9}, std::size_t{70}, std::size_t{200}}) {
        const std::vector<std::uint32_t> tokens = tokens_of(n, 40 + n, 64);
        // Reference in float64, per layer.
        std::vector<TensorD> ladder;
        auto parent = ref::empty_snapshot<double>(cfg);
        const auto cont = ref::continue_snapshot<double>(*provider, cfg, wd, *parent, tokens, 0, &ladder);
        std::vector<std::size_t> offsets(n);
        for (std::size_t i = 0; i < n; ++i) offsets[i] = i;
        const TensorD final_norm = ref::rms_norm<double>(cont.last_layer.mat(), wd[final_norm_name].span(), cfg.rms_norm_eps);

        for (const std::size_t delta_chunk : {std::size_t{0}, std::size_t{64}}) {
            TrunkOptions opt;
            opt.threads = 3;
            opt.chunk_tokens = 64;  // several prefill chunks even for the short requests
            opt.step_tokens = 128;
            opt.delta_chunk = delta_chunk;
            CpuTrunk trunk(cfg, wf, embedding_of(wf, 64), opt);
            std::vector<std::vector<float>> layer_out(cfg.num_layers, std::vector<float>(n * 64));
            LayerObserver observer = [&](std::size_t, std::size_t layer, std::size_t pos, std::size_t rows, const float* x) {
                std::memcpy(layer_out[layer].data() + pos * 64, x, rows * 64 * sizeof(float));
            };
            std::vector<std::vector<float>> hidden;
            const std::span<const std::uint32_t> req(tokens);
            trunk.forward({&req, 1}, hidden, &observer);
            for (std::size_t l = 0; l < cfg.num_layers; ++l) {
                double worst = 0, scale = 0;
                for (std::size_t i = 0; i < n * 64; ++i) {
                    worst = std::max(worst, std::fabs(layer_out[l][i] - ladder[l][i]));
                    scale = std::max(scale, std::fabs(ladder[l][i]));
                }
                CHECK(worst < 2e-4 * (1.0 + scale));
            }
            double worst = 0;
            for (std::size_t i = 0; i < n * 64; ++i) worst = std::max(worst, std::fabs(hidden[0][i] - final_norm[i]));
            CHECK(worst < 5e-4);
        }
    }
}

SEPAL_TEST(cpu_trunk, no_thread_count_or_batch_changes_a_bit) {
    const ModelConfig cfg = small_config();
    const Weights<float> w = random_weights<float>(cfg, 5);
    const std::vector<std::uint32_t> a = tokens_of(150, 1, 64), b = tokens_of(37, 2, 64), c = tokens_of(260, 3, 64);
    const std::vector<std::span<const std::uint32_t>> all = {a, b, c};
    for (const Profile profile : {Profile::F32, Profile::BF16, Profile::Q8}) {
        TrunkOptions opt;
        opt.profile = profile;
        opt.chunk_tokens = 64;
        opt.step_tokens = 160;
        opt.threads = 1;
        CpuTrunk one(cfg, w, embedding_of(w, 64), opt);
        std::vector<std::vector<float>> h1;
        one.forward(all, h1);
        opt.threads = 7;
        CpuTrunk many(cfg, w, embedding_of(w, 64), opt);
        std::vector<std::vector<float>> h7;
        many.forward(all, h7);
        for (std::size_t i = 0; i < 3; ++i) CHECK(same_bits(h1[i], h7[i]));
        // A request alone gives the same bits as inside the batch.
        for (std::size_t i = 0; i < 3; ++i) {
            std::vector<std::vector<float>> alone;
            many.forward({&all[i], 1}, alone);
            CHECK(same_bits(alone[0], h7[i]));
        }
    }
}

SEPAL_TEST(cpu_trunk, reduced_precision_profiles_stay_near_float32) {
    const ModelConfig cfg = small_config();
    const Weights<float> w = random_weights<float>(cfg, 8);
    const std::vector<std::uint32_t> t = tokens_of(120, 4, 64);
    const std::span<const std::uint32_t> req(t);
    std::vector<std::vector<float>> base;
    {
        TrunkOptions opt;
        opt.threads = 2;
        CpuTrunk trunk(cfg, w, embedding_of(w, 64), opt);
        trunk.forward({&req, 1}, base);
    }
    for (const Profile profile : {Profile::BF16, Profile::Q8}) {
        TrunkOptions opt;
        opt.profile = profile;
        opt.threads = 2;
        CpuTrunk trunk(cfg, w, embedding_of(w, 64), opt);
        std::vector<std::vector<float>> got;
        trunk.forward({&req, 1}, got);
        double err = 0, norm = 0;
        for (std::size_t i = 0; i < base[0].size(); ++i) {
            err += (got[0][i] - base[0][i]) * (got[0][i] - base[0][i]);
            norm += base[0][i] * base[0][i];
        }
        CHECK(std::sqrt(err / norm) < (profile == Profile::BF16 ? 0.05 : 0.2));
    }
}

SEPAL_TEST(cpu_trunk, malformed_requests_are_refused) {
    const ModelConfig cfg = small_config();
    const Weights<float> w = random_weights<float>(cfg, 8);
    TrunkOptions opt;
    opt.threads = 1;
    CpuTrunk trunk(cfg, w, embedding_of(w, 64), opt);
    std::vector<std::uint32_t> empty, bad = {1, 2, 64};
    std::vector<std::vector<float>> out;
    const std::span<const std::uint32_t> e(empty), b(bad);
    CHECK_THROWS(trunk.forward({&e, 1}, out), std::invalid_argument);
    CHECK_THROWS(trunk.forward({&b, 1}, out), std::invalid_argument);
}
