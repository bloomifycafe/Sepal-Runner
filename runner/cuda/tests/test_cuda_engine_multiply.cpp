// The engine's multiply options on the tiny geometry: the batch-invariant kernel (MultiplyPath::Invariant), binary16
// accumulation (Fp16Accumulate) on either path, and what strict mode still guarantees with them. Random weights.
//
// The tiny preset's DeltaNet value width (4 heads of 5) is not a multiple of 8, which the kernel needs of an inner
// dimension, so these tests use a geometry of the same shape with head dimensions the kernel takes.

#include <filesystem>
#include <fstream>
#include <functional>
#include <map>

#include "engine_test_util.hpp"

using namespace sepal;
using namespace engine_test;
using sepal::cuda::CudaEngine;
using sepal::cuda::EngineOptions;
using sepal::cuda::EngineRow;
using sepal::cuda::EngineRowResult;
using sepal::cuda::MultiplyPath;
using sepal::cuda::ParentRow;
using sepal::cuda::Precision;

namespace {

ModelConfig kernel_config() {
    ModelConfig c = tiny_test_config();
    c.name = "tiny-kernel";
    c.dn_key_head_dim = 8;   // key width 16, convolution width 64
    c.dn_value_head_dim = 8; // value width 32, input projection width 104
    return c;
}

const Weights<float>& weights_of(const ModelConfig& cfg, std::uint64_t seed) {
    static std::map<std::pair<std::string, std::uint64_t>, Weights<float>> cache;
    const auto key = std::make_pair(cfg.name, seed);
    auto it = cache.find(key);
    if (it == cache.end()) it = cache.emplace(key, random_weights<float>(cfg, seed)).first;
    return it->second;
}

EngineOptions options_with(MultiplyPath path, bool binary16, std::size_t gemm_rows, unsigned flush_k = 64) {
    EngineOptions o;
    o.precision = Precision::F16;
    o.gemm_rows = gemm_rows;
    o.multiply_path = path;
    o.fp16_accumulate = {binary16, binary16, binary16, binary16, flush_k};
    o.memory_budget_bytes = std::size_t{192} << 20;
    o.max_call_tokens = 4096;
    o.max_call_rows = 256;
    o.state_slots = 128;
    return o;
}

std::vector<EngineRowResult> run(CudaEngine& engine, const std::vector<EngineRow>& rows) {
    return engine.run(std::span<const EngineRow>(rows));
}

std::vector<std::uint32_t> tokens(Rng& rng, std::size_t n) {
    std::vector<std::uint32_t> t(n);
    for (auto& v : t) v = static_cast<std::uint32_t>(rng.below(64));
    return t;
}

EngineRow make_row(std::vector<std::uint32_t> t, std::vector<std::size_t> readouts, bool keep = false) {
    EngineRow r;
    r.token_ids = std::move(t);
    r.readout_offsets = std::move(readouts);
    r.keep_snapshot = keep;
    return r;
}

// Rows of other requests to share a call with.
std::vector<EngineRow> others(Rng& rng, std::size_t max_len) {
    FuzzOptions fo;
    fo.max_len = max_len;
    fo.allow_store = false;
    return make_request(rng, {}, fo).engine;
}

// The rows of `subject` (whose parents count within it) placed after `before` other rows.
std::vector<EngineRow> splice(const std::vector<EngineRow>& before, const std::vector<EngineRow>& subject, const std::vector<EngineRow>& after) {
    std::vector<EngineRow> out = before;
    for (EngineRow r : subject) {
        if (auto* p = std::get_if<ParentRow>(&r.parent)) p->index += before.size();
        out.push_back(std::move(r));
    }
    for (EngineRow r : after) {
        if (auto* p = std::get_if<ParentRow>(&r.parent)) p->index += before.size() + subject.size();
        out.push_back(std::move(r));
    }
    return out;
}

struct Counts {
    std::size_t trials = 0, identical = 0;
};

// A state with branches: alone, then among random other rows at random places, then with the state stored by an earlier call.
Counts family_invariance(CudaEngine& engine, std::uint64_t seed, std::size_t contexts) {
    Rng rng(seed);
    Counts counts;
    for (std::size_t trial = 0; trial < contexts; ++trial) {
        const std::vector<std::uint32_t> state = tokens(rng, 20 + rng.below(120));
        std::vector<std::vector<std::uint32_t>> suffixes;
        for (std::size_t b = 0, nb = 1 + rng.below(5); b < nb; ++b) suffixes.push_back(tokens(rng, 1 + rng.below(30)));
        std::vector<EngineRow> family = {make_row(state, {})};
        for (const auto& s : suffixes) {
            EngineRow b = make_row(s, {s.size() - 1, 0});
            b.parent = ParentRow{0};
            family.push_back(b);
        }
        const auto alone = run(engine, family);
        // Among other rows, before and after.
        const std::vector<EngineRow> before = others(rng, 150), after = others(rng, 60);
        const auto among = run(engine, splice(before, family, after));
        for (std::size_t i = 0; i < family.size(); ++i) {
            ++counts.trials;
            if (same_bits(alone[i].hidden, among[before.size() + i].hidden)) ++counts.identical;
        }
        // The state stored by one call, the branches continuing it in another.
        const auto stored = run(engine, {make_row(state, {}, true)});
        std::vector<EngineRow> branches;
        for (const auto& s : suffixes) {
            EngineRow b = make_row(s, {s.size() - 1, 0});
            b.parent = stored[0].snapshot;
            branches.push_back(b);
        }
        const std::vector<EngineRow> pre = others(rng, 40);
        const auto hit = run(engine, splice(pre, branches, {}));
        for (std::size_t i = 0; i < branches.size(); ++i) {
            ++counts.trials;
            if (same_bits(alone[i + 1].hidden, hit[pre.size() + i].hidden)) ++counts.identical;
        }
    }
    return counts;
}

double mean_distance(const std::vector<EngineRowResult>& a, const std::vector<EngineRowResult>& b, double* worst) {
    double sum = 0;
    std::size_t count = 0;
    for (std::size_t r = 0; r < a.size(); ++r) {
        for (std::size_t i = 0; i < a[r].hidden.size(); ++i) {
            const double d = std::fabs(static_cast<double>(a[r].hidden[i]) - b[r].hidden[i]);
            *worst = std::max(*worst, d);
            sum += d;
            ++count;
        }
    }
    return sum / static_cast<double>(std::max<std::size_t>(count, 1));
}

} // namespace

SEPAL_TEST(engine_multiply, defaults_and_refusals) {
    const EngineOptions defaults;
    CHECK(defaults.multiply_path == MultiplyPath::Auto);
    CHECK(!defaults.fp16_accumulate.gate_up && !defaults.fp16_accumulate.down && !defaults.fp16_accumulate.mixer_in && !defaults.fp16_accumulate.mixer_out);
    CHECK_EQ(defaults.fp16_accumulate.flush_k, 128u);
    const ModelConfig cfg = kernel_config();
    bool refused = false;
    try {
        CudaEngine engine(cfg, weights_of(cfg, 1), options_with(MultiplyPath::Invariant, true, 0, 48));
    } catch (const std::invalid_argument&) {
        refused = true;
    }
    CHECK(refused); // a flush interval that is not a multiple of 32
}

namespace {

EngineOptions default_options(std::size_t gemm_rows) {
    EngineOptions o = options_with(MultiplyPath::Auto, false, gemm_rows);
    o.fp16_accumulate = sepal::cuda::Fp16Accumulate{}; // untouched: Auto chooses it
    return o;
}

bool all_four(const sepal::cuda::Fp16Accumulate& f) { return f.gate_up && f.down && f.mixer_in && f.mixer_out; }
bool none_of(const sepal::cuda::Fp16Accumulate& f) { return !f.gate_up && !f.down && !f.mixer_in && !f.mixer_out; }

} // namespace

SEPAL_TEST(engine_multiply, the_default_path_resolves_by_mode) {
    const ModelConfig cfg = kernel_config();
    const Weights<float>& weights = weights_of(cfg, 9);
    {
        // Natural mode: cuBLAS with binary16 accumulation in all four groups.
        CudaEngine engine(cfg, weights, default_options(0));
        CHECK(engine.options().multiply_path == MultiplyPath::Cublas);
        CHECK(all_four(engine.options().fp16_accumulate));
    }
    for (const std::size_t gemm_rows : {std::size_t{16}, std::size_t{48}}) {
        // Strict mode: the own kernel, binary16 accumulators flushed every 64 values of k, all four groups.
        CudaEngine engine(cfg, weights, default_options(gemm_rows));
        CHECK(engine.options().multiply_path == MultiplyPath::Invariant);
        CHECK(all_four(engine.options().fp16_accumulate));
        CHECK_EQ(engine.options().fp16_accumulate.flush_k, 64u);
    }
    {
        // The float32 path has nothing to choose.
        EngineOptions o = default_options(0);
        o.precision = Precision::F32;
        CudaEngine engine(cfg, weights, o);
        CHECK(engine.options().multiply_path == MultiplyPath::Cublas);
        CHECK(none_of(engine.options().fp16_accumulate));
    }
    {
        // Explicit settings are left as they are.
        CudaEngine engine(cfg, weights, options_with(MultiplyPath::Cublas, false, 0));
        CHECK(engine.options().multiply_path == MultiplyPath::Cublas);
        CHECK(none_of(engine.options().fp16_accumulate));
    }
    {
        // An accumulation setting next to Auto is refused rather than ignored.
        EngineOptions o = default_options(0);
        o.fp16_accumulate.gate_up = true;
        bool refused = false;
        try {
            CudaEngine engine(cfg, weights, o);
        } catch (const std::invalid_argument&) {
            refused = true;
        }
        CHECK(refused);
    }
}

SEPAL_TEST(engine_multiply, the_default_keeps_strict_mode_bit_identical_and_natural_mode_close_to_float32) {
    // Strict mode with the default path: the own kernel where it takes the shapes (kernel_config), padded cuBLAS with binary16
    // accumulation where it does not (the tiny preset, whose DeltaNet value width is not a multiple of 8).
    for (const ModelConfig& cfg : {kernel_config(), tiny_test_config()}) {
        const Weights<float>& weights = weights_of(cfg, 10);
        for (const std::size_t gemm_rows : {std::size_t{16}, std::size_t{48}}) {
            CudaEngine engine(cfg, weights, default_options(gemm_rows));
            const Counts c = family_invariance(engine, 1100 + gemm_rows, 8);
            std::cout << "  default engine on " << cfg.name << ", strict " << gemm_rows << ": " << c.identical << " of " << c.trials
                      << " rows bit-identical alone, among other rows, and continued from a stored snapshot\n";
            CHECK_EQ(c.identical, c.trials);
        }
    }
    // Natural mode with the default: stays close to the float32 engine.
    const ModelConfig cfg = kernel_config();
    const Weights<float>& weights = weights_of(cfg, 11);
    EngineOptions f32;
    f32.precision = Precision::F32;
    f32.memory_budget_bytes = std::size_t{192} << 20;
    f32.max_call_tokens = 4096;
    f32.state_slots = 128;
    CudaEngine reference(cfg, weights, f32);
    CudaEngine engine(cfg, weights, default_options(0));
    Rng rng(41);
    double worst = 0, mean_sum = 0;
    for (std::size_t call = 0; call < 24; ++call) {
        const std::vector<EngineRow> request = others(rng, 200);
        mean_sum += mean_distance(run(engine, request), run(reference, request), &worst);
    }
    std::cout << "  default engine, natural, against the float32 engine on the tiny geometry: mean |difference| " << std::scientific << std::setprecision(2) << mean_sum / 24.0
              << ", largest " << worst << std::fixed << "\n";
    CHECK(mean_sum / 24.0 < 0.06);
}

SEPAL_TEST(engine_multiply, the_variants_stay_close_to_the_float32_engine) {
    const ModelConfig cfg = kernel_config();
    const Weights<float>& weights = weights_of(cfg, 5);
    EngineOptions f32;
    f32.precision = Precision::F32;
    f32.memory_budget_bytes = std::size_t{192} << 20;
    f32.max_call_tokens = 4096;
    f32.state_slots = 128;
    CudaEngine reference(cfg, weights, f32);
    struct Variant {
        const char* name;
        EngineOptions options;
        double mean_allowed;
    };
    const std::vector<Variant> variants = {
        {"cuBLAS, float32 accumulation", options_with(MultiplyPath::Cublas, false, 0), 0.03},
        {"cuBLAS, binary16 accumulation everywhere", options_with(MultiplyPath::Cublas, true, 0), 0.06},
        {"cuBLAS, binary16 accumulation, one group at a time (mixer_in)", [] { EngineOptions o = options_with(MultiplyPath::Cublas, false, 0); o.fp16_accumulate.mixer_in = true; return o; }(), 0.06},
        {"cuBLAS, binary16 accumulation, one group at a time (mixer_out + down)", [] { EngineOptions o = options_with(MultiplyPath::Cublas, false, 0); o.fp16_accumulate.mixer_out = o.fp16_accumulate.down = true; return o; }(), 0.06},
        {"own kernel, float32 accumulation", options_with(MultiplyPath::Invariant, false, 0), 0.03},
        {"own kernel, binary16 accumulation, flush 64", options_with(MultiplyPath::Invariant, true, 0, 64), 0.04},
        {"own kernel, binary16 accumulation, flush 256", options_with(MultiplyPath::Invariant, true, 0, 256), 0.05},
        {"own kernel, binary16 accumulation, no flush", options_with(MultiplyPath::Invariant, true, 0, 0), 0.06},
    };
    for (const Variant& v : variants) {
        CudaEngine engine(cfg, weights, v.options);
        Rng rng(31);
        double worst = 0, mean_sum = 0;
        std::size_t calls = 0;
        for (std::size_t call = 0; call < 24; ++call) {
            const std::vector<EngineRow> request = others(rng, 200);
            const auto want = run(reference, request);
            const auto got = run(engine, request);
            for (const auto& g : got) CHECK(all_finite(g.hidden));
            mean_sum += mean_distance(got, want, &worst);
            ++calls;
        }
        const double mean = mean_sum / static_cast<double>(calls);
        std::cout << "  " << v.name << " against the float32 engine on the tiny geometry: mean |difference| " << std::scientific << std::setprecision(2) << mean << ", largest "
                  << worst << std::fixed << "\n";
        CHECK(mean < v.mean_allowed);
    }
}

SEPAL_TEST(engine_multiply, the_own_kernel_makes_every_batch_composition_give_the_same_bits) {
    const ModelConfig cfg = kernel_config();
    const Weights<float>& weights = weights_of(cfg, 6);
    for (const bool binary16 : {false, true}) {
        // gemm_rows 0 is natural mode: with the own kernel it is invariant as well; 16 and 48 are strict mode, which pads nothing.
        for (const std::size_t gemm_rows : {std::size_t{0}, std::size_t{16}, std::size_t{48}}) {
            CudaEngine engine(cfg, weights, options_with(MultiplyPath::Invariant, binary16, gemm_rows));
            const Counts c = family_invariance(engine, 700 + gemm_rows, 10);
            std::cout << "  own kernel, " << (binary16 ? "binary16" : "float32") << " accumulation, gemm_rows " << gemm_rows << ": " << c.identical << " of " << c.trials
                      << " rows bit-identical alone, among other rows, and continued from a stored snapshot\n";
            CHECK_EQ(c.identical, c.trials);
        }
    }
}

SEPAL_TEST(engine_multiply, strict_cublas_with_binary16_accumulation_is_still_bit_identical) {
    // cuBLAS picks its kernel from the shape and the binary16 kernels round differently from the float32 ones, so the
    // last layer must run the same multiplies whether or not it is pruned to the read-out tokens.
    // (And with float32 accumulation, explicitly: it was the default of strict mode, and padding is what keeps it invariant.)
    const ModelConfig cfg = tiny_test_config();
    const Weights<float>& weights = weights_of(cfg, 7);
    for (const bool binary16 : {true, false})
    for (const std::size_t gemm_rows : {std::size_t{16}, std::size_t{48}}) {
        EngineOptions with = options_with(MultiplyPath::Cublas, binary16, gemm_rows);
        EngineOptions without = with;
        without.prune_last_layer = false;
        CudaEngine pruned(cfg, weights, with);
        CudaEngine full(cfg, weights, without);
        const Counts c = family_invariance(pruned, 800 + gemm_rows, 8);
        Rng rng(gemm_rows);
        std::size_t compared = 0, identical = 0;
        for (std::size_t trial = 0; trial < 30; ++trial) {
            const std::vector<EngineRow> request = others(rng, trial % 2 == 0 ? 300 : 40);
            const auto a = run(pruned, request);
            const auto b = run(full, request);
            for (std::size_t r = 0; r < a.size(); ++r) {
                ++compared;
                if (same_bits(a[r].hidden, b[r].hidden)) ++identical;
            }
        }
        std::cout << "  strict cuBLAS, " << (binary16 ? "binary16" : "float32") << " accumulation, gemm_rows " << gemm_rows << ": " << c.identical << " of " << c.trials
                  << " rows bit-identical alone / among others / from a snapshot; " << identical << " of " << compared << " rows identical between pruned and full last layer\n";
        CHECK_EQ(c.identical, c.trials);
        CHECK_EQ(identical, compared);
    }
}

SEPAL_TEST(engine_multiply, repeated_runs_of_the_own_kernel_are_bit_identical) {
    // Repeated runs of the same call give the same bits (no race in the kernel's pipeline), in both modes and both accumulations.
    const ModelConfig cfg = kernel_config();
    const Weights<float>& weights = weights_of(cfg, 8);
    for (const bool binary16 : {false, true}) {
        CudaEngine engine(cfg, weights, options_with(MultiplyPath::Invariant, binary16, 0));
        Rng rng(5);
        for (std::size_t trial = 0; trial < 8; ++trial) {
            const std::vector<EngineRow> request = others(rng, 250);
            const auto first = run(engine, request);
            for (std::size_t rep = 0; rep < 3; ++rep) {
                const auto again = run(engine, request);
                for (std::size_t r = 0; r < request.size(); ++r) CHECK(same_bits(first[r].hidden, again[r].hidden));
            }
        }
    }
}

// ---- int8 multiplies (EngineOptions::int8_multiply) ---------------------------------------------------------------------

namespace {

// kernel_config() has H = 24, which the int8 kernels refuse (K a multiple of 16); this one has H = 32 and I = 48.
ModelConfig int8_config() {
    ModelConfig c = kernel_config();
    c.name = "tiny-int8";
    c.hidden_size = 32;
    c.intermediate_size = 48;
    return c;
}

sepal::cuda::Int8Multiply int8_groups(bool gate_up, bool down, bool mixer_in, bool mixer_out, bool int8_activations) {
    sepal::cuda::Int8Multiply q;
    q.gate_up = gate_up;
    q.down = down;
    q.mixer_in = mixer_in;
    q.mixer_out = mixer_out;
    q.int8_activations = int8_activations;
    return q;
}

EngineOptions int8_options(const sepal::cuda::Int8Multiply& q, MultiplyPath path, std::size_t gemm_rows) {
    EngineOptions o = options_with(path, false, gemm_rows);
    o.int8_multiply = q;
    return o;
}

// The same groups with Int8Smoothing from the float16 engine's statistics over random tokens.
sepal::cuda::Int8Multiply smoothed(sepal::cuda::Int8Multiply q, const ModelConfig& cfg, const Weights<float>& weights, float alpha = 0.5f,
                                   sepal::cuda::Int8Smoothing::Statistic statistic = sepal::cuda::Int8Smoothing::Statistic::Rms) {
    q.smoothing.stats = collect_stats(cfg, weights);
    q.smoothing.alpha = alpha;
    q.smoothing.statistic = statistic;
    return q;
}

bool refused_by_engine(const std::function<void()>& make) {
    try {
        make();
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

} // namespace

SEPAL_TEST(engine_int8, defaults_and_refusals) {
    const EngineOptions defaults;
    CHECK(!defaults.int8_multiply.any());
    CHECK(defaults.int8_multiply.int8_activations);
    const auto refused = [](auto&& make) {
        try {
            make();
        } catch (const std::invalid_argument&) {
            return true;
        }
        return false;
    };
    const ModelConfig good = int8_config();
    // The float32 path has no int8 multiply.
    CHECK(refused([&] {
        EngineOptions o = int8_options(int8_groups(true, true, true, true, true), MultiplyPath::Cublas, 0);
        o.precision = Precision::F32;
        CudaEngine engine(good, weights_of(good, 1), o);
    }));
    // H = 24 is not a multiple of 16, whichever group is chosen first.
    const ModelConfig bad = kernel_config();
    CHECK(refused([&] { CudaEngine engine(bad, weights_of(bad, 1), int8_options(int8_groups(true, false, false, false, true), MultiplyPath::Cublas, 0)); }));
    CHECK(refused([&] { CudaEngine engine(bad, weights_of(bad, 1), int8_options(int8_groups(false, false, true, false, false), MultiplyPath::Invariant, 0)); }));
    CudaEngine fine(good, weights_of(good, 1), int8_options(int8_groups(true, true, true, true, true), MultiplyPath::Cublas, 0));
    CHECK(fine.options().int8_multiply.any());
}

SEPAL_TEST(engine_int8, weights_take_less_memory_and_the_variants_stay_close_to_the_float32_engine) {
    const ModelConfig cfg = int8_config();
    const Weights<float>& weights = weights_of(cfg, 15);
    EngineOptions f32;
    f32.precision = Precision::F32;
    f32.memory_budget_bytes = std::size_t{192} << 20;
    f32.max_call_tokens = 4096;
    f32.state_slots = 128;
    CudaEngine reference(cfg, weights, f32);
    CudaEngine plain(cfg, weights, options_with(MultiplyPath::Cublas, false, 0));
    struct Variant {
        const char* name;
        sepal::cuda::Int8Multiply q;
        double mean_allowed;
    };
    const std::vector<Variant> variants = {
        {"W8A16, all four", int8_groups(true, true, true, true, false), 0.06},
        {"W8A8, all four", int8_groups(true, true, true, true, true), 0.15},
        {"W8A8, all four, smoothed (rms, alpha 1/2)", smoothed(int8_groups(true, true, true, true, true), cfg, weights), 0.15},
        {"W8A8, all four, smoothed (max, alpha 3/4)", smoothed(int8_groups(true, true, true, true, true), cfg, weights, 0.75f, sepal::cuda::Int8Smoothing::Statistic::Max), 0.15},
        {"W8A8, gate_up and mixer_in, smoothed (mean, alpha 1/2)", smoothed(int8_groups(true, false, true, false, true), cfg, weights, 0.5f, sepal::cuda::Int8Smoothing::Statistic::MeanAbs), 0.08},
        {"W8A8, gate_up", int8_groups(true, false, false, false, true), 0.08},
        {"W8A8, down", int8_groups(false, true, false, false, true), 0.08},
        {"W8A8, mixer_in", int8_groups(false, false, true, false, true), 0.08},
        {"W8A8, mixer_out", int8_groups(false, false, false, true, true), 0.08},
        {"W8A16, mixer_in", int8_groups(false, false, true, false, false), 0.06},
    };
    for (const Variant& v : variants) {
        CudaEngine engine(cfg, weights, int8_options(v.q, MultiplyPath::Cublas, 0));
        const auto mem = engine.memory();
        CHECK(mem.weights_bytes < plain.memory().weights_bytes);
        Rng rng(31);
        double worst = 0, mean_sum = 0;
        std::size_t calls = 0;
        for (std::size_t call = 0; call < 24; ++call) {
            const std::vector<EngineRow> request = others(rng, 200);
            const auto want = run(reference, request);
            const auto got = run(engine, request);
            for (const auto& g : got) CHECK(all_finite(g.hidden));
            mean_sum += mean_distance(got, want, &worst);
            ++calls;
        }
        const double mean = mean_sum / static_cast<double>(calls);
        std::cout << "  " << v.name << " against the float32 engine on the tiny geometry: mean |difference| " << std::scientific << std::setprecision(2) << mean << ", largest "
                  << worst << std::fixed << "; weights " << mem.weights_bytes << " bytes (binary16: " << plain.memory().weights_bytes << ")\n";
        CHECK(mean < v.mean_allowed);
    }
}

SEPAL_TEST(engine_int8, every_batch_composition_gives_the_same_bits) {
    const ModelConfig cfg = int8_config();
    const Weights<float>& weights = weights_of(cfg, 16);
    struct Setup {
        const char* name;
        sepal::cuda::Int8Multiply q;
        MultiplyPath path;
        bool acc16_down = false; // down on the own kernel with binary16 accumulation (flush 64)
    };
    const std::vector<Setup> setups = {
        {"W8A8 all four", int8_groups(true, true, true, true, true), MultiplyPath::Cublas},
        {"W8A16 all four", int8_groups(true, true, true, true, false), MultiplyPath::Cublas},
        {"W8A8 mixer_in only, cuBLAS for the rest", int8_groups(false, false, true, false, true), MultiplyPath::Cublas},
        {"W8A8 down only, own kernel for the rest", int8_groups(false, true, false, false, true), MultiplyPath::Invariant},
        {"W8A8 all four, smoothed", smoothed(int8_groups(true, true, true, true, true), cfg, weights), MultiplyPath::Cublas},
        {"W8A8 gate_up, mixer_in and mixer_out smoothed, own kernel for down", smoothed(int8_groups(true, false, true, true, true), cfg, weights, 0.6f), MultiplyPath::Invariant},
        {"the recommended profile (down: own kernel, binary16 accumulation)", smoothed(int8_groups(true, false, true, true, true), cfg, weights, 0.6f), MultiplyPath::Invariant, true},
    };
    for (const Setup& s : setups) {
        for (const std::size_t gemm_rows : {std::size_t{0}, std::size_t{16}, std::size_t{48}}) {
            // Natural mode with cuBLAS groups left in is not invariant (cuBLAS picks its kernel by shape); strict mode is.
            const bool all_own = s.path == MultiplyPath::Invariant || (s.q.gate_up && s.q.down && s.q.mixer_in && s.q.mixer_out);
            if (gemm_rows == 0 && !all_own) continue;
            EngineOptions options = int8_options(s.q, s.path, gemm_rows);
            if (s.acc16_down) options.fp16_accumulate = {false, true, false, false, 64};
            CudaEngine engine(cfg, weights, options);
            const Counts c = family_invariance(engine, 900 + gemm_rows, 8);
            std::cout << "  " << s.name << ", gemm_rows " << gemm_rows << ": " << c.identical << " of " << c.trials
                      << " rows bit-identical alone, among other rows, and continued from a stored snapshot\n";
            CHECK_EQ(c.identical, c.trials);
        }
    }
}

SEPAL_TEST(engine_int8, strict_pruned_and_full_last_layers_agree_and_repeated_runs_are_identical) {
    const ModelConfig cfg = int8_config();
    const Weights<float>& weights = weights_of(cfg, 17);
    for (const int kind : {0, 1, 2, 3}) { // W8A8, W8A16, W8A8 smoothed, the recommended profile (three groups smoothed, down on the own kernel, binary16 accumulation)
        const bool w8a8 = kind != 1;
        sepal::cuda::Int8Multiply q = int8_groups(true, kind != 3, true, true, w8a8);
        if (kind >= 2) q = smoothed(q, cfg, weights, kind == 3 ? 0.6f : 0.5f);
        EngineOptions with = int8_options(q, kind == 3 ? MultiplyPath::Invariant : MultiplyPath::Cublas, 16);
        if (kind == 3) with.fp16_accumulate = {false, true, false, false, 64};
        EngineOptions without = with;
        without.prune_last_layer = false;
        CudaEngine pruned(cfg, weights, with);
        CudaEngine full(cfg, weights, without);
        Rng rng(77);
        std::size_t compared = 0, identical = 0;
        for (std::size_t trial = 0; trial < 20; ++trial) {
            const std::vector<EngineRow> request = others(rng, trial % 2 == 0 ? 300 : 40);
            const auto a = run(pruned, request);
            const auto again = run(pruned, request);
            const auto b = run(full, request);
            for (std::size_t r = 0; r < a.size(); ++r) {
                ++compared;
                if (same_bits(a[r].hidden, b[r].hidden) && same_bits(a[r].hidden, again[r].hidden)) ++identical;
            }
        }
        std::cout << "  " << (kind == 1 ? "W8A16" : kind == 2 ? "W8A8 smoothed" : kind == 3 ? "recommended profile" : "W8A8") << " strict: " << identical << " of " << compared << " rows identical between pruned and full last layer and across repeated runs\n";
        CHECK_EQ(identical, compared);
    }
}

SEPAL_TEST(engine_int8, smoothing_refusals_and_the_statistics_it_needs) {
    const ModelConfig cfg = int8_config();
    const Weights<float>& weights = weights_of(cfg, 18);
    const auto stats = collect_stats(cfg, weights);
    const sepal::cuda::Int8Multiply all = int8_groups(true, true, true, true, true);
    const auto make = [&](const sepal::cuda::Int8Multiply& q) { CudaEngine engine(cfg, weights, int8_options(q, MultiplyPath::Cublas, 0)); };
    // The statistics: right shapes, ordered, and the same bits for the same calls.
    CHECK_EQ(stats->layers.size(), static_cast<std::size_t>(cfg.num_layers));
    for (std::size_t l = 0; l < stats->layers.size(); ++l) {
        const std::size_t widths[4] = {cfg.hidden_size, cfg.intermediate_size, cfg.hidden_size,
                                       cfg.is_full_attention(l) ? static_cast<std::size_t>(cfg.attn_width()) : static_cast<std::size_t>(cfg.dn_value_dim())};
        for (std::size_t g = 0; g < 4; ++g) {
            const sepal::cuda::ChannelStats& c = stats->layers[l][g];
            CHECK_EQ(c.amax.size(), widths[g]);
            CHECK_EQ(c.mean_abs.size(), widths[g]);
            CHECK_EQ(c.rms.size(), widths[g]);
            CHECK(stats->tokens[l][g] == (l + 1 == stats->layers.size() && g != 2 ? 6u : 6u * 90u)); // the pruned last layer sees the read-out rows only
            bool ordered = true;
            for (std::size_t k = 0; k < widths[g]; ++k) ordered = ordered && c.mean_abs[k] <= c.rms[k] * 1.0001f && c.rms[k] <= c.amax[k] * 1.0001f && c.amax[k] >= 0.0f && std::isfinite(c.rms[k]);
            CHECK(ordered);
        }
    }
    const auto again = collect_stats(cfg, weights);
    const auto other = collect_stats(cfg, weights, 2);
    bool identical = true, differs = false;
    for (std::size_t l = 0; l < stats->layers.size(); ++l)
        for (std::size_t g = 0; g < 4; ++g) {
            identical = identical && stats->layers[l][g].amax == again->layers[l][g].amax && stats->layers[l][g].rms == again->layers[l][g].rms;
            differs = differs || stats->layers[l][g].rms != other->layers[l][g].rms;
        }
    CHECK(identical);
    CHECK(differs);
    // Smoothing needs int8 activations, a group, alpha in [0, 1], statistics of this geometry.
    sepal::cuda::Int8Multiply q = smoothed(all, cfg, weights);
    make(q);
    q.int8_activations = false;
    CHECK(refused_by_engine([&] { make(q); }));
    q = smoothed(int8_groups(false, false, false, false, true), cfg, weights);
    CHECK(refused_by_engine([&] { make(q); }));
    for (const float alpha : {-0.1f, 1.1f, std::nanf("")}) CHECK(refused_by_engine([&] { make(smoothed(all, cfg, weights, alpha)); }));
    {
        auto fewer = std::make_shared<sepal::cuda::ActivationStats>(*stats);
        fewer->layers.pop_back();
        fewer->tokens.pop_back();
        q = all;
        q.smoothing.stats = fewer;
        CHECK(refused_by_engine([&] { make(q); }));
        auto ragged = std::make_shared<sepal::cuda::ActivationStats>(*stats);
        ragged->layers[1][2].rms.pop_back();
        q.smoothing.stats = ragged;
        CHECK(refused_by_engine([&] { make(q); }));
        auto narrow = std::make_shared<sepal::cuda::ActivationStats>(*stats);
        narrow->layers[0][1].amax.pop_back();
        narrow->layers[0][1].mean_abs.pop_back();
        narrow->layers[0][1].rms.pop_back();
        q.smoothing.stats = narrow;
        CHECK(refused_by_engine([&] { make(q); }));
    }
    // Collecting statistics needs the float16 path, and reading them needs collecting.
    {
        EngineOptions o = options_with(MultiplyPath::Cublas, false, 0);
        o.precision = Precision::F32;
        o.collect_activation_stats = true;
        CHECK(refused_by_engine([&] { CudaEngine engine(cfg, weights, o); }));
        CudaEngine plain(cfg, weights, options_with(MultiplyPath::Cublas, false, 0));
        bool threw = false;
        try {
            (void)plain.activation_stats();
        } catch (const std::logic_error&) {
            threw = true;
        }
        CHECK(threw);
    }
}

SEPAL_TEST(engine_int8, smoothing_changes_the_numbers_and_the_weight_fingerprint) {
    const ModelConfig cfg = int8_config();
    const Weights<float>& weights = weights_of(cfg, 19);
    CudaEngine plain(cfg, weights, int8_options(int8_groups(true, true, true, true, true), MultiplyPath::Cublas, 0));
    CudaEngine half_alpha(cfg, weights, int8_options(smoothed(int8_groups(true, true, true, true, true), cfg, weights, 0.5f), MultiplyPath::Cublas, 0));
    CudaEngine three_quarters(cfg, weights, int8_options(smoothed(int8_groups(true, true, true, true, true), cfg, weights, 0.75f), MultiplyPath::Cublas, 0));
    Rng rng(5);
    const std::vector<EngineRow> request = others(rng, 120);
    const auto a = run(plain, request), b = run(half_alpha, request), c = run(three_quarters, request);
    bool ab = false, bc = false;
    for (std::size_t r = 0; r < a.size(); ++r) {
        ab = ab || !same_bits(a[r].hidden, b[r].hidden);
        bc = bc || !same_bits(b[r].hidden, c[r].hidden);
    }
    CHECK(ab);
    CHECK(bc);
    CHECK(half_alpha.weight_fingerprint() != plain.weight_fingerprint());
    CHECK(half_alpha.weight_fingerprint() != three_quarters.weight_fingerprint());
}

SEPAL_TEST(engine_int8, statistics_files_round_trip_and_refuse_damage) {
    namespace fs = std::filesystem;
    const ModelConfig cfg = int8_config();
    const auto stats = collect_stats(cfg, weights_of(cfg, 20));
    const fs::path dir = fs::temp_directory_path() / "sepal-act-stats-test";
    fs::create_directories(dir);
    const fs::path file = dir / "stats.bin";
    sepal::cuda::save_activation_stats(*stats, file);
    const sepal::cuda::ActivationStats back = sepal::cuda::load_activation_stats(file);
    bool same = back.layers.size() == stats->layers.size() && back.tokens == stats->tokens;
    for (std::size_t l = 0; same && l < back.layers.size(); ++l)
        for (std::size_t g = 0; g < 4; ++g)
            same = same && back.layers[l][g].amax == stats->layers[l][g].amax && back.layers[l][g].mean_abs == stats->layers[l][g].mean_abs && back.layers[l][g].rms == stats->layers[l][g].rms;
    CHECK(same);
    std::ifstream in(file, std::ios::binary);
    std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    in.close();
    const auto refused_bytes = [&](const std::vector<char>& damaged) {
        const fs::path bad = dir / "bad.bin";
        {
            std::ofstream out(bad, std::ios::binary | std::ios::trunc);
            out.write(damaged.data(), static_cast<std::streamsize>(damaged.size()));
        }
        try {
            (void)sepal::cuda::load_activation_stats(bad);
        } catch (const std::invalid_argument&) {
            return true;
        }
        return false;
    };
    CHECK(!refused_bytes(bytes));
    std::size_t refused = 0, tried = 0;
    for (std::size_t len = 0; len < bytes.size(); len += 1 + len / 20) { // truncations
        ++tried;
        refused += refused_bytes(std::vector<char>(bytes.begin(), bytes.begin() + static_cast<std::ptrdiff_t>(len))) ? 1 : 0;
    }
    for (std::size_t at = 0; at < bytes.size(); at += 1 + at / 25) { // flipped bytes
        std::vector<char> bad = bytes;
        bad[at] = static_cast<char>(bad[at] ^ 0x40);
        ++tried;
        refused += refused_bytes(bad) ? 1 : 0;
    }
    std::vector<char> longer = bytes;
    longer.push_back('x');
    ++tried;
    refused += refused_bytes(longer) ? 1 : 0;
    std::cout << "  " << refused << " of " << tried << " damaged statistics files refused\n";
    CHECK_EQ(refused, tried);
    CHECK_THROWS(sepal::cuda::load_activation_stats(dir / "missing.bin"), std::runtime_error);
    fs::remove_all(dir);
}
