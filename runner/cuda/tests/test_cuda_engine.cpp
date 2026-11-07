// The GPU engine on the tiny test geometry with random weights: against the CPU reference executor over many
// random request DAGs (float32), the residual ladder, bit-invariance in strict mode, natural-mode drift, memory
// and lifetime of snapshots, and refused requests. The real checkpoint is in test_cuda_engine_real.cpp.
//
// Environment: SEPAL_FUZZ_CALLS (default 400) sets the number of random calls of the fuzz suite.

#include <chrono>
#include <map>
#include <set>

#include "engine_test_util.hpp"

using namespace sepal;
using namespace engine_test;
using sepal::cuda::BudgetExceeded;
using sepal::cuda::CudaEngine;
using sepal::cuda::DeviceSnapshot;
using sepal::cuda::EngineOptions;
using sepal::cuda::EngineRow;
using sepal::cuda::EngineRowResult;
using sepal::cuda::Precision;

namespace {

constexpr double kHiddenTolerance = 1e-4;
constexpr double kFloat32Floor = 2.5e-4; // see fuzz_against_reference

EngineOptions tiny_options(Precision precision, std::size_t gemm_rows = 0) {
    EngineOptions o;
    o.precision = precision;
    o.gemm_rows = gemm_rows;
    o.memory_budget_bytes = std::size_t{192} << 20;
    o.max_call_tokens = 4096;
    o.max_call_rows = 256;
    o.state_slots = 96;
    return o;
}

const Weights<float>& tiny_weights(std::uint64_t seed) {
    static std::map<std::uint64_t, Weights<float>> cache;
    auto it = cache.find(seed);
    if (it == cache.end()) it = cache.emplace(seed, random_weights<float>(tiny_test_config(), seed)).first;
    return it->second;
}

std::vector<EngineRowResult> run(CudaEngine& engine, const std::vector<EngineRow>& rows) {
    return engine.run(std::span<const EngineRow>(rows));
}

EngineRow row_of(std::vector<std::uint32_t> tokens, std::vector<std::size_t> readouts = {}) {
    EngineRow r;
    r.token_ids = std::move(tokens);
    r.readout_offsets = std::move(readouts);
    return r;
}

std::vector<std::uint32_t> random_tokens(Rng& rng, std::size_t n, std::size_t vocab = 64) {
    std::vector<std::uint32_t> t(n);
    for (auto& v : t) v = static_cast<std::uint32_t>(rng.below(vocab));
    return t;
}

// Compares every readout of a request with the reference. Returns the largest difference.
double compare_request(const Request& request, const std::vector<EngineRowResult>& got,
                       const std::vector<ref::RowResult<float>>& want, std::size_t hidden) {
    double worst = 0.0;
    for (std::size_t r = 0; r < request.engine.size(); ++r) {
        const std::size_t count = request.engine[r].readout_offsets.size();
        CHECK_EQ(got[r].hidden.size(), count * hidden);
        if (got[r].hidden.size() != count * hidden) continue;
        CHECK_EQ(want[r].hidden.size(), count * hidden);
        worst = std::max(worst, max_abs_diff(got[r].hidden.data(), want[r].hidden.data(), count * hidden));
        CHECK(all_finite(got[r].hidden));
    }
    return worst;
}

// On a mismatch: runs the request again with the ladder on and prints the first layer that differs, per row.
void locate_mismatch(CudaEngine& engine, const ModelConfig& cfg, const Weights<float>& weights, const Request& request,
                     const std::vector<ref::RowResult<float>>& want, std::uint64_t seed) {
    const Weights<double> weights64 = double_weights(cfg, weights);
    const auto exact = run_reference64(cfg, weights64, request.reference64);
    std::vector<EngineRow> rows = request.engine;
    for (EngineRow& r : rows) r.capture_layers = true;
    const auto got = run(engine, rows);
    const std::size_t hidden = ref::dim(cfg.hidden_size);
    for (std::size_t r = 0; r < rows.size(); ++r) {
        const auto ladder = reference_ladder<float>(cfg, weights, request.reference, want, r);
        const auto ladder64 = reference_ladder<double>(cfg, weights64, request.reference64, exact, r);
        const std::size_t tokens = rows[r].token_ids.size();
        for (std::size_t l = 0; l < ladder.size(); ++l) {
            const double diff = max_abs_diff(got[r].layers.data() + l * tokens * hidden, ladder[l].data(), tokens * hidden);
            double e_engine = 0, e_ref = 0, peak = 0;
            for (std::size_t i = 0; i < tokens * hidden; ++i) {
                e_engine = std::max(e_engine, std::fabs(got[r].layers[l * tokens * hidden + i] - ladder64[l].data()[i]));
                e_ref = std::max(e_ref, std::fabs(static_cast<double>(ladder[l].data()[i]) - ladder64[l].data()[i]));
                peak = std::max(peak, std::fabs(ladder64[l].data()[i]));
            }
            if (diff > 3e-5) {
                std::cerr << "      layer " << l << ": engine-ref32 " << diff << ", engine-exact " << e_engine << ", ref32-exact " << e_ref
                          << ", largest value " << peak << "\n";
            }
            if (diff > kHiddenTolerance) {
                std::cerr << "    seed " << seed << ": row " << r << " (" << tokens << " tokens) first differs at layer " << l << " ("
                          << (cfg.is_full_attention(l) ? "attention" : "deltanet") << ") by " << diff << "\n";
                break;
            }
        }
    }
}

} // namespace

SEPAL_TEST(engine_f32, fuzz_against_reference) {
    const ModelConfig cfg = tiny_test_config();
    const std::size_t hidden = ref::dim(cfg.hidden_size);
    std::size_t calls = 400;
    if (const auto env = environment("SEPAL_FUZZ_CALLS")) calls = std::stoul(*env);
    const std::size_t weight_sets = 5;

    double worst = 0.0, worst_over_bar = 0.0, engine_vs_exact = 0.0, ref32_vs_exact = 0.0, peak = 0.0;
    std::size_t rows_total = 0, tokens_total = 0, snapshots_made = 0, from_store = 0, over_bar = 0;
    for (std::size_t w = 0; w < weight_sets; ++w) {
        const Weights<float>& weights = tiny_weights(100 + w);
        const Weights<double> weights64 = double_weights(cfg, weights);
        CudaEngine engine(cfg, weights, tiny_options(Precision::F32));
        std::vector<Stored> stored;
        Rng rng(1000 + w);
        for (std::size_t call = 0; call < calls / weight_sets; ++call) {
            const std::uint64_t seed = 1000 * (w + 1) + call;
            FuzzOptions fo;
            const Request request = make_request(rng, stored, fo);
            const auto want = run_reference(cfg, weights, request.reference);
            const auto exact = run_reference64(cfg, weights64, request.reference64);
            const auto got = run(engine, request.engine);
            const double diff = compare_request(request, got, want, hidden);
            worst = std::max(worst, diff);
            double call_engine_exact = 0.0, call_ref32_exact = 0.0;
            for (std::size_t r = 0; r < got.size(); ++r) {
                for (std::size_t i = 0; i < got[r].hidden.size(); ++i) {
                    const double e = exact[r].hidden.data()[i];
                    call_engine_exact = std::max(call_engine_exact, std::fabs(static_cast<double>(got[r].hidden[i]) - e));
                    call_ref32_exact = std::max(call_ref32_exact, std::fabs(static_cast<double>(want[r].hidden.data()[i]) - e));
                    peak = std::max(peak, std::fabs(e));
                }
            }
            engine_vs_exact = std::max(engine_vs_exact, call_engine_exact);
            ref32_vs_exact = std::max(ref32_vs_exact, call_ref32_exact);
            if (diff > kHiddenTolerance) {
                // Two float32 computations cannot agree more closely than their own distance from the true value. The
                // random weights amplify float32 rounding noise through the DeltaNet layers (the ladder shows the error
                // growing in DeltaNet layers only, in the engine and in the reference alike), so in a few calls the
                // float32 reference is itself far from the float64 answer. Such a call passes when the engine is no further
                // from float64 than kFloat32Floor plus 3 times the reference's own distance. The floor is the float32 noise this
                // chain can reach on these random weights: the float32 reference alone is 2.0e-4 from float64 in its worst call
                // (CUDA 12.8 cuBLAS on sm_86 sums in another order than the CPU, so which calls are the bad ones differs from
                // machine to machine), so a bar of 1e-4 was below what two correct float32 implementations can promise. A fault
                // in the engine would be orders of magnitude over it.
                ++over_bar;
                worst_over_bar = std::max(worst_over_bar, diff);
                std::cerr << "  call " << call << " of weight set " << w << " differs from the float32 reference by " << diff
                          << " (engine from float64: " << call_engine_exact << ", float32 reference from float64: " << call_ref32_exact << ")\n";
                CHECK(call_engine_exact <= 3.0 * call_ref32_exact + kFloat32Floor);
                if (call_engine_exact > 3.0 * call_ref32_exact + kFloat32Floor) locate_mismatch(engine, cfg, weights, request, want, seed);
            }
            for (std::size_t r = 0; r < request.engine.size(); ++r) {
                ++rows_total;
                tokens_total += request.engine[r].token_ids.size();
                if (!std::holds_alternative<std::monostate>(request.engine[r].parent) && !std::holds_alternative<cuda::ParentRow>(request.engine[r].parent)) ++from_store;
                if (!request.engine[r].keep_snapshot) {
                    CHECK(!got[r].snapshot.valid());
                    continue;
                }
                CHECK(got[r].snapshot.valid());
                ++snapshots_made;
                CHECK_EQ(got[r].snapshot.tokens(), request.tokens[r]);
                CHECK_EQ(got[r].snapshot.end_position(), request.end_position[r]);
                stored.push_back({got[r].snapshot, want[r].snapshot, exact[r].snapshot, request.end_position[r], request.tokens[r]});
            }
            while (stored.size() > 8) stored.erase(stored.begin() + static_cast<std::ptrdiff_t>(rng.below(stored.size())));
        }
        const cuda::EngineMemory mem = engine.memory();
        CHECK(mem.slots_in_use <= 8);
    }
    std::cout << "  float32 engine against the float32 reference: " << calls << " random calls, " << rows_total << " rows, " << tokens_total
              << " tokens, " << snapshots_made << " snapshots kept, " << from_store << " rows continuing a stored snapshot;\n"
              << "  largest |difference| of any hidden state " << std::scientific << std::setprecision(3) << worst << " (bar " << kHiddenTolerance
              << "); " << over_bar << " calls over the bar, the worst of them " << worst_over_bar << "\n"
              << "  for scale: against the float64 reference the engine is off by at most " << engine_vs_exact
              << " and the float32 reference by " << ref32_vs_exact << " (largest hidden value " << std::fixed << std::setprecision(2)
              << peak << ")\n";
    CHECK(over_bar * 100 <= calls * 2); // at most 2% of the calls may need the float64 yardstick
    CHECK(engine_vs_exact <= 2.0 * ref32_vs_exact + 1e-4); // over the whole run the engine is as close to float64 as the float32 reference
}

SEPAL_TEST(engine_f32, ladder_matches_reference_layer_by_layer) {
    const ModelConfig cfg = tiny_test_config();
    const Weights<float>& weights = tiny_weights(7);
    CudaEngine engine(cfg, weights, tiny_options(Precision::F32));
    Rng rng(5);
    // A state and two branches, so the ladder crosses a snapshot boundary inside the call.
    std::vector<EngineRow> rows(3);
    rows[0] = row_of(random_tokens(rng, 90));
    rows[0].capture_layers = true;
    for (std::size_t b = 1; b < 3; ++b) {
        rows[b] = row_of(random_tokens(rng, 11 + b), {10});
        rows[b].parent = cuda::ParentRow{0};
        rows[b].capture_layers = true;
    }
    std::vector<ref::Row<float>> reference(3);
    for (std::size_t r = 0; r < 3; ++r) {
        reference[r].token_ids = rows[r].token_ids;
        reference[r].readout_offsets = rows[r].readout_offsets;
        if (r > 0) {
            reference[r].parent = ref::ParentRow{0};
            reference[r].start_position = 90;
        }
    }
    const auto want = run_reference(cfg, weights, reference);
    const auto got = run(engine, rows);
    const std::size_t hidden = ref::dim(cfg.hidden_size);
    double worst = 0.0;
    for (std::size_t r = 0; r < 3; ++r) {
        const auto ladder = reference_ladder<float>(cfg, weights, reference, want, r);
        CHECK_EQ(ladder.size(), ref::dim(cfg.num_layers));
        CHECK_EQ(got[r].layers.size(), ladder.size() * rows[r].token_ids.size() * hidden);
        for (std::size_t l = 0; l < ladder.size(); ++l) {
            const double diff = max_abs_diff(got[r].layers.data() + l * rows[r].token_ids.size() * hidden, ladder[l].data(), ladder[l].size());
            worst = std::max(worst, diff);
            // Two float32 computations of the residual stream (values up to about 4) in different summation orders: the bound grows with
            // the layer's largest value (cuBLAS 12 on a GTX 1080 Ti gave 1.29e-4 at a peak of 3.9; the float32 reference itself is
            // 2e-4 from float64 there, see fuzz_against_reference).
            float peak = 0.0f;
            for (std::size_t i = 0; i < ladder[l].size(); ++i) peak = std::max(peak, std::fabs(ladder[l].data()[i]));
            CHECK(diff <= std::max(kHiddenTolerance, 5e-5 * static_cast<double>(peak)));
        }
    }
    std::cout << "  residual ladder (" << cfg.num_layers << " layers x 3 rows): largest difference from the reference " << std::scientific
              << std::setprecision(3) << worst << std::fixed << "\n";
}

SEPAL_TEST(engine_f32, snapshot_chain_across_calls_matches_one_long_row) {
    // A row run in three pieces, each piece continuing the snapshot of the one before (three calls), must give the
    // reference's answer for the whole row run at once.
    const ModelConfig cfg = tiny_test_config();
    const Weights<float>& weights = tiny_weights(8);
    CudaEngine engine(cfg, weights, tiny_options(Precision::F32));
    Rng rng(6);
    const std::vector<std::uint32_t> all = random_tokens(rng, 150);
    ref::Row<float> whole;
    whole.token_ids = all;
    whole.readout_offsets = {149, 100, 99};
    const auto want = run_reference(cfg, weights, {whole});

    DeviceSnapshot snapshot;
    std::size_t at = 0;
    std::vector<EngineRowResult> last;
    for (const std::size_t len : {60u, 40u, 50u}) {
        EngineRow r;
        r.token_ids.assign(all.begin() + static_cast<std::ptrdiff_t>(at), all.begin() + static_cast<std::ptrdiff_t>(at + len));
        if (snapshot.valid()) r.parent = snapshot;
        r.keep_snapshot = true;
        // Readouts at global offsets 149, 100, 99 where they fall in this piece.
        for (const std::size_t g : {149u, 100u, 99u}) {
            if (g >= at && g < at + len) r.readout_offsets.push_back(g - at);
        }
        last = run(engine, {r});
        snapshot = last[0].snapshot;
        at += len;
    }
    CHECK_EQ(snapshot.tokens(), 150u);
    // The last piece (tokens 100..149) holds readouts 149 and 100, in that order; 99 was in the second piece.
    const std::size_t hidden = ref::dim(cfg.hidden_size);
    CHECK_EQ(last[0].hidden.size(), 2 * hidden);
    CHECK(max_abs_diff(last[0].hidden.data(), want[0].hidden.mat().row(0).data(), hidden) <= kHiddenTolerance);
    CHECK(max_abs_diff(last[0].hidden.data() + hidden, want[0].hidden.mat().row(1).data(), hidden) <= kHiddenTolerance);
}

SEPAL_TEST(engine_f16, tiny_against_reference_for_information) {
    const ModelConfig cfg = tiny_test_config();
    const std::size_t hidden = ref::dim(cfg.hidden_size);
    const Weights<float>& weights = tiny_weights(100);
    CudaEngine engine(cfg, weights, tiny_options(Precision::F16));
    std::vector<Stored> stored;
    Rng rng(77);
    double worst = 0.0, sum = 0.0;
    std::size_t count = 0;
    for (std::size_t call = 0; call < 40; ++call) {
        FuzzOptions fo;
        fo.allow_store = false;
        const Request request = make_request(rng, stored, fo);
        const auto want = run_reference(cfg, weights, request.reference);
        const auto got = run(engine, request.engine);
        for (std::size_t r = 0; r < got.size(); ++r) {
            for (std::size_t i = 0; i < got[r].hidden.size(); ++i) {
                const double d = std::fabs(static_cast<double>(got[r].hidden[i]) - static_cast<double>(want[r].hidden.data()[i]));
                worst = std::max(worst, d);
                sum += d;
                ++count;
            }
            CHECK(all_finite(got[r].hidden));
        }
    }
    std::cout << "  float16 engine against the float32 reference on the tiny geometry: largest |difference| " << std::scientific << std::setprecision(3)
              << worst << ", mean " << sum / static_cast<double>(std::max<std::size_t>(count, 1)) << std::fixed
              << " (hidden values are of order 1; this random model amplifies rounding noise through its DeltaNet layers, so the largest value is a\n"
                 "  tail; the real checkpoint, which is well conditioned, is compared in real_f16)\n";
    CHECK(sum / static_cast<double>(std::max<std::size_t>(count, 1)) < 0.03); // a broken path is off by order 1 on average
    (void)hidden;
}
