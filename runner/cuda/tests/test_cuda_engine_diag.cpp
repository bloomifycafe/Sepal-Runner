#include "engine_test_util.hpp"

using namespace sepal;
using namespace engine_test;
using sepal::cuda::CudaEngine;
using sepal::cuda::EngineOptions;
using sepal::cuda::EngineRow;
using sepal::cuda::Precision;

namespace {
EngineOptions diag_options() {
    EngineOptions o;
    o.precision = Precision::F32;
    o.memory_budget_bytes = std::size_t{192} << 20;
    o.max_call_tokens = 4096;
    o.max_call_rows = 256;
    o.state_slots = 16;
    return o;
}

// A float32 multiply that accumulates in float32, like a GPU does (sequential order: only the noise level matters).
class FloatAccumGemm final : public GemmProvider {
public:
    GemmCaps caps() const override { return {}; }
    void gemm_f(MatRef x, MatRef w, MatView y, float) const override {
        const float* xs = static_cast<const float*>(x.data);
        const float* ws = static_cast<const float*>(w.data);
        for (std::size_t t = 0; t < x.rows; ++t) {
            for (std::size_t n = 0; n < w.rows; ++n) {
                float acc = 0.0f;
                for (std::size_t k = 0; k < x.cols; ++k) acc += xs[t * x.cols + k] * ws[n * x.cols + k];
                y(t, n) = acc;
            }
        }
    }
    void gemm_f64(ConstMatViewD x, ConstMatViewD w, MatViewD y, double beta) const override { make_reference_gemm()->gemm_f64(x, w, y, beta); }
};
} // namespace

SEPAL_TEST(diag, noise_floor_of_float32_accumulation) {
    ModelConfig cfg = tiny_test_config();
    double engine_vs_exact = 0, ref32_vs_exact = 0, acc_vs_exact = 0, engine_vs_acc = 0;
    const FloatAccumGemm acc_gemm;
    for (std::uint64_t seed = 0; seed < 60; ++seed) {
        const Weights<float> weights = random_weights<float>(cfg, 700 + seed);
        const Weights<double> weights64 = double_weights(cfg, weights);
        CudaEngine engine(cfg, weights, diag_options());
        Rng rng(seed);
        ref::Row<float> r;
        r.token_ids.resize(129);
        for (auto& t : r.token_ids) t = static_cast<std::uint32_t>(rng.below(64));
        r.readout_offsets = {128, 64, 3};
        ref::Row<double> r64;
        r64.token_ids = r.token_ids;
        r64.readout_offsets = r.readout_offsets;
        cuda::EngineRow e;
        e.token_ids = r.token_ids;
        e.readout_offsets = r.readout_offsets;
        const auto want = run_reference(cfg, weights, {r});
        const std::vector<ref::Row<float>> rows = {r};
        const auto acc = ref::run_rows<float>(acc_gemm, cfg, weights, std::span<const ref::Row<float>>(rows));
        const auto exact = run_reference64(cfg, weights64, {r64});
        const std::vector<cuda::EngineRow> erows = {e};
        const auto got = engine.run(std::span<const cuda::EngineRow>(erows));
        for (std::size_t i = 0; i < got[0].hidden.size(); ++i) {
            const double x = exact[0].hidden.data()[i];
            engine_vs_exact = std::max(engine_vs_exact, std::fabs(got[0].hidden[i] - x));
            ref32_vs_exact = std::max(ref32_vs_exact, std::fabs(want[0].hidden.data()[i] - x));
            acc_vs_exact = std::max(acc_vs_exact, std::fabs(acc[0].hidden.data()[i] - x));
            engine_vs_acc = std::max(engine_vs_acc, static_cast<double>(std::fabs(got[0].hidden[i] - acc[0].hidden.data()[i])));
        }
    }
    std::cout << "  max over 60 models: engine-exact " << engine_vs_exact << ", ref32 (double-accum gemm)-exact " << ref32_vs_exact
              << ", ref32 (float-accum gemm)-exact " << acc_vs_exact << ", engine-ref32(float-accum) " << engine_vs_acc << "\n";
}

SEPAL_TEST(diag, no_stored_parents_statistics) {
    ModelConfig cfg = tiny_test_config();
    std::size_t calls = 0, over = 0;
    double worst = 0, eng_exact = 0, ref_exact = 0;
    for (std::uint64_t w = 0; w < 5; ++w) {
        const Weights<float> weights = random_weights<float>(cfg, 100 + w);
        const Weights<double> weights64 = double_weights(cfg, weights);
        CudaEngine engine(cfg, weights, diag_options());
        Rng rng(5000 + w);
        for (std::size_t c = 0; c < 200; ++c) {
            FuzzOptions fo;
            fo.allow_store = false;
            const Request request = make_request(rng, {}, fo);
            const auto want = run_reference(cfg, weights, request.reference);
            const auto exact = run_reference64(cfg, weights64, request.reference64);
            const auto got = engine.run(std::span<const cuda::EngineRow>(request.engine));
            double d = 0, ee = 0, re = 0;
            for (std::size_t r = 0; r < got.size(); ++r) {
                for (std::size_t i = 0; i < got[r].hidden.size(); ++i) {
                    const double x = exact[r].hidden.data()[i];
                    d = std::max(d, std::fabs(got[r].hidden[i] - static_cast<double>(want[r].hidden.data()[i])));
                    ee = std::max(ee, std::fabs(got[r].hidden[i] - x));
                    re = std::max(re, std::fabs(want[r].hidden.data()[i] - x));
                }
            }
            ++calls;
            if (d > 1e-4) ++over;
            worst = std::max(worst, d);
            eng_exact = std::max(eng_exact, ee);
            ref_exact = std::max(ref_exact, re);
        }
    }
    std::cout << "  " << calls << " calls without stored parents: " << over << " over 1e-4, worst engine-ref32 " << worst << ", engine-exact "
              << eng_exact << ", ref32-exact " << ref_exact << "\n";
}
