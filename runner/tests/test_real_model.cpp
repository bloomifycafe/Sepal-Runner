// The CPU reference path on the real jeff-base checkpoint against transformers.
//
// Skipped (and reported as skipped) unless two environment variables are set:
//     SEPAL_REAL_MODEL    the checkpoint directory (config.json, model.safetensors, readout.safetensors, ...)
//     SEPAL_REAL_GOLDEN   the directory tools/export_real_golden.py wrote (case_i.safetensors ... case_iv.safetensors)
//
// The reference path runs a few tokens per second on one thread, so the token counts are those of the cases:
// 16 + 64 + 300 + (300 + 3 x about 60) tokens in float32, and the 16-token case again in float64.
// Every comparison prints the largest difference it saw, and the run time of each case.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "sepal/loader.hpp"
#include "sepal/ref/executor.hpp"
#include "test.hpp"

using namespace sepal;

namespace {

using Clock = std::chrono::steady_clock;

// The bar for float32 hidden states and for probabilities (see the task: about 1e-4 absolute).
constexpr double hidden_tolerance = 1e-4;
constexpr double probability_tolerance = 1e-4;
// Float64 against transformers' float64 run: both are exact to rounding.
constexpr double hidden_tolerance_f64 = 1e-9;

std::optional<std::string> environment(const char* name) {
#if defined(_MSC_VER)
    char* buffer = nullptr;
    std::size_t length = 0;
    if (_dupenv_s(&buffer, &length, name) != 0 || buffer == nullptr) return std::nullopt;
    const std::string value(buffer);
    std::free(buffer);
#else
    const char* raw = std::getenv(name);
    const std::string value = raw == nullptr ? std::string() : std::string(raw);
#endif
    if (value.empty()) return std::nullopt;
    return value;
}

// True (after printing why) when the real-model tests have nothing to run on.
bool skipped() {
    const auto model = environment("SEPAL_REAL_MODEL");
    const auto golden = environment("SEPAL_REAL_GOLDEN");
    if (model && golden) return false;
    std::cout << "SKIPPED: real-model test; set SEPAL_REAL_MODEL (checkpoint directory) and SEPAL_REAL_GOLDEN (tools/export_real_golden.py "
                 "output directory) to run it\n";
    return true;
}

double seconds_since(Clock::time_point start) { return std::chrono::duration<double>(Clock::now() - start).count(); }

struct Real {
    LoadedModel model;
    Weights<float> weights32;
    std::unique_ptr<GemmProvider> gemm = make_reference_gemm();
};

const Real& real() {
    static const Real r = [] {
        const auto start = Clock::now();
        Real built{LoadedModel::open(*environment("SEPAL_REAL_MODEL")), {}, make_reference_gemm()};
        built.weights32 = built.model.weights<float>();
        std::cout << "  loaded " << built.model.raw().size() << " tensors, converted to float32 in " << std::fixed << std::setprecision(1)
                  << seconds_since(start) << " s\n";
        return built;
    }();
    return r;
}

SafetensorsFile golden(const char* name) { return SafetensorsFile::open(*environment("SEPAL_REAL_GOLDEN") + "/" + name + ".safetensors"); }

std::vector<std::uint32_t> tokens_of(const TensorView& view) {
    std::vector<std::uint32_t> out;
    for (const std::int32_t t : view.values<std::int32_t>()) out.push_back(static_cast<std::uint32_t>(t));
    return out;
}

template <class T>
double max_abs_diff(std::span<const T> got, std::span<const float> want) {
    double worst = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) worst = std::max(worst, std::fabs(static_cast<double>(got[i]) - static_cast<double>(want[i])));
    return worst;
}

double max_abs_diff(std::span<const double> got, std::span<const double> want) {
    double worst = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) worst = std::max(worst, std::fabs(got[i] - want[i]));
    return worst;
}

template <class T>
double max_abs(std::span<const T> values) {
    double peak = 0.0;
    for (const T v : values) peak = std::max(peak, std::fabs(static_cast<double>(v)));
    return peak;
}

// Runs `tokens` through the model as one row and returns every layer's output [T, hidden] and the final hidden state.
template <class T>
struct Pass {
    std::vector<BasicTensor<T>> layers;
    std::vector<T> final_hidden;
    double seconds = 0.0;
};

template <class T>
Pass<T> run_all_layers(const ModelConfig& config, const Weights<T>& weights, const GemmProvider& gemm, const std::vector<std::uint32_t>& tokens) {
    Pass<T> pass;
    const auto start = Clock::now();
    const ref::SnapshotPtr<T> nothing = ref::empty_snapshot<T>(config);
    const ref::Continuation<T> run =
        ref::continue_snapshot<T>(gemm, config, weights, *nothing, std::span<const std::uint32_t>(tokens), 0, &pass.layers);
    pass.seconds = seconds_since(start);
    const std::size_t hidden = ref::dim(config.hidden_size);
    BasicTensor<T> picked({1, hidden});
    std::copy(run.last_layer.data() + (tokens.size() - 1) * hidden, run.last_layer.data() + tokens.size() * hidden, picked.data());
    const BasicTensor<T> normed = ref::rms_norm<T>(picked.mat(), weights[final_norm_name].span(), config.rms_norm_eps);
    pass.final_hidden.assign(normed.data(), normed.data() + hidden);
    return pass;
}

// Compares every layer's output with a reference [L, T, hidden] and prints the table. Returns the largest difference.
template <class T, class R>
double compare_layers(const char* label, const Pass<T>& pass, std::span<const R> reference, std::size_t layer_count, std::size_t tokens,
                      std::size_t hidden, double tolerance) {
    std::cout << "  " << label << ": layer, largest |difference|, largest |value| (" << tokens << " tokens)\n";
    double worst = 0.0;
    std::size_t first_bad = layer_count;
    for (std::size_t l = 0; l < layer_count; ++l) {
        const std::span<const R> want = reference.subspan(l * tokens * hidden, tokens * hidden);
        double diff = 0.0;
        for (std::size_t i = 0; i < want.size(); ++i) diff = std::max(diff, std::fabs(static_cast<double>(pass.layers[l][i]) - static_cast<double>(want[i])));
        const double peak = max_abs<R>(want);
        std::cout << "    " << std::setw(2) << l << "  " << std::scientific << std::setprecision(3) << diff << "  " << std::fixed
                  << std::setprecision(2) << peak << "\n";
        worst = std::max(worst, diff);
        if (diff > tolerance && first_bad == layer_count) first_bad = l;
        CHECK(diff <= tolerance);
    }
    std::cout << "  " << label << ": largest difference over all layers " << std::scientific << std::setprecision(3) << worst << " (tolerance "
              << tolerance << (first_bad == layer_count ? ")" : ", first layer over: " + std::to_string(first_bad) + ")") << std::fixed << "\n";
    return worst;
}

void plain_case(const char* name, const char* with_float64) {
    const Real& r = real();
    const SafetensorsFile file = golden(name);
    const std::vector<std::uint32_t> tokens = tokens_of(file.at("token_ids"));
    const std::size_t hidden = ref::dim(r.model.config().hidden_size);
    const std::size_t layers = ref::dim(r.model.config().num_layers);
    CHECK_EQ(file.at("layers").elements(), layers * tokens.size() * hidden);

    const Pass<float> pass = run_all_layers<float>(r.model.config(), r.weights32, *r.gemm, tokens);
    std::cout << "  " << name << ": " << tokens.size() << " tokens in " << std::fixed << std::setprecision(1) << pass.seconds << " s\n";
    compare_layers<float, float>("float32 against transformers float32", pass, file.at("layers").values<float>(), layers, tokens.size(), hidden,
                                 hidden_tolerance);
    const double final_diff = max_abs_diff<float>(pass.final_hidden, file.at("final_hidden").values<float>());
    std::cout << "  " << name << ": final hidden state largest difference " << std::scientific << std::setprecision(3) << final_diff << std::fixed << "\n";
    CHECK(final_diff <= hidden_tolerance);

    const DecisionReadout readout = DecisionReadout::open_directory(*environment("SEPAL_REAL_MODEL"));
    const std::vector<double> p = readout.probabilities(std::span<const float>(pass.final_hidden), 4);
    const double p_diff = max_abs_diff(p, file.at("probs").values<double>());
    std::cout << "  " << name << ": probabilities largest difference " << std::scientific << std::setprecision(3) << p_diff << std::fixed << "\n";
    CHECK(p_diff <= probability_tolerance);

    if (with_float64 != nullptr) {
        const auto start = Clock::now();
        const Weights<double> weights64 = r.model.weights<double>();
        std::cout << "  weights converted to float64 in " << std::setprecision(1) << seconds_since(start) << " s\n";
        const Pass<double> pass64 = run_all_layers<double>(r.model.config(), weights64, *r.gemm, tokens);
        std::cout << "  " << name << " in float64: " << std::setprecision(1) << pass64.seconds << " s\n";
        compare_layers<double, double>("float64 against transformers float64", pass64, file.at("layers_f64").values<double>(), layers, tokens.size(),
                                       hidden, hidden_tolerance_f64);
        // And the float64 engine against transformers' float32 run: the engine is the more exact of the two.
        compare_layers<double, float>("float64 against transformers float32", pass64, file.at("layers").values<float>(), layers, tokens.size(),
                                      hidden, hidden_tolerance);
    }
}

} // namespace

SEPAL_TEST(real, case_i_sixteen_tokens_every_layer) {
    if (skipped()) return;
    plain_case("case_i", "with float64");
}

SEPAL_TEST(real, case_ii_sixty_four_tokens_every_layer) {
    if (skipped()) return;
    plain_case("case_ii", nullptr);
}

SEPAL_TEST(real, case_iii_decision_prompt_final_hidden_and_probabilities) {
    if (skipped()) return;
    const Real& r = real();
    const SafetensorsFile file = golden("case_iii");
    const std::vector<std::uint32_t> tokens = tokens_of(file.at("token_ids"));

    const auto start = Clock::now();
    ref::Row<float> row;
    row.token_ids = tokens;
    row.readout_offsets = {tokens.size() - 1};
    const std::vector<ref::Row<float>> rows = {row};
    const auto result = ref::run_rows<float>(*r.gemm, r.model.config(), r.weights32, rows);
    std::cout << "  case_iii: " << tokens.size() << " tokens in " << std::fixed << std::setprecision(1) << seconds_since(start) << " s\n";
    const std::span<const float> hidden = result[0].hidden.mat().row(0);

    const double h_diff = max_abs_diff<float>(hidden, file.at("final_hidden").values<float>());
    const DecisionReadout readout = DecisionReadout::open_directory(*environment("SEPAL_REAL_MODEL"));
    const std::vector<double> p = readout.probabilities(hidden, 4);
    const double p_diff = max_abs_diff(p, file.at("probs").values<double>());
    std::cout << "  case_iii: final hidden largest difference " << std::scientific << std::setprecision(3) << h_diff << " (largest value "
              << std::fixed << std::setprecision(2) << max_abs<float>(hidden) << "), probabilities " << std::scientific << std::setprecision(3)
              << p_diff << std::fixed << "\n";
    CHECK(h_diff <= hidden_tolerance);
    CHECK(p_diff <= probability_tolerance);

    // For information: how far transformers' half-precision GPU runs are from the float32 reference.
    for (const char* label : {"gpu_fp16", "gpu_bf16"}) {
        if (file.find(std::string(label) + ".final_hidden") == nullptr) continue;
        std::cout << "  case_iii: " << label << " against this engine's float32: hidden " << std::scientific << std::setprecision(3)
                  << max_abs_diff<float>(hidden, file.at(std::string(label) + ".final_hidden").values<float>()) << ", probabilities "
                  << max_abs_diff(p, file.at(std::string(label) + ".probs").values<double>()) << std::fixed << "\n";
    }
}

SEPAL_TEST(real, case_iv_one_state_continued_by_three_branches) {
    if (skipped()) return;
    const Real& r = real();
    const SafetensorsFile file = golden("case_iv");
    const std::size_t split = std::stoul(file.metadata().at("split"));
    const std::vector<std::uint32_t> state = tokens_of(file.at("state_ids"));
    CHECK_EQ(state.size(), split);
    const DecisionReadout readout = DecisionReadout::open_directory(*environment("SEPAL_REAL_MODEL"));
    const std::size_t hidden_size = ref::dim(r.model.config().hidden_size);

    // One request: row 0 is the state, rows 1 to 3 continue it from where it ended (ParentRow).
    std::vector<ref::Row<float>> rows(4);
    rows[0].token_ids = state;
    std::vector<std::vector<std::uint32_t>> full(3);
    std::vector<std::size_t> options = {2, 4, 3};
    std::size_t branch_tokens = 0;
    for (std::size_t k = 0; k < 3; ++k) {
        full[k] = tokens_of(file.at("q" + std::to_string(k) + ".token_ids"));
        CHECK(full[k].size() > split);
        CHECK(std::equal(state.begin(), state.end(), full[k].begin())); // the state is an exact prefix
        rows[k + 1].token_ids.assign(full[k].begin() + static_cast<std::ptrdiff_t>(split), full[k].end());
        rows[k + 1].start_position = split;
        rows[k + 1].parent = ref::ParentRow{0};
        rows[k + 1].readout_offsets = {rows[k + 1].token_ids.size() - 1};
        branch_tokens += rows[k + 1].token_ids.size();
    }
    const auto start = Clock::now();
    const auto results = ref::run_rows<float>(*r.gemm, r.model.config(), r.weights32, rows);
    std::cout << "  case_iv: state of " << split << " tokens and three branches of " << branch_tokens << " tokens in total, " << std::fixed
              << std::setprecision(1) << seconds_since(start) << " s\n";

    // The same branches again, continuing the stored snapshot of the state in a second request: identical bits.
    std::vector<ref::Row<float>> from_store(3);
    for (std::size_t k = 0; k < 3; ++k) {
        from_store[k] = rows[k + 1];
        from_store[k].parent = results[0].snapshot;
    }
    const auto again = ref::run_rows<float>(*r.gemm, r.model.config(), r.weights32, from_store);

    for (std::size_t k = 0; k < 3; ++k) {
        const std::string q = "q" + std::to_string(k);
        const std::span<const float> hidden = results[k + 1].hidden.mat().row(0);
        CHECK_EQ(hidden.size(), hidden_size);
        const double h_diff = max_abs_diff<float>(hidden, file.at(q + ".final_hidden").values<float>());
        const std::vector<double> p = readout.probabilities(hidden, options[k]);
        const double p_diff = max_abs_diff(p, file.at(q + ".probs").values<double>());
        bool identical = true;
        for (std::size_t i = 0; i < hidden_size; ++i) identical = identical && hidden[i] == again[k].hidden.mat().row(0)[i];
        std::cout << "  case_iv " << q << " (" << full[k].size() << " tokens in transformers): final hidden largest difference " << std::scientific
                  << std::setprecision(3) << h_diff << ", probabilities " << p_diff << std::fixed << ", stored-snapshot run identical: "
                  << (identical ? "yes" : "NO") << "\n";
        CHECK(h_diff <= hidden_tolerance);
        CHECK(p_diff <= probability_tolerance);
        CHECK(identical);
        for (const char* label : {"gpu_fp16", "gpu_bf16"}) {
            const std::string name = q + "." + label;
            if (file.find(name + ".final_hidden") == nullptr) continue;
            std::cout << "    " << label << " against this engine's float32: hidden " << std::scientific << std::setprecision(3)
                      << max_abs_diff<float>(hidden, file.at(name + ".final_hidden").values<float>()) << ", probabilities "
                      << max_abs_diff(p, file.at(name + ".probs").values<double>()) << std::fixed << "\n";
        }
    }
}
