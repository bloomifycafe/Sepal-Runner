// The GPU engine on the real jeff-base checkpoint against the transformers goldens (tools/export_real_golden.py).
//
// Skipped (and reported as skipped) unless two environment variables are set:
//     SEPAL_REAL_MODEL    the checkpoint directory (config.json, model.safetensors, readout.safetensors, ...)
//     SEPAL_REAL_GOLDEN   the directory tools/export_real_golden.py wrote (case_i.safetensors ... case_iv.safetensors)
//
// Suites: real_f32 (float32 engine against the float32 goldens, layer by layer where exported), real_f16 (float16
// engine against the float32 golden and against transformers' own float16 run), real_strict (bit-invariance on the
// real model), real_natural (drift of the natural mode with the composition of the call).

#include <chrono>

#include "engine_test_util.hpp"
#include "sepal/loader.hpp"
#include "sepal/safetensors.hpp"

using namespace sepal;
using namespace engine_test;
using sepal::cuda::CudaEngine;
using sepal::cuda::DeviceSnapshot;
using sepal::cuda::EngineOptions;
using sepal::cuda::EngineRow;
using sepal::cuda::EngineRowResult;
using sepal::cuda::ParentRow;
using sepal::cuda::Precision;

namespace {

using Clock = std::chrono::steady_clock;

constexpr double kHiddenTolerance = 1e-4;
constexpr double kProbabilityTolerance = 1e-4;
constexpr double kF16ProbabilityGate = 0.03;

bool skipped() {
    if (environment("SEPAL_REAL_MODEL") && environment("SEPAL_REAL_GOLDEN")) return false;
    std::cout << "SKIPPED: real-model test; set SEPAL_REAL_MODEL (checkpoint directory) and SEPAL_REAL_GOLDEN (tools/export_real_golden.py "
                 "output directory) to run it\n";
    return true;
}

double seconds_since(Clock::time_point start) { return std::chrono::duration<double>(Clock::now() - start).count(); }

struct Real {
    LoadedModel model;
    DecisionReadout readout;
};

Real& real() {
    static Real r{LoadedModel::open(*environment("SEPAL_REAL_MODEL")), DecisionReadout::open_directory(*environment("SEPAL_REAL_MODEL"))};
    return r;
}

SafetensorsFile golden(const char* name) { return SafetensorsFile::open(*environment("SEPAL_REAL_GOLDEN") + "/" + name + ".safetensors"); }

std::vector<std::uint32_t> tokens_of(const TensorView& view) {
    std::vector<std::uint32_t> out;
    for (const std::int32_t t : view.values<std::int32_t>()) out.push_back(static_cast<std::uint32_t>(t));
    return out;
}

EngineOptions real_options(Precision precision, std::size_t gemm_rows, std::size_t budget_mib) {
    EngineOptions o;
    o.precision = precision;
    o.gemm_rows = gemm_rows;
    o.memory_budget_bytes = budget_mib << 20;
    o.max_call_tokens = 4096;
    o.max_call_rows = 128;
    o.state_slots = 24;
    return o;
}

std::vector<EngineRowResult> run(CudaEngine& engine, const std::vector<EngineRow>& rows) {
    return engine.run(std::span<const EngineRow>(rows));
}

EngineRow row_of(const std::vector<std::uint32_t>& t, std::vector<std::size_t> readouts, bool keep = false) {
    EngineRow r;
    r.token_ids = t;
    r.readout_offsets = std::move(readouts);
    r.keep_snapshot = keep;
    return r;
}

std::vector<double> probabilities(const std::vector<float>& hidden, std::size_t options) {
    return real().readout.probabilities(std::span<const float>(hidden), options);
}

double max_diff(const std::vector<double>& a, std::span<const double> b) {
    double worst = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) worst = std::max(worst, std::fabs(a[i] - b[i]));
    return worst;
}

std::size_t argmax(std::span<const double> p) { return static_cast<std::size_t>(std::max_element(p.begin(), p.end()) - p.begin()); }

// The prompts of case iv as the state and the three suffixes.
struct CaseIV {
    std::vector<std::uint32_t> state;
    std::vector<std::vector<std::uint32_t>> suffix, full;
    std::vector<std::size_t> options = {2, 4, 3};
    SafetensorsFile file;
};

CaseIV load_case_iv() {
    CaseIV c{{}, {}, {}, {2, 4, 3}, golden("case_iv")};
    const std::size_t split = std::stoul(c.file.metadata().at("split"));
    c.state = tokens_of(c.file.at("state_ids"));
    CHECK_EQ(c.state.size(), split);
    for (std::size_t k = 0; k < 3; ++k) {
        c.full.push_back(tokens_of(c.file.at("q" + std::to_string(k) + ".token_ids")));
        CHECK(std::equal(c.state.begin(), c.state.end(), c.full[k].begin()));
        c.suffix.emplace_back(c.full[k].begin() + static_cast<std::ptrdiff_t>(split), c.full[k].end());
    }
    return c;
}

// State row + three branch rows in one call; branch k is row k + 1.
std::vector<EngineRow> case_iv_call(const CaseIV& c) {
    std::vector<EngineRow> rows = {row_of(c.state, {})};
    for (std::size_t k = 0; k < 3; ++k) {
        EngineRow b = row_of(c.suffix[k], {c.suffix[k].size() - 1});
        b.parent = ParentRow{0};
        b.start_position = c.state.size();
        rows.push_back(b);
    }
    return rows;
}

// The same branches continuing a stored snapshot of the state (a later call).
std::vector<EngineRow> case_iv_from_store(const CaseIV& c, const DeviceSnapshot& state) {
    std::vector<EngineRow> rows;
    for (std::size_t k = 0; k < 3; ++k) {
        EngineRow b = row_of(c.suffix[k], {c.suffix[k].size() - 1});
        b.parent = state;
        b.start_position = c.state.size();
        rows.push_back(b);
    }
    return rows;
}

} // namespace

// ---- float32 engine against the float32 goldens ----

SEPAL_TEST(real_f32, cases_i_and_ii_every_layer_final_hidden_and_probabilities) {
    if (skipped()) return;
    Real& r = real();
    const auto start = Clock::now();
    CudaEngine engine(r.model, real_options(Precision::F32, 0, 6000));
    std::cout << "  float32 engine created (weights uploaded) in " << std::fixed << std::setprecision(1) << seconds_since(start) << " s; reserved "
              << engine.memory().reserved_bytes() / (1 << 20) << " MiB\n";
    const std::size_t hidden = ref::dim(r.model.config().hidden_size);
    const std::size_t layers = ref::dim(r.model.config().num_layers);
    for (const char* name : {"case_i", "case_ii"}) {
        const SafetensorsFile file = golden(name);
        const std::vector<std::uint32_t> tokens = tokens_of(file.at("token_ids"));
        EngineRow row = row_of(tokens, {tokens.size() - 1});
        row.capture_layers = true;
        const auto t0 = Clock::now();
        const auto got = run(engine, {row});
        const double seconds = seconds_since(t0);
        const std::span<const float> want = file.at("layers").values<float>();
        double worst = 0.0;
        std::size_t first_bad = layers;
        std::cout << "  " << name << " (" << tokens.size() << " tokens, " << std::setprecision(3) << seconds << " s): layer, largest |difference|\n";
        for (std::size_t l = 0; l < layers; ++l) {
            const double diff = max_abs_diff(got[0].layers.data() + l * tokens.size() * hidden, want.data() + l * tokens.size() * hidden, tokens.size() * hidden);
            std::cout << "    " << std::setw(2) << l << "  " << std::scientific << std::setprecision(3) << diff << std::fixed << "\n";
            worst = std::max(worst, diff);
            if (diff > kHiddenTolerance && first_bad == layers) first_bad = l;
            CHECK(diff <= kHiddenTolerance);
        }
        const double final_diff = max_abs_diff(got[0].hidden.data(), file.at("final_hidden").values<float>().data(), hidden);
        const std::vector<double> p = probabilities(got[0].hidden, 4);
        const double p_diff = max_diff(p, file.at("probs").values<double>());
        std::cout << "  " << name << ": largest layer difference " << std::scientific << std::setprecision(3) << worst << ", final hidden " << final_diff
                  << ", probabilities " << p_diff << std::fixed << (first_bad == layers ? "" : "  FIRST LAYER OVER THE BAR: " + std::to_string(first_bad)) << "\n";
        CHECK(final_diff <= kHiddenTolerance);
        CHECK(p_diff <= kProbabilityTolerance);
    }
}

SEPAL_TEST(real_f32, case_iii_and_iv_final_hidden_and_probabilities) {
    if (skipped()) return;
    Real& r = real();
    CudaEngine engine(r.model, real_options(Precision::F32, 0, 6000));
    const std::size_t hidden = ref::dim(r.model.config().hidden_size);

    {
        const SafetensorsFile file = golden("case_iii");
        const std::vector<std::uint32_t> tokens = tokens_of(file.at("token_ids"));
        const auto got = run(engine, {row_of(tokens, {tokens.size() - 1})});
        const double h_diff = max_abs_diff(got[0].hidden.data(), file.at("final_hidden").values<float>().data(), hidden);
        const double p_diff = max_diff(probabilities(got[0].hidden, 4), file.at("probs").values<double>());
        std::cout << "  case_iii (" << tokens.size() << " tokens): final hidden " << std::scientific << std::setprecision(3) << h_diff << ", probabilities " << p_diff << std::fixed << "\n";
        CHECK(h_diff <= kHiddenTolerance);
        CHECK(p_diff <= kProbabilityTolerance);
    }

    const CaseIV c = load_case_iv();
    // One call: the state row and three branch rows.
    const auto together = run(engine, case_iv_call(c));
    // A stored snapshot of the state, continued by a later call.
    EngineRow state_row = row_of(c.state, {}, true);
    const auto stored = run(engine, {state_row});
    CHECK_EQ(stored[0].snapshot.tokens(), c.state.size());
    const auto later = run(engine, case_iv_from_store(c, stored[0].snapshot));
    for (std::size_t k = 0; k < 3; ++k) {
        const std::string q = "q" + std::to_string(k);
        const std::vector<float>& h = together[k + 1].hidden;
        const double h_diff = max_abs_diff(h.data(), c.file.at(q + ".final_hidden").values<float>().data(), hidden);
        const double p_diff = max_diff(probabilities(h, c.options[k]), c.file.at(q + ".probs").values<double>());
        const double hit_diff = max_abs_diff(later[k].hidden.data(), h.data(), hidden);
        const double hit_p = max_diff(probabilities(later[k].hidden, c.options[k]), c.file.at(q + ".probs").values<double>());
        std::cout << "  case_iv " << q << " (" << c.full[k].size() << " tokens): one call final hidden " << std::scientific << std::setprecision(3) << h_diff
                  << ", probabilities " << p_diff << "; stored snapshot continued by a later call: hidden " << max_abs_diff(later[k].hidden.data(), c.file.at(q + ".final_hidden").values<float>().data(), hidden)
                  << ", probabilities " << hit_p << ", difference between the two ways " << hit_diff << std::fixed << "\n";
        CHECK(h_diff <= kHiddenTolerance);
        CHECK(p_diff <= kProbabilityTolerance);
        CHECK(hit_p <= kProbabilityTolerance);
        CHECK(max_abs_diff(later[k].hidden.data(), c.file.at(q + ".final_hidden").values<float>().data(), hidden) <= kHiddenTolerance);
    }
}

// ---- float16 engine against the float32 golden, and against transformers' own float16 ----

namespace {

struct Distances {
    double max = 0.0, sum = 0.0;
    std::size_t count = 0;
    void add(double d) {
        max = std::max(max, d);
        sum += d;
        ++count;
    }
    double mean() const { return count == 0 ? 0.0 : sum / static_cast<double>(count); }
};

struct F16Report {
    Distances hidden, probability;          // this engine against the float32 golden
    Distances torch_hidden, torch_probability; // transformers' float16 run against the float32 golden
    std::size_t prompts = 0, flips = 0;
};

void compare_prompt(F16Report& report, const std::vector<float>& hidden, std::size_t options, const TensorView& golden_hidden,
                    const TensorView& golden_probs, const SafetensorsFile& file, const std::string& prefix) {
    const std::span<const float> want = golden_hidden.values<float>();
    for (std::size_t i = 0; i < hidden.size(); ++i) report.hidden.add(std::fabs(static_cast<double>(hidden[i]) - want[i]));
    const std::vector<double> p = probabilities(hidden, options);
    const std::span<const double> want_p = golden_probs.values<double>();
    for (std::size_t i = 0; i < p.size(); ++i) report.probability.add(std::fabs(p[i] - want_p[i]));
    ++report.prompts;
    if (argmax(p) != argmax(want_p)) ++report.flips;
    const std::string torch = prefix + "gpu_fp16.";
    if (file.find(torch + "final_hidden") != nullptr) {
        const std::span<const float> th = file.at(torch + "final_hidden").values<float>();
        for (std::size_t i = 0; i < th.size(); ++i) report.torch_hidden.add(std::fabs(static_cast<double>(th[i]) - want[i]));
        const std::span<const double> tp = file.at(torch + "probs").values<double>();
        for (std::size_t i = 0; i < tp.size(); ++i) report.torch_probability.add(std::fabs(tp[i] - want_p[i]));
    }
}

void print_report(const char* label, const F16Report& r) {
    std::cout << std::scientific << std::setprecision(2) << "  " << label << ": " << r.prompts << " prompts, answer flips " << r.flips << "\n"
              << "    this engine, float16 vs float32 golden:    |dh| max " << r.hidden.max << " mean " << r.hidden.mean() << "   |dp| max " << r.probability.max << " mean "
              << r.probability.mean() << "\n"
              << "    transformers float16 vs float32 golden:    |dh| max " << r.torch_hidden.max << " mean " << r.torch_hidden.mean() << "   |dp| max "
              << r.torch_probability.max << " mean " << r.torch_probability.mean() << std::fixed << "\n";
}

} // namespace

SEPAL_TEST(real_f16, float16_engine_against_the_float32_golden) {
    if (skipped()) return;
    Real& r = real();
    const auto start = Clock::now();
    CudaEngine engine(r.model, real_options(Precision::F16, 0, 4000));
    std::cout << "  float16 engine created in " << std::fixed << std::setprecision(1) << seconds_since(start) << " s; reserved " << engine.memory().reserved_bytes() / (1 << 20) << " MiB\n";
    F16Report plain, iv, all;
    for (const char* name : {"case_i", "case_ii", "case_iii"}) {
        const SafetensorsFile file = golden(name);
        const std::vector<std::uint32_t> tokens = tokens_of(file.at("token_ids"));
        const auto got = run(engine, {row_of(tokens, {tokens.size() - 1})});
        CHECK(all_finite(got[0].hidden));
        compare_prompt(plain, got[0].hidden, 4, file.at("final_hidden"), file.at("probs"), file, "");
        compare_prompt(all, got[0].hidden, 4, file.at("final_hidden"), file.at("probs"), file, "");
    }
    {
        const CaseIV c = load_case_iv();
        const auto together = run(engine, case_iv_call(c));
        for (std::size_t k = 0; k < 3; ++k) {
            const std::string q = "q" + std::to_string(k);
            CHECK(all_finite(together[k + 1].hidden));
            compare_prompt(iv, together[k + 1].hidden, c.options[k], c.file.at(q + ".final_hidden"), c.file.at(q + ".probs"), c.file, q + ".");
            compare_prompt(all, together[k + 1].hidden, c.options[k], c.file.at(q + ".final_hidden"), c.file.at(q + ".probs"), c.file, q + ".");
        }
    }
    print_report("cases i, ii, iii (plain prompts; transformers fp16 reference only for iii)", plain);
    print_report("case iv (state + 3 branches in one call)", iv);
    print_report("all prompts", all);
    CHECK(all.probability.max <= kF16ProbabilityGate);
    CHECK_EQ(all.flips, 0u);
}

// ---- invariance on the real model ----

namespace {

// Other rows to share a call with: token runs of the goldens (real text), cut at various lengths.
std::vector<EngineRow> real_others(Rng& rng, const std::vector<std::vector<std::uint32_t>>& corpus, std::size_t count) {
    std::vector<EngineRow> rows;
    for (std::size_t i = 0; i < count; ++i) {
        const auto& source = corpus[rng.below(corpus.size())];
        const std::size_t len = 1 + rng.below(std::min<std::size_t>(source.size(), 200));
        const std::size_t begin = rng.below(source.size() - len + 1);
        rows.push_back(row_of(std::vector<std::uint32_t>(source.begin() + static_cast<std::ptrdiff_t>(begin), source.begin() + static_cast<std::ptrdiff_t>(begin + len)),
                              {len - 1}));
    }
    return rows;
}

std::vector<std::vector<std::uint32_t>> corpus_of_goldens() {
    std::vector<std::vector<std::uint32_t>> corpus;
    for (const char* name : {"case_i", "case_ii", "case_iii"}) corpus.push_back(tokens_of(golden(name).at("token_ids")));
    const CaseIV c = load_case_iv();
    corpus.push_back(c.full[0]);
    corpus.push_back(c.full[1]);
    return corpus;
}

} // namespace

SEPAL_TEST(real_strict, real_model_rows_are_bit_identical_whatever_shares_the_call) {
    if (skipped()) return;
    Real& r = real();
    const CaseIV c = load_case_iv();
    const auto corpus = corpus_of_goldens();
    for (const std::size_t t : {std::size_t{128}, std::size_t{1024}}) {
        CudaEngine engine(r.model, real_options(Precision::F16, t, 4000));
        // The reference: case iv's family alone.
        const auto alone = run(engine, case_iv_call(c));
        Rng rng(2024 + t);
        std::size_t trials = 0, identical = 0, hits = 0, hit_identical = 0, permuted = 0, permuted_identical = 0;
        double p_diff = 0.0;
        for (std::size_t trial = 0; trial < 6; ++trial) {
            // Among other rows, at a random place.
            const std::vector<EngineRow> others = real_others(rng, corpus, 2 + rng.below(8));
            const std::size_t pos = rng.below(others.size() + 1);
            std::vector<EngineRow> family = case_iv_call(c);
            std::vector<EngineRow> shifted = others;
            for (EngineRow& o : shifted) o.parent = std::monostate{};
            std::vector<EngineRow> call(shifted.begin(), shifted.begin() + static_cast<std::ptrdiff_t>(pos));
            for (EngineRow f : family) {
                if (auto* p = std::get_if<ParentRow>(&f.parent)) p->index += pos;
                call.push_back(f);
            }
            call.insert(call.end(), shifted.begin() + static_cast<std::ptrdiff_t>(pos), shifted.end());
            const auto among = run(engine, call);
            for (std::size_t k = 1; k <= 3; ++k) {
                ++trials;
                if (same_bits(alone[k].hidden, among[pos + k].hidden)) ++identical;
                p_diff = std::max(p_diff, max_diff(probabilities(among[pos + k].hidden, c.options[k - 1]), probabilities(alone[k].hidden, c.options[k - 1])));
            }

            // Cache hit: the state stored by one call, the branches (with other rows) in a later call.
            const auto stored = run(engine, {row_of(c.state, {}, true)});
            std::vector<EngineRow> branch_call = real_others(rng, corpus, 1 + rng.below(6));
            const std::size_t at = rng.below(branch_call.size() + 1);
            std::vector<EngineRow> branches = case_iv_from_store(c, stored[0].snapshot);
            branch_call.insert(branch_call.begin() + static_cast<std::ptrdiff_t>(at), branches.begin(), branches.end());
            const auto hit = run(engine, branch_call);
            for (std::size_t k = 0; k < 3; ++k) {
                ++hits;
                if (same_bits(alone[k + 1].hidden, hit[at + k].hidden)) ++hit_identical;
            }

            // Permuted rows.
            std::vector<std::size_t> order(call.size());
            for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
            for (std::size_t i = order.size(); i > 1; --i) std::swap(order[i - 1], order[rng.below(i)]);
            std::vector<std::size_t> new_index(call.size());
            for (std::size_t i = 0; i < order.size(); ++i) new_index[order[i]] = i;
            std::vector<EngineRow> shuffled;
            for (const std::size_t old : order) {
                EngineRow row = call[old];
                if (auto* p = std::get_if<ParentRow>(&row.parent)) p->index = new_index[p->index];
                shuffled.push_back(std::move(row));
            }
            const auto again = run(engine, shuffled);
            for (std::size_t i = 0; i < order.size(); ++i) {
                ++permuted;
                if (same_bits(among[order[i]].hidden, again[i].hidden)) ++permuted_identical;
            }
        }
        std::cout << "  strict float16, gemm_rows " << t << ": " << identical << " of " << trials << " branch rows bit-identical alone vs among other rows (largest |dp| " << p_diff
                  << "); " << hit_identical << " of " << hits << " branches from a stored snapshot identical to the same branch run with its state; " << permuted_identical << " of "
                  << permuted << " rows identical after permuting the call\n";
        CHECK_EQ(identical, trials);
        CHECK_EQ(hit_identical, hits);
        CHECK_EQ(permuted_identical, permuted);
    }
}

// ---- natural mode: how far results move with the composition of the call ----

SEPAL_TEST(real_natural, drift_of_probabilities_across_batch_compositions) {
    if (skipped()) return;
    Real& r = real();
    const CaseIV c = load_case_iv();
    const auto corpus = corpus_of_goldens();
    for (const Precision precision : {Precision::F16, Precision::F32}) {
        CudaEngine engine(r.model, real_options(precision, 0, precision == Precision::F16 ? 4000 : 6000));
        const auto alone = run(engine, case_iv_call(c));
        Rng rng(31337);
        double worst_p = 0.0, worst_h = 0.0;
        std::size_t differing = 0, trials = 0;
        for (std::size_t trial = 0; trial < 12; ++trial) {
            const std::vector<EngineRow> others = real_others(rng, corpus, 1 + rng.below(12));
            const std::size_t pos = rng.below(others.size() + 1);
            std::vector<EngineRow> call(others.begin(), others.begin() + static_cast<std::ptrdiff_t>(pos));
            for (EngineRow f : case_iv_call(c)) {
                if (auto* p = std::get_if<ParentRow>(&f.parent)) p->index += pos;
                call.push_back(f);
            }
            call.insert(call.end(), others.begin() + static_cast<std::ptrdiff_t>(pos), others.end());
            const auto among = run(engine, call);
            for (std::size_t k = 1; k <= 3; ++k) {
                ++trials;
                if (!same_bits(alone[k].hidden, among[pos + k].hidden)) ++differing;
                worst_h = std::max(worst_h, max_abs_diff(alone[k].hidden.data(), among[pos + k].hidden.data(), alone[k].hidden.size()));
                worst_p = std::max(worst_p, max_diff(probabilities(among[pos + k].hidden, c.options[k - 1]), probabilities(alone[k].hidden, c.options[k - 1])));
            }
        }
        std::cout << "  natural " << (precision == Precision::F16 ? "float16" : "float32") << ": " << differing << " of " << trials
                  << " branch rows differ in bits between alone and among other rows (call sizes 4 to 16 rows); largest |dh| " << std::scientific << std::setprecision(2) << worst_h
                  << ", largest |dp| " << worst_p << std::fixed << "\n";
        CHECK(worst_p < (precision == Precision::F16 ? 0.01 : 1e-4));
    }
}

// ---- a second geometry: Qwen3.5-2B-Base (optional; there is no transformers golden for it) ----
// SEPAL_REAL_MODEL_2B names its checkpoint directory. The float32 engine is compared with the CPU reference on a few
// dozen tokens (the reference is slow), and the float16 engine with the float32 engine.

SEPAL_TEST(real_2b, second_geometry_against_the_cpu_reference_and_float16_against_float32) {
    const auto dir = environment("SEPAL_REAL_MODEL_2B");
    if (!dir || !environment("SEPAL_REAL_GOLDEN")) {
        std::cout << "SKIPPED: real-model test; set SEPAL_REAL_MODEL_2B (a Qwen3.5-2B-Base checkpoint directory) and SEPAL_REAL_GOLDEN to run this\n";
        return;
    }
    const LoadedModel model = LoadedModel::open(*dir);
    const ModelConfig& cfg = model.config();
    const std::size_t hidden = ref::dim(cfg.hidden_size);
    std::vector<std::uint32_t> text = tokens_of(golden("case_iii").at("token_ids"));
    text.resize(40);
    // A 28-token state with two branches, and a separate 12-token row, in one call.
    std::vector<EngineRow> rows = {row_of({text.begin(), text.begin() + 28}, {27}), row_of({text.begin() + 28, text.end()}, {11, 0})};
    for (std::size_t b = 0; b < 2; ++b) {
        EngineRow branch = row_of({text.begin() + 3 * (b + 1), text.begin() + 3 * (b + 1) + 5 + b}, {4 + b});
        branch.parent = ParentRow{0};
        branch.start_position = 28;
        rows.push_back(branch);
    }

    const auto start = Clock::now();
    const Weights<float> weights32 = model.weights<float>();
    std::vector<ref::Row<float>> reference(rows.size());
    for (std::size_t r = 0; r < rows.size(); ++r) {
        reference[r].token_ids = rows[r].token_ids;
        reference[r].readout_offsets = rows[r].readout_offsets;
        reference[r].start_position = rows[r].start_position.value_or(0);
        if (std::holds_alternative<ParentRow>(rows[r].parent)) reference[r].parent = ref::ParentRow{std::get<ParentRow>(rows[r].parent).index};
    }
    const std::unique_ptr<GemmProvider> gemm = make_reference_gemm();
    const auto want = ref::run_rows<float>(*gemm, cfg, weights32, std::span<const ref::Row<float>>(reference));
    std::cout << "  " << cfg.name << ": CPU reference on " << 28 + 12 + 5 + 6 << " tokens in " << std::fixed << std::setprecision(1) << seconds_since(start) << " s\n";

    std::vector<std::vector<EngineRowResult>> got;
    for (const Precision precision : {Precision::F32, Precision::F16}) {
        CudaEngine engine(model, real_options(precision, 0, precision == Precision::F32 ? 9000 : 6000));
        got.push_back(run(engine, rows));
    }
    double d32 = 0.0, d16_ref = 0.0, d16_32 = 0.0, sum16 = 0.0;
    std::size_t count = 0;
    for (std::size_t r = 0; r < rows.size(); ++r) {
        d32 = std::max(d32, max_abs_diff(got[0][r].hidden.data(), want[r].hidden.data(), got[0][r].hidden.size()));
        d16_ref = std::max(d16_ref, max_abs_diff(got[1][r].hidden.data(), want[r].hidden.data(), got[1][r].hidden.size()));
        for (std::size_t i = 0; i < got[1][r].hidden.size(); ++i) {
            const double d = std::fabs(static_cast<double>(got[1][r].hidden[i]) - got[0][r].hidden[i]);
            d16_32 = std::max(d16_32, d);
            sum16 += d;
            ++count;
        }
    }
    std::cout << "  " << cfg.name << " (hidden " << hidden << ", " << cfg.num_layers << " layers): float32 engine vs CPU reference, largest |difference| " << std::scientific
              << std::setprecision(3) << d32 << "; float16 engine vs float32 engine: largest " << d16_32 << ", mean " << sum16 / static_cast<double>(count) << std::fixed << "\n";
    CHECK(d32 <= kHiddenTolerance);
    CHECK(d16_32 < 0.05);
    (void)d16_ref;
}
