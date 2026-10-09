// The int8 multiplies (EngineOptions::int8_multiply) on the real checkpoints: accuracy against the float32 goldens for every
// variant and group combination worth reporting.
//
// Skipped (reported as skipped) unless the environment names the data:
//     real_int8      SEPAL_REAL_MODEL, SEPAL_REAL_GOLDEN (tools/export_real_golden.py: the six prompts), SEPAL_ALERTS and
//                    SEPAL_ALERT_GOLDEN (the 512-prompt alert workload and its float32 transformers hidden states)
//     real_int8_2b   SEPAL_REAL_MODEL_2B, SEPAL_ALERTS (the 2B model has no golden: the float32 engine is the reference)
// Optional: SEPAL_INT8_VARIANTS, a comma list of variant specs (default: a table of them); SEPAL_INT8_WAYS=all also runs the
// workload one alert per call and with the states cached; SEPAL_ALERT_ALERTS limits the alerts (default 64).
// Smoothing variants (a "~<statistic>:<alpha>" suffix) need calibration statistics: the float16 engine's run over a calibration workload that is NOT
// the workload it is judged on, SEPAL_CALIB_ALERTS (or calibration_workload.json next to SEPAL_ALERTS; make it with
// `tools/make_alert_workload.py --seed 4242 --out runs/calibration_workload.json`). Without one the smoothing variants are skipped, with a message.
//
// A variant spec is "base" (the default engine), "acc16" (own kernel, binary16 accumulation, flush 64, all four groups) or
// "w8a8:<groups>" / "w8a16:<groups>" with groups joined by '+': gate_up, down, mixer_in, mixer_out, mlp (gate_up+down),
// noout (all but mixer_out), all. The design's gate for an int8 profile (docs/RUNNER.md, "q8"): largest |dp| <= 0.05 and at most
// 1% answer flips against the float32 golden. The suite prints every figure and the verdict; it asserts only what must hold
// whatever the verdict (finite results, the profile that the design expects to pass, and the batch invariance of the
// int8 profile).
//
// Memory: the 0.8B engine takes about 3 GiB (weights 1.4 GiB, binary16); take the GPU lock when something else is running.

#include <chrono>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>

#include "engine_test_util.hpp"
#include "sepal/json.hpp"
#include "sepal/loader.hpp"
#include "sepal/safetensors.hpp"

using namespace sepal;
using namespace engine_test;
using sepal::cuda::CudaEngine;
using sepal::cuda::DeviceSnapshot;
using sepal::cuda::EngineOptions;
using sepal::cuda::EngineRow;
using sepal::cuda::ActivationStats;
using sepal::cuda::EngineRowResult;
using sepal::cuda::Int8Multiply;
using sepal::cuda::Int8Smoothing;
using sepal::cuda::MultiplyPath;
using sepal::cuda::ParentRow;
using sepal::cuda::Precision;

namespace {

using Clock = std::chrono::steady_clock;
double since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

constexpr std::size_t kQuestions = 8;
constexpr std::size_t kOptions = 2;
constexpr double kGateDp = 0.05;
constexpr double kGateFlips = 0.01;

// ---- variants ----

struct Variant {
    std::string spec;
    EngineOptions options;
};

std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> out;
    std::stringstream in(s);
    std::string item;
    while (std::getline(in, item, sep))
        if (!item.empty()) out.push_back(item);
    return out;
}

EngineOptions base_options(std::size_t budget_mib, std::size_t slots) {
    EngineOptions o;
    o.precision = Precision::F16;
    o.max_call_tokens = 16384;
    o.max_call_rows = 256;
    o.state_slots = slots;
    o.memory_budget_bytes = budget_mib << 20;
    return o;
}

// The statistics a "~<max|mean|rms>:<alpha>" suffix smooths with: the float16 engine run over the calibration alerts (SEPAL_CALIB_ALERTS),
// which are not the evaluation alerts. Collected on first use.

std::shared_ptr<const ActivationStats> calibration_stats(const LoadedModel& model);

Variant parse_variant(const LoadedModel& model, const std::string& whole_spec, std::size_t budget_mib, std::size_t slots) {
    // spec ::= base | acc16 | (w8a8|w8a16):<group>{+<group>}[~<max|mean|rms>:<alpha>][/<cublas16|own16-<flush>|own32>]
    //   ~: smoothing (W8A8 only); /: how the groups that are not int8 multiply
    const std::size_t slash = whole_spec.find('/');
    const std::string full_spec = whole_spec.substr(0, slash);
    const std::string rest_spec = slash == std::string::npos ? std::string() : whole_spec.substr(slash + 1);
    const std::size_t tilde = full_spec.find('~');
    const std::string spec = full_spec.substr(0, tilde);
    Variant v{whole_spec, base_options(budget_mib, slots)};
    if (rest_spec == "cublas16") {
        v.options.fp16_accumulate = {true, true, true, true, 64};
    } else if (rest_spec.rfind("own16-", 0) == 0) {
        v.options.multiply_path = MultiplyPath::Invariant;
        v.options.fp16_accumulate = {true, true, true, true, static_cast<unsigned>(std::stoul(rest_spec.substr(6)))};
    } else if (rest_spec == "own32") {
        v.options.multiply_path = MultiplyPath::Invariant;
    } else if (!rest_spec.empty()) {
        throw std::invalid_argument("bad variant suffix: " + rest_spec);
    }
    if (spec == "base") return v;
    if (spec == "acc16") {
        v.options.multiply_path = MultiplyPath::Invariant;
        v.options.fp16_accumulate = {true, true, true, true, 64};
        return v;
    }
    const std::size_t colon = spec.find(':');
    if (colon == std::string::npos || (spec.substr(0, colon) != "w8a8" && spec.substr(0, colon) != "w8a16")) throw std::invalid_argument("bad int8 variant spec: " + spec);
    Int8Multiply q;
    q.int8_activations = spec.substr(0, colon) == "w8a8";
    if (tilde != std::string::npos) {
        const std::string rest = full_spec.substr(tilde + 1);
        const std::size_t c2 = rest.find(':');
        if (c2 == std::string::npos) throw std::invalid_argument("bad smoothing spec: " + rest);
        const std::string stat = rest.substr(0, c2);
        q.smoothing.statistic = stat == "max" ? Int8Smoothing::Statistic::Max : stat == "mean" ? Int8Smoothing::Statistic::MeanAbs : Int8Smoothing::Statistic::Rms;
        q.smoothing.alpha = std::stof(rest.substr(c2 + 1));
        q.smoothing.stats = calibration_stats(model);
    }
    for (const std::string& g : split(spec.substr(colon + 1), '+')) {
        if (g == "gate_up" || g == "mlp" || g == "noout" || g == "all") q.gate_up = true;
        if (g == "down" || g == "mlp" || g == "noout" || g == "all") q.down = true;
        if (g == "mixer_in" || g == "noout" || g == "all") q.mixer_in = true;
        if (g == "mixer_out" || g == "all") q.mixer_out = true;
        if (g != "gate_up" && g != "down" && g != "mixer_in" && g != "mixer_out" && g != "mlp" && g != "noout" && g != "all") throw std::invalid_argument("bad int8 group: " + g);
    }
    v.options.int8_multiply = q;
    return v;
}

// The recommended int8 profile (README.md, "int8 multiplies"): W8A8 with smoothing for the MLP's gate/up and the mixers' two projections, the MLP's
// down projection on the engine's own kernel with binary16 accumulation flushed every 64 (all multiplies on the engine's own kernels: batch-invariant).
constexpr const char* kRecommended = "w8a8:gate_up+mixer_in+mixer_out~rms:0.6/own16-64";

std::filesystem::path calibration_path() {
    if (const auto e = environment("SEPAL_CALIB_ALERTS")) return *e;
    if (const auto a = environment("SEPAL_ALERTS")) {
        const std::filesystem::path beside = std::filesystem::path(*a).parent_path() / "calibration_workload.json";
        if (std::filesystem::exists(beside)) return beside;
    }
    return {};
}

std::vector<std::string> variant_specs() {
    if (const auto e = environment("SEPAL_INT8_VARIANTS")) return split(*e, ',');
    std::vector<std::string> specs = {"base",
                                      "acc16",
                                      "w8a16:all",
                                      "w8a16:noout",
                                      "w8a8:gate_up",
                                      "w8a8:down",
                                      "w8a8:mixer_in",
                                      "w8a8:mixer_out",
                                      "w8a8:mlp",
                                      "w8a8:gate_up+mixer_in",
                                      "w8a8:noout",
                                      "w8a8:all"};
    if (!calibration_path().empty()) {
        for (const char* s : {"w8a8:all~rms:0.6", "w8a8:gate_up~rms:0.6", "w8a8:down~rms:0.6", "w8a8:mixer_in~rms:0.6", "w8a8:mixer_out~rms:0.6",
                              "w8a8:gate_up+mixer_in+mixer_out~rms:0.6/own16-64", "w8a8:gate_up+mixer_in+mixer_out~rms:0.6/cublas16"})
            specs.push_back(s);
    } else {
        std::cout << "SMOOTHING VARIANTS NOT RUN: no calibration workload (set SEPAL_CALIB_ALERTS, see the header of this file)\n";
    }
    return specs;
}

// ---- data ----

struct Alert {
    std::vector<std::uint32_t> state;
    std::vector<std::vector<std::uint32_t>> suffixes;
};

std::vector<Alert> load_alerts(std::size_t limit, const std::filesystem::path& path = *environment("SEPAL_ALERTS")) {
    std::ifstream in(path, std::ios::binary);
    std::stringstream buffer;
    buffer << in.rdbuf();
    const json::Value root = json::parse(buffer.str());
    std::vector<Alert> alerts;
    for (const json::Value& a : root.at("alerts").items()) {
        if (alerts.size() == limit) break;
        Alert alert;
        for (const std::uint64_t id : a.at("state").as_uint_list()) alert.state.push_back(static_cast<std::uint32_t>(id));
        for (const json::Value& s : a.at("suffixes").items()) {
            std::vector<std::uint32_t> suffix;
            for (const std::uint64_t id : s.as_uint_list()) suffix.push_back(static_cast<std::uint32_t>(id));
            alert.suffixes.push_back(std::move(suffix));
        }
        alerts.push_back(std::move(alert));
    }
    return alerts;
}

std::size_t alert_count_of(const char* variable, std::size_t fallback) {
    if (const auto e = environment(variable)) return std::stoul(*e);
    return fallback;
}
std::size_t alert_count(std::size_t fallback) { return alert_count_of("SEPAL_ALERT_ALERTS", fallback); }

using Hidden = std::vector<float>;

std::vector<EngineRowResult> run(CudaEngine& e, const std::vector<EngineRow>& rows) { return e.run(std::span<const EngineRow>(rows)); }

EngineRow state_row(const Alert& a, bool keep) {
    EngineRow r;
    r.token_ids = a.state;
    r.keep_snapshot = keep;
    return r;
}

EngineRow question_row(const Alert& a, std::size_t q) {
    EngineRow r;
    r.token_ids = a.suffixes[q];
    r.readout_offsets = {a.suffixes[q].size() - 1};
    return r;
}

Hidden run_batched(CudaEngine& e, const std::vector<Alert>& alerts, std::size_t per_call, std::size_t hidden) {
    Hidden out(alerts.size() * kQuestions * hidden, 0.0f);
    for (std::size_t first = 0; first < alerts.size(); first += per_call) {
        const std::size_t count = std::min(per_call, alerts.size() - first);
        std::vector<EngineRow> rows;
        std::vector<std::pair<std::size_t, std::size_t>> owner;
        for (std::size_t i = 0; i < count; ++i) {
            const std::size_t state_at = rows.size();
            rows.push_back(state_row(alerts[first + i], false));
            owner.emplace_back(0, 0);
            for (std::size_t q = 0; q < kQuestions; ++q) {
                EngineRow r = question_row(alerts[first + i], q);
                r.parent = ParentRow{state_at};
                rows.push_back(std::move(r));
                owner.emplace_back(first + i, q);
            }
        }
        const auto results = run(e, rows);
        for (std::size_t r = 0; r < rows.size(); ++r) {
            if (rows[r].readout_offsets.empty()) continue;
            CHECK_EQ(results[r].hidden.size(), hidden);
            std::copy(results[r].hidden.begin(), results[r].hidden.end(), out.begin() + static_cast<std::ptrdiff_t>((owner[r].first * kQuestions + owner[r].second) * hidden));
        }
    }
    return out;
}

Hidden run_cached(CudaEngine& e, const std::vector<Alert>& alerts, std::size_t per_call, std::size_t hidden) {
    Hidden out(alerts.size() * kQuestions * hidden, 0.0f);
    std::vector<DeviceSnapshot> states(alerts.size());
    for (std::size_t first = 0; first < alerts.size(); first += per_call) {
        const std::size_t count = std::min(per_call, alerts.size() - first);
        std::vector<EngineRow> rows;
        for (std::size_t i = 0; i < count; ++i) rows.push_back(state_row(alerts[first + i], true));
        const auto results = run(e, rows);
        for (std::size_t i = 0; i < count; ++i) states[first + i] = results[i].snapshot;
    }
    for (std::size_t first = 0; first < alerts.size(); first += per_call) {
        const std::size_t count = std::min(per_call, alerts.size() - first);
        std::vector<EngineRow> rows;
        std::vector<std::pair<std::size_t, std::size_t>> owner;
        for (std::size_t i = 0; i < count; ++i)
            for (std::size_t q = 0; q < kQuestions; ++q) {
                EngineRow r = question_row(alerts[first + i], q);
                r.parent = states[first + i];
                rows.push_back(std::move(r));
                owner.emplace_back(first + i, q);
            }
        const auto results = run(e, rows);
        for (std::size_t r = 0; r < rows.size(); ++r)
            std::copy(results[r].hidden.begin(), results[r].hidden.end(), out.begin() + static_cast<std::ptrdiff_t>((owner[r].first * kQuestions + owner[r].second) * hidden));
    }
    return out;
}

const std::vector<Alert>& calibration_alerts() {
    static const std::vector<Alert> alerts = [] {
        if (calibration_path().empty()) throw std::invalid_argument("a smoothing variant needs SEPAL_CALIB_ALERTS (a calibration workload that is not SEPAL_ALERTS)");
        return load_alerts(alert_count_of("SEPAL_CALIB_COUNT", 64), calibration_path());
    }();
    return alerts;
}

std::shared_ptr<const ActivationStats> calibration_stats(const LoadedModel& model) {
    static std::map<std::pair<std::uint64_t, std::uint64_t>, std::shared_ptr<const ActivationStats>> cache; // by layers and hidden size
    auto& stats = cache[{model.config().num_layers, model.config().hidden_size}];
    if (stats) return stats;
    EngineOptions o = base_options(6000, 80);
    o.collect_activation_stats = true;
    o.max_call_tokens = 4096; // four alerts per call fit in it, on either model
    o.multiply_path = sepal::cuda::MultiplyPath::Invariant; // a row's activations do not depend on how the calibration was batched
    CudaEngine engine(model, o);
    (void)run_batched(engine, calibration_alerts(), 4, ref::dim(model.config().hidden_size));
    stats = std::make_shared<const ActivationStats>(engine.activation_stats());
    std::cout << "calibration: " << calibration_alerts().size() << " alerts, " << stats->tokens[0][0] << " tokens per layer\n";
    if (environment("SEPAL_CALIB_REPORT")) {
        // How peaked the channels are: for each layer and group, the median over channels of the largest |x|, the largest of them and how many are over 10 times the median.
        std::cout << "calibration report: layer group K median(amax) max(amax) max/median n>10x median(mean|x|) max(mean|x|) median(rms) max(rms)\n";
        const auto median = [](std::vector<float> v) {
            std::sort(v.begin(), v.end());
            return v.empty() ? 0.0f : v[v.size() / 2];
        };
        for (std::size_t l = 0; l < stats->layers.size(); ++l) {
            for (std::size_t g = 0; g < ActivationStats::kGroups; ++g) {
                const auto& c = stats->layers[l][g];
                const float med = median(c.amax), mx = *std::max_element(c.amax.begin(), c.amax.end());
                std::size_t over = 0;
                for (const float a : c.amax) over += a > 10.0f * med ? 1 : 0;
                std::cout << "  calib " << l << " " << g << " " << c.amax.size() << " " << std::setprecision(4) << med << " " << mx << " " << mx / std::max(med, 1e-30f) << " " << over << " "
                          << median(c.mean_abs) << " " << *std::max_element(c.mean_abs.begin(), c.mean_abs.end()) << " " << median(c.rms) << " " << *std::max_element(c.rms.begin(), c.rms.end()) << "\n";
            }
        }
    }
    return stats;
}

// ---- statistics ----

struct Dist {
    std::vector<double> v;
    void add(double x) { v.push_back(x); }
    double at(double f) const {
        if (v.empty()) return 0.0;
        std::vector<double> s = v;
        std::sort(s.begin(), s.end());
        return s[std::min(s.size() - 1, static_cast<std::size_t>(f * static_cast<double>(s.size())))];
    }
    double max() const { return v.empty() ? 0.0 : *std::max_element(v.begin(), v.end()); }
    double mean() const {
        double s = 0;
        for (const double x : v) s += x;
        return v.empty() ? 0.0 : s / static_cast<double>(v.size());
    }
};

struct Report {
    Dist dp, dh_linf, dh_rel;
    std::size_t prompts = 0, flips = 0;
    std::vector<std::string> flip_notes;
    std::size_t flips_within_001 = 0; // flips where the golden's probability is within 0.01 of one half
    double flip_limit() const { return kGateFlips * static_cast<double>(prompts); }
    bool gate() const { return dp.max() <= kGateDp && static_cast<double>(flips) <= flip_limit(); }
};

void print(const std::string& what, const Report& r) {
    std::cout << std::scientific << std::setprecision(2) << "  " << what << " (" << r.prompts << " prompts)\n"
              << "      |dp|  median " << r.dp.at(0.5) << "  p99 " << r.dp.at(0.99) << "  max " << r.dp.max() << "  mean " << r.dp.mean() << "\n"
              << "      |dh|inf max " << r.dh_linf.max() << "  mean " << r.dh_linf.mean() << "   rel L2 median " << r.dh_rel.at(0.5) << " max " << r.dh_rel.max() << "\n"
              << std::fixed << "      answer flips " << r.flips << " (" << std::setprecision(2) << 100.0 * static_cast<double>(r.flips) / static_cast<double>(std::max<std::size_t>(r.prompts, 1))
              << "%), of which with the golden within 0.01 of one half: " << r.flips_within_001 << "; gate (max |dp| <= " << kGateDp << ", flips <= 1%): " << (r.gate() ? "PASS" : "FAIL") << "\n";
    for (const std::string& note : r.flip_notes) std::cout << "      flip: " << note << "\n";
}

void summary_line(const std::string& model, const std::string& way, const std::string& spec, const Report& r) {
    std::cout << std::scientific << std::setprecision(2) << "RESULT " << model << " | " << way << " | " << spec << " | dp median " << r.dp.at(0.5) << " p99 " << r.dp.at(0.99) << " max "
              << r.dp.max() << " | rel L2 median " << r.dh_rel.at(0.5) << " | flips " << r.flips << "/" << r.prompts << " (near 0.5: " << r.flips_within_001 << ") | " << (r.gate() ? "PASS" : "FAIL")
              << std::fixed << "\n";
}

// ---- the 512-prompt workload against the transformers float32 hidden states ----

struct Alerts {
    std::vector<Alert> alerts;
    Hidden gold;
    std::size_t hidden = 0;
};

Report compare_alerts(const Hidden& got, const Hidden& want, std::size_t hidden, const DecisionReadout& readout) {
    Report rep;
    const std::size_t n = got.size() / hidden;
    for (std::size_t r = 0; r < n; ++r) {
        const float* x = got.data() + r * hidden;
        const float* y = want.data() + r * hidden;
        double linf = 0, num = 0, den = 0;
        for (std::size_t i = 0; i < hidden; ++i) {
            const double d = static_cast<double>(x[i]) - static_cast<double>(y[i]);
            linf = std::max(linf, std::fabs(d));
            num += d * d;
            den += static_cast<double>(y[i]) * static_cast<double>(y[i]);
        }
        const auto pa = readout.probabilities(std::span<const float>(x, hidden), kOptions);
        const auto pb = readout.probabilities(std::span<const float>(y, hidden), kOptions);
        rep.dp.add(std::fabs(pa[0] - pb[0]));
        rep.dh_linf.add(linf);
        rep.dh_rel.add(std::sqrt(num / std::max(den, 1e-30)));
        ++rep.prompts;
        if ((pa[0] > pa[1]) != (pb[0] > pb[1])) {
            ++rep.flips;
            if (std::fabs(pb[0] - 0.5) <= 0.01) ++rep.flips_within_001;
            std::ostringstream note;
            note << "alert " << r / kQuestions << " q" << r % kQuestions << ": golden p0 " << std::fixed << std::setprecision(5) << pb[0] << ", this engine " << pa[0] << " (|dp| "
                 << std::fabs(pa[0] - pb[0]) << ")";
            rep.flip_notes.push_back(note.str());
        }
    }
    return rep;
}

} // namespace

SEPAL_TEST(real_int8, the_six_prompts_and_the_512_prompt_workload_against_the_float32_golden) {
    if (!(environment("SEPAL_REAL_MODEL") && environment("SEPAL_REAL_GOLDEN") && environment("SEPAL_ALERTS") && environment("SEPAL_ALERT_GOLDEN"))) {
        std::cout << "SKIPPED: real-model test; set SEPAL_REAL_MODEL, SEPAL_REAL_GOLDEN, SEPAL_ALERTS and SEPAL_ALERT_GOLDEN\n";
        return;
    }
    const LoadedModel model = LoadedModel::open(*environment("SEPAL_REAL_MODEL"));
    const DecisionReadout readout = DecisionReadout::open_directory(*environment("SEPAL_REAL_MODEL"));
    const std::size_t hidden = ref::dim(model.config().hidden_size);
    const std::vector<Alert> alerts = load_alerts(alert_count(64));
    Hidden gold;
    {
        const SafetensorsFile file = SafetensorsFile::open(*environment("SEPAL_ALERT_GOLDEN"));
        const auto v = file.at("final_hidden").values<float>();
        gold.assign(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(alerts.size() * kQuestions * hidden));
    }
    const std::string golden_dir = *environment("SEPAL_REAL_GOLDEN");
    const auto open_golden = [&](const char* name) { return SafetensorsFile::open(golden_dir + "/" + name + ".safetensors"); };
    const auto tokens_of = [](const TensorView& view) {
        std::vector<std::uint32_t> out;
        for (const std::int32_t t : view.values<std::int32_t>()) out.push_back(static_cast<std::uint32_t>(t));
        return out;
    };
    const bool all_ways = environment("SEPAL_INT8_WAYS").value_or("") == "all";

    for (const std::string& spec : variant_specs()) {
        const Variant v = parse_variant(model, spec, 6000, 80);
        const auto start = Clock::now();
        CudaEngine engine(model, v.options);
        std::cout << "variant " << spec << ": engine created in " << std::fixed << std::setprecision(1) << since(start) << " s, weights " << engine.memory().weights_bytes / (1 << 20) << " MiB\n";

        // The six prompts of the real_f16 test.
        {
            Report six;
            const auto add = [&](const std::vector<float>& h, std::size_t options, const TensorView& gh, const TensorView& gp) {
                CHECK(all_finite(h));
                const auto want_h = gh.values<float>();
                double linf = 0, num = 0, den = 0;
                for (std::size_t i = 0; i < h.size(); ++i) {
                    const double d = static_cast<double>(h[i]) - want_h[i];
                    linf = std::max(linf, std::fabs(d));
                    num += d * d;
                    den += static_cast<double>(want_h[i]) * want_h[i];
                }
                const auto p = readout.probabilities(std::span<const float>(h), options);
                const auto want_p = gp.values<double>();
                double worst = 0;
                for (std::size_t i = 0; i < p.size(); ++i) worst = std::max(worst, std::fabs(p[i] - want_p[i]));
                six.dp.add(worst);
                six.dh_linf.add(linf);
                six.dh_rel.add(std::sqrt(num / std::max(den, 1e-30)));
                ++six.prompts;
                const auto argmax = [](std::span<const double> s) { return static_cast<std::size_t>(std::max_element(s.begin(), s.end()) - s.begin()); };
                if (argmax(p) != argmax(want_p)) ++six.flips;
            };
            for (const char* name : {"case_i", "case_ii", "case_iii"}) {
                const SafetensorsFile file = open_golden(name);
                const std::vector<std::uint32_t> tokens = tokens_of(file.at("token_ids"));
                EngineRow row;
                row.token_ids = tokens;
                row.readout_offsets = {tokens.size() - 1};
                const auto got = run(engine, {row});
                add(got[0].hidden, 4, file.at("final_hidden"), file.at("probs"));
            }
            {
                const SafetensorsFile file = open_golden("case_iv");
                const std::size_t split_at = std::stoul(file.metadata().at("split"));
                const std::vector<std::uint32_t> state = tokens_of(file.at("state_ids"));
                std::vector<EngineRow> rows(1);
                rows[0].token_ids = state;
                std::vector<std::vector<std::uint32_t>> full;
                for (std::size_t k = 0; k < 3; ++k) {
                    full.push_back(tokens_of(file.at("q" + std::to_string(k) + ".token_ids")));
                    EngineRow b;
                    b.token_ids.assign(full[k].begin() + static_cast<std::ptrdiff_t>(split_at), full[k].end());
                    b.readout_offsets = {b.token_ids.size() - 1};
                    b.parent = ParentRow{0};
                    b.start_position = split_at;
                    rows.push_back(b);
                }
                const auto got = run(engine, rows);
                const std::size_t options[3] = {2, 4, 3};
                for (std::size_t k = 0; k < 3; ++k) {
                    const std::string q = "q" + std::to_string(k);
                    add(got[k + 1].hidden, options[k], file.at(q + ".final_hidden"), file.at(q + ".probs"));
                }
            }
            print(spec + ", the six real_f16 prompts", six);
            summary_line("0.8B", "6 prompts", spec, six);
            if (spec == "base") CHECK(six.gate());
        }

        // The workload.
        const auto t = Clock::now();
        const Hidden x16 = run_batched(engine, alerts, 16, hidden);
        std::cout << "    workload (16 alerts per call) in " << std::fixed << std::setprecision(1) << since(t) << " s\n";
        CHECK(all_finite(x16));
        const Report r16 = compare_alerts(x16, gold, hidden, readout);
        print(spec + ", " + std::to_string(alerts.size() * kQuestions) + " alert prompts, 16 alerts per call", r16);
        summary_line("0.8B", std::to_string(r16.prompts) + " prompts x16", spec, r16);
        if (spec == "base" || spec == kRecommended) CHECK(r16.gate());
        if ((all_ways || spec == kRecommended) && spec != "base" && spec != "acc16") {
            // Row independence of the int8 profile on the real model: other ways of batching give the same bits when every group is
            // run by the own kernels (strict mode needs no padding for them); with cuBLAS groups left in they differ a little.
            const Hidden alone = run_batched(engine, alerts, 1, hidden);
            const Hidden cached = run_cached(engine, alerts, 16, hidden);
            const Report ra = compare_alerts(alone, gold, hidden, readout), rc = compare_alerts(cached, gold, hidden, readout);
            print(spec + ", one alert per call", ra);
            print(spec + ", states cached, 16 per call", rc);
            summary_line("0.8B", "alone", spec, ra);
            summary_line("0.8B", "cached x16", spec, rc);
            const bool all_int8 = v.options.int8_multiply.gate_up && v.options.int8_multiply.down && v.options.int8_multiply.mixer_in && v.options.int8_multiply.mixer_out;
            if (all_int8 || v.options.multiply_path == MultiplyPath::Invariant) {
                CHECK(same_bits(alone, x16));
                CHECK(same_bits(cached, x16));
            }
        }
    }
}

SEPAL_TEST(real_int8_2b, int8_variants_against_the_float32_engine_on_the_second_geometry) {
    const auto dir = environment("SEPAL_REAL_MODEL_2B");
    if (!dir || !environment("SEPAL_ALERTS")) {
        std::cout << "SKIPPED: set SEPAL_REAL_MODEL_2B (a Qwen3.5-2B-Base checkpoint directory) and SEPAL_ALERTS to run this\n";
        return;
    }
    const LoadedModel model = LoadedModel::open(*dir);
    const std::size_t hidden = ref::dim(model.config().hidden_size);
    const std::vector<Alert> alerts = load_alerts(alert_count(8));
    Hidden want;
    {
        EngineOptions f32 = base_options(9000, 40);
        f32.precision = Precision::F32;
        f32.max_call_tokens = 4096;
        CudaEngine engine(model, f32);
        want = run_batched(engine, alerts, 4, hidden);
    }
    std::vector<std::string> specs = environment("SEPAL_INT8_VARIANTS") ? variant_specs()
                                                                        : std::vector<std::string>{"base", "acc16", "w8a16:all", "w8a8:gate_up", "w8a8:down", "w8a8:mixer_in", "w8a8:mixer_out", "w8a8:mlp", "w8a8:noout", "w8a8:all"};
    if (!environment("SEPAL_INT8_VARIANTS") && !calibration_path().empty()) {
        for (const char* s : {"w8a8:all~rms:0.6", "w8a8:gate_up+mixer_in+mixer_out~rms:0.6/cublas16", kRecommended}) specs.push_back(s);
    }
    for (const std::string& spec : specs) {
        Variant v = parse_variant(model, spec, 6000, 40);
        v.options.max_call_tokens = 4096;
        CudaEngine engine(model, v.options);
        const Hidden got = run_batched(engine, alerts, 4, hidden);
        CHECK(all_finite(got));
        double sum = 0, worst = 0, rel_sum = 0;
        std::size_t count = 0, rows = got.size() / hidden;
        for (std::size_t r = 0; r < rows; ++r) {
            double num = 0, den = 0;
            for (std::size_t i = 0; i < hidden; ++i) {
                const double d = static_cast<double>(got[r * hidden + i]) - want[r * hidden + i];
                worst = std::max(worst, std::fabs(d));
                sum += std::fabs(d);
                num += d * d;
                den += static_cast<double>(want[r * hidden + i]) * want[r * hidden + i];
                ++count;
            }
            rel_sum += std::sqrt(num / std::max(den, 1e-30));
        }
        std::cout << "  Qwen3.5-2B, " << spec << ", " << rows << " alert prompts against the float32 engine: weights " << engine.memory().weights_bytes / (1 << 20) << " MiB, mean |dh| "
                  << std::scientific << std::setprecision(2) << sum / static_cast<double>(count) << ", largest |dh| " << worst << ", mean relative L2 " << rel_sum / static_cast<double>(rows)
                  << std::fixed << "\n";
        std::cout << "RESULT 2B | " << rows << " prompts | " << spec << " | mean rel L2 " << std::scientific << std::setprecision(2) << rel_sum / static_cast<double>(rows) << std::fixed << "\n";
    }
}
