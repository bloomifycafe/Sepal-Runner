// The GPU engine on the real jeff-base checkpoint over the whole 64-alert workload (64 states x 8 questions = 512
// prompts), run in several ways and compared with transformers' float32 hidden states for every prompt
// (tools/export_alert_golden.py), and for a sample of alerts with the CPU float32 reference.
//
// Skipped (reported as skipped) unless all three are set:
//     SEPAL_REAL_MODEL     the jeff-base checkpoint directory
//     SEPAL_ALERTS         runs/alert_workload.json (tools/make_alert_workload.py)
//     SEPAL_ALERT_GOLDEN   runs/alert-golden/alert_golden.safetensors
// Optional: SEPAL_ALERT_CPU_SAMPLES (alerts to run through the CPU reference, default 0), SEPAL_ALERT_ALERTS (use only
// the first N alerts, default 64).
//
// Suite real_alerts. It prints the distributions it measures; its checks are the claims of the engine's README.
// Memory: about 6 GiB of the device for the float32 engine; take the GPU lock before running it next to a benchmark.

#include <chrono>
#include <fstream>
#include <sstream>

#include "engine_test_util.hpp"
#include "sepal/json.hpp"
#include "sepal/loader.hpp"
#include "sepal/safetensors.hpp"

using namespace sepal;
using namespace engine_test;
using sepal::cuda::CudaEngine;
using sepal::cuda::DeviceSnapshot;
using sepal::cuda::EngineMemory;
using sepal::cuda::EngineOptions;
using sepal::cuda::EngineRow;
using sepal::cuda::EngineRowResult;
using sepal::cuda::ParentRow;
using sepal::cuda::Precision;

namespace {

using Clock = std::chrono::steady_clock;
double since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

constexpr std::size_t kQuestions = 8;
constexpr std::size_t kOptions = 2;

struct Alert {
    std::vector<std::uint32_t> state;
    std::vector<std::vector<std::uint32_t>> suffixes;
};

bool skipped() {
    if (environment("SEPAL_REAL_MODEL") && environment("SEPAL_ALERTS") && environment("SEPAL_ALERT_GOLDEN")) return false;
    std::cout << "SKIPPED: real-model test; set SEPAL_REAL_MODEL, SEPAL_ALERTS (runs/alert_workload.json) and SEPAL_ALERT_GOLDEN (runs/alert-golden/"
                 "alert_golden.safetensors, from tools/export_alert_golden.py)\n";
    return true;
}

std::vector<Alert> load_alerts(std::size_t limit) {
    std::ifstream in(*environment("SEPAL_ALERTS"), std::ios::binary);
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

// Hidden states [alert][question][hidden] flattened.
using Hidden = std::vector<float>;

struct Real {
    LoadedModel model;
    DecisionReadout readout;
    std::size_t hidden;
};

Real& real() {
    static Real r{LoadedModel::open(*environment("SEPAL_REAL_MODEL")), DecisionReadout::open_directory(*environment("SEPAL_REAL_MODEL")), 0};
    r.hidden = ref::dim(r.model.config().hidden_size);
    return r;
}

EngineOptions options(Precision p, std::size_t gemm_rows) {
    EngineOptions o;
    o.precision = p;
    o.gemm_rows = gemm_rows;
    o.max_call_tokens = 16384;
    o.max_call_rows = 256;
    o.state_slots = 80;
    o.memory_budget_bytes = (p == Precision::F32 ? std::size_t{7600} : std::size_t{6000}) << 20;
    return o;
}

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

void store(Hidden& out, std::size_t alert, std::size_t q, const std::vector<float>& h) {
    CHECK_EQ(h.size(), real().hidden);
    std::copy(h.begin(), h.end(), out.begin() + static_cast<std::ptrdiff_t>((alert * kQuestions + q) * real().hidden));
}

// ---- the ways of running the workload ----

// n alerts per call, each a state row and its eight questions as children. `reverse` reverses the order of alerts and of
// questions inside the call.
Hidden run_batched(CudaEngine& e, const std::vector<Alert>& alerts, std::size_t per_call, bool reverse) {
    Hidden out(alerts.size() * kQuestions * real().hidden, 0.0f);
    for (std::size_t first = 0; first < alerts.size(); first += per_call) {
        const std::size_t count = std::min(per_call, alerts.size() - first);
        std::vector<EngineRow> rows;
        std::vector<std::pair<std::size_t, std::size_t>> owner; // row -> (alert, question), questions only
        std::vector<std::size_t> order(count);
        for (std::size_t i = 0; i < count; ++i) order[i] = reverse ? count - 1 - i : i;
        for (const std::size_t i : order) {
            const std::size_t state_at = rows.size();
            rows.push_back(state_row(alerts[first + i], false));
            owner.emplace_back(0, 0);
            for (std::size_t k = 0; k < kQuestions; ++k) {
                const std::size_t q = reverse ? kQuestions - 1 - k : k;
                EngineRow r = question_row(alerts[first + i], q);
                r.parent = ParentRow{state_at};
                rows.push_back(std::move(r));
                owner.emplace_back(first + i, q);
            }
        }
        const auto results = run(e, rows);
        for (std::size_t r = 0; r < rows.size(); ++r) {
            if (!rows[r].readout_offsets.empty()) store(out, owner[r].first, owner[r].second, results[r].hidden);
        }
    }
    return out;
}

// States first (kept as snapshots, n per call), then the questions in later calls: `groups` lists the questions of each
// later call (for example {0..7}, or {0,1,2} then {3..7}, or one at a time).
Hidden run_cached(CudaEngine& e, const std::vector<Alert>& alerts, std::size_t per_call, const std::vector<std::vector<std::size_t>>& groups) {
    Hidden out(alerts.size() * kQuestions * real().hidden, 0.0f);
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
        for (const auto& group : groups) {
            std::vector<EngineRow> rows;
            std::vector<std::pair<std::size_t, std::size_t>> owner;
            for (std::size_t i = 0; i < count; ++i) {
                for (const std::size_t q : group) {
                    EngineRow r = question_row(alerts[first + i], q);
                    r.parent = states[first + i];
                    rows.push_back(std::move(r));
                    owner.emplace_back(first + i, q);
                }
            }
            const auto results = run(e, rows);
            for (std::size_t r = 0; r < rows.size(); ++r) store(out, owner[r].first, owner[r].second, results[r].hidden);
        }
    }
    return out;
}

// ---- statistics ----

struct Dist {
    std::vector<double> v;
    void add(double x) { v.push_back(x); }
    std::string summary() const {
        if (v.empty()) return "(none)";
        std::vector<double> s = v;
        std::sort(s.begin(), s.end());
        const auto q = [&](double f) { return s[std::min(s.size() - 1, static_cast<std::size_t>(f * static_cast<double>(s.size())))]; };
        double sum = 0;
        for (const double x : s) sum += x;
        std::ostringstream o;
        o << std::scientific << std::setprecision(2) << "n=" << s.size() << " min " << s.front() << " median " << q(0.5) << " mean " << sum / static_cast<double>(s.size()) << " p90 " << q(0.9)
          << " p99 " << q(0.99) << " max " << s.back();
        return o.str();
    }
    double max() const { return v.empty() ? 0.0 : *std::max_element(v.begin(), v.end()); }
    double p99() const {
        std::vector<double> s = v;
        std::sort(s.begin(), s.end());
        return s.empty() ? 0.0 : s[std::min(s.size() - 1, static_cast<std::size_t>(0.99 * static_cast<double>(s.size())))];
    }
};

struct Comparison {
    Dist dp, dh, dh_rel;      // |Δp| of option 0, max |Δ hidden|, relative L2 distance of the hidden states
    std::size_t flips = 0;    // answer flips (argmax of the two probabilities)
    std::size_t bit_equal = 0;
    std::size_t rows = 0;
    std::vector<std::pair<double, std::size_t>> worst; // (|Δp|, row)
};

std::vector<double> prob_of(const Hidden& h, std::size_t row) {
    return real().readout.probabilities(std::span<const float>(h.data() + row * real().hidden, real().hidden), kOptions);
}

Comparison compare(const Hidden& a, const Hidden& b) {
    Comparison c;
    const std::size_t H = real().hidden, n = a.size() / H;
    for (std::size_t r = 0; r < n; ++r) {
        const float* x = a.data() + r * H;
        const float* y = b.data() + r * H;
        double linf = 0, num = 0, den = 0;
        for (std::size_t i = 0; i < H; ++i) {
            const double d = static_cast<double>(x[i]) - static_cast<double>(y[i]);
            linf = std::max(linf, std::fabs(d));
            num += d * d;
            den += static_cast<double>(y[i]) * static_cast<double>(y[i]);
        }
        const auto pa = prob_of(a, r), pb = prob_of(b, r);
        const double dp = std::fabs(pa[0] - pb[0]);
        c.dp.add(dp);
        c.dh.add(linf);
        c.dh_rel.add(std::sqrt(num / std::max(den, 1e-30)));
        if ((pa[0] > pa[1]) != (pb[0] > pb[1])) ++c.flips;
        if (std::memcmp(x, y, H * sizeof(float)) == 0) ++c.bit_equal;
        c.worst.emplace_back(dp, r);
        ++c.rows;
    }
    std::sort(c.worst.rbegin(), c.worst.rend());
    return c;
}

void report(const std::string& what, const Comparison& c) {
    std::cout << "  " << what << " (" << c.rows << " prompts)\n"
              << "      |dp|      " << c.dp.summary() << "\n"
              << "      |dh|inf   " << c.dh.summary() << "\n"
              << "      rel L2    " << c.dh_rel.summary() << "\n"
              << "      answer flips " << c.flips << ", bit-identical rows " << c.bit_equal << ", largest |dp| at";
    for (std::size_t i = 0; i < std::min<std::size_t>(3, c.worst.size()); ++i) {
        std::cout << " (alert " << c.worst[i].second / kQuestions << " q" << c.worst[i].second % kQuestions << ": " << std::scientific << std::setprecision(2) << c.worst[i].first << ")";
    }
    std::cout << std::fixed << "\n";
}

// A single row is an outlier when its |dp| exceeds 8x the 99th percentile and the 99th percentile is itself nonzero.
bool has_outlier(const Comparison& c) { return c.dp.max() > 8.0 * std::max(c.dp.p99(), 1e-7); }

Hidden golden_hidden(std::size_t alerts) {
    static const SafetensorsFile file = SafetensorsFile::open(*environment("SEPAL_ALERT_GOLDEN"));
    const auto done = file.at("done").values<std::int32_t>();
    for (std::size_t a = 0; a < alerts; ++a) CHECK(done[a] == 1);
    const auto v = file.at("final_hidden").values<float>();
    return Hidden(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(alerts * kQuestions * real().hidden));
}

std::size_t alert_count() {
    std::size_t n = 64;
    if (const auto e = environment("SEPAL_ALERT_ALERTS")) n = std::stoul(*e);
    return n;
}

} // namespace

SEPAL_TEST(real_alerts, the_64_alert_workload_in_every_way_against_transformers_float32) {
    if (skipped()) return;
    const std::vector<Alert> alerts = load_alerts(alert_count());
    const std::size_t H = real().hidden;
    const Hidden gold = golden_hidden(alerts.size());
    std::cout << "  " << alerts.size() << " alerts, " << alerts.size() * kQuestions << " prompts; golden = transformers float32 (TF32 off)\n";

    // Sanity: the golden's own probabilities are the readout of its hidden states.
    {
        const SafetensorsFile file = SafetensorsFile::open(*environment("SEPAL_ALERT_GOLDEN"));
        const auto p = file.at("probs").values<double>();
        double worst = 0;
        for (std::size_t r = 0; r < alerts.size() * kQuestions; ++r) {
            const auto mine = prob_of(gold, r);
            for (std::size_t o = 0; o < kOptions; ++o) worst = std::max(worst, std::fabs(mine[o] - p[r * kOptions + o]));
        }
        std::cout << "  golden probabilities vs the readout of the golden hidden states: " << std::scientific << worst << std::fixed << "\n";
        CHECK(worst < 1e-6);
    }

    const std::vector<std::vector<std::size_t>> all_q = {{0, 1, 2, 3, 4, 5, 6, 7}};
    const std::vector<std::vector<std::size_t>> split_q = {{0, 1, 2}, {3, 4, 5, 6, 7}};
    std::vector<std::vector<std::size_t>> one_q;
    for (std::size_t q = 0; q < kQuestions; ++q) one_q.push_back({q});

    std::map<std::string, Hidden> natural32, natural16, strict32, strict16;
    const auto timed = [&](std::map<std::string, Hidden>& into, const std::string& name, auto&& fn) {
        const auto t = Clock::now();
        into[name] = fn();
        std::cout << "    ran " << name << " in " << std::fixed << std::setprecision(1) << since(t) << " s" << "\n";
    };
    const auto suite = [&](Precision p, std::size_t gemm_rows, std::map<std::string, Hidden>& into) {
        CudaEngine engine(real().model, options(p, gemm_rows));
        std::cout << "  engine " << (p == Precision::F32 ? "float32" : "float16") << (gemm_rows == 0 ? " natural" : " strict " + std::to_string(gemm_rows)) << ": reserved "
                  << engine.memory().reserved_bytes() / (1 << 20) << " MiB" << "\n";
        timed(into, "alone", [&] { return run_batched(engine, alerts, 1, false); });
        timed(into, "x16", [&] { return run_batched(engine, alerts, 16, false); });
        timed(into, "x16 reversed", [&] { return run_batched(engine, alerts, 16, true); });
        timed(into, "cached x16", [&] { return run_cached(engine, alerts, 16, all_q); });
        timed(into, "cached x16 split 3+5", [&] { return run_cached(engine, alerts, 16, split_q); });
        timed(into, "cached x1 one question per call", [&] { return run_cached(engine, alerts, 1, one_q); });
        const EngineMemory m = engine.memory();
        std::cout << "    peak: slots " << m.slots_peak << " of " << m.slots_total << ", heap " << m.kv_peak_bytes / (1 << 20) << " of " << m.kv_heap_bytes / (1 << 20) << " MiB" << "\n";
    };

    // ---- float32 engine, natural ----
    suite(Precision::F32, 0, natural32);
    for (const auto& [name, h] : natural32) {
        const Comparison c = compare(h, gold);
        report("float32 natural '" + name + "' vs transformers float32", c);
        CHECK(c.dh.max() < 1e-4);
        CHECK(c.dp.max() < 1e-5);
        CHECK_EQ(c.flips, std::size_t{0});
        CHECK(!has_outlier(c));
    }
    {
        const auto& base = natural32.at("alone");
        for (const auto& [name, h] : natural32) {
            if (name == "alone") continue;
            const Comparison c = compare(h, base);
            report("float32 natural '" + name + "' vs float32 natural 'alone'", c);
            CHECK(c.dh.max() < 1e-4);
            CHECK_EQ(c.flips, std::size_t{0});
        }
    }
    // ---- float16 engine, natural ----
    suite(Precision::F16, 0, natural16);
    for (const auto& [name, h] : natural16) {
        const Comparison c = compare(h, gold);
        report("float16 natural '" + name + "' vs transformers float32", c);
        CHECK(c.dp.max() < 0.02);
        CHECK(!has_outlier(c));
    }
    {
        const Comparison c = compare(natural16.at("alone"), natural32.at("alone"));
        report("float16 natural 'alone' vs float32 natural 'alone'", c);
        CHECK(!has_outlier(c));
        CHECK_EQ(c.flips, std::size_t{0});
        for (const auto& [name, h] : natural16) {
            if (name == "alone") continue;
            const Comparison d = compare(h, natural16.at("alone"));
            report("float16 natural '" + name + "' vs float16 natural 'alone' (batch composition)", d);
            CHECK_EQ(d.flips, std::size_t{0});
        }
    }
    // ---- strict ----
    for (const Precision p : {Precision::F32, Precision::F16}) {
        auto& into = p == Precision::F32 ? strict32 : strict16;
        suite(p, 128, into);
        const Hidden& base = into.at("alone");
        for (const auto& [name, h] : into) {
            const bool same = h.size() == base.size() && std::memcmp(h.data(), base.data(), h.size() * sizeof(float)) == 0;
            std::cout << "  " << (p == Precision::F32 ? "float32" : "float16") << " strict 128 '" << name << "' vs 'alone': " << (same ? "BIT-IDENTICAL" : "DIFFERENT") << "\n";
            CHECK(same);
        }
        const Comparison c = compare(base, gold);
        report(std::string(p == Precision::F32 ? "float32" : "float16") + " strict 128 vs transformers float32", c);
        const Comparison d = compare(base, p == Precision::F32 ? natural32.at("alone") : natural16.at("alone"));
        report(std::string(p == Precision::F32 ? "float32" : "float16") + " strict 128 vs the same precision natural 'alone'", d);
    }

    // ---- CPU reference on a sample ----
    std::size_t samples = 0;
    if (const auto e = environment("SEPAL_ALERT_CPU_SAMPLES")) samples = std::stoul(*e);
    if (samples != 0) {
        const Weights<float> weights = real().model.weights<float>();
        const std::unique_ptr<GemmProvider> gemm = make_reference_gemm();
        Hidden cpu;
        Hidden mine, theirs, ours16;
        double total = 0;
        for (std::size_t s = 0; s < samples; ++s) {
            const std::size_t a = samples == 1 ? 0 : s * (alerts.size() - 1) / (samples - 1);
            std::vector<ref::Row<float>> rows(1 + kQuestions);
            rows[0].token_ids = alerts[a].state;
            for (std::size_t q = 0; q < kQuestions; ++q) {
                rows[1 + q].token_ids = alerts[a].suffixes[q];
                rows[1 + q].readout_offsets = {alerts[a].suffixes[q].size() - 1};
                rows[1 + q].start_position = alerts[a].state.size();
                rows[1 + q].parent = ref::ParentRow{0};
            }
            const auto t = Clock::now();
            const auto want = ref::run_rows<float>(*gemm, real().model.config(), weights, rows);
            const double sec = since(t);
            total += sec;
            std::size_t tokens = alerts[a].state.size();
            for (const auto& suffix : alerts[a].suffixes) tokens += suffix.size();
            std::cout << "    CPU reference, alert " << a << " (" << tokens << " tokens): " << std::fixed << std::setprecision(1) << sec << " s" << "\n";
            for (std::size_t q = 0; q < kQuestions; ++q) {
                const auto v = want[1 + q].hidden.span();
                cpu.insert(cpu.end(), v.begin(), v.end());
                const float* g = gold.data() + (a * kQuestions + q) * H;
                theirs.insert(theirs.end(), g, g + H);
                const float* m = natural32.at("alone").data() + (a * kQuestions + q) * H;
                mine.insert(mine.end(), m, m + H);
                const float* m16 = natural16.at("alone").data() + (a * kQuestions + q) * H;
                ours16.insert(ours16.end(), m16, m16 + H);
            }
        }
        std::cout << "  CPU reference: " << samples << " alerts in " << std::fixed << std::setprecision(0) << total << " s" << "\n";
        report("CPU float32 reference vs transformers float32", compare(cpu, theirs));
        const Comparison c = compare(mine, cpu);
        report("float32 engine (natural, alone) vs CPU float32 reference", c);
        CHECK(c.dh.max() < 1e-4);
        report("float16 engine (natural, alone) vs CPU float32 reference", compare(ours16, cpu));
    }
}
