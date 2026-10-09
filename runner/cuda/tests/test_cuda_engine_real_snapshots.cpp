// Snapshot density and tiers on the real checkpoints (the numbers quoted in README.md, "Snapshot density and tiers").
//
// Skipped (reported as skipped) unless set:
//     SEPAL_REAL_MODEL, SEPAL_REAL_GOLDEN   as for the other real-model suites (real_half_state, real_tiers)
//     SEPAL_ALERTS, SEPAL_ALERT_GOLDEN      the alert workload and its float32 golden (real_half_state)
//     SEPAL_REAL_MODEL_2B                   Qwen3.5-2B-Base (real_tiers; skipped for the 2B when absent)
//
// real_half_state: the accuracy cost of keeping the DeltaNet state of snapshots in binary16: the six prompts of
// real_f16, the 512 prompts of the alert workload run from stored states, a long row (8k tokens), a chain of 30 links,
// and the largest magnitudes of the state on real data. Every comparison is against the same engine with float32 state.
// real_tiers: bytes per snapshot, how many fit a 12 GB card, and the times of every tier against recomputing the state.
// These use 4 to 10 GiB of the device: hold the GPU lock.

#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <tuple>

#include "engine_test_util.hpp"
#include "sepal/json.hpp"
#include "sepal/loader.hpp"
#include "sepal/safetensors.hpp"

using namespace sepal;
using namespace engine_test;
using sepal::cuda::BudgetExceeded;
using sepal::cuda::CudaEngine;
using sepal::cuda::DeviceSnapshot;
using sepal::cuda::EngineMemory;
using sepal::cuda::EngineOptions;
using sepal::cuda::EngineRow;
using sepal::cuda::EngineRowResult;
using sepal::cuda::HostMemory;
using sepal::cuda::HostSnapshot;
using sepal::cuda::ParentRow;
using sepal::cuda::Precision;

namespace {

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

constexpr std::size_t kQuestions = 8;

struct Alert {
    std::vector<std::uint32_t> state;
    std::vector<std::vector<std::uint32_t>> suffixes;
};

bool skipped_half() {
    if (environment("SEPAL_REAL_MODEL") && environment("SEPAL_REAL_GOLDEN") && environment("SEPAL_ALERTS") && environment("SEPAL_ALERT_GOLDEN")) return false;
    std::cout << "SKIPPED: real-model test; set SEPAL_REAL_MODEL, SEPAL_REAL_GOLDEN, SEPAL_ALERTS and SEPAL_ALERT_GOLDEN\n";
    return true;
}

bool skipped_tiers() {
    if (environment("SEPAL_REAL_MODEL") && environment("SEPAL_ALERTS")) return false;
    std::cout << "SKIPPED: real-model test; set SEPAL_REAL_MODEL and SEPAL_ALERTS (SEPAL_REAL_MODEL_2B for the 2B)\n";
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

struct Real {
    LoadedModel model;
    DecisionReadout readout;
};
Real& real() {
    static Real r{LoadedModel::open(*environment("SEPAL_REAL_MODEL")), DecisionReadout::open_directory(*environment("SEPAL_REAL_MODEL"))};
    return r;
}
std::size_t hidden_size() { return ref::dim(real().model.config().hidden_size); }

SafetensorsFile golden(const char* name) { return SafetensorsFile::open(*environment("SEPAL_REAL_GOLDEN") + "/" + name + ".safetensors"); }
std::vector<std::uint32_t> tokens_of(const TensorView& view) {
    std::vector<std::uint32_t> out;
    for (const std::int32_t t : view.values<std::int32_t>()) out.push_back(static_cast<std::uint32_t>(t));
    return out;
}

EngineOptions options(bool half_state, std::size_t budget_mib = 6200, std::size_t slots = 80, std::size_t max_tokens = 16384) {
    EngineOptions o;
    o.precision = Precision::F16;
    o.gemm_rows = 0;
    o.max_call_tokens = max_tokens;
    o.max_call_rows = 256;
    o.state_slots = slots;
    o.memory_budget_bytes = budget_mib << 20;
    o.half_state = half_state;
    return o;
}

std::vector<EngineRowResult> run(CudaEngine& e, const std::vector<EngineRow>& rows) { return e.run(std::span<const EngineRow>(rows)); }

EngineRow row_of(const std::vector<std::uint32_t>& t, std::vector<std::size_t> readouts, bool keep = false) {
    EngineRow r;
    r.token_ids = t;
    r.readout_offsets = std::move(readouts);
    r.keep_snapshot = keep;
    return r;
}

struct Dist {
    std::vector<double> v;
    void add(double x) { v.push_back(x); }
    double q(double f) const {
        if (v.empty()) return 0;
        std::vector<double> s = v;
        std::sort(s.begin(), s.end());
        return s[std::min(s.size() - 1, static_cast<std::size_t>(f * static_cast<double>(s.size())))];
    }
    double max() const { return v.empty() ? 0.0 : *std::max_element(v.begin(), v.end()); }
    std::string summary() const {
        std::ostringstream o;
        o << std::scientific << std::setprecision(2) << "n=" << v.size() << " median " << q(0.5) << " p99 " << q(0.99) << " max " << max();
        return o.str();
    }
};

std::vector<double> probs(const float* h, std::size_t options_count) { return real().readout.probabilities(std::span<const float>(h, hidden_size()), options_count); }

// Compares two sets of hidden states [n, H]: |dp| of option 0, hidden L-inf and relative L2 distance, answer flips.
struct Cmp {
    Dist dp, dh, rel;
    std::size_t flips = 0, hard_flips = 0, n = 0;
};

Cmp compare(const std::vector<float>& a, const std::vector<float>& b, std::size_t options_count = 2) {
    Cmp c;
    const std::size_t H = hidden_size();
    c.n = a.size() / H;
    for (std::size_t r = 0; r < c.n; ++r) {
        double linf = 0, num = 0, den = 0;
        for (std::size_t i = 0; i < H; ++i) {
            const double d = static_cast<double>(a[r * H + i]) - static_cast<double>(b[r * H + i]);
            linf = std::max(linf, std::fabs(d));
            num += d * d;
            den += static_cast<double>(b[r * H + i]) * static_cast<double>(b[r * H + i]);
        }
        const auto pa = probs(a.data() + r * H, options_count), pb = probs(b.data() + r * H, options_count);
        c.dp.add(std::fabs(pa[0] - pb[0]));
        c.dh.add(linf);
        c.rel.add(std::sqrt(num / std::max(den, 1e-30)));
        const std::size_t ma = static_cast<std::size_t>(std::max_element(pa.begin(), pa.end()) - pa.begin());
        const std::size_t mb = static_cast<std::size_t>(std::max_element(pb.begin(), pb.end()) - pb.begin());
        if (ma != mb) {
            ++c.flips;
            if (std::fabs(pa[0] - 0.5) > 0.01 && std::fabs(pb[0] - 0.5) > 0.01) ++c.hard_flips;
        }
    }
    return c;
}

void print(const std::string& what, const Cmp& c) {
    std::cout << "  [measure] " << what << " (" << c.n << " prompts)\n"
              << "      |dp|   " << c.dp.summary() << "\n      |dh|inf " << c.dh.summary() << "\n      relL2  " << c.rel.summary() << "\n      flips " << c.flips
              << " (away from a tie " << c.hard_flips << ")\n";
}

// ---- magnitudes of the state, from a float32 snapshot on the host ----

struct Range {
    double max_recurrent = 0, max_conv = 0;
    std::size_t values = 0, nonzero = 0, below_subnormal_edge = 0 /* |x| < 2^-14: binary16 subnormal */, flushed = 0 /* |x| < 2^-25: rounds to zero */;
    std::vector<double> per_layer_recurrent;
    void add(const HostSnapshot& snap, const ModelConfig& cfg) {
        const std::span<const std::byte> bytes = snap.state_bytes();
        const std::size_t dn_layers = ref::dim(cfg.num_deltanet_layers());
        const std::size_t conv = ref::dim(cfg.conv_kernel - 1) * ref::dim(cfg.dn_conv_dim());
        const std::size_t rec = ref::dim(cfg.dn_value_heads * cfg.dn_key_head_dim * cfg.dn_value_head_dim);
        const auto* f = reinterpret_cast<const float*>(bytes.data());
        CHECK_EQ(bytes.size(), dn_layers * (conv + rec) * 4);
        if (per_layer_recurrent.size() < dn_layers) per_layer_recurrent.assign(dn_layers, 0.0);
        for (std::size_t d = 0; d < dn_layers; ++d) {
            const float* c = f + d * (conv + rec);
            const float* r = c + conv;
            for (std::size_t i = 0; i < conv; ++i) max_conv = std::max(max_conv, static_cast<double>(std::fabs(c[i])));
            for (std::size_t i = 0; i < rec; ++i) {
                const double x = std::fabs(static_cast<double>(r[i]));
                max_recurrent = std::max(max_recurrent, x);
                per_layer_recurrent[d] = std::max(per_layer_recurrent[d], x);
                ++values;
                if (x == 0.0) continue;
                ++nonzero;
                if (x < 6.103515625e-5) {
                    ++below_subnormal_edge;
                    if (x < 2.98023223876953125e-8) ++flushed;
                }
            }
        }
    }
};

void print_range(const std::string& what, const Range& r) {
    std::cout << "  [measure] range, " << what << ": max |recurrent| " << std::scientific << std::setprecision(3) << r.max_recurrent << ", max |conv tail| " << r.max_conv
              << " (binary16 max 65504); recurrent values " << std::defaultfloat << r.values << ", nonzero " << r.nonzero << ", below 2^-14 (binary16 subnormal) " << r.below_subnormal_edge
              << " (" << std::fixed << std::setprecision(3) << 100.0 * static_cast<double>(r.below_subnormal_edge) / static_cast<double>(std::max<std::size_t>(r.nonzero, 1)) << "% of nonzero), below 2^-25 (rounds to 0) "
              << r.flushed << " (" << 100.0 * static_cast<double>(r.flushed) / static_cast<double>(std::max<std::size_t>(r.nonzero, 1)) << "%)\n";
}

// Everything one engine setting produces, for comparison with the other.
struct Outputs {
    std::vector<float> case_iv_store;                    // 3 branches from a stored case-iv state, in H floats each
    std::vector<float> case_iv_call;                     // 3 branches in the call of their state
    std::vector<float> plain;                            // case i-iii
    std::vector<float> alerts;                           // 64 x 8 hidden states, run from stored states
    std::vector<float> long_row;                         // 8 questions after an 8k row
    std::vector<std::vector<float>> chain;               // per probed depth, 8 questions
    std::vector<std::size_t> chain_depths;
    std::uint64_t saturations = 0;
};

struct CaseIV {
    std::vector<std::uint32_t> state;
    std::vector<std::vector<std::uint32_t>> suffix;
    std::vector<std::size_t> options = {2, 4, 3};
    SafetensorsFile file;
};
CaseIV load_case_iv() {
    CaseIV c{{}, {}, {2, 4, 3}, golden("case_iv")};
    const std::size_t split = std::stoul(c.file.metadata().at("split"));
    c.state = tokens_of(c.file.at("state_ids"));
    for (std::size_t k = 0; k < 3; ++k) {
        const auto full = tokens_of(c.file.at("q" + std::to_string(k) + ".token_ids"));
        c.suffix.emplace_back(full.begin() + static_cast<std::ptrdiff_t>(split), full.end());
    }
    return c;
}

Outputs measure_all(bool half_state, const std::vector<Alert>& alerts, Range* range) {
    Outputs out;
    CudaEngine engine(real().model, options(half_state));
    const ModelConfig& cfg = real().model.config();
    const std::size_t H = hidden_size();
    std::cout << "  engine float16, state " << (half_state ? "binary16" : "float32") << ": slot " << engine.memory().slot_bytes / 1024.0 / 1024.0 << " MiB\n";
    const auto note_range = [&](const DeviceSnapshot& s) {
        if (range != nullptr) range->add(engine.to_host(s), cfg);
    };

    // 1. The six prompts: cases i-iii plain, case iv's three branches in the call of their state and from a stored state.
    for (const char* name : {"case_i", "case_ii", "case_iii"}) {
        const auto f = golden(name);
        const auto t = tokens_of(f.at("token_ids"));
        const auto r = run(engine, {row_of(t, {t.size() - 1})});
        out.plain.insert(out.plain.end(), r[0].hidden.begin(), r[0].hidden.end());
    }
    {
        const CaseIV c = load_case_iv();
        std::vector<EngineRow> rows = {row_of(c.state, {}, true)};
        for (std::size_t k = 0; k < 3; ++k) {
            EngineRow b = row_of(c.suffix[k], {c.suffix[k].size() - 1});
            b.parent = ParentRow{0};
            b.start_position = c.state.size();
            rows.push_back(b);
        }
        const auto r = run(engine, rows);
        for (std::size_t k = 1; k <= 3; ++k) out.case_iv_call.insert(out.case_iv_call.end(), r[k].hidden.begin(), r[k].hidden.end());
        note_range(r[0].snapshot);
        for (std::size_t k = 0; k < 3; ++k) {
            EngineRow b = row_of(c.suffix[k], {c.suffix[k].size() - 1});
            b.parent = r[0].snapshot;
            b.start_position = c.state.size();
            const auto g = run(engine, {b});
            out.case_iv_store.insert(out.case_iv_store.end(), g[0].hidden.begin(), g[0].hidden.end());
        }
    }

    // 2. The 512 prompts of the workload, every question run from a stored state (16 states per call, then 8 questions each).
    out.alerts.assign(alerts.size() * kQuestions * H, 0.0f);
    for (std::size_t first = 0; first < alerts.size(); first += 16) {
        const std::size_t count = std::min<std::size_t>(16, alerts.size() - first);
        std::vector<EngineRow> rows;
        for (std::size_t i = 0; i < count; ++i) rows.push_back(row_of(alerts[first + i].state, {}, true));
        const auto states = run(engine, rows);
        rows.clear();
        for (std::size_t i = 0; i < count; ++i) {
            for (std::size_t q = 0; q < kQuestions; ++q) {
                EngineRow r = row_of(alerts[first + i].suffixes[q], {alerts[first + i].suffixes[q].size() - 1});
                r.parent = states[i].snapshot;
                rows.push_back(std::move(r));
            }
        }
        const auto answers = run(engine, rows);
        for (std::size_t i = 0; i < count; ++i) {
            for (std::size_t q = 0; q < kQuestions; ++q) {
                const auto& h = answers[i * kQuestions + q].hidden;
                std::copy(h.begin(), h.end(), out.alerts.begin() + static_cast<std::ptrdiff_t>(((first + i) * kQuestions + q) * H));
            }
            if (first + i < 12) note_range(states[i].snapshot);
        }
    }

    // 3. A long row: the states of the first alerts joined to 8000 tokens, kept, then the questions of alert 0 continue it.
    {
        std::vector<std::uint32_t> big;
        for (std::size_t a = 1; a < alerts.size() && big.size() < 8000; ++a) big.insert(big.end(), alerts[a].state.begin(), alerts[a].state.end());
        big.resize(8000);
        const auto kept = run(engine, {row_of(big, {}, true)});
        note_range(kept[0].snapshot);
        std::vector<EngineRow> rows;
        for (std::size_t q = 0; q < kQuestions; ++q) {
            EngineRow r = row_of(alerts[0].suffixes[q], {alerts[0].suffixes[q].size() - 1});
            r.parent = kept[0].snapshot;
            rows.push_back(std::move(r));
        }
        for (const auto& r : run(engine, rows)) out.long_row.insert(out.long_row.end(), r.hidden.begin(), r.hidden.end());
    }

    // 4. A deep chain: alert 0's state, then 30 links of 64 tokens (text of other alerts), each continuing the stored snapshot of the
    // one before (the state is rounded at every store). After some links the questions of alert 0 continue the chain's snapshot.
    {
        DeviceSnapshot prev;
        std::size_t at = 0;
        const std::vector<std::size_t> probe_at = {1, 2, 5, 10, 20, 30};
        for (std::size_t link = 0; link <= 30; ++link) {
            EngineRow r;
            if (link == 0) {
                r = row_of(alerts[0].state, {}, true);
            } else {
                const Alert& a = alerts[1 + (link % (alerts.size() - 1))];
                const std::size_t begin = (link * 64) % (a.state.size() - 64);
                r = row_of(std::vector<std::uint32_t>(a.state.begin() + static_cast<std::ptrdiff_t>(begin), a.state.begin() + static_cast<std::ptrdiff_t>(begin + 64)), {}, true);
                r.parent = prev;
            }
            prev = run(engine, {r})[0].snapshot;
            at = link;
            if (std::find(probe_at.begin(), probe_at.end(), link) != probe_at.end()) {
                std::vector<EngineRow> rows;
                for (std::size_t q = 0; q < kQuestions; ++q) {
                    EngineRow p = row_of(alerts[0].suffixes[q], {alerts[0].suffixes[q].size() - 1});
                    p.parent = prev;
                    rows.push_back(std::move(p));
                }
                std::vector<float> hs;
                for (const auto& x : run(engine, rows)) hs.insert(hs.end(), x.hidden.begin(), x.hidden.end());
                out.chain.push_back(std::move(hs));
                out.chain_depths.push_back(link);
            }
        }
        note_range(prev);
        (void)at;
    }
    out.saturations = engine.state_saturations();
    return out;
}

} // namespace

SEPAL_TEST(real_half_state, the_accuracy_cost_of_binary16_snapshot_state_on_the_real_model) {
    if (skipped_half()) return;
    const std::vector<Alert> alerts = load_alerts(64);
    const ModelConfig& cfg = real().model.config();
    Range range;
    const Outputs full = measure_all(false, alerts, &range);
    const Outputs half = measure_all(true, alerts, nullptr);
    const std::size_t H = hidden_size();

    print_range("float32 snapshots: alert states 0-11, case iv, an 8000-token row, the end of a 30-link chain", range);
    std::cout << "  per DeltaNet layer, largest |recurrent|:";
    for (const double m : range.per_layer_recurrent) std::cout << " " << std::scientific << std::setprecision(2) << m;
    std::cout << std::fixed << "\n";
    CHECK(range.max_recurrent < 65504.0);
    CHECK(range.max_conv < 65504.0);
    CHECK_EQ(half.saturations, std::uint64_t{0});

    // The six prompts. i-iii have no stored parent: bit-identical. The three branches of iv see the rounding.
    CHECK(same_bits(full.plain, half.plain));
    {
        const CaseIV c = load_case_iv();
        for (const auto& [label, a, b] : {std::tuple<const char*, const std::vector<float>&, const std::vector<float>&>{"case iv branches in the call of their state", half.case_iv_call, full.case_iv_call},
                                          {"case iv branches from a stored state", half.case_iv_store, full.case_iv_store}}) {
            double worst_dp = 0, worst_dh = 0, worst_gold_half = 0, worst_gold_full = 0;
            for (std::size_t k = 0; k < 3; ++k) {
                const auto ph = real().readout.probabilities(std::span<const float>(a.data() + k * H, H), c.options[k]);
                const auto pf = real().readout.probabilities(std::span<const float>(b.data() + k * H, H), c.options[k]);
                const auto pg = c.file.at("q" + std::to_string(k) + ".probs").values<double>();
                for (std::size_t o = 0; o < c.options[k]; ++o) {
                    worst_dp = std::max(worst_dp, std::fabs(ph[o] - pf[o]));
                    worst_gold_half = std::max(worst_gold_half, std::fabs(ph[o] - pg[o]));
                    worst_gold_full = std::max(worst_gold_full, std::fabs(pf[o] - pg[o]));
                }
                worst_dh = std::max(worst_dh, max_abs_diff(a.data() + k * H, b.data() + k * H, H));
            }
            std::cout << "  [measure] " << label << ": binary16 vs float32 state: max |dp| " << std::scientific << std::setprecision(2) << worst_dp << ", max |dh| " << worst_dh
                      << "; against the float32 golden: binary16 state " << worst_gold_half << ", float32 state " << worst_gold_full << std::fixed << "\n";
            CHECK(worst_dp < 5e-3);
        }
    }

    // The workload.
    const SafetensorsFile gfile = SafetensorsFile::open(*environment("SEPAL_ALERT_GOLDEN"));
    const auto gv = gfile.at("final_hidden").values<float>();
    const std::vector<float> gold(gv.begin(), gv.begin() + static_cast<std::ptrdiff_t>(alerts.size() * kQuestions * H));
    const Cmp hv_full = compare(half.alerts, full.alerts);
    print("512 prompts from stored states: binary16 state vs float32 state", hv_full);
    print("512 prompts: float32 state vs transformers float32 golden", compare(full.alerts, gold));
    print("512 prompts: binary16 state vs transformers float32 golden", compare(half.alerts, gold));
    CHECK(hv_full.dp.max() < 5e-3);
    CHECK_EQ(hv_full.hard_flips, std::size_t{0});

    const Cmp lr = compare(half.long_row, full.long_row);
    print("8000-token row, 8 questions after it: binary16 vs float32 state", lr);
    CHECK(lr.dp.max() < 5e-2);
    for (std::size_t i = 0; i < full.chain.size(); ++i) {
        print("deep chain, after " + std::to_string(full.chain_depths[i]) + " links, 8 questions: binary16 vs float32 state", compare(half.chain[i], full.chain[i]));
        CHECK(all_finite(half.chain[i]));
    }
    (void)cfg;
}

// ---- tiers ----

namespace {

struct Timing {
    std::vector<double> ms;
    void add(double x) { ms.push_back(x); }
    double best() const { return *std::min_element(ms.begin(), ms.end()); }
    double median() const {
        std::vector<double> s = ms;
        std::sort(s.begin(), s.end());
        return s[s.size() / 2];
    }
};

void tiers_for(const char* label, const std::string& model_dir, const std::vector<Alert>& alerts, std::size_t budget_mib, std::size_t slots) {
    LoadedModel model = LoadedModel::open(model_dir);
    const ModelConfig& cfg = model.config();
    std::cout << "  ==== " << label << " (" << cfg.name << ", " << cfg.num_layers << " layers, hidden " << cfg.hidden_size << ") ====\n";
    const fs::path dir = fs::path("snapshot-test-files");
    fs::create_directories(dir);
    const fs::path file = dir / (std::string("tier-") + label + ".snap");
    const std::vector<std::uint32_t>& state = alerts[0].state;

    for (const bool half_state : {false, true}) {
        EngineOptions o = options(half_state, budget_mib, slots, 4096);
        CudaEngine engine(model, o);
        const EngineMemory m = engine.memory();
        // Bytes of one alert snapshot (state of 650-ish tokens): the slot and the keys and values.
        const auto kept = engine.run(std::span<const EngineRow>(std::vector<EngineRow>{row_of(state, {}, true)}));
        const DeviceSnapshot snap = kept[0].snapshot;
        const double slot_mib = static_cast<double>(m.slot_bytes) / 1048576.0;
        const double kv_mib = static_cast<double>(snap.own_bytes() - m.slot_bytes) / 1048576.0;
        const double total_mib = slot_mib + kv_mib;
        const double fixed_mib = static_cast<double>(m.weights_bytes + m.call_buffer_bytes) / 1048576.0;
        std::cout << "  [measure] " << label << " state " << (half_state ? "binary16" : "float32") << ": " << state.size() << "-token alert: slot " << std::fixed << std::setprecision(2) << slot_mib
                  << " MiB + keys/values " << kv_mib << " MiB = " << total_mib << " MiB per snapshot; weights " << m.weights_bytes / 1048576.0 << " MiB, call buffers (4096 tokens) "
                  << m.call_buffer_bytes / 1048576.0 << " MiB\n";
        const double card_mib = 12288.0, reserve_mib = 600.0; // 12 GB card, driver and display take about 600 MiB
        const double fit = std::floor((card_mib - reserve_mib - fixed_mib) / total_mib);
        std::cout << "  [measure] " << label << " " << (half_state ? "binary16" : "float32") << ": alerts of this size resident on a 12 GB card next to the weights: " << fit << "\n";
        // Slot overhead check: pool bytes per slot against the state itself.
        std::cout << "  [measure] pool bytes per slot " << static_cast<double>(m.slot_pool_bytes) / static_cast<double>(m.slots_total) / 1048576.0 << " MiB vs slot " << slot_mib << " MiB\n";

        // Reference answers from the original snapshot.
        const auto probe = [&](const DeviceSnapshot& s) {
            std::vector<EngineRow> rows;
            for (std::size_t q = 0; q < kQuestions; ++q) {
                EngineRow r = row_of(alerts[0].suffixes[q], {alerts[0].suffixes[q].size() - 1});
                r.parent = s;
                rows.push_back(std::move(r));
            }
            return engine.run(std::span<const EngineRow>(rows));
        };
        const auto want = probe(snap);
        const auto same = [&](const std::vector<EngineRowResult>& got) {
            for (std::size_t i = 0; i < got.size(); ++i) {
                if (!same_bits(got[i].hidden, want[i].hidden)) return false;
            }
            return true;
        };

        constexpr int kReps = 7;
        Timing recompute, host_pageable, host_pinned, to_dev, save, load_host, load;
        // Warm up, then take the best and the median of kReps.
        for (int i = 0; i < kReps + 1; ++i) {
            auto t = Clock::now();
            { auto r = engine.run(std::span<const EngineRow>(std::vector<EngineRow>{row_of(state, {}, true)})); }
            if (i > 0) recompute.add(ms_since(t));

            t = Clock::now();
            HostSnapshot hp = engine.to_host(snap, HostMemory::Pageable);
            if (i > 0) host_pageable.add(ms_since(t));

            t = Clock::now();
            HostSnapshot hq = engine.to_host(snap, HostMemory::Pinned);
            if (i > 0) host_pinned.add(ms_since(t));

            t = Clock::now();
            DeviceSnapshot back = engine.to_device(hp);
            if (i > 0) to_dev.add(ms_since(t));
            if (i == 1) CHECK(same(probe(back)));

            t = Clock::now();
            engine.save(snap, file);
            if (i > 0) save.add(ms_since(t));

            t = Clock::now();
            HostSnapshot lh = engine.load_host(file);
            if (i > 0) load_host.add(ms_since(t));

            t = Clock::now();
            DeviceSnapshot ld = engine.load(file);
            if (i > 0) load.add(ms_since(t));
            if (i == 1) CHECK(same(probe(ld)));
        }
        const auto line = [&](const char* what, const Timing& t) {
            std::cout << "  [measure] " << label << " " << (half_state ? "binary16" : "float32") << " " << std::left << std::setw(28) << what << std::right << std::fixed << std::setprecision(2) << "best "
                      << std::setw(8) << t.best() << " ms   median " << std::setw(8) << t.median() << " ms\n";
        };
        line("recompute the state", recompute);
        line("to_host (pageable)", host_pageable);
        line("to_host (pinned)", host_pinned);
        line("to_device", to_dev);
        line("save (device -> file)", save);
        line("load_host (file -> host)", load_host);
        line("load (file -> device)", load);
        std::cout << "  [measure] file size " << fs::file_size(file) / 1048576.0 << " MiB (host snapshot " << engine.to_host(snap).bytes() / 1048576.0 << " MiB)\n";
        fs::remove(file);
    }
}

} // namespace

SEPAL_TEST(real_tiers, bytes_per_snapshot_and_the_time_of_every_tier_against_recomputing) {
    if (skipped_tiers()) return;
    const std::vector<Alert> alerts = load_alerts(4);
    tiers_for("0.8B", *environment("SEPAL_REAL_MODEL"), alerts, 5000, 24);
    if (const auto m2 = environment("SEPAL_REAL_MODEL_2B")) {
        tiers_for("2B", *m2, alerts, 9000, 24);
    } else {
        std::cout << "  (SEPAL_REAL_MODEL_2B not set: the 2B is skipped)\n";
    }
}
