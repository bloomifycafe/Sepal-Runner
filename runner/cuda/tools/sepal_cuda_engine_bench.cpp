// sepal-cuda-engine-bench: speed of the GPU engine on a real checkpoint.
//
//   sepal-cuda-engine-bench --model <checkpoint dir> --workload <alert_workload.json>
//       [--precision f16|f32] [--gemm-rows N] [--reps 7] [--warmup 2] [--json out.json]
//       [--sections prefill,alerts,cached,profile] [--max-call-tokens 16384] [--budget-mib N]
//
// The workload is the file tools/make_alert_workload.py writes. Sections:
//   prefill  tokens per second of one row of 64, 128, 300, 600, 1000, 2048 and 4800 tokens, and of 16 rows of 640;
//   alerts   alerts per second where an alert is one state row plus its 8 question rows (a readout at the last
//            token of each question), with 1, 2, 4, 8 and 16 alerts per call;
//   cached   the same with the states already stored as snapshots: a call carries only the 8 branches per alert;
//   profile  where the time of a call goes (multiplies, DeltaNet kernels, attention kernels, elementwise, host
//            overhead and launches), measured in a separate synchronised mode so that it does not disturb the rest.
// Every figure is the best and the median of the repeated timed runs. The device memory of the engine is reported
// at the end. Output is a table, and JSON with --json.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "sepal/cuda/cuda_engine.hpp"
#include "sepal/json.hpp"
#include "sepal/loader.hpp"

namespace {

using namespace sepal;
using cuda::CudaEngine;
using cuda::DeviceSnapshot;
using cuda::EngineOptions;
using cuda::EngineRow;
using cuda::ParentRow;
using cuda::Precision;

struct Options {
    std::string model;
    std::string workload;
    std::string json;
    Precision precision = Precision::F16;
    std::size_t gemm_rows = 0;
    int reps = 7;
    int warmup = 2;
    std::size_t max_call_tokens = 16384;
    std::size_t budget_mib = 0;
    std::size_t state_slots = 0;
    bool half_state = false;
    bool prune_last_layer = true;
    bool fp16_gate_up = true;
    sepal::cuda::Fp16Accumulate fp16_accumulate;
    sepal::cuda::MultiplyPath multiply_path = sepal::cuda::MultiplyPath::Cublas;
    sepal::cuda::Int8Multiply int8_multiply;
    bool prefill = true, alerts = true, cached = true, profile = true;
};

[[noreturn]] void usage(const std::string& why) {
    std::fprintf(stderr,
                 "%s\nusage: sepal-cuda-engine-bench --model <dir> --workload <json> [--precision f16|f32] [--gemm-rows N] [--reps N]\n"
                 "       [--warmup N] [--json out.json] [--sections prefill,alerts,cached,profile] [--max-call-tokens N] [--budget-mib N]\n"
                 "       [--state-slots N] [--prune-last-layer 0|1] [--fp16-gate-up 0|1]\n",
                 why.c_str());
    std::exit(2);
}

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        if (i + 1 >= argc) usage("option " + flag + " needs a value");
        const std::string value = argv[++i];
        if (flag == "--model") {
            o.model = value;
        } else if (flag == "--workload") {
            o.workload = value;
        } else if (flag == "--json") {
            o.json = value;
        } else if (flag == "--precision") {
            if (value == "f16") {
                o.precision = Precision::F16;
            } else if (value == "f32") {
                o.precision = Precision::F32;
            } else {
                usage("unknown precision " + value);
            }
        } else if (flag == "--gemm-rows") {
            o.gemm_rows = std::stoul(value);
        } else if (flag == "--reps") {
            o.reps = std::stoi(value);
        } else if (flag == "--warmup") {
            o.warmup = std::stoi(value);
        } else if (flag == "--max-call-tokens") {
            o.max_call_tokens = std::stoul(value);
        } else if (flag == "--budget-mib") {
            o.budget_mib = std::stoul(value);
        } else if (flag == "--state-slots") {
            o.state_slots = std::stoul(value);
        } else if (flag == "--half-state") { // keep the DeltaNet state of snapshots in binary16
            o.half_state = value != "0";
        } else if (flag == "--fp16-gate-up") {
            o.fp16_gate_up = value != "0";
        } else if (flag == "--multiply-path") { // cublas (default) or invariant (the engine's own batch-invariant kernel)
            if (value != "cublas" && value != "invariant") usage("unknown multiply path " + value);
            o.multiply_path = value == "invariant" ? sepal::cuda::MultiplyPath::Invariant : sepal::cuda::MultiplyPath::Cublas;
        } else if (flag == "--fp16-flush-k") {
            o.fp16_accumulate.flush_k = static_cast<unsigned>(std::stoul(value));
        } else if (flag == "--int8") { // w8a8:<groups> or w8a16:<groups>, groups joined by '+' (gate_up, down, mixer_in, mixer_out, mlp, noout, all); none
            if (value != "none") {
                const std::size_t colon = value.find(':');
                const std::string kind = value.substr(0, colon), groups = colon == std::string::npos ? "all" : value.substr(colon + 1);
                if (kind != "w8a8" && kind != "w8a16") usage("--int8 takes w8a8:<groups>, w8a16:<groups> or none");
                const auto has = [&](const char* g) { return groups.find(g) != std::string::npos; };
                o.int8_multiply.int8_activations = kind == "w8a8";
                o.int8_multiply.gate_up = has("gate_up") || has("mlp") || has("noout") || has("all");
                o.int8_multiply.down = has("down") || has("mlp") || has("noout") || has("all");
                o.int8_multiply.mixer_in = has("mixer_in") || has("noout") || has("all");
                o.int8_multiply.mixer_out = has("mixer_out") || has("all");
            }
        } else if (flag == "--fp16-accumulate") { // comma list of gate_up,down,mixer_in,mixer_out, or all, or none
            o.fp16_accumulate.gate_up = value.find("gate_up") != std::string::npos || value == "all";
            o.fp16_accumulate.down = value.find("down") != std::string::npos || value == "all";
            o.fp16_accumulate.mixer_in = value.find("mixer_in") != std::string::npos || value == "all";
            o.fp16_accumulate.mixer_out = value.find("mixer_out") != std::string::npos || value == "all";
        } else if (flag == "--prune-last-layer") {
            o.prune_last_layer = value != "0";
        } else if (flag == "--sections") {
            o.prefill = o.alerts = o.cached = o.profile = false;
            std::stringstream ss(value);
            std::string part;
            while (std::getline(ss, part, ',')) {
                if (part == "prefill") o.prefill = true;
                else if (part == "alerts") o.alerts = true;
                else if (part == "cached") o.cached = true;
                else if (part == "profile") o.profile = true;
                else usage("unknown section " + part);
            }
        } else {
            usage("unknown option " + flag);
        }
    }
    if (o.model.empty() || o.workload.empty()) usage("--model and --workload are required");
    if (o.reps < 1) usage("--reps must be positive");
    return o;
}

struct Alert {
    std::vector<std::uint32_t> state;
    std::vector<std::vector<std::uint32_t>> suffixes;
};

std::vector<Alert> load_workload(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) usage("cannot open " + path);
    std::stringstream buffer;
    buffer << in.rdbuf();
    const json::Value root = json::parse(buffer.str());
    std::vector<Alert> alerts;
    for (const json::Value& a : root.at("alerts").items()) {
        Alert alert;
        for (const std::uint64_t id : a.at("state").as_uint_list()) alert.state.push_back(static_cast<std::uint32_t>(id));
        for (const json::Value& s : a.at("suffixes").items()) {
            std::vector<std::uint32_t> suffix;
            for (const std::uint64_t id : s.as_uint_list()) suffix.push_back(static_cast<std::uint32_t>(id));
            alert.suffixes.push_back(std::move(suffix));
        }
        alerts.push_back(std::move(alert));
    }
    if (alerts.empty()) usage("the workload has no alerts");
    return alerts;
}

struct Timing {
    double min_ms = 0, median_ms = 0, max_ms = 0;
};

Timing summarise(std::vector<double> ms) {
    std::sort(ms.begin(), ms.end());
    Timing t;
    t.min_ms = ms.front();
    t.max_ms = ms.back();
    const std::size_t mid = ms.size() / 2;
    t.median_ms = ms.size() % 2 == 1 ? ms[mid] : 0.5 * (ms[mid - 1] + ms[mid]);
    return t;
}

// Times `make_rows` + engine.run per repetition: rows are built before the clock starts; `after` (not timed) runs
// with the results (for example to release snapshots).
template <class Make>
Timing time_calls(CudaEngine& engine, int warmup, int reps, Make&& make_rows) {
    std::vector<double> ms;
    for (int i = -warmup; i < reps; ++i) {
        const std::vector<EngineRow> rows = make_rows(static_cast<std::size_t>(i + warmup));
        const auto begin = std::chrono::steady_clock::now();
        const auto results = engine.run(std::span<const EngineRow>(rows));
        const auto end = std::chrono::steady_clock::now();
        (void)results;
        if (i >= 0) ms.push_back(std::chrono::duration<double, std::milli>(end - begin).count());
    }
    return summarise(ms);
}

EngineRow plain_row(const std::vector<std::uint32_t>& tokens, std::size_t begin, std::size_t length, bool readout) {
    EngineRow r;
    r.token_ids.assign(tokens.begin() + static_cast<std::ptrdiff_t>(begin), tokens.begin() + static_cast<std::ptrdiff_t>(begin + length));
    if (readout) r.readout_offsets = {length - 1};
    return r;
}

// A pool of real tokens to cut rows of any length from: the workload's states one after another.
std::vector<std::uint32_t> token_pool(const std::vector<Alert>& alerts, std::size_t at_least) {
    std::vector<std::uint32_t> pool;
    for (std::size_t i = 0; pool.size() < at_least; ++i) pool.insert(pool.end(), alerts[i % alerts.size()].state.begin(), alerts[i % alerts.size()].state.end());
    return pool;
}

// One alert as rows appended to `rows`: the state row, then its questions as ParentRow children.
void add_alert(std::vector<EngineRow>& rows, const Alert& alert) {
    const std::size_t state_index = rows.size();
    EngineRow state;
    state.token_ids = alert.state;
    rows.push_back(std::move(state));
    for (const auto& suffix : alert.suffixes) {
        EngineRow q;
        q.token_ids = suffix;
        q.readout_offsets = {suffix.size() - 1};
        q.parent = ParentRow{state_index};
        q.start_position = alert.state.size();
        rows.push_back(std::move(q));
    }
}

std::size_t tokens_of(const std::vector<EngineRow>& rows) {
    std::size_t n = 0;
    for (const EngineRow& r : rows) n += r.token_ids.size();
    return n;
}

struct Result {
    std::string section, name;
    std::size_t tokens_per_call = 0, alerts_per_call = 0;
    Timing time;
    double best_rate = 0, median_rate = 0; // tokens/s or alerts/s
    const char* unit = "";
};

std::vector<Result> g_results;

void report(Result r, double work /* tokens or alerts per call */) {
    r.best_rate = work / (r.time.min_ms / 1000.0);
    r.median_rate = work / (r.time.median_ms / 1000.0);
    std::printf("  %-34s %6zu tokens/call  min %9.2f ms  median %9.2f ms   best %10.1f %s   median %10.1f %s\n", r.name.c_str(), r.tokens_per_call, r.time.min_ms,
                r.time.median_ms, r.best_rate, r.unit, r.median_rate, r.unit);
    std::fflush(stdout);
    g_results.push_back(std::move(r));
}

std::string json_escape(const std::string& s) {
    std::string out;
    for (const char c : s) {
        if (c == '"' || c == '\\') out += '\\';
        out += c;
    }
    return out;
}

struct ProfileRow {
    std::string name;
    std::size_t tokens = 0;
    cuda::PhaseTimes times;
};
std::vector<ProfileRow> g_profiles;

// Runs `rows` with profiling on a few times and keeps the run with the shortest wall time.
void profile_call(CudaEngine& engine, const std::string& name, const std::vector<EngineRow>& rows, int runs) {
    engine.set_profiling(true);
    cuda::PhaseTimes best;
    best.wall_ms = 1e300;
    for (int i = 0; i < runs + 1; ++i) {
        engine.run(std::span<const EngineRow>(rows));
        if (i > 0 && engine.last_profile().wall_ms < best.wall_ms) best = engine.last_profile();
    }
    engine.set_profiling(false);
    g_profiles.push_back({name, tokens_of(rows), best});
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options opt = parse(argc, argv);
        const std::vector<Alert> alerts = load_workload(opt.workload);
        const LoadedModel model = LoadedModel::open(opt.model);

        EngineOptions eo;
        eo.precision = opt.precision;
        eo.gemm_rows = opt.gemm_rows;
        eo.max_call_tokens = opt.max_call_tokens;
        eo.max_call_rows = 16 * 9 + 16;
        eo.memory_budget_bytes = opt.budget_mib << 20;
        eo.state_slots = opt.state_slots;
        eo.half_state = opt.half_state;
        eo.prune_last_layer = opt.prune_last_layer;
        eo.fp16_gate_up = opt.fp16_gate_up;
        eo.fp16_accumulate = opt.fp16_accumulate;
        eo.multiply_path = opt.multiply_path;
        eo.int8_multiply = opt.int8_multiply;
        const auto created = std::chrono::steady_clock::now();
        CudaEngine engine(model, eo);
        const double create_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - created).count();
        const cuda::DeviceInfo info = engine.device_info();
        const cuda::EngineMemory mem0 = engine.memory();
        const ModelConfig& cfg = model.config();

        const auto cublas_text = [&] {
            std::ostringstream s;
            s << info.cublas_version / 10000 << "." << (info.cublas_version / 100) % 100 << "." << info.cublas_version % 100;
            return s.str();
        };
        std::printf("ENVIRONMENT\n  GPU %s (compute %d.%d, %.1f GiB), driver supports CUDA %d.%d, runtime %d.%d, cuBLAS %s\n", info.name.c_str(), info.compute_major,
                    info.compute_minor, static_cast<double>(info.total_bytes) / (1 << 30), info.driver_version / 1000, info.driver_version % 1000 / 10,
                    info.runtime_version / 1000, info.runtime_version % 1000 / 10, cublas_text().c_str());
        std::printf("  model %s (%s): %llu layers, hidden %llu; precision %s; multiply batching %s; engine created in %.1f s\n", cfg.name.c_str(), opt.model.c_str(),
                    static_cast<unsigned long long>(cfg.num_layers), static_cast<unsigned long long>(cfg.hidden_size), opt.precision == Precision::F16 ? "float16" : "float32",
                    opt.gemm_rows == 0 ? "natural" : ("strict, gemm_rows " + std::to_string(opt.gemm_rows)).c_str(), create_s);
        std::printf("  last layer computed for the read-out tokens only: %s\n", opt.prune_last_layer ? "yes" : "no");
        std::printf("  gate and up of the MLP written as binary16 by the multiply: %s\n", opt.precision == Precision::F16 && opt.fp16_gate_up ? "yes" : "no");
        std::printf("  multiplies by %s", opt.multiply_path == sepal::cuda::MultiplyPath::Invariant ? "the engine's own batch-invariant kernel" : "cuBLAS");
        if (opt.multiply_path == sepal::cuda::MultiplyPath::Invariant) std::printf(" (binary16 flush every %u)", opt.fp16_accumulate.flush_k);
        if (opt.int8_multiply.any())
            std::printf("\n  int8 weights (%s): gate_up %s, down %s, mixer_in %s, mixer_out %s", opt.int8_multiply.int8_activations ? "int8 activations" : "binary16 activations",
                        opt.int8_multiply.gate_up ? "yes" : "no", opt.int8_multiply.down ? "yes" : "no", opt.int8_multiply.mixer_in ? "yes" : "no",
                        opt.int8_multiply.mixer_out ? "yes" : "no");
        std::printf("\n  multiplies accumulating in binary16: gate_up %s, down %s, mixer_in %s, mixer_out %s\n", opt.fp16_accumulate.gate_up ? "yes" : "no",
                    opt.fp16_accumulate.down ? "yes" : "no", opt.fp16_accumulate.mixer_in ? "yes" : "no", opt.fp16_accumulate.mixer_out ? "yes" : "no");
        std::printf("  memory: weights %.0f MiB, call buffers %.0f MiB, %zu state slots of %.1f MiB, key/value heap %.0f MiB (budget %.0f MiB)\n",
                    static_cast<double>(mem0.weights_bytes) / (1 << 20), static_cast<double>(mem0.call_buffer_bytes) / (1 << 20), mem0.slots_total,
                    static_cast<double>(mem0.slot_bytes) / (1 << 20), static_cast<double>(mem0.kv_heap_bytes) / (1 << 20), static_cast<double>(mem0.budget_bytes) / (1 << 20));
        std::printf("  workload %s: %zu alerts, state %zu tokens (first), suffix %zu tokens (first)\n\n", opt.workload.c_str(), alerts.size(), alerts[0].state.size(),
                    alerts[0].suffixes[0].size());

        engine.reset_peaks();

        if (opt.prefill) {
            std::printf("PREFILL (rows without a parent; one readout at the last token)\n");
            const std::vector<std::uint32_t> pool = token_pool(alerts, 16 * 640 + 64);
            for (const std::size_t len : {64u, 128u, 300u, 600u, 1000u, 2048u, 4800u}) {
                if (len > opt.max_call_tokens) continue;
                Result r;
                r.section = "prefill";
                r.name = "1 row x " + std::to_string(len);
                r.tokens_per_call = len;
                r.unit = "tok/s";
                r.time = time_calls(engine, opt.warmup, opt.reps, [&](std::size_t rep) {
                    return std::vector<EngineRow>{plain_row(pool, (rep * 37) % 64, len, true)};
                });
                report(std::move(r), static_cast<double>(len));
            }
            if (16 * 640 <= opt.max_call_tokens) {
                Result r;
                r.section = "prefill";
                r.name = "16 rows x 640";
                r.tokens_per_call = 16 * 640;
                r.unit = "tok/s";
                r.time = time_calls(engine, opt.warmup, opt.reps, [&](std::size_t rep) {
                    std::vector<EngineRow> rows;
                    for (std::size_t i = 0; i < 16; ++i) rows.push_back(plain_row(pool, i * 640 + (rep % 8), 640, true));
                    return rows;
                });
                report(std::move(r), 16.0 * 640.0);
            }
            std::printf("\n");
        }

        const std::size_t per_call[] = {1, 2, 4, 8, 16};
        if (opt.alerts) {
            std::printf("ALERTS (one state row + 8 question rows per alert, all in the same call)\n");
            for (const std::size_t a : per_call) {
                std::size_t tokens = 0;
                {
                    std::vector<EngineRow> probe;
                    for (std::size_t i = 0; i < a; ++i) add_alert(probe, alerts[i]);
                    tokens = tokens_of(probe);
                }
                if (tokens > opt.max_call_tokens) continue;
                Result r;
                r.section = "alerts";
                r.name = std::to_string(a) + " alert(s) per call";
                r.tokens_per_call = tokens;
                r.alerts_per_call = a;
                r.unit = "alerts/s";
                r.time = time_calls(engine, opt.warmup, opt.reps, [&](std::size_t rep) {
                    std::vector<EngineRow> rows;
                    for (std::size_t i = 0; i < a; ++i) add_alert(rows, alerts[(rep * a + i) % alerts.size()]);
                    return rows;
                });
                report(std::move(r), static_cast<double>(a));
            }
            std::printf("\n");
        }

        if (opt.cached) {
            std::printf("STATE CACHED (the states are stored snapshots; a call carries only the 8 question rows of each alert)\n");
            for (const std::size_t a : per_call) {
                // Store the states of the first `a` alerts (not timed), then time calls that only carry the questions.
                std::vector<EngineRow> state_rows;
                for (std::size_t i = 0; i < a; ++i) {
                    EngineRow s;
                    s.token_ids = alerts[i].state;
                    s.keep_snapshot = true;
                    state_rows.push_back(std::move(s));
                }
                auto stored = engine.run(std::span<const EngineRow>(state_rows));
                std::vector<EngineRow> probe;
                for (std::size_t i = 0; i < a; ++i) {
                    for (const auto& suffix : alerts[i].suffixes) {
                        EngineRow q;
                        q.token_ids = suffix;
                        q.readout_offsets = {suffix.size() - 1};
                        q.parent = stored[i].snapshot;
                        probe.push_back(std::move(q));
                    }
                }
                Result r;
                r.section = "cached";
                r.name = std::to_string(a) + " alert(s) per call, state cached";
                r.tokens_per_call = tokens_of(probe);
                r.alerts_per_call = a;
                r.unit = "alerts/s";
                r.time = time_calls(engine, opt.warmup, opt.reps, [&](std::size_t) { return probe; });
                report(std::move(r), static_cast<double>(a));
            }
            std::printf("\n");
        }

        if (opt.profile) {
            std::printf("TIME BREAKDOWN (synchronised profiling mode; the best of 5 runs; ms per call)\n");
            const std::vector<std::uint32_t> pool = token_pool(alerts, 16 * 640 + 64);
            profile_call(engine, "1 row x 64", {plain_row(pool, 0, 64, true)}, 5);
            profile_call(engine, "1 row x 600", {plain_row(pool, 0, 600, true)}, 5);
            profile_call(engine, "1 row x 2048", {plain_row(pool, 0, 2048, true)}, 5);
            {
                std::vector<EngineRow> rows;
                for (std::size_t i = 0; i < 16; ++i) rows.push_back(plain_row(pool, i * 640, 640, true));
                profile_call(engine, "16 rows x 640", rows, 5);
            }
            for (const std::size_t a : {std::size_t{1}, std::size_t{16}}) {
                std::vector<EngineRow> rows;
                for (std::size_t i = 0; i < a; ++i) add_alert(rows, alerts[i]);
                if (tokens_of(rows) <= opt.max_call_tokens) profile_call(engine, std::to_string(a) + " alert(s) per call", rows, 5);
            }
            {
                std::vector<EngineRow> state_rows;
                for (std::size_t i = 0; i < 16; ++i) {
                    EngineRow s;
                    s.token_ids = alerts[i].state;
                    s.keep_snapshot = true;
                    state_rows.push_back(std::move(s));
                }
                auto stored = engine.run(std::span<const EngineRow>(state_rows));
                std::vector<EngineRow> rows;
                for (std::size_t i = 0; i < 16; ++i) {
                    for (const auto& suffix : alerts[i].suffixes) {
                        EngineRow q;
                        q.token_ids = suffix;
                        q.readout_offsets = {suffix.size() - 1};
                        q.parent = stored[i].snapshot;
                        rows.push_back(std::move(q));
                    }
                }
                profile_call(engine, "16 alerts per call, state cached", rows, 5);
            }
            std::printf("  %-34s %7s | %10s %10s %10s %10s %10s | %9s\n", "", "tokens", "multiplies", "deltanet", "attention", "elementw.", "host+launch", "wall ms");
            for (const ProfileRow& p : g_profiles) {
                const cuda::PhaseTimes& t = p.times;
                std::printf("  %-34s %7zu | %6.2f %3.0f%% %6.2f %3.0f%% %6.2f %3.0f%% %6.2f %3.0f%% %6.2f %3.0f%% | %9.2f\n", p.name.c_str(), p.tokens, t.multiplies_ms,
                            100 * t.multiplies_ms / t.wall_ms, t.deltanet_ms, 100 * t.deltanet_ms / t.wall_ms, t.attention_ms, 100 * t.attention_ms / t.wall_ms,
                            t.elementwise_ms, 100 * t.elementwise_ms / t.wall_ms, t.host_ms(), 100 * t.host_ms() / t.wall_ms, t.wall_ms);
            }
            std::printf("\n");
        }

        const cuda::EngineMemory mem = engine.memory();
        std::printf("DEVICE MEMORY\n  reserved at creation %.0f MiB (weights %.0f, call buffers %.0f, state slots %.0f, key/value heap %.0f); peak in use %.0f MiB "
                    "(slots in use at most %zu of %zu, key/value heap at most %.0f MiB)\n  device total %.0f MiB, free at start %.0f MiB, lowest free %.0f MiB, so the process "
                    "took at most %.0f MiB\n",
                    static_cast<double>(mem.reserved_bytes()) / (1 << 20), static_cast<double>(mem.weights_bytes) / (1 << 20), static_cast<double>(mem.call_buffer_bytes) / (1 << 20),
                    static_cast<double>(mem.slot_pool_bytes) / (1 << 20), static_cast<double>(mem.kv_heap_bytes) / (1 << 20), static_cast<double>(mem.peak_in_use_bytes()) / (1 << 20),
                    mem.slots_peak, mem.slots_total, static_cast<double>(mem.kv_peak_bytes) / (1 << 20), static_cast<double>(mem.device_total) / (1 << 20),
                    static_cast<double>(mem.device_free_at_start) / (1 << 20), static_cast<double>(mem.device_free_min) / (1 << 20),
                    static_cast<double>(mem.device_free_at_start - mem.device_free_min) / (1 << 20));

        if (!opt.json.empty()) {
            std::ofstream out(opt.json);
            out << std::fixed << std::setprecision(3);
            out << "{\n \"environment\": {\"gpu\": \"" << json_escape(info.name) << "\", \"compute\": \"" << info.compute_major << "." << info.compute_minor
                << "\", \"driver_cuda\": " << info.driver_version << ", \"runtime_cuda\": " << info.runtime_version << ", \"cublas\": " << info.cublas_version
                << ", \"model\": \"" << json_escape(cfg.name) << "\", \"model_dir\": \"" << json_escape(opt.model) << "\", \"precision\": \""
                << (opt.precision == Precision::F16 ? "f16" : "f32") << "\", \"gemm_rows\": " << opt.gemm_rows << ", \"max_call_tokens\": " << opt.max_call_tokens
                << ", \"reps\": " << opt.reps << "},\n \"results\": [\n";
            for (std::size_t i = 0; i < g_results.size(); ++i) {
                const Result& r = g_results[i];
                out << "  {\"section\": \"" << r.section << "\", \"name\": \"" << json_escape(r.name) << "\", \"tokens_per_call\": " << r.tokens_per_call
                    << ", \"alerts_per_call\": " << r.alerts_per_call << ", \"min_ms\": " << r.time.min_ms << ", \"median_ms\": " << r.time.median_ms
                    << ", \"max_ms\": " << r.time.max_ms << ", \"best_rate\": " << r.best_rate << ", \"median_rate\": " << r.median_rate << ", \"unit\": \"" << r.unit << "\"}"
                    << (i + 1 < g_results.size() ? ",\n" : "\n");
            }
            out << " ],\n \"profile\": [\n";
            for (std::size_t i = 0; i < g_profiles.size(); ++i) {
                const ProfileRow& p = g_profiles[i];
                out << "  {\"name\": \"" << json_escape(p.name) << "\", \"tokens\": " << p.tokens << ", \"multiplies_ms\": " << p.times.multiplies_ms << ", \"deltanet_ms\": "
                    << p.times.deltanet_ms << ", \"attention_ms\": " << p.times.attention_ms << ", \"elementwise_ms\": " << p.times.elementwise_ms << ", \"host_ms\": "
                    << p.times.host_ms() << ", \"wall_ms\": " << p.times.wall_ms << "}" << (i + 1 < g_profiles.size() ? ",\n" : "\n");
            }
            out << " ],\n \"memory\": {\"budget_bytes\": " << mem.budget_bytes << ", \"reserved_bytes\": " << mem.reserved_bytes() << ", \"weights_bytes\": " << mem.weights_bytes
                << ", \"call_buffer_bytes\": " << mem.call_buffer_bytes << ", \"slot_pool_bytes\": " << mem.slot_pool_bytes << ", \"kv_heap_bytes\": " << mem.kv_heap_bytes
                << ", \"peak_in_use_bytes\": " << mem.peak_in_use_bytes() << ", \"slots_peak\": " << mem.slots_peak << ", \"slots_total\": " << mem.slots_total
                << ", \"kv_peak_bytes\": " << mem.kv_peak_bytes << ", \"device_total\": " << mem.device_total << ", \"device_free_at_start\": " << mem.device_free_at_start
                << ", \"device_free_min\": " << mem.device_free_min << "}\n}\n";
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
