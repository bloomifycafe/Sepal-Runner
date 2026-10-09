// sepal-cuda-engine-ab: the engine's multiply options compared inside one process, rotating through them so that a slow
// moment of a shared GPU hits every option alike, and reporting the best rate of each option over the rounds.
//
//   sepal-cuda-engine-ab --model <dir> --workload <json> [--rounds 2] [--reps 5] [--gemm-rows N] [--sizes 64,600,2048]
//                        [--options cublas,cublas16,cublas16-mlp,own32,own16-64,own16-128,...]
//
// Rows: one row of N tokens (one readout), 16 rows of 640 tokens, and the alert shapes (a state with eight questions, one
// alert and sixteen alerts per call). Rates are tokens per second, alerts per second for the alert rows.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "sepal/cuda/cuda_engine.hpp"
#include "sepal/json.hpp"
#include "sepal/loader.hpp"

namespace {

using namespace sepal;
using cuda::CudaEngine;
using cuda::EngineOptions;
using cuda::EngineRow;
using cuda::MultiplyPath;
using cuda::ParentRow;

struct Alert {
    std::vector<std::uint32_t> state;
    std::vector<std::vector<std::uint32_t>> suffixes;
};

std::vector<Alert> load_workload(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::fprintf(stderr, "cannot open %s\n", path.c_str());
        std::exit(1);
    }
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
    return alerts;
}

std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> parts;
    std::stringstream ss(s);
    std::string part;
    while (std::getline(ss, part, sep)) {
        if (!part.empty()) parts.push_back(part);
    }
    return parts;
}

EngineOptions make_options(const std::string& name, std::size_t gemm_rows) {
    EngineOptions o;
    o.gemm_rows = gemm_rows;
    o.max_call_tokens = 16384;
    // name: cublas | cublas16 (all four binary16) | cublas16-mlp | own32 | own16-<flush> (all four) | own16-mlp-<flush>
    const auto flags = [&](bool gate_up, bool down, bool mixer_in, bool mixer_out) { o.fp16_accumulate = {gate_up, down, mixer_in, mixer_out, o.fp16_accumulate.flush_k}; };
    if (name == "cublas") {
    } else if (name == "cublas16") {
        flags(true, true, true, true);
    } else if (name == "cublas16-mlp") {
        flags(true, true, false, false);
    } else if (name == "own32") {
        o.multiply_path = MultiplyPath::Invariant;
    } else if (name.rfind("own16-mlp-", 0) == 0) {
        o.multiply_path = MultiplyPath::Invariant;
        o.fp16_accumulate.flush_k = static_cast<unsigned>(std::stoul(name.substr(10)));
        flags(true, true, false, false);
    } else if (name.rfind("own16-", 0) == 0) {
        o.multiply_path = MultiplyPath::Invariant;
        o.fp16_accumulate.flush_k = static_cast<unsigned>(std::stoul(name.substr(6)));
        flags(true, true, true, true);
    } else if (name.rfind("w8a8-", 0) == 0 || name.rfind("w8a16-", 0) == 0) {
        // w8a8-<groups> | w8a16-<groups>, groups joined by '+': gate_up, down, mixer_in, mixer_out, mlp, noout, all
        const bool a8 = name.rfind("w8a8-", 0) == 0;
        const std::string groups = name.substr(a8 ? 5 : 6);
        const auto has = [&](const char* g) { return groups.find(g) != std::string::npos; };
        o.int8_multiply.int8_activations = a8;
        o.int8_multiply.gate_up = has("gate_up") || has("mlp") || has("noout") || has("all");
        o.int8_multiply.down = has("down") || has("mlp") || has("noout") || has("all");
        o.int8_multiply.mixer_in = has("mixer_in") || has("noout") || has("all");
        o.int8_multiply.mixer_out = has("mixer_out") || has("all");
    } else {
        std::fprintf(stderr, "unknown option set %s\n", name.c_str());
        std::exit(1);
    }
    return o;
}

EngineRow plain_row(const std::vector<std::uint32_t>& tokens, std::size_t begin, std::size_t length) {
    EngineRow r;
    r.token_ids.assign(tokens.begin() + static_cast<std::ptrdiff_t>(begin), tokens.begin() + static_cast<std::ptrdiff_t>(begin + length));
    r.readout_offsets = {length - 1};
    return r;
}

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

// Best (shortest) time in ms of `reps` calls after two warm-up calls.
template <class Make>
double best_ms(CudaEngine& engine, int reps, Make&& make_rows) {
    double best = 1e300;
    for (int i = -2; i < reps; ++i) {
        const std::vector<EngineRow> rows = make_rows(static_cast<std::size_t>(i + 2));
        const auto begin = std::chrono::steady_clock::now();
        const auto results = engine.run(std::span<const EngineRow>(rows));
        const auto end = std::chrono::steady_clock::now();
        (void)results;
        if (i >= 0) best = std::min(best, std::chrono::duration<double, std::milli>(end - begin).count());
    }
    return best;
}

} // namespace

int main(int argc, char** argv) {
    std::string model, workload, options = "cublas,cublas16,own32,own16-64,own16-128";
    int rounds = 2, reps = 5;
    std::size_t gemm_rows = 0;
    std::vector<std::size_t> sizes = {64, 128, 600, 2048, 4800};
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string flag = argv[i], value = argv[i + 1];
        if (flag == "--model") model = value;
        else if (flag == "--workload") workload = value;
        else if (flag == "--rounds") rounds = std::stoi(value);
        else if (flag == "--reps") reps = std::stoi(value);
        else if (flag == "--gemm-rows") gemm_rows = std::stoul(value);
        else if (flag == "--options") options = value;
        else if (flag == "--sizes") {
            sizes.clear();
            for (const auto& s : split(value, ',')) sizes.push_back(std::stoul(s));
        } else {
            std::fprintf(stderr, "unknown option %s\n", flag.c_str());
            return 1;
        }
    }
    if (model.empty() || workload.empty()) {
        std::fprintf(stderr, "usage: sepal-cuda-engine-ab --model <dir> --workload <json> [--rounds N] [--reps N] [--gemm-rows N] [--sizes a,b] [--options a,b]\n");
        return 1;
    }
    try {
        const std::vector<Alert> alerts = load_workload(workload);
        const LoadedModel loaded = LoadedModel::open(model);
        std::vector<std::uint32_t> pool;
        for (std::size_t i = 0; pool.size() < 16 * 640 + 64; ++i) pool.insert(pool.end(), alerts[i % alerts.size()].state.begin(), alerts[i % alerts.size()].state.end());
        const std::vector<std::string> names = split(options, ',');
        // rate[row label][option] = best rate over the rounds
        std::vector<std::string> labels;
        std::map<std::string, std::map<std::string, double>> rate;
        std::map<std::string, std::size_t> weights_mib;
        const auto note = [&](const std::string& label, const std::string& name, double r) {
            if (std::find(labels.begin(), labels.end(), label) == labels.end()) labels.push_back(label);
            double& slot = rate[label][name];
            slot = std::max(slot, r);
        };
        for (int round = 0; round < rounds; ++round) {
            for (const std::string& name : names) {
                CudaEngine engine(loaded, make_options(name, gemm_rows));
                weights_mib[name] = engine.memory().weights_bytes / (1 << 20);
                for (const std::size_t len : sizes) {
                    const double ms = best_ms(engine, reps, [&](std::size_t rep) { return std::vector<EngineRow>{plain_row(pool, (rep * 37) % 64, len)}; });
                    note("1 row x " + std::to_string(len), name, static_cast<double>(len) / (ms / 1000.0));
                }
                {
                    const double ms = best_ms(engine, reps, [&](std::size_t rep) {
                        std::vector<EngineRow> rows;
                        for (std::size_t i = 0; i < 16; ++i) rows.push_back(plain_row(pool, i * 640 + (rep % 8), 640));
                        return rows;
                    });
                    note("16 rows x 640", name, 16.0 * 640.0 / (ms / 1000.0));
                }
                for (const std::size_t per_call : {std::size_t{1}, std::size_t{16}}) {
                    const double ms = best_ms(engine, reps, [&](std::size_t rep) {
                        std::vector<EngineRow> rows;
                        for (std::size_t a = 0; a < per_call; ++a) add_alert(rows, alerts[(rep * per_call + a) % alerts.size()]);
                        return rows;
                    });
                    note(std::to_string(per_call) + " alert(s) per call [alerts/s]", name, static_cast<double>(per_call) / (ms / 1000.0));
                }
            }
            std::fprintf(stderr, "round %d done\n", round);
        }
        std::printf("best of %d rounds x %d reps; multiply batching %s\n%-34s", rounds, reps, gemm_rows == 0 ? "natural" : ("strict " + std::to_string(gemm_rows)).c_str(), "");
        for (const auto& n : names) std::printf("%12s", n.c_str());
        std::printf("\n");
        for (const auto& label : labels) {
            std::printf("%-34s", label.c_str());
            for (const auto& n : names) std::printf("%12.1f", rate[label][n]);
            std::printf("\n");
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
    return 0;
}
