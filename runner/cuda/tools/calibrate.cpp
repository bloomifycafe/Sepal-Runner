// sepal-cuda-calibrate: runs a calibration workload through the float16 engine and writes the per-channel statistics of the
// activations that feed the four multiply groups (cuda::ActivationStats) to a file, for Int8Smoothing.
//
//   sepal-cuda-calibrate --model <dir> --workload <json> --out <file> [--alerts N]
//
// The workload is the JSON of tools/make_alert_workload.py; it must not be the one the engine is judged on (README.md, "int8 multiplies").
// The run is deterministic: the same model, workload and card give the same file.

#include <chrono>
#include <cstdio>
#include <fstream>
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
using cuda::ParentRow;

} // namespace

int main(int argc, char** argv) {
    std::string model, workload, out;
    std::size_t limit = 0;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string flag = argv[i], value = argv[i + 1];
        if (flag == "--model") model = value;
        else if (flag == "--workload") workload = value;
        else if (flag == "--out") out = value;
        else if (flag == "--alerts") limit = std::stoul(value);
        else {
            std::fprintf(stderr, "unknown option %s\n", flag.c_str());
            return 1;
        }
    }
    if (model.empty() || workload.empty() || out.empty()) {
        std::fprintf(stderr, "usage: sepal-cuda-calibrate --model <dir> --workload <json> --out <file> [--alerts N]\n");
        return 1;
    }
    try {
        std::ifstream in(workload, std::ios::binary);
        if (!in) throw std::runtime_error("cannot open " + workload);
        std::stringstream buffer;
        buffer << in.rdbuf();
        const json::Value root = json::parse(buffer.str());
        const LoadedModel loaded = LoadedModel::open(model);
        EngineOptions o;
        o.max_call_tokens = 16384;
        o.max_call_rows = 256;
        o.collect_activation_stats = true;
        o.multiply_path = cuda::MultiplyPath::Invariant; // the activations of a row do not depend on how the calibration is batched
        CudaEngine engine(loaded, o);
        std::size_t alerts = 0;
        std::vector<EngineRow> rows;
        const auto flush = [&] {
            if (rows.empty()) return;
            (void)engine.run(std::span<const EngineRow>(rows));
            rows.clear();
        };
        std::size_t in_call = 0;
        for (const json::Value& a : root.at("alerts").items()) {
            if (limit != 0 && alerts == limit) break;
            const std::size_t state_index = rows.size();
            EngineRow state;
            for (const std::uint64_t id : a.at("state").as_uint_list()) state.token_ids.push_back(static_cast<std::uint32_t>(id));
            rows.push_back(std::move(state));
            const std::size_t state_len = rows.back().token_ids.size();
            for (const json::Value& s : a.at("suffixes").items()) {
                EngineRow q;
                for (const std::uint64_t id : s.as_uint_list()) q.token_ids.push_back(static_cast<std::uint32_t>(id));
                q.readout_offsets = {q.token_ids.size() - 1};
                q.parent = ParentRow{state_index};
                q.start_position = state_len;
                rows.push_back(std::move(q));
            }
            ++alerts;
            if (++in_call == 16) {
                flush();
                in_call = 0;
            }
        }
        flush();
        const cuda::ActivationStats stats = engine.activation_stats();
        cuda::save_activation_stats(stats, out);
        std::printf("%zu alerts, %llu tokens per layer, %zu layers -> %s\n", alerts, static_cast<unsigned long long>(stats.tokens[0][0]), stats.layers.size(), out.c_str());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
    return 0;
}
