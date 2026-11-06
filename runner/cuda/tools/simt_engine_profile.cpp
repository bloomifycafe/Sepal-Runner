// Where the time of the float32 SIMT engine goes on a real checkpoint: one row of T tokens, or a long request cut into calls of
// C tokens that continue each other's snapshots (as GpuDecisionEngine runs it), with the engine's phase profiling.
//
//   sepal-cuda-simt-profile --model <checkpoint dir> [--storage bf16|int8g|f32|f16] [--tokens 1024,2048] [--long 16384]
//                           [--call 2048] [--budget-mib 9000] [--reps 3]
//
// Prints per case: wall time, tokens/s, and the synchronised phase times (multiplies, DeltaNet, attention, elementwise, host).
// --storage f16 runs the binary16 engine (tensor cores) for comparison. Random token ids; the numbers are not checked here.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "sepal/cuda/cuda_engine.hpp"
#include "sepal/loader.hpp"

using namespace sepal;
using namespace sepal::cuda;

namespace {

std::vector<std::size_t> list_of(const std::string& s) {
    std::vector<std::size_t> v;
    std::stringstream ss(s);
    std::string part;
    while (std::getline(ss, part, ',')) v.push_back(std::stoull(part));
    return v;
}

void print(const char* what, std::size_t tokens, double wall_ms, const PhaseTimes& p) {
    std::printf("%-28s %7zu tok %9.1f ms %8.1f tok/s | mult %8.1f  dn %7.1f  attn %7.1f  elem %6.1f  host %6.1f ms\n", what, tokens, wall_ms,
                static_cast<double>(tokens) / wall_ms * 1e3, p.multiplies_ms, p.deltanet_ms, p.attention_ms, p.elementwise_ms, p.host_ms());
}

} // namespace

int main(int argc, char** argv) {
    std::string model_dir, storage = "bf16";
    std::vector<std::size_t> tokens = {1024, 2048};
    std::size_t long_tokens = 0, call = 2048, budget_mib = 9000;
    int reps = 3;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i], v = argv[i + 1];
        if (a == "--model") model_dir = v;
        else if (a == "--storage") storage = v;
        else if (a == "--tokens") tokens = list_of(v);
        else if (a == "--long") long_tokens = std::stoull(v);
        else if (a == "--call") call = std::stoull(v);
        else if (a == "--budget-mib") budget_mib = std::stoull(v);
        else if (a == "--reps") reps = std::atoi(v.c_str());
        else {
            std::fprintf(stderr, "unknown option %s\n", a.c_str());
            return 2;
        }
    }
    if (model_dir.empty()) {
        std::fprintf(stderr, "usage: sepal-cuda-simt-profile --model <checkpoint dir> [--storage bf16|int8g|f32|f16] [--tokens 1024,2048] [--long N] [--call C]\n");
        return 2;
    }
    const LoadedModel model = LoadedModel::open(model_dir);
    EngineOptions eo;
    eo.precision = storage == "f16" ? Precision::F16 : Precision::F32;
    eo.weight_storage = storage == "bf16" ? WeightStorage::BF16 : storage == "int8g" ? WeightStorage::Int8Group : WeightStorage::Float32;
    if (storage == "f32") eo.multiply_path = MultiplyPath::Invariant;
    eo.max_call_tokens = std::max(call, *std::max_element(tokens.begin(), tokens.end()));
    eo.memory_budget_bytes = budget_mib << 20;
    eo.state_slots = 8;
    eo.prune_last_layer = false;
    const auto t0 = std::chrono::steady_clock::now();
    CudaEngine engine(model, eo);
    const double load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const DeviceInfo info = engine.device_info();
    const EngineMemory mem = engine.memory();
    std::printf("%s sm_%d%d, storage %s, load %.1f s, weights %.2f GiB, call buffers %.2f GiB, kv heap %.2f GiB, slots %zu\n", info.name.c_str(),
                info.compute_major, info.compute_minor, storage.c_str(), load_s, static_cast<double>(mem.weights_bytes) / (1 << 30),
                static_cast<double>(mem.call_buffer_bytes) / (1 << 30), static_cast<double>(mem.kv_heap_bytes) / (1 << 30), mem.slots_total);
    std::mt19937 rng(5);
    std::uniform_int_distribution<std::uint32_t> tok(1000, 150000);
    const auto row_of = [&](std::size_t n) {
        EngineRow r;
        r.token_ids.resize(n);
        for (auto& t : r.token_ids) t = tok(rng);
        r.readout_offsets = {n - 1};
        return r;
    };
    for (const std::size_t n : tokens) {
        const std::vector<EngineRow> rows = {row_of(n)};
        engine.run(rows); // warm-up
        double best = 1e30;
        for (int r = 0; r < reps; ++r) {
            const auto s = std::chrono::steady_clock::now();
            engine.run(rows);
            best = std::min(best, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s).count());
        }
        engine.set_profiling(true);
        engine.run(rows);
        const PhaseTimes p = engine.last_profile();
        engine.set_profiling(false);
        print(("one row of " + std::to_string(n)).c_str(), n, best, p);
    }
    if (long_tokens != 0) {
        // One long request in calls of `call` tokens, each continuing the previous call's snapshot.
        PhaseTimes sum;
        double wall = 0;
        engine.set_profiling(true);
        DeviceSnapshot prev;
        for (std::size_t at = 0; at < long_tokens; at += call) {
            EngineRow r = row_of(std::min(call, long_tokens - at));
            if (prev.valid()) r.parent = prev;
            r.keep_snapshot = at + call < long_tokens;
            const auto s = std::chrono::steady_clock::now();
            auto res = engine.run(std::vector<EngineRow>{r});
            wall += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s).count();
            const PhaseTimes p = engine.last_profile();
            sum.multiplies_ms += p.multiplies_ms;
            sum.deltanet_ms += p.deltanet_ms;
            sum.attention_ms += p.attention_ms;
            sum.elementwise_ms += p.elementwise_ms;
            sum.wall_ms += p.wall_ms;
            prev = res[0].snapshot;
        }
        engine.set_profiling(false);
        print(("long request, calls of " + std::to_string(call)).c_str(), long_tokens, wall, sum);
        std::printf("  (phase times are synchronised: the wall time of this case includes the synchronisation)\n");
    }
    const EngineMemory m2 = engine.memory();
    std::printf("peak in use %.2f GiB (device total %.2f GiB)\n", static_cast<double>(m2.peak_in_use_bytes()) / (1 << 30), static_cast<double>(m2.device_total) / (1 << 30));
    return 0;
}
