// The CUDA library's profiles (sepal_runner_cuda): gpu-f16 and gpu-f32 on GpuDecisionEngine.
#include "sepal/cuda/gpu_decision_engine.hpp"
#include "sepal/engine.hpp"

namespace sepal {

namespace {

std::unique_ptr<DecisionEngine> make_gpu(const EngineOptions& o, const char* profile) {
    if (o.max_tokens == 0) throw std::invalid_argument("max_tokens must be positive");
    cuda::GpuDecisionOptions g;
    g.profile = profile;
    g.device = o.device;
    g.max_tokens = o.max_tokens;
    for (const auto& [key, value] : o.extra) {
        if (key == "prefix_cache") g.prefix_cache = value.as_bool();
        else if (key == "max_call_tokens") g.max_call_tokens = static_cast<std::size_t>(value.as_uint());
        else if (key == "trunk_budget_mib") g.trunk_budget_bytes = static_cast<std::size_t>(value.as_uint()) << 20;
        else if (key == "prefix_entries") g.prefix_entries = static_cast<std::size_t>(value.as_uint());
        else if (key == "deltanet") {
            const std::string& mode = value.as_string();
            if (mode == "scan") g.deltanet = cuda::DeltaNetMode::Scan;
            else if (mode == "auto") g.deltanet = cuda::DeltaNetMode::Auto;
            else if (mode == "chunked") g.deltanet = cuda::DeltaNetMode::Chunked;
            else throw std::invalid_argument("deltanet must be scan, auto or chunked");
        }
    }
    if (g.profile == "gpu-q8w") { // int8 weights, binary16 activations, all four multiply groups (README.md, "int8 multiplies")
        g.int8.gate_up = g.int8.down = g.int8.mixer_in = g.int8.mixer_out = true;
        g.int8.int8_activations = false;
    }
    TrainedModel model = TrainedModel::open(o.final_dir, o.base_dir);
    return std::make_unique<cuda::GpuDecisionEngine>(model, g);
}

} // namespace

void register_backend_profiles() {
    const std::vector<std::string> keys = {"prefix_cache", "max_call_tokens", "trunk_budget_mib", "prefix_entries", "deltanet"};
    register_profile({"gpu-f16", "cuda", keys, [](const EngineOptions& o) { return make_gpu(o, "gpu-f16"); }});
    register_profile({"gpu-q8w", "cuda", keys, [](const EngineOptions& o) { return make_gpu(o, "gpu-q8w"); }});
    register_profile({"gpu-f16-strict", "cuda", keys, [](const EngineOptions& o) { return make_gpu(o, "gpu-f16-strict"); }});
    register_profile({"gpu-f32-strict", "cuda", keys, [](const EngineOptions& o) { return make_gpu(o, "gpu-f32-strict"); }});
    register_profile({"gpu-f32", "cuda", keys, [](const EngineOptions& o) { return make_gpu(o, "gpu-f32"); }});
}

} // namespace sepal
