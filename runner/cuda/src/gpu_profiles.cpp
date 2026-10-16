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
    }
    TrainedModel model = TrainedModel::open(o.final_dir, o.base_dir);
    return std::make_unique<cuda::GpuDecisionEngine>(model, g);
}

} // namespace

void register_backend_profiles() {
    const std::vector<std::string> keys = {"prefix_cache", "max_call_tokens", "trunk_budget_mib", "prefix_entries"};
    register_profile({"gpu-f16", "cuda", keys, [](const EngineOptions& o) { return make_gpu(o, "gpu-f16"); }});
    register_profile({"gpu-f32", "cuda", keys, [](const EngineOptions& o) { return make_gpu(o, "gpu-f32"); }});
}

} // namespace sepal
