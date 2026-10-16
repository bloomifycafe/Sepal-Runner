// The CPU library's profiles (sepal_runner_cpu): the C++ reference engines and the fast engines.
//   cpu-f32, cpu-bf16, cpu-q8: packed-GEMM trunk + CPU Clef head (sepal/cpu/engine.hpp).
// Extra option keys: "chunk_tokens", "step_tokens", "delta_chunk", "group_tokens" (numbers) and, for cpu-q8, "q8"
// (a string naming the int8 groups: any of "mlp", "attn", "dn_in"; default all three).
#include "sepal/cpu/engine.hpp"
#include "sepal/engine.hpp"

namespace sepal {

namespace {

std::unique_ptr<DecisionEngine> make_fast(const std::string& profile, const EngineOptions& o) {
    cpu::CpuEngineOptions opt;
    opt.profile = profile;
    opt.threads = o.threads;
    opt.max_tokens = o.max_tokens;
    const auto number = [&](const char* key, std::size_t& out) {
        if (const auto it = o.extra.find(key); it != o.extra.end()) out = static_cast<std::size_t>(it->second.as_uint());
    };
    number("chunk_tokens", opt.chunk_tokens);
    number("step_tokens", opt.step_tokens);
    number("delta_chunk", opt.delta_chunk);
    number("group_tokens", opt.group_tokens);
    if (const auto it = o.extra.find("q8"); it != o.extra.end()) {
        const std::string& s = it->second.as_string();
        opt.q8.mlp = s.find("mlp") != std::string::npos;
        opt.q8.attention = s.find("attn") != std::string::npos;
        opt.q8.deltanet_in = s.find("dn_in") != std::string::npos;
    }
    return cpu::make_cpu_engine(o.final_dir, o.base_dir, opt);
}

} // namespace

void register_backend_profiles() {
    register_ref_profiles();
    const std::vector<std::string> keys = {"chunk_tokens", "step_tokens", "delta_chunk", "group_tokens"};
    for (const char* name : {"cpu-f32", "cpu-bf16"}) {
        const std::string profile = name;
        register_profile({profile, "cpu", keys, [profile](const EngineOptions& o) { return make_fast(profile, o); }});
    }
    std::vector<std::string> q8_keys = keys;
    q8_keys.push_back("q8");
    register_profile({"cpu-q8", "cpu", q8_keys, [](const EngineOptions& o) { return make_fast("cpu-q8", o); }});
}

} // namespace sepal
