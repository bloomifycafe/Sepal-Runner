// Engine options and the profile registry behind the C ABI (c_api.h).
//
// Every backend library (sepal_runner_cpu, sepal_runner_cuda) is the same c_api.cpp plus its own engines. A
// backend makes its engines reachable by profile name by calling `register_profile` for each of them from its own
// `register_backend_profiles()`, which c_api.cpp calls exactly once, before the first `sepal_open`:
//
//     // my_backend_profiles.cpp (compiled into the backend library only)
//     #include "sepal/engine.hpp"
//     void sepal::register_backend_profiles() {
//         register_profile({"gpu-f16", "cuda", {"streams", "prefix_cache_mib"},
//                           [](const EngineOptions& o) -> std::unique_ptr<DecisionEngine> { return make_gpu_engine(o); }});
//     }
//
// The factory receives the parsed options: the common keys as fields, the backend's own keys (the ones named in
// `extra_keys`, no others are accepted) as parsed JSON values in `extra`. A factory throws on any problem; c_api.cpp
// turns the exception into a message for sepal_last_error().
#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "sepal/decision.hpp"
#include "sepal/json.hpp"

namespace sepal {

struct EngineOptions {
    std::string final_dir;             // required
    std::string base_dir;              // required
    std::string profile;               // required
    std::uint32_t threads = 0;         // 0: the backend's choice
    std::uint64_t max_tokens = 16384;  // longest sequence one request may have
    std::int32_t device = 0;           // GPU ordinal; ignored by CPU backends
    std::map<std::string, json::Value, std::less<>> extra; // backend-specific keys
};

using EngineFactory = std::function<std::unique_ptr<DecisionEngine>(const EngineOptions&)>;

struct ProfileInfo {
    std::string name;                    // "cpu-ref", "gpu-f16", ...
    std::string backend;                 // "cpu", "cuda"; reported by sepal_info
    std::vector<std::string> extra_keys; // option keys beyond the common ones that this profile accepts
    EngineFactory factory;
};

/// Adds a profile. Thread-safe. Throws std::invalid_argument for an empty name, a missing factory or a name taken.
void register_profile(ProfileInfo info);
/// The registered profile, or nullptr. The pointer stays valid for the life of the process.
const ProfileInfo* find_profile(std::string_view name);
/// All registered names, sorted.
std::vector<std::string> registered_profiles();

/// Defined by each backend library (not by sepal_core): registers that library's profiles.
void register_backend_profiles();

/// "cpu-ref" (fp32 trunk) and "cpu-ref64" (fp64 trunk): the C++ oracle on the sepal::ref executor. Registered by the
/// CPU library's register_backend_profiles(); tests may also call it directly (once per process).
void register_ref_profiles();

/// The reference engine itself: merged weights (float32 or float64) through sepal::ref, the final hidden state of every
/// position, then clef_head_reference. Slow by design.
std::unique_ptr<DecisionEngine> make_ref_engine(const EngineOptions& options, bool fp64);

} // namespace sepal
