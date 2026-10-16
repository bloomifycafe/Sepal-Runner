#include <algorithm>
#include <map>
#include <mutex>
#include <stdexcept>

#include "sepal/engine.hpp"

namespace sepal {

namespace {

struct Registry {
    std::mutex mutex;
    std::map<std::string, ProfileInfo, std::less<>> profiles;
};

Registry& registry() {
    static Registry r;
    return r;
}

} // namespace

void register_profile(ProfileInfo info) {
    if (info.name.empty()) throw std::invalid_argument("register_profile: empty profile name");
    if (!info.factory) throw std::invalid_argument("register_profile: profile '" + info.name + "' has no factory");
    Registry& r = registry();
    const std::lock_guard<std::mutex> lock(r.mutex);
    const std::string name = info.name;
    if (!r.profiles.emplace(name, std::move(info)).second) {
        throw std::invalid_argument("register_profile: profile '" + name + "' is already registered");
    }
}

const ProfileInfo* find_profile(std::string_view name) {
    Registry& r = registry();
    const std::lock_guard<std::mutex> lock(r.mutex);
    const auto it = r.profiles.find(name);
    return it == r.profiles.end() ? nullptr : &it->second;
}

std::vector<std::string> registered_profiles() {
    Registry& r = registry();
    const std::lock_guard<std::mutex> lock(r.mutex);
    std::vector<std::string> names;
    for (const auto& [name, info] : r.profiles) names.push_back(name);
    return names;
}

void register_ref_profiles() {
    register_profile({"cpu-ref", "cpu", {}, [](const EngineOptions& o) { return make_ref_engine(o, false); }});
    register_profile({"cpu-ref64", "cpu", {}, [](const EngineOptions& o) { return make_ref_engine(o, true); }});
}

} // namespace sepal
