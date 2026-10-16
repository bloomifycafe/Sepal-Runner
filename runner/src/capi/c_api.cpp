// The C ABI (include/sepal/c_api.h) over DecisionEngine and the profile registry (include/sepal/engine.hpp).
// One implementation for every backend library: a backend adds its own `register_backend_profiles()` and links this file.
// No exception crosses the ABI: every entry point catches everything and reports through sepal_last_error().
#include "sepal/c_api.h"

#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "sepal/decision.hpp"
#include "sepal/engine.hpp"
#include "sepal/json.hpp"

struct sepal_engine {
    std::unique_ptr<sepal::DecisionEngine> engine;
    std::string backend;
    std::mutex mutex; // calls on one handle are serialised
};

namespace {

thread_local std::string g_last_error;

int fail(const std::string& message) {
    g_last_error = message;
    return 1;
}

int fail_current() {
    try {
        throw;
    } catch (const std::exception& e) {
        return fail(e.what());
    } catch (...) {
        return fail("unknown error");
    }
}

std::once_flag g_register_once;
std::string g_register_error;

void ensure_registered() {
    std::call_once(g_register_once, [] {
        try {
            sepal::register_backend_profiles();
        } catch (const std::exception& e) {
            g_register_error = e.what();
        } catch (...) {
            g_register_error = "unknown error";
        }
    });
    if (!g_register_error.empty()) throw std::runtime_error("registering the backend's profiles failed: " + g_register_error);
}

std::string need_string(const sepal::json::Value& options, const char* key) {
    const sepal::json::Value* v = options.find(key);
    if (v == nullptr) throw std::invalid_argument(std::string("options: \"") + key + "\" is required");
    if (v->kind() != sepal::json::Kind::String || v->as_string().empty()) {
        throw std::invalid_argument(std::string("options: \"") + key + "\" must be a non-empty string");
    }
    return v->as_string();
}

std::uint64_t uint_option(const sepal::json::Value& v, const char* key) {
    try {
        return v.as_uint();
    } catch (const std::runtime_error&) {
        throw std::invalid_argument(std::string("options: \"") + key + "\" must be a non-negative whole number");
    }
}

sepal::EngineOptions parse_options(const char* options_json, const sepal::ProfileInfo*& profile_out) {
    if (options_json == nullptr) throw std::invalid_argument("options_json is NULL");
    sepal::json::Value root;
    try {
        root = sepal::json::parse(options_json);
    } catch (const std::runtime_error& e) {
        throw std::invalid_argument(std::string("options_json is not valid JSON: ") + e.what());
    }
    if (root.kind() != sepal::json::Kind::Object) throw std::invalid_argument("options_json must be a JSON object");
    sepal::EngineOptions o;
    o.profile = need_string(root, "profile");
    const sepal::ProfileInfo* profile = sepal::find_engine_profile(o.profile);
    if (profile == nullptr) {
        std::string known;
        for (const std::string& n : sepal::registered_profiles()) known += (known.empty() ? "" : ", ") + n;
        throw std::invalid_argument("unknown profile '" + o.profile + "' (this library has: " + (known.empty() ? "none" : known) + ")");
    }
    o.final_dir = need_string(root, "final_dir");
    o.base_dir = need_string(root, "base_dir");
    const std::set<std::string> allowed(profile->extra_keys.begin(), profile->extra_keys.end());
    const std::vector<std::string>& keys = root.keys();
    const std::vector<sepal::json::Value>& values = root.items();
    for (std::size_t i = 0; i < keys.size(); ++i) {
        const std::string& key = keys[i];
        if (key == "profile" || key == "final_dir" || key == "base_dir") continue;
        if (key == "threads") {
            const std::uint64_t n = uint_option(values[i], "threads");
            if (n > 4096) throw std::invalid_argument("options: \"threads\" is implausibly large");
            o.threads = static_cast<std::uint32_t>(n);
        } else if (key == "max_tokens") {
            o.max_tokens = uint_option(values[i], "max_tokens");
            if (o.max_tokens == 0) throw std::invalid_argument("options: \"max_tokens\" must be positive");
        } else if (key == "device") {
            const std::uint64_t n = uint_option(values[i], "device");
            if (n > 1024) throw std::invalid_argument("options: \"device\" is implausibly large");
            o.device = static_cast<std::int32_t>(n);
        } else if (allowed.contains(key)) {
            o.extra.emplace(key, values[i]);
        } else {
            throw std::invalid_argument("options: unknown key \"" + key + "\" for profile '" + o.profile + "'");
        }
    }
    profile_out = profile;
    return o;
}

std::string escaped(const std::string& s) {
    std::string out;
    for (const char c : s) {
        if (c == '"' || c == '\\') {
            out += '\\';
            out += c;
        } else if (static_cast<unsigned char>(c) < 0x20) {
            out += ' ';
        } else {
            out += c;
        }
    }
    return out;
}

} // namespace

extern "C" {

SEPAL_API int32_t sepal_api_version(void) { return SEPAL_C_API_VERSION; }

SEPAL_API const char* sepal_last_error(void) { return g_last_error.c_str(); }

SEPAL_API int32_t sepal_open(const char* options_json, sepal_engine** out) {
    g_last_error.clear();
    try {
        if (out == nullptr) return fail("out is NULL");
        *out = nullptr;
        ensure_registered();
        const sepal::ProfileInfo* profile = nullptr;
        const sepal::EngineOptions options = parse_options(options_json, profile);
        std::unique_ptr<sepal::DecisionEngine> engine = profile->factory(options);
        if (!engine) return fail("the backend returned no engine for profile '" + options.profile + "'");
        auto handle = std::make_unique<sepal_engine>();
        handle->engine = std::move(engine);
        handle->backend = profile->backend;
        *out = handle.release();
        return 0;
    } catch (...) {
        return fail_current();
    }
}

SEPAL_API void sepal_close(sepal_engine* engine) {
    try {
        delete engine;
    } catch (...) {
    }
}

SEPAL_API int32_t sepal_info(sepal_engine* engine, char* buffer, uint64_t capacity, uint64_t* needed) {
    g_last_error.clear();
    try {
        if (engine == nullptr) return fail("engine is NULL");
        if (needed == nullptr) return fail("needed is NULL");
        std::string text;
        {
            const std::lock_guard<std::mutex> lock(engine->mutex);
            text = "{\"profile\": \"" + escaped(engine->engine->profile()) + "\", \"max_tokens\": " + std::to_string(engine->engine->max_tokens()) +
                   ", \"hidden_size\": " + std::to_string(engine->engine->hidden_size()) + ", \"backend\": \"" + escaped(engine->backend) +
                   "\", \"api_version\": " + std::to_string(SEPAL_C_API_VERSION) + "}";
        }
        *needed = text.size() + 1;
        if (buffer == nullptr && capacity == 0) return 0; // size query
        if (buffer == nullptr || capacity < text.size() + 1) {
            return fail("the info buffer holds " + std::to_string(capacity) + " bytes, " + std::to_string(text.size() + 1) + " are needed");
        }
        std::memcpy(buffer, text.c_str(), text.size() + 1);
        return 0;
    } catch (...) {
        return fail_current();
    }
}

SEPAL_API int32_t sepal_score(sepal_engine* engine, uint64_t n_requests, const uint32_t* token_ids, const uint64_t* token_offsets,
                              const uint64_t* question_offsets, const uint32_t* question_types, const uint32_t* instruction_spans,
                              const uint64_t* option_offsets, const uint32_t* option_spans, float* logits_out, double* times_out) {
    g_last_error.clear();
    try {
        if (engine == nullptr) return fail("engine is NULL");
        if (n_requests == 0) return fail("n_requests is 0");
        if (token_offsets == nullptr || question_offsets == nullptr || option_offsets == nullptr)
            return fail("an offsets array is NULL");
        if (token_offsets[0] != 0 || question_offsets[0] != 0) return fail("token_offsets[0] and question_offsets[0] must be 0");
        // Offsets: monotone, and the totals size the other arrays.
        for (std::uint64_t r = 0; r < n_requests; ++r) {
            if (token_offsets[r + 1] <= token_offsets[r]) return fail("request " + std::to_string(r) + " has no tokens (token_offsets must increase)");
            if (question_offsets[r + 1] <= question_offsets[r])
                return fail("request " + std::to_string(r) + " has no questions (question_offsets must increase)");
        }
        const std::uint64_t n_tokens = token_offsets[n_requests];
        const std::uint64_t n_questions = question_offsets[n_requests];
        if (token_ids == nullptr || question_types == nullptr || instruction_spans == nullptr || option_spans == nullptr || logits_out == nullptr)
            return fail("a data array is NULL");
        if (option_offsets[0] != 0) return fail("option_offsets[0] must be 0");
        for (std::uint64_t q = 0; q < n_questions; ++q) {
            if (option_offsets[q + 1] <= option_offsets[q]) return fail("question " + std::to_string(q) + " has no options (option_offsets must increase)");
        }
        const std::uint64_t n_options = option_offsets[n_questions];
        if (n_tokens > (std::uint64_t{1} << 40) || n_options > (std::uint64_t{1} << 32)) return fail("the request set is implausibly large");

        std::vector<sepal::EncodedRequest> requests(n_requests);
        for (std::uint64_t r = 0; r < n_requests; ++r) {
            sepal::EncodedRequest& request = requests[r];
            request.token_ids.assign(token_ids + token_offsets[r], token_ids + token_offsets[r + 1]);
            for (std::uint64_t q = question_offsets[r]; q < question_offsets[r + 1]; ++q) {
                if (question_types[q] > 2) return fail("question " + std::to_string(q) + " has type " + std::to_string(question_types[q]) + ", not 0, 1 or 2");
                sepal::EncodedQuestion question;
                question.type = static_cast<sepal::QuestionType>(question_types[q]);
                question.instruction = {instruction_spans[2 * q], instruction_spans[2 * q + 1]};
                for (std::uint64_t k = option_offsets[q]; k < option_offsets[q + 1]; ++k) {
                    question.options.push_back({option_spans[2 * k], option_spans[2 * k + 1]});
                }
                request.questions.push_back(std::move(question));
            }
        }

        std::vector<sepal::RequestLogits> results;
        sepal::ScoreTimes times;
        {
            const std::lock_guard<std::mutex> lock(engine->mutex);
            results = engine->engine->score(requests);
            times = engine->engine->last_times();
        }
        if (results.size() != n_requests) return fail("the engine returned the wrong number of results");
        std::uint64_t at = 0;
        for (std::uint64_t r = 0; r < n_requests; ++r) {
            if (results[r].logits.size() != requests[r].questions.size()) return fail("the engine returned the wrong number of questions");
            for (std::size_t q = 0; q < results[r].logits.size(); ++q) {
                if (results[r].logits[q].size() != requests[r].questions[q].options.size()) return fail("the engine returned the wrong number of options");
                for (const float v : results[r].logits[q]) logits_out[at++] = v;
            }
        }
        if (times_out != nullptr) {
            times_out[0] = times.trunk_ms;
            times_out[1] = times.head_ms;
            times_out[2] = times.total_ms;
            times_out[3] = static_cast<double>(times.tokens);
            times_out[4] = static_cast<double>(times.tokens_reused);
        }
        return 0;
    } catch (...) {
        return fail_current();
    }
}

} // extern "C"
