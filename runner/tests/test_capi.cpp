// The C ABI of sepal_runner_cpu, called through include/sepal/c_api.h only (sepal_core is linked just to read the
// fixture's request and to compare with the in-process engine). Own main: the library is a separate DLL, so its
// registry is not the test harness's.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "head_case.hpp"
#include "sepal/c_api.h"
#include "sepal/engine.hpp"

#ifndef SEPAL_FIXTURE_DIR
#error "SEPAL_FIXTURE_DIR must be defined by the build"
#endif

namespace {

int g_checks = 0, g_failures = 0;

void check(bool ok, const char* text, int line) {
    ++g_checks;
    if (ok) return;
    ++g_failures;
    std::fprintf(stderr, "test_capi.cpp:%d: FAILED: %s\n", line, text);
}
#define CHECK(cond) check(static_cast<bool>(cond), #cond, __LINE__)

const std::string final_dir = std::string(SEPAL_FIXTURE_DIR) + "/tiny_trained";
const std::string base_dir = std::string(SEPAL_FIXTURE_DIR) + "/tiny_checkpoint";

std::string options(const std::string& extra = "", const std::string& profile = "cpu-ref") {
    return "{\"final_dir\": \"" + final_dir + "\", \"base_dir\": \"" + base_dir + "\", \"profile\": \"" + profile + "\"" + extra + "}";
}

// A request set in the flat layout of sepal_score.
struct Flat {
    std::vector<uint32_t> tokens, types, instruction, spans;
    std::vector<uint64_t> token_offsets{0}, question_offsets{0}, option_offsets{0};
    std::size_t n_options = 0;
    void add(const sepal::EncodedRequest& r) {
        tokens.insert(tokens.end(), r.token_ids.begin(), r.token_ids.end());
        token_offsets.push_back(tokens.size());
        for (const sepal::EncodedQuestion& q : r.questions) {
            types.push_back(static_cast<uint32_t>(q.type));
            instruction.push_back(q.instruction.begin);
            instruction.push_back(q.instruction.end);
            for (const sepal::TokenSpan& s : q.options) {
                spans.push_back(s.begin);
                spans.push_back(s.end);
                ++n_options;
            }
            option_offsets.push_back(n_options);
        }
        question_offsets.push_back(types.size());
    }
    int score(sepal_engine* e, std::vector<float>& out, double* times = nullptr) const {
        out.assign(n_options, -1.0f);
        return sepal_score(e, token_offsets.size() - 1, tokens.data(), token_offsets.data(), question_offsets.data(), types.data(),
                           instruction.data(), option_offsets.data(), spans.data(), out.data(), times);
    }
};

bool last_error_contains(const char* what) { return std::strstr(sepal_last_error(), what) != nullptr; }

void test_open_errors() {
    sepal_engine* e = reinterpret_cast<sepal_engine*>(0x1);
    CHECK(sepal_open(nullptr, &e) != 0);
    CHECK(e == nullptr);
    CHECK(last_error_contains("NULL"));
    CHECK(sepal_open(options().c_str(), nullptr) != 0);
    CHECK(sepal_open("{not json", &e) != 0);
    CHECK(last_error_contains("JSON"));
    CHECK(sepal_open("[1]", &e) != 0);
    CHECK(last_error_contains("object"));
    CHECK(sepal_open("{}", &e) != 0);
    CHECK(last_error_contains("profile"));
    CHECK(sepal_open(options("", "no-such-profile").c_str(), &e) != 0);
    CHECK(last_error_contains("unknown profile"));
    CHECK(last_error_contains("cpu-ref")); // lists what exists
    CHECK(sepal_open(("{\"profile\": \"cpu-ref\", \"base_dir\": \"" + base_dir + "\"}").c_str(), &e) != 0);
    CHECK(last_error_contains("final_dir"));
    CHECK(sepal_open(options(", \"bogus\": 1").c_str(), &e) != 0);
    CHECK(last_error_contains("unknown key"));
    CHECK(last_error_contains("bogus"));
    CHECK(sepal_open(options(", \"threads\": \"many\"").c_str(), &e) != 0);
    CHECK(last_error_contains("threads"));
    CHECK(sepal_open(options(", \"max_tokens\": 0").c_str(), &e) != 0);
    CHECK(last_error_contains("max_tokens"));
    CHECK(sepal_open(options(", \"device\": -1").c_str(), &e) != 0);
    CHECK(sepal_open(("{\"final_dir\": \"/no/such/dir\", \"base_dir\": \"" + base_dir + "\", \"profile\": \"cpu-ref\"}").c_str(), &e) != 0);
    CHECK(std::strlen(sepal_last_error()) > 0);
    CHECK(e == nullptr);
    // a good call clears the error
    CHECK(sepal_open(options().c_str(), &e) == 0);
    CHECK(std::strlen(sepal_last_error()) == 0);
    sepal_close(e);
    sepal_close(nullptr); // allowed
    // the accepted common keys
    CHECK(sepal_open(options(", \"threads\": 2, \"max_tokens\": 100, \"device\": 0").c_str(), &e) == 0);
    sepal_close(e);
}

void test_info() {
    sepal_engine* e = nullptr;
    CHECK(sepal_open(options(", \"max_tokens\": 77").c_str(), &e) == 0);
    uint64_t needed = 0;
    CHECK(sepal_info(e, nullptr, 0, &needed) == 0); // size query
    CHECK(needed > 10);
    std::vector<char> buffer(static_cast<std::size_t>(needed));
    CHECK(sepal_info(e, buffer.data(), buffer.size(), &needed) == 0);
    const std::string text(buffer.data());
    CHECK(text.size() + 1 == needed);
    CHECK(text.find("\"profile\": \"cpu-ref\"") != std::string::npos);
    CHECK(text.find("\"max_tokens\": 77") != std::string::npos);
    CHECK(text.find("\"hidden_size\": 24") != std::string::npos);
    CHECK(text.find("\"backend\": \"cpu\"") != std::string::npos);
    char small[4];
    CHECK(sepal_info(e, small, sizeof small, &needed) != 0);
    CHECK(last_error_contains("needed"));
    CHECK(needed > sizeof small);
    CHECK(sepal_info(nullptr, buffer.data(), buffer.size(), &needed) != 0);
    CHECK(sepal_info(e, buffer.data(), buffer.size(), nullptr) != 0);
    sepal_close(e);
}

void test_score(const sepal::test::GoldenCase& c) {
    sepal_engine* e = nullptr;
    CHECK(sepal_open(options(", \"max_tokens\": 64").c_str(), &e) == 0);
    sepal::EncodedRequest reversed = c.request;
    std::reverse(reversed.token_ids.begin(), reversed.token_ids.end());

    // against the in-process engine, bit for bit
    sepal::EngineOptions in_process;
    in_process.final_dir = final_dir;
    in_process.base_dir = base_dir;
    in_process.profile = "cpu-ref";
    const auto engine = sepal::make_ref_engine(in_process, false);
    const sepal::EncodedRequest both[] = {c.request, reversed};
    const std::vector<sepal::RequestLogits> expected = engine->score(both);

    Flat flat;
    flat.add(c.request);
    flat.add(reversed);
    std::vector<float> got;
    double times[5] = {0, 0, 0, 0, 0};
    CHECK(flat.score(e, got, times) == 0);
    std::vector<float> want;
    for (const sepal::RequestLogits& r : expected)
        for (const auto& q : r.logits) want.insert(want.end(), q.begin(), q.end());
    CHECK(got.size() == want.size());
    bool same = got.size() == want.size();
    for (std::size_t i = 0; same && i < got.size(); ++i) same = got[i] == want[i];
    CHECK(same);
    CHECK(times[2] > 0 && times[3] == 2.0 * static_cast<double>(c.request.token_ids.size()) && times[4] == 0.0);
    CHECK(flat.score(e, got, nullptr) == 0); // times_out may be NULL

    // one handle from several threads gives the same answers
    std::vector<std::thread> threads;
    std::vector<int> ok(4, 0);
    for (std::size_t t = 0; t < ok.size(); ++t) {
        threads.emplace_back([&, t] {
            std::vector<float> mine;
            ok[t] = flat.score(e, mine) == 0 && mine == want;
        });
    }
    for (std::thread& t : threads) t.join();
    for (const int o : ok) CHECK(o == 1);

    // errors do not poison the handle, and the message is thread-local
    Flat bad = flat;
    bad.types[1] = 3;
    CHECK(bad.score(e, got) != 0);
    CHECK(last_error_contains("type"));
    std::string other_thread_error = "x";
    std::thread([&] { other_thread_error = sepal_last_error(); }).join();
    CHECK(other_thread_error.empty());

    bad = flat;
    bad.tokens[2] = 64; // outside the 64-token vocabulary: the engine refuses
    CHECK(bad.score(e, got) != 0);
    CHECK(last_error_contains("vocabulary"));

    bad = flat;
    bad.spans[1] = 999; // span end outside the request
    CHECK(bad.score(e, got) != 0);
    CHECK(last_error_contains("span"));

    bad = flat;
    bad.token_offsets[1] = bad.token_offsets[0]; // an empty request
    CHECK(bad.score(e, got) != 0);
    CHECK(last_error_contains("no tokens"));

    bad = flat;
    bad.question_offsets[2] = bad.question_offsets[1]; // a request without questions
    CHECK(bad.score(e, got) != 0);
    CHECK(last_error_contains("no questions"));

    bad = flat;
    bad.option_offsets[2] = bad.option_offsets[1]; // a question without options
    CHECK(bad.score(e, got) != 0);
    CHECK(last_error_contains("no options"));

    bad = flat;
    bad.token_offsets[0] = 1;
    CHECK(bad.score(e, got) != 0);

    CHECK(sepal_score(e, 0, flat.tokens.data(), flat.token_offsets.data(), flat.question_offsets.data(), flat.types.data(), flat.instruction.data(),
                      flat.option_offsets.data(), flat.spans.data(), got.data(), nullptr) != 0);
    CHECK(sepal_score(e, 2, nullptr, flat.token_offsets.data(), flat.question_offsets.data(), flat.types.data(), flat.instruction.data(),
                      flat.option_offsets.data(), flat.spans.data(), got.data(), nullptr) != 0);
    CHECK(last_error_contains("NULL"));
    CHECK(sepal_score(e, 2, flat.tokens.data(), nullptr, flat.question_offsets.data(), flat.types.data(), flat.instruction.data(),
                      flat.option_offsets.data(), flat.spans.data(), got.data(), nullptr) != 0);
    CHECK(sepal_score(nullptr, 2, flat.tokens.data(), flat.token_offsets.data(), flat.question_offsets.data(), flat.types.data(),
                      flat.instruction.data(), flat.option_offsets.data(), flat.spans.data(), got.data(), nullptr) != 0);
    CHECK(last_error_contains("engine"));

    // the max_tokens limit
    Flat longer;
    sepal::EncodedRequest big = c.request;
    big.token_ids.resize(65, 1);
    longer.add(big);
    CHECK(longer.score(e, got) != 0);
    CHECK(last_error_contains("limit"));

    // still healthy
    CHECK(flat.score(e, got) == 0);
    CHECK(got == want);
    sepal_close(e);
}

} // namespace

int main() {
    CHECK(sepal_api_version() == SEPAL_C_API_VERSION);
    CHECK(std::strlen(sepal_last_error()) == 0);
    const sepal::test::GoldenCase c = sepal::test::read_case(final_dir + "/head_case.safetensors");
    test_open_errors();
    test_info();
    test_score(c);
    std::printf("%d check(s), %d failed\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
