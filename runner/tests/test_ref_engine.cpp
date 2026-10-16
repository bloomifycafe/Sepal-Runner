// Suite `ref_engine`: the cpu-ref / cpu-ref64 DecisionEngine on the tiny fixture; suite `ref_engine_golden`: the same on
// the shortest cases of the golden set (needs SEPAL_TRAINED_FINAL, SEPAL_TRAINED_BASE, SEPAL_GOLDEN_DIR).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>

#include "head_case.hpp"
#include "sepal/engine.hpp"
#include "sepal/json.hpp"
#include "test.hpp"

#ifndef SEPAL_FIXTURE_DIR
#error "SEPAL_FIXTURE_DIR must be defined by the build"
#endif

using namespace sepal;
namespace fs = std::filesystem;
using sepal::test::env_value;

namespace {

EngineOptions tiny_options(const char* profile) {
    EngineOptions o;
    o.final_dir = std::string(SEPAL_FIXTURE_DIR) + "/tiny_trained";
    o.base_dir = std::string(SEPAL_FIXTURE_DIR) + "/tiny_checkpoint";
    o.profile = profile;
    return o;
}

bool same_bits(const RequestLogits& a, const RequestLogits& b) {
    if (a.logits.size() != b.logits.size()) return false;
    for (std::size_t q = 0; q < a.logits.size(); ++q) {
        if (a.logits[q].size() != b.logits[q].size()) return false;
        for (std::size_t k = 0; k < a.logits[q].size(); ++k) {
            if (a.logits[q][k] != b.logits[q][k]) return false;
        }
    }
    return true;
}

} // namespace

SEPAL_TEST(ref_engine, profiles_register_once) {
    // Other suites do not register; this is the only place register_ref_profiles runs in the test binary.
    register_ref_profiles();
    CHECK(find_engine_profile("cpu-ref") != nullptr);
    CHECK(find_engine_profile("cpu-ref64") != nullptr);
    CHECK(find_engine_profile("nope") == nullptr);
    CHECK_EQ(registered_profiles().size(), std::size_t{2});
    CHECK_THROWS(register_ref_profiles(), std::invalid_argument);
    CHECK_THROWS(register_profile({"", "x", {}, [](const EngineOptions&) { return std::unique_ptr<DecisionEngine>(); }}), std::invalid_argument);
    CHECK_THROWS(register_profile({"no-factory", "x", {}, {}}), std::invalid_argument);
}

SEPAL_TEST(ref_engine, scores_the_fixture_and_batches_invariantly) {
    const sepal::test::GoldenCase c = sepal::test::read_case(std::string(SEPAL_FIXTURE_DIR) + "/tiny_trained/head_case.safetensors");
    const auto engine = make_ref_engine(tiny_options("cpu-ref"), false);
    CHECK_EQ(engine->profile(), std::string("cpu-ref"));
    CHECK_EQ(engine->hidden_size(), std::uint64_t{24});
    EncodedRequest other = c.request;
    std::reverse(other.token_ids.begin(), other.token_ids.end());
    const EncodedRequest batch[] = {c.request, other};
    const std::vector<RequestLogits> together = engine->score(batch);
    CHECK_EQ(together.size(), std::size_t{2});
    CHECK_EQ(together[0].logits.size(), std::size_t{3});
    CHECK_EQ(together[0].logits[2].size(), std::size_t{3});
    for (const RequestLogits& r : together) {
        for (const auto& q : r.logits) {
            for (const float v : q) CHECK(std::isfinite(v));
        }
    }
    CHECK(engine->last_times().tokens == 2 * c.request.token_ids.size());
    CHECK(engine->last_times().total_ms > 0);
    const std::vector<RequestLogits> alone0 = engine->score(std::span<const EncodedRequest>(&batch[0], 1));
    const std::vector<RequestLogits> alone1 = engine->score(std::span<const EncodedRequest>(&batch[1], 1));
    CHECK(same_bits(together[0], alone0[0]));
    CHECK(same_bits(together[1], alone1[0]));
    CHECK(!same_bits(together[0], together[1]));

    // the float64 trunk agrees with the float32 trunk on a tiny model
    const auto engine64 = make_ref_engine(tiny_options("cpu-ref64"), true);
    CHECK_EQ(engine64->profile(), std::string("cpu-ref64"));
    const std::vector<RequestLogits> wide = engine64->score(std::span<const EncodedRequest>(&batch[0], 1));
    const sepal::test::Parity parity = sepal::test::compare_logits(alone0[0], [&] {
        std::vector<float> flat;
        for (const auto& q : wide[0].logits) flat.insert(flat.end(), q.begin(), q.end());
        return flat;
    }());
    std::cout << "  cpu-ref vs cpu-ref64 (tiny): max |dlogit| " << std::scientific << parity.max_dlogit << "\n";
    CHECK(parity.max_dlogit <= 1e-3);
}

SEPAL_TEST(ref_engine, refuses_bad_requests) {
    const sepal::test::GoldenCase c = sepal::test::read_case(std::string(SEPAL_FIXTURE_DIR) + "/tiny_trained/head_case.safetensors");
    EngineOptions options = tiny_options("cpu-ref");
    options.max_tokens = 10;
    const auto engine = make_ref_engine(options, false);
    const EncodedRequest too_long[] = {c.request};
    CHECK_THROWS(engine->score(too_long), std::invalid_argument);
    options.max_tokens = 100;
    const auto roomy = make_ref_engine(options, false);
    EncodedRequest bad = c.request;
    bad.token_ids[3] = 64; // outside the 64-token vocabulary
    const EncodedRequest bad_token[] = {bad};
    CHECK_THROWS(roomy->score(bad_token), std::invalid_argument);
    bad = c.request;
    bad.questions[0].options[0] = {5, 99};
    const EncodedRequest bad_span[] = {bad};
    CHECK_THROWS(roomy->score(bad_span), std::invalid_argument);
    options.max_tokens = 0;
    CHECK_THROWS(make_ref_engine(options, false), std::invalid_argument);
}

SEPAL_TEST(ref_engine_golden, shortest_cases) {
    const auto golden = env_value("SEPAL_GOLDEN_DIR");
    if (!golden || !sepal::test::real_model_available("ref_engine golden")) {
        if (!golden) std::cout << "SKIPPED: real-model ref_engine golden test; set SEPAL_GOLDEN_DIR (runs/runner-golden/full08b)\n";
        return;
    }
    std::ifstream in(fs::path(*golden) / "manifest.json", std::ios::binary);
    const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const json::Value manifest = json::parse(text);
    std::vector<std::pair<std::uint64_t, std::string>> cases;
    for (const json::Value& entry : manifest.at("cases").items()) {
        cases.emplace_back(entry.at("tokens").as_uint(), entry.at("file").as_string());
    }
    std::sort(cases.begin(), cases.end());
    std::size_t count = 3;
    if (const auto n = env_value("SEPAL_REF_CASES")) count = std::stoull(*n);
    cases.resize(std::min(count, cases.size()));
    // SEPAL_REF_ONLY=<case file name>: just that case (lets a driver run one process per case in parallel)
    if (const auto only = env_value("SEPAL_REF_ONLY")) {
        cases.clear();
        for (const json::Value& entry : manifest.at("cases").items()) {
            if (entry.at("file").as_string() == *only) cases.emplace_back(entry.at("tokens").as_uint(), *only);
        }
    }
    const bool fp64 = env_value("SEPAL_REF_FP64").has_value();

    EngineOptions options;
    options.final_dir = *env_value("SEPAL_TRAINED_FINAL");
    options.base_dir = *env_value("SEPAL_TRAINED_BASE");
    options.profile = fp64 ? "cpu-ref64" : "cpu-ref";
    const auto load_start = std::chrono::steady_clock::now();
    const auto engine = make_ref_engine(options, fp64);
    std::cout << "  " << engine->profile() << " loaded in " << std::fixed << std::setprecision(1)
              << std::chrono::duration<double>(std::chrono::steady_clock::now() - load_start).count() << " s\n";
    double worst_logit = 0, worst_p = 0;
    std::size_t flips = 0;
    for (const auto& [tokens, file] : cases) {
        const sepal::test::GoldenCase c = sepal::test::read_case((fs::path(*golden) / file).string());
        const EncodedRequest one[] = {c.request};
        const std::vector<RequestLogits> got = engine->score(one);
        const sepal::test::Parity parity = sepal::test::compare_logits(got[0], c.logits);
        std::cout << "  " << file << ": L=" << tokens << " max |dlogit| " << std::scientific << parity.max_dlogit << " max |dp| " << parity.max_dp
                  << " flips " << parity.flips << " (" << std::fixed << std::setprecision(1) << engine->last_times().total_ms / 1000.0
                  << " s, trunk " << engine->last_times().trunk_ms / 1000.0 << " s)\n";
        worst_logit = std::fmax(worst_logit, parity.max_dlogit);
        worst_p = std::fmax(worst_p, parity.max_dp);
        flips += parity.flips;
    }
    std::cout << "  " << engine->profile() << " vs golden: max |dlogit| " << std::scientific << worst_logit << ", max |dp| " << worst_p << ", flips " << flips << "\n";
    CHECK(!cases.empty());
    CHECK(worst_p <= 1e-4);
    CHECK_EQ(flips, std::size_t{0});
}
