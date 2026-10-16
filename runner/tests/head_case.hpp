// Shared by the `trained`, `clef` and `ref_engine` suites: the golden-set case layout (safetensors with input_ids,
// question_types, instruction_spans, option_offsets, option_spans, hidden_fp32, logits_fp32), environment lookups,
// and the real-model paths.
#pragma once

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "sepal/decision.hpp"
#include "sepal/safetensors.hpp"

namespace sepal::test {

inline std::optional<std::string> env_value(const char* name) {
#if defined(_MSC_VER)
    char* buffer = nullptr;
    std::size_t length = 0;
    if (_dupenv_s(&buffer, &length, name) != 0 || buffer == nullptr) return std::nullopt;
    const std::string value(buffer);
    std::free(buffer);
#else
    const char* raw = std::getenv(name);
    const std::string value = raw == nullptr ? std::string() : std::string(raw);
#endif
    if (value.empty()) return std::nullopt;
    return value;
}

/// SEPAL_TRAINED_FINAL / SEPAL_TRAINED_BASE both set (the trained model and its base snapshot).
inline bool real_model_available(const char* suite) {
    if (env_value("SEPAL_TRAINED_FINAL") && env_value("SEPAL_TRAINED_BASE")) return true;
    std::cout << "SKIPPED: real-model " << suite << " test; set SEPAL_TRAINED_FINAL (the final/ directory) and SEPAL_TRAINED_BASE (the base snapshot)\n";
    return false;
}

struct GoldenCase {
    EncodedRequest request;
    std::vector<float> hidden;  // [L, hidden]
    std::vector<float> logits;  // [P]
};

inline GoldenCase read_case(const std::string& path) {
    const SafetensorsFile file = SafetensorsFile::open(path);
    GoldenCase c;
    const auto ids = file.at("input_ids").values<std::int32_t>();
    for (const std::int32_t t : ids) c.request.token_ids.push_back(static_cast<std::uint32_t>(t));
    const auto types = file.at("question_types").values<std::int32_t>();
    const auto instr = file.at("instruction_spans").values<std::int32_t>();
    const auto offsets = file.at("option_offsets").values<std::int32_t>();
    const auto spans = file.at("option_spans").values<std::int32_t>();
    for (std::size_t q = 0; q < types.size(); ++q) {
        EncodedQuestion question;
        question.type = static_cast<QuestionType>(types[q]);
        question.instruction = {static_cast<std::uint32_t>(instr[2 * q]), static_cast<std::uint32_t>(instr[2 * q + 1])};
        for (std::int32_t k = offsets[q]; k < offsets[q + 1]; ++k) {
            const std::size_t i = static_cast<std::size_t>(k);
            question.options.push_back({static_cast<std::uint32_t>(spans[2 * i]), static_cast<std::uint32_t>(spans[2 * i + 1])});
        }
        c.request.questions.push_back(std::move(question));
    }
    const auto hidden = file.at("hidden_fp32").values<float>();
    c.hidden.assign(hidden.begin(), hidden.end());
    const auto logits = file.at("logits_fp32").values<float>();
    c.logits.assign(logits.begin(), logits.end());
    return c;
}

/// Softmax (float64) of one question's logits.
inline std::vector<double> softmax64(const float* logits, std::size_t n) {
    double top = -1e300;
    for (std::size_t i = 0; i < n; ++i) top = std::fmax(top, static_cast<double>(logits[i]));
    std::vector<double> p(n);
    double sum = 0;
    for (std::size_t i = 0; i < n; ++i) {
        p[i] = std::exp(static_cast<double>(logits[i]) - top);
        sum += p[i];
    }
    for (double& v : p) v /= sum;
    return p;
}

/// Largest |logit difference| and |probability difference| of `got` against the case's expected logits.
struct Parity {
    double max_dlogit = 0;
    double max_dp = 0;
    std::size_t flips = 0;
    std::size_t questions = 0;
};

inline Parity compare_logits(const RequestLogits& got, const std::vector<float>& expected) {
    Parity parity;
    std::size_t at = 0;
    for (const std::vector<float>& question : got.logits) {
        const std::vector<double> pg = softmax64(question.data(), question.size());
        const std::vector<double> pe = softmax64(expected.data() + at, question.size());
        std::size_t best_g = 0, best_e = 0;
        for (std::size_t k = 0; k < question.size(); ++k) {
            parity.max_dlogit = std::fmax(parity.max_dlogit, std::fabs(static_cast<double>(question[k]) - static_cast<double>(expected[at + k])));
            parity.max_dp = std::fmax(parity.max_dp, std::fabs(pg[k] - pe[k]));
            if (pg[k] > pg[best_g]) best_g = k;
            if (pe[k] > pe[best_e]) best_e = k;
        }
        if (best_g != best_e) ++parity.flips;
        ++parity.questions;
        at += question.size();
    }
    return parity;
}

} // namespace sepal::test
