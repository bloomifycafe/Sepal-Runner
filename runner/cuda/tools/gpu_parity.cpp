// End-to-end parity of the GPU decision engine against the golden set (logits_fp32 of the fp32 oracle):
//   sepal-gpu-parity --final <dir> --base <dir> --golden <dir> [--profile gpu-f16|gpu-f32] [--max-tokens N] [--batch K]
//                    [--multiply cublas|invariant|auto] [--cases a,b,c]
// Prints, per case and overall, max |dlogit|, max and p99 |dp| at temperature 1 and at the calibrated temperatures of sepal.json,
// and the answer flips (argmax differs from the oracle's). Cases are scored K at a time (default 1).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <string>

#include "head_case.hpp"
#include "sepal/cuda/gpu_decision_engine.hpp"
#include "sepal/json.hpp"

using namespace sepal;
namespace fs = std::filesystem;

namespace {

struct Temps {
    double single = 1;
    std::map<std::string, double> types, buckets;
};

std::string read_text(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

Temps load_temps(const fs::path& sepal_json) {
    Temps t;
    const json::Value v = json::parse(read_text(sepal_json));
    t.single = v.at("temperature").as_number();
    if (const json::Value* tt = v.find("temperatures")) {
        t.single = tt->at("single").as_number();
        const json::Value& types = tt->at("types");
        for (std::size_t i = 0; i < types.keys().size(); ++i) t.types[types.keys()[i]] = types.items()[i].as_number();
        const json::Value& buckets = tt->at("buckets");
        for (std::size_t i = 0; i < buckets.keys().size(); ++i) t.buckets[buckets.keys()[i]] = buckets.items()[i].as_number();
    }
    return t;
}

const char* kind_name(QuestionType t) { return t == QuestionType::Noul ? "noul" : t == QuestionType::Choice ? "choice" : "score"; }

double temperature_for(const Temps& t, QuestionType type, std::size_t k) {
    const std::string kind = kind_name(type);
    const char* bucket = k <= 2 ? "2" : k <= 5 ? "3-5" : k <= 12 ? "6-12" : "13+";
    if (const auto it = t.buckets.find(kind + ":" + bucket); it != t.buckets.end()) return it->second;
    if (const auto it = t.types.find(kind); it != t.types.end()) return it->second;
    return t.single;
}

std::vector<double> softmax(const float* z, std::size_t n, double temperature) {
    double top = -1e300;
    for (std::size_t i = 0; i < n; ++i) top = std::fmax(top, static_cast<double>(z[i]));
    std::vector<double> p(n);
    double sum = 0;
    for (std::size_t i = 0; i < n; ++i) {
        p[i] = std::exp((static_cast<double>(z[i]) - top) / temperature);
        sum += p[i];
    }
    for (double& x : p) x /= sum;
    return p;
}

double percentile(std::vector<double> v, double q) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, static_cast<std::size_t>(q * static_cast<double>(v.size())))];
}

struct Accum {
    double max_dlogit = 0;
    std::vector<double> dp1, dpc;
    std::size_t flips1 = 0, flipsc = 0, questions = 0;
};

} // namespace

int main(int argc, char** argv) {
    std::string final_dir, base_dir, golden_dir, profile = "gpu-f16", multiply = "auto", only, int8_groups;
    bool int8_act = true, skip_without_env = false;
    std::string deltanet = "auto";
    std::uint64_t max_tokens = 40000;
    std::size_t batch = 1;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i], v = argv[i + 1];
        if (a == "--final") final_dir = v;
        else if (a == "--base") base_dir = v;
        else if (a == "--golden") golden_dir = v;
        else if (a == "--profile") profile = v;
        else if (a == "--max-tokens") max_tokens = std::stoull(v);
        else if (a == "--batch") batch = std::stoull(v);
        else if (a == "--multiply") multiply = v;
        else if (a == "--int8") int8_groups = v;
        else if (a == "--deltanet") deltanet = v;
        else if (a == "--int8-act") int8_act = v == "on";
        else if (a == "--cases") only = "," + v + ",";
        else if (a == "--skip-without-env") skip_without_env = v == "on";
        else { std::cerr << "unknown option " << a << "\n"; return 2; }
    }
    const auto from_env = [](const char* name) {
#if defined(_MSC_VER)
        char* buffer = nullptr;
        std::size_t length = 0;
        if (_dupenv_s(&buffer, &length, name) != 0 || buffer == nullptr) return std::string();
        const std::string value(buffer);
        std::free(buffer);
        return value;
#else
        const char* v = std::getenv(name);
        return v != nullptr ? std::string(v) : std::string();
#endif
    };
    if (final_dir.empty()) final_dir = from_env("SEPAL_TRAINED_FINAL");
    if (base_dir.empty()) base_dir = from_env("SEPAL_TRAINED_BASE");
    if (golden_dir.empty()) golden_dir = from_env("SEPAL_GOLDEN_DIR");
    if (skip_without_env && (final_dir.empty() || base_dir.empty() || golden_dir.empty())) {
        std::cout << "SKIPPED: real-model parity; set SEPAL_TRAINED_FINAL, SEPAL_TRAINED_BASE and SEPAL_GOLDEN_DIR\n";
        return 77;
    }
    if (final_dir.empty() || base_dir.empty() || golden_dir.empty()) {
        std::cerr << "usage: sepal-gpu-parity --final D --base D --golden D [--profile P] [--max-tokens N] [--batch K] [--multiply cublas|invariant|auto] [--cases a,b]\n";
        return 2;
    }
    try {
        const TrainedModel model = TrainedModel::open(final_dir, base_dir);
        const Temps temps = load_temps(fs::path(final_dir) / "sepal.json");
        cuda::GpuDecisionOptions options;
        options.profile = profile;
        if (multiply == "cublas") options.multiply_path = cuda::MultiplyPath::Cublas;
        else if (multiply == "invariant") options.multiply_path = cuda::MultiplyPath::Invariant;
        options.head.memory_tokens = 98304;
        options.int8.gate_up = int8_groups.find("gate_up") != std::string::npos;
        options.int8.down = int8_groups.find("down") != std::string::npos;
        options.int8.mixer_in = int8_groups.find("mixer_in") != std::string::npos;
        options.int8.mixer_out = int8_groups.find("mixer_out") != std::string::npos;
        options.int8.int8_activations = int8_act;
        if (deltanet == "scan") options.deltanet = cuda::DeltaNetMode::Scan;
        else if (deltanet == "chunked") options.deltanet = cuda::DeltaNetMode::Chunked;
        cuda::GpuDecisionEngine engine(model, options);
        const json::Value manifest = json::parse(read_text(fs::path(golden_dir) / "manifest.json"));
        std::vector<std::string> names;
        std::vector<sepal::test::GoldenCase> cases;
        for (const json::Value& entry : manifest.at("cases").items()) {
            if (entry.at("tokens").as_uint() > max_tokens) continue;
            const std::string name = entry.at("name").as_string();
            if (!only.empty() && only.find("," + name + ",") == std::string::npos) continue;
            names.push_back(name);
            cases.push_back(sepal::test::read_case((fs::path(golden_dir) / entry.at("file").as_string()).string()));
        }
        Accum all;
        std::cout << std::scientific << std::setprecision(2);
        for (std::size_t at = 0; at < cases.size(); at += batch) {
            const std::size_t n = std::min(batch, cases.size() - at);
            std::vector<EncodedRequest> reqs;
            for (std::size_t i = 0; i < n; ++i) reqs.push_back(cases[at + i].request);
            const auto t0 = std::chrono::steady_clock::now();
            const std::vector<RequestLogits> got = engine.score(reqs);
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            for (std::size_t i = 0; i < n; ++i) {
                const auto& c = cases[at + i];
                Accum one;
                std::size_t off = 0;
                for (std::size_t q = 0; q < got[i].logits.size(); ++q) {
                    const auto& z = got[i].logits[q];
                    const float* e = c.logits.data() + off;
                    const QuestionType type = c.request.questions[q].type;
                    for (std::size_t k = 0; k < z.size(); ++k) one.max_dlogit = std::fmax(one.max_dlogit, std::fabs(static_cast<double>(z[k]) - static_cast<double>(e[k])));
                    const double tc = temperature_for(temps, type, z.size());
                    for (int pass = 0; pass < 2; ++pass) {
                        const double temp = pass == 0 ? 1.0 : tc;
                        const auto pg = softmax(z.data(), z.size(), temp), pe = softmax(e, z.size(), temp);
                        std::size_t bg = 0, be = 0;
                        for (std::size_t k = 0; k < z.size(); ++k) {
                            (pass == 0 ? one.dp1 : one.dpc).push_back(std::fabs(pg[k] - pe[k]));
                            if (pg[k] > pg[bg]) bg = k;
                            if (pe[k] > pe[be]) be = k;
                        }
                        if (bg != be) ++(pass == 0 ? one.flips1 : one.flipsc);
                    }
                    ++one.questions;
                    off += z.size();
                }
                std::cout << "  " << std::left << std::setw(24) << names[at + i] << " L=" << std::setw(6) << c.request.token_ids.size() << " dlogit " << one.max_dlogit
                          << " | T=1 max dp " << *std::max_element(one.dp1.begin(), one.dp1.end()) << " flips " << one.flips1 << " | cal max dp "
                          << *std::max_element(one.dpc.begin(), one.dpc.end()) << " flips " << one.flipsc << (i == 0 ? "" : "") << "\n";
                all.max_dlogit = std::fmax(all.max_dlogit, one.max_dlogit);
                all.dp1.insert(all.dp1.end(), one.dp1.begin(), one.dp1.end());
                all.dpc.insert(all.dpc.end(), one.dpc.begin(), one.dpc.end());
                all.flips1 += one.flips1;
                all.flipsc += one.flipsc;
                all.questions += one.questions;
            }
            std::cout << std::fixed << std::setprecision(1) << "    (call of " << n << " case(s): " << ms << " ms)" << std::scientific << std::setprecision(2) << "\n";
        }
        std::cout << "profile " << profile << " (" << engine.profile() << "), " << cases.size() << " cases, " << all.questions << " questions, batch " << batch << "\n";
        std::cout << "  max |dlogit| " << all.max_dlogit << "\n";
        std::cout << "  T=1        : max |dp| " << *std::max_element(all.dp1.begin(), all.dp1.end()) << "  p99 " << percentile(all.dp1, 0.99) << "  flips " << all.flips1 << " ("
                  << std::fixed << std::setprecision(2) << 100.0 * static_cast<double>(all.flips1) / static_cast<double>(all.questions) << "%)\n" << std::scientific << std::setprecision(2);
        std::cout << "  calibrated : max |dp| " << *std::max_element(all.dpc.begin(), all.dpc.end()) << "  p99 " << percentile(all.dpc, 0.99) << "  flips " << all.flipsc << " ("
                  << std::fixed << std::setprecision(2) << 100.0 * static_cast<double>(all.flipsc) / static_cast<double>(all.questions) << "%)\n";
        // The gates of docs/RUNNER.md 14: float32 profiles 1e-4 and no flips; gpu-f16 0.03, p99 0.01, flips at most 0.5%; at T=1 and calibrated.
        const bool f32 = profile.rfind("gpu-f32", 0) == 0;
        const double questions = static_cast<double>(all.questions);
        const double worst_dp = std::fmax(*std::max_element(all.dp1.begin(), all.dp1.end()), *std::max_element(all.dpc.begin(), all.dpc.end()));
        const double worst_p99 = std::fmax(percentile(all.dp1, 0.99), percentile(all.dpc, 0.99));
        const std::size_t worst_flips = std::max(all.flips1, all.flipsc);
        const bool pass = f32 ? (worst_dp <= 1e-4 && worst_flips == 0)
                              : (worst_dp <= 0.03 && worst_p99 <= 0.01 && static_cast<double>(worst_flips) <= 0.005 * questions);
        std::cout << "  gate (" << (f32 ? "max dp <= 1e-4, no flips" : "max dp <= 0.03, p99 <= 0.01, flips <= 0.5%") << "): " << (pass ? "PASS" : "FAIL") << "\n";
        if (!pass) return 3;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
