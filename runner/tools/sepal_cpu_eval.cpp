// Parity of the CPU profiles against the golden set (runs/runner-golden/<name>/): the trunk's hidden states against
// hidden_fp32, the CPU Clef head fed with the golden hidden states and with the CPU trunk's own, and the resulting
// probabilities (temperature 1 and the calibrated temperatures of sepal.json) against softmax(logits_fp32).
//
//   sepal-cpu-eval --final <final dir> --base <base dir> --golden <golden dir>
//                  [--profiles cpu-f32,cpu-bf16,cpu-q8] [--q8 mlp,attn,dn_in] [--threads N] [--max-tokens N]
//                  [--delta-chunk N] [--head-oracle]
//
// --head-oracle also runs sepal::clef_head_reference (float64, single thread) on the golden hidden states.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "head_case.hpp"
#include "sepal/clef_head.hpp"
#include "sepal/cpu/clef_head.hpp"
#include "sepal/cpu/engine.hpp"
#include "sepal/cpu/trunk.hpp"
#include "sepal/json.hpp"
#include "sepal/trained_model.hpp"

using namespace sepal;
using Clock = std::chrono::steady_clock;

namespace {

std::string read_text(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

struct Temperatures {
    double single = 1.0;
    std::map<std::string, double> types, buckets;

    static Temperatures parse(const std::string& sepal_json) {
        const json::Value root = json::parse(sepal_json);
        Temperatures t;
        t.single = root.at("temperature").as_number();
        if (const json::Value* tt = root.find("temperatures")) {
            if (const json::Value* s = tt->find("single")) t.single = s->as_number();
            if (const json::Value* types = tt->find("types"))
                for (std::size_t i = 0; i < types->keys().size(); ++i) t.types[types->keys()[i]] = types->items()[i].as_number();
            if (const json::Value* buckets = tt->find("buckets"))
                for (std::size_t i = 0; i < buckets->keys().size(); ++i) t.buckets[buckets->keys()[i]] = buckets->items()[i].as_number();
        }
        return t;
    }
    double of(QuestionType type, std::size_t options) const {
        const char* name = type == QuestionType::Choice ? "choice" : type == QuestionType::Score ? "score" : "noul";
        const std::string bucket = options == 2 ? "2" : options <= 5 ? "3-5" : options <= 12 ? "6-12" : "13+";
        if (const auto it = buckets.find(std::string(name) + ":" + bucket); it != buckets.end()) return it->second;
        if (const auto it = types.find(name); it != types.end()) return it->second;
        return single;
    }
};

std::vector<double> softmax_t(const float* logits, std::size_t n, double temperature) {
    std::vector<double> p(n);
    double top = -1e300;
    for (std::size_t i = 0; i < n; ++i) top = std::max(top, static_cast<double>(logits[i]) / temperature);
    double sum = 0;
    for (std::size_t i = 0; i < n; ++i) {
        p[i] = std::exp(static_cast<double>(logits[i]) / temperature - top);
        sum += p[i];
    }
    for (double& v : p) v /= sum;
    return p;
}

struct Accumulator {
    std::vector<double> dp;  // |dp| per option
    std::size_t flips = 0, questions = 0;
    double max_dp() const { return dp.empty() ? 0 : *std::max_element(dp.begin(), dp.end()); }
    double p99() const {
        if (dp.empty()) return 0;
        std::vector<double> s = dp;
        std::sort(s.begin(), s.end());
        return s[static_cast<std::size_t>(0.99 * static_cast<double>(s.size() - 1))];
    }
    void add(const RequestLogits& got, const std::vector<float>& expected, const EncodedRequest& request, const Temperatures* temps) {
        std::size_t at = 0;
        for (std::size_t q = 0; q < got.logits.size(); ++q) {
            const std::size_t n = got.logits[q].size();
            const double t = temps != nullptr ? temps->of(request.questions[q].type, n) : 1.0;
            const std::vector<double> pg = softmax_t(got.logits[q].data(), n, t), pe = softmax_t(expected.data() + at, n, t);
            std::size_t bg = 0, be = 0;
            for (std::size_t k = 0; k < n; ++k) {
                dp.push_back(std::fabs(pg[k] - pe[k]));
                if (pg[k] > pg[bg]) bg = k;
                if (pe[k] > pe[be]) be = k;
            }
            if (bg != be) ++flips;
            ++questions;
            at += n;
        }
    }
};

void report(const char* label, const Accumulator& a) {
    std::printf("    %-34s max |dp| %.3e  p99 %.3e  flips %zu/%zu (%.2f%%)\n", label, a.max_dp(), a.p99(), a.flips, a.questions,
                a.questions == 0 ? 0.0 : 100.0 * static_cast<double>(a.flips) / static_cast<double>(a.questions));
}

} // namespace

int main(int argc, char** argv) {
    std::string final_dir, base_dir, golden_dir, profiles = "cpu-f32,cpu-bf16,cpu-q8", q8 = "mlp,attn,dn_in";
    unsigned threads = 0;
    std::uint64_t max_tokens = 100000;
    std::size_t delta_chunk = 64;
    bool oracle = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const auto value = [&]() -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument(a + " needs a value");
            return argv[++i];
        };
        if (a == "--final") final_dir = value();
        else if (a == "--base") base_dir = value();
        else if (a == "--golden") golden_dir = value();
        else if (a == "--profiles") profiles = value();
        else if (a == "--q8") q8 = value();
        else if (a == "--threads") threads = static_cast<unsigned>(std::stoul(value()));
        else if (a == "--max-tokens") max_tokens = std::stoull(value());
        else if (a == "--delta-chunk") delta_chunk = std::stoul(value());
        else if (a == "--head-oracle") oracle = true;
        else throw std::invalid_argument("unknown argument " + a);
    }
    if (final_dir.empty() || base_dir.empty() || golden_dir.empty()) {
        std::fprintf(stderr, "usage: sepal-cpu-eval --final DIR --base DIR --golden DIR [options]\n");
        return 2;
    }
    cpu::Q8Groups groups;
    groups.mlp = q8.find("mlp") != std::string::npos;
    groups.attention = q8.find("attn") != std::string::npos;
    groups.deltanet_in = q8.find("dn_in") != std::string::npos;

    const TrainedModel model = TrainedModel::open(final_dir, base_dir);
    const Temperatures temps = Temperatures::parse(model.sepal_json());
    const json::Value manifest = json::parse(read_text(golden_dir + "/manifest.json"));
    std::vector<std::pair<std::string, sepal::test::GoldenCase>> cases;
    for (const json::Value& entry : manifest.at("cases").items()) {
        if (entry.at("tokens").as_uint() > max_tokens) continue;
        cases.emplace_back(entry.at("name").as_string(), sepal::test::read_case(golden_dir + "/" + entry.at("file").as_string()));
    }
    std::printf("golden: %zu case(s) from %s\n", cases.size(), golden_dir.c_str());

    const Weights<float> merged = model.merged_weights<float>();
    const auto embed = [&model](std::uint32_t token, float* out) { model.embedding_row(token, out); };

    // The head alone, on the golden hidden states.
    {
        cpu::ThreadPool pool(threads);
        cpu::CpuClefHead head(model.head(), pool, cpu::best_isa());
        Accumulator t1, tc, ora1;
        for (const auto& [name, c] : cases) {
            const RequestLogits got = head.run(c.hidden, c.request, embed);
            t1.add(got, c.logits, c.request, nullptr);
            tc.add(got, c.logits, c.request, &temps);
            if (oracle) {
                const RequestLogits ref = clef_head_reference(
                    model.head(), c.hidden, c.request,
                    [](const void* ctx, std::uint32_t token, float* out) { static_cast<const TrainedModel*>(ctx)->embedding_row(token, out); }, &model);
                ora1.add(ref, c.logits, c.request, nullptr);
            }
        }
        std::printf("head on golden hidden states (cpu fp32 head):\n");
        report("T = 1", t1);
        report("calibrated temperatures", tc);
        if (oracle) report("clef_head_reference (fp64), T = 1", ora1);
    }

    std::stringstream ss(profiles);
    std::string profile;
    while (std::getline(ss, profile, ',')) {
        cpu::TrunkOptions opt;
        opt.profile = cpu::parse_profile(profile);
        opt.q8 = groups;
        opt.threads = threads;
        opt.delta_chunk = delta_chunk;
        cpu::CpuTrunk trunk(model.config(), merged, embed, opt);
        cpu::CpuClefHead head(model.head(), trunk.pool(), trunk.isa());
        std::printf("%s: isa %s, %u threads, weights %.2f GB, delta chunk %zu%s\n", profile.c_str(), cpu::name_of(trunk.isa()), trunk.threads(),
                    static_cast<double>(trunk.weight_bytes()) / 1e9, delta_chunk,
                    opt.profile == cpu::Profile::Q8 ? (" q8 groups: " + q8).c_str() : "");
        Accumulator t1, tc;
        double worst_rel = 0, worst_abs = 0, total_s = 0;
        std::size_t tokens = 0;
        for (const auto& [name, c] : cases) {
            std::vector<std::vector<float>> hidden;
            const std::span<const std::uint32_t> req(c.request.token_ids);
            const auto start = Clock::now();
            trunk.forward({&req, 1}, hidden);
            total_s += std::chrono::duration<double>(Clock::now() - start).count();
            tokens += req.size();
            double err = 0, norm = 0;
            for (std::size_t i = 0; i < hidden[0].size(); ++i) {
                const double d = static_cast<double>(hidden[0][i]) - c.hidden[i];
                err += d * d;
                norm += static_cast<double>(c.hidden[i]) * c.hidden[i];
                worst_abs = std::max(worst_abs, std::fabs(d));
            }
            const double rel = std::sqrt(err / norm);
            worst_rel = std::max(worst_rel, rel);
            const RequestLogits got = head.run(hidden[0], c.request, embed);
            Accumulator one;
            one.add(got, c.logits, c.request, nullptr);
            t1.add(got, c.logits, c.request, nullptr);
            tc.add(got, c.logits, c.request, &temps);
            std::printf("  %-22s L=%-5zu hidden rel err %.2e  max |dp| %.2e  flips %zu\n", name.c_str(), req.size(), rel, one.max_dp(), one.flips);
        }
        std::printf("  trunk hidden vs hidden_fp32: worst relative RMS error %.3e, worst abs %.3e; %zu tokens in %.2f s (%.0f tok/s)\n", worst_rel,
                    worst_abs, tokens, total_s, static_cast<double>(tokens) / total_s);
        report("T = 1", t1);
        report("calibrated temperatures", tc);
    }
    return 0;
}
