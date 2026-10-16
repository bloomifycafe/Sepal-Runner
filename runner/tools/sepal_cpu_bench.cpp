// sepal-cpu-bench: throughput and latency of the CPU engine on an encoded workload (tools/runner_cpu_encode_workloads.py).
//
//   sepal-cpu-bench --final DIR --base DIR --input encoded.jsonl [--profile cpu-bf16] [--threads N] [--batch N]
//                   [--limit N] [--repeats N] [--warmup] [--max-tokens N] [--label text] [--q8 mlp,attn,dn_in]
//
// One JSON line on stdout per repeat: {"label", "profile", "threads", "isa", "batch", "requests", "tokens", "wall_s",
// "tokens_per_s", "requests_per_s", "call_ms_p50", "call_ms_max", "trunk_ms", "head_ms", "peak_rss_mb", "load_s"}.
// Call latency is the wall time of one score() call (a batch of `batch` requests); with --batch 1 it is the per-request latency.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "sepal/cpu/engine.hpp"
#include "sepal/json.hpp"

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#  include <psapi.h>
#endif

using namespace sepal;
using Clock = std::chrono::steady_clock;

namespace {

double peak_rss_mb() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc{};
    GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof pmc);
    return static_cast<double>(pmc.PeakWorkingSetSize) / 1048576.0;
#else
    std::ifstream in("/proc/self/status");
    std::string line;
    while (std::getline(in, line))
        if (line.rfind("VmHWM:", 0) == 0) return std::stod(line.substr(6)) / 1024.0;
    return 0.0;
#endif
}

EncodedRequest parse(const std::string& line) {
    const json::Value v = json::parse(line);
    EncodedRequest r;
    for (const json::Value& t : v.at("input_ids").items()) r.token_ids.push_back(static_cast<std::uint32_t>(t.as_uint()));
    const auto& types = v.at("question_types").items();
    const auto& ins = v.at("instruction_spans").items();
    const auto& off = v.at("option_offsets").items();
    const auto& spans = v.at("option_spans").items();
    for (std::size_t q = 0; q < types.size(); ++q) {
        EncodedQuestion e;
        e.type = static_cast<QuestionType>(types[q].as_uint());
        e.instruction = {static_cast<std::uint32_t>(ins[q].items()[0].as_uint()), static_cast<std::uint32_t>(ins[q].items()[1].as_uint())};
        for (std::uint64_t o = off[q].as_uint(); o < off[q + 1].as_uint(); ++o)
            e.options.push_back({static_cast<std::uint32_t>(spans[o].items()[0].as_uint()), static_cast<std::uint32_t>(spans[o].items()[1].as_uint())});
        r.questions.push_back(std::move(e));
    }
    return r;
}

} // namespace

int main(int argc, char** argv) {
    std::string final_dir, base_dir, input, profile = "cpu-bf16", label, q8 = "mlp,attn,dn_in";
    unsigned threads = 0;
    std::size_t batch = 1, limit = 0, repeats = 1;
    std::uint64_t max_tokens = 65536;
    bool warmup = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const auto value = [&]() -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument(a + " needs a value");
            return argv[++i];
        };
        if (a == "--final") final_dir = value();
        else if (a == "--base") base_dir = value();
        else if (a == "--input") input = value();
        else if (a == "--profile") profile = value();
        else if (a == "--threads") threads = static_cast<unsigned>(std::stoul(value()));
        else if (a == "--batch") batch = std::stoul(value());
        else if (a == "--limit") limit = std::stoul(value());
        else if (a == "--repeats") repeats = std::stoul(value());
        else if (a == "--max-tokens") max_tokens = std::stoull(value());
        else if (a == "--label") label = value();
        else if (a == "--q8") q8 = value();
        else if (a == "--warmup") warmup = true;
        else throw std::invalid_argument("unknown argument " + a);
    }
    if (final_dir.empty() || base_dir.empty() || input.empty() || batch == 0) {
        std::fprintf(stderr, "usage: sepal-cpu-bench --final DIR --base DIR --input FILE [options]\n");
        return 2;
    }
    std::vector<EncodedRequest> requests;
    {
        std::ifstream in(input);
        std::string line;
        while (std::getline(in, line) && (limit == 0 || requests.size() < limit))
            if (!line.empty()) requests.push_back(parse(line));
    }
    std::uint64_t total_tokens = 0;
    for (const auto& r : requests) total_tokens += r.token_ids.size();

    cpu::CpuEngineOptions opt;
    opt.profile = profile;
    opt.threads = threads;
    opt.max_tokens = max_tokens;
    opt.q8.mlp = q8.find("mlp") != std::string::npos;
    opt.q8.attention = q8.find("attn") != std::string::npos;
    opt.q8.deltanet_in = q8.find("dn_in") != std::string::npos;
    const auto load_start = Clock::now();
    const auto engine = cpu::make_cpu_engine(final_dir, base_dir, opt);
    const double load_s = std::chrono::duration<double>(Clock::now() - load_start).count();
    const unsigned used = threads != 0 ? threads : std::thread::hardware_concurrency();

    if (warmup) engine->score({requests.data(), std::min<std::size_t>(batch, requests.size())});
    for (std::size_t rep = 0; rep < repeats; ++rep) {
        std::vector<double> calls;
        double trunk = 0, head = 0;
        const auto start = Clock::now();
        for (std::size_t i = 0; i < requests.size(); i += batch) {
            const std::size_t n = std::min(batch, requests.size() - i);
            const auto t0 = Clock::now();
            engine->score({requests.data() + i, n});
            calls.push_back(std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
            trunk += engine->last_times().trunk_ms;
            head += engine->last_times().head_ms;
        }
        const double wall = std::chrono::duration<double>(Clock::now() - start).count();
        std::vector<double> sorted = calls;
        std::sort(sorted.begin(), sorted.end());
        std::printf("{\"label\": \"%s\", \"profile\": \"%s\", \"threads\": %u, \"batch\": %zu, \"requests\": %zu, \"tokens\": %llu, "
                    "\"wall_s\": %.4f, \"tokens_per_s\": %.1f, \"requests_per_s\": %.3f, \"call_ms_p50\": %.1f, \"call_ms_max\": %.1f, "
                    "\"trunk_ms\": %.1f, \"head_ms\": %.1f, \"peak_rss_mb\": %.0f, \"load_s\": %.2f}\n",
                    label.c_str(), profile.c_str(), used, batch, requests.size(), static_cast<unsigned long long>(total_tokens), wall,
                    static_cast<double>(total_tokens) / wall, static_cast<double>(requests.size()) / wall, sorted[sorted.size() / 2],
                    sorted.back(), trunk, head, peak_rss_mb(), load_s);
        std::fflush(stdout);
    }
    return 0;
}
