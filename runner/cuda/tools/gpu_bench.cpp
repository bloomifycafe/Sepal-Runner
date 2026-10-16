// Throughput of the GPU decision engine over a workload file of runs/runner-bench/workloads (one JSON object per line with
// input_ids, question_types, instruction_spans, option_offsets, option_spans):
//   sepal-gpu-bench --final D --base D --workload F.jsonl [--profile gpu-f16] [--batch N] [--reps R] [--warmup W]
//                   [--multiply auto|cublas|invariant] [--prefix-cache on|off] [--out file.json] [--label text]
// --batch N scores N requests per call (0: all of them in one call). One repetition scores the whole file once; the report
// is the best repetition's requests/s and tokens/s, and the p50/p95 of the latency of a call (every request of a call
// waits for the whole call). Nothing else may use the GPU while this runs (take the lock).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include <cuda_runtime.h>

#include "sepal/cuda/gpu_decision_engine.hpp"
#include "sepal/json.hpp"

using namespace sepal;
using Clock = std::chrono::steady_clock;

namespace {

std::vector<EncodedRequest> load(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::vector<EncodedRequest> out;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        const json::Value v = json::parse(line);
        EncodedRequest r;
        for (const json::Value& t : v.at("input_ids").items()) r.token_ids.push_back(static_cast<std::uint32_t>(t.as_uint()));
        const auto& types = v.at("question_types").items();
        const auto& instr = v.at("instruction_spans").items();
        const auto& off = v.at("option_offsets").items();
        const auto& spans = v.at("option_spans").items();
        for (std::size_t q = 0; q < types.size(); ++q) {
            EncodedQuestion e;
            e.type = static_cast<QuestionType>(types[q].as_uint());
            e.instruction = {static_cast<std::uint32_t>(instr[q].at(0).as_uint()), static_cast<std::uint32_t>(instr[q].at(1).as_uint())};
            for (std::uint64_t k = off[q].as_uint(); k < off[q + 1].as_uint(); ++k)
                e.options.push_back({static_cast<std::uint32_t>(spans[k].at(0).as_uint()), static_cast<std::uint32_t>(spans[k].at(1).as_uint())});
            r.questions.push_back(std::move(e));
        }
        out.push_back(std::move(r));
    }
    return out;
}

double pct(std::vector<double> v, double q) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, static_cast<std::size_t>(q * static_cast<double>(v.size())))];
}

} // namespace

int main(int argc, char** argv) {
    std::string final_dir, base_dir, workload, profile = "gpu-f16", multiply = "auto", out_path, label, prefix = "on", dump_path, compare_path;
    bool profile_phases = false;
    std::size_t call_tokens = 0;
    std::size_t batch = 1, reps = 3, warmup = 1;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i], v = argv[i + 1];
        if (a == "--final") final_dir = v;
        else if (a == "--base") base_dir = v;
        else if (a == "--workload") workload = v;
        else if (a == "--profile") profile = v;
        else if (a == "--batch") batch = std::stoull(v);
        else if (a == "--reps") reps = std::stoull(v);
        else if (a == "--warmup") warmup = std::stoull(v);
        else if (a == "--multiply") multiply = v;
        else if (a == "--prefix-cache") prefix = v;
        else if (a == "--out") out_path = v;
        else if (a == "--dump") dump_path = v;
        else if (a == "--call-tokens") call_tokens = std::stoull(v);
        else if (a == "--phases") profile_phases = v == "on";
        else if (a == "--compare") compare_path = v;
        else if (a == "--label") label = v;
        else { std::cerr << "unknown option " << a << "\n"; return 2; }
    }
    if (final_dir.empty() || base_dir.empty() || workload.empty()) {
        std::cerr << "usage: sepal-gpu-bench --final D --base D --workload F [--profile P] [--batch N] [--reps R] [--warmup W] [--multiply M] [--out F]\n";
        return 2;
    }
    try {
        const TrainedModel model = TrainedModel::open(final_dir, base_dir);
        cuda::GpuDecisionOptions options;
        options.profile = profile;
        if (multiply == "cublas") options.multiply_path = cuda::MultiplyPath::Cublas;
        else if (multiply == "invariant") options.multiply_path = cuda::MultiplyPath::Invariant;
        options.prefix_cache = prefix == "on";
        if (call_tokens != 0) options.max_call_tokens = call_tokens;
        cuda::GpuDecisionEngine engine(model, options);
        if (profile_phases) engine.trunk().set_profiling(true);
        cuda::PhaseTimes ph;
        const std::vector<EncodedRequest> reqs = load(workload);
        std::uint64_t tokens = 0, questions = 0, option_count = 0;
        for (const EncodedRequest& r : reqs) {
            tokens += r.token_ids.size();
            questions += r.questions.size();
            for (const auto& q : r.questions) option_count += q.options.size();
        }
        const std::size_t per_call = batch == 0 ? reqs.size() : batch;
        double best_wall = 1e300, best_trunk = 0, best_head = 0;
        std::uint64_t reused = 0;
        std::vector<double> latencies;
        std::vector<float> flat;
        for (std::size_t rep = 0; rep < warmup + reps; ++rep) {
            std::vector<double> lat;
            flat.clear();
            const auto t0 = Clock::now();
            double trunk_ms = 0, head_ms = 0;
            std::uint64_t reused_now = 0;
            for (std::size_t at = 0; at < reqs.size(); at += per_call) {
                const std::size_t n = std::min(per_call, reqs.size() - at);
                const auto c0 = Clock::now();
                const auto got = engine.score(std::span<const EncodedRequest>(reqs.data() + at, n));
                lat.push_back(std::chrono::duration<double, std::milli>(Clock::now() - c0).count());
                if (profile_phases) {
                    const cuda::PhaseTimes q = engine.trunk().last_profile();
                    ph.multiplies_ms += q.multiplies_ms; ph.deltanet_ms += q.deltanet_ms; ph.attention_ms += q.attention_ms;
                    ph.elementwise_ms += q.elementwise_ms; ph.wall_ms += q.wall_ms; ph.tokens += q.tokens;
                }
                const ScoreTimes t = engine.last_times();
                trunk_ms += t.trunk_ms;
                head_ms += t.head_ms;
                reused_now += t.tokens_reused;
                if (got.size() != n) throw std::runtime_error("wrong number of answers");
                for (const RequestLogits& g : got)
                    for (const std::vector<float>& q : g.logits) flat.insert(flat.end(), q.begin(), q.end());
            }
            const double wall = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
            std::cout << "  rep " << rep << (rep < warmup ? " (warmup)" : "") << ": " << std::fixed << std::setprecision(1) << wall << " ms, "
                      << static_cast<double>(reqs.size()) / wall * 1000.0 << " requests/s, " << static_cast<double>(tokens) / wall * 1000.0 << " tokens/s\n";
            if (rep >= warmup && wall < best_wall) {
                best_wall = wall;
                best_trunk = trunk_ms;
                best_head = head_ms;
                latencies = lat;
                reused = reused_now;
            }
        }
        if (!dump_path.empty()) {
            std::ofstream d(dump_path, std::ios::binary);
            d.write(reinterpret_cast<const char*>(flat.data()), static_cast<std::streamsize>(flat.size() * sizeof(float)));
        }
        if (!compare_path.empty()) {
            std::ifstream c(compare_path, std::ios::binary);
            std::vector<float> ref(flat.size());
            c.read(reinterpret_cast<char*>(ref.data()), static_cast<std::streamsize>(ref.size() * sizeof(float)));
            if (!c || c.peek() != std::ifstream::traits_type::eof()) throw std::runtime_error("--compare file has a different size");
            double worst = 0;
            for (std::size_t i = 0; i < flat.size(); ++i) worst = std::max(worst, static_cast<double>(std::fabs(flat[i] - ref[i])));
            std::cout << "  compare: " << flat.size() << " logits, max |dlogit| vs " << compare_path << " = " << std::scientific << worst << std::fixed << "\n";
        }
        if (profile_phases)
            std::cout << "  phases (all reps, summed over trunk calls; synchronised): multiplies " << ph.multiplies_ms << " ms, deltanet " << ph.deltanet_ms
                      << " ms, attention " << ph.attention_ms << " ms, elementwise " << ph.elementwise_ms << " ms, host " << ph.host_ms() << " ms, wall " << ph.wall_ms << " ms\n";
        const cuda::EngineMemory mem = engine.trunk_memory();
        std::size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        const double rps = static_cast<double>(reqs.size()) / best_wall * 1000.0;
        const double tps = static_cast<double>(tokens) / best_wall * 1000.0;
        std::ostringstream js;
        js << std::fixed << std::setprecision(3);
        js << "{\"label\":\"" << label << "\",\"workload\":\"" << workload << "\",\"profile\":\"" << profile << "\",\"batch\":" << batch << ",\"requests\":" << reqs.size()
           << ",\"tokens\":" << tokens << ",\"questions\":" << questions << ",\"options\":" << option_count << ",\"best_wall_ms\":" << best_wall << ",\"requests_per_s\":" << rps
           << ",\"tokens_per_s\":" << tps << ",\"call_latency_p50_ms\":" << pct(latencies, 0.5) << ",\"call_latency_p95_ms\":" << pct(latencies, 0.95)
           << ",\"calls\":" << latencies.size() << ",\"tokens_reused\":" << reused << ",\"trunk_ms\":" << best_trunk << ",\"head_ms\":" << best_head
           << ",\"peak_in_use_mib\":" << static_cast<double>(mem.peak_in_use_bytes()) / 1048576.0 << ",\"reserved_mib\":" << static_cast<double>(mem.reserved_bytes()) / 1048576.0
           << ",\"device_used_mib\":" << static_cast<double>(total_b - free_b) / 1048576.0 << "}";
        std::cout << js.str() << "\n";
        if (!out_path.empty()) {
            std::ofstream out(out_path, std::ios::app);
            out << js.str() << "\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
