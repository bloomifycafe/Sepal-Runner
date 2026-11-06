// Speed of the CPU causal attention kernels on one request prefilled in chunks, as the trunk runs it.
//
//   sepal-cpu-attn-bench [--tokens 16384] [--chunk 2048] [--heads 16] [--kv-heads 4] [--head-dim 256] [--threads 12]
//                        [--cpus 0-11] [--reps 3] [--kernels tiled,flash] [--q-scale 4] [--check-rows 24] [--isa avx2-tuned]
//
// Fills a K/V cache with random keys and values for `tokens` positions, then runs attention_forward chunk by chunk
// (queries [c, c + chunk) against keys [0, c + chunk)), every listed kernel in turn, repeat by repeat in a rotating order.
// Prints per kernel the median time of the whole prefill, its GFLOP/s (4 * head_dim per query head and visible key:
// Q K^T and P V), the speedup over the first kernel, the largest difference from the first kernel's output, and the
// largest error against a float64 softmax on `check-rows` sampled query rows (all heads).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

#include "sepal/cpu/attention.hpp"
#include "sepal/cpu/isa.hpp"
#include "sepal/cpu/thread_pool.hpp"

using namespace sepal::cpu;

namespace {

double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

struct Rng {
    std::uint64_t s;
    explicit Rng(std::uint64_t seed) : s(seed) {}
    float next() {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        return static_cast<float>(static_cast<std::int32_t>(s >> 33)) / 1073741824.0f - 1.0f;
    }
};

} // namespace

int main(int argc, char** argv) {
    std::size_t tokens = 16384, chunk = 2048, heads = 16, kv = 4, hd = 256, check_rows = 24;
    unsigned threads = 12;
    int reps = 3;
    double q_scale = 4.0;
    std::string cpus, kernels_arg = "tiled,flash";
    Isa isa = best_isa();
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i], v = argv[i + 1];
        if (a == "--tokens") tokens = std::stoul(v);
        else if (a == "--chunk") chunk = std::stoul(v);
        else if (a == "--heads") heads = std::stoul(v);
        else if (a == "--kv-heads") kv = std::stoul(v);
        else if (a == "--head-dim") hd = std::stoul(v);
        else if (a == "--threads") threads = static_cast<unsigned>(std::stoul(v));
        else if (a == "--cpus") cpus = v;
        else if (a == "--reps") reps = std::stoi(v);
        else if (a == "--kernels") kernels_arg = v;
        else if (a == "--q-scale") q_scale = std::stod(v);
        else if (a == "--check-rows") check_rows = std::stoul(v);
        else if (a == "--isa") isa = parse_isa(v);
    }
    require_isa(isa);
    std::vector<AttentionKernel> kernels;
    {
        std::stringstream ss(kernels_arg);
        std::string tok;
        while (std::getline(ss, tok, ',')) kernels.push_back(parse_attention_kernel(tok));
    }
    ThreadPool pool(threads, cpus);
    const std::size_t qw = heads * hd;
    Rng rng(7);
    KvCache cache(kv, hd, tokens, isa);
    std::vector<float> keys(tokens * kv * hd), vals(tokens * kv * hd);
    for (float& x : keys) x = rng.next();
    for (float& x : vals) x = rng.next();
    for (std::size_t p = 0; p < tokens; ++p)
        for (std::size_t g = 0; g < kv; ++g) cache.append(p, g, keys.data() + (p * kv + g) * hd, vals.data() + (p * kv + g) * hd);
    // Queries already scaled by 1 / sqrt(head_dim), times q_scale (peakier softmax: more rescaling).
    std::vector<float> q(tokens * qw), gate(tokens * 2 * qw);
    const float qs = static_cast<float>(q_scale / std::sqrt(static_cast<double>(hd)));
    for (float& x : q) x = rng.next() * qs;
    for (float& x : gate) x = rng.next() * 2.0f;
    double flops = 0.0;
    for (std::size_t p = 0; p < tokens; ++p) flops += 4.0 * static_cast<double>(hd * heads) * static_cast<double>(p + 1);

    const std::size_t nk = kernels.size();
    std::vector<std::vector<float>> outs(nk, std::vector<float>(tokens * qw, 0.0f));
    std::vector<std::vector<double>> times(nk);
    const auto prefill = [&](std::size_t k) {
        set_attention_kernel(kernels[k]);
        for (std::size_t c0 = 0; c0 < tokens; c0 += chunk) {
            AttentionSegment seg;
            seg.cache = &cache;
            seg.start = c0;
            seg.rows = std::min(chunk, tokens - c0);
            seg.q = q.data() + c0 * qw;
            seg.gate = gate.data() + c0 * 2 * qw + hd;
            seg.out = outs[k].data() + c0 * qw;
            attention_forward({heads, kv, hd}, isa, {&seg, 1}, qw, 2 * qw, 2 * hd, qw, pool);
        }
        set_attention_kernel(AttentionKernel::Default);
    };
    for (std::size_t k = 0; k < nk; ++k) prefill(k); // warm up
    for (int r = 0; r < reps; ++r) {
        for (std::size_t i = 0; i < nk; ++i) {
            const std::size_t k = (i + static_cast<std::size_t>(r)) % nk;
            const double t0 = now_s();
            prefill(k);
            times[k].push_back(now_s() - t0);
        }
    }
    // float64 reference on sampled rows (spread over the request, the last one included).
    std::vector<std::size_t> sample;
    for (std::size_t i = 0; i < check_rows; ++i) sample.push_back(check_rows == 1 ? tokens - 1 : (tokens - 1) * i / (check_rows - 1));
    std::vector<double> ref_err(nk, 0.0);
    double ref_max = 0.0;
    for (const std::size_t t : sample) {
        for (std::size_t h = 0; h < heads; ++h) {
            const std::size_t g = h / (heads / kv);
            std::vector<double> s(t + 1);
            double peak = -1e300;
            for (std::size_t j = 0; j <= t; ++j) {
                double dot = 0;
                for (std::size_t d = 0; d < hd; ++d) dot += static_cast<double>(q[t * qw + h * hd + d]) * keys[(j * kv + g) * hd + d];
                s[j] = dot;
                peak = std::max(peak, dot);
            }
            double sum = 0;
            for (double& x : s) {
                x = std::exp(x - peak);
                sum += x;
            }
            for (std::size_t d = 0; d < hd; ++d) {
                double o = 0;
                for (std::size_t j = 0; j <= t; ++j) o += s[j] * vals[(j * kv + g) * hd + d];
                const double gt = gate[t * 2 * qw + h * 2 * hd + hd + d];
                const double expect = o / sum / (1.0 + std::exp(-gt));
                ref_max = std::max(ref_max, std::fabs(expect));
                for (std::size_t k = 0; k < nk; ++k) ref_err[k] = std::max(ref_err[k], std::fabs(expect - outs[k][t * qw + h * hd + d]));
            }
        }
    }
    double ymax = 0.0;
    for (const float v : outs[0]) ymax = std::max(ymax, static_cast<double>(std::fabs(v)));
    std::printf("attention: %zu tokens in chunks of %zu, %zu heads / %zu kv heads x %zu, %u threads (%s), q scale %.1f, %.3g GFLOP per prefill\n",
                tokens, chunk, heads, kv, hd, threads, pool.layout().c_str(), q_scale, flops / 1e9);
    double first = 0.0;
    for (std::size_t k = 0; k < nk; ++k) {
        std::sort(times[k].begin(), times[k].end());
        const double med = times[k][times[k].size() / 2];
        if (k == 0) first = med;
        double diff = 0.0;
        for (std::size_t i = 0; i < outs[k].size(); ++i) diff = std::max(diff, static_cast<double>(std::fabs(outs[k][i] - outs[0][i])));
        std::printf("ATTN %-6s | median %8.1f ms (best %8.1f) | %7.1f GFLOP/s | vs %s %+6.1f%% | max|d out| vs %s %.2e (max|out| %.3f) | "
                    "max err vs float64 %.2e (%zu rows)\n",
                    name_of(kernels[k]), med * 1e3, times[k].front() * 1e3, flops / med / 1e9, name_of(kernels[0]), 100.0 * (first / med - 1.0),
                    name_of(kernels[0]), diff, ymax, ref_err[k], sample.size());
    }
    return 0;
}
