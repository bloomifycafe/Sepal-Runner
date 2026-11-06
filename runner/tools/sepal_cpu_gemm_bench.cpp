// GFLOP/s of the packed CPU GEMM against the core count and the peak.
//
//   sepal-cpu-gemm-bench [--kind f32|bf16|i8|all] [--isa avx2|avx2-tuned|avx512] [--threads 1,2,4,6,12] [--rows 2048]
//                        [--n 3584] [--k 1024] [--reps 7] [--shapes custom|4b|0.8b] [--peak-core GFLOPS]
//                        [--fma-latency 4]
//
// Prints, per shape, kind and thread count, the best and median of `reps` runs (the packed weights are built once;
// the product is timed alone, and the packing of the activations separately) and the percentage of the peak. The peak
// per core is --peak-core (fp32 FMA GFLOP/s of one core, measured with independent FMA streams) or, without it, derived
// from the clock measured by a dependent FMA chain (--fma-latency cycles: 4 on Zen 4 and Skylake, 5 on Haswell and
// Broadwell) and an assumed issue rate per core and cycle (fp32 FMA: 32 FLOP on AVX2 and Zen 4; bf16 with vdpbf16ps
// 64; int8 with vpdpbusd 128). The bf16 and int8 kernels of the AVX2 sets run at the fp32 FMA rate.
// --shapes 4b / 0.8b run the trunk's products of Qwen3.5-4B / 0.8B (gate_up as one SwiGLU pair, down, DeltaNet in and
// out, attention qkv and o) and print the FLOP-weighted total of one layer mix.
#include <immintrin.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include "sepal/cpu/gemm.hpp"
#include "sepal/cpu/isa.hpp"

using namespace sepal::cpu;
using Clock = std::chrono::steady_clock;

static double now_s() { return std::chrono::duration<double>(Clock::now().time_since_epoch()).count(); }

static double measure_ghz(double latency) {
    // A chain of dependent FMAs runs at clock / latency FMAs per second.
    const long long n = 200'000'000;
    __m256 v = _mm256_set1_ps(1.0f);
    const __m256 mul = _mm256_set1_ps(1.0f), add = _mm256_set1_ps(0.0f);
    double best = 1e30;
    for (int rep = 0; rep < 3; ++rep) {
        const double t0 = now_s();
        for (long long i = 0; i < n; ++i) v = _mm256_fmadd_ps(v, mul, add);
        best = std::min(best, now_s() - t0);
    }
    volatile float sink = _mm256_cvtss_f32(v);
    (void)sink;
    return latency * static_cast<double>(n) / best / 1e9;
}

struct Shape {
    const char* name;
    std::size_t n, k;
    bool swiglu;
    double per_layer; // how many of this product one average layer runs (FLOP weighting of the total)
};

int main(int argc, char** argv) {
    std::string kind_arg = "all", threads_arg = "1,2,4,6,12", shapes_arg = "custom";
    Isa isa = best_isa();
    std::size_t rows = 2048, n = 3584, k = 1024;
    int reps = 7;
    double peak_core = 0.0, latency = 4.0;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        const std::string v = argv[i + 1];
        if (a == "--kind") kind_arg = v;
        else if (a == "--isa") isa = parse_isa(v);
        else if (a == "--threads") threads_arg = v;
        else if (a == "--rows") rows = std::stoul(v);
        else if (a == "--n") n = std::stoul(v);
        else if (a == "--k") k = std::stoul(v);
        else if (a == "--reps") reps = std::stoi(v);
        else if (a == "--shapes") shapes_arg = v;
        else if (a == "--peak-core") peak_core = std::stod(v);
        else if (a == "--fma-latency") latency = std::stod(v);
    }
    require_isa(isa);
    std::vector<unsigned> thread_counts;
    {
        std::stringstream ss(threads_arg);
        std::string tok;
        while (std::getline(ss, tok, ',')) thread_counts.push_back(static_cast<unsigned>(std::stoul(tok)));
    }
    std::vector<Shape> shapes;
    if (shapes_arg == "4b") {
        // Qwen3.5-4B: hidden 2560, intermediate 9216, 24 DeltaNet + 8 attention layers of 32.
        shapes = {{"gate_up", 9216, 2560, true, 1.0},  {"down", 2560, 9216, false, 1.0},  {"dn_in", 12288, 2560, false, 0.75},
                  {"dn_out", 2560, 4096, false, 0.75}, {"attn_qkv", 10240, 2560, false, 0.25}, {"attn_o", 2560, 4096, false, 0.25}};
    } else if (shapes_arg == "0.8b") {
        // Qwen3.5-0.8B: hidden 1024, intermediate 3584, DeltaNet 16 + 16 heads x 128, attention 8 heads x 256 (2 kv).
        shapes = {{"gate_up", 3584, 1024, true, 1.0},  {"down", 1024, 3584, false, 1.0},  {"dn_in", 8192, 1024, false, 0.75},
                  {"dn_out", 1024, 2048, false, 0.75}, {"attn_qkv", 5120, 1024, false, 0.25}, {"attn_o", 1024, 2048, false, 0.25}};
    } else {
        shapes = {{"custom", n, k, false, 1.0}};
    }
    const CacheSizes& cs = cache_sizes();
    const double ghz = peak_core > 0.0 ? 0.0 : measure_ghz(latency);
    std::printf("cpu: %s | isa %s | L1D %zu KiB, L2 %zu KiB, L3 %zu KiB | ", features().brand.c_str(), name_of(isa), cs.l1d / 1024, cs.l2 / 1024,
                cs.l3 / 1024);
    if (peak_core > 0.0) std::printf("peak %.1f GFLOP/s per core (given)\n", peak_core);
    else std::printf("clock %.2f GHz (dependent FMA chain, latency %.0f; single thread)\n", ghz, latency);

    std::vector<float> x(rows * 16384);
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = static_cast<float>(static_cast<int>((i * 40503u + 7) >> 4 & 1023) - 512) / 512.0f;

    for (const WeightKind kind : {WeightKind::F32, WeightKind::BF16, WeightKind::I8}) {
        if (kind_arg != "all" && kind_arg != name_of(kind)) continue;
        const bool avx512 = isa == Isa::Avx512;
        const double per_core_cycle = !avx512 || kind == WeightKind::F32 ? 32.0 : kind == WeightKind::BF16 ? 64.0 : 128.0;
        for (const unsigned threads : thread_counts) {
            ThreadPool pool(threads);
            double flops_total = 0.0, time_total = 0.0, pack_total = 0.0;
            for (const Shape& s : shapes) {
                const std::size_t wn = s.swiglu ? 2 * s.n : s.n;
                std::vector<float> w(wn * s.k);
                for (std::size_t i = 0; i < w.size(); ++i) w[i] = static_cast<float>(static_cast<int>((i * 2654435761u) >> 20 & 1023) - 512) / 4096.0f;
                const PackedWeight pw = s.swiglu ? pack_weight_swiglu(w.data(), w.data() + s.n * s.k, s.k, s.n, s.k, kind, isa, pool)
                                                 : pack_weight(w.data(), s.k, s.n, s.k, kind, isa, pool);
                PackedActs pa;
                pack_acts(pa, x.data(), s.k, rows, s.k, pw.variant(), pool);
                std::vector<float> y(rows * s.n);
                gemm_packed(pw, pa, y.data(), s.n, false, pool); // warm up
                std::vector<double> times, packs;
                for (int r = 0; r < reps; ++r) {
                    const double t0 = now_s();
                    pack_acts(pa, x.data(), s.k, rows, s.k, pw.variant(), pool);
                    const double t1 = now_s();
                    gemm_packed(pw, pa, y.data(), s.n, false, pool);
                    times.push_back(now_s() - t1);
                    packs.push_back(t1 - t0);
                }
                std::sort(times.begin(), times.end());
                std::sort(packs.begin(), packs.end());
                const double flops = gemm_flops(rows, s.swiglu ? 2 * s.n : s.n, s.k);
                const double best = flops / times.front() / 1e9;
                const double median = flops / times[times.size() / 2] / 1e9;
                const double worst = flops / times.back() / 1e9;
                // Two threads on a core share its units: the peak counts physical cores (the caller pins one thread per core).
                const double peak = peak_core > 0.0 ? threads * peak_core : threads * per_core_cycle * ghz;
                std::printf("%-8s %-5s %-20s threads %2u | rows %zu n %zu k %zu | best %7.1f median %7.1f worst %7.1f GFLOP/s | peak %7.1f | "
                            "median %5.1f%% of peak | %.2f ms, pack %.2f ms\n",
                            s.name, name_of(kind), name_of(pw.variant()), threads, rows, s.swiglu ? 2 * s.n : s.n, s.k, best, median, worst, peak,
                            100.0 * median / peak, times[times.size() / 2] * 1e3, packs[packs.size() / 2] * 1e3);
                flops_total += flops * s.per_layer;
                time_total += times[times.size() / 2] * s.per_layer;
                pack_total += packs[packs.size() / 2] * s.per_layer;
            }
            if (shapes.size() > 1) {
                const double peak = peak_core > 0.0 ? threads * peak_core : threads * per_core_cycle * ghz;
                std::printf("LAYERMIX %-5s threads %2u | %7.1f GFLOP/s, %5.1f%% of peak (median times, FLOP weighted); packing adds %.1f%% of the GEMM time\n",
                            name_of(kind), threads, flops_total / time_total / 1e9, 100.0 * flops_total / time_total / 1e9 / peak,
                            100.0 * pack_total / time_total);
            }
            std::fflush(stdout);
        }
    }
    return 0;
}
