// GFLOP/s of the packed CPU GEMM against the core count and the theoretical peak.
//
//   sepal-cpu-gemm-bench [--kind f32|bf16|i8|all] [--isa avx2|avx512] [--threads 1,2,4,6,12] [--rows 2048]
//                        [--n 3584] [--k 1024] [--reps 7]
//
// Prints, per kind and thread count, the best and median of `reps` runs (the packed weights and activations are
// built once; only the product is timed) and the percentage of the derived peak. The peak is derived from the
// clock frequency measured by a dependent-add chain and an assumed issue rate per core and cycle (Zen 4:
// 32 fp32 FLOP, 64 bf16 FLOP with vdpbf16ps, 128 int8 ops with vpdpbusd), over min(threads, 6) physical cores.
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

static double measure_ghz() {
    // A chain of dependent FMAs: Zen 4 has a fused multiply-add latency of 4 cycles (uops.info), so the chain runs
    // at clock / 4 FMAs per second.
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
    return 4.0 * static_cast<double>(n) / best / 1e9;
}

int main(int argc, char** argv) {
    std::string kind_arg = "all", threads_arg = "1,2,4,6,12";
    Isa isa = best_isa();
    std::size_t rows = 2048, n = 3584, k = 1024;
    int reps = 7;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        const std::string v = argv[i + 1];
        if (a == "--kind") kind_arg = v;
        else if (a == "--isa") isa = v == "avx2" ? Isa::Avx2 : Isa::Avx512;
        else if (a == "--threads") threads_arg = v;
        else if (a == "--rows") rows = std::stoul(v);
        else if (a == "--n") n = std::stoul(v);
        else if (a == "--k") k = std::stoul(v);
        else if (a == "--reps") reps = std::stoi(v);
    }
    std::vector<unsigned> thread_counts;
    {
        std::stringstream ss(threads_arg);
        std::string tok;
        while (std::getline(ss, tok, ',')) thread_counts.push_back(static_cast<unsigned>(std::stoul(tok)));
    }
    const double ghz = measure_ghz();
    std::printf("cpu: %s | isa %s | measured clock %.2f GHz (dependent FMA chain, latency 4; single thread)\n", features().brand.c_str(),
                name_of(isa), ghz);
    std::printf("shape: rows %zu, n %zu, k %zu (%.2f GFLOP per product)\n", rows, n, k, 2.0 * static_cast<double>(rows * n * k) / 1e9);

    std::vector<float> w(n * k), x(rows * k);
    for (std::size_t i = 0; i < w.size(); ++i) w[i] = static_cast<float>(static_cast<int>((i * 2654435761u) >> 20 & 1023) - 512) / 512.0f;
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = static_cast<float>(static_cast<int>((i * 40503u + 7) >> 4 & 1023) - 512) / 512.0f;

    for (const WeightKind kind : {WeightKind::F32, WeightKind::BF16, WeightKind::I8}) {
        if (kind_arg != "all" && kind_arg != name_of(kind)) continue;
        const double per_core_cycle = kind == WeightKind::F32 ? 32.0 : kind == WeightKind::BF16 ? 64.0 : 128.0;
        for (const unsigned threads : thread_counts) {
            ThreadPool pool(threads);
            const PackedWeight pw = pack_weight(w.data(), k, n, k, kind, isa, pool);
            PackedActs pa;
            pack_acts(pa, x.data(), k, rows, k, pw.variant(), pool);
            std::vector<float> y(rows * n);
            gemm_packed(pw, pa, y.data(), n, false, pool); // warm up
            std::vector<double> times;
            for (int r = 0; r < reps; ++r) {
                const double t0 = now_s();
                gemm_packed(pw, pa, y.data(), n, false, pool);
                times.push_back(now_s() - t0);
            }
            std::sort(times.begin(), times.end());
            const double flops = gemm_flops(rows, n, k);
            const double best = flops / times.front() / 1e9;
            const double median = flops / times[times.size() / 2] / 1e9;
            const double worst = flops / times.back() / 1e9;
            const double cores = std::min<double>(threads, 6.0);
            // Two threads on a core share its units: the peak counts physical cores.
            const double peak = cores * per_core_cycle * ghz;
            std::printf("%-5s %-18s threads %2u | best %8.1f median %8.1f worst %8.1f GFLOP/s | peak %8.1f | best %5.1f%% of peak | %.2f ms\n",
                        name_of(kind), name_of(pw.variant()), threads, best, median, worst, peak, 100.0 * best / peak,
                        times.front() * 1e3);
        }
    }
    return 0;
}
