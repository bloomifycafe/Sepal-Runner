// The float32 SIMT multiply (src/gemm_simt_kernels.cu) against cuBLAS SGEMM on the Sepal-4B shapes.
//
//   sepal-cuda-simt-bench [--tokens T] [--reps R] [--check on]
//
// For each shape [T x K] by [N x K]: the SIMT kernel with float32, bfloat16 and int8-group weights, and cublasSgemm on
// float32 copies of the same (dequantised) weights, in TFLOP/s (2 T N K per multiply). With --check on, the cuBLAS result is
// compared with the SIMT result on the same float32 weights (max relative difference; the summation orders differ).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "device.hpp"
#include "gemm_simt_kernels.hpp"

using namespace sepal::cuda;

namespace {

struct Shape {
    const char* name;
    std::size_t n, k;
    SimtEpilogue ep;
};

double time_ms(cudaStream_t s, int reps, const auto& fn) {
    fn();
    check(cudaStreamSynchronize(s), "sync");
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    cudaEventRecord(a, s);
    for (int i = 0; i < reps; ++i) fn();
    cudaEventRecord(b, s);
    cudaEventSynchronize(b);
    float ms = 0;
    cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    return static_cast<double>(ms) / reps;
}

} // namespace

int main(int argc, char** argv) {
    std::size_t T = 2048;
    int reps = 10;
    bool do_check = false;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "--tokens") T = std::strtoull(argv[i + 1], nullptr, 10);
        else if (a == "--reps") reps = std::atoi(argv[i + 1]);
        else if (a == "--check") do_check = std::string(argv[i + 1]) == "on";
    }
    cudaDeviceProp prop{};
    check(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
    int rt = 0, drv = 0, blas_ver = 0;
    cudaRuntimeGetVersion(&rt);
    cudaDriverGetVersion(&drv);
    cudaStream_t s;
    check(cudaStreamCreate(&s), "cudaStreamCreate");
    cublasHandle_t h;
    if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) return 2;
    cublasSetStream(h, s);
    cublasGetVersion(h, &blas_ver);
    std::printf("device %s sm_%d%d, %d SMs; CUDA runtime %d, driver %d, cuBLAS %d; T = %zu\n", prop.name, prop.major, prop.minor,
                prop.multiProcessorCount, rt, drv, blas_ver, T);

    const Shape shapes[] = {
        {"mlp gate+up (SwiGLU)", 18432, 2560, SimtEpilogue::SwiGLU},
        {"mlp down (+residual)", 2560, 9216, SimtEpilogue::Add},
        {"deltanet in (qkv,z)", 12288, 2560, SimtEpilogue::Store},
        {"deltanet out", 2560, 4096, SimtEpilogue::Add},
        {"attention in (q,g,k,v)", 10240, 2560, SimtEpilogue::Store},
        {"attention out", 2560, 4096, SimtEpilogue::Add},
    };
    std::mt19937 rng(1);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::printf("%-24s %8s %8s %10s %10s %10s %10s %10s\n", "shape", "N", "K", "simt-f32", "simt-bf16", "simt-w8g", "cublasSgemm",
                "w8g/sgemm");
    double sum_sgemm = 0, sum_bf16 = 0, sum_w8 = 0;
    for (const Shape& sh : shapes) {
        const std::size_t N = sh.n, K = sh.k;
        std::vector<float> x(T * K), w(N * K);
        for (auto& v : x) v = nd(rng);
        for (auto& v : w) v = nd(rng) / std::sqrt(static_cast<float>(K));
        DeviceBuffer<float> dx(x), dw(w), dy(T * N), dy2(T * N), dr(T * N);
        DeviceBuffer<std::uint16_t> dw16(N * K);
        DeviceBuffer<std::int8_t> dq(N * K);
        DeviceBuffer<float> dsc(N * (K / 128));
        launch_float_to_bf16(dw.data(), dw16.data(), N * K, s);
        launch_quantize_group_s8(dw.data(), K, dq.data(), K, dsc.data(), K / 128, N, K, s);
        dr.zero();
        const std::size_t ld_y = sh.ep == SimtEpilogue::SwiGLU ? N / 2 : N;
        const SimtMatrix m32{dw.data(), K, nullptr, 0, SimtWeight::F32};
        const SimtMatrix m16{dw16.data(), K, nullptr, 0, SimtWeight::BF16};
        const SimtMatrix m8{dq.data(), K, dsc.data(), K / 128, SimtWeight::Int8Group};
        const double flop = 2.0 * static_cast<double>(T) * static_cast<double>(N) * static_cast<double>(K);
        const auto simt = [&](const SimtMatrix& m) {
            return time_ms(s, reps, [&] { launch_gemm_simt(dx.data(), K, m, sh.ep == SimtEpilogue::Add ? dr.data() : dy.data(), ld_y, T, N, K, sh.ep, -1, s); });
        };
        const double t32 = simt(m32), t16 = simt(m16), t8 = simt(m8);
        const float alpha = 1.0f, beta = 0.0f;
        const double tb = time_ms(s, reps, [&] {
            cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(N), static_cast<int>(T), static_cast<int>(K), &alpha, dw.data(),
                        static_cast<int>(K), dx.data(), static_cast<int>(K), &beta, dy2.data(), static_cast<int>(N));
        });
        sum_sgemm += tb;
        sum_bf16 += t16;
        sum_w8 += t8;
        std::printf("%-24s %8zu %8zu %10.2f %10.2f %10.2f %10.2f %10.2f\n", sh.name, N, K, flop / t32 * 1e-9, flop / t16 * 1e-9,
                    flop / t8 * 1e-9, flop / tb * 1e-9, tb / t8);
        if (do_check) {
            launch_gemm_simt(dx.data(), K, m32, dy.data(), N, T, N, K, SimtEpilogue::Store, -1, s);
            check(cudaStreamSynchronize(s), "sync");
            const std::vector<float> a = dy.download(), b = dy2.download();
            double worst = 0, rms = 0;
            for (std::size_t i = 0; i < a.size(); ++i) {
                worst = std::max(worst, std::fabs(static_cast<double>(a[i]) - b[i]));
                rms += static_cast<double>(b[i]) * b[i];
            }
            std::printf("    check: max |simt - cublas| = %.3e (rms of the result %.3e)\n", worst, std::sqrt(rms / static_cast<double>(a.size())));
        }
    }
    std::printf("all shapes (sum of times): bf16 %.2f ms, w8g %.2f ms, cublasSgemm %.2f ms; simt-bf16 at %.0f%% of SGEMM speed\n", sum_bf16, sum_w8,
                sum_sgemm, 100.0 * sum_sgemm / sum_bf16);
    cublasDestroy(h);
    cudaStreamDestroy(s);
    return 0;
}
