// sepal-cuda-gemm-probe: what do cuBLAS and cuBLASLt deliver at the engine's multiply shapes, and what does each entry
// point cost in accuracy? Measurement tool; the engine does not link it.
//
//   sepal-cuda-gemm-probe peak                         big shapes: the best this card does, per compute type
//   sepal-cuda-gemm-probe shapes --model 0.8b|2b [--tokens 64,256,...] [--only substring] [--lt 1] [--variants a,b]
//   sepal-cuda-gemm-probe spin [--seconds 20]          a long 4096^3 loop, to watch clocks and power from nvidia-smi
//
// Every case is Y[T, N] = X[T, K] W[N, K]^T in row-major terms, issued the way the engine issues it: column-major
// Y^T (N x T) = op_T(W stored [N][K]) * X^T. Time per call is the minimum over three batches of back-to-back calls.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "gemm_kernels.hpp"

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

#define CK(call)                                                                                      \
    do {                                                                                              \
        const cudaError_t e_ = (call);                                                                \
        if (e_ != cudaSuccess) {                                                                      \
            std::fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(e_)); \
            std::exit(2);                                                                             \
        }                                                                                             \
    } while (0)
#define CB(call)                                                                                      \
    do {                                                                                              \
        const cublasStatus_t s_ = (call);                                                             \
        if (s_ != CUBLAS_STATUS_SUCCESS) {                                                            \
            std::fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #call, cublasGetStatusString(s_)); \
            std::exit(2);                                                                             \
        }                                                                                             \
    } while (0)

cublasHandle_t g_blas = nullptr;
cublasLtHandle_t g_lt = nullptr;
cudaStream_t g_stream = nullptr;
void* g_workspace = nullptr;
int g_batches = 3; // batches of back-to-back calls timed; the minimum is reported
int g_config = -1; // tile configuration of the custom kernel: -1 picks by T
constexpr std::size_t kWorkspace = std::size_t{32} << 20;

std::vector<std::string> split(const std::string& s) {
    std::vector<std::string> parts;
    std::size_t begin = 0;
    while (begin <= s.size()) {
        const std::size_t comma = s.find(',', begin);
        const std::size_t end = comma == std::string::npos ? s.size() : comma;
        if (end > begin) parts.push_back(s.substr(begin, end - begin));
        begin = end + 1;
    }
    return parts;
}

// Minimum over three batches of the mean time of `reps` back-to-back calls, in ms.
double time_ms(const std::function<void()>& fn, int reps) {
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    for (int i = 0; i < 3; ++i) fn();
    CK(cudaStreamSynchronize(g_stream));
    double best = 1e30;
    for (int batch = 0; batch < g_batches; ++batch) {
        CK(cudaEventRecord(a, g_stream));
        for (int i = 0; i < reps; ++i) fn();
        CK(cudaEventRecord(b, g_stream));
        CK(cudaEventSynchronize(b));
        float ms = 0;
        CK(cudaEventElapsedTime(&ms, a, b));
        best = std::min(best, static_cast<double>(ms) / reps);
    }
    CK(cudaEventDestroy(a));
    CK(cudaEventDestroy(b));
    return best;
}

int reps_for(double flops) { return flops > 2e10 ? 8 : (flops > 2e9 ? 20 : 60); }

struct Buffers {
    __half *w16 = nullptr, *x16 = nullptr, *y16 = nullptr; // w16 holds [N][K]; w16t its transpose [K][N]
    __half* w16t = nullptr;
    float *w32 = nullptr, *x32 = nullptr, *y32 = nullptr;
    std::size_t w_count = 0, x_count = 0, y_count = 0;
    std::vector<float> hx, hw; // host copies of the fp16-rounded values, for the reference
};

void alloc_buffers(Buffers& b, int t, int n, int k, bool want32) {
    b.w_count = static_cast<std::size_t>(n) * k;
    b.x_count = static_cast<std::size_t>(t) * k;
    b.y_count = static_cast<std::size_t>(t) * n;
    std::mt19937 rng(12345);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    const float wscale = 1.0f / std::sqrt(static_cast<float>(k));
    b.hw.resize(b.w_count);
    b.hx.resize(b.x_count);
    std::vector<__half> w16(b.w_count), w16t(b.w_count), x16(b.x_count);
    for (std::size_t i = 0; i < b.w_count; ++i) {
        const __half h = __float2half(nd(rng) * wscale);
        w16[i] = h;
        b.hw[i] = __half2float(h);
    }
    for (int r = 0; r < n; ++r)
        for (int c = 0; c < k; ++c) w16t[static_cast<std::size_t>(c) * n + r] = w16[static_cast<std::size_t>(r) * k + c];
    for (std::size_t i = 0; i < b.x_count; ++i) {
        const __half h = __float2half(nd(rng));
        x16[i] = h;
        b.hx[i] = __half2float(h);
    }
    CK(cudaMalloc(&b.w16, b.w_count * 2));
    CK(cudaMalloc(&b.w16t, b.w_count * 2));
    CK(cudaMalloc(&b.x16, b.x_count * 2));
    CK(cudaMalloc(&b.y16, b.y_count * 2));
    CK(cudaMalloc(&b.y32, b.y_count * 4));
    CK(cudaMemcpy(b.w16, w16.data(), b.w_count * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(b.w16t, w16t.data(), b.w_count * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(b.x16, x16.data(), b.x_count * 2, cudaMemcpyHostToDevice));
    if (want32) {
        CK(cudaMalloc(&b.w32, b.w_count * 4));
        CK(cudaMalloc(&b.x32, b.x_count * 4));
        CK(cudaMemcpy(b.w32, b.hw.data(), b.w_count * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(b.x32, b.hx.data(), b.x_count * 4, cudaMemcpyHostToDevice));
    }
}

void free_buffers(Buffers& b) {
    cudaFree(b.w16);
    cudaFree(b.w16t);
    cudaFree(b.x16);
    cudaFree(b.y16);
    cudaFree(b.y32);
    cudaFree(b.w32);
    cudaFree(b.x32);
    b = Buffers{};
}

// Error of an output against a float64 product of the (fp16-rounded) operands, over the first `rows` rows.
struct Err {
    double max_abs = 0, rms_rel = 0;
};

Err measure(const Buffers& b, int t, int n, int k, bool out_half) {
    const int rows = std::min(t, 6);
    std::vector<float> y32;
    std::vector<__half> y16;
    if (out_half) {
        y16.resize(b.y_count);
        CK(cudaMemcpy(y16.data(), b.y16, b.y_count * 2, cudaMemcpyDeviceToHost));
    } else {
        y32.resize(b.y_count);
        CK(cudaMemcpy(y32.data(), b.y32, b.y_count * 4, cudaMemcpyDeviceToHost));
    }
    double sum_sq_err = 0, sum_sq_ref = 0, max_abs = 0;
    for (int r = 0; r < rows; ++r) {
        const float* x = &b.hx[static_cast<std::size_t>(r) * k];
        for (int c = 0; c < n; ++c) {
            const float* w = &b.hw[static_cast<std::size_t>(c) * k];
            double acc = 0;
            for (int i = 0; i < k; ++i) acc += static_cast<double>(x[i]) * w[i];
            const double got = out_half ? static_cast<double>(__half2float(y16[static_cast<std::size_t>(r) * n + c]))
                                        : static_cast<double>(y32[static_cast<std::size_t>(r) * n + c]);
            const double d = got - acc;
            max_abs = std::max(max_abs, std::fabs(d));
            sum_sq_err += d * d;
            sum_sq_ref += acc * acc;
        }
    }
    return Err{max_abs, std::sqrt(sum_sq_err / std::max(sum_sq_ref, 1e-30))};
}

// ---- one cuBLASLt case: all heuristic candidates, each timed ----
struct LtCase {
    cublasComputeType_t compute;
    cudaDataType scale;
    cudaDataType a_type, b_type, c_type;
    bool trans_a; // true: A is W stored [N][K]; false: A is W stored [K][N]
    int t, n, k;
    const void* a;
    const void* b;
    void* c;
    bool half_scale;
};

struct LtResult {
    int candidates = 0;
    double default_ms = -1; // the heuristic's first choice
    double best_ms = -1;
    int best_index = -1;
};

LtResult run_lt(const LtCase& lc, int max_algos) {
    LtResult res;
    cublasLtMatmulDesc_t desc = nullptr;
    CB(cublasLtMatmulDescCreate(&desc, lc.compute, lc.scale));
    const cublasOperation_t op_t = CUBLAS_OP_T, op_n = CUBLAS_OP_N;
    CB(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, lc.trans_a ? &op_t : &op_n, sizeof(op_t)));
    CB(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_n, sizeof(op_n)));
    cublasLtMatrixLayout_t la = nullptr, lb = nullptr, lcm = nullptr;
    if (lc.trans_a) {
        CB(cublasLtMatrixLayoutCreate(&la, lc.a_type, static_cast<uint64_t>(lc.k), static_cast<uint64_t>(lc.n), lc.k));
    } else {
        CB(cublasLtMatrixLayoutCreate(&la, lc.a_type, static_cast<uint64_t>(lc.n), static_cast<uint64_t>(lc.k), lc.n));
    }
    CB(cublasLtMatrixLayoutCreate(&lb, lc.b_type, static_cast<uint64_t>(lc.k), static_cast<uint64_t>(lc.t), lc.k));
    CB(cublasLtMatrixLayoutCreate(&lcm, lc.c_type, static_cast<uint64_t>(lc.n), static_cast<uint64_t>(lc.t), lc.n));
    cublasLtMatmulPreference_t pref = nullptr;
    CB(cublasLtMatmulPreferenceCreate(&pref));
    std::size_t ws = kWorkspace;
    CB(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)));
    std::vector<cublasLtMatmulHeuristicResult_t> found(static_cast<std::size_t>(max_algos));
    int count = 0;
    const cublasStatus_t hs = cublasLtMatmulAlgoGetHeuristic(g_lt, desc, la, lb, lcm, lcm, pref, max_algos, found.data(), &count);
    if (hs == CUBLAS_STATUS_SUCCESS && count > 0) {
        res.candidates = count;
        const float af = 1.0f, bf = 0.0f;
        const __half ah = __float2half(1.0f), bh = __float2half(0.0f);
        const double flops = 2.0 * lc.t * lc.n * lc.k;
        const int reps = std::max(3, reps_for(flops) / 3);
        for (int i = 0; i < count; ++i) {
            if (found[static_cast<std::size_t>(i)].state != CUBLAS_STATUS_SUCCESS) continue;
            const cublasLtMatmulAlgo_t* algo = &found[static_cast<std::size_t>(i)].algo;
            const auto call = [&]() {
                const cublasStatus_t s = cublasLtMatmul(g_lt, desc, lc.half_scale ? static_cast<const void*>(&ah) : &af, lc.a, la, lc.b, lb,
                                                        lc.half_scale ? static_cast<const void*>(&bh) : &bf, lc.c, lcm, lc.c, lcm, algo,
                                                        g_workspace, kWorkspace, g_stream);
                if (s != CUBLAS_STATUS_SUCCESS) {
                    std::fprintf(stderr, "cublasLtMatmul: %s\n", cublasGetStatusString(s));
                    std::exit(2);
                }
            };
            const double ms = time_ms(call, reps);
            if (i == 0) res.default_ms = ms;
            if (res.best_ms < 0 || ms < res.best_ms) {
                res.best_ms = ms;
                res.best_index = i;
            }
        }
    }
    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(la);
    cublasLtMatrixLayoutDestroy(lb);
    cublasLtMatrixLayoutDestroy(lcm);
    cublasLtMatmulDescDestroy(desc);
    return res;
}

// ---- cuBLAS variants ----
struct Variant {
    std::string name;
    bool out_half;
    bool exact_check; // false: skip the accuracy measurement (no meaning)
    std::function<void()> run;
};

std::vector<Variant> make_blas_variants(const Buffers& b, int t, int n, int k) {
    static const float one = 1.0f, zero = 0.0f;
    static const __half one_h = __float2half(1.0f), zero_h = __float2half(0.0f);
    std::vector<Variant> v;
    v.push_back({"gemmex_f32acc_out32", false, true, [=, &b]() {
                     CB(cublasGemmEx(g_blas, CUBLAS_OP_T, CUBLAS_OP_N, n, t, k, &one, b.w16, CUDA_R_16F, k, b.x16, CUDA_R_16F, k, &zero, b.y32,
                                     CUDA_R_32F, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
                 }});
    v.push_back({"gemmex_f32acc_out16", true, true, [=, &b]() {
                     CB(cublasGemmEx(g_blas, CUBLAS_OP_T, CUBLAS_OP_N, n, t, k, &one, b.w16, CUDA_R_16F, k, b.x16, CUDA_R_16F, k, &zero, b.y16,
                                     CUDA_R_16F, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
                 }});
    if (t % 64 == 0 && t >= 128) {
        // The same product as t/64 batched multiplies of 64 rows: the weights shared (stride 0), X and Y cut into row blocks.
        v.push_back({"gemmex_f32acc_batched64_out32", false, true, [=, &b]() {
                         CB(cublasGemmStridedBatchedEx(g_blas, CUBLAS_OP_T, CUBLAS_OP_N, n, 64, k, &one, b.w16, CUDA_R_16F, k, 0, b.x16, CUDA_R_16F, k,
                                                       static_cast<long long>(64) * k, &zero, b.y32, CUDA_R_32F, n, static_cast<long long>(64) * n, t / 64,
                                                       CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
                     }});
    }
    v.push_back({"gemmex_f32acc_NN_out32", false, true, [=, &b]() { // weights stored [K][N]
                     CB(cublasGemmEx(g_blas, CUBLAS_OP_N, CUBLAS_OP_N, n, t, k, &one, b.w16t, CUDA_R_16F, n, b.x16, CUDA_R_16F, k, &zero, b.y32,
                                     CUDA_R_32F, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
                 }});
    v.push_back({"gemmex_f16acc_out16", true, true, [=, &b]() {
                     CB(cublasGemmEx(g_blas, CUBLAS_OP_T, CUBLAS_OP_N, n, t, k, &one_h, b.w16, CUDA_R_16F, k, b.x16, CUDA_R_16F, k, &zero_h, b.y16,
                                     CUDA_R_16F, n, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT));
                 }});
    v.push_back({"gemmex_fast16f_out32", false, true, [=, &b]() { // fp32 compute that may down-convert (operands are fp16 already)
                     CB(cublasGemmEx(g_blas, CUBLAS_OP_T, CUBLAS_OP_N, n, t, k, &one, b.w16, CUDA_R_16F, k, b.x16, CUDA_R_16F, k, &zero, b.y32,
                                     CUDA_R_32F, n, CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT));
                 }});
    for (const unsigned flush : {0u, 64u, 128u, 256u, 512u}) {
        v.push_back({"custom_fl" + std::to_string(flush) + "_out32", false, true, [=, &b]() {
                         sepal::cuda::launch_gemm_nt(b.x16, static_cast<std::size_t>(k), b.w16, static_cast<std::size_t>(k), b.y32,
                                                     static_cast<std::size_t>(n), false, static_cast<std::size_t>(t), static_cast<std::size_t>(n),
                                                     static_cast<std::size_t>(k), sepal::cuda::GemmAccumulate::Float16Flush, flush, g_config, g_stream);
                     }});
    }
    v.push_back({"custom_fl128_out16", true, true, [=, &b]() {
                     sepal::cuda::launch_gemm_nt(b.x16, static_cast<std::size_t>(k), b.w16, static_cast<std::size_t>(k), b.y16,
                                                 static_cast<std::size_t>(n), true, static_cast<std::size_t>(t), static_cast<std::size_t>(n),
                                                 static_cast<std::size_t>(k), sepal::cuda::GemmAccumulate::Float16Flush, 128, g_config, g_stream);
                 }});
    v.push_back({"custom_f32acc_out32", false, true, [=, &b]() {
                     sepal::cuda::launch_gemm_nt(b.x16, static_cast<std::size_t>(k), b.w16, static_cast<std::size_t>(k), b.y32,
                                                 static_cast<std::size_t>(n), false, static_cast<std::size_t>(t), static_cast<std::size_t>(n),
                                                 static_cast<std::size_t>(k), sepal::cuda::GemmAccumulate::Float32, 0, g_config, g_stream);
                 }});
    if (b.w32 != nullptr) {
        v.push_back({"gemmex_tf32_out32", false, true, [=, &b]() {
                         CB(cublasGemmEx(g_blas, CUBLAS_OP_T, CUBLAS_OP_N, n, t, k, &one, b.w32, CUDA_R_32F, k, b.x32, CUDA_R_32F, k, &zero, b.y32,
                                         CUDA_R_32F, n, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT));
                     }});
        v.push_back({"sgemm_fp32", false, true, [=, &b]() {
                         CB(cublasSgemm(g_blas, CUBLAS_OP_T, CUBLAS_OP_N, n, t, k, &one, b.w32, k, b.x32, k, &zero, b.y32, n));
                     }});
    }
    return v;
}

struct LtVariant {
    std::string name;
    bool out_half;
    LtCase lc;
};

std::vector<LtVariant> make_lt_variants(const Buffers& b, int t, int n, int k) {
    std::vector<LtVariant> v;
    LtCase base{CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16F, CUDA_R_16F, CUDA_R_32F, true, t, n, k, b.w16, b.x16, b.y32, false};
    v.push_back({"lt_f32acc_out32", false, base});
    LtCase o16 = base;
    o16.c_type = CUDA_R_16F;
    o16.c = b.y16;
    v.push_back({"lt_f32acc_out16", true, o16});
    LtCase nn = base;
    nn.trans_a = false;
    nn.a = b.w16t;
    v.push_back({"lt_f32acc_NN_out32", false, nn});
    LtCase h = o16;
    h.compute = CUBLAS_COMPUTE_16F;
    h.scale = CUDA_R_16F;
    h.half_scale = true;
    v.push_back({"lt_f16acc_out16", true, h});
    LtCase hnn = h;
    hnn.trans_a = false;
    hnn.a = b.w16t;
    v.push_back({"lt_f16acc_NN_out16", true, hnn});
    return v;
}

double tflops(int t, int n, int k, double ms) { return 2.0 * t * n * k / (ms * 1e-3) / 1e12; }

bool want(const std::vector<std::string>& filter, const std::string& name) {
    if (filter.empty()) return true;
    for (const auto& f : filter)
        if (name.find(f) != std::string::npos) return true;
    return false;
}

void run_case(const char* label, int t, int n, int k, bool with_lt, bool want32, const std::vector<std::string>& filter, int max_algos) {
    Buffers b;
    alloc_buffers(b, t, n, k, want32);
    const double flops = 2.0 * t * n * k;
    const int reps = reps_for(flops);
    for (auto& v : make_blas_variants(b, t, n, k)) {
        if (!want(filter, v.name)) continue;
        const double ms = time_ms(v.run, reps);
        v.run();
        CK(cudaStreamSynchronize(g_stream));
        const Err e = measure(b, t, n, k, v.out_half);
        std::printf("%-22s T=%-6d N=%-5d K=%-5d %-24s %9.4f ms %7.2f TF  err max %.2e rel %.2e\n", label, t, n, k, v.name.c_str(), ms,
                    tflops(t, n, k, ms), e.max_abs, e.rms_rel);
        std::fflush(stdout);
    }
    if (with_lt) {
        for (auto& v : make_lt_variants(b, t, n, k)) {
            if (!want(filter, v.name)) continue;
            const LtResult r = run_lt(v.lc, max_algos);
            if (r.candidates == 0) {
                std::printf("%-22s T=%-6d N=%-5d K=%-5d %-24s no algorithm\n", label, t, n, k, v.name.c_str());
                continue;
            }
            // Re-run the winning algorithm once more for the accuracy figure: the buffers hold the last candidate's result,
            // which has the same accumulation type, so the error is representative of the variant.
            CK(cudaStreamSynchronize(g_stream));
            const Err e = measure(b, t, n, k, v.out_half);
            std::printf("%-22s T=%-6d N=%-5d K=%-5d %-24s %9.4f ms %7.2f TF  (heuristic #0 %.4f ms %.2f TF, best of %d is #%d)  err max %.2e rel %.2e\n",
                        label, t, n, k, v.name.c_str(), r.best_ms, tflops(t, n, k, r.best_ms), r.default_ms, tflops(t, n, k, r.default_ms),
                        r.candidates, r.best_index, e.max_abs, e.rms_rel);
            std::fflush(stdout);
        }
    }
    free_buffers(b);
}

struct ShapeSpec {
    const char* name;
    int n, k, per_model; // projections of this shape in a 24-layer model
};

std::vector<ShapeSpec> shapes_for(const std::string& model) {
    if (model == "0.8b") {
        return {{"mlp_gate_up", 7168, 1024, 24}, {"mlp_down", 1024, 3584, 24},     {"deltanet_in", 8224, 1024, 18},
                {"deltanet_out", 1024, 2048, 18}, {"attn_in", 5120, 1024, 6},       {"attn_out", 1024, 2048, 6}, {"attn_kv", 1024, 1024, 1}, {"attn_qg", 4096, 1024, 1}};
    }
    return {{"mlp_gate_up", 12288, 2048, 24}, {"mlp_down", 2048, 6144, 24},     {"deltanet_in", 8224, 2048, 18},
            {"deltanet_out", 2048, 2048, 18}, {"attn_in", 5120, 2048, 6},        {"attn_out", 2048, 2048, 6}, {"attn_kv", 1024, 2048, 1}, {"attn_qg", 4096, 2048, 1}};
}

// Does a row's result depend on where the row sits in the batch, or on the other rows? Bits compared, per variant.
void rows_check(const char* label, int t, int n, int k, const std::vector<std::string>& filter) {
    Buffers b;
    alloc_buffers(b, t, n, k, false);
    std::mt19937 rng(777);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    const auto upload_x = [&](const std::vector<float>& rows_f) {
        std::vector<__half> h(rows_f.size());
        for (std::size_t i = 0; i < h.size(); ++i) h[i] = __float2half(rows_f[i]);
        CK(cudaMemcpy(b.x16, h.data(), h.size() * 2, cudaMemcpyHostToDevice));
    };
    const auto fetch = [&](bool out_half) {
        std::vector<unsigned> bits(b.y_count);
        CK(cudaStreamSynchronize(g_stream));
        if (out_half) {
            std::vector<unsigned short> y(b.y_count);
            CK(cudaMemcpy(y.data(), b.y16, b.y_count * 2, cudaMemcpyDeviceToHost));
            for (std::size_t i = 0; i < y.size(); ++i) bits[i] = y[i];
        } else {
            CK(cudaMemcpy(bits.data(), b.y32, b.y_count * 4, cudaMemcpyDeviceToHost));
        }
        return bits;
    };
    for (auto& v : make_blas_variants(b, t, n, k)) {
        if (!want(filter, v.name)) continue;
        std::vector<float> x0(b.x_count);
        for (auto& e : x0) e = nd(rng);
        upload_x(x0);
        v.run();
        const auto y0 = fetch(v.out_half);
        v.run();
        const auto y0b = fetch(v.out_half);
        // (a) rotate the rows by 37 places
        std::vector<float> xa(b.x_count);
        const int shift = 37 % t;
        for (int r = 0; r < t; ++r) std::copy_n(&x0[static_cast<std::size_t>(r) * k], k, &xa[static_cast<std::size_t>((r + shift) % t) * k]);
        upload_x(xa);
        v.run();
        const auto ya = fetch(v.out_half);
        // (b) every row but row 0 replaced
        std::vector<float> xb = x0;
        for (std::size_t i = static_cast<std::size_t>(k); i < xb.size(); ++i) xb[i] = nd(rng);
        upload_x(xb);
        v.run();
        const auto yb = fetch(v.out_half);
        int rot_diff = 0, same_run_diff = 0, other_diff = 0;
        for (int r = 0; r < t; ++r) {
            bool d = false, d2 = false;
            for (int c = 0; c < n; ++c) {
                d = d || y0[static_cast<std::size_t>(r) * n + c] != ya[static_cast<std::size_t>((r + shift) % t) * n + c];
                d2 = d2 || y0[static_cast<std::size_t>(r) * n + c] != y0b[static_cast<std::size_t>(r) * n + c];
            }
            rot_diff += d;
            same_run_diff += d2;
        }
        for (int c = 0; c < n; ++c) other_diff += y0[static_cast<std::size_t>(c)] != yb[static_cast<std::size_t>(c)];
        std::printf("%-14s T=%-5d N=%-5d K=%-5d %-24s rows differing after rotation by %d: %d of %d; same input twice: %d; row 0 with other rows replaced: %d of %d columns\n",
                    label, t, n, k, v.name.c_str(), shift, rot_diff, t, same_run_diff, other_diff, n);
        std::fflush(stdout);
    }
    free_buffers(b);
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: sepal-cuda-gemm-probe peak|shapes|spin [options]\n");
        return 1;
    }
    const std::string mode = argv[1];
    std::string model = "0.8b";
    std::vector<int> tokens = {64, 256, 1024, 4096};
    std::vector<std::string> filter, only;
    bool with_lt = true, want32 = false;
    int seconds = 20, max_algos = 48;
    for (int i = 2; i + 1 < argc; i += 2) {
        const std::string key = argv[i], val = argv[i + 1];
        if (key == "--model") model = val;
        else if (key == "--tokens") {
            tokens.clear();
            for (const auto& s : split(val)) tokens.push_back(std::atoi(s.c_str()));
        } else if (key == "--variants") filter = split(val);
        else if (key == "--only") only = split(val);
        else if (key == "--lt") with_lt = std::atoi(val.c_str()) != 0;
        else if (key == "--fp32") want32 = std::atoi(val.c_str()) != 0;
        else if (key == "--seconds") seconds = std::atoi(val.c_str());
        else if (key == "--algos") max_algos = std::atoi(val.c_str());
        else if (key == "--config") g_config = std::atoi(val.c_str());
        else if (key == "--batches") g_batches = std::atoi(val.c_str());
        else {
            std::fprintf(stderr, "unknown option %s\n", key.c_str());
            return 1;
        }
    }
    CK(cudaSetDevice(0));
    CK(cudaStreamCreateWithFlags(&g_stream, cudaStreamNonBlocking));
    CB(cublasCreate(&g_blas));
    CB(cublasSetStream(g_blas, g_stream));
    CK(cudaMalloc(&g_workspace, kWorkspace));
    CB(cublasSetWorkspace(g_blas, g_workspace, kWorkspace));
    CB(cublasLtCreate(&g_lt));

    if (mode == "peak") {
        // Large shapes: the card's best. fp32 operands only at the sizes that fit.
        const int big[][3] = {{8192, 8192, 4096}, {4096, 4096, 4096}, {8192, 4096, 8192}, {2048, 8192, 2048}};
        for (const auto& s : big) run_case("peak", s[0], s[1], s[2], with_lt, s[0] * s[2] <= 4096 * 4096 * 4, filter, max_algos);
    } else if (mode == "shapes") {
        for (const ShapeSpec& s : shapes_for(model)) {
            if (!only.empty() && !want(only, s.name)) continue;
            for (int t : tokens) run_case(s.name, t, s.n, s.k, with_lt, want32, filter, max_algos);
        }
    } else if (mode == "rows") {
        for (const ShapeSpec& s : shapes_for(model)) {
            if (!only.empty() && !want(only, s.name)) continue;
            for (int t : tokens) rows_check(s.name, t, s.n, s.k, filter);
        }
    } else if (mode == "spin") {
        Buffers b;
        alloc_buffers(b, 4096, 4096, 4096, false);
        auto vs = make_blas_variants(b, 4096, 4096, 4096);
        const auto begin = std::chrono::steady_clock::now();
        double done = 0;
        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count() < seconds) {
            for (int i = 0; i < 10; ++i) vs[0].run();
            CK(cudaStreamSynchronize(g_stream));
            done += 10;
        }
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
        std::printf("spin: %.2f TFLOP/s sustained over %.1f s\n", done * 2.0 * 4096.0 * 4096.0 * 4096.0 / s / 1e12, s);
    }
    return 0;
}
