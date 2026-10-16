// sepal-cuda-int8-probe: what does this card deliver for int8 multiplies at the engine's shapes, against the binary16 ones?
// Measurement tool; the engine does not link it.
//
//   sepal-cuda-int8-probe peak                                big shapes: the best each entry point does
//   sepal-cuda-int8-probe shapes --model 0.8b|2b [--tokens 64,256,...] [--only substring] [--lt 1] [--variants a,b]
//
// Every case is Y[T, N] = X[T, K] W[N, K]^T, issued like the engine issues it (column-major Y^T = op_T(W) X^T for cuBLAS).
// Time per call is the minimum over three batches of back-to-back calls. "TF" is 2*T*N*K per second whatever the operand type
// (so int8 figures are TOP/s). Accuracy is the relative rms error against a float64 product of the *unquantised* operands (the
// binary16 values), over the first six rows: for the int8 variants that is the quantisation error of the whole multiply.

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
#include "gemm_s8_kernels.hpp"

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
int g_batches = 3;
int g_config = -1;
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
    __half *w16 = nullptr, *x16 = nullptr, *y16 = nullptr;
    float* y32 = nullptr;
    std::int8_t *wq = nullptr, *wqp = nullptr, *xq = nullptr; // weights plain / fp16-pairs, quantised activations
    float *sw = nullptr, *swp = nullptr, *sx = nullptr;
    std::int32_t* yi = nullptr;
    std::size_t w_count = 0, x_count = 0, y_count = 0;
    std::vector<float> hx, hw;
};

void alloc_buffers(Buffers& b, int t, int n, int k) {
    b.w_count = static_cast<std::size_t>(n) * static_cast<std::size_t>(k);
    b.x_count = static_cast<std::size_t>(t) * static_cast<std::size_t>(k);
    b.y_count = static_cast<std::size_t>(t) * static_cast<std::size_t>(n);
    std::mt19937 rng(12345);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    const float wscale = 1.0f / std::sqrt(static_cast<float>(k));
    b.hw.resize(b.w_count);
    b.hx.resize(b.x_count);
    std::vector<__half> w16(b.w_count), x16(b.x_count);
    for (std::size_t i = 0; i < b.w_count; ++i) {
        const __half h = __float2half(nd(rng) * wscale);
        w16[i] = h;
        b.hw[i] = __half2float(h);
    }
    for (std::size_t i = 0; i < b.x_count; ++i) {
        const __half h = __float2half(nd(rng));
        x16[i] = h;
        b.hx[i] = __half2float(h);
    }
    CK(cudaMalloc(&b.w16, b.w_count * 2));
    CK(cudaMalloc(&b.x16, b.x_count * 2));
    CK(cudaMalloc(&b.y16, b.y_count * 2));
    CK(cudaMalloc(&b.y32, b.y_count * 4));
    CK(cudaMalloc(&b.yi, b.y_count * 4));
    CK(cudaMalloc(&b.wq, b.w_count));
    CK(cudaMalloc(&b.wqp, b.w_count));
    CK(cudaMalloc(&b.xq, b.x_count));
    CK(cudaMalloc(&b.sw, static_cast<std::size_t>(n) * 4));
    CK(cudaMalloc(&b.swp, static_cast<std::size_t>(n) * 4));
    CK(cudaMalloc(&b.sx, static_cast<std::size_t>(t) * 4));
    CK(cudaMemcpy(b.w16, w16.data(), b.w_count * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(b.x16, x16.data(), b.x_count * 2, cudaMemcpyHostToDevice));
    sepal::cuda::launch_quantize_weights_s8(b.w16, static_cast<std::size_t>(k), b.wq, b.sw, static_cast<std::size_t>(n), static_cast<std::size_t>(k),
                                            sepal::cuda::S8WeightLayout::Plain, g_stream);
    sepal::cuda::launch_quantize_weights_s8(b.w16, static_cast<std::size_t>(k), b.wqp, b.swp, static_cast<std::size_t>(n), static_cast<std::size_t>(k),
                                            sepal::cuda::S8WeightLayout::Fp16Pairs, g_stream);
    sepal::cuda::launch_quantize_rows_s8(b.x16, static_cast<std::size_t>(k), b.xq, static_cast<std::size_t>(k), b.sx, static_cast<std::size_t>(t),
                                         static_cast<std::size_t>(k), g_stream);
    CK(cudaStreamSynchronize(g_stream));
}

void free_buffers(Buffers& b) {
    cudaFree(b.w16);
    cudaFree(b.x16);
    cudaFree(b.y16);
    cudaFree(b.y32);
    cudaFree(b.yi);
    cudaFree(b.wq);
    cudaFree(b.wqp);
    cudaFree(b.xq);
    cudaFree(b.sw);
    cudaFree(b.swp);
    cudaFree(b.sx);
    b = Buffers{};
}

enum class Out { F32, F16, I32 };

double rel_rms(const Buffers& b, int t, int n, int k, Out out) {
    const int rows = std::min(t, 6);
    std::vector<float> y32;
    std::vector<__half> y16;
    if (out == Out::F16) {
        y16.resize(b.y_count);
        CK(cudaMemcpy(y16.data(), b.y16, b.y_count * 2, cudaMemcpyDeviceToHost));
    } else if (out == Out::F32) {
        y32.resize(b.y_count);
        CK(cudaMemcpy(y32.data(), b.y32, b.y_count * 4, cudaMemcpyDeviceToHost));
    } else {
        return -1.0;
    }
    double sq_err = 0, sq_ref = 0;
    for (int r = 0; r < rows; ++r) {
        const float* x = &b.hx[static_cast<std::size_t>(r) * static_cast<std::size_t>(k)];
        for (int c = 0; c < n; ++c) {
            const float* w = &b.hw[static_cast<std::size_t>(c) * static_cast<std::size_t>(k)];
            double acc = 0;
            for (int i = 0; i < k; ++i) acc += static_cast<double>(x[i]) * w[i];
            const double got = out == Out::F16 ? static_cast<double>(__half2float(y16[static_cast<std::size_t>(r) * static_cast<std::size_t>(n) + static_cast<std::size_t>(c)]))
                                               : static_cast<double>(y32[static_cast<std::size_t>(r) * static_cast<std::size_t>(n) + static_cast<std::size_t>(c)]);
            sq_err += (got - acc) * (got - acc);
            sq_ref += acc * acc;
        }
    }
    return std::sqrt(sq_err / std::max(sq_ref, 1e-30));
}

struct Variant {
    std::string name;
    Out out;
    std::function<void()> run;
};

std::vector<Variant> make_variants(const Buffers& b, int t, int n, int k) {
    static const float one = 1.0f, zero = 0.0f;
    static const __half one_h = __float2half(1.0f), zero_h = __float2half(0.0f);
    static const std::int32_t one_i = 1, zero_i = 0;
    const std::size_t T = static_cast<std::size_t>(t), N = static_cast<std::size_t>(n), K = static_cast<std::size_t>(k);
    std::vector<Variant> v;
    v.push_back({"fp16_cublas_f32acc", Out::F32, [=, &b]() {
                     CB(cublasGemmEx(g_blas, CUBLAS_OP_T, CUBLAS_OP_N, n, t, k, &one, b.w16, CUDA_R_16F, k, b.x16, CUDA_R_16F, k, &zero, b.y32, CUDA_R_32F, n,
                                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
                 }});
    v.push_back({"fp16_cublas_f16acc", Out::F16, [=, &b]() {
                     CB(cublasGemmEx(g_blas, CUBLAS_OP_T, CUBLAS_OP_N, n, t, k, &one_h, b.w16, CUDA_R_16F, k, b.x16, CUDA_R_16F, k, &zero_h, b.y16, CUDA_R_16F,
                                     n, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT));
                 }});
    v.push_back({"fp16_own_f32acc", Out::F32, [=, &b]() {
                     sepal::cuda::launch_gemm_nt(b.x16, K, b.w16, K, b.y32, N, false, T, N, K, sepal::cuda::GemmAccumulate::Float32, 0, g_config, g_stream);
                 }});
    v.push_back({"fp16_own_f16acc_fl128", Out::F32, [=, &b]() {
                     sepal::cuda::launch_gemm_nt(b.x16, K, b.w16, K, b.y32, N, false, T, N, K, sepal::cuda::GemmAccumulate::Float16Flush, 128, g_config,
                                                 g_stream);
                 }});
    v.push_back({"s8_own_w8a8_gemm", Out::F32, [=, &b]() {
                     sepal::cuda::launch_gemm_s8(b.xq, K, b.sx, b.wq, K, b.sw, b.y32, N, false, T, N, K, g_config, g_stream);
                 }});
    v.push_back({"s8_own_w8a8_quant+gemm", Out::F32, [=, &b]() {
                     sepal::cuda::launch_quantize_rows_s8(b.x16, K, b.xq, K, b.sx, T, K, g_stream);
                     sepal::cuda::launch_gemm_s8(b.xq, K, b.sx, b.wq, K, b.sw, b.y32, N, false, T, N, K, g_config, g_stream);
                 }});
    v.push_back({"s8_own_w8a8_quant_only", Out::I32, [=, &b]() { sepal::cuda::launch_quantize_rows_s8(b.x16, K, b.xq, K, b.sx, T, K, g_stream); }});
    v.push_back({"s8_own_w8a16", Out::F32, [=, &b]() {
                     sepal::cuda::launch_gemm_w8a16(b.x16, K, b.wqp, K, b.swp, b.y32, N, false, T, N, K, g_config, g_stream);
                 }});
    v.push_back({"s8_cublas_gemmex_int32out", Out::I32, [=, &b]() {
                     CB(cublasGemmEx(g_blas, CUBLAS_OP_T, CUBLAS_OP_N, n, t, k, &one_i, b.wq, CUDA_R_8I, k, b.xq, CUDA_R_8I, k, &zero_i, b.yi, CUDA_R_32I, n,
                                     CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT));
                 }});
    return v;
}

// cuBLASLt int8: A = W [N][K] (transposed), B = X, C int32; every heuristic candidate timed.
struct LtResult {
    int candidates = 0;
    double default_ms = -1, best_ms = -1;
    int best_index = -1;
};

LtResult run_lt_int8(const Buffers& b, int t, int n, int k, int max_algos) {
    LtResult res;
    cublasLtMatmulDesc_t desc = nullptr;
    CB(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32I, CUDA_R_32I));
    const cublasOperation_t op_t = CUBLAS_OP_T, op_n = CUBLAS_OP_N;
    CB(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_t, sizeof(op_t)));
    CB(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_n, sizeof(op_n)));
    cublasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr;
    CB(cublasLtMatrixLayoutCreate(&la, CUDA_R_8I, static_cast<uint64_t>(k), static_cast<uint64_t>(n), k));
    CB(cublasLtMatrixLayoutCreate(&lb, CUDA_R_8I, static_cast<uint64_t>(k), static_cast<uint64_t>(t), k));
    CB(cublasLtMatrixLayoutCreate(&lc, CUDA_R_32I, static_cast<uint64_t>(n), static_cast<uint64_t>(t), n));
    cublasLtMatmulPreference_t pref = nullptr;
    CB(cublasLtMatmulPreferenceCreate(&pref));
    std::size_t ws = kWorkspace;
    CB(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)));
    std::vector<cublasLtMatmulHeuristicResult_t> found(static_cast<std::size_t>(max_algos));
    int count = 0;
    const cublasStatus_t hs = cublasLtMatmulAlgoGetHeuristic(g_lt, desc, la, lb, lc, lc, pref, max_algos, found.data(), &count);
    if (hs == CUBLAS_STATUS_SUCCESS && count > 0) {
        res.candidates = count;
        const std::int32_t alpha = 1, beta = 0;
        const int reps = std::max(3, reps_for(2.0 * t * n * k) / 3);
        for (int i = 0; i < count; ++i) {
            if (found[static_cast<std::size_t>(i)].state != CUBLAS_STATUS_SUCCESS) continue;
            const cublasLtMatmulAlgo_t* algo = &found[static_cast<std::size_t>(i)].algo;
            const auto call = [&]() {
                const cublasStatus_t s = cublasLtMatmul(g_lt, desc, &alpha, b.wq, la, b.xq, lb, &beta, b.yi, lc, b.yi, lc, algo, g_workspace, kWorkspace, g_stream);
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
    cublasLtMatrixLayoutDestroy(lc);
    cublasLtMatmulDescDestroy(desc);
    return res;
}

double tflops(int t, int n, int k, double ms) { return 2.0 * t * n * k / (ms * 1e-3) / 1e12; }

bool want(const std::vector<std::string>& filter, const std::string& name) {
    if (filter.empty()) return true;
    for (const auto& f : filter)
        if (name.find(f) != std::string::npos) return true;
    return false;
}

void run_case(const char* label, int t, int n, int k, bool with_lt, const std::vector<std::string>& filter, int max_algos) {
    Buffers b;
    alloc_buffers(b, t, n, k);
    const int reps = reps_for(2.0 * t * n * k);
    const double weight_mb = static_cast<double>(n) * k / 1e6;
    for (auto& v : make_variants(b, t, n, k)) {
        if (!want(filter, v.name)) continue;
        const double ms = time_ms(v.run, reps);
        v.run();
        CK(cudaStreamSynchronize(g_stream));
        const double e = rel_rms(b, t, n, k, v.out);
        const bool is_fp16 = v.name.rfind("fp16", 0) == 0;
        std::printf("%-14s T=%-5d N=%-5d K=%-5d %-26s %9.4f ms %7.2f TF  %7.1f GB/s(w)  err rel %.2e\n", label, t, n, k, v.name.c_str(), ms,
                    tflops(t, n, k, ms), weight_mb * (is_fp16 ? 2.0 : 1.0) / ms, e);
        std::fflush(stdout);
    }
    if (with_lt && want(filter, "s8_cublaslt")) {
        const LtResult r = run_lt_int8(b, t, n, k, max_algos);
        if (r.candidates == 0) {
            std::printf("%-14s T=%-5d N=%-5d K=%-5d %-26s no algorithm\n", label, t, n, k, "s8_cublaslt_int32out");
        } else {
            std::printf("%-14s T=%-5d N=%-5d K=%-5d %-26s %9.4f ms %7.2f TF  (heuristic #0 %.4f ms %.2f TF, best of %d is #%d)\n", label, t, n, k,
                        "s8_cublaslt_int32out", r.best_ms, tflops(t, n, k, r.best_ms), r.default_ms, tflops(t, n, k, r.default_ms), r.candidates, r.best_index);
        }
        std::fflush(stdout);
    }
    free_buffers(b);
}

struct ShapeSpec {
    const char* name;
    int n, k;
};

std::vector<ShapeSpec> shapes_for(const std::string& model) {
    if (model == "0.8b") {
        return {{"mlp_gate_up", 7168, 1024}, {"mlp_down", 1024, 3584}, {"deltanet_in", 8224, 1024}, {"deltanet_out", 1024, 2048}, {"attn_in", 5120, 1024}, {"attn_out", 1024, 2048}};
    }
    return {{"mlp_gate_up", 12288, 2048}, {"mlp_down", 2048, 6144}, {"deltanet_in", 8224, 2048}, {"deltanet_out", 2048, 2048}, {"attn_in", 5120, 2048}, {"attn_out", 2048, 2048}};
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: sepal-cuda-int8-probe peak|shapes [options]\n");
        return 1;
    }
    const std::string mode = argv[1];
    std::string model = "0.8b";
    std::vector<int> tokens = {64, 256, 1024, 4096};
    std::vector<std::string> filter, only;
    bool with_lt = true;
    int max_algos = 24;
    for (int i = 2; i + 1 < argc; i += 2) {
        const std::string key = argv[i], val = argv[i + 1];
        if (key == "--model") model = val;
        else if (key == "--tokens") {
            tokens.clear();
            for (const auto& s : split(val)) tokens.push_back(std::atoi(s.c_str()));
        } else if (key == "--variants") filter = split(val);
        else if (key == "--only") only = split(val);
        else if (key == "--lt") with_lt = std::atoi(val.c_str()) != 0;
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
        const int big[][3] = {{8192, 8192, 4096}, {4096, 4096, 4096}, {8192, 4096, 8192}, {2048, 8192, 2048}};
        for (const auto& s : big) run_case("peak", s[0], s[1], s[2], with_lt, filter, max_algos);
    } else if (mode == "shapes") {
        for (const ShapeSpec& s : shapes_for(model)) {
            if (!only.empty() && !want(only, s.name)) continue;
            for (int t : tokens) run_case(s.name, t, s.n, s.k, with_lt, filter, max_algos);
        }
    }
    return 0;
}
