// The CUDA matrix-multiply provider (docs/RUNNER.md section 6, "GEMM provider"; stage R2).
//
// It implements sepal::GemmProvider on one GPU through cuBLAS:
//
//     Y[t, n] = beta * Y[t, n] + sum over k of X[t, k] * W[n, k],   Y float32.
//
// Memory: one device allocation (the arena) is made when the provider is created and nothing is allocated on the
// device afterwards. Weights and device-resident matrices are carved from the bottom of the arena and stay for
// the provider's life; everything a call needs temporarily is carved from the remainder and released when the
// call returns. When the arena is too small the call throws and names the shortfall.
//
// Precision: the provider computes on one path, chosen at creation.
//   F32: operands are float32 on the device (binary16 and bfloat16 inputs are widened exactly); cublasSgemm.
//   F16: operands are binary16 on the device (float32 and bfloat16 inputs are rounded to nearest); cublasGemmEx
//        with float32 accumulation and float32 output.
//
// All device work runs on a stream the provider owns, never on the default stream.
//
// Row independence, as measured (tests/test_cuda_gemm.cpp, suite `rows`; RTX 3060, cuBLAS 13.2.1):
//   - A row's result never depends on the VALUES of the other rows in the batch, nor on the row's POSITION in
//     the batch: at a fixed batch size both are bit-identical.
//   - A row's BITS do depend on the batch size T, because cuBLAS chooses its kernel from the shape. A row
//     computed alone and the same row in a batch differ in most elements, by at most 1.3e-5 absolute on outputs
//     of rms 0.33 (4e-5 of the rms). cuBLAS's pedantic mode does not change this.
// So this provider does not give "the same bits for a row whether it is computed alone or inside any batch", which
// gemm.hpp asks of a provider. An engine that needs bit-identical results across batches must call it with one
// fixed T (padding the batch), or use a kernel with a fixed summation order instead.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

#include "sepal/dtype.hpp"
#include "sepal/gemm.hpp"
#include "sepal/tensor.hpp"

namespace sepal::cuda {

/// A CUDA runtime, cuBLAS or arena failure. what() reads "<call> failed: <reason> (code <n>)[: <context>]".
class CudaError : public std::runtime_error {
public:
    CudaError(std::string call, long long code, const std::string& reason, const std::string& context = {});
    /// The name of the failing call, for example "cudaMalloc" or "cublasGemmEx".
    const std::string& call() const noexcept { return call_; }
    long long code() const noexcept { return code_; }

private:
    std::string call_;
    long long code_;
};

enum class Precision { F16, F32 };

constexpr DType dtype_of(Precision p) noexcept { return p == Precision::F16 ? DType::F16 : DType::F32; }

struct CudaGemmOptions {
    int device = 0;
    Precision precision = Precision::F16;
    /// Size of the single device allocation. 0 takes 80% of the memory free on the device at creation.
    std::size_t arena_bytes = 0;
    /// Part of the arena handed to cuBLAS as its workspace, so that cuBLAS does not allocate either.
    /// 0 leaves cuBLAS on the pool it allocates once when its handle is created.
    std::size_t cublas_workspace_bytes = std::size_t{32} << 20;
    /// Asks cuBLAS for its pedantic mode (CUBLAS_PEDANTIC_MATH, CUBLAS_COMPUTE_32F_PEDANTIC). Measured on an
    /// RTX 3060 with cuBLAS 13.2.1: it does not make a row's bits independent of the batch (see README.md).
    bool pedantic = false;
};

/// A weight matrix that lives on the device. Valid for the provider that returned it.
struct WeightId {
    std::uint32_t index = 0xFFFFFFFFu;
    constexpr bool valid() const noexcept { return index != 0xFFFFFFFFu; }
};

/// A row-major matrix inside the provider's arena. `data` is a device pointer: never dereference it on the host.
struct DeviceMatrix {
    void* data = nullptr;
    DType dtype = DType::F32;
    std::size_t rows = 0;
    std::size_t cols = 0;
};

struct DeviceInfo {
    int device = 0;
    std::string name;
    int compute_major = 0;
    int compute_minor = 0;
    std::size_t total_bytes = 0;
    int runtime_version = 0; // cudaRuntimeGetVersion: 1000 * major + 10 * minor
    int driver_version = 0;  // cudaDriverGetVersion: the newest CUDA version the installed driver supports
    int cublas_version = 0;  // cublasGetVersion
};

struct MemoryStats {
    std::size_t device_total = 0;
    std::size_t device_free_at_start = 0; // after the CUDA context exists, before the arena and the cuBLAS handle
    std::size_t device_free_min = 0;      // lowest value seen at creation and at each memory() call
    std::size_t arena_bytes = 0;
    std::size_t resident_bytes = 0;     // weights, device matrices and the cuBLAS workspace
    std::size_t scratch_peak_bytes = 0; // largest temporary use by one call
    std::size_t arena_peak_bytes = 0;   // largest resident + temporary use seen
};

struct DeviceMemory {
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
};

/// Free and total memory of a device, as the driver reports them now. Creates a CUDA context on the device.
DeviceMemory query_device_memory(int device = 0);

class CudaGemm final : public GemmProvider {
public:
    explicit CudaGemm(const CudaGemmOptions& options = {});
    ~CudaGemm() override;
    CudaGemm(const CudaGemm&) = delete;
    CudaGemm& operator=(const CudaGemm&) = delete;

    GemmCaps caps() const override;

    /// The interface call. X is uploaded, Y is downloaded, and the call returns when Y is complete.
    /// W is taken from the device when `w` was registered with register_weights (same pointer, element type and
    /// shape); otherwise it is uploaded for this call only.
    /// Throws std::invalid_argument on a shape mismatch, a null pointer or a non-float element type, as the
    /// reference provider does, and CudaError when a device call fails or the arena is too small.
    void gemm_f(MatRef x, MatRef w, MatView y, float beta) const override;

    /// Copies W [N, K] to the device once, converted to the provider's precision, and keeps it there.
    WeightId upload_weights(MatRef w);

    /// upload_weights, and from now on gemm_f(x, w, ...) with this exact MatRef uses the device copy. Only the
    /// pointer, element type and shape are compared: the caller must not change or free the host matrix while the
    /// provider is alive and still called with it.
    WeightId register_weights(MatRef w);

    /// The interface call against a weight matrix that is already on the device.
    void gemm_f(MatRef x, WeightId w, MatView y, float beta) const;

    // ---- device-resident operands: no host transfer per call ----

    /// Copies X to the device, converted to the provider's precision, and keeps it there.
    DeviceMatrix upload_matrix(MatRef x);
    /// Reserves an uninitialised float32 [rows, cols] matrix on the device.
    DeviceMatrix alloc_output(std::size_t rows, std::size_t cols);
    /// Enqueues Y = beta * Y + X W^T on the provider's stream and returns without waiting.
    /// X must have the provider's element type and Y must be float32; both must come from this provider.
    void gemm_device(const DeviceMatrix& x, WeightId w, const DeviceMatrix& y, float beta) const;
    /// Waits until everything enqueued on the provider's stream has finished.
    void synchronize() const;
    /// Copies a float32 device matrix to the host.
    void download(const DeviceMatrix& m, MatView out) const;

    Precision precision() const noexcept;
    DeviceInfo device_info() const;
    MemoryStats memory() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/// A CUDA provider behind the engine's interface type.
std::unique_ptr<GemmProvider> make_cuda_gemm(const CudaGemmOptions& options = {});

} // namespace sepal::cuda
