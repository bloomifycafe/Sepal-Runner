// Conventions shared by every device kernel and by the pass executor (docs/RUNNER.md section 6; stage R2).
//
// A pass runs many rows at once. Their tokens lie one row after another in ONE flat list of N tokens, with no
// padding between rows, and every activation is a row-major matrix [N, width] over that list.
//
// Element types
//   - The residual stream, every matrix-multiply output and every state a snapshot keeps: float32.
//   - The input of a matrix multiply: the provider's operand type, binary16 (`__half`) on the F16 path and float32
//     on the F32 path. The kernel that produces such an input writes that type itself; there is no separate
//     conversion launch. Such kernels are therefore written for both output types.
//
// Matrices
//   - A matrix argument is a device pointer and a row stride in elements, so a kernel can read a band of columns
//     out of a wider matrix (one multiply computes several projections of the same input into one output).
//
// Rows
//   - A kernel that needs to know where rows start takes a device array of RowSpan and, when it does not run on
//     every row, a device array of the row indices to run (a "wave": rows whose parent is another row of the same
//     pass run after it).
//
// Determinism
//   - A token's result depends on its own row and on that row's parent chain, and on nothing else: not on the
//     other rows of the pass, not on their number, not on the row's place in the flat list. Every reduction has
//     one fixed order per output element. No fast-math, no atomics on floating-point values.
//
// Launching
//   - A launch function enqueues on the stream it is given and returns without waiting. It throws
//     std::runtime_error (through `check`) when the launch fails. Kernels have external linkage (see
//     convert_kernels.cu for why).
#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

// Tensor-core code (mma.sync, ldmatrix, cp.async, wmma) is compiled only into the sm_80-and-later images of a fat binary; an older
// image (sm_61, sm_75) gets __trap() in its place. The host never launches such a kernel on an older card: the binary16
// profiles are refused below compute capability 8.0 (gpu_decision_engine.cpp), and the float32 path uses the SIMT kernels.
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
#define SEPAL_TENSOR_CORES 1
#else
#define SEPAL_TENSOR_CORES 0
#endif

namespace sepal::cuda {

/// Whether the current device can run the tensor-core kernels: compute capability 8.0 or later AND this build holds an
/// image with them (SEPAL_CUDA_TC_IMAGE, set by CMakeLists.txt when an architecture of 80 or more is built; a Pascal-only
/// build JIT-compiled on a newer card has only the traps). Cached per device.
inline bool device_has_tensor_cores() {
#if defined(SEPAL_CUDA_TC_IMAGE) && SEPAL_CUDA_TC_IMAGE == 0
    return false;
#else
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) {
        (void)cudaGetLastError();
        return false;
    }
    static int cache[64] = {}; // 0 unknown, 1 no, 2 yes (a benign race: every writer writes the same value)
    if (device >= 0 && device < 64 && cache[device] != 0) return cache[device] == 2;
    int major = 0;
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess) {
        (void)cudaGetLastError();
        return false;
    }
    const bool yes = major >= 8;
    if (device >= 0 && device < 64) cache[device] = yes ? 2 : 1;
    return yes;
#endif
}

/// One row of a pass: tokens [offset, offset + length) of the flat token list.
struct RowSpan {
    std::uint32_t offset;
    std::uint32_t length;
};

/// "No slot": a row that starts from nothing has this as the index of its incoming state.
inline constexpr std::uint32_t kNoSlot = 0xFFFFFFFFu;

inline void check(cudaError_t status, const char* call) {
    if (status != cudaSuccess) {
        (void)cudaGetLastError(); // the error is reported by the exception: do not leave it for the next call's check
        throw std::runtime_error(std::string(call) + " failed: " + cudaGetErrorString(status));
    }
}

/// `count` elements of T on the device, freed with the object. Not initialised.
template <class T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t count) : count_(count) {
        if (count_ != 0) check(cudaMalloc(&data_, count_ * sizeof(T)), "cudaMalloc");
    }
    /// Allocates and copies `host` to the device; returns when the copy is complete.
    explicit DeviceBuffer(const std::vector<T>& host) : DeviceBuffer(host.size()) { upload(host.data(), host.size()); }
    ~DeviceBuffer() {
        if (data_ != nullptr) cudaFree(data_);
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept : data_(other.data_), count_(other.count_) {
        other.data_ = nullptr;
        other.count_ = 0;
    }
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (data_ != nullptr) cudaFree(data_);
            data_ = other.data_;
            count_ = other.count_;
            other.data_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    T* data() const noexcept { return static_cast<T*>(data_); }
    std::size_t size() const noexcept { return count_; }

    /// Blocking copies. They wait for the default stream only: synchronise the stream a kernel ran on first.
    void upload(const T* host, std::size_t count) {
        if (count > count_) throw std::invalid_argument("DeviceBuffer::upload: more elements than the buffer holds");
        if (count != 0) check(cudaMemcpy(data_, host, count * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy");
    }
    std::vector<T> download() const {
        std::vector<T> host(count_);
        if (count_ != 0) check(cudaMemcpy(host.data(), data_, count_ * sizeof(T), cudaMemcpyDeviceToHost), "cudaMemcpy");
        return host;
    }
    void zero() {
        if (count_ != 0) check(cudaMemset(data_, 0, count_ * sizeof(T)), "cudaMemset");
    }

private:
    void* data_ = nullptr;
    std::size_t count_ = 0;
};

} // namespace sepal::cuda
