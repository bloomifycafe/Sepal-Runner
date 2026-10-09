#include "engine_kernels.hpp"

#include <stdexcept>

#include "device.hpp"

namespace sepal::cuda {

// The kernel has external linkage (a kernel in an anonymous namespace makes the compiler's generated stubs trip the
// warnings-as-errors build, see convert_kernels.cu).
namespace detail {

constexpr unsigned kGatherThreads = 128;

// One block per destination row; the threads copy it 32 bits at a time.
__global__ void gather_rows_kernel(const char* src, std::size_t src_stride, const std::uint32_t* rows, std::size_t row_bytes,
                                   char* dst, std::size_t dst_stride) {
    const char* from = src + static_cast<std::size_t>(rows[blockIdx.x]) * src_stride;
    char* to = dst + static_cast<std::size_t>(blockIdx.x) * dst_stride;
    const std::size_t words = row_bytes / 4;
    for (std::size_t i = threadIdx.x; i < words; i += blockDim.x) {
        reinterpret_cast<std::uint32_t*>(to)[i] = reinterpret_cast<const std::uint32_t*>(from)[i];
    }
}

constexpr unsigned kHashThreads = 256;
constexpr unsigned kHashBlocks = 256;

// Thread t hashes the 8-byte words t, t + stride, ... into one value; the host folds the values in order. Integer
// arithmetic only, so the result does not depend on the schedule.
__global__ void hash_words_kernel(const unsigned long long* words, std::size_t count, unsigned long long* partials) {
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    const std::size_t first = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    unsigned long long h = 0x9E3779B97F4A7C15ull ^ (first * 0xC2B2AE3D27D4EB4Full);
    for (std::size_t i = first; i < count; i += stride) {
        unsigned long long w = words[i] * 0xC2B2AE3D27D4EB4Full;
        w = (w << 31) | (w >> 33);
        h ^= w * 0x9E3779B185EBCA87ull;
        h = ((h << 27) | (h >> 37)) * 0x9E3779B185EBCA87ull + 0x165667B19E3779F9ull;
    }
    partials[first] = h;
}

} // namespace detail

std::size_t hash_partial_count() { return static_cast<std::size_t>(detail::kHashThreads) * detail::kHashBlocks; }

void launch_hash_words(const void* data, std::size_t bytes, unsigned long long* partials, cudaStream_t stream) {
    if (data == nullptr || partials == nullptr) throw std::invalid_argument("hash_words: null pointer");
    if (bytes % 8 != 0 || reinterpret_cast<std::uintptr_t>(data) % 8 != 0) throw std::invalid_argument("hash_words: size and pointer must be multiples of 8");
    detail::hash_words_kernel<<<dim3(detail::kHashBlocks), dim3(detail::kHashThreads), 0, stream>>>(
        static_cast<const unsigned long long*>(data), bytes / 8, partials);
    check(cudaGetLastError(), "hash_words kernel launch");
}

void launch_gather_rows(const void* src, std::size_t src_stride_bytes, const std::uint32_t* rows, std::size_t count,
                        std::size_t row_bytes, void* dst, std::size_t dst_stride_bytes, cudaStream_t stream) {
    if (count == 0 || row_bytes == 0) return;
    if (row_bytes % 4 != 0 || src_stride_bytes % 4 != 0 || dst_stride_bytes % 4 != 0 ||
        reinterpret_cast<std::uintptr_t>(src) % 4 != 0 || reinterpret_cast<std::uintptr_t>(dst) % 4 != 0) {
        throw std::invalid_argument("gather_rows: row width, strides and pointers must be multiples of 4 bytes");
    }
    if (src == nullptr || dst == nullptr || rows == nullptr) throw std::invalid_argument("gather_rows: null pointer");
    detail::gather_rows_kernel<<<dim3(static_cast<unsigned>(count)), dim3(detail::kGatherThreads), 0, stream>>>(
        static_cast<const char*>(src), src_stride_bytes, rows, row_bytes, static_cast<char*>(dst), dst_stride_bytes);
    check(cudaGetLastError(), "gather_rows kernel launch");
}

} // namespace sepal::cuda
