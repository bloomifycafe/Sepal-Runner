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

} // namespace detail

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
