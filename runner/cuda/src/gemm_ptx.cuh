// PTX wrappers shared by the multiply kernels (gemm_kernels.cu, gemm_s8_kernels.cu): cp.async, ldmatrix and the
// XOR swizzle of 64-byte shared-memory tile rows.
#pragma once

#include <cuda_runtime.h>
#include "device.hpp"

namespace sepal::cuda::detail {

// Chunk c (of 4) of tile row r lives at chunk c ^ ((r >> 1) & 3): the eight rows of an ldmatrix read, or the eight
// consecutive threads of a cp.async write, then fall into different bank groups.
__device__ inline int swz(int r, int c) { return c ^ ((r >> 1) & 3); }

__device__ inline void cp_async16(void* smem, const void* gmem, bool valid) {
#if SEPAL_TENSOR_CORES
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    const int bytes = valid ? 16 : 0; // zero-fill outside the matrix
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(bytes));
#else
    __trap();
#endif
}
__device__ inline void cp_async_commit() {
#if SEPAL_TENSOR_CORES
    asm volatile("cp.async.commit_group;\n" ::);
#else
    __trap();
#endif
}
template <int N>
__device__ inline void cp_async_wait() {
#if SEPAL_TENSOR_CORES
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
#else
    __trap();
#endif
}
__device__ inline void ldmatrix_x4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, const void* smem) {
#if SEPAL_TENSOR_CORES
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(s));
#else
    __trap();
#endif
}
__device__ inline void ldmatrix_x2(unsigned& r0, unsigned& r1, const void* smem) {
#if SEPAL_TENSOR_CORES
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n" : "=r"(r0), "=r"(r1) : "r"(s));
#else
    __trap();
#endif
}

} // namespace sepal::cuda::detail
