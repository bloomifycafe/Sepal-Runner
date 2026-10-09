// Device kernels that belong to the multiplies (cuda_gemm.cpp, cuda_engine.cpp's gemm). Conventions: device.hpp.
#pragma once

#include <cstddef>

#include <cuda_runtime.h>

namespace sepal::cuda {

/// dst[r, c] = float(src[r, c]) for r in [0, rows), c in [0, cols); rows are `ld_src` / `ld_dst` elements apart.
/// src is binary16, dst float32. Each element depends on one element, so rows cannot couple. Enqueued on `stream`;
/// throws std::runtime_error when the launch fails.
void launch_half_to_float_2d(const void* src, std::size_t ld_src, void* dst, std::size_t ld_dst, std::size_t rows,
                             std::size_t cols, cudaStream_t stream);

/// How launch_gemm_nt accumulates the products of one output element.
enum class GemmAccumulate {
    Float32,      // float32 accumulators (tensor cores at half rate on GeForce cards)
    Float16Flush, // binary16 accumulators, added to float32 accumulators every `flush_k` values of k (about twice as fast)
};

/// Y[t, n] = sum over k of X[t, k] * W[n, k]: X [T, K] and W [N, K] binary16 with K contiguous (row strides ld_x and
/// ld_w), Y row-major with row stride ld_y, binary16 when y_half and float32 otherwise. With Float16Flush the products are
/// summed by tensor-core steps in binary16 accumulators that are added to float32 ones after every `flush_k` values of k
/// (a multiple of 32; 0 never flushes, so the whole sum is accumulated in binary16). A row's bits depend on its own X row,
/// W, K, the mode and flush_k only: not on T, N, the other rows, the row's place in the batch, or the tile shape picked for
/// the call (`config` < 0 picks one by T and N; 0 to 5 force 128x128, 64x128, 32x128, 64x64, 32x64 and 32x32 tiles, for tests).
/// Requires K and the strides of X and W multiples of 8 with 16-byte aligned pointers, N and the stride of Y even.
/// Enqueued on `stream`; throws std::invalid_argument for a violated requirement and std::runtime_error when the launch fails.
void launch_gemm_nt(const void* x, std::size_t ld_x, const void* w, std::size_t ld_w, void* y, std::size_t ld_y, bool y_half, std::size_t t,
                    std::size_t n, std::size_t k, GemmAccumulate accumulate, unsigned flush_k, int config, cudaStream_t stream);

/// Whether launch_gemm_nt accepts these dimensions and strides (the pointers are the caller's to align).
constexpr bool gemm_nt_supports(std::size_t n, std::size_t k, std::size_t ld_x, std::size_t ld_w, std::size_t ld_y) noexcept {
    return k != 0 && k % 8 == 0 && ld_x % 8 == 0 && ld_w % 8 == 0 && ld_x >= k && ld_w >= k && n % 2 == 0 && ld_y % 2 == 0 && ld_y >= n;
}

} // namespace sepal::cuda
