// int8 multiplies of the float16 path (docs/RUNNER.md precision profiles; runner/cuda/README.md "int8 multiplies").
// Conventions: device.hpp. Every row's result depends on that row, the weights and K only (not on T, N, the other rows,
// the place in the batch or the tile shape picked): the integer sums are exact, and the float32 steps after them are
// the same for every row.
#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace sepal::cuda {

/// How the int8 weights of a matrix are stored within every group of 16 values of k.
enum class S8WeightLayout {
    Plain,      // as they are: for the int8-activation multiply (launch_gemm_s8)
    Fp16Pairs,  // positions 4q..4q+3 of a group hold k = 2q, 2q+1, 2q+8, 2q+9: for the binary16-activation multiply (launch_gemm_w8a16)
};

/// Per-row symmetric int8 quantisation of binary16 activations: scale[r] = max |x[r, :]| / 127, q[r, c] = round-to-nearest-even
/// of x[r, c] / scale[r] (a zero row gets scale 0 and zeros). x is [rows, cols] binary16 with row stride ld_x, q is int8 with
/// row stride ld_q. Requires cols a multiple of 16, ld_x and ld_q multiples of 16 and 16-byte aligned pointers.
/// Enqueued on `stream`; throws std::invalid_argument / std::runtime_error as launch_gemm_nt.
void launch_quantize_rows_s8(const void* x, std::size_t ld_x, void* q, std::size_t ld_q, float* scales, std::size_t rows, std::size_t cols,
                             cudaStream_t stream);

/// Per-output-channel symmetric int8 quantisation of a weight matrix w [n, k] (binary16, row stride ld_w, k contiguous):
/// scales[r] = max |w[r, :]| / 127, wq[r, c] = round-to-nearest-even of w[r, c] / scales[r], stored contiguously with row
/// stride k in `layout`. Requires k a multiple of 16, ld_w a multiple of 16, 16-byte aligned pointers.
void launch_quantize_weights_s8(const void* w, std::size_t ld_w, void* wq, float* scales, std::size_t n, std::size_t k, S8WeightLayout layout,
                                cudaStream_t stream);

/// Y[t, n] = sx[t] * sw[n] * sum over k of xq[t, k] * wq[n, k]: int8 operands (xq [T, K] with row stride ld_x, wq [N, K] with row
/// stride ld_w, layout Plain), exact int32 sums on the tensor cores, then (float(sum) * sx[t]) * sw[n] in float32. Y is row-major
/// with row stride ld_y, binary16 when y_half and float32 otherwise. `config` < 0 picks a tile shape by T and N; 0 to 5 force
/// 128x128, 64x128, 32x128, 64x64, 32x64 and 32x32 tiles (tests). Requires K, ld_x, ld_w multiples of 16, N and ld_y even,
/// 16-byte aligned xq and wq, 4-byte aligned scales, 8-byte aligned y.
void launch_gemm_s8(const void* xq, std::size_t ld_x, const float* sx, const void* wq, std::size_t ld_w, const float* sw, void* y, std::size_t ld_y,
                    bool y_half, std::size_t t, std::size_t n, std::size_t k, int config, cudaStream_t stream);

/// Y[t, n] = sw[n] * sum over k of x[t, k] * wq[n, k]: x [T, K] binary16 (row stride ld_x), wq [N, K] int8 in layout Fp16Pairs
/// (row stride ld_w). The weights are widened to binary16 in registers (exact) and multiplied on the tensor cores with float32
/// accumulators, so a weight byte is the only thing that is read at a quarter of the cost of the binary16 multiply's; the
/// result is scaled by the channel's scale afterwards. Requires K a multiple of 16, ld_x a multiple of 8, ld_w a multiple of 16,
/// N and ld_y even, 16-byte aligned x and wq.
void launch_gemm_w8a16(const void* x, std::size_t ld_x, const void* wq, std::size_t ld_w, const float* sw, void* y, std::size_t ld_y, bool y_half,
                       std::size_t t, std::size_t n, std::size_t k, int config, cudaStream_t stream);

/// Whether the two multiplies accept these dimensions and strides (pointers are the caller's to align).
constexpr bool gemm_s8_supports(std::size_t n, std::size_t k, std::size_t ld_x, std::size_t ld_w, std::size_t ld_y) noexcept {
    return k != 0 && k % 16 == 0 && ld_x % 16 == 0 && ld_w % 16 == 0 && ld_x >= k && ld_w >= k && n % 2 == 0 && ld_y % 2 == 0 && ld_y >= n;
}
constexpr bool gemm_w8a16_supports(std::size_t n, std::size_t k, std::size_t ld_x, std::size_t ld_w, std::size_t ld_y) noexcept {
    return k != 0 && k % 16 == 0 && ld_x % 8 == 0 && ld_w % 16 == 0 && ld_x >= k && ld_w >= k && n % 2 == 0 && ld_y % 2 == 0 && ld_y >= n;
}

} // namespace sepal::cuda
