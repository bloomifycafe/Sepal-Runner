// Fused elementwise kernels of the CPU trunk (AVX2 + FMA; they are memory bound, so one width serves both ISAs).
// Every function is deterministic: an output element depends only on its own row and weights, with a fixed order
// of operations, whatever the thread count.
#pragma once

#include <cstddef>
#include <cstdint>

#include "sepal/cpu/thread_pool.hpp"

namespace sepal::cpu {

/// Zero-centred RMSNorm of `rows` rows of `cols` values: y = x / sqrt(mean(x^2) + eps) * w1, where `w1` is the
/// ALREADY ADDED scale (1 + stored weight). cols must be a multiple of 8. y may alias x.
void rmsnorm_rows(const float* x, std::size_t ldx, float* y, std::size_t ldy, std::size_t rows, std::size_t cols,
                  const float* w1, double eps, ThreadPool& pool);

/// Gated RMSNorm of the DeltaNet read: for every (row, head) of `dim` values, y = x / sqrt(mean(x^2) + eps) * w *
/// silu(z). x is [rows, heads * dim] (ld ldx), z is [rows, heads * dim] (ld ldz), w is [dim] (the stored weight
/// itself). y may alias x. dim must be a multiple of 8.
void gated_rmsnorm_heads(const float* x, std::size_t ldx, const float* z, std::size_t ldz, float* y, std::size_t ldy,
                         std::size_t rows, std::size_t heads, std::size_t dim, const float* w, double eps, ThreadPool& pool);

/// x / sqrt(sum(x^2) + eps) over every (row, head) of `dim` values, in place (eps inside the root).
void l2norm_heads(float* x, std::size_t ld, std::size_t rows, std::size_t heads, std::size_t dim, double eps, ThreadPool& pool);

/// Depthwise causal conv with SiLU: out[t, c] = silu(sum_j w[j][c] * in[t + j - (taps - 1)][c]) where in rows before
/// the segment come from `tail` [taps - 1, channels] (oldest first). `w_t` is [taps][channels] (tap j of channel c;
/// tap taps-1 multiplies the current input). channels must be a multiple of 8. After the call `tail` holds the last
/// taps - 1 rows of tail-then-input.
void causal_conv_silu(const float* in, std::size_t ldin, float* out, std::size_t ldout, std::size_t rows,
                      std::size_t channels, const float* w_t, std::size_t taps, float* tail, ThreadPool& pool);

/// SiLU of the rows' values in place. (Used for tests and small tensors.)
void silu_inplace(float* x, std::size_t n);

} // namespace sepal::cpu
