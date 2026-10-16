#include "sepal/cpu/elementwise.hpp"

#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "vmath.hpp"

namespace sepal::cpu {

namespace {

constexpr std::size_t row_block = 8;

std::size_t ceil_div(std::size_t a, std::size_t b) { return (a + b - 1) / b; }

// Sum of squares of n values (n a multiple of 8) in a fixed order: four 8-lane accumulators, then lanes.
float sum_squares(const float* x, std::size_t n) {
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps(), a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
    std::size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        const __m256 v0 = _mm256_loadu_ps(x + i), v1 = _mm256_loadu_ps(x + i + 8);
        const __m256 v2 = _mm256_loadu_ps(x + i + 16), v3 = _mm256_loadu_ps(x + i + 24);
        a0 = _mm256_fmadd_ps(v0, v0, a0);
        a1 = _mm256_fmadd_ps(v1, v1, a1);
        a2 = _mm256_fmadd_ps(v2, v2, a2);
        a3 = _mm256_fmadd_ps(v3, v3, a3);
    }
    for (; i + 8 <= n; i += 8) {
        const __m256 v = _mm256_loadu_ps(x + i);
        a0 = _mm256_fmadd_ps(v, v, a0);
    }
    const __m256 s = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
    return hsum_ps(s);
}

} // namespace

void rmsnorm_rows(const float* x, std::size_t ldx, float* y, std::size_t ldy, std::size_t rows, std::size_t cols,
                  const float* w1, double eps, ThreadPool& pool) {
    if (cols % 8 != 0) throw std::invalid_argument("rmsnorm_rows: cols must be a multiple of 8");
    pool.parallel_for(ceil_div(rows, row_block), [&](std::size_t blk, unsigned) {
        const std::size_t end = std::min(rows, (blk + 1) * row_block);
        for (std::size_t r = blk * row_block; r < end; ++r) {
            const float* xr = x + r * ldx;
            float* yr = y + r * ldy;
            const double mean = static_cast<double>(sum_squares(xr, cols)) / static_cast<double>(cols);
            const __m256 inv = _mm256_set1_ps(static_cast<float>(1.0 / std::sqrt(mean + eps)));
            for (std::size_t c = 0; c < cols; c += 8) {
                const __m256 v = _mm256_mul_ps(_mm256_loadu_ps(xr + c), inv);
                _mm256_storeu_ps(yr + c, _mm256_mul_ps(v, _mm256_loadu_ps(w1 + c)));
            }
        }
    });
}

void gated_rmsnorm_heads(const float* x, std::size_t ldx, const float* z, std::size_t ldz, float* y, std::size_t ldy,
                         std::size_t rows, std::size_t heads, std::size_t dim, const float* w, double eps, ThreadPool& pool) {
    if (dim % 8 != 0) throw std::invalid_argument("gated_rmsnorm_heads: dim must be a multiple of 8");
    pool.parallel_for(ceil_div(rows, row_block), [&](std::size_t blk, unsigned) {
        const std::size_t end = std::min(rows, (blk + 1) * row_block);
        for (std::size_t r = blk * row_block; r < end; ++r) {
            for (std::size_t h = 0; h < heads; ++h) {
                const float* xr = x + r * ldx + h * dim;
                const float* zr = z + r * ldz + h * dim;
                float* yr = y + r * ldy + h * dim;
                const double mean = static_cast<double>(sum_squares(xr, dim)) / static_cast<double>(dim);
                const __m256 inv = _mm256_set1_ps(static_cast<float>(1.0 / std::sqrt(mean + eps)));
                for (std::size_t c = 0; c < dim; c += 8) {
                    __m256 v = _mm256_mul_ps(_mm256_loadu_ps(xr + c), inv);
                    v = _mm256_mul_ps(v, _mm256_loadu_ps(w + c));
                    _mm256_storeu_ps(yr + c, _mm256_mul_ps(v, silu_ps(_mm256_loadu_ps(zr + c))));
                }
            }
        }
    });
}

void l2norm_heads(float* x, std::size_t ld, std::size_t rows, std::size_t heads, std::size_t dim, double eps, ThreadPool& pool) {
    if (dim % 8 != 0) throw std::invalid_argument("l2norm_heads: dim must be a multiple of 8");
    pool.parallel_for(ceil_div(rows, row_block), [&](std::size_t blk, unsigned) {
        const std::size_t end = std::min(rows, (blk + 1) * row_block);
        for (std::size_t r = blk * row_block; r < end; ++r) {
            for (std::size_t h = 0; h < heads; ++h) {
                float* xr = x + r * ld + h * dim;
                const double total = static_cast<double>(sum_squares(xr, dim));
                const __m256 inv = _mm256_set1_ps(static_cast<float>(1.0 / std::sqrt(total + eps)));
                for (std::size_t c = 0; c < dim; c += 8) _mm256_storeu_ps(xr + c, _mm256_mul_ps(_mm256_loadu_ps(xr + c), inv));
            }
        }
    });
}

void causal_conv_silu(const float* in, std::size_t ldin, float* out, std::size_t ldout, std::size_t rows,
                      std::size_t channels, const float* w_t, std::size_t taps, float* tail, ThreadPool& pool) {
    if (channels % 8 != 0 || taps == 0) throw std::invalid_argument("causal_conv_silu: channels must be a multiple of 8");
    const std::size_t history = taps - 1;
    constexpr std::size_t block = 16;
    pool.parallel_for(ceil_div(rows, block), [&](std::size_t blk, unsigned) {
        const std::size_t end = std::min(rows, (blk + 1) * block);
        for (std::size_t t = blk * block; t < end; ++t) {
            // Row of the window at tap j: window index t + j; indices below `history` come from the tail.
            const float* src[16];
            for (std::size_t j = 0; j < taps; ++j) {
                const std::size_t idx = t + j;
                src[j] = idx < history ? tail + idx * channels : in + (idx - history) * ldin;
            }
            float* dst = out + t * ldout;
            for (std::size_t c = 0; c < channels; c += 8) {
                __m256 acc = _mm256_mul_ps(_mm256_loadu_ps(w_t + c), _mm256_loadu_ps(src[0] + c));
                for (std::size_t j = 1; j < taps; ++j) {
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(w_t + j * channels + c), _mm256_loadu_ps(src[j] + c), acc);
                }
                _mm256_storeu_ps(dst + c, silu_ps(acc));
            }
        }
    });
    // New tail: the last `history` rows of tail-then-input.
    if (history == 0) return;
    std::vector<float> next(history * channels);
    for (std::size_t i = 0; i < history; ++i) {
        const std::size_t idx = rows + i; // window index
        const float* s = idx < history ? tail + idx * channels : in + (idx - history) * ldin;
        std::memcpy(next.data() + i * channels, s, channels * sizeof(float));
    }
    std::memcpy(tail, next.data(), next.size() * sizeof(float));
}

void silu_inplace(float* x, std::size_t n) {
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) _mm256_storeu_ps(x + i, silu_ps(_mm256_loadu_ps(x + i)));
    for (; i < n; ++i) x[i] = silu_scalar(x[i]);
}

} // namespace sepal::cpu
