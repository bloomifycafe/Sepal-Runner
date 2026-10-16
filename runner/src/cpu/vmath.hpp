// AVX2 + FMA vector math shared by the CPU backend's elementwise kernels (internal header).
//
// One algorithm, used lane-wise everywhere: an element's result never depends on its position in a vector or on
// the instruction set, so tails and thread partitions cannot change a value.
#pragma once

#include <immintrin.h>

#include <cmath>
#include <cstddef>

namespace sepal::cpu {

/// exp(x) for float lanes, Cephes-style range reduction and degree-5 polynomial; relative error about 1e-7.
/// Inputs are clamped to [-87.3, 88.3], so the result is never zero, infinity or NaN for finite inputs.
inline __m256 exp_ps(__m256 x) {
    x = _mm256_min_ps(x, _mm256_set1_ps(88.3762626647949f));
    x = _mm256_max_ps(x, _mm256_set1_ps(-87.3f));
    const __m256 fx = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(1.44269504088896341f)),
                                      _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 r = _mm256_fnmadd_ps(fx, _mm256_set1_ps(0.693359375f), x);
    r = _mm256_fnmadd_ps(fx, _mm256_set1_ps(-2.12194440e-4f), r);
    __m256 p = _mm256_set1_ps(1.9875691500E-4f);
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.3981999507E-3f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(8.3334519073E-3f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(4.1665795894E-2f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.6666665459E-1f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(5.0000001201E-1f));
    p = _mm256_fmadd_ps(p, _mm256_mul_ps(r, r), _mm256_add_ps(r, _mm256_set1_ps(1.0f)));
    const __m256i n = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(fx), _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(p, _mm256_castsi256_ps(n));
}

/// x * sigmoid(x) = x / (1 + exp(-x)).
inline __m256 silu_ps(__m256 x) {
    const __m256 e = exp_ps(_mm256_sub_ps(_mm256_setzero_ps(), x));
    return _mm256_div_ps(x, _mm256_add_ps(_mm256_set1_ps(1.0f), e));
}

/// 1 / (1 + exp(-x)).
inline __m256 sigmoid_ps(__m256 x) {
    const __m256 e = exp_ps(_mm256_sub_ps(_mm256_setzero_ps(), x));
    return _mm256_div_ps(_mm256_set1_ps(1.0f), _mm256_add_ps(_mm256_set1_ps(1.0f), e));
}

/// Scalar versions with the same algorithm (used for tails: lane 0 of a one-lane vector).
inline float exp_scalar(float x) { return _mm256_cvtss_f32(exp_ps(_mm256_set1_ps(x))); }
inline float silu_scalar(float x) { return _mm256_cvtss_f32(silu_ps(_mm256_set1_ps(x))); }
inline float sigmoid_scalar(float x) { return _mm256_cvtss_f32(sigmoid_ps(_mm256_set1_ps(x))); }

/// Horizontal sum in a fixed order (lanes 0..7 pairwise).
inline float hsum_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    return _mm_cvtss_f32(s);
}

} // namespace sepal::cpu
