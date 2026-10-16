#include <immintrin.h>

#include "gemm_kernels.hpp"

namespace sepal::cpu {

namespace {

// ---- F32_Y: mr 6, nr 16, group 1 ----------------------------------------------------------------------------

#define Y6_DECL(r)                                                                            \
    __m256 c##r##0 = args.accumulate ? _mm256_load_ps(tile + (r) * 16) : _mm256_setzero_ps(); \
    __m256 c##r##1 = args.accumulate ? _mm256_load_ps(tile + (r) * 16 + 8) : _mm256_setzero_ps();
#define Y6_STEP(r)                                      \
    {                                                   \
        const __m256 ar = _mm256_broadcast_ss(pa + (r)); \
        c##r##0 = _mm256_fmadd_ps(ar, b0, c##r##0);     \
        c##r##1 = _mm256_fmadd_ps(ar, b1, c##r##1);     \
    }
#define Y6_STORE(r)                                    \
    _mm256_store_ps(tile + (r) * 16, c##r##0);         \
    _mm256_store_ps(tile + (r) * 16 + 8, c##r##1);

void kernel_f32_y(const KernelArgs& args) {
    const float* pa = static_cast<const float*>(args.a);
    const float* pb = static_cast<const float*>(args.b);
    float* tile = args.tile;
    Y6_DECL(0) Y6_DECL(1) Y6_DECL(2) Y6_DECL(3) Y6_DECL(4) Y6_DECL(5)
    for (std::size_t k = 0; k < args.kgroups; ++k) {
        const __m256 b0 = _mm256_load_ps(pb);
        const __m256 b1 = _mm256_load_ps(pb + 8);
        Y6_STEP(0) Y6_STEP(1) Y6_STEP(2) Y6_STEP(3) Y6_STEP(4) Y6_STEP(5)
        pa += 6;
        pb += 16;
    }
    Y6_STORE(0) Y6_STORE(1) Y6_STORE(2) Y6_STORE(3) Y6_STORE(4) Y6_STORE(5)
    _mm256_zeroupper();
}

// ---- I8_Y: mr 4, nr 16, group 2 (int16 pairs of int8 values); vpmaddwd into int32 -------------------------------

#define I4_DECL(r) __m256i c##r##0 = _mm256_setzero_si256(); __m256i c##r##1 = _mm256_setzero_si256();
#define I4_STEP(r)                                                          \
    {                                                                       \
        const __m256i ad = _mm256_set1_epi32(pa[r]);                        \
        c##r##0 = _mm256_add_epi32(c##r##0, _mm256_madd_epi16(ad, b0));     \
        c##r##1 = _mm256_add_epi32(c##r##1, _mm256_madd_epi16(ad, b1));     \
    }
#define I4_STORE(r)                                                                                                  \
    {                                                                                                                \
        const __m256 xs = _mm256_broadcast_ss(args.xscale + (r));                                                    \
        _mm256_store_ps(tile + (r) * 16, _mm256_mul_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(c##r##0), ws0), xs));        \
        _mm256_store_ps(tile + (r) * 16 + 8, _mm256_mul_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(c##r##1), ws1), xs));    \
    }

void kernel_i8_y(const KernelArgs& args) {
    const int* pa = static_cast<const int*>(args.a);
    const __m256i* pb = static_cast<const __m256i*>(args.b);
    float* tile = args.tile;
    I4_DECL(0) I4_DECL(1) I4_DECL(2) I4_DECL(3)
    for (std::size_t k = 0; k < args.kgroups; ++k) {
        const __m256i b0 = _mm256_load_si256(pb);
        const __m256i b1 = _mm256_load_si256(pb + 1);
        I4_STEP(0) I4_STEP(1) I4_STEP(2) I4_STEP(3)
        pa += 4;
        pb += 2;
    }
    const __m256 ws0 = _mm256_load_ps(args.wscale);
    const __m256 ws1 = _mm256_load_ps(args.wscale + 8);
    I4_STORE(0) I4_STORE(1) I4_STORE(2) I4_STORE(3)
    _mm256_zeroupper();
}

} // namespace

KernelFn kernel_avx2(Variant v) {
    switch (v) {
    case Variant::F32_Y: return &kernel_f32_y;
    case Variant::I8_Y: return &kernel_i8_y;
    default: return nullptr;
    }
}

} // namespace sepal::cpu
