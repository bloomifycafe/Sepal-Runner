#include <immintrin.h>

#include "gemm_kernels.hpp"

namespace sepal::cpu {

namespace {

#define ROWS12(X) X(0) X(1) X(2) X(3) X(4) X(5) X(6) X(7) X(8) X(9) X(10) X(11)

// ---- F32_Z: mr 12, nr 32, group 1 ---------------------------------------------------------------------------

#define Z_DECL_F(r) __m512 c##r##0 = _mm512_setzero_ps(); __m512 c##r##1 = _mm512_setzero_ps();
#define Z_STEP_F(r)                                       \
    {                                                     \
        const __m512 ar = _mm512_set1_ps(pa[r]);          \
        c##r##0 = _mm512_fmadd_ps(ar, b0, c##r##0);       \
        c##r##1 = _mm512_fmadd_ps(ar, b1, c##r##1);       \
    }
#define Z_STORE_F(r)                                      \
    _mm512_store_ps(tile + (r) * 32, c##r##0);            \
    _mm512_store_ps(tile + (r) * 32 + 16, c##r##1);

void kernel_f32_z(const KernelArgs& args) {
    const float* pa = static_cast<const float*>(args.a);
    const float* pb = static_cast<const float*>(args.b);
    float* tile = args.tile;
    ROWS12(Z_DECL_F)
    for (std::size_t k = 0; k < args.kgroups; ++k) {
        const __m512 b0 = _mm512_load_ps(pb);
        const __m512 b1 = _mm512_load_ps(pb + 16);
        ROWS12(Z_STEP_F)
        pa += 12;
        pb += 32;
    }
    ROWS12(Z_STORE_F)
}

// ---- BF16_Z: mr 12, nr 32, group 2; vdpbf16ps ---------------------------------------------------------------

#define Z_STEP_B(r)                                                                              \
    {                                                                                            \
        const __m512bh ar = (__m512bh)_mm512_set1_epi32(pa[r]);                                  \
        c##r##0 = _mm512_dpbf16_ps(c##r##0, ar, b0);                                             \
        c##r##1 = _mm512_dpbf16_ps(c##r##1, ar, b1);                                             \
    }

void kernel_bf16_z(const KernelArgs& args) {
    const int* pa = static_cast<const int*>(args.a);
    const __m512i* pb = static_cast<const __m512i*>(args.b);
    float* tile = args.tile;
    ROWS12(Z_DECL_F)
    for (std::size_t k = 0; k < args.kgroups; ++k) {
        const __m512bh b0 = (__m512bh)_mm512_load_si512(pb);
        const __m512bh b1 = (__m512bh)_mm512_load_si512(pb + 1);
        ROWS12(Z_STEP_B)
        pa += 12;
        pb += 2;
    }
    ROWS12(Z_STORE_F)
}

// ---- I8_Z: mr 12, nr 32, group 4; vpdpbusd with u8 activations (value + 128) and s8 weights ------------------

#define Z_DECL_I(r) __m512i c##r##0 = _mm512_setzero_si512(); __m512i c##r##1 = _mm512_setzero_si512();
#define Z_STEP_I(r)                                                \
    {                                                              \
        const __m512i ar = _mm512_set1_epi32(pa[r]);               \
        c##r##0 = _mm512_dpbusd_epi32(c##r##0, ar, b0);            \
        c##r##1 = _mm512_dpbusd_epi32(c##r##1, ar, b1);            \
    }
#define Z_STORE_I(r)                                                                                                   \
    {                                                                                                                  \
        const __m512 xs = _mm512_set1_ps(args.xscale[r]);                                                              \
        _mm512_store_ps(tile + (r) * 32, _mm512_mul_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(c##r##0, corr0)), ws0), xs));      \
        _mm512_store_ps(tile + (r) * 32 + 16, _mm512_mul_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(c##r##1, corr1)), ws1), xs)); \
    }

void kernel_i8_z(const KernelArgs& args) {
    const int* pa = static_cast<const int*>(args.a);
    const __m512i* pb = static_cast<const __m512i*>(args.b);
    float* tile = args.tile;
    ROWS12(Z_DECL_I)
    for (std::size_t k = 0; k < args.kgroups; ++k) {
        const __m512i b0 = _mm512_load_si512(pb);
        const __m512i b1 = _mm512_load_si512(pb + 1);
        ROWS12(Z_STEP_I)
        pa += 12;
        pb += 2;
    }
    const __m512 ws0 = _mm512_load_ps(args.wscale);
    const __m512 ws1 = _mm512_load_ps(args.wscale + 16);
    const __m512i corr0 = _mm512_load_si512(args.corr);
    const __m512i corr1 = _mm512_load_si512(args.corr + 16);
    ROWS12(Z_STORE_I)
}

} // namespace

KernelFn kernel_avx512(Variant v) {
    switch (v) {
    case Variant::F32_Z: return &kernel_f32_z;
    case Variant::BF16_Z: return &kernel_bf16_z;
    case Variant::I8_Z: return &kernel_i8_z;
    default: return nullptr;
    }
}

} // namespace sepal::cpu
