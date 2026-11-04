// The tuned AVX2 kernel set for x86-64-v3 CPUs (AVX2, FMA, F16C, BMI1/2, LZCNT, MOVBE; no AVX-512): Haswell,
// Broadwell, Skylake client, Zen 1 to 3. GCC and Clang build this file with the x86-64-v3 flags; `best_isa()` only
// picks it after checking every one of those features with cpuid.
//
// Both kernels are 6 x 16 tiles with the twelve accumulators in ymm0..ymm11 and no spill, the k loop unrolled four
// times so the 4-wide front end of Haswell/Broadwell keeps both FMA ports busy (20 fused uops per k step against 6
// cycles of FMAs). BF16_Y3 widens the bf16 weights inside the loop: the panel stores the 16 columns of a k step in
// the order [0..3, 8..11, 4..7, 12..15], so vpunpcklwd / vpunpckhwd of zero and the packed words (port 5, the load
// folded in) give the float32 columns 0..7 and 8..15 exactly. The products and the order of the sums are those of
// F32_Y: every result is bit-identical to the widen-then-F32_Y path of the generic AVX2 set.
//
// Measured on a Xeon E5-2698 v4 (operands in cache, one core): this assembly 97-109 GFLOP/s for bf16 against 81-96 for
// the same kernel from intrinsics (GCC loads the packed words into a register instead of folding them into the
// unpacks), and 84.5% against 77.2% of the FMA peak in the whole 4B GEMM on 10 cores. Other tile shapes for 16 ymm:
// 4 x 24 is no faster in fp32 and cannot widen bf16 in registers (12 accumulators + 3 B + broadcast + zero = 17);
// 12 x 8 needs 12 broadcasts per 12 FMAs and is load-port bound (66-80 GFLOP/s).
#include <immintrin.h>

#include "gemm_kernels.hpp"

#if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__) && !defined(SEPAL_NO_INLINE_ASM)
#define SEPAL_V3_ASM 1
#endif

namespace sepal::cpu {

namespace {

#if defined(SEPAL_V3_ASM)

// ---- GCC / Clang: inline assembly (AT&T) ------------------------------------------------------------------------

#define V3_LOAD_TILE                                                                                                         \
    "vmovaps   0(%[tile]), %%ymm0\n\t" "vmovaps  32(%[tile]), %%ymm1\n\t" "vmovaps  64(%[tile]), %%ymm2\n\t"                   \
    "vmovaps  96(%[tile]), %%ymm3\n\t" "vmovaps 128(%[tile]), %%ymm4\n\t" "vmovaps 160(%[tile]), %%ymm5\n\t"                   \
    "vmovaps 192(%[tile]), %%ymm6\n\t" "vmovaps 224(%[tile]), %%ymm7\n\t" "vmovaps 256(%[tile]), %%ymm8\n\t"                   \
    "vmovaps 288(%[tile]), %%ymm9\n\t" "vmovaps 320(%[tile]), %%ymm10\n\t" "vmovaps 352(%[tile]), %%ymm11\n\t"
#define V3_ZERO_TILE                                                                                                         \
    "vxorps %%ymm0, %%ymm0, %%ymm0\n\t" "vxorps %%ymm1, %%ymm1, %%ymm1\n\t" "vxorps %%ymm2, %%ymm2, %%ymm2\n\t"               \
    "vxorps %%ymm3, %%ymm3, %%ymm3\n\t" "vxorps %%ymm4, %%ymm4, %%ymm4\n\t" "vxorps %%ymm5, %%ymm5, %%ymm5\n\t"               \
    "vxorps %%ymm6, %%ymm6, %%ymm6\n\t" "vxorps %%ymm7, %%ymm7, %%ymm7\n\t" "vxorps %%ymm8, %%ymm8, %%ymm8\n\t"               \
    "vxorps %%ymm9, %%ymm9, %%ymm9\n\t" "vxorps %%ymm10, %%ymm10, %%ymm10\n\t" "vxorps %%ymm11, %%ymm11, %%ymm11\n\t"
#define V3_STORE_TILE                                                                                                        \
    "vmovaps %%ymm0,   0(%[tile])\n\t" "vmovaps %%ymm1,  32(%[tile])\n\t" "vmovaps %%ymm2,  64(%[tile])\n\t"                   \
    "vmovaps %%ymm3,  96(%[tile])\n\t" "vmovaps %%ymm4, 128(%[tile])\n\t" "vmovaps %%ymm5, 160(%[tile])\n\t"                   \
    "vmovaps %%ymm6, 192(%[tile])\n\t" "vmovaps %%ymm7, 224(%[tile])\n\t" "vmovaps %%ymm8, 256(%[tile])\n\t"                   \
    "vmovaps %%ymm9, 288(%[tile])\n\t" "vmovaps %%ymm10, 320(%[tile])\n\t" "vmovaps %%ymm11, 352(%[tile])\n\t"

// Six rows of one k step: A value r of the step is at A_OFF + 4 r bytes; B columns 0..7 in ymm12, 8..15 in ymm13.
#define V3_FMAS(A_OFF)                                                                                                       \
    "vbroadcastss " #A_OFF "+0(%[a]), %%ymm14\n\t"  "vfmadd231ps %%ymm12, %%ymm14, %%ymm0\n\t"  "vfmadd231ps %%ymm13, %%ymm14, %%ymm1\n\t"  \
    "vbroadcastss " #A_OFF "+4(%[a]), %%ymm14\n\t"  "vfmadd231ps %%ymm12, %%ymm14, %%ymm2\n\t"  "vfmadd231ps %%ymm13, %%ymm14, %%ymm3\n\t"  \
    "vbroadcastss " #A_OFF "+8(%[a]), %%ymm14\n\t"  "vfmadd231ps %%ymm12, %%ymm14, %%ymm4\n\t"  "vfmadd231ps %%ymm13, %%ymm14, %%ymm5\n\t"  \
    "vbroadcastss " #A_OFF "+12(%[a]), %%ymm14\n\t" "vfmadd231ps %%ymm12, %%ymm14, %%ymm6\n\t"  "vfmadd231ps %%ymm13, %%ymm14, %%ymm7\n\t"  \
    "vbroadcastss " #A_OFF "+16(%[a]), %%ymm14\n\t" "vfmadd231ps %%ymm12, %%ymm14, %%ymm8\n\t"  "vfmadd231ps %%ymm13, %%ymm14, %%ymm9\n\t"  \
    "vbroadcastss " #A_OFF "+20(%[a]), %%ymm14\n\t" "vfmadd231ps %%ymm12, %%ymm14, %%ymm10\n\t" "vfmadd231ps %%ymm13, %%ymm14, %%ymm11\n\t"

// One k step of BF16_Y3 (B: 32 bytes per step) and of F32_Y3 (B: 64 bytes per step).
#define V3_BF16_STEP(A_OFF, B_OFF)                                                                                           \
    "vpunpcklwd " #B_OFF "(%[b]), %%ymm15, %%ymm12\n\t" "vpunpckhwd " #B_OFF "(%[b]), %%ymm15, %%ymm13\n\t" V3_FMAS(A_OFF)
#define V3_F32_STEP(A_OFF, B_OFF)                                                                                            \
    "vmovaps " #B_OFF "(%[b]), %%ymm12\n\t" "vmovaps " #B_OFF "+32(%[b]), %%ymm13\n\t" V3_FMAS(A_OFF)

#define V3_CLOBBERS                                                                                                          \
    "cc", "memory", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7", "xmm8", "xmm9", "xmm10", "xmm11",       \
        "xmm12", "xmm13", "xmm14", "xmm15"

void kernel_bf16_y3(const KernelArgs& args) {
    const float* a = static_cast<const float*>(args.a);
    const unsigned char* b = static_cast<const unsigned char*>(args.b);
    float* tile = args.tile;
    std::size_t quads = args.kgroups / 4;
    std::size_t rest = args.kgroups % 4;
    const std::size_t accumulate = args.accumulate ? 1 : 0;
    __asm__ volatile(
        "vpxor %%ymm15, %%ymm15, %%ymm15\n\t"
        "test %[acc], %[acc]\n\t"
        "jz 1f\n\t"
        V3_LOAD_TILE
        "jmp 2f\n\t"
        "1:\n\t"
        V3_ZERO_TILE
        "2:\n\t"
        "test %[quads], %[quads]\n\t"
        "jz 4f\n\t"
        ".p2align 5\n\t"
        "3:\n\t"
        "prefetcht0 384(%[a])\n\t"
        V3_BF16_STEP(0, 0)
        V3_BF16_STEP(24, 32)
        V3_BF16_STEP(48, 64)
        V3_BF16_STEP(72, 96)
        "add $96, %[a]\n\t"
        "add $128, %[b]\n\t"
        "dec %[quads]\n\t"
        "jnz 3b\n\t"
        "4:\n\t"
        "test %[rest], %[rest]\n\t"
        "jz 6f\n\t"
        "5:\n\t"
        V3_BF16_STEP(0, 0)
        "add $24, %[a]\n\t"
        "add $32, %[b]\n\t"
        "dec %[rest]\n\t"
        "jnz 5b\n\t"
        "6:\n\t"
        V3_STORE_TILE
        "vzeroupper\n\t"
        : [a] "+r"(a), [b] "+r"(b), [quads] "+r"(quads), [rest] "+r"(rest)
        : [tile] "r"(tile), [acc] "r"(accumulate)
        : V3_CLOBBERS);
}

void kernel_f32_y3(const KernelArgs& args) {
    const float* a = static_cast<const float*>(args.a);
    const unsigned char* b = static_cast<const unsigned char*>(args.b);
    float* tile = args.tile;
    std::size_t quads = args.kgroups / 4;
    std::size_t rest = args.kgroups % 4;
    const std::size_t accumulate = args.accumulate ? 1 : 0;
    __asm__ volatile(
        "test %[acc], %[acc]\n\t"
        "jz 1f\n\t"
        V3_LOAD_TILE
        "jmp 2f\n\t"
        "1:\n\t"
        V3_ZERO_TILE
        "2:\n\t"
        "test %[quads], %[quads]\n\t"
        "jz 4f\n\t"
        ".p2align 5\n\t"
        "3:\n\t"
        "prefetcht0 384(%[a])\n\t"
        V3_F32_STEP(0, 0)
        V3_F32_STEP(24, 64)
        V3_F32_STEP(48, 128)
        V3_F32_STEP(72, 192)
        "add $96, %[a]\n\t"
        "add $256, %[b]\n\t"
        "dec %[quads]\n\t"
        "jnz 3b\n\t"
        "4:\n\t"
        "test %[rest], %[rest]\n\t"
        "jz 6f\n\t"
        "5:\n\t"
        V3_F32_STEP(0, 0)
        "add $24, %[a]\n\t"
        "add $64, %[b]\n\t"
        "dec %[rest]\n\t"
        "jnz 5b\n\t"
        "6:\n\t"
        V3_STORE_TILE
        "vzeroupper\n\t"
        : [a] "+r"(a), [b] "+r"(b), [quads] "+r"(quads), [rest] "+r"(rest)
        : [tile] "r"(tile), [acc] "r"(accumulate)
        : V3_CLOBBERS);
}

#else

// ---- MSVC (no x64 inline assembly) and other compilers: intrinsics, the same operations in the same order ----------

// Named accumulators (an `__m256 c[12]` array can be kept in memory by the compiler, which stores it every step).
#define V3I_DECL(r)                                                                                    \
    __m256 c##r##0 = args.accumulate ? _mm256_load_ps(tile + (r) * 16) : _mm256_setzero_ps();           \
    __m256 c##r##1 = args.accumulate ? _mm256_load_ps(tile + (r) * 16 + 8) : _mm256_setzero_ps();
#define V3I_ROW(r)                                                                                     \
    {                                                                                                  \
        const __m256 ar = _mm256_broadcast_ss(pa + (r));                                               \
        c##r##0 = _mm256_fmadd_ps(ar, b0, c##r##0);                                                    \
        c##r##1 = _mm256_fmadd_ps(ar, b1, c##r##1);                                                    \
    }
#define V3I_STEP                                                                                       \
    {                                                                                                  \
        __m256 b0, b1;                                                                                 \
        if constexpr (BF16) {                                                                          \
            const __m256i raw = _mm256_load_si256(reinterpret_cast<const __m256i*>(pb));               \
            b0 = _mm256_castsi256_ps(_mm256_unpacklo_epi16(_mm256_setzero_si256(), raw));              \
            b1 = _mm256_castsi256_ps(_mm256_unpackhi_epi16(_mm256_setzero_si256(), raw));              \
        } else {                                                                                       \
            b0 = _mm256_load_ps(reinterpret_cast<const float*>(pb));                                   \
            b1 = _mm256_load_ps(reinterpret_cast<const float*>(pb) + 8);                               \
        }                                                                                              \
        V3I_ROW(0) V3I_ROW(1) V3I_ROW(2) V3I_ROW(3) V3I_ROW(4) V3I_ROW(5)                              \
        pa += 6;                                                                                       \
        pb += b_step;                                                                                  \
    }
#define V3I_STORE(r)                                                                                   \
    _mm256_store_ps(tile + (r) * 16, c##r##0);                                                         \
    _mm256_store_ps(tile + (r) * 16 + 8, c##r##1);

template <bool BF16>
void kernel_v3(const KernelArgs& args) {
    const float* pa = static_cast<const float*>(args.a);
    const unsigned char* pb = static_cast<const unsigned char*>(args.b);
    constexpr std::size_t b_step = BF16 ? 32 : 64;
    float* tile = args.tile;
    V3I_DECL(0) V3I_DECL(1) V3I_DECL(2) V3I_DECL(3) V3I_DECL(4) V3I_DECL(5)
    std::size_t k = 0;
    for (; k + 4 <= args.kgroups; k += 4) {
        _mm_prefetch(reinterpret_cast<const char*>(pa) + 384, _MM_HINT_T0);
        V3I_STEP V3I_STEP V3I_STEP V3I_STEP
    }
    for (; k < args.kgroups; ++k) V3I_STEP
    V3I_STORE(0) V3I_STORE(1) V3I_STORE(2) V3I_STORE(3) V3I_STORE(4) V3I_STORE(5)
    _mm256_zeroupper();
}

void kernel_bf16_y3(const KernelArgs& args) { kernel_v3<true>(args); }
void kernel_f32_y3(const KernelArgs& args) { kernel_v3<false>(args); }

#endif

} // namespace

KernelFn kernel_avx2_v3(Variant v) {
    switch (v) {
    case Variant::F32_Y3: return &kernel_f32_y3;
    case Variant::BF16_Y3: return &kernel_bf16_y3;
    default: return nullptr;
    }
}

} // namespace sepal::cpu
