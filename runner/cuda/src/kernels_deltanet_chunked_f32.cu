// The float32 SIMT chunked Gated DeltaNet kernels; see kernels_deltanet_chunked_f32.hpp for the algorithm and the numbers.
//
// Layout of the work. All matrices of a (chunk, value head) tile are 64 tokens by 128 features, kept row-major (feature index
// fastest) so that a thread can read four consecutive features with one 128-bit access. Every product is computed by a
// register tile of 4 x 4 outputs per thread (two such tiles sharing an operand in the state kernel), fed from shared memory with
// 128-bit loads; the operands are staged through small panels of shared memory, and no kernel uses more than 48 KB of it. The
// layouts a product can use for its operands are written in the comments as A[m][k] (k fastest: "k-minor") or A[k][m]
// ("k-major"). Measured on an RTX 3060, one LDS.128 per 10 multiply-adds, as these tiles need, costs about a third of the
// float32 peak (the shared-memory pipe, not the arithmetic, limits the products); 8 x 8 tiles would need 64 accumulators per
// thread and 64-thread blocks, which the 48 KB of shared memory per block does not leave room for.
//
//   conv:     one block per (chunk, vector): the q or k of a key head, or the v of a value head. 64 tokens by 128 channels.
//   prepare:  one block per (chunk, value head), 256 threads: the gates, K K^T and Q K^T (k-major panels), T (one thread per
//             column, the column in registers), W = T (beta gamma K) and U = T (beta V) (T k-minor, X k-major).
//   state:    one block per (row, value head, 32 of the 128 value columns), 128 threads, the state slice [128][32] and the
//             chunk's Y [64][32] in shared memory; the serial loop over the row's chunks.
//   norm:     one block per (chunk, value head), in place on the output matrix.
#include "kernels_deltanet_chunked_f32.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "device.hpp"

namespace sepal::cuda {
namespace chunked_f32 {

constexpr int kC = 64;        // tokens per chunk
constexpr int kD = 128;       // head dimensions are padded to this inside
constexpr int kSplit = 4;     // slices of the value columns run by separate blocks of the state kernel
constexpr int kSlice = kD / kSplit;
constexpr int kMaxTaps = static_cast<int>(kDeltaNetMaxConvKernel);
constexpr int kScStride = 132; // floats of per-tile scalars: gamma[64], kdec[64], gend, 3 spare
static_assert(kC == static_cast<int>(kDeltaNetChunk), "the chunk length is the tensor-core version's");
static_assert(kScStride % 4 == 0, "16-byte aligned tiles");

// ---- scalar functions: the scan kernel's formulas ----

__device__ inline float softplus_f(float x) { return __fadd_rn(fmaxf(x, 0.0f), log1pf(expf(-fabsf(x)))); }
__device__ __forceinline__ float reciprocal(float d) {
    float r;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(d));
    return __fmaf_rn(__fmaf_rn(-d, r, 1.0f), r, r);
}
__device__ __forceinline__ float divide_by(float x, float d, float recip) {
    const float q = __fmul_rn(x, recip);
    return __fmaf_rn(__fmaf_rn(-d, q, x), recip, q);
}
__device__ __forceinline__ float silu_f(float x) {
    const float d = fminf(__fadd_rn(1.0f, expf(-x)), 1e38f);
    return divide_by(x, d, reciprocal(d));
}
__device__ inline float sigmoid_f(float x) { return expf(-softplus_f(-x)); }
__device__ inline float warp_sum(float x) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) x = __fadd_rn(x, __shfl_xor_sync(0xFFFFFFFFu, x, offset));
    return x;
}

// ---- state storage: float32, or binary16 (computation stays float32: widen on load, round on store) ----

__device__ __forceinline__ float load_state(const void* base, std::size_t index, std::true_type) {
    return __half2float(static_cast<const __half*>(base)[index]);
}
__device__ __forceinline__ float load_state(const void* base, std::size_t index, std::false_type) {
    return static_cast<const float*>(base)[index];
}
__device__ __forceinline__ void store_state(void* base, std::size_t index, float value, unsigned* saturations, std::true_type) {
    constexpr float kMax = 65504.0f;
    float v = value;
    if (v > kMax || v < -kMax) {
        v = v > 0.0f ? kMax : -kMax;
        if (saturations != nullptr) atomicAdd(saturations, 1u);
    }
    static_cast<__half*>(base)[index] = __float2half_rn(v);
}
__device__ __forceinline__ void store_state(void* base, std::size_t index, float value, unsigned*, std::false_type) {
    static_cast<float*>(base)[index] = value;
}

// ---- helpers ----

__device__ __forceinline__ float4 ld4(const float* p) { return *reinterpret_cast<const float4*>(p); }
__device__ __forceinline__ void st4(float* p, float4 v) { *reinterpret_cast<float4*>(p) = v; }
__device__ __forceinline__ void load4(float (&d)[4], const float* p) {
    const float4 v = ld4(p);
    d[0] = v.x;
    d[1] = v.y;
    d[2] = v.z;
    d[3] = v.w;
}
__device__ __forceinline__ float4 make4(const float (&v)[4]) { return make_float4(v[0], v[1], v[2], v[3]); }

// The (wave position, chunk) of a linear chunk number.
__device__ __forceinline__ void locate(const std::uint32_t* chunk_base, std::uint32_t wave_size, std::uint32_t lin,
                                       std::uint32_t& pos, std::uint32_t& j) {
    std::uint32_t lo = 0, hi = wave_size - 1;
    while (lo < hi) {
        const std::uint32_t mid = (lo + hi + 1) / 2;
        if (chunk_base[mid] <= lin) lo = mid;
        else hi = mid - 1;
    }
    pos = lo;
    j = lin - chunk_base[lo];
}

struct Args {
    std::uint32_t key_heads, value_heads, repeat, taps, dk, dv, wave_size;
    float read_scale, l2_eps, norm_eps;
    const float *conv_w, *a_log, *dt_bias, *norm_w;
    const float *mixed, *z, *b, *a;
    std::size_t mixed_stride, z_stride, b_stride, a_stride;
    const RowSpan* rows;
    const std::uint32_t *in_slots, *out_slots, *wave, *chunk_base;
    void* conv_tails;
    void* recurrent;
    unsigned* saturations;
    // workspace
    float* act; // [tokens][(2 key heads + value heads) * 128]: normalised q (times the read scale), normalised k, v
    std::size_t act_stride;
    float *Wt, *Ut, *Mt, *sc; // per (chunk, value head) tile: W [64][128], U [64][128], M [64][64], scalars [kScStride]
    float* out;
    std::size_t out_stride;
    bool vec_out; // out and out_stride allow 16-byte stores of four value columns
};

// ================= kernel 1: convolution, SiLU, L2 norm =================
//
// A block is one (chunk, vector): job = blockIdx.y runs over the q of each key head, the k of each key head, the v of each
// value head, in the order of the channels of `mixed`. Each warp takes 16 tokens, four at a time; a lane owns the channels
// lane, lane + 32, lane + 64, lane + 96 (those past the head dimension are zero). The taps are applied in the scan kernel's
// order; the result goes to the scratch `act` at the vector's 128-wide slot, zero-padded.

template <bool HS>
__global__ void __launch_bounds__(128) conv_kernel(const Args args) {
    constexpr std::integral_constant<bool, HS> kHalfState{};
    const std::uint32_t tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const std::uint32_t job = blockIdx.y;
    std::uint32_t pos, j;
    locate(args.chunk_base, args.wave_size, blockIdx.x, pos, j);
    const std::uint32_t row = args.wave[pos];
    const RowSpan span = args.rows[row];
    const std::uint32_t in_slot = args.in_slots[row];
    const std::uint32_t c0 = j * kC;
    const std::uint32_t n = min(static_cast<std::uint32_t>(kC), span.length - c0);
    const std::uint32_t taps = args.taps;
    const std::uint32_t kh = args.key_heads;
    const bool is_v = job >= 2 * kh;
    const bool is_q = job < kh;
    const std::uint32_t dim = is_v ? args.dv : args.dk;
    const std::uint32_t ch_base = is_v ? 2 * kh * args.dk + (job - 2 * kh) * args.dv : job * args.dk;
    const std::uint32_t conv_dim = 2 * kh * args.dk + args.value_heads * args.dv;

    constexpr int kTok = 4;
    constexpr int kRows = kTok + kMaxTaps - 1;
    float wt[4][kMaxTaps];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const std::uint32_t ch = lane + 32u * static_cast<std::uint32_t>(q);
#pragma unroll
        for (int k = 0; k < kMaxTaps; ++k) {
            wt[q][k] = (static_cast<std::uint32_t>(k) < taps && ch < dim)
                           ? args.conv_w[static_cast<std::size_t>(ch_base + ch) * taps + static_cast<std::uint32_t>(k)] : 0.0f;
        }
    }
    float* dst = args.act + job * kD + lane;

    for (std::uint32_t it = 0; it < 4; ++it) {
        const std::uint32_t t_first = warp * 16 + it * 4;
        if (t_first >= n) break; // uniform across the warp
        float x[kRows][4];
        if (c0 + t_first >= taps - 1) {
            // every row of the window is an input of this row
            const float* src = args.mixed + static_cast<std::size_t>(span.offset + c0 + t_first - (taps - 1)) * args.mixed_stride + ch_base + lane;
#pragma unroll
            for (int r = 0; r < kRows; ++r) {
                const bool live = static_cast<std::uint32_t>(r) < kTok + taps - 1 && t_first + static_cast<std::uint32_t>(r) < n + taps - 1;
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    x[r][q] = (live && lane + 32u * static_cast<std::uint32_t>(q) < dim)
                                  ? src[static_cast<std::size_t>(r) * args.mixed_stride + 32 * q] : 0.0f;
                }
            }
        } else {
#pragma unroll
            for (int r = 0; r < kRows; ++r) {
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    float v = 0.0f;
                    const std::uint32_t ch = lane + 32u * static_cast<std::uint32_t>(q);
                    if (static_cast<std::uint32_t>(r) < kTok + taps - 1 && t_first + static_cast<std::uint32_t>(r) < n + taps - 1 && ch < dim) {
                        const std::uint32_t w = c0 + t_first + static_cast<std::uint32_t>(r); // position in (tail, inputs)
                        if (w < taps - 1) {
                            if (in_slot != kNoSlot) {
                                v = load_state(args.conv_tails, (static_cast<std::size_t>(in_slot) * (taps - 1) + w) * conv_dim + ch_base + ch, kHalfState);
                            }
                        } else {
                            v = args.mixed[static_cast<std::size_t>(span.offset + w - (taps - 1)) * args.mixed_stride + ch_base + ch];
                        }
                    }
                    x[r][q] = v;
                }
            }
        }
#pragma unroll
        for (int u = 0; u < kTok; ++u) {
            const std::uint32_t tok = t_first + static_cast<std::uint32_t>(u);
            float v[4];
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                float acc = 0.0f;
#pragma unroll
                for (int k = 0; k < kMaxTaps; ++k) {
                    if (static_cast<std::uint32_t>(k) < taps) acc = __fmaf_rn(wt[q][k], x[u + k][q], acc);
                }
                v[q] = silu_f(acc);
            }
            if (!is_v) {
                float sum = 0.0f;
#pragma unroll
                for (int q = 0; q < 4; ++q) sum = __fmaf_rn(v[q], v[q], sum);
                sum = warp_sum(sum);
                const float norm = sqrtf(__fadd_rn(sum, args.l2_eps));
                const float rn = reciprocal(norm);
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    v[q] = divide_by(v[q], norm, rn);
                    if (is_q) v[q] = __fmul_rn(v[q], args.read_scale);
                }
            }
            if (tok < n) {
                float* d = dst + static_cast<std::size_t>(span.offset + c0 + tok) * args.act_stride;
#pragma unroll
                for (int q = 0; q < 4; ++q) d[32 * q] = v[q];
            }
        }
    }
}

// ================= kernel 2: prepare =================

constexpr int kPrepThreads = 256;
constexpr int kLdT = 68; // leading dimension of the [64][64] tiles in shared memory (a multiple of 4: 16-byte rows)

// Two blocks per multiprocessor: at most 128 registers per thread (no spills; the column of T is the largest consumer).
__global__ void __launch_bounds__(kPrepThreads, 2) prepare_kernel(const Args args) {
    __shared__ __align__(16) float R[kC * kLdT]; // K^T and Q^T panels, then A, then the X panels
    __shared__ __align__(16) float Ts[kC * kLdT]; // T, row-major, zero above the diagonal
    __shared__ float sg[kC], sCum[kC], sBeta[kC], sGamma[kC];

    const std::uint32_t tid = threadIdx.x;
    const std::uint32_t head = blockIdx.y;
    std::uint32_t pos, j;
    locate(args.chunk_base, args.wave_size, blockIdx.x, pos, j);
    const RowSpan span = args.rows[args.wave[pos]];
    const std::uint32_t c0 = j * kC;
    const std::uint32_t n = min(static_cast<std::uint32_t>(kC), span.length - c0);
    const std::uint32_t kv = head / args.repeat;
    const std::size_t tile = static_cast<std::size_t>(blockIdx.x) * args.value_heads + head;
    const float* const act0 = args.act + static_cast<std::size_t>(span.offset + c0) * args.act_stride;
    const float* const act_q = act0 + static_cast<std::size_t>(kv) * kD;
    const float* const act_k = act0 + static_cast<std::size_t>(args.key_heads + kv) * kD;
    const float* const act_v = act0 + static_cast<std::size_t>(2 * args.key_heads + head) * kD;
    float* const Wg = args.Wt + tile * (kC * kD);
    float* const Ug = args.Ut + tile * (kC * kD);
    float* const Mg = args.Mt + tile * (kC * kC);
    float* const scg = args.sc + tile * kScStride;

    // ---- gates: g_i, beta_i, G_i = g_0 + ... + g_i, gamma_i = exp(G_i) ----
    if (tid < kC) {
        float g = 0.0f, bt = 0.0f;
        if (tid < n) {
            const std::size_t token = span.offset + c0 + tid;
            const float a_in = args.a[token * args.a_stride + head];
            const float b_in = args.b[token * args.b_stride + head];
            g = __fmul_rn(-expf(args.a_log[head]), softplus_f(__fadd_rn(a_in, args.dt_bias[head])));
            // exp(g) is 0 in float32 below about -103.97, so clamping at -110 changes no decay; it keeps G finite, so that G_i - G_j is
            // never -inf - (-inf) = NaN when exp(A_log) overflows. A NaN g stays NaN (the comparison is false), as in the scan.
            if (g < -110.0f) g = -110.0f;
            bt = sigmoid_f(b_in);
        }
        sg[tid] = g;
        sBeta[tid] = bt;
    }
    __syncthreads();
    if (tid == 0) {
        float c = 0.0f;
        for (int t = 0; t < kC; ++t) {
            if (static_cast<std::uint32_t>(t) < n) c = __fadd_rn(c, sg[t]);
            sCum[t] = c;
        }
    }
    __syncthreads();
    if (tid < kC) {
        const float last = sCum[n - 1];
        const float gamma = tid < n ? expf(sCum[tid]) : 0.0f;
        sGamma[tid] = gamma;
        scg[tid] = gamma;
        scg[kC + tid] = tid < n ? expf(__fsub_rn(last, sCum[tid])) : 0.0f;
        if (tid == 0) {
            scg[2 * kC] = expf(last);
            scg[2 * kC + 1] = 0.0f;
            scg[2 * kC + 2] = 0.0f;
            scg[2 * kC + 3] = 0.0f;
        }
    }

    // ---- K K^T and Q K^T over the 128 features, panels of 32 (k-major panels: [feature][token]) ----
    // 136 threads own the 4 x 4 tiles (ti, tj) with tj <= ti of the 16 x 16 grid of tiles.
    float accA[4][4], accM[4][4];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            accA[r][q] = 0.0f;
            accM[r][q] = 0.0f;
        }
    }
    int ti = 0, tj = 0;
    const bool lower_tile = tid < 136;
    if (lower_tile) {
        while ((ti + 1) * (ti + 2) / 2 <= static_cast<int>(tid)) ++ti;
        tj = static_cast<int>(tid) - ti * (ti + 1) / 2;
    }
    float* const Kt = R;
    float* const Qt = R + 32 * kLdT;
    for (int p = 0; p < 4; ++p) {
#pragma unroll
        for (int u = 0; u < 2; ++u) {
            const std::uint32_t e = tid + 256u * static_cast<std::uint32_t>(u);
            const std::uint32_t t = e >> 3, f4 = e & 7;
            float4 kk = make_float4(0.0f, 0.0f, 0.0f, 0.0f), qq = kk;
            if (t < n) {
                kk = ld4(act_k + static_cast<std::size_t>(t) * args.act_stride + 32 * p + 4 * f4);
                qq = ld4(act_q + static_cast<std::size_t>(t) * args.act_stride + 32 * p + 4 * f4);
            }
            Kt[(4 * f4 + 0) * kLdT + t] = kk.x;
            Kt[(4 * f4 + 1) * kLdT + t] = kk.y;
            Kt[(4 * f4 + 2) * kLdT + t] = kk.z;
            Kt[(4 * f4 + 3) * kLdT + t] = kk.w;
            Qt[(4 * f4 + 0) * kLdT + t] = qq.x;
            Qt[(4 * f4 + 1) * kLdT + t] = qq.y;
            Qt[(4 * f4 + 2) * kLdT + t] = qq.z;
            Qt[(4 * f4 + 3) * kLdT + t] = qq.w;
        }
        __syncthreads();
        if (lower_tile) {
#pragma unroll 4
            for (int f = 0; f < 32; ++f) {
                float ak[4], aq[4], b[4];
                load4(ak, Kt + f * kLdT + 4 * ti);
                load4(aq, Qt + f * kLdT + 4 * ti);
                load4(b, Kt + f * kLdT + 4 * tj);
#pragma unroll
                for (int r = 0; r < 4; ++r) {
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        accA[r][q] = __fmaf_rn(ak[r], b[q], accA[r][q]);
                        accM[r][q] = __fmaf_rn(aq[r], b[q], accM[r][q]);
                    }
                }
            }
        }
        __syncthreads();
    }

    // ---- A = diag(beta) tril(Gam KK, -1) into R (row-major), M = tril(Gam QK) to the workspace ----
    if (lower_tile) {
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const std::uint32_t i = 4u * static_cast<std::uint32_t>(ti) + static_cast<std::uint32_t>(r);
            float av[4], mv[4];
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const std::uint32_t jj = 4u * static_cast<std::uint32_t>(tj) + static_cast<std::uint32_t>(q);
                float a = 0.0f, m = 0.0f;
                if (i < n && jj <= i) {
                    const float gam = expf(__fsub_rn(sCum[i], sCum[jj]));
                    m = __fmul_rn(gam, accM[r][q]);
                    if (jj < i) a = __fmul_rn(__fmul_rn(sBeta[i], gam), accA[r][q]);
                }
                av[q] = a;
                mv[q] = m;
            }
            st4(R + i * kLdT + 4 * tj, make4(av));
            st4(Mg + i * kC + 4 * tj, make4(mv));
        }
    } else {
        // the 120 tiles above the diagonal: zeros (tile (b, a), b < a)
        const int m = static_cast<int>(tid) - 136;
        int a = 1;
        while (a * (a + 1) / 2 <= m) ++a;
        const int b = m - a * (a - 1) / 2;
        const float4 zero = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            st4(R + (4 * b + r) * kLdT + 4 * a, zero);
            st4(Mg + (4 * b + r) * kC + 4 * a, zero);
        }
    }
    __syncthreads();

    // ---- T = (I + A)^-1: thread jc solves column jc, row by row, l increasing; the column stays in registers ----
    if (tid < kC) {
        const int jc = static_cast<int>(tid);
        float tc[kC];
#pragma unroll
        for (int i = 0; i < kC; ++i) tc[i] = 0.0f;
#pragma unroll
        for (int i = 0; i < kC; ++i) {
            float s = 0.0f;
#pragma unroll
            for (int l4 = 0; l4 * 4 < i; ++l4) {
                float a4[4];
                load4(a4, R + i * kLdT + 4 * l4); // A(i, l) = 0 for l >= i; tc[l] = 0 for l >= i here
                s = __fmaf_rn(a4[0], tc[4 * l4 + 0], s);
                s = __fmaf_rn(a4[1], tc[4 * l4 + 1], s);
                s = __fmaf_rn(a4[2], tc[4 * l4 + 2], s);
                s = __fmaf_rn(a4[3], tc[4 * l4 + 3], s);
            }
            tc[i] = i < jc ? 0.0f : (i == jc ? 1.0f : -s);
        }
#pragma unroll
        for (int i = 0; i < kC; ++i) Ts[i * kLdT + jc] = tc[i];
    }
    __syncthreads();

    // ---- W = T diag(beta gamma) K and U = T diag(beta) V, in panels of 64 features: X is k-major [token][feature] ----
    {
        const int tr = static_cast<int>(tid >> 4), tf = static_cast<int>(tid & 15); // output tile: rows 4 tr.., features 4 tf..
        for (int prod = 0; prod < 4; ++prod) {
            const bool is_u = prod >= 2;
            const int h = prod & 1;
            const float* const src = is_u ? act_v : act_k;
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const std::uint32_t e = tid + 256u * static_cast<std::uint32_t>(u);
                const std::uint32_t l = e >> 4, f4 = e & 15;
                float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                if (l < n) {
                    const float scale = is_u ? sBeta[l] : __fmul_rn(sBeta[l], sGamma[l]);
                    v = ld4(src + static_cast<std::size_t>(l) * args.act_stride + 64 * h + 4 * f4);
                    v.x = __fmul_rn(scale, v.x);
                    v.y = __fmul_rn(scale, v.y);
                    v.z = __fmul_rn(scale, v.z);
                    v.w = __fmul_rn(scale, v.w);
                }
                st4(R + l * kLdT + 4 * f4, v);
            }
            __syncthreads();
            float c[4][4];
#pragma unroll
            for (int r = 0; r < 4; ++r) {
#pragma unroll
                for (int q = 0; q < 4; ++q) c[r][q] = 0.0f;
            }
            for (int l4 = 0; l4 <= tr; ++l4) {
                float a[4][4];
#pragma unroll
                for (int r = 0; r < 4; ++r) load4(a[r], Ts + (4 * tr + r) * kLdT + 4 * l4);
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    float b[4];
                    load4(b, R + (4 * l4 + u) * kLdT + 4 * tf);
#pragma unroll
                    for (int r = 0; r < 4; ++r) {
#pragma unroll
                        for (int q = 0; q < 4; ++q) c[r][q] = __fmaf_rn(a[r][u], b[q], c[r][q]);
                    }
                }
            }
            float* const dst = is_u ? Ug : Wg;
#pragma unroll
            for (int r = 0; r < 4; ++r) st4(dst + (4 * tr + r) * kD + 64 * h + 4 * tf, make4(c[r]));
            __syncthreads();
        }
    }
}

// ================= kernel 3: the state loop =================
//
// One block per (row, value head, slice of 32 value columns); 128 threads; thread (tg = tid / 8, cg = tid % 8) owns the 4 x 4
// tile of rows 4 tg.. and columns 4 cg.. of the chunk's [64][32] matrices, and the state rows 4 tg.. and 4 (tg + 16)..
// (4 columns each). Per chunk:
//   (1) P = W S0 and O = Q S0 for the thread's rows over the 128 keys (W, Q k-minor panels of 32 keys, S k-major), then
//       O = gamma O;
//   (2) Y = U - P, to shared memory;
//   (3) O += M Y over the tokens j <= i (M k-minor panels of 32 tokens, Y k-major), the reads are stored;
//   (4) S = gamma_last S0 + Kd^T Y (Kd k-major panels of 32 tokens), Kd_j = Gam(last, j) k_j.
// The global loads of the next panel are issued before the multiply-adds of the current one.

constexpr int kLdP = 36;   // leading dimension of the k-minor panels [64 rows][32] in shared memory
constexpr int kLdK = 132;  // of the Kd panel [32 tokens][128]
constexpr int kPanelFloats = 2 * kC * kLdP; // W and Q panels (4608 floats) hold the larger of the three layouts

template <bool HS>
__global__ void __launch_bounds__(128) state_kernel(const Args args) {
    constexpr std::integral_constant<bool, HS> kHalfState{};
    __shared__ __align__(16) float Sm[kD * kSlice]; // S(key, column of the slice)
    __shared__ __align__(16) float Ym[kC * kSlice]; // Y(token, column of the slice)
    __shared__ __align__(16) float Pn[kPanelFloats];

    const std::uint32_t sp = blockIdx.x, head = blockIdx.y, pos = blockIdx.z;
    if (sp * kSlice >= args.dv) return; // no column of this slice exists (uniform for the block)
    const std::uint32_t tid = threadIdx.x;
    const std::uint32_t tg = tid >> 3, cg = tid & 7;
    const std::uint32_t i0 = 4 * tg, c0 = 4 * cg;
    const std::uint32_t row = args.wave[pos];
    const RowSpan span = args.rows[row];
    const std::uint32_t in_slot = args.in_slots[row], out_slot = args.out_slots[row];
    const std::uint32_t nchunks = (span.length + kC - 1) / kC;
    const std::uint32_t base = args.chunk_base != nullptr ? args.chunk_base[pos] : 0u;
    const std::uint32_t dk = args.dk, dv = args.dv;
    const std::size_t state_base_in = (static_cast<std::size_t>(in_slot) * args.value_heads + head) * dk * dv;
    const std::size_t state_base_out = (static_cast<std::size_t>(out_slot) * args.value_heads + head) * dk * dv;

    for (std::uint32_t e = tid; e < kD * kSlice; e += 128) {
        const std::uint32_t f = e / kSlice, col = sp * kSlice + e % kSlice;
        Sm[e] = (in_slot != kNoSlot && f < dk && col < dv) ? load_state(args.recurrent, state_base_in + f * dv + col, kHalfState) : 0.0f;
    }
    __syncthreads();

    const std::uint32_t kv = head / args.repeat;
    const std::size_t qoff = static_cast<std::size_t>(kv) * kD;
    const std::size_t koff = static_cast<std::size_t>(args.key_heads + kv) * kD;
    __shared__ float sKd[kC]; // Gam(last, j) of the chunk's tokens
    float* const Wp = Pn;                 // (1): W panel [64][kLdP], then Q panel
    float* const Qp = Pn + kC * kLdP;
    float* const Mp = Pn;                 // (3): M panel [64][kLdP]
    float* const Kp = Pn;                 // (4): Kd panel [32][kLdK]
    const float4 zero4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    const std::uint32_t kd_t = tid >> 5, kd_f = tid & 31; // the (token, float4) a thread stages of the Kd panel: token kd_t + 4 u

    float4 rg[8]; // staging registers: the global loads of the next panel wait here while the current one is multiplied
    auto tile_of = [&](std::uint32_t jj) { return (static_cast<std::size_t>(base) + jj) * args.value_heads + head; };
    auto act_row = [&](std::uint32_t jj) { return args.act + static_cast<std::size_t>(span.offset + jj * kC) * args.act_stride; };
    auto load_wq = [&](std::uint32_t jj, int p) {
        const float* const wg = args.Wt + tile_of(jj) * (kC * kD);
        const float* const aq = act_row(jj) + qoff;
        const std::uint32_t nn = min(static_cast<std::uint32_t>(kC), span.length - jj * kC);
#pragma unroll
        for (std::uint32_t u = 0; u < 4; ++u) {
            const std::uint32_t t = tg + 16 * u;
            rg[u] = ld4(wg + t * kD + 32 * p + 4 * cg);
            rg[4 + u] = t < nn ? ld4(aq + static_cast<std::size_t>(t) * args.act_stride + 32 * p + 4 * cg) : zero4;
        }
    };
    auto store_wq = [&]() {
#pragma unroll
        for (std::uint32_t u = 0; u < 4; ++u) {
            const std::uint32_t t = tg + 16 * u;
            st4(Wp + t * kLdP + 4 * cg, rg[u]);
            st4(Qp + t * kLdP + 4 * cg, rg[4 + u]);
        }
    };
    auto load_m = [&](const float* mg, int pp) {
#pragma unroll
        for (std::uint32_t u = 0; u < 4; ++u) rg[u] = ld4(mg + (tg + 16 * u) * kC + 32 * pp + 4 * cg);
    };
    auto store_m = [&]() {
#pragma unroll
        for (std::uint32_t u = 0; u < 4; ++u) st4(Mp + (tg + 16 * u) * kLdP + 4 * cg, rg[u]);
    };
    auto load_kd = [&](const float* ak, int pt, std::uint32_t nn) {
#pragma unroll
        for (std::uint32_t u = 0; u < 8; ++u) {
            const std::uint32_t tt = 32 * static_cast<std::uint32_t>(pt) + kd_t + 4 * u;
            rg[u] = tt < nn ? ld4(ak + static_cast<std::size_t>(tt) * args.act_stride + 4 * kd_f) : zero4;
        }
    };
    auto store_kd = [&](int pt) {
#pragma unroll
        for (std::uint32_t u = 0; u < 8; ++u) {
            const std::uint32_t t = kd_t + 4 * u;
            const float kd = sKd[32 * static_cast<std::uint32_t>(pt) + t];
            float4 v = rg[u];
            v.x = __fmul_rn(kd, v.x);
            v.y = __fmul_rn(kd, v.y);
            v.z = __fmul_rn(kd, v.z);
            v.w = __fmul_rn(kd, v.w);
            st4(Kp + t * kLdK + 4 * kd_f, v);
        }
    };

    if (nchunks != 0) load_wq(0, 0);
    for (std::uint32_t j = 0; j < nchunks; ++j) {
        const std::size_t tile = tile_of(j);
        const std::uint32_t n = min(static_cast<std::uint32_t>(kC), span.length - j * kC);
        const float* const Ug = args.Ut + tile * (kC * kD);
        const float* const Mg = args.Mt + tile * (kC * kC);
        const float* const scg = args.sc + tile * kScStride;
        const float* const actk = act_row(j) + koff;
        const float gend = scg[2 * kC];
        float gm[4];
        float4 u4[4];
#pragma unroll
        for (std::uint32_t r = 0; r < 4; ++r) {
            gm[r] = scg[i0 + r];
            u4[r] = ld4(Ug + (i0 + r) * kD + sp * kSlice + c0);
        }
        if (tid < kC) sKd[tid] = scg[kC + tid];

        // ---- (1) P = W S0, O = gamma (Q S0) ----
        float P[4][4], O[4][4];
#pragma unroll
        for (int r = 0; r < 4; ++r) {
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                P[r][q] = 0.0f;
                O[r][q] = 0.0f;
            }
        }
        for (int p = 0; p < 4; ++p) {
            store_wq();
            __syncthreads();
            if (p < 3) load_wq(j, p + 1);
            else load_m(Mg, 0);
#pragma unroll 4
            for (int kk4 = 0; kk4 < 8; ++kk4) {
                float aw[4][4], aq[4][4];
#pragma unroll
                for (std::uint32_t r = 0; r < 4; ++r) {
                    load4(aw[r], Wp + (i0 + r) * kLdP + 4 * kk4);
                    load4(aq[r], Qp + (i0 + r) * kLdP + 4 * kk4);
                }
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    float b[4];
                    load4(b, Sm + (32 * p + 4 * kk4 + u) * kSlice + c0);
#pragma unroll
                    for (int r = 0; r < 4; ++r) {
#pragma unroll
                        for (int q = 0; q < 4; ++q) {
                            P[r][q] = __fmaf_rn(aw[r][u], b[q], P[r][q]);
                            O[r][q] = __fmaf_rn(aq[r][u], b[q], O[r][q]);
                        }
                    }
                }
            }
            __syncthreads();
        }

        // ---- (2) Y = U - P ----
#pragma unroll
        for (std::uint32_t r = 0; r < 4; ++r) {
            const float uu[4] = {u4[r].x, u4[r].y, u4[r].z, u4[r].w};
            float y[4];
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                y[q] = (i0 + r < n) ? __fsub_rn(uu[q], P[r][q]) : 0.0f;
                O[r][q] = __fmul_rn(gm[r], O[r][q]);
            }
            st4(Ym + (i0 + r) * kSlice + c0, make4(y));
        }
        __syncthreads();

        // ---- (3) O += M Y (j <= i) ----
        const int npan = n > 32 ? 2 : 1;
        for (int pp = 0; pp < npan; ++pp) {
            store_m();
            __syncthreads();
            if (pp + 1 < npan) load_m(Mg, pp + 1);
            else load_kd(actk, 0, n);
#pragma unroll 2
            for (int kk4 = 0; kk4 < 8; ++kk4) {
                if (8 * pp + kk4 <= static_cast<int>(tg)) {
                    float am[4][4];
#pragma unroll
                    for (std::uint32_t r = 0; r < 4; ++r) load4(am[r], Mp + (i0 + r) * kLdP + 4 * kk4);
#pragma unroll
                    for (int u = 0; u < 4; ++u) {
                        float b[4];
                        load4(b, Ym + (32 * pp + 4 * kk4 + u) * kSlice + c0);
#pragma unroll
                        for (int r = 0; r < 4; ++r) {
#pragma unroll
                            for (int q = 0; q < 4; ++q) O[r][q] = __fmaf_rn(am[r][u], b[q], O[r][q]);
                        }
                    }
                }
            }
            __syncthreads();
        }
        // the reads, raw: the norm kernel finishes them in place
        {
            const std::uint32_t col = sp * kSlice + c0;
#pragma unroll
            for (std::uint32_t r = 0; r < 4; ++r) {
                if (i0 + r < n) {
                    float* const d = args.out + static_cast<std::size_t>(span.offset + j * kC + i0 + r) * args.out_stride + head * dv + col;
                    if (args.vec_out) {
                        if (col < dv) st4(d, make4(O[r]));
                    } else {
#pragma unroll
                        for (std::uint32_t q = 0; q < 4; ++q) {
                            if (col + q < dv) d[q] = O[r][q];
                        }
                    }
                }
            }
        }

        // ---- (4) S = gamma_last S0 + Kd^T Y ----
        float acc0[4][4], acc1[4][4];
#pragma unroll
        for (std::uint32_t r = 0; r < 4; ++r) {
            float s0[4], s1[4];
            load4(s0, Sm + (4 * tg + r) * kSlice + c0);
            load4(s1, Sm + (4 * (tg + 16) + r) * kSlice + c0);
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                acc0[r][q] = __fmul_rn(gend, s0[q]);
                acc1[r][q] = __fmul_rn(gend, s1[q]);
            }
        }
        for (int pt = 0; pt < npan; ++pt) {
            store_kd(pt);
            __syncthreads();
            if (pt + 1 < npan) load_kd(actk, pt + 1, n);
            else if (j + 1 < nchunks) load_wq(j + 1, 0);
            const int tcount = static_cast<int>(min(32u, n - 32u * static_cast<std::uint32_t>(pt)));
#pragma unroll 4
            for (int t = 0; t < tcount; ++t) {
                float a0[4], a1[4], b[4];
                load4(a0, Kp + t * kLdK + 4 * tg);
                load4(a1, Kp + t * kLdK + 4 * (tg + 16));
                load4(b, Ym + (32 * pt + t) * kSlice + c0);
#pragma unroll
                for (int r = 0; r < 4; ++r) {
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        acc0[r][q] = __fmaf_rn(a0[r], b[q], acc0[r][q]);
                        acc1[r][q] = __fmaf_rn(a1[r], b[q], acc1[r][q]);
                    }
                }
            }
            __syncthreads();
        }
#pragma unroll
        for (std::uint32_t r = 0; r < 4; ++r) {
            st4(Sm + (4 * tg + r) * kSlice + c0, make4(acc0[r]));
            st4(Sm + (4 * (tg + 16) + r) * kSlice + c0, make4(acc1[r]));
        }
        __syncthreads();
    }

    if (out_slot != kNoSlot) {
        for (std::uint32_t e = tid; e < kD * kSlice; e += 128) {
            const std::uint32_t f = e / kSlice, col = sp * kSlice + e % kSlice;
            if (f < dk && col < dv) store_state(args.recurrent, state_base_out + f * dv + col, Sm[e], args.saturations, kHalfState);
        }
    }
}

// ================= kernel 4: gated RMSNorm, in place on the raw reads =================

__global__ void __launch_bounds__(128) norm_kernel(const Args args) {
    const std::uint32_t tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const std::uint32_t head = blockIdx.y;
    std::uint32_t pos, j;
    locate(args.chunk_base, args.wave_size, blockIdx.x, pos, j);
    const RowSpan span = args.rows[args.wave[pos]];
    const std::uint32_t c0 = j * kC;
    const std::uint32_t n = min(static_cast<std::uint32_t>(kC), span.length - c0);
    const std::uint32_t dv = args.dv;
    float nw[4];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const std::uint32_t jj = lane + 32u * static_cast<std::uint32_t>(q);
        nw[q] = jj < dv ? args.norm_w[jj] : 0.0f;
    }
    for (std::uint32_t u = 0; u < 16; ++u) {
        const std::uint32_t t = warp * 16 + u;
        if (t >= n) break;
        const std::size_t token = span.offset + c0 + t;
        float* const o = args.out + token * args.out_stride + head * dv;
        const float* const zz = args.z + token * args.z_stride + head * dv;
        float x[4], sum = 0.0f;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const std::uint32_t jj = lane + 32u * static_cast<std::uint32_t>(q);
            x[q] = jj < dv ? o[jj] : 0.0f;
            sum = __fmaf_rn(x[q], x[q], sum);
        }
        sum = warp_sum(sum);
        const float rms = sqrtf(__fadd_rn(__fdiv_rn(sum, static_cast<float>(dv)), args.norm_eps));
        const float recip = reciprocal(rms);
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const std::uint32_t jj = lane + 32u * static_cast<std::uint32_t>(q);
            if (jj < dv) o[jj] = __fmul_rn(__fmul_rn(divide_by(x[q], rms, recip), nw[q]), silu_f(zz[jj]));
        }
    }
}

} // namespace chunked_f32

// ---- host side ----

bool deltanet_chunked_f32_supported(const DeltaNetGeometry& g) {
    return g.key_heads != 0 && g.value_heads != 0 && g.value_heads % g.key_heads == 0 && g.key_head_dim >= 1 &&
           g.key_head_dim <= chunked_f32::kD && g.value_head_dim >= 1 && g.value_head_dim <= chunked_f32::kD && g.conv_kernel >= 1 &&
           g.conv_kernel <= static_cast<std::uint32_t>(chunked_f32::kMaxTaps);
}

namespace {

std::size_t align_up(std::size_t v, std::size_t a) { return (v + a - 1) / a * a; }

struct Layout {
    std::size_t act, w, u, m, sc, total;
    std::size_t act_stride;
};

Layout layout(const DeltaNetGeometry& g, std::size_t chunks, std::size_t tokens) {
    using namespace chunked_f32;
    const std::size_t tiles = chunks * g.value_heads;
    Layout l{};
    l.act_stride = static_cast<std::size_t>(2 * g.key_heads + g.value_heads) * kD;
    std::size_t at = 0;
    l.act = at; at += align_up(tokens * l.act_stride * sizeof(float), 256);
    l.w = at; at += tiles * kC * kD * sizeof(float);
    l.u = at; at += tiles * kC * kD * sizeof(float);
    l.m = at; at += tiles * kC * kC * sizeof(float);
    l.sc = at; at += align_up(tiles * kScStride * sizeof(float), 256);
    l.total = at;
    return l;
}

} // namespace

std::size_t deltanet_chunked_f32_workspace_bytes(const DeltaNetGeometry& g, std::size_t chunks, std::size_t tokens) {
    if (!deltanet_chunked_f32_supported(g)) return 0;
    return layout(g, chunks, tokens).total;
}

void launch_deltanet_chunked_f32(const DeltaNetGeometry& g, const DeltaNetWeights& w, const DeltaNetInputs& in, const DeltaNetRows& rows,
                                 const DeltaNetChunkPlan& plan, const DeltaNetStates& states, void* workspace, std::size_t workspace_bytes,
                                 std::size_t workspace_tokens, float* out, std::size_t out_stride, cudaStream_t stream) {
    using namespace chunked_f32;
    if (!deltanet_chunked_f32_supported(g)) throw std::invalid_argument("deltanet chunked f32: unsupported geometry");
    if (rows.wave_size == 0) return;
    if (rows.rows == nullptr || rows.in_slots == nullptr || rows.out_slots == nullptr || rows.wave == nullptr) {
        throw std::invalid_argument("deltanet chunked f32: a wave needs the row, slot and wave arrays");
    }
    if (rows.wave_size > 65535u) throw std::invalid_argument("deltanet chunked f32: a wave of more than 65535 rows");
    if (2 * g.key_heads + g.value_heads > 65535u || g.value_heads > 65535u) {
        throw std::invalid_argument("deltanet chunked f32: more than 65535 heads");
    }
    if (w.conv == nullptr || w.a_log == nullptr || w.dt_bias == nullptr || w.norm == nullptr || in.mixed.data == nullptr || in.z.data == nullptr ||
        in.b.data == nullptr || in.a.data == nullptr || states.recurrent == nullptr || out == nullptr ||
        (plan.total_chunks != 0 && plan.chunk_base == nullptr) || (g.conv_kernel > 1 && states.conv_tails == nullptr)) {
        throw std::invalid_argument("deltanet chunked f32: a null pointer");
    }
    const Layout l = layout(g, plan.total_chunks, workspace_tokens);
    if (plan.total_chunks != 0 && (workspace == nullptr || l.total > workspace_bytes)) {
        throw std::invalid_argument("deltanet chunked f32: the workspace is too small");
    }
    if (plan.total_chunks != 0 && reinterpret_cast<std::uintptr_t>(workspace) % 256 != 0) {
        throw std::invalid_argument("deltanet chunked f32: the workspace must be 256-byte aligned (as cudaMalloc returns it)");
    }
    char* ws = static_cast<char*>(workspace);
    Args a{};
    a.key_heads = g.key_heads;
    a.value_heads = g.value_heads;
    a.repeat = g.value_heads / g.key_heads;
    a.taps = g.conv_kernel;
    a.dk = g.key_head_dim;
    a.dv = g.value_head_dim;
    a.wave_size = rows.wave_size;
    a.read_scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(g.key_head_dim)));
    a.l2_eps = static_cast<float>(1e-6);
    a.norm_eps = static_cast<float>(g.rms_norm_eps);
    a.conv_w = w.conv;
    a.a_log = w.a_log;
    a.dt_bias = w.dt_bias;
    a.norm_w = w.norm;
    a.mixed = in.mixed.data;
    a.mixed_stride = in.mixed.stride;
    a.z = in.z.data;
    a.z_stride = in.z.stride;
    a.b = in.b.data;
    a.b_stride = in.b.stride;
    a.a = in.a.data;
    a.a_stride = in.a.stride;
    a.rows = rows.rows;
    a.in_slots = rows.in_slots;
    a.out_slots = rows.out_slots;
    a.wave = rows.wave;
    a.chunk_base = plan.chunk_base;
    a.conv_tails = states.conv_tails;
    a.recurrent = states.recurrent;
    a.saturations = states.saturations;
    if (plan.total_chunks != 0) {
        a.act = reinterpret_cast<float*>(ws + l.act);
        a.Wt = reinterpret_cast<float*>(ws + l.w);
        a.Ut = reinterpret_cast<float*>(ws + l.u);
        a.Mt = reinterpret_cast<float*>(ws + l.m);
        a.sc = reinterpret_cast<float*>(ws + l.sc);
    }
    a.act_stride = l.act_stride;
    a.out = out;
    a.out_stride = out_stride;
    a.vec_out = g.value_head_dim % 4 == 0 && out_stride % 4 == 0 && reinterpret_cast<std::uintptr_t>(out) % 16 == 0;

    // SEPAL_CHUNK_F32_PROFILE in the environment prints the time of each kernel (and synchronises the stream).
    static const bool profile = std::getenv("SEPAL_CHUNK_F32_PROFILE") != nullptr;
    cudaEvent_t ev[5] = {nullptr, nullptr, nullptr, nullptr, nullptr};
    if (profile) {
        for (cudaEvent_t& e : ev) cudaEventCreate(&e);
        cudaEventRecord(ev[0], stream);
    }
    if (plan.total_chunks != 0) {
        const dim3 conv_grid(plan.total_chunks, 2 * g.key_heads + g.value_heads);
        const dim3 tile_grid(plan.total_chunks, g.value_heads);
        if (states.half_storage) conv_kernel<true><<<conv_grid, 128, 0, stream>>>(a);
        else conv_kernel<false><<<conv_grid, 128, 0, stream>>>(a);
        if (profile) cudaEventRecord(ev[1], stream);
        prepare_kernel<<<tile_grid, kPrepThreads, 0, stream>>>(a);
    }
    if (profile) cudaEventRecord(ev[2], stream);
    const dim3 state_grid(kSplit, g.value_heads, rows.wave_size);
    if (states.half_storage) state_kernel<true><<<state_grid, 128, 0, stream>>>(a);
    else state_kernel<false><<<state_grid, 128, 0, stream>>>(a);
    if (profile) cudaEventRecord(ev[3], stream);
    if (plan.total_chunks != 0) norm_kernel<<<dim3(plan.total_chunks, g.value_heads), 128, 0, stream>>>(a);
    if (profile) {
        cudaEventRecord(ev[4], stream);
        cudaEventSynchronize(ev[4]);
        float t[4] = {0, 0, 0, 0};
        for (int i = 0; i < 4; ++i) cudaEventElapsedTime(&t[i], ev[i], ev[i + 1]);
        std::fprintf(stderr, "[chunked f32] conv %.3f ms, prepare %.3f ms, state %.3f ms, norm %.3f ms\n", t[0], t[1], t[2], t[3]);
        for (cudaEvent_t e : ev) cudaEventDestroy(e);
    }
    check(cudaGetLastError(), "deltanet chunked f32 kernels launch");
    launch_deltanet_conv_tail(g, in, rows, states, stream);
}

} // namespace sepal::cuda
