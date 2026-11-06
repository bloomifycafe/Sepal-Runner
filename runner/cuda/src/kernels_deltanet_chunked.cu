// The chunked Gated DeltaNet kernels; see kernels_deltanet_chunked.hpp for the algorithm and the numbers.
#include "kernels_deltanet_chunked.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include "device.hpp"

// An image without tensor cores (sm_61, sm_75) keeps only traps of the kernels below; their constants are then unused.
#if !SEPAL_TENSOR_CORES
#pragma nv_diag_suppress 177
#endif

namespace sepal::cuda {
namespace chunked {

#if SEPAL_TENSOR_CORES
using namespace nvcuda; // wmma exists only in the sm_70+ images
#endif

constexpr int kC = 64;          // tokens per chunk
constexpr int kD = 128;         // key and value head dimension
constexpr int kSplit = 4;       // slices of the value columns (32 each) run by separate blocks of the state kernel
constexpr int kSliceCols = kD / kSplit;
constexpr int kMaxTaps = 8;
// Leading dimensions in shared memory, padded so that the 8 or 16 rows of a tensor-core fragment fall in different banks.
constexpr int kPadD = 136; // [64][128] binary16 operands
constexpr int kPadC = 72;  // [64][64] binary16
constexpr int kPadA = 68;  // [64][64] float32
constexpr std::size_t kTileBytesBig = static_cast<std::size_t>(kC) * kD * sizeof(__half); // W, U, Q, K of one (chunk, head)
constexpr std::size_t kTileBytesM = static_cast<std::size_t>(kC) * kC * sizeof(__half);

// ---- helpers (the scan kernel's formulas) ----

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
__device__ __forceinline__ float widen_in(float v) { return v; }
__device__ __forceinline__ float widen_in(__half v) { return __half2float(v); }

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

// The (wave position, chunk) of a linear chunk number.
__device__ __forceinline__ void locate(const std::uint32_t* chunk_base, std::uint32_t wave_size, std::uint32_t lin, std::uint32_t& pos,
                                       std::uint32_t& j) {
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
    std::uint32_t key_heads, value_heads, repeat, taps, wave_size;
    float read_scale, l2_eps, norm_eps;
    const float *conv_w, *a_log, *dt_bias, *norm_w;
    const void *mixed, *z, *b, *a;
    std::size_t mixed_stride, z_stride, b_stride, a_stride;
    const RowSpan* rows;
    const std::uint32_t *in_slots, *out_slots, *wave, *chunk_base;
    void* conv_tails;
    void* recurrent;
    unsigned* saturations;
    // workspace
    __half *Wt, *Ut, *Qg, *Kd, *Mt;
    float* gend;
    __half* reads;
    __half* out;
    std::size_t out_stride;
};

// ---- kernel 1: prepare ----

constexpr int kPrepThreads = 512;
constexpr int kPrepSmem = 16384 + 17408 + 17408 + 17408 + 9216 + 1536;

template <class InT, bool HS>
__global__ void __launch_bounds__(kPrepThreads) prepare_kernel(const Args args) {
#if SEPAL_TENSOR_CORES
    extern __shared__ __align__(128) unsigned char smem_raw[];
    constexpr std::integral_constant<bool, HS> kHalfState{};
    float* F = reinterpret_cast<float*>(smem_raw);                        // [64][64]: T
    __half* ks = reinterpret_cast<__half*>(smem_raw + 16384);              // [64][kPadD]
    __half* qx = reinterpret_cast<__half*>(smem_raw + 33792);              // [64][kPadD]: X for W, then Q, then X for U
    float* AT = reinterpret_cast<float*>(smem_raw + 51200);                // [64][kPadA]: A / products staging
    __half* Th = reinterpret_cast<__half*>(smem_raw + 68608);              // [64][kPadC]
    float* sg = reinterpret_cast<float*>(smem_raw + 77824);                // g[64]
    float* sbeta = sg + 64;
    float* scum = sbeta + 64;
    float* sex = scum + 64;                                                // exp(cum)
    float* skd = sex + 64;                                                 // exp(cum_last - cum)

    const std::uint32_t tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const std::uint32_t head = blockIdx.y;
    std::uint32_t pos, j;
    locate(args.chunk_base, args.wave_size, blockIdx.x, pos, j);
    const std::uint32_t row = args.wave[pos];
    const RowSpan span = args.rows[row];
    const std::uint32_t in_slot = args.in_slots[row];
    const std::uint32_t c0 = j * kC;
    const std::uint32_t n = min(static_cast<std::uint32_t>(kC), span.length - c0);
    const std::uint32_t taps = args.taps;
    const std::uint32_t key_head = head / args.repeat;
    const std::uint32_t conv_dim = 2 * args.key_heads * kD + args.value_heads * kD;
    const std::size_t tile = static_cast<std::size_t>(blockIdx.x) * args.value_heads + head;
    __half* Qg_t = args.Qg + tile * (kC * kD);
    __half* Kd_t = args.Kd + tile * (kC * kD);
    __half* W_t = args.Wt + tile * (kC * kD);
    __half* U_t = args.Ut + tile * (kC * kD);
    __half* M_t = args.Mt + tile * (kC * kC);

    // ---- gates ----
    if (tid < kC) {
        float g = 0.0f, bt = 0.0f;
        if (tid < n) {
            const std::size_t token = span.offset + c0 + tid;
            const float a_in = widen_in(static_cast<const InT*>(args.a)[token * args.a_stride + head]);
            const float b_in = widen_in(static_cast<const InT*>(args.b)[token * args.b_stride + head]);
            g = __fmul_rn(-expf(args.a_log[head]), softplus_f(__fadd_rn(a_in, args.dt_bias[head])));
            bt = sigmoid_f(b_in);
        }
        sg[tid] = g;
        sbeta[tid] = bt;
    }
    __syncthreads();
    if (tid == 0) {
        float c = 0.0f;
        for (int t = 0; t < kC; ++t) {
            if (static_cast<std::uint32_t>(t) < n) c = __fadd_rn(c, sg[t]);
            scum[t] = c;
        }
    }
    __syncthreads();
    if (tid < kC) {
        const float last = scum[n - 1];
        sex[tid] = tid < n ? expf(scum[tid]) : 0.0f;
        skd[tid] = tid < n ? expf(last - scum[tid]) : 0.0f;
        if (tid == 0) args.gend[tile] = expf(last);
    }
    __syncthreads();

    // The 4 tokens [4 warp, 4 warp + 4) of this warp: convolution + SiLU of a channel group (the lane's channels lane + 32 q) in the
    // scan kernel's tap order, handed to consume(token, v[4]). All the rows of the window are requested before any is used.
    constexpr int kTokensPerWarp = kC / (kPrepThreads / 32);
    constexpr int kRows = kTokensPerWarp + kMaxTaps - 1;
    auto conv_tokens = [&](std::uint32_t ch_base, auto&& consume) {
        const std::uint32_t t_first = warp * kTokensPerWarp;
        float wt[4][kMaxTaps];
#pragma unroll
        for (int q = 0; q < 4; ++q)
#pragma unroll
            for (int k = 0; k < kMaxTaps; ++k)
                wt[q][k] = static_cast<std::uint32_t>(k) < taps ? args.conv_w[static_cast<std::size_t>(ch_base + lane + 32 * q) * taps + k] : 0.0f;
        float x[kRows][4];
        if (c0 + t_first >= taps - 1) {
            // every row of the window is an input of this row: plain predicated loads
            const InT* src = static_cast<const InT*>(args.mixed) + static_cast<std::size_t>(span.offset + c0 + t_first - (taps - 1)) * args.mixed_stride + ch_base + lane;
#pragma unroll
            for (int r = 0; r < kRows; ++r) {
                const bool live = static_cast<std::uint32_t>(r) < kTokensPerWarp + taps - 1 && t_first + r < n + taps - 1;
#pragma unroll
                for (int q = 0; q < 4; ++q) x[r][q] = live ? widen_in(src[static_cast<std::size_t>(r) * args.mixed_stride + 32 * q]) : 0.0f;
            }
        } else {
#pragma unroll
            for (int r = 0; r < kRows; ++r) {
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    float v = 0.0f;
                    if (static_cast<std::uint32_t>(r) < kTokensPerWarp + taps - 1 && t_first + r < n + taps - 1) {
                        const std::uint32_t w = c0 + t_first + r; // position in (tail, inputs)
                        const std::uint32_t ch = ch_base + lane + 32 * q;
                        if (w < taps - 1) {
                            v = in_slot != kNoSlot ? load_state(args.conv_tails, (static_cast<std::size_t>(in_slot) * (taps - 1) + w) * conv_dim + ch, kHalfState) : 0.0f;
                        } else {
                            v = widen_in(static_cast<const InT*>(args.mixed)[static_cast<std::size_t>(span.offset + w - (taps - 1)) * args.mixed_stride + ch]);
                        }
                    }
                    x[r][q] = v;
                }
            }
        }
#pragma unroll
        for (int u = 0; u < kTokensPerWarp; ++u) {
            const std::uint32_t tok = t_first + u;
            float v[4];
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                float acc = 0.0f;
#pragma unroll
                for (int k = 0; k < kMaxTaps; ++k) {
                    if (static_cast<std::uint32_t>(k) < taps) acc = __fmaf_rn(wt[q][k], x[u + k][q], acc);
                }
                v[q] = tok < n ? silu_f(acc) : 0.0f;
            }
            consume(tok, v);
        }
    };
    // L2 normalisation of the token's 128 values (4 per lane), handed to store(token, feature, normalised value).
    auto normalised = [&](auto&& store) {
        return [&store, this_n = n, l2 = args.l2_eps, lane](std::uint32_t tok, float (&f)[4]) {
            float sum = 0.0f;
#pragma unroll
            for (int q = 0; q < 4; ++q) sum = __fmaf_rn(f[q], f[q], sum);
            sum = warp_sum(sum);
            const float norm = sqrtf(__fadd_rn(sum, l2));
            const float rn = reciprocal(norm);
#pragma unroll
            for (int q = 0; q < 4; ++q) store(tok, lane + 32 * q, tok < this_n ? divide_by(f[q], norm, rn) : 0.0f);
        };
    };

    // ---- k: convolution, L2 norm; ks, Kd (global), X for W ----
    conv_tokens(args.key_heads * kD + key_head * kD, normalised([&](std::uint32_t tok, std::uint32_t i, float kh) {
                    const float kd = tok < n ? __fmul_rn(skd[tok], kh) : 0.0f;
                    const float xw = tok < n ? __fmul_rn(__fmul_rn(sbeta[tok], sex[tok]), kh) : 0.0f;
                    ks[tok * kPadD + i] = __float2half_rn(kh);
                    Kd_t[tok * kD + i] = __float2half_rn(kd);
                    qx[tok * kPadD + i] = __float2half_rn(xw);
                }));
    __syncthreads();

    // ---- A = diag(beta) tril(K K^T * ratio, -1) ----
    {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
        const int rt = warp & 3, nt = warp >> 2;
        wmma::fill_fragment(acc, 0.0f);
        for (int kk = 0; kk < 8; ++kk) {
            wmma::load_matrix_sync(fa, ks + (16 * rt) * kPadD + 16 * kk, kPadD);
            wmma::load_matrix_sync(fb, ks + (16 * nt) * kPadD + 16 * kk, kPadD);
            wmma::mma_sync(acc, fa, fb, acc);
        }
        wmma::store_matrix_sync(AT + (16 * rt) * kPadA + 16 * nt, acc, kPadA, wmma::mem_row_major);
    }
    __syncthreads();
    for (std::uint32_t e = tid; e < kC * kC; e += kPrepThreads) {
        const std::uint32_t i = e / kC, jj = e % kC;
        float& a = AT[i * kPadA + jj];
        a = (jj < i && i < n) ? __fmul_rn(__fmul_rn(sbeta[i], expf(__fsub_rn(scum[i], scum[jj]))), a) : 0.0f;
    }
    for (std::uint32_t e = tid; e < kC * kC; e += kPrepThreads) F[e] = 0.0f;
    __syncthreads();

    // ---- T = (I + A)^-1 by 16 x 16 blocks: the four diagonal blocks (a column per thread), then block rows 1..3, each as
    //      S = sum_k A_rk T_kc and T_rc = -T_rr S ----
    if (tid < kC) {
        const std::uint32_t b = tid >> 4, jc = tid & 15;
        float tcol[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            float s = 0.0f;
#pragma unroll
            for (int m = 0; m < 15; ++m) {
                if (m < i && static_cast<std::uint32_t>(m) >= jc) s = __fmaf_rn(AT[(b * 16 + i) * kPadA + b * 16 + m], tcol[m], s);
            }
            tcol[i] = static_cast<std::uint32_t>(i) < jc ? 0.0f : (static_cast<std::uint32_t>(i) == jc ? 1.0f : -s);
            F[(b * 16 + i) * kC + b * 16 + jc] = tcol[i];
        }
    }
    __syncthreads();
    float* Sbuf = reinterpret_cast<float*>(Th); // 3 x 256 floats, free until Th is filled
    for (std::uint32_t r = 1; r < 4; ++r) {
        for (std::uint32_t e = tid; e < 256 * r; e += kPrepThreads) {
            const std::uint32_t c = e >> 8, i = (e >> 4) & 15, jc = e & 15;
            float s = 0.0f;
            for (std::uint32_t k = c; k < r; ++k) {
#pragma unroll
                for (int q = 0; q < 16; ++q) s = __fmaf_rn(AT[(r * 16 + i) * kPadA + k * 16 + q], F[(k * 16 + q) * kC + c * 16 + jc], s);
            }
            Sbuf[e] = s;
        }
        __syncthreads();
        for (std::uint32_t e = tid; e < 256 * r; e += kPrepThreads) {
            const std::uint32_t c = e >> 8, i = (e >> 4) & 15, jc = e & 15;
            float s = 0.0f;
            for (std::uint32_t q = 0; q <= i; ++q) s = __fmaf_rn(F[(r * 16 + i) * kC + r * 16 + q], Sbuf[c * 256 + q * 16 + jc], s);
            F[(r * 16 + i) * kC + c * 16 + jc] = -s;
        }
        __syncthreads();
    }
    for (std::uint32_t e = tid; e < kC * kC; e += kPrepThreads) Th[(e / kC) * kPadC + e % kC] = __float2half_rn(F[e]);
    __syncthreads();

    // C[64][128] = Th[64][64] * X[64][128] (X in qx), written as binary16 to `dst` (a [64][128] tile). The staging area is AT.
    auto product_to_global = [&](__half* dst) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fb;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
        float* stage = AT + warp * 256;
        const int rt = warp & 3;
        for (int np = 0; np < 2; ++np) {
            const int col0 = ((warp >> 2) * 2 + np) * 16;
            wmma::fill_fragment(acc, 0.0f);
            for (int kk = 0; kk < 4; ++kk) {
                wmma::load_matrix_sync(fa, Th + (16 * rt) * kPadC + 16 * kk, kPadC);
                wmma::load_matrix_sync(fb, qx + (16 * kk) * kPadD + col0, kPadD);
                wmma::mma_sync(acc, fa, fb, acc);
            }
            wmma::store_matrix_sync(stage, acc, 16, wmma::mem_row_major);
            __syncwarp();
            for (int idx = lane; idx < 256; idx += 32) {
                const int r = idx / 16, c = idx % 16;
                dst[(16 * rt + r) * kD + col0 + c] = __float2half_rn(stage[idx]);
            }
            __syncwarp();
        }
    };
    product_to_global(W_t);
    __syncthreads();

    // ---- q: convolution, L2 norm, times the read scale; qx, Qg (global) ----
    conv_tokens(key_head * kD, normalised([&](std::uint32_t tok, std::uint32_t i, float qh) {
                    const float qs = tok < n ? __fmul_rn(qh, args.read_scale) : 0.0f;
                    const float qg = tok < n ? __fmul_rn(sex[tok], qs) : 0.0f;
                    qx[tok * kPadD + i] = __float2half_rn(qs);
                    Qg_t[tok * kD + i] = __float2half_rn(qg);
                }));
    __syncthreads();
    // M = tril(Q K^T * ratio)
    {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
        const int rt = warp & 3, nt = warp >> 2;
        wmma::fill_fragment(acc, 0.0f);
        for (int kk = 0; kk < 8; ++kk) {
            wmma::load_matrix_sync(fa, qx + (16 * rt) * kPadD + 16 * kk, kPadD);
            wmma::load_matrix_sync(fb, ks + (16 * nt) * kPadD + 16 * kk, kPadD);
            wmma::mma_sync(acc, fa, fb, acc);
        }
        wmma::store_matrix_sync(AT + (16 * rt) * kPadA + 16 * nt, acc, kPadA, wmma::mem_row_major);
    }
    __syncthreads();
    for (std::uint32_t e = tid; e < kC * kC; e += kPrepThreads) {
        const std::uint32_t i = e / kC, jj = e % kC;
        const float m = (jj <= i && i < n) ? __fmul_rn(expf(__fsub_rn(scum[i], scum[jj])), AT[i * kPadA + jj]) : 0.0f;
        M_t[e] = __float2half_rn(m);
    }
    __syncthreads(); // qx and AT are free

    // ---- v: convolution; X for U = diag(beta) V ----
    conv_tokens(2 * args.key_heads * kD + head * kD, [&](std::uint32_t tok, float (&v)[4]) {
#pragma unroll
        for (int q = 0; q < 4; ++q) qx[tok * kPadD + lane + 32 * q] = __float2half_rn(tok < n ? __fmul_rn(sbeta[tok], v[q]) : 0.0f);
    });
    __syncthreads();
    product_to_global(U_t);
#else
    __trap();
#endif
}

// ---- kernel 2: the state loop ----
//
// One block per (row, head, slice of 32 value columns); it walks the row's chunks. Per chunk (all products on the tensor cores, float32
// accumulation, operands single binary16): P = W S, Y = U - P, out = Qg S + M Y, S = gend S + Kd^T Y. The four chunk operands of the next chunk
// are fetched with cp.async as soon as the buffer they replace is free, so the loads hide behind the arithmetic of the current chunk.

constexpr int kYld = 40; // leading dimension of Y [64][32] binary16 in shared memory
constexpr int kOld = 36; // of the float32 staging [64][32]
constexpr int kStateSmem = 100352;

__device__ __forceinline__ void cp_async16(void* shared_dst, const void* global_src) {
#if SEPAL_TENSOR_CORES
    const unsigned dst = static_cast<unsigned>(__cvta_generic_to_shared(shared_dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(global_src));
#else
    __trap();
#endif
}
__device__ __forceinline__ void cp_async_commit() {
#if SEPAL_TENSOR_CORES
    asm volatile("cp.async.commit_group;\n" ::);
#else
    __trap();
#endif
}
__device__ __forceinline__ void cp_async_wait_all() {
#if SEPAL_TENSOR_CORES
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
#else
    __trap();
#endif
}

// A [64][128] tile of the workspace into a padded shared buffer, by the 128 threads of the block.
__device__ __forceinline__ void fetch_big(__half* shared, const __half* global, std::uint32_t tid) {
    for (std::uint32_t e = tid; e < kTileBytesBig / 16; e += 128) {
        cp_async16(shared + (e / 16) * kPadD + (e % 16) * 8, global + static_cast<std::size_t>(e) * 8);
    }
}
__device__ __forceinline__ void fetch_mask(__half* shared, const __half* global, std::uint32_t tid) {
    for (std::uint32_t e = tid; e < kTileBytesM / 16; e += 128) cp_async16(shared + (e / 8) * kPadC + (e % 8) * 8, global + static_cast<std::size_t>(e) * 8);
}

template <bool HS>
__global__ void __launch_bounds__(128) state_kernel(const Args args) {
#if SEPAL_TENSOR_CORES
    extern __shared__ __align__(128) unsigned char smem_raw[];
    constexpr std::integral_constant<bool, HS> kHalfState{};
    float* Sf = reinterpret_cast<float*>(smem_raw);                 // [128][32]
    __half* Sh = reinterpret_cast<__half*>(smem_raw + 16384);       // [128][32]
    __half* Wb = reinterpret_cast<__half*>(smem_raw + 24576);       // [64][kPadD]
    __half* Qb = reinterpret_cast<__half*>(smem_raw + 41984);       // [64][kPadD]: Qg
    __half* Kb = reinterpret_cast<__half*>(smem_raw + 59392);       // [64][kPadD]: Kd
    __half* Mb = reinterpret_cast<__half*>(smem_raw + 76800);       // [64][kPadC]
    float* Ost = reinterpret_cast<float*>(smem_raw + 86016);        // [64][kOld]
    __half* Yh = reinterpret_cast<__half*>(smem_raw + 95232);       // [64][kYld]

    const std::uint32_t tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const std::uint32_t pos = blockIdx.x, head = blockIdx.y, split = blockIdx.z;
    const std::uint32_t row = args.wave[pos];
    const RowSpan span = args.rows[row];
    const std::uint32_t in_slot = args.in_slots[row], out_slot = args.out_slots[row];
    const std::uint32_t nchunks = (span.length + kC - 1) / kC;
    const std::uint32_t base = args.chunk_base[pos];
    const std::size_t value_dim = static_cast<std::size_t>(args.value_heads) * kD;
    auto tile_of = [&](std::uint32_t j) { return static_cast<std::size_t>(base + j) * args.value_heads + head; };

    if (nchunks != 0) {
        const std::size_t t0 = tile_of(0);
        fetch_big(Wb, args.Wt + t0 * (kC * kD), tid);
        fetch_big(Qb, args.Qg + t0 * (kC * kD), tid);
        fetch_big(Kb, args.Kd + t0 * (kC * kD), tid);
        fetch_mask(Mb, args.Mt + t0 * (kC * kC), tid);
        cp_async_commit();
    }
    for (std::uint32_t e = tid; e < kD * kSliceCols; e += 128) {
        const std::uint32_t key = e / kSliceCols, col = e % kSliceCols;
        Sf[e] = in_slot != kNoSlot
                    ? load_state(args.recurrent, (static_cast<std::size_t>(in_slot) * args.value_heads + head) * kD * kD + key * kD + split * kSliceCols + col, kHalfState)
                    : 0.0f;
    }
    __syncthreads();

    for (std::uint32_t j = 0; j < nchunks; ++j) {
        const std::size_t tile = tile_of(j);
        const std::uint32_t n = min(static_cast<std::uint32_t>(kC), span.length - j * kC);
        // this warp's U values and the chunk's decay, requested early
        float u[16];
        {
            const __half* ug = args.Ut + tile * (kC * kD) + split * kSliceCols + lane;
#pragma unroll
            for (int i = 0; i < 16; ++i) u[i] = __half2float(ug[(16 * warp + i) * kD]);
        }
        const float gend = args.gend[tile];
        // binary16 copy of the state for the tensor cores
        for (std::uint32_t e = tid; e < kD * kSliceCols; e += 128) Sh[e] = __float2half_rn(Sf[e]);
        cp_async_wait_all();
        __syncthreads(); // the fetched operands and the state copy are complete for everyone

        // P = W S0 and O = Qg S0 for this warp's 16 tokens
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> accP[2], accO[2];
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fw, fq;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> bh;
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                wmma::fill_fragment(accP[nt], 0.0f);
                wmma::fill_fragment(accO[nt], 0.0f);
            }
            for (int kk = 0; kk < 8; ++kk) {
                wmma::load_matrix_sync(fw, Wb + (16 * warp) * kPadD + 16 * kk, kPadD);
                wmma::load_matrix_sync(fq, Qb + (16 * warp) * kPadD + 16 * kk, kPadD);
#pragma unroll
                for (int nt = 0; nt < 2; ++nt) {
                    wmma::load_matrix_sync(bh, Sh + (16 * kk) * kSliceCols + 16 * nt, kSliceCols);
                    wmma::mma_sync(accP[nt], fw, bh, accP[nt]);
                    wmma::mma_sync(accO[nt], fq, bh, accO[nt]);
                }
            }
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) wmma::store_matrix_sync(Ost + (16 * warp) * kOld + 16 * nt, accP[nt], kOld, wmma::mem_row_major);
        }
        __syncwarp();
        // Y = U - P for this warp's rows
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const int rowi = 16 * warp + i;
            Yh[rowi * kYld + lane] = __float2half_rn(__fsub_rn(u[i], Ost[rowi * kOld + lane]));
        }
        __syncthreads(); // Y complete; W and Qg are no longer needed
        if (j + 1 < nchunks) {
            const std::size_t next = tile_of(j + 1);
            fetch_big(Wb, args.Wt + next * (kC * kD), tid);
            fetch_big(Qb, args.Qg + next * (kC * kD), tid);
            cp_async_commit();
        }

        // O += M Y (M is lower triangular: tiles right of the diagonal are zero)
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fm;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> bh;
            for (int kk = 0; kk <= static_cast<int>(warp); ++kk) {
                wmma::load_matrix_sync(fm, Mb + (16 * warp) * kPadC + 16 * kk, kPadC);
#pragma unroll
                for (int nt = 0; nt < 2; ++nt) {
                    wmma::load_matrix_sync(bh, Yh + (16 * kk) * kYld + 16 * nt, kYld);
                    wmma::mma_sync(accO[nt], fm, bh, accO[nt]);
                }
            }
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) wmma::store_matrix_sync(Ost + (16 * warp) * kOld + 16 * nt, accO[nt], kOld, wmma::mem_row_major);
        }
        __syncwarp();
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const int rowi = 16 * warp + i;
            if (static_cast<std::uint32_t>(rowi) < n) {
                args.reads[(static_cast<std::size_t>(span.offset) + j * kC + rowi) * value_dim + head * kD + split * kSliceCols + lane] =
                    __float2half_rn(Ost[rowi * kOld + lane]);
            }
        }

        // S = gend S + Kd^T Y for this warp's 32 keys
        for (int idx = lane; idx < 32 * kSliceCols; idx += 32) Sf[(32 * warp) * kSliceCols + idx] = __fmul_rn(Sf[(32 * warp) * kSliceCols + idx], gend);
        __syncwarp();
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::col_major> fk;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> bh;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> accS[2][2];
#pragma unroll
            for (int mt = 0; mt < 2; ++mt)
#pragma unroll
                for (int nt = 0; nt < 2; ++nt)
                    wmma::load_matrix_sync(accS[mt][nt], Sf + (32 * warp + 16 * mt) * kSliceCols + 16 * nt, kSliceCols, wmma::mem_row_major);
            for (int kk = 0; kk < 4; ++kk) {
                if (static_cast<std::uint32_t>(16 * kk) >= n) break;
#pragma unroll
                for (int nt = 0; nt < 2; ++nt) {
                    wmma::load_matrix_sync(bh, Yh + (16 * kk) * kYld + 16 * nt, kYld);
#pragma unroll
                    for (int mt = 0; mt < 2; ++mt) {
                        wmma::load_matrix_sync(fk, Kb + (16 * kk) * kPadD + 32 * warp + 16 * mt, kPadD);
                        wmma::mma_sync(accS[mt][nt], fk, bh, accS[mt][nt]);
                    }
                }
            }
#pragma unroll
            for (int mt = 0; mt < 2; ++mt)
#pragma unroll
                for (int nt = 0; nt < 2; ++nt)
                    wmma::store_matrix_sync(Sf + (32 * warp + 16 * mt) * kSliceCols + 16 * nt, accS[mt][nt], kSliceCols, wmma::mem_row_major);
        }
        __syncthreads(); // S complete; Kd, M and Y are no longer needed
        if (j + 1 < nchunks) {
            const std::size_t next = tile_of(j + 1);
            fetch_big(Kb, args.Kd + next * (kC * kD), tid);
            fetch_mask(Mb, args.Mt + next * (kC * kC), tid);
            cp_async_commit();
        }
    }

    if (out_slot != kNoSlot) {
        for (std::uint32_t e = tid; e < kD * kSliceCols; e += 128) {
            const std::uint32_t key = e / kSliceCols, col = e % kSliceCols;
            store_state(args.recurrent, (static_cast<std::size_t>(out_slot) * args.value_heads + head) * kD * kD + key * kD + split * kSliceCols + col, Sf[e],
                        args.saturations, kHalfState);
        }
    }
#else
    __trap();
#endif
}

// ---- kernel 3: gated RMSNorm ----

template <class InT>
__global__ void __launch_bounds__(128) norm_kernel(const Args args) {
    const std::uint32_t tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const std::uint32_t head = blockIdx.y;
    std::uint32_t pos, j;
    locate(args.chunk_base, args.wave_size, blockIdx.x, pos, j);
    const RowSpan span = args.rows[args.wave[pos]];
    const std::uint32_t c0 = j * kC;
    const std::uint32_t n = min(static_cast<std::uint32_t>(kC), span.length - c0);
    const std::size_t value_dim = static_cast<std::size_t>(args.value_heads) * kD;
    float nw[4];
#pragma unroll
    for (int q = 0; q < 4; ++q) nw[q] = args.norm_w[lane + 32 * q];
    for (std::uint32_t u = 0; u < 16; ++u) {
        const std::uint32_t t = warp * 16 + u;
        if (t >= n) break;
        const std::size_t token = span.offset + c0 + t;
        float x[4], sum = 0.0f;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            x[q] = __half2float(args.reads[token * value_dim + head * kD + lane + 32 * q]);
            sum = __fmaf_rn(x[q], x[q], sum);
        }
        sum = warp_sum(sum);
        const float rms = sqrtf(__fadd_rn(__fdiv_rn(sum, static_cast<float>(kD)), args.norm_eps));
        const float recip = reciprocal(rms);
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const std::uint32_t jj = lane + 32 * q;
            const float zz = widen_in(static_cast<const InT*>(args.z)[token * args.z_stride + head * kD + jj]);
            const float y = __fmul_rn(__fmul_rn(divide_by(x[q], rms, recip), nw[q]), silu_f(zz));
            args.out[token * args.out_stride + head * kD + jj] = __float2half_rn(y);
        }
    }
}

template <class InT, bool HS>
void launch_pair(const dim3& prep_grid, const dim3& state_grid, const Args& a, cudaStream_t stream, cudaEvent_t* ev = nullptr) {
    static const bool configured = [] {
        check(cudaFuncSetAttribute(prepare_kernel<InT, HS>, cudaFuncAttributeMaxDynamicSharedMemorySize, kPrepSmem), "cudaFuncSetAttribute");
        check(cudaFuncSetAttribute(state_kernel<HS>, cudaFuncAttributeMaxDynamicSharedMemorySize, kStateSmem), "cudaFuncSetAttribute");
        return true;
    }();
    (void)configured;
    prepare_kernel<InT, HS><<<prep_grid, kPrepThreads, kPrepSmem, stream>>>(a);
    if (ev != nullptr) cudaEventRecord(ev[1], stream);
    state_kernel<HS><<<state_grid, 128, kStateSmem, stream>>>(a);
    if (ev != nullptr) cudaEventRecord(ev[2], stream);
}

} // namespace chunked

// ---- host side ----

bool deltanet_chunked_supported(const DeltaNetGeometry& g) {
    return g.key_head_dim == 128 && g.value_head_dim == 128 && g.conv_kernel >= 1 && g.conv_kernel <= chunked::kMaxTaps && g.key_heads != 0 &&
           g.value_heads % g.key_heads == 0;
}

std::vector<std::uint32_t> deltanet_chunk_prefix(const std::vector<std::uint32_t>& lengths) {
    std::vector<std::uint32_t> base(lengths.size() + 1, 0);
    for (std::size_t i = 0; i < lengths.size(); ++i) base[i + 1] = base[i] + (lengths[i] + kDeltaNetChunk - 1) / kDeltaNetChunk;
    return base;
}

namespace {
std::size_t align_up(std::size_t v, std::size_t a) { return (v + a - 1) / a * a; }
struct Layout {
    std::size_t w, u, qg, kd, m, gend, reads, total;
};
Layout layout(const DeltaNetGeometry& g, std::size_t chunks, std::size_t tokens) {
    const std::size_t tiles = chunks * g.value_heads;
    Layout l{};
    std::size_t at = 0;
    l.w = at; at += tiles * chunked::kTileBytesBig;
    l.u = at; at += tiles * chunked::kTileBytesBig;
    l.qg = at; at += tiles * chunked::kTileBytesBig;
    l.kd = at; at += tiles * chunked::kTileBytesBig;
    l.m = at; at += tiles * chunked::kTileBytesM;
    l.gend = at; at += align_up(tiles * sizeof(float), 256);
    l.reads = at; at += tokens * g.value_dim() * sizeof(__half);
    l.total = at;
    return l;
}
} // namespace

std::size_t deltanet_chunked_workspace_bytes(const DeltaNetGeometry& g, std::size_t chunks, std::size_t tokens) { return layout(g, chunks, tokens).total; }

template <class InT>
void launch_deltanet_chunked(const DeltaNetGeometry& g, const DeltaNetWeights& w, const DeltaNetInputsT<InT>& in, const DeltaNetRows& rows,
                             const DeltaNetChunkPlan& plan, const DeltaNetStates& states, void* workspace, std::size_t workspace_bytes,
                             std::size_t workspace_tokens, __half* out, std::size_t out_stride, cudaStream_t stream) {
    if (!deltanet_chunked_supported(g)) throw std::invalid_argument("deltanet chunked: unsupported geometry");
    if (rows.wave_size == 0 || plan.total_chunks == 0) return;
    if (workspace == nullptr || layout(g, plan.total_chunks, workspace_tokens).total > workspace_bytes) {
        throw std::invalid_argument("deltanet chunked: the workspace is too small");
    }
    if (w.conv == nullptr || w.a_log == nullptr || w.dt_bias == nullptr || w.norm == nullptr || in.mixed.data == nullptr || in.z.data == nullptr ||
        in.b.data == nullptr || in.a.data == nullptr || states.recurrent == nullptr || out == nullptr || plan.chunk_base == nullptr ||
        (g.conv_kernel > 1 && states.conv_tails == nullptr)) {
        throw std::invalid_argument("deltanet chunked: a null pointer");
    }
    const Layout l = layout(g, plan.total_chunks, workspace_tokens);
    char* ws = static_cast<char*>(workspace);
    chunked::Args a{};
    a.key_heads = g.key_heads;
    a.value_heads = g.value_heads;
    a.repeat = g.value_heads / g.key_heads;
    a.taps = g.conv_kernel;
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
    a.Wt = reinterpret_cast<__half*>(ws + l.w);
    a.Ut = reinterpret_cast<__half*>(ws + l.u);
    a.Qg = reinterpret_cast<__half*>(ws + l.qg);
    a.Kd = reinterpret_cast<__half*>(ws + l.kd);
    a.Mt = reinterpret_cast<__half*>(ws + l.m);
    a.gend = reinterpret_cast<float*>(ws + l.gend);
    a.reads = reinterpret_cast<__half*>(ws + l.reads);
    a.out = out;
    a.out_stride = out_stride;

    const dim3 prep_grid(plan.total_chunks, g.value_heads);
    const dim3 state_grid(rows.wave_size, g.value_heads, chunked::kSplit);
    static const bool profile = std::getenv("SEPAL_CHUNK_PROFILE") != nullptr; // prints the time of each kernel; synchronises
    cudaEvent_t ev[4] = {nullptr, nullptr, nullptr, nullptr};
    if (profile) {
        for (cudaEvent_t& e : ev) cudaEventCreate(&e);
        cudaEventRecord(ev[0], stream);
    }
    if (states.half_storage) chunked::launch_pair<InT, true>(prep_grid, state_grid, a, stream, profile ? ev : nullptr);
    else chunked::launch_pair<InT, false>(prep_grid, state_grid, a, stream, profile ? ev : nullptr);
    chunked::norm_kernel<InT><<<prep_grid, 128, 0, stream>>>(a);
    if (profile) {
        cudaEventRecord(ev[3], stream);
        cudaEventSynchronize(ev[3]);
        float p = 0, st = 0, nm = 0;
        cudaEventElapsedTime(&p, ev[0], ev[1]);
        cudaEventElapsedTime(&st, ev[1], ev[2]);
        cudaEventElapsedTime(&nm, ev[2], ev[3]);
        std::fprintf(stderr, "[chunked] prepare %.3f ms, state %.3f ms, norm %.3f ms\n", p, st, nm);
        for (cudaEvent_t e : ev) cudaEventDestroy(e);
    }
    check(cudaGetLastError(), "deltanet chunked kernels launch");
    launch_deltanet_conv_tail(g, in, rows, states, stream);
}

template void launch_deltanet_chunked<__half>(const DeltaNetGeometry&, const DeltaNetWeights&, const DeltaNetInputsT<__half>&, const DeltaNetRows&,
                                              const DeltaNetChunkPlan&, const DeltaNetStates&, void*, std::size_t, std::size_t, __half*, std::size_t,
                                              cudaStream_t);
template void launch_deltanet_chunked<float>(const DeltaNetGeometry&, const DeltaNetWeights&, const DeltaNetInputsT<float>&, const DeltaNetRows&,
                                             const DeltaNetChunkPlan&, const DeltaNetStates&, void*, std::size_t, std::size_t, __half*, std::size_t,
                                             cudaStream_t);

} // namespace sepal::cuda
