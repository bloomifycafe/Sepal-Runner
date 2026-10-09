#include "gemm_s8_kernels.hpp"

#include <cstdint>
#include <stdexcept>

#include <cuda_fp16.h>

#include "device.hpp"
#include "gemm_ptx.cuh"

namespace sepal::cuda {
namespace detail {

constexpr int kS8Group = 4; // blocks run in groups of this many row tiles (as in gemm_kernels.cu)

// ---- quantisation ---------------------------------------------------------------------------------------------------

__device__ inline float block_max(float v, float* shared) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, o));
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    __syncthreads(); // `shared` may still be read from a previous use
    if (lane == 0) shared[warp] = v;
    __syncthreads();
    float r = shared[0];
    for (int i = 1; i < (blockDim.x >> 5); ++i) r = fmaxf(r, shared[i]);
    return r;
}

__device__ inline int quantize_one(float v, float inv) {
    const int q = __float2int_rn(v * inv);
    return min(127, max(-127, q));
}

// One block per row. The maximum does not depend on the order it is taken in, and each element is quantised on its own.
__global__ void quantize_rows_s8_kernel(const uint4* __restrict__ x, std::size_t ld_x8, uint2* __restrict__ q, std::size_t ld_q8, float* __restrict__ scales,
                                        int cols8) {
    __shared__ float shared[8];
    const std::size_t row = blockIdx.x;
    const uint4* xr = x + row * ld_x8;
    float amax = 0.0f;
    for (int c = threadIdx.x; c < cols8; c += blockDim.x) {
        const uint4 v = xr[c];
        const __half2* h = reinterpret_cast<const __half2*>(&v);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 f = __half22float2(h[i]);
            amax = fmaxf(amax, fmaxf(fabsf(f.x), fabsf(f.y)));
        }
    }
    amax = block_max(amax, shared);
    const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
    if (threadIdx.x == 0) scales[row] = amax / 127.0f;
    uint2* qr = q + row * ld_q8;
    for (int c = threadIdx.x; c < cols8; c += blockDim.x) {
        const uint4 v = xr[c];
        const __half2* h = reinterpret_cast<const __half2*>(&v);
        unsigned packed[2] = {0u, 0u};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 f = __half22float2(h[i]);
            const unsigned lo = static_cast<unsigned>(quantize_one(f.x, inv)) & 0xFFu, hi = static_cast<unsigned>(quantize_one(f.y, inv)) & 0xFFu;
            packed[i >> 1] |= (lo << ((i & 1) * 16)) | (hi << ((i & 1) * 16 + 8));
        }
        qr[c] = make_uint2(packed[0], packed[1]);
    }
}

// One block per row; a thread takes groups of 16 values of k (two 16-byte loads, one 16-byte store).
__global__ void quantize_weights_s8_kernel(const uint4* __restrict__ w, std::size_t ld_w8, uint4* __restrict__ wq, float* __restrict__ scales, int groups,
                                           bool pairs) {
    __shared__ float shared[8];
    const std::size_t row = blockIdx.x;
    const uint4* wr = w + row * ld_w8;
    float amax = 0.0f;
    for (int g = threadIdx.x; g < groups; g += blockDim.x) {
        for (int part = 0; part < 2; ++part) {
            const uint4 v = wr[2 * g + part];
            const __half2* h = reinterpret_cast<const __half2*>(&v);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float2 f = __half22float2(h[i]);
                amax = fmaxf(amax, fmaxf(fabsf(f.x), fabsf(f.y)));
            }
        }
    }
    amax = block_max(amax, shared);
    const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
    if (threadIdx.x == 0) scales[row] = amax / 127.0f;
    uint4* qr = wq + row * static_cast<std::size_t>(groups);
    for (int g = threadIdx.x; g < groups; g += blockDim.x) {
        unsigned char b[16];
#pragma unroll
        for (int part = 0; part < 2; ++part) {
            const uint4 v = wr[2 * g + part];
            const __half2* h = reinterpret_cast<const __half2*>(&v);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float2 f = __half22float2(h[i]);
                b[part * 8 + 2 * i] = static_cast<unsigned char>(quantize_one(f.x, inv));
                b[part * 8 + 2 * i + 1] = static_cast<unsigned char>(quantize_one(f.y, inv));
            }
        }
        __align__(16) unsigned char o[16];
#pragma unroll
        for (int s = 0; s < 16; ++s) {
            const int pos = pairs ? 4 * ((s & 7) >> 1) + (s & 1) + 2 * (s >> 3) : s;
            o[pos] = b[s];
        }
        qr[g] = *reinterpret_cast<const uint4*>(o);
    }
}

// ---- the int8-activation multiply -----------------------------------------------------------------------------------
//
// mma.sync.m16n8k32 with s8 operands and s32 accumulators: A (X tile) and B (W tile) both have k contiguous, 64 values (bytes)
// per shared-memory row = four 16-byte chunks, the very layout the binary16 kernel has for 32 halves, so the same swizzle and
// the same ldmatrix addressing carry over (one k32 step is two chunks).
constexpr int kBK8 = 64;

__device__ inline void mma_s8(int* c, unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

template <int BM, int BN, int WM, int WN, int STAGES, bool OUT_HALF, int MINB>
__global__ void __launch_bounds__(WM* WN * 32, MINB)
    gemm_s8_kernel(const signed char* __restrict__ x, int ldx, const float* __restrict__ sx, const signed char* __restrict__ w, int ldw,
                   const float* __restrict__ sw, void* __restrict__ y, int ldy, int t, int n, int k) {
    constexpr int kThreads = WM * WN * 32;
    constexpr int MI = BM / WM / 16;
    constexpr int NI = BN / WN / 8;
    static_assert(MI >= 1 && NI >= 2 && NI % 2 == 0, "warp tile");
    constexpr int kAStage = BM * kBK8, kBStage = BN * kBK8; // bytes
    extern __shared__ __align__(16) signed char smem8[];
    signed char* sa = smem8;
    signed char* sb = smem8 + STAGES * kAStage;

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int warp_m = warp / WN, warp_n = warp % WN;
    const int tiles_m = (t + BM - 1) / BM, tiles_n = (n + BN - 1) / BN;
    const int per_group = kS8Group * tiles_n;
    const int group = blockIdx.x / per_group, first_m = group * kS8Group;
    const int group_m = min(tiles_m - first_m, kS8Group);
    const int in_group = blockIdx.x % per_group;
    const int m0 = (first_m + in_group % group_m) * BM, n0 = (in_group / group_m) * BN;
    const int tiles = (k + kBK8 - 1) / kBK8;
    const int a_row0 = warp_m * MI * 16, b_row0 = warp_n * NI * 8;

    const auto load_tile = [&](int tile, int stage) {
        const int k0 = tile * kBK8;
        for (int c = tid; c < BM * 4; c += kThreads) {
            const int r = c >> 2, kc = (c & 3) * 16;
            const bool ok = m0 + r < t && k0 + kc < k;
            cp_async16(sa + stage * kAStage + r * kBK8 + swz(r, kc >> 4) * 16, ok ? x + static_cast<std::size_t>(m0 + r) * ldx + k0 + kc : x, ok);
        }
        for (int c = tid; c < BN * 4; c += kThreads) {
            const int r = c >> 2, kc = (c & 3) * 16;
            const bool ok = n0 + r < n && k0 + kc < k;
            cp_async16(sb + stage * kBStage + r * kBK8 + swz(r, kc >> 4) * 16, ok ? w + static_cast<std::size_t>(n0 + r) * ldw + k0 + kc : w, ok);
        }
    };

    int acc[MI][NI][4];
#pragma unroll
    for (int i = 0; i < MI; ++i)
#pragma unroll
        for (int j = 0; j < NI; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0;

#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < tiles) load_tile(s, s);
        cp_async_commit();
    }

    for (int tile = 0; tile < tiles; ++tile) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();
        {
            const int next = tile + STAGES - 1;
            if (next < tiles) load_tile(next, next % STAGES);
            cp_async_commit();
        }
        const signed char* a_base = sa + (tile % STAGES) * kAStage + a_row0 * kBK8;
        const signed char* b_base = sb + (tile % STAGES) * kBStage + b_row0 * kBK8;
        unsigned a[2][MI][4], b[2][NI][2];
#pragma unroll
        for (int step = 0; step < 2; ++step) {
#pragma unroll
            for (int i = 0; i < MI; ++i) {
                const int row = i * 16 + (lane & 15);
                ldmatrix_x4(a[step][i][0], a[step][i][1], a[step][i][2], a[step][i][3], a_base + row * kBK8 + swz(a_row0 + row, step * 2 + (lane >> 4)) * 16);
            }
#pragma unroll
            for (int j = 0; j < NI; j += 2) {
                const int row = j * 8 + (lane & 7) + (lane >> 4) * 8;
                ldmatrix_x4(b[step][j][0], b[step][j][1], b[step][j + 1][0], b[step][j + 1][1],
                            b_base + row * kBK8 + swz(b_row0 + row, step * 2 + ((lane >> 3) & 1)) * 16);
            }
        }
#pragma unroll
        for (int step = 0; step < 2; ++step)
#pragma unroll
            for (int i = 0; i < MI; ++i)
#pragma unroll
                for (int j = 0; j < NI; ++j) mma_s8(acc[i][j], a[step][i][0], a[step][i][1], a[step][i][2], a[step][i][3], b[step][j][0], b[step][j][1]);
    }
    cp_async_wait<0>();

    const int g = lane >> 2, q = lane & 3;
#pragma unroll
    for (int i = 0; i < MI; ++i) {
#pragma unroll
        for (int j = 0; j < NI; ++j) {
            const int col = n0 + warp_n * NI * 8 + j * 8 + 2 * q;
            if (col >= n) continue;
            const float s0 = sw[col], s1 = sw[col + 1];
#pragma unroll
            for (int half_row = 0; half_row < 2; ++half_row) {
                const int row = m0 + warp_m * MI * 16 + i * 16 + g + half_row * 8;
                if (row < t) {
                    const float rs = sx[row];
                    const float v0 = static_cast<float>(acc[i][j][half_row * 2]) * rs * s0, v1 = static_cast<float>(acc[i][j][half_row * 2 + 1]) * rs * s1;
                    if constexpr (OUT_HALF) {
                        *reinterpret_cast<__half2*>(static_cast<__half*>(y) + static_cast<std::size_t>(row) * ldy + col) = __floats2half2_rn(v0, v1);
                    } else {
                        *reinterpret_cast<float2*>(static_cast<float*>(y) + static_cast<std::size_t>(row) * ldy + col) = make_float2(v0, v1);
                    }
                }
            }
        }
    }
}

// ---- the binary16-activation multiply with int8 weights ---------------------------------------------------------------
//
// X tile: 32 halves per row (64 bytes), exactly as the binary16 kernel. W tile: 32 int8 per row (32 bytes = two 16-byte
// chunks, one per k16 step). ldmatrix of a W chunk hands thread (g, q) four bytes of row g: because the weights were stored
// in layout Fp16Pairs these are k = 2q, 2q+1, 2q+8, 2q+9 of the step, which is exactly what the two B registers of
// mma.m16n8k16 hold, so they only have to be widened: a byte b becomes the binary16 1024 + (b ^ 0x80) by or-ing the exponent
// of 1024 in front of it, and 1152 (= 1024 + 128) is subtracted again, exactly.
constexpr int kBK16 = 32;

__device__ inline int swz_w8(int r, int c) { return c ^ ((r >> 2) & 1); }

__device__ inline void widen_s8x4(unsigned v, unsigned& lo, unsigned& hi) {
    v ^= 0x80808080u;
    const unsigned a = __byte_perm(v, 0x64646464u, 0x7150u), b = __byte_perm(v, 0x64646464u, 0x7352u);
    const __half2 bias = __half2half2(__ushort_as_half(static_cast<unsigned short>(0x6480)));
    const __half2 ha = __hsub2(*reinterpret_cast<const __half2*>(&a), bias), hb = __hsub2(*reinterpret_cast<const __half2*>(&b), bias);
    lo = *reinterpret_cast<const unsigned*>(&ha);
    hi = *reinterpret_cast<const unsigned*>(&hb);
}

__device__ inline void mma_f32_k16(float* c, unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

template <int BM, int BN, int WM, int WN, int STAGES, bool OUT_HALF, int MINB>
__global__ void __launch_bounds__(WM* WN * 32, MINB)
    gemm_w8a16_kernel(const __half* __restrict__ x, int ldx, const signed char* __restrict__ w, int ldw, const float* __restrict__ sw, void* __restrict__ y,
                      int ldy, int t, int n, int k) {
    constexpr int kThreads = WM * WN * 32;
    constexpr int MI = BM / WM / 16;
    constexpr int NI = BN / WN / 8;
    static_assert(MI >= 1 && NI >= 2 && NI % 2 == 0, "warp tile");
    constexpr int kAStage = BM * 64, kBStage = BN * 32; // bytes
    extern __shared__ __align__(16) signed char smem8[];
    signed char* sa = smem8;
    signed char* sb = smem8 + STAGES * kAStage;

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int warp_m = warp / WN, warp_n = warp % WN;
    const int tiles_m = (t + BM - 1) / BM, tiles_n = (n + BN - 1) / BN;
    const int per_group = kS8Group * tiles_n;
    const int group = blockIdx.x / per_group, first_m = group * kS8Group;
    const int group_m = min(tiles_m - first_m, kS8Group);
    const int in_group = blockIdx.x % per_group;
    const int m0 = (first_m + in_group % group_m) * BM, n0 = (in_group / group_m) * BN;
    const int tiles = (k + kBK16 - 1) / kBK16;
    const int a_row0 = warp_m * MI * 16, b_row0 = warp_n * NI * 8;

    const auto load_tile = [&](int tile, int stage) {
        const int k0 = tile * kBK16;
        for (int c = tid; c < BM * 4; c += kThreads) {
            const int r = c >> 2, kc = (c & 3) * 8;
            const bool ok = m0 + r < t && k0 + kc < k;
            cp_async16(sa + stage * kAStage + r * 64 + swz(r, kc >> 3) * 16, ok ? x + static_cast<std::size_t>(m0 + r) * ldx + k0 + kc : x, ok);
        }
        for (int c = tid; c < BN * 2; c += kThreads) {
            const int r = c >> 1, kc = (c & 1) * 16;
            const bool ok = n0 + r < n && k0 + kc < k;
            cp_async16(sb + stage * kBStage + r * 32 + swz_w8(r, kc >> 4) * 16, ok ? w + static_cast<std::size_t>(n0 + r) * ldw + k0 + kc : w, ok);
        }
    };

    float acc[MI][NI][4];
#pragma unroll
    for (int i = 0; i < MI; ++i)
#pragma unroll
        for (int j = 0; j < NI; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.0f;

#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < tiles) load_tile(s, s);
        cp_async_commit();
    }

    for (int tile = 0; tile < tiles; ++tile) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();
        {
            const int next = tile + STAGES - 1;
            if (next < tiles) load_tile(next, next % STAGES);
            cp_async_commit();
        }
        const signed char* a_base = sa + (tile % STAGES) * kAStage + a_row0 * 64;
        const signed char* b_base = sb + (tile % STAGES) * kBStage + b_row0 * 32;
        unsigned a[2][MI][4], braw[2][NI];
#pragma unroll
        for (int step = 0; step < 2; ++step) {
#pragma unroll
            for (int i = 0; i < MI; ++i) {
                const int row = i * 16 + (lane & 15);
                ldmatrix_x4(a[step][i][0], a[step][i][1], a[step][i][2], a[step][i][3], a_base + row * 64 + swz(a_row0 + row, step * 2 + (lane >> 4)) * 16);
            }
#pragma unroll
            for (int j = 0; j < NI; j += 4) {
                if (NI - j >= 4) {
                    const int row = (j + (lane >> 3)) * 8 + (lane & 7);
                    ldmatrix_x4(braw[step][j], braw[step][j + 1], braw[step][j + 2], braw[step][j + 3], b_base + row * 32 + swz_w8(b_row0 + row, step) * 16);
                } else {
                    const int row = (j + ((lane >> 3) & 1)) * 8 + (lane & 7);
                    ldmatrix_x2(braw[step][j], braw[step][j + 1], b_base + row * 32 + swz_w8(b_row0 + row, step) * 16);
                }
            }
        }
#pragma unroll
        for (int step = 0; step < 2; ++step) {
            unsigned b[NI][2];
#pragma unroll
            for (int j = 0; j < NI; ++j) widen_s8x4(braw[step][j], b[j][0], b[j][1]);
#pragma unroll
            for (int i = 0; i < MI; ++i)
#pragma unroll
                for (int j = 0; j < NI; ++j) mma_f32_k16(acc[i][j], a[step][i][0], a[step][i][1], a[step][i][2], a[step][i][3], b[j][0], b[j][1]);
        }
    }
    cp_async_wait<0>();

    const int g = lane >> 2, q = lane & 3;
#pragma unroll
    for (int i = 0; i < MI; ++i) {
#pragma unroll
        for (int j = 0; j < NI; ++j) {
            const int col = n0 + warp_n * NI * 8 + j * 8 + 2 * q;
            if (col >= n) continue;
            const float s0 = sw[col], s1 = sw[col + 1];
#pragma unroll
            for (int half_row = 0; half_row < 2; ++half_row) {
                const int row = m0 + warp_m * MI * 16 + i * 16 + g + half_row * 8;
                if (row < t) {
                    const float v0 = acc[i][j][half_row * 2] * s0, v1 = acc[i][j][half_row * 2 + 1] * s1;
                    if constexpr (OUT_HALF) {
                        *reinterpret_cast<__half2*>(static_cast<__half*>(y) + static_cast<std::size_t>(row) * ldy + col) = __floats2half2_rn(v0, v1);
                    } else {
                        *reinterpret_cast<float2*>(static_cast<float*>(y) + static_cast<std::size_t>(row) * ldy + col) = make_float2(v0, v1);
                    }
                }
            }
        }
    }
}

// ---- launching ------------------------------------------------------------------------------------------------------

template <int BM, int BN, int WM, int WN, int STAGES, int MINB>
void launch_s8_config(bool y_half, const signed char* x, int ldx, const float* sx, const signed char* w, int ldw, const float* sw, void* y, int ldy, int t,
                      int n, int k, cudaStream_t stream) {
    constexpr std::size_t smem = static_cast<std::size_t>(STAGES) * (BM + BN) * kBK8;
    const auto go = [&](auto kernel) {
        if constexpr (smem > 48 * 1024) {
            check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)), "cudaFuncSetAttribute");
        }
        const dim3 grid(static_cast<unsigned>(((t + BM - 1) / BM) * ((n + BN - 1) / BN)));
        kernel<<<grid, dim3(WM * WN * 32), smem, stream>>>(x, ldx, sx, w, ldw, sw, y, ldy, t, n, k);
    };
    if (y_half) go(gemm_s8_kernel<BM, BN, WM, WN, STAGES, true, MINB>);
    else go(gemm_s8_kernel<BM, BN, WM, WN, STAGES, false, MINB>);
}

template <int BM, int BN, int WM, int WN, int STAGES, int MINB>
void launch_w8a16_config(bool y_half, const __half* x, int ldx, const signed char* w, int ldw, const float* sw, void* y, int ldy, int t, int n, int k,
                         cudaStream_t stream) {
    constexpr std::size_t smem = static_cast<std::size_t>(STAGES) * (BM * 64 + BN * 32);
    const auto go = [&](auto kernel) {
        if constexpr (smem > 48 * 1024) {
            check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)), "cudaFuncSetAttribute");
        }
        const dim3 grid(static_cast<unsigned>(((t + BM - 1) / BM) * ((n + BN - 1) / BN)));
        kernel<<<grid, dim3(WM * WN * 32), smem, stream>>>(x, ldx, w, ldw, sw, y, ldy, t, n, k);
    };
    if (y_half) go(gemm_w8a16_kernel<BM, BN, WM, WN, STAGES, true, MINB>);
    else go(gemm_w8a16_kernel<BM, BN, WM, WN, STAGES, false, MINB>);
}

// The same choice by problem size as the binary16 kernel (only the speed depends on it, not the bits).
inline int pick_s8_config(std::size_t t, std::size_t n) {
    struct Candidate {
        int config, bm, bn;
    };
    static constexpr Candidate candidates[] = {{0, 128, 128}, {1, 64, 128}, {3, 64, 64}, {4, 32, 64}, {5, 32, 32}};
    const std::size_t height_cap = t > 192 ? 128 : (t > 32 ? 64 : 32);
    constexpr std::size_t kWantedBlocks = 56;
    int fallback = 5;
    for (const Candidate& c : candidates) {
        if (static_cast<std::size_t>(c.bm) > height_cap) continue;
        const std::size_t blocks = (t + static_cast<std::size_t>(c.bm) - 1) / static_cast<std::size_t>(c.bm) * ((n + static_cast<std::size_t>(c.bn) - 1) / static_cast<std::size_t>(c.bn));
        if (blocks >= kWantedBlocks) return c.config;
        fallback = c.config;
    }
    return fallback;
}

} // namespace detail

void launch_quantize_rows_s8(const void* x, std::size_t ld_x, void* q, std::size_t ld_q, float* scales, std::size_t rows, std::size_t cols,
                             cudaStream_t stream) {
    if (rows == 0 || cols == 0) return;
    if (x == nullptr || q == nullptr || scales == nullptr) throw std::invalid_argument("quantize_rows_s8: null pointer");
    if (cols % 16 != 0 || ld_x % 16 != 0 || ld_q % 16 != 0 || ld_x < cols || ld_q < cols || reinterpret_cast<std::uintptr_t>(x) % 16 != 0 ||
        reinterpret_cast<std::uintptr_t>(q) % 16 != 0) {
        throw std::invalid_argument("quantize_rows_s8: cols and both row strides must be multiples of 16, strides at least cols, pointers 16-byte aligned");
    }
    if (rows > 0x7FFFFFFF || cols > 0x7FFFFFFF) throw std::invalid_argument("quantize_rows_s8: dimension beyond 32 bits");
    const int cols8 = static_cast<int>(cols / 8);
    const unsigned threads = cols8 >= 256 ? 256u : (static_cast<unsigned>(cols8) + 31u) / 32u * 32u;
    detail::quantize_rows_s8_kernel<<<dim3(static_cast<unsigned>(rows)), dim3(threads), 0, stream>>>(
        static_cast<const uint4*>(x), ld_x / 8, static_cast<uint2*>(q), ld_q / 8, scales, cols8);
    check(cudaGetLastError(), "quantize_rows_s8 kernel launch");
}

void launch_quantize_weights_s8(const void* w, std::size_t ld_w, void* wq, float* scales, std::size_t n, std::size_t k, S8WeightLayout layout,
                                cudaStream_t stream) {
    if (n == 0 || k == 0) return;
    if (w == nullptr || wq == nullptr || scales == nullptr) throw std::invalid_argument("quantize_weights_s8: null pointer");
    if (k % 16 != 0 || ld_w % 16 != 0 || ld_w < k || reinterpret_cast<std::uintptr_t>(w) % 16 != 0 || reinterpret_cast<std::uintptr_t>(wq) % 16 != 0) {
        throw std::invalid_argument("quantize_weights_s8: k and the row stride must be multiples of 16, stride at least k, pointers 16-byte aligned");
    }
    if (n > 0x7FFFFFFF || k > 0x7FFFFFFF) throw std::invalid_argument("quantize_weights_s8: dimension beyond 32 bits");
    const int groups = static_cast<int>(k / 16);
    const unsigned threads = groups >= 256 ? 256u : (static_cast<unsigned>(groups) + 31u) / 32u * 32u;
    detail::quantize_weights_s8_kernel<<<dim3(static_cast<unsigned>(n)), dim3(threads), 0, stream>>>(
        static_cast<const uint4*>(w), ld_w / 8, static_cast<uint4*>(wq), scales, groups, layout == S8WeightLayout::Fp16Pairs);
    check(cudaGetLastError(), "quantize_weights_s8 kernel launch");
}

void launch_gemm_s8(const void* xq, std::size_t ld_x, const float* sx, const void* wq, std::size_t ld_w, const float* sw, void* y, std::size_t ld_y,
                    bool y_half, std::size_t t, std::size_t n, std::size_t k, int config, cudaStream_t stream) {
    if (t == 0 || n == 0) return;
    if (xq == nullptr || wq == nullptr || sx == nullptr || sw == nullptr || y == nullptr) throw std::invalid_argument("gemm_s8: null pointer");
    if (!gemm_s8_supports(n, k, ld_x, ld_w, ld_y) || reinterpret_cast<std::uintptr_t>(xq) % 16 != 0 || reinterpret_cast<std::uintptr_t>(wq) % 16 != 0 ||
        reinterpret_cast<std::uintptr_t>(y) % 8 != 0 || reinterpret_cast<std::uintptr_t>(sx) % 4 != 0 || reinterpret_cast<std::uintptr_t>(sw) % 4 != 0) {
        throw std::invalid_argument("gemm_s8: K and the row strides of X and W must be multiples of 16, N and the row stride of Y even, pointers aligned");
    }
    if (t > 0x7FFFFFFF || n > 0x7FFFFFFF || k > 0x7FFFFFFF) throw std::invalid_argument("gemm_s8: dimension beyond 32 bits");
    const auto* xs = static_cast<const signed char*>(xq);
    const auto* ws = static_cast<const signed char*>(wq);
    const int it = static_cast<int>(t), in = static_cast<int>(n), ik = static_cast<int>(k), ilx = static_cast<int>(ld_x), ilw = static_cast<int>(ld_w),
              ily = static_cast<int>(ld_y);
    if (config < 0) config = detail::pick_s8_config(t, n);
    switch (config) {
    case 0: detail::launch_s8_config<128, 128, 2, 4, 3, 2>(y_half, xs, ilx, sx, ws, ilw, sw, y, ily, it, in, ik, stream); break;
    case 1: detail::launch_s8_config<64, 128, 1, 4, 4, 1>(y_half, xs, ilx, sx, ws, ilw, sw, y, ily, it, in, ik, stream); break;
    case 2: detail::launch_s8_config<32, 128, 1, 4, 4, 1>(y_half, xs, ilx, sx, ws, ilw, sw, y, ily, it, in, ik, stream); break;
    case 3: detail::launch_s8_config<64, 64, 2, 2, 6, 4>(y_half, xs, ilx, sx, ws, ilw, sw, y, ily, it, in, ik, stream); break;
    case 4: detail::launch_s8_config<32, 64, 1, 2, 6, 4>(y_half, xs, ilx, sx, ws, ilw, sw, y, ily, it, in, ik, stream); break;
    case 5: detail::launch_s8_config<32, 32, 1, 2, 8, 6>(y_half, xs, ilx, sx, ws, ilw, sw, y, ily, it, in, ik, stream); break;
    default: throw std::invalid_argument("gemm_s8: unknown configuration");
    }
    check(cudaGetLastError(), "gemm_s8 kernel launch");
}

void launch_gemm_w8a16(const void* x, std::size_t ld_x, const void* wq, std::size_t ld_w, const float* sw, void* y, std::size_t ld_y, bool y_half,
                       std::size_t t, std::size_t n, std::size_t k, int config, cudaStream_t stream) {
    if (t == 0 || n == 0) return;
    if (x == nullptr || wq == nullptr || sw == nullptr || y == nullptr) throw std::invalid_argument("gemm_w8a16: null pointer");
    if (!gemm_w8a16_supports(n, k, ld_x, ld_w, ld_y) || reinterpret_cast<std::uintptr_t>(x) % 16 != 0 || reinterpret_cast<std::uintptr_t>(wq) % 16 != 0 ||
        reinterpret_cast<std::uintptr_t>(y) % 8 != 0 || reinterpret_cast<std::uintptr_t>(sw) % 4 != 0) {
        throw std::invalid_argument("gemm_w8a16: K and the row stride of W must be multiples of 16, the stride of X of 8, N and the stride of Y even, pointers aligned");
    }
    if (t > 0x7FFFFFFF || n > 0x7FFFFFFF || k > 0x7FFFFFFF) throw std::invalid_argument("gemm_w8a16: dimension beyond 32 bits");
    const auto* xh = static_cast<const __half*>(x);
    const auto* ws = static_cast<const signed char*>(wq);
    const int it = static_cast<int>(t), in = static_cast<int>(n), ik = static_cast<int>(k), ilx = static_cast<int>(ld_x), ilw = static_cast<int>(ld_w),
              ily = static_cast<int>(ld_y);
    if (config < 0) config = detail::pick_s8_config(t, n);
    switch (config) {
    case 0: detail::launch_w8a16_config<128, 128, 2, 4, 3, 2>(y_half, xh, ilx, ws, ilw, sw, y, ily, it, in, ik, stream); break;
    case 1: detail::launch_w8a16_config<64, 128, 1, 4, 4, 1>(y_half, xh, ilx, ws, ilw, sw, y, ily, it, in, ik, stream); break;
    case 2: detail::launch_w8a16_config<32, 128, 1, 4, 4, 1>(y_half, xh, ilx, ws, ilw, sw, y, ily, it, in, ik, stream); break;
    case 3: detail::launch_w8a16_config<64, 64, 2, 2, 6, 4>(y_half, xh, ilx, ws, ilw, sw, y, ily, it, in, ik, stream); break;
    case 4: detail::launch_w8a16_config<32, 64, 1, 2, 6, 4>(y_half, xh, ilx, ws, ilw, sw, y, ily, it, in, ik, stream); break;
    case 5: detail::launch_w8a16_config<32, 32, 1, 2, 8, 6>(y_half, xh, ilx, ws, ilw, sw, y, ily, it, in, ik, stream); break;
    default: throw std::invalid_argument("gemm_w8a16: unknown configuration");
    }
    check(cudaGetLastError(), "gemm_w8a16 kernel launch");
}

} // namespace sepal::cuda
