#include "gemm_kernels.hpp"

#include <cstdint>

#include <cuda_fp16.h>

#include "device.hpp"
#include "gemm_ptx.cuh"

namespace sepal::cuda {

// External linkage on purpose, see convert_kernels.cu.
namespace detail {

constexpr unsigned kConvertBlock = 256;

// Eight elements per thread: one 16-byte load, two 16-byte stores.
__global__ void half_to_float_2d_vec8_kernel(const uint4* src, std::size_t ld_src8, float4* dst, std::size_t ld_dst4,
                                             std::size_t vec_per_row, std::size_t total) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const std::size_t r = i / vec_per_row, c = i % vec_per_row;
    const uint4 v = src[r * ld_src8 + c];
    const __half2* h = reinterpret_cast<const __half2*>(&v);
    const float2 a = __half22float2(h[0]), b = __half22float2(h[1]), d = __half22float2(h[2]), e = __half22float2(h[3]);
    float4* out = dst + r * ld_dst4 + c * 2;
    out[0] = make_float4(a.x, a.y, b.x, b.y);
    out[1] = make_float4(d.x, d.y, e.x, e.y);
}

__global__ void half_to_float_2d_scalar_kernel(const __half* src, std::size_t ld_src, float* dst, std::size_t ld_dst,
                                               std::size_t cols, std::size_t total) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const std::size_t r = i / cols, c = i % cols;
    dst[r * ld_dst + c] = __half2float(src[r * ld_src + c]);
}

// ---- the batch-invariant tensor-core multiply ------------------------------------------------------------------------
//
// Y[t, n] = sum over k of X[t, k] * W[n, k], X [T, K] and W [N, K] binary16 with K contiguous, which is exactly the
// operand layout mma.sync.m16n8k16.row.col wants, so both tiles go from global memory to shared memory (cp.async) and
// from there to registers (ldmatrix) without a transpose.
//
// Arithmetic. Each output element is a chain of m16n8k16 tensor-core steps over k = 0, 16, 32, ... in that order (a K that
// is not a multiple of 32 is zero-padded). Two accumulation modes:
//   float32:  the chain accumulates in float32 registers (mma ...f32.f16.f16.f32);
//   binary16: the chain accumulates in binary16 registers (mma ...f16.f16.f16.f16), which run twice as fast on GeForce
//             cards, and every `flush_tiles` tiles of 32 (never when 0) the binary16 partial sum is added to a float32
//             accumulator and cleared.
// The chain of an element depends on K, the mode and the flush interval only, and not on T, N, the tile shape, the
// element's place in the tile, or the other rows: a row has the same bits whatever kernel configuration was chosen.
constexpr int kBK = 32;
constexpr int kGroupM = 4;  // blocks run in groups of this many row tiles, so that the blocks resident together share panels in L2
constexpr int kPitch = kBK; // halves per shared-memory row: 64 bytes, four 16-byte chunks, stored XOR-swizzled (see swz)

__device__ inline void mma_f16(unsigned& c0, unsigned& c1, unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
#if SEPAL_TENSOR_CORES
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%0, %1};\n"
                 : "+r"(c0), "+r"(c1)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
    __trap();
#endif
}
__device__ inline void mma_f32(float* c, unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
#if SEPAL_TENSOR_CORES
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
    __trap();
#endif
}

__device__ inline void flush_pair(unsigned& h, float& lo, float& hi) {
    const float2 f = __half22float2(*reinterpret_cast<__half2*>(&h));
    lo += f.x;
    hi += f.y;
    h = 0u;
}

template <int BM, int BN, int WM, int WN, int STAGES, bool ACC16, bool OUT_HALF, int MINB>
__global__ void __launch_bounds__(WM* WN * 32, MINB)
    gemm_nt_kernel(const __half* __restrict__ x, int ldx, const __half* __restrict__ w, int ldw, void* __restrict__ y, int ldy, int t, int n,
                   int k, int flush_tiles) {
    constexpr int kThreads = WM * WN * 32;
    constexpr int MI = BM / WM / 16; // m16 tiles per warp
    constexpr int NI = BN / WN / 8;  // n8 tiles per warp
    static_assert(MI >= 1 && NI >= 2 && NI % 2 == 0, "warp tile");
    constexpr int kAStage = BM * kPitch, kBStage = BN * kPitch;
    extern __shared__ __align__(16) __half smem[];
    __half* sa = smem;
    __half* sb = smem + STAGES * kAStage;

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int warp_m = warp / WN, warp_n = warp % WN;
    const int tiles_m = (t + BM - 1) / BM, tiles_n = (n + BN - 1) / BN;
    const int per_group = kGroupM * tiles_n;
    const int group = blockIdx.x / per_group, first_m = group * kGroupM;
    const int group_m = min(tiles_m - first_m, kGroupM);
    const int in_group = blockIdx.x % per_group;
    const int m0 = (first_m + in_group % group_m) * BM, n0 = (in_group / group_m) * BN;
    const int tiles = (k + kBK - 1) / kBK;
    const int a_row0 = warp_m * MI * 16, b_row0 = warp_n * NI * 8; // first tile row of this warp's A and B fragments

    // Global -> shared: each tile row is 32 halves = 4 chunks of 16 bytes; chunks outside the matrix are zero-filled.
    const auto load_tile = [&](int tile, int stage) {
        const int k0 = tile * kBK;
        for (int c = tid; c < BM * 4; c += kThreads) {
            const int r = c >> 2, kc = (c & 3) * 8;
            const bool ok = m0 + r < t && k0 + kc < k;
            cp_async16(sa + stage * kAStage + r * kPitch + swz(r, kc >> 3) * 8, ok ? x + static_cast<std::size_t>(m0 + r) * ldx + k0 + kc : x, ok);
        }
        for (int c = tid; c < BN * 4; c += kThreads) {
            const int r = c >> 2, kc = (c & 3) * 8;
            const bool ok = n0 + r < n && k0 + kc < k;
            cp_async16(sb + stage * kBStage + r * kPitch + swz(r, kc >> 3) * 8, ok ? w + static_cast<std::size_t>(n0 + r) * ldw + k0 + kc : w, ok);
        }
    };

    unsigned acc16[MI][NI][2];
    float acc32[MI][NI][4];
#pragma unroll
    for (int i = 0; i < MI; ++i)
#pragma unroll
        for (int j = 0; j < NI; ++j) {
            acc16[i][j][0] = acc16[i][j][1] = 0u;
            acc32[i][j][0] = acc32[i][j][1] = acc32[i][j][2] = acc32[i][j][3] = 0.0f;
        }

#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < tiles) load_tile(s, s);
        cp_async_commit();
    }

    int since_flush = 0;
    for (int tile = 0; tile < tiles; ++tile) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();
        {
            const int next = tile + STAGES - 1;
            if (next < tiles) load_tile(next, next % STAGES);
            cp_async_commit();
        }
        const __half* a_base = sa + (tile % STAGES) * kAStage + a_row0 * kPitch;
        const __half* b_base = sb + (tile % STAGES) * kBStage + b_row0 * kPitch;
        // Both k16 steps of the tile are loaded before the first multiply, so that the shared-memory latency of the second
        // overlaps the multiplies of the first (the asm statements are volatile: the order written is the order issued).
        unsigned a[2][MI][4], b[2][NI][2];
#pragma unroll
        for (int step = 0; step < 2; ++step) {
            const int kk = step * 16;
#pragma unroll
            for (int i = 0; i < MI; ++i) {
                const int row = i * 16 + (lane & 15);
                ldmatrix_x4(a[step][i][0], a[step][i][1], a[step][i][2], a[step][i][3], a_base + row * kPitch + swz(a_row0 + row, (kk >> 3) + (lane >> 4)) * 8);
            }
#pragma unroll
            for (int j = 0; j < NI; j += 2) {
                const int row = j * 8 + (lane & 7) + (lane >> 4) * 8;
                ldmatrix_x4(b[step][j][0], b[step][j][1], b[step][j + 1][0], b[step][j + 1][1],
                            b_base + row * kPitch + swz(b_row0 + row, (kk >> 3) + ((lane >> 3) & 1)) * 8);
            }
        }
#pragma unroll
        for (int step = 0; step < 2; ++step)
#pragma unroll
            for (int i = 0; i < MI; ++i)
#pragma unroll
                for (int j = 0; j < NI; ++j) {
                    if constexpr (ACC16) {
                        mma_f16(acc16[i][j][0], acc16[i][j][1], a[step][i][0], a[step][i][1], a[step][i][2], a[step][i][3], b[step][j][0], b[step][j][1]);
                    } else {
                        mma_f32(acc32[i][j], a[step][i][0], a[step][i][1], a[step][i][2], a[step][i][3], b[step][j][0], b[step][j][1]);
                    }
                }
        if constexpr (ACC16) {
            if (flush_tiles != 0 && ++since_flush == flush_tiles) {
                since_flush = 0;
#pragma unroll
                for (int i = 0; i < MI; ++i)
#pragma unroll
                    for (int j = 0; j < NI; ++j) {
                        flush_pair(acc16[i][j][0], acc32[i][j][0], acc32[i][j][1]);
                        flush_pair(acc16[i][j][1], acc32[i][j][2], acc32[i][j][3]);
                    }
            }
        }
    }
    cp_async_wait<0>();
    if constexpr (ACC16) {
        // The tail (or the whole sum, when flush_tiles is 0).
#pragma unroll
        for (int i = 0; i < MI; ++i)
#pragma unroll
            for (int j = 0; j < NI; ++j) {
                flush_pair(acc16[i][j][0], acc32[i][j][0], acc32[i][j][1]);
                flush_pair(acc16[i][j][1], acc32[i][j][2], acc32[i][j][3]);
            }
    }

    // Epilogue: fragment element (g, 2q), (g, 2q + 1) and (g + 8, ...) of each m16n8 tile.
    const int g = lane >> 2, q = lane & 3;
#pragma unroll
    for (int i = 0; i < MI; ++i) {
#pragma unroll
        for (int j = 0; j < NI; ++j) {
            const int col = n0 + warp_n * NI * 8 + j * 8 + 2 * q;
#pragma unroll
            for (int half_row = 0; half_row < 2; ++half_row) {
                const int row = m0 + warp_m * MI * 16 + i * 16 + g + half_row * 8;
                if (row < t && col < n) {
                    const float v0 = acc32[i][j][half_row * 2], v1 = acc32[i][j][half_row * 2 + 1];
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

template <int BM, int BN, int WM, int WN, int STAGES, int MINB>
void launch_config(bool acc16, bool y_half, const __half* x, int ldx, const __half* w, int ldw, void* y, int ldy, int t, int n, int k,
                   int flush_tiles, cudaStream_t stream) {
    constexpr std::size_t smem = static_cast<std::size_t>(STAGES) * (BM + BN) * kPitch * sizeof(__half);
    const auto go = [&](auto kernel) {
        if constexpr (smem > 48 * 1024) {
            check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)), "cudaFuncSetAttribute");
        }
        const dim3 grid(static_cast<unsigned>(((t + BM - 1) / BM) * ((n + BN - 1) / BN)));
        kernel<<<grid, dim3(WM * WN * 32), smem, stream>>>(x, ldx, w, ldw, y, ldy, t, n, k, flush_tiles);
    };
    if (acc16) {
        if (y_half) go(gemm_nt_kernel<BM, BN, WM, WN, STAGES, true, true, MINB>);
        else go(gemm_nt_kernel<BM, BN, WM, WN, STAGES, true, false, MINB>);
    } else {
        if (y_half) go(gemm_nt_kernel<BM, BN, WM, WN, STAGES, false, true, MINB>);
        else go(gemm_nt_kernel<BM, BN, WM, WN, STAGES, false, false, MINB>);
    }
}

// Tile shape by problem size: the largest tile that still gives every SM two or more blocks, never taller than the batch needs.
// Only the speed depends on this choice, not the bits (see the note on the arithmetic above).
inline int pick_config(std::size_t t, std::size_t n) {
    // Up to 128 rows the count of blocks is not the whole story (a block that streams K = 1,024 to 6,144 of a narrow output is long and
    // there are few of them): the shape per size below is the one that was fastest on the engine's multiplies of both geometries
    // (sepal-cuda-gemm-probe shapes --config, shorter of two passes; narrow = N below 4,096), 15 to 45% of the multiplies' time
    // against the rule that follows. Above it the rule holds.
    if (t <= 128) {
        const bool wide = n >= 4096;
        if (t <= 32) return 5;
        if (t <= 64) return wide ? 4 : 5;
        if (t <= 96) return wide ? 2 : 4;
        return wide ? 0 : 3;
    }
    struct Candidate {
        int config, bm, bn;
    };
    static constexpr Candidate candidates[] = {{0, 128, 128}, {1, 64, 128}, {3, 64, 64}, {4, 32, 64}, {5, 32, 32}};
    const std::size_t height_cap = t > 192 ? 128 : (t > 32 ? 64 : 32);
    constexpr std::size_t kWantedBlocks = 56; // two per SM of the RTX 3060; other cards just get a similar tile
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

void launch_half_to_float_2d(const void* src, std::size_t ld_src, void* dst, std::size_t ld_dst, std::size_t rows,
                             std::size_t cols, cudaStream_t stream) {
    if (rows == 0 || cols == 0) return;
    if (src == nullptr || dst == nullptr) throw std::invalid_argument("half_to_float_2d: null pointer");
    if (ld_src < cols || ld_dst < cols) throw std::invalid_argument("half_to_float_2d: a row stride is shorter than the row");
    const bool vec = cols % 8 == 0 && ld_src % 8 == 0 && ld_dst % 4 == 0 && reinterpret_cast<std::uintptr_t>(src) % 16 == 0 &&
                     reinterpret_cast<std::uintptr_t>(dst) % 16 == 0;
    if (vec) {
        const std::size_t per_row = cols / 8, total = rows * per_row;
        detail::half_to_float_2d_vec8_kernel<<<dim3(static_cast<unsigned>((total + detail::kConvertBlock - 1) / detail::kConvertBlock)),
                                               dim3(detail::kConvertBlock), 0, stream>>>(
            static_cast<const uint4*>(src), ld_src / 8, static_cast<float4*>(dst), ld_dst / 4, per_row, total);
    } else {
        const std::size_t total = rows * cols;
        detail::half_to_float_2d_scalar_kernel<<<dim3(static_cast<unsigned>((total + detail::kConvertBlock - 1) / detail::kConvertBlock)),
                                                 dim3(detail::kConvertBlock), 0, stream>>>(
            static_cast<const __half*>(src), ld_src, static_cast<float*>(dst), ld_dst, cols, total);
    }
    check(cudaGetLastError(), "half_to_float_2d kernel launch");
}

void launch_gemm_nt(const void* x, std::size_t ld_x, const void* w, std::size_t ld_w, void* y, std::size_t ld_y, bool y_half, std::size_t t,
                    std::size_t n, std::size_t k, GemmAccumulate accumulate, unsigned flush_k, int config, cudaStream_t stream) {
    if (t == 0 || n == 0) return;
    if (x == nullptr || w == nullptr || y == nullptr) throw std::invalid_argument("gemm_nt: null pointer");
    if (!gemm_nt_supports(n, k, ld_x, ld_w, ld_y) || flush_k % detail::kBK != 0 || reinterpret_cast<std::uintptr_t>(x) % 16 != 0 ||
        reinterpret_cast<std::uintptr_t>(w) % 16 != 0 || reinterpret_cast<std::uintptr_t>(y) % 8 != 0) {
        throw std::invalid_argument(
            "gemm_nt: K and the row strides of X and W must be multiples of 8, N and the row stride of Y even, flush_k a multiple of 32, pointers aligned");
    }
    if (t > 0x7FFFFFFF || n > 0x7FFFFFFF || k > 0x7FFFFFFF) throw std::invalid_argument("gemm_nt: dimension beyond 32 bits");
    const __half* xh = static_cast<const __half*>(x);
    const __half* wh = static_cast<const __half*>(w);
    const int it = static_cast<int>(t), in = static_cast<int>(n), ik = static_cast<int>(k), ft = static_cast<int>(flush_k / detail::kBK);
    const int ilx = static_cast<int>(ld_x), ilw = static_cast<int>(ld_w), ily = static_cast<int>(ld_y);
    const bool acc16 = accumulate == GemmAccumulate::Float16Flush;
    if (config < 0) config = detail::pick_config(t, n);
    switch (config) {
    case 0: detail::launch_config<128, 128, 2, 4, 3, 2>(acc16, y_half, xh, ilx, wh, ilw, y, ily, it, in, ik, ft, stream); break;
    case 1: detail::launch_config<64, 128, 1, 4, 4, 1>(acc16, y_half, xh, ilx, wh, ilw, y, ily, it, in, ik, ft, stream); break;
    case 2: detail::launch_config<32, 128, 1, 4, 4, 1>(acc16, y_half, xh, ilx, wh, ilw, y, ily, it, in, ik, ft, stream); break;
    case 3: detail::launch_config<64, 64, 2, 2, 6, 4>(acc16, y_half, xh, ilx, wh, ilw, y, ily, it, in, ik, ft, stream); break;
    case 4: detail::launch_config<32, 64, 1, 2, 6, 4>(acc16, y_half, xh, ilx, wh, ilw, y, ily, it, in, ik, ft, stream); break;
    case 5: detail::launch_config<32, 32, 1, 2, 8, 6>(acc16, y_half, xh, ilx, wh, ilw, y, ily, it, in, ik, ft, stream); break;
    default: throw std::invalid_argument("gemm_nt: unknown configuration");
    }
    check(cudaGetLastError(), "gemm_nt kernel launch");
}

} // namespace sepal::cuda
