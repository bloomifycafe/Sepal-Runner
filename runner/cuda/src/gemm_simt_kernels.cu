// The float32 SIMT multiply; see gemm_simt_kernels.hpp for the contract.
//
// Kernel shape. A block computes a BM x 128 tile of Y (BM = 128, 64 or 32 rows of X; 128 weight rows) with 2 BM threads,
// each owning 8 x 8 outputs: rows ty*4 + i and BM/2 + ty*4 + i, columns tx*4 + j and 64 + tx*4 + j (i, j < 4). k runs in
// tiles of 16: X and the widened weights are staged transposed in shared memory ([k][row], [k][column]) so that a thread
// reads its four rows and four columns as float4, and the next tile is fetched from global memory into registers while
// the current one is multiplied (two shared buffers, one barrier per tile). Global loads are 16 bytes wide.
#include "gemm_simt_kernels.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

#include "device.hpp"

namespace sepal::cuda {

// The kernels have external linkage on purpose (see convert_kernels.cu): a named namespace, not an anonymous one.
namespace simt {

constexpr int kBN = 128;
constexpr int kBK = 16;
constexpr int kPad = 4;

// silu(g) * u exactly as kernels_elementwise.cu computes it (launch_swiglu).
__device__ __forceinline__ float softplus_e(float x) { return fmaxf(x, 0.0f) + log1pf(expf(-fabsf(x))); }
__device__ __forceinline__ float swiglu_e(float g, float u) { return __fmul_rn(__fmul_rn(g, expf(-softplus_e(-g))), u); }

__device__ __forceinline__ float bf16_lo(unsigned v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float bf16_hi(unsigned v) { return __uint_as_float(v & 0xFFFF0000u); }
// The signed byte `byte` of v as a float, exactly: 2^23 + 128 + b is a float with b + 128 in its low mantissa bits.
__device__ __forceinline__ float s8_at(unsigned v, int byte) {
    return __fsub_rn(__uint_as_float(0x4B000000u | (((v >> (8 * byte)) & 0xFFu) ^ 0x80u)), 8388736.0f);
}

template <SimtWeight WT>
struct WeightTraits;
template <>
struct WeightTraits<SimtWeight::F32> {
    static constexpr int kVecs = 4;   // 16-byte vectors per weight row per k tile
    static constexpr int kPerVec = 4; // values per vector
    static constexpr int kBytes = 4;
};
template <>
struct WeightTraits<SimtWeight::BF16> {
    static constexpr int kVecs = 2;
    static constexpr int kPerVec = 8;
    static constexpr int kBytes = 2;
};
template <>
struct WeightTraits<SimtWeight::Int8Group> {
    static constexpr int kVecs = 2;   // 8-byte vectors here, so that every thread of a 128-row block converts the same amount
    static constexpr int kPerVec = 8;
    static constexpr int kBytes = 1;
};

struct Params {
    const float* x;
    std::size_t ld_x;
    const unsigned char* w;
    std::size_t ld_w_bytes;
    const float* scales;
    std::size_t ld_s;
    float* y;
    std::size_t ld_y;
    int t, n, k;
};

template <int BM, SimtWeight WT, SimtEpilogue EP>
__global__ void __launch_bounds__(2 * BM, BM == 128 ? 2 : BM == 64 ? 3 : 6) gemm_simt_kernel(const Params p) {
    constexpr int kThreads = 2 * BM;
    using Tr = WeightTraits<WT>;
    constexpr int kWVecs = kBN * Tr::kVecs;                        // weight vectors per tile
    constexpr int kWPer = (kWVecs + kThreads - 1) / kThreads;      // per thread
    constexpr int kXPer = 2;                                        // X float4 per thread: BM * 16 / 4 / (2 BM)

    __shared__ __align__(16) float As[2][kBK][BM + kPad];
    __shared__ __align__(16) float Bs[2][kBK][kBN + kPad];

    const int tid = static_cast<int>(threadIdx.x);
    const int tx = tid % 16;
    const int ty = tid / 16;
    const int n0 = static_cast<int>(blockIdx.x) * kBN;
    const int m0 = static_cast<int>(blockIdx.y) * BM;

    float acc[8][8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[i][j] = 0.0f;

    float4 xr[kXPer];
    uint4 wr[kWPer];
    [[maybe_unused]] float sr[kWPer];

    // This thread's source rows, fixed for the whole k loop (null: outside the matrix, the tile gets zeros).
    const float* xsrc[kXPer];
#pragma unroll
    for (int e = 0; e < kXPer; ++e) {
        const int idx = tid + e * kThreads;
        const int m = m0 + idx / 4;
        xsrc[e] = m < p.t ? p.x + static_cast<std::size_t>(m) * p.ld_x + (idx % 4) * 4 : nullptr;
    }
    const unsigned char* wsrc[kWPer];
    [[maybe_unused]] const float* ssrc[kWPer];
#pragma unroll
    for (int e = 0; e < kWPer; ++e) {
        const int v = tid + e * kThreads;
        const int nn = n0 + v / Tr::kVecs;
        const bool on = v < kWVecs && nn < p.n;
        wsrc[e] = on ? p.w + static_cast<std::size_t>(nn) * p.ld_w_bytes + static_cast<std::size_t>((v % Tr::kVecs) * Tr::kPerVec) * Tr::kBytes : nullptr;
        if constexpr (WT == SimtWeight::Int8Group) ssrc[e] = on ? p.scales + static_cast<std::size_t>(nn) * p.ld_s : nullptr;
    }

    const auto load = [&](int k0) {
#pragma unroll
        for (int e = 0; e < kXPer; ++e) {
            xr[e] = xsrc[e] != nullptr ? *reinterpret_cast<const float4*>(xsrc[e] + k0) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
#pragma unroll
        for (int e = 0; e < kWPer; ++e) {
            if (wsrc[e] != nullptr) {
                const unsigned char* src = wsrc[e] + static_cast<std::size_t>(k0) * Tr::kBytes;
                if constexpr (WT == SimtWeight::Int8Group) {
                    const uint2 v8 = *reinterpret_cast<const uint2*>(src);
                    wr[e] = make_uint4(v8.x, v8.y, 0u, 0u);
                } else {
                    wr[e] = *reinterpret_cast<const uint4*>(src);
                }
                if constexpr (WT == SimtWeight::Int8Group) sr[e] = ssrc[e][k0 / 128];
            } else {
                wr[e] = make_uint4(0u, 0u, 0u, 0u);
                if constexpr (WT == SimtWeight::Int8Group) sr[e] = 0.0f;
            }
        }
    };

    const auto store = [&](int buf) {
#pragma unroll
        for (int e = 0; e < kXPer; ++e) {
            const int idx = tid + e * kThreads;
            const int row = idx / 4;
            const int kq = (idx % 4) * 4;
            As[buf][kq + 0][row] = xr[e].x;
            As[buf][kq + 1][row] = xr[e].y;
            As[buf][kq + 2][row] = xr[e].z;
            As[buf][kq + 3][row] = xr[e].w;
        }
#pragma unroll
        for (int e = 0; e < kWPer; ++e) {
            const int v = tid + e * kThreads;
            if (v >= kWVecs) continue;
            const int row = v / Tr::kVecs;
            const int kb = (v % Tr::kVecs) * Tr::kPerVec;
            const uint4 q = wr[e];
            if constexpr (WT == SimtWeight::F32) {
                Bs[buf][kb + 0][row] = __uint_as_float(q.x);
                Bs[buf][kb + 1][row] = __uint_as_float(q.y);
                Bs[buf][kb + 2][row] = __uint_as_float(q.z);
                Bs[buf][kb + 3][row] = __uint_as_float(q.w);
            } else if constexpr (WT == SimtWeight::BF16) {
                Bs[buf][kb + 0][row] = bf16_lo(q.x);
                Bs[buf][kb + 1][row] = bf16_hi(q.x);
                Bs[buf][kb + 2][row] = bf16_lo(q.y);
                Bs[buf][kb + 3][row] = bf16_hi(q.y);
                Bs[buf][kb + 4][row] = bf16_lo(q.z);
                Bs[buf][kb + 5][row] = bf16_hi(q.z);
                Bs[buf][kb + 6][row] = bf16_lo(q.w);
                Bs[buf][kb + 7][row] = bf16_hi(q.w);
            } else {
                const float s = sr[e];
#pragma unroll
                for (int b = 0; b < 4; ++b) Bs[buf][kb + b][row] = __fmul_rn(s8_at(q.x, b), s);
#pragma unroll
                for (int b = 0; b < 4; ++b) Bs[buf][kb + 4 + b][row] = __fmul_rn(s8_at(q.y, b), s);
            }
        }
    };

    const int tiles = p.k / kBK;
    load(0);
    store(0);
    __syncthreads();
    for (int kt = 0; kt < tiles; ++kt) {
        const int buf = kt & 1;
        if (kt + 1 < tiles) load((kt + 1) * kBK);
#pragma unroll
        for (int kk = 0; kk < kBK; ++kk) {
            const float4 a0 = *reinterpret_cast<const float4*>(&As[buf][kk][ty * 4]);
            const float4 a1 = *reinterpret_cast<const float4*>(&As[buf][kk][BM / 2 + ty * 4]);
            const float4 b0 = *reinterpret_cast<const float4*>(&Bs[buf][kk][tx * 4]);
            const float4 b1 = *reinterpret_cast<const float4*>(&Bs[buf][kk][64 + tx * 4]);
            const float a[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            const float b[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
            for (int i = 0; i < 8; ++i)
#pragma unroll
                for (int j = 0; j < 8; ++j) acc[i][j] = __fmaf_rn(a[i], b[j], acc[i][j]);
        }
        if (kt + 1 < tiles) store(buf ^ 1);
        __syncthreads();
    }

    // ---- epilogue ----
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int m = m0 + (i < 4 ? ty * 4 + i : BM / 2 + ty * 4 + (i - 4));
        if (m >= p.t) continue;
        if constexpr (EP == SimtEpilogue::SwiGLU) {
            // packed columns n0 + tx*4 + j (gate) and n0 + 64 + tx*4 + j (up) -> feature n0/2 + tx*4 + j
            float* out = p.y + static_cast<std::size_t>(m) * p.ld_y + n0 / 2 + tx * 4;
            *reinterpret_cast<float4*>(out) = make_float4(swiglu_e(acc[i][0], acc[i][4]), swiglu_e(acc[i][1], acc[i][5]),
                                                          swiglu_e(acc[i][2], acc[i][6]), swiglu_e(acc[i][3], acc[i][7]));
        } else {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int nn = n0 + h * 64 + tx * 4;
                float* out = p.y + static_cast<std::size_t>(m) * p.ld_y + nn;
                if (nn + 3 < p.n) {
                    float4 v = make_float4(acc[i][h * 4 + 0], acc[i][h * 4 + 1], acc[i][h * 4 + 2], acc[i][h * 4 + 3]);
                    if constexpr (EP == SimtEpilogue::Add) {
                        const float4 r4 = *reinterpret_cast<const float4*>(out);
                        v = make_float4(__fadd_rn(r4.x, v.x), __fadd_rn(r4.y, v.y), __fadd_rn(r4.z, v.z), __fadd_rn(r4.w, v.w));
                    }
                    *reinterpret_cast<float4*>(out) = v;
                } else {
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        if (nn + j >= p.n) break;
                        float v = acc[i][h * 4 + j];
                        if constexpr (EP == SimtEpilogue::Add) v = __fadd_rn(out[j], v);
                        out[j] = v;
                    }
                }
            }
        }
    }
}

template <int BM, SimtWeight WT>
void launch_epilogue(const Params& p, SimtEpilogue ep, cudaStream_t stream) {
    const dim3 grid(static_cast<unsigned>((p.n + kBN - 1) / kBN), static_cast<unsigned>((p.t + BM - 1) / BM));
    const dim3 block(2 * BM);
    switch (ep) {
    case SimtEpilogue::Store: gemm_simt_kernel<BM, WT, SimtEpilogue::Store><<<grid, block, 0, stream>>>(p); break;
    case SimtEpilogue::Add: gemm_simt_kernel<BM, WT, SimtEpilogue::Add><<<grid, block, 0, stream>>>(p); break;
    case SimtEpilogue::SwiGLU: gemm_simt_kernel<BM, WT, SimtEpilogue::SwiGLU><<<grid, block, 0, stream>>>(p); break;
    }
}

template <int BM>
void launch_bm(const Params& p, SimtWeight wt, SimtEpilogue ep, cudaStream_t stream) {
    switch (wt) {
    case SimtWeight::F32: launch_epilogue<BM, SimtWeight::F32>(p, ep, stream); break;
    case SimtWeight::BF16: launch_epilogue<BM, SimtWeight::BF16>(p, ep, stream); break;
    case SimtWeight::Int8Group: launch_epilogue<BM, SimtWeight::Int8Group>(p, ep, stream); break;
    }
}

inline std::size_t elem_bytes(SimtWeight wt) { return wt == SimtWeight::F32 ? 4 : wt == SimtWeight::BF16 ? 2 : 1; }

// ---- weight preparation ----

// One warp per (row, group of 128): each lane holds 4 values; the maximum is exact whatever the order.
__global__ void quantize_group_s8_kernel(const float* src, std::size_t ld_src, std::int8_t* q, std::size_t ld_q, float* s,
                                         std::size_t ld_s, std::size_t rows, std::size_t groups) {
    const std::size_t warp = (static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    const unsigned lane = threadIdx.x % 32;
    if (warp >= rows * groups) return;
    const std::size_t row = warp / groups, g = warp % groups;
    const float4 v = *reinterpret_cast<const float4*>(src + row * ld_src + g * 128 + lane * 4);
    float m = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFu, m, o));
    const float scale = __fdiv_rn(m, 127.0f);
    const auto quant = [&](float w) -> int {
        if (!(scale > 0.0f)) return 0;
        const float r = rintf(__fdiv_rn(w, scale));
        return static_cast<int>(fminf(fmaxf(r, -127.0f), 127.0f));
    };
    const unsigned packed = (static_cast<unsigned>(quant(v.x)) & 0xFFu) | ((static_cast<unsigned>(quant(v.y)) & 0xFFu) << 8) |
                            ((static_cast<unsigned>(quant(v.z)) & 0xFFu) << 16) | ((static_cast<unsigned>(quant(v.w)) & 0xFFu) << 24);
    *reinterpret_cast<unsigned*>(q + row * ld_q + g * 128 + lane * 4) = packed;
    if (lane == 0) s[row * ld_s + g] = scale;
}

__global__ void float_to_bf16_kernel(const float* src, std::uint16_t* dst, std::size_t count) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const unsigned bits = __float_as_uint(src[i]);
    if ((bits & 0x7F800000u) == 0x7F800000u && (bits & 0x007FFFFFu) != 0) {
        dst[i] = static_cast<std::uint16_t>((bits >> 16) | 0x0040u); // quiet NaN
        return;
    }
    dst[i] = static_cast<std::uint16_t>((bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16);
}

} // namespace simt

using namespace simt;

bool gemm_simt_supports(SimtWeight type, std::size_t n, std::size_t k, std::size_t ld_x, std::size_t ld_w, std::size_t ld_y,
                        SimtEpilogue epilogue) {
    const std::size_t kq = type == SimtWeight::Int8Group ? kSimtGroup : std::size_t{16};
    if (k == 0 || n == 0 || k % kq != 0 || ld_x % 4 != 0 || ld_x < k || ld_w < k || ld_y % 4 != 0) return false;
    if ((ld_w * elem_bytes(type)) % 16 != 0) return false;
    if (epilogue == SimtEpilogue::SwiGLU) return n % 128 == 0 && ld_y >= n / 2;
    return ld_y >= n;
}

void launch_gemm_simt(const float* x, std::size_t ld_x, const SimtMatrix& w, float* y, std::size_t ld_y, std::size_t t,
                      std::size_t n, std::size_t k, SimtEpilogue epilogue, int config, cudaStream_t stream) {
    if (!gemm_simt_supports(w.type, n, k, ld_x, w.ld_w, ld_y, epilogue)) {
        throw std::invalid_argument("launch_gemm_simt: unsupported shape [" + std::to_string(t) + " x " + std::to_string(k) + "] by [" +
                                    std::to_string(n) + " x " + std::to_string(k) + "] (K a multiple of 16, 128 for int8 groups; strides "
                                    "multiples of 4 floats and 16 bytes; SwiGLU needs N a multiple of 128)");
    }
    if (w.type == SimtWeight::Int8Group && w.scales == nullptr) throw std::invalid_argument("launch_gemm_simt: int8 weights need scales");
    if (t == 0) return;
    if (t > 0x7FFFFFFFu || n > 0x7FFFFFFFu || k > 0x7FFFFFFFu) throw std::invalid_argument("launch_gemm_simt: dimension beyond 2^31");
    Params p{x, ld_x, static_cast<const unsigned char*>(w.w), w.ld_w * elem_bytes(w.type), w.scales, w.ld_s, y, ld_y,
             static_cast<int>(t), static_cast<int>(n), static_cast<int>(k)};
    if (config < 0) config = t >= 512 ? 0 : t >= 96 ? 1 : 2;
    switch (config) {
    case 0: launch_bm<128>(p, w.type, epilogue, stream); break;
    case 1: launch_bm<64>(p, w.type, epilogue, stream); break;
    case 2: launch_bm<32>(p, w.type, epilogue, stream); break;
    default: throw std::invalid_argument("launch_gemm_simt: config must be -1, 0, 1 or 2");
    }
    check(cudaGetLastError(), "gemm_simt_kernel launch");
}

void launch_quantize_group_s8(const float* src, std::size_t ld_src, std::int8_t* q, std::size_t ld_q, float* s,
                              std::size_t ld_s, std::size_t rows, std::size_t k, cudaStream_t stream) {
    if (k % kSimtGroup != 0 || ld_src % 4 != 0 || ld_q % 4 != 0) throw std::invalid_argument("launch_quantize_group_s8: K must be a multiple of 128");
    const std::size_t groups = k / kSimtGroup;
    const std::size_t warps = rows * groups;
    if (warps == 0) return;
    const std::size_t blocks = (warps * 32 + 255) / 256;
    quantize_group_s8_kernel<<<static_cast<unsigned>(blocks), 256, 0, stream>>>(src, ld_src, q, ld_q, s, ld_s, rows, groups);
    check(cudaGetLastError(), "quantize_group_s8_kernel launch");
}

void launch_float_to_bf16(const float* src, std::uint16_t* dst, std::size_t count, cudaStream_t stream) {
    if (count == 0) return;
    float_to_bf16_kernel<<<static_cast<unsigned>((count + 255) / 256), 256, 0, stream>>>(src, dst, count);
    check(cudaGetLastError(), "float_to_bf16_kernel launch");
}

} // namespace sepal::cuda
