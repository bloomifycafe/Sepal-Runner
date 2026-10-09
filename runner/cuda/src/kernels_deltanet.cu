// The Gated DeltaNet mixer's kernels (see kernels_deltanet.hpp for what they compute and the rules for waves).
//
// The scan kernel. One block runs one (row, value head). A thread owns COLUMN j of that head's state, the dk
// values s[i * dv + j], and keeps them in registers for the whole row: the recurrence for a column depends on
// nothing but k, q, v[j], decay and beta, so columns never talk to each other. Where `lanes_per_column` (SPLIT)
// is more than one, SPLIT adjacent lanes of a warp hold interleaved parts of the same column and meet only for
// the two dot products of a token, by warp shuffles.
//
// The block walks its row in chunks of kChunk tokens:
//   A. all threads convolve + SiLU the chunk's q, k, v channels into shared memory and compute decay and beta;
//   B. one warp per vector L2-normalises q and k;
//   C. the sequential part: token after token, every thread updates its column and reads it;
//   D. one warp per token applies the gated RMSNorm to the chunk's reads and writes the kernel's output.
//
// One token costs four operations per state element (decay multiply, k.S, update, q.S). Token n's update and
// read and token n+1's decay and k.S are one pass over the registers, which halves the chain of dependent
// reductions per token without changing a single operation.
//
// Summation order. A dot product over the key index i = 4 * (8 * mb + vl) + c (mb in 0..3, virtual lane vl in
// 0..7, c in 0..3) goes into accumulator u = 2 * vl + (c & 1), in increasing i; the 16 accumulators are then
// added by the tree ((u0 + u1) + (u2 + u3)) + ... The thread mapping decides only which thread holds which
// accumulators (a thread holds the virtual lanes vl = p * (8 / SPLIT) + v), so every mapping computes the same
// bits. Padding of the key dimension to 128 adds exact zeros.
#include "kernels_deltanet.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace sepal::cuda::detail {

// The kernels have external linkage on purpose (see convert_kernels.cu).

constexpr int kChunk = 16;       // tokens staged in shared memory at a time
constexpr int kConvGroup = 8;    // tokens convolved from one load of their window
constexpr int kMaxTaps = static_cast<int>(kDeltaNetMaxConvKernel);
constexpr int kKeyWidth = 128;   // key dimension padded to this
constexpr int kVirtualLanes = 8; // lanes of the canonical dot-product order
constexpr int kRowGroups = kKeyWidth / (4 * kVirtualLanes); // 4

// Everything a launch needs, by value.
struct WaveArgs {
    std::uint32_t key_heads, value_heads, dk, dv, taps, repeat;
    float l2_eps, norm_eps, read_scale;
    const float *conv_w, *a_log, *dt_bias, *norm_w;
    const float *mixed, *z, *b, *a;
    std::size_t mixed_stride, z_stride, b_stride, a_stride;
    const RowSpan* rows;
    const std::uint32_t *in_slots, *out_slots, *wave;
    float* conv_tails;
    float* recurrent;
    void* out;
    std::size_t out_stride;
    float* tap_conv;
    std::size_t tap_conv_stride;
    float* tap_reads;
    std::size_t tap_reads_stride;
};

// ---- scalar functions, each the reference's formula with the rounding of every step spelled out ----

__device__ inline float softplus_f(float x) {
    return __fadd_rn(fmaxf(x, 0.0f), log1pf(expf(-fabsf(x))));
}

// x * sigmoid(x) written as x / (1 + exp(-x)): the same function with one transcendental instead of three. For
// a large negative x the quotient underflows to zero a little before the reference's product does.
__device__ inline float silu_f(float x) { return __fdiv_rn(x, __fadd_rn(1.0f, expf(-x))); }

__device__ inline float sigmoid_f(float x) { return expf(-softplus_f(-x)); }

// Sum over the 32 lanes of a warp in a fixed butterfly order; every lane returns the same bits.
__device__ inline float warp_sum(float x) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) x = __fadd_rn(x, __shfl_xor_sync(0xFFFFFFFFu, x, offset));
    return x;
}

// Adds the NA accumulators of this thread by a tree, then the partial sums of the SPLIT lanes of the column by a
// butterfly. See the note on summation order at the top of the file.
template <int SPLIT, int NA>
__device__ inline float combine(float (&acc)[NA]) {
    #pragma unroll
    for (int stride = 1; stride < NA; stride *= 2) {
        #pragma unroll
        for (int u = 0; u + stride < NA; u += 2 * stride) acc[u] = __fadd_rn(acc[u], acc[u + stride]);
    }
    float x = acc[0];
    #pragma unroll
    for (int offset = 1; offset < SPLIT; offset *= 2) x = __fadd_rn(x, __shfl_xor_sync(0xFFFFFFFFu, x, offset));
    return x;
}

__device__ inline void store_out(float* p, float v) { *p = v; }
__device__ inline void store_out(__half* p, float v) { *p = __float2half_rn(v); }

// ---- the scan kernel ----

template <int SPLIT, class OutT>
__global__ void __launch_bounds__(128 * SPLIT) deltanet_scan_kernel(const WaveArgs args) {
    constexpr int LOC = kVirtualLanes / SPLIT; // virtual lanes held by one thread
    constexpr int PER = kRowGroups * LOC * 4;  // state values per thread
    constexpr int NA = LOC * 2;                // accumulators per dot product per thread
    constexpr int NT = 128 * SPLIT;            // largest block

    extern __shared__ float smem[];

    const std::uint32_t head = blockIdx.x;
    const std::uint32_t row = args.wave[blockIdx.y];
    const RowSpan span = args.rows[row];
    const std::uint32_t in_slot = args.in_slots[row];
    const std::uint32_t out_slot = args.out_slots[row];
    const std::uint32_t dk = args.dk, dv = args.dv, taps = args.taps, key_heads = args.key_heads;
    const std::uint32_t key_head = head / args.repeat;
    const std::uint32_t conv_dim = 2 * key_heads * dk + args.value_heads * dv;

    const std::uint32_t threads = blockDim.x;
    const std::uint32_t tid = threadIdx.x;
    const std::uint32_t warps = threads / 32;
    const std::uint32_t warp = tid / 32;
    const std::uint32_t lane = tid % 32;
    const std::uint32_t p = tid % SPLIT; // this thread's share of its column
    const std::uint32_t col = tid / SPLIT;
    const bool valid = col < dv;
    const bool tap_owner = head % args.repeat == 0; // writes the shared key head's tap values

    // Shared memory: q and k of the chunk [kChunk][128], v and the reads [kChunk][dv], decay and beta [kChunk].
    float* qs = smem;
    float* ks = qs + kChunk * kKeyWidth;
    float* vs = ks + kChunk * kKeyWidth;
    float* rd = vs + kChunk * dv;
    float* dec = rd + kChunk * dv;
    float* bet = dec + kChunk;
    const float4* qs4 = reinterpret_cast<const float4*>(qs);
    const float4* ks4 = reinterpret_cast<const float4*>(ks);

    // The key dimension beyond dk stays zero for the whole row.
    for (std::uint32_t e = tid; e < kChunk * kKeyWidth; e += threads) {
        if (e % kKeyWidth >= dk) {
            qs[e] = 0.0f;
            ks[e] = 0.0f;
        }
    }

    // The column's state values: float s[PER], element (mb, v, c) is key index 4 * (8 * mb + p * LOC + v) + c.
    float s[PER];
    {
        const float* source = in_slot == kNoSlot ? nullptr
                                                 : args.recurrent + (static_cast<std::size_t>(in_slot) * args.value_heads + head) * dk * dv;
        #pragma unroll
        for (int mb = 0; mb < kRowGroups; ++mb) {
            #pragma unroll
            for (int v = 0; v < LOC; ++v) {
                #pragma unroll
                for (int c = 0; c < 4; ++c) {
                    const std::uint32_t i = 4 * (8 * mb + p * LOC + v) + c;
                    s[(mb * LOC + v) * 4 + c] = (source != nullptr && valid && i < dk) ? source[i * dv + col] : 0.0f;
                }
            }
        }
    }

    for (std::uint32_t c0 = 0; c0 < span.length; c0 += kChunk) {
        const std::uint32_t n = min(static_cast<std::uint32_t>(kChunk), span.length - c0);

        // ---- A: convolution + SiLU of q, k, v for the chunk, decay and beta ----
        // Thread = channel. It loads the window of inputs for kConvGroup tokens at once (independent loads) and
        // convolves them from registers, so the memory latency is paid once per group, not once per tap.
        const std::uint32_t per_token = 2 * dk + dv;
        for (std::uint32_t c = tid; c < per_token; c += threads) {
            const std::uint32_t channel = c < dk       ? key_head * dk + c
                                          : c < 2 * dk ? key_heads * dk + key_head * dk + (c - dk)
                                                       : 2 * key_heads * dk + head * dv + (c - 2 * dk);
            float weight[kMaxTaps];
            #pragma unroll
            for (int j = 0; j < kMaxTaps; ++j) {
                weight[j] = static_cast<std::uint32_t>(j) < taps ? args.conv_w[static_cast<std::size_t>(channel) * taps + j] : 0.0f;
            }
            for (std::uint32_t g0 = 0; g0 < n; g0 += kConvGroup) {
                const std::uint32_t m = min(static_cast<std::uint32_t>(kConvGroup), n - g0);
                float x[kConvGroup + kMaxTaps - 1];
                #pragma unroll
                for (int i = 0; i < kConvGroup + kMaxTaps - 1; ++i) {
                    x[i] = 0.0f;
                    if (static_cast<std::uint32_t>(i) < m + taps - 1) {
                        const std::uint32_t w = c0 + g0 + static_cast<std::uint32_t>(i); // position in (tail, inputs)
                        if (w < taps - 1) {
                            if (in_slot != kNoSlot) {
                                x[i] = args.conv_tails[(static_cast<std::size_t>(in_slot) * (taps - 1) + w) * conv_dim + channel];
                            }
                        } else {
                            x[i] = args.mixed[static_cast<std::size_t>(span.offset + w - (taps - 1)) * args.mixed_stride + channel];
                        }
                    }
                }
                #pragma unroll
                for (int k = 0; k < kConvGroup; ++k) {
                    if (static_cast<std::uint32_t>(k) < m) {
                        float acc = 0.0f;
                        #pragma unroll
                        for (int j = 0; j < kMaxTaps; ++j) {
                            if (static_cast<std::uint32_t>(j) < taps) acc = __fmaf_rn(weight[j], x[k + j], acc);
                        }
                        const float value = silu_f(acc);
                        const std::uint32_t tt = g0 + static_cast<std::uint32_t>(k);
                        if (c < dk) {
                            qs[tt * kKeyWidth + c] = value;
                        } else if (c < 2 * dk) {
                            ks[tt * kKeyWidth + (c - dk)] = value;
                        } else {
                            vs[tt * dv + (c - 2 * dk)] = value;
                            if (args.tap_conv != nullptr) {
                                args.tap_conv[static_cast<std::size_t>(span.offset + c0 + tt) * args.tap_conv_stride + channel] = value;
                            }
                        }
                    }
                }
            }
        }
        if (tid < n) {
            const std::size_t token = span.offset + c0 + tid;
            const float a_in = args.a[token * args.a_stride + head];
            const float b_in = args.b[token * args.b_stride + head];
            const float g = __fmul_rn(-expf(args.a_log[head]), softplus_f(__fadd_rn(a_in, args.dt_bias[head])));
            dec[tid] = expf(g);
            bet[tid] = sigmoid_f(b_in);
        }
        __syncthreads();

        // ---- B: L2 normalisation of q and k, one warp per vector ----
        for (std::uint32_t vec = warp; vec < 2 * n; vec += warps) {
            const std::uint32_t tt = vec >> 1;
            const bool is_k = (vec & 1u) != 0;
            float* x = (is_k ? ks : qs) + tt * kKeyWidth;
            float sum = 0.0f;
            for (std::uint32_t i = lane; i < dk; i += 32) sum = __fmaf_rn(x[i], x[i], sum);
            sum = warp_sum(sum);
            const float norm = sqrtf(__fadd_rn(sum, args.l2_eps));
            for (std::uint32_t i = lane; i < dk; i += 32) {
                const float value = __fdiv_rn(x[i], norm);
                x[i] = value;
                if (args.tap_conv != nullptr && tap_owner) {
                    const std::size_t channel = (is_k ? key_heads * dk : 0) + key_head * dk + i;
                    args.tap_conv[static_cast<std::size_t>(span.offset + c0 + tt) * args.tap_conv_stride + channel] = value;
                }
            }
        }
        __syncthreads();

        // ---- C: the recurrence, token after token ----
        // Token 0 of the chunk: decay the state and take k.S.
        float corr;
        {
            float acc_k[NA];
            #pragma unroll
            for (int u = 0; u < NA; ++u) acc_k[u] = 0.0f;
            const float decay0 = dec[0];
            #pragma unroll
            for (int mb = 0; mb < kRowGroups; ++mb) {
                #pragma unroll
                for (int v = 0; v < LOC; ++v) {
                    const float4 kq = ks4[mb * 8 + p * LOC + v];
                    const float kk[4] = {kq.x, kq.y, kq.z, kq.w};
                    #pragma unroll
                    for (int c = 0; c < 4; ++c) {
                        const int idx = (mb * LOC + v) * 4 + c;
                        s[idx] = __fmul_rn(s[idx], decay0);
                        acc_k[v * 2 + (c & 1)] = __fmaf_rn(kk[c], s[idx], acc_k[v * 2 + (c & 1)]);
                    }
                }
            }
            const float recalled = combine<SPLIT, NA>(acc_k);
            corr = __fmul_rn(bet[0], __fsub_rn(valid ? vs[col] : 0.0f, recalled));
        }

        for (std::uint32_t tt = 0; tt < n; ++tt) {
            float acc_q[NA];
            float acc_k[NA];
            #pragma unroll
            for (int u = 0; u < NA; ++u) {
                acc_q[u] = 0.0f;
                acc_k[u] = 0.0f;
            }
            const float4* kn = ks4 + tt * (kKeyWidth / 4);
            const float4* qn = qs4 + tt * (kKeyWidth / 4);
            if (tt + 1 < n) {
                // Update with token tt, read it, then decay for token tt + 1 and take its k.S.
                const float4* kx = ks4 + (tt + 1) * (kKeyWidth / 4);
                const float decay_next = dec[tt + 1];
                #pragma unroll
                for (int mb = 0; mb < kRowGroups; ++mb) {
                    #pragma unroll
                    for (int v = 0; v < LOC; ++v) {
                        const int at = mb * 8 + static_cast<int>(p) * LOC + v;
                        const float4 k4 = kn[at];
                        const float4 q4 = qn[at];
                        const float4 x4 = kx[at];
                        const float kk[4] = {k4.x, k4.y, k4.z, k4.w};
                        const float qq[4] = {q4.x, q4.y, q4.z, q4.w};
                        const float xx[4] = {x4.x, x4.y, x4.z, x4.w};
                        #pragma unroll
                        for (int c = 0; c < 4; ++c) {
                            const int idx = (mb * LOC + v) * 4 + c;
                            const int u = v * 2 + (c & 1);
                            float sv = __fmaf_rn(kk[c], corr, s[idx]);
                            acc_q[u] = __fmaf_rn(qq[c], sv, acc_q[u]);
                            sv = __fmul_rn(sv, decay_next);
                            acc_k[u] = __fmaf_rn(xx[c], sv, acc_k[u]);
                            s[idx] = sv;
                        }
                    }
                }
            } else {
                // Last token of the chunk: update and read only.
                #pragma unroll
                for (int mb = 0; mb < kRowGroups; ++mb) {
                    #pragma unroll
                    for (int v = 0; v < LOC; ++v) {
                        const int at = mb * 8 + static_cast<int>(p) * LOC + v;
                        const float4 k4 = kn[at];
                        const float4 q4 = qn[at];
                        const float kk[4] = {k4.x, k4.y, k4.z, k4.w};
                        const float qq[4] = {q4.x, q4.y, q4.z, q4.w};
                        #pragma unroll
                        for (int c = 0; c < 4; ++c) {
                            const int idx = (mb * LOC + v) * 4 + c;
                            const int u = v * 2 + (c & 1);
                            const float sv = __fmaf_rn(kk[c], corr, s[idx]);
                            acc_q[u] = __fmaf_rn(qq[c], sv, acc_q[u]);
                            s[idx] = sv;
                        }
                    }
                }
            }
            const float read = __fmul_rn(args.read_scale, combine<SPLIT, NA>(acc_q));
            if (valid && p == 0) rd[tt * dv + col] = read;
            if (tt + 1 < n) {
                const float recalled = combine<SPLIT, NA>(acc_k);
                corr = __fmul_rn(bet[tt + 1], __fsub_rn(valid ? vs[(tt + 1) * dv + col] : 0.0f, recalled));
            }
        }
        __syncthreads();

        // ---- D: gated RMSNorm of the chunk's reads, one warp per token ----
        for (std::uint32_t tt = warp; tt < n; tt += warps) {
            const std::size_t token = span.offset + c0 + tt;
            const float* x = rd + tt * dv;
            // z first, all loads together.
            float zv[kKeyWidth / 32];
            #pragma unroll
            for (int m = 0; m < kKeyWidth / 32; ++m) {
                const std::uint32_t j = lane + 32u * static_cast<std::uint32_t>(m);
                zv[m] = j < dv ? args.z[token * args.z_stride + head * dv + j] : 0.0f;
            }
            float sum = 0.0f;
            for (std::uint32_t j = lane; j < dv; j += 32) sum = __fmaf_rn(x[j], x[j], sum);
            sum = warp_sum(sum);
            const float rms = sqrtf(__fadd_rn(__fdiv_rn(sum, static_cast<float>(dv)), args.norm_eps));
            #pragma unroll
            for (int m = 0; m < kKeyWidth / 32; ++m) {
                const std::uint32_t j = lane + 32u * static_cast<std::uint32_t>(m);
                if (j < dv) {
                    const float y = __fmul_rn(__fmul_rn(__fdiv_rn(x[j], rms), args.norm_w[j]), silu_f(zv[m]));
                    store_out(static_cast<OutT*>(args.out) + token * args.out_stride + head * dv + j, y);
                    if (args.tap_reads != nullptr) args.tap_reads[token * args.tap_reads_stride + head * dv + j] = x[j];
                }
            }
        }
        // No barrier needed before the next chunk's step A: A writes q, k, v, decay and beta, which C and D no
        // longer read after the barrier above, and the barrier after A orders it before the next C writes `rd`.
    }

    if (out_slot != kNoSlot && valid) {
        float* target = args.recurrent + (static_cast<std::size_t>(out_slot) * args.value_heads + head) * dk * dv;
        #pragma unroll
        for (int mb = 0; mb < kRowGroups; ++mb) {
            #pragma unroll
            for (int v = 0; v < LOC; ++v) {
                #pragma unroll
                for (int c = 0; c < 4; ++c) {
                    const std::uint32_t i = 4 * (8 * mb + p * LOC + v) + c;
                    if (i < dk) target[i * dv + col] = s[(mb * LOC + v) * 4 + c];
                }
            }
        }
    }
    (void)NT;
}

// ---- the conv tail kernel ----

struct TailArgs {
    std::uint32_t taps, conv_dim;
    const float* mixed;
    std::size_t mixed_stride;
    const RowSpan* rows;
    const std::uint32_t *in_slots, *out_slots, *wave;
    float* conv_tails;
};

// Thread (row, channel) writes the channel's K-1 new tail values in increasing order. When a row is continued in
// place that order reads every old value before it is overwritten.
__global__ void deltanet_conv_tail_kernel(const TailArgs args) {
    const std::uint32_t channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= args.conv_dim) return;
    const std::uint32_t row = args.wave[blockIdx.y];
    const std::uint32_t out_slot = args.out_slots[row];
    if (out_slot == kNoSlot) return;
    const std::uint32_t in_slot = args.in_slots[row];
    const RowSpan span = args.rows[row];
    const std::uint32_t history = args.taps - 1;
    for (std::uint32_t i = 0; i < history; ++i) {
        const std::uint32_t w = span.length + i; // position in (tail, then inputs)
        float value;
        if (w < history) {
            value = in_slot == kNoSlot ? 0.0f
                                       : args.conv_tails[(static_cast<std::size_t>(in_slot) * history + w) * args.conv_dim + channel];
        } else {
            value = args.mixed[static_cast<std::size_t>(span.offset + w - history) * args.mixed_stride + channel];
        }
        args.conv_tails[(static_cast<std::size_t>(out_slot) * history + i) * args.conv_dim + channel] = value;
    }
}

} // namespace sepal::cuda::detail

namespace sepal::cuda {

namespace {

void validate(const DeltaNetGeometry& g, const DeltaNetRows& rows) {
    if (g.key_heads == 0 || g.value_heads == 0 || g.key_head_dim == 0 || g.value_head_dim == 0 || g.conv_kernel == 0) {
        throw std::invalid_argument("deltanet: every size of the geometry must be positive");
    }
    if (g.value_heads % g.key_heads != 0) {
        throw std::invalid_argument("deltanet: value heads must be a multiple of key heads");
    }
    if (g.key_head_dim > kDeltaNetMaxHeadDim || g.value_head_dim > kDeltaNetMaxHeadDim) {
        throw std::invalid_argument("deltanet: head dimensions above " + std::to_string(kDeltaNetMaxHeadDim) +
                                    " are not supported");
    }
    if (g.conv_kernel > kDeltaNetMaxConvKernel) {
        throw std::invalid_argument("deltanet: more than " + std::to_string(kDeltaNetMaxConvKernel) +
                                    " conv taps are not supported");
    }
    if (rows.wave_size != 0 && (rows.rows == nullptr || rows.in_slots == nullptr || rows.out_slots == nullptr ||
                                rows.wave == nullptr)) {
        throw std::invalid_argument("deltanet: a wave needs the row, slot and wave arrays");
    }
    if (rows.wave_size > 65535u) throw std::invalid_argument("deltanet: a wave of more than 65535 rows");
}

// Lanes per column when the caller does not say. Measured on an RTX 3060 (28 SMs, see the benchmark): one lane per
// column is fastest whenever there are enough blocks to fill the GPU, two lanes when a wave has few blocks (one
// long row). More lanes cost more in shuffles than they gain in parallelism. The result bits do not depend on it.
int choose_lanes(const DeltaNetGeometry& g, std::uint32_t wave_size) {
    const std::uint64_t blocks = static_cast<std::uint64_t>(wave_size) * g.value_heads;
    return blocks >= 48 ? 1 : 2;
}

template <int SPLIT, class OutT>
void run_scan(const detail::WaveArgs& args, std::uint32_t wave_size, std::uint32_t value_heads, std::size_t smem_bytes,
              std::uint32_t dv, cudaStream_t stream) {
    const std::uint32_t threads = std::max(32u, (dv * SPLIT + 31) / 32 * 32);
    const dim3 grid(value_heads, wave_size);
    detail::deltanet_scan_kernel<SPLIT, OutT><<<grid, dim3(threads), smem_bytes, stream>>>(args);
}

} // namespace

template <class OutT>
void launch_deltanet_scan(const DeltaNetGeometry& g, const DeltaNetWeights& w, const DeltaNetInputs& in,
                          const DeltaNetRows& rows, const DeltaNetStates& states, OutT* out, std::size_t out_stride,
                          cudaStream_t stream, const DeltaNetOptions& options) {
    validate(g, rows);
    if (rows.wave_size == 0) return;
    if (w.conv == nullptr || w.a_log == nullptr || w.dt_bias == nullptr || w.norm == nullptr ||
        in.mixed.data == nullptr || in.z.data == nullptr || in.b.data == nullptr || in.a.data == nullptr ||
        states.recurrent == nullptr || out == nullptr || (g.conv_kernel > 1 && states.conv_tails == nullptr)) {
        throw std::invalid_argument("deltanet: a null weight, input, state or output pointer");
    }
    const int lanes = options.lanes_per_column != 0 ? options.lanes_per_column : choose_lanes(g, rows.wave_size);
    if (lanes != 1 && lanes != 2 && lanes != 4 && lanes != 8) {
        throw std::invalid_argument("deltanet: lanes_per_column must be 0, 1, 2, 4 or 8");
    }

    detail::WaveArgs args{};
    args.key_heads = g.key_heads;
    args.value_heads = g.value_heads;
    args.dk = g.key_head_dim;
    args.dv = g.value_head_dim;
    args.taps = g.conv_kernel;
    args.repeat = g.value_heads / g.key_heads;
    args.l2_eps = static_cast<float>(1e-6);
    args.norm_eps = static_cast<float>(g.rms_norm_eps);
    args.read_scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(g.key_head_dim)));
    args.conv_w = w.conv;
    args.a_log = w.a_log;
    args.dt_bias = w.dt_bias;
    args.norm_w = w.norm;
    args.mixed = in.mixed.data;
    args.mixed_stride = in.mixed.stride;
    args.z = in.z.data;
    args.z_stride = in.z.stride;
    args.b = in.b.data;
    args.b_stride = in.b.stride;
    args.a = in.a.data;
    args.a_stride = in.a.stride;
    args.rows = rows.rows;
    args.in_slots = rows.in_slots;
    args.out_slots = rows.out_slots;
    args.wave = rows.wave;
    args.conv_tails = states.conv_tails;
    args.recurrent = states.recurrent;
    args.out = out;
    args.out_stride = out_stride;
    args.tap_conv = options.taps.conv_act;
    args.tap_conv_stride = options.taps.conv_act_stride;
    args.tap_reads = options.taps.reads;
    args.tap_reads_stride = options.taps.reads_stride;

    const std::size_t smem_floats =
        2u * detail::kChunk * detail::kKeyWidth + 2u * detail::kChunk * g.value_head_dim + 2u * detail::kChunk;
    const std::size_t smem = smem_floats * sizeof(float);
    switch (lanes) {
        case 1: run_scan<1, OutT>(args, rows.wave_size, g.value_heads, smem, g.value_head_dim, stream); break;
        case 2: run_scan<2, OutT>(args, rows.wave_size, g.value_heads, smem, g.value_head_dim, stream); break;
        case 4: run_scan<4, OutT>(args, rows.wave_size, g.value_heads, smem, g.value_head_dim, stream); break;
        default: run_scan<8, OutT>(args, rows.wave_size, g.value_heads, smem, g.value_head_dim, stream); break;
    }
    check(cudaGetLastError(), "deltanet scan kernel launch");
}

template void launch_deltanet_scan<float>(const DeltaNetGeometry&, const DeltaNetWeights&, const DeltaNetInputs&,
                                          const DeltaNetRows&, const DeltaNetStates&, float*, std::size_t,
                                          cudaStream_t, const DeltaNetOptions&);
template void launch_deltanet_scan<__half>(const DeltaNetGeometry&, const DeltaNetWeights&, const DeltaNetInputs&,
                                           const DeltaNetRows&, const DeltaNetStates&, __half*, std::size_t,
                                           cudaStream_t, const DeltaNetOptions&);

void launch_deltanet_conv_tail(const DeltaNetGeometry& g, const DeltaNetInputs& in, const DeltaNetRows& rows,
                               const DeltaNetStates& states, cudaStream_t stream) {
    validate(g, rows);
    if (rows.wave_size == 0 || g.conv_kernel == 1) return;
    if (in.mixed.data == nullptr || states.conv_tails == nullptr) {
        throw std::invalid_argument("deltanet: a null input or conv tail pointer");
    }
    detail::TailArgs args{};
    args.taps = g.conv_kernel;
    args.conv_dim = g.conv_dim();
    args.mixed = in.mixed.data;
    args.mixed_stride = in.mixed.stride;
    args.rows = rows.rows;
    args.in_slots = rows.in_slots;
    args.out_slots = rows.out_slots;
    args.wave = rows.wave;
    args.conv_tails = states.conv_tails;
    constexpr std::uint32_t kThreads = 128;
    const dim3 grid((g.conv_dim() + kThreads - 1) / kThreads, rows.wave_size);
    detail::deltanet_conv_tail_kernel<<<grid, dim3(kThreads), 0, stream>>>(args);
    check(cudaGetLastError(), "deltanet conv tail kernel launch");
}

} // namespace sepal::cuda
