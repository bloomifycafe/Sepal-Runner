// The Gated DeltaNet mixer's kernels (see kernels_deltanet.hpp for what they compute and the rules for waves).
//
// The scan kernel. One block runs one (row, value head) and holds two kinds of warps that work at the same time:
//
//   SCAN warps (the first 4 * ceil(dv / 4) threads, 128 for dv = 128). A thread owns a TILE of the head's state: 4
//   adjacent columns j and a quarter of the keys i (see Tile). Columns never talk to each other (the recurrence for a
//   column depends on nothing but k, q, v[j], decay and beta); the 4 lanes that share a column quad meet only when a
//   dot product is completed, by two shuffles. Every k or q vector read from shared memory serves 4 columns: reading
//   those operands, not the arithmetic, limited the one-column-per-thread form this replaced. The scan warps do nothing
//   but the recurrence.
//
//   AUX warps (the last kAuxThreads threads). They prepare the scan's input and finish its output, in chunks of
//   kChunk tokens, one chunk ahead of the scan:
//     A. convolve + SiLU the chunk's q, k, v channels; the scalars of each token (decay bookkeeping, beta);
//     B. L2-normalise q and k (one warp per four vectors);
//     D. the gated RMSNorm of the chunk's reads, written as the kernel's output (one warp per four tokens).
//   The scan reads a chunk's q, k, v and scalars from one of two stage buffers in shared memory and writes its reads to
//   one of two read buffers; named barriers pass the buffers back and forth (full/empty pairs), so the global-memory
//   latency and the arithmetic of A, B and D hide behind the recurrence instead of stopping it. Everything the aux warps
//   load from global memory is requested one step ahead, and their code has no data-dependent branch, because they are
//   few and get their speed from overlapping independent instructions.
//
// The recurrence. The state is kept as tile = S / g, where g is the "outstanding decay scale": the decay of a token
// multiplies the scalar g instead of the 128 values of a column, which removes one of the four operations per state
// element (the others: k.S, the update, q.S). With true recalled = g (k.tile) the correction is beta (v - g (k.tile)),
// the update tile += k (correction / g), and the read scale * g (q.tile). When g would fall below 2^-40 the aux warps
// hand the scan a factor to multiply the tile by (the whole decay since the last such point) and g restarts at 1; at the
// end of a row the tile times g is stored as the state. Token n's update and read and token n+1's k.S are one pass over
// the registers; the last token of a chunk takes the next chunk's first token, so the scan never stops between chunks.
//
// Determinism. Every value depends on its own row and nothing else: the aux scalars are a sequential function of the
// row's own tokens, the reductions have one fixed order (each lane's two accumulators in increasing key order, added;
// then the lanes of the quad by xor 1 and xor 2), and no atomic touches a float. A row's bits do not depend on the
// other rows of the wave, on its place in the pass, or on how many blocks run. They do depend on where the row STARTS:
// a row cut in two and continued from the first part's state differs from the uncut row in the last bits (the decay
// scale restarts at the cut), like every row continued from a stored state; the engine always cuts at the same place.
#include "kernels_deltanet.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace sepal::cuda::detail {

// The kernels have external linkage on purpose (see convert_kernels.cu).

constexpr int kChunk = 16;       // tokens staged in shared memory at a time
constexpr int kAuxThreads = 128; // threads of the aux role
constexpr int kAuxChannels = (3 * 128 + kAuxThreads - 1) / kAuxThreads; // channels per aux thread (q, k, v: 3 * 128)
constexpr int kMaxTaps = static_cast<int>(kDeltaNetMaxConvKernel);
constexpr float kRescale = 9.094947017729282e-13f; // 2^-40: below this the decay scale is folded into the state
constexpr int kKeyWidth = 128;   // key dimension padded to this
constexpr int kAcc = 2;         // accumulators per dot product per thread (keys alternate between them)
constexpr int kCols = 4;         // columns of the state one scan thread holds
constexpr int kKeyLanes = 4;     // lanes of a warp that share a column quad, each with a quarter of the keys
constexpr int kGroups = kKeyWidth / 4 / kKeyLanes; // key groups (of 4 keys) per lane: 8

// Named barriers (0 is __syncthreads). Pairs of stage buffers: full = filled by aux, empty = consumed by scan;
// read buffers: rdfull = filled by scan, rdempty = consumed by aux.
constexpr int kBarFull = 1;    // + buffer
constexpr int kBarEmpty = 3;   // + buffer
constexpr int kBarRdFull = 5;  // + buffer
constexpr int kBarRdEmpty = 7; // + buffer
constexpr int kBarAux = 9;     // the aux warps among themselves

__device__ __forceinline__ void bar_sync(int id, int count) {
    asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(count) : "memory");
}
__device__ __forceinline__ void bar_arrive(int id, int count) {
    asm volatile("bar.arrive %0, %1;" ::"r"(id), "r"(count) : "memory");
}

// Everything a launch needs, by value.
struct WaveArgs {
    std::uint32_t key_heads, value_heads, dk, dv, taps, repeat;
    float l2_eps, norm_eps, read_scale;
    const float *conv_w, *a_log, *dt_bias, *norm_w;
    const void *mixed, *z, *b, *a; // float32 or binary16 matrices (the kernel's InT)
    std::size_t mixed_stride, z_stride, b_stride, a_stride;
    const RowSpan* rows;
    const std::uint32_t *in_slots, *out_slots, *wave;
    void* conv_tails;  // float, or __half when the kernel is built for binary16 state storage
    void* recurrent;
    unsigned* saturations; // counts values that did not fit binary16 on a store (null: not counted)
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

// x / d correctly rounded, without the branch to a slow path that `__fdiv_rn` carries: the reciprocal estimate refined
// by two fused steps, the same sequence the fast path of `__fdiv_rn` uses (so the same bits for finite, normal d). The
// aux warps need straight-line code. `recip` is the refined reciprocal of d (see reciprocal()).
__device__ __forceinline__ float reciprocal(float d) {
    float r;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(d));
    return __fmaf_rn(__fmaf_rn(-d, r, 1.0f), r, r);
}
__device__ __forceinline__ float divide_by(float x, float d, float recip) {
    const float q = __fmul_rn(x, recip);
    return __fmaf_rn(__fmaf_rn(-d, q, x), recip, q);
}

// x * sigmoid(x) written as x / (1 + exp(-x)): the same function with one transcendental instead of three. For
// a large negative x the quotient underflows to zero a little before the reference's product does (the denominator is
// capped so that its reciprocal stays finite).
__device__ __forceinline__ float silu_f(float x) {
    const float d = fminf(__fadd_rn(1.0f, expf(-x)), 1e38f);
    return divide_by(x, d, reciprocal(d));
}

__device__ inline float sigmoid_f(float x) { return expf(-softplus_f(-x)); }

// Sum over the 32 lanes of a warp in a fixed butterfly order; every lane returns the same bits.
__device__ inline float warp_sum(float x) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) x = __fadd_rn(x, __shfl_xor_sync(0xFFFFFFFFu, x, offset));
    return x;
}

__device__ inline void store_out(float* p, float v) { *p = v; }
__device__ inline void store_out(__half* p, float v) { *p = __float2half_rn(v); }

// ---- the scan role ----

__host__ __device__ constexpr int pad4(int n) { return (n + 3) & ~3; }

// Floats of one stage buffer: q and k [kChunk][128], v [kChunk][dv], then five arrays of [kChunk] scalars (pre, inv, beta, read scale,
// cumulative decay: see the scan role); padded so that the
// next buffer starts on a 16-byte boundary.
__host__ __device__ constexpr int stage_floats(int dv) { return pad4(2 * kChunk * kKeyWidth + kChunk * dv + 5 * kChunk + 1); }

// The state tile of one scan thread: kCols columns of the head's state, and in each the keys of kGroups groups of
// four, group g = p + kKeyLanes * m (m = 0 .. kGroups - 1) for the thread's key lane p. s[cc][4 m + c] is the state
// element of key 4 * (p + kKeyLanes * m) + c, column 4 * quad + cc. The kKeyLanes lanes of a column quad are adjacent
// lanes of a warp; they meet only when a dot product is completed, by two shuffles. Reading the operand vectors
// (k, q) from shared memory is the bottleneck of the recurrence, so every operand loaded is used for kCols columns.
struct Tile {
    float s[kCols][kGroups * 4];
};
struct Acc {
    float a[kCols][kAcc];
};

// Completes one dot product: the thread's two accumulators, then the partial sums of the kKeyLanes lanes by a
// butterfly. Every one of the lanes ends with the same bits (addition is commutative).
__device__ __forceinline__ float finish(const float (&acc)[kAcc]) {
    float x = __fadd_rn(acc[0], acc[1]);
    x = __fadd_rn(x, __shfl_xor_sync(0xFFFFFFFFu, x, 1));
    x = __fadd_rn(x, __shfl_xor_sync(0xFFFFFFFFu, x, 2));
    return x;
}

// One token's pass over this thread's state tile: update with the token's correction, take q.S, and (NEXT) take the next
// token's k.S of the updated state.
template <bool NEXT>
__device__ __forceinline__ void token_pass(Tile& t, Acc& acc_q, Acc& acc_k, const float4* kn, const float4* qn,
                                           const float4* kx, const float (&corr)[kCols], int p) {
    #pragma unroll
    for (int m = 0; m < kGroups; ++m) {
        const int at = p + kKeyLanes * m;
        const float4 k4 = kn[at];
        const float4 q4 = qn[at];
        float4 x4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (NEXT) x4 = kx[at];
        const float kk[4] = {k4.x, k4.y, k4.z, k4.w};
        const float qq[4] = {q4.x, q4.y, q4.z, q4.w};
        const float xx[4] = {x4.x, x4.y, x4.z, x4.w};
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            // The four columns are independent: each step is issued for all four before the next one.
            float sv[kCols];
            #pragma unroll
            for (int cc = 0; cc < kCols; ++cc) sv[cc] = __fmaf_rn(kk[c], corr[cc], t.s[cc][m * 4 + c]);
            #pragma unroll
            for (int cc = 0; cc < kCols; ++cc) acc_q.a[cc][c % kAcc] = __fmaf_rn(qq[c], sv[cc], acc_q.a[cc][c % kAcc]);
            if (NEXT) {
                #pragma unroll
                for (int cc = 0; cc < kCols; ++cc) acc_k.a[cc][c % kAcc] = __fmaf_rn(xx[c], sv[cc], acc_k.a[cc][c % kAcc]);
            }
            #pragma unroll
            for (int cc = 0; cc < kCols; ++cc) t.s[cc][m * 4 + c] = sv[cc];
        }
    }
}

// ---- state storage: float32, or binary16 (computation stays float32: widen on load, round on store) ----

__device__ __forceinline__ float load_state(const void* base, std::size_t index, std::true_type /*half*/) {
    return __half2float(static_cast<const __half*>(base)[index]);
}
__device__ __forceinline__ float load_state(const void* base, std::size_t index, std::false_type) {
    return static_cast<const float*>(base)[index];
}
// Round to nearest even; a value beyond the binary16 range is clamped to +-65504 (never infinity) and counted.
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

// ---- the scan kernel ----

// TAPS: the number of conv taps the kernel is built for; kMaxTaps means any number up to that (a runtime count).
// An element of an input matrix, widened (exactly, for binary16) to float32.
__device__ __forceinline__ float widen_in(float v) { return v; }
__device__ __forceinline__ float widen_in(__half v) { return __half2float(v); }

template <class OutT, class InT, int TAPS, bool HS>
__global__ void __launch_bounds__(128 + kAuxThreads) deltanet_scan_kernel(const WaveArgs args) {
    extern __shared__ float smem[];
    constexpr std::integral_constant<bool, HS> kHalfState{};

    const std::uint32_t head = blockIdx.x;
    const std::uint32_t row = args.wave[blockIdx.y];
    const RowSpan span = args.rows[row];
    const std::uint32_t in_slot = args.in_slots[row];
    const std::uint32_t out_slot = args.out_slots[row];
    const std::uint32_t dk = args.dk, dv = args.dv;
    const std::uint32_t tid = threadIdx.x;
    const std::uint32_t nthreads = blockDim.x;
    const std::uint32_t scan_threads = nthreads - kAuxThreads;
    const int bar_count = static_cast<int>(nthreads);
    const std::uint32_t nchunks = (span.length + kChunk - 1) / kChunk;

    // Shared memory: two stage buffers, then two read buffers [kChunk][dv].
    const int stage_size = stage_floats(static_cast<int>(dv));
    // Pointers are formed from `smem` by arithmetic and never kept in arrays, so the compiler knows they are in
    // shared memory and uses the shared-memory loads.
    auto stage_of = [&](int b) { return smem + b * stage_size; };
    auto reads_of = [&](int b) { return smem + 2 * stage_size + b * (kChunk * static_cast<int>(dv)); };
    // Inside a stage buffer.
    constexpr int kQ = 0, kK = kChunk * kKeyWidth, kV = 2 * kChunk * kKeyWidth;
    const int kPre = kV + kChunk * static_cast<int>(dv); // factor applied to the state before the token (1 = none)
    const int kInv = kPre + kChunk;                      // 1 / (outstanding decay scale after the token)
    const int kBet = kInv + kChunk;                      // beta
    const int kRs = kBet + kChunk;                       // read scale times the outstanding decay scale
    const int kG = kRs + kChunk;                         // outstanding decay scale after the token
    const int kSlow = kG + kChunk;                       // 1 if a token other than the first of the chunk has pre != 1

    if (tid < scan_threads) {
        // ================= SCAN role =================
        const int p = static_cast<int>(tid % kKeyLanes);  // this thread's key lane
        const std::uint32_t quad = tid / kKeyLanes;        // its column quad: columns 4 * quad .. 4 * quad + 3
        std::uint32_t col[kCols];
        bool valid[kCols];
        #pragma unroll
        for (int cc = 0; cc < kCols; ++cc) {
            col[cc] = kCols * quad + static_cast<std::uint32_t>(cc);
            valid[cc] = col[cc] < dv;
        }
        const bool quad_valid = valid[0];
        Tile tile;
        {
            const std::size_t source_base = (static_cast<std::size_t>(in_slot) * args.value_heads + head) * dk * dv;
            const bool has_source = in_slot != kNoSlot;
            #pragma unroll
            for (int m = 0; m < kGroups; ++m) {
                #pragma unroll
                for (int c = 0; c < 4; ++c) {
                    const std::uint32_t i = 4u * static_cast<std::uint32_t>(p + kKeyLanes * m + 0) + static_cast<std::uint32_t>(c);
                    #pragma unroll
                    for (int cc = 0; cc < kCols; ++cc) {
                        tile.s[cc][m * 4 + c] = (has_source && valid[cc] && i < dk) ? load_state(args.recurrent, source_base + i * dv + col[cc], kHalfState) : 0.0f;
                    }
                }
            }
        }

        // The state is kept as tile = S / g, with g the "outstanding decay scale" (1 at the start of the row): the decay
        // of a token multiplies g, not the 128 state values. The aux warps hand over, for each token, the factor
        // `pre` to multiply the tile by before the token, and 1/g, beta, scale * g after it:
        //   true recalled = g * (k.tile),  correction = beta * (v - g * (k.tile)),  tile += k * (correction / g),
        //   read = scale * g * (q.tile).
        // The first token of every chunk folds the whole decay since the last fold into the tile (pre != 1, g restarts
        // at 1); so does any other token whose decay would take g below kRescale, but that is rare and the aux warps flag
        // such a chunk ("slow"), which runs a loop that tests every token's `pre`. At the end of the row the tile
        // is multiplied by g to give the state.
        // start_token takes the dot products `acc_k` of the next token (k.tile before the token's pre-factor) from the
        // last pass and returns the correction the token applies; it scales the tile when the token asks for it.
        // MODE 0: the token always folds (first of a chunk); 1: it never does; 2: when pre != 1.
        auto start_token = [&](auto mode, const float* nx, std::uint32_t nt, const Acc& acc_k, float (&corr_out)[kCols]) {
            constexpr int MODE = decltype(mode)::value;
            const float inv = nx[kInv + nt];
            const float bet = nx[kBet + nt];
            float recalled[kCols];
            #pragma unroll
            for (int cc = 0; cc < kCols; ++cc) recalled[cc] = finish(acc_k.a[cc]);
            if (MODE != 1) {
                const float pre = nx[kPre + nt];
                if (MODE == 0 || pre != 1.0f) { // uniform across the block
                    #pragma unroll
                    for (int cc = 0; cc < kCols; ++cc) {
                        recalled[cc] = __fmul_rn(recalled[cc], pre);
                        #pragma unroll
                        for (int i = 0; i < kGroups * 4; ++i) tile.s[cc][i] = __fmul_rn(tile.s[cc][i], pre);
                    }
                }
            }
            #pragma unroll
            for (int cc = 0; cc < kCols; ++cc) {
                corr_out[cc] = __fmul_rn(bet, __fsub_rn(__fmul_rn(valid[cc] ? nx[kV + nt * dv + col[cc]] : 0.0f, inv), recalled[cc]));
            }
        };

        float corr[kCols] = {0.0f, 0.0f, 0.0f, 0.0f};
        if (nchunks != 0) {
            // Token 0 of the row: k.S of the incoming state.
            bar_sync(kBarFull, bar_count);
            const float* st = stage_of(0);
            const float4* ks4 = reinterpret_cast<const float4*>(st + kK);
            Acc acc_k;
            #pragma unroll
            for (int cc = 0; cc < kCols; ++cc) for (int u = 0; u < kAcc; ++u) acc_k.a[cc][u] = 0.0f;
            #pragma unroll
            for (int m = 0; m < kGroups; ++m) {
                const float4 kq = ks4[p + kKeyLanes * m];
                const float kk[4] = {kq.x, kq.y, kq.z, kq.w};
                #pragma unroll
                for (int c = 0; c < 4; ++c) {
                    #pragma unroll
                    for (int cc = 0; cc < kCols; ++cc) acc_k.a[cc][c % kAcc] = __fmaf_rn(kk[c], tile.s[cc][m * 4 + c], acc_k.a[cc][c % kAcc]);
                }
            }
            start_token(std::integral_constant<int, 0>{}, st, 0, acc_k, corr);
        }
        float g_end = 1.0f;

        for (std::uint32_t n = 0; n < nchunks; ++n) {
            const int b = static_cast<int>(n & 1u);
            const std::uint32_t count = min(static_cast<std::uint32_t>(kChunk), span.length - n * kChunk);
            const bool more_chunks = n + 1 < nchunks;
            const float* st = stage_of(b);
            float* rd = reads_of(b);
            const float4* ks4 = reinterpret_cast<const float4*>(st + kK);
            const float4* qs4 = reinterpret_cast<const float4*>(st + kQ);
            if (n >= 2) bar_sync(kBarRdEmpty + b, bar_count); // the aux warps are done with chunk n - 2's reads

            // The chunk's tokens. SLOW: the tokens inside the chunk may fold the scale (see start_token).
            auto run_tokens = [&](auto slow) {
                constexpr int INNER = decltype(slow)::value ? 2 : 1;
                for (std::uint32_t tt = 0; tt < count; ++tt) {
                    Acc acc_q, acc_k;
                    #pragma unroll
                    for (int cc = 0; cc < kCols; ++cc) {
                        for (int u = 0; u < kAcc; ++u) acc_q.a[cc][u] = 0.0f;
                        for (int u = 0; u < kAcc; ++u) acc_k.a[cc][u] = 0.0f;
                    }
                    const float4* kn = ks4 + tt * (kKeyWidth / 4);
                    const float4* qn = qs4 + tt * (kKeyWidth / 4);
                    const bool next = tt + 1 < count || more_chunks;
                    const bool cross = tt + 1 == count;
                    if (next) {
                        // Update with token tt, read it, then take the next token's k.S. After a chunk's last token
                        // the next token is the first of the next chunk: wait for its stage buffer.
                        if (cross) bar_sync(kBarFull + (b ^ 1), bar_count);
                        const float* nx = cross ? stage_of(b ^ 1) : st;
                        const std::uint32_t nt = cross ? 0u : tt + 1;
                        token_pass<true>(tile, acc_q, acc_k, kn, qn,
                                         reinterpret_cast<const float4*>(nx + kK) + nt * (kKeyWidth / 4), corr, p);
                    } else {
                        // The row's last token: update and read only.
                        token_pass<false>(tile, acc_q, acc_k, kn, qn, nullptr, corr, p);
                        g_end = st[kG + tt];
                    }
                    // Reads, then (for the next token) the correction.
                    const float rs = st[kRs + tt];
                    float read[kCols];
                    #pragma unroll
                    for (int cc = 0; cc < kCols; ++cc) read[cc] = __fmul_rn(rs, finish(acc_q.a[cc]));
                    if (p == 0 && valid[0]) {
                        if (dv % kCols == 0) { // 16-byte aligned: one store for the quad's four reads
                            *reinterpret_cast<float4*>(rd + tt * dv + col[0]) = make_float4(read[0], read[1], read[2], read[3]);
                        } else {
                            #pragma unroll
                            for (int cc = 0; cc < kCols; ++cc) {
                                if (valid[cc]) rd[tt * dv + col[cc]] = read[cc];
                            }
                        }
                    }
                    if (next) {
                        const float* nx = cross ? stage_of(b ^ 1) : st;
                        const std::uint32_t nt = cross ? 0u : tt + 1;
                        if (cross) {
                            start_token(std::integral_constant<int, 0>{}, nx, nt, acc_k, corr);
                        } else {
                            start_token(std::integral_constant<int, INNER>{}, nx, nt, acc_k, corr);
                        }
                    }
                }
            };
            if (st[kSlow] != 0.0f) { // uniform across the block
                run_tokens(std::true_type{});
            } else {
                run_tokens(std::false_type{});
            }
            if (n + 2 < nchunks) bar_arrive(kBarEmpty + b, bar_count); // the aux warps may refill this stage buffer
            bar_arrive(kBarRdFull + b, bar_count);                     // and normalise these reads
        }

        if (out_slot != kNoSlot && quad_valid) {
            if (nchunks != 0) { // the state is the tile times the outstanding decay scale
                #pragma unroll
                for (int cc = 0; cc < kCols; ++cc) {
                    #pragma unroll
                    for (int i = 0; i < kGroups * 4; ++i) tile.s[cc][i] = __fmul_rn(tile.s[cc][i], g_end);
                }
            }
            const std::size_t target_base = (static_cast<std::size_t>(out_slot) * args.value_heads + head) * dk * dv;
            #pragma unroll
            for (int m = 0; m < kGroups; ++m) {
                #pragma unroll
                for (int c = 0; c < 4; ++c) {
                    const std::uint32_t i = 4u * static_cast<std::uint32_t>(p + kKeyLanes * m) + static_cast<std::uint32_t>(c);
                    if (i < dk) {
                        #pragma unroll
                        for (int cc = 0; cc < kCols; ++cc) {
                            if (valid[cc]) store_state(args.recurrent, target_base + i * dv + col[cc], tile.s[cc][m * 4 + c], args.saturations, kHalfState);
                        }
                    }
                }
            }
        }
    } else {
        // ================= AUX role =================
        // Everything that waits on global memory is issued one step ahead of its use: the convolution windows and the
        // decay/beta inputs of chunk n + 1 while chunk n is finished, the z inputs of chunk n - 1 before chunk n starts.
        // The code is straight-line over the chunk's token slots (no branch that depends on the data), because the
        // aux warps are few and get their speed from the overlap of independent instructions; the optional test taps
        // are written by separate loops.
        const std::uint32_t atid = tid - scan_threads;
        const std::uint32_t warp = atid / 32;
        const std::uint32_t lane = atid % 32;
        constexpr std::uint32_t awarps = kAuxThreads / 32;
        const std::uint32_t taps = TAPS < kMaxTaps ? static_cast<std::uint32_t>(TAPS) : args.taps;
        const std::uint32_t key_heads = args.key_heads;
        const std::uint32_t key_head = head / args.repeat;
        const std::uint32_t conv_dim = 2 * key_heads * dk + args.value_heads * dv;
        const bool tap_owner = head % args.repeat == 0; // writes the shared key head's tap values
        const std::uint32_t per_token = 2 * dk + dv;

        // The key dimension beyond dk stays zero for the whole row, in both stage buffers.
        for (int b = 0; b < 2; ++b) {
            for (std::uint32_t e = atid; e < 2 * kChunk * kKeyWidth; e += kAuxThreads) {
                if (e % kKeyWidth >= dk) stage_of(b)[e] = 0.0f;
            }
        }

        // Thread = channel: this thread's channels (at most kAuxChannels) and their conv weights. A channel past the
        // end (head dimensions below 128) reads channel 0 and its results are never stored.
        std::uint32_t channel[kAuxChannels];
        bool active[kAuxChannels];
        float weight[kAuxChannels][TAPS];
        #pragma unroll
        for (int r = 0; r < kAuxChannels; ++r) {
            const std::uint32_t c = atid + static_cast<std::uint32_t>(r) * kAuxThreads;
            active[r] = c < per_token;
            channel[r] = !active[r] ? 0u
                         : c < dk   ? key_head * dk + c
                         : c < 2 * dk ? key_heads * dk + key_head * dk + (c - dk)
                                      : 2 * key_heads * dk + head * dv + (c - 2 * dk);
            #pragma unroll
            for (int j = 0; j < TAPS; ++j) {
                weight[r][j] = (active[r] && static_cast<std::uint32_t>(j) < taps)
                                   ? args.conv_w[static_cast<std::size_t>(channel[r]) * taps + j] : 0.0f;
            }
        }

        float g_run = 1.0f; // the outstanding decay scale, carried from chunk to chunk (warp 0 only)
        constexpr int kWindow = kChunk + TAPS - 1;
        float xcur[kAuxChannels][kWindow], xnext[kAuxChannels][kWindow];
        float a_cur = 0.0f, b_cur = 0.0f, a_next = 0.0f, b_next = 0.0f;

        // Loads the conv input window of chunk `chunk` for this thread's channels, and a, b of its token `atid`.
        auto load_chunk = [&](std::uint32_t chunk, float (&x)[kAuxChannels][kWindow], float& a_in, float& b_in) {
            const std::uint32_t c0 = chunk * kChunk;
            const std::uint32_t count = min(static_cast<std::uint32_t>(kChunk), span.length - c0);
            if (chunk > 0 && count == kChunk) {
                // The whole window is made of inputs of this row.
                const InT* base = static_cast<const InT*>(args.mixed) + static_cast<std::size_t>(span.offset + c0 - (taps - 1)) * args.mixed_stride;
                #pragma unroll
                for (int r = 0; r < kAuxChannels; ++r) {
                    #pragma unroll
                    for (int i = 0; i < kWindow; ++i) x[r][i] = widen_in(base[static_cast<std::size_t>(i) * args.mixed_stride + channel[r]]);
                }
            } else {
                #pragma unroll
                for (int r = 0; r < kAuxChannels; ++r) {
                    #pragma unroll
                    for (int i = 0; i < kWindow; ++i) {
                        x[r][i] = 0.0f;
                        if (active[r] && static_cast<std::uint32_t>(i) < count + taps - 1) {
                            const std::uint32_t w = c0 + static_cast<std::uint32_t>(i); // position in (tail, inputs)
                            if (w < taps - 1) {
                                if (in_slot != kNoSlot) {
                                    x[r][i] = load_state(args.conv_tails, (static_cast<std::size_t>(in_slot) * (taps - 1) + w) * conv_dim + channel[r], kHalfState);
                                }
                            } else {
                                x[r][i] = widen_in(static_cast<const InT*>(args.mixed)[static_cast<std::size_t>(span.offset + w - (taps - 1)) * args.mixed_stride + channel[r]]);
                            }
                        }
                    }
                }
            }
            a_in = 0.0f;
            b_in = 0.0f;
            if (atid < count) {
                const std::size_t token = span.offset + c0 + atid;
                a_in = widen_in(static_cast<const InT*>(args.a)[token * args.a_stride + head]);
                b_in = widen_in(static_cast<const InT*>(args.b)[token * args.b_stride + head]);
            }
        };

        if (nchunks != 0) load_chunk(0, xcur, a_cur, b_cur);

        for (std::uint32_t n = 0; n <= nchunks; ++n) {
            // z of the previous chunk's tokens, for step D below: this warp's tokens warp, warp + awarps, ...
            constexpr int kTokens = kChunk / static_cast<int>(awarps);
            float zr[kTokens][kKeyWidth / 32];
            if (n >= 1) {
                const std::uint32_t m = n - 1;
                const std::uint32_t count = min(static_cast<std::uint32_t>(kChunk), span.length - m * kChunk);
                #pragma unroll
                for (int u = 0; u < kTokens; ++u) {
                    const std::uint32_t tt = warp + awarps * static_cast<std::uint32_t>(u);
                    const std::size_t token = span.offset + m * kChunk + tt;
                    #pragma unroll
                    for (int q = 0; q < kKeyWidth / 32; ++q) {
                        const std::uint32_t j = lane + 32u * static_cast<std::uint32_t>(q);
                        zr[u][q] = (tt < count && j < dv) ? widen_in(static_cast<const InT*>(args.z)[token * args.z_stride + head * dv + j]) : 0.0f;
                    }
                }
            }

            if (n < nchunks) {
                const int b = static_cast<int>(n & 1u);
                const std::uint32_t c0 = n * kChunk;
                const std::uint32_t count = min(static_cast<std::uint32_t>(kChunk), span.length - c0);
                float* qs = stage_of(b) + kQ;
                float* ks = stage_of(b) + kK;
                float* vs = stage_of(b) + kV;
                float* sc = stage_of(b); // the per-token scalars: sc[kPre + t], ...
                if (n >= 2) bar_sync(kBarEmpty + b, bar_count); // the scan warps are done with chunk n - 2

                // ---- A: convolution + SiLU of q, k, v for the chunk, decay and beta ----
                // Straight-line code over the chunk's kChunk token slots (the slots beyond `count` hold zeros and
                // are never read by the scan).
                #pragma unroll
                for (int r = 0; r < kAuxChannels; ++r) {
                    if (!active[r]) continue;
                    const std::uint32_t c = atid + static_cast<std::uint32_t>(r) * kAuxThreads;
                    float* dst = c < dk ? qs + c : c < 2 * dk ? ks + (c - dk) : vs + (c - 2 * dk);
                    const std::uint32_t row_stride = c >= 2 * dk ? dv : static_cast<std::uint32_t>(kKeyWidth);
                    #pragma unroll
                    for (int k = 0; k < kChunk; ++k) {
                        float acc = 0.0f;
                        #pragma unroll
                        for (int j = 0; j < TAPS; ++j) {
                            if (TAPS < kMaxTaps || static_cast<std::uint32_t>(j) < taps) acc = __fmaf_rn(weight[r][j], xcur[r][k + j], acc);
                        }
                        dst[static_cast<std::uint32_t>(k) * row_stride] = silu_f(acc);
                    }
                }
                if (warp == 0) {
                    // The first warp turns the chunk's decays into the scan's scalars (see the scan role): lane t
                    // computes token t's decay and beta, then every lane walks the chunk's tokens in order, through
                    // the scale carried from the previous chunk. The first token folds the carried scale and its own
                    // decay into the state; a later token does the same when the scale would fall below kRescale
                    // (a very strong decay), which flags the chunk "slow". The walk is straight-line code: the decays
                    // are fetched first, then the short chain of products runs.
                    float decay = 1.0f;
                    if (lane < count) {
                        const float g = __fmul_rn(-expf(args.a_log[head]), softplus_f(__fadd_rn(a_cur, args.dt_bias[head])));
                        decay = expf(g);
                    }
                    float d_all[kChunk];
                    #pragma unroll
                    for (int t = 0; t < kChunk; ++t) d_all[t] = __shfl_sync(0xFFFFFFFFu, decay, t);
                    float pre_l = 1.0f, g_l = 1.0f;
                    bool slow_flag = false;
                    #pragma unroll
                    for (int t = 0; t < kChunk; ++t) {
                        const bool live = static_cast<std::uint32_t>(t) < count;
                        const float attempt = __fmul_rn(g_run, d_all[t]);
                        const bool small = attempt < kRescale;
                        const bool reset = small || t == 0; // the first token of a chunk always folds the scale in
                        slow_flag = slow_flag || (live && small && t != 0);
                        const float after = reset ? 1.0f : attempt;
                        if (lane == static_cast<std::uint32_t>(t)) {
                            pre_l = reset ? attempt : 1.0f;
                            g_l = after;
                        }
                        g_run = live ? after : g_run;
                    }
                    const float inv_l = __frcp_rn(g_l);
                    const float rs_l = __fmul_rn(args.read_scale, g_l);
                    const float slow = slow_flag ? 1.0f : 0.0f;                    if (lane < count) {
                        sc[kPre + lane] = pre_l;
                        sc[kInv + lane] = inv_l;
                        sc[kBet + lane] = sigmoid_f(b_cur);
                        sc[kRs + lane] = rs_l;
                        sc[kG + lane] = g_l;
                    }
                    if (lane == 0) sc[kSlow] = slow;
                }
                // The next chunk's inputs are on their way while B and D run.
                if (n + 1 < nchunks) load_chunk(n + 1, xnext, a_next, b_next);
                bar_sync(kBarAux, kAuxThreads);

                // ---- B: L2 normalisation of q and k, four vectors per warp at a time ----
                #pragma unroll
                for (int j0 = 0; j0 < 2 * kChunk / static_cast<int>(awarps); j0 += 4) {
                    float xv[4][kKeyWidth / 32];
                    float sum[4];
                    #pragma unroll
                    for (int u = 0; u < 4; ++u) {
                        const std::uint32_t vec = warp + awarps * static_cast<std::uint32_t>(j0 + u);
                        const float* x = ((vec & 1u) != 0 ? ks : qs) + (vec >> 1) * kKeyWidth;
                        sum[u] = 0.0f;
                        #pragma unroll
                        for (int q = 0; q < kKeyWidth / 32; ++q) {
                            const std::uint32_t i = lane + 32u * static_cast<std::uint32_t>(q);
                            xv[u][q] = (vec < 2 * count && i < dk) ? x[i] : 0.0f;
                            sum[u] = __fmaf_rn(xv[u][q], xv[u][q], sum[u]);
                        }
                    }
                    #pragma unroll
                    for (int u = 0; u < 4; ++u) sum[u] = warp_sum(sum[u]);
                    #pragma unroll
                    for (int u = 0; u < 4; ++u) {
                        const std::uint32_t vec = warp + awarps * static_cast<std::uint32_t>(j0 + u);
                        float* x = ((vec & 1u) != 0 ? ks : qs) + (vec >> 1) * kKeyWidth;
                        const float norm = sqrtf(__fadd_rn(sum[u], args.l2_eps));
                        const float recip_norm = reciprocal(norm);
                        #pragma unroll
                        for (int q = 0; q < kKeyWidth / 32; ++q) {
                            const std::uint32_t i = lane + 32u * static_cast<std::uint32_t>(q);
                            if (vec < 2 * count && i < dk) x[i] = divide_by(xv[u][q], norm, recip_norm);
                        }
                    }
                }
                if (args.tap_conv != nullptr) {
                    // Test taps: the activated convolution as the scan sees it (q and k per KEY head).
                    bar_sync(kBarAux, kAuxThreads);
                    for (std::uint32_t e = atid; e < count * per_token; e += kAuxThreads) {
                        const std::uint32_t tt = e / per_token, c = e % per_token;
                        float value;
                        std::size_t ch;
                        if (c < dk) {
                            if (!tap_owner) continue;
                            value = qs[tt * kKeyWidth + c];
                            ch = key_head * dk + c;
                        } else if (c < 2 * dk) {
                            if (!tap_owner) continue;
                            value = ks[tt * kKeyWidth + (c - dk)];
                            ch = key_heads * dk + key_head * dk + (c - dk);
                        } else {
                            value = vs[tt * dv + (c - 2 * dk)];
                            ch = 2 * key_heads * dk + head * dv + (c - 2 * dk);
                        }
                        args.tap_conv[static_cast<std::size_t>(span.offset + c0 + tt) * args.tap_conv_stride + ch] = value;
                    }
                }
                bar_arrive(kBarFull + b, bar_count); // the scan warps may take this chunk
            }

            if (n >= 1) {
                // ---- D: gated RMSNorm of chunk n - 1's reads, one warp per token (all of the warp's tokens together) ----
                const std::uint32_t m = n - 1;
                const int b = static_cast<int>(m & 1u);
                const std::uint32_t c0 = m * kChunk;
                const std::uint32_t count = min(static_cast<std::uint32_t>(kChunk), span.length - c0);
                bar_sync(kBarRdFull + b, bar_count); // the scan warps have written the reads
                float xv[kTokens][kKeyWidth / 32];
                float sum[kTokens];
                #pragma unroll
                for (int u = 0; u < kTokens; ++u) {
                    const std::uint32_t tt = warp + awarps * static_cast<std::uint32_t>(u);
                    const float* x = reads_of(b) + tt * dv;
                    sum[u] = 0.0f;
                    #pragma unroll
                    for (int q = 0; q < kKeyWidth / 32; ++q) {
                        const std::uint32_t j = lane + 32u * static_cast<std::uint32_t>(q);
                        xv[u][q] = (tt < count && j < dv) ? x[j] : 0.0f;
                        sum[u] = __fmaf_rn(xv[u][q], xv[u][q], sum[u]);
                    }
                }
                #pragma unroll
                for (int u = 0; u < kTokens; ++u) sum[u] = warp_sum(sum[u]);
                #pragma unroll
                for (int u = 0; u < kTokens; ++u) {
                    const std::uint32_t tt = warp + awarps * static_cast<std::uint32_t>(u);
                    const std::size_t token = span.offset + c0 + tt;
                    const float rms = sqrtf(__fadd_rn(__fdiv_rn(sum[u], static_cast<float>(dv)), args.norm_eps));
                    const float recip_rms = reciprocal(rms);
                    #pragma unroll
                    for (int q = 0; q < kKeyWidth / 32; ++q) {
                        const std::uint32_t j = lane + 32u * static_cast<std::uint32_t>(q);
                        const float y = __fmul_rn(__fmul_rn(divide_by(xv[u][q], rms, recip_rms), j < dv ? args.norm_w[j] : 0.0f), silu_f(zr[u][q]));
                        if (tt < count && j < dv) store_out(static_cast<OutT*>(args.out) + token * args.out_stride + head * dv + j, y);
                    }
                }
                if (args.tap_reads != nullptr) {
                    // Test taps: the reads before the gated norm.
                    for (std::uint32_t e = atid; e < count * dv; e += kAuxThreads) {
                        const std::uint32_t tt = e / dv, j = e % dv;
                        args.tap_reads[(span.offset + c0 + tt) * args.tap_reads_stride + head * dv + j] = reads_of(b)[tt * dv + j];
                    }
                }
                if (m + 2 < nchunks) bar_arrive(kBarRdEmpty + b, bar_count); // the scan warps may overwrite them
            }

            if (n + 1 < nchunks) { // the inputs loaded for chunk n + 1 are now the current ones
                #pragma unroll
                for (int r = 0; r < kAuxChannels; ++r) {
                    #pragma unroll
                    for (int i = 0; i < kWindow; ++i) xcur[r][i] = xnext[r][i];
                }
                a_cur = a_next;
                b_cur = b_next;
            }
        }
    }
}

// ---- the conv tail kernel ----

struct TailArgs {
    std::uint32_t taps, conv_dim;
    const void* mixed; // float32 or binary16 (the kernel's InT)
    std::size_t mixed_stride;
    const RowSpan* rows;
    const std::uint32_t *in_slots, *out_slots, *wave;
    void* conv_tails;
    unsigned* saturations;
};

// Thread (row, channel) writes the channel's K-1 new tail values in increasing order. When a row is continued in
// place that order reads every old value before it is overwritten.
template <class InT, bool HS>
__global__ void deltanet_conv_tail_kernel(const TailArgs args) {
    constexpr std::integral_constant<bool, HS> kHalfState{};
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
                                       : load_state(args.conv_tails, (static_cast<std::size_t>(in_slot) * history + w) * args.conv_dim + channel, kHalfState);
        } else {
            value = widen_in(static_cast<const InT*>(args.mixed)[static_cast<std::size_t>(span.offset + w - history) * args.mixed_stride + channel]);
        }
        store_state(args.conv_tails, (static_cast<std::size_t>(out_slot) * history + i) * args.conv_dim + channel, value, args.saturations, kHalfState);
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

template <class OutT, class InT, int TAPS, bool HS>
void run_scan(const detail::WaveArgs& args, std::uint32_t wave_size, std::uint32_t value_heads, std::uint32_t dv,
              cudaStream_t stream) {
    const std::size_t smem_bytes =
        (2u * static_cast<std::size_t>(detail::stage_floats(static_cast<int>(dv))) + 2u * detail::kChunk * dv) * sizeof(float);
    // More than 48 KB of dynamic shared memory has to be allowed per kernel; allow the largest head dimension once.
    static const bool configured = [] {
        constexpr int kMaxDv = static_cast<int>(kDeltaNetMaxHeadDim);
        constexpr int max_bytes = (2 * detail::stage_floats(kMaxDv) + 2 * detail::kChunk * kMaxDv) * static_cast<int>(sizeof(float));
        check(cudaFuncSetAttribute(detail::deltanet_scan_kernel<OutT, InT, TAPS, HS>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   max_bytes),
              "cudaFuncSetAttribute");
        return true;
    }();
    (void)configured;
    // One scan thread per column quad and key lane, rounded up to whole warps.
    const std::uint32_t quads = (dv + detail::kCols - 1) / detail::kCols;
    const std::uint32_t scan_threads = std::max(32u, (quads * detail::kKeyLanes + 31) / 32 * 32);
    const dim3 grid(value_heads, wave_size);
    detail::deltanet_scan_kernel<OutT, InT, TAPS, HS><<<grid, dim3(scan_threads + detail::kAuxThreads), smem_bytes, stream>>>(args);
}

} // namespace

template <class OutT, class InT>
void launch_deltanet_scan(const DeltaNetGeometry& g, const DeltaNetWeights& w, const DeltaNetInputsT<InT>& in,
                          const DeltaNetRows& rows, const DeltaNetStates& states, OutT* out, std::size_t out_stride,
                          cudaStream_t stream, const DeltaNetOptions& options) {
    validate(g, rows);
    if (rows.wave_size == 0) return;
    if (w.conv == nullptr || w.a_log == nullptr || w.dt_bias == nullptr || w.norm == nullptr ||
        in.mixed.data == nullptr || in.z.data == nullptr || in.b.data == nullptr || in.a.data == nullptr ||
        states.recurrent == nullptr || out == nullptr || (g.conv_kernel > 1 && states.conv_tails == nullptr)) {
        throw std::invalid_argument("deltanet: a null weight, input, state or output pointer");
    }
    // `lanes_per_column` once chose among thread mappings; there is one mapping now, with one set of result bits.
    const int lanes = options.lanes_per_column;
    if (lanes != 0 && lanes != 1 && lanes != 2 && lanes != 4 && lanes != 8) {
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
    args.saturations = states.saturations;
    args.out = out;
    args.out_stride = out_stride;
    args.tap_conv = options.taps.conv_act;
    args.tap_conv_stride = options.taps.conv_act_stride;
    args.tap_reads = options.taps.reads;
    args.tap_reads_stride = options.taps.reads_stride;

    constexpr int kMaxTapsT = static_cast<int>(kDeltaNetMaxConvKernel);
    if (g.conv_kernel == 4) {
        if (states.half_storage) run_scan<OutT, InT, 4, true>(args, rows.wave_size, g.value_heads, g.value_head_dim, stream);
        else run_scan<OutT, InT, 4, false>(args, rows.wave_size, g.value_heads, g.value_head_dim, stream);
    } else {
        if (states.half_storage) run_scan<OutT, InT, kMaxTapsT, true>(args, rows.wave_size, g.value_heads, g.value_head_dim, stream);
        else run_scan<OutT, InT, kMaxTapsT, false>(args, rows.wave_size, g.value_heads, g.value_head_dim, stream);
    }
    check(cudaGetLastError(), "deltanet scan kernel launch");
}

// float32 inputs with either output type, and binary16 inputs (the multiply's result left in binary16) with binary16 output.
template void launch_deltanet_scan<float, float>(const DeltaNetGeometry&, const DeltaNetWeights&, const DeltaNetInputsT<float>&,
                                                 const DeltaNetRows&, const DeltaNetStates&, float*, std::size_t,
                                                 cudaStream_t, const DeltaNetOptions&);
template void launch_deltanet_scan<__half, float>(const DeltaNetGeometry&, const DeltaNetWeights&, const DeltaNetInputsT<float>&,
                                                  const DeltaNetRows&, const DeltaNetStates&, __half*, std::size_t,
                                                  cudaStream_t, const DeltaNetOptions&);
template void launch_deltanet_scan<__half, __half>(const DeltaNetGeometry&, const DeltaNetWeights&, const DeltaNetInputsT<__half>&,
                                                   const DeltaNetRows&, const DeltaNetStates&, __half*, std::size_t,
                                                   cudaStream_t, const DeltaNetOptions&);

template <class InT>
void launch_deltanet_conv_tail(const DeltaNetGeometry& g, const DeltaNetInputsT<InT>& in, const DeltaNetRows& rows,
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
    args.saturations = states.saturations;
    constexpr std::uint32_t kThreads = 128;
    const dim3 grid((g.conv_dim() + kThreads - 1) / kThreads, rows.wave_size);
    if (states.half_storage) detail::deltanet_conv_tail_kernel<InT, true><<<grid, dim3(kThreads), 0, stream>>>(args);
    else detail::deltanet_conv_tail_kernel<InT, false><<<grid, dim3(kThreads), 0, stream>>>(args);
    check(cudaGetLastError(), "deltanet conv tail kernel launch");
}

template void launch_deltanet_conv_tail<float>(const DeltaNetGeometry&, const DeltaNetInputsT<float>&, const DeltaNetRows&,
                                               const DeltaNetStates&, cudaStream_t);
template void launch_deltanet_conv_tail<__half>(const DeltaNetGeometry&, const DeltaNetInputsT<__half>&, const DeltaNetRows&,
                                                const DeltaNetStates&, cudaStream_t);

} // namespace sepal::cuda
