// The full-attention mixer's non-projection work on the device; see kernels_attention.hpp for the contract.
#include "kernels_attention.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include "device.hpp"
#include <string>

// The kernels have external linkage on purpose (see convert_kernels.cu).

namespace sepal::cuda {

namespace {

constexpr std::uint32_t kMaxHeadDim = 256;
constexpr std::uint32_t kMaxRotaryPairs = 128;

} // namespace

// ---- shared device helpers ----

__device__ inline float to_float(float x) { return x; }
__device__ inline float to_float(__half x) { return __half2float(x); }

template <class T>
__device__ inline T from_float(float x);
template <>
__device__ inline float from_float<float>(float x) {
    return x;
}
template <>
__device__ inline __half from_float<__half>(float x) {
    return __float2half_rn(x);
}

// The INFINITY macro is a double constant under MSVC and does not convert to float in device code.
__device__ inline float neg_inf() { return __int_as_float(0xFF800000); }

// sigmoid(x) = 1 / (1 + exp(-x)), written with exp of a non-positive number so that it cannot overflow: for
// negative x it is exp(x) / (1 + exp(x)). The reference writes exp(-softplus(-x)); the two agree to a few ulp.
__device__ inline float sigmoid_stable(float x) {
    const float e = expf(-fabsf(x));
    const float s = 1.0f / (1.0f + e);
    return x >= 0.0f ? s : e * s;
}

// The row a block works on: blockIdx.z (tiled) or blockIdx.y (the other kernels) indexes the run list.
__device__ inline std::uint32_t run_row(const AttentionRows& rows, std::uint32_t slot) {
    return rows.run_rows != nullptr ? rows.run_rows[slot] : slot;
}

// Tokens before the row's own segment.
__device__ inline std::uint32_t context_tokens(const KvSegment* segments, const AttentionRowDesc& desc) {
    if (desc.indexed != 0) return segments[desc.first_segment + desc.segment_count - 1].before;
    std::uint32_t n = 0;
    for (std::uint32_t k = 0; k + 1 < desc.segment_count; ++k) n += segments[desc.first_segment + k].tokens;
    return n;
}

// The segment that holds key j of the chain, and j's index inside it. j must be below the chain's token count.
__device__ inline const KvSegment* locate_key(const KvSegment* segments, const AttentionRowDesc& desc, std::uint32_t j,
                                              std::uint32_t& local) {
    if (desc.indexed != 0) {
        // The first context segment that ends after key j (segment ends never decrease); none: the row's own segment.
        std::uint32_t lo = 0, hi = desc.segment_count - 1;
        const KvSegment* first = segments + desc.first_segment;
        while (lo < hi) {
            const std::uint32_t mid = (lo + hi) / 2;
            if (first[mid].before + first[mid].tokens > j) hi = mid;
            else lo = mid + 1;
        }
        local = j - first[lo].before;
        return first + lo;
    }
    std::uint32_t base = 0;
    const KvSegment* seg = segments + desc.first_segment;
    for (std::uint32_t k = 0; k + 1 < desc.segment_count; ++k, ++seg) {
        if (j < base + seg->tokens) {
            local = j - base;
            return seg;
        }
        base += seg->tokens;
    }
    local = j - base; // the row's own segment
    return seg;
}

// ---- prepare: per-head norm, RoPE, storing keys and values ----

constexpr unsigned kPrepareThreads = 256; // 8 warps, one head vector at a time per warp

struct PrepareParams {
    AttentionRows rows;
    std::uint32_t heads;
    std::uint32_t kv_heads;
    std::uint32_t head_dim;
    std::uint32_t pairs; // rotary_dims / 2
    float eps;
    FloatBand q_gate;
    FloatBand k;
    FloatBand v;
    const float* q_norm;
    const float* k_norm;
    void* q_out;
    std::size_t q_out_stride;
    bool do_queries;
    bool do_keys_values;
    double inverse_frequency[kMaxRotaryPairs];
};

template <class T>
__global__ void attention_prepare_kernel(const PrepareParams p) {
    __shared__ float cosines[kMaxRotaryPairs];
    __shared__ float sines[kMaxRotaryPairs];

    const std::uint32_t row = run_row(p.rows, blockIdx.y);
    const RowSpan span = p.rows.spans[row];
    const std::uint32_t t = blockIdx.x;
    if (t >= span.length) return;
    const AttentionRowDesc desc = p.rows.descs[row];
    const KvSegment own = p.rows.segments[desc.first_segment + desc.segment_count - 1];
    const std::size_t token = static_cast<std::size_t>(span.offset) + t;
    const std::uint32_t head_dim = p.head_dim;
    const std::uint32_t pairs = p.pairs;
    const std::uint32_t kv_width = p.kv_heads * head_dim;

    // The angle is a float64 product of the position and the frequency, as in the reference; the cosine and sine
    // are taken in float64 and rounded to float once.
    for (std::uint32_t i = threadIdx.x; i < pairs; i += blockDim.x) {
        const double angle = static_cast<double>(desc.start_position + t) * p.inverse_frequency[i];
        double sine, cosine;
        sincos(angle, &sine, &cosine);
        cosines[i] = static_cast<float>(cosine);
        sines[i] = static_cast<float>(sine);
    }
    __syncthreads();

    const unsigned warp = threadIdx.x / 32;
    const unsigned lane = threadIdx.x % 32;
    T* const keys = static_cast<T*>(own.keys) + static_cast<std::size_t>(t) * kv_width;
    T* const values = static_cast<T*>(own.values) + static_cast<std::size_t>(t) * kv_width;

    for (std::uint32_t v = warp; v < p.heads + p.kv_heads; v += kPrepareThreads / 32) {
        const bool is_query = v < p.heads;
        if (is_query ? !p.do_queries : !p.do_keys_values) continue;
        const float* src = is_query ? p.q_gate.data + token * p.q_gate.stride + static_cast<std::size_t>(v) * 2 * head_dim
                                    : p.k.data + token * p.k.stride + static_cast<std::size_t>(v - p.heads) * head_dim;
        const float* weight = is_query ? p.q_norm : p.k_norm;
        T* dst = is_query ? static_cast<T*>(p.q_out) + token * p.q_out_stride + static_cast<std::size_t>(v) * head_dim
                          : keys + static_cast<std::size_t>(v - p.heads) * head_dim;

        // Sum of squares: each lane adds its elements in order, then a fixed butterfly across the lanes.
        float sum_sq = 0.0f;
        for (std::uint32_t d = lane; d < head_dim; d += 32) sum_sq = fmaf(src[d], src[d], sum_sq);
        for (int offset = 16; offset > 0; offset >>= 1) sum_sq += __shfl_xor_sync(0xFFFFFFFFu, sum_sq, offset);
        const float rms = sqrtf(__fadd_rn(__fdiv_rn(sum_sq, static_cast<float>(head_dim)), p.eps));

        for (std::uint32_t i = lane; i < pairs; i += 32) {
            const float first = __fmul_rn(__fdiv_rn(src[i], rms), __fadd_rn(1.0f, weight[i]));
            const float second = __fmul_rn(__fdiv_rn(src[pairs + i], rms), __fadd_rn(1.0f, weight[pairs + i]));
            const float c = cosines[i];
            const float s = sines[i];
            dst[i] = from_float<T>(__fsub_rn(__fmul_rn(first, c), __fmul_rn(second, s)));
            dst[pairs + i] = from_float<T>(__fadd_rn(__fmul_rn(second, c), __fmul_rn(first, s)));
        }
        for (std::uint32_t d = 2 * pairs + lane; d < head_dim; d += 32) {
            dst[d] = from_float<T>(__fmul_rn(__fdiv_rn(src[d], rms), __fadd_rn(1.0f, weight[d])));
        }
    }

    if (!p.do_keys_values) return;
    const float* v_src = p.v.data + token * p.v.stride;
    for (std::uint32_t j = threadIdx.x; j < kv_width; j += blockDim.x) values[j] = from_float<T>(v_src[j]);
}

// ---- the plain attention kernel ----
// One block per (token, query head). The keys are walked in chunks of 128 counted from the chain's first token,
// each chunk updating a running maximum, sum and weighted sum (online softmax). Scores are reduced by a fixed
// butterfly across a warp; the weighted sum of values is accumulated key by key by the thread that owns the
// dimension. Nothing depends on anything outside the row's chain.

constexpr unsigned kRefThreads = 128;
constexpr unsigned kRefChunk = 128; // one score per thread

struct ReferenceParams {
    AttentionRows rows;
    std::uint32_t heads;
    std::uint32_t kv_heads;
    std::uint32_t head_dim;
    float scale;
    const void* q;
    std::size_t q_stride;
    FloatBand gate;
    void* out;
    std::size_t out_stride;
};

template <class TKV, class TOut>
__global__ void attention_reference_kernel(const ReferenceParams p) {
    extern __shared__ float ref_smem[];
    __shared__ float red_max[kRefThreads / 32];
    __shared__ float red_sum[kRefThreads / 32];

    const std::uint32_t row = run_row(p.rows, blockIdx.y);
    const RowSpan span = p.rows.spans[row];
    const std::uint32_t t = blockIdx.x;
    if (t >= span.length) return;
    const std::uint32_t head = blockIdx.z;
    const AttentionRowDesc desc = p.rows.descs[row];
    const std::uint32_t head_dim = p.head_dim;
    const std::uint32_t kv_width = p.kv_heads * head_dim;
    const std::uint32_t kv_offset = (head / (p.heads / p.kv_heads)) * head_dim;
    const std::size_t token = static_cast<std::size_t>(span.offset) + t;
    const unsigned tid = threadIdx.x;
    const unsigned warp = tid / 32;
    const unsigned lane = tid % 32;

    float* const qs = ref_smem;
    float* const ss = ref_smem + head_dim; // scores, then probabilities, of the current chunk
    for (std::uint32_t d = tid; d < head_dim; d += kRefThreads) {
        qs[d] = to_float(static_cast<const TKV*>(p.q)[token * p.q_stride + static_cast<std::size_t>(head) * head_dim + d]);
    }
    __syncthreads();

    const std::uint32_t visible = context_tokens(p.rows.segments, desc) + t + 1;
    float running_max = neg_inf();
    float running_sum = 0.0f;
    float acc[2] = {0.0f, 0.0f}; // dimensions tid and tid + 128 (head_dim is at most 256)

    for (std::uint32_t c0 = 0; c0 < visible; c0 += kRefChunk) {
        const std::uint32_t n = min(kRefChunk, visible - c0);
        for (std::uint32_t jj = warp; jj < n; jj += kRefThreads / 32) {
            std::uint32_t local;
            const KvSegment* seg = locate_key(p.rows.segments, desc, c0 + jj, local);
            const TKV* key = static_cast<const TKV*>(seg->keys) + static_cast<std::size_t>(local) * kv_width + kv_offset;
            float s = 0.0f;
            for (std::uint32_t d = lane; d < head_dim; d += 32) s = fmaf(qs[d], to_float(key[d]), s);
            for (int offset = 16; offset > 0; offset >>= 1) s += __shfl_xor_sync(0xFFFFFFFFu, s, offset);
            if (lane == 0) ss[jj] = __fmul_rn(s, p.scale);
        }
        __syncthreads();

        // Thread tid owns the chunk's key tid: the chunk maximum, the new probabilities and their sum.
        const float x = tid < n ? ss[tid] : neg_inf();
        float m = x;
        for (int offset = 16; offset > 0; offset >>= 1) m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFu, m, offset));
        if (lane == 0) red_max[warp] = m;
        __syncthreads();
        const float chunk_max = fmaxf(fmaxf(red_max[0], red_max[1]), fmaxf(red_max[2], red_max[3]));
        const float new_max = fmaxf(running_max, chunk_max);
        const float alpha = expf(running_max - new_max); // 0 on the first chunk
        const float prob = tid < n ? expf(x - new_max) : 0.0f;
        ss[tid] = prob;
        float sum = prob;
        for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_xor_sync(0xFFFFFFFFu, sum, offset);
        if (lane == 0) red_sum[warp] = sum;
        __syncthreads();
        running_sum = running_sum * alpha + (((red_sum[0] + red_sum[1]) + red_sum[2]) + red_sum[3]);
        running_max = new_max;

        acc[0] *= alpha;
        acc[1] *= alpha;
        for (std::uint32_t j = 0; j < n; ++j) {
            std::uint32_t local;
            const KvSegment* seg = locate_key(p.rows.segments, desc, c0 + j, local);
            const TKV* value = static_cast<const TKV*>(seg->values) + static_cast<std::size_t>(local) * kv_width + kv_offset;
            const float pj = ss[j];
            if (tid < head_dim) acc[0] = fmaf(pj, to_float(value[tid]), acc[0]);
            if (tid + kRefThreads < head_dim) acc[1] = fmaf(pj, to_float(value[tid + kRefThreads]), acc[1]);
        }
        __syncthreads();
    }

    TOut* const out = static_cast<TOut*>(p.out) + token * p.out_stride + static_cast<std::size_t>(head) * head_dim;
    const float* const gate = p.gate.data + token * p.gate.stride + static_cast<std::size_t>(head) * 2 * head_dim + head_dim;
    for (int k = 0; k < 2; ++k) {
        const std::uint32_t d = tid + static_cast<std::uint32_t>(k) * kRefThreads;
        if (d < head_dim) out[d] = from_float<TOut>(__fmul_rn(acc[k] / running_sum, sigmoid_stable(gate[d])));
    }
}

// ---- the tiled tensor-core kernel (F16 keys and values) ----
// One block (4 warps) takes one KV head of one row and 64 "query rows": the (token, head) pairs of one KV group
// for 64 / group consecutive tokens, token-major. Each warp owns 16 of them. The block walks the chain's keys 32 at
// a time (tiles counted from the chain's first token, so they do not depend on segment boundaries or on the
// query tile): S = Q K^T and O += P V on mma.sync m16n8k16 with fp32 accumulation, an online softmax in
// registers, P rounded to binary16 for the second product. Tiles in which a row sees nothing contribute exactly
// zero, so a token's bits do not depend on the other tokens of its block.
//
// Shared memory: the query tile (64 rows) and one 32-row buffer that holds the keys of a tile and then its values,
// 48 KB at head_dim 256, so that two blocks share an SM and one block's softmax and loads overlap the other's
// multiplies. Rows are not padded; the 16-byte chunks of a row are permuted by the row number modulo 8 (head_dim 64
// and up) so that ldmatrix reads are free of bank conflicts. A 16-wide head is padded instead. Tiles are fetched with
// cp.async.

constexpr unsigned kTiledThreads = 128;
constexpr unsigned kTileRows = 64;
constexpr unsigned kTileKeys = 32;

struct TiledParams {
    AttentionRows rows;
    std::uint32_t kv_heads;
    std::uint32_t group_log2;
    float scale_log2; // scale * log2(e): scores live in the base-2 exponent domain
    const __half* q;
    std::size_t q_stride;
    FloatBand gate;
    void* out;
    std::size_t out_stride;
};

__device__ inline void cp_async16(void* shared, const void* global) {
#if SEPAL_TENSOR_CORES
    const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(shared));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(address), "l"(global) : "memory");
#else
    __trap();
#endif
}
__device__ inline void cp_async_commit() {
#if SEPAL_TENSOR_CORES
    asm volatile("cp.async.commit_group;\n" ::: "memory");
#else
    __trap();
#endif
}
template <int N>
__device__ inline void cp_async_wait() {
#if SEPAL_TENSOR_CORES
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
#else
    __trap();
#endif
}
__device__ inline void zero16(void* shared) { *static_cast<uint4*>(shared) = make_uint4(0u, 0u, 0u, 0u); }

__device__ inline void ldmatrix_x4(std::uint32_t (&r)[4], const void* shared) {
#if SEPAL_TENSOR_CORES
    const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(shared));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(address));
#else
    __trap();
#endif
}
__device__ inline void ldmatrix_x4_trans(std::uint32_t (&r)[4], const void* shared) {
#if SEPAL_TENSOR_CORES
    const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(shared));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(address));
#else
    __trap();
#endif
}
__device__ inline void mma_16816(float (&c)[4], const std::uint32_t (&a)[4], std::uint32_t b0, std::uint32_t b1) {
#if SEPAL_TENSOR_CORES
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
                 "{%0, %1, %2, %3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    __trap();
#endif
}
__device__ inline std::uint32_t pack_half2(float low, float high) {
    const __half2 h = __floats2half2_rn(low, high);
    return *reinterpret_cast<const std::uint32_t*>(&h);
}

// Row stride and element offset of a 16-byte chunk of a row of a shared tile, in halves.
template <int HD>
__host__ __device__ constexpr int tile_stride() {
    return HD >= 64 ? HD : HD + 8;
}
template <int HD>
__device__ inline int tile_offset(unsigned row, unsigned chunk) {
    if constexpr (HD >= 64) return static_cast<int>(row) * HD + static_cast<int>((chunk ^ (row & 7u)) * 8u);
    else return static_cast<int>(row) * (HD + 8) + static_cast<int>(chunk) * 8;
}

template <int HD>
__host__ __device__ constexpr std::size_t tiled_smem_bytes() {
    return (static_cast<std::size_t>(kTileRows) + kTileKeys) * tile_stride<HD>() * sizeof(__half);
}

// Loads key tile `tile` (or its values) of the chain into a 32-row shared buffer; positions past the end
// of the chain become zeros, so that a probability of zero never meets a stale value.
template <int HD, unsigned THREADS = kTiledThreads>
__device__ inline void load_kv_tile(__half* buffer, const KvSegment* segments, const AttentionRowDesc& desc,
                                    std::uint32_t total, std::uint32_t tile, std::uint32_t kv_head,
                                    std::uint32_t kv_width, bool values) {
    constexpr int CPR = HD / 8;
    for (unsigned c = threadIdx.x; c < kTileKeys * CPR; c += THREADS) {
        const unsigned r = c / CPR;
        const unsigned ch = c % CPR;
        const std::uint32_t j = tile * kTileKeys + r;
        __half* dst = buffer + tile_offset<HD>(r, ch);
        if (j < total) {
            std::uint32_t local;
            const KvSegment* seg = locate_key(segments, desc, j, local);
            const __half* base = static_cast<const __half*>(values ? seg->values : seg->keys);
            cp_async16(dst, base + static_cast<std::size_t>(local) * kv_width + kv_head * HD + ch * 8);
        } else {
            zero16(dst);
        }
    }
}

template <int HD, class TOut>
__global__ void __launch_bounds__(kTiledThreads, 2) attention_tiled_kernel(const TiledParams p) {
    constexpr int CPR = HD / 8;
    constexpr int NT = HD / 8; // n-tiles of the output
    extern __shared__ __align__(16) unsigned char tiled_smem[];
    __half* const Qs = reinterpret_cast<__half*>(tiled_smem);
    __half* const KVs = Qs + kTileRows * tile_stride<HD>(); // keys of a tile, then its values

    const std::uint32_t row = run_row(p.rows, blockIdx.z);
    const RowSpan span = p.rows.spans[row];
    const std::uint32_t len = span.length;
    const std::uint32_t lg = p.group_log2;
    const std::uint32_t group = 1u << lg;
    const std::uint32_t tokens_per_tile = kTileRows >> lg;
    const std::uint32_t tiles = (len + tokens_per_tile - 1) / tokens_per_tile;
    if (blockIdx.x >= tiles) return;
    const std::uint32_t tile = tiles - 1 - blockIdx.x; // the heaviest query tile of a row first
    const std::uint32_t kv_head = blockIdx.y;
    const AttentionRowDesc desc = p.rows.descs[row];
    const std::uint32_t kv_width = p.kv_heads * HD;
    const std::uint32_t context = context_tokens(p.rows.segments, desc);
    const std::uint32_t total = context + len;
    const std::uint32_t t0 = tile * tokens_per_tile;
    const std::uint32_t last_visible = context + min(len, t0 + tokens_per_tile);
    const std::uint32_t key_tiles = (last_visible + kTileKeys - 1) / kTileKeys;

    const unsigned tid = threadIdx.x;
    const unsigned warp = tid / 32;
    const unsigned lane = tid % 32;
    const unsigned g = lane >> 2;  // the quad's row
    const unsigned qd = lane & 3;  // the lane within the quad

    // The query tile.
    for (unsigned c = tid; c < kTileRows * CPR; c += kTiledThreads) {
        const unsigned r = c / CPR;
        const unsigned ch = c % CPR;
        const std::uint32_t tl = t0 + (r >> lg);
        __half* dst = Qs + tile_offset<HD>(r, ch);
        if (tl < len) {
            const std::size_t head = static_cast<std::size_t>(kv_head) * group + (r & (group - 1));
            cp_async16(dst, p.q + (static_cast<std::size_t>(span.offset) + tl) * p.q_stride + head * HD + ch * 8);
        } else {
            zero16(dst);
        }
    }
    load_kv_tile<HD>(KVs, p.rows.segments, desc, total, 0, kv_head, kv_width, false);
    cp_async_commit();

    // Rows g and g + 8 of this warp's 16: the tokens they belong to and how many keys each sees.
    const unsigned row_a = warp * 16 + g;
    const unsigned row_b = row_a + 8;
    const std::uint32_t token_a = min(t0 + (row_a >> lg), len - 1);
    const std::uint32_t token_b = min(t0 + (row_b >> lg), len - 1);
    const std::uint32_t visible_a = context + token_a + 1;
    const std::uint32_t visible_b = context + token_b + 1;

    float o[NT][4];
#pragma unroll
    for (int i = 0; i < NT; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
    float max_a = neg_inf(), max_b = neg_inf();
    float sum_a = 0.0f, sum_b = 0.0f; // this lane's share of the row sums

    for (std::uint32_t kt = 0; kt < key_tiles; ++kt) {
        cp_async_wait<0>(); // the query tile (first time) and this tile's keys
        __syncthreads();

        float s[4][4];
#pragma unroll
        for (int i = 0; i < 4; ++i) s[i][0] = s[i][1] = s[i][2] = s[i][3] = 0.0f;
#pragma unroll
        for (int ks = 0; ks < HD / 16; ++ks) {
            std::uint32_t a[4];
            ldmatrix_x4(a, Qs + tile_offset<HD>(warp * 16 + (lane & 7) + ((lane >> 3) & 1) * 8, ks * 2 + (lane >> 4)));
#pragma unroll
            for (int np = 0; np < 2; ++np) {
                std::uint32_t b[4];
                ldmatrix_x4(b, KVs + tile_offset<HD>(np * 16 + (lane >> 4) * 8 + (lane & 7), ks * 2 + ((lane >> 3) & 1)));
                mma_16816(s[2 * np], a, b[0], b[1]);
                mma_16816(s[2 * np + 1], a, b[2], b[3]);
            }
        }

        __syncthreads(); // every warp has read the keys: the buffer takes the values
        load_kv_tile<HD>(KVs, p.rows.segments, desc, total, kt, kv_head, kv_width, true);
        cp_async_commit();

        // Mask, scale, online softmax. Row a owns s[.][0..1], row b s[.][2..3].
        float x[4][4];
        float tile_max_a = neg_inf(), tile_max_b = neg_inf();
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const std::uint32_t j = kt * kTileKeys + nt * 8 + qd * 2 + (e & 1);
                const std::uint32_t visible = (e >> 1) ? visible_b : visible_a;
                x[nt][e] = j < visible ? s[nt][e] * p.scale_log2 : neg_inf();
            }
            tile_max_a = fmaxf(tile_max_a, fmaxf(x[nt][0], x[nt][1]));
            tile_max_b = fmaxf(tile_max_b, fmaxf(x[nt][2], x[nt][3]));
        }
        tile_max_a = fmaxf(tile_max_a, __shfl_xor_sync(0xFFFFFFFFu, tile_max_a, 1));
        tile_max_a = fmaxf(tile_max_a, __shfl_xor_sync(0xFFFFFFFFu, tile_max_a, 2));
        tile_max_b = fmaxf(tile_max_b, __shfl_xor_sync(0xFFFFFFFFu, tile_max_b, 1));
        tile_max_b = fmaxf(tile_max_b, __shfl_xor_sync(0xFFFFFFFFu, tile_max_b, 2));
        const float new_max_a = fmaxf(max_a, tile_max_a);
        const float new_max_b = fmaxf(max_b, tile_max_b);
        const float alpha_a = exp2f(max_a - new_max_a); // 0 on the first tile, 1 when the maximum did not move
        const float alpha_b = exp2f(max_b - new_max_b);
        max_a = new_max_a;
        max_b = new_max_b;

        float tile_sum_a = 0.0f, tile_sum_b = 0.0f;
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            x[nt][0] = exp2f(x[nt][0] - new_max_a);
            x[nt][1] = exp2f(x[nt][1] - new_max_a);
            x[nt][2] = exp2f(x[nt][2] - new_max_b);
            x[nt][3] = exp2f(x[nt][3] - new_max_b);
            tile_sum_a += x[nt][0] + x[nt][1];
            tile_sum_b += x[nt][2] + x[nt][3];
        }
        sum_a = sum_a * alpha_a + tile_sum_a;
        sum_b = sum_b * alpha_b + tile_sum_b;
#pragma unroll
        for (int i = 0; i < NT; ++i) {
            o[i][0] *= alpha_a;
            o[i][1] *= alpha_a;
            o[i][2] *= alpha_b;
            o[i][3] *= alpha_b;
        }

        cp_async_wait<0>(); // this tile's values
        __syncthreads();

        // P V: the 16 x 32 probabilities of this warp, as A operands straight from the S accumulators.
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            std::uint32_t a[4];
            a[0] = pack_half2(x[2 * kk][0], x[2 * kk][1]);
            a[1] = pack_half2(x[2 * kk][2], x[2 * kk][3]);
            a[2] = pack_half2(x[2 * kk + 1][0], x[2 * kk + 1][1]);
            a[3] = pack_half2(x[2 * kk + 1][2], x[2 * kk + 1][3]);
#pragma unroll
            for (int dp = 0; dp < HD / 16; ++dp) {
                std::uint32_t b[4];
                ldmatrix_x4_trans(b, KVs + tile_offset<HD>(kk * 16 + ((lane >> 3) & 1) * 8 + (lane & 7), dp * 2 + (lane >> 4)));
                mma_16816(o[2 * dp], a, b[0], b[1]);
                mma_16816(o[2 * dp + 1], a, b[2], b[3]);
            }
        }

        if (kt + 1 < key_tiles) {
            __syncthreads(); // every warp has read the values: the buffer takes the next keys
            load_kv_tile<HD>(KVs, p.rows.segments, desc, total, kt + 1, kv_head, kv_width, false);
            cp_async_commit();
        }
    }

    // The row sums: add the four lanes of a quad in a fixed order.
    sum_a += __shfl_xor_sync(0xFFFFFFFFu, sum_a, 1);
    sum_a += __shfl_xor_sync(0xFFFFFFFFu, sum_a, 2);
    sum_b += __shfl_xor_sync(0xFFFFFFFFu, sum_b, 1);
    sum_b += __shfl_xor_sync(0xFFFFFFFFu, sum_b, 2);

#pragma unroll
    for (int half_row = 0; half_row < 2; ++half_row) {
        const unsigned r = half_row == 0 ? row_a : row_b;
        const std::uint32_t tl = t0 + (r >> lg);
        if (tl >= len) continue;
        const float denominator = half_row == 0 ? sum_a : sum_b;
        const std::size_t head = static_cast<std::size_t>(kv_head) * group + (r & (group - 1));
        const std::size_t token = static_cast<std::size_t>(span.offset) + tl;
        TOut* const out = static_cast<TOut*>(p.out) + token * p.out_stride + head * HD;
        const float* const gate = p.gate.data + token * p.gate.stride + head * 2 * HD + HD;
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int d = nt * 8 + static_cast<int>(qd) * 2 + e;
                out[d] = from_float<TOut>(__fmul_rn(o[nt][half_row * 2 + e] / denominator, sigmoid_stable(gate[d])));
            }
        }
    }
}

// ---- the tiled float32 kernel (F32 keys and values, no tensor cores: every architecture from sm_61 up) ----
// One block takes one KV head of one row and 64 "query rows": the (token, head) pairs of one KV group for 64 / group
// consecutive tokens, token-major, so the group's heads share every key and value the block loads. A thread holds two query
// rows over SLICE = 32 head dimensions (query and output in registers): a head of D dimensions takes D / 32 threads per row pair
// (a narrower head is one slice padded with zeros), a block 32 * D / 32 threads. Every key or value read from shared memory feeds
// eight multiply-adds (two rows times four dimensions). The block walks the chain's keys 16 at a time (tiles counted from the
// chain's first token, so they do not depend on segment boundaries or on the query tile) up to the last key its last token sees
// (causal skip). Per tile: the keys and values go to shared memory (fetched into registers during the previous tile, so the loads
// overlap the arithmetic); a thread's partial score over its 32 dimensions is one fmaf chain in dimension order, the slices are
// added by a fixed butterfly (the same bits in every thread of the row); every thread of the row then runs the same online
// softmax (running maximum, rescale factor exp(m_old - m_new), sum of the probabilities in key order) and O = O * alpha + P V on its
// dimensions (one fmaf chain per output over the keys in order). A tile in which a row sees nothing leaves that row's bits
// unchanged (alpha = exp(0) = 1, p = 0), so a token's result depends on its row's chain only, never on the other tokens of its
// block (proofs/ExactTrack/OnlineSoftmax.lean has the softmax identity).

constexpr unsigned kF32Rows = 64;  // query rows per block
constexpr unsigned kF32Keys = 16;  // keys per tile
constexpr unsigned kF32Slice = 32; // head dimensions per thread
constexpr unsigned kF32Pitch = kF32Slice + 4;                  // floats per key row of a slice in shared memory
constexpr unsigned kF32SliceStride = kF32Keys * kF32Pitch + 4; // 580 = 4 mod 32: the eight slices of one key in different banks

struct TiledF32Params {
    AttentionRows rows;
    std::uint32_t kv_heads;
    std::uint32_t group_log2;
    float scale;
    const float* q;
    std::size_t q_stride;
    FloatBand gate;
    void* out;
    std::size_t out_stride;
};

template <int HD>
__host__ __device__ constexpr int f32_slices() {
    return HD <= static_cast<int>(kF32Slice) ? 1 : HD / static_cast<int>(kF32Slice);
}
template <int HD>
__host__ __device__ constexpr unsigned f32_threads() {
    return kF32Rows / 2 * static_cast<unsigned>(f32_slices<HD>());
}

template <int HD, class TOut>
__global__ void __launch_bounds__(f32_threads<HD>(), 1) attention_f32_tiled_kernel(const TiledF32Params p) {
    constexpr int kSlices = f32_slices<HD>();
    constexpr int kThreads = static_cast<int>(f32_threads<HD>());
    constexpr int kSW = static_cast<int>(kF32Slice);
    constexpr int kKeys = static_cast<int>(kF32Keys);
    constexpr int kHDP = kSlices * kSW;                            // padded head dimension
    constexpr int kVec = kHDP / 4;                                 // float4 per key row
    constexpr int kPer = kKeys * kVec / kThreads;                  // float4 of K (and of V) per thread per tile: 4
    static_assert(kPer * kThreads == kKeys * kVec, "tile loads must divide evenly");
    __shared__ __align__(16) float Ks[kSlices * kF32SliceStride];
    __shared__ __align__(16) float Vs[kSlices * kF32SliceStride];

    const std::uint32_t row = run_row(p.rows, blockIdx.z);
    const RowSpan span = p.rows.spans[row];
    const std::uint32_t lg = p.group_log2;
    const std::uint32_t group = 1u << lg;
    const std::uint32_t tokens_per_tile = kF32Rows >> lg;
    const std::uint32_t tok0 = blockIdx.x * tokens_per_tile;
    if (tok0 >= span.length) return;
    const std::uint32_t kvh = blockIdx.y;
    const AttentionRowDesc desc = p.rows.descs[row];
    const std::uint32_t ctx = context_tokens(p.rows.segments, desc);
    const std::uint32_t n_tok = min(tokens_per_tile, span.length - tok0);
    const std::uint32_t block_visible = ctx + tok0 + n_tok; // keys the block's last token sees
    const std::uint32_t key_tiles = (block_visible + kF32Keys - 1) / kF32Keys;
    const std::size_t kv_width = static_cast<std::size_t>(p.kv_heads) * HD;
    const unsigned tid = threadIdx.x;
    const unsigned slice = tid % kSlices;
    const unsigned pair = tid / kSlices; // query rows 2 pair and 2 pair + 1
    const unsigned d0 = slice * kF32Slice; // this thread's first head dimension
    std::uint32_t t[2], head[2];
    bool live[2];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        const unsigned qrow = 2 * pair + static_cast<unsigned>(r);
        t[r] = tok0 + (qrow >> lg);
        live[r] = t[r] < span.length;
        head[r] = kvh * group + (qrow & (group - 1));
    }

    // ---- this thread's 32 dimensions of its two queries ----
    float q[2][kF32Slice];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        const float* src = live[r] ? p.q + (static_cast<std::size_t>(span.offset) + t[r]) * p.q_stride + static_cast<std::size_t>(head[r]) * HD : nullptr;
#pragma unroll
        for (int i = 0; i < kSW; i += 4) {
            float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (src != nullptr && d0 + static_cast<unsigned>(i) < static_cast<unsigned>(HD)) v = *reinterpret_cast<const float4*>(src + d0 + i);
            q[r][i] = v.x;
            q[r][i + 1] = v.y;
            q[r][i + 2] = v.z;
            q[r][i + 3] = v.w;
        }
    }
    float o[2][kF32Slice];
#pragma unroll
    for (int r = 0; r < 2; ++r)
#pragma unroll
        for (int i = 0; i < kSW; ++i) o[r][i] = 0.0f;
    float m_run[2] = {neg_inf(), neg_inf()};
    float l_run[2] = {0.0f, 0.0f};

    // ---- the tile fetch: kPer float4 of keys and of values per thread ----
    float4 kr[kPer], vr[kPer];
    const auto fetch = [&](std::uint32_t key0) {
#pragma unroll
        for (int e = 0; e < kPer; ++e) {
            const int idx = static_cast<int>(tid) + e * kThreads;
            const std::uint32_t j = key0 + static_cast<std::uint32_t>(idx / kVec);
            const int d = (idx % kVec) * 4;
            kr[e] = vr[e] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (j < block_visible && d < HD) {
                std::uint32_t local;
                const KvSegment* seg = locate_key(p.rows.segments, desc, j, local);
                const std::size_t at = static_cast<std::size_t>(local) * kv_width + static_cast<std::size_t>(kvh) * HD + static_cast<std::size_t>(d);
                kr[e] = *reinterpret_cast<const float4*>(static_cast<const float*>(seg->keys) + at);
                vr[e] = *reinterpret_cast<const float4*>(static_cast<const float*>(seg->values) + at);
            }
        }
    };
    const auto stage = [&]() {
#pragma unroll
        for (int e = 0; e < kPer; ++e) {
            const int idx = static_cast<int>(tid) + e * kThreads;
            const int j = idx / kVec;
            const int d = (idx % kVec) * 4;
            const int at = (d / kSW) * static_cast<int>(kF32SliceStride) + j * static_cast<int>(kF32Pitch) + d % kSW;
            *reinterpret_cast<float4*>(&Ks[at]) = kr[e];
            *reinterpret_cast<float4*>(&Vs[at]) = vr[e];
        }
    };

    fetch(0);
    const float* const ks = Ks + slice * kF32SliceStride;
    const float* const vs = Vs + slice * kF32SliceStride;
    for (std::uint32_t kt = 0; kt < key_tiles; ++kt) {
        const std::uint32_t key0 = kt * kF32Keys;
        __syncthreads(); // the previous tile's arithmetic is done with the buffers
        stage();
        __syncthreads();
        if (kt + 1 < key_tiles) fetch(key0 + kF32Keys); // in flight during this tile's arithmetic

        // S: the partial scores of this thread's 32 dimensions, then the slices added by a fixed butterfly.
        float s[2][kF32Keys];
#pragma unroll
        for (int r = 0; r < 2; ++r)
#pragma unroll
            for (int j = 0; j < kKeys; ++j) s[r][j] = 0.0f;
#pragma unroll
        for (int i = 0; i < kSW; i += 4) {
#pragma unroll
            for (int j = 0; j < kKeys; ++j) {
                const float4 kv = *reinterpret_cast<const float4*>(ks + j * static_cast<int>(kF32Pitch) + i);
#pragma unroll
                for (int r = 0; r < 2; ++r) {
                    s[r][j] = __fmaf_rn(q[r][i], kv.x, s[r][j]);
                    s[r][j] = __fmaf_rn(q[r][i + 1], kv.y, s[r][j]);
                    s[r][j] = __fmaf_rn(q[r][i + 2], kv.z, s[r][j]);
                    s[r][j] = __fmaf_rn(q[r][i + 3], kv.w, s[r][j]);
                }
            }
        }
#pragma unroll
        for (int offset = 1; offset < kSlices; offset <<= 1) {
#pragma unroll
            for (int r = 0; r < 2; ++r)
#pragma unroll
                for (int j = 0; j < kKeys; ++j) s[r][j] = __fadd_rn(s[r][j], __shfl_xor_sync(0xFFFFFFFFu, s[r][j], offset));
        }
        // Scale, mask (a key is visible to token t when its chain index is at most ctx + t), online softmax. A row past the
        // row's end (the last tile of a short row) keeps zeros and is never written out.
        float alpha[2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            float mx = neg_inf();
#pragma unroll
            for (int j = 0; j < kKeys; ++j) {
                const std::uint32_t key = key0 + static_cast<std::uint32_t>(j);
                s[r][j] = !live[r] ? 0.0f : key <= ctx + t[r] ? __fmul_rn(s[r][j], p.scale) : neg_inf();
                mx = fmaxf(mx, s[r][j]);
            }
            const float new_max = fmaxf(m_run[r], mx);
            alpha[r] = expf(m_run[r] - new_max); // 0 on the first tile, 1 on a tile this row does not see
            float sum = 0.0f;
#pragma unroll
            for (int j = 0; j < kKeys; ++j) {
                s[r][j] = expf(s[r][j] - new_max);
                sum = __fadd_rn(sum, s[r][j]);
            }
            l_run[r] = __fadd_rn(__fmul_rn(l_run[r], alpha[r]), sum);
            m_run[r] = new_max;
        }
        // O = O * alpha + P V on this thread's dimensions.
#pragma unroll
        for (int r = 0; r < 2; ++r)
#pragma unroll
            for (int i = 0; i < kSW; ++i) o[r][i] = __fmul_rn(o[r][i], alpha[r]);
#pragma unroll
        for (int j = 0; j < kKeys; ++j) {
#pragma unroll
            for (int i = 0; i < kSW; i += 4) {
                const float4 vv = *reinterpret_cast<const float4*>(vs + j * static_cast<int>(kF32Pitch) + i);
#pragma unroll
                for (int r = 0; r < 2; ++r) {
                    o[r][i] = __fmaf_rn(s[r][j], vv.x, o[r][i]);
                    o[r][i + 1] = __fmaf_rn(s[r][j], vv.y, o[r][i + 1]);
                    o[r][i + 2] = __fmaf_rn(s[r][j], vv.z, o[r][i + 2]);
                    o[r][i + 3] = __fmaf_rn(s[r][j], vv.w, o[r][i + 3]);
                }
            }
        }
    }

#pragma unroll
    for (int r = 0; r < 2; ++r) {
        if (!live[r]) continue;
        const std::size_t token = static_cast<std::size_t>(span.offset) + t[r];
        TOut* const out = static_cast<TOut*>(p.out) + token * p.out_stride + static_cast<std::size_t>(head[r]) * HD;
        const float* const gate = p.gate.data + token * p.gate.stride + static_cast<std::size_t>(head[r]) * 2 * HD + HD;
#pragma unroll
        for (int i = 0; i < kSW; ++i) {
            const unsigned d = d0 + static_cast<unsigned>(i);
            if (d < static_cast<unsigned>(HD)) out[d] = from_float<TOut>(__fmul_rn(o[r][i] / l_run[r], sigmoid_stable(gate[d])));
        }
    }
}

// ---- the float32 split kernel (F32 keys and values, head_dim 256, every architecture from sm_61 up) ----
// The first float32 kernel gives each key or value it reads from shared memory to two query rows only, so it is bound by the
// shared-memory pipe at a quarter of the cores' rate (Pascal: 32 words per clock per SM against 128 multiply-adds), and it
// all-reduces every score across eight lanes and then runs the softmax eight times over. This one:
//   * A block is 64 query rows (one KV group of 64 / group tokens) and 8 warps. A half-warp owns four rows; lane s of it holds the
//     four rows' queries and outputs over the 16 dimensions 4s + 64i + c (i, c < 4), so each float4 it reads from shared memory
//     feeds 16 multiply-adds and a quarter-warp's eight float4 fall in 32 different banks.
//   * Keys 8 at a time. The 16 lanes' partial scores (4 rows x 8 keys) are reduce-scattered over the half-warp (masks 8, 4, 2, 1:
//     30 shuffles instead of 128 for an all-reduce); lane s ends with row s / 4, keys 2 (s % 4) and 2 (s % 4) + 1, does the
//     softmax for those two (exp2 on the fast path, ex2.approx), and the probabilities and rescale factors go back to the 16 lanes
//     by 36 shuffles for P V. Each lane keeps its own share of the row sum; the four shares are added by a fixed butterfly at the end.
//   * Two shared buffers of 8 keys and 8 values (32 KB): tile k + 1 is stored from registers while tile k is multiplied and
//     tile k + 2 is fetched from global memory, one barrier per tile.
//   * Split K/V: the keys of a row are cut into chunks of kSplitChunk counted from the chain's first key (the last may be short),
//     and each (query tile, chunk) is its own block, so a short call over a long cache still fills the card. A token whose
//     visible keys fit in one chunk is finished by its block. Otherwise each chunk's block writes its partial state (m, l, acc:
//     the running maximum in the base-2 domain, the sum of the probabilities, the weighted values) to a workspace and a second
//     launch merges the partials by a fixed pairwise tree over the chunk index (merge: M = max(m1, m2), l = l1 2^(m1 - M) + l2 2^(m2 - M), acc likewise).
// Every score, sum and product runs in an order that depends on the key's index in its row's chain and on nothing else (the tiles
// and chunks are counted from the chain's first key; a tile a row cannot see leaves its state's bits as they were: the rescale
// factor is exactly 1 and the probabilities exactly 0), so a token's bits depend on its own chain only: not on its call, its
// block, the other rows, the window of tiles, or the grid. proofs/GpuTrack/GpuTrack/AttnSplit.lean proves the merge is a monoid,
// that the chunked fold equals softmax attention for any chunking, that skipped tiles and chunks drop only zero-weight terms, and
// that the operation tree of a token does not depend on the launch.

constexpr unsigned kSplitThreads = 256;
constexpr unsigned kSplitRows = 64;
constexpr unsigned kSplitKeys = 8;
constexpr int kSplitHD = 256;
constexpr std::uint32_t kSplitChunk = 2048;
constexpr unsigned kSplitSlot = kSplitHD + 4; // floats per partial: acc[256], m, l, two pad (16-byte aligned slots)

struct SplitF32Params {
    AttentionRows rows;
    std::uint32_t heads;
    std::uint32_t kv_heads;
    std::uint32_t group_log2;
    float scale_log2; // scale * log2(e)
    const float* q;
    std::size_t q_stride;
    FloatBand gate;
    void* out;
    std::size_t out_stride;
    float* part;             // the chunk partials of this window, or null when every token fits one chunk
    std::uint32_t chunk;     // keys per chunk
    std::uint32_t chunks;    // chunk slots per (token, head): ceil(max_row_keys / chunk)
    std::uint32_t tile0;     // the window's first query tile
    std::uint32_t win_tiles; // query tiles in the window
};

// 2^x on the special-function unit. PTX ISA: ex2.approx.f32 has a maximum relative error of 2^-22 over the full range (ftz: results
// below 2^-126 are zero). The test suite `exp2` measures it against float64. ex2(-inf) = +0.
__device__ inline float ex2_fast(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ inline std::uint32_t split_chunks_of(std::uint32_t visible, std::uint32_t chunk) { return (visible + chunk - 1) / chunk; }

// The output's row stride and base need not be 16-byte aligned: four scalar stores.
template <class TOut>
__device__ inline void store4(TOut* dst, float4 v) {
    dst[0] = from_float<TOut>(v.x);
    dst[1] = from_float<TOut>(v.y);
    dst[2] = from_float<TOut>(v.z);
    dst[3] = from_float<TOut>(v.w);
}

template <class TOut>
__global__ void __launch_bounds__(kSplitThreads, 1) attention_f32_split_kernel(const SplitF32Params p) {
    __shared__ __align__(16) float Ks[2][kSplitKeys][kSplitHD];
    __shared__ __align__(16) float Vs[2][kSplitKeys][kSplitHD];

    const std::uint32_t row = run_row(p.rows, blockIdx.z);
    const RowSpan span = p.rows.spans[row];
    const std::uint32_t lg = p.group_log2;
    const std::uint32_t group = 1u << lg;
    const std::uint32_t tpt = kSplitRows >> lg; // tokens per query tile
    const std::uint32_t tile = p.tile0 + blockIdx.x / p.chunks;
    const std::uint32_t chunk = blockIdx.x % p.chunks;
    const std::uint32_t tok0 = tile * tpt;
    if (tok0 >= span.length) return;
    const AttentionRowDesc desc = p.rows.descs[row];
    const std::uint32_t ctx = context_tokens(p.rows.segments, desc);
    const std::uint32_t n_tok = min(tpt, span.length - tok0);
    const std::uint32_t block_visible = ctx + tok0 + n_tok; // keys the block's last token sees
    const std::uint32_t key_begin = chunk * p.chunk;
    if (key_begin >= block_visible) return;
    const std::uint32_t key_end = min(key_begin + p.chunk, block_visible);
    const std::uint32_t tiles = (key_end - key_begin + kSplitKeys - 1) / kSplitKeys;
    const std::size_t kv_width = static_cast<std::size_t>(p.kv_heads) * kSplitHD;
    const unsigned kvh = blockIdx.y;
    const unsigned tid = threadIdx.x;
    const unsigned lane = tid & 31u;
    const unsigned warp = tid >> 5;
    const unsigned hbase = lane & 16u; // the half-warp's first lane
    const unsigned s = lane & 15u;
    const unsigned rbase = warp * 8 + (lane >> 4) * 4; // the half-warp's first query row in the block

    // Row r of this half-warp (r < 4) is block row rbase + r: token tok0 + (row >> lg), head kvh * group + (row & (group - 1)).
    // The row whose softmax this lane runs (row s / 4 of the four) and the keys it sees.
    const unsigned own = rbase + (s >> 2);
    const std::uint32_t own_t = tok0 + (own >> lg);
    const std::uint32_t own_visible = own_t < span.length ? ctx + own_t + 1 : 0;

    // Queries: q[r][4 i + c] is dimension 4 s + 64 i + c of row r.
    float q[4][16];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const unsigned qrow = rbase + static_cast<unsigned>(r);
        const std::uint32_t tr = tok0 + (qrow >> lg);
        const bool live = tr < span.length;
        const std::uint32_t hr = kvh * group + (qrow & (group - 1));
        const float* src = p.q + (static_cast<std::size_t>(span.offset) + (live ? tr : 0)) * p.q_stride +
                           static_cast<std::size_t>(hr) * kSplitHD + 4 * s;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (live) v = *reinterpret_cast<const float4*>(src + 64 * i);
            q[r][4 * i] = v.x;
            q[r][4 * i + 1] = v.y;
            q[r][4 * i + 2] = v.z;
            q[r][4 * i + 3] = v.w;
        }
    }
    float o[4][16];
#pragma unroll
    for (int r = 0; r < 4; ++r)
#pragma unroll
        for (int i = 0; i < 16; ++i) o[r][i] = 0.0f;
    float m_run = neg_inf(); // the own row's running maximum (base-2 domain), the same in the row's four lanes
    float l_run = 0.0f;      // this lane's share of the own row's sum

    // Tile fetch: keys (tid / 64) and (tid / 64) + 4 of the tile, float4 (tid % 64) of each, keys and values.
    float4 kr[2], vr[2];
    const auto fetch = [&](std::uint32_t key0) {
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            const std::uint32_t j = key0 + (tid >> 6) + 4u * static_cast<unsigned>(e);
            kr[e] = vr[e] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (j < key_end) {
                std::uint32_t local;
                const KvSegment* seg = locate_key(p.rows.segments, desc, j, local);
                const std::size_t at = static_cast<std::size_t>(local) * kv_width + static_cast<std::size_t>(kvh) * kSplitHD + 4 * (tid & 63u);
                kr[e] = *reinterpret_cast<const float4*>(static_cast<const float*>(seg->keys) + at);
                vr[e] = *reinterpret_cast<const float4*>(static_cast<const float*>(seg->values) + at);
            }
        }
    };
    const auto stage = [&](int b) {
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            const unsigned j = (tid >> 6) + 4u * static_cast<unsigned>(e);
            *reinterpret_cast<float4*>(&Ks[b][j][4 * (tid & 63u)]) = kr[e];
            *reinterpret_cast<float4*>(&Vs[b][j][4 * (tid & 63u)]) = vr[e];
        }
    };

    fetch(key_begin);
    stage(0);
    if (tiles > 1) fetch(key_begin + kSplitKeys);
    __syncthreads();

    const unsigned b8 = (s >> 3) & 1u, b4 = (s >> 2) & 1u, b2 = (s >> 1) & 1u, b1 = s & 1u;
    for (std::uint32_t kt = 0; kt < tiles; ++kt) {
        const int b = static_cast<int>(kt & 1u);
        const std::uint32_t key0 = key_begin + kt * kSplitKeys;

        // Partial scores over this lane's 16 dimensions: one fmaf chain per (row, key) in dimension order.
        float sp[4][8];
#pragma unroll
        for (int r = 0; r < 4; ++r)
#pragma unroll
            for (int j = 0; j < 8; ++j) sp[r][j] = 0.0f;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float4 k4 = *reinterpret_cast<const float4*>(&Ks[b][j][4 * s + 64 * i]);
#pragma unroll
                for (int r = 0; r < 4; ++r) {
                    sp[r][j] = __fmaf_rn(q[r][4 * i], k4.x, sp[r][j]);
                    sp[r][j] = __fmaf_rn(q[r][4 * i + 1], k4.y, sp[r][j]);
                    sp[r][j] = __fmaf_rn(q[r][4 * i + 2], k4.z, sp[r][j]);
                    sp[r][j] = __fmaf_rn(q[r][4 * i + 3], k4.w, sp[r][j]);
                }
            }
        }

        // The next tile goes into the other buffer (read during the previous tile, which every thread has finished: the barrier
        // ending that tile), and the one after it is fetched.
        if (kt + 1 < tiles) {
            stage(b ^ 1);
            if (kt + 2 < tiles) fetch(key0 + 2 * kSplitKeys);
        }

        // Reduce-scatter over the half-warp. Mask 8 splits the rows {0,1} / {2,3}, mask 4 row parity, mask 2 keys 0-3 / 4-7,
        // mask 1 keys {0,1,4,5} / {2,3,6,7}; a lane keeps the half its mask bit selects and adds its partner's partial of it.
        float a1[2][8];
#pragma unroll
        for (int r = 0; r < 2; ++r)
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float send = b8 ? sp[r][j] : sp[r + 2][j];
                const float keep = b8 ? sp[r + 2][j] : sp[r][j];
                a1[r][j] = __fadd_rn(keep, __shfl_xor_sync(0xFFFFFFFFu, send, 8));
            }
        float a2[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float send = b4 ? a1[0][j] : a1[1][j];
            const float keep = b4 ? a1[1][j] : a1[0][j];
            a2[j] = __fadd_rn(keep, __shfl_xor_sync(0xFFFFFFFFu, send, 4));
        }
        float a3[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float send = b2 ? a2[j] : a2[j + 4];
            const float keep = b2 ? a2[j + 4] : a2[j];
            a3[j] = __fadd_rn(keep, __shfl_xor_sync(0xFFFFFFFFu, send, 2));
        }
        float a4[2];
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const float send = b1 ? a3[j] : a3[j + 2];
            const float keep = b1 ? a3[j + 2] : a3[j];
            a4[j] = __fadd_rn(keep, __shfl_xor_sync(0xFFFFFFFFu, send, 1));
        }
        // This lane now holds the full scores of its own row for keys key0 + 2 (s % 4) and key0 + 2 (s % 4) + 1.
        const std::uint32_t kj = key0 + 2 * (s & 3u);
        const float x0 = kj < own_visible ? __fmul_rn(a4[0], p.scale_log2) : neg_inf();
        const float x1 = kj + 1 < own_visible ? __fmul_rn(a4[1], p.scale_log2) : neg_inf();
        float mx = fmaxf(x0, x1);
        mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, 1));
        mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, 2));
        const float m_new = fmaxf(m_run, mx);
        const float m_use = m_new == neg_inf() ? 0.0f : m_new; // a row that has seen nothing yet: p = 2^-inf = 0
        const float alpha = m_new == m_run ? 1.0f : ex2_fast(m_run - m_use); // 0 on the first visible tile
        const float p0 = ex2_fast(x0 - m_use);
        const float p1 = ex2_fast(x1 - m_use);
        l_run = __fadd_rn(__fmul_rn(l_run, alpha), __fadd_rn(p0, p1));
        m_run = m_new;

        // Back to the 16 lanes: the rescale factors of the four rows, then (key by key) their probabilities.
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const float al = __shfl_sync(0xFFFFFFFFu, alpha, hbase + 4 * r);
            if (al != 1.0f) {
#pragma unroll
                for (int i = 0; i < 16; ++i) o[r][i] = __fmul_rn(o[r][i], al);
            }
        }
        // O += P V: one fmaf chain per output over the keys in order.
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            float pj[4];
#pragma unroll
            for (int r = 0; r < 4; ++r) pj[r] = __shfl_sync(0xFFFFFFFFu, (j & 1) ? p1 : p0, hbase + 4 * r + (j >> 1));
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float4 v4 = *reinterpret_cast<const float4*>(&Vs[b][j][4 * s + 64 * i]);
#pragma unroll
                for (int r = 0; r < 4; ++r) {
                    o[r][4 * i] = __fmaf_rn(pj[r], v4.x, o[r][4 * i]);
                    o[r][4 * i + 1] = __fmaf_rn(pj[r], v4.y, o[r][4 * i + 1]);
                    o[r][4 * i + 2] = __fmaf_rn(pj[r], v4.z, o[r][4 * i + 2]);
                    o[r][4 * i + 3] = __fmaf_rn(pj[r], v4.w, o[r][4 * i + 3]);
                }
            }
        }
        __syncthreads();
    }

    // The own row's sum: its four lanes' shares by a fixed butterfly (the same bits in the four).
    float l_row = __fadd_rn(l_run, __shfl_xor_sync(0xFFFFFFFFu, l_run, 1));
    l_row = __fadd_rn(l_row, __shfl_xor_sync(0xFFFFFFFFu, l_row, 2));
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const float l_r = __shfl_sync(0xFFFFFFFFu, l_row, hbase + 4 * r);
        const float m_r = __shfl_sync(0xFFFFFFFFu, m_run, hbase + 4 * r);
        const unsigned qrow = rbase + static_cast<unsigned>(r);
        const std::uint32_t tr = tok0 + (qrow >> lg);
        if (tr >= span.length) continue;
        const std::uint32_t hr = kvh * group + (qrow & (group - 1));
        const std::uint32_t visible = ctx + tr + 1;
        const std::uint32_t n_chunks = split_chunks_of(visible, p.chunk);
        if (chunk >= n_chunks) continue; // the row sees no key of this chunk
        const std::size_t token = static_cast<std::size_t>(span.offset) + tr;
        if (n_chunks == 1) {
            TOut* const out = static_cast<TOut*>(p.out) + token * p.out_stride + static_cast<std::size_t>(hr) * kSplitHD + 4 * s;
            const float* const gate = p.gate.data + token * p.gate.stride + static_cast<std::size_t>(hr) * 2 * kSplitHD + kSplitHD + 4 * s;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                float4 v;
                v.x = __fmul_rn(__fdiv_rn(o[r][4 * i], l_r), sigmoid_stable(gate[64 * i]));
                v.y = __fmul_rn(__fdiv_rn(o[r][4 * i + 1], l_r), sigmoid_stable(gate[64 * i + 1]));
                v.z = __fmul_rn(__fdiv_rn(o[r][4 * i + 2], l_r), sigmoid_stable(gate[64 * i + 2]));
                v.w = __fmul_rn(__fdiv_rn(o[r][4 * i + 3], l_r), sigmoid_stable(gate[64 * i + 3]));
                store4<TOut>(out + 64 * i, v);
            }
        } else {
            const std::size_t wt = static_cast<std::size_t>(tr) - static_cast<std::size_t>(p.tile0) * tpt; // token in the window
            const std::size_t slot =
                ((static_cast<std::size_t>(blockIdx.z) * p.win_tiles * tpt + wt) * p.heads + hr) * p.chunks + chunk;
            float* const dst = p.part + slot * kSplitSlot;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                *reinterpret_cast<float4*>(dst + 4 * s + 64 * i) =
                    make_float4(o[r][4 * i], o[r][4 * i + 1], o[r][4 * i + 2], o[r][4 * i + 3]);
            }
            if (s == 0) {
                dst[kSplitHD] = m_r;
                dst[kSplitHD + 1] = l_r;
            }
        }
    }
}

// The merge of the chunk partials of one (token, head): a block of 256 threads, one per dimension, each combining the partials by the
// same fixed tree (every thread computes the same maxima and sums, so they agree bit for bit), then normalising and gating as the split
// kernel does for a token that fits one chunk.
template <class TOut>
__global__ void __launch_bounds__(kSplitHD) attention_f32_merge_kernel(const SplitF32Params p) {
    const std::uint32_t row = run_row(p.rows, blockIdx.z);
    const RowSpan span = p.rows.spans[row];
    const std::uint32_t tpt = kSplitRows >> p.group_log2;
    const std::uint32_t wt = blockIdx.x;
    const std::uint32_t t = p.tile0 * tpt + wt;
    if (t >= span.length) return;
    const AttentionRowDesc desc = p.rows.descs[row];
    const std::uint32_t visible = context_tokens(p.rows.segments, desc) + t + 1;
    const std::uint32_t n_chunks = split_chunks_of(visible, p.chunk);
    if (n_chunks <= 1) return;
    const std::uint32_t head = blockIdx.y;
    const unsigned d = threadIdx.x;
    const float* src = p.part + ((static_cast<std::size_t>(blockIdx.z) * p.win_tiles * tpt + wt) * p.heads + head) * p.chunks * kSplitSlot;
    // The tree of AttnSplit.lean (`pairUp` / `treeFold`): partials 0+1, 2+3, ..., an odd last one carried, repeated. Its node at
    // level k covers the chunks [i 2^k, (i + 1) 2^k) cut at n_chunks, so a binary-counter stack builds it left to right (two
    // complete neighbours of equal size merge as soon as the right one is complete) and the incomplete blocks left at the end merge
    // right to left. merge(x, y) and merge(y, x) give the same bits (max and + commute), so only the tree's shape matters.
    // Every chunk below n_chunks holds at least one visible key, so every maximum is finite.
    constexpr int kDepth = 32;
    float sm[kDepth], sl[kDepth], sa[kDepth];
    std::uint32_t ssize[kDepth];
    int top = 0;
    const auto merge = [](float& m, float& l, float& acc, float m2, float l2, float acc2) {
        const float mn = fmaxf(m, m2);
        const float f1 = m == mn ? 1.0f : ex2_fast(m - mn);
        const float f2 = m2 == mn ? 1.0f : ex2_fast(m2 - mn);
        l = __fadd_rn(__fmul_rn(l, f1), __fmul_rn(l2, f2));
        acc = __fadd_rn(__fmul_rn(acc, f1), __fmul_rn(acc2, f2));
        m = mn;
    };
    for (std::uint32_t c = 0; c < n_chunks; ++c, src += kSplitSlot) {
        sm[top] = src[kSplitHD];
        sl[top] = src[kSplitHD + 1];
        sa[top] = src[d];
        ssize[top] = 1;
        ++top;
        while (top >= 2 && ssize[top - 1] == ssize[top - 2]) {
            merge(sm[top - 2], sl[top - 2], sa[top - 2], sm[top - 1], sl[top - 1], sa[top - 1]);
            ssize[top - 2] *= 2;
            --top;
        }
    }
    while (top >= 2) {
        merge(sm[top - 2], sl[top - 2], sa[top - 2], sm[top - 1], sl[top - 1], sa[top - 1]);
        --top;
    }
    const float l = sl[0], acc = sa[0];
    const std::size_t token = static_cast<std::size_t>(span.offset) + t;
    const float gate = p.gate.data[token * p.gate.stride + static_cast<std::size_t>(head) * 2 * kSplitHD + kSplitHD + d];
    static_cast<TOut*>(p.out)[token * p.out_stride + static_cast<std::size_t>(head) * kSplitHD + d] =
        from_float<TOut>(__fmul_rn(__fdiv_rn(acc, l), sigmoid_stable(gate)));
}

// The most keys a token of the running rows sees, for a launch whose host did not give it.
__global__ void attention_max_keys_kernel(const AttentionRows rows, std::uint32_t* result) {
    for (std::uint32_t i = threadIdx.x; i < rows.run_count; i += blockDim.x) {
        const std::uint32_t row = run_row(rows, i);
        const std::uint32_t keys = context_tokens(rows.segments, rows.descs[row]) + rows.spans[row].length;
        atomicMax(result, keys);
    }
}

// exp2 on the fast path against float64, for the test suite: out[i] = ex2_fast(in[i]).
__global__ void attention_ex2_probe_kernel(const float* in, float* out, std::uint32_t n) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = ex2_fast(in[i]);
}

// ---- the second tiled kernel (F16 keys and values, head_dim 64/128/256) ----
// The same arithmetic contract as attention_tiled_kernel (key tiles of 32 counted from the chain's first token, an
// online softmax in float32 registers in the base-2 domain, P rounded to binary16, tiles a row cannot see contribute
// exactly zero, so a token's bits depend on its own chain only), with three changes made for speed:
//   * 8 warps and 128 query rows per block: every key and value tile fetched from L2 serves twice the rows, which halves
//     the L2-to-SM traffic that bounds the first kernel at long contexts;
//   * separate key and value buffers: the values of a tile load while its scores are computed and the next tile's keys
//     load while the values are multiplied, instead of one buffer used in turn;
//   * optionally (PV16) P V on binary16 accumulators: each tile's 32-key product is formed in binary16 (two roundings,
//     P <= 1) and added to the float32 output accumulators as they are rescaled. GeForce Ampere runs binary16-accumulate
//     mma.sync at twice its float32-accumulate rate. The scores Q K^T always accumulate in float32.
// The order of the additions inside a tile differs from the first kernel's, so the bits differ from it in the last places.
constexpr unsigned kV2Threads = 256;
constexpr unsigned kV2Rows = 128;

template <int HD>
__host__ __device__ constexpr std::size_t v2_smem_bytes() {
    return (static_cast<std::size_t>(kV2Rows) + 2 * kTileKeys) * HD * sizeof(__half);
}

__device__ inline void mma_16816_f16(std::uint32_t (&c)[2], const std::uint32_t (&a)[4], std::uint32_t b0, std::uint32_t b1) {
#if SEPAL_TENSOR_CORES
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%0, %1};\n"
                 : "+r"(c[0]), "+r"(c[1])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    __trap();
#endif
}

template <int HD, class TOut, bool PV16, bool QK16 = false>
__global__ void __launch_bounds__(kV2Threads, 1) attention_v2_kernel(const TiledParams p) {
    static_assert(HD >= 64, "the second tiled kernel takes head_dim 64, 128 or 256");
    constexpr int CPR = HD / 8;
    constexpr int NT = HD / 8;
    extern __shared__ __align__(16) unsigned char v2_smem[];
    __half* const Qs = reinterpret_cast<__half*>(v2_smem);
    __half* const Ks = Qs + kV2Rows * HD;
    __half* const Vs = Ks + kTileKeys * HD;

    const std::uint32_t row = run_row(p.rows, blockIdx.z);
    const RowSpan span = p.rows.spans[row];
    const std::uint32_t len = span.length;
    const std::uint32_t lg = p.group_log2;
    const std::uint32_t group = 1u << lg;
    const std::uint32_t tokens_per_tile = kV2Rows >> lg;
    const std::uint32_t tiles = (len + tokens_per_tile - 1) / tokens_per_tile;
    if (blockIdx.x >= tiles) return;
    const std::uint32_t tile = tiles - 1 - blockIdx.x; // the heaviest query tile of a row first
    const std::uint32_t kv_head = blockIdx.y;
    const AttentionRowDesc desc = p.rows.descs[row];
    const std::uint32_t kv_width = p.kv_heads * HD;
    const std::uint32_t context = context_tokens(p.rows.segments, desc);
    const std::uint32_t total = context + len;
    const std::uint32_t t0 = tile * tokens_per_tile;
    const std::uint32_t last_visible = context + min(len, t0 + tokens_per_tile);
    const std::uint32_t key_tiles = (last_visible + kTileKeys - 1) / kTileKeys;

    const unsigned tid = threadIdx.x;
    const unsigned warp = tid / 32;
    const unsigned lane = tid % 32;
    const unsigned g = lane >> 2;
    const unsigned qd = lane & 3;

    for (unsigned c = tid; c < kV2Rows * CPR; c += kV2Threads) {
        const unsigned r = c / CPR;
        const unsigned ch = c % CPR;
        const std::uint32_t tl = t0 + (r >> lg);
        __half* dst = Qs + tile_offset<HD>(r, ch);
        if (tl < len) {
            const std::size_t head = static_cast<std::size_t>(kv_head) * group + (r & (group - 1));
            cp_async16(dst, p.q + (static_cast<std::size_t>(span.offset) + tl) * p.q_stride + head * HD + ch * 8);
        } else {
            zero16(dst);
        }
    }
    load_kv_tile<HD, kV2Threads>(Ks, p.rows.segments, desc, total, 0, kv_head, kv_width, false);
    cp_async_commit(); // the queries and the first keys
    load_kv_tile<HD, kV2Threads>(Vs, p.rows.segments, desc, total, 0, kv_head, kv_width, true);
    cp_async_commit(); // the first values

    const unsigned row_a = warp * 16 + g;
    const unsigned row_b = row_a + 8;
    const std::uint32_t token_a = min(t0 + (row_a >> lg), len - 1);
    const std::uint32_t token_b = min(t0 + (row_b >> lg), len - 1);
    const std::uint32_t visible_a = context + token_a + 1;
    const std::uint32_t visible_b = context + token_b + 1;

    float o[NT][4];
#pragma unroll
    for (int i = 0; i < NT; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
    float max_a = neg_inf(), max_b = neg_inf();
    float sum_a = 0.0f, sum_b = 0.0f; // this lane's share of the row sums

    for (std::uint32_t kt = 0; kt < key_tiles; ++kt) {
        // In flight: the keys of tile kt, then its values (one commit group each, in that order). Wait for the keys only.
        cp_async_wait<1>();
        __syncthreads();

        float s[4][4];
#pragma unroll
        for (int i = 0; i < 4; ++i) s[i][0] = s[i][1] = s[i][2] = s[i][3] = 0.0f;
#pragma unroll
        for (int kg = 0; kg < HD / 64; ++kg) {
            // QK16: binary16 accumulators over 64 dimensions at a time, then added into the float32 scores.
            std::uint32_t h[4][2] = {{0u, 0u}, {0u, 0u}, {0u, 0u}, {0u, 0u}};
#pragma unroll
            for (int ki = 0; ki < 4; ++ki) {
                const int ks = kg * 4 + ki;
                std::uint32_t a[4];
                ldmatrix_x4(a, Qs + tile_offset<HD>(warp * 16 + (lane & 7) + ((lane >> 3) & 1) * 8, ks * 2 + (lane >> 4)));
#pragma unroll
                for (int np = 0; np < 2; ++np) {
                    std::uint32_t b[4];
                    ldmatrix_x4(b, Ks + tile_offset<HD>(np * 16 + (lane >> 4) * 8 + (lane & 7), ks * 2 + ((lane >> 3) & 1)));
                    if constexpr (QK16) {
                        mma_16816_f16(h[2 * np], a, b[0], b[1]);
                        mma_16816_f16(h[2 * np + 1], a, b[2], b[3]);
                    } else {
                        mma_16816(s[2 * np], a, b[0], b[1]);
                        mma_16816(s[2 * np + 1], a, b[2], b[3]);
                    }
                }
            }
            if constexpr (QK16) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const float2 lo = __half22float2(*reinterpret_cast<const __half2*>(&h[i][0]));
                    const float2 hi = __half22float2(*reinterpret_cast<const __half2*>(&h[i][1]));
                    s[i][0] += lo.x;
                    s[i][1] += lo.y;
                    s[i][2] += hi.x;
                    s[i][3] += hi.y;
                }
            }
        }

        __syncthreads(); // every warp has read the keys: the buffer takes the next tile's keys
        if (kt + 1 < key_tiles) load_kv_tile<HD, kV2Threads>(Ks, p.rows.segments, desc, total, kt + 1, kv_head, kv_width, false);
        cp_async_commit(); // committed even when empty, so that the group count stays fixed

        float x[4][4];
        float tile_max_a = neg_inf(), tile_max_b = neg_inf();
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const std::uint32_t j = kt * kTileKeys + nt * 8 + qd * 2 + (e & 1);
                const std::uint32_t visible = (e >> 1) ? visible_b : visible_a;
                x[nt][e] = j < visible ? s[nt][e] * p.scale_log2 : neg_inf();
            }
            tile_max_a = fmaxf(tile_max_a, fmaxf(x[nt][0], x[nt][1]));
            tile_max_b = fmaxf(tile_max_b, fmaxf(x[nt][2], x[nt][3]));
        }
        tile_max_a = fmaxf(tile_max_a, __shfl_xor_sync(0xFFFFFFFFu, tile_max_a, 1));
        tile_max_a = fmaxf(tile_max_a, __shfl_xor_sync(0xFFFFFFFFu, tile_max_a, 2));
        tile_max_b = fmaxf(tile_max_b, __shfl_xor_sync(0xFFFFFFFFu, tile_max_b, 1));
        tile_max_b = fmaxf(tile_max_b, __shfl_xor_sync(0xFFFFFFFFu, tile_max_b, 2));
        const float new_max_a = fmaxf(max_a, tile_max_a);
        const float new_max_b = fmaxf(max_b, tile_max_b);
        const float alpha_a = exp2f(max_a - new_max_a); // 0 on the first tile, 1 when the maximum did not move
        const float alpha_b = exp2f(max_b - new_max_b);
        max_a = new_max_a;
        max_b = new_max_b;

        float tile_sum_a = 0.0f, tile_sum_b = 0.0f;
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            x[nt][0] = exp2f(x[nt][0] - new_max_a);
            x[nt][1] = exp2f(x[nt][1] - new_max_a);
            x[nt][2] = exp2f(x[nt][2] - new_max_b);
            x[nt][3] = exp2f(x[nt][3] - new_max_b);
            tile_sum_a += x[nt][0] + x[nt][1];
            tile_sum_b += x[nt][2] + x[nt][3];
        }
        sum_a = sum_a * alpha_a + tile_sum_a;
        sum_b = sum_b * alpha_b + tile_sum_b;

        // The 16 x 32 probabilities of this warp as A operands, straight from the score accumulators.
        std::uint32_t pa[2][4];
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            pa[kk][0] = pack_half2(x[2 * kk][0], x[2 * kk][1]);
            pa[kk][1] = pack_half2(x[2 * kk][2], x[2 * kk][3]);
            pa[kk][2] = pack_half2(x[2 * kk + 1][0], x[2 * kk + 1][1]);
            pa[kk][3] = pack_half2(x[2 * kk + 1][2], x[2 * kk + 1][3]);
        }

        cp_async_wait<1>(); // this tile's values (the next tile's keys may still be in flight)
        __syncthreads();

        if constexpr (PV16) {
#pragma unroll
            for (int dp = 0; dp < HD / 16; ++dp) {
                std::uint32_t c0[2] = {0u, 0u}, c1[2] = {0u, 0u};
#pragma unroll
                for (int kk = 0; kk < 2; ++kk) {
                    std::uint32_t b[4];
                    ldmatrix_x4_trans(b, Vs + tile_offset<HD>(kk * 16 + ((lane >> 3) & 1) * 8 + (lane & 7), dp * 2 + (lane >> 4)));
                    mma_16816_f16(c0, pa[kk], b[0], b[1]);
                    mma_16816_f16(c1, pa[kk], b[2], b[3]);
                }
                const float2 c0a = __half22float2(*reinterpret_cast<const __half2*>(&c0[0]));
                const float2 c0b = __half22float2(*reinterpret_cast<const __half2*>(&c0[1]));
                const float2 c1a = __half22float2(*reinterpret_cast<const __half2*>(&c1[0]));
                const float2 c1b = __half22float2(*reinterpret_cast<const __half2*>(&c1[1]));
                o[2 * dp][0] = fmaf(o[2 * dp][0], alpha_a, c0a.x);
                o[2 * dp][1] = fmaf(o[2 * dp][1], alpha_a, c0a.y);
                o[2 * dp][2] = fmaf(o[2 * dp][2], alpha_b, c0b.x);
                o[2 * dp][3] = fmaf(o[2 * dp][3], alpha_b, c0b.y);
                o[2 * dp + 1][0] = fmaf(o[2 * dp + 1][0], alpha_a, c1a.x);
                o[2 * dp + 1][1] = fmaf(o[2 * dp + 1][1], alpha_a, c1a.y);
                o[2 * dp + 1][2] = fmaf(o[2 * dp + 1][2], alpha_b, c1b.x);
                o[2 * dp + 1][3] = fmaf(o[2 * dp + 1][3], alpha_b, c1b.y);
            }
        } else {
#pragma unroll
            for (int i = 0; i < NT; ++i) {
                o[i][0] *= alpha_a;
                o[i][1] *= alpha_a;
                o[i][2] *= alpha_b;
                o[i][3] *= alpha_b;
            }
#pragma unroll
            for (int kk = 0; kk < 2; ++kk) {
#pragma unroll
                for (int dp = 0; dp < HD / 16; ++dp) {
                    std::uint32_t b[4];
                    ldmatrix_x4_trans(b, Vs + tile_offset<HD>(kk * 16 + ((lane >> 3) & 1) * 8 + (lane & 7), dp * 2 + (lane >> 4)));
                    mma_16816(o[2 * dp], pa[kk], b[0], b[1]);
                    mma_16816(o[2 * dp + 1], pa[kk], b[2], b[3]);
                }
            }
        }

        __syncthreads(); // every warp has read the values: the buffer takes the next tile's values
        if (kt + 1 < key_tiles) load_kv_tile<HD, kV2Threads>(Vs, p.rows.segments, desc, total, kt + 1, kv_head, kv_width, true);
        cp_async_commit();
    }
    cp_async_wait<0>();

    // The row sums: add the four lanes of a quad in a fixed order.
    sum_a += __shfl_xor_sync(0xFFFFFFFFu, sum_a, 1);
    sum_a += __shfl_xor_sync(0xFFFFFFFFu, sum_a, 2);
    sum_b += __shfl_xor_sync(0xFFFFFFFFu, sum_b, 1);
    sum_b += __shfl_xor_sync(0xFFFFFFFFu, sum_b, 2);

#pragma unroll
    for (int half_row = 0; half_row < 2; ++half_row) {
        const unsigned r = half_row == 0 ? row_a : row_b;
        const std::uint32_t tl = t0 + (r >> lg);
        if (tl >= len) continue;
        const float denominator = half_row == 0 ? sum_a : sum_b;
        const std::size_t head = static_cast<std::size_t>(kv_head) * group + (r & (group - 1));
        const std::size_t token = static_cast<std::size_t>(span.offset) + tl;
        TOut* const out = static_cast<TOut*>(p.out) + token * p.out_stride + head * HD;
        const float* const gate = p.gate.data + token * p.gate.stride + head * 2 * HD + HD;
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int d = nt * 8 + static_cast<int>(qd) * 2 + e;
                out[d] = from_float<TOut>(__fmul_rn(o[nt][half_row * 2 + e] / denominator, sigmoid_stable(gate[d])));
            }
        }
    }
}

// ---- host side ----

namespace {

void validate_shape(const AttentionShape& shape) {
    if (shape.heads == 0 || shape.kv_heads == 0 || shape.heads % shape.kv_heads != 0) {
        throw std::invalid_argument("attention: heads must be a positive multiple of kv_heads");
    }
    if (shape.head_dim == 0 || shape.head_dim > kMaxHeadDim) {
        throw std::invalid_argument("attention: head_dim must be in 1 .. " + std::to_string(kMaxHeadDim));
    }
    if (shape.rotary_dims % 2 != 0 || shape.rotary_dims > shape.head_dim || shape.rotary_dims / 2 > kMaxRotaryPairs) {
        throw std::invalid_argument("attention: rotary_dims must be even and at most head_dim");
    }
}

void validate_rows(const AttentionRows& rows) {
    if (rows.run_count > 65535u) throw std::invalid_argument("attention: more than 65535 rows in one launch");
}

bool log2_group(const AttentionShape& shape, std::uint32_t& lg) {
    const std::uint32_t group = shape.heads / shape.kv_heads;
    if (group == 0 || (group & (group - 1)) != 0 || group > 16) return false;
    lg = 0;
    while ((1u << lg) < group) ++lg;
    return true;
}

template <int HD, class TOut>
void launch_tiled_instance(const TiledParams& params, unsigned max_tiles, const AttentionShape& shape,
                           std::uint32_t run_count, cudaStream_t stream) {
    constexpr std::size_t smem = tiled_smem_bytes<HD>();
    static const cudaError_t configured = cudaFuncSetAttribute(
        attention_tiled_kernel<HD, TOut>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
    check(configured, "cudaFuncSetAttribute");
    const dim3 grid(max_tiles, shape.kv_heads, run_count);
    attention_tiled_kernel<HD, TOut><<<grid, dim3(kTiledThreads), smem, stream>>>(params);
}

template <int HD, class TOut, bool PV16, bool QK16>
void launch_v2_instance(const TiledParams& params, const AttentionRows& rows, const AttentionShape& shape, std::uint32_t lg,
                        cudaStream_t stream) {
    constexpr std::size_t smem = v2_smem_bytes<HD>();
    static const cudaError_t configured = cudaFuncSetAttribute(
        attention_v2_kernel<HD, TOut, PV16, QK16>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
    check(configured, "cudaFuncSetAttribute");
    const unsigned tokens_per_tile = kV2Rows >> lg;
    const unsigned max_tiles = (rows.max_row_tokens + tokens_per_tile - 1) / tokens_per_tile;
    const dim3 grid(max_tiles, shape.kv_heads, rows.run_count);
    attention_v2_kernel<HD, TOut, PV16, QK16><<<grid, dim3(kV2Threads), smem, stream>>>(params);
}

template <class TOut, bool PV16, bool QK16 = false>
void dispatch_v2(const TiledParams& params, const AttentionRows& rows, const AttentionShape& shape, std::uint32_t lg,
                 cudaStream_t stream) {
    switch (shape.head_dim) {
    case 64: launch_v2_instance<64, TOut, PV16, QK16>(params, rows, shape, lg, stream); break;
    case 128: launch_v2_instance<128, TOut, PV16, QK16>(params, rows, shape, lg, stream); break;
    default: launch_v2_instance<256, TOut, PV16, QK16>(params, rows, shape, lg, stream); break;
    }
}

template <class TOut>
void dispatch_tiled(const TiledParams& params, unsigned max_tiles, const AttentionShape& shape,
                    std::uint32_t run_count, cudaStream_t stream) {
    switch (shape.head_dim) {
    case 16: launch_tiled_instance<16, TOut>(params, max_tiles, shape, run_count, stream); break;
    case 64: launch_tiled_instance<64, TOut>(params, max_tiles, shape, run_count, stream); break;
    case 128: launch_tiled_instance<128, TOut>(params, max_tiles, shape, run_count, stream); break;
    default: launch_tiled_instance<256, TOut>(params, max_tiles, shape, run_count, stream); break;
    }
}

template <int HD, class TOut>
void launch_f32_tiled_instance(const TiledF32Params& params, unsigned max_tiles, const AttentionShape& shape, std::uint32_t run_count,
                               cudaStream_t stream) {
    const dim3 grid(max_tiles, shape.kv_heads, run_count);
    attention_f32_tiled_kernel<HD, TOut><<<grid, dim3(f32_threads<HD>()), 0, stream>>>(params);
}

template <class TOut>
void dispatch_f32_tiled(const TiledF32Params& params, unsigned max_tiles, const AttentionShape& shape, std::uint32_t run_count,
                        cudaStream_t stream) {
    switch (shape.head_dim) {
    case 16: launch_f32_tiled_instance<16, TOut>(params, max_tiles, shape, run_count, stream); break;
    case 32: launch_f32_tiled_instance<32, TOut>(params, max_tiles, shape, run_count, stream); break;
    case 64: launch_f32_tiled_instance<64, TOut>(params, max_tiles, shape, run_count, stream); break;
    case 128: launch_f32_tiled_instance<128, TOut>(params, max_tiles, shape, run_count, stream); break;
    default: launch_f32_tiled_instance<256, TOut>(params, max_tiles, shape, run_count, stream); break;
    }
}

template <class TKV, class TOut>
void launch_reference_instance(const ReferenceParams& params, const AttentionShape& shape, std::uint32_t run_count,
                               cudaStream_t stream) {
    const dim3 grid(params.rows.max_row_tokens, run_count, shape.heads);
    const std::size_t smem = (static_cast<std::size_t>(shape.head_dim) + kRefChunk) * sizeof(float);
    attention_reference_kernel<TKV, TOut><<<grid, dim3(kRefThreads), smem, stream>>>(params);
}

} // namespace

bool attention_f32_tiled_supported(const AttentionShape& shape, AttnElem kv_type) {
    std::uint32_t lg = 0;
    if (kv_type != AttnElem::F32 || shape.kv_heads == 0 || shape.heads % shape.kv_heads != 0) return false;
    if (!log2_group(shape, lg)) return false;
    return shape.head_dim == 16 || shape.head_dim == 32 || shape.head_dim == 64 || shape.head_dim == 128 || shape.head_dim == 256;
}

namespace {
std::atomic<int> g_tiled_override{-1};
} // namespace

void set_tiled_kernel(TiledKernel kernel) { g_tiled_override.store(static_cast<int>(kernel)); }
void reset_tiled_kernel() { g_tiled_override.store(-1); }

TiledKernel tiled_kernel_choice() {
    if (const int o = g_tiled_override.load(); o >= 0) return static_cast<TiledKernel>(o);
    // SEPAL_ATTENTION_KERNEL=v1 | v2 | v2h | v2hh; read once per process. Unset: v2, which gives v1's bits faster. v2h and v2hh
    // are faster still but miss the gpu-f16 parity gate on SEC-dev (max |dp| 0.057 and 0.064 against 0.03): opt-in.
    static const TiledKernel choice = [] {
        const char* v = std::getenv("SEPAL_ATTENTION_KERNEL");
        if (v == nullptr || *v == '\0') return TiledKernel::Second; // v1's bits, faster (README.md, "Ampere")
        const std::string s(v);
        if (s == "v1") return TiledKernel::First;
        if (s == "v2") return TiledKernel::Second;
        if (s == "v2h") return TiledKernel::SecondPv16;
        if (s == "v2hh") return TiledKernel::SecondFp16;
        throw std::invalid_argument("SEPAL_ATTENTION_KERNEL must be v1, v2, v2h or v2hh, not " + s);
    }();
    return choice;
}

namespace {
std::atomic<int> g_f32_override{-1};
std::atomic<std::uint32_t> g_split_chunk{0};
std::atomic<std::size_t> g_split_cap{0};
constexpr std::size_t kSplitDefaultCap = std::size_t{32} << 20;

// The split kernel's device workspace: the chunk partials of one window (and four bytes for the read-back of max_row_keys).
struct SplitWorkspace {
    std::mutex mutex;
    float* data = nullptr;
    std::size_t bytes = 0;
    float* ensure(std::size_t need) {
        if (need > bytes) {
            if (data != nullptr) check(cudaFree(data), "cudaFree (attention split workspace)");
            data = nullptr;
            bytes = 0;
            check(cudaMalloc(&data, need), "cudaMalloc (attention split workspace)");
            bytes = need;
        }
        return data;
    }
    void release() {
        if (data != nullptr) cudaFree(data);
        data = nullptr;
        bytes = 0;
    }
};
SplitWorkspace& split_workspace() {
    static SplitWorkspace w;
    return w;
}
} // namespace

void set_f32_kernel(F32Kernel kernel) { g_f32_override.store(static_cast<int>(kernel)); }
void reset_f32_kernel() { g_f32_override.store(-1); }
void set_f32_split_limits(std::uint32_t chunk_keys, std::size_t workspace_cap_bytes) {
    if (chunk_keys % kSplitKeys != 0) throw std::invalid_argument("set_f32_split_limits: the chunk must be a multiple of 8 keys");
    g_split_chunk.store(chunk_keys);
    g_split_cap.store(workspace_cap_bytes);
}
void release_f32_split_workspace() {
    SplitWorkspace& w = split_workspace();
    std::lock_guard<std::mutex> lock(w.mutex);
    w.release();
}

F32Kernel f32_kernel_choice() {
    if (const int o = g_f32_override.load(); o >= 0) return static_cast<F32Kernel>(o);
    // SEPAL_ATTENTION_F32_KERNEL=v1 | v2; read once per process. Unset: v2.
    static const F32Kernel choice = [] {
        const char* v = std::getenv("SEPAL_ATTENTION_F32_KERNEL");
        if (v == nullptr || *v == '\0') return F32Kernel::Split;
        const std::string s(v);
        if (s == "v1") return F32Kernel::First;
        if (s == "v2") return F32Kernel::Split;
        throw std::invalid_argument("SEPAL_ATTENTION_F32_KERNEL must be v1 or v2, not " + s);
    }();
    return choice;
}

namespace {

template <class TOut>
void launch_f32_split(SplitF32Params p, const AttentionShape& shape, std::uint32_t lg, cudaStream_t stream) {
    const AttentionRows& rows = p.rows;
    SplitWorkspace& ws = split_workspace();
    std::lock_guard<std::mutex> lock(ws.mutex);
    std::uint32_t max_keys = rows.max_row_keys;
    if (max_keys == 0) {
        auto* d = reinterpret_cast<std::uint32_t*>(ws.ensure(sizeof(std::uint32_t)));
        check(cudaMemsetAsync(d, 0, sizeof(std::uint32_t), stream), "cudaMemsetAsync");
        attention_max_keys_kernel<<<1, 256, 0, stream>>>(rows, d);
        check(cudaGetLastError(), "attention_max_keys_kernel launch");
        check(cudaMemcpyAsync(&max_keys, d, sizeof max_keys, cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync");
        check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    }
    const std::uint32_t chunk = g_split_chunk.load() != 0 ? g_split_chunk.load() : kSplitChunk;
    const std::uint32_t tpt = kSplitRows >> lg;
    const std::uint32_t tiles = (rows.max_row_tokens + tpt - 1) / tpt;
    p.chunk = chunk;
    p.chunks = std::max<std::uint32_t>(1, (max_keys + chunk - 1) / chunk);
    if (p.chunks == 1) { // every token fits one chunk: one launch, no partials
        p.part = nullptr;
        p.tile0 = 0;
        p.win_tiles = tiles;
        attention_f32_split_kernel<TOut><<<dim3(tiles, shape.kv_heads, rows.run_count), dim3(kSplitThreads), 0, stream>>>(p);
        check(cudaGetLastError(), "attention_f32_split_kernel launch");
        return;
    }
    // Windows of query tiles whose partials fit the cap (at least one tile). The window changes where a partial is stored, not a bit.
    const std::size_t cap = g_split_cap.load() != 0 ? g_split_cap.load() : kSplitDefaultCap;
    const std::size_t per_tile = static_cast<std::size_t>(rows.run_count) * tpt * shape.heads * p.chunks * kSplitSlot * sizeof(float);
    const std::uint32_t win = static_cast<std::uint32_t>(std::clamp<std::size_t>(cap / per_tile, 1, tiles));
    p.part = ws.ensure(per_tile * win);
    p.win_tiles = win;
    for (std::uint32_t tile0 = 0; tile0 < tiles; tile0 += win) {
        p.tile0 = tile0;
        const std::uint32_t n = std::min(win, tiles - tile0);
        attention_f32_split_kernel<TOut><<<dim3(n * p.chunks, shape.kv_heads, rows.run_count), dim3(kSplitThreads), 0, stream>>>(p);
        check(cudaGetLastError(), "attention_f32_split_kernel launch");
        attention_f32_merge_kernel<TOut><<<dim3(n * tpt, shape.heads, rows.run_count), dim3(kSplitHD), 0, stream>>>(p);
        check(cudaGetLastError(), "attention_f32_merge_kernel launch");
    }
}

} // namespace

void attention_ex2_probe(const float* in, float* out, std::uint32_t n, cudaStream_t stream) {
    if (n == 0) return;
    attention_ex2_probe_kernel<<<(n + 255) / 256, 256, 0, stream>>>(in, out, n);
    check(cudaGetLastError(), "attention_ex2_probe_kernel launch");
}

bool attention_tiled_supported(const AttentionShape& shape, AttnElem kv_type) {
    std::uint32_t lg = 0;
    if (kv_type != AttnElem::F16 || shape.kv_heads == 0 || shape.heads % shape.kv_heads != 0) return false;
    if (!log2_group(shape, lg)) return false;
    return shape.head_dim == 16 || shape.head_dim == 64 || shape.head_dim == 128 || shape.head_dim == 256;
}

void launch_attention_prepare(const AttentionShape& shape, const AttentionRows& rows, AttnElem kv_type, FloatBand q_gate,
                              FloatBand k, FloatBand v, const float* q_norm, const float* k_norm, ElemMatrix q_out,
                              cudaStream_t stream, PreparePart part) {
    validate_shape(shape);
    validate_rows(rows);
    if (rows.run_count == 0 || rows.max_row_tokens == 0) return;

    PrepareParams p{};
    p.rows = rows;
    p.heads = shape.heads;
    p.kv_heads = shape.kv_heads;
    p.head_dim = shape.head_dim;
    p.pairs = shape.rotary_dims / 2;
    p.eps = static_cast<float>(shape.rms_norm_eps);
    p.q_gate = q_gate;
    p.k = k;
    p.v = v;
    p.q_norm = q_norm;
    p.k_norm = k_norm;
    p.q_out = q_out.data;
    p.q_out_stride = q_out.stride;
    p.do_queries = part != PreparePart::KeysValuesOnly;
    p.do_keys_values = part != PreparePart::QueriesOnly;
    // The reference's own expression, evaluated in float64 on the host: theta ** (-2i / rotary_dims).
    const double rotary = static_cast<double>(2 * p.pairs);
    for (std::uint32_t i = 0; i < p.pairs; ++i) {
        p.inverse_frequency[i] = std::pow(shape.rope_theta, -static_cast<double>(2 * i) / rotary);
    }

    const dim3 grid(rows.max_row_tokens, rows.run_count);
    if (kv_type == AttnElem::F16) {
        attention_prepare_kernel<__half><<<grid, dim3(kPrepareThreads), 0, stream>>>(p);
    } else {
        attention_prepare_kernel<float><<<grid, dim3(kPrepareThreads), 0, stream>>>(p);
    }
    check(cudaGetLastError(), "attention_prepare_kernel launch");
}

AttentionPath launch_attention(const AttentionShape& shape, const AttentionRows& rows, AttnElem kv_type,
                               AttentionPath path, ConstElemMatrix q, FloatBand gate, ElemMatrix out,
                               AttnElem out_type, cudaStream_t stream) {
    validate_shape(shape);
    validate_rows(rows);
    const bool tiled_ok = attention_tiled_supported(shape, kv_type) && q.stride % 8 == 0 && device_has_tensor_cores();
    const bool f32_ok = attention_f32_tiled_supported(shape, kv_type) && q.stride % 4 == 0 &&
                        reinterpret_cast<std::uintptr_t>(q.data) % 16 == 0;
    if (path == AttentionPath::Tiled && !tiled_ok) {
        throw std::invalid_argument(
            "attention: the tiled path needs F16 keys and values, head_dim 16/64/128/256, a group size of 1, 2, 4, 8 "
            "or 16, a q row stride that is a multiple of 8 and a card with tensor cores (compute capability 8.0 or later)");
    }
    if (path == AttentionPath::TiledF32 && !f32_ok) {
        throw std::invalid_argument(
            "attention: the float32 tiled path needs F32 keys and values, head_dim 16/32/64/128/256, a group size of 1, 2, 4, 8 "
            "or 16 and a 16-byte aligned q with a row stride that is a multiple of 4");
    }
    AttentionPath used = AttentionPath::Reference;
    if (path == AttentionPath::Tiled || (path == AttentionPath::Auto && tiled_ok)) used = AttentionPath::Tiled;
    else if (path == AttentionPath::TiledF32 || (path == AttentionPath::Auto && f32_ok)) used = AttentionPath::TiledF32;
    if (rows.run_count == 0 || rows.max_row_tokens == 0) return used;

    if (used == AttentionPath::TiledF32) {
        std::uint32_t lg = 0;
        log2_group(shape, lg);
        TiledF32Params p{};
        p.rows = rows;
        p.kv_heads = shape.kv_heads;
        p.group_log2 = lg;
        p.scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(shape.head_dim)));
        p.q = static_cast<const float*>(q.data);
        p.q_stride = q.stride;
        p.gate = gate;
        p.out = out.data;
        p.out_stride = out.stride;
        if (shape.head_dim == static_cast<std::uint32_t>(kSplitHD) && f32_kernel_choice() == F32Kernel::Split) {
            SplitF32Params sp{};
            sp.rows = rows;
            sp.heads = shape.heads;
            sp.kv_heads = shape.kv_heads;
            sp.group_log2 = lg;
            sp.scale_log2 = static_cast<float>(1.4426950408889634 / std::sqrt(static_cast<double>(shape.head_dim)));
            sp.q = p.q;
            sp.q_stride = q.stride;
            sp.gate = gate;
            sp.out = out.data;
            sp.out_stride = out.stride;
            if (out_type == AttnElem::F16) launch_f32_split<__half>(sp, shape, lg, stream);
            else launch_f32_split<float>(sp, shape, lg, stream);
            return used;
        }
        const unsigned tokens_per_tile = kF32Rows >> lg;
        const unsigned max_tiles = (rows.max_row_tokens + tokens_per_tile - 1) / tokens_per_tile;
        if (out_type == AttnElem::F16) dispatch_f32_tiled<__half>(p, max_tiles, shape, rows.run_count, stream);
        else dispatch_f32_tiled<float>(p, max_tiles, shape, rows.run_count, stream);
        check(cudaGetLastError(), "attention_f32_tiled_kernel launch");
        return used;
    }

    if (used == AttentionPath::Tiled) {
        std::uint32_t lg = 0;
        log2_group(shape, lg);
        TiledParams p{};
        p.rows = rows;
        p.kv_heads = shape.kv_heads;
        p.group_log2 = lg;
        p.scale_log2 = static_cast<float>(1.0 / std::sqrt(static_cast<double>(shape.head_dim))) * 1.44269504088896341f;
        p.q = static_cast<const __half*>(q.data);
        p.q_stride = q.stride;
        p.gate = gate;
        p.out = out.data;
        p.out_stride = out.stride;
        const TiledKernel kernel = tiled_kernel_choice();
        if (kernel != TiledKernel::First && shape.head_dim >= 64) {
            const bool pv16 = kernel == TiledKernel::SecondPv16;
            if (kernel == TiledKernel::SecondFp16) {
                if (out_type == AttnElem::F16) dispatch_v2<__half, true, true>(p, rows, shape, lg, stream);
                else dispatch_v2<float, true, true>(p, rows, shape, lg, stream);
            } else if (out_type == AttnElem::F16) {
                if (pv16) dispatch_v2<__half, true>(p, rows, shape, lg, stream);
                else dispatch_v2<__half, false>(p, rows, shape, lg, stream);
            } else {
                if (pv16) dispatch_v2<float, true>(p, rows, shape, lg, stream);
                else dispatch_v2<float, false>(p, rows, shape, lg, stream);
            }
            check(cudaGetLastError(), "attention_v2_kernel launch");
            return used;
        }
        const unsigned tokens_per_tile = kTileRows >> lg;
        const unsigned max_tiles = (rows.max_row_tokens + tokens_per_tile - 1) / tokens_per_tile;
        if (out_type == AttnElem::F16) {
            dispatch_tiled<__half>(p, max_tiles, shape, rows.run_count, stream);
        } else {
            dispatch_tiled<float>(p, max_tiles, shape, rows.run_count, stream);
        }
        check(cudaGetLastError(), "attention_tiled_kernel launch");
        return used;
    }

    ReferenceParams p{};
    p.rows = rows;
    p.heads = shape.heads;
    p.kv_heads = shape.kv_heads;
    p.head_dim = shape.head_dim;
    p.scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(shape.head_dim)));
    p.q = q.data;
    p.q_stride = q.stride;
    p.gate = gate;
    p.out = out.data;
    p.out_stride = out.stride;
    if (kv_type == AttnElem::F16) {
        if (out_type == AttnElem::F16) launch_reference_instance<__half, __half>(p, shape, rows.run_count, stream);
        else launch_reference_instance<__half, float>(p, shape, rows.run_count, stream);
    } else {
        if (out_type == AttnElem::F16) launch_reference_instance<float, __half>(p, shape, rows.run_count, stream);
        else launch_reference_instance<float, float>(p, shape, rows.run_count, stream);
    }
    check(cudaGetLastError(), "attention_reference_kernel launch");
    return used;
}

void check_attention_rows(std::span<const RowSpan> spans, std::span<const AttentionRowDesc> descs,
                          std::span<const KvSegment> segments) {
    if (spans.size() != descs.size()) throw std::invalid_argument("check_attention_rows: spans and descs differ in size");
    for (std::size_t r = 0; r < spans.size(); ++r) {
        const std::string row = "check_attention_rows: row " + std::to_string(r) + ": ";
        const AttentionRowDesc& d = descs[r];
        if (d.segment_count == 0) throw std::invalid_argument(row + "no segments (the row's own is required)");
        if (static_cast<std::size_t>(d.first_segment) + d.segment_count > segments.size()) {
            throw std::invalid_argument(row + "segments lie outside the segment array");
        }
        const KvSegment& own = segments[d.first_segment + d.segment_count - 1];
        if (own.tokens != spans[r].length) {
            throw std::invalid_argument(row + "the last segment holds " + std::to_string(own.tokens) +
                                        " tokens but the row has " + std::to_string(spans[r].length));
        }
        std::uint64_t total = 0;
        for (std::uint32_t k = 0; k < d.segment_count; ++k) {
            const KvSegment& s = segments[d.first_segment + k];
            if (s.tokens != 0 && (s.keys == nullptr || s.values == nullptr)) {
                throw std::invalid_argument(row + "a segment with tokens has a null keys or values pointer");
            }
            if (d.indexed != 0 && s.before != total) {
                throw std::invalid_argument(row + "an indexed row's segment " + std::to_string(k) + " says " + std::to_string(s.before) +
                                            " tokens before it but " + std::to_string(total) + " precede it");
            }
            total += s.tokens;
        }
        if (static_cast<std::uint64_t>(d.start_position) + total > 0xFFFFFFFFull) {
            throw std::invalid_argument(row + "positions overflow 32 bits");
        }
    }
}

} // namespace sepal::cuda
