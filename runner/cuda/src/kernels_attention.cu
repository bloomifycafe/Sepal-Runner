// The full-attention mixer's non-projection work on the device; see kernels_attention.hpp for the contract.
#include "kernels_attention.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
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
    const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(shared));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(address), "l"(global) : "memory");
}
__device__ inline void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N>
__device__ inline void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
}
__device__ inline void zero16(void* shared) { *static_cast<uint4*>(shared) = make_uint4(0u, 0u, 0u, 0u); }

__device__ inline void ldmatrix_x4(std::uint32_t (&r)[4], const void* shared) {
    const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(shared));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(address));
}
__device__ inline void ldmatrix_x4_trans(std::uint32_t (&r)[4], const void* shared) {
    const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(shared));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(address));
}
__device__ inline void mma_16816(float (&c)[4], const std::uint32_t (&a)[4], std::uint32_t b0, std::uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
                 "{%0, %1, %2, %3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
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
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%0, %1};\n"
                 : "+r"(c[0]), "+r"(c[1])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
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

template <class TKV, class TOut>
void launch_reference_instance(const ReferenceParams& params, const AttentionShape& shape, std::uint32_t run_count,
                               cudaStream_t stream) {
    const dim3 grid(params.rows.max_row_tokens, run_count, shape.heads);
    const std::size_t smem = (static_cast<std::size_t>(shape.head_dim) + kRefChunk) * sizeof(float);
    attention_reference_kernel<TKV, TOut><<<grid, dim3(kRefThreads), smem, stream>>>(params);
}

} // namespace

TiledKernel tiled_kernel_choice() {
    // SEPAL_ATTENTION_KERNEL=v1 | v2 | v2h; read once per process. Unset: the default below.
    static const TiledKernel choice = [] {
        const char* v = std::getenv("SEPAL_ATTENTION_KERNEL");
        if (v == nullptr || *v == '\0') return TiledKernel::SecondPv16;
        const std::string s(v);
        if (s == "v1") return TiledKernel::First;
        if (s == "v2") return TiledKernel::Second;
        if (s == "v2h") return TiledKernel::SecondPv16;
        if (s == "v2hh") return TiledKernel::SecondFp16;
        throw std::invalid_argument("SEPAL_ATTENTION_KERNEL must be v1, v2, v2h or v2hh, not " + s);
    }();
    return choice;
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
    const bool tiled_ok = attention_tiled_supported(shape, kv_type) && q.stride % 8 == 0;
    if (path == AttentionPath::Tiled && !tiled_ok) {
        throw std::invalid_argument(
            "attention: the tiled path needs F16 keys and values, head_dim 16/64/128/256, a group size of 1, 2, 4, 8 "
            "or 16 and a q row stride that is a multiple of 8");
    }
    const AttentionPath used = (path == AttentionPath::Reference || !tiled_ok) ? AttentionPath::Reference : AttentionPath::Tiled;
    if (rows.run_count == 0 || rows.max_row_tokens == 0) return used;

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
