// The elementwise, norm, MLP, embedding and readout kernels (see kernels_elementwise.hpp for the contracts).
//
// All of them are memory-bound, so the data is touched as few times as possible: a norm reads its row once into
// registers, reduces, and writes once; the residual addition is fused with the norm that follows it.
//
// Reduction order. A row is cut into chunks of four consecutive elements. Chunk c belongs to thread (c mod 256) of
// the row's block, and a thread adds the squares of its chunks in ascending chunk order, element by element, with
// fused multiply-adds written out by hand. The 256 partial sums are then added by a fixed tree: a butterfly over
// each warp, then the eight warp sums in order. Nothing in that order depends on the number of rows, on where the
// row is, on how the row is addressed, or on the alignment of its memory (a chunk past the end of the row, or a
// row that cannot be read four at a time, is read element by element and zero-filled, which adds exact zeros).
#include "kernels_elementwise.hpp"

#include "device.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <limits>
#include <stdexcept>

namespace sepal::cuda {

// The kernels have external linkage on purpose, as in convert_kernels.cu.

constexpr unsigned kBlock = 256;              // threads per block in every kernel here
constexpr unsigned kWarps = kBlock / 32;
constexpr int kRegChunks = 4;                 // chunks a norm thread keeps in registers (rows up to 4,096 wide)
constexpr unsigned kMaxGridY = 65535;

// ---- element access in groups of four ----

/// Four consecutive elements of `row` starting at `base`, zero past `width`. Reads a float4 when the whole group
/// is inside the row and `aligned` says the row's address allows it.
__device__ inline float4 load4(const float* row, std::size_t base, std::size_t width, bool aligned) {
    if (aligned && base + 4 <= width) return *reinterpret_cast<const float4*>(row + base);
    float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (base < width) v.x = row[base];
    if (base + 1 < width) v.y = row[base + 1];
    if (base + 2 < width) v.z = row[base + 2];
    if (base + 3 < width) v.w = row[base + 3];
    return v;
}

__device__ inline void store4(float* row, std::size_t base, std::size_t width, bool aligned, float4 v) {
    if (aligned && base + 4 <= width) {
        *reinterpret_cast<float4*>(row + base) = v;
        return;
    }
    if (base < width) row[base] = v.x;
    if (base + 1 < width) row[base + 1] = v.y;
    if (base + 2 < width) row[base + 2] = v.z;
    if (base + 3 < width) row[base + 3] = v.w;
}

/// Four float32 results rounded to binary16 (to nearest, ties to even), stored as one 8-byte write when possible.
__device__ inline void store4(__half* row, std::size_t base, std::size_t width, bool aligned, float4 v) {
    if (aligned && base + 4 <= width) {
        const __half2 lo = __floats2half2_rn(v.x, v.y);
        const __half2 hi = __floats2half2_rn(v.z, v.w);
        uint2 packed;
        packed.x = *reinterpret_cast<const unsigned int*>(&lo);
        packed.y = *reinterpret_cast<const unsigned int*>(&hi);
        *reinterpret_cast<uint2*>(row + base) = packed;
        return;
    }
    if (base < width) row[base] = __float2half_rn(v.x);
    if (base + 1 < width) row[base + 1] = __float2half_rn(v.y);
    if (base + 2 < width) row[base + 2] = __float2half_rn(v.z);
    if (base + 3 < width) row[base + 3] = __float2half_rn(v.w);
}

/// Four elements of a table row, widened to float32, zero past `width`.
__device__ inline float4 load4_widened(const float* row, std::size_t base, std::size_t width, bool aligned) {
    return load4(row, base, width, aligned);
}

__device__ inline float4 load4_widened(const __half* row, std::size_t base, std::size_t width, bool aligned) {
    if (aligned && base + 4 <= width) {
        const uint2 packed = *reinterpret_cast<const uint2*>(row + base);
        const float2 lo = __half22float2(*reinterpret_cast<const __half2*>(&packed.x));
        const float2 hi = __half22float2(*reinterpret_cast<const __half2*>(&packed.y));
        return make_float4(lo.x, lo.y, hi.x, hi.y);
    }
    float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (base < width) v.x = __half2float(row[base]);
    if (base + 1 < width) v.y = __half2float(row[base + 1]);
    if (base + 2 < width) v.z = __half2float(row[base + 2]);
    if (base + 3 < width) v.w = __half2float(row[base + 3]);
    return v;
}

// ---- the norm ----

/// acc + v.x^2 + v.y^2 + v.z^2 + v.w^2, in that order, each step one fused multiply-add.
__device__ inline float add_squares(float acc, float4 v) {
    acc = __fmaf_rn(v.x, v.x, acc);
    acc = __fmaf_rn(v.y, v.y, acc);
    acc = __fmaf_rn(v.z, v.z, acc);
    return __fmaf_rn(v.w, v.w, acc);
}

/// The sum of every thread's `partial` in the fixed order described at the top of the file; every thread of the
/// block gets the same value.
__device__ inline float block_sum(float partial, float* warp_sums) {
    for (int offset = 16; offset > 0; offset >>= 1) partial += __shfl_xor_sync(0xFFFFFFFFu, partial, offset);
    if ((threadIdx.x & 31u) == 0) warp_sums[threadIdx.x >> 5] = partial;
    __syncthreads();
    float total = warp_sums[0];
    for (unsigned w = 1; w < kWarps; ++w) total += warp_sums[w];
    return total;
}

/// sqrt(sum_sq / width + eps), each step rounded to float32 as in sepal::ref::rms_norm<float>.
__device__ inline float rms_of(float sum_sq, std::size_t width, float eps) {
    return __fsqrt_rn(__fadd_rn(__fdiv_rn(sum_sq, static_cast<float>(width)), eps));
}

/// (x / rms) * (1 + weight), the reference's order of operations, four elements at a time.
__device__ inline float4 scale4(float4 v, float rms, float4 w) {
    return make_float4(__fmul_rn(__fdiv_rn(v.x, rms), __fadd_rn(1.0f, w.x)),
                       __fmul_rn(__fdiv_rn(v.y, rms), __fadd_rn(1.0f, w.y)),
                       __fmul_rn(__fdiv_rn(v.z, rms), __fadd_rn(1.0f, w.z)),
                       __fmul_rn(__fdiv_rn(v.w, rms), __fadd_rn(1.0f, w.w)));
}

/// One block per output row. Output row j is the norm of row `rows[j]` of x, or of row j when `rows` is null.
template <class Out>
__global__ void rms_norm_kernel(const float* x, std::size_t x_stride, const std::uint32_t* rows, const float* weight,
                                std::size_t width, float eps, Out* out, std::size_t out_stride, bool aligned) {
    __shared__ float warp_sums[kWarps];
    const std::size_t j = blockIdx.x;
    const std::size_t source = rows != nullptr ? rows[j] : j;
    const float* xr = x + source * x_stride;
    Out* yr = out + j * out_stride;
    const std::size_t chunks = (width + 3) / 4;
    const std::size_t first = threadIdx.x;

    float4 held[kRegChunks];
    float partial = 0.0f;
#pragma unroll
    for (int k = 0; k < kRegChunks; ++k) {
        const std::size_t c = first + static_cast<std::size_t>(k) * kBlock;
        if (c < chunks) {
            held[k] = load4(xr, c * 4, width, aligned);
            partial = add_squares(partial, held[k]);
        }
    }
    for (std::size_t c = first + static_cast<std::size_t>(kRegChunks) * kBlock; c < chunks; c += kBlock) {
        partial = add_squares(partial, load4(xr, c * 4, width, aligned));
    }
    const float rms = rms_of(block_sum(partial, warp_sums), width, eps);

#pragma unroll
    for (int k = 0; k < kRegChunks; ++k) {
        const std::size_t c = first + static_cast<std::size_t>(k) * kBlock;
        if (c < chunks) store4(yr, c * 4, width, aligned, scale4(held[k], rms, load4(weight, c * 4, width, aligned)));
    }
    for (std::size_t c = first + static_cast<std::size_t>(kRegChunks) * kBlock; c < chunks; c += kBlock) {
        store4(yr, c * 4, width, aligned, scale4(load4(xr, c * 4, width, aligned), rms, load4(weight, c * 4, width, aligned)));
    }
}

/// r = x + m written to r_out, then the norm of r. Rows of r beyond the registers are read back from r_out by the
/// thread that wrote them, so no synchronisation is needed for them and r_out may be x or m (float32 m only). `MIn` is
/// the type of m: float, or __half (a multiply result that was left in binary16; widening is exact, so the sum is the
/// same as with the widened values).
template <class Out, class MIn>
__global__ void add_rms_norm_kernel(const float* x, std::size_t x_stride, const MIn* m, std::size_t m_stride,
                                    float* r_out, std::size_t r_stride, const float* weight, std::size_t width,
                                    float eps, Out* out, std::size_t out_stride, bool aligned) {
    __shared__ float warp_sums[kWarps];
    const std::size_t row = blockIdx.x;
    const float* xr = x + row * x_stride;
    const MIn* mr = m + row * m_stride;
    float* rr = r_out + row * r_stride;
    Out* yr = out + row * out_stride;
    const std::size_t chunks = (width + 3) / 4;
    const std::size_t first = threadIdx.x;

    float4 held[kRegChunks];
    float partial = 0.0f;
#pragma unroll
    for (int k = 0; k < kRegChunks; ++k) {
        const std::size_t c = first + static_cast<std::size_t>(k) * kBlock;
        if (c < chunks) {
            const float4 a = load4(xr, c * 4, width, aligned);
            const float4 b = load4_widened(mr, c * 4, width, aligned);
            held[k] = make_float4(__fadd_rn(a.x, b.x), __fadd_rn(a.y, b.y), __fadd_rn(a.z, b.z), __fadd_rn(a.w, b.w));
            store4(rr, c * 4, width, aligned, held[k]);
            partial = add_squares(partial, held[k]);
        }
    }
    for (std::size_t c = first + static_cast<std::size_t>(kRegChunks) * kBlock; c < chunks; c += kBlock) {
        const float4 a = load4(xr, c * 4, width, aligned);
        const float4 b = load4_widened(mr, c * 4, width, aligned);
        const float4 sum = make_float4(__fadd_rn(a.x, b.x), __fadd_rn(a.y, b.y), __fadd_rn(a.z, b.z), __fadd_rn(a.w, b.w));
        store4(rr, c * 4, width, aligned, sum);
        partial = add_squares(partial, sum);
    }
    const float rms = rms_of(block_sum(partial, warp_sums), width, eps);

#pragma unroll
    for (int k = 0; k < kRegChunks; ++k) {
        const std::size_t c = first + static_cast<std::size_t>(k) * kBlock;
        if (c < chunks) store4(yr, c * 4, width, aligned, scale4(held[k], rms, load4(weight, c * 4, width, aligned)));
    }
    for (std::size_t c = first + static_cast<std::size_t>(kRegChunks) * kBlock; c < chunks; c += kBlock) {
        store4(yr, c * 4, width, aligned, scale4(load4(rr, c * 4, width, aligned), rms, load4(weight, c * 4, width, aligned)));
    }
}

// ---- the elementwise kernels ----

template <class MIn>
__global__ void add_kernel(const float* x, std::size_t x_stride, const MIn* m, std::size_t m_stride, float* out,
                           std::size_t out_stride, std::size_t n, std::size_t width, bool aligned) {
    const std::size_t c = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c * 4 >= width) return;
    for (std::size_t row = blockIdx.y; row < n; row += gridDim.y) {
        const float4 a = load4(x + row * x_stride, c * 4, width, aligned);
        const float4 b = load4_widened(m + row * m_stride, c * 4, width, aligned);
        store4(out + row * out_stride, c * 4, width, aligned,
               make_float4(__fadd_rn(a.x, b.x), __fadd_rn(a.y, b.y), __fadd_rn(a.z, b.z), __fadd_rn(a.w, b.w)));
    }
}

/// softplus(x) = max(x, 0) + log1p(exp(-|x|)), as in sepal::ref (never exponentiates a positive number).
__device__ inline float softplus_f(float x) { return fmaxf(x, 0.0f) + log1pf(expf(-fabsf(x))); }

/// x * sigmoid(x) with sigmoid(x) = exp(-softplus(-x)).
__device__ inline float silu_f(float x) { return __fmul_rn(x, expf(-softplus_f(-x))); }

template <class In, class Out>
__global__ void swiglu_kernel(const In* gate, std::size_t gate_stride, const In* up, std::size_t up_stride,
                              std::size_t n, std::size_t width, Out* out, std::size_t out_stride, bool aligned) {
    const std::size_t c = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c * 4 >= width) return;
    for (std::size_t row = blockIdx.y; row < n; row += gridDim.y) {
        const float4 g = load4_widened(gate + row * gate_stride, c * 4, width, aligned);
        const float4 u = load4_widened(up + row * up_stride, c * 4, width, aligned);
        store4(out + row * out_stride, c * 4, width, aligned,
               make_float4(__fmul_rn(silu_f(g.x), u.x), __fmul_rn(silu_f(g.y), u.y), __fmul_rn(silu_f(g.z), u.z),
                           __fmul_rn(silu_f(g.w), u.w)));
    }
}

/// One block per token.
template <class Src>
__global__ void embed_kernel(const std::uint32_t* token_ids, const Src* table, std::size_t vocab, std::size_t hidden,
                             float* out, std::size_t out_stride, bool aligned) {
    const std::size_t t = blockIdx.x;
    const std::uint32_t id = token_ids[t];
    float* yr = out + t * out_stride;
    const std::size_t chunks = (hidden + 3) / 4;
    if (id >= vocab) {
        for (std::size_t c = threadIdx.x; c < chunks; c += blockDim.x) {
            store4(yr, c * 4, hidden, aligned, make_float4(0.0f, 0.0f, 0.0f, 0.0f));
        }
        return;
    }
    const Src* row = table + static_cast<std::size_t>(id) * hidden;
    for (std::size_t c = threadIdx.x; c < chunks; c += blockDim.x) {
        store4(yr, c * 4, hidden, aligned, load4_widened(row, c * 4, hidden, aligned));
    }
}

// ---- host side ----

namespace {

bool aligned16(const void* p) { return reinterpret_cast<std::uintptr_t>(p) % 16 == 0; }

void require(bool ok, const char* what) {
    if (!ok) throw std::invalid_argument(what);
}

/// A matrix of n rows of `width` elements at `stride`: nothing to do when empty, and the stride must be able to
/// hold a row.
void require_matrix(std::size_t n, std::size_t width, std::size_t stride, const char* what) {
    require(n <= static_cast<std::size_t>(std::numeric_limits<int>::max()), "elementwise kernels: too many rows");
    require(n == 0 || width == 0 || stride >= width, what);
}

/// The bytes a matrix of float32 occupies, as [begin, end).
struct Extent {
    std::uintptr_t begin;
    std::uintptr_t end;
};

Extent extent_of(const void* p, std::size_t n, std::size_t width, std::size_t stride, std::size_t element_bytes) {
    const auto begin = reinterpret_cast<std::uintptr_t>(p);
    return {begin, begin + ((n - 1) * stride + width) * element_bytes};
}

bool overlap(Extent a, Extent b) { return a.begin < b.end && b.begin < a.end; }

/// True when an output matrix is the same matrix as an input (allowed: in place) or shares no byte with it.
bool disjoint_or_same(const void* a, std::size_t a_stride, std::size_t a_bytes, const void* b, std::size_t b_stride,
                      std::size_t b_bytes, std::size_t n, std::size_t width) {
    if (a == b && a_stride == b_stride && a_bytes == b_bytes) return true;
    return !overlap(extent_of(a, n, width, a_stride, a_bytes), extent_of(b, n, width, b_stride, b_bytes));
}

dim3 grid_for_chunks(std::size_t n, std::size_t width) {
    const std::size_t chunks = (width + 3) / 4;
    const std::size_t blocks_x = (chunks + kBlock - 1) / kBlock;
    const std::size_t blocks_y = n < kMaxGridY ? n : kMaxGridY;
    return dim3(static_cast<unsigned>(blocks_x), static_cast<unsigned>(blocks_y));
}

void check_launch() { check(cudaGetLastError(), "kernel launch"); }

void require_output_type(Elem type) {
    require(type == Elem::F32 || type == Elem::F16, "elementwise kernels: the output type must be F32 or F16");
}

} // namespace

void launch_embed(const std::uint32_t* token_ids, std::size_t n, const void* table, Elem table_type,
                  std::size_t vocab, std::size_t hidden, float* out, std::size_t out_stride, cudaStream_t stream) {
    require(table_type == Elem::F32 || table_type == Elem::F16, "launch_embed: the table must be F32 or F16");
    require_matrix(n, hidden, out_stride, "launch_embed: out_stride is smaller than hidden");
    if (n == 0 || hidden == 0) return;
    const bool aligned = hidden % 4 == 0 && out_stride % 4 == 0 && aligned16(table) && aligned16(out);
    if (table_type == Elem::F32) {
        embed_kernel<float><<<static_cast<unsigned>(n), kBlock, 0, stream>>>(
            token_ids, static_cast<const float*>(table), vocab, hidden, out, out_stride, aligned);
    } else {
        embed_kernel<__half><<<static_cast<unsigned>(n), kBlock, 0, stream>>>(
            token_ids, static_cast<const __half*>(table), vocab, hidden, out, out_stride, aligned);
    }
    check_launch();
}

namespace {

/// Shared by the norm launchers.
template <class Out>
void run_rms_norm(const float* x, std::size_t x_stride, const std::uint32_t* rows, const float* weight, std::size_t count,
                  std::size_t width, float eps, Out* out, std::size_t out_stride, bool aligned, cudaStream_t stream) {
    rms_norm_kernel<Out><<<static_cast<unsigned>(count), kBlock, 0, stream>>>(x, x_stride, rows, weight, width, eps,
                                                                                  out, out_stride, aligned);
    check_launch();
}

} // namespace

void launch_rms_norm(const float* x, std::size_t x_stride, const float* weight, std::size_t n, std::size_t width,
                     float eps, void* out, Elem out_type, std::size_t out_stride, cudaStream_t stream) {
    require_output_type(out_type);
    require_matrix(n, width, x_stride, "launch_rms_norm: x_stride is smaller than width");
    require_matrix(n, width, out_stride, "launch_rms_norm: out_stride is smaller than width");
    if (n == 0 || width == 0) return;
    const bool aligned = x_stride % 4 == 0 && out_stride % 4 == 0 && aligned16(x) && aligned16(weight) && aligned16(out);
    if (out_type == Elem::F32) {
        run_rms_norm<float>(x, x_stride, nullptr, weight, n, width, eps, static_cast<float*>(out), out_stride, aligned, stream);
    } else {
        run_rms_norm<__half>(x, x_stride, nullptr, weight, n, width, eps, static_cast<__half*>(out), out_stride, aligned, stream);
    }
}

namespace {

// launch_add_rms_norm for m of float32 or binary16 (`MIn`); the overlap rules are those of the float32 original.
template <class MIn>
void add_rms_norm_launch(const float* x, std::size_t x_stride, const MIn* m, std::size_t m_stride, float* r_out,
                         std::size_t r_stride, const float* weight, std::size_t n, std::size_t width, float eps,
                         void* norm_out, Elem out_type, std::size_t norm_stride, cudaStream_t stream) {
    require_output_type(out_type);
    require_matrix(n, width, x_stride, "launch_add_rms_norm: x_stride is smaller than width");
    require_matrix(n, width, m_stride, "launch_add_rms_norm: m_stride is smaller than width");
    require_matrix(n, width, r_stride, "launch_add_rms_norm: r_stride is smaller than width");
    require_matrix(n, width, norm_stride, "launch_add_rms_norm: norm_stride is smaller than width");
    if (n == 0 || width == 0) return;
    const std::size_t out_bytes = out_type == Elem::F32 ? sizeof(float) : sizeof(std::uint16_t);
    // r_out may be x or (float32 m only) m; binary16 m is another matrix of other bytes.
    require(disjoint_or_same(r_out, r_stride, 4, x, x_stride, 4, n, width) &&
                disjoint_or_same(r_out, r_stride, 4, m, m_stride, sizeof(MIn), n, width),
            "launch_add_rms_norm: r_out overlaps x or m other than as the same matrix");
    require(!overlap(extent_of(norm_out, n, width, norm_stride, out_bytes), extent_of(r_out, n, width, r_stride, 4)) &&
                !overlap(extent_of(norm_out, n, width, norm_stride, out_bytes), extent_of(x, n, width, x_stride, 4)) &&
                !overlap(extent_of(norm_out, n, width, norm_stride, out_bytes), extent_of(m, n, width, m_stride, sizeof(MIn))),
            "launch_add_rms_norm: norm_out overlaps an input or r_out");
    const bool aligned = x_stride % 4 == 0 && m_stride % 4 == 0 && r_stride % 4 == 0 && norm_stride % 4 == 0 &&
                         aligned16(x) && aligned16(m) && aligned16(r_out) && aligned16(weight) && aligned16(norm_out);
    const unsigned blocks = static_cast<unsigned>(n);
    if (out_type == Elem::F32) {
        add_rms_norm_kernel<float, MIn><<<blocks, kBlock, 0, stream>>>(x, x_stride, m, m_stride, r_out, r_stride, weight,
                                                                         width, eps, static_cast<float*>(norm_out),
                                                                         norm_stride, aligned);
    } else {
        add_rms_norm_kernel<__half, MIn><<<blocks, kBlock, 0, stream>>>(x, x_stride, m, m_stride, r_out, r_stride, weight,
                                                                          width, eps, static_cast<__half*>(norm_out),
                                                                          norm_stride, aligned);
    }
    check_launch();
}

template <class MIn>
void add_launch(const float* x, std::size_t x_stride, const MIn* m, std::size_t m_stride, float* out, std::size_t out_stride,
                std::size_t n, std::size_t width, cudaStream_t stream) {
    require_matrix(n, width, x_stride, "launch_add: x_stride is smaller than width");
    require_matrix(n, width, m_stride, "launch_add: m_stride is smaller than width");
    require_matrix(n, width, out_stride, "launch_add: out_stride is smaller than width");
    if (n == 0 || width == 0) return;
    require(disjoint_or_same(out, out_stride, 4, x, x_stride, 4, n, width) &&
                disjoint_or_same(out, out_stride, 4, m, m_stride, sizeof(MIn), n, width),
            "launch_add: out overlaps x or m other than as the same matrix");
    const bool aligned = x_stride % 4 == 0 && m_stride % 4 == 0 && out_stride % 4 == 0 && aligned16(x) && aligned16(m) &&
                         aligned16(out);
    add_kernel<MIn><<<grid_for_chunks(n, width), kBlock, 0, stream>>>(x, x_stride, m, m_stride, out, out_stride, n, width,
                                                                       aligned);
    check_launch();
}

} // namespace

void launch_add_rms_norm(const float* x, std::size_t x_stride, const float* m, std::size_t m_stride, float* r_out,
                         std::size_t r_stride, const float* weight, std::size_t n, std::size_t width, float eps,
                         void* norm_out, Elem out_type, std::size_t norm_stride, cudaStream_t stream) {
    add_rms_norm_launch<float>(x, x_stride, m, m_stride, r_out, r_stride, weight, n, width, eps, norm_out, out_type, norm_stride,
                               stream);
}

void launch_add_rms_norm_half_m(const float* x, std::size_t x_stride, const void* m, std::size_t m_stride, float* r_out,
                                std::size_t r_stride, const float* weight, std::size_t n, std::size_t width, float eps,
                                void* norm_out, Elem out_type, std::size_t norm_stride, cudaStream_t stream) {
    add_rms_norm_launch<__half>(x, x_stride, static_cast<const __half*>(m), m_stride, r_out, r_stride, weight, n, width, eps,
                                norm_out, out_type, norm_stride, stream);
}

void launch_add(const float* x, std::size_t x_stride, const float* m, std::size_t m_stride, float* out,
                std::size_t out_stride, std::size_t n, std::size_t width, cudaStream_t stream) {
    add_launch<float>(x, x_stride, m, m_stride, out, out_stride, n, width, stream);
}

void launch_add_half_m(const float* x, std::size_t x_stride, const void* m, std::size_t m_stride, float* out,
                       std::size_t out_stride, std::size_t n, std::size_t width, cudaStream_t stream) {
    add_launch<__half>(x, x_stride, static_cast<const __half*>(m), m_stride, out, out_stride, n, width, stream);
}

void launch_swiglu(const float* gate, std::size_t gate_stride, const float* up, std::size_t up_stride, std::size_t n,
                   std::size_t width, void* out, Elem out_type, std::size_t out_stride, cudaStream_t stream) {
    require_output_type(out_type);
    require_matrix(n, width, gate_stride, "launch_swiglu: gate_stride is smaller than width");
    require_matrix(n, width, up_stride, "launch_swiglu: up_stride is smaller than width");
    require_matrix(n, width, out_stride, "launch_swiglu: out_stride is smaller than width");
    if (n == 0 || width == 0) return;
    const std::size_t out_bytes = out_type == Elem::F32 ? sizeof(float) : sizeof(std::uint16_t);
    const Extent o = extent_of(out, n, width, out_stride, out_bytes);
    require(!overlap(o, extent_of(gate, n, width, gate_stride, 4)) && !overlap(o, extent_of(up, n, width, up_stride, 4)),
            "launch_swiglu: out overlaps gate or up");
    const bool aligned = gate_stride % 4 == 0 && up_stride % 4 == 0 && out_stride % 4 == 0 && aligned16(gate) &&
                         aligned16(up) && aligned16(out);
    if (out_type == Elem::F32) {
        swiglu_kernel<float, float><<<grid_for_chunks(n, width), kBlock, 0, stream>>>(
            gate, gate_stride, up, up_stride, n, width, static_cast<float*>(out), out_stride, aligned);
    } else {
        swiglu_kernel<float, __half><<<grid_for_chunks(n, width), kBlock, 0, stream>>>(
            gate, gate_stride, up, up_stride, n, width, static_cast<__half*>(out), out_stride, aligned);
    }
    check_launch();
}

void launch_swiglu_half(const void* gate, std::size_t gate_stride, const void* up, std::size_t up_stride, std::size_t n,
                        std::size_t width, void* out, Elem out_type, std::size_t out_stride, cudaStream_t stream) {
    require_output_type(out_type);
    require_matrix(n, width, gate_stride, "launch_swiglu_half: gate_stride is smaller than width");
    require_matrix(n, width, up_stride, "launch_swiglu_half: up_stride is smaller than width");
    require_matrix(n, width, out_stride, "launch_swiglu_half: out_stride is smaller than width");
    if (n == 0 || width == 0) return;
    const std::size_t out_bytes = out_type == Elem::F32 ? sizeof(float) : sizeof(std::uint16_t);
    const Extent o = extent_of(out, n, width, out_stride, out_bytes);
    require(!overlap(o, extent_of(gate, n, width, gate_stride, 2)) && !overlap(o, extent_of(up, n, width, up_stride, 2)),
            "launch_swiglu_half: out overlaps gate or up");
    const bool aligned = gate_stride % 4 == 0 && up_stride % 4 == 0 && out_stride % 4 == 0 && aligned16(gate) &&
                         aligned16(up) && aligned16(out);
    if (out_type == Elem::F32) {
        swiglu_kernel<__half, float><<<grid_for_chunks(n, width), kBlock, 0, stream>>>(
            static_cast<const __half*>(gate), gate_stride, static_cast<const __half*>(up), up_stride, n, width,
            static_cast<float*>(out), out_stride, aligned);
    } else {
        swiglu_kernel<__half, __half><<<grid_for_chunks(n, width), kBlock, 0, stream>>>(
            static_cast<const __half*>(gate), gate_stride, static_cast<const __half*>(up), up_stride, n, width,
            static_cast<__half*>(out), out_stride, aligned);
    }
    check_launch();
}

void launch_readout_norm(const float* x, std::size_t x_stride, const std::uint32_t* rows, std::size_t count,
                         const float* weight, std::size_t width, float eps, float* out, std::size_t out_stride,
                         cudaStream_t stream) {
    require(x_stride >= width, "launch_readout_norm: x_stride is smaller than width");
    require_matrix(count, width, out_stride, "launch_readout_norm: out_stride is smaller than width");
    if (count == 0 || width == 0) return;
    const bool aligned = x_stride % 4 == 0 && out_stride % 4 == 0 && aligned16(x) && aligned16(weight) && aligned16(out);
    run_rms_norm<float>(x, x_stride, rows, weight, count, width, eps, out, out_stride, aligned, stream);
}

} // namespace sepal::cuda
