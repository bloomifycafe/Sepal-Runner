// Device kernels of the Clef head. See clef_head_kernels.hpp.
#include "clef_head_kernels.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace sepal::cuda {

namespace {

constexpr int kThreads = 256;

inline void ck(cudaError_t status, const char* what) {
    if (status != cudaSuccess) throw std::runtime_error(std::string(what) + " failed: " + cudaGetErrorString(status));
}

inline unsigned blocks_for(std::size_t n, std::size_t per_block) {
    return static_cast<unsigned>((n + per_block - 1) / per_block);
}

} // namespace

// Kernels have external linkage (see convert_kernels.cu); the names carry a prefix of their own.

// Sum of one double per thread over a block of kThreads threads; every thread gets the total. Fixed order.
__device__ __forceinline__ double clef_block_sum(double v, double* shared) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xFFFFFFFFu, v, off);
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    if (lane == 0) shared[warp] = v;
    __syncthreads();
    double total = 0.0;
    for (int w = 0; w < kThreads / 32; ++w) total += shared[w];
    __syncthreads();
    return total;
}

__global__ void clef_k_layer_norm(const float* x, std::size_t x_stride, float* out, std::size_t out_stride, std::size_t width,
                             const float* gamma, const float* beta, float eps) {
    __shared__ double red[kThreads / 32];
    const std::size_t r = blockIdx.x;
    const float* row = x + r * x_stride;
    double s = 0.0;
    for (std::size_t c = threadIdx.x; c < width; c += kThreads) s += static_cast<double>(row[c]);
    const double mean = clef_block_sum(s, red) / static_cast<double>(width);
    double q = 0.0;
    for (std::size_t c = threadIdx.x; c < width; c += kThreads) {
        const double d = static_cast<double>(row[c]) - mean;
        q += d * d;
    }
    const double var = clef_block_sum(q, red) / static_cast<double>(width);
    const double rstd = 1.0 / sqrt(var + static_cast<double>(eps));
    float* o = out + r * out_stride;
    for (std::size_t c = threadIdx.x; c < width; c += kThreads) {
        double v = (static_cast<double>(row[c]) - mean) * rstd;
        if (gamma != nullptr) v *= static_cast<double>(gamma[c]);
        if (beta != nullptr) v += static_cast<double>(beta[c]);
        o[c] = static_cast<float>(v);
    }
}

__global__ void clef_k_broadcast_bias(float* out, std::size_t stride, std::size_t width, const float* bias) {
    float* row = out + blockIdx.x * stride;
    for (std::size_t c = threadIdx.x; c < width; c += kThreads) row[c] = bias[c];
}

__global__ void clef_k_add_bias(float* x, std::size_t stride, std::size_t width, const float* bias) {
    float* row = x + blockIdx.x * stride;
    for (std::size_t c = threadIdx.x; c < width; c += kThreads) row[c] += bias[c];
}

__global__ void clef_k_gelu(float* x, std::size_t n) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * kThreads + threadIdx.x;
    if (i >= n) return;
    const double v = static_cast<double>(x[i]);
    x[i] = static_cast<float>(0.5 * v * (1.0 + erf(v * 0.7071067811865476)));
}

__global__ void clef_k_scale_shift(float* x, std::size_t n, std::size_t width, const float* gamma, const float* beta) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * kThreads + threadIdx.x;
    if (i >= n * width) return;
    const std::size_t c = i % width;
    float v = x[i];
    if (gamma != nullptr) v *= gamma[c];
    if (beta != nullptr) v += beta[c];
    x[i] = v;
}

__global__ void clef_k_scale(float* x, std::size_t n, float s) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * kThreads + threadIdx.x;
    if (i < n) x[i] *= s;
}

__global__ void clef_k_add(float* x, const float* y, std::size_t n) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * kThreads + threadIdx.x;
    if (i < n) x[i] += y[i];
}

__global__ void clef_k_span_accumulate(const ClefSpan* spans, std::size_t s0, std::uint32_t first, std::uint32_t n,
                                  const float* x, std::size_t x_stride, float* acc, std::size_t width) {
    const std::size_t s = s0 + blockIdx.x;
    const std::uint32_t lo = spans[s].begin > first ? spans[s].begin : first;
    const std::uint32_t hi_span = spans[s].end;
    const std::uint32_t hi_call = first + n;
    const std::uint32_t hi = hi_span < hi_call ? hi_span : hi_call;
    if (lo >= hi) return;
    for (std::size_t c = threadIdx.x; c < width; c += kThreads) {
        double v = 0.0;
        for (std::uint32_t t = lo; t < hi; ++t) v += static_cast<double>(x[static_cast<std::size_t>(t - first) * x_stride + c]);
        acc[s * width + c] = static_cast<float>(static_cast<double>(acc[s * width + c]) + v);
    }
}

__global__ void clef_k_span_finalize(const ClefSpan* spans, float* acc, std::size_t width) {
    const std::size_t s = blockIdx.x;
    const double len = static_cast<double>(spans[s].end - spans[s].begin);
    for (std::size_t c = threadIdx.x; c < width; c += kThreads) {
        acc[s * width + c] = static_cast<float>(static_cast<double>(acc[s * width + c]) / len);
    }
}

__global__ void clef_k_lexical_means(const std::uint32_t* tok_begin, const std::uint32_t* token_index, const float* rows,
                                std::size_t width, float* out) {
    const std::size_t o = blockIdx.x;
    const std::uint32_t b = tok_begin[o], e = tok_begin[o + 1];
    const double len = static_cast<double>(e - b);
    for (std::size_t c = threadIdx.x; c < width; c += kThreads) {
        double v = 0.0;
        for (std::uint32_t j = b; j < e; ++j) v += static_cast<double>(rows[static_cast<std::size_t>(token_index[j]) * width + c]);
        out[o * width + c] = static_cast<float>(v / len);
    }
}

__global__ void clef_k_gather_add(float* dst, const float* src, const std::uint32_t* idx, std::size_t width) {
    const std::size_t o = blockIdx.x;
    const float* s = src + static_cast<std::size_t>(idx[o]) * width;
    for (std::size_t c = threadIdx.x; c < width; c += kThreads) dst[o * width + c] += s[c];
}

__global__ void clef_k_softmax_rows(float* x, std::size_t len, float scale) {
    __shared__ double red[kThreads / 32];
    __shared__ float shared_max[kThreads / 32];
    float* row = x + static_cast<std::size_t>(blockIdx.x) * len;
    float m = -3.0e38f;
    for (std::size_t i = threadIdx.x; i < len; i += kThreads) m = fmaxf(m, row[i] * scale);
    for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_down_sync(0xFFFFFFFFu, m, off));
    if ((threadIdx.x & 31) == 0) shared_max[threadIdx.x >> 5] = m;
    __syncthreads();
    float mx = shared_max[0];
    for (int w = 1; w < kThreads / 32; ++w) mx = fmaxf(mx, shared_max[w]);
    double s = 0.0;
    for (std::size_t i = threadIdx.x; i < len; i += kThreads) {
        const float e = expf(row[i] * scale - mx);
        row[i] = e;
        s += static_cast<double>(e);
    }
    const double total = clef_block_sum(s, red);
    const float inv = static_cast<float>(1.0 / total);
    for (std::size_t i = threadIdx.x; i < len; i += kThreads) row[i] *= inv;
}

// One block per (field, head); head_dim threads.
__global__ void clef_k_self_attention(const float* qkv, const std::uint32_t* field_request, const std::uint32_t* request_first,
                                 const std::uint32_t* request_count, std::size_t width, std::size_t heads, float* out) {
    extern __shared__ float scores[];
    const std::size_t f = blockIdx.x;
    const std::size_t h = blockIdx.y;
    const std::size_t hd = width / heads;
    const std::uint32_t req = field_request[f];
    const std::uint32_t first = request_first[req];
    const std::uint32_t count = request_count[req];
    const std::size_t stride = 3 * width;
    const float* q = qkv + f * stride + h * hd;
    const float inv_sqrt = 1.0f / sqrtf(static_cast<float>(hd));
    for (std::uint32_t j = threadIdx.x; j < count; j += blockDim.x) {
        const float* k = qkv + (static_cast<std::size_t>(first) + j) * stride + width + h * hd;
        double s = 0.0;
        for (std::size_t d = 0; d < hd; ++d) s += static_cast<double>(q[d]) * static_cast<double>(k[d]);
        scores[j] = static_cast<float>(s) * inv_sqrt;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float mx = -3.0e38f;
        for (std::uint32_t j = 0; j < count; ++j) mx = fmaxf(mx, scores[j]);
        double sum = 0.0;
        for (std::uint32_t j = 0; j < count; ++j) {
            const float e = expf(scores[j] - mx);
            scores[j] = e;
            sum += static_cast<double>(e);
        }
        const float inv = static_cast<float>(1.0 / sum);
        for (std::uint32_t j = 0; j < count; ++j) scores[j] *= inv;
    }
    __syncthreads();
    for (std::size_t d = threadIdx.x; d < hd; d += blockDim.x) {
        double v = 0.0;
        for (std::uint32_t j = 0; j < count; ++j) {
            v += static_cast<double>(scores[j]) * static_cast<double>(qkv[(static_cast<std::size_t>(first) + j) * stride + 2 * width + h * hd + d]);
        }
        out[f * width + h * hd + d] = static_cast<float>(v);
    }
}

__global__ void clef_k_option_summary(const float* opt, const float* base, const std::uint32_t* q_first, const std::uint32_t* q_count,
                                 std::size_t width, float* summary) {
    extern __shared__ float w[];
    __shared__ double red[kThreads / 32];
    const std::size_t q = blockIdx.x;
    const std::uint32_t first = q_first[q], count = q_count[q];
    const float inv_sqrt = 1.0f / sqrtf(static_cast<float>(width));
    for (std::uint32_t k = 0; k < count; ++k) {
        double s = 0.0;
        for (std::size_t c = threadIdx.x; c < width; c += kThreads) {
            s += static_cast<double>(opt[(static_cast<std::size_t>(first) + k) * width + c]) * static_cast<double>(base[q * width + c]);
        }
        const double total = clef_block_sum(s, red);
        if (threadIdx.x == 0) w[k] = static_cast<float>(total) * inv_sqrt;
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float mx = -3.0e38f;
        for (std::uint32_t k = 0; k < count; ++k) mx = fmaxf(mx, w[k]);
        double sum = 0.0;
        for (std::uint32_t k = 0; k < count; ++k) {
            const float e = expf(w[k] - mx);
            w[k] = e;
            sum += static_cast<double>(e);
        }
        const float inv = static_cast<float>(1.0 / sum);
        for (std::uint32_t k = 0; k < count; ++k) w[k] *= inv;
    }
    __syncthreads();
    for (std::size_t c = threadIdx.x; c < width; c += kThreads) {
        double v = 0.0;
        for (std::uint32_t k = 0; k < count; ++k) v += static_cast<double>(w[k]) * static_cast<double>(opt[(static_cast<std::size_t>(first) + k) * width + c]);
        summary[q * width + c] = static_cast<float>(v);
    }
}

__global__ void clef_k_init_fields(const float* base, const float* summary_norm, const float* global_proj, const std::uint32_t* q_request,
                              const float* type_embedding, const std::uint32_t* q_type, std::size_t width, float* fields) {
    const std::size_t q = blockIdx.x;
    for (std::size_t c = threadIdx.x; c < width; c += kThreads) {
        float v = base[q * width + c] + summary_norm[q * width + c];
        v += global_proj[static_cast<std::size_t>(q_request[q]) * width + c];
        v += type_embedding[static_cast<std::size_t>(q_type[q]) * width + c];
        fields[q * width + c] = v;
    }
}

__global__ void clef_k_scorer_features(const float* fields, const float* options, const std::uint32_t* option_question, std::size_t width,
                                  float* features, float* cosine) {
    __shared__ double red[kThreads / 32];
    const std::size_t o = blockIdx.x;
    const float* f = fields + static_cast<std::size_t>(option_question[o]) * width;
    const float* p = options + o * width;
    float* out = features + o * 4 * width;
    double dot = 0.0, nf = 0.0, np = 0.0;
    for (std::size_t c = threadIdx.x; c < width; c += kThreads) {
        const float a = f[c], b = p[c];
        out[c] = a;
        out[width + c] = b;
        out[2 * width + c] = a * b;
        out[3 * width + c] = fabsf(a - b);
        dot += static_cast<double>(a) * static_cast<double>(b);
        nf += static_cast<double>(a) * static_cast<double>(a);
        np += static_cast<double>(b) * static_cast<double>(b);
    }
    const double d = clef_block_sum(dot, red);
    const double sf = clef_block_sum(nf, red);
    const double sp = clef_block_sum(np, red);
    if (threadIdx.x == 0) {
        const double a = fmax(sqrt(sf), 1e-8), b = fmax(sqrt(sp), 1e-8);
        cosine[o] = static_cast<float>(d / (a * b));
    }
}

__global__ void clef_k_final_logits(const float* lex, const float* qvec, const float* global_vec, const std::uint32_t* option_question,
                               const std::uint32_t* question_request, const float* cosine, const float* hidden2, const float* w2,
                               float b2, std::size_t hidden_size, std::size_t width, float prior_scale, float joint_scale,
                               float gate, float* logits) {
    __shared__ double red[kThreads / 32];
    const std::size_t o = blockIdx.x;
    const std::size_t q = option_question[o];
    const std::size_t r = question_request[q];
    double dl = 0.0, da = 0.0, cross = 0.0;
    for (std::size_t c = threadIdx.x; c < hidden_size; c += kThreads) {
        const double l = static_cast<double>(lex[o * hidden_size + c]);
        const double a = static_cast<double>(qvec[q * hidden_size + c]) + static_cast<double>(global_vec[r * hidden_size + c]);
        dl += l * l;
        da += a * a;
        cross += l * a;
    }
    const double sl = clef_block_sum(dl, red);
    const double sa = clef_block_sum(da, red);
    const double sc = clef_block_sum(cross, red);
    double rs = 0.0;
    for (std::size_t c = threadIdx.x; c < width; c += kThreads) rs += static_cast<double>(hidden2[o * width + c]) * static_cast<double>(w2[c]);
    const double residual = clef_block_sum(rs, red) + static_cast<double>(b2);
    if (threadIdx.x == 0) {
        const double prior = static_cast<double>(prior_scale) * (sc / (fmax(sqrt(sl), 1e-12) * fmax(sqrt(sa), 1e-12)));
        const double joint = static_cast<double>(joint_scale) * static_cast<double>(cosine[o]) + residual;
        const double sig = 1.0 / (1.0 + exp(-static_cast<double>(gate)));
        logits[o] = static_cast<float>(prior + sig * joint);
    }
}

void clef_layer_norm(const float* x, std::size_t x_stride, float* out, std::size_t out_stride, std::size_t n, std::size_t width,
                     const float* gamma, const float* beta, float eps, cudaStream_t stream) {
    if (n == 0) return;
    clef_k_layer_norm<<<static_cast<unsigned>(n), kThreads, 0, stream>>>(x, x_stride, out, out_stride, width, gamma, beta, eps);
    ck(cudaGetLastError(), "clef_layer_norm");
}

void clef_broadcast_bias(float* out, std::size_t stride, std::size_t n, std::size_t width, const float* bias, cudaStream_t stream) {
    if (n == 0) return;
    clef_k_broadcast_bias<<<static_cast<unsigned>(n), kThreads, 0, stream>>>(out, stride, width, bias);
    ck(cudaGetLastError(), "clef_broadcast_bias");
}

void clef_add_bias(float* x, std::size_t stride, std::size_t n, std::size_t width, const float* bias, cudaStream_t stream) {
    if (n == 0) return;
    clef_k_add_bias<<<static_cast<unsigned>(n), kThreads, 0, stream>>>(x, stride, width, bias);
    ck(cudaGetLastError(), "clef_add_bias");
}

void clef_gelu(float* x, std::size_t n, cudaStream_t stream) {
    if (n == 0) return;
    clef_k_gelu<<<blocks_for(n, kThreads), kThreads, 0, stream>>>(x, n);
    ck(cudaGetLastError(), "clef_gelu");
}

void clef_scale_shift_cols(float* x, std::size_t n, std::size_t width, const float* gamma, const float* beta, cudaStream_t stream) {
    if (n == 0) return;
    clef_k_scale_shift<<<blocks_for(n * width, kThreads), kThreads, 0, stream>>>(x, n, width, gamma, beta);
    ck(cudaGetLastError(), "clef_scale_shift_cols");
}

void clef_scale(float* x, std::size_t n, float s, cudaStream_t stream) {
    if (n == 0) return;
    clef_k_scale<<<blocks_for(n, kThreads), kThreads, 0, stream>>>(x, n, s);
    ck(cudaGetLastError(), "clef_scale");
}

void clef_add(float* x, const float* y, std::size_t n, cudaStream_t stream) {
    if (n == 0) return;
    clef_k_add<<<blocks_for(n, kThreads), kThreads, 0, stream>>>(x, y, n);
    ck(cudaGetLastError(), "clef_add");
}

void clef_span_accumulate(const ClefSpan* spans, std::size_t s0, std::size_t count, std::uint32_t first, std::uint32_t n,
                          const float* x, std::size_t x_stride, float* acc, std::size_t width, cudaStream_t stream) {
    if (count == 0 || n == 0) return;
    clef_k_span_accumulate<<<static_cast<unsigned>(count), kThreads, 0, stream>>>(spans, s0, first, n, x, x_stride, acc, width);
    ck(cudaGetLastError(), "clef_span_accumulate");
}

void clef_span_finalize(const ClefSpan* spans, std::size_t count, float* acc, std::size_t width, cudaStream_t stream) {
    if (count == 0) return;
    clef_k_span_finalize<<<static_cast<unsigned>(count), kThreads, 0, stream>>>(spans, acc, width);
    ck(cudaGetLastError(), "clef_span_finalize");
}

void clef_lexical_means(const std::uint32_t* tok_begin, const std::uint32_t* token_index, const float* rows, std::size_t options,
                        std::size_t width, float* out, cudaStream_t stream) {
    if (options == 0) return;
    clef_k_lexical_means<<<static_cast<unsigned>(options), kThreads, 0, stream>>>(tok_begin, token_index, rows, width, out);
    ck(cudaGetLastError(), "clef_lexical_means");
}

void clef_gather_add(float* dst, const float* src, const std::uint32_t* idx, std::size_t n, std::size_t width, cudaStream_t stream) {
    if (n == 0) return;
    clef_k_gather_add<<<static_cast<unsigned>(n), kThreads, 0, stream>>>(dst, src, idx, width);
    ck(cudaGetLastError(), "clef_gather_add");
}

void clef_softmax_rows(float* x, std::size_t rows, std::size_t len, float scale, cudaStream_t stream) {
    if (rows == 0 || len == 0) return;
    clef_k_softmax_rows<<<static_cast<unsigned>(rows), kThreads, 0, stream>>>(x, len, scale);
    ck(cudaGetLastError(), "clef_softmax_rows");
}

void clef_self_attention(const float* qkv, const std::uint32_t* field_request, const std::uint32_t* request_first,
                         const std::uint32_t* request_count, std::size_t fields, std::size_t width, std::size_t heads, float* out,
                         cudaStream_t stream, std::size_t max_count) {
    if (fields == 0) return;
    const std::size_t hd = width / heads;
    const dim3 grid(static_cast<unsigned>(fields), static_cast<unsigned>(heads));
    const unsigned threads = static_cast<unsigned>(hd < 64 ? 64 : hd);
    clef_k_self_attention<<<grid, threads, max_count * sizeof(float), stream>>>(qkv, field_request, request_first, request_count, width, heads, out);
    ck(cudaGetLastError(), "clef_self_attention");
}

void clef_option_summary(const float* opt, const float* base, const std::uint32_t* q_first, const std::uint32_t* q_count,
                         std::size_t questions, std::size_t width, float* summary, cudaStream_t stream, std::size_t max_count) {
    if (questions == 0) return;
    clef_k_option_summary<<<static_cast<unsigned>(questions), kThreads, max_count * sizeof(float), stream>>>(opt, base, q_first, q_count, width, summary);
    ck(cudaGetLastError(), "clef_option_summary");
}

void clef_init_fields(const float* base, const float* summary_norm, const float* global_proj, const std::uint32_t* q_request,
                      const float* type_embedding, const std::uint32_t* q_type, std::size_t questions, std::size_t width, float* fields,
                      cudaStream_t stream) {
    if (questions == 0) return;
    clef_k_init_fields<<<static_cast<unsigned>(questions), kThreads, 0, stream>>>(base, summary_norm, global_proj, q_request, type_embedding, q_type, width, fields);
    ck(cudaGetLastError(), "clef_init_fields");
}

void clef_scorer_features(const float* fields, const float* options, const std::uint32_t* option_question, std::size_t n_options,
                          std::size_t width, float* features, float* cosine, cudaStream_t stream) {
    if (n_options == 0) return;
    clef_k_scorer_features<<<static_cast<unsigned>(n_options), kThreads, 0, stream>>>(fields, options, option_question, width, features, cosine);
    ck(cudaGetLastError(), "clef_scorer_features");
}

void clef_final_logits(const float* lex, const float* qvec, const float* global_vec, const std::uint32_t* option_question,
                       const std::uint32_t* question_request, const float* cosine, const float* hidden2, const float* w2, float b2,
                       std::size_t n_options, std::size_t hidden_size, std::size_t width, float prior_scale, float joint_scale,
                       float gate, float* logits, cudaStream_t stream) {
    if (n_options == 0) return;
    clef_k_final_logits<<<static_cast<unsigned>(n_options), kThreads, 0, stream>>>(lex, qvec, global_vec, option_question, question_request, cosine,
                                                                               hidden2, w2, b2, hidden_size, width, prior_scale, joint_scale, gate, logits);
    ck(cudaGetLastError(), "clef_final_logits");
}

} // namespace sepal::cuda
