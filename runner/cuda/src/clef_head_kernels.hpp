// Device kernels of the Clef head (include/sepal/cuda/clef_head_cuda.hpp). Plain types only, so that the host compiler
// and nvcc read the same header. Every launch enqueues on `stream` and returns without waiting. All data is float32.
//
// Determinism: every output element is reduced in one fixed order that depends on the element's own row only
// (no atomics on floating-point values). Sums that PyTorch takes in float32 are taken here in double and rounded once,
// which is as close to the exact value as the float32 oracle is.
#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace sepal::cuda {

/// out[r, :] = (x[r, :] - mean) / sqrt(var + eps) * gamma + beta (gamma and beta may be null: no affine part).
/// x: [n, width] with row stride x_stride; out: row stride out_stride (may be x).
void clef_layer_norm(const float* x, std::size_t x_stride, float* out, std::size_t out_stride, std::size_t n,
                     std::size_t width, const float* gamma, const float* beta, float eps, cudaStream_t stream);

/// out[r, :] = bias[:] for r in [0, n): the start of a linear layer whose multiply then accumulates (beta = 1).
void clef_broadcast_bias(float* out, std::size_t stride, std::size_t n, std::size_t width, const float* bias,
                         cudaStream_t stream);

/// x[r, :] += bias[:] over the first `width` columns of each row.
void clef_add_bias(float* x, std::size_t stride, std::size_t n, std::size_t width, const float* bias, cudaStream_t stream);

/// x = gelu(x) (the exact, erf form), n elements.
void clef_gelu(float* x, std::size_t n, cudaStream_t stream);

/// x[r, c] = x[r, c] * gamma[c] + beta[c] (either may be null).
void clef_scale_shift_cols(float* x, std::size_t n, std::size_t width, const float* gamma, const float* beta,
                           cudaStream_t stream);

/// x[i] *= s.
void clef_scale(float* x, std::size_t n, float s, cudaStream_t stream);

/// x += y, n elements.
void clef_add(float* x, const float* y, std::size_t n, cudaStream_t stream);

/// The description of one span the head averages over the final-normed hidden states: tokens [begin, end) of request
/// `request`; `out` indexes the row of the accumulator it fills.
struct ClefSpan {
    std::uint32_t begin;
    std::uint32_t end;
};

/// acc[s, :] += sum over t in [max(begin, first), min(end, first + n)) of x[t - first, :], for the spans [s0, s0 + count).
/// acc is [spans, width] float32, x is [n, width] row-major (stride x_stride). Accumulation in token order.
void clef_span_accumulate(const ClefSpan* spans, std::size_t s0, std::size_t count, std::uint32_t first, std::uint32_t n,
                          const float* x, std::size_t x_stride, float* acc, std::size_t width, cudaStream_t stream);

/// acc[s, :] /= (end - begin) for all spans.
void clef_span_finalize(const ClefSpan* spans, std::size_t count, float* acc, std::size_t width, cudaStream_t stream);

/// Lexical vectors: out[o, :] = mean over the option's tokens of rows[token_index[tok_begin[o] + j], :].
void clef_lexical_means(const std::uint32_t* tok_begin, const std::uint32_t* token_index, const float* rows,
                        std::size_t options, std::size_t width, float* out, cudaStream_t stream);

/// dst[o, :] += src[idx[o], :] for o in [0, n).
void clef_gather_add(float* dst, const float* src, const std::uint32_t* idx, std::size_t n, std::size_t width,
                     cudaStream_t stream);

/// Row softmax in place, ragged: `rows` rows of length `len`, each multiplied by `scale` first.
void clef_softmax_rows(float* x, std::size_t rows, std::size_t len, float scale, cudaStream_t stream);

/// Field self-attention (no mask) over the fields of each request. qkv: [fields, 3 * width] (q | k | v, q already
/// biased; the kernel applies 1/sqrt(head_dim)); request_first/request_count: per request, the first field and the
/// number of fields. out: [fields, width].
void clef_self_attention(const float* qkv, const std::uint32_t* field_request, const std::uint32_t* request_first,
                         const std::uint32_t* request_count, std::size_t fields, std::size_t width, std::size_t heads,
                         float* out, cudaStream_t stream, std::size_t max_count);

/// Option routing summary: for each question, softmax over its options of (option . base) / sqrt(width), then the weighted
/// sum of the options: summary[q, :] = sum_k w_k opt_k. opt: [options, width]; base: [questions, width];
/// q_first/q_count: each question's first option and option count.
void clef_option_summary(const float* opt, const float* base, const std::uint32_t* q_first, const std::uint32_t* q_count,
                         std::size_t questions, std::size_t width, float* summary, cudaStream_t stream,
                         std::size_t max_count);

/// fields[q, :] = base[q, :] + summary_norm[q, :] + global[request_of_q, :] + type_embedding[type[q], :].
void clef_init_fields(const float* base, const float* summary_norm, const float* global_proj,
                      const std::uint32_t* q_request, const float* type_embedding, const std::uint32_t* q_type,
                      std::size_t questions, std::size_t width, float* fields, cudaStream_t stream);

/// features[o, :] = [f, p, f * p, |f - p|] with f = fields[option_question[o]], p = options[o]; cosine[o] = cos(f, p).
void clef_scorer_features(const float* fields, const float* options, const std::uint32_t* option_question,
                          std::size_t n_options, std::size_t width, float* features, float* cosine, cudaStream_t stream);

/// logit[o] = prior_scale * dot(normalize(lex[o]), normalize(qvec[q] + global[r])) +
///            sigmoid(gate) * (joint_scale * cosine[o] + residual[o]); residual[o] = hidden2[o, :] . w2 + b2.
void clef_final_logits(const float* lex, const float* qvec, const float* global_vec, const std::uint32_t* option_question,
                       const std::uint32_t* question_request, const float* cosine, const float* hidden2,
                       const float* w2, float b2, std::size_t n_options, std::size_t hidden_size, std::size_t width,
                       float prior_scale, float joint_scale, float gate, float* logits, cudaStream_t stream);

} // namespace sepal::cuda
