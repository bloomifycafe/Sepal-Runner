// The full-attention mixer: gated grouped-query attention with per-head norms and partial RoPE.
// A port of sepal/refengine/attention.py, function for function (that file is adapted from
// jaredpalmer/kev@5e42a7a kev/fused_qwen35.py, Apache-2.0).
//
// The oracle marks four details of this mixer as not yet verified against the real model (see
// `sepal.refengine.UNVERIFIED`). Each lives in exactly one function here, marked UNVERIFIED with the oracle's
// name for it.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "sepal/config.hpp"
#include "sepal/gemm.hpp"
#include "sepal/ref/primitives.hpp"
#include "sepal/ref/snapshot.hpp"
#include "sepal/tensor.hpp"
#include "sepal/weights.hpp"

namespace sepal::ref {

template <class T>
struct QueryAndGate {
    BasicTensor<T> query; // [T, heads, head_dim]
    BasicTensor<T> gate;  // [T, heads, head_dim]
};

/// The `q_proj` output [T, 2 * heads * head_dim] as the query and its output gate, each [T, heads, head_dim].
/// The packing is per head: each head's 2 * head_dim outputs are its query, then its gate.
template <class T>
QueryAndGate<T> unpack_query_and_gate(BasicConstMatView<T> q_gate, const ModelConfig& config) {
    const std::size_t heads = dim(config.attn_heads);
    const std::size_t head_dim = dim(config.attn_head_dim);
    if (q_gate.cols != 2 * heads * head_dim) throw std::invalid_argument("unpack_query_and_gate: wrong q_proj width");
    const std::vector<std::size_t> shape = {q_gate.rows, heads, head_dim};
    QueryAndGate<T> out{BasicTensor<T>(shape), BasicTensor<T>(shape)};
    for (std::size_t r = 0; r < q_gate.rows * heads; ++r) {
        const T* packed = q_gate.data + r * 2 * head_dim;
        std::copy(packed, packed + head_dim, out.query.data() + r * head_dim);
        std::copy(packed + head_dim, packed + 2 * head_dim, out.gate.data() + r * head_dim);
    }
    return out;
}

/// UNVERIFIED (attention.rope_angles): one position axis, frequencies theta ** (-2i / rotary_dim).
///
/// [T, rotary_dim / 2] rotation angles: position times `theta ** (-2i / rotary_dim)`. One position per token:
/// for text the model's three position axes carry the same value, so its sectioned multi-axis RoPE reduces to
/// this. Angles are float64 whatever the element type of the pass: a position is a large whole number and its
/// product with a frequency must not be rounded before the cosine and sine are taken.
inline TensorD rope_angles(std::span<const double> positions, const ModelConfig& config) {
    const std::size_t half = dim(config.attn_rotary_dims()) / 2;
    const double rotary = static_cast<double>(2 * half);
    TensorD angles({positions.size(), half});
    for (std::size_t t = 0; t < positions.size(); ++t) {
        for (std::size_t i = 0; i < half; ++i) {
            const double inverse_frequency = std::pow(config.rope_theta, -static_cast<double>(2 * i) / rotary);
            angles[t * half + i] = positions[t] * inverse_frequency;
        }
    }
    return angles;
}

/// UNVERIFIED (attention.apply_rope): the FIRST rotary_dim dims rotate, dim i paired with dim i + rotary_dim / 2.
///
/// Rotate the first rotary_dim dims of every head, in place; the rest pass through. x [T, heads, head_dim];
/// angles [T, rotary_dim / 2]. Dim i is paired with dim i + rotary_dim / 2 (the two halves of the rotated part,
/// not neighbouring dims): `(a, b) -> (a cos - b sin, b cos + a sin)`.
template <class T>
void apply_rope(BasicTensor<T>& x, const TensorD& angles) {
    const std::size_t tokens = x.shape().at(0);
    const std::size_t heads = x.shape().at(1);
    const std::size_t head_dim = x.shape().at(2);
    const std::size_t half = angles.shape().at(1);
    if (angles.shape().at(0) != tokens || 2 * half > head_dim) throw std::invalid_argument("apply_rope: angles do not fit x");
    for (std::size_t t = 0; t < tokens; ++t) {
        for (std::size_t h = 0; h < heads; ++h) {
            T* head = x.data() + (t * heads + h) * head_dim;
            for (std::size_t i = 0; i < half; ++i) {
                const T cosine = static_cast<T>(std::cos(angles[t * half + i]));
                const T sine = static_cast<T>(std::sin(angles[t * half + i]));
                const T first = head[i];
                const T second = head[half + i];
                head[i] = first * cosine - second * sine;
                head[half + i] = second * cosine + first * sine;
            }
        }
    }
}

/// UNVERIFIED (attention.attention_scale): scores are scaled by 1 / sqrt(head_dim).
inline double attention_scale(const ModelConfig& config) {
    return 1.0 / std::sqrt(static_cast<double>(config.attn_head_dim));
}

/// UNVERIFIED (attention.grouped_query_attention): query head h reads kv head h / (heads / kv heads)
/// (repeat-interleave, not tiling).
///
/// Causal grouped-query attention of a run of tokens over its context and itself.
/// q [T, heads, head_dim]; `state` holds the keys and values of context + T tokens, the first `context` of them
/// from the parent chain and the last T this run's own. Token t attends to every context token and to own
/// tokens 0..t. Returns [T, heads, head_dim].
///
/// Both products go through the GEMM provider: scores = Q K^T for one head, then probabilities times V. Tokens
/// a query may not see get probability exactly 0, as the oracle's -inf scores do.
template <class T>
BasicTensor<T> grouped_query_attention(const GemmProvider& provider, const BasicTensor<T>& q, const AttentionState<T>& state,
                                       std::size_t context, double scale) {
    const std::size_t tokens = q.shape().at(0);
    const std::size_t heads = q.shape().at(1);
    const std::size_t head_dim = q.shape().at(2);
    const std::size_t kv_heads = state.keys.shape().at(1);
    const std::size_t total = state.tokens;
    if (total != context + tokens) throw std::invalid_argument("grouped_query_attention: the state is not context + T tokens");
    const std::size_t group = heads / kv_heads;
    const T score_scale = static_cast<T>(scale);
    const std::vector<const AttentionState<T>*> segments = attention_segments(state);

    BasicTensor<T> output({tokens, heads, head_dim});
    BasicTensor<T> keys({total, head_dim});            // one kv head's keys, a row per visible-or-not token
    BasicTensor<T> values_transposed({head_dim, total}); // the same head's values, a row per head dim
    BasicTensor<T> q_head({tokens, head_dim});
    BasicTensor<T> scores({tokens, total});
    BasicTensor<T> attended({tokens, head_dim});

    for (std::size_t kv_head = 0; kv_head < kv_heads; ++kv_head) {
        // Read this head's keys and values through the parent chain: the parent's tensors are read, not copied
        // into a new state.
        std::size_t row = 0;
        for (const AttentionState<T>* segment : segments) {
            for (std::size_t t = 0; t < segment->own_tokens(); ++t, ++row) {
                const T* key = segment->keys.data() + (t * kv_heads + kv_head) * head_dim;
                const T* value = segment->values.data() + (t * kv_heads + kv_head) * head_dim;
                std::copy(key, key + head_dim, keys.data() + row * head_dim);
                for (std::size_t i = 0; i < head_dim; ++i) values_transposed[i * total + row] = value[i];
            }
        }
        for (std::size_t head = kv_head * group; head < (kv_head + 1) * group; ++head) {
            for (std::size_t t = 0; t < tokens; ++t) {
                const T* source = q.data() + (t * heads + head) * head_dim;
                std::copy(source, source + head_dim, q_head.data() + t * head_dim);
            }
            gemm(provider, q_head.mat(), keys.mat(), scores.mat());
            for (std::size_t t = 0; t < tokens; ++t) {
                const std::span<T> row_scores = scores.mat().row(t);
                const std::size_t visible = context + t + 1;
                for (std::size_t j = 0; j < visible; ++j) row_scores[j] *= score_scale;
                softmax_inplace(row_scores.first(visible));
                std::fill(row_scores.begin() + static_cast<std::ptrdiff_t>(visible), row_scores.end(), T(0));
            }
            gemm(provider, scores.mat(), values_transposed.mat(), attended.mat());
            for (std::size_t t = 0; t < tokens; ++t) {
                const T* source = attended.data() + t * head_dim;
                std::copy(source, source + head_dim, output.data() + (t * heads + head) * head_dim);
            }
        }
    }
    return output;
}

template <class T>
struct AttentionResult {
    BasicTensor<T> output; // [T, hidden]
    AttentionStatePtr<T> state;
};

/// The mixer on the normed layer input x [T, hidden] at `positions` [T], continuing `state`, which is only
/// read and is shared, not copied: the returned state links to it and adds this run's keys and values.
template <class T>
AttentionResult<T> attention_mixer(const GemmProvider& provider, BasicConstMatView<T> x, std::span<const double> positions,
                                   const Weights<T>& weights, const AttentionStatePtr<T>& state, const ModelConfig& config) {
    const std::size_t tokens = x.rows;
    const std::size_t head_dim = dim(config.attn_head_dim);
    const double eps = config.rms_norm_eps;
    const std::vector<std::size_t> kv_shape = {tokens, dim(config.attn_kv_heads), head_dim};

    const BasicTensor<T> q_gate = project(provider, x, weights["q_proj.weight"]);
    const BasicTensor<T> k_flat = project(provider, x, weights["k_proj.weight"]);
    BasicTensor<T> v = project(provider, x, weights["v_proj.weight"]);
    const QueryAndGate<T> unpacked = unpack_query_and_gate<T>(q_gate.mat(), config);
    const TensorD angles = rope_angles(positions, config);

    BasicTensor<T> q = rms_norm<T>(unpacked.query.as_mat(head_dim), weights["q_norm.weight"].span(), eps);
    q.reshape({tokens, dim(config.attn_heads), head_dim});
    apply_rope(q, angles);
    BasicTensor<T> k = rms_norm<T>(k_flat.as_mat(head_dim), weights["k_norm.weight"].span(), eps);
    k.reshape(kv_shape);
    apply_rope(k, angles);
    v.reshape(kv_shape);

    // The oracle concatenates the parent's keys and values with this run's. Here the new state links to the
    // parent's instead, so the parent's tensors are shared by every child and copied by none.
    auto new_state = std::make_shared<AttentionState<T>>();
    if (state->tokens != 0) new_state->parent = state;
    new_state->keys = std::move(k);
    new_state->values = std::move(v);
    new_state->tokens = state->tokens + tokens;

    BasicTensor<T> attended = grouped_query_attention(provider, q, *new_state, state->tokens, attention_scale(config));
    for (std::size_t i = 0; i < attended.size(); ++i) attended[i] *= sigmoid(unpacked.gate[i]);
    BasicTensor<T> output = project<T>(provider, attended.as_mat(dim(config.attn_width())), weights["o_proj.weight"]);
    return {std::move(output), std::move(new_state)};
}

} // namespace sepal::ref
