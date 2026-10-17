// The Gated DeltaNet mixer, token by token. A port of sepal_runner/refengine/deltanet.py, function for function
// (that file is adapted from jaredpalmer/kev@5e42a7a kev/fused_qwen35.py, Apache-2.0).
//
// The oracle marks eight details of this mixer as not yet verified against the real model (see
// `sepal.refengine.UNVERIFIED`). Each lives in exactly one function here, marked UNVERIFIED with the oracle's
// name for it, so that when one changes there the same single function changes here.
#pragma once

#include <cmath>
#include <cstddef>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>

#include "sepal/config.hpp"
#include "sepal/gemm.hpp"
#include "sepal/ref/primitives.hpp"
#include "sepal/ref/snapshot.hpp"
#include "sepal/tensor.hpp"
#include "sepal/weights.hpp"

namespace sepal::ref {

inline constexpr double l2_eps = 1e-6;

template <class T>
struct ConvResult {
    BasicTensor<T> outputs; // [T, C]
    BasicTensor<T> tail;    // [K-1, C]
};

/// UNVERIFIED (deltanet.causal_conv): tap order, tail order, no bias.
///
/// Depthwise causal convolution continued from `tail`, without activation. inputs [T, C]; kernel [C, K];
/// tail [K-1, C], the K-1 inputs before this run, oldest first. Output t is `sum_j kernel[c, j] * window[t + j]`
/// over the window of the K inputs ending at t, so the LAST tap multiplies the current token (the
/// cross-correlation a left-padded Conv1d computes). Returns the outputs and the new tail: the last K-1 inputs
/// of tail-then-inputs.
template <class T>
ConvResult<T> causal_conv(BasicConstMatView<T> inputs, BasicConstMatView<T> kernel, BasicConstMatView<T> tail) {
    const std::size_t channels = kernel.rows;
    const std::size_t taps = kernel.cols;
    if (taps == 0) throw std::invalid_argument("causal_conv: the kernel has no taps");
    const std::size_t history = taps - 1;
    const std::size_t tokens = inputs.rows;
    if (inputs.cols != channels) throw std::invalid_argument("causal_conv: input width differs from the channels");
    if (tail.rows != history || (history != 0 && tail.cols != channels)) {
        throw std::invalid_argument("causal_conv: the tail must be [K-1, C]");
    }

    // Row i of the window: tail first, then the inputs.
    const auto window = [&](std::size_t i, std::size_t c) { return i < history ? tail(i, c) : inputs(i - history, c); };

    ConvResult<T> result{BasicTensor<T>({tokens, channels}), BasicTensor<T>({history, channels})};
    const BasicMatView<T> outputs = result.outputs.mat();
    for (std::size_t t = 0; t < tokens; ++t) {
        for (std::size_t c = 0; c < channels; ++c) {
            T acc = T(0);
            for (std::size_t j = 0; j < taps; ++j) acc += kernel(c, j) * window(t + j, c);
            outputs(t, c) = acc;
        }
    }
    const BasicMatView<T> new_tail = result.tail.mat();
    for (std::size_t i = 0; i < history; ++i) {
        for (std::size_t c = 0; c < channels; ++c) new_tail(i, c) = window(tokens + i, c);
    }
    return result;
}

/// UNVERIFIED (deltanet.l2_normalise): eps inside the root, not max(norm, eps).
///
/// `x / sqrt(sum(x^2) + eps)` over every row of `x`, in place. A zero vector stays zero.
template <class T>
void l2_normalise(BasicMatView<T> x) {
    for (std::size_t r = 0; r < x.rows; ++r) {
        T sum_sq = T(0);
        for (std::size_t i = 0; i < x.cols; ++i) sum_sq += x(r, i) * x(r, i);
        const T norm = std::sqrt(sum_sq + static_cast<T>(l2_eps));
        for (std::size_t i = 0; i < x.cols; ++i) x(r, i) /= norm;
    }
}

/// UNVERIFIED (deltanet.share_key_heads): repeat-interleave, not tiling.
///
/// [T, key heads, dk] -> [T, value heads, dk]: value head h reads key head `h / (value heads / key heads)`.
template <class T>
BasicTensor<T> share_key_heads(const BasicTensor<T>& per_key_head, const ModelConfig& config) {
    const std::size_t key_heads = dim(config.dn_key_heads);
    const std::size_t value_heads = dim(config.dn_value_heads);
    const std::size_t dk = dim(config.dn_key_head_dim);
    const std::size_t tokens = per_key_head.size() / (key_heads * dk);
    const std::size_t repeat = value_heads / key_heads;
    BasicTensor<T> shared({tokens, value_heads, dk});
    for (std::size_t t = 0; t < tokens; ++t) {
        for (std::size_t h = 0; h < value_heads; ++h) {
            const T* source = per_key_head.data() + (t * key_heads + h / repeat) * dk;
            std::copy(source, source + dk, shared.data() + (t * value_heads + h) * dk);
        }
    }
    return shared;
}

template <class T>
struct SplitHeads {
    BasicTensor<T> q; // [T, key heads, dk]
    BasicTensor<T> k; // [T, key heads, dk]
    BasicTensor<T> v; // [T, value heads, dv]
};

/// The activated conv output [T, conv_dim] as q, k [T, key heads, dk] and v [T, value heads, dv].
template <class T>
SplitHeads<T> split_heads(BasicConstMatView<T> conv_output, const ModelConfig& config) {
    const std::size_t key_dim = dim(config.dn_key_dim());
    const std::size_t value_dim = dim(config.dn_value_dim());
    const std::size_t tokens = conv_output.rows;
    const std::vector<std::size_t> key_shape = {tokens, dim(config.dn_key_heads), dim(config.dn_key_head_dim)};
    SplitHeads<T> split{BasicTensor<T>(key_shape), BasicTensor<T>(key_shape),
                        BasicTensor<T>({tokens, dim(config.dn_value_heads), dim(config.dn_value_head_dim)})};
    for (std::size_t t = 0; t < tokens; ++t) {
        const T* row = conv_output.data + t * conv_output.cols;
        std::copy(row, row + key_dim, split.q.data() + t * key_dim);
        std::copy(row + key_dim, row + 2 * key_dim, split.k.data() + t * key_dim);
        std::copy(row + 2 * key_dim, row + 2 * key_dim + value_dim, split.v.data() + t * value_dim);
    }
    return split;
}

/// The per-head decay in (0, 1]: `exp(g)` with `g = -exp(A_log) * softplus(a + dt_bias)`. a [T, H] -> [T, H].
template <class T>
BasicTensor<T> decay_gate(BasicConstMatView<T> a, std::span<const T> a_log, std::span<const T> dt_bias) {
    BasicTensor<T> decay({a.rows, a.cols});
    const BasicMatView<T> out = decay.mat();
    for (std::size_t t = 0; t < a.rows; ++t) {
        for (std::size_t h = 0; h < a.cols; ++h) out(t, h) = std::exp(-std::exp(a_log[h]) * softplus(a(t, h) + dt_bias[h]));
    }
    return decay;
}

/// UNVERIFIED (deltanet.delta_rule_update): operand order and state axes.
///
/// One token of the gated delta rule, all value heads at once, updating `state` in place.
/// state [H, dk, dv]; k [H, dk]; v [H, dv]; decay, beta [H]. The old state is decayed FIRST; the memory then
/// replaces a fraction beta of what it returns for key k by v:
/// `S' = decay * S;  S_new = S' + k (x) beta * (v - k . S')`.
template <class T>
void delta_rule_update(BasicTensor<T>& state, const T* k, const T* v, const T* decay, const T* beta, std::size_t heads,
                       std::size_t dk, std::size_t dv) {
    std::vector<T> correction(dv);
    for (std::size_t h = 0; h < heads; ++h) {
        T* s = state.data() + h * dk * dv; // this head's [dk, dv]
        const T* kh = k + h * dk;
        const T* vh = v + h * dv;
        for (std::size_t i = 0; i < dk * dv; ++i) s[i] *= decay[h];
        for (std::size_t j = 0; j < dv; ++j) {
            T recalled = T(0);
            for (std::size_t i = 0; i < dk; ++i) recalled += kh[i] * s[i * dv + j];
            correction[j] = beta[h] * (vh[j] - recalled);
        }
        for (std::size_t i = 0; i < dk; ++i) {
            for (std::size_t j = 0; j < dv; ++j) s[i * dv + j] += kh[i] * correction[j];
        }
    }
}

/// UNVERIFIED (deltanet.delta_rule_output_scale): the read is scaled by 1 / sqrt(key head dim).
inline double delta_rule_output_scale(const ModelConfig& config) {
    return 1.0 / std::sqrt(static_cast<double>(config.dn_key_head_dim));
}

template <class T>
struct ScanResult {
    BasicTensor<T> reads; // [T, H, dv]
    BasicTensor<T> state; // [H, dk, dv], after the last token
};

/// UNVERIFIED (deltanet.delta_rule_scan): a token's output reads the state AFTER its own update.
///
/// The recurrence over a run of tokens, strictly one token after another. q, k [T, H, dk]; v [T, H, dv];
/// decay, beta [T, H]; `state` [H, dk, dv] is the caller's own copy and is consumed. Each token updates the
/// state, then reads it with its own query (so a token sees its own write).
template <class T>
ScanResult<T> delta_rule_scan(const BasicTensor<T>& q, const BasicTensor<T>& k, const BasicTensor<T>& v,
                              const BasicTensor<T>& decay, const BasicTensor<T>& beta, BasicTensor<T> state, double scale) {
    const std::size_t heads = state.shape().at(0);
    const std::size_t dk = state.shape().at(1);
    const std::size_t dv = state.shape().at(2);
    const std::size_t tokens = v.size() / (heads * dv);
    const T read_scale = static_cast<T>(scale);
    BasicTensor<T> reads({tokens, heads, dv});
    for (std::size_t t = 0; t < tokens; ++t) {
        delta_rule_update(state, k.data() + t * heads * dk, v.data() + t * heads * dv, decay.data() + t * heads,
                          beta.data() + t * heads, heads, dk, dv);
        for (std::size_t h = 0; h < heads; ++h) {
            const T* s = state.data() + h * dk * dv;
            const T* qh = q.data() + (t * heads + h) * dk;
            T* out = reads.data() + (t * heads + h) * dv;
            for (std::size_t j = 0; j < dv; ++j) {
                T read = T(0);
                for (std::size_t i = 0; i < dk; ++i) read += qh[i] * s[i * dv + j];
                out[j] = read_scale * read;
            }
        }
    }
    return {std::move(reads), std::move(state)};
}

/// UNVERIFIED (deltanet.gated_norm_scale): the stored weight itself, NOT `1 + weight` like the other norms.
template <class T>
T gated_norm_scale(T weight) {
    return weight;
}

/// UNVERIFIED (deltanet.gated_norm_gate): `silu(z)` (Kev names the activation "swish"), not sigmoid(z).
template <class T>
T gated_norm_gate(T z) {
    return silu(z);
}

/// RMSNorm of each value head's read, scaled by the norm weight and gated by z. x, z [T * H, dv] as rows.
template <class T>
BasicTensor<T> gated_rms_norm(BasicConstMatView<T> x, BasicConstMatView<T> z, std::span<const T> weight, double eps) {
    if (weight.size() != x.cols || z.rows != x.rows || z.cols != x.cols) {
        throw std::invalid_argument("gated_rms_norm: x, z and the weight do not agree in shape");
    }
    BasicTensor<T> out({x.rows, x.cols});
    const BasicMatView<T> y = out.mat();
    for (std::size_t r = 0; r < x.rows; ++r) {
        T sum_sq = T(0);
        for (std::size_t i = 0; i < x.cols; ++i) sum_sq += x(r, i) * x(r, i);
        const T rms = std::sqrt(sum_sq / static_cast<T>(x.cols) + static_cast<T>(eps));
        for (std::size_t i = 0; i < x.cols; ++i) y(r, i) = x(r, i) / rms * gated_norm_scale(weight[i]) * gated_norm_gate(z(r, i));
    }
    return out;
}

template <class T>
struct DeltaNetResult {
    BasicTensor<T> output; // [T, hidden]
    DeltaNetStatePtr<T> state;
};

/// The mixer on the normed layer input x [T, hidden], continuing `state`, which is only read.
/// Returns the mixer output [T, hidden] and the state after the last token.
template <class T>
DeltaNetResult<T> deltanet_mixer(const GemmProvider& provider, BasicConstMatView<T> x, const Weights<T>& weights,
                                 const DeltaNetState<T>& state, const ModelConfig& config) {
    const std::size_t dv = dim(config.dn_value_head_dim);
    const BasicTensor<T> mixed = project(provider, x, weights["in_proj_qkv.weight"]);
    const BasicTensor<T> z = project(provider, x, weights["in_proj_z.weight"]);
    BasicTensor<T> b = project(provider, x, weights["in_proj_b.weight"]);
    const BasicTensor<T> a = project(provider, x, weights["in_proj_a.weight"]);

    // conv1d.weight is [channels, 1, taps]: the same values as a [channels, taps] matrix.
    ConvResult<T> conv = causal_conv<T>(mixed.mat(), weights["conv1d.weight"].as_mat(dim(config.conv_kernel)), state.conv_tail.mat());
    for (T& value : conv.outputs.span()) value = silu(value);
    SplitHeads<T> split = split_heads<T>(conv.outputs.mat(), config);
    l2_normalise(split.q.as_mat(dim(config.dn_key_head_dim)));
    l2_normalise(split.k.as_mat(dim(config.dn_key_head_dim)));
    const BasicTensor<T> q = share_key_heads(split.q, config);
    const BasicTensor<T> k = share_key_heads(split.k, config);

    const BasicTensor<T> decay = decay_gate<T>(a.mat(), weights["A_log"].span(), weights["dt_bias"].span());
    for (T& value : b.span()) value = sigmoid(value); // beta
    ScanResult<T> scan = delta_rule_scan(q, k, split.v, decay, b, state.recurrent, delta_rule_output_scale(config));

    const BasicTensor<T> normed = gated_rms_norm<T>(scan.reads.as_mat(dv), z.as_mat(dv), weights["norm.weight"].span(), config.rms_norm_eps);
    BasicTensor<T> output = project<T>(provider, normed.as_mat(dim(config.dn_value_dim())), weights["out_proj.weight"]);

    auto new_state = std::make_shared<DeltaNetState<T>>();
    new_state->recurrent = std::move(scan.state);
    new_state->conv_tail = std::move(conv.tail);
    return {std::move(output), std::move(new_state)};
}

} // namespace sepal::ref
