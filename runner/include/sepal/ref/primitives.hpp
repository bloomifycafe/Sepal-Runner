// Elementwise functions, norms and the projection helper shared by every layer of the reference path.
// A port of sepal/refengine/primitives.py, function for function.
//
// Everything in sepal::ref is a template over the element type T. With T = double it is the FP64 reference
// forward pass, compared with the NumPy oracle to 1e-9; with T = float the same code is the unoptimised float32
// path. Nothing here is tuned: clarity beats speed throughout, as in the oracle.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>

#include "sepal/gemm.hpp"
#include "sepal/tensor.hpp"

namespace sepal::ref {

/// A geometry size as an index type.
inline std::size_t dim(std::uint64_t value) { return static_cast<std::size_t>(value); }

/// log(1 + exp(x)), written so that neither a large positive nor a large negative input overflows.
template <class T>
T softplus(T x) {
    return std::max(x, T(0)) + std::log1p(std::exp(-std::fabs(x)));
}

/// 1 / (1 + exp(-x)), written through softplus so a large negative input does not overflow.
template <class T>
T sigmoid(T x) {
    return std::exp(-softplus(-x));
}

/// x * sigmoid(x), also called swish.
template <class T>
T silu(T x) {
    return x * sigmoid(x);
}

/// Softmax of one vector, in place, computed against its maximum.
template <class T>
void softmax_inplace(std::span<T> logits) {
    if (logits.empty()) throw std::invalid_argument("softmax: empty input");
    const T peak = *std::max_element(logits.begin(), logits.end());
    T sum = T(0);
    for (T& value : logits) {
        value = std::exp(value - peak);
        sum += value;
    }
    for (T& value : logits) value /= sum;
}

/// Zero-centred RMSNorm of every row: `x / sqrt(mean(x^2) + eps) * (1 + weight)`.
/// The stored weight is the offset from one, so a zero weight is the identity scale. A zero row stays zero.
template <class T>
BasicTensor<T> rms_norm(BasicConstMatView<T> x, std::span<const T> weight, double eps) {
    if (weight.size() != x.cols) throw std::invalid_argument("rms_norm: weight length differs from the row width");
    BasicTensor<T> out({x.rows, x.cols});
    const BasicMatView<T> y = out.mat();
    for (std::size_t r = 0; r < x.rows; ++r) {
        T sum_sq = T(0);
        for (std::size_t i = 0; i < x.cols; ++i) sum_sq += x(r, i) * x(r, i);
        const T rms = std::sqrt(sum_sq / static_cast<T>(x.cols) + static_cast<T>(eps));
        for (std::size_t i = 0; i < x.cols; ++i) y(r, i) = x(r, i) / rms * (T(1) + weight[i]);
    }
    return out;
}

/// One bias-free projection: x [T, in] times weight [out, in], giving [T, out], through the GEMM provider.
///
/// The oracle's `project` concatenates the weights of several projections of one input and computes one product.
/// Under the provider contract (an output depends on its input row and its weight row only) one product per
/// weight gives the same values, so the mixers call this once per weight and nothing is copied.
template <class T>
BasicTensor<T> project(const GemmProvider& provider, BasicConstMatView<T> x, const BasicTensor<T>& weight) {
    const BasicConstMatView<T> w = weight.mat();
    BasicTensor<T> out({x.rows, w.rows});
    gemm(provider, x, w, out.mat());
    return out;
}

/// Bias-free SwiGLU: `down(silu(gate(x)) * up(x))`.
template <class T>
BasicTensor<T> swiglu_mlp(const GemmProvider& provider, BasicConstMatView<T> x, const BasicTensor<T>& gate_weight,
                          const BasicTensor<T>& up_weight, const BasicTensor<T>& down_weight) {
    BasicTensor<T> gate = project(provider, x, gate_weight);
    const BasicTensor<T> up = project(provider, x, up_weight);
    for (std::size_t i = 0; i < gate.size(); ++i) gate[i] = silu(gate[i]) * up[i];
    return project<T>(provider, gate.mat(), down_weight);
}

} // namespace sepal::ref
