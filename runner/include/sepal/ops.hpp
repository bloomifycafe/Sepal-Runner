// The small exact operations of the model (docs/RUNNER.md section 3). Scalar, single-threaded, float64
// intermediates rounded to float32 once per output. All shape errors throw std::invalid_argument.
#pragma once

#include <span>

#include "sepal/gemm.hpp"
#include "sepal/tensor.hpp"

namespace sepal {

/// Zero-centred RMSNorm, per row: y = x / sqrt(mean(x^2) + eps) * (1 + w).
/// The stored weight `w` is centred on zero, so the scale applied is `1 + w`, not `w`. y may be x.
void rmsnorm(ConstMatView x, std::span<const float> w, double eps, MatView y);

/// SiLU (swish): x * sigmoid(x).
float silu(float x);
void silu_inplace(std::span<float> x);

/// Bias-free SwiGLU MLP: y = beta * y + down(silu(gate(x)) * up(x)).
///
/// `w_gate_up` is [2 I, H]: the gate rows first, then the up rows, so gate and up are one GEMM.
/// `w_down` is [H, I]. x is [T, H]; y is [T, H]. With beta == 1 and y preloaded with the residual stream this
/// is the second half of a decoder layer. y must not overlap x.
void swiglu_mlp(const GemmProvider& gemm, ConstMatView x, MatRef w_gate_up, MatRef w_down, MatView y, float beta);

/// Softmax over one vector, computed against the maximum so large logits do not overflow. probs may be logits.
void softmax(std::span<const float> logits, std::span<float> probs);

/// Depthwise causal convolution over a sequence, continued from a carried tail.
///
///   x        [T, C]    the new inputs, oldest first
///   w        [C, K]    one K-tap filter per channel; tap K-1 multiplies the current input, tap 0 the oldest
///                      (the layout of a PyTorch Conv1d weight [C, 1, K] with left padding K-1)
///   tail_in  [K-1, C]  the K-1 inputs that preceded x, oldest first; all zeros at the start of a sequence
///   y        [T, C]    y[t, c] = sum over j of w[c, j] * in[t + j, c], where in = concat(tail_in, x)
///   tail_out [K-1, C]  the last K-1 rows of concat(tail_in, x); pass an empty view to skip it
///
/// Running a sequence in pieces with the tail carried over gives the same bits as running it whole.
/// tail_out may be tail_in. y must not overlap x.
void causal_conv1d_depthwise(ConstMatView x, ConstMatView w, ConstMatView tail_in, MatView y, MatView tail_out);

} // namespace sepal
