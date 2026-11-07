// The Gated DeltaNet mixer for long rows, in its chunked (WY) form on ordinary float32 cores: no tensor cores, no wmma, no
// mma.sync, no cp.async, no ldmatrix, so it compiles for every architecture (sm_61 and up) and is the long-row kernel of the
// Pascal profiles. Same inputs, outputs and state slots as launch_deltanet_wave (kernels_deltanet.hpp) and the same chunk
// plan and workspace convention as the tensor-core version (kernels_deltanet_chunked.hpp); the output is float32.
//
// The arithmetic is the formulation proofs/GpuTrack/GpuTrack/DeltaRule.lean proves equal to the token-serial recurrence.
// Per (row, value head) the row is cut into chunks of C = 64 tokens counted from the row's first token (the last chunk may
// be shorter, n <= 64). State S is dk x dv (key index, value index), the slot layout [value heads, dk, dv]. For the tokens
// i, j of a chunk: k_i and q_i are the L2-normalised key and query (q times the read scale 1/sqrt(dk)), v_i the value,
// beta_i = sigmoid(b_i), g_i = -exp(A_log) softplus(a_i + dt_bias) the log of the decay (used directly, never a log of the
// decay),
//
//   G_i = g_0 + g_1 + ... + g_i  (float32, in this order)      gamma_i = exp(G_i)      Gam(i, j) = exp(G_i - G_j), j <= i
//   A(i, j) = beta_i Gam(i, j) (k_i . k_j) for j < i, else 0
//   T = (I + A)^-1, unit lower triangular, by forward substitution row by row, l increasing:
//       T(i, j) = [i == j] - sum_{l = j}^{i-1} A(i, l) T(l, j)
//   W = T diag(beta gamma) K (n x dk)        U = T diag(beta) V (n x dv)        (independent of the incoming state)
//   from S0, the state at the chunk's start:
//       Y = U - W S0
//       out_i = gamma_i (q_i^T S0) + sum_{j <= i} Gam(i, j) (q_i . k_j) Y_j
//       S_end = gamma_{n-1} S0 + sum_j Gam(n-1, j) k_j Y_j^T          (S_end is the next chunk's S0)
//
// and the gated RMSNorm of out with z, exactly the scan kernel's, is the result. Kernels per wave: conv (causal convolution,
// SiLU and L2 norm of q, k, v once per chunk and head into a float32 scratch), prepare (gates, T, W, U and the masks for all
// chunks and value heads in parallel), state (the serial loop over a row's chunks, one block per (row, value head, quarter of
// the value columns)), norm; then the scan's conv-tail kernel.
//
// Numbers. Everything is float32 on the ordinary cores with explicit round-to-nearest intrinsics (the file is built with
// --fmad=false). Every sum has one fixed order per output element: dot products run over the feature, key or token index in
// increasing order into one accumulator; no atomic touches a float; nothing depends on the grid, on the other rows of the
// wave, on the row's place in the flat token list or on timing. A row's bits therefore depend only on its own tokens and its
// incoming state (the chunks start at the row's first token), so a batch changes nothing, but a row cut in two and continued
// from the first part's stored state is not bit for bit the uncut row (the chunk grid restarts at the cut), like the scan
// kernel's decay scale. Against the scan kernel the results differ by float32 rounding only (see tools/chunk_check_f32.cpp).
//
// Limits: key and value head dimensions at most 128 (shorter heads are padded with zeros inside), conv taps 1..8, float32
// inputs and output. Shared memory per block stays below 48 KB (the limit of Pascal).
#pragma once

#include <cstddef>
#include <cstdint>

#include "kernels_deltanet_chunked.hpp"

namespace sepal::cuda {

/// True when the geometry is one the float32 chunked kernels take.
bool deltanet_chunked_f32_supported(const DeltaNetGeometry& geometry);

/// Bytes of workspace for a wave of at most `chunks` chunks (deltanet_chunk_prefix) and `tokens` tokens.
std::size_t deltanet_chunked_f32_workspace_bytes(const DeltaNetGeometry& geometry, std::size_t chunks, std::size_t tokens);

/// Conv, prepare, state and norm kernels, then the conv tail. `plan` is the same DeltaNetChunkPlan as the tensor-core
/// version takes (row `p` of the wave has ceil(length / 64) chunks, numbered from chunk_base[p]). `workspace` has
/// deltanet_chunked_f32_workspace_bytes for the plan's chunks and for `workspace_tokens` tokens (the largest offset + length
/// of any row of the wave must be within it; the pointer must be 256-byte aligned, as cudaMalloc returns it). `out` is [N, value_dim] float32 with row stride `out_stride`; it also holds the
/// raw reads between the kernels (they are normalised in place), so it must not alias an input. Enqueues on `stream` and
/// returns without waiting; throws std::invalid_argument for an unsupported geometry, a null pointer or a too small
/// workspace, std::runtime_error if a launch fails.
void launch_deltanet_chunked_f32(const DeltaNetGeometry& geometry, const DeltaNetWeights& weights, const DeltaNetInputs& inputs,
                                 const DeltaNetRows& rows, const DeltaNetChunkPlan& plan, const DeltaNetStates& states,
                                 void* workspace, std::size_t workspace_bytes, std::size_t workspace_tokens, float* out,
                                 std::size_t out_stride, cudaStream_t stream);

} // namespace sepal::cuda
