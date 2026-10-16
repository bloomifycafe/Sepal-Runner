// The Gated DeltaNet mixer for long rows, in its chunked (WY) form on the tensor cores. Same inputs, outputs and state
// slots as launch_deltanet_wave (kernels_deltanet.hpp), different arithmetic: the token-serial recurrence of the scan
// kernel is cut into chunks of 64 tokens. Per chunk (tokens i = 1..n, decay gamma_i = prod of the chunk's decays up to i,
// S0 the state at the chunk's start, K, Q the L2-normalised keys and queries, Q already times the read scale):
//
//   A = diag(beta) tril(K K^T (gamma_i / gamma_j), -1)         T = (I + A)^-1
//   W = T diag(beta gamma) K          U = T diag(beta) V        (both independent of S0: computed for every chunk in parallel)
//   Y = U - W S0                                                (the chunk's corrections, in the decay-scaled form)
//   out = diag(gamma) Q S0 + tril(Q K^T (gamma_i / gamma_j)) Y
//   S_end = gamma_n S0 + (K diag(gamma_n / gamma))^T Y
//
// Only S_end depends on the previous chunk, so the serial part is one short loop per (row, head, slice of the value
// columns) of three small matrix products per chunk; everything else runs in parallel over chunks. Three kernels per
// wave: prepare (convolution, SiLU, L2 norm, gates, T, W, U and the masks, per chunk and head), state (the loop over
// chunks), and the gated RMSNorm; the conv tail kernel of the scan follows as before.
//
// Numbers. Products run on binary16 tensor cores with float32 accumulation; the state and Y are split into a high and a
// low binary16 part (so they keep about 21 bits); the chunk matrices W, U, M, Q, K are single binary16. A row's results
// depend on its own tokens and its starting state only (the chunks start at the row's first token), so a batch changes
// nothing, but a row cut in two and continued from the first part's state is not bit for bit the uncut row (the chunk
// grid restarts at the cut), like the scan kernel's decay scale. Measured against the scan kernel in the tests.
//
// Limits: key and value head dimension 128, conv taps at most 8, outputs binary16, binary16 or float32 inputs.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "kernels_deltanet.hpp"

namespace sepal::cuda {

inline constexpr std::uint32_t kDeltaNetChunk = 64;

/// True when the geometry is one the chunked kernels take.
bool deltanet_chunked_supported(const DeltaNetGeometry& geometry);

/// The chunks of a wave: row `p` of the wave has ceil(length / 64) chunks, numbered from chunk_base[p].
struct DeltaNetChunkPlan {
    const std::uint32_t* chunk_base = nullptr; // device, [wave_size + 1]: prefix sums of the chunk counts
    std::uint32_t total_chunks = 0;
};

/// Host side: the prefix sums for rows of the given lengths.
std::vector<std::uint32_t> deltanet_chunk_prefix(const std::vector<std::uint32_t>& lengths);

/// Bytes of workspace for a wave of at most `chunks` chunks and `tokens` tokens.
std::size_t deltanet_chunked_workspace_bytes(const DeltaNetGeometry& geometry, std::size_t chunks, std::size_t tokens);

/// Prepare, state and norm kernels, then the conv tail. `workspace` has deltanet_chunked_workspace_bytes for the plan's
/// chunks and for `workspace_tokens` tokens (the largest offset + length of any row of the wave must be within it).
template <class InT>
void launch_deltanet_chunked(const DeltaNetGeometry& geometry, const DeltaNetWeights& weights, const DeltaNetInputsT<InT>& inputs,
                             const DeltaNetRows& rows, const DeltaNetChunkPlan& plan, const DeltaNetStates& states, void* workspace,
                             std::size_t workspace_bytes, std::size_t workspace_tokens, __half* out, std::size_t out_stride,
                             cudaStream_t stream);

} // namespace sepal::cuda
