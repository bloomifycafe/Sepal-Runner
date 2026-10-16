// The Gated DeltaNet recurrence on the CPU, two ways that are mathematically the same function:
//
//   * `delta_scan_sequential`: the token-by-token recurrence of sepal::ref::delta_rule_scan (vectorised). It is the
//     test oracle of the chunked form and a valid production path.
//   * `delta_scan_chunked`: the chunked / WY form of flash-linear-attention's chunk_gated_delta_rule. A chunk of C
//     tokens is solved with one triangular system (pseudo values U and pseudo keys W) and a few small products, so
//     the long-sequence work is matrix multiplication instead of C dependent state updates.
//
// Per value head h, with state S [dk, dv], decay_t = exp(g_t), and token t of the chunk:
//     S'  = decay_t S;      S = S' + k_t (x) beta_t (v_t - k_t . S');      out_t = scale * q_t . S
// Chunked: with G_t = sum_{s<=t} g_s, Gam_t = exp(G_t), E_ti = exp(G_t - G_i) (never above one, so nothing overflows),
//     A_ti = beta_t E_ti (k_t . k_i)   (i < t),     (I + A) [U | W] = [beta V | beta Gam K]
//     D = U - W S_0,   out_t = scale (Gam_t q_t . S_0 + sum_{i<=t} E_ti (q_t . k_i) D_i),
//     S_C = Gam_C S_0 + sum_i E_Ci k_i (x) D_i.
//
// Both write the output rows directly and update the segment's state in place. Parallelism is over (segment, value
// head, column slice of the value dimension): the columns of S evolve independently, so a thread count changes no sum.
#pragma once

#include <cstddef>
#include <span>

#include "sepal/cpu/thread_pool.hpp"

namespace sepal::cpu {

struct DeltaGeometry {
    std::size_t key_heads = 0;
    std::size_t value_heads = 0;
    std::size_t dk = 0;  // key head dim, multiple of 8
    std::size_t dv = 0;  // value head dim, multiple of 8
};

/// One run of consecutive tokens of one request in one DeltaNet layer.
///   q, k   [rows, key_heads * dk]   rows at stride `ld_in` (already convolved, activated and L2-normalised)
///   v      [rows, value_heads * dv] rows at stride `ld_in`
///   g      [rows, value_heads]      log decay, <= 0 (contiguous rows)
///   beta   [rows, value_heads]      in (0, 1)
///   state  [value_heads, dk, dv]    carried state, updated in place
///   out    [rows, value_heads * dv] rows at stride `ld_out`
struct DeltaSegment {
    std::size_t rows = 0;
    const float* q = nullptr;
    const float* k = nullptr;
    const float* v = nullptr;
    const float* g = nullptr;
    const float* beta = nullptr;
    float* state = nullptr;
    float* out = nullptr;
};

void delta_scan_sequential(const DeltaGeometry& geometry, std::span<const DeltaSegment> segments, std::size_t ld_in,
                           std::size_t ld_out, float scale, ThreadPool& pool);

/// `chunk` is the number of tokens solved together (at most 64).
void delta_scan_chunked(const DeltaGeometry& geometry, std::span<const DeltaSegment> segments, std::size_t ld_in,
                        std::size_t ld_out, float scale, std::size_t chunk, ThreadPool& pool);

} // namespace sepal::cpu
