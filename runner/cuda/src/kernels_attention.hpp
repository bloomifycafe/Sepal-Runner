// The non-projection work of the full-attention mixer on the device: per-head RMSNorm of q and k, partial RoPE,
// storing the row's keys and values, causal grouped-query attention over the row's parent chain, and the output
// gate. A port of `attention_mixer` in sepal/ref/attention.hpp from the moment the three projections have been
// computed to the moment `o_proj` is about to be. Conventions: device.hpp.
//
// Two launches per layer, both over many rows at once:
//
//   launch_attention_prepare   q, k, v bands of the multiply output -> normed and rotated q; the row's keys and
//                              values written into its own segment (keys after norm and RoPE).
//   launch_attention           q, the segments, the gate band -> attention output times sigmoid(gate), in the
//                              element type the `o_proj` multiply takes.
//
// Because the prepare launch writes every row's keys and values before any row attends, a branch row whose
// context is another row of the same pass needs no wave ordering here: launch prepare for all rows, then attention
// for all rows.
//
// K/V layout. A segment is a run of consecutive tokens' keys and values, [tokens, kv_heads, head_dim] row-major
// each, in the reference's own layout, so a snapshot's gather is a concatenation. The element type is float32 on
// the F32 path and binary16 on the F16 path. Segments are never copied: a row lists the segments of its parent
// chain, oldest first, and last its own, which the prepare launch fills. Eight branches that continue one state
// all list the state's segment and read it in place.
//
// Segment description. Three device arrays, indexed as follows (all of them are built by the executor):
//   spans[row]      RowSpan of the row's tokens in the flat list.
//   descs[row]      AttentionRowDesc: where the row's segments start in `segments`, how many (context segments
//                   plus its own), and the position of its first token.
//   segments[i]     KvSegment: keys pointer, values pointer, token count.
// The row's tokens are positions start_position + t. Token t of a row attends to the keys of all its context
// segments and to its own keys 0..t. The row's own segment is the last of its list and holds exactly
// spans[row].length tokens (check_attention_rows tests this on the host).
//
// The q buffer between the two launches has the K/V element type, [N, heads * head_dim] with a row stride.
//
// Determinism. Every output element has one reduction order that depends on the row's own length and context
// only: scores, softmax and the weighted sum run over the row's keys in order, in key blocks counted from the
// row's first context token (the blocks do not depend on where one segment ends and the next begins). No other
// row, and no place in the flat list, enters. Both paths keep this; their bits differ from each other.
//
// Paths. `Reference` is a plain kernel, one block per (token, query head), for both element types and any
// head_dim up to 256. `Tiled` is a tensor-core kernel (mma.sync, fp32 accumulation) for F16 only: a block
// takes 64 rows of (token, head of one KV group) of one row and walks the keys 32 at a time with an online
// softmax. It needs head_dim 16, 64, 128 or 256 and a group size heads / kv_heads of 1, 2, 4, 8 or 16.
// Alignment the tiled kernel relies on (cp.async, 16 bytes): segment pointers and the q buffer are 16-byte
// aligned and the q row stride is a multiple of 8 elements.
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "device.hpp"

namespace sepal::cuda {

/// Element type of a device matrix handed to or from the attention launches.
enum class AttnElem : int { F32 = 0, F16 = 1 };

enum class AttentionPath : int {
    Reference = 0, ///< the plain kernel: any head_dim up to 256, either element type
    Tiled = 1,     ///< tensor cores, F16 only; throws when the geometry is not supported
    Auto = 2,      ///< Tiled when supported, otherwise Reference
};

/// Which tensor-core kernel the Tiled path runs (head_dim 64 and up; a 16-wide head always takes the first). Chosen once per
/// process by the environment variable SEPAL_ATTENTION_KERNEL (v1, v2, v2h); unset means SecondPv16.
enum class TiledKernel : int {
    First = 0,      ///< v1: 4 warps, 64 query rows, one buffer for a tile's keys and then its values
    Second = 1,     ///< v2: 8 warps, 128 query rows, separate key and value buffers; P V in float32
    SecondPv16 = 2, ///< v2h: the same with each tile's P V on binary16 accumulators, added into float32
};
TiledKernel tiled_kernel_choice();

/// The attention geometry, read from the model configuration.
struct AttentionShape {
    std::uint32_t heads;       ///< query heads
    std::uint32_t kv_heads;    ///< key and value heads; divides heads (query head h reads KV head h / (heads / kv_heads))
    std::uint32_t head_dim;
    std::uint32_t rotary_dims; ///< the first rotary_dims of each head rotate (even, at most head_dim)
    double rope_theta;
    double rms_norm_eps;
};

/// A run of consecutive tokens' keys and values: [tokens, kv_heads, head_dim] each, in the K/V element type.
struct KvSegment {
    void* keys;
    void* values;
    std::uint32_t tokens;
    std::uint32_t before; ///< tokens in the row's earlier segments (the sum of `tokens` over the segments before this one); read only when the row is `indexed`
};

struct AttentionRowDesc {
    std::uint32_t first_segment;  ///< index of the row's oldest segment in the segment array
    std::uint32_t segment_count;  ///< context segments plus the row's own, which comes last; at least 1
    std::uint32_t start_position; ///< position of the row's first token
    /// 1: every segment of the row carries `before`, and a kernel finds the segment of a key by binary search. 0: `before` is
    /// not set, and the kernel scans the segments from the first, which costs O(segments) per key (O(S^2) for a query over a
    /// chain of S one-token links: 60 ms at 1,000 links). Both give the same bits; `check_attention_rows` verifies `before`.
    std::uint32_t indexed;
};

/// The device tables of one launch. `run_rows` (device, may be null) lists the rows to run, in any order; null
/// runs rows 0 .. run_count - 1. `max_row_tokens` is the largest length among the rows that run (a host value
/// that sizes the grid).
struct AttentionRows {
    const RowSpan* spans;
    const AttentionRowDesc* descs;
    const KvSegment* segments;
    const std::uint32_t* run_rows;
    std::uint32_t run_count;
    std::uint32_t max_row_tokens;
};

/// A band of columns of a float32 matrix: element (token, c) is at data[token * stride + c].
struct FloatBand {
    const float* data;
    std::size_t stride;
};

/// A matrix of the K/V element type (q) or of the o_proj operand type (output), with a row stride in elements.
struct ElemMatrix {
    void* data;
    std::size_t stride;
};
struct ConstElemMatrix {
    const void* data;
    std::size_t stride;
};

/// Which of the two jobs of the prepare launch to do. A pass that needs the queries of only some of the tokens (the last
/// layer, where only the tokens that are read out matter) stores every token's keys and values with KeysValuesOnly and
/// prepares those tokens' queries separately with QueriesOnly: the bands and q_out that a part does not use may be null.
enum class PreparePart : int { All = 0, QueriesOnly = 1, KeysValuesOnly = 2 };

/// Per-head norm and RoPE of q and k, then stores k and v.
///   q_gate  [N, 2 * heads * head_dim]: per head, the query then its gate (the gate is read by launch_attention).
///   k, v    [N, kv_heads * head_dim].
///   q_norm, k_norm  device float32 [head_dim]: the stored RMSNorm weights ("1 + weight" form).
///   q_out   [N, heads * head_dim] in `kv_type`: the normed, rotated queries.
/// The row's keys and values go to its own (last) segment.
void launch_attention_prepare(const AttentionShape& shape, const AttentionRows& rows, AttnElem kv_type, FloatBand q_gate,
                              FloatBand k, FloatBand v, const float* q_norm, const float* k_norm, ElemMatrix q_out,
                              cudaStream_t stream, PreparePart part = PreparePart::All);

/// Causal grouped-query attention of every token, multiplied by sigmoid(gate); `gate` is the same band as the
/// q_gate of the prepare launch (it is read at column head * 2 * head_dim + head_dim). out is [N, heads * head_dim]
/// in `out_type`. Returns the path that ran (Reference or Tiled).
AttentionPath launch_attention(const AttentionShape& shape, const AttentionRows& rows, AttnElem kv_type,
                               AttentionPath path, ConstElemMatrix q, FloatBand gate, ElemMatrix out,
                               AttnElem out_type, cudaStream_t stream);

/// True when the tiled kernel supports this geometry and K/V type.
bool attention_tiled_supported(const AttentionShape& shape, AttnElem kv_type);

/// Host-side check of the tables before they are uploaded: every row has at least one segment, its segments lie
/// inside the array, its own (last) segment holds exactly as many tokens as the row, and positions do not
/// overflow. Throws std::invalid_argument naming the row.
void check_attention_rows(std::span<const RowSpan> spans, std::span<const AttentionRowDesc> descs,
                          std::span<const KvSegment> segments);

} // namespace sepal::cuda
