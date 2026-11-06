// Flash-style tiled causal grouped-query attention for the CPU trunk (head_dim a multiple of 32, 256 at 0.8B).
//
// The keys and values of a request live in a `KvCache` packed in the micro-kernel panel layout, so that
// S = Q K^T and O += P V are both done by the packed FMA micro kernels of the GEMM (no softmax matrix is ever
// materialised beyond one 128-key tile). Per (segment, kv head, block of query tokens) the kernel keeps the online
// softmax state (row max, row sum, rescaled output) for all query heads of the group at once, so a K or V panel is
// read once per group, not once per head. Each query row sees its keys in the same absolute 128-key tiles
// whatever the batch, so a row's result does not depend on the other rows or on the thread count.
#pragma once

#include <cstddef>
#include <span>
#include <string>

#include "sepal/cpu/aligned.hpp"
#include "sepal/cpu/isa.hpp"
#include "sepal/cpu/thread_pool.hpp"

namespace sepal::cpu {

inline constexpr std::size_t attention_key_tile = 128;

/// Keys and values of one request in one full-attention layer (float32), zero initialised.
class KvCache {
public:
    KvCache(std::size_t kv_heads, std::size_t head_dim, std::size_t capacity_tokens, Isa isa);
    std::size_t kv_heads() const { return kv_heads_; }
    std::size_t head_dim() const { return head_dim_; }
    std::size_t capacity() const { return capacity_; }   // rounded up to a multiple of the key tile
    std::size_t nr() const { return nr_; }               // panel width of the micro kernel
    std::size_t bytes() const { return (k_.size() + v_.size()) * sizeof(float); }

    /// Copies the keys and values of tokens [0, tokens) from `src` (same kv heads, head dim and key tile; any capacity of at least `tokens`).
    void copy_prefix_from(const KvCache& src, std::size_t tokens);

    /// Stores the key and value of token `pos` for kv head `kvh` (head_dim floats each).
    void append(std::size_t pos, std::size_t kvh, const float* k, const float* v);

    const float* key_panel(std::size_t kvh, std::size_t panel) const { return k_.data() + (kvh * (capacity_ / nr_) + panel) * head_dim_ * nr_; }
    const float* value_panel(std::size_t kvh, std::size_t dim_panel) const { return v_.data() + (kvh * (head_dim_ / nr_) + dim_panel) * capacity_ * nr_; }

private:
    std::size_t kv_heads_, head_dim_, capacity_, nr_;
    AlignedBuffer<float> k_, v_;
};

struct AttentionGeometry {
    std::size_t heads = 0;
    std::size_t kv_heads = 0;
    std::size_t head_dim = 0;
};

/// One run of consecutive query tokens of one request. Token i of the segment is at position start + i; the cache
/// must already hold the keys and values of positions [0, start + rows).
///   q     [rows, heads * head_dim]   queries after norm and RoPE, ALREADY multiplied by 1 / sqrt(head_dim)
///   gate  rows at stride ld_gate, head h at offset h * gate_head_stride: the output gate (before the sigmoid)
///   out   [rows, heads * head_dim]   sigmoid(gate) * softmax(q k^T) v
struct AttentionSegment {
    const KvCache* cache = nullptr;
    std::size_t start = 0;
    std::size_t rows = 0;
    const float* q = nullptr;
    const float* gate = nullptr;
    float* out = nullptr;
};

/// The attention kernel. Tiled: the 0.4 kernel. Flash: the same online softmax with every buffer in the micro-kernel tile
/// layout (no copies between the products), O accumulated inside the micro kernel, panel-outer loops and the last key tile
/// cut to the visible keys; not bit-identical to Tiled, held to the cpu-bf16 parity gates (RUNNER.md 14).
/// Default resolves to SEPAL_ATTENTION (tiled|flash) or the built-in choice.
enum class AttentionKernel { Default, Tiled, Flash };
const char* name_of(AttentionKernel k);
AttentionKernel parse_attention_kernel(const std::string& name); // throws std::invalid_argument
/// Process-wide override (bench and tests); Default restores the environment / built-in choice.
void set_attention_kernel(AttentionKernel k);
/// The kernel attention_forward uses now (never Default).
AttentionKernel attention_kernel();

void attention_forward(const AttentionGeometry& geometry, Isa isa, std::span<const AttentionSegment> segments, std::size_t ld_q,
                       std::size_t ld_gate, std::size_t gate_head_stride, std::size_t ld_out, ThreadPool& pool);

} // namespace sepal::cpu
