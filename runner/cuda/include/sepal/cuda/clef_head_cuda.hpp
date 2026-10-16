// The Clef JointSchemaHead on the GPU (math: sepal/clef_head.hpp and Cloudflare's Apache-2.0 joint_schema_model.py).
//
// It reads the final-normed hidden state of every position while it is still on the device, for many requests at once:
//
//   begin(requests)            plans a batch: spans, option tokens, offsets; nothing runs yet
//   consume(...) / adopt(...)  feeds hidden states of tokens (the trunk, in one or several calls) or cached memory rows
//   finish()                   routing layers, decoder layers, scorer; returns the raw logits of every request
//
// Layout. The memory of a request (memory_projection(LayerNorm(h)), [tokens, width] float32) lives in one arena, a request after
// another with no padding. Everything that has few rows (options, fields) is batched over all requests of the batch: one
// multiply per layer for all of them. Only the attention of those few queries over the long memory is per request.
//
// Few queries, long memory. For the few queries q of a cross-attention the keys and values of the memory are never
// materialised. With K = M Wk^T + bk and V = M Wv^T + bv per head, the scores are (Wk_h^T q_h) . M[t] (+ a constant per query,
// which the softmax removes), and the output is Wv_h (sum_t p_t M[t]) + bv_h because the probabilities sum to one. So each
// layer costs two multiplies of [queries x 512] by [512 x L] instead of two of [L x 512] by [512 x 512]. The routing layers
// attend to LayerNorm(memory) = gamma * n + beta (n the normalised memory): gamma is folded into the query side and beta
// into the output side, n is kept in a second arena.
//
// Precision. Weights and arithmetic are float32 (cuBLAS float32 multiplies, no TF32); sums of norms, softmaxes and
// means are accumulated in double and rounded once.
//
// Threads: not thread safe; the caller (the engine adapter) serialises batches. All work is on the stream given at creation.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "sepal/clef_head.hpp"
#include "sepal/decision.hpp"

namespace sepal::cuda {

struct ClefHeadCapacity {
    std::size_t memory_tokens = 131072;  // tokens of all requests of one batch (two float32 arenas of width 512 each)
    std::size_t call_tokens = 16384;     // tokens one consume() may carry
    std::size_t requests = 512;
    std::size_t fields = 2048;           // questions of one batch
    std::size_t options = 4096;          // options of one batch
    std::size_t option_tokens = 262144;  // tokens of all option spans of one batch
    std::size_t unique_tokens = 32768;   // distinct tokens among them
    std::size_t score_floats = std::size_t{1} << 25; // scratch of the attention scores (heads * queries * L per piece; 128 MiB)
};

/// A piece of hidden states: tokens [first_token, first_token + count) of request `request` are rows
/// [row, row + count) of the matrix handed to consume().
struct HeadSegment {
    std::size_t request = 0;
    std::size_t first_token = 0;
    std::size_t count = 0;
    std::size_t row = 0;
};

/// Device memory the head takes for a capacity (weights, arenas, scratch), with a small margin, in bytes.
std::size_t clef_head_device_bytes(const ClefHeadConfig& config, const ClefHeadCapacity& capacity);

class CudaClefHead {
public:
    /// `stream` is a cudaStream_t (as void*, to keep CUDA headers out of this one). Uploads the weights.
    CudaClefHead(const ClefHeadWeights& weights, const ClefHeadCapacity& capacity, void* stream);
    ~CudaClefHead();
    CudaClefHead(const CudaClefHead&) = delete;
    CudaClefHead& operator=(const CudaClefHead&) = delete;

    const ClefHeadConfig& config() const;
    const ClefHeadCapacity& capacity() const;

    /// Plans a batch. Every request's spans and sizes are checked (std::invalid_argument naming the problem; nothing is
    /// changed then, and a batch that exceeds the capacity is refused the same way). The lexical rows are gathered here
    /// from the host (`embedding_row`). The requests must stay alive until finish().
    void begin(std::span<const EncodedRequest* const> requests, EmbeddingRowFn embedding_row, const void* context);

    /// Feeds hidden states: `hidden` is a device matrix [rows, hidden_size] of final-normed states (float32). Each segment
    /// names where its tokens belong. Enqueued on the stream.
    void consume(const float* hidden, std::size_t rows, std::span<const HeadSegment> segments);

    /// Installs memory rows computed earlier (device pointer to [count, width], e.g. from memory_rows of a previous batch)
    /// for tokens [0, count) of a request.
    void adopt(std::size_t request, const float* memory, std::size_t count);

    /// Device pointer to the memory rows of a request (its token 0), valid until the next begin().
    const float* memory_rows(std::size_t request) const;
    std::size_t width() const;

    /// Runs everything after the memory is complete and returns the logits of every request, in order. Waits.
    std::vector<RequestLogits> finish();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace sepal::cuda
