// The CPU trunk: Qwen3.5 (Gated DeltaNet + full attention) prefill on the packed kernels.
//
// Requests are cut into prefill chunks (`chunk_tokens`); a step runs the next chunk of one or several requests
// through all layers, carrying the DeltaNet state, the conv tail and the K/V cache of each request, so only
// chunk-sized activations are alive whatever the request length. The result per request is the final-normed hidden
// state of EVERY position (the Clef head reads them all).
//
// Determinism: every output element is computed by one task with a fixed order of operations, so the result is the
// same for any thread count; a request's hidden states do not depend on the other requests of the call.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "sepal/config.hpp"
#include "sepal/cpu/aligned.hpp"
#include "sepal/cpu/attention.hpp"
#include "sepal/cpu/isa.hpp"
#include "sepal/cpu/thread_pool.hpp"
#include "sepal/weights.hpp"

namespace sepal::cpu {

/// Precision profile of the trunk's matrix multiplies.
enum class Profile { F32, BF16, Q8 };
const char* name_of(Profile p);

/// Which weight groups become int8 in the Q8 profile (the others stay bf16). The DeltaNet out_proj, in_proj_a and
/// in_proj_b are never int8 (RUNNER.md 11 keeps them at 16 bits).
struct Q8Groups {
    bool mlp = true;          // gate, up, down
    bool gate_up = false;     // gate and up alone (the option string "gate_up"); "down" likewise; "mlp" means all three
    bool attention = true;    // q, k, v, o
    bool deltanet_in = true;  // in_proj_qkv, in_proj_z
    bool down = false;
};

struct TrunkOptions {
    Profile profile = Profile::F32;
    Q8Groups q8;
    unsigned threads = 0;                    // 0: hardware concurrency
    std::size_t chunk_tokens = 2048;         // prefill chunk of one request
    std::size_t step_tokens = 4096;          // rows processed together in one layer pass
    std::size_t delta_chunk = 64;            // DeltaNet chunk (WY form); 0 runs the token-sequential recurrence
    std::size_t state_budget_bytes = 8ull << 30; // request state (K/V caches, DeltaNet states) alive at once
    bool force_isa = false;                  // use `isa` instead of the best one
    Isa isa = Isa::Avx2;
    std::string cpus;                        // CPU list for the thread pool ("0-11"); empty: all allowed (or SEPAL_CPU_SET)
};

/// Wall-clock milliseconds of the last `forward`, by kind of work.
struct TrunkTimes {
    double gemm_ms = 0, delta_ms = 0, attention_ms = 0, other_ms = 0, total_ms = 0;
    std::uint64_t tokens = 0, steps = 0;
};

/// Called after every layer for every segment of a step: (request, layer, first position, rows, x [rows, hidden]).
using LayerObserver = std::function<void(std::size_t, std::size_t, std::size_t, std::size_t, const float*)>;

/// The state of one request after its first `tokens` tokens: DeltaNet states, conv tails, K/V caches and the final-normed hidden state of every
/// position so far. A request that starts with the same tokens resumes from it and only computes the rest.
struct PrefixState {
    std::size_t tokens = 0;
    std::vector<AlignedBuffer<float>> dn_state, conv_tail;
    std::vector<std::unique_ptr<KvCache>> kv;
    std::vector<float> hidden;   // [tokens, hidden size]
    std::uint64_t bytes() const;
};

/// Per request of a `forward` call: resume from `from` (the request's first `from->tokens` tokens must be the ones the state was made from),
/// and/or leave the state after the request's last token in `capture`.
struct Resume {
    const PrefixState* from = nullptr;
    PrefixState* capture = nullptr;
};

class CpuTrunk {
public:
    /// Fills `out` with the embedding row of a token (hidden_size floats). Must be thread safe.
    using EmbeddingFn = std::function<void(std::uint32_t, float*)>;

    /// Packs the body of `weights` (every tensor except the embedding and the pointer head) for the profile. The
    /// head dimensions must be multiples of 8 (attention head_dim a multiple of 32) and the hidden size a multiple of 8.
    CpuTrunk(const ModelConfig& config, const Weights<float>& weights, EmbeddingFn embed, const TrunkOptions& options);

    /// The same, asking `source` for one part at a time ("layers.3", then "norm"; each answer addressed relative to the scope). Each part is
    /// packed and then released, so peak host memory is the packed weights plus one layer in float32, not the whole model in float32.
    using WeightSource = std::function<Weights<float>(const std::string&)>;
    CpuTrunk(const ModelConfig& config, const WeightSource& source, EmbeddingFn embed, const TrunkOptions& options);
    ~CpuTrunk();
    CpuTrunk(const CpuTrunk&) = delete;
    CpuTrunk& operator=(const CpuTrunk&) = delete;

    /// Runs the requests. `hidden[i]` is resized to [tokens_i, hidden_size] and filled with the final-normed hidden
    /// states. Throws std::invalid_argument for an empty request or a token outside the vocabulary.
    void forward(std::span<const std::span<const std::uint32_t>> requests, std::vector<std::vector<float>>& hidden,
                 const LayerObserver* observer = nullptr, const std::vector<Resume>* resume = nullptr);

    const ModelConfig& config() const;
    const TrunkOptions& options() const;
    Isa isa() const;
    unsigned threads() const;
    ThreadPool& pool();
    std::uint64_t weight_bytes() const;
    const TrunkTimes& last_times() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace sepal::cpu
