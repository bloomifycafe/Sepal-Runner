// sepal::DecisionEngine on the GPU: the trained model (LoRA merged in float32, rounded once to the profile's storage type on
// upload), the CUDA trunk (CudaEngine) and the CUDA Clef head (CudaClefHead), joined without a host round trip of the hidden states.
//
// Profiles: "gpu-f16" (binary16 weights and activations, the default) and "gpu-f32" (float32 throughout; the parity profile).
//
// A call of `score` takes many requests. They are cut into groups that fit the head's capacity; in a group every request is one
// row of the trunk (a request longer than one call's token budget is cut into rows that continue each other's snapshots), the
// rows of up to `max_call_tokens` tokens go through the trunk together, and the final-normed hidden state of every token goes from
// the trunk's buffer into the head on the device. Only the logits come back.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "sepal/cuda/clef_head_cuda.hpp"
#include "sepal/cuda/cuda_engine.hpp"
#include "sepal/decision.hpp"
#include "sepal/trained_model.hpp"

namespace sepal::cuda {

struct GpuDecisionOptions {
    std::string profile = "gpu-f16"; // or "gpu-f32"
    int device = 0;
    /// The longest request. The trunk's call buffers are sized for max_call_tokens, not for this: a longer request is cut.
    std::uint64_t max_tokens = 32768;
    /// Tokens of one trunk call. 0: chosen from the free device memory (65536 from 20 GiB, 32768 from 10 GiB, else 16384). Several long rows
    /// in one call fill the card's SMs (a Gated DeltaNet scan is one block per row and head), so larger is faster for long states.
    std::size_t max_call_tokens = 0;
    std::size_t max_call_rows = 1024;
    ClefHeadCapacity head;
    /// Device memory the trunk may take (weights, call buffers, pools); 0: what is free, less the head and a margin.
    std::size_t trunk_budget_bytes = 0;
    /// Trunk multiply configuration; see EngineOptions.
    MultiplyPath multiply_path = MultiplyPath::Auto;
    Fp16Accumulate fp16_accumulate;
    bool half_state = false;
    std::size_t gemm_rows = 0;
    /// int8 weights for chosen multiply groups of the gpu-f16 trunk (off by default; see the int8 section of README.md).
    Int8Multiply int8;
    /// The Gated DeltaNet layers: scan kernel, chunked tensor-core kernels, or a per-wave choice (see DeltaNetMode). gpu-f16 only.
    DeltaNetMode deltanet = DeltaNetMode::Auto;
    /// Prefix reuse (RUNNER.md section 7): a snapshot of the trunk plus the head's memory rows at a cut before the first question
    /// span. The cut is the last occurrence of `schema_marker` (token ids; empty: the longest common prefix with a recent request).
    bool prefix_cache = false;
    std::vector<std::uint32_t> schema_marker;
    std::size_t prefix_min_tokens = 256;  // no entry shorter than this
    std::size_t prefix_min_gain = 128;    // nor one this few tokens longer than the hit it would extend
    std::size_t prefix_entries = 16;      // device snapshots kept (LRU)
    std::size_t prefix_history = 8;       // recent requests the common-prefix search looks at
};

class GpuDecisionEngine : public DecisionEngine {
public:
    GpuDecisionEngine(const TrainedModel& model, const GpuDecisionOptions& options);
    ~GpuDecisionEngine() override;

    std::vector<RequestLogits> score(std::span<const EncodedRequest> requests) override;
    std::string profile() const override;
    std::uint64_t max_tokens() const override;
    ScoreTimes last_times() const override;

    CudaEngine& trunk();
    const GpuDecisionOptions& options() const;
    EngineMemory trunk_memory() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace sepal::cuda
