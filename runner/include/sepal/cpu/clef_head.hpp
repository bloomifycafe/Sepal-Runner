// The Clef JointSchemaHead on the CPU: the same math as sepal::clef_head_reference (the oracle), in float32 with the
// large products (the memory projection, the key and value projections over every position) on the packed GEMM.
#pragma once

#include <functional>
#include <memory>
#include <span>

#include "sepal/clef_head.hpp"
#include "sepal/cpu/isa.hpp"
#include "sepal/cpu/thread_pool.hpp"

namespace sepal::cpu {

class CpuClefHead {
public:
    /// Packs the head weights (float32). `pool` must outlive the head; calls are serialised by the caller.
    CpuClefHead(const ClefHeadWeights& weights, ThreadPool& pool, Isa isa);
    ~CpuClefHead();
    CpuClefHead(const CpuClefHead&) = delete;
    CpuClefHead& operator=(const CpuClefHead&) = delete;

    /// `hidden` is [L, hidden_size] (the trunk's final-normed hidden states of every position); `embed` gives the
    /// output-embedding row of a token. Same contract as clef_head_reference.
    RequestLogits run(std::span<const float> hidden, const EncodedRequest& request,
                      const std::function<void(std::uint32_t, float*)>& embed);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace sepal::cpu
