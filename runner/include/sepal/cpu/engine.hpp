// The CPU DecisionEngine: trained model (base + merged LoRA) on the packed CPU trunk and the CPU Clef head.
// Profiles: "cpu-f32" (float32), "cpu-bf16" (bf16 weights and activations), "cpu-q8" (int8 weights x dynamic int8
// activations for the groups named in `q8`, bf16 elsewhere).
#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "sepal/cpu/trunk.hpp"
#include "sepal/decision.hpp"
#include "sepal/trained_model.hpp"

namespace sepal::cpu {

struct CpuEngineOptions {
    std::string profile = "cpu-f32";
    unsigned threads = 0;                    // 0: hardware concurrency
    std::uint64_t max_tokens = 16384;        // longest request accepted
    Q8Groups q8;                             // cpu-q8 only
    std::size_t chunk_tokens = 2048;
    std::size_t step_tokens = 4096;
    std::size_t delta_chunk = 64;            // 0: token-sequential DeltaNet
    std::size_t group_tokens = 65536;        // requests per trunk call are grouped up to this many tokens
    std::size_t prefix_cache_mib = 0;        // host memory for prefix states (0: off). A request that shares its first tokens with an earlier one
                                             // (the same log asked with other questions) resumes from the state after them
    std::size_t prefix_min_tokens = 512;     // shortest shared prefix worth a state
    std::size_t prefix_min_gain = 256;       // a new, longer state is only made when it adds at least this many tokens to the best hit
    bool force_isa = false;
    Isa isa = Isa::Avx2;
    std::string cpus;                        // CPU list of the thread pool, e.g. "0-11" (Linux); empty: every allowed CPU
    std::string gemm;                        // GEMM mode (TrunkOptions::gemm): empty = SEPAL_CPU_GEMM, else classical
};

/// Parses "cpu-f32" / "cpu-bf16" / "cpu-q8"; throws std::invalid_argument for anything else.
Profile parse_profile(const std::string& id);

std::unique_ptr<DecisionEngine> make_cpu_engine(TrainedModel model, const CpuEngineOptions& options);
std::unique_ptr<DecisionEngine> make_cpu_engine(const std::string& final_dir, const std::string& base_dir, const CpuEngineOptions& options);

} // namespace sepal::cpu
