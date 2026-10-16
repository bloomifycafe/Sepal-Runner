// The packed CPU GEMM behind the engine's GemmProvider interface (docs/RUNNER.md section 6). The provider packs the
// weight matrix on every call, so it is meant for tests, tools and one-off products; the engine keeps its weights
// packed once and calls gemm_packed directly.
#pragma once

#include <memory>

#include "sepal/cpu/gemm.hpp"
#include "sepal/gemm.hpp"

namespace sepal::cpu {

/// `kind` selects the arithmetic: F32 (float32 FMA), BF16 (weights and activations rounded to bfloat16),
/// I8 (dynamic int8). `threads` 0 means the hardware concurrency. gemm_f64 delegates to the reference provider.
std::unique_ptr<GemmProvider> make_cpu_gemm(WeightKind kind, Isa isa, unsigned threads);

} // namespace sepal::cpu
