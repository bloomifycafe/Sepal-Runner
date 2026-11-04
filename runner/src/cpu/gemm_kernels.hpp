// Micro kernels of the packed GEMM (internal header). A kernel multiplies one packed activation block
// [kgroups][mr][group] by one packed weight panel [kgroups][nr][group] over `kgroups` k groups, in increasing k
// order, and writes the finished float tile [mr][nr] (row major, 64-byte aligned). The int8 kernels apply the
// offset correction and the weight and activation scales on the way out.
#pragma once

#include <cstddef>
#include <cstdint>

#include "sepal/cpu/gemm.hpp"

namespace sepal::cpu {

struct KernelArgs {
    const void* a = nullptr;
    const void* b = nullptr;
    std::size_t kgroups = 0;
    float* tile = nullptr;
    bool accumulate = false;             // float kinds: start from the values already in `tile` (k blocking); int8 ignores it
    const float* wscale = nullptr;       // I8: nr weight scales of the panel
    const std::int32_t* corr = nullptr;  // I8_Z: nr offset corrections
    const float* xscale = nullptr;       // I8: mr activation scales of the block
};

using KernelFn = void (*)(const KernelArgs&);

KernelFn kernel_avx2(Variant v);    // nullptr when `v` is not an AVX2 variant
KernelFn kernel_avx512(Variant v);  // nullptr when `v` is not an AVX-512 variant
KernelFn kernel_avx2_v3(Variant v); // nullptr when `v` is not a tuned AVX2 (x86-64-v3) variant

/// The kernel of any variant (nullptr for BF16_Y, which `gemm_packed` runs as F32_Y on widened slices).
inline KernelFn kernel_for(Variant v) {
    switch (v) {
    case Variant::F32_Z:
    case Variant::BF16_Z:
    case Variant::I8_Z: return kernel_avx512(v);
    case Variant::F32_Y3:
    case Variant::BF16_Y3: return kernel_avx2_v3(v);
    default: return kernel_avx2(v);
    }
}

} // namespace sepal::cpu
