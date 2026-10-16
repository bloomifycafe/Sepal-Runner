// Micro kernels of the packed GEMM (internal header). A kernel multiplies one packed activation block
// [kgroups][mr][group] by one packed weight panel [kgroups][nr][group] over the whole k range, in increasing k
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
    const float* wscale = nullptr;       // I8: nr weight scales of the panel
    const std::int32_t* corr = nullptr;  // I8_Z: nr offset corrections
    const float* xscale = nullptr;       // I8: mr activation scales of the block
};

using KernelFn = void (*)(const KernelArgs&);

KernelFn kernel_avx2(Variant v);    // nullptr when `v` is not an AVX2 variant
KernelFn kernel_avx512(Variant v);  // nullptr when `v` is not an AVX-512 variant

} // namespace sepal::cpu
