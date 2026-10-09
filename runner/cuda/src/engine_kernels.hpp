// Small device kernels of the engine that the layer kernels do not provide. Conventions: device.hpp.
#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace sepal::cuda {

/// Row gather: dst row j = src row rows[j], for j in [0, count). Rows are `row_bytes` bytes wide (a multiple of 4)
/// and start `src_stride_bytes` / `dst_stride_bytes` apart; pointers and strides must be multiples of 4. The host has
/// checked the indices against the source's row count. Pure copies: the bytes are the source's, whatever their type.
void launch_gather_rows(const void* src, std::size_t src_stride_bytes, const std::uint32_t* rows, std::size_t count,
                        std::size_t row_bytes, void* dst, std::size_t dst_stride_bytes, cudaStream_t stream);

} // namespace sepal::cuda
