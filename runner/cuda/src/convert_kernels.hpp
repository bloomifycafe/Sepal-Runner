// Elementwise element-type conversions on the device. Declared with plain types only, so that this header is
// read the same way by the host compiler and by nvcc.
#pragma once

#include <cstddef>

namespace sepal::cuda::detail {

enum class Elem : int { F32 = 0, F16 = 1, BF16 = 2 };

/// dst[i] = convert(src[i]) for i in [0, count), enqueued on `stream` (a cudaStream_t). Supported pairs:
/// F16 -> F32 and BF16 -> F32 (exact), F32 -> F16 and BF16 -> F16 (round to nearest, ties to even; values beyond
/// the binary16 range become infinities), F32 -> BF16 (round to nearest, ties to even; NaN stays NaN). src and dst must not overlap.
/// Returns the cudaError_t of the launch as an int (0 is success); -1 for an unsupported pair.
int launch_convert(const void* src, Elem src_type, void* dst, Elem dst_type, std::size_t count, void* stream);

} // namespace sepal::cuda::detail
