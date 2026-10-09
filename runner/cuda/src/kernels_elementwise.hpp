// Device kernels for everything in a layer that is not a matrix multiply, a recurrence or attention: the
// embedding gather, RMSNorm, the residual additions, the SwiGLU product and the readout. Declared with plain
// types only (`void*` plus an element tag for outputs that can be binary16 or float32), so that this header is
// read the same way by the host compiler and by nvcc. The conventions are those of device.hpp.
//
// Every launch function enqueues on `stream`, returns without waiting, and throws std::runtime_error when the
// launch fails or std::invalid_argument for arguments that cannot work (an element type that is not F32 or F16,
// an overlapping output). A matrix is a pointer and a row stride in elements, and the first `width` elements of
// each row are the ones used. A pointer must be valid for the rows and columns the call names; strides may be
// larger than the width.
//
// Output element type. A kernel whose output feeds a matrix multiply takes `out_type`: Elem::F16 writes
// binary16 (round to nearest, ties to even; magnitudes beyond 65504 become infinities, as in the conversion
// kernels) and Elem::F32 writes float32. Anything else is refused. The residual stream is always float32.
//
// Determinism. An output element depends on its own row and on the weights, and on nothing else. The sum of
// squares of a row is accumulated in one fixed order that depends on the row's width only: not on the number of
// rows, not on the row's place in the list, not on whether the row is read through an index, and not on the
// alignment or stride of the data.
#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "convert_kernels.hpp"

namespace sepal::cuda {

using detail::Elem;

/// Embedding gather: out[t, :] = float32(table[token_ids[t], :]) for t in [0, n).
/// token_ids: device array of n ids. table: device matrix [vocab, hidden], dense (row stride hidden), stored as
/// `table_type` (F32 or F16). out: float32 matrix [n, hidden] with row stride `out_stride`. An id outside the
/// vocabulary writes a zero row (the host admits ids before a pass; the kernel only guarantees not to read out of
/// the table).
void launch_embed(const std::uint32_t* token_ids, std::size_t n, const void* table, Elem table_type,
                  std::size_t vocab, std::size_t hidden, float* out, std::size_t out_stride, cudaStream_t stream);

/// RMSNorm to a multiply input: out[r, i] = x[r, i] / sqrt(mean(x[r, :]^2) + eps) * (1 + weight[i]), r in [0, n).
/// x: float32 [n, width], row stride x_stride. weight: float32 [width], the offset from one. out: `out_type`
/// matrix [n, width] with row stride out_stride. A zero row gives a zero row. The sum of squares is accumulated in
/// float32 and overflows to infinity (and then gives zeros) for entries beyond about 1e19 in magnitude.
void launch_rms_norm(const float* x, std::size_t x_stride, const float* weight, std::size_t n, std::size_t width,
                     float eps, void* out, Elem out_type, std::size_t out_stride, cudaStream_t stream);

/// Residual addition fused with the next norm, one pass over the data:
///   r[t, :] = x[t, :] + m[t, :]  written as float32 (the new residual stream), and
///   norm_out[t, :] = rms_norm(r[t, :]) as in launch_rms_norm, written as `out_type`.
/// The sum is rounded to float32 before the norm sees it, so the result is the same as a launch_add followed by a
/// launch_rms_norm, bit for bit. r_out may be the same matrix as x or as m (same pointer, same stride); any other
/// overlap between an output and an input, or between the two outputs, is refused where it can be seen.
void launch_add_rms_norm(const float* x, std::size_t x_stride, const float* m, std::size_t m_stride, float* r_out,
                         std::size_t r_stride, const float* weight, std::size_t n, std::size_t width, float eps,
                         void* norm_out, Elem out_type, std::size_t norm_stride, cudaStream_t stream);

/// Plain residual addition: out[t, i] = x[t, i] + m[t, i], float32 throughout. out may be x or m.
void launch_add(const float* x, std::size_t x_stride, const float* m, std::size_t m_stride, float* out,
                std::size_t out_stride, std::size_t n, std::size_t width, cudaStream_t stream);

/// SwiGLU combine: out[t, i] = silu(gate[t, i]) * up[t, i], the input of the down projection.
/// gate and up are float32 column bands of one multiply output (each a pointer and a row stride), `width` columns
/// of n rows. out: `out_type` matrix [n, width] with row stride out_stride. silu(x) = x * sigmoid(x) with
/// sigmoid(x) = exp(-softplus(-x)) and softplus(x) = max(x, 0) + log1p(exp(-|x|)), as in sepal::ref, so no input
/// overflows an intermediate.
void launch_swiglu(const float* gate, std::size_t gate_stride, const float* up, std::size_t up_stride, std::size_t n,
                   std::size_t width, void* out, Elem out_type, std::size_t out_stride, cudaStream_t stream);

/// launch_swiglu with gate and up given as binary16 (the output of a multiply that rounds to binary16, which halves the
/// bytes this launch and that multiply move). The inputs are widened exactly, so the result is the same as
/// launch_swiglu's on the same values as float32. `gate` and `up` are `const __half*`.
void launch_swiglu_half(const void* gate, std::size_t gate_stride, const void* up, std::size_t up_stride, std::size_t n,
                        std::size_t width, void* out, Elem out_type, std::size_t out_stride, cudaStream_t stream);

/// Readout: the final norm of chosen rows. out[j, :] = rms_norm(x[rows[j], :]) with the final norm's weight, for j
/// in [0, count); float32 [count, width], row stride out_stride. x: the last layer's output, float32, row stride
/// x_stride. rows: device array of `count` row indices into x (the host has checked them against x's row count).
/// A row's result is bit-identical to the same row through launch_rms_norm.
void launch_readout_norm(const float* x, std::size_t x_stride, const std::uint32_t* rows, std::size_t count,
                         const float* weight, std::size_t width, float eps, float* out, std::size_t out_stride,
                         cudaStream_t stream);

} // namespace sepal::cuda
