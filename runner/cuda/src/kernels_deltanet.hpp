// The Gated DeltaNet mixer's kernels, everything between the four input projections and the output projection.
// The CPU oracle is `deltanet_mixer` in sepal/ref/deltanet.hpp; read it first.
//
// What one launch does, for each row of a wave and each value head:
//   1. depthwise causal convolution of the row's `mixed` input, continued from the row's incoming conv tail, then
//      SiLU; the same for q, k and v channels;
//   2. L2 normalisation of q and k per key head (shared by the value heads that repeat-interleave it);
//   3. decay = exp(-exp(A_log) * softplus(a + dt_bias)), beta = sigmoid(b);
//   4. the gated delta rule token after token, from the row's incoming recurrent state;
//   5. gated RMSNorm of the reads with z, written as the input of the output projection (float32 or binary16);
// and, in a second kernel of the same call, the row's new conv tail. Steps 1 to 5 are ONE kernel: the convolved
// q, k, v and the raw reads exist only in shared memory (tap pointers can copy them out for tests).
//
// Rows, slots and waves (see device.hpp). `rows` has one RowSpan per row of the pass and `in_slots`/`out_slots`
// one slot index per row, all device arrays indexed by ROW INDEX. A slot is one conv tail [K-1, conv_dim] in
// `conv_tails` and one recurrent state [value heads, dk, dv] in `recurrent` (both float32; slot s starts at
// s * floats-per-slot). An incoming kNoSlot means zeros; an outgoing kNoSlot means "nobody continues this row":
// nothing is stored. `wave` lists the rows to run. Rules for a wave, which the caller guarantees:
//   - several rows may read the same incoming slot;
//   - no row may write a slot that another row of the wave reads or writes;
//   - one row may read and write the same slot (it is then continued in place).
// A row whose parent is a row of the same pass goes in a later wave, after the launch that wrote the parent's
// outgoing slot.
//
// Numbers. Every arithmetic step is an explicit round-to-nearest intrinsic (`__fmaf_rn`, `__fmul_rn`, ...), so the
// compiler cannot contract or reassociate anything; the fused multiply-adds are where the code says so. Every
// sum has one fixed order per output element, independent of the other rows, the wave, the launch, the chunking of
// tokens and the thread mapping: the dot products over the key dimension use 16 interleaved accumulators that are
// combined by a fixed binary tree, whatever `lanes_per_column` is. So a row's results are bit-identical alone or
// in any batch, in one row or continued across rows, and under every thread mapping.
//
// Limits: key and value head dimensions at most 128, at most 16 conv taps.
#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "device.hpp"

namespace sepal::cuda {

inline constexpr std::uint32_t kDeltaNetMaxHeadDim = 128;
inline constexpr std::uint32_t kDeltaNetMaxConvKernel = 8;

/// The mixer's sizes (ModelConfig: dn_key_heads, dn_value_heads, dn_key_head_dim, dn_value_head_dim, conv_kernel,
/// rms_norm_eps). Value heads must be a multiple of key heads.
struct DeltaNetGeometry {
    std::uint32_t key_heads = 0;
    std::uint32_t value_heads = 0;
    std::uint32_t key_head_dim = 0;
    std::uint32_t value_head_dim = 0;
    std::uint32_t conv_kernel = 0; // taps K; the conv tail has K-1 rows
    double rms_norm_eps = 1e-6;

    std::uint32_t key_dim() const { return key_heads * key_head_dim; }
    std::uint32_t value_dim() const { return value_heads * value_head_dim; }
    /// Channels of `mixed`: q, k, v.
    std::uint32_t conv_dim() const { return 2 * key_dim() + value_dim(); }
    /// Floats of one conv-tail slot and of one recurrent-state slot.
    std::size_t conv_tail_floats() const { return static_cast<std::size_t>(conv_kernel - 1) * conv_dim(); }
    std::size_t recurrent_floats() const {
        return static_cast<std::size_t>(value_heads) * key_head_dim * value_head_dim;
    }
};

/// Small weights, float32, on the device.
struct DeltaNetWeights {
    const float* conv = nullptr;    // [conv_dim, K], tap K-1 multiplies the current token
    const float* a_log = nullptr;   // [value heads]
    const float* dt_bias = nullptr; // [value heads]
    const float* norm = nullptr;    // [value head dim], the stored weight itself (not 1 + weight)
};

/// A float32 matrix [N, width] over the flat token list: a pointer to its first element and a row stride in
/// elements, so each input can be a band of columns of the multiply output.
struct DeviceMatrixF32 {
    const float* data = nullptr;
    std::size_t stride = 0;
};

/// The four projections of the normed layer input.
struct DeltaNetInputs {
    DeviceMatrixF32 mixed; // [N, conv_dim]
    DeviceMatrixF32 z;     // [N, value_dim]
    DeviceMatrixF32 b;     // [N, value heads]
    DeviceMatrixF32 a;     // [N, value heads]
};

/// The rows to run and where their states are. All arrays are on the device.
struct DeltaNetRows {
    const RowSpan* rows = nullptr;            // every row of the pass
    const std::uint32_t* in_slots = nullptr;  // per row index; kNoSlot = start from zeros
    const std::uint32_t* out_slots = nullptr; // per row index; kNoSlot = do not store
    const std::uint32_t* wave = nullptr;      // row indices to run in this launch
    std::uint32_t wave_size = 0;
};

/// The slot arrays.
struct DeltaNetStates {
    float* conv_tails = nullptr; // [slots, K-1, conv_dim]
    float* recurrent = nullptr;  // [slots, value heads, dk, dv]
};

/// Optional copies of intermediate values, for tests. Null pointers cost nothing.
struct DeltaNetTaps {
    /// [N, conv_dim]: the activated convolution output as the scan sees it: q and k after L2 normalisation (one
    /// copy per KEY head, not per value head), v as convolved and activated.
    float* conv_act = nullptr;
    std::size_t conv_act_stride = 0;
    /// [N, value_dim]: the scan's reads (scaled by 1/sqrt(dk)), before the gated norm.
    float* reads = nullptr;
    std::size_t reads_stride = 0;
};

struct DeltaNetOptions {
    /// Threads cooperating on one column of the recurrent state (splitting the dot product over the key
    /// dimension): 0 chooses by the size of the wave, or 1, 2, 4 or 8. It changes speed, never the results.
    int lanes_per_column = 0;
    DeltaNetTaps taps;
};

/// Steps 1 to 5 above for the rows of the wave: the whole scan kernel. `out` is [N, value_dim] with row stride
/// `out_stride` elements, of type float or __half. Enqueues on `stream` and returns without waiting; throws
/// std::invalid_argument for an unsupported geometry and std::runtime_error if the launch fails.
template <class OutT>
void launch_deltanet_scan(const DeltaNetGeometry& geometry, const DeltaNetWeights& weights,
                          const DeltaNetInputs& inputs, const DeltaNetRows& rows, const DeltaNetStates& states,
                          OutT* out, std::size_t out_stride, cudaStream_t stream,
                          const DeltaNetOptions& options = {});

/// The rows' new conv tails: the last K-1 conv inputs of (incoming tail, then the row's own inputs). Reads only
/// `mixed` and the incoming tails, and may overwrite the incoming tail it reads (a row continued in place), so it
/// must run AFTER launch_deltanet_scan of the same wave, on the same stream.
void launch_deltanet_conv_tail(const DeltaNetGeometry& geometry, const DeltaNetInputs& inputs,
                               const DeltaNetRows& rows, const DeltaNetStates& states, cudaStream_t stream);

/// Both of the above: everything the mixer does for one wave.
template <class OutT>
void launch_deltanet_wave(const DeltaNetGeometry& geometry, const DeltaNetWeights& weights,
                          const DeltaNetInputs& inputs, const DeltaNetRows& rows, const DeltaNetStates& states,
                          OutT* out, std::size_t out_stride, cudaStream_t stream,
                          const DeltaNetOptions& options = {}) {
    launch_deltanet_scan(geometry, weights, inputs, rows, states, out, out_stride, stream, options);
    launch_deltanet_conv_tail(geometry, inputs, rows, states, stream);
}

} // namespace sepal::cuda
