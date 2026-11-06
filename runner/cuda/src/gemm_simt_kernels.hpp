// The float32 SIMT multiply of the Pascal profiles (gpu-bf16, gpu-w8g): no tensor cores, no cp.async, no ldmatrix, so it
// runs on every architecture from sm_61 up. Conventions: device.hpp.
//
//   Y[t, n] = epilogue( sum over k of X[t, k] * Wd[n, k] )
//
// X is float32 [T, K] (row stride ld_x). The weights are stored narrow and widened when a tile is copied to shared memory:
//   F32        Wd = W, float32 [N, K];
//   BF16       Wd = float32(W), bfloat16 [N, K] (exact widening);
//   Int8Group  Wd[n, k] = float32(q[n, k]) * s[n, k / 128], q int8 [N, K], s float32 [N, K / 128]: one scale per group of
//              128 values of k; the product is rounded to float32 once, when the tile is stored (so the multiply is the
//              float32 multiply of the dequantised matrix, bit for bit).
//
// Schedule and determinism (proofs/GpuTrack/GpuTrack/GemmSchedule.lean, GroupQuant.lean). Each output element is owned by
// one thread, which runs acc = fma(X[t, k], Wd[n, k], acc) for k = 0, 1, ..., K-1 from acc = 0 (round-to-nearest fused
// multiply-adds, written as intrinsics so nothing is contracted or reordered) and then applies the epilogue. There is no
// split of k across threads or blocks, so an element's bits depend on its own X row, its own weight row and K only: not on
// T, N, the tile shape (`config`), the grid, the other rows of the batch or the row's place in it.
//
// Epilogues:
//   Store    Y[t, n] = acc
//   Add      Y[t, n] = Y[t, n] + acc (the residual stream updated in place; one float32 addition, as launch_add)
//   SwiGLU   the weight rows are the MLP's gate and up projections interleaved in blocks of 64 (pack_gate_up_rows):
//            packed row 128 b + r is gate row 64 b + r and packed row 128 b + 64 + r is up row 64 b + r. N is the packed
//            width 2 I and Y has I columns: Y[t, 64 b + r] = silu(acc[128 b + r]) * acc[128 b + 64 + r], silu as in
//            launch_swiglu (kernels_elementwise.hpp), so the result is that of the unfused multiply and launch_swiglu.
//
// Requirements: K a multiple of 16 (128 for Int8Group), ld_x and the weight row stride multiples of 16 elements with
// 16-byte aligned pointers, ld_y a multiple of 4 with a 16-byte aligned Y; SwiGLU needs N a multiple of 128.
// Enqueued on `stream`; throws std::invalid_argument for a violated requirement, std::runtime_error when the launch fails.
#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace sepal::cuda {

enum class SimtWeight : int { F32 = 0, BF16 = 1, Int8Group = 2 };
enum class SimtEpilogue : int { Store = 0, Add = 1, SwiGLU = 2 };

inline constexpr std::size_t kSimtGroup = 128; // values of k per Int8Group scale

/// A weight matrix as stored on the device.
struct SimtMatrix {
    const void* w = nullptr;        // [N, ld_w] elements of the stored type (bytes for Int8Group)
    std::size_t ld_w = 0;
    const float* scales = nullptr;  // Int8Group: [N, ld_s] scales, one per group of 128 values of k
    std::size_t ld_s = 0;
    SimtWeight type = SimtWeight::F32;
};

/// Tile shapes, for tests and benchmarks: -1 picks one from T, 0 is 128 x 128 (256 threads), 1 is 64 x 128 (128 threads),
/// 2 is 32 x 128 (64 threads). All give the same bits.
void launch_gemm_simt(const float* x, std::size_t ld_x, const SimtMatrix& w, float* y, std::size_t ld_y, std::size_t t,
                      std::size_t n, std::size_t k, SimtEpilogue epilogue, int config, cudaStream_t stream);

/// Whether launch_gemm_simt takes these dimensions and strides (pointer alignment is the caller's).
bool gemm_simt_supports(SimtWeight type, std::size_t n, std::size_t k, std::size_t ld_x, std::size_t ld_w, std::size_t ld_y,
                        SimtEpilogue epilogue);

/// Quantises float32 weights [rows, K] (row stride ld_src) to int8 with one float32 scale per row and group of 128 values
/// of k: s = max |w| / 127 over the group, q = w / s rounded to nearest even and clamped to [-127, 127] (an all-zero group
/// gets s = 0 and q = 0). q goes to [rows, ld_q] bytes, s to [rows, ld_s]. K a multiple of 128.
/// Deterministic (one thread per group; the maximum and the divisions do not depend on the launch).
void launch_quantize_group_s8(const float* src, std::size_t ld_src, std::int8_t* q, std::size_t ld_q, float* s,
                              std::size_t ld_s, std::size_t rows, std::size_t k, cudaStream_t stream);

/// Rounds float32 to bfloat16 (nearest, ties to even; NaN stays NaN): dst[i] = bf16(src[i]), `count` elements.
void launch_float_to_bf16(const float* src, std::uint16_t* dst, std::size_t count, cudaStream_t stream);

/// The row order of the fused SwiGLU multiply: packed row p of [gate; up] (2 I rows, I a multiple of 64) comes from
/// original row pack_gate_up_source(p, I), where rows 0 .. I-1 are gate and I .. 2I-1 are up.
constexpr std::size_t pack_gate_up_source(std::size_t p, std::size_t inter) noexcept {
    const std::size_t b = p / 128, r = p % 128;
    return r < 64 ? 64 * b + r : inter + 64 * b + (r - 64);
}

} // namespace sepal::cuda
