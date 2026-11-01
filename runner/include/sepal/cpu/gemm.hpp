// Blocked, packed GEMM of the CPU backend: Y[t, n] = (accumulate ? Y[t, n] : 0) + sum_k X[t, k] W[n, k].
//
// Weights are packed once at load into panels of `nr` output features; activations are packed per call (or once
// for several weights that read the same input) into blocks of `mr` rows. Three weight kinds:
//   F32   float32 weights, float32 activations, FMA.
//   BF16  bfloat16 weights, activations rounded to bfloat16 (round to nearest even), float32 accumulation
//         (vdpbf16ps on AVX-512 BF16; an exact-product FMA emulation on AVX2).
//   I8    int8 weights with one scale per output feature, int8 activations quantised dynamically with one scale per
//         row, int32 accumulation (vpdpbusd on AVX-512 VNNI with the u8 x s8 offset trick; vpmaddwd on AVX2).
//         Integer sums are exact, so any order gives the same result.
//
// Determinism: a given output element is the sum over k in increasing order in one chain (the float kinds) and
// is computed by one task, so no thread count or batch composition changes any bit of a row's result. The
// result of a row does not depend on the other rows of the batch (they only share the packing of their block).
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "sepal/cpu/aligned.hpp"
#include "sepal/cpu/isa.hpp"
#include "sepal/cpu/thread_pool.hpp"

namespace sepal::cpu {

enum class WeightKind { F32, BF16, I8 };
const char* name_of(WeightKind kind);

/// One kernel: the weight kind on one instruction-set family. Z = AVX-512, Y = AVX2.
enum class Variant { F32_Z, F32_Y, BF16_Z, BF16_Y, I8_Z, I8_Y };

struct VariantInfo {
    WeightKind kind;
    Isa isa;
    int mr;             // activation rows per block (rows of the micro tile)
    int nr;             // weight columns per panel (columns of the micro tile)
    int group;          // consecutive k values stored together (1 float, 2 bf16 pair, 4 u8 / 2 i16)
    std::size_t weight_bytes; // bytes per stored weight element
    std::size_t act_bytes;    // bytes per stored activation element
};
VariantInfo info(Variant v);
Variant pick_variant(WeightKind kind, Isa isa);
const char* name_of(Variant v);

/// Bfloat16 of a float32, round to nearest even; NaN stays a NaN.
std::uint16_t float_to_bf16(float x);

/// A weight matrix W [n, k] packed for one variant. Immutable after construction.
class PackedWeight {
public:
    Variant variant() const { return variant_; }
    std::size_t n() const { return n_; }          // output features
    std::size_t k() const { return k_; }          // input features
    std::size_t kgroups() const { return kgroups_; }
    bool swiglu() const { return swiglu_; }       // gate/up interleaved: the product is silu(gate) * up, n = intermediate size
    std::size_t panels() const { return panels_; }
    std::size_t bytes() const { return data_.size() + (scale_.size() + corr_.size()) * 4; }
    const unsigned char* panel(std::size_t p) const { return data_.data() + p * panel_bytes_; }
    /// I8_Z has one scale (and offset correction) per output feature and block of `kBlockValues` input values, laid out [kblock][nr] per panel;
    /// I8_Y has one per output feature (kblocks() == 1).
    std::size_t kblocks() const { return kblocks_; }
    const float* scale(std::size_t p) const { return scale_.data() + p * kblocks_ * static_cast<std::size_t>(info(variant_).nr); }
    const std::int32_t* corr(std::size_t p) const { return corr_.data() + p * kblocks_ * static_cast<std::size_t>(info(variant_).nr); }

private:
    friend PackedWeight pack_weight(const float*, std::size_t, std::size_t, std::size_t, WeightKind, Isa, ThreadPool&);
    friend PackedWeight pack_weight_swiglu(const float*, const float*, std::size_t, std::size_t, std::size_t, WeightKind, Isa, ThreadPool&);
    Variant variant_ = Variant::F32_Y;
    std::size_t n_ = 0, k_ = 0, panels_ = 0, panel_bytes_ = 0, kgroups_ = 0, kblocks_ = 1;
    bool swiglu_ = false;
    AlignedBuffer<unsigned char> data_;
    AlignedBuffer<float> scale_;        // I8: per padded output feature
    AlignedBuffer<std::int32_t> corr_;  // I8_Z: 128 * sum of the feature's quantised weights (u8 offset correction)
};

/// Input values per quantisation block of the AVX-512 int8 kernel (weights and activations each get one scale per block, so one outlier
/// only costs the precision of its own 128 values, not of the whole row).
constexpr std::size_t kBlockValues = 128;

/// Packs `w` [n, k] with row stride `ldw` (floats). BF16 rounds each weight to nearest even; I8 quantises each
/// output feature symmetrically with scale max|w| / 127.
PackedWeight pack_weight(const float* w, std::size_t ldw, std::size_t n, std::size_t k, WeightKind kind, Isa isa,
                         ThreadPool& pool);

/// Packs a SwiGLU gate/up pair (both [n, k], n = intermediate size) so that one GEMM yields
/// silu(gate x) * (up x) as [rows, n]. The product is the output of `gemm_packed` with Store.
PackedWeight pack_weight_swiglu(const float* gate, const float* up, std::size_t ldw, std::size_t n, std::size_t k,
                                WeightKind kind, Isa isa, ThreadPool& pool);

/// Activations X [rows, k] packed for one variant (row blocks of `mr`).
struct PackedActs {
    Variant variant = Variant::F32_Y;
    std::size_t rows = 0, k = 0, blocks = 0, block_bytes = 0, kgroups = 0, kblocks = 1;
    AlignedBuffer<unsigned char> data;
    AlignedBuffer<float> row_scale; // I8: activation scale per (padded) row; I8_Z: per row block [mr-block][kblock][mr]
};

/// (Re)packs `x` [rows, k] with row stride `ldx` into `out`, reusing its buffers.
void pack_acts(PackedActs& out, const float* x, std::size_t ldx, std::size_t rows, std::size_t k, Variant variant,
               ThreadPool& pool);

/// y [rows, n] (row stride ldy) = (accumulate ? y : 0) + x w^T. For SwiGLU weights n is the intermediate size and
/// the result is silu(gate) * up (accumulate must be false). The variants of `w` and `x` and their k must agree.
void gemm_packed(const PackedWeight& w, const PackedActs& x, float* y, std::size_t ldy, bool accumulate, ThreadPool& pool);

/// Floating-point operations of one product (2 m n k).
inline double gemm_flops(std::size_t rows, std::size_t n, std::size_t k) {
    return 2.0 * static_cast<double>(rows) * static_cast<double>(n) * static_cast<double>(k);
}

} // namespace sepal::cpu
