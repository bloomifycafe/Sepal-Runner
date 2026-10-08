// The matrix-multiply provider interface (docs/RUNNER.md section 6, "GEMM provider").
//
// Every matrix multiply of the engine goes through a provider, so that vendor libraries and GPU back ends can
// be swapped in. The contract that makes batching invisible:
//
//     output row t depends on input row t and on the weights, and on nothing else.
//
// A provider must give the same bits for a row whether it is computed alone or inside any batch. The reference
// provider in this file's companion source is the one later providers are tested against.
#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "sepal/dtype.hpp"
#include "sepal/tensor.hpp"

namespace sepal {

/// Non-owning, read-only, dense row-major matrix of a given element type. `data` must be aligned for that type.
struct MatRef {
    const void* data = nullptr;
    DType dtype = DType::F32;
    std::size_t rows = 0;
    std::size_t cols = 0;

    constexpr MatRef() = default;
    constexpr MatRef(const void* d, DType t, std::size_t r, std::size_t c) : data(d), dtype(t), rows(r), cols(c) {}
    constexpr MatRef(ConstMatView v) : data(v.data), dtype(DType::F32), rows(v.rows), cols(v.cols) {}
    constexpr MatRef(MatView v) : data(v.data), dtype(DType::F32), rows(v.rows), cols(v.cols) {}
};

struct GemmCaps {
    std::string name;
    std::string version;
    std::string isa;                // instruction set the kernels use; "scalar" for the reference
    std::vector<DType> float_dtypes; // element types gemm_f accepts for X and W
    bool integer_exact = false;     // true once an exact int32-accumulate gemm_qd8 exists
    bool fixed_summation_order = false;
};

class GemmProvider {
public:
    virtual ~GemmProvider() = default;

    virtual GemmCaps caps() const = 0;

    /// Y[t, n] = beta * Y[t, n] + sum over k of X[t, k] * W[n, k].
    ///
    /// X is [T, K], W is [N, K] (one output feature per row, as weights are stored), Y is [T, N] float32.
    /// With beta == 0, Y is written without being read. Y must not overlap X or W.
    /// Throws std::invalid_argument on a shape mismatch or an unsupported element type.
    virtual void gemm_f(MatRef x, MatRef w, MatView y, float beta) const = 0;

    /// The same product with X, W and Y all float64: the reference path the float32 engine is compared with.
    /// Shapes, beta and errors as gemm_f.
    virtual void gemm_f64(ConstMatViewD x, ConstMatViewD w, MatViewD y, double beta) const = 0;
};

/// Y = X W^T in the element type of the arguments, so that code written once for both precisions reaches the
/// provider through one name.
inline void gemm(const GemmProvider& provider, ConstMatView x, ConstMatView w, MatView y) {
    provider.gemm_f(x, w, y, 0.0f);
}
inline void gemm(const GemmProvider& provider, ConstMatViewD x, ConstMatViewD w, MatViewD y) {
    provider.gemm_f64(x, w, y, 0.0);
}

/// Plain single-threaded provider. Each output is accumulated in float64 over k = 0, 1, ..., K-1 in that order
/// and (in gemm_f) rounded to float32 once, so its result is a fixed function of the row and the weights.
std::unique_ptr<GemmProvider> make_reference_gemm();

} // namespace sepal
