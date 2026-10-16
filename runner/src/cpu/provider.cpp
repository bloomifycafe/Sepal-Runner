#include "sepal/cpu/provider.hpp"

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "sepal/dtype.hpp"

namespace sepal::cpu {

namespace {

// Float32 copy of a matrix of any float element type.
std::vector<float> to_float(MatRef m, const char* what) {
    std::vector<float> out(m.rows * m.cols);
    const auto* bytes = static_cast<const unsigned char*>(m.data);
    switch (m.dtype) {
    case DType::F32:
        std::memcpy(out.data(), m.data, out.size() * sizeof(float));
        return out;
    case DType::F16:
    case DType::BF16:
        for (std::size_t i = 0; i < out.size(); ++i) {
            std::uint16_t bits = 0;
            std::memcpy(&bits, bytes + 2 * i, sizeof bits);
            out[i] = m.dtype == DType::F16 ? half_to_float(bits) : bf16_to_float(bits);
        }
        return out;
    case DType::I8:
        break;
    }
    throw std::invalid_argument(std::string("cpu gemm_f: ") + what + " is not a float matrix");
}

class CpuGemm final : public GemmProvider {
public:
    CpuGemm(WeightKind kind, Isa isa, unsigned threads) : kind_(kind), isa_(isa), pool_(threads), reference_(make_reference_gemm()) {}

    GemmCaps caps() const override {
        GemmCaps c;
        c.name = std::string("cpu-") + name_of(kind_);
        c.version = "1";
        c.isa = name_of(isa_);
        c.float_dtypes = {DType::F32, DType::F16, DType::BF16};
        c.integer_exact = kind_ == WeightKind::I8;
        c.fixed_summation_order = true;
        return c;
    }

    void gemm_f(MatRef x, MatRef w, MatView y, float beta) const override {
        if (x.cols != w.cols || y.rows != x.rows || y.cols != w.rows) throw std::invalid_argument("cpu gemm_f: shape mismatch");
        if (beta != 0.0f && beta != 1.0f) throw std::invalid_argument("cpu gemm_f: beta must be 0 or 1");
        const std::vector<float> xf = to_float(x, "x");
        const std::vector<float> wf = to_float(w, "w");
        const PackedWeight pw = pack_weight(wf.data(), w.cols, w.rows, w.cols, kind_, isa_, pool_);
        PackedActs pa;
        pack_acts(pa, xf.data(), x.cols, x.rows, x.cols, pw.variant(), pool_);
        gemm_packed(pw, pa, y.data, y.cols, beta == 1.0f, pool_);
    }

    void gemm_f64(ConstMatViewD x, ConstMatViewD w, MatViewD y, double beta) const override {
        reference_->gemm_f64(x, w, y, beta);
    }

private:
    WeightKind kind_;
    Isa isa_;
    mutable ThreadPool pool_;
    std::unique_ptr<GemmProvider> reference_;
};

} // namespace

std::unique_ptr<GemmProvider> make_cpu_gemm(WeightKind kind, Isa isa, unsigned threads) {
    return std::make_unique<CpuGemm>(kind, isa, threads);
}

} // namespace sepal::cpu
