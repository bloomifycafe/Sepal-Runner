#include "sepal/gemm.hpp"

#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace sepal {

namespace {

// A float32 view of a matrix: the original storage for F32, an exactly decoded copy for the 16-bit formats.
class F32Matrix {
public:
    F32Matrix(MatRef m, const char* what) {
        const std::size_t n = m.rows * m.cols;
        switch (m.dtype) {
        case DType::F32:
            data_ = static_cast<const float*>(m.data);
            return;
        case DType::F16:
        case DType::BF16: {
            decoded_.resize(n);
            const auto* bytes = static_cast<const unsigned char*>(m.data);
            for (std::size_t i = 0; i < n; ++i) {
                std::uint16_t bits = 0;
                std::memcpy(&bits, bytes + 2 * i, sizeof bits);
                decoded_[i] = m.dtype == DType::F16 ? half_to_float(bits) : bf16_to_float(bits);
            }
            data_ = decoded_.data();
            return;
        }
        case DType::I8:
            break;
        }
        throw std::invalid_argument(std::string("gemm_f: ") + what + " has element type " +
                                    std::string(name_of(m.dtype)) + ", which is not a float type");
    }
    const float* data() const { return data_; }

private:
    const float* data_ = nullptr;
    std::vector<float> decoded_;
};

class ReferenceGemm final : public GemmProvider {
public:
    GemmCaps caps() const override {
        GemmCaps c;
        c.name = "reference";
        c.version = "1";
        c.isa = "scalar";
        c.float_dtypes = {DType::F32, DType::F16, DType::BF16};
        c.integer_exact = false;
        c.fixed_summation_order = true;
        return c;
    }

    void gemm_f(MatRef x, MatRef w, MatView y, float beta) const override {
        check_shapes("gemm_f", x.data, x.rows, x.cols, w.data, w.rows, w.cols, y.data, y.rows, y.cols);

        const F32Matrix xf(x, "X");
        const F32Matrix wf(w, "W");
        const std::size_t width = x.cols;

        for (std::size_t t = 0; t < x.rows; ++t) {
            const float* x_row = xf.data() + t * width;
            for (std::size_t n = 0; n < w.rows; ++n) {
                const float* w_row = wf.data() + n * width;
                double acc = 0.0;
                for (std::size_t k = 0; k < width; ++k) {
                    acc += static_cast<double>(x_row[k]) * static_cast<double>(w_row[k]);
                }
                float& out = y(t, n);
                if (beta != 0.0f) acc += static_cast<double>(beta) * static_cast<double>(out);
                out = static_cast<float>(acc);
            }
        }
    }

    void gemm_f64(ConstMatViewD x, ConstMatViewD w, MatViewD y, double beta) const override {
        check_shapes("gemm_f64", x.data, x.rows, x.cols, w.data, w.rows, w.cols, y.data, y.rows, y.cols);
        const std::size_t width = x.cols;
        for (std::size_t t = 0; t < x.rows; ++t) {
            const double* x_row = x.data + t * width;
            for (std::size_t n = 0; n < w.rows; ++n) {
                const double* w_row = w.data + n * width;
                double acc = 0.0;
                for (std::size_t k = 0; k < width; ++k) acc += x_row[k] * w_row[k];
                double& out = y(t, n);
                if (beta != 0.0) acc += beta * out;
                out = acc;
            }
        }
    }

private:
    static void check_shapes(const char* who, const void* x, std::size_t x_rows, std::size_t x_cols, const void* w,
                             std::size_t w_rows, std::size_t w_cols, const void* y, std::size_t y_rows,
                             std::size_t y_cols) {
        const std::string name(who);
        if (x_cols != w_cols) {
            throw std::invalid_argument(name + ": X has " + std::to_string(x_cols) + " columns but W has " +
                                        std::to_string(w_cols));
        }
        if (y_rows != x_rows || y_cols != w_rows) {
            throw std::invalid_argument(name + ": Y must be [" + std::to_string(x_rows) + ", " +
                                        std::to_string(w_rows) + "] but is [" + std::to_string(y_rows) + ", " +
                                        std::to_string(y_cols) + "]");
        }
        if ((x == nullptr && x_rows * x_cols != 0) || (w == nullptr && w_rows * w_cols != 0) ||
            (y == nullptr && y_rows * y_cols != 0)) {
            throw std::invalid_argument(name + ": null data pointer");
        }
    }
};

} // namespace

std::unique_ptr<GemmProvider> make_reference_gemm() { return std::make_unique<ReferenceGemm>(); }

} // namespace sepal
