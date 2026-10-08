// A minimal owning float32 buffer with a shape, and non-owning row-major matrix views.
#pragma once

#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace sepal {

/// Non-owning read-only view of a dense row-major float32 matrix.
struct ConstMatView {
    const float* data = nullptr;
    std::size_t rows = 0;
    std::size_t cols = 0;

    constexpr ConstMatView() = default;
    constexpr ConstMatView(const float* d, std::size_t r, std::size_t c) : data(d), rows(r), cols(c) {}

    constexpr std::size_t size() const { return rows * cols; }
    const float& operator()(std::size_t r, std::size_t c) const {
        assert(r < rows && c < cols);
        return data[r * cols + c];
    }
    std::span<const float> row(std::size_t r) const {
        assert(r < rows);
        return {data + r * cols, cols};
    }
    /// Rows [begin, begin + count) of this view.
    ConstMatView slice_rows(std::size_t begin, std::size_t count) const {
        if (begin > rows || count > rows - begin) throw std::out_of_range("ConstMatView::slice_rows");
        return {data + begin * cols, count, cols};
    }
};

/// Non-owning mutable view of a dense row-major float32 matrix.
struct MatView {
    float* data = nullptr;
    std::size_t rows = 0;
    std::size_t cols = 0;

    constexpr MatView() = default;
    constexpr MatView(float* d, std::size_t r, std::size_t c) : data(d), rows(r), cols(c) {}

    constexpr operator ConstMatView() const { return {data, rows, cols}; }

    constexpr std::size_t size() const { return rows * cols; }
    float& operator()(std::size_t r, std::size_t c) const {
        assert(r < rows && c < cols);
        return data[r * cols + c];
    }
    std::span<float> row(std::size_t r) const {
        assert(r < rows);
        return {data + r * cols, cols};
    }
    MatView slice_rows(std::size_t begin, std::size_t count) const {
        if (begin > rows || count > rows - begin) throw std::out_of_range("MatView::slice_rows");
        return {data + begin * cols, count, cols};
    }
};

/// Owning float32 buffer with a shape. Row-major; zero-filled unless values are given.
class Tensor {
public:
    Tensor() = default;

    explicit Tensor(std::vector<std::size_t> shape, float fill = 0.0f)
        : shape_(std::move(shape)), data_(element_count(shape_), fill) {}

    Tensor(std::vector<std::size_t> shape, std::initializer_list<float> values)
        : shape_(std::move(shape)), data_(values) {
        if (data_.size() != element_count(shape_)) {
            throw std::invalid_argument("Tensor: value count does not match the shape");
        }
    }

    const std::vector<std::size_t>& shape() const { return shape_; }
    std::size_t rank() const { return shape_.size(); }
    std::size_t size() const { return data_.size(); }

    float* data() { return data_.data(); }
    const float* data() const { return data_.data(); }
    std::span<float> span() { return data_; }
    std::span<const float> span() const { return data_; }

    float& operator[](std::size_t i) {
        assert(i < data_.size());
        return data_[i];
    }
    const float& operator[](std::size_t i) const {
        assert(i < data_.size());
        return data_[i];
    }

    /// Matrix view of a rank-2 tensor.
    MatView mat() {
        require_rank2();
        return {data_.data(), shape_[0], shape_[1]};
    }
    ConstMatView mat() const {
        require_rank2();
        return {data_.data(), shape_[0], shape_[1]};
    }

private:
    static std::size_t element_count(const std::vector<std::size_t>& shape) {
        std::size_t n = 1;
        for (const std::size_t d : shape) {
            if (d != 0 && n > static_cast<std::size_t>(-1) / d) {
                throw std::length_error("Tensor: shape overflows size_t");
            }
            n *= d;
        }
        return n; // a rank-0 tensor holds one element
    }
    void require_rank2() const {
        if (shape_.size() != 2) throw std::logic_error("Tensor::mat: tensor is not rank 2");
    }

    std::vector<std::size_t> shape_;
    std::vector<float> data_;
};

} // namespace sepal
