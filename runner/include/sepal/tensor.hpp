// A minimal owning buffer with a shape, and non-owning row-major matrix views. Each is a template over the
// element type; the unsuffixed names are the float32 ones the engine computes in, and the D-suffixed names are
// the float64 ones of the reference path.
#pragma once

#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace sepal {

/// Non-owning read-only view of a dense row-major matrix.
template <class T>
struct BasicConstMatView {
    const T* data = nullptr;
    std::size_t rows = 0;
    std::size_t cols = 0;

    constexpr BasicConstMatView() = default;
    constexpr BasicConstMatView(const T* d, std::size_t r, std::size_t c) : data(d), rows(r), cols(c) {}

    constexpr std::size_t size() const { return rows * cols; }
    const T& operator()(std::size_t r, std::size_t c) const {
        assert(r < rows && c < cols);
        return data[r * cols + c];
    }
    std::span<const T> row(std::size_t r) const {
        assert(r < rows);
        return {data + r * cols, cols};
    }
    /// Rows [begin, begin + count) of this view.
    BasicConstMatView slice_rows(std::size_t begin, std::size_t count) const {
        if (begin > rows || count > rows - begin) throw std::out_of_range("ConstMatView::slice_rows");
        return {data + begin * cols, count, cols};
    }
};

/// Non-owning mutable view of a dense row-major matrix.
template <class T>
struct BasicMatView {
    T* data = nullptr;
    std::size_t rows = 0;
    std::size_t cols = 0;

    constexpr BasicMatView() = default;
    constexpr BasicMatView(T* d, std::size_t r, std::size_t c) : data(d), rows(r), cols(c) {}

    constexpr operator BasicConstMatView<T>() const { return {data, rows, cols}; }

    constexpr std::size_t size() const { return rows * cols; }
    T& operator()(std::size_t r, std::size_t c) const {
        assert(r < rows && c < cols);
        return data[r * cols + c];
    }
    std::span<T> row(std::size_t r) const {
        assert(r < rows);
        return {data + r * cols, cols};
    }
    BasicMatView slice_rows(std::size_t begin, std::size_t count) const {
        if (begin > rows || count > rows - begin) throw std::out_of_range("MatView::slice_rows");
        return {data + begin * cols, count, cols};
    }
};

/// Owning buffer with a shape. Row-major; zero-filled unless values are given.
template <class T>
class BasicTensor {
public:
    BasicTensor() = default;

    explicit BasicTensor(std::vector<std::size_t> shape, T fill = T(0))
        : shape_(std::move(shape)), data_(element_count(shape_), fill) {}

    BasicTensor(std::vector<std::size_t> shape, std::initializer_list<T> values)
        : shape_(std::move(shape)), data_(values) {
        if (data_.size() != element_count(shape_)) {
            throw std::invalid_argument("Tensor: value count does not match the shape");
        }
    }

    const std::vector<std::size_t>& shape() const { return shape_; }
    std::size_t rank() const { return shape_.size(); }
    std::size_t size() const { return data_.size(); }

    T* data() { return data_.data(); }
    const T* data() const { return data_.data(); }
    std::span<T> span() { return data_; }
    std::span<const T> span() const { return data_; }

    T& operator[](std::size_t i) {
        assert(i < data_.size());
        return data_[i];
    }
    const T& operator[](std::size_t i) const {
        assert(i < data_.size());
        return data_[i];
    }

    /// Matrix view of a rank-2 tensor.
    BasicMatView<T> mat() {
        require_rank2();
        return {data_.data(), shape_[0], shape_[1]};
    }
    BasicConstMatView<T> mat() const {
        require_rank2();
        return {data_.data(), shape_[0], shape_[1]};
    }

    /// The whole buffer as a matrix with `cols` columns, whatever the rank: [T, heads, dim] read as
    /// [T * heads, dim] or as [T, heads * dim]. Throws if the size is not a multiple of `cols`.
    BasicMatView<T> as_mat(std::size_t cols) { return {data_.data(), rows_for(cols), cols}; }
    BasicConstMatView<T> as_mat(std::size_t cols) const { return {data_.data(), rows_for(cols), cols}; }

    /// Gives the same values another shape of the same size.
    void reshape(std::vector<std::size_t> shape) {
        if (element_count(shape) != data_.size()) throw std::invalid_argument("Tensor::reshape: the size differs");
        shape_ = std::move(shape);
    }

    /// A tensor that takes ownership of `values`.
    static BasicTensor from_values(std::vector<std::size_t> shape, std::vector<T> values) {
        if (values.size() != element_count(shape)) {
            throw std::invalid_argument("Tensor: value count does not match the shape");
        }
        BasicTensor t;
        t.shape_ = std::move(shape);
        t.data_ = std::move(values);
        return t;
    }

private:
    std::size_t rows_for(std::size_t cols) const {
        if (cols == 0 || data_.size() % cols != 0) {
            throw std::invalid_argument("Tensor::as_mat: the size is not a multiple of the column count");
        }
        return data_.size() / cols;
    }
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
    std::vector<T> data_;
};

using ConstMatView = BasicConstMatView<float>;
using MatView = BasicMatView<float>;
using Tensor = BasicTensor<float>;

using ConstMatViewD = BasicConstMatView<double>;
using MatViewD = BasicMatView<double>;
using TensorD = BasicTensor<double>;

} // namespace sepal
