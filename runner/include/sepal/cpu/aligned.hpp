// A 64-byte aligned, move-only buffer (cache line and zmm aligned). Contents are uninitialised after `resize`.
#pragma once

#ifdef _WIN32
#include <malloc.h>
#else
#include <cstdlib>
#endif

#include <cstddef>
#include <cstring>
#include <new>
#include <type_traits>
#include <utility>

namespace sepal::cpu {

namespace detail {
inline void* aligned_alloc64(std::size_t bytes) {
#ifdef _WIN32
    return _aligned_malloc(bytes, 64);
#else
    return std::aligned_alloc(64, (bytes + 63) / 64 * 64);
#endif
}
inline void aligned_free64(void* p) {
#ifdef _WIN32
    _aligned_free(p);
#else
    std::free(p);
#endif
}
} // namespace detail

template <class T>
class AlignedBuffer {
    static_assert(std::is_trivially_copyable_v<T>);

public:
    AlignedBuffer() = default;
    explicit AlignedBuffer(std::size_t n) { resize(n); }
    ~AlignedBuffer() { release(); }
    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;
    AlignedBuffer(AlignedBuffer&& o) noexcept : data_(std::exchange(o.data_, nullptr)), size_(std::exchange(o.size_, 0)) {}
    AlignedBuffer& operator=(AlignedBuffer&& o) noexcept {
        if (this != &o) {
            release();
            data_ = std::exchange(o.data_, nullptr);
            size_ = std::exchange(o.size_, 0);
        }
        return *this;
    }

    /// Reallocates to `n` elements (old contents are dropped) unless the capacity already suffices.
    void resize(std::size_t n) {
        if (n <= capacity_) {
            size_ = n;
            return;
        }
        release();
        data_ = static_cast<T*>(detail::aligned_alloc64((n == 0 ? 1 : n) * sizeof(T)));
        if (data_ == nullptr) throw std::bad_alloc();
        size_ = n;
        capacity_ = n;
    }
    void zero() {
        if (size_ != 0) std::memset(data_, 0, size_ * sizeof(T));
    }
    T* data() { return data_; }
    const T* data() const { return data_; }
    std::size_t size() const { return size_; }
    T& operator[](std::size_t i) { return data_[i]; }
    const T& operator[](std::size_t i) const { return data_[i]; }

private:
    void release() {
        if (data_ != nullptr) detail::aligned_free64(data_);
        data_ = nullptr;
        size_ = 0;
        capacity_ = 0;
    }
    T* data_ = nullptr;
    std::size_t size_ = 0;
    std::size_t capacity_ = 0;
};

} // namespace sepal::cpu
