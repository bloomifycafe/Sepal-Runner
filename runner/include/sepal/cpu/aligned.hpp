// A 64-byte aligned, move-only buffer (cache line and zmm aligned). Contents are uninitialised after `resize`.
// On Linux a buffer of 8 MiB and more (packed weights, step workspaces) is 2 MiB aligned and advised for transparent huge
// pages (madvise), so streaming the weight panels does not walk a 4 KiB page table entry per 4 KiB; SEPAL_HUGEPAGES=0 turns
// that off.
#pragma once

#ifdef _WIN32
#include <malloc.h>
#else
#include <cstdlib>
#endif
#if defined(__linux__)
#include <sys/mman.h>
#endif

#include <cstddef>
#include <cstring>
#include <new>
#include <type_traits>
#include <utility>

namespace sepal::cpu {

namespace detail {
#if defined(__linux__)
// Linux only (MSVC never compiles this): std::getenv is fine here.
inline bool huge_pages_wanted() {
    static const bool wanted = [] {
        const char* env = std::getenv("SEPAL_HUGEPAGES");  // NOLINT(concurrency-mt-unsafe): read once
        return env == nullptr || std::strcmp(env, "0") != 0;
    }();
    return wanted;
}
#endif
inline void* aligned_alloc64(std::size_t bytes) {
#ifdef _WIN32
    return _aligned_malloc(bytes, 64);
#else
#if defined(__linux__)
    constexpr std::size_t huge = std::size_t{2} << 20;
    if (bytes >= 4 * huge && huge_pages_wanted()) {
        const std::size_t rounded = (bytes + huge - 1) / huge * huge;
        void* p = std::aligned_alloc(huge, rounded);
        if (p != nullptr) madvise(p, rounded, MADV_HUGEPAGE);
        return p;
    }
#endif
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
    AlignedBuffer(AlignedBuffer&& o) noexcept
        : data_(std::exchange(o.data_, nullptr)), size_(std::exchange(o.size_, 0)), capacity_(std::exchange(o.capacity_, 0)) {}
    AlignedBuffer& operator=(AlignedBuffer&& o) noexcept {
        if (this != &o) {
            release();
            data_ = std::exchange(o.data_, nullptr);
            size_ = std::exchange(o.size_, 0);
            capacity_ = std::exchange(o.capacity_, 0);
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
