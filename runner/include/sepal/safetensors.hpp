// Reader for the safetensors format (https://github.com/huggingface/safetensors), written against the format
// description and nothing else:
//
//     u64 (little endian)  N, the length of the header in bytes
//     N bytes              the header: one JSON object. Each member names a tensor and holds its "dtype", its
//                          "shape" and its "data_offsets" [begin, end), counted from the end of the header.
//                          The member "__metadata__", when present, maps strings to strings.
//     the rest             the tensors' bytes, little endian, row-major, one after the other
//
// A file is mapped read-only and parsed once; each tensor is then a view (name, element type, shape, pointer)
// into the mapping, never a copy. The reader checks everything it later relies on, so a view that was handed out
// is inside the file, matches its shape, is aligned for its element type and overlaps no other tensor. A file
// that breaks a rule is refused with a message that names the file and the tensor.
//
// A checkpoint is one such file, or several (`model-00001-of-0000N.safetensors`, usually with an index).
#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "sepal/dtype.hpp"
#include "sepal/mapped_file.hpp"

namespace sepal {

/// The element types the format defines for numbers. Narrower than the format's whole list on purpose: the
/// 8-bit float formats and complex numbers are refused as unknown.
enum class StType { Bool, U8, I8, I16, U16, F16, BF16, I32, U32, F32, F64, I64, U64 };

std::size_t bytes_of(StType type) noexcept;
/// The name the format uses ("F32", "BF16", ...).
std::string_view name_of(StType type) noexcept;
std::optional<StType> parse_st_type(std::string_view name) noexcept;
/// The engine's element type for a stored type, when it has one (F32, F16, BF16, I8).
std::optional<DType> as_dtype(StType type) noexcept;

/// Thrown for a file that is not a valid safetensors file, with the reason in the message.
class SafetensorsError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

/// One tensor of a file: where its bytes are, and what they mean. Valid while the file object lives.
struct TensorView {
    std::string name;
    StType type = StType::F32;
    std::vector<std::size_t> shape;
    const unsigned char* data = nullptr;
    std::size_t bytes = 0;

    std::size_t elements() const;

    /// The values as a span of T, when T is the C++ type of this tensor's element type (float for F32, double for
    /// F64, std::int32_t for I32, ...). The 16-bit float formats have no C++ type: use `bits()`.
    /// Throws std::logic_error on another type.
    template <class T>
    std::span<const T> values() const;

    /// The 16-bit patterns of an F16 or BF16 tensor (also valid for I16 and U16).
    std::span<const std::uint16_t> bits() const;
};

namespace detail {
template <class T>
struct StTypeOf;
template <> struct StTypeOf<float> { static constexpr StType value = StType::F32; };
template <> struct StTypeOf<double> { static constexpr StType value = StType::F64; };
template <> struct StTypeOf<std::int8_t> { static constexpr StType value = StType::I8; };
template <> struct StTypeOf<std::uint8_t> { static constexpr StType value = StType::U8; };
template <> struct StTypeOf<std::int16_t> { static constexpr StType value = StType::I16; };
template <> struct StTypeOf<std::uint16_t> { static constexpr StType value = StType::U16; };
template <> struct StTypeOf<std::int32_t> { static constexpr StType value = StType::I32; };
template <> struct StTypeOf<std::uint32_t> { static constexpr StType value = StType::U32; };
template <> struct StTypeOf<std::int64_t> { static constexpr StType value = StType::I64; };
template <> struct StTypeOf<std::uint64_t> { static constexpr StType value = StType::U64; };
} // namespace detail

template <class T>
std::span<const T> TensorView::values() const {
    if (type != detail::StTypeOf<T>::value) {
        throw std::logic_error("tensor '" + name + "' holds " + std::string(name_of(type)) + ", not the requested type");
    }
    return {reinterpret_cast<const T*>(data), bytes / sizeof(T)};
}

/// A parsed file.
class SafetensorsFile {
public:
    /// Maps the file and parses it. Throws SafetensorsError for a malformed file and std::runtime_error when it
    /// cannot be mapped.
    static SafetensorsFile open(const std::string& path);

    /// Parses bytes the caller keeps alive, and keeps alive for as long as the views are used. `origin` names
    /// the bytes in error messages.
    static SafetensorsFile from_memory(std::span<const unsigned char> bytes, std::string origin = "memory");

    /// Tensors in file order (by position of their bytes).
    const std::vector<TensorView>& tensors() const { return tensors_; }
    const TensorView* find(std::string_view name) const;
    /// Throws std::out_of_range naming the tensor and the file.
    const TensorView& at(std::string_view name) const;
    const std::map<std::string, std::string, std::less<>>& metadata() const { return metadata_; }
    const std::string& origin() const { return origin_; }

private:
    static SafetensorsFile parse(std::span<const unsigned char> bytes, std::string origin);

    std::unique_ptr<MappedFile> mapping_; // null when the bytes are borrowed
    std::string origin_;
    std::vector<TensorView> tensors_;
    std::map<std::string, std::size_t, std::less<>> index_;
    std::map<std::string, std::string, std::less<>> metadata_;
};

/// All the tensors of a checkpoint directory (or list of files) under their checkpoint names.
class Checkpoint {
public:
    /// A directory holding `model.safetensors`, or `model.safetensors.index.json` with the shards it names, or
    /// the shards `model-00001-of-0000N.safetensors` alone (all N must be present). Throws when none of these
    /// is found, when the index and the shards disagree, and when two files hold the same tensor name.
    static Checkpoint open_directory(const std::string& directory);

    /// Exactly these files, in this order.
    static Checkpoint open_files(const std::vector<std::string>& paths);

    const std::vector<std::shared_ptr<const SafetensorsFile>>& files() const { return files_; }
    const TensorView* find(std::string_view name) const;
    const TensorView& at(std::string_view name) const;
    /// Every tensor name, sorted.
    std::vector<std::string> names() const;
    std::size_t tensor_count() const { return by_name_.size(); }

private:
    void add(std::shared_ptr<const SafetensorsFile> file);

    std::vector<std::shared_ptr<const SafetensorsFile>> files_;
    std::map<std::string, const TensorView*, std::less<>> by_name_;
};

/// The values of an F32, F16, BF16 or F64 tensor converted to T (float or double), written to `out`, which must
/// hold `tensor.elements()` values. Conversions from F16 and BF16 are exact. Throws std::invalid_argument for
/// another type.
void convert_floating(const TensorView& tensor, float* out);
void convert_floating(const TensorView& tensor, double* out);

} // namespace sepal
