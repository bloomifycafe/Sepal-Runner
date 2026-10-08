// Reader for the golden-vector files written by runner/tools/export_golden.py from the NumPy reference engine.
//
// Format "SEPALGV1"; every integer and float is little-endian:
//
//     8 bytes   magic, the ASCII text "SEPALGV1"
//     u32       H, the length of the header in bytes
//     H bytes   the header: one UTF-8 JSON object
//     u32       N, the number of arrays
//     N times:  u32 name length, the name (UTF-8), u32 rank R, R x u64 shape,
//               prod(shape) x f64 values in row-major order
//
// The header holds the geometry, the seed and the test cases; each case names the arrays that hold its expected
// results. The reader is byte-order independent and checks every length against the file size before it reads.
#pragma once

#include <cstddef>
#include <functional>
#include <map>
#include <span>
#include <string>
#include <string_view>

#include "sepal/config.hpp"
#include "sepal/json.hpp"
#include "sepal/tensor.hpp"

namespace sepal {

struct GoldenFile {
    json::Value header;
    std::map<std::string, TensorD, std::less<>> arrays;

    /// Throws std::out_of_range naming the array when it is absent.
    const TensorD& array(std::string_view name) const;

    /// The geometry in the header (field names of the reference engine's ModelConfig), validated.
    ModelConfig config() const;

    /// The arrays under "weights/", keyed by tensor name without that prefix.
    std::map<std::string, TensorD, std::less<>> weights() const;
};

/// Throws std::runtime_error describing the first thing wrong with the bytes.
GoldenFile parse_golden(std::span<const unsigned char> bytes);

/// Reads and parses a file. Throws std::runtime_error when it cannot be opened or is malformed.
GoldenFile load_golden(const std::string& path);

} // namespace sepal
