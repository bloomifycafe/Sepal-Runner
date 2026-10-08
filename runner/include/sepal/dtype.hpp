// Element types of stored tensors, and exact decoding of the 16-bit float formats to float32.
#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace sepal {

enum class DType { F32, F16, BF16, I8 };

constexpr std::size_t bytes_of(DType d) noexcept {
    switch (d) {
    case DType::F32: return 4;
    case DType::F16: return 2;
    case DType::BF16: return 2;
    case DType::I8: return 1;
    }
    return 0;
}

constexpr std::string_view name_of(DType d) noexcept {
    switch (d) {
    case DType::F32: return "f32";
    case DType::F16: return "f16";
    case DType::BF16: return "bf16";
    case DType::I8: return "i8";
    }
    return "?";
}

/// IEEE 754 binary16 bits to float32. Exact: every binary16 value is representable in binary32.
constexpr float half_to_float(std::uint16_t h) noexcept {
    const std::uint32_t x = h;
    const std::uint32_t sign = (x & 0x8000u) << 16;
    const std::uint32_t exponent = (x >> 10) & 0x1Fu;
    std::uint32_t mantissa = x & 0x3FFu;
    std::uint32_t bits = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign; // signed zero
        } else {
            // Subnormal: value = mantissa * 2^-24. Shift until the implicit bit appears.
            std::uint32_t shift = 0;
            while ((mantissa & 0x400u) == 0) {
                mantissa <<= 1;
                ++shift;
            }
            bits = sign | ((113u - shift) << 23) | ((mantissa & 0x3FFu) << 13);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7F800000u | (mantissa << 13); // infinity or NaN
    } else {
        bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
    }
    return std::bit_cast<float>(bits);
}

/// bfloat16 bits to float32. Exact: bfloat16 is the upper half of a float32.
constexpr float bf16_to_float(std::uint16_t b) noexcept {
    return std::bit_cast<float>(static_cast<std::uint32_t>(b) << 16);
}

} // namespace sepal
