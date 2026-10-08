#include <cmath>
#include <limits>

#include "sepal/dtype.hpp"
#include "test.hpp"

using namespace sepal;

SEPAL_TEST(dtype, sizes_and_names) {
    CHECK_EQ(bytes_of(DType::F32), 4u);
    CHECK_EQ(bytes_of(DType::F16), 2u);
    CHECK_EQ(bytes_of(DType::BF16), 2u);
    CHECK_EQ(bytes_of(DType::I8), 1u);
    CHECK_EQ(name_of(DType::BF16), "bf16");
}

SEPAL_TEST(dtype, half_decodes_exactly) {
    CHECK_EQ(half_to_float(0x0000), 0.0f);
    CHECK(std::signbit(half_to_float(0x8000)));
    CHECK_EQ(half_to_float(0x3C00), 1.0f);
    CHECK_EQ(half_to_float(0xC000), -2.0f);
    CHECK_EQ(half_to_float(0x3800), 0.5f);
    CHECK_EQ(half_to_float(0x3555), 0.333251953125f); // nearest half to 1/3
    CHECK_EQ(half_to_float(0x7BFF), 65504.0f);        // largest finite half
    CHECK_EQ(half_to_float(0x0400), 6.103515625e-05f); // smallest normal, 2^-14
    CHECK_EQ(half_to_float(0x0001), 5.9604644775390625e-08f); // smallest subnormal, 2^-24
    CHECK_EQ(half_to_float(0x03FF), 1023.0f * 5.9604644775390625e-08f); // largest subnormal
    CHECK_EQ(half_to_float(0x7C00), std::numeric_limits<float>::infinity());
    CHECK_EQ(half_to_float(0xFC00), -std::numeric_limits<float>::infinity());
    CHECK(std::isnan(half_to_float(0x7E00)));
}

SEPAL_TEST(dtype, bf16_decodes_exactly) {
    CHECK_EQ(bf16_to_float(0x0000), 0.0f);
    CHECK_EQ(bf16_to_float(0x3F80), 1.0f);
    CHECK_EQ(bf16_to_float(0xBF80), -1.0f);
    CHECK_EQ(bf16_to_float(0x4000), 2.0f);
    CHECK_EQ(bf16_to_float(0x4049), 3.140625f); // pi truncated to bf16
    CHECK(std::isinf(bf16_to_float(0x7F80)));
}
