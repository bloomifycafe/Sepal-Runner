#include <stdexcept>

#include "sepal/tensor.hpp"
#include "test.hpp"

using namespace sepal;

SEPAL_TEST(tensor, owns_a_zero_filled_buffer_with_a_shape) {
    Tensor t({2, 3, 4});
    CHECK_EQ(t.rank(), 3u);
    CHECK_EQ(t.size(), 24u);
    CHECK_EQ(t.shape()[1], 3u);
    for (const float v : t.span()) CHECK_EQ(v, 0.0f);

    const Tensor filled({2, 2}, 1.5f);
    CHECK_EQ(filled[3], 1.5f);

    const Tensor empty;
    CHECK_EQ(empty.size(), 0u);

    const Tensor no_rows({0, 5});
    CHECK_EQ(no_rows.size(), 0u);
}

SEPAL_TEST(tensor, takes_values_and_checks_their_count) {
    const Tensor t({2, 3}, {1, 2, 3, 4, 5, 6});
    CHECK_EQ(t[0], 1.0f);
    CHECK_EQ(t[5], 6.0f);
    CHECK_THROWS(Tensor({2, 3}, {1, 2, 3}), std::invalid_argument);
}

SEPAL_TEST(tensor, matrix_views_are_row_major_and_do_not_own) {
    Tensor t({2, 3}, {1, 2, 3, 4, 5, 6});
    const MatView m = t.mat();
    CHECK_EQ(m.rows, 2u);
    CHECK_EQ(m.cols, 3u);
    CHECK_EQ(m(1, 0), 4.0f);
    CHECK_EQ(m.row(1)[2], 6.0f);
    CHECK_EQ(m.row(0).size(), 3u);

    m(0, 1) = 20.0f; // writes through to the owner
    CHECK_EQ(t[1], 20.0f);

    const ConstMatView c = m;
    CHECK_EQ(c(0, 1), 20.0f);
    CHECK(c.data == t.data());

    const Tensor& const_ref = t;
    CHECK_EQ(const_ref.mat()(1, 2), 6.0f);
}

SEPAL_TEST(tensor, row_slices) {
    Tensor t({4, 2}, {0, 1, 10, 11, 20, 21, 30, 31});
    const MatView tail = t.mat().slice_rows(2, 2);
    CHECK_EQ(tail.rows, 2u);
    CHECK_EQ(tail(0, 0), 20.0f);
    CHECK_EQ(tail(1, 1), 31.0f);
    const ConstMatView none = ConstMatView(t.mat()).slice_rows(4, 0);
    CHECK_EQ(none.rows, 0u);
    CHECK_THROWS(t.mat().slice_rows(3, 2), std::out_of_range);
    CHECK_THROWS(ConstMatView(t.mat()).slice_rows(5, 0), std::out_of_range);
}

SEPAL_TEST(tensor, mat_requires_rank_two) {
    Tensor t({2, 3, 4});
    CHECK_THROWS(t.mat(), std::logic_error);
}
