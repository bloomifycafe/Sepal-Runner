// The two readouts: from final hidden states at the readout positions to a probability per option.
// A port of sepal_runner/refengine/readout.py, function for function (that file is adapted from
// jaredpalmer/kev@5e42a7a kev/model.py, Apache-2.0).
//
// Every question is read out on its own, from its own hidden states: nothing here mixes questions, so a
// question's probabilities cannot depend on which other questions share its request.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <set>
#include <span>
#include <stdexcept>
#include <vector>

#include "sepal/config.hpp"
#include "sepal/gemm.hpp"
#include "sepal/ref/primitives.hpp"
#include "sepal/tensor.hpp"
#include "sepal/weights.hpp"

namespace sepal::ref {

inline void check_temperature(double temperature) {
    if (!(temperature > 0.0)) throw std::invalid_argument("temperature must be positive");
}

/// One question's logits [K] from the hidden state at its decide token [hidden] and at the end of each of its
/// K options [K, hidden].
///
/// Two linear maps WITH bias into the pointer dim, one for the decide vector and one for the option vectors;
/// the logit of an option is their dot product divided by `sqrt(pointer_dim)` and by the calibration
/// temperature.
template <class T>
std::vector<T> pointer_logits(const GemmProvider& provider, const ModelConfig& config, const Weights<T>& weights,
                              std::span<const T> h_decide, BasicConstMatView<T> h_options, double temperature = 1.0) {
    check_temperature(temperature);
    const Weights<T> head = weights.scope(head_prefix);
    const std::size_t pointer_dim = dim(config.pointer_dim);

    BasicTensor<T> decide = project<T>(provider, {h_decide.data(), 1, h_decide.size()}, head["q.weight"]); // [1, P]
    for (std::size_t i = 0; i < pointer_dim; ++i) decide[i] += head["q.bias"][i];
    BasicTensor<T> options = project(provider, h_options, head["k.weight"]); // [K, P]
    for (std::size_t r = 0; r < h_options.rows; ++r) {
        for (std::size_t i = 0; i < pointer_dim; ++i) options[r * pointer_dim + i] += head["k.bias"][i];
    }

    BasicTensor<T> dots({h_options.rows, 1});
    gemm(provider, BasicConstMatView<T>(options.mat()), BasicConstMatView<T>(decide.mat()), dots.mat());
    std::vector<T> logits(h_options.rows);
    const T root = std::sqrt(static_cast<T>(pointer_dim));
    for (std::size_t r = 0; r < logits.size(); ++r) logits[r] = dots[r] / root / static_cast<T>(temperature);
    return logits;
}

/// One question's probability per option [K]: the softmax of its pointer logits.
template <class T>
std::vector<T> pointer_probabilities(const GemmProvider& provider, const ModelConfig& config, const Weights<T>& weights,
                                     std::span<const T> h_decide, BasicConstMatView<T> h_options, double temperature = 1.0) {
    std::vector<T> probabilities = pointer_logits(provider, config, weights, h_decide, h_options, temperature);
    softmax_inplace(std::span<T>(probabilities));
    return probabilities;
}

/// The embedding rows [K, hidden] of a question's valid label tokens (the vocabulary projection's rows where it
/// is tied to the embedding). Labels must be distinct, or one answer would be counted twice.
template <class T>
BasicTensor<T> label_rows(const ModelConfig& config, const Weights<T>& weights, std::span<const std::uint32_t> label_token_ids) {
    const std::set<std::uint32_t> distinct(label_token_ids.begin(), label_token_ids.end());
    if (label_token_ids.empty() || distinct.size() != label_token_ids.size()) {
        throw std::invalid_argument("a question needs at least one label token, each listed once");
    }
    for (const std::uint32_t token : label_token_ids) {
        if (token >= config.vocab_size) throw std::invalid_argument("a label token id is outside the vocabulary");
    }
    const BasicConstMatView<T> table = weights[embedding_name].mat();
    BasicTensor<T> rows({label_token_ids.size(), table.cols});
    for (std::size_t r = 0; r < label_token_ids.size(); ++r) {
        const std::span<const T> row = table.row(label_token_ids[r]);
        std::copy(row.begin(), row.end(), rows.data() + r * table.cols);
    }
    return rows;
}

/// One question's logits [K]: the hidden state at its answer slot [hidden] times its label rows [K, hidden],
/// divided by the calibration temperature. Only these K rows of the vocabulary are ever read.
template <class T>
std::vector<T> label_row_logits(const GemmProvider& provider, std::span<const T> h_slot, const BasicTensor<T>& rows,
                                double temperature = 1.0) {
    check_temperature(temperature);
    const BasicTensor<T> product = project<T>(provider, {h_slot.data(), 1, h_slot.size()}, rows); // [1, K]
    std::vector<T> logits(product.size());
    for (std::size_t r = 0; r < logits.size(); ++r) logits[r] = product[r] / static_cast<T>(temperature);
    return logits;
}

/// One question's probability per label [K]: the softmax over its valid labels only.
template <class T>
std::vector<T> label_row_probabilities(const GemmProvider& provider, std::span<const T> h_slot, const BasicTensor<T>& rows,
                                       double temperature = 1.0) {
    std::vector<T> probabilities = label_row_logits(provider, h_slot, rows, temperature);
    softmax_inplace(std::span<T>(probabilities));
    return probabilities;
}

} // namespace sepal::ref
