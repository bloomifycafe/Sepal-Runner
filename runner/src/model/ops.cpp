#include "sepal/ops.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace sepal {

void rmsnorm(ConstMatView x, std::span<const float> w, double eps, MatView y) {
    if (w.size() != x.cols) throw std::invalid_argument("rmsnorm: weight length differs from the row width");
    if (y.rows != x.rows || y.cols != x.cols) throw std::invalid_argument("rmsnorm: y shape differs from x");
    if (x.cols == 0) return;

    for (std::size_t t = 0; t < x.rows; ++t) {
        double sum_sq = 0.0;
        for (std::size_t i = 0; i < x.cols; ++i) {
            const double v = static_cast<double>(x(t, i));
            sum_sq += v * v;
        }
        const double inv_rms = 1.0 / std::sqrt(sum_sq / static_cast<double>(x.cols) + eps);
        for (std::size_t i = 0; i < x.cols; ++i) {
            const double scale = 1.0 + static_cast<double>(w[i]);
            y(t, i) = static_cast<float>(static_cast<double>(x(t, i)) * inv_rms * scale);
        }
    }
}

float silu(float x) {
    const double v = static_cast<double>(x);
    return static_cast<float>(v / (1.0 + std::exp(-v)));
}

void silu_inplace(std::span<float> x) {
    for (float& v : x) v = silu(v);
}

void swiglu_mlp(const GemmProvider& gemm, ConstMatView x, MatRef w_gate_up, MatRef w_down, MatView y, float beta) {
    if (w_gate_up.rows % 2 != 0) throw std::invalid_argument("swiglu_mlp: w_gate_up must have 2 I rows");
    const std::size_t inter = w_gate_up.rows / 2;
    if (w_down.cols != inter) {
        throw std::invalid_argument("swiglu_mlp: w_down has " + std::to_string(w_down.cols) +
                                    " columns but the intermediate size is " + std::to_string(inter));
    }

    Tensor gate_up({x.rows, 2 * inter});
    gemm.gemm_f(x, w_gate_up, gate_up.mat(), 0.0f);

    Tensor act({x.rows, inter});
    const MatView gu = gate_up.mat();
    const MatView a = act.mat();
    for (std::size_t t = 0; t < x.rows; ++t) {
        for (std::size_t i = 0; i < inter; ++i) {
            a(t, i) = silu(gu(t, i)) * gu(t, inter + i);
        }
    }
    gemm.gemm_f(act.mat(), w_down, y, beta);
}

void softmax(std::span<const float> logits, std::span<float> probs) {
    if (logits.empty()) throw std::invalid_argument("softmax: empty input");
    if (probs.size() != logits.size()) throw std::invalid_argument("softmax: output length differs from input");

    const double peak = static_cast<double>(*std::max_element(logits.begin(), logits.end()));
    std::vector<double> e(logits.size());
    double sum = 0.0;
    for (std::size_t i = 0; i < logits.size(); ++i) {
        e[i] = std::exp(static_cast<double>(logits[i]) - peak);
        sum += e[i];
    }
    for (std::size_t i = 0; i < logits.size(); ++i) probs[i] = static_cast<float>(e[i] / sum);
}

void causal_conv1d_depthwise(ConstMatView x, ConstMatView w, ConstMatView tail_in, MatView y, MatView tail_out) {
    const std::size_t channels = w.rows;
    const std::size_t taps = w.cols;
    if (taps == 0) throw std::invalid_argument("causal_conv1d_depthwise: the filter has no taps");
    const std::size_t history = taps - 1;
    const std::size_t steps = x.rows;

    if (x.cols != channels) throw std::invalid_argument("causal_conv1d_depthwise: x width differs from channels");
    if (y.rows != steps || y.cols != channels) {
        throw std::invalid_argument("causal_conv1d_depthwise: y shape differs from x");
    }
    if (tail_in.rows != history || (history != 0 && tail_in.cols != channels)) {
        throw std::invalid_argument("causal_conv1d_depthwise: tail_in must be [K-1, C]");
    }
    const bool write_tail = tail_out.rows != 0;
    if (write_tail && (tail_out.rows != history || tail_out.cols != channels)) {
        throw std::invalid_argument("causal_conv1d_depthwise: tail_out must be [K-1, C] or empty");
    }

    // Element i of concat(tail_in, x), for channel c.
    const auto input = [&](std::size_t i, std::size_t c) -> float {
        return i < history ? tail_in(i, c) : x(i - history, c);
    };

    for (std::size_t t = 0; t < steps; ++t) {
        for (std::size_t c = 0; c < channels; ++c) {
            double acc = 0.0;
            for (std::size_t j = 0; j < taps; ++j) {
                acc += static_cast<double>(w(c, j)) * static_cast<double>(input(t + j, c));
            }
            y(t, c) = static_cast<float>(acc);
        }
    }

    if (write_tail) {
        // Built in a temporary because tail_out may be tail_in, and a short x leaves part of the old tail in place.
        std::vector<float> next(history * channels);
        for (std::size_t i = 0; i < history; ++i) {
            for (std::size_t c = 0; c < channels; ++c) next[i * channels + c] = input(steps + i, c);
        }
        std::copy(next.begin(), next.end(), tail_out.data);
    }
}

} // namespace sepal
