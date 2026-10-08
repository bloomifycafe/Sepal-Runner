// Shared by the suites that compare the reference path with the NumPy oracle: the golden file, the model built
// from it in either element type, and comparison helpers that report the largest difference they saw.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "sepal/golden.hpp"
#include "sepal/ref/executor.hpp"
#include "sepal/weights.hpp"
#include "test.hpp"

#ifndef SEPAL_GOLDEN_DIR
#error "SEPAL_GOLDEN_DIR must be defined by the build"
#endif

namespace sepal::test {

/// The tolerance of every float64 comparison with the oracle.
inline constexpr double oracle_tolerance = 1e-9;

inline const GoldenFile& golden() {
    static const GoldenFile file = load_golden(std::string(SEPAL_GOLDEN_DIR) + "/tiny.sgv");
    return file;
}

inline const json::Value& golden_case(const char* name) { return golden().header.at("cases").at(name); }

/// The array a header entry names.
inline const TensorD& named(const json::Value& entry, const char* key) {
    return golden().array(entry.at(key).as_string());
}

template <class T>
struct Model {
    ModelConfig config;
    Weights<T> weights;
    std::unique_ptr<GemmProvider> gemm;
};

template <class T>
const Model<T>& model() {
    static const Model<T> m = [] {
        Model<T> built;
        built.config = golden().config();
        built.weights = Weights<T>::from_arrays(built.config, golden().weights());
        built.gemm = make_reference_gemm();
        return built;
    }();
    return m;
}

template <class T>
BasicTensor<T> cast(const TensorD& source) {
    BasicTensor<T> out(source.shape());
    for (std::size_t i = 0; i < source.size(); ++i) out[i] = static_cast<T>(source[i]);
    return out;
}

inline std::vector<std::uint32_t> token_list(const json::Value& list) {
    std::vector<std::uint32_t> out;
    for (const std::uint64_t v : list.as_uint_list()) out.push_back(static_cast<std::uint32_t>(v));
    return out;
}

inline std::vector<std::size_t> index_list(const json::Value& list) {
    std::vector<std::size_t> out;
    for (const std::uint64_t v : list.as_uint_list()) out.push_back(static_cast<std::size_t>(v));
    return out;
}

/// The largest |got - want| over two buffers of the same length; infinity when the lengths differ or a value
/// is not finite, so such a result can never pass a tolerance.
template <class T>
double max_abs_diff(std::span<const T> got, std::span<const double> want) {
    const double never = std::numeric_limits<double>::infinity();
    if (got.size() != want.size()) return never;
    double worst = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double diff = std::fabs(static_cast<double>(got[i]) - want[i]);
        if (!std::isfinite(diff)) return never;
        worst = std::max(worst, diff);
    }
    return worst;
}

/// Collects the largest difference over the comparisons of one test case and prints it once, so the numbers a
/// run measured can be read from its output.
class Worst {
public:
    explicit Worst(std::string label) : label_(std::move(label)) {}
    ~Worst() {
        std::ostringstream line;
        line << "    " << label_ << ": max |diff| = " << std::scientific << std::setprecision(3) << worst_ << " over "
             << arrays_ << " array(s)\n";
        std::cout << line.str();
    }
    Worst(const Worst&) = delete;
    Worst& operator=(const Worst&) = delete;

    /// Records the difference and returns it. Element counts must agree; shapes may differ in how the same
    /// values are grouped (the oracle keeps [T, heads, dim] where a kernel here works on rows).
    template <class T>
    double see(std::span<const T> got, std::span<const double> want) {
        const double diff = max_abs_diff(got, want);
        worst_ = std::max(worst_, diff);
        ++arrays_;
        return diff;
    }
    double value() const { return worst_; }

private:
    std::string label_;
    double worst_ = 0.0;
    int arrays_ = 0;
};

template <class T>
void expect_close(Worst& worst, const std::string& what, std::span<const T> got, std::span<const double> want, double tolerance,
                  const char* file, int line) {
    const double diff = worst.see(got, want);
    std::ostringstream text;
    text << what << " within " << tolerance << " of the oracle (max |diff| " << diff << ", " << got.size() << " vs "
         << want.size() << " values)";
    report(diff <= tolerance, text.str(), file, line);
}

template <class T>
void expect_close(Worst& worst, const std::string& what, const BasicTensor<T>& got, const TensorD& want, double tolerance,
                  const char* file, int line) {
    expect_close<T>(worst, what, got.span(), want.span(), tolerance, file, line);
}

template <class T>
void expect_close(Worst& worst, const std::string& what, const std::vector<T>& got, const TensorD& want, double tolerance,
                  const char* file, int line) {
    expect_close<T>(worst, what, std::span<const T>(got), want.span(), tolerance, file, line);
}

/// A snapshot built from the arrays of a header entry {"n_tokens", "layers": [...]}: the path a stored snapshot
/// takes into the engine.
template <class T>
ref::SnapshotPtr<T> snapshot_from_golden(const json::Value& entry) {
    auto snapshot = std::make_shared<ref::Snapshot<T>>();
    snapshot->n_tokens = static_cast<std::size_t>(entry.at("n_tokens").as_uint());
    for (const json::Value& layer : entry.at("layers").items()) {
        if (layer.at("kind").as_string() == "deltanet") {
            auto state = std::make_shared<ref::DeltaNetState<T>>();
            state->recurrent = cast<T>(named(layer, "recurrent"));
            state->conv_tail = cast<T>(named(layer, "conv_tail"));
            snapshot->layers.emplace_back(ref::DeltaNetStatePtr<T>(std::move(state)));
        } else {
            auto state = std::make_shared<ref::AttentionState<T>>();
            state->keys = cast<T>(named(layer, "keys"));
            state->values = cast<T>(named(layer, "values"));
            state->tokens = state->keys.shape().at(0);
            snapshot->layers.emplace_back(ref::AttentionStatePtr<T>(std::move(state)));
        }
    }
    return snapshot;
}

/// The rows of the golden request, in request order.
template <class T>
std::vector<ref::Row<T>> request_rows() {
    std::vector<ref::Row<T>> rows;
    for (const json::Value& entry : golden_case("request").at("rows").items()) {
        ref::Row<T> row;
        row.token_ids = token_list(entry.at("token_ids"));
        row.start_position = entry.at("start_position").as_uint();
        if (!entry.at("parent").is_null()) row.parent = ref::ParentRow{static_cast<std::size_t>(entry.at("parent").as_uint())};
        row.readout_offsets = index_list(entry.at("readout_offsets"));
        rows.push_back(std::move(row));
    }
    return rows;
}

/// Index of the request row with this name.
inline std::size_t request_row_index(const std::string& name) {
    const std::vector<json::Value>& rows = golden_case("request").at("rows").items();
    for (std::size_t i = 0; i < rows.size(); ++i) {
        if (rows[i].at("name").as_string() == name) return i;
    }
    throw std::runtime_error("golden file: no request row named " + name);
}

} // namespace sepal::test

#define EXPECT_CLOSE(worst, what, got, want, tolerance) \
    ::sepal::test::expect_close((worst), (what), (got), (want), (tolerance), __FILE__, __LINE__)
