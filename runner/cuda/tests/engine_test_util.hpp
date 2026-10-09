// Shared helpers of the engine tests: random request DAGs for the engine and the CPU reference at once,
// comparisons, and the environment of the real-model tests.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "sepal/config.hpp"
#include "sepal/cuda/cuda_engine.hpp"
#include "sepal/gemm.hpp"
#include "sepal/ref/executor.hpp"
#include "sepal/weights.hpp"
#include "test.hpp"

namespace engine_test {

using namespace sepal;

/// splitmix64: the same seeded sequence on every platform.
class Rng {
public:
    explicit Rng(std::uint64_t seed) : state_(seed * 0x9E3779B97F4A7C15ull + 12345) {}
    std::uint64_t next() {
        state_ += 0x9E3779B97F4A7C15ull;
        std::uint64_t z = state_;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    /// Uniform in [0, n).
    std::size_t below(std::size_t n) { return static_cast<std::size_t>(next() % n); }
    bool chance(int percent) { return below(100) < static_cast<std::size_t>(percent); }
    template <class T>
    const T& pick(const std::vector<T>& v) { return v[below(v.size())]; }

private:
    std::uint64_t state_;
};

inline std::optional<std::string> environment(const char* name) {
#if defined(_MSC_VER)
    char* buffer = nullptr;
    std::size_t length = 0;
    if (_dupenv_s(&buffer, &length, name) != 0 || buffer == nullptr) return std::nullopt;
    const std::string value(buffer);
    std::free(buffer);
#else
    const char* raw = std::getenv(name);
    const std::string value = raw == nullptr ? std::string() : std::string(raw);
#endif
    if (value.empty()) return std::nullopt;
    return value;
}

inline double max_abs_diff(const float* a, const float* b, std::size_t n) {
    double worst = 0.0;
    for (std::size_t i = 0; i < n; ++i) worst = std::max(worst, std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
    return worst;
}

inline bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}

inline bool all_finite(const std::vector<float>& v) {
    for (const float f : v) {
        if (!std::isfinite(f)) return false;
    }
    return true;
}

/// A snapshot kept by both implementations.
struct Stored {
    cuda::DeviceSnapshot device;
    ref::SnapshotPtr<float> reference;
    ref::SnapshotPtr<double> reference64;
    std::uint64_t end_position = 0;
    std::uint64_t tokens = 0;
};

/// One request, as the engine and the reference see it.
struct Request {
    std::vector<cuda::EngineRow> engine;
    std::vector<ref::Row<float>> reference;
    std::vector<ref::Row<double>> reference64; // the same request in float64, for telling rounding noise from errors
    std::vector<std::uint64_t> end_position; // per row
    std::vector<std::uint64_t> tokens;       // per row, with its ancestors
};

/// Options of the random request generator.
struct FuzzOptions {
    std::size_t vocab = 64;
    std::size_t max_len = 300;
    std::size_t max_groups = 3;
    bool shuffle = true;     // parents may come after their children in the request
    bool allow_store = true; // rows may continue snapshots of `stored`
};

inline std::size_t random_length(Rng& rng, std::size_t max_len, bool branch) {
    static const std::vector<std::size_t> any = {1, 1, 2, 3, 5, 8, 13, 17, 31, 33, 64, 100, 129, 200, 300};
    static const std::vector<std::size_t> shortish = {1, 2, 3, 5, 8, 13, 17, 27, 31};
    const std::vector<std::size_t>& from = branch ? shortish : any;
    std::size_t len = 0;
    do {
        len = rng.pick(from);
    } while (len > max_len);
    return len;
}

/// Builds a random request: unrelated groups, each a single row, a state with branches, a chain, or rows that
/// continue snapshots stored by earlier calls; random readouts. Parents may follow their children in the list.
inline Request make_request(Rng& rng, const std::vector<Stored>& stored, const FuzzOptions& fo) {
    struct Node {
        std::vector<std::uint32_t> tokens;
        int parent = -1;              // index among nodes
        int stored_parent = -1;       // index into `stored`
        std::optional<std::uint64_t> start; // explicit start position
        bool keep = false;
    };
    std::vector<Node> nodes;
    const auto tokens_of_len = [&](std::size_t len) {
        std::vector<std::uint32_t> t(len);
        for (auto& v : t) v = static_cast<std::uint32_t>(rng.below(fo.vocab));
        return t;
    };
    const auto add = [&](std::size_t len, int parent, int stored_parent, bool branch) {
        Node node;
        node.tokens = tokens_of_len(random_length(rng, std::min(len == 0 ? fo.max_len : len, fo.max_len), branch));
        node.parent = parent;
        node.stored_parent = stored_parent;
        node.keep = rng.chance(30);
        nodes.push_back(std::move(node));
        return static_cast<int>(nodes.size() - 1);
    };
    const std::size_t groups = 1 + rng.below(fo.max_groups);
    for (std::size_t g = 0; g < groups; ++g) {
        const bool use_stored = fo.allow_store && !stored.empty() && rng.chance(35);
        const int stored_parent = use_stored ? static_cast<int>(rng.below(stored.size())) : -1;
        switch (rng.below(5)) {
        case 0: { // a single row
            add(0, -1, stored_parent, false);
            break;
        }
        case 1: { // a state and several branches
            const int root = add(0, -1, stored_parent, false);
            const std::size_t branches = 1 + rng.below(8);
            for (std::size_t b = 0; b < branches; ++b) add(0, root, -1, true);
            break;
        }
        case 2: { // a chain three deep, with a branch off the middle
            const int a = add(0, -1, stored_parent, false);
            const int b = add(0, a, -1, false);
            add(0, b, -1, false);
            if (rng.chance(50)) add(0, b, -1, true);
            break;
        }
        case 3: { // several rows continuing one stored snapshot (or nothing)
            const std::size_t branches = 1 + rng.below(4);
            for (std::size_t b = 0; b < branches; ++b) add(0, -1, stored_parent, true);
            break;
        }
        default: { // a state with a child and a grandchild, and a second child
            const int root = add(0, -1, stored_parent, false);
            const int child = add(0, root, -1, true);
            add(0, child, -1, true);
            add(0, root, -1, true);
            break;
        }
        }
    }

    // Order of the request: nodes in a random permutation when asked to.
    std::vector<std::size_t> order(nodes.size());
    for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
    if (fo.shuffle) {
        for (std::size_t i = order.size(); i > 1; --i) std::swap(order[i - 1], order[rng.below(i)]);
    }
    std::vector<std::size_t> position(nodes.size()); // node -> index in the request
    for (std::size_t i = 0; i < order.size(); ++i) position[order[i]] = i;

    // Start positions and chain lengths are resolved parents first.
    std::vector<std::uint64_t> end(nodes.size(), 0), total(nodes.size(), 0);
    std::vector<bool> done(nodes.size(), false);
    const auto resolve = [&](auto&& self, std::size_t i) -> void {
        if (done[i]) return;
        Node& node = nodes[i];
        std::uint64_t parent_end = 0, parent_total = 0;
        if (node.parent >= 0) {
            self(self, static_cast<std::size_t>(node.parent));
            parent_end = end[static_cast<std::size_t>(node.parent)];
            parent_total = total[static_cast<std::size_t>(node.parent)];
        } else if (node.stored_parent >= 0) {
            parent_end = stored[static_cast<std::size_t>(node.stored_parent)].end_position;
            parent_total = stored[static_cast<std::size_t>(node.stored_parent)].tokens;
        }
        // Mostly continue where the parent ended; sometimes start somewhere else (positions only enter through RoPE).
        node.start = rng.chance(70) ? std::optional<std::uint64_t>() : std::optional<std::uint64_t>(parent_end + rng.below(50));
        const std::uint64_t start = node.start.value_or(parent_end);
        end[i] = start + node.tokens.size();
        total[i] = parent_total + node.tokens.size();
        done[i] = true;
    };
    for (std::size_t i = 0; i < nodes.size(); ++i) resolve(resolve, i);

    Request request;
    request.engine.resize(nodes.size());
    request.reference.resize(nodes.size());
    request.reference64.resize(nodes.size());
    request.end_position.resize(nodes.size());
    request.tokens.resize(nodes.size());
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        const Node& node = nodes[i];
        const std::size_t at = position[i];
        cuda::EngineRow& e = request.engine[at];
        ref::Row<float>& r = request.reference[at];
        ref::Row<double>& r64 = request.reference64[at];
        e.token_ids = node.tokens;
        r.token_ids = node.tokens;
        r64.token_ids = node.tokens;
        e.start_position = node.start;
        std::uint64_t parent_end = 0;
        if (node.parent >= 0) {
            e.parent = cuda::ParentRow{position[static_cast<std::size_t>(node.parent)]};
            r.parent = ref::ParentRow{position[static_cast<std::size_t>(node.parent)]};
            r64.parent = ref::ParentRow{position[static_cast<std::size_t>(node.parent)]};
            parent_end = end[static_cast<std::size_t>(node.parent)];
        } else if (node.stored_parent >= 0) {
            const Stored& s = stored[static_cast<std::size_t>(node.stored_parent)];
            e.parent = s.device;
            r.parent = s.reference;
            r64.parent = s.reference64;
            parent_end = s.end_position;
        }
        r.start_position = node.start.value_or(parent_end);
        r64.start_position = r.start_position;
        const std::size_t readouts = rng.below(4);
        for (std::size_t k = 0; k < readouts; ++k) {
            const std::size_t offset = rng.below(node.tokens.size());
            e.readout_offsets.push_back(offset);
            r.readout_offsets.push_back(offset);
            r64.readout_offsets.push_back(offset);
        }
        e.keep_snapshot = node.keep;
        request.end_position[at] = end[i];
        request.tokens[at] = total[i];
    }
    return request;
}

/// The reference's answer for the rows of a request, with a ladder per row on request.
inline std::vector<ref::RowResult<float>> run_reference(const ModelConfig& config, const Weights<float>& weights,
                                                        const std::vector<ref::Row<float>>& rows) {
    static const std::unique_ptr<GemmProvider> gemm = make_reference_gemm();
    return ref::run_rows<float>(*gemm, config, weights, rows);
}

inline std::vector<ref::RowResult<double>> run_reference64(const ModelConfig& config, const Weights<double>& weights,
                                                           const std::vector<ref::Row<double>>& rows) {
    static const std::unique_ptr<GemmProvider> gemm = make_reference_gemm();
    return ref::run_rows<double>(*gemm, config, weights, rows);
}

/// The same weights in float64.
inline Weights<double> double_weights(const ModelConfig& cfg, const Weights<float>& weights) {
    Weights<double>::Map map;
    for (const auto& [name, shape] : weight_shapes(cfg)) {
        const BasicTensor<float>& t = weights[name];
        BasicTensor<double> d(shape);
        for (std::size_t i = 0; i < t.size(); ++i) d[i] = static_cast<double>(t[i]);
        map.emplace(name, std::move(d));
    }
    return Weights<double>::adopt(cfg, std::move(map));
}

/// The layer outputs of one row of a request given the reference results of the request (for locating a mismatch).
template <class T>
std::vector<BasicTensor<T>> reference_ladder(const ModelConfig& config, const Weights<T>& weights,
                                             const std::vector<ref::Row<T>>& rows,
                                             const std::vector<ref::RowResult<T>>& results, std::size_t row) {
    static const std::unique_ptr<GemmProvider> gemm = make_reference_gemm();
    const ref::Row<T>& r = rows[row];
    ref::SnapshotPtr<T> parent;
    if (std::holds_alternative<std::monostate>(r.parent)) {
        parent = ref::empty_snapshot<T>(config);
    } else if (const auto* s = std::get_if<ref::SnapshotPtr<T>>(&r.parent)) {
        parent = *s;
    } else {
        parent = results[std::get<ref::ParentRow>(r.parent).index].snapshot;
    }
    std::vector<BasicTensor<T>> ladder;
    ref::continue_snapshot<T>(*gemm, config, weights, *parent, r.token_ids, r.start_position, &ladder);
    return ladder;
}

/// Calibration statistics of an engine's float16 run over random tokens (EngineOptions::collect_activation_stats): what
/// Int8Smoothing needs for a geometry that has no real calibration set. `salt` changes the tokens.
inline std::shared_ptr<const sepal::cuda::ActivationStats> collect_stats(const ModelConfig& config, const Weights<float>& weights, std::uint64_t salt = 1,
                                                                         std::size_t rows = 6, std::size_t length = 90) {
    sepal::cuda::EngineOptions o;
    o.precision = sepal::cuda::Precision::F16;
    o.collect_activation_stats = true;
    o.memory_budget_bytes = std::size_t{192} << 20;
    o.max_call_tokens = 4096;
    o.max_call_rows = 64;
    o.state_slots = 64;
    sepal::cuda::CudaEngine engine(config, weights, o);
    Rng rng(salt * 7919 + 13);
    std::vector<sepal::cuda::EngineRow> request(rows);
    for (auto& r : request) {
        r.token_ids.resize(length);
        for (auto& t : r.token_ids) t = static_cast<std::uint32_t>(rng.below(static_cast<std::size_t>(config.vocab_size)));
        r.readout_offsets = {length - 1};
    }
    (void)engine.run(std::span<const sepal::cuda::EngineRow>(request));
    return std::make_shared<const sepal::cuda::ActivationStats>(engine.activation_stats());
}

} // namespace engine_test
