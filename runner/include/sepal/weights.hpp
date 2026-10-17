// Weights: the tensor names and shapes a geometry implies, and a read-only container keyed by tensor name.
// A port of sepal_runner/refengine/weights.py.
//
// Names follow the transformers Qwen3.5 text model (`layers.<i>.linear_attn.in_proj_qkv.weight` and so on).
// Projections are stored one tensor per projection, as checkpoints hold them. Three groups: the embedding table,
// the body (everything a pass multiplies by) and the pointer head, which belongs to an adapter. The vocabulary
// projection is never part of the weights: a prefill-only engine does not have one.
#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "sepal/config.hpp"
#include "sepal/tensor.hpp"

namespace sepal {

using Shape = std::vector<std::size_t>;
using NamedShapes = std::vector<std::pair<std::string, Shape>>;

inline constexpr std::string_view embedding_name = "embed_tokens.weight";
inline constexpr std::string_view final_norm_name = "norm.weight";
inline constexpr std::string_view head_prefix = "head";

NamedShapes embedding_shapes(const ModelConfig& cfg);
/// One Gated DeltaNet mixer. `conv1d.weight` keeps the checkpoint layout [channels, 1, taps].
NamedShapes deltanet_shapes(const ModelConfig& cfg);
/// One full-attention mixer. `q_proj` is twice the attention width: the output gate is packed with the query.
NamedShapes attention_shapes(const ModelConfig& cfg);
NamedShapes mlp_shapes(const ModelConfig& cfg);
/// Every tensor of the decoder stack and the final norm: what "non-embedding parameters" counts.
NamedShapes body_shapes(const ModelConfig& cfg);
/// The pointer head: two linear maps WITH bias from the hidden size to the pointer dim.
NamedShapes head_shapes(const ModelConfig& cfg);
/// Every tensor `Weights` holds for this geometry, in the order the reference engine lists them.
NamedShapes weight_shapes(const ModelConfig& cfg);

/// Sum of the element counts of `shapes`.
std::uint64_t count_elements(const NamedShapes& shapes);

/// Empty when `names_and_shapes` is exactly what the geometry implies; otherwise one sentence naming the first
/// missing, unexpected or mis-shaped tensors. Strict, because a missing or mis-shaped tensor must not load.
std::string weights_mismatch(const ModelConfig& cfg, const NamedShapes& names_and_shapes);

/// Read-only tensors keyed by name, in one element type. `scope` gives one module's tensors under their short
/// names. Copies share the tensors, which are never written after construction.
template <class T>
class Weights {
public:
    using Map = std::map<std::string, BasicTensor<T>, std::less<>>;

    Weights() : tensors_(std::make_shared<const Map>()) {}

    /// Takes the tensors as they are, after the strict check against the geometry. Throws std::invalid_argument.
    static Weights adopt(const ModelConfig& cfg, Map tensors) {
        NamedShapes found;
        found.reserve(tensors.size());
        for (const auto& [name, tensor] : tensors) found.emplace_back(name, tensor.shape());
        const std::string problem = weights_mismatch(cfg, found);
        if (!problem.empty()) throw std::invalid_argument(problem);
        Weights w;
        w.tensors_ = std::make_shared<const Map>(std::move(tensors));
        return w;
    }

    /// The loader interface: any mapping of name to array (a checkpoint, a test fixture) becomes the model's
    /// weights in element type T, each value converted once.
    template <class U>
    static Weights from_arrays(const ModelConfig& cfg, const std::map<std::string, BasicTensor<U>, std::less<>>& arrays) {
        Map tensors;
        for (const auto& [name, source] : arrays) {
            BasicTensor<T> converted(source.shape());
            for (std::size_t i = 0; i < source.size(); ++i) converted[i] = static_cast<T>(source[i]);
            tensors.emplace(name, std::move(converted));
        }
        return adopt(cfg, std::move(tensors));
    }

    /// The tensors under `prefix.`, addressed without it.
    Weights scope(std::string_view prefix) const {
        Weights w;
        w.tensors_ = tensors_;
        w.prefix_ = prefix_ + std::string(prefix) + ".";
        return w;
    }

    bool contains(std::string_view name) const { return tensors_->find(prefix_ + std::string(name)) != tensors_->end(); }

    const BasicTensor<T>& operator[](std::string_view name) const {
        const auto it = tensors_->find(prefix_ + std::string(name));
        if (it == tensors_->end()) throw std::out_of_range("weights: no tensor named " + prefix_ + std::string(name));
        return it->second;
    }

    /// Number of tensors in the whole container, whatever the scope.
    std::size_t size() const { return tensors_->size(); }

private:
    std::shared_ptr<const Map> tensors_;
    std::string prefix_;
};

/// Seeded pseudo-random values in [-sqrt(3), sqrt(3)) with unit variance: one fixed generator (splitmix64), so
/// the same seed gives the same values on every platform. NOT the reference engine's `random_weights` (that one
/// draws NumPy normals and reaches C++ through the golden file); this exists for benchmarks and for tests that
/// need some model of a geometry.
class UnitRandom {
public:
    explicit UnitRandom(std::uint64_t seed) : state_(seed) {}
    double next();

private:
    std::uint64_t state_;
};

/// The scale `sepal.refengine.weights._random_tensor` gives a tensor by its role, as (multiplier, offset):
/// value = offset + multiplier * unit draw. Projections and conv kernels are scaled by fan-in so activations
/// keep their size through the stack; norm weights are small; the gated norm's weight is near one.
std::pair<double, double> random_tensor_scale(std::string_view name, const Shape& shape);

/// Seeded random weights of a geometry, built directly in element type T (so a real geometry costs its float32
/// size and no more). This allocates every tensor.
template <class T>
Weights<T> random_weights(const ModelConfig& cfg, std::uint64_t seed) {
    UnitRandom rng(seed);
    typename Weights<T>::Map tensors;
    for (const auto& [name, shape] : weight_shapes(cfg)) {
        const auto [multiplier, offset] = random_tensor_scale(name, shape);
        BasicTensor<T> tensor(shape);
        for (T& value : tensor.span()) value = static_cast<T>(offset + multiplier * rng.next());
        tensors.emplace(name, std::move(tensor));
    }
    return Weights<T>::adopt(cfg, std::move(tensors));
}

} // namespace sepal
