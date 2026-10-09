// Loading a real checkpoint: its config.json as a ModelConfig, its tensors under the engine's names, and the
// decision readout that comes with a fine-tuned decision model.
// A port of sepal/refengine/upstream.py (`model_config`, `text_model_arrays`, `engine_weights`).
//
// A checkpoint directory holds `config.json` and safetensors files. The text model's tensors sit under a prefix
// that depends on how the checkpoint was saved (`model.language_model.`, `language_model.`, `model.` or none);
// the prefix is found and removed and nothing else is renamed. What a text prefill engine never loads (the
// vision tower, the vocabulary projection, the multi-token-prediction head) is listed and left out. Any other
// tensor is an error: a checkpoint is either understood or refused.
//
// Two views of the same files:
//   * `raw()`: every loaded tensor by engine name, pointing into the mapped files in the stored element type,
//     with nothing copied. A GPU engine uploads bf16 or f16 weights from here without widening them on the host.
//   * `weights<T>()`: the reference path's `Weights<float>` or `Weights<double>`, each tensor converted once.
#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "sepal/config.hpp"
#include "sepal/dtype.hpp"
#include "sepal/safetensors.hpp"
#include "sepal/weights.hpp"

namespace sepal {

/// The pointer-head width given to a loaded geometry. The head belongs to an adapter and is not in the
/// checkpoint; the loader fills it with zeros.
inline constexpr std::uint64_t default_pointer_dim = 256;

/// A geometry read from a config.json, and how it differs from the preset it is closest to.
struct ConfigReading {
    ModelConfig config;
    std::string preset;                    // the closest preset's name
    std::vector<std::string> differences;  // "field: checkpoint X, preset Y", empty when they agree
};

/// Reads the text model's geometry from the text of a config.json (the multimodal layout with a `text_config`
/// member, or a text-only one). Throws std::invalid_argument naming the field when a value is missing, invalid
/// or something this engine does not compute (no output gate, attention bias, an activation other than SiLU,
/// layers that are not a fixed full-attention interval, ...). `name` becomes the geometry's name.
ConfigReading read_model_config(std::string_view json_text, const std::string& name);

/// One line per field in which `config` differs from `preset` (the name and the pointer head are not compared).
std::vector<std::string> compare_with_preset(const ModelConfig& config, const ModelConfig& preset);

/// A stored tensor under its engine name: where the bytes are and what they mean. The bytes belong to the files
/// the LoadedModel keeps mapped.
struct RawTensor {
    DType dtype = DType::F32;
    std::vector<std::size_t> shape;
    const unsigned char* data = nullptr;
    std::size_t bytes = 0;
    const TensorView* source = nullptr; // the checkpoint tensor, for conversion
    std::size_t elements() const;
};

/// A checkpoint tensor the engine does not load, and why.
struct IgnoredTensor {
    std::string name;
    std::string reason;
    std::uint64_t bytes = 0;
};

class LoadedModel {
public:
    /// Opens `directory`, reads and validates its config.json, finds the text model's tensors and checks them
    /// against the geometry (every expected tensor present, no tensor with another shape, nothing unexpected).
    /// Throws std::runtime_error, std::invalid_argument or SafetensorsError with a message naming the cause.
    static LoadedModel open(const std::string& directory);

    const ModelConfig& config() const { return reading_.config; }
    const ConfigReading& config_reading() const { return reading_; }
    /// The text model's prefix in the checkpoint's tensor names, e.g. "language_model."; may be empty.
    const std::string& prefix() const { return prefix_; }
    const Checkpoint& checkpoint() const { return *checkpoint_; }

    /// Every loaded tensor by engine name, in stored precision, without a copy. The pointer head is not here:
    /// the checkpoint has none.
    const std::map<std::string, RawTensor, std::less<>>& raw() const { return raw_; }
    const RawTensor& raw_tensor(std::string_view engine_name) const;

    const std::vector<IgnoredTensor>& ignored() const { return ignored_; }

    /// Bytes of the loaded tensors as stored.
    std::uint64_t raw_bytes() const;

    /// The reference path's weights: each tensor converted to T once, and the pointer head (which a checkpoint of
    /// the base model does not have) all zeros. Passes the strict shape check of `Weights::adopt`.
    /// Instantiated for float and double.
    template <class T>
    Weights<T> weights() const;

private:
    std::shared_ptr<const Checkpoint> checkpoint_;
    ConfigReading reading_;
    std::string prefix_;
    std::map<std::string, RawTensor, std::less<>> raw_;
    std::vector<IgnoredTensor> ignored_;
};

/// The decision readout of a fine-tuned decision model: rows of a matrix [rows, hidden], one per answer code,
/// and a temperature. The probability of option k among N is the softmax over the first N rows of
/// `hidden . row_k / temperature`, in float64.
class DecisionReadout {
public:
    /// `<directory>/readout.safetensors` (tensor "weight") with the temperature from `decision_config.json`.
    static DecisionReadout open_directory(const std::string& directory);
    /// A readout file (tensor "weight") and a given temperature.
    static DecisionReadout open(const std::string& path, double temperature);

    double temperature() const { return temperature_; }
    std::size_t rows() const { return weight_.shape.at(0); }
    std::size_t hidden_size() const { return weight_.shape.at(1); }
    const RawTensor& weight() const { return weight_; }

    /// Logits of the first `options` rows for a hidden state of `hidden_size()` values.
    std::vector<double> logits(std::span<const double> hidden, std::size_t options) const;
    /// Probabilities over the first `options` rows. Throws std::invalid_argument when `options` is 0 or above
    /// `rows()`, or the hidden state has another width.
    std::vector<double> probabilities(std::span<const double> hidden, std::size_t options) const;
    std::vector<double> probabilities(std::span<const float> hidden, std::size_t options) const;

private:
    std::shared_ptr<const SafetensorsFile> file_;
    RawTensor weight_;
    double temperature_ = 1.0;
};

} // namespace sepal
