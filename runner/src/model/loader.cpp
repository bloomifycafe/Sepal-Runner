#include "sepal/loader.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "sepal/gemm.hpp"
#include "sepal/json.hpp"
#include "sepal/ref/readout.hpp"

namespace sepal {

namespace {

namespace fs = std::filesystem;

// Where the text model's tensors can sit, first match wins: a multimodal checkpoint saved by transformers 5, an
// older multimodal one, a text-only model, a bare one.
constexpr std::string_view text_prefixes[] = {"model.language_model.", "language_model.", "model.", ""};

struct IgnoreRule {
    std::string_view prefix;
    std::string_view reason;
};

// What a text prefill engine does not load. Anything not listed here and not a text-model tensor is an error.
constexpr IgnoreRule ignore_rules[] = {
    {"model.visual.", "vision tower"},
    {"visual.", "vision tower"},
    {"lm_head.", "vocabulary projection (a prefill engine reads the label rows of the embedding instead)"},
    {"model.mtp.", "multi-token-prediction head"},
    {"mtp.", "multi-token-prediction head"},
};

std::string read_text_file(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot read '" + path.string() + "'");
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

fs::path to_path(const std::string& utf8) {
    return fs::path(std::u8string(reinterpret_cast<const char8_t*>(utf8.data()), utf8.size()));
}

std::string to_utf8(const fs::path& path) {
    const std::u8string text = path.u8string();
    return std::string(reinterpret_cast<const char*>(text.data()), text.size());
}

[[noreturn]] void bad_config(const std::string& what) { throw std::invalid_argument("config.json: " + what); }

std::uint64_t need_uint(const json::Value& object, const char* key) {
    const json::Value* found = object.find(key);
    if (found == nullptr) bad_config(std::string("\"") + key + "\" is missing");
    try {
        return found->as_uint();
    } catch (const std::runtime_error&) {
        bad_config(std::string("\"") + key + "\" is not a non-negative whole number");
    }
}

double need_number(const json::Value& object, const char* key) {
    const json::Value* found = object.find(key);
    if (found == nullptr) bad_config(std::string("\"") + key + "\" is missing");
    if (found->kind() != json::Kind::Number) bad_config(std::string("\"") + key + "\" is not a number");
    return found->as_number();
}

std::string number_text(double value) {
    std::ostringstream s;
    s.precision(17);
    s << value;
    return s.str();
}

} // namespace

std::vector<std::string> compare_with_preset(const ModelConfig& config, const ModelConfig& preset) {
    std::vector<std::string> out;
    const auto integer = [&out](const char* field, std::uint64_t got, std::uint64_t want) {
        if (got != want) out.push_back(std::string(field) + ": checkpoint " + std::to_string(got) + ", preset " + std::to_string(want));
    };
    const auto real = [&out](const char* field, double got, double want) {
        if (got != want) out.push_back(std::string(field) + ": checkpoint " + number_text(got) + ", preset " + number_text(want));
    };
    integer("num_layers", config.num_layers, preset.num_layers);
    integer("full_attention_interval", config.full_attention_interval, preset.full_attention_interval);
    integer("hidden_size", config.hidden_size, preset.hidden_size);
    integer("intermediate_size", config.intermediate_size, preset.intermediate_size);
    integer("attn_heads", config.attn_heads, preset.attn_heads);
    integer("attn_kv_heads", config.attn_kv_heads, preset.attn_kv_heads);
    integer("attn_head_dim", config.attn_head_dim, preset.attn_head_dim);
    integer("dn_key_heads", config.dn_key_heads, preset.dn_key_heads);
    integer("dn_value_heads", config.dn_value_heads, preset.dn_value_heads);
    integer("dn_key_head_dim", config.dn_key_head_dim, preset.dn_key_head_dim);
    integer("dn_value_head_dim", config.dn_value_head_dim, preset.dn_value_head_dim);
    integer("conv_kernel", config.conv_kernel, preset.conv_kernel);
    integer("vocab_size", config.vocab_size, preset.vocab_size);
    real("rms_norm_eps", config.rms_norm_eps, preset.rms_norm_eps);
    real("rope_theta", config.rope_theta, preset.rope_theta);
    real("partial_rotary_factor", config.partial_rotary_factor, preset.partial_rotary_factor);
    if (config.lm_head_tied != preset.lm_head_tied) {
        out.push_back(std::string("lm_head_tied: checkpoint ") + (config.lm_head_tied ? "true" : "false") + ", preset " +
                      (preset.lm_head_tied ? "true" : "false"));
    }
    return out;
}

ConfigReading read_model_config(std::string_view json_text, const std::string& name) {
    json::Value root;
    try {
        root = json::parse(json_text);
    } catch (const std::runtime_error& error) {
        bad_config(std::string("not valid JSON: ") + error.what());
    }
    if (root.kind() != json::Kind::Object) bad_config("the top level is not an object");
    const json::Value* nested = root.find("text_config");
    const json::Value& c = nested != nullptr ? *nested : root;
    if (c.kind() != json::Kind::Object) bad_config("\"text_config\" is not an object");

    if (const json::Value* type = c.find("model_type")) {
        const std::string& model_type = type->as_string();
        if (model_type != "qwen3_5_text" && model_type != "qwen3_5") {
            bad_config("model_type '" + model_type + "' is not a Qwen3.5 text model");
        }
    }
    // Things this engine's arithmetic is built on. A checkpoint that departs from one is refused, not misread.
    if (const json::Value* gate = c.find("attn_output_gate")) {
        if (!gate->as_bool()) bad_config("attn_output_gate is false, but the engine's q_proj carries the output gate");
    }
    if (const json::Value* bias = c.find("attention_bias")) {
        if (bias->as_bool()) bad_config("attention_bias is true, but the engine's projections have no bias");
    }
    if (const json::Value* act = c.find("hidden_act")) {
        if (act->as_string() != "silu") bad_config("hidden_act '" + act->as_string() + "' is not silu");
    }
    if (const json::Value* only = c.find("mlp_only_layers")) {
        if (!only->items().empty()) bad_config("mlp_only_layers is not empty: the engine has no layer without a token mixer");
    }

    ModelConfig cfg;
    cfg.name = name;
    cfg.num_layers = need_uint(c, "num_hidden_layers");
    cfg.hidden_size = need_uint(c, "hidden_size");
    cfg.intermediate_size = need_uint(c, "intermediate_size");
    cfg.attn_heads = need_uint(c, "num_attention_heads");
    cfg.attn_kv_heads = need_uint(c, "num_key_value_heads");
    cfg.attn_head_dim = need_uint(c, "head_dim");
    cfg.dn_key_heads = need_uint(c, "linear_num_key_heads");
    cfg.dn_value_heads = need_uint(c, "linear_num_value_heads");
    cfg.dn_key_head_dim = need_uint(c, "linear_key_head_dim");
    cfg.dn_value_head_dim = need_uint(c, "linear_value_head_dim");
    cfg.conv_kernel = need_uint(c, "linear_conv_kernel_dim");
    cfg.vocab_size = need_uint(c, "vocab_size");
    cfg.rms_norm_eps = need_number(c, "rms_norm_eps");
    cfg.pointer_dim = default_pointer_dim;

    // RoPE: transformers 5 keeps theta and the rotary fraction under "rope_parameters"; older files at the top.
    const json::Value* rope = c.find("rope_parameters");
    if (rope != nullptr) {
        if (const json::Value* kind = rope->find("rope_type")) {
            if (kind->as_string() != "default") bad_config("rope_type '" + kind->as_string() + "' is not default");
        }
    }
    const auto rope_number = [&](const char* key) {
        if (rope != nullptr && rope->find(key) != nullptr) return need_number(*rope, key);
        return need_number(c, key);
    };
    cfg.rope_theta = rope_number("rope_theta");
    cfg.partial_rotary_factor = rope_number("partial_rotary_factor");

    // Tied embeddings are stated in the text config of a multimodal file and sometimes only at the top level.
    const json::Value* tied = c.find("tie_word_embeddings");
    if (tied == nullptr) tied = root.find("tie_word_embeddings");
    cfg.lm_head_tied = tied != nullptr ? tied->as_bool() : true;

    // The full-attention interval: stated, derived from the layer types, or both and then they must agree.
    std::uint64_t interval = 0;
    if (const json::Value* types = c.find("layer_types")) {
        const std::vector<json::Value>& kinds = types->items();
        if (kinds.size() != cfg.num_layers) bad_config("layer_types lists " + std::to_string(kinds.size()) + " layers, num_hidden_layers is " + std::to_string(cfg.num_layers));
        for (std::size_t i = 0; i < kinds.size(); ++i) {
            const std::string& kind = kinds[i].as_string();
            if (kind != "full_attention" && kind != "linear_attention") bad_config("layer_types[" + std::to_string(i) + "] is '" + kind + "'");
            if (interval == 0 && kind == "full_attention") interval = i + 1;
        }
        if (interval == 0) bad_config("layer_types has no full-attention layer");
    }
    if (const json::Value* stated = c.find("full_attention_interval")) {
        const std::uint64_t value = stated->as_uint();
        if (interval != 0 && interval != value) bad_config("full_attention_interval " + std::to_string(value) + " disagrees with layer_types (" + std::to_string(interval) + ")");
        interval = value;
    }
    if (interval == 0) bad_config("neither layer_types nor full_attention_interval is given");
    cfg.full_attention_interval = interval;
    if (const json::Value* types = c.find("layer_types")) {
        for (std::size_t i = 0; i < types->items().size(); ++i) {
            const bool full = types->items()[i].as_string() == "full_attention";
            if (full != cfg.is_full_attention(i)) bad_config("layer_types is not a fixed full-attention interval: layer " + std::to_string(i) + " breaks the pattern of every " + std::to_string(interval));
        }
    }

    require_valid(cfg);

    // The closest preset: the one that differs in the fewest fields.
    ConfigReading reading;
    reading.config = cfg;
    std::size_t fewest = std::numeric_limits<std::size_t>::max();
    for (const ModelConfig& preset : presets()) {
        std::vector<std::string> differences = compare_with_preset(cfg, preset);
        if (differences.size() < fewest) {
            fewest = differences.size();
            reading.preset = preset.name;
            reading.differences = std::move(differences);
        }
    }
    return reading;
}

std::size_t RawTensor::elements() const {
    std::size_t n = 1;
    for (const std::size_t d : shape) n *= d;
    return n;
}

namespace {

// The prefix in front of "embed_tokens.weight", or nullopt when no known prefix has one.
std::optional<std::string> find_text_prefix(const Checkpoint& checkpoint) {
    for (const std::string_view prefix : text_prefixes) {
        if (checkpoint.find(std::string(prefix) + std::string(embedding_name)) != nullptr) return std::string(prefix);
    }
    return std::nullopt;
}

std::string shape_text(const std::vector<std::size_t>& shape) {
    std::string text = "[";
    for (std::size_t i = 0; i < shape.size(); ++i) text += (i == 0 ? "" : ", ") + std::to_string(shape[i]);
    return text + "]";
}

} // namespace

LoadedModel LoadedModel::open(const std::string& directory) {
    const fs::path dir = to_path(directory);
    LoadedModel model;
    const fs::path config_path = dir / "config.json";
    std::error_code ec;
    if (!fs::is_regular_file(config_path, ec)) throw std::runtime_error("no config.json in '" + directory + "'");
    std::string leaf = to_utf8(dir.filename());
    if (leaf.empty()) leaf = to_utf8(dir.parent_path().filename());
    model.reading_ = read_model_config(read_text_file(config_path), leaf);
    const ModelConfig& cfg = model.reading_.config;

    model.checkpoint_ = std::make_shared<const Checkpoint>(Checkpoint::open_directory(directory));
    const Checkpoint& checkpoint = *model.checkpoint_;

    const std::optional<std::string> prefix = find_text_prefix(checkpoint);
    if (!prefix) {
        throw std::runtime_error("the checkpoint has no embed_tokens.weight under any of the prefixes 'model.language_model.', 'language_model.', 'model.' or none");
    }
    model.prefix_ = *prefix;

    // Sort every checkpoint tensor into: a text-model tensor the geometry expects, an ignorable one, or an error.
    std::set<std::string> expected;
    for (const auto& [name, shape] : embedding_shapes(cfg)) expected.insert(name);
    for (const auto& [name, shape] : body_shapes(cfg)) expected.insert(name);
    std::vector<std::string> unexpected;
    for (const std::string& name : checkpoint.names()) {
        const TensorView& tensor = checkpoint.at(name);
        const IgnoreRule* rule = nullptr;
        for (const IgnoreRule& candidate : ignore_rules) {
            if (name.starts_with(candidate.prefix)) {
                rule = &candidate;
                break;
            }
        }
        if (rule != nullptr) {
            model.ignored_.push_back({name, std::string(rule->reason), tensor.bytes});
        } else if (name.starts_with(model.prefix_) && expected.contains(name.substr(model.prefix_.size()))) {
            const std::string engine_name = name.substr(model.prefix_.size());
            const std::optional<DType> dtype = as_dtype(tensor.type);
            if (!dtype || *dtype == DType::I8) {
                throw std::runtime_error("tensor '" + name + "' is stored as " + std::string(name_of(tensor.type)) + "; the engine loads F32, F16 and BF16 weights");
            }
            model.raw_.emplace(engine_name, RawTensor{*dtype, tensor.shape, tensor.data, tensor.bytes, &tensor});
        } else {
            unexpected.push_back(name);
        }
    }
    if (!unexpected.empty()) {
        std::string text = "the checkpoint holds " + std::to_string(unexpected.size()) + " tensor(s) the engine neither loads nor knows to ignore: ";
        for (std::size_t i = 0; i < unexpected.size() && i < 5; ++i) text += (i == 0 ? "" : ", ") + unexpected[i];
        throw std::runtime_error(text);
    }

    // The strict shape check, on the raw tensors plus the pointer head the engine's container also requires.
    NamedShapes found;
    for (const auto& [name, tensor] : model.raw_) found.emplace_back(name, tensor.shape);
    for (const auto& [name, shape] : head_shapes(cfg)) found.emplace_back(name, shape);
    const std::string problem = weights_mismatch(cfg, found);
    if (!problem.empty()) throw std::runtime_error("checkpoint does not fit its config.json: " + problem);
    return model;
}

const RawTensor& LoadedModel::raw_tensor(std::string_view engine_name) const {
    const auto it = raw_.find(engine_name);
    if (it == raw_.end()) throw std::out_of_range("no loaded tensor named " + std::string(engine_name));
    return it->second;
}

std::uint64_t LoadedModel::raw_bytes() const {
    std::uint64_t total = 0;
    for (const auto& [name, tensor] : raw_) total += tensor.bytes;
    return total;
}

template <class T>
Weights<T> LoadedModel::weights() const {
    typename Weights<T>::Map tensors;
    for (const auto& [name, raw] : raw_) {
        BasicTensor<T> converted(raw.shape);
        convert_floating(*raw.source, converted.data());
        tensors.emplace(name, std::move(converted));
    }
    for (const auto& [name, shape] : head_shapes(reading_.config)) tensors.emplace(name, BasicTensor<T>(shape));
    return Weights<T>::adopt(reading_.config, std::move(tensors));
}

template Weights<float> LoadedModel::weights<float>() const;
template Weights<double> LoadedModel::weights<double>() const;

DecisionReadout DecisionReadout::open(const std::string& path, double temperature) {
    if (!(temperature > 0.0)) throw std::invalid_argument("the readout temperature must be positive");
    DecisionReadout readout;
    readout.file_ = std::make_shared<const SafetensorsFile>(SafetensorsFile::open(path));
    const TensorView* weight = readout.file_->find("weight");
    if (weight == nullptr) throw std::runtime_error("'" + path + "' has no tensor named weight");
    const std::optional<DType> dtype = as_dtype(weight->type);
    if (!dtype || *dtype == DType::I8) throw std::runtime_error("the readout weight is stored as " + std::string(name_of(weight->type)) + ", not F32, F16 or BF16");
    if (weight->shape.size() != 2) throw std::runtime_error("the readout weight is " + shape_text(weight->shape) + ", not a matrix");
    readout.weight_ = RawTensor{*dtype, weight->shape, weight->data, weight->bytes, weight};
    readout.temperature_ = temperature;
    return readout;
}

DecisionReadout DecisionReadout::open_directory(const std::string& directory) {
    const fs::path dir = to_path(directory);
    const fs::path config_path = dir / "decision_config.json";
    std::error_code ec;
    if (!fs::is_regular_file(config_path, ec)) throw std::runtime_error("no decision_config.json in '" + directory + "'");
    const json::Value config = json::parse(read_text_file(config_path));
    return open(to_utf8(dir / "readout.safetensors"), need_number(config, "temperature"));
}

std::vector<double> DecisionReadout::logits(std::span<const double> hidden, std::size_t options) const {
    if (options == 0 || options > rows()) {
        throw std::invalid_argument("the readout has " + std::to_string(rows()) + " rows, so " + std::to_string(options) + " options do not fit");
    }
    if (hidden.size() != hidden_size()) {
        throw std::invalid_argument("the hidden state has " + std::to_string(hidden.size()) + " values, the readout rows have " + std::to_string(hidden_size()));
    }
    // The first `options` rows are contiguous at the start of the matrix, so converting them is one pass.
    BasicTensor<double> rows_used({options, hidden_size()});
    TensorView head = *weight_.source;
    head.shape = {options, hidden_size()};
    head.bytes = options * hidden_size() * bytes_of(head.type);
    convert_floating(head, rows_used.data());
    const std::unique_ptr<GemmProvider> provider = make_reference_gemm();
    return ref::label_row_logits<double>(*provider, hidden, rows_used, temperature_);
}

std::vector<double> DecisionReadout::probabilities(std::span<const double> hidden, std::size_t options) const {
    std::vector<double> p = logits(hidden, options);
    ref::softmax_inplace(std::span<double>(p));
    return p;
}

std::vector<double> DecisionReadout::probabilities(std::span<const float> hidden, std::size_t options) const {
    const std::vector<double> widened(hidden.begin(), hidden.end());
    return probabilities(std::span<const double>(widened), options);
}

} // namespace sepal
