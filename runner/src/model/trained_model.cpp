#include "sepal/trained_model.hpp"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <stdexcept>

#include "sepal/json.hpp"
#include "sepal/parallel.hpp"

namespace sepal {

namespace {

namespace fs = std::filesystem;

fs::path to_path(const std::string& utf8) {
    return fs::path(std::u8string(reinterpret_cast<const char8_t*>(utf8.data()), utf8.size()));
}

std::string to_utf8(const fs::path& path) {
    const std::u8string text = path.u8string();
    return std::string(reinterpret_cast<const char*>(text.data()), text.size());
}

std::string read_text(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot read '" + to_utf8(path) + "'");
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

[[noreturn]] void refuse(const std::string& what) { throw std::runtime_error("adapter_config.json: " + what); }

bool is_empty_or_null(const json::Value* v) {
    if (v == nullptr || v->is_null()) return true;
    if (v->kind() == json::Kind::Object || v->kind() == json::Kind::Array) return v->items().empty();
    return false;
}

void require_false(const json::Value& c, const char* key) {
    const json::Value* v = c.find(key);
    if (v == nullptr) return; // absent means the PEFT default, which is false for all of these
    if (v->kind() != json::Kind::Bool || v->as_bool()) refuse(std::string(key) + " is set; only plain LoRA is supported");
}

void require_null(const json::Value& c, const char* key) {
    if (!is_empty_or_null(c.find(key))) refuse(std::string(key) + " is set; this engine does not support it");
}

LoraInfo validate_adapter_config(const std::string& text) {
    json::Value c;
    try {
        c = json::parse(text);
    } catch (const std::runtime_error& e) {
        refuse(std::string("not valid JSON: ") + e.what());
    }
    if (c.kind() != json::Kind::Object) refuse("not an object");
    if (const json::Value* t = c.find("peft_type")) {
        if (t->kind() != json::Kind::String || t->as_string() != "LORA") refuse("peft_type is not LORA");
    }
    const json::Value* r = c.find("r");
    const json::Value* alpha = c.find("lora_alpha");
    if (r == nullptr || alpha == nullptr) refuse("r or lora_alpha is missing");
    LoraInfo info;
    try {
        info.rank = static_cast<std::uint32_t>(r->as_uint());
        info.alpha = alpha->as_number();
    } catch (const std::runtime_error&) {
        refuse("r or lora_alpha is not a number");
    }
    if (info.rank == 0 || !(info.alpha > 0)) refuse("r and lora_alpha must be positive");
    require_false(c, "use_rslora");
    require_false(c, "use_dora");
    require_false(c, "lora_bias");
    require_false(c, "fan_in_fan_out");
    const json::Value* bias = c.find("bias");
    if (bias != nullptr && (bias->kind() != json::Kind::String || bias->as_string() != "none")) refuse("bias is not \"none\"");
    require_null(c, "rank_pattern");
    require_null(c, "alpha_pattern");
    require_null(c, "modules_to_save");
    require_null(c, "layer_replication");
    require_null(c, "trainable_token_indices");
    require_null(c, "target_parameters");
    require_null(c, "use_bdlora");
    return info;
}

struct LoraPair {
    const TensorView* a = nullptr;
    const TensorView* b = nullptr;
};

[[noreturn]] void unsupported(const std::string& tensor) {
    throw std::runtime_error("adapter tensor '" + tensor + "' is not a plain LoRA A/B weight (embedding, DoRA magnitude, bias and "
                             "modules_to_save entries are not supported)");
}

double seconds_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}

} // namespace

struct TrainedModel::Impl {
    LoadedModel base;
    LoraInfo lora;
    ClefHeadConfig head_config;
    ClefHeadWeights head;
    std::shared_ptr<const SafetensorsFile> adapter;
    std::map<std::string, LoraPair, std::less<>> pairs; // engine tensor name ("layers.0.mlp.down_proj.weight") -> A, B
    std::string sepal_json;
    double load_seconds = 0;
};

TrainedModel TrainedModel::open(const std::string& final_dir, const std::string& base_dir) {
    const auto start = std::chrono::steady_clock::now();
    const fs::path dir = to_path(final_dir);
    auto impl = std::make_shared<Impl>(Impl{LoadedModel::open(base_dir), {}, {}, {}, nullptr, {}, {}, 0});
    const ModelConfig& cfg = impl->base.config();

    impl->lora = validate_adapter_config(read_text(dir / "adapter_config.json"));
    impl->head_config = ClefHeadConfig::parse(read_text(dir / "joint_head_config.json"));
    if (impl->head_config.hidden_size != cfg.hidden_size) {
        throw std::runtime_error("the head reads hidden size " + std::to_string(impl->head_config.hidden_size) + ", the base has " +
                                 std::to_string(cfg.hidden_size));
    }
    impl->head = ClefHeadWeights::open(to_utf8(dir / "joint_head.safetensors"), impl->head_config);
    {
        fs::path sepal = dir / "sepal.json";
        std::error_code ec;
        if (fs::is_regular_file(sepal, ec)) impl->sepal_json = read_text(sepal);
    }

    // The adapter: map every key to an engine tensor and check every shape against its base module.
    impl->adapter = std::make_shared<const SafetensorsFile>(SafetensorsFile::open(to_utf8(dir / "adapter_model.safetensors")));
    const std::string peft_prefix = "base_model.model.";
    const std::string text_prefix = impl->base.prefix();
    for (const TensorView& t : impl->adapter->tensors()) {
        std::string name = t.name;
        bool is_a = false;
        if (name.ends_with(".lora_A.weight")) {
            is_a = true;
            name.resize(name.size() - std::string(".lora_A.weight").size());
        } else if (name.ends_with(".lora_B.weight")) {
            name.resize(name.size() - std::string(".lora_B.weight").size());
        } else {
            unsupported(t.name);
        }
        if (!name.starts_with(peft_prefix)) unsupported(t.name);
        name = name.substr(peft_prefix.size());
        if (!name.starts_with(text_prefix)) {
            throw std::runtime_error("adapter tensor '" + t.name + "' does not sit under the base's text-model prefix '" + text_prefix + "'");
        }
        const std::string engine = name.substr(text_prefix.size()) + ".weight";
        if (impl->base.raw().find(engine) == impl->base.raw().end()) {
            throw std::runtime_error("adapter tensor '" + t.name + "' targets '" + engine + "', which the base model does not have");
        }
        if (t.type != StType::F32 && t.type != StType::BF16 && t.type != StType::F16) {
            throw std::runtime_error("adapter tensor '" + t.name + "' is stored as " + std::string(name_of(t.type)));
        }
        LoraPair& pair = impl->pairs[engine];
        (is_a ? pair.a : pair.b) = &t;
    }
    for (const auto& [engine, pair] : impl->pairs) {
        if (pair.a == nullptr || pair.b == nullptr) {
            throw std::runtime_error("adapter has only one of lora_A / lora_B for '" + engine + "'");
        }
        const RawTensor& w = impl->base.raw_tensor(engine);
        if (w.shape.size() != 2) throw std::runtime_error("adapter targets '" + engine + "', which is not a matrix");
        const std::size_t out = w.shape[0], in = w.shape[1], r = impl->lora.rank;
        if (pair.a->shape != std::vector<std::size_t>{r, in}) {
            throw std::runtime_error("lora_A of '" + engine + "' has the wrong shape (want [" + std::to_string(r) + ", " + std::to_string(in) +
                                     "] from r and the base module)");
        }
        if (pair.b->shape != std::vector<std::size_t>{out, r}) {
            throw std::runtime_error("lora_B of '" + engine + "' has the wrong shape (want [" + std::to_string(out) + ", " + std::to_string(r) +
                                     "] from r and the base module)");
        }
    }
    impl->lora.modules = impl->pairs.size();
    impl->load_seconds = seconds_since(start);
    TrainedModel model;
    model.impl_ = std::move(impl);
    return model;
}

const ModelConfig& TrainedModel::config() const { return impl_->base.config(); }
const LoadedModel& TrainedModel::base() const { return impl_->base; }
const LoraInfo& TrainedModel::lora() const { return impl_->lora; }
const ClefHeadConfig& TrainedModel::head_config() const { return impl_->head_config; }
const ClefHeadWeights& TrainedModel::head() const { return impl_->head; }
const std::string& TrainedModel::sepal_json() const { return impl_->sepal_json; }
double TrainedModel::load_seconds() const { return impl_->load_seconds; }

void TrainedModel::embedding_row(std::uint32_t token, float* out) const {
    const RawTensor& e = impl_->base.raw_tensor(embedding_name);
    const std::size_t hidden = e.shape.at(1);
    if (token >= e.shape[0]) throw std::out_of_range("embedding_row: token " + std::to_string(token) + " is outside the vocabulary");
    TensorView row = *e.source;
    row.shape = {1, hidden};
    row.bytes = hidden * bytes_of(row.type);
    row.data = e.source->data + static_cast<std::size_t>(token) * row.bytes;
    convert_floating(row, out);
}

template <class T>
Weights<T> TrainedModel::merged_weights() const {
    const Impl& m = *impl_;
    typename Weights<T>::Map tensors;
    for (const auto& [name, raw] : m.base.raw()) {
        BasicTensor<T> w(raw.shape);
        convert_floating(*raw.source, w.data());
        const auto it = m.pairs.find(name);
        if (it != m.pairs.end()) {
            const std::size_t out = raw.shape[0], in = raw.shape[1], r = m.lora.rank;
            std::vector<double> a(r * in);
            {
                std::vector<float> tmp(r * in);
                convert_floating(*it->second.a, tmp.data());
                for (std::size_t i = 0; i < tmp.size(); ++i) a[i] = static_cast<double>(tmp[i]);
            }
            std::vector<float> b(out * r);
            convert_floating(*it->second.b, b.data());
            const T scale = static_cast<T>(m.lora.scale());
            T* weights = w.data();
            // W' = W + (alpha / r) * (B @ A); the product of each element is accumulated in double, rows in parallel.
            parallel_for(out, [&](std::size_t o) {
                std::vector<double> acc(in, 0.0);
                for (std::size_t k = 0; k < r; ++k) {
                    const double bk = static_cast<double>(b[o * r + k]);
                    const double* ak = a.data() + k * in;
                    for (std::size_t i = 0; i < in; ++i) acc[i] += bk * ak[i];
                }
                T* row = weights + o * in;
                for (std::size_t i = 0; i < in; ++i) row[i] = row[i] + scale * static_cast<T>(acc[i]);
            }, 16);
        }
        tensors.emplace(name, std::move(w));
    }
    for (const auto& [name, shape] : head_shapes(m.base.config())) tensors.emplace(name, BasicTensor<T>(shape));
    return Weights<T>::adopt(m.base.config(), std::move(tensors));
}

template Weights<float> TrainedModel::merged_weights<float>() const;
template Weights<double> TrainedModel::merged_weights<double>() const;

} // namespace sepal
