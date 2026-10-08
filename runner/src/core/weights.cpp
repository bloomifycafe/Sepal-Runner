#include "sepal/weights.hpp"

#include <algorithm>
#include <cmath>

namespace sepal {

namespace {

std::size_t dim(std::uint64_t value) { return static_cast<std::size_t>(value); }

bool ends_with(std::string_view text, std::string_view suffix) {
    return text.size() >= suffix.size() && text.substr(text.size() - suffix.size()) == suffix;
}

void append_scoped(NamedShapes& out, const std::string& prefix, const NamedShapes& shapes) {
    for (const auto& [name, shape] : shapes) out.emplace_back(prefix + name, shape);
}

std::string shape_text(const Shape& shape) {
    std::string text = "[";
    for (std::size_t i = 0; i < shape.size(); ++i) text += (i == 0 ? "" : ", ") + std::to_string(shape[i]);
    return text + "]";
}

} // namespace

NamedShapes embedding_shapes(const ModelConfig& cfg) {
    return {{std::string(embedding_name), {dim(cfg.vocab_size), dim(cfg.hidden_size)}}};
}

NamedShapes deltanet_shapes(const ModelConfig& cfg) {
    const std::size_t d = dim(cfg.hidden_size);
    const std::size_t hv = dim(cfg.dn_value_heads);
    return {
        {"in_proj_qkv.weight", {dim(cfg.dn_conv_dim()), d}},
        {"in_proj_z.weight", {dim(cfg.dn_value_dim()), d}},
        {"in_proj_b.weight", {hv, d}},
        {"in_proj_a.weight", {hv, d}},
        {"conv1d.weight", {dim(cfg.dn_conv_dim()), 1, dim(cfg.conv_kernel)}},
        {"A_log", {hv}},
        {"dt_bias", {hv}},
        {"norm.weight", {dim(cfg.dn_value_head_dim)}},
        {"out_proj.weight", {d, dim(cfg.dn_value_dim())}},
    };
}

NamedShapes attention_shapes(const ModelConfig& cfg) {
    const std::size_t d = dim(cfg.hidden_size);
    return {
        {"q_proj.weight", {2 * dim(cfg.attn_width()), d}},
        {"k_proj.weight", {dim(cfg.attn_kv_width()), d}},
        {"v_proj.weight", {dim(cfg.attn_kv_width()), d}},
        {"q_norm.weight", {dim(cfg.attn_head_dim)}},
        {"k_norm.weight", {dim(cfg.attn_head_dim)}},
        {"o_proj.weight", {d, dim(cfg.attn_width())}},
    };
}

NamedShapes mlp_shapes(const ModelConfig& cfg) {
    const std::size_t d = dim(cfg.hidden_size);
    const std::size_t m = dim(cfg.intermediate_size);
    return {{"gate_proj.weight", {m, d}}, {"up_proj.weight", {m, d}}, {"down_proj.weight", {d, m}}};
}

NamedShapes body_shapes(const ModelConfig& cfg) {
    NamedShapes shapes;
    for (std::uint64_t i = 0; i < cfg.num_layers; ++i) {
        const std::string layer = "layers." + std::to_string(i) + ".";
        if (cfg.is_full_attention(i)) {
            append_scoped(shapes, layer + "self_attn.", attention_shapes(cfg));
        } else {
            append_scoped(shapes, layer + "linear_attn.", deltanet_shapes(cfg));
        }
        append_scoped(shapes, layer + "mlp.", mlp_shapes(cfg));
        shapes.emplace_back(layer + "input_layernorm.weight", Shape{dim(cfg.hidden_size)});
        shapes.emplace_back(layer + "post_attention_layernorm.weight", Shape{dim(cfg.hidden_size)});
    }
    shapes.emplace_back(std::string(final_norm_name), Shape{dim(cfg.hidden_size)});
    return shapes;
}

NamedShapes head_shapes(const ModelConfig& cfg) {
    const std::size_t d = dim(cfg.hidden_size);
    const std::size_t p = dim(cfg.pointer_dim);
    const std::string prefix = std::string(head_prefix) + ".";
    return {{prefix + "q.weight", {p, d}}, {prefix + "q.bias", {p}}, {prefix + "k.weight", {p, d}}, {prefix + "k.bias", {p}}};
}

NamedShapes weight_shapes(const ModelConfig& cfg) {
    NamedShapes shapes = embedding_shapes(cfg);
    append_scoped(shapes, "", body_shapes(cfg));
    append_scoped(shapes, "", head_shapes(cfg));
    return shapes;
}

std::uint64_t count_elements(const NamedShapes& shapes) {
    std::uint64_t total = 0;
    for (const auto& [name, shape] : shapes) {
        std::uint64_t n = 1;
        for (const std::size_t d : shape) n *= d;
        total += n;
    }
    return total;
}

std::string weights_mismatch(const ModelConfig& cfg, const NamedShapes& names_and_shapes) {
    std::map<std::string, Shape> expected;
    for (auto& [name, shape] : weight_shapes(cfg)) expected.emplace(name, shape);
    std::map<std::string, Shape> found;
    for (const auto& [name, shape] : names_and_shapes) found.emplace(name, shape);

    std::vector<std::string> missing;
    std::vector<std::string> unexpected;
    std::vector<std::string> wrong;
    for (const auto& [name, shape] : expected) {
        const auto it = found.find(name);
        if (it == found.end()) {
            missing.push_back(name);
        } else if (it->second != shape) {
            wrong.push_back(name + ": " + shape_text(it->second) + ", expected " + shape_text(shape));
        }
    }
    for (const auto& [name, shape] : found) {
        if (expected.find(name) == expected.end()) unexpected.push_back(name);
    }

    if (!missing.empty() || !unexpected.empty()) {
        std::string text = "weights do not match the geometry:";
        if (!missing.empty()) text += " missing " + missing.front() + " (" + std::to_string(missing.size()) + " in all)";
        if (!unexpected.empty()) {
            text += " unexpected " + unexpected.front() + " (" + std::to_string(unexpected.size()) + " in all)";
        }
        return text;
    }
    if (!wrong.empty()) {
        return std::to_string(wrong.size()) + " tensors have another shape, e.g. " + wrong.front();
    }
    return {};
}

double UnitRandom::next() {
    // splitmix64, then the top 53 bits as a uniform value in [0, 1).
    state_ += 0x9E3779B97F4A7C15ull;
    std::uint64_t z = state_;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z ^= z >> 31;
    const double uniform = static_cast<double>(z >> 11) * (1.0 / 9007199254740992.0);
    return (2.0 * uniform - 1.0) * 1.7320508075688772; // variance 1
}

std::pair<double, double> random_tensor_scale(std::string_view name, const Shape& shape) {
    if (name == embedding_name) return {1.0, 0.0};
    if (ends_with(name, "A_log") || ends_with(name, "dt_bias")) return {0.5, 0.0};
    if (ends_with(name, "linear_attn.norm.weight")) return {0.1, 1.0};
    if (ends_with(name, "norm.weight") || ends_with(name, "layernorm.weight") || ends_with(name, ".bias")) return {0.1, 0.0};
    const double fan_in = static_cast<double>(std::max<std::size_t>(shape.empty() ? 1 : shape.back(), 1));
    return {1.0 / std::sqrt(fan_in), 0.0};
}

} // namespace sepal
