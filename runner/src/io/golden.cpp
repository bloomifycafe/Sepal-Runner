#include "sepal/golden.hpp"

#include <bit>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <vector>

namespace sepal {

namespace {

constexpr std::string_view magic = "SEPALGV1";
constexpr std::string_view weights_prefix = "weights/";
constexpr std::uint32_t max_rank = 8;

// A bounds-checked little-endian cursor over the file's bytes.
class Reader {
public:
    explicit Reader(std::span<const unsigned char> bytes) : bytes_(bytes) {}

    std::size_t remaining() const { return bytes_.size() - pos_; }

    std::span<const unsigned char> take(std::size_t count, const char* what) {
        if (count > remaining()) {
            throw std::runtime_error(std::string("golden file: truncated while reading ") + what);
        }
        const std::span<const unsigned char> out = bytes_.subspan(pos_, count);
        pos_ += count;
        return out;
    }

    std::uint64_t uint(std::size_t width, const char* what) {
        const std::span<const unsigned char> raw = take(width, what);
        std::uint64_t value = 0;
        for (std::size_t i = 0; i < width; ++i) value |= static_cast<std::uint64_t>(raw[i]) << (8 * i);
        return value;
    }

    std::string text(std::size_t count, const char* what) {
        const std::span<const unsigned char> raw = take(count, what);
        return std::string(raw.begin(), raw.end());
    }

private:
    std::span<const unsigned char> bytes_;
    std::size_t pos_ = 0;
};

} // namespace

const TensorD& GoldenFile::array(std::string_view name) const {
    const auto it = arrays.find(name);
    if (it == arrays.end()) throw std::out_of_range("golden file: no array named " + std::string(name));
    return it->second;
}

ModelConfig GoldenFile::config() const {
    const json::Value& c = header.at("config");
    ModelConfig cfg;
    cfg.name = header.at("preset").as_string();
    cfg.num_layers = c.at("num_layers").as_uint();
    cfg.full_attention_interval = c.at("full_attention_interval").as_uint();
    cfg.hidden_size = c.at("hidden_size").as_uint();
    cfg.intermediate_size = c.at("intermediate_size").as_uint();
    cfg.attn_heads = c.at("attention_heads").as_uint();
    cfg.attn_kv_heads = c.at("attention_kv_heads").as_uint();
    cfg.attn_head_dim = c.at("attention_head_dim").as_uint();
    cfg.dn_key_heads = c.at("deltanet_key_heads").as_uint();
    cfg.dn_value_heads = c.at("deltanet_value_heads").as_uint();
    cfg.dn_key_head_dim = c.at("deltanet_key_head_dim").as_uint();
    cfg.dn_value_head_dim = c.at("deltanet_value_head_dim").as_uint();
    cfg.conv_kernel = c.at("conv_kernel").as_uint();
    cfg.vocab_size = c.at("vocab_size").as_uint();
    cfg.rms_norm_eps = c.at("rms_norm_eps").as_number();
    cfg.rope_theta = c.at("rope_theta").as_number();
    cfg.partial_rotary_factor = c.at("partial_rotary_factor").as_number();
    cfg.pointer_dim = c.at("pointer_dim").as_uint();
    cfg.lm_head_tied = c.at("tied_lm_head").as_bool();
    require_valid(cfg);
    return cfg;
}

std::map<std::string, TensorD, std::less<>> GoldenFile::weights() const {
    std::map<std::string, TensorD, std::less<>> out;
    for (const auto& [name, tensor] : arrays) {
        if (std::string_view(name).substr(0, weights_prefix.size()) == weights_prefix) {
            out.emplace(name.substr(weights_prefix.size()), tensor);
        }
    }
    return out;
}

GoldenFile parse_golden(std::span<const unsigned char> bytes) {
    Reader in(bytes);
    if (in.text(magic.size(), "the magic") != magic) throw std::runtime_error("golden file: wrong magic, not SEPALGV1");

    GoldenFile file;
    const std::size_t header_bytes = static_cast<std::size_t>(in.uint(4, "the header length"));
    const std::string header_text = in.text(header_bytes, "the header");
    file.header = json::parse(header_text);
    if (file.header.at("format").as_string() != "sepal-golden" || file.header.at("version").as_uint() != 1) {
        throw std::runtime_error("golden file: unknown format or version in the header");
    }

    const std::uint64_t count = in.uint(4, "the array count");
    for (std::uint64_t n = 0; n < count; ++n) {
        const std::size_t name_bytes = static_cast<std::size_t>(in.uint(4, "an array name length"));
        std::string name = in.text(name_bytes, "an array name");
        const std::uint64_t rank = in.uint(4, "an array rank");
        if (rank > max_rank) throw std::runtime_error("golden file: array " + name + " has a rank above 8");

        std::vector<std::size_t> shape;
        std::uint64_t elements = 1;
        for (std::uint64_t r = 0; r < rank; ++r) {
            const std::uint64_t extent = in.uint(8, "an array shape");
            // Each extent is bounded by the bytes left, so the running product cannot overflow unnoticed.
            if (extent > in.remaining() || (extent != 0 && elements > in.remaining() / extent)) {
                throw std::runtime_error("golden file: array " + name + " is larger than the file");
            }
            elements *= extent;
            shape.push_back(static_cast<std::size_t>(extent));
        }
        if (elements > in.remaining() / 8) {
            throw std::runtime_error("golden file: array " + name + " is larger than the file");
        }

        std::vector<double> values(static_cast<std::size_t>(elements));
        for (double& value : values) value = std::bit_cast<double>(in.uint(8, "array values"));
        if (!file.arrays.emplace(std::move(name), TensorD::from_values(std::move(shape), std::move(values))).second) {
            throw std::runtime_error("golden file: an array name appears twice");
        }
    }
    if (in.remaining() != 0) throw std::runtime_error("golden file: bytes left over after the last array");
    return file;
}

GoldenFile load_golden(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("golden file: cannot open " + path);
    const std::vector<char> raw((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
    const std::vector<unsigned char> bytes(raw.begin(), raw.end());
    return parse_golden(bytes);
}

} // namespace sepal
