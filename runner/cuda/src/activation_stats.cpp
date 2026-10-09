// Calibration statistics of the int8 multiplies (cuda_engine.hpp, ActivationStats): the file they travel in.
//
//     0   8  "SEPALACT"
//     8   4  version (1)
//    12   4  layers
//    16   4  groups (4)
//    then, per layer and group: u32 K, u64 tokens, then K float32 each of amax, mean_abs, rms (little endian)
//    last 8  FNV-1a 64 over everything before it
//
// The file is untrusted input when it is read: every size is checked against what is left of the file before it is used.

#include <cstring>
#include <fstream>
#include <stdexcept>

#include "sepal/cuda/cuda_engine.hpp"

namespace sepal::cuda {
namespace {

constexpr char kMagic[8] = {'S', 'E', 'P', 'A', 'L', 'A', 'C', 'T'};
constexpr std::uint32_t kVersion = 1;
constexpr std::uint32_t kMaxLayers = 4096;
constexpr std::uint32_t kMaxChannels = 1u << 24;

std::uint64_t fnv1a(const std::vector<char>& bytes, std::size_t count) {
    std::uint64_t h = 1469598103934665603ull;
    for (std::size_t i = 0; i < count; ++i) h = (h ^ static_cast<unsigned char>(bytes[i])) * 1099511628211ull;
    return h;
}

template <class T>
void put(std::vector<char>& out, const T& v) {
    const char* p = reinterpret_cast<const char*>(&v);
    out.insert(out.end(), p, p + sizeof(T));
}

class Reader {
public:
    explicit Reader(const std::vector<char>& bytes) : bytes_(bytes) {}
    template <class T>
    T get() {
        need(sizeof(T));
        T v;
        std::memcpy(&v, bytes_.data() + at_, sizeof(T));
        at_ += sizeof(T);
        return v;
    }
    void floats(std::vector<float>& out, std::size_t count) {
        need(count * sizeof(float));
        out.resize(count);
        std::memcpy(out.data(), bytes_.data() + at_, count * sizeof(float));
        at_ += count * sizeof(float);
    }
    std::size_t left() const { return bytes_.size() - at_; }

private:
    void need(std::size_t n) const {
        if (n > bytes_.size() - at_) throw std::invalid_argument("activation statistics file: truncated");
    }
    const std::vector<char>& bytes_;
    std::size_t at_ = 0;
};

} // namespace

void save_activation_stats(const ActivationStats& stats, const std::filesystem::path& path) {
    std::vector<char> out(kMagic, kMagic + sizeof kMagic);
    put(out, kVersion);
    put(out, static_cast<std::uint32_t>(stats.layers.size()));
    put(out, static_cast<std::uint32_t>(ActivationStats::kGroups));
    if (stats.tokens.size() != stats.layers.size()) throw std::invalid_argument("save_activation_stats: tokens and layers differ in length");
    for (std::size_t l = 0; l < stats.layers.size(); ++l) {
        for (std::size_t g = 0; g < ActivationStats::kGroups; ++g) {
            const ChannelStats& c = stats.layers[l][g];
            if (c.mean_abs.size() != c.amax.size() || c.rms.size() != c.amax.size()) throw std::invalid_argument("save_activation_stats: ragged channel statistics");
            put(out, static_cast<std::uint32_t>(c.amax.size()));
            put(out, stats.tokens[l][g]);
            for (const std::vector<float>* v : {&c.amax, &c.mean_abs, &c.rms}) {
                const char* p = reinterpret_cast<const char*>(v->data());
                out.insert(out.end(), p, p + v->size() * sizeof(float));
            }
        }
    }
    put(out, fnv1a(out, out.size()));
    const std::filesystem::path temp = path.string() + ".tmp";
    {
        std::ofstream file(temp, std::ios::binary | std::ios::trunc);
        file.write(out.data(), static_cast<std::streamsize>(out.size()));
        file.flush();
        if (!file) throw std::runtime_error("save_activation_stats: cannot write " + temp.string());
    }
    std::filesystem::rename(temp, path);
}

ActivationStats load_activation_stats(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("load_activation_stats: cannot open " + path.string());
    const std::streamoff size = file.tellg();
    if (size < 32 || size > (std::streamoff{1} << 30)) throw std::invalid_argument("activation statistics file: bad size");
    std::vector<char> bytes(static_cast<std::size_t>(size));
    file.seekg(0);
    file.read(bytes.data(), size);
    if (!file) throw std::runtime_error("load_activation_stats: cannot read " + path.string());
    std::uint64_t want = 0;
    std::memcpy(&want, bytes.data() + bytes.size() - 8, 8);
    if (fnv1a(bytes, bytes.size() - 8) != want) throw std::invalid_argument("activation statistics file: hash mismatch (corrupted)");
    bytes.resize(bytes.size() - 8);
    Reader in(bytes);
    char magic[8];
    for (char& c : magic) c = in.get<char>();
    if (std::memcmp(magic, kMagic, sizeof kMagic) != 0) throw std::invalid_argument("activation statistics file: not one (bad magic)");
    if (in.get<std::uint32_t>() != kVersion) throw std::invalid_argument("activation statistics file: unknown version");
    const std::uint32_t layers = in.get<std::uint32_t>();
    if (layers == 0 || layers > kMaxLayers) throw std::invalid_argument("activation statistics file: bad layer count");
    if (in.get<std::uint32_t>() != ActivationStats::kGroups) throw std::invalid_argument("activation statistics file: bad group count");
    ActivationStats out;
    out.layers.resize(layers);
    out.tokens.resize(layers);
    for (std::uint32_t l = 0; l < layers; ++l) {
        for (std::size_t g = 0; g < ActivationStats::kGroups; ++g) {
            const std::uint32_t k = in.get<std::uint32_t>();
            if (k == 0 || k > kMaxChannels) throw std::invalid_argument("activation statistics file: bad channel count");
            out.tokens[l][g] = in.get<std::uint64_t>();
            ChannelStats& c = out.layers[l][g];
            in.floats(c.amax, k);
            in.floats(c.mean_abs, k);
            in.floats(c.rms, k);
        }
    }
    if (in.left() != 0) throw std::invalid_argument("activation statistics file: trailing bytes");
    return out;
}

} // namespace sepal::cuda
