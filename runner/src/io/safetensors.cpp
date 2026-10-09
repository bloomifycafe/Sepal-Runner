#include "sepal/safetensors.hpp"

#include <algorithm>
#include <bit>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <regex>
#include <set>
#include <utility>

#include "sepal/json.hpp"

namespace sepal {

namespace {

namespace fs = std::filesystem;

// The reference implementation refuses headers above 100 MB; a larger one is a damaged or hostile file.
constexpr std::uint64_t max_header_bytes = 100'000'000;
constexpr std::size_t max_rank = 32;

constexpr std::string_view metadata_key = "__metadata__";

struct TypeInfo {
    StType type;
    std::string_view name;
    std::size_t bytes;
};

constexpr TypeInfo type_table[] = {
    {StType::Bool, "BOOL", 1}, {StType::U8, "U8", 1},   {StType::I8, "I8", 1},   {StType::I16, "I16", 2},
    {StType::U16, "U16", 2},   {StType::F16, "F16", 2}, {StType::BF16, "BF16", 2}, {StType::I32, "I32", 4},
    {StType::U32, "U32", 4},   {StType::F32, "F32", 4}, {StType::F64, "F64", 8}, {StType::I64, "I64", 8},
    {StType::U64, "U64", 8},
};

[[noreturn]] void refuse(const std::string& origin, const std::string& what) {
    throw SafetensorsError("safetensors file '" + origin + "': " + what);
}

[[noreturn]] void refuse_tensor(const std::string& origin, const std::string& tensor, const std::string& what) {
    refuse(origin, "tensor '" + tensor + "': " + what);
}

std::uint64_t read_u64_le(const unsigned char* p) {
    std::uint64_t value = 0;
    for (int i = 0; i < 8; ++i) value |= static_cast<std::uint64_t>(p[i]) << (8 * i);
    return value;
}

std::string shape_text(const std::vector<std::size_t>& shape) {
    std::string text = "[";
    for (std::size_t i = 0; i < shape.size(); ++i) text += (i == 0 ? "" : ", ") + std::to_string(shape[i]);
    return text + "]";
}

fs::path to_path(const std::string& utf8) {
    return fs::path(std::u8string(reinterpret_cast<const char8_t*>(utf8.data()), utf8.size()));
}

std::string to_utf8(const fs::path& path) {
    const std::u8string text = path.u8string();
    return std::string(reinterpret_cast<const char*>(text.data()), text.size());
}

struct Placement {
    std::size_t tensor; // index into the tensor list
    std::uint64_t begin;
    std::uint64_t end;
};

} // namespace

std::size_t bytes_of(StType type) noexcept {
    for (const TypeInfo& info : type_table) {
        if (info.type == type) return info.bytes;
    }
    return 0;
}

std::string_view name_of(StType type) noexcept {
    for (const TypeInfo& info : type_table) {
        if (info.type == type) return info.name;
    }
    return "?";
}

std::optional<StType> parse_st_type(std::string_view name) noexcept {
    for (const TypeInfo& info : type_table) {
        if (info.name == name) return info.type;
    }
    return std::nullopt;
}

std::optional<DType> as_dtype(StType type) noexcept {
    switch (type) {
    case StType::F32: return DType::F32;
    case StType::F16: return DType::F16;
    case StType::BF16: return DType::BF16;
    case StType::I8: return DType::I8;
    default: return std::nullopt;
    }
}

std::size_t TensorView::elements() const {
    std::size_t n = 1;
    for (const std::size_t d : shape) n *= d; // the reader has checked that this does not overflow
    return n;
}

std::span<const std::uint16_t> TensorView::bits() const {
    if (bytes_of(type) != 2) {
        throw std::logic_error("tensor '" + name + "' holds " + std::string(name_of(type)) + ", which is not a 16-bit type");
    }
    return {reinterpret_cast<const std::uint16_t*>(data), bytes / 2};
}

SafetensorsFile SafetensorsFile::open(const std::string& path) {
    auto mapping = std::make_unique<MappedFile>(MappedFile::open(path));
    SafetensorsFile file = parse(mapping->bytes(), path);
    file.mapping_ = std::move(mapping);
    return file;
}

SafetensorsFile SafetensorsFile::from_memory(std::span<const unsigned char> bytes, std::string origin) {
    return parse(bytes, std::move(origin));
}

SafetensorsFile SafetensorsFile::parse(std::span<const unsigned char> bytes, std::string origin) {
    if (bytes.size() < 8) refuse(origin, "shorter than the 8 bytes that hold the header length");
    const std::uint64_t header_length = read_u64_le(bytes.data());
    if (header_length > max_header_bytes) {
        refuse(origin, "the header claims " + std::to_string(header_length) + " bytes, above the limit of 100 MB");
    }
    if (header_length > bytes.size() - 8) {
        refuse(origin, "the header claims " + std::to_string(header_length) + " bytes but the file holds only " +
                           std::to_string(bytes.size() - 8) + " after the length");
    }
    const std::size_t data_begin = 8 + static_cast<std::size_t>(header_length);
    const std::uint64_t data_size = bytes.size() - data_begin;
    if (header_length == 0 || bytes[8] != '{') refuse(origin, "the header does not start with '{'");

    json::Value header;
    try {
        header = json::parse(std::string_view(reinterpret_cast<const char*>(bytes.data()) + 8, static_cast<std::size_t>(header_length)));
    } catch (const std::runtime_error& error) {
        refuse(origin, std::string("the header is not valid JSON: ") + error.what());
    }
    if (header.kind() != json::Kind::Object) refuse(origin, "the header is not a JSON object");

    SafetensorsFile file;
    file.origin_ = origin;
    std::vector<Placement> placements;
    std::set<std::string> seen;
    for (std::size_t i = 0; i < header.keys().size(); ++i) {
        const std::string& name = header.keys()[i];
        const json::Value& entry = header.items()[i];
        if (!seen.insert(name).second) refuse(origin, "the name '" + name + "' appears twice in the header");

        if (name == metadata_key) {
            if (entry.kind() != json::Kind::Object) refuse(origin, "__metadata__ is not an object");
            for (std::size_t m = 0; m < entry.keys().size(); ++m) {
                if (entry.items()[m].kind() != json::Kind::String) {
                    refuse(origin, "__metadata__ member '" + entry.keys()[m] + "' is not a string");
                }
                file.metadata_.emplace(entry.keys()[m], entry.items()[m].as_string());
            }
            continue;
        }

        if (entry.kind() != json::Kind::Object) refuse_tensor(origin, name, "the header entry is not an object");
        const json::Value* dtype_value = entry.find("dtype");
        const json::Value* shape_value = entry.find("shape");
        const json::Value* offsets_value = entry.find("data_offsets");
        if (dtype_value == nullptr || shape_value == nullptr || offsets_value == nullptr) {
            refuse_tensor(origin, name, "the entry needs \"dtype\", \"shape\" and \"data_offsets\"");
        }
        if (dtype_value->kind() != json::Kind::String) refuse_tensor(origin, name, "\"dtype\" is not a string");
        const std::optional<StType> type = parse_st_type(dtype_value->as_string());
        if (!type) refuse_tensor(origin, name, "unknown element type '" + dtype_value->as_string() + "'");

        std::vector<std::uint64_t> dims;
        std::vector<std::uint64_t> offsets;
        try {
            dims = shape_value->as_uint_list();
            offsets = offsets_value->as_uint_list();
        } catch (const std::runtime_error&) {
            refuse_tensor(origin, name, "\"shape\" and \"data_offsets\" must be lists of non-negative whole numbers");
        }
        if (offsets.size() != 2) refuse_tensor(origin, name, "\"data_offsets\" must hold exactly two numbers");
        if (dims.size() > max_rank) refuse_tensor(origin, name, "rank " + std::to_string(dims.size()) + " is above the limit of 32");

        TensorView view;
        view.name = name;
        view.type = *type;
        const std::uint64_t element_bytes = bytes_of(*type);
        std::uint64_t expected_bytes = element_bytes;
        for (const std::uint64_t d : dims) {
            if (d != 0 && expected_bytes > std::numeric_limits<std::uint64_t>::max() / d) {
                refuse_tensor(origin, name, "the shape overflows 64 bits");
            }
            expected_bytes *= d;
            view.shape.push_back(static_cast<std::size_t>(d));
        }

        const std::uint64_t begin = offsets[0];
        const std::uint64_t end = offsets[1];
        if (begin > end) refuse_tensor(origin, name, "data_offsets [" + std::to_string(begin) + ", " + std::to_string(end) + ") end before they begin");
        if (end > data_size) {
            refuse_tensor(origin, name, "data_offsets end at " + std::to_string(end) + ", past the " + std::to_string(data_size) +
                                            " bytes of data in the file");
        }
        if (end - begin != expected_bytes) {
            refuse_tensor(origin, name, "shape " + shape_text(view.shape) + " of " + std::string(name_of(*type)) + " needs " +
                                            std::to_string(expected_bytes) + " bytes but data_offsets span " +
                                            std::to_string(end - begin));
        }
        const std::uint64_t file_offset = data_begin + begin;
        const std::uintptr_t address = reinterpret_cast<std::uintptr_t>(bytes.data()) + static_cast<std::uintptr_t>(file_offset);
        if (file_offset % element_bytes != 0 || address % element_bytes != 0) {
            refuse_tensor(origin, name, "its bytes start at file offset " + std::to_string(file_offset) +
                                            ", which is not a multiple of the " + std::to_string(element_bytes) +
                                            "-byte element size");
        }
        view.data = bytes.data() + file_offset;
        view.bytes = static_cast<std::size_t>(expected_bytes);
        placements.push_back({file.tensors_.size(), begin, end});
        file.tensors_.push_back(std::move(view));
    }

    // The data region is the tensors one after the other with nothing between and nothing after.
    std::sort(placements.begin(), placements.end(), [](const Placement& a, const Placement& b) {
        return a.begin != b.begin ? a.begin < b.begin : a.end < b.end;
    });
    std::uint64_t covered = 0;
    for (const Placement& p : placements) {
        const std::string& name = file.tensors_[p.tensor].name;
        if (p.begin < covered) refuse_tensor(origin, name, "its bytes overlap those of the tensor before it");
        if (p.begin > covered) {
            refuse_tensor(origin, name, "bytes " + std::to_string(covered) + " to " + std::to_string(p.begin) + " belong to no tensor");
        }
        covered = p.end;
    }
    if (covered != data_size) {
        refuse(origin, "the last " + std::to_string(data_size - covered) + " bytes of the file belong to no tensor");
    }

    // File order: the order of the bytes, which is the order a sequential read (or an upload) wants.
    std::vector<TensorView> ordered;
    ordered.reserve(file.tensors_.size());
    for (const Placement& p : placements) ordered.push_back(std::move(file.tensors_[p.tensor]));
    file.tensors_ = std::move(ordered);
    for (std::size_t i = 0; i < file.tensors_.size(); ++i) file.index_.emplace(file.tensors_[i].name, i);
    return file;
}

const TensorView* SafetensorsFile::find(std::string_view name) const {
    const auto it = index_.find(name);
    return it == index_.end() ? nullptr : &tensors_[it->second];
}

const TensorView& SafetensorsFile::at(std::string_view name) const {
    const TensorView* found = find(name);
    if (found == nullptr) throw std::out_of_range("safetensors file '" + origin_ + "' has no tensor named " + std::string(name));
    return *found;
}

void Checkpoint::add(std::shared_ptr<const SafetensorsFile> file) {
    for (const TensorView& tensor : file->tensors()) {
        const auto inserted = by_name_.emplace(tensor.name, &tensor);
        if (!inserted.second) {
            std::string first = "an earlier file";
            for (const auto& earlier : files_) {
                if (earlier->find(tensor.name) != nullptr) first = "'" + earlier->origin() + "'";
            }
            throw SafetensorsError("tensor '" + tensor.name + "' is in two files of the checkpoint: " + first + " and '" + file->origin() + "'");
        }
    }
    files_.push_back(std::move(file));
}

Checkpoint Checkpoint::open_files(const std::vector<std::string>& paths) {
    if (paths.empty()) throw SafetensorsError("a checkpoint needs at least one file");
    Checkpoint checkpoint;
    for (const std::string& path : paths) {
        checkpoint.add(std::make_shared<const SafetensorsFile>(SafetensorsFile::open(path)));
    }
    return checkpoint;
}

Checkpoint Checkpoint::open_directory(const std::string& directory) {
    const fs::path dir = to_path(directory);
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) throw SafetensorsError("'" + directory + "' is not a directory");

    const fs::path index_path = dir / "model.safetensors.index.json";
    if (fs::is_regular_file(index_path, ec)) {
        // The index maps each tensor to its shard. It is data: a shard name is a plain file name or nothing.
        std::ifstream in(index_path, std::ios::binary);
        const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        json::Value index;
        try {
            index = json::parse(text);
        } catch (const std::runtime_error& error) {
            throw SafetensorsError("'" + to_utf8(index_path) + "' is not valid JSON: " + error.what());
        }
        const json::Value& weight_map = index.at("weight_map");
        std::map<std::string, std::string> shard_of;
        std::vector<std::string> shard_names;
        for (std::size_t i = 0; i < weight_map.keys().size(); ++i) {
            const std::string& shard = weight_map.items()[i].as_string();
            if (shard.empty() || shard.find_first_of("/\\") != std::string::npos || shard == "." || shard == "..") {
                throw SafetensorsError("the index names the shard '" + shard + "', which is not a plain file name");
            }
            shard_of.emplace(weight_map.keys()[i], shard);
            if (std::find(shard_names.begin(), shard_names.end(), shard) == shard_names.end()) shard_names.push_back(shard);
        }
        std::sort(shard_names.begin(), shard_names.end());
        std::vector<std::string> paths;
        for (const std::string& shard : shard_names) paths.push_back(to_utf8(dir / to_path(shard)));
        Checkpoint checkpoint = open_files(paths);
        for (const auto& [tensor, shard] : shard_of) {
            const TensorView* found = checkpoint.find(tensor);
            if (found == nullptr) throw SafetensorsError("the index puts tensor '" + tensor + "' in " + shard + ", which does not hold it");
        }
        for (std::size_t f = 0; f < checkpoint.files_.size(); ++f) {
            for (const TensorView& tensor : checkpoint.files_[f]->tensors()) {
                const auto it = shard_of.find(tensor.name);
                if (it == shard_of.end() || it->second != shard_names[f]) {
                    throw SafetensorsError("tensor '" + tensor.name + "' in " + shard_names[f] + " is not listed for that shard in the index");
                }
            }
        }
        return checkpoint;
    }

    const fs::path single = dir / "model.safetensors";
    if (fs::is_regular_file(single, ec)) return open_files({to_utf8(single)});

    // Shards without an index: model-00001-of-00004.safetensors ... all of them or none.
    static const std::regex shard_name(R"(model-(\d+)-of-(\d+)\.safetensors)");
    std::map<unsigned long, std::string> found;
    unsigned long total = 0;
    for (const fs::directory_entry& entry : fs::directory_iterator(dir, ec)) {
        const std::string file_name = to_utf8(entry.path().filename());
        std::smatch match;
        if (!std::regex_match(file_name, match, shard_name)) continue;
        const unsigned long number = std::stoul(match[1].str());
        const unsigned long of = std::stoul(match[2].str());
        if (total != 0 && of != total) throw SafetensorsError("the shards in '" + directory + "' disagree on how many there are");
        total = of;
        found.emplace(number, to_utf8(entry.path()));
    }
    if (found.empty()) throw SafetensorsError("no model.safetensors, index or model-*-of-*.safetensors shards in '" + directory + "'");
    std::vector<std::string> paths;
    for (unsigned long number = 1; number <= total; ++number) {
        const auto it = found.find(number);
        if (it == found.end()) {
            throw SafetensorsError("shard " + std::to_string(number) + " of " + std::to_string(total) + " is missing from '" + directory + "'");
        }
        paths.push_back(it->second);
    }
    if (found.size() != paths.size()) throw SafetensorsError("a shard number in '" + directory + "' is above the total of " + std::to_string(total));
    return open_files(paths);
}

const TensorView* Checkpoint::find(std::string_view name) const {
    const auto it = by_name_.find(name);
    return it == by_name_.end() ? nullptr : it->second;
}

const TensorView& Checkpoint::at(std::string_view name) const {
    const TensorView* found = find(name);
    if (found == nullptr) throw std::out_of_range("the checkpoint has no tensor named " + std::string(name));
    return *found;
}

std::vector<std::string> Checkpoint::names() const {
    std::vector<std::string> out;
    out.reserve(by_name_.size());
    for (const auto& [name, view] : by_name_) out.push_back(name);
    return out;
}

namespace {

template <class T>
void convert_floating_impl(const TensorView& tensor, T* out) {
    const std::size_t n = tensor.elements();
    switch (tensor.type) {
    case StType::F32: {
        const std::span<const float> in = tensor.values<float>();
        for (std::size_t i = 0; i < n; ++i) out[i] = static_cast<T>(in[i]);
        return;
    }
    case StType::F64: {
        const std::span<const double> in = tensor.values<double>();
        for (std::size_t i = 0; i < n; ++i) out[i] = static_cast<T>(in[i]);
        return;
    }
    case StType::F16: {
        const std::span<const std::uint16_t> in = tensor.bits();
        for (std::size_t i = 0; i < n; ++i) out[i] = static_cast<T>(half_to_float(in[i]));
        return;
    }
    case StType::BF16: {
        const std::span<const std::uint16_t> in = tensor.bits();
        for (std::size_t i = 0; i < n; ++i) out[i] = static_cast<T>(bf16_to_float(in[i]));
        return;
    }
    default:
        throw std::invalid_argument("tensor '" + tensor.name + "' holds " + std::string(name_of(tensor.type)) +
                                    ", which is not a floating-point type this reader converts");
    }
}

} // namespace

void convert_floating(const TensorView& tensor, float* out) { convert_floating_impl(tensor, out); }
void convert_floating(const TensorView& tensor, double* out) { convert_floating_impl(tensor, out); }

} // namespace sepal
