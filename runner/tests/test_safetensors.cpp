// The safetensors reader: valid files, every kind of malformed file, files on disk, and checkpoints of several
// files. All files are built here, byte by byte, from the format description.

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "sepal/mapped_file.hpp"
#include "sepal/safetensors.hpp"
#include "test.hpp"

using namespace sepal;

namespace {

using Bytes = std::vector<unsigned char>;

// A header padded with spaces to a multiple of 8 bytes (so tensor data is 8-byte aligned), then `data`.
Bytes file_of(std::string header, const Bytes& data, bool pad = true) {
    if (pad) header.append((8 - header.size() % 8) % 8, ' ');
    Bytes out(8);
    const std::uint64_t n = header.size();
    for (int i = 0; i < 8; ++i) out[static_cast<std::size_t>(i)] = static_cast<unsigned char>(n >> (8 * i));
    out.insert(out.end(), header.begin(), header.end());
    out.insert(out.end(), data.begin(), data.end());
    return out;
}

Bytes zeros(std::size_t n) { return Bytes(n, 0); }

void put_f32(Bytes& out, float v) {
    unsigned char raw[4];
    std::memcpy(raw, &v, 4);
    out.insert(out.end(), raw, raw + 4);
}

void put_u16(Bytes& out, std::uint16_t v) {
    out.push_back(static_cast<unsigned char>(v & 0xFF));
    out.push_back(static_cast<unsigned char>(v >> 8));
}

// Whether parsing `bytes` throws a SafetensorsError whose message contains `needle`.
bool refused(const Bytes& bytes, const std::string& needle) {
    try {
        (void)SafetensorsFile::from_memory(bytes, "test");
    } catch (const SafetensorsError& error) {
        return std::string(error.what()).find(needle) != std::string::npos;
    }
    return false;
}

// Two tensors: a: F32 [2,3] (24 bytes), b: BF16 [4] (8 bytes), c: I32 [] a scalar (4 bytes) plus metadata.
Bytes valid_file() {
    Bytes data;
    for (int i = 0; i < 6; ++i) put_f32(data, static_cast<float>(i) + 0.5f);
    put_u16(data, 0x3F80); // bf16 1.0
    put_u16(data, 0xC000); // bf16 -2.0
    put_u16(data, 0x0000);
    put_u16(data, 0x4040); // bf16 3.0
    for (const int byte : {7, 0, 0, 0}) data.push_back(static_cast<unsigned char>(byte));
    const std::string header =
        R"({"__metadata__":{"format":"pt","split":"12"},)"
        R"("b":{"dtype":"BF16","shape":[4],"data_offsets":[24,32]},)"
        R"("c":{"dtype":"I32","shape":[],"data_offsets":[32,36]},)"
        R"("a":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}})";
    return file_of(header, data);
}

std::filesystem::path scratch_directory(const std::string& name) {
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / ("sepal_test_" + name);
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    return dir;
}

void write_file(const std::filesystem::path& path, const Bytes& bytes) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
}

void write_text(const std::filesystem::path& path, const std::string& text) {
    std::ofstream out(path, std::ios::binary);
    out << text;
}

// One-tensor file holding the F32 values `first`, `first + 1`, ...
Bytes one_tensor_file(const std::string& name, int count, float first) {
    Bytes data;
    for (int i = 0; i < count; ++i) put_f32(data, first + static_cast<float>(i));
    return file_of("{\"" + name + "\":{\"dtype\":\"F32\",\"shape\":[" + std::to_string(count) + "],\"data_offsets\":[0," +
                       std::to_string(count * 4) + "]}}",
                   data);
}

} // namespace

SEPAL_TEST(safetensors, valid_file_is_read_without_copying) {
    const Bytes bytes = valid_file();
    const SafetensorsFile file = SafetensorsFile::from_memory(bytes, "valid");
    CHECK_EQ(file.tensors().size(), std::size_t(3));
    // File order, not header order.
    CHECK_EQ(file.tensors()[0].name, std::string("a"));
    CHECK_EQ(file.tensors()[1].name, std::string("b"));
    CHECK_EQ(file.tensors()[2].name, std::string("c"));

    const TensorView& a = file.at("a");
    CHECK(a.type == StType::F32);
    CHECK_EQ(a.shape.size(), std::size_t(2));
    CHECK_EQ(a.shape[0], std::size_t(2));
    CHECK_EQ(a.shape[1], std::size_t(3));
    CHECK_EQ(a.elements(), std::size_t(6));
    CHECK_EQ(a.bytes, std::size_t(24));
    CHECK_EQ(a.values<float>()[4], 4.5f);
    // The view points into the caller's bytes: nothing was copied.
    CHECK(a.data >= bytes.data() && a.data + a.bytes <= bytes.data() + bytes.size());

    const TensorView& b = file.at("b");
    CHECK(b.type == StType::BF16);
    CHECK_EQ(b.bits()[1], std::uint16_t(0xC000));
    float widened[4];
    convert_floating(b, widened);
    CHECK_EQ(widened[0], 1.0f);
    CHECK_EQ(widened[1], -2.0f);
    CHECK_EQ(widened[3], 3.0f);

    const TensorView& c = file.at("c"); // a scalar: rank 0, one element
    CHECK(c.type == StType::I32);
    CHECK_EQ(c.shape.size(), std::size_t(0));
    CHECK_EQ(c.elements(), std::size_t(1));
    CHECK_EQ(c.values<std::int32_t>()[0], 7);

    CHECK_EQ(file.metadata().at("format"), std::string("pt"));
    CHECK_EQ(file.metadata().at("split"), std::string("12"));
    CHECK(file.find("missing") == nullptr);
    CHECK_THROWS(file.at("missing"), std::out_of_range);
    CHECK_THROWS(a.values<std::int32_t>(), std::logic_error); // asking for the wrong type is a bug, not a cast
    CHECK_THROWS(a.bits(), std::logic_error);
}

SEPAL_TEST(safetensors, f16_values_are_decoded_exactly) {
    Bytes data;
    put_u16(data, 0x3C00); // 1.0
    put_u16(data, 0xC000); // -2.0
    put_u16(data, 0x0001); // the smallest subnormal, 2^-24
    put_u16(data, 0x7BFF); // 65504
    const Bytes bytes = file_of(R"({"h":{"dtype":"F16","shape":[2,2],"data_offsets":[0,8]}})", data);
    const SafetensorsFile file = SafetensorsFile::from_memory(bytes);
    double out[4];
    convert_floating(file.at("h"), out);
    CHECK_EQ(out[0], 1.0);
    CHECK_EQ(out[1], -2.0);
    CHECK_EQ(out[2], 5.9604644775390625e-08);
    CHECK_EQ(out[3], 65504.0);
    CHECK_THROWS(convert_floating(SafetensorsFile::from_memory(valid_file()).at("c"), out), std::invalid_argument);
}

SEPAL_TEST(safetensors, an_empty_tensor_and_an_empty_file_are_valid) {
    const Bytes with_empty = file_of(
        R"({"e":{"dtype":"F32","shape":[0,5],"data_offsets":[0,0]},"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})", zeros(4));
    const SafetensorsFile file = SafetensorsFile::from_memory(with_empty);
    CHECK_EQ(file.at("e").elements(), std::size_t(0));
    CHECK_EQ(file.at("x").elements(), std::size_t(1));
    const SafetensorsFile none = SafetensorsFile::from_memory(file_of("{}", {}));
    CHECK_EQ(none.tensors().size(), std::size_t(0));
}

SEPAL_TEST(safetensors, malformed_headers_are_refused) {
    CHECK(refused(Bytes{1, 2, 3}, "shorter than"));
    Bytes too_long = file_of("{}", {});
    too_long[0] = 200; // claims 200 header bytes, the file has 8
    CHECK(refused(too_long, "claims 200 bytes"));
    Bytes enormous = file_of("{}", {});
    enormous[7] = 0x7F;
    CHECK(refused(enormous, "above the limit"));
    CHECK(refused(file_of("[]", {}), "does not start with '{'"));
    CHECK(refused(file_of("", {}), "does not start with '{'"));
    CHECK(refused(file_of("{\"a\":", {}), "not valid JSON"));
    CHECK(refused(file_of(R"({"a":3})", {}), "not an object"));
    CHECK(refused(file_of(R"({"a":{"dtype":"F32"}})", zeros(4)), "needs \"dtype\", \"shape\" and \"data_offsets\""));
    CHECK(refused(file_of(R"({"__metadata__":{"k":1}})", {}), "is not a string"));
    CHECK(refused(file_of(R"({"__metadata__":[]})", {}), "__metadata__ is not an object"));
    CHECK(refused(file_of(R"({"a":{"dtype":"F32","shape":[-1],"data_offsets":[0,4]}})", zeros(4)), "non-negative whole numbers"));
    CHECK(refused(file_of(R"({"a":{"dtype":"F32","shape":[1.5],"data_offsets":[0,4]}})", zeros(4)), "non-negative whole numbers"));
    CHECK(refused(file_of(R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4,8]}})", zeros(4)), "exactly two numbers"));
    CHECK(refused(file_of(R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"a":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}})", zeros(8)),
                  "appears twice"));
}

SEPAL_TEST(safetensors, unknown_element_types_are_refused) {
    CHECK(refused(file_of(R"({"a":{"dtype":"F8_E4M3","shape":[4],"data_offsets":[0,4]}})", zeros(4)), "unknown element type 'F8_E4M3'"));
    CHECK(refused(file_of(R"({"a":{"dtype":"float32","shape":[1],"data_offsets":[0,4]}})", zeros(4)), "unknown element type"));
    CHECK(refused(file_of(R"({"a":{"dtype":7,"shape":[1],"data_offsets":[0,4]}})", zeros(4)), "\"dtype\" is not a string"));
}

SEPAL_TEST(safetensors, offsets_outside_the_file_are_refused) {
    CHECK(refused(file_of(R"({"a":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}})", zeros(12)), "past the 12 bytes"));
    CHECK(refused(file_of(R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[4,0]}})", zeros(4)), "end before they begin"));
    CHECK(refused(file_of(R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[1000000000000,1000000000004]}})", zeros(4)), "past the"));
}

SEPAL_TEST(safetensors, shape_and_byte_size_must_agree) {
    CHECK(refused(file_of(R"({"a":{"dtype":"F32","shape":[2,3],"data_offsets":[0,20]}})", zeros(20)), "needs 24 bytes but data_offsets span 20"));
    CHECK(refused(file_of(R"({"a":{"dtype":"F16","shape":[3],"data_offsets":[0,8]}})", zeros(8)), "needs 6 bytes"));
    CHECK(refused(file_of(R"({"a":{"dtype":"F32","shape":[4294967296,4294967296,4294967296],"data_offsets":[0,4]}})", zeros(4)), "overflows 64 bits"));
}

SEPAL_TEST(safetensors, overlapping_tensors_are_refused) {
    CHECK(refused(file_of(R"({"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"b":{"dtype":"F32","shape":[2],"data_offsets":[4,12]}})", zeros(12)),
                  "overlap"));
    // The same bytes under two names.
    CHECK(refused(file_of(R"({"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"b":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}})", zeros(8)),
                  "overlap"));
}

SEPAL_TEST(safetensors, bytes_that_belong_to_no_tensor_are_refused) {
    CHECK(refused(file_of(R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}})", zeros(8)), "belong to no tensor")); // hole at the start
    CHECK(refused(file_of(R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"b":{"dtype":"F32","shape":[1],"data_offsets":[8,12]}})", zeros(12)),
                  "belong to no tensor")); // hole between
    CHECK(refused(file_of(R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})", zeros(8)), "last 4 bytes")); // trailing bytes
}

SEPAL_TEST(safetensors, misaligned_tensors_are_refused) {
    // An unpadded header of odd length makes the data start at an odd file offset, where an F32 tensor is not
    // 4-byte aligned.
    std::string header = R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})";
    if (header.size() % 2 == 0) header += ' ';
    CHECK(refused(file_of(header, zeros(4), false), "not a multiple of the 4-byte element size"));
    // A BF16 tensor after a 1-byte tensor starts at an odd offset.
    CHECK(refused(file_of(R"({"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"b":{"dtype":"BF16","shape":[1],"data_offsets":[1,3]}})", zeros(3)),
                  "not a multiple of the 2-byte element size"));
}

SEPAL_TEST(safetensors, files_are_mapped_from_disk) {
    const std::filesystem::path dir = scratch_directory("st_files");
    write_file(dir / "ok.safetensors", valid_file());
    {
        SafetensorsFile file = SafetensorsFile::open((dir / "ok.safetensors").string());
        CHECK_EQ(file.at("a").values<float>()[5], 5.5f);
        CHECK_EQ(file.at("b").bits()[3], std::uint16_t(0x4040));
        // Moving the file keeps the views valid: the mapping does not move.
        const SafetensorsFile moved = std::move(file);
        CHECK_EQ(moved.at("a").values<float>()[0], 0.5f);
    }
    Bytes truncated = valid_file();
    truncated.resize(truncated.size() - 4);
    write_file(dir / "truncated.safetensors", truncated);
    CHECK_THROWS(SafetensorsFile::open((dir / "truncated.safetensors").string()), SafetensorsError);
    write_file(dir / "empty.safetensors", {});
    CHECK_THROWS(SafetensorsFile::open((dir / "empty.safetensors").string()), SafetensorsError);
    CHECK_THROWS(SafetensorsFile::open((dir / "absent.safetensors").string()), std::runtime_error);
    CHECK_THROWS(MappedFile::open(dir.string()), std::runtime_error); // a directory is not a file
    {
        const MappedFile empty = MappedFile::open((dir / "empty.safetensors").string());
        CHECK_EQ(empty.bytes().size(), std::size_t(0));
    }
    std::filesystem::remove_all(dir);
}

SEPAL_TEST(safetensors, a_checkpoint_of_shards_without_an_index) {
    const std::filesystem::path dir = scratch_directory("st_shards");
    write_file(dir / "model-00001-of-00002.safetensors", one_tensor_file("x", 3, 1.0f));
    write_file(dir / "model-00002-of-00002.safetensors", one_tensor_file("y", 2, 10.0f));
    {
        const Checkpoint checkpoint = Checkpoint::open_directory(dir.string());
        CHECK_EQ(checkpoint.files().size(), std::size_t(2));
        CHECK_EQ(checkpoint.tensor_count(), std::size_t(2));
        CHECK_EQ(checkpoint.at("x").values<float>()[2], 3.0f);
        CHECK_EQ(checkpoint.at("y").values<float>()[1], 11.0f);
        CHECK(checkpoint.find("z") == nullptr);
        CHECK_EQ(checkpoint.names().front(), std::string("x"));
    } // closed before the files are removed: Windows will not delete a mapped file

    // One shard missing.
    std::filesystem::remove(dir / "model-00001-of-00002.safetensors");
    CHECK_THROWS(Checkpoint::open_directory(dir.string()), SafetensorsError);
    std::filesystem::remove_all(dir);
}

SEPAL_TEST(safetensors, a_checkpoint_with_an_index) {
    const std::filesystem::path dir = scratch_directory("st_index");
    write_file(dir / "a.safetensors", one_tensor_file("x", 3, 1.0f));
    write_file(dir / "b.safetensors", one_tensor_file("y", 2, 10.0f));
    write_text(dir / "model.safetensors.index.json", R"({"metadata":{},"weight_map":{"x":"a.safetensors","y":"b.safetensors"}})");
    {
        const Checkpoint checkpoint = Checkpoint::open_directory(dir.string());
        CHECK_EQ(checkpoint.tensor_count(), std::size_t(2));
        CHECK_EQ(checkpoint.at("y").values<float>()[0], 10.0f);
    }

    // The index disagrees with the shards: a tensor in the wrong file, a listed tensor that is absent.
    write_text(dir / "model.safetensors.index.json", R"({"weight_map":{"x":"b.safetensors","y":"a.safetensors"}})");
    CHECK_THROWS(Checkpoint::open_directory(dir.string()), SafetensorsError);
    write_text(dir / "model.safetensors.index.json", R"({"weight_map":{"x":"a.safetensors","y":"b.safetensors","z":"b.safetensors"}})");
    CHECK_THROWS(Checkpoint::open_directory(dir.string()), SafetensorsError);
    // A shard name that leaves the directory is refused before anything is opened.
    write_text(dir / "model.safetensors.index.json", R"({"weight_map":{"x":"../a.safetensors"}})");
    CHECK_THROWS(Checkpoint::open_directory(dir.string()), SafetensorsError);
    std::filesystem::remove_all(dir);
}

SEPAL_TEST(safetensors, one_file_a_missing_directory_and_a_duplicate_name) {
    const std::filesystem::path dir = scratch_directory("st_single");
    write_file(dir / "model.safetensors", one_tensor_file("x", 2, 1.0f));
    CHECK_EQ(Checkpoint::open_directory(dir.string()).tensor_count(), std::size_t(1));
    CHECK_THROWS(Checkpoint::open_directory((dir / "nowhere").string()), SafetensorsError);
    write_file(dir / "other.safetensors", one_tensor_file("x", 2, 5.0f));
    CHECK_THROWS((Checkpoint::open_files({(dir / "model.safetensors").string(), (dir / "other.safetensors").string()})), SafetensorsError);
    CHECK_THROWS(Checkpoint::open_files({}), SafetensorsError);
    std::filesystem::remove_all(dir);
    const std::filesystem::path empty_dir = scratch_directory("st_none");
    CHECK_THROWS(Checkpoint::open_directory(empty_dir.string()), SafetensorsError);
    std::filesystem::remove_all(empty_dir);
}
