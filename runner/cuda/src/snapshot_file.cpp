#include "snapshot_file.hpp"

#include <algorithm>
#include <bit>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <new>
#include <string>

#include <cuda_runtime.h>

namespace sepal::cuda {

static_assert(std::endian::native == std::endian::little, "snapshot files are little-endian and written by memcpy of host integers");

namespace {

constexpr char kMagic[8] = {'S', 'E', 'P', 'A', 'L', 'S', 'N', 'P'};
constexpr char kEnd[8] = {'S', 'E', 'P', 'A', 'L', 'E', 'N', 'D'};
constexpr std::uint32_t kVersion = 1;
constexpr std::size_t kHeaderBytes = 168;
constexpr std::size_t kBlockEntryBytes = 24;
constexpr std::size_t kTrailerBytes = 16;
constexpr std::uint32_t kMaxBlocks = 65536;
constexpr std::size_t kChunk = std::size_t{4} << 20;

constexpr std::uint64_t P1 = 0x9E3779B185EBCA87ull;
constexpr std::uint64_t P2 = 0xC2B2AE3D27D4EB4Full;
constexpr std::uint64_t P3 = 0x165667B19E3779F9ull;

constexpr std::uint64_t rotl(std::uint64_t x, int r) { return (x << r) | (x >> (64 - r)); }

[[noreturn]] void refuse(const std::string& why) { throw SnapshotFileError("snapshot file refused: " + why); }

bool mul_overflows(std::uint64_t a, std::uint64_t b, std::uint64_t& out) {
    if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a) return true;
    out = a * b;
    return false;
}
bool add_overflows(std::uint64_t a, std::uint64_t b, std::uint64_t& out) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) return true;
    out = a + b;
    return false;
}

// Little-endian field writer / reader over a fixed buffer.
class Out {
public:
    explicit Out(std::size_t size) : bytes_(size, 0) {}
    void u32(std::size_t at, std::uint32_t v) { std::memcpy(bytes_.data() + at, &v, 4); }
    void u64(std::size_t at, std::uint64_t v) { std::memcpy(bytes_.data() + at, &v, 8); }
    std::vector<unsigned char>& bytes() { return bytes_; }

private:
    std::vector<unsigned char> bytes_;
};

std::uint32_t get32(const unsigned char* p, std::size_t at) {
    std::uint32_t v;
    std::memcpy(&v, p + at, 4);
    return v;
}
std::uint64_t get64(const unsigned char* p, std::size_t at) {
    std::uint64_t v;
    std::memcpy(&v, p + at, 8);
    return v;
}

} // namespace

std::uint64_t hash64(const void* data, std::size_t bytes, std::uint64_t seed) {
    const auto* p = static_cast<const unsigned char*>(data);
    std::uint64_t v[4] = {seed + P1 + P2, seed + P2, seed, seed - P1};
    std::size_t at = 0;
    for (; at + 32 <= bytes; at += 32) {
        for (int lane = 0; lane < 4; ++lane) {
            std::uint64_t w;
            std::memcpy(&w, p + at + 8 * static_cast<std::size_t>(lane), 8);
            v[lane] = rotl(v[lane] + w * P2, 31) * P1;
        }
    }
    std::uint64_t h = rotl(v[0], 1) + rotl(v[1], 7) + rotl(v[2], 12) + rotl(v[3], 18);
    for (int lane = 0; lane < 4; ++lane) h = (h ^ (rotl(v[lane] * P2, 31) * P1)) * P1 + P3;
    h += bytes;
    for (; at + 8 <= bytes; at += 8) {
        std::uint64_t w;
        std::memcpy(&w, p + at, 8);
        h = rotl(h ^ (rotl(w * P2, 31) * P1), 27) * P1 + P3;
    }
    for (; at < bytes; ++at) h = rotl(h ^ (p[at] * P3), 11) * P1;
    h ^= h >> 33;
    h *= P2;
    h ^= h >> 29;
    h *= P3;
    h ^= h >> 32;
    return h;
}

std::uint64_t fold_hashes(std::span<const unsigned long long> values) {
    std::uint64_t h = 0x243F6A8885A308D3ull;
    for (const unsigned long long v : values) h = hash64(&v, sizeof v, h);
    return h;
}

std::shared_ptr<std::byte> allocate_host_memory(std::size_t bytes, bool pinned) {
    const std::size_t n = std::max<std::size_t>(bytes, 1);
    if (pinned) {
        void* p = nullptr;
        const cudaError_t status = cudaHostAlloc(&p, n, cudaHostAllocDefault);
        if (status != cudaSuccess) {
            cudaGetLastError();
            throw std::runtime_error(std::string("cudaHostAlloc failed: ") + cudaGetErrorString(status));
        }
        return std::shared_ptr<std::byte>(static_cast<std::byte*>(p), [](std::byte* q) { cudaFreeHost(q); });
    }
    return std::shared_ptr<std::byte>(new std::byte[n], [](std::byte* q) { delete[] q; });
}

namespace {

std::uint64_t table_and_payload_hash(const std::vector<unsigned char>& table, const std::byte* payload, std::size_t payload_bytes) {
    std::uint64_t h = hash64(table.data(), table.size(), 0);
    // The payload in chunks so that a reader that streams the file gets the same value.
    for (std::size_t at = 0; at < payload_bytes; at += kChunk) {
        h = hash64(payload + at, std::min(kChunk, payload_bytes - at), h);
    }
    return h;
}

} // namespace

void write_snapshot_file(const HostSnapshot::Data& d, const std::filesystem::path& path) {
    const SnapshotIdentity& id = d.identity;
    const std::uint64_t kv_total = d.total_bytes - d.state_bytes;
    Out header(kHeaderBytes);
    std::memcpy(header.bytes().data(), kMagic, 8);
    header.u32(8, kVersion);
    header.u32(12, static_cast<std::uint32_t>(kHeaderBytes));
    header.u64(16, id.geometry_fp);
    header.u64(24, id.weight_fp);
    header.u64(32, id.compute_fp);
    header.u32(40, id.precision);
    header.u32(44, id.state_format);
    header.u32(48, id.kv_format);
    header.u32(52, id.dn_layers);
    header.u32(56, id.attn_layers);
    header.u64(64, id.conv_tail_elems);
    header.u64(72, id.recurrent_elems);
    header.u64(80, id.kv_unit_bytes);
    header.u64(88, d.tokens);
    header.u64(96, d.end_position);
    header.u32(104, static_cast<std::uint32_t>(d.blocks.size()));
    header.u32(108, d.state_bytes != 0 ? 1 : 0);
    header.u64(112, d.state_bytes);
    header.u64(120, kv_total);
    header.u64(128, d.total_bytes);
    header.u64(160, hash64(header.bytes().data(), 160));

    Out table(d.blocks.size() * kBlockEntryBytes);
    for (std::size_t i = 0; i < d.blocks.size(); ++i) {
        const HostBlock& b = d.blocks[i];
        table.u32(i * kBlockEntryBytes, b.tokens);
        table.u64(i * kBlockEntryBytes + 8, b.start_position);
        table.u64(i * kBlockEntryBytes + 16, b.kv_bytes);
    }
    const std::uint64_t hash = table_and_payload_hash(table.bytes(), d.memory.get(), d.total_bytes);

    std::filesystem::path temp = path;
    temp += ".partial";
    bool ok = false;
    {
        std::ofstream out(temp, std::ios::binary | std::ios::trunc);
        if (!out) throw SnapshotFileError("snapshot file: cannot create " + temp.string());
        out.write(reinterpret_cast<const char*>(header.bytes().data()), static_cast<std::streamsize>(header.bytes().size()));
        out.write(reinterpret_cast<const char*>(table.bytes().data()), static_cast<std::streamsize>(table.bytes().size()));
        for (std::size_t at = 0; at < d.total_bytes; at += kChunk) {
            out.write(reinterpret_cast<const char*>(d.memory.get() + at), static_cast<std::streamsize>(std::min(kChunk, d.total_bytes - at)));
        }
        unsigned char trailer[kTrailerBytes];
        std::memcpy(trailer, &hash, 8);
        std::memcpy(trailer + 8, kEnd, 8);
        out.write(reinterpret_cast<const char*>(trailer), kTrailerBytes);
        out.flush();
        ok = static_cast<bool>(out);
    }
    std::error_code ec;
    if (ok) {
        std::filesystem::rename(temp, path, ec);
        if (ec) ok = false;
    }
    if (!ok) {
        std::filesystem::remove(temp, ec);
        throw SnapshotFileError("snapshot file: writing " + path.string() + " failed");
    }
}

HostSnapshot read_snapshot_file(const std::filesystem::path& path, const SnapshotIdentity& want, bool pinned) {
    std::error_code ec;
    if (!std::filesystem::is_regular_file(path, ec)) refuse(path.string() + " is not a regular file");
    const std::uint64_t file_size = std::filesystem::file_size(path, ec);
    if (ec) refuse("cannot get the size of " + path.string());
    if (file_size < kHeaderBytes + kBlockEntryBytes + kTrailerBytes) refuse("the file is too short to be a snapshot (" + std::to_string(file_size) + " bytes)");

    std::ifstream in(path, std::ios::binary);
    if (!in) refuse("cannot open " + path.string());
    const auto read_exact = [&](void* dst, std::size_t bytes) {
        in.read(static_cast<char*>(dst), static_cast<std::streamsize>(bytes));
        if (static_cast<std::size_t>(in.gcount()) != bytes) refuse("the file ended early or could not be read");
    };

    unsigned char h[kHeaderBytes];
    read_exact(h, kHeaderBytes);
    if (std::memcmp(h, kMagic, 8) != 0) refuse("bad magic: not a Sepal snapshot");
    if (get64(h, 160) != hash64(h, 160)) refuse("the header is corrupted (hash mismatch)");
    if (get32(h, 8) != kVersion) refuse("version " + std::to_string(get32(h, 8)) + ", this build reads version " + std::to_string(kVersion));
    if (get32(h, 12) != kHeaderBytes) refuse("unexpected header size");
    if (get32(h, 60) != 0 || get64(h, 136) != 0 || get64(h, 144) != 0 || get64(h, 152) != 0) refuse("reserved header fields are not zero");

    // Identity: what the bytes mean must be what this engine means by them.
    const auto expect = [&](const char* name, std::uint64_t got, std::uint64_t wanted) {
        if (got != wanted) refuse(std::string(name) + " differs (file " + std::to_string(got) + ", engine " + std::to_string(wanted) + ")");
    };
    if (get64(h, 16) != want.geometry_fp) refuse("the snapshot was made for another model geometry");
    if (get64(h, 24) != want.weight_fp) refuse("the snapshot was made with other weights");
    expect("precision", get32(h, 40), want.precision);
    expect("state format (EngineOptions::half_state)", get32(h, 44), want.state_format);
    expect("key/value format", get32(h, 48), want.kv_format);
    expect("DeltaNet layer count", get32(h, 52), want.dn_layers);
    expect("attention layer count", get32(h, 56), want.attn_layers);
    expect("conv-tail elements per layer", get64(h, 64), want.conv_tail_elems);
    expect("recurrent-state elements per layer", get64(h, 72), want.recurrent_elems);
    expect("key/value bytes per token and layer", get64(h, 80), want.kv_unit_bytes);
    if (get64(h, 32) != want.compute_fp) refuse("the snapshot was made with other compute options (multiply path, accumulation, gemm_rows ...)");

    const std::uint64_t tokens = get64(h, 88);
    const std::uint64_t end_position = get64(h, 96);
    const std::uint32_t blocks = get32(h, 104);
    const std::uint32_t has_state = get32(h, 108);
    const std::uint64_t state_bytes = get64(h, 112);
    const std::uint64_t kv_total = get64(h, 120);
    const std::uint64_t payload_bytes = get64(h, 128);
    if (blocks == 0 || blocks > kMaxBlocks) refuse("implausible block count " + std::to_string(blocks));
    if (has_state != (want.dn_layers != 0 ? 1u : 0u)) refuse("the state flag does not match the model");
    if (state_bytes != want.state_bytes()) refuse("state size " + std::to_string(state_bytes) + " does not match the model's " + std::to_string(want.state_bytes()));
    std::uint64_t payload_check = 0;
    if (add_overflows(state_bytes, kv_total, payload_check) || payload_check != payload_bytes) refuse("payload size is not state + keys and values");

    // The file must be exactly header + table + payload + trailer, before anything is allocated.
    std::uint64_t table_bytes = 0, expected_size = 0;
    if (mul_overflows(blocks, kBlockEntryBytes, table_bytes)) refuse("block table size overflows");
    if (add_overflows(kHeaderBytes, table_bytes, expected_size) || add_overflows(expected_size, payload_bytes, expected_size) ||
        add_overflows(expected_size, kTrailerBytes, expected_size)) refuse("file size overflows");
    if (expected_size != file_size) {
        refuse(std::string(file_size < expected_size ? "truncated" : "longer than its header says") + " (" + std::to_string(file_size) + " bytes, header says " +
               std::to_string(expected_size) + ")");
    }

    std::vector<unsigned char> table(static_cast<std::size_t>(table_bytes));
    read_exact(table.data(), table.size());
    auto data = std::make_shared<HostSnapshot::Data>();
    data->identity = want;
    data->tokens = tokens;
    data->end_position = end_position;
    data->state_bytes = static_cast<std::size_t>(state_bytes);
    data->total_bytes = static_cast<std::size_t>(payload_bytes);
    data->pinned = pinned;
    std::uint64_t token_sum = 0, kv_sum = 0, last_end = 0;
    std::size_t offset = data->state_bytes;
    for (std::uint32_t i = 0; i < blocks; ++i) {
        const unsigned char* e = table.data() + static_cast<std::size_t>(i) * kBlockEntryBytes;
        HostBlock b;
        b.tokens = get32(e, 0);
        b.start_position = get64(e, 8);
        b.kv_bytes = get64(e, 16);
        if (b.tokens == 0) refuse("block " + std::to_string(i) + " has no tokens");
        if (get32(e, 4) != 0) refuse("reserved block field is not zero");
        if (b.start_position > 0xFFFFFFFEull || b.start_position + b.tokens > 0xFFFFFFFEull) refuse("block " + std::to_string(i) + " has positions beyond 32 bits");
        if (b.kv_bytes != want.kv_bytes(b.tokens)) refuse("block " + std::to_string(i) + " has a key/value size that does not match its tokens");
        if (add_overflows(token_sum, b.tokens, token_sum) || add_overflows(kv_sum, b.kv_bytes, kv_sum)) refuse("sums overflow");
        last_end = b.start_position + b.tokens;
        b.kv_offset = offset;
        offset += static_cast<std::size_t>(b.kv_bytes);
        data->blocks.push_back(b);
    }
    if (token_sum != tokens) refuse("the blocks' tokens do not add up to the header's token count");
    if (kv_sum != kv_total) refuse("the blocks' key/value bytes do not add up to the header's");
    if (last_end != end_position) refuse("the end position does not follow from the last block");
    if (offset != data->total_bytes) refuse("internal size mismatch");

    try {
        data->memory = allocate_host_memory(data->total_bytes, pinned);
    } catch (const std::exception& e) {
        refuse(std::string("not enough host memory for the snapshot: ") + e.what());
    }
    std::uint64_t hash = hash64(table.data(), table.size(), 0);
    for (std::size_t at = 0; at < data->total_bytes; at += kChunk) {
        const std::size_t n = std::min(kChunk, data->total_bytes - at);
        read_exact(data->memory.get() + at, n);
        hash = hash64(data->memory.get() + at, n, hash);
    }
    unsigned char trailer[kTrailerBytes];
    read_exact(trailer, kTrailerBytes);
    if (std::memcmp(trailer + 8, kEnd, 8) != 0) refuse("bad end marker");
    if (get64(trailer, 0) != hash) refuse("the payload is corrupted (hash mismatch)");
    return HostSnapshot(std::move(data));
}

} // namespace sepal::cuda
