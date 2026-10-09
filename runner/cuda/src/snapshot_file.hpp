// Host side of the snapshot tiers: the in-memory form of a snapshot (HostSnapshot::Data) and the file format. No CUDA
// calls except the allocation of pinned memory. The device side (copying to and from the pools) is in cuda_engine.cpp.
//
// File format, version 1. Little-endian; sizes in bytes.
//
//   header, 168 bytes
//      0  8  magic "SEPALSNP"
//      8  4  version (1)
//     12  4  header size (168)
//     16  8  geometry fingerprint   (every field of the ModelConfig that sizes the state or the keys and values)
//     24  8  weight fingerprint     (hash of the weights as they sit on the device)
//     32  8  compute fingerprint    (options that change the bits of the state: multiply path, accumulation, gemm_rows, ...)
//     40  4  precision              (0 float16 path, 1 float32 path)
//     44  4  state format           (0 float32, 1 binary16)
//     48  4  key/value format       (0: the operand type of the precision)
//     52  4  DeltaNet layers
//     56  4  attention layers
//     60  4  reserved (0)
//     64  8  conv-tail elements per DeltaNet layer
//     72  8  recurrent-state elements per DeltaNet layer
//     80  8  bytes of the keys (or values) of one token in one attention layer
//     88  8  tokens of the chain
//     96  8  end position
//    104  4  number of blocks
//    108  4  has state (0 or 1)
//    112  8  state bytes
//    120  8  key/value bytes of all blocks
//    128  8  payload bytes (state + keys and values)
//    136 24  reserved (0)
//    160  8  hash of bytes 0 .. 159
//   block table: number of blocks x 24 bytes: tokens u32, reserved u32 (0), start position u64, key/value bytes u64
//   payload: the state (for each DeltaNet layer: conv tail, recurrent state), then for each block, oldest first, for each
//            attention layer: the keys of its tokens, the values of its tokens (no padding)
//   trailer, 16 bytes: hash of (block table + payload), end marker "SEPALEND"
//
// The hash is a 64-bit multiply-rotate hash of our own (hash64): it catches truncation and corruption, it does not
// authenticate.
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <vector>

#include "sepal/cuda/cuda_engine.hpp"

namespace sepal::cuda {

/// What a snapshot's bytes mean: everything that has to agree between the engine that wrote them and the one that reads them.
struct SnapshotIdentity {
    std::uint64_t geometry_fp = 0;
    std::uint64_t weight_fp = 0;
    std::uint64_t compute_fp = 0;
    std::uint32_t precision = 0;
    std::uint32_t state_format = 0;
    std::uint32_t kv_format = 0;
    std::uint32_t dn_layers = 0;
    std::uint32_t attn_layers = 0;
    std::uint64_t conv_tail_elems = 0;
    std::uint64_t recurrent_elems = 0;
    std::uint64_t kv_unit_bytes = 0;

    bool operator==(const SnapshotIdentity&) const = default;
    std::size_t state_elem_bytes() const { return state_format == 1 ? 2 : 4; }
    /// Bytes of the DeltaNet state of one snapshot (0 without DeltaNet layers).
    std::uint64_t state_bytes() const {
        return static_cast<std::uint64_t>(dn_layers) * (conv_tail_elems + recurrent_elems) * state_elem_bytes();
    }
    /// Bytes of the keys and values of `tokens` tokens in all attention layers.
    std::uint64_t kv_bytes(std::uint64_t tokens) const { return static_cast<std::uint64_t>(attn_layers) * 2 * tokens * kv_unit_bytes; }
};

struct HostBlock {
    std::uint32_t tokens = 0;
    std::uint64_t start_position = 0;
    std::uint64_t kv_bytes = 0;  // compact, as in the payload
    std::size_t kv_offset = 0;   // into Data::memory
};

struct HostSnapshot::Data {
    SnapshotIdentity identity;
    std::uint64_t tokens = 0;
    std::uint64_t end_position = 0;
    std::vector<HostBlock> blocks; // oldest first
    std::shared_ptr<std::byte> memory; // [state][block 0 keys and values][block 1 ...]
    std::size_t state_bytes = 0;
    std::size_t total_bytes = 0;
    bool pinned = false;
};

/// 64-bit hash of a byte range (see the file comment). `seed` chains calls.
std::uint64_t hash64(const void* data, std::size_t bytes, std::uint64_t seed = 0);
/// Folds the per-thread values of the weight hash kernel into one number.
std::uint64_t fold_hashes(std::span<const unsigned long long> values);

/// Memory for a snapshot's bytes: pageable (operator new) or pinned (cudaHostAlloc); freed when the last owner lets go.
/// Throws std::bad_alloc / std::runtime_error when it cannot be had.
std::shared_ptr<std::byte> allocate_host_memory(std::size_t bytes, bool pinned);

/// Writes `data` to `path` (temporary file, then rename). SnapshotFileError on failure.
void write_snapshot_file(const HostSnapshot::Data& data, const std::filesystem::path& path);

/// Reads and validates a snapshot file against the identity of the reading engine. SnapshotFileError on anything
/// wrong. `pinned` chooses the kind of host memory. Allocates at most the size of the file.
HostSnapshot read_snapshot_file(const std::filesystem::path& path, const SnapshotIdentity& expect, bool pinned);

} // namespace sepal::cuda
