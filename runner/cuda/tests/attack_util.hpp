// Helpers of the adversarial engine tests (test_cuda_engine_attack*.cpp, test_cuda_engine_soak.cpp): engines of the
// tiny geometry in every option combination, a CPU-reference "world" that follows the snapshots an engine hands out,
// and the small utilities the properties are written with.
#pragma once

#include <cuda_runtime.h>

#include <map>
#include <set>
#include <sstream>
#include <thread>

#include "engine_test_util.hpp"

namespace attack {

using namespace sepal;
using namespace engine_test;
using sepal::cuda::BudgetExceeded;
using sepal::cuda::CudaEngine;
using sepal::cuda::DeviceSnapshot;
using sepal::cuda::EngineMemory;
using sepal::cuda::EngineOptions;
using sepal::cuda::EngineRow;
using sepal::cuda::EngineRowResult;
using sepal::cuda::ParentRow;
using sepal::cuda::Precision;

/// Every knob of the engine the tests turn.
struct Knobs {
    Precision precision = Precision::F32;
    std::size_t gemm_rows = 0;
    bool prune = true;
    bool gate_up = true; // fp16_gate_up
    std::size_t max_tokens = 4096;
    std::size_t max_rows = 256;
    std::size_t slots = 96;
    std::size_t budget_mib = 192;
};

inline EngineOptions options_of(const Knobs& k) {
    EngineOptions o;
    o.precision = k.precision;
    o.gemm_rows = k.gemm_rows;
    o.prune_last_layer = k.prune;
    o.fp16_gate_up = k.gate_up;
    o.memory_budget_bytes = k.budget_mib << 20;
    o.max_call_tokens = k.max_tokens;
    o.max_call_rows = k.max_rows;
    o.state_slots = k.slots;
    return o;
}

inline const char* name_of(Precision p) { return p == Precision::F16 ? "float16" : "float32"; }

inline std::string describe(const Knobs& k) {
    std::ostringstream s;
    s << name_of(k.precision) << (k.gemm_rows == 0 ? " natural" : " strict " + std::to_string(k.gemm_rows)) << (k.prune ? "" : " no-prune")
      << (k.precision == Precision::F16 && !k.gate_up ? " f32-gate-up" : "");
    return s.str();
}

inline const Weights<float>& tiny_weights(std::uint64_t seed) {
    static std::map<std::uint64_t, Weights<float>> cache;
    auto it = cache.find(seed);
    if (it == cache.end()) it = cache.emplace(seed, random_weights<float>(tiny_test_config(), seed)).first;
    return it->second;
}

inline std::unique_ptr<CudaEngine> make_engine(const Knobs& k, std::uint64_t weight_seed = 31) {
    return std::make_unique<CudaEngine>(tiny_test_config(), tiny_weights(weight_seed), options_of(k));
}

inline std::vector<EngineRowResult> run(CudaEngine& engine, const std::vector<EngineRow>& rows) {
    return engine.run(std::span<const EngineRow>(rows));
}

inline std::vector<std::uint32_t> tokens(Rng& rng, std::size_t n, std::size_t vocab = 64) {
    std::vector<std::uint32_t> t(n);
    for (auto& v : t) v = static_cast<std::uint32_t>(rng.below(vocab));
    return t;
}

inline EngineRow make_row(std::vector<std::uint32_t> t, std::vector<std::size_t> readouts = {}, bool keep = false) {
    EngineRow r;
    r.token_ids = std::move(t);
    r.readout_offsets = std::move(readouts);
    r.keep_snapshot = keep;
    return r;
}

inline EngineRow child_of(EngineRow r, std::size_t parent_index) {
    r.parent = ParentRow{parent_index};
    return r;
}

inline EngineRow child_of(EngineRow r, const DeviceSnapshot& s) {
    r.parent = s;
    return r;
}

inline std::vector<std::size_t> all_offsets(std::size_t n) {
    std::vector<std::size_t> o(n);
    for (std::size_t i = 0; i < n; ++i) o[i] = i;
    return o;
}

/// Bits of every float of every row's hidden state, for comparing two runs byte for byte.
inline bool same_hidden(const std::vector<EngineRowResult>& a, const std::vector<EngineRowResult>& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (!same_bits(a[i].hidden, b[i].hidden)) return false;
    }
    return true;
}

inline double hidden_distance(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return 1e30;
    return a.empty() ? 0.0 : max_abs_diff(a.data(), b.data(), a.size());
}

/// FNV-1a over bytes, for digests that two processes can compare.
inline std::uint64_t fnv(std::uint64_t h, const void* data, std::size_t bytes) {
    const auto* p = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < bytes; ++i) {
        h ^= p[i];
        h *= 0x100000001B3ull;
    }
    return h;
}

inline std::uint64_t digest(const std::vector<EngineRowResult>& results, std::uint64_t h = 0xCBF29CE484222325ull) {
    for (const EngineRowResult& r : results) {
        const std::uint64_t n = r.hidden.size();
        h = fnv(h, &n, sizeof n);
        h = fnv(h, r.hidden.data(), r.hidden.size() * sizeof(float));
    }
    return h;
}

/// The CPU float32 reference together with the snapshots an engine handed out: lets a test replay any request on
/// both and keep following the chains it builds across calls.
class World {
public:
    explicit World(std::uint64_t weight_seed = 31) : cfg_(tiny_test_config()), weights_(tiny_weights(weight_seed)) {}
    World(const ModelConfig& cfg, const Weights<float>& weights) : cfg_(cfg), weights_(weights) {}

    const ModelConfig& config() const { return cfg_; }
    std::size_t hidden() const { return ref::dim(cfg_.hidden_size); }

    /// The reference form of a request. Snapshot parents must have been registered by `run_checked`.
    std::vector<ref::Row<float>> convert(const std::vector<EngineRow>& rows) const {
        std::vector<ref::Row<float>> out(rows.size());
        std::vector<std::uint64_t> end(rows.size(), 0);
        // Start positions default to the parent's end, which for a ParentRow depends on the parent's own start.
        std::vector<bool> done(rows.size(), false);
        const auto resolve = [&](auto&& self, std::size_t i) -> void {
            if (done[i]) return;
            const EngineRow& row = rows[i];
            ref::Row<float>& r = out[i];
            r.token_ids = row.token_ids;
            r.readout_offsets = row.readout_offsets;
            std::uint64_t parent_end = 0;
            if (const auto* p = std::get_if<ParentRow>(&row.parent)) {
                self(self, p->index);
                r.parent = ref::ParentRow{p->index};
                parent_end = end[p->index];
            } else if (const auto* s = std::get_if<DeviceSnapshot>(&row.parent)) {
                const auto it = snapshots_.find(s->data().get());
                if (it == snapshots_.end()) throw std::logic_error("World: a parent snapshot the reference does not know");
                r.parent = it->second.first;
                parent_end = it->second.second;
            }
            r.start_position = row.start_position.value_or(parent_end);
            end[i] = r.start_position + row.token_ids.size();
            done[i] = true;
        };
        for (std::size_t i = 0; i < rows.size(); ++i) resolve(resolve, i);
        return out;
    }

    /// Runs the reference alone.
    std::vector<ref::RowResult<float>> reference(const std::vector<EngineRow>& rows) const { return run_reference(cfg_, weights_, convert(rows)); }

    /// Remembers the reference snapshots of the rows that kept one on the device. The device snapshots are not kept alive
    /// by the world: an entry whose device snapshot is gone is dropped.
    void adopt(const std::vector<EngineRow>& rows, const std::vector<EngineRowResult>& got, const std::vector<ref::RowResult<float>>& want) {
        std::vector<ref::Row<float>> converted = convert(rows);
        for (std::size_t i = 0; i < rows.size(); ++i) {
            if (!rows[i].keep_snapshot) continue;
            const auto& data = got[i].snapshot.data();
            snapshots_[data.get()] = {want[i].snapshot, converted[i].start_position + rows[i].token_ids.size()};
            alive_[data.get()] = data;
        }
        if (snapshots_.size() > 64) {
            for (auto it = alive_.begin(); it != alive_.end();) {
                if (it->second.expired()) {
                    snapshots_.erase(it->first);
                    it = alive_.erase(it);
                } else {
                    ++it;
                }
            }
        }
    }

    /// Forgets snapshots.
    void forget_all() {
        snapshots_.clear();
        alive_.clear();
    }

    struct Checked {
        std::vector<EngineRowResult> got;
        double worst = 0.0; // largest |engine - reference| over every readout
    };

    /// Runs the engine and the reference on the request and compares every readout.
    Checked run_checked(CudaEngine& engine, const std::vector<EngineRow>& rows, bool adopt_kept = true) {
        Checked c;
        c.got = run(engine, rows);
        const auto want = reference(rows);
        for (std::size_t i = 0; i < rows.size(); ++i) {
            const std::size_t count = rows[i].readout_offsets.size() * hidden();
            CHECK_EQ(c.got[i].hidden.size(), count);
            if (c.got[i].hidden.size() != count) continue;
            CHECK(all_finite(c.got[i].hidden));
            if (count != 0) c.worst = std::max(c.worst, max_abs_diff(c.got[i].hidden.data(), want[i].hidden.data(), count));
        }
        if (adopt_kept) adopt(rows, c.got, want);
        return c;
    }

private:
    ModelConfig cfg_;
    const Weights<float>& weights_;
    std::map<const void*, std::pair<ref::SnapshotPtr<float>, std::uint64_t>> snapshots_;
    std::map<const void*, std::weak_ptr<const DeviceSnapshot::Data>> alive_; // tells whether a key still names a snapshot
};

/// Free device memory, for before/after comparisons.
inline std::size_t device_free() {
    std::size_t free_bytes = 0, total = 0;
    cudaDeviceSynchronize();
    cudaMemGetInfo(&free_bytes, &total);
    return free_bytes;
}

} // namespace attack
