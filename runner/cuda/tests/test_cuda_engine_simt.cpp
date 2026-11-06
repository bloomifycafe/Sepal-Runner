// The float32 SIMT path of the GPU engine (EngineOptions::weight_storage: BF16, Int8Group, and float32 weights with
// multiply_path Invariant): the path of the Pascal profiles gpu-bf16 and gpu-w8g. Runs on any card from sm_61 up.
//
//   engine_simt   against the CPU reference executor over random request DAGs, on the weights as the engine stores them
//                 (bfloat16-rounded or int8-group-dequantised on the host with the engine's formulas), so that the comparison
//                 measures the arithmetic only; bit-invariance (alone, in a batch, natural and strict mode, a branch from a
//                 stored snapshot against the same branch in one call); refused options; weight memory per storage type.
//
// The geometry is a small one the SIMT kernels take (hidden 256, intermediate 384, head dims 64 and 32), random weights.

#include <cmath>
#include <cstring>
#include <map>

#include "engine_test_util.hpp"

using namespace sepal;
using namespace engine_test;
using sepal::cuda::CudaEngine;
using sepal::cuda::DeviceSnapshot;
using sepal::cuda::EngineOptions;
using sepal::cuda::EngineRow;
using sepal::cuda::EngineRowResult;
using sepal::cuda::MultiplyPath;
using sepal::cuda::ParentRow;
using sepal::cuda::Precision;
using sepal::cuda::WeightStorage;

namespace {

ModelConfig simt_config() {
    ModelConfig c = tiny_test_config();
    c.name = "simt-small";
    c.num_layers = 8;
    c.full_attention_interval = 4;
    c.hidden_size = 256;
    c.intermediate_size = 384;
    c.attn_heads = 4;
    c.attn_kv_heads = 2;
    c.attn_head_dim = 64;
    c.dn_key_heads = 2;
    c.dn_value_heads = 4;
    c.dn_key_head_dim = 32;
    c.dn_value_head_dim = 32;
    c.vocab_size = 96;
    return c;
}

EngineOptions simt_options(WeightStorage storage, std::size_t gemm_rows = 0) {
    EngineOptions o;
    o.precision = Precision::F32;
    o.weight_storage = storage;
    if (storage == WeightStorage::Float32) o.multiply_path = MultiplyPath::Invariant;
    o.gemm_rows = gemm_rows;
    o.memory_budget_bytes = std::size_t{256} << 20;
    o.max_call_tokens = 2048;
    o.max_call_rows = 256;
    o.state_slots = 64;
    return o;
}

float bf16_round(float f) {
    std::uint32_t bits;
    std::memcpy(&bits, &f, 4);
    bits = (bits + 0x7FFFu + ((bits >> 16) & 1u)) & 0xFFFF0000u;
    float r;
    std::memcpy(&r, &bits, 4);
    return r;
}

// The engine's int8 group quantiser on the host (same IEEE float operations), dequantised: s * q, rounded once.
void quantize_dequantize(BasicTensor<float>& t, std::size_t k) {
    const std::size_t rows = t.size() / k;
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t g = 0; g < k / 128; ++g) {
            float* w = t.data() + r * k + g * 128;
            float m = 0.0f;
            for (std::size_t i = 0; i < 128; ++i) m = std::max(m, std::fabs(w[i]));
            const float s = m / 127.0f;
            for (std::size_t i = 0; i < 128; ++i) {
                float q = 0.0f;
                if (s > 0.0f) q = std::min(std::max(std::nearbyint(w[i] / s), -127.0f), 127.0f);
                w[i] = q * s;
            }
        }
    }
}

bool is_matrix(const std::string& name) {
    for (const char* m : {"q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight", "in_proj_qkv.weight", "in_proj_z.weight",
                          "in_proj_b.weight", "in_proj_a.weight", "out_proj.weight", "gate_proj.weight", "up_proj.weight", "down_proj.weight"}) {
        if (name.size() >= std::strlen(m) && name.compare(name.size() - std::strlen(m), std::string::npos, m) == 0) return true;
    }
    return false;
}

// The weights as the engine of `storage` holds them (float32 values).
Weights<float> stored_weights(const ModelConfig& cfg, const Weights<float>& w, WeightStorage storage) {
    Weights<float>::Map map;
    for (const auto& [name, shape] : weight_shapes(cfg)) {
        BasicTensor<float> t = w[name];
        const bool matrix = is_matrix(name);
        const bool embedding = name == embedding_name;
        if (storage == WeightStorage::BF16 && (matrix || embedding)) {
            for (std::size_t i = 0; i < t.size(); ++i) t[i] = bf16_round(t[i]);
        } else if (storage == WeightStorage::Int8Group) {
            const bool ba = name.find("in_proj_b.weight") != std::string::npos || name.find("in_proj_a.weight") != std::string::npos;
            if (embedding || (matrix && ba)) {
                for (std::size_t i = 0; i < t.size(); ++i) t[i] = bf16_round(t[i]);
            } else if (matrix) {
                quantize_dequantize(t, shape.back());
            }
        }
        map.emplace(name, std::move(t));
    }
    return Weights<float>::adopt(cfg, std::move(map));
}

const Weights<float>& weights_of(std::uint64_t seed) {
    static std::map<std::uint64_t, Weights<float>> cache;
    auto it = cache.find(seed);
    if (it == cache.end()) it = cache.emplace(seed, random_weights<float>(simt_config(), seed)).first;
    return it->second;
}

std::vector<EngineRowResult> run(CudaEngine& engine, const std::vector<EngineRow>& rows) { return engine.run(std::span<const EngineRow>(rows)); }

const char* name_of(WeightStorage s) { return s == WeightStorage::Float32 ? "float32" : s == WeightStorage::BF16 ? "bf16" : "int8-group"; }

} // namespace

SEPAL_TEST(engine_simt, fuzz_against_reference) {
    const ModelConfig cfg = simt_config();
    const std::size_t hidden = ref::dim(cfg.hidden_size);
    std::size_t calls = 60;
    if (const auto env = environment("SEPAL_FUZZ_CALLS")) calls = std::max<std::size_t>(std::stoul(*env) / 4, 6);
    for (const WeightStorage storage : {WeightStorage::Float32, WeightStorage::BF16, WeightStorage::Int8Group}) {
        double worst = 0.0;
        std::size_t rows_total = 0, tokens_total = 0;
        for (std::uint64_t w = 0; w < 2; ++w) {
            const Weights<float>& original = weights_of(300 + w);
            const Weights<float> held = stored_weights(cfg, original, storage);
            CudaEngine engine(cfg, original, simt_options(storage));
            CHECK(engine.options().multiply_path == MultiplyPath::Invariant);
            std::vector<Stored> stored;
            Rng rng(3000 + w);
            for (std::size_t call = 0; call < calls / 2; ++call) {
                FuzzOptions fo;
                fo.vocab = cfg.vocab_size;
                const Request request = make_request(rng, stored, fo);
                const auto want = run_reference(cfg, held, request.reference);
                const auto got = run(engine, request.engine);
                for (std::size_t r = 0; r < got.size(); ++r) {
                    const std::size_t count = request.engine[r].readout_offsets.size() * hidden;
                    CHECK_EQ(got[r].hidden.size(), count);
                    if (got[r].hidden.size() != count) continue;
                    worst = std::max(worst, max_abs_diff(got[r].hidden.data(), want[r].hidden.data(), count));
                    CHECK(all_finite(got[r].hidden));
                    ++rows_total;
                    tokens_total += request.engine[r].token_ids.size();
                }
                for (std::size_t r = 0; r < got.size(); ++r) {
                    if (!request.engine[r].keep_snapshot) continue;
                    CHECK(got[r].snapshot.valid());
                    // The float64 reference is not run here: its snapshot stays empty (only the float32 reference runs).
                    stored.push_back({got[r].snapshot, want[r].snapshot, nullptr, request.end_position[r], request.tokens[r]});
                }
                while (stored.size() > 8) stored.erase(stored.begin() + static_cast<std::ptrdiff_t>(rng.below(stored.size())));
            }
        }
        std::printf("  [measured] %-10s %zu rows, %zu tokens: largest |engine - reference| over the read-out hidden states %.3e\n", name_of(storage),
                    rows_total, tokens_total, worst);
        CHECK(worst < 1e-4);
    }
}

SEPAL_TEST(engine_simt, chunked_deltanet_against_reference) {
    // DeltaNetMode::Chunked on the float32 path: the float32 chunked kernels (WY form) instead of the scan.
    const ModelConfig cfg = simt_config();
    const std::size_t hidden = ref::dim(cfg.hidden_size);
    const Weights<float>& original = weights_of(320);
    const Weights<float> held = stored_weights(cfg, original, WeightStorage::BF16);
    EngineOptions o = simt_options(WeightStorage::BF16);
    o.deltanet = sepal::cuda::DeltaNetMode::Chunked;
    CudaEngine chunked(cfg, original, o);
    CudaEngine scan(cfg, original, simt_options(WeightStorage::BF16));
    std::vector<Stored> stored;
    Rng rng(3200);
    double worst = 0.0, worst_scan = 0.0;
    for (std::size_t call = 0; call < 20; ++call) {
        FuzzOptions fo;
        fo.vocab = cfg.vocab_size;
        fo.allow_store = false;
        const Request request = make_request(rng, stored, fo);
        const auto want = run_reference(cfg, held, request.reference);
        const auto got = run(chunked, request.engine);
        const auto base = run(scan, request.engine);
        for (std::size_t r = 0; r < got.size(); ++r) {
            const std::size_t count = request.engine[r].readout_offsets.size() * hidden;
            CHECK_EQ(got[r].hidden.size(), count);
            if (got[r].hidden.size() != count) continue;
            worst = std::max(worst, max_abs_diff(got[r].hidden.data(), want[r].hidden.data(), count));
            worst_scan = std::max(worst_scan, max_abs_diff(got[r].hidden.data(), base[r].hidden.data(), count));
        }
    }
    std::printf("  [measured] chunked float32 DeltaNet: largest |engine - reference| %.3e, |chunked - scan| %.3e\n", worst, worst_scan);
    CHECK(worst < 1e-4);
    CHECK(worst_scan < 1e-4);
}

SEPAL_TEST(engine_simt, rows_are_bit_invariant) {
    const ModelConfig cfg = simt_config();
    for (const WeightStorage storage : {WeightStorage::BF16, WeightStorage::Int8Group}) {
        const Weights<float>& w = weights_of(410);
        CudaEngine natural(cfg, w, simt_options(storage));
        CudaEngine strict(cfg, w, simt_options(storage, 64));
        Rng rng(77);
        std::vector<EngineRow> rows;
        for (std::size_t r = 0; r < 6; ++r) {
            EngineRow row;
            row.token_ids.resize(5 + 37 * r);
            for (auto& t : row.token_ids) t = static_cast<std::uint32_t>(rng.below(cfg.vocab_size));
            row.readout_offsets = {0, row.token_ids.size() / 2, row.token_ids.size() - 1};
            rows.push_back(row);
        }
        const auto batch = run(natural, rows);
        const auto batch_strict = run(strict, rows);
        bool same = true;
        for (std::size_t r = 0; r < rows.size(); ++r) {
            const auto alone = run(natural, {rows[r]});
            same = same && same_bits(alone[0].hidden, batch[r].hidden) && same_bits(batch_strict[r].hidden, batch[r].hidden);
        }
        // Reversed order: every row keeps its bits.
        std::vector<EngineRow> reversed(rows.rbegin(), rows.rend());
        const auto back = run(natural, reversed);
        for (std::size_t r = 0; r < rows.size(); ++r) same = same && same_bits(back[rows.size() - 1 - r].hidden, batch[r].hidden);
        CHECK(same);

        // A branch from a stored snapshot against the same branch run with its parent in one call.
        EngineRow parent = rows[3];
        parent.keep_snapshot = true;
        parent.readout_offsets.clear();
        EngineRow child = rows[2];
        const auto first = run(natural, {parent});
        child.parent = first[0].snapshot;
        const auto from_store = run(natural, {child});
        EngineRow child_in_call = rows[2];
        child_in_call.parent = ParentRow{0};
        EngineRow parent_in_call = rows[3];
        parent_in_call.readout_offsets.clear();
        const auto together = run(natural, {parent_in_call, child_in_call});
        CHECK(same_bits(from_store[0].hidden, together[1].hidden));
    }
}

SEPAL_TEST(engine_simt, host_embedding_gives_the_same_bits) {
    const ModelConfig cfg = simt_config();
    const Weights<float>& w = weights_of(450);
    for (const WeightStorage storage : {WeightStorage::BF16, WeightStorage::Int8Group}) {
        EngineOptions on_device = simt_options(storage);
        EngineOptions on_host = simt_options(storage);
        // The host function hands the table's values: the bfloat16 rounding the device table would hold.
        const BasicTensor<float>& table = w[std::string(embedding_name)];
        const std::size_t hidden = ref::dim(cfg.hidden_size);
        on_host.host_embedding = [&table, hidden](std::uint32_t token, float* out) {
            for (std::size_t i = 0; i < hidden; ++i) out[i] = bf16_round(table[token * hidden + i]);
        };
        CudaEngine a(cfg, w, on_device);
        CudaEngine b(cfg, w, on_host);
        CHECK(b.memory().weights_bytes + cfg.vocab_size * hidden * 2 <= a.memory().weights_bytes + 256);
        Rng rng(91);
        std::vector<EngineRow> rows;
        for (std::size_t r = 0; r < 3; ++r) {
            EngineRow row;
            row.token_ids.resize(40 + 30 * r);
            for (auto& t : row.token_ids) t = static_cast<std::uint32_t>(rng.below(cfg.vocab_size));
            row.readout_offsets = {row.token_ids.size() - 1, 0};
            rows.push_back(row);
        }
        const auto x = run(a, rows);
        const auto y = run(b, rows);
        bool same = true;
        for (std::size_t r = 0; r < rows.size(); ++r) same = same && same_bits(x[r].hidden, y[r].hidden);
        CHECK(same);
    }
}

SEPAL_TEST(engine_simt, options_and_memory) {
    const ModelConfig cfg = simt_config();
    const Weights<float>& w = weights_of(500);
    // Narrow weights need the float32 path; the tiny geometry is not one the SIMT kernels take.
    EngineOptions f16 = simt_options(WeightStorage::BF16);
    f16.precision = Precision::F16;
    CHECK_THROWS(CudaEngine(cfg, w, f16), std::invalid_argument);
    CHECK_THROWS(CudaEngine(tiny_test_config(), random_weights<float>(tiny_test_config(), 1), simt_options(WeightStorage::BF16)), std::invalid_argument);
    std::size_t bytes[3] = {0, 0, 0};
    int i = 0;
    for (const WeightStorage storage : {WeightStorage::Float32, WeightStorage::BF16, WeightStorage::Int8Group}) {
        CudaEngine engine(cfg, w, simt_options(storage));
        bytes[i++] = engine.memory().weights_bytes;
    }
    std::printf("  [measured] weight bytes: float32 %zu, bf16 %zu, int8-group %zu\n", bytes[0], bytes[1], bytes[2]);
    CHECK(bytes[1] < bytes[0]);
    CHECK(bytes[2] < bytes[1]);
}
