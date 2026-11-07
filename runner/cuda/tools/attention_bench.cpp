// Benchmark of the attention kernels (kernels_attention.hpp): time per layer of the prepare launch (norm, RoPE,
// K/V store) and the attention launch (with the gate), for each path, on three request shapes at the attention
// geometry of a preset:
//
//   (a) 16 rows x 640 tokens, no context
//   (b) 1 row x 4096 tokens, no context
//   (c) 8 state rows x 640 tokens and 64 branch rows x 27 tokens, eight per state, in the same launch
//
// Each time is the minimum over the repetitions of a synchronised run (the GPU is shared, so the minimum is the
// figure that is least disturbed by other work). Inputs are random; this measures speed, not accuracy.
//
//   sepal-cuda-attention-bench [--presets 0.8b,2b] [--reps 20]

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

#include "kernels_attention.hpp"
#include "sepal/config.hpp"

namespace {

using namespace sepal::cuda;

struct Variant {
    AttnElem kv;
    AttentionPath path;
    const char* name;
    int f32 = -1; // TiledF32: the float32 kernel forced (F32Kernel), -1 = the environment's choice
};

struct RowSpec {
    std::uint32_t length;
    int parent; // index of the row whose keys and values are the context, or -1
};

struct Scenario {
    const char* name;
    std::vector<RowSpec> rows;
    std::uint32_t links = 0;  // the first row continues a chain of this many one-token snapshot links (rows[0].parent is -1)
    bool indexed = false;     // the rows' segments carry `before` (binary search instead of a scan for a key's segment)
    std::uint32_t context = 0; // the first row continues one external segment of this many tokens (random keys and values)
};

std::vector<Scenario> scenarios() {
    std::vector<Scenario> all;
    Scenario a{"(a) 16 x 640", {}};
    for (int i = 0; i < 16; ++i) a.rows.push_back({640, -1});
    all.push_back(a);
    all.push_back({"(b) 1 x 4096", {{4096, -1}}});
    Scenario c{"(c) 8 states x 640 + 64 branches x 27", {}};
    for (int s = 0; s < 8; ++s) c.rows.push_back({640, -1});
    for (int s = 0; s < 8; ++s) {
        for (int b = 0; b < 8; ++b) c.rows.push_back({27, s});
    }
    all.push_back(c);
    // One new token over a chain of 1,000 one-token snapshot links, found by a scan from the first segment or by binary search.
    all.push_back({"(d) 1 x 1 over 1000 links, scanned", {{1, -1}}, 1000, false});
    all.push_back({"(e) 1 x 1 over 1000 links, indexed", {{1, -1}}, 1000, true});
    return all;
}

// --f32: the Sepal-4B shapes of the Pascal profiles (prefill calls of up to 2,048 tokens over a growing cache).
std::vector<Scenario> f32_scenarios() {
    std::vector<Scenario> all;
    all.push_back({"(f) 1 x 1024, no context (t1k)", {{1024, -1}}});
    all.push_back({"(g) 1 x 2048 over 2048", {{2048, -1}}, 0, true, 2048});
    all.push_back({"(h) 1 x 2048 over 14336 (16k, last call)", {{2048, -1}}, 0, true, 14336});
    all.push_back({"(i) 1 x 200 over 16000", {{200, -1}}, 0, true, 16000});
    all.push_back({"(j) 1 x 1 over 16383 (read-out)", {{1, -1}}, 0, true, 16383});
    all.push_back({"(k) 4 x 512 over 4096 each", {{512, -1}, {512, -1}, {512, -1}, {512, -1}}, 0, true, 4096});
    return all;
}

std::uint64_t g_state = 12345;
float random_unit() {
    std::uint64_t z = (g_state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z ^= z >> 31;
    return static_cast<float>(z >> 40) * (1.0f / 16777216.0f) * 2.0f - 1.0f;
}

struct Timing {
    double prepare_ms = 0, attention_ms = 0, total_ms = 0;
    AttentionPath used = AttentionPath::Reference;
    double flops = 0; // causal attention arithmetic: 4 head_dim per (token, head, visible key)
};

Timing run(const sepal::ModelConfig& cfg, const Scenario& scenario, const Variant& variant, int reps) {
    const AttentionShape shape{static_cast<std::uint32_t>(cfg.attn_heads), static_cast<std::uint32_t>(cfg.attn_kv_heads),
                               static_cast<std::uint32_t>(cfg.attn_head_dim), static_cast<std::uint32_t>(cfg.attn_rotary_dims()),
                               cfg.rope_theta, cfg.rms_norm_eps};
    const std::size_t q_width = static_cast<std::size_t>(shape.heads) * shape.head_dim;
    const std::size_t kv_width = static_cast<std::size_t>(shape.kv_heads) * shape.head_dim;
    const std::size_t esize = variant.kv == AttnElem::F16 ? 2 : 4;

    std::vector<RowSpan> spans;
    std::uint32_t total = 0, max_len = 0;
    for (const RowSpec& r : scenario.rows) {
        spans.push_back({total, r.length});
        total += r.length;
        max_len = std::max(max_len, r.length);
    }
    // The multiply output [q with gate | k | v], as one matrix with a row stride.
    const std::size_t stride = 2 * q_width + 2 * kv_width;
    std::vector<float> mult(static_cast<std::size_t>(total) * stride);
    for (float& x : mult) x = random_unit() * 2.0f;
    DeviceBuffer<float> d_mult(mult);
    std::vector<float> weights(shape.head_dim);
    for (float& w : weights) w = random_unit() * 0.3f;
    DeviceBuffer<float> d_qn(weights), d_kn(weights);

    std::vector<DeviceBuffer<unsigned char>> keys, values;
    for (const RowSpec& r : scenario.rows) {
        keys.emplace_back(static_cast<std::size_t>(r.length) * kv_width * esize);
        values.emplace_back(static_cast<std::size_t>(r.length) * kv_width * esize);
    }
    std::vector<AttentionRowDesc> descs;
    std::vector<KvSegment> segments;
    // The links of a chain all point at one token's keys and values: the timing does not depend on where they live.
    DeviceBuffer<unsigned char> link_keys(kv_width * esize), link_values(kv_width * esize);
    link_keys.zero();
    link_values.zero();
    // The external context of every row (scenario.context tokens of random keys and values; zeros for binary16).
    DeviceBuffer<unsigned char> ctx_keys(std::max<std::size_t>(1, static_cast<std::size_t>(scenario.context) * kv_width * esize));
    DeviceBuffer<unsigned char> ctx_values(ctx_keys.size());
    if (esize == 4 && scenario.context != 0) {
        std::vector<float> r(static_cast<std::size_t>(scenario.context) * kv_width);
        for (float& x : r) x = random_unit();
        ctx_keys.upload(reinterpret_cast<const unsigned char*>(r.data()), ctx_keys.size());
        for (float& x : r) x = random_unit();
        ctx_values.upload(reinterpret_cast<const unsigned char*>(r.data()), ctx_values.size());
    } else {
        ctx_keys.zero();
        ctx_values.zero();
    }
    std::uint32_t max_keys = 0;
    double flops = 0;
    for (std::size_t i = 0; i < scenario.rows.size(); ++i) {
        const RowSpec& r = scenario.rows[i];
        const std::uint32_t parent_tokens = r.parent >= 0 ? scenario.rows[static_cast<std::size_t>(r.parent)].length : 0;
        const std::uint32_t links = i == 0 ? scenario.links : 0;
        const std::uint32_t ext = scenario.context != 0 ? 1u : 0u;
        descs.push_back({static_cast<std::uint32_t>(segments.size()), (r.parent >= 0 ? 2u : 1u) + links + ext,
                         parent_tokens + links + scenario.context, scenario.indexed ? 1u : 0u});
        std::uint32_t before = 0;
        if (ext != 0) {
            segments.push_back({ctx_keys.data(), ctx_values.data(), scenario.context, before});
            before += scenario.context;
        }
        for (std::uint32_t l = 0; l < links; ++l) segments.push_back({link_keys.data(), link_values.data(), 1, before++});
        if (r.parent >= 0) {
            const std::size_t p = static_cast<std::size_t>(r.parent);
            segments.push_back({keys[p].data(), values[p].data(), scenario.rows[p].length, before});
            before += scenario.rows[p].length;
        }
        segments.push_back({keys[i].data(), values[i].data(), r.length, before});
        max_keys = std::max(max_keys, before + r.length);
        for (std::uint32_t t = 0; t < r.length; ++t) flops += 4.0 * shape.head_dim * shape.heads * (before + t + 1);
    }
    check_attention_rows(spans, descs, segments);
    DeviceBuffer<RowSpan> d_spans(spans);
    DeviceBuffer<AttentionRowDesc> d_descs(descs);
    DeviceBuffer<KvSegment> d_segments(segments);
    const AttentionRows table{d_spans.data(), d_descs.data(), d_segments.data(), nullptr,
                              static_cast<std::uint32_t>(scenario.rows.size()), max_len, max_keys};

    DeviceBuffer<unsigned char> d_q(static_cast<std::size_t>(total) * q_width * esize);
    DeviceBuffer<unsigned char> d_out(static_cast<std::size_t>(total) * q_width * esize);
    const FloatBand q_gate{d_mult.data(), stride};
    const FloatBand k_band{d_mult.data() + 2 * q_width, stride};
    const FloatBand v_band{d_mult.data() + 2 * q_width + kv_width, stride};

    // The default (blocking) stream, so that the uploads above have landed before the first kernel.
    cudaStream_t stream = nullptr;
    cudaEvent_t e0, e1, e2;
    check(cudaEventCreate(&e0), "cudaEventCreate");
    check(cudaEventCreate(&e1), "cudaEventCreate");
    check(cudaEventCreate(&e2), "cudaEventCreate");
    Timing best{1e30, 1e30, 1e30, AttentionPath::Reference, flops};
    if (variant.f32 >= 0) set_f32_kernel(static_cast<F32Kernel>(variant.f32));
    for (int rep = -2; rep < reps; ++rep) { // two warm-up repetitions
        check(cudaEventRecord(e0, stream), "cudaEventRecord");
        launch_attention_prepare(shape, table, variant.kv, q_gate, k_band, v_band, d_qn.data(), d_kn.data(),
                                 {d_q.data(), q_width}, stream);
        check(cudaEventRecord(e1, stream), "cudaEventRecord");
        best.used = launch_attention(shape, table, variant.kv, variant.path, {d_q.data(), q_width}, q_gate,
                                     {d_out.data(), q_width}, variant.kv, stream);
        check(cudaEventRecord(e2, stream), "cudaEventRecord");
        check(cudaEventSynchronize(e2), "cudaEventSynchronize");
        if (rep < 0) continue;
        float p = 0, a = 0, t = 0;
        check(cudaEventElapsedTime(&p, e0, e1), "cudaEventElapsedTime");
        check(cudaEventElapsedTime(&a, e1, e2), "cudaEventElapsedTime");
        check(cudaEventElapsedTime(&t, e0, e2), "cudaEventElapsedTime");
        best.prepare_ms = std::min(best.prepare_ms, static_cast<double>(p));
        best.attention_ms = std::min(best.attention_ms, static_cast<double>(a));
        best.total_ms = std::min(best.total_ms, static_cast<double>(t));
    }
    if (variant.f32 >= 0) reset_f32_kernel();
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cudaEventDestroy(e2);
    return best;
}

} // namespace

int main(int argc, char** argv) {
    std::vector<std::string> presets = {"0.8b", "2b"};
    int reps = 20;
    int rounds = 3;
    bool f32 = false;
    for (int i = 1; i < argc; ++i) {
        const bool has_value = i + 1 < argc;
        if (std::strcmp(argv[i], "--f32") == 0) {
            f32 = true;
        } else if (std::strcmp(argv[i], "--presets") == 0 && has_value) {
            presets.clear();
            std::string list = argv[++i];
            for (std::size_t pos = 0; pos <= list.size();) {
                const std::size_t comma = list.find(',', pos);
                presets.push_back(list.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos));
                pos = comma == std::string::npos ? list.size() + 1 : comma + 1;
            }
        } else if (std::strcmp(argv[i], "--reps") == 0 && has_value) {
            reps = std::max(1, std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "--rounds") == 0 && has_value) {
            rounds = std::max(1, std::atoi(argv[++i]));
        } else {
            std::fprintf(stderr, "usage: sepal-cuda-attention-bench [--presets 0.8b,2b] [--reps 20] [--f32 [--rounds 3]]\n"
                                 "  --f32: the Sepal-4B float32 shapes (default preset 4b), the two float32 kernels interleaved over\n"
                                 "         rounds (each round runs every variant once with --reps repetitions); TFLOP/s of the causal arithmetic\n");
            return 2;
        }
    }
    try {
        if (f32) {
            if (presets == std::vector<std::string>{"0.8b", "2b"}) presets = {"4b"};
            const std::vector<Variant> variants = {{AttnElem::F32, AttentionPath::TiledF32, "f32 v1", 0},
                                                   {AttnElem::F32, AttentionPath::TiledF32, "f32 v2", 1}};
            std::printf("float32 kernels, %d interleaved rounds of %d synchronised repetitions: ms per layer (minimum; every round's in brackets)\n",
                        rounds, reps);
            std::printf("%-6s %-42s %-8s %10s %10s  %s\n", "preset", "scenario", "kernel", "attention", "TFLOP/s", "rounds");
            for (const std::string& name : presets) {
                const auto cfg = sepal::find_preset(name);
                if (!cfg) {
                    std::fprintf(stderr, "unknown preset %s\n", name.c_str());
                    return 2;
                }
                for (const Scenario& scenario : f32_scenarios()) {
                    std::vector<std::vector<double>> ms(variants.size());
                    double flops = 0;
                    for (int round = 0; round < rounds; ++round) {
                        for (std::size_t v = 0; v < variants.size(); ++v) {
                            const Timing t = run(*cfg, scenario, variants[v], reps);
                            ms[v].push_back(t.attention_ms);
                            flops = t.flops;
                        }
                    }
                    for (std::size_t v = 0; v < variants.size(); ++v) {
                        const double best = *std::min_element(ms[v].begin(), ms[v].end());
                        std::string all;
                        for (const double x : ms[v]) all += (all.empty() ? "" : " ") + std::to_string(x).substr(0, 7);
                        std::printf("%-6s %-42s %-8s %10.3f %10.3f  [%s]\n", name.c_str(), scenario.name, variants[v].name, best,
                                    flops / (best * 1e-3) / 1e12, all.c_str());
                        std::fflush(stdout);
                    }
                }
            }
            return 0;
        }
        const std::vector<Variant> variants = {{AttnElem::F32, AttentionPath::Reference, "f32 plain"},
                                               {AttnElem::F16, AttentionPath::Reference, "f16 plain"},
                                               {AttnElem::F16, AttentionPath::Tiled, "f16 tiled"}};
        std::printf("minimum of %d synchronised repetitions, ms per layer (prepare = norm + RoPE + K/V store)\n", reps);
        std::printf("%-6s %-42s %-10s %10s %10s %10s\n", "preset", "scenario", "path", "prepare", "attention", "total");
        for (const std::string& name : presets) {
            const auto cfg = sepal::find_preset(name);
            if (!cfg) {
                std::fprintf(stderr, "unknown preset %s\n", name.c_str());
                return 2;
            }
            for (const Scenario& scenario : scenarios()) {
                for (const Variant& variant : variants) {
                    const Timing t = run(*cfg, scenario, variant, reps);
                    std::printf("%-6s %-42s %-10s %10.3f %10.3f %10.3f\n", name.c_str(), scenario.name, variant.name,
                                t.prepare_ms, t.attention_ms, t.total_ms);
                    std::fflush(stdout);
                }
            }
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}