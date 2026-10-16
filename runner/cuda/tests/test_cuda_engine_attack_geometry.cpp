// The engine on geometries other than the tiny test one and the two real ones: every shape of the layer pattern (all
// attention, one attention layer, interval 2), convolution kernels of 1 to 8 taps, multi-head, grouped and multi-query
// attention, head widths 8 to 256, DeltaNet head widths 4 to 128, and widths that are not multiples of 8 (a half-precision
// matrix with an odd row stride is a place for vectorised loads to go wrong). Float32 engine against the CPU reference;
// float16 engine against float32. A geometry the engine cannot take must be refused when the engine is created.
//
// Suite attack_geometry.

#include "attack_util.hpp"

using namespace attack;

namespace {

struct Variant {
    std::string name;
    ModelConfig cfg;
};

ModelConfig base() {
    ModelConfig c = tiny_test_config();
    return c;
}

std::vector<Variant> variants() {
    std::vector<Variant> v;
    const auto add = [&](const std::string& name, auto&& edit) {
        ModelConfig c = base();
        c.name = name;
        edit(c);
        v.push_back({name, c});
    };
    add("all attention (interval 1)", [](ModelConfig& c) { c.num_layers = 3; c.full_attention_interval = 1; });
    add("one attention layer, three DeltaNet (interval 4, 4 layers)", [](ModelConfig& c) { c.num_layers = 4; c.full_attention_interval = 4; });
    add("interval 2, 6 layers", [](ModelConfig& c) { c.num_layers = 6; c.full_attention_interval = 2; });
    add("interval 6, 6 layers", [](ModelConfig& c) { c.num_layers = 6; c.full_attention_interval = 6; });
    add("single layer (attention only)", [](ModelConfig& c) { c.num_layers = 1; c.full_attention_interval = 1; });
    for (const std::uint64_t k : {1U, 2U, 3U, 5U, 8U}) {
        add("conv_kernel " + std::to_string(k), [k](ModelConfig& c) { c.conv_kernel = k; });
    }
    add("multi-head attention (kv heads = heads = 4)", [](ModelConfig& c) { c.attn_kv_heads = 4; });
    add("multi-query attention (1 kv head, 4 heads)", [](ModelConfig& c) { c.attn_kv_heads = 1; });
    add("8 heads / 2 kv heads", [](ModelConfig& c) { c.attn_heads = 8; c.attn_kv_heads = 2; });
    for (const std::uint64_t d : {8U, 32U, 64U, 128U, 256U}) {
        add("attention head dim " + std::to_string(d), [d](ModelConfig& c) { c.attn_head_dim = d; c.attn_heads = 2; c.attn_kv_heads = 1; });
    }
    add("attention head dim 256, rotary factor 1", [](ModelConfig& c) { c.attn_head_dim = 256; c.attn_heads = 2; c.attn_kv_heads = 1; c.partial_rotary_factor = 1.0; });
    add("attention head dim 64, rotary factor 0.5", [](ModelConfig& c) { c.attn_head_dim = 64; c.partial_rotary_factor = 0.5; });
    for (const std::uint64_t d : {4U, 16U, 32U, 64U, 128U}) {
        add("DeltaNet key dim " + std::to_string(d), [d](ModelConfig& c) { c.dn_key_head_dim = d; });
        add("DeltaNet value dim " + std::to_string(d), [d](ModelConfig& c) { c.dn_value_head_dim = d; });
    }
    add("DeltaNet 128/128 with 2 heads", [](ModelConfig& c) { c.dn_key_head_dim = 128; c.dn_value_head_dim = 128; c.dn_key_heads = 2; c.dn_value_heads = 2; });
    add("DeltaNet 1 key head, 4 value heads", [](ModelConfig& c) { c.dn_key_heads = 1; c.dn_value_heads = 4; });
    add("DeltaNet 4 key heads, 4 value heads", [](ModelConfig& c) { c.dn_key_heads = 4; c.dn_value_heads = 4; });
    add("hidden 20", [](ModelConfig& c) { c.hidden_size = 20; });
    add("hidden 33", [](ModelConfig& c) { c.hidden_size = 33; });
    add("hidden 64, intermediate 128", [](ModelConfig& c) { c.hidden_size = 64; c.intermediate_size = 128; });
    add("intermediate 33", [](ModelConfig& c) { c.intermediate_size = 33; });
    add("intermediate 7", [](ModelConfig& c) { c.intermediate_size = 7; });
    add("vocabulary 1000", [](ModelConfig& c) { c.vocab_size = 1000; });
    add("rope theta 10000", [](ModelConfig& c) { c.rope_theta = 10000.0; });
    return v;
}

struct Outcome {
    bool refused = false;
    std::string why;
    double f32_vs_ref = 0.0;
    double f16_vs_f32 = 0.0;
};

Outcome try_variant(const Variant& var, std::size_t gemm_rows, bool prune) {
    Outcome out;
    const Weights<float> weights = random_weights<float>(var.cfg, 77);
    Knobs k;
    k.max_tokens = 1024;
    k.max_rows = 32;
    k.slots = 24;
    k.budget_mib = 96;
    k.gemm_rows = gemm_rows;
    k.prune = prune;
    std::unique_ptr<CudaEngine> e32, e16;
    try {
        e32 = std::make_unique<CudaEngine>(var.cfg, weights, options_of(k));
        k.precision = Precision::F16;
        e16 = std::make_unique<CudaEngine>(var.cfg, weights, options_of(k));
    } catch (const std::exception& e) {
        out.refused = true;
        out.why = e.what();
        return out;
    }
    World world(var.cfg, weights);
    Rng rng(404 + gemm_rows);
    const std::size_t vocab = static_cast<std::size_t>(var.cfg.vocab_size);
    for (int call = 0; call < 6; ++call) {
        std::vector<EngineRow> rows;
        EngineRow state = make_row(tokens(rng, 1 + rng.below(70), vocab), {}, true);
        rows.push_back(state);
        for (std::size_t b = 0, n = 1 + rng.below(3); b < n; ++b) {
            const std::size_t len = 1 + rng.below(9);
            EngineRow r = make_row(tokens(rng, len, vocab), {len - 1, 0}, rng.chance(30));
            r.parent = ParentRow{0};
            rows.push_back(r);
        }
        rows.push_back(make_row(tokens(rng, 1 + rng.below(20), vocab), {0}, true)); // an unrelated row
        const World::Checked c = world.run_checked(*e32, rows);
        out.f32_vs_ref = std::max(out.f32_vs_ref, c.worst);
        const auto g16 = run(*e16, rows);
        for (std::size_t i = 0; i < rows.size(); ++i) {
            CHECK(all_finite(g16[i].hidden));
            out.f16_vs_f32 = std::max(out.f16_vs_f32, hidden_distance(g16[i].hidden, c.got[i].hidden));
        }
        // And a later call continuing the kept snapshots, with two branches.
        std::vector<EngineRow> next;
        for (std::size_t i = 0; i < rows.size(); ++i) {
            if (!rows[i].keep_snapshot) continue;
            for (int b = 0; b < 2; ++b) {
                EngineRow r = make_row(tokens(rng, 1 + rng.below(5), vocab), {0});
                r.parent = c.got[i].snapshot;
                next.push_back(r);
            }
        }
        if (!next.empty()) {
            const World::Checked d = world.run_checked(*e32, next, false);
            out.f32_vs_ref = std::max(out.f32_vs_ref, d.worst);
        }
        world.forget_all();
    }
    return out;
}

} // namespace

SEPAL_TEST(attack_geometry, other_geometries_agree_with_the_reference_or_are_refused_when_the_engine_is_made) {
    std::size_t ok = 0, refused = 0;
    for (const Variant& var : variants()) {
        for (const auto& [g, prune] : {std::pair<std::size_t, bool>{0, true}, {16, true}, {0, false}, {16, false}}) {
            Outcome o;
            try {
                o = try_variant(var, g, prune);
            } catch (const std::exception& e) {
                std::cout << "  [" << var.name << ", gemm_rows " << g << (prune ? "" : ", no prune") << "] THREW AT RUN: " << e.what() << "\n";
                CHECK(false);
                continue;
            }
            if (o.refused) {
                ++refused;
                std::cout << "  [" << var.name << ", gemm_rows " << g << (prune ? "" : ", no prune") << "] refused at creation: " << o.why.substr(0, 160) << "\n";
                continue;
            }
            ++ok;
            std::cout << "  [" << var.name << ", gemm_rows " << g << (prune ? "" : ", no prune") << "] float32 vs reference " << std::scientific << std::setprecision(2) << o.f32_vs_ref << ", float16 vs float32 " << o.f16_vs_f32 << std::fixed
                      << "\n";
            CHECK(o.f32_vs_ref <= 2e-3);
            CHECK(o.f16_vs_f32 <= 1.0);
        }
    }
    std::cout << "  " << ok << " geometry runs agreed, " << refused << " were refused at creation\n";
}

// Strict mode promises that a row's bits do not depend on the other rows of the call. Last-layer pruning changes which
// multiplies run (and, in the pruned layer, their widths) depending on how many rows are read out, so it must not change a
// single bit. The tiny geometry cannot show a breach (its multiplies are narrow); the geometries below are wider.
SEPAL_TEST(attack_geometry, last_layer_pruning_changes_no_bit_in_strict_mode_on_wide_geometries) {
    struct Wide {
        std::string name;
        ModelConfig cfg;
    };
    std::vector<Wide> wide;
    {
        ModelConfig c = tiny_test_config();
        c.name = "mid";
        c.num_layers = 4;
        c.hidden_size = 256;
        c.intermediate_size = 768;
        c.attn_heads = 8;
        c.attn_kv_heads = 2;
        c.attn_head_dim = 64;
        c.dn_key_heads = 4;
        c.dn_value_heads = 8;
        c.dn_key_head_dim = 32;
        c.dn_value_head_dim = 32;
        wide.push_back({"hidden 256, 8 heads of 64", c});
        c.name = "wide";
        c.hidden_size = 512;
        c.intermediate_size = 1792;
        c.attn_head_dim = 256;
        c.attn_heads = 8;
        c.dn_key_heads = 8;
        c.dn_value_heads = 8;
        c.dn_key_head_dim = 64;
        c.dn_value_head_dim = 64;
        wide.push_back({"hidden 512, 8 heads of 256 (0.8B-like attention)", c});
    }
    for (const Wide& w : wide) {
        const Weights<float> weights = random_weights<float>(w.cfg, 5);
        for (const Precision precision : {Precision::F32, Precision::F16}) {
            for (const std::size_t g : {std::size_t{32}, std::size_t{128}}) {
                Knobs on;
                on.precision = precision;
                on.gemm_rows = g;
                on.max_tokens = 2048;
                on.max_rows = 64;
                on.slots = 24;
                on.budget_mib = 360;
                Knobs off = on;
                off.prune = false;
                CudaEngine pruned(w.cfg, weights, options_of(on)), full(w.cfg, weights, options_of(off));
                Rng rng(g + 3);
                const std::size_t vocab = static_cast<std::size_t>(w.cfg.vocab_size);
                std::size_t same = 0, total = 0;
                DeviceSnapshot sp, sf;
                for (int call = 0; call < 24; ++call) {
                    // A state and a few branches (readouts at the ends), sometimes continuing the stored state of an earlier call:
                    // so the number of read-out rows and of tokens varies from call to call.
                    std::vector<EngineRow> rows;
                    const std::size_t n_state = 20 + rng.below(300);
                    rows.push_back(make_row(tokens(rng, n_state, vocab), {}, true));
                    for (std::size_t b = 0, n = 1 + rng.below(6); b < n; ++b) {
                        const std::size_t len = 1 + rng.below(30);
                        rows.push_back(child_of(make_row(tokens(rng, len, vocab), {len - 1}, b == 0), 0));
                    }
                    std::vector<EngineRow> rows_p = rows, rows_f = rows;
                    if (sp.valid() && rng.chance(50)) {
                        rows_p[0].parent = sp;
                        rows_f[0].parent = sf;
                        rows_p[0].readout_offsets = {0};
                        rows_f[0].readout_offsets = {0};
                    }
                    const auto rp = run(pruned, rows_p), rf = run(full, rows_f);
                    for (std::size_t i = 0; i < rows.size(); ++i) {
                        ++total;
                        same += same_bits(rp[i].hidden, rf[i].hidden) ? 1U : 0U;
                    }
                    sp = rp[1].snapshot;
                    sf = rf[1].snapshot;
                }
                std::cout << "  " << w.name << ", " << name_of(precision) << " strict " << g << ": " << same << " of " << total << " rows bit-identical, pruned vs full last layer\n";
                CHECK_EQ(same, total);
            }
        }
    }
}
