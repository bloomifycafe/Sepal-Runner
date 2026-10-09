// Adversarial tests of the GPU engine on the tiny geometry (random weights): shapes the random fuzz of
// test_cuda_engine.cpp does not reach, and the engine's invariances stated in cuda_engine.hpp, each tried in every
// option combination (precision, natural / strict, prune_last_layer, fp16_gate_up).
//
// Suites: attack_shapes (rows and chains of unusual shape, limits, positions), attack_invariance (what must not change a
// result, bit for bit in strict mode), attack_lifetime (snapshots used after other snapshots came and went).
// The float32 engine is checked against the CPU reference, the float16 engine against the float32 engine.

#include "attack_util.hpp"

using namespace attack;

namespace {

constexpr double kF32Bar = 5e-4;   // float32 engine vs CPU reference (random weights amplify rounding; fuzz saw 1.8e-4)
constexpr double kF16Bar = 0.4;    // float16 engine vs float32 engine on the tiny geometry (information, not accuracy)

std::vector<Knobs> f32_modes() {
    std::vector<Knobs> v;
    for (const std::size_t g : {std::size_t{0}, std::size_t{8}, std::size_t{24}}) {
        for (const bool prune : {true, false}) {
            Knobs k;
            k.gemm_rows = g;
            k.prune = prune;
            v.push_back(k);
        }
    }
    return v;
}

std::vector<Knobs> f16_modes() {
    std::vector<Knobs> v;
    for (const std::size_t g : {std::size_t{0}, std::size_t{16}}) {
        for (const bool prune : {true, false}) {
            for (const bool gate_up : {true, false}) {
                Knobs k;
                k.precision = Precision::F16;
                k.gemm_rows = g;
                k.prune = prune;
                k.gate_up = gate_up;
                v.push_back(k);
            }
        }
    }
    return v;
}

std::vector<Knobs> all_modes() {
    std::vector<Knobs> v = f32_modes();
    for (const Knobs& k : f16_modes()) v.push_back(k);
    return v;
}

Knobs strict_of(Precision p, std::size_t g, bool prune, bool gate_up) {
    Knobs k;
    k.precision = p;
    k.gemm_rows = g;
    k.prune = prune;
    k.gate_up = gate_up;
    return k;
}

std::vector<Knobs> strict_modes() {
    std::vector<Knobs> v;
    for (const bool prune : {true, false}) {
        v.push_back(strict_of(Precision::F32, 8, prune, true));
        v.push_back(strict_of(Precision::F32, 24, prune, true));
        for (const bool gate_up : {true, false}) {
            v.push_back(strict_of(Precision::F16, 8, prune, gate_up));
            v.push_back(strict_of(Precision::F16, 24, prune, gate_up));
        }
    }
    return v;
}

// Float32 engine against the reference, float16 engine against the float32 engine, on the same sequence of calls.
// `build(world_or_null)` is not used: the sequences below are written out where they are needed.

} // namespace

// ---- rows shorter than the convolution kernel, continuing snapshots that are themselves shorter ----

SEPAL_TEST(attack_shapes, rows_shorter_than_the_conv_kernel_continue_snapshots_shorter_than_it) {
    // conv_kernel is 4 in the tiny geometry: lengths 1..6 on each side of the boundary, three calls deep.
    double worst = 0.0;
    std::size_t cases = 0;
    for (const Knobs& k : f32_modes()) {
        auto engine = make_engine(k);
        World world;
        Rng rng(700 + k.gemm_rows + (k.prune ? 1 : 0));
        for (std::size_t a = 1; a <= 6; ++a) {
            for (std::size_t b = 1; b <= 6; ++b) {
                for (std::size_t c = 1; c <= 3; ++c) {
                    // Three calls: a root, a child, a grandchild; each keeps its snapshot and reads every token.
                    World::Checked r1 = world.run_checked(*engine, {make_row(tokens(rng, a), all_offsets(a), true)});
                    World::Checked r2 = world.run_checked(*engine, {child_of(make_row(tokens(rng, b), all_offsets(b), true), r1.got[0].snapshot)});
                    World::Checked r3 = world.run_checked(*engine, {child_of(make_row(tokens(rng, c), all_offsets(c), true), r2.got[0].snapshot)});
                    // And the same three as one call, rows chained by index.
                    std::vector<EngineRow> chain = {make_row(tokens(rng, a), all_offsets(a)), make_row(tokens(rng, b), all_offsets(b)),
                                                    make_row(tokens(rng, c), all_offsets(c))};
                    chain[1].parent = ParentRow{0};
                    chain[2].parent = ParentRow{1};
                    World::Checked r4 = world.run_checked(*engine, chain);
                    const double this_worst = std::max({r1.worst, r2.worst, r3.worst, r4.worst});
                    if (this_worst > 3e-4) {
                        // Who is right? The same chain in float64.
                        std::vector<ref::Row<double>> rows64(3);
                        for (std::size_t i = 0; i < 3; ++i) {
                            rows64[i].token_ids = chain[i].token_ids;
                            rows64[i].readout_offsets = chain[i].readout_offsets;
                            if (i > 0) rows64[i].parent = ref::ParentRow{i - 1};
                        }
                        rows64[1].start_position = a;
                        rows64[2].start_position = a + b;
                        const auto exact = run_reference64(tiny_test_config(), double_weights(tiny_test_config(), tiny_weights(31)), rows64);
                        const auto want32 = world.reference(chain);
                        const World::Checked again = world.run_checked(*engine, chain, false);
                        double e_engine = 0, e_ref = 0;
                        for (std::size_t i = 0; i < 3; ++i) {
                            for (std::size_t j = 0; j < again.got[i].hidden.size(); ++j) {
                                e_engine = std::max(e_engine, std::fabs(again.got[i].hidden[j] - exact[i].hidden.data()[j]));
                                e_ref = std::max(e_ref, std::fabs(static_cast<double>(want32[i].hidden.data()[j]) - exact[i].hidden.data()[j]));
                            }
                        }
                        std::cout << "    " << describe(k) << " lengths " << a << "," << b << "," << c << ": engine-ref32 " << this_worst << ", engine-exact " << e_engine
                                  << ", ref32-exact " << e_ref << "\n";
                    }
                    worst = std::max(worst, this_worst);
                    ++cases;
                    world.forget_all();
                }
            }
        }
    }
    std::cout << "  " << cases << " short-row chains (float32, " << f32_modes().size() << " modes): largest difference from the reference " << worst << "\n";
    // Rows this short through random weights amplify rounding: the engine and the float32 reference are equally far (6e-4)
    // from float64 where they differ by 1.2e-3, so the bar is the noise of the reference itself.
    CHECK(worst <= 2e-3);
}

SEPAL_TEST(attack_shapes, float16_short_chains_stay_near_float32) {
    Rng rng(41);
    double worst = 0.0;
    auto e32 = make_engine(Knobs{});
    for (const Knobs& k : f16_modes()) {
        auto e16 = make_engine(k);
        double mode_worst = 0.0;
        for (std::size_t a = 1; a <= 5; ++a) {
            for (std::size_t b = 1; b <= 5; ++b) {
                const auto ta = tokens(rng, a), tb = tokens(rng, b);
                double d = 0;
                for (CudaEngine* e : {e32.get(), e16.get()}) {
                    (void)e;
                }
                const auto r32a = run(*e32, {make_row(ta, all_offsets(a), true)});
                const auto r32b = run(*e32, {child_of(make_row(tb, all_offsets(b)), r32a[0].snapshot)});
                const auto r16a = run(*e16, {make_row(ta, all_offsets(a), true)});
                const auto r16b = run(*e16, {child_of(make_row(tb, all_offsets(b)), r16a[0].snapshot)});
                d = std::max(hidden_distance(r32a[0].hidden, r16a[0].hidden), hidden_distance(r32b[0].hidden, r16b[0].hidden));
                CHECK(all_finite(r16b[0].hidden));
                worst = std::max(worst, d);
                mode_worst = std::max(mode_worst, d);
            }
        }
        std::cout << "    " << describe(k) << ": " << mode_worst << "\n";
    }
    std::cout << "  float16 (8 modes) vs float32 on 1..5 + 1..5 token chains: largest difference " << worst << "\n";
    CHECK(worst <= kF16Bar);
}

// ---- a chain built one token at a time against the same tokens in one row ----

namespace {

struct ChainResult {
    double chain_vs_oneshot = 0.0;   // engine chain vs engine one-shot, worst over positions
    double chain_vs_reference = 0.0; // only filled in float32
    double oneshot_vs_reference = 0.0;
    std::size_t positions = 0;
    std::size_t exact_positions = 0; // positions at which the two agree to the bit
};

ChainResult chain_vs_oneshot(const Knobs& k, std::size_t total, std::size_t max_piece, std::uint64_t seed, bool release_parents) {
    auto engine = make_engine(k);
    World world;
    Rng rng(seed);
    const auto all = tokens(rng, total);
    ChainResult out;
    // One shot.
    const auto one = run(*engine, {make_row(all, all_offsets(total))});
    // The chain: pieces of 1..max_piece tokens, each continuing the previous snapshot.
    std::vector<float> chained(total * world.hidden());
    DeviceSnapshot prev;
    std::size_t at = 0;
    std::vector<EngineRow> calls_rows; // for the reference, replayed below
    while (at < total) {
        const std::size_t len = std::min(total - at, max_piece == 1 ? std::size_t{1} : 1 + rng.below(max_piece));
        EngineRow row = make_row({all.begin() + static_cast<std::ptrdiff_t>(at), all.begin() + static_cast<std::ptrdiff_t>(at + len)}, all_offsets(len), true);
        if (prev.valid()) row.parent = prev;
        const auto r = run(*engine, {row});
        std::copy(r[0].hidden.begin(), r[0].hidden.end(), chained.begin() + static_cast<std::ptrdiff_t>(at * world.hidden()));
        if (release_parents) prev = r[0].snapshot;
        else prev = r[0].snapshot; // handles of older pieces are dropped either way once `prev` moves on
        at += len;
    }
    for (std::size_t p = 0; p < total; ++p) {
        const double d = max_abs_diff(chained.data() + p * world.hidden(), one[0].hidden.data() + p * world.hidden(), world.hidden());
        out.chain_vs_oneshot = std::max(out.chain_vs_oneshot, d);
        ++out.positions;
        if (d == 0.0) ++out.exact_positions;
    }
    if (k.precision == Precision::F32) {
        const auto want = world.reference({make_row(all, all_offsets(total))});
        for (std::size_t p = 0; p < total; ++p) {
            out.oneshot_vs_reference = std::max(out.oneshot_vs_reference,
                                                max_abs_diff(one[0].hidden.data() + p * world.hidden(), want[0].hidden.data() + p * world.hidden(), world.hidden()));
            out.chain_vs_reference = std::max(out.chain_vs_reference,
                                              max_abs_diff(chained.data() + p * world.hidden(), want[0].hidden.data() + p * world.hidden(), world.hidden()));
        }
    }
    return out;
}

} // namespace

SEPAL_TEST(attack_shapes, a_chain_one_token_at_a_time_equals_the_same_tokens_in_one_row) {
    for (const Knobs& k : {strict_of(Precision::F32, 0, true, true), strict_of(Precision::F32, 8, true, true), strict_of(Precision::F32, 8, false, true)}) {
        for (const std::size_t total : {std::size_t{3}, std::size_t{4}, std::size_t{5}, std::size_t{37}, std::size_t{150}}) {
            const ChainResult r = chain_vs_oneshot(k, total, 1, 1000 + total, true);
            std::cout << "  " << describe(k) << ", " << total << " tokens one call each vs one row: chain-vs-oneshot " << r.chain_vs_oneshot << " (" << r.exact_positions
                      << "/" << r.positions << " positions bit-identical); one-shot vs reference " << r.oneshot_vs_reference << ", chain vs reference "
                      << r.chain_vs_reference << "\n";
            CHECK(r.chain_vs_oneshot <= 5e-4);
            CHECK(r.chain_vs_reference <= kF32Bar);
            CHECK(r.oneshot_vs_reference <= kF32Bar);
        }
    }
    // Pieces of random size.
    for (const std::size_t g : {std::size_t{0}, std::size_t{8}}) {
        const ChainResult r = chain_vs_oneshot(strict_of(Precision::F32, g, true, true), 200, 9, 77, true);
        std::cout << "  float32 gemm_rows " << g << ", 200 tokens in pieces of 1..9: chain-vs-oneshot " << r.chain_vs_oneshot << " (" << r.exact_positions << "/" << r.positions
                  << " bit-identical)\n";
        CHECK(r.chain_vs_oneshot <= 5e-4);
    }
}

SEPAL_TEST(attack_shapes, float16_chain_one_token_at_a_time_stays_near_the_one_shot_row) {
    for (const Knobs& k : {strict_of(Precision::F16, 0, true, true), strict_of(Precision::F16, 16, true, false)}) {
        const ChainResult r = chain_vs_oneshot(k, 100, 1, 4242, true);
        std::cout << "  " << describe(k) << ", 100 tokens: chain-vs-oneshot " << r.chain_vs_oneshot << " (" << r.exact_positions << "/" << r.positions << " bit-identical)\n";
        CHECK(r.chain_vs_oneshot <= kF16Bar);
    }
}

// ---- waves and fan-out ----

SEPAL_TEST(attack_shapes, parent_chains_many_waves_deep_and_one_state_with_255_branches) {
    for (const Knobs& base : {strict_of(Precision::F32, 0, true, true), strict_of(Precision::F32, 8, true, true), strict_of(Precision::F32, 0, false, true)}) {
        Knobs k = base;
        k.max_rows = 512;
        k.max_tokens = 4096;
        k.slots = 400;
        k.budget_mib = 256;
        auto engine = make_engine(k);
        World world;
        Rng rng(5150);
        // A chain 60 rows deep (waves 60), rows of 1..6 tokens, readouts at random offsets, kept snapshots here and there.
        std::vector<EngineRow> chain;
        for (std::size_t i = 0; i < 60; ++i) {
            const std::size_t len = 1 + rng.below(6);
            EngineRow row = make_row(tokens(rng, len), {rng.below(len)}, rng.chance(20));
            if (i > 0) row.parent = ParentRow{i - 1};
            chain.push_back(std::move(row));
        }
        const World::Checked deep = world.run_checked(*engine, chain);
        // The same chain with the rows listed deepest first (parents come after children in the request).
        std::vector<EngineRow> reversed(chain.size());
        for (std::size_t i = 0; i < chain.size(); ++i) {
            reversed[chain.size() - 1 - i] = chain[i];
            if (auto* p = std::get_if<ParentRow>(&reversed[chain.size() - 1 - i].parent)) p->index = chain.size() - 1 - p->index;
            reversed[chain.size() - 1 - i].keep_snapshot = false;
        }
        const World::Checked backwards = world.run_checked(*engine, reversed, false);

        // One state, 255 branches of 1..8 tokens (the call has 256 rows).
        std::vector<EngineRow> fan = {make_row(tokens(rng, 40), {39}, true)};
        for (std::size_t b = 0; b < 255; ++b) {
            const std::size_t len = 1 + rng.below(8);
            fan.push_back(child_of(make_row(tokens(rng, len), {len - 1}), 0));
        }
        const World::Checked wide = world.run_checked(*engine, fan);
        // Deep and wide: every row of a 40-chain has a few leaves.
        std::vector<EngineRow> tree;
        for (std::size_t i = 0; i < 40; ++i) {
            EngineRow row = make_row(tokens(rng, 2 + rng.below(4)), {0});
            if (i > 0) row.parent = ParentRow{(i - 1) * 4};
            tree.push_back(row);
            for (std::size_t leaf = 0; leaf < 3; ++leaf) tree.push_back(child_of(make_row(tokens(rng, 1 + rng.below(3)), {0}), tree.size() - 1 - leaf));
        }
        // (the leaf's parent index above is arbitrary but valid; the chain spine is rows 0, 4, 8, ...)
        const World::Checked bushy = world.run_checked(*engine, tree);
        std::cout << "  " << describe(base) << ": 60 waves " << deep.worst << ", reversed " << backwards.worst << ", 255 branches " << wide.worst << ", 40 spine + leaves " << bushy.worst << "\n";
        CHECK(deep.worst <= kF32Bar);
        CHECK(backwards.worst <= kF32Bar);
        CHECK(wide.worst <= kF32Bar);
        CHECK(bushy.worst <= kF32Bar);
        // Branch rows continuing the state later, from the stored snapshot.
        std::vector<EngineRow> later;
        for (std::size_t b = 1; b < 256; b += 5) {
            EngineRow row = fan[b];
            row.parent = wide.got[0].snapshot;
            later.push_back(row);
        }
        const World::Checked from_store = world.run_checked(*engine, later);
        CHECK(from_store.worst <= kF32Bar);
        // Same bits as in the call that held the state (strict mode only).
        if (k.gemm_rows != 0) {
            for (std::size_t i = 0; i < later.size(); ++i) CHECK(same_bits(from_store.got[i].hidden, wide.got[1 + 5 * i].hidden));
        }
    }
}

// ---- the limits of a call ----

SEPAL_TEST(attack_shapes, a_call_at_exactly_the_limits_runs_and_one_over_is_refused_without_harm) {
    for (const std::size_t g : {std::size_t{0}, std::size_t{8}, std::size_t{24}, std::size_t{64}}) {
        Knobs k;
        k.gemm_rows = g;
        k.max_tokens = 500;
        k.max_rows = 12;
        auto engine = make_engine(k);
        auto fresh = make_engine(k);
        World world;
        Rng rng(31 + g);
        const std::vector<EngineRow> at_limit_one_row = {make_row(tokens(rng, 500), {0, 249, 499})};
        std::vector<EngineRow> at_limit_rows;
        std::size_t left = 500;
        for (std::size_t i = 0; i < 12; ++i) {
            const std::size_t len = i == 11 ? left : 41;
            left -= len;
            at_limit_rows.push_back(make_row(tokens(rng, len), {len - 1, 0}));
        }
        CHECK_EQ(left, std::size_t{0});
        // Every token read out, at the limit.
        const std::vector<EngineRow> all_read = {make_row(tokens(rng, 500), all_offsets(500))};
        double worst = 0.0;
        const std::vector<const std::vector<EngineRow>*> cases = {&at_limit_one_row, &at_limit_rows, &all_read};
        for (const auto* rows : cases) {
            const World::Checked c = world.run_checked(*engine, *rows, false);
            worst = std::max(worst, c.worst);
            // One token over, one row over.
            std::vector<EngineRow> over = *rows;
            over.back().token_ids.push_back(1);
            CHECK_THROWS((void)run(*engine, over), std::invalid_argument);
            const auto again = run(*engine, *rows);
            CHECK(same_hidden(again, c.got));
            if (g != 0) CHECK(same_hidden(run(*fresh, *rows), c.got));
        }
        std::vector<EngineRow> too_many_rows(13, make_row({1}, {0}));
        CHECK_THROWS((void)run(*engine, too_many_rows), std::invalid_argument);
        std::vector<EngineRow> twelve_rows(12, make_row({1}, {0}));
        CHECK_EQ(run(*engine, twelve_rows).size(), std::size_t{12});
        CHECK(worst <= kF32Bar);
        const EngineMemory m = engine->memory();
        CHECK_EQ(m.slots_in_use, std::size_t{0});
        CHECK_EQ(m.kv_in_use_bytes, std::size_t{0});
        std::cout << "  gemm_rows " << g << ": 500 tokens in one row, in 12 rows, and all 500 read out; one over refused; largest difference from the reference " << worst << "\n";
    }
}

// ---- positions ----

SEPAL_TEST(attack_shapes, start_positions_far_out_restarting_and_at_the_edge_of_the_range) {
    for (const Knobs& k : {strict_of(Precision::F32, 0, true, true), strict_of(Precision::F32, 8, false, true)}) {
        auto engine = make_engine(k);
        World world;
        Rng rng(606);
        std::vector<EngineRow> rows;
        for (const std::uint64_t start : {std::uint64_t{0}, std::uint64_t{1}, std::uint64_t{4095}, std::uint64_t{60000}, std::uint64_t{1000000}, std::uint64_t{100000000},
                                         std::uint64_t{4000000000ull}}) {
            EngineRow r = make_row(tokens(rng, 40), {0, 20, 39}, true);
            r.start_position = start;
            rows.push_back(r);
        }
        const World::Checked far = world.run_checked(*engine, rows);
        std::cout << "  " << describe(k) << ", start positions 0 .. 4000000000: largest difference from the reference " << far.worst << "\n";
        CHECK(far.worst <= kF32Bar);
        // A branch of a long state whose positions restart at 0, and one that jumps ahead; from a stored snapshot and in-call.
        std::vector<EngineRow> branches;
        EngineRow state = make_row(tokens(rng, 90), {}, true);
        state.start_position = 60000;
        branches.push_back(state);
        for (const std::optional<std::uint64_t> start : {std::optional<std::uint64_t>(0), std::optional<std::uint64_t>(60000), std::optional<std::uint64_t>(60090),
                                                        std::optional<std::uint64_t>(61000), std::optional<std::uint64_t>(5), std::optional<std::uint64_t>()}) {
            EngineRow b = child_of(make_row(tokens(rng, 7), {0, 6}), 0);
            b.start_position = start;
            branches.push_back(b);
        }
        const World::Checked restart = world.run_checked(*engine, branches);
        CHECK(restart.worst <= kF32Bar);
        std::vector<EngineRow> later;
        for (std::size_t i = 1; i < branches.size(); ++i) {
            EngineRow b = branches[i];
            b.parent = restart.got[0].snapshot;
            later.push_back(b);
        }
        const World::Checked from_store = world.run_checked(*engine, later);
        CHECK(from_store.worst <= kF32Bar);
        std::cout << "    restarting branches in-call " << restart.worst << ", from the stored state " << from_store.worst << "\n";
        // A snapshot's end_position follows an explicit start.
        CHECK_EQ(restart.got[0].snapshot.end_position(), std::uint64_t{60090});
    }
}

SEPAL_TEST(attack_shapes, start_positions_that_do_not_fit_the_device_are_refused) {
    auto engine = make_engine(Knobs{});
    Rng rng(7);
    const std::uint64_t max_u32 = 0xFFFFFFFFull;
    // Ends beyond 32 bits, by a little and by a lot, including a start so large that start + length wraps around 2^64.
    for (const std::uint64_t start : {max_u32, max_u32 + 1, std::uint64_t{1} << 40, std::uint64_t{0} - 1, std::uint64_t{0} - 2, std::uint64_t{0} - 3, std::uint64_t{1} << 63}) {
        EngineRow r = make_row(tokens(rng, 3), {0});
        r.start_position = start;
        bool refused = false;
        try {
            (void)run(*engine, {r});
        } catch (const std::invalid_argument&) {
            refused = true;
        }
        if (!refused) std::cout << "  NOT REFUSED: start_position " << start << " with 3 tokens\n";
        CHECK(refused);
    }
    // The engine is still fine and nothing is held.
    CHECK_EQ(run(*engine, {make_row({1, 2, 3}, {2})}).size(), std::size_t{1});
    const EngineMemory m = engine->memory();
    CHECK_EQ(m.kv_in_use_bytes, std::size_t{0});
    CHECK_EQ(m.slots_in_use, std::size_t{0});
}

// ---- what must not change a result ----

namespace {

struct Subject {
    std::string name;
    std::vector<EngineRow> rows; // ParentRow indices within the subject
    std::vector<std::size_t> watched;
};

// Subjects whose watched rows read out every offset (so single-offset runs can be compared), none of them continuing a stored snapshot.
std::vector<Subject> plain_subjects(Rng& rng) {
    std::vector<Subject> out(3);
    out[0].name = "one row of 37";
    out[0].rows = {make_row(tokens(rng, 37), all_offsets(37))};
    out[0].watched = {0};
    out[1].name = "state of 50 and two branches";
    out[1].rows = {make_row(tokens(rng, 50), all_offsets(50)), child_of(make_row(tokens(rng, 6), all_offsets(6)), 0), child_of(make_row(tokens(rng, 13), all_offsets(13)), 0)};
    out[1].watched = {0, 1, 2};
    out[2].name = "three rows deep, with lengths 1, 2, 3";
    out[2].rows = {make_row(tokens(rng, 1), {0}), child_of(make_row(tokens(rng, 2), {0, 1}), 0), child_of(make_row(tokens(rng, 3), {2, 0}), 1)};
    out[2].watched = {0, 1, 2};
    return out;
}

} // namespace

SEPAL_TEST(attack_invariance, readout_at_every_offset_equals_a_readout_at_one_offset_bit_for_bit_in_strict_mode) {
    for (const Knobs& k : strict_modes()) {
        auto engine = make_engine(k);
        Rng rng(2024);
        std::size_t compared = 0, differing = 0;
        for (const Subject& s : plain_subjects(rng)) {
            const auto full = run(*engine, s.rows);
            for (const std::size_t w : s.watched) {
                const std::size_t hidden = tiny_test_config().hidden_size;
                for (std::size_t slot = 0; slot < s.rows[w].readout_offsets.size(); ++slot) {
                    // Same subject, but row w reads out only this one offset.
                    std::vector<EngineRow> rows = s.rows;
                    for (EngineRow& r : rows) r.readout_offsets.clear();
                    rows[w].readout_offsets = {s.rows[w].readout_offsets[slot]};
                    const auto one = run(*engine, rows);
                    ++compared;
                    const std::vector<float> want(full[w].hidden.begin() + static_cast<std::ptrdiff_t>(slot * hidden),
                                                  full[w].hidden.begin() + static_cast<std::ptrdiff_t>((slot + 1) * hidden));
                    if (!same_bits(one[w].hidden, want)) ++differing;
                }
            }
        }
        std::cout << "  " << describe(k) << ": " << compared - differing << " of " << compared << " single-offset readouts bit-identical to the all-offsets run\n";
        CHECK_EQ(differing, std::size_t{0});
    }
}

SEPAL_TEST(attack_invariance, capture_keep_snapshot_and_a_row_twice_change_nothing_in_strict_mode) {
    for (const Knobs& k : strict_modes()) {
        auto engine = make_engine(k);
        Rng rng(99);
        std::size_t trials = 0, bad = 0;
        for (const Subject& s : plain_subjects(rng)) {
            const auto base = run(*engine, s.rows);
            // capture_layers on one row, on all rows
            for (const bool all : {false, true}) {
                std::vector<EngineRow> rows = s.rows;
                for (std::size_t i = 0; i < rows.size(); ++i) rows[i].capture_layers = all || i == rows.size() - 1;
                const auto got = run(*engine, rows);
                ++trials;
                if (!same_hidden(got, base)) ++bad;
                for (std::size_t i = 0; i < rows.size(); ++i) {
                    if (rows[i].capture_layers) {
                        CHECK_EQ(got[i].layers.size(), tiny_test_config().num_layers * rows[i].token_ids.size() * tiny_test_config().hidden_size);
                        CHECK(all_finite(got[i].layers));
                    }
                }
            }
            // keep_snapshot on every row and on none
            for (const bool keep : {true, false}) {
                std::vector<EngineRow> rows = s.rows;
                for (EngineRow& r : rows) r.keep_snapshot = keep;
                const auto got = run(*engine, rows);
                ++trials;
                if (!same_hidden(got, base)) ++bad;
            }
            // The whole subject twice in one call (second copy's ParentRow indices shifted), also with another copy first.
            std::vector<EngineRow> twice = s.rows;
            for (EngineRow r : s.rows) {
                if (auto* p = std::get_if<ParentRow>(&r.parent)) p->index += s.rows.size();
                twice.push_back(r);
            }
            const auto got = run(*engine, twice);
            ++trials;
            std::vector<EngineRowResult> first(got.begin(), got.begin() + static_cast<std::ptrdiff_t>(s.rows.size())),
                second(got.begin() + static_cast<std::ptrdiff_t>(s.rows.size()), got.end());
            if (!same_hidden(first, base) || !same_hidden(second, base)) ++bad;
            // Readout list with repeats and another order: the rows come back in the order asked.
            std::vector<EngineRow> reordered = s.rows;
            const std::size_t w = s.watched.back();
            std::vector<std::size_t> order = {s.rows[w].readout_offsets.back(), s.rows[w].readout_offsets.front(), s.rows[w].readout_offsets.back()};
            reordered[w].readout_offsets = order;
            const auto ro = run(*engine, reordered);
            const std::size_t hidden = tiny_test_config().hidden_size;
            for (std::size_t j = 0; j < order.size(); ++j) {
                std::size_t slot = 0;
                while (s.rows[w].readout_offsets[slot] != order[j]) ++slot;
                const std::vector<float> want(base[w].hidden.begin() + static_cast<std::ptrdiff_t>(slot * hidden),
                                              base[w].hidden.begin() + static_cast<std::ptrdiff_t>((slot + 1) * hidden));
                const std::vector<float> have(ro[w].hidden.begin() + static_cast<std::ptrdiff_t>(j * hidden), ro[w].hidden.begin() + static_cast<std::ptrdiff_t>((j + 1) * hidden));
                ++trials;
                if (!same_bits(want, have)) ++bad;
            }
        }
        std::cout << "  " << describe(k) << ": " << trials - bad << " of " << trials << " variants bit-identical (capture_layers, keep_snapshot, row twice, readout order)\n";
        CHECK_EQ(bad, std::size_t{0});
    }
}

SEPAL_TEST(attack_invariance, prune_last_layer_and_capture_layers_in_natural_mode_are_close) {
    // Natural mode promises no bits; this measures how far a capture_layers row elsewhere in the call (which turns the
    // pruning off for the whole call) moves the others, per option set.
    for (const Knobs& k : all_modes()) {
        if (k.gemm_rows != 0) continue;
        auto engine = make_engine(k);
        Rng rng(4);
        double worst = 0.0;
        std::size_t differing = 0, rows_seen = 0;
        for (const Subject& s : plain_subjects(rng)) {
            const auto base = run(*engine, s.rows);
            std::vector<EngineRow> rows = s.rows;
            rows[0].capture_layers = true;
            const auto got = run(*engine, rows);
            for (std::size_t i = 0; i < rows.size(); ++i) {
                ++rows_seen;
                if (!same_bits(got[i].hidden, base[i].hidden)) ++differing;
                worst = std::max(worst, hidden_distance(got[i].hidden, base[i].hidden));
            }
        }
        std::cout << "  " << describe(k) << ": capture_layers on row 0 moves other rows by at most " << worst << " (" << differing << " of " << rows_seen << " rows differ in bits)\n";
        CHECK(worst <= (k.precision == Precision::F32 ? 2e-4 : kF16Bar));
    }
}

SEPAL_TEST(attack_invariance, pruned_and_unpruned_last_layer_agree_in_strict_mode_with_snapshots_and_far_positions) {
    for (const Precision p : {Precision::F32, Precision::F16}) {
        for (const std::size_t g : {std::size_t{8}, std::size_t{24}}) {
            auto pruned = make_engine(strict_of(p, g, true, true));
            auto full = make_engine(strict_of(p, g, false, true));
            Rng rng(1234 + g);
            std::size_t same = 0, total = 0;
            DeviceSnapshot sp, sf;
            for (int step = 0; step < 40; ++step) {
                std::vector<EngineRow> rows;
                const std::size_t n_rows = 1 + rng.below(5);
                for (std::size_t i = 0; i < n_rows; ++i) {
                    const std::size_t len = 1 + rng.below(40);
                    std::vector<std::size_t> readouts;
                    for (std::size_t j = 0, c = rng.below(4); j < c; ++j) readouts.push_back(rng.below(len));
                    EngineRow r = make_row(tokens(rng, len), readouts, i == 0);
                    if (i > 0 && rng.chance(60)) r.parent = ParentRow{rng.below(i)};
                    if (rng.chance(25)) r.start_position = rng.below(70000);
                    rows.push_back(r);
                }
                std::vector<EngineRow> rows_f = rows, rows_p = rows;
                if (sp.valid() && rng.chance(50)) {
                    rows_p[0].parent = sp;
                    rows_f[0].parent = sf;
                }
                const auto rp = run(*pruned, rows_p);
                const auto rf = run(*full, rows_f);
                for (std::size_t i = 0; i < rows.size(); ++i) {
                    ++total;
                    if (same_bits(rp[i].hidden, rf[i].hidden)) ++same;
                }
                sp = rp[0].snapshot;
                sf = rf[0].snapshot;
            }
            std::cout << "  strict " << name_of(p) << " gemm_rows " << g << ": " << same << " of " << total << " rows bit-identical, pruned vs full last layer (random calls, snapshots, far positions)\n";
            CHECK_EQ(same, total);
        }
    }
}

// ---- snapshots in use while others come and go ----

SEPAL_TEST(attack_lifetime, a_snapshot_used_by_many_later_calls_keeps_its_values_while_others_are_made_and_released) {
    for (const Knobs& base : {strict_of(Precision::F32, 8, true, true), strict_of(Precision::F16, 8, true, true), strict_of(Precision::F32, 24, false, true)}) {
        Knobs k = base;
        k.slots = 10;   // tiny pools: freed slots and heap blocks are reused at once
        k.max_tokens = 512;
        k.budget_mib = 48;
        auto engine = make_engine(k);
        Rng rng(8008);
        // The chain under watch: A (80 tokens), B continues A (30), C continues B (10); then A and B handles are dropped.
        DeviceSnapshot a = run(*engine, {make_row(tokens(rng, 80), {}, true)})[0].snapshot;
        DeviceSnapshot b = run(*engine, {child_of(make_row(tokens(rng, 30), {}, true), a)})[0].snapshot;
        DeviceSnapshot c = run(*engine, {child_of(make_row(tokens(rng, 10), {}, true), b)})[0].snapshot;
        const auto probe_tokens = tokens(rng, 6);
        const auto probe = [&](const DeviceSnapshot& s) {
            std::vector<EngineRow> rows;
            for (std::size_t j = 0; j < 3; ++j) {
                EngineRow r = make_row({probe_tokens.begin(), probe_tokens.begin() + static_cast<std::ptrdiff_t>(2 + 2 * j)}, {1 + 2 * j, 0});
                r.parent = s;
                rows.push_back(r);
            }
            return run(*engine, rows);
        };
        const auto want_a = probe(a), want_b = probe(b), want_c = probe(c);
        const EngineMemory before_churn = engine->memory();
        std::vector<DeviceSnapshot> junk;
        std::size_t checks = 0, bad = 0;
        a = DeviceSnapshot(); // only B's and C's chains keep A's keys and values from here on
        for (int step = 0; step < 400; ++step) {
            switch (rng.below(6)) {
            case 0:
            case 1: { // make new snapshots, big and small, sometimes continuing junk
                std::vector<EngineRow> rows;
                for (std::size_t i = 0, n = 1 + rng.below(3); i < n; ++i) {
                    EngineRow r = make_row(tokens(rng, 1 + rng.below(120)), {}, true);
                    if (!junk.empty() && rng.chance(40)) r.parent = junk[rng.below(junk.size())];
                    rows.push_back(r);
                }
                try {
                    const auto got = run(*engine, rows);
                    for (const auto& g : got) junk.push_back(g.snapshot);
                } catch (const BudgetExceeded&) {
                    // pools full: release a few and go on
                    for (std::size_t i = 0, n = junk.size() / 2; i < n; ++i) {
                        const std::size_t at = rng.below(junk.size());
                        junk.erase(junk.begin() + static_cast<std::ptrdiff_t>(at));
                    }
                }
                break;
            }
            case 2: { // release some
                for (std::size_t i = 0, n = rng.below(4); i < n && !junk.empty(); ++i) junk.erase(junk.begin() + static_cast<std::ptrdiff_t>(rng.below(junk.size())));
                break;
            }
            case 3: {
                ++checks;
                if (!same_hidden(probe(c), want_c)) ++bad;
                break;
            }
            case 4: {
                if (!b.valid()) break;
                ++checks;
                if (!same_hidden(probe(b), want_b)) ++bad;
                break;
            }
            default: { // drop B's handle mid-way: C must go on working through its parent link
                if (step > 200 && b.valid()) b = DeviceSnapshot();
                ++checks;
                if (!same_hidden(probe(c), want_c)) ++bad;
                break;
            }
            }
        }
        (void)want_a;
        junk.clear();
        c = DeviceSnapshot();
        b = DeviceSnapshot();
        const EngineMemory after = engine->memory();
        std::cout << "  " << describe(k) << ": " << checks - bad << " of " << checks << " probes of a long-lived chain bit-identical after heavy churn (pools: " << before_churn.slots_total
                  << " slots, heap " << before_churn.kv_heap_bytes / 1024 << " KiB; peak slots " << after.slots_peak << ")\n";
        CHECK_EQ(bad, std::size_t{0});
        CHECK_EQ(after.slots_in_use, std::size_t{0});
        CHECK_EQ(after.kv_in_use_bytes, std::size_t{0});
        CHECK_EQ(after.kv_largest_free_bytes, after.kv_heap_bytes);
    }
}
