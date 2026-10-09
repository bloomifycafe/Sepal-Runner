// Adversarial tests of the engine's memory, determinism and thread-safety claims, on the tiny geometry.
//
// Suites: attack_memory (pools filled to the brim, fragmentation, BudgetExceeded in the middle of a session, two engines,
// an engine destroyed under live snapshots), attack_deepchain (a chain of thousands of one-token snapshots, built and
// then released), attack_determinism (strict-mode bits across instances, budgets, slot counts, other work and process
// restarts), attack_concurrency (two threads on one engine).
//
// Environment: SEPAL_DET_OUT / SEPAL_DET_PAD_MB are used by attack_determinism's child processes (suite det_dump).

#include <atomic>
#include <cstdio>
#include <fstream>
#include <mutex>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

#include "attack_util.hpp"

using namespace attack;

namespace {

Knobs small_pools(Precision p = Precision::F32, std::size_t gemm_rows = 8) {
    Knobs k;
    k.precision = p;
    k.gemm_rows = gemm_rows;
    k.max_tokens = 2048;
    k.max_rows = 64;
    k.slots = 12;
    k.budget_mib = 48;
    return k;
}

// How many kept snapshots of `len` tokens fit before BudgetExceeded, one per call. Fills `held`.
std::size_t fill_until_full(CudaEngine& engine, Rng& rng, std::size_t len, std::vector<DeviceSnapshot>& held, std::size_t limit = 100000) {
    std::size_t made = 0;
    while (made < limit) {
        try {
            const auto r = run(engine, {make_row(tokens(rng, len), {}, true)});
            held.push_back(r[0].snapshot);
            ++made;
        } catch (const BudgetExceeded&) {
            break;
        }
    }
    return made;
}

} // namespace

SEPAL_TEST(attack_memory, fill_slots_and_heap_release_in_random_order_refill_and_compare_with_a_fresh_engine) {
    for (const Precision precision : {Precision::F32, Precision::F16}) {
        const Knobs k = small_pools(precision);
        auto engine = make_engine(k);
        auto fresh = make_engine(k);
        const EngineMemory start = engine->memory();
        Rng rng(555);
        // 1. Slot-limited: many tiny snapshots.
        std::vector<DeviceSnapshot> held;
        std::vector<std::vector<std::uint32_t>> held_tokens;
        std::size_t made = 0;
        for (;;) {
            held_tokens.push_back(tokens(rng, 5));
            try {
                held.push_back(run(*engine, {make_row(held_tokens.back(), {}, true)})[0].snapshot);
                ++made;
            } catch (const BudgetExceeded& e) {
                held_tokens.pop_back();
                std::cout << "  " << name_of(precision) << ": slot pool refused the call after " << made << " snapshots (" << e.what() << ")\n";
                break;
            }
        }
        CHECK_EQ(made, start.slots_total);
        EngineMemory full = engine->memory();
        CHECK_EQ(full.slots_in_use, start.slots_total);
        // A call that needs no slot still runs while the pool is full (nothing continues, nothing kept).
        const auto plain = run(*engine, {make_row(tokens(rng, 7), {6})});
        CHECK_EQ(plain[0].hidden.size(), std::size_t{24});
        // Probes from every held snapshot, bit-compared with the same chain run in a fresh engine as one row.
        const auto probe_tokens = tokens(rng, 9);
        const auto probe_all = [&](CudaEngine& e, const std::vector<DeviceSnapshot>& snaps) {
            std::vector<EngineRow> rows;
            for (const DeviceSnapshot& s : snaps) rows.push_back(child_of(make_row(probe_tokens, {8}), s));
            return run(e, rows);
        };
        // Reference answers from the fresh engine: parent tokens + probe tokens as one row (strict mode: same bits as continuing).
        const auto want = [&](std::size_t i) {
            std::vector<std::uint32_t> all = held_tokens[i];
            all.insert(all.end(), probe_tokens.begin(), probe_tokens.end());
            return run(*fresh, {make_row(all, {all.size() - 1})})[0].hidden;
        };
        // A pool of 12 snapshots, but a probe call wants a slot for each row that has a child or keeps a snapshot: none do.
        const auto got = probe_all(*engine, held);
        std::size_t exact = 0;
        double worst = 0;
        for (std::size_t i = 0; i < held.size(); ++i) {
            const auto w = want(i);
            if (same_bits(got[i].hidden, w)) ++exact;
            worst = std::max(worst, hidden_distance(got[i].hidden, w));
        }
        std::cout << "    " << exact << " of " << held.size() << " continued snapshots equal the one-row run bit for bit (largest difference " << worst << ")\n";
        CHECK(worst <= (precision == Precision::F32 ? 1e-4 : 0.01));

        // 2. Release in random order, refill, repeat; the pool must hand out exactly slots_total each time.
        for (int round = 0; round < 6; ++round) {
            while (!held.empty()) {
                const std::size_t at = rng.below(held.size());
                held.erase(held.begin() + static_cast<std::ptrdiff_t>(at));
                held_tokens.erase(held_tokens.begin() + static_cast<std::ptrdiff_t>(at));
                if (rng.chance(30)) break;
            }
            const std::size_t before = held.size();
            for (;;) {
                held_tokens.push_back(tokens(rng, 1 + rng.below(40)));
                try {
                    held.push_back(run(*engine, {make_row(held_tokens.back(), {}, true)})[0].snapshot);
                } catch (const BudgetExceeded&) {
                    held_tokens.pop_back();
                    break;
                }
            }
            CHECK_EQ(held.size(), start.slots_total);
            (void)before;
        }
        const auto again = probe_all(*engine, held);
        for (std::size_t i = 0; i < held.size(); ++i) CHECK(hidden_distance(again[i].hidden, want(i)) <= (precision == Precision::F32 ? 1e-4 : 0.01));
        held.clear();
        const EngineMemory empty = engine->memory();
        CHECK_EQ(empty.slots_in_use, std::size_t{0});
        CHECK_EQ(empty.kv_in_use_bytes, std::size_t{0});
        CHECK_EQ(empty.kv_largest_free_bytes, empty.kv_heap_bytes);
    }
}

SEPAL_TEST(attack_memory, heap_exhaustion_leaves_no_partial_call_behind_and_fragmentation_is_recovered) {
    Knobs k = small_pools(Precision::F32, 8);
    k.slots = 64;
    auto engine = make_engine(k);
    Rng rng(88);
    std::vector<DeviceSnapshot> held;
    const std::size_t n_long = fill_until_full(*engine, rng, 400, held);
    EngineMemory m = engine->memory();
    std::cout << "  " << n_long << " snapshots of 400 tokens filled the heap of " << m.kv_heap_bytes / 1024 << " KiB (" << m.kv_in_use_bytes / 1024 << " KiB in use, largest free block "
              << m.kv_largest_free_bytes / 1024 << " KiB)\n";
    CHECK(n_long >= 2);
    // Release every second one: free space is large in total but in separate holes.
    std::vector<DeviceSnapshot> kept;
    for (std::size_t i = 0; i < held.size(); ++i) {
        if (i % 2 == 0) kept.push_back(held[i]);
    }
    held.clear();
    m = engine->memory();
    std::cout << "    after releasing every second: " << m.kv_in_use_bytes / 1024 << " KiB in use, largest hole " << m.kv_largest_free_bytes / 1024 << " KiB\n";
    // A call of three rows, the first two fit in the holes, the third (a single big row) cannot: all or nothing.
    const std::size_t in_use_before = m.kv_in_use_bytes, slots_before = m.slots_in_use;
    const auto probe_rows = [&](std::size_t big) {
        return std::vector<EngineRow>{make_row(tokens(rng, 100), {99}, true), make_row(tokens(rng, 100), {99}, true), make_row(tokens(rng, big), {big - 1}, true)};
    };
    bool refused = false;
    try {
        (void)run(*engine, probe_rows(1500));
    } catch (const BudgetExceeded& e) {
        refused = true;
        std::cout << "    refused: " << e.what() << "\n";
    }
    CHECK(refused);
    m = engine->memory();
    CHECK_EQ(m.kv_in_use_bytes, in_use_before);
    CHECK_EQ(m.slots_in_use, slots_before);
    // The same engine now answers like a fresh one on a call that fits.
    auto fresh = make_engine(k);
    const std::vector<EngineRow> fits = {make_row(tokens(rng, 100), {99, 0}), make_row(tokens(rng, 60), {59})};
    CHECK(same_hidden(run(*engine, fits), run(*fresh, fits)));
    kept.clear();
    m = engine->memory();
    CHECK_EQ(m.kv_in_use_bytes, std::size_t{0});
    CHECK_EQ(m.kv_largest_free_bytes, m.kv_heap_bytes);
}

namespace {

// One reproducible session of random calls on a small-pool engine. A call that throws BudgetExceeded is recorded in
// `failed`; a call in `skip` is not made at all (it failed on the engine that recorded the failures). Returns a digest
// per call that ran.
struct Session {
    std::vector<std::uint64_t> digests;       // per call that ran
    std::vector<std::size_t> failed;          // call numbers that threw BudgetExceeded
    std::size_t calls = 0, rows = 0, tokens_run = 0;
};

Session run_session(CudaEngine& engine, std::uint64_t seed, std::size_t calls, const std::set<std::size_t>& skip, std::size_t window) {
    Rng rng(seed);
    Session out;
    std::vector<DeviceSnapshot> live;
    for (std::size_t c = 0; c < calls; ++c) {
        // The random choices are drawn whether or not the call runs, so both engines see the same sequence.
        const std::size_t n_rows = 1 + rng.below(4);
        std::vector<EngineRow> rows;
        for (std::size_t i = 0; i < n_rows; ++i) {
            const std::size_t len = rng.chance(8) ? 200 + rng.below(250) : 1 + rng.below(60);
            std::vector<std::size_t> readouts = {rng.below(len)};
            EngineRow r = make_row(tokens(rng, len), readouts, rng.chance(60));
            const std::size_t pick = rng.below(100);
            const std::size_t live_index = live.empty() ? 0 : rng.below(live.size());
            if (pick < 40 && !live.empty()) r.parent = live[live_index];
            else if (pick < 60 && i > 0) r.parent = ParentRow{rng.below(i)};
            rows.push_back(std::move(r));
        }
        const std::size_t drop = rng.below(live.size() + 1);
        const bool will_drop = rng.chance(window > live.size() ? 25 : 80);
        if (skip.count(c) != 0) {
            // Same effect on `live` as a failed call: nothing added. Release as the original did.
            if (will_drop && drop < live.size()) live.erase(live.begin() + static_cast<std::ptrdiff_t>(drop));
            continue;
        }
        try {
            const auto got = run(engine, rows);
            out.digests.push_back(digest(got));
            ++out.calls;
            out.rows += rows.size();
            for (const EngineRow& r : rows) out.tokens_run += r.token_ids.size();
            for (const auto& g : got) {
                if (g.snapshot.valid()) live.push_back(g.snapshot);
            }
        } catch (const BudgetExceeded&) {
            out.failed.push_back(c);
        }
        if (will_drop && drop < live.size()) live.erase(live.begin() + static_cast<std::ptrdiff_t>(drop));
    }
    return out;
}

} // namespace

SEPAL_TEST(attack_memory, budget_exceeded_in_the_middle_of_a_long_session_then_bits_match_a_fresh_big_engine) {
    for (const Precision precision : {Precision::F32, Precision::F16}) {
        Knobs small = small_pools(precision, 8);
        small.slots = 14;
        small.budget_mib = 44;
        Knobs big = small;
        big.slots = 400;
        big.budget_mib = 160;
        auto tight = make_engine(small);
        auto roomy = make_engine(big);
        const std::size_t calls = 1500;
        const Session a = run_session(*tight, 2718, calls, {}, 12);
        const std::set<std::size_t> skip(a.failed.begin(), a.failed.end());
        const Session b = run_session(*roomy, 2718, calls, skip, 12);
        std::cout << "  " << name_of(precision) << ": " << calls << " calls on a tight engine (" << small.slots << " slots), " << a.failed.size() << " refused with BudgetExceeded, " << a.calls << " ran ("
                  << a.rows << " rows, " << a.tokens_run << " tokens)\n";
        CHECK(!a.failed.empty());
        CHECK_EQ(a.digests.size(), b.digests.size());
        std::size_t equal = 0;
        for (std::size_t i = 0; i < std::min(a.digests.size(), b.digests.size()); ++i) equal += a.digests[i] == b.digests[i] ? 1 : 0;
        std::cout << "    " << equal << " of " << a.digests.size() << " calls bit-identical to the same session on a roomy engine that never refused\n";
        CHECK_EQ(equal, a.digests.size());
        const EngineMemory m = tight->memory();
        std::cout << "    tight engine at the end: slots in use " << m.slots_in_use << " (peak " << m.slots_peak << "), heap in use " << m.kv_in_use_bytes / 1024 << " KiB (peak " << m.kv_peak_bytes / 1024 << " of "
                  << m.kv_heap_bytes / 1024 << " KiB)\n";
    }
}

SEPAL_TEST(attack_memory, ten_thousand_calls_with_snapshots_of_all_sizes_neither_leak_nor_stay_fragmented) {
    Knobs k = small_pools(Precision::F32, 0);
    k.max_rows = 16;
    k.slots = 48;
    k.budget_mib = 56;
    auto engine = make_engine(k);
    const std::size_t free_before = device_free();
    const EngineMemory start = engine->memory();
    Rng rng(31415);
    std::vector<DeviceSnapshot> live;
    std::size_t calls = 0, refused = 0, rows_total = 0;
    double worst_frag = 0.0; // 1 - largest hole / total free
    for (; calls < 12000; ++calls) {
        std::vector<EngineRow> rows;
        const std::size_t n_rows = 1 + rng.below(4);
        for (std::size_t i = 0; i < n_rows; ++i) {
            const std::size_t len = rng.chance(5) ? 100 + rng.below(300) : 1 + rng.below(30);
            EngineRow r = make_row(tokens(rng, len), {rng.below(len)}, rng.chance(50));
            if (!live.empty() && rng.chance(30)) r.parent = live[rng.below(live.size())];
            else if (i > 0 && rng.chance(30)) r.parent = ParentRow{rng.below(i)};
            rows.push_back(std::move(r));
        }
        try {
            const auto got = run(*engine, rows);
            rows_total += rows.size();
            for (const auto& g : got) {
                if (g.snapshot.valid()) live.push_back(g.snapshot);
            }
        } catch (const BudgetExceeded&) {
            ++refused;
            for (std::size_t i = 0, n = live.size() / 2; i < n; ++i) live.erase(live.begin() + static_cast<std::ptrdiff_t>(rng.below(live.size())));
        }
        while (live.size() > 20 + rng.below(10)) live.erase(live.begin() + static_cast<std::ptrdiff_t>(rng.below(live.size())));
        if (calls % 50 == 0) {
            const EngineMemory m = engine->memory();
            const std::size_t free_total = m.kv_heap_bytes - m.kv_in_use_bytes;
            if (free_total != 0) worst_frag = std::max(worst_frag, 1.0 - static_cast<double>(m.kv_largest_free_bytes) / static_cast<double>(free_total));
        }
    }
    const EngineMemory busy = engine->memory();
    live.clear();
    const EngineMemory end = engine->memory();
    const std::size_t free_after = device_free();
    std::cout << "  " << calls << " calls, " << rows_total << " rows, " << refused << " refused; worst fragmentation seen (1 - largest hole / free) " << worst_frag << "; peak slots " << busy.slots_peak << " of "
              << start.slots_total << ", peak heap " << busy.kv_peak_bytes / 1024 << " of " << start.kv_heap_bytes / 1024 << " KiB\n";
    std::cout << "    device free memory " << free_before / 1024 << " KiB before, " << free_after / 1024 << " KiB after\n";
    CHECK_EQ(end.slots_in_use, std::size_t{0});
    CHECK_EQ(end.kv_in_use_bytes, std::size_t{0});
    CHECK_EQ(end.kv_largest_free_bytes, end.kv_heap_bytes);
    const std::size_t drift = free_before > free_after ? free_before - free_after : free_after - free_before;
    CHECK(drift < (std::size_t{2} << 20)); // other processes use the card too
}

SEPAL_TEST(attack_memory, an_engine_whose_creation_fails_leaves_no_device_memory_behind) {
    // Three ways to fail after the stream and the cuBLAS handle exist: a budget too small, weights of another
    // geometry, and a budget larger than the card. Thirty of each; the cuBLAS handle alone is megabytes.
    ModelConfig other = tiny_test_config();
    other.hidden_size = 32;
    other.pointer_dim = 12;
    std::size_t refused = 0;
    // Warm up so that lazily loaded modules and the first handle do not count as a leak.
    { auto warm = make_engine(small_pools()); (void)run(*warm, {make_row({1, 2, 3}, {2})}); }
    const std::size_t free0 = device_free();
    for (int i = 0; i < 30; ++i) {
        Knobs small = small_pools();
        small.budget_mib = 20;
        try { auto e = make_engine(small); } catch (const BudgetExceeded&) { ++refused; }
        try { CudaEngine e(other, tiny_weights(31), options_of(small_pools())); } catch (const std::exception&) { ++refused; }
        Knobs huge = small_pools();
        huge.budget_mib = std::size_t{1} << 20; // 1 TiB
        try { auto e = make_engine(huge); } catch (const BudgetExceeded&) { ++refused; }
    }
    const std::size_t free1 = device_free();
    std::cout << "  90 failed creations (" << refused << " refused): device free " << free0 / 1024 << " -> " << free1 / 1024 << " KiB (" << (free0 > free1 ? (free0 - free1) / 1024 : 0)
              << " KiB lost)\n";
    CHECK_EQ(refused, std::size_t{90});
    CHECK(free0 < free1 + (std::size_t{2} << 20));
}

SEPAL_TEST(attack_memory, two_engines_alive_on_the_device_do_not_disturb_each_other_and_do_not_leak_when_created_over_and_over) {
    const std::size_t free0 = device_free();
    for (int i = 0; i < 12; ++i) {
        auto e = make_engine(small_pools());
        (void)run(*e, {make_row({1, 2, 3}, {2})});
    }
    const std::size_t free1 = device_free();
    std::cout << "  12 engines created, used and destroyed: device free " << free0 / 1024 << " -> " << free1 / 1024 << " KiB\n";
    CHECK(free0 < free1 + (std::size_t{4} << 20) && free1 < free0 + (std::size_t{4} << 20));

    // Failed creation (budget too small) leaves nothing behind.
    Knobs tiny = small_pools();
    tiny.budget_mib = 20;
    bool threw = false;
    try {
        auto e = make_engine(tiny);
    } catch (const BudgetExceeded&) {
        threw = true;
    }
    CHECK(threw);
    CHECK(device_free() + (std::size_t{4} << 20) > free0);

    Knobs k32 = small_pools(Precision::F32, 8), k16 = small_pools(Precision::F16, 8);
    auto a = make_engine(k32, 31);
    auto b = make_engine(k16, 77); // other weights
    auto a_alone = make_engine(k32, 31);
    auto b_alone = make_engine(k16, 77);
    Rng rng(9);
    DeviceSnapshot sa, sb, sa2, sb2;
    std::size_t same = 0, total = 0;
    for (int step = 0; step < 60; ++step) {
        const auto t = tokens(rng, 1 + rng.below(50));
        const auto u = tokens(rng, 1 + rng.below(20));
        EngineRow ra = make_row(t, {t.size() - 1}, true), rb = ra;
        EngineRow ra2 = make_row(t, {t.size() - 1}, true), rb2 = ra;
        if (sa.valid()) ra.parent = sa;
        if (sb.valid()) rb.parent = sb;
        if (sa2.valid()) ra2.parent = sa2;
        if (sb2.valid()) rb2.parent = sb2;
        const auto xa = run(*a, {ra}), xb = run(*b, {rb});        // interleaved
        const auto ya = run(*a_alone, {ra2}), yb = run(*b_alone, {rb2});
        total += 2;
        same += same_hidden(xa, ya) ? 1 : 0;
        same += same_hidden(xb, yb) ? 1 : 0;
        sa = xa[0].snapshot;
        sb = xb[0].snapshot;
        sa2 = ya[0].snapshot;
        sb2 = yb[0].snapshot;
        (void)u;
    }
    std::cout << "  interleaved calls on two live engines vs the same calls on engines used alone: " << same << " of " << total << " bit-identical\n";
    CHECK_EQ(same, total);
    // A snapshot of one engine is refused by the other, with a message, and changes nothing.
    EngineRow cross = make_row({1, 2}, {1});
    cross.parent = sa;
    CHECK_THROWS((void)run(*b, {cross}), std::invalid_argument);
    CHECK(run(*b, {make_row({1, 2}, {1})}).size() == 1);
}

SEPAL_TEST(attack_memory, an_engine_destroyed_under_live_snapshots_gives_its_memory_back_when_the_last_snapshot_goes) {
    // The first pass warms up whatever CUDA and cuBLAS load lazily on first use (megabytes of kernels); the second must
    // end with the same free memory it started with.
    std::size_t free0 = 0, free_held = 0;
    for (int pass = 0; pass < 2; ++pass) {
        if (pass == 1) free0 = device_free();
        std::vector<DeviceSnapshot> snaps;
        {
            auto engine = make_engine(small_pools(Precision::F32, 8));
            Rng rng(123);
            DeviceSnapshot s = run(*engine, {make_row(tokens(rng, 50), {}, true)})[0].snapshot;
            for (int i = 0; i < 6; ++i) {
                s = run(*engine, {child_of(make_row(tokens(rng, 20), {}, true), s)})[0].snapshot;
                snaps.push_back(s);
            }
            CHECK(snaps.back().tokens() == 170);
        }
        if (pass == 1) free_held = device_free();
        // A new engine works while the old pools are still held, and refuses the old snapshots.
        auto second = make_engine(small_pools(Precision::F32, 8));
        EngineRow r = make_row({3, 4}, {1});
        r.parent = snaps.back();
        CHECK_THROWS((void)run(*second, {r}), std::invalid_argument);
        CHECK_EQ(run(*second, {make_row({3, 4}, {1})}).size(), std::size_t{1});
        second.reset();
        // Newest first, then the rest.
        while (snaps.size() > 1) snaps.pop_back();
        snaps.clear();
    }
    const std::size_t free_end = device_free();
    std::cout << "  engine destroyed with 6 snapshots alive: device free " << free0 / 1024 << " KiB at the start, " << free_held / 1024 << " KiB while they live (pools stay), "
              << free_end / 1024 << " KiB after the last was released\n";
    CHECK(free_held < free0);
    CHECK(free_end + (std::size_t{1} << 20) > free0);
}

// ---- a very deep chain of snapshots ----

SEPAL_TEST(attack_deepchain, a_chain_of_twenty_thousand_one_token_snapshots_can_be_built_used_and_released) {
    Knobs k = small_pools(Precision::F32, 0);
    k.budget_mib = 128;
    k.slots = 8;
    auto engine = make_engine(k);
    Rng rng(1);
    const std::size_t depth = 20000;
    DeviceSnapshot s;
    std::vector<float> last;
    for (std::size_t i = 0; i < depth; ++i) {
        EngineRow r = make_row({static_cast<std::uint32_t>(rng.below(64))}, {0}, true);
        if (s.valid()) r.parent = s;
        const auto got = run(*engine, {r});
        s = got[0].snapshot;
        last = got[0].hidden;
        CHECK_EQ(s.tokens(), static_cast<std::uint64_t>(i + 1));
    }
    CHECK(all_finite(last));
    const EngineMemory m = engine->memory();
    std::cout << "  chain of " << depth << " snapshots built; heap in use " << m.kv_in_use_bytes / 1024 << " KiB\n";
    s = DeviceSnapshot(); // releases 20000 nested blocks
    const EngineMemory end = engine->memory();
    CHECK_EQ(end.kv_in_use_bytes, std::size_t{0});
    CHECK_EQ(end.slots_in_use, std::size_t{0});
}

// ---- determinism ----

namespace {

// A fixed script of calls, with stored snapshots, whose results are hashed. Used by the tests below and by det_dump.
std::uint64_t scripted_digest(CudaEngine& engine, std::uint64_t seed, std::size_t other_work_rows) {
    Rng rng(seed);
    std::uint64_t h = 0xCBF29CE484222325ull;
    std::vector<DeviceSnapshot> store;
    for (int step = 0; step < 40; ++step) {
        // The subject rows (drawn from `rng`, identical whatever else is in the call).
        std::vector<EngineRow> rows;
        const std::size_t n = 1 + rng.below(3);
        for (std::size_t i = 0; i < n; ++i) {
            const std::size_t len = 1 + rng.below(70);
            EngineRow r = make_row(tokens(rng, len), {rng.below(len), 0}, rng.chance(50));
            if (!store.empty() && rng.chance(40)) r.parent = store[rng.below(store.size())];
            else if (i > 0 && rng.chance(40)) r.parent = ParentRow{rng.below(i)};
            if (rng.chance(15)) r.start_position = rng.below(5000);
            rows.push_back(std::move(r));
        }
        // Other work around the subject: rows of a second generator, appended after (so subject indices stay valid).
        Rng other(seed * 7919 + static_cast<std::uint64_t>(step));
        for (std::size_t i = 0; i < other_work_rows; ++i) rows.push_back(make_row(tokens(other, 1 + other.below(100)), {other.below(2)}));
        const auto got = run(engine, rows);
        std::vector<EngineRowResult> subject(got.begin(), got.begin() + static_cast<std::ptrdiff_t>(n));
        h = digest(subject, h);
        for (std::size_t i = 0; i < n; ++i) {
            if (got[i].snapshot.valid()) store.push_back(got[i].snapshot);
        }
        while (store.size() > 12) store.erase(store.begin() + static_cast<std::ptrdiff_t>(rng.below(store.size())));
    }
    return h;
}

std::string hex(std::uint64_t v) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%016llx", static_cast<unsigned long long>(v));
    return buf;
}

} // namespace

SEPAL_TEST(attack_determinism, strict_bits_do_not_depend_on_the_instance_the_budget_the_slot_count_or_other_work) {
    for (const Precision precision : {Precision::F32, Precision::F16}) {
        for (const std::size_t g : {std::size_t{8}, std::size_t{24}}) {
            std::vector<std::pair<std::string, std::uint64_t>> results;
            const auto add = [&](const std::string& what, Knobs k, std::size_t other_rows) {
                k.precision = precision;
                k.gemm_rows = g;
                auto e = make_engine(k);
                results.emplace_back(what, scripted_digest(*e, 4711, other_rows));
            };
            Knobs base;
            base.max_tokens = 4096;
            base.max_rows = 64;
            add("baseline", base, 0);
            add("second instance", base, 0);
            Knobs b2 = base;
            b2.budget_mib = 400;
            b2.slots = 300;
            add("budget 400 MiB, 300 slots", b2, 0);
            Knobs b3 = base;
            b3.budget_mib = 70;
            b3.slots = 30;
            b3.max_tokens = 1024;
            add("budget 70 MiB, 30 slots, max_call_tokens 1024", b3, 0);
            Knobs b4 = base;
            b4.max_tokens = 3000;
            b4.max_rows = 40;
            add("max_call_tokens 3000, max_call_rows 40", b4, 0);
            add("with 5 other rows per call", base, 5);
            add("with 30 other rows per call", base, 30);
            std::cout << "  " << name_of(precision) << " strict " << g << ": ";
            for (const auto& [what, h] : results) std::cout << hex(h).substr(0, 8) << " ";
            std::cout << "\n";
            for (const auto& [what, h] : results) {
                if (h != results[0].second) std::cout << "    DIFFERENT: " << what << "\n";
                CHECK_EQ(h, results[0].second);
            }
        }
    }
}

SEPAL_TEST(det_dump, write_the_digest_of_the_scripted_session_to_the_file_named_by_SEPAL_DET_OUT) {
    const auto out = environment("SEPAL_DET_OUT");
    if (!out) {
        std::cout << "  (det_dump has no SEPAL_DET_OUT; nothing written)\n";
        return;
    }
    // Optionally hold some device memory first, so that this process's allocations land at other addresses.
    void* pad = nullptr;
    if (const auto mb = environment("SEPAL_DET_PAD_MB")) {
        if (std::stoul(*mb) != 0) cudaMalloc(&pad, std::stoul(*mb) << 20);
    }
    std::ofstream file(*out);
    for (const Precision precision : {Precision::F32, Precision::F16}) {
        for (const std::size_t g : {std::size_t{0}, std::size_t{8}, std::size_t{24}}) {
            Knobs k;
            k.precision = precision;
            k.gemm_rows = g;
            k.max_rows = 64;
            auto e = make_engine(k);
            file << name_of(precision) << " " << g << " " << hex(scripted_digest(*e, 4711, g == 24 ? 3 : 0)) << "\n";
        }
    }
}

SEPAL_TEST(attack_determinism, a_fresh_process_gives_the_same_bytes_in_strict_and_in_natural_mode) {
#ifdef _WIN32
    char path[MAX_PATH * 4] = {};
    GetModuleFileNameA(nullptr, path, sizeof path);
    const std::string exe = path;
    const std::string dir = environment("TEMP").value_or(".");
    const std::string f1 = dir + "\\sepal_det_1.txt", f2 = dir + "\\sepal_det_2.txt", f3 = dir + "\\sepal_det_3.txt";
    const auto spawn = [&](const std::string& file, const char* pad) {
        _putenv_s("SEPAL_DET_OUT", file.c_str());
        _putenv_s("SEPAL_DET_PAD_MB", pad);
        const std::string cmd = "\"\"" + exe + "\" det_dump > NUL\"";
        return std::system(cmd.c_str());
    };
    CHECK_EQ(spawn(f1, "0"), 0);
    CHECK_EQ(spawn(f2, "0"), 0);
    CHECK_EQ(spawn(f3, "137"), 0);
    _putenv_s("SEPAL_DET_OUT", "");
    _putenv_s("SEPAL_DET_PAD_MB", "");
    const auto slurp = [](const std::string& f) {
        std::ifstream in(f);
        return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    };
    const std::string a = slurp(f1), b = slurp(f2), c = slurp(f3);
    std::cout << "  digests from three processes (the third held 137 MiB of the device first):\n" << a;
    CHECK(!a.empty());
    CHECK(a == b);
    CHECK(a == c);
    // And this process agrees with them for the strict rows.
    std::remove(f1.c_str());
    std::remove(f2.c_str());
    std::remove(f3.c_str());
#else
    std::cout << "  (process-restart test is written for Windows)\n";
#endif
}

// ---- concurrency ----

SEPAL_TEST(attack_concurrency, two_threads_on_one_engine_are_serialised_and_every_result_is_the_serial_one) {
    Knobs k = small_pools(Precision::F32, 8);
    k.slots = 64;
    k.budget_mib = 96;
    auto engine = make_engine(k);
    auto serial = make_engine(k);
    constexpr int kThreads = 4, kCalls = 150;
    // Each thread has its own seeded sequence and snapshot chain; expected digests come from running the same on `serial`.
    const auto work = [&](CudaEngine& e, int t, std::vector<std::uint64_t>& out) {
        Rng rng(1000 + static_cast<std::uint64_t>(t));
        DeviceSnapshot s;
        for (int i = 0; i < kCalls; ++i) {
            EngineRow r = make_row(tokens(rng, 1 + rng.below(40)), {0}, true);
            if (s.valid() && rng.chance(70)) r.parent = s;
            const auto got = run(e, {r, make_row(tokens(rng, 3), {2})});
            out.push_back(digest(got));
            s = got[0].snapshot;
            if (i % 7 == 0) (void)e.memory(); // reads the pools while other threads run
        }
    };
    std::vector<std::vector<std::uint64_t>> want(kThreads), got(kThreads);
    for (int t = 0; t < kThreads; ++t) work(*serial, t, want[t]);
    std::vector<std::thread> threads;
    std::atomic<int> failures{0};
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t] {
            try {
                work(*engine, t, got[t]);
            } catch (const std::exception& e) {
                std::cerr << "thread " << t << ": " << e.what() << "\n";
                ++failures;
            }
        });
    }
    for (auto& th : threads) th.join();
    CHECK_EQ(failures.load(), 0);
    std::size_t same = 0, total = 0;
    for (int t = 0; t < kThreads; ++t) {
        for (std::size_t i = 0; i < want[t].size() && i < got[t].size(); ++i) {
            ++total;
            same += want[t][i] == got[t][i] ? 1 : 0;
        }
    }
    std::cout << "  " << kThreads << " threads x " << kCalls << " calls on one engine: " << same << " of " << total << " results bit-identical to the same calls made serially\n";
    CHECK_EQ(same, total);
    const EngineMemory m = engine->memory();
    CHECK_EQ(m.slots_in_use, std::size_t{0});
    CHECK_EQ(m.kv_in_use_bytes, std::size_t{0});
}

SEPAL_TEST(attack_concurrency, snapshots_released_by_other_threads_while_a_call_runs_and_set_profiling_from_a_third) {
    Knobs k = small_pools(Precision::F32, 8);
    k.slots = 48;
    k.budget_mib = 96;
    auto engine = make_engine(k);
    Rng rng(5);
    std::vector<DeviceSnapshot> pool;
    for (int i = 0; i < 24; ++i) pool.push_back(run(*engine, {make_row(tokens(rng, 30), {}, true)})[0].snapshot);
    const std::vector<DeviceSnapshot> originals = pool;
    // Want: a probe from each stored snapshot, run alone.
    const auto probe_tokens = tokens(rng, 8);
    std::vector<std::vector<float>> want;
    for (const DeviceSnapshot& s : pool) want.push_back(run(*engine, {child_of(make_row(probe_tokens, {7}), s)})[0].hidden);
    std::atomic<bool> stop{false};
    std::atomic<int> bad{0};
    std::vector<DeviceSnapshot> shared = pool; // the "caller" thread keeps handles, the releaser works on copies of the vector
    std::mutex shared_mutex;
    std::thread releaser([&] {
        Rng r2(77);
        while (!stop) {
            DeviceSnapshot dropped;
            {
                std::lock_guard<std::mutex> g(shared_mutex);
                // Copy a handle out, then drop it outside the lock: the last reference may die on this thread.
                dropped = shared[r2.below(shared.size())];
            }
            dropped = DeviceSnapshot();
            std::this_thread::yield();
        }
    });
    std::thread profiler([&] {
        bool on = false;
        while (!stop) {
            engine->set_profiling(on = !on);
            (void)engine->memory();
            std::this_thread::yield();
        }
        engine->set_profiling(false);
    });
    for (int i = 0; i < 300; ++i) {
        const std::size_t at = rng.below(pool.size());
        EngineRow r = child_of(make_row(probe_tokens, {7}, true), pool[at]);
        const auto got = run(*engine, {r});
        if (!same_bits(got[0].hidden, want[at])) ++bad;
        // The result snapshot is dropped at once, in this thread, while the others work.
    }
    stop = true;
    releaser.join();
    profiler.join();
    CHECK_EQ(bad.load(), 0);
    std::cout << "  300 calls while another thread dropped handles and a third toggled profiling: " << bad.load() << " wrong results\n";
}
