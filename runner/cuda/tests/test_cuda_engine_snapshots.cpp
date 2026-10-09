// Tests of the snapshot density and tier features on the tiny geometry: binary16 state storage (EngineOptions::half_state),
// and the host and file tiers (to_host, to_device, save, load_host, load).
//
// Suites: snap_half (binary16 state: sizes, strict-mode hit = miss, closeness to float32 state, deep chains),
//         snap_tiers (round trips bit for bit, chains, lifetime, memory back to zero, refusals, pools full),
//         snap_fuzz (truncated, corrupted and lying files: refused, never a crash).
//
// Files are written under snapshot-test-files/ in the working directory (SEPAL_SNAPSHOT_DIR overrides) and removed.

#include <filesystem>
#include <fstream>

#include "attack_util.hpp"
#include "snapshot_file.hpp"

using namespace attack;
using sepal::cuda::HostMemory;
using sepal::cuda::HostSnapshot;
using sepal::cuda::SnapshotFileError;

namespace {

namespace fs = std::filesystem;

fs::path scratch_dir() {
    const auto env = environment("SEPAL_SNAPSHOT_DIR");
    fs::path dir = env ? fs::path(*env) : fs::path("snapshot-test-files");
    fs::create_directories(dir);
    return dir;
}

struct Variant {
    Precision precision = Precision::F32;
    std::size_t gemm_rows = 8;
    bool half_state = false;
};

std::unique_ptr<CudaEngine> engine_of(const Variant& v, std::uint64_t weight_seed = 31, std::size_t slots = 48, std::size_t budget_mib = 96) {
    Knobs k;
    k.precision = v.precision;
    k.gemm_rows = v.gemm_rows;
    k.slots = slots;
    k.budget_mib = budget_mib;
    k.max_tokens = 2048;
    k.max_rows = 64;
    EngineOptions o = options_of(k);
    o.half_state = v.half_state;
    return std::make_unique<CudaEngine>(tiny_test_config(), tiny_weights(weight_seed), o);
}

std::vector<std::uint32_t> random_tokens(Rng& rng, std::size_t n) { return tokens(rng, n); }

// The children of a state, run in the same call as the state (the "miss": state computed in the call) and from the stored
// snapshot in a later call (the "hit"). Returns both hidden-state sets.
struct HitMiss {
    std::vector<EngineRowResult> miss, hit;
};

HitMiss hit_and_miss(CudaEngine& engine, const std::vector<std::uint32_t>& state, const std::vector<std::vector<std::uint32_t>>& children) {
    HitMiss r;
    std::vector<EngineRow> rows;
    rows.push_back(make_row(state, {}, true));
    for (const auto& c : children) rows.push_back(child_of(make_row(c, {c.size() - 1}), 0));
    const auto miss_all = run(engine, rows);
    r.miss.assign(miss_all.begin() + 1, miss_all.end());
    std::vector<EngineRow> second;
    for (const auto& c : children) second.push_back(child_of(make_row(c, {c.size() - 1}), miss_all[0].snapshot));
    r.hit = run(engine, second);
    return r;
}

} // namespace

// ---- snap_half ----

SEPAL_TEST(snap_half, slots_are_half_the_size) {
    for (const Precision p : {Precision::F32, Precision::F16}) {
        Variant a{p, 8, false}, b{p, 8, true};
        const auto full = engine_of(a), half = engine_of(b);
        const EngineMemory mf = full->memory(), mh = half->memory();
        CHECK_EQ(mh.slot_bytes * 2, mf.slot_bytes);
        std::cout << "  " << name_of(p) << ": slot " << mf.slot_bytes << " bytes -> " << mh.slot_bytes << " bytes\n";
        Rng rng(5);
        const auto r = run(*half, {make_row(random_tokens(rng, 20), {19}, true)});
        CHECK(r[0].snapshot.own_bytes() >= mh.slot_bytes);
        CHECK(half->state_saturations() == 0);
    }
}

SEPAL_TEST(snap_half, strict_mode_hit_equals_miss_with_half_state) {
    std::size_t differs_from_float = 0, trials = 0;
    for (const Precision p : {Precision::F32, Precision::F16}) {
        for (const std::size_t gemm_rows : {std::size_t{8}, std::size_t{24}}) {
            const auto half = engine_of({p, gemm_rows, true});
            const auto full = engine_of({p, gemm_rows, false});
            Rng rng(100 + gemm_rows);
            for (int trial = 0; trial < 6; ++trial) {
                const auto state = random_tokens(rng, 5 + rng.below(80));
                std::vector<std::vector<std::uint32_t>> children;
                for (std::size_t c = 0; c < 1 + rng.below(4); ++c) children.push_back(random_tokens(rng, 1 + rng.below(30)));
                const HitMiss h = hit_and_miss(*half, state, children);
                CHECK(same_hidden(h.miss, h.hit));
                // And a second hit, later: the stored snapshot is immutable.
                const HitMiss f = hit_and_miss(*full, state, children);
                CHECK(same_hidden(f.miss, f.hit)); // the float32 storage keeps its guarantee
                for (std::size_t i = 0; i < children.size(); ++i) {
                    ++trials;
                    if (!same_bits(h.hit[i].hidden, f.hit[i].hidden)) ++differs_from_float;
                    CHECK(all_finite(h.hit[i].hidden));
                }
            }
        }
    }
    // The test is not vacuous: the rounding does reach the continuations.
    CHECK(differs_from_float > trials / 2);
    std::cout << "  continuations that differ from float32 state: " << differs_from_float << " of " << trials << "\n";
}

SEPAL_TEST(snap_half, deep_in_call_chain_equals_the_chain_across_calls) {
    for (const Precision p : {Precision::F32, Precision::F16}) {
        const auto engine = engine_of({p, 8, true});
        Rng rng(77);
        // One call: S -> A -> B -> C (ParentRow), readout on the last token of each.
        std::vector<std::vector<std::uint32_t>> parts;
        for (int i = 0; i < 4; ++i) parts.push_back(random_tokens(rng, 3 + rng.below(40)));
        std::vector<EngineRow> rows;
        for (std::size_t i = 0; i < parts.size(); ++i) {
            EngineRow r = make_row(parts[i], {parts[i].size() - 1}, i + 1 == parts.size());
            if (i > 0) r.parent = ParentRow{i - 1};
            rows.push_back(std::move(r));
        }
        const auto one = run(*engine, rows);
        // The same, one call per link, each continuing the stored snapshot of the one before.
        DeviceSnapshot prev;
        std::vector<EngineRowResult> chain;
        for (std::size_t i = 0; i < parts.size(); ++i) {
            EngineRow r = make_row(parts[i], {parts[i].size() - 1}, true);
            if (i > 0) r.parent = prev;
            auto res = run(*engine, {r});
            prev = res[0].snapshot;
            chain.push_back(std::move(res[0]));
        }
        for (std::size_t i = 0; i < parts.size(); ++i) CHECK(same_bits(one[i].hidden, chain[i].hidden));
    }
}

SEPAL_TEST(snap_half, rows_without_a_stored_parent_do_not_see_the_rounding) {
    for (const Precision p : {Precision::F32, Precision::F16}) {
        const auto half = engine_of({p, 8, true});
        const auto full = engine_of({p, 8, false});
        Rng rng(9);
        for (int trial = 0; trial < 5; ++trial) {
            const auto t = random_tokens(rng, 3 + rng.below(100));
            const auto a = run(*half, {make_row(t, {t.size() - 1}, true)});
            const auto b = run(*full, {make_row(t, {t.size() - 1}, true)});
            CHECK(same_bits(a[0].hidden, b[0].hidden));
        }
    }
}

SEPAL_TEST(snap_half, closeness_and_a_deep_chain) {
    for (const Precision p : {Precision::F32, Precision::F16}) {
        const auto half = engine_of({p, 8, true});
        const auto full = engine_of({p, 8, false});
        Rng rng(2024);
        double worst_one = 0.0, worst_deep = 0.0;
        for (int trial = 0; trial < 5; ++trial) {
            const auto state = random_tokens(rng, 40);
            const auto child = random_tokens(rng, 9);
            const HitMiss a = hit_and_miss(*half, state, {child}), b = hit_and_miss(*full, state, {child});
            worst_one = std::max(worst_one, hidden_distance(a.hit[0].hidden, b.hit[0].hidden));
        }
        // 24 links, each continuing the previous snapshot (rounding is applied at every store).
        DeviceSnapshot sh, sf;
        for (int link = 0; link < 24; ++link) {
            const auto t = random_tokens(rng, 6 + rng.below(10));
            EngineRow rh = make_row(t, {t.size() - 1}, true), rf = rh;
            if (link > 0) {
                rh.parent = sh;
                rf.parent = sf;
            }
            auto a = run(*half, {rh}), b = run(*full, {rf});
            sh = a[0].snapshot;
            sf = b[0].snapshot;
            CHECK(all_finite(a[0].hidden));
            worst_deep = std::max(worst_deep, hidden_distance(a[0].hidden, b[0].hidden));
        }
        std::cout << "  " << name_of(p) << ": half vs float32 state, one hand-over " << worst_one << ", after 24 links " << worst_deep << "\n";
        CHECK(worst_one < 0.1);
        CHECK(worst_deep < 0.2);
        CHECK_EQ(half->state_saturations(), std::uint64_t{0});
    }
}

// ---- snap_tiers ----

namespace {

// Children of `snapshot` with a few shapes, one call; the readout of every token.
std::vector<EngineRowResult> probe(CudaEngine& engine, const DeviceSnapshot& snapshot, std::uint64_t seed) {
    Rng rng(seed);
    std::vector<EngineRow> rows;
    for (const std::size_t len : {std::size_t{1}, std::size_t{7}, std::size_t{19}}) {
        rows.push_back(child_of(make_row(random_tokens(rng, len), all_offsets(len)), snapshot));
    }
    return run(engine, rows);
}

bool same_bytes(std::span<const std::byte> a, std::span<const std::byte> b) {
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size()) == 0);
}

std::vector<char> slurp(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    return std::vector<char>((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

void spit(const fs::path& p, const std::vector<char>& bytes) {
    std::ofstream out(p, std::ios::binary | std::ios::trunc);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

// A chain of `links` snapshots (each continues the previous one's stored snapshot); the last is returned.
DeviceSnapshot make_chain(CudaEngine& engine, std::size_t links, std::uint64_t seed, bool gaps = false) {
    Rng rng(seed);
    DeviceSnapshot prev;
    for (std::size_t i = 0; i < links; ++i) {
        EngineRow r = make_row(random_tokens(rng, 5 + rng.below(30)), {}, true);
        if (i > 0) r.parent = prev;
        if (gaps && i % 2 == 1) r.start_position = prev.end_position() + 7; // a hole in the positions
        prev = run(engine, {r})[0].snapshot;
    }
    return prev;
}

} // namespace

SEPAL_TEST(snap_tiers, host_round_trip_is_bit_exact_for_every_setting) {
    for (const bool half_state : {false, true}) {
        for (const Precision p : {Precision::F32, Precision::F16}) {
            const auto engine = engine_of({p, 8, half_state});
            const DeviceSnapshot original = make_chain(*engine, 3, 11);
            for (const HostMemory mem : {HostMemory::Pageable, HostMemory::Pinned}) {
                const HostSnapshot host = engine->to_host(original, mem);
                CHECK(host.valid());
                CHECK_EQ(host.pinned(), mem == HostMemory::Pinned);
                CHECK_EQ(host.chain_length(), std::size_t{3});
                CHECK_EQ(host.tokens(), original.tokens());
                CHECK_EQ(host.end_position(), original.end_position());
                const DeviceSnapshot back = engine->to_device(host);
                CHECK_EQ(back.tokens(), original.tokens());
                CHECK_EQ(back.end_position(), original.end_position());
                CHECK_EQ(back.own_bytes(), original.own_bytes());
                CHECK(same_hidden(probe(*engine, original, 1), probe(*engine, back, 1)));
                // The device memory of the promoted copy is the same bytes: exporting it again gives the very same host bytes.
                const HostSnapshot again = engine->to_host(back, HostMemory::Pageable);
                CHECK(same_bytes(host.state_bytes(), again.state_bytes()));
                CHECK_EQ(host.bytes(), again.bytes());
            }
        }
    }
}

SEPAL_TEST(snap_tiers, file_round_trip_chain_gaps_and_deterministic_files) {
    const fs::path dir = scratch_dir();
    for (const bool half_state : {false, true}) {
        const auto engine = engine_of({Precision::F16, 8, half_state});
        for (const std::size_t links : {std::size_t{1}, std::size_t{2}, std::size_t{5}}) {
            const DeviceSnapshot original = make_chain(*engine, links, 40 + links, /*gaps=*/true);
            const fs::path a = dir / "a.snap", b = dir / "b.snap";
            engine->save(original, a);
            const DeviceSnapshot loaded = engine->load(a);
            CHECK_EQ(loaded.tokens(), original.tokens());
            CHECK_EQ(loaded.end_position(), original.end_position());
            CHECK(same_hidden(probe(*engine, original, 3), probe(*engine, loaded, 3)));
            // save(load(save(x))) is the same file: the file does not depend on heap addresses or padding.
            engine->save(loaded, b);
            CHECK(slurp(a) == slurp(b));
            // A host snapshot read from the file saves identically too.
            const HostSnapshot host = engine->load_host(a);
            CHECK_EQ(host.chain_length(), links);
            engine->save(host, b);
            CHECK(slurp(a) == slurp(b));
            // Deeper chains continue from the loaded snapshot.
            EngineRow next = make_row({1, 2, 3, 4, 5}, {4}, true);
            next.parent = loaded;
            EngineRow next_o = next;
            next_o.parent = original;
            const auto n1 = run(*engine, {next}), n2 = run(*engine, {next_o});
            CHECK(same_bits(n1[0].hidden, n2[0].hidden));
            CHECK(same_hidden(probe(*engine, n1[0].snapshot, 5), probe(*engine, n2[0].snapshot, 5)));
            fs::remove(a);
            fs::remove(b);
        }
    }
}

SEPAL_TEST(snap_tiers, a_second_engine_and_snapshots_that_outlive_things) {
    const fs::path file = scratch_dir() / "second.snap";
    HostSnapshot host;
    DeviceSnapshot loaded_in_b;
    std::vector<EngineRowResult> want;
    {
        const auto a = engine_of({Precision::F16, 8, true});
        const DeviceSnapshot original = make_chain(*a, 3, 61);
        want = probe(*a, original, 9);
        a->save(original, file);
        host = a->to_host(original, HostMemory::Pinned);
    } // engine a is gone; the host snapshot still holds its bytes
    CHECK(host.valid());
    std::size_t free_before = 0;
    {
        const auto b = engine_of({Precision::F16, 8, true}); // same weights, same options: a different engine
        free_before = device_free();
        loaded_in_b = b->load(file);
        CHECK(same_hidden(want, probe(*b, loaded_in_b, 9)));
        const DeviceSnapshot from_host = b->to_device(host);
        CHECK(same_hidden(want, probe(*b, from_host, 9)));
        // A snapshot made by engine a (it only exists as bytes now) is not a handle of b: a handle of another engine is refused.
        const auto c = engine_of({Precision::F16, 8, true});
        CHECK_THROWS(c->to_host(loaded_in_b), std::invalid_argument);
        CHECK_THROWS(run(*c, {child_of(make_row({1, 2}, {1}), loaded_in_b)}), std::invalid_argument);
    } // engine b is destroyed while loaded_in_b is alive: the handle keeps the pools; destroying it afterwards must be fine
    loaded_in_b = DeviceSnapshot();
    (void)free_before;
    fs::remove(file);
}

SEPAL_TEST(snap_tiers, memory_goes_back_to_zero) {
    const auto engine = engine_of({Precision::F32, 8, true});
    const EngineMemory start = engine->memory();
    const std::size_t free_start = device_free();
    const fs::path file = scratch_dir() / "mem.snap";
    {
        std::vector<DeviceSnapshot> held;
        for (int i = 0; i < 6; ++i) {
            const DeviceSnapshot s = make_chain(*engine, 1 + i % 3, 70 + static_cast<std::uint64_t>(i));
            engine->save(s, file);
            held.push_back(engine->load(file));
            held.push_back(engine->to_device(engine->to_host(s)));
        }
        CHECK(engine->memory().slots_in_use > 0);
        CHECK(engine->memory().kv_in_use_bytes > 0);
    }
    const EngineMemory end = engine->memory();
    CHECK_EQ(end.slots_in_use, std::size_t{0});
    CHECK_EQ(end.kv_in_use_bytes, std::size_t{0});
    CHECK_EQ(end.kv_largest_free_bytes, start.kv_largest_free_bytes);
    CHECK_EQ(device_free(), free_start);
    fs::remove(file);
}

SEPAL_TEST(snap_tiers, pools_full_gives_budget_exceeded_and_changes_nothing) {
    const auto engine = engine_of({Precision::F32, 8, false}, 31, /*slots=*/4);
    const DeviceSnapshot s = make_chain(*engine, 2, 80);
    const HostSnapshot host = engine->to_host(s);
    std::vector<DeviceSnapshot> held;
    for (;;) {
        try {
            held.push_back(engine->to_device(host));
        } catch (const BudgetExceeded&) {
            break;
        }
    }
    CHECK_EQ(held.size(), std::size_t{3}); // 4 slots, one is `s`
    const EngineMemory full = engine->memory();
    CHECK_THROWS(engine->to_device(host), BudgetExceeded);
    const EngineMemory after = engine->memory();
    CHECK_EQ(after.slots_in_use, full.slots_in_use);
    CHECK_EQ(after.kv_in_use_bytes, full.kv_in_use_bytes);
    held.pop_back();
    const DeviceSnapshot again = engine->to_device(host); // room again
    CHECK(same_hidden(probe(*engine, s, 2), probe(*engine, again, 2)));
    // A heap that cannot hold the blocks: fill the heap with big rows instead (slots stay free).
    const auto small_heap = engine_of({Precision::F32, 8, false}, 31, /*slots=*/400, /*budget_mib=*/50);
    Rng rng(4);
    std::vector<DeviceSnapshot> keep;
    for (;;) {
        try {
            keep.push_back(run(*small_heap, {make_row(random_tokens(rng, 400), {}, true)})[0].snapshot);
        } catch (const BudgetExceeded&) {
            break;
        }
    }
    const DeviceSnapshot big = keep.back();
    const HostSnapshot big_host = small_heap->to_host(big);
    keep.clear();
    keep.push_back(run(*small_heap, {make_row(random_tokens(rng, 400), {}, true)})[0].snapshot);
    const std::size_t slots_before = small_heap->memory().slots_in_use;
    std::vector<DeviceSnapshot> more;
    for (;;) {
        try {
            more.push_back(small_heap->to_device(big_host));
        } catch (const BudgetExceeded&) {
            break;
        }
    }
    CHECK_EQ(small_heap->memory().slots_in_use, slots_before + more.size());
}

SEPAL_TEST(snap_tiers, refusals) {
    const fs::path dir = scratch_dir();
    const fs::path file = dir / "r.snap";
    const auto engine = engine_of({Precision::F16, 8, true});
    const DeviceSnapshot s = make_chain(*engine, 2, 90);
    engine->save(s, file);

    CHECK_THROWS(engine->to_host(DeviceSnapshot()), std::invalid_argument);
    CHECK_THROWS(engine->to_device(HostSnapshot()), std::invalid_argument);
    CHECK_THROWS(engine->save(DeviceSnapshot(), file), std::invalid_argument);
    CHECK_THROWS(engine->load(dir / "does-not-exist.snap"), SnapshotFileError);
    CHECK_THROWS(engine->load(dir), SnapshotFileError); // a directory

    const auto expect_refused = [&](CudaEngine& e, const char* why, const char* fragment) {
        try {
            (void)e.load(file);
            CHECK(false);
            std::cout << "  NOT refused: " << why << "\n";
        } catch (const SnapshotFileError& err) {
            const std::string m = err.what();
            const bool found = m.find(fragment) != std::string::npos;
            CHECK(found);
            std::cout << "  refused (" << why << "): " << m << "\n";
        }
    };
    expect_refused(*engine_of({Precision::F16, 8, false}), "other state storage", "state format");
    expect_refused(*engine_of({Precision::F16, 8, true}, 32), "other weights", "other weights");
    expect_refused(*engine_of({Precision::F16, 16, true}), "other gemm_rows", "compute options");
    expect_refused(*engine_of({Precision::F32, 8, true}), "other precision", "");
    {
        // Another geometry.
        sepal::ModelConfig cfg = tiny_test_config();
        cfg.intermediate_size = 48;
        Knobs k;
        k.precision = Precision::F16;
        k.gemm_rows = 8;
        EngineOptions o = options_of(k);
        o.half_state = true;
        CudaEngine other(cfg, random_weights<float>(cfg, 31), o);
        expect_refused(other, "other geometry", "geometry");
    }
    {
        // A host snapshot of another kind cannot be promoted either.
        const auto f32 = engine_of({Precision::F16, 8, false});
        const HostSnapshot host = engine->to_host(s);
        CHECK_THROWS(f32->to_device(host), SnapshotFileError);
    }
    // The file itself is untouched by all that, and still loads.
    CHECK(engine->load(file).valid());
    fs::remove(file);
}

// ---- snap_fuzz ----

namespace {

using sepal::cuda::hash64;

// A small valid file and the engine that reads it.
struct Fixture {
    std::unique_ptr<CudaEngine> engine = engine_of({Precision::F32, 8, true});
    fs::path dir = scratch_dir();
    fs::path file = dir / "fz.snap";
    fs::path work = dir / "fz-work.snap";
    std::vector<char> good;

    Fixture() {
        Rng rng(33);
        EngineRow a = make_row(random_tokens(rng, 6), {}, true);
        const DeviceSnapshot sa = run(*engine, {a})[0].snapshot;
        EngineRow b = make_row(random_tokens(rng, 5), {}, true);
        b.parent = sa;
        const DeviceSnapshot sb = run(*engine, {b})[0].snapshot;
        engine->save(sb, file);
        good = slurp(file);
    }
    ~Fixture() {
        std::error_code ec;
        fs::remove(file, ec);
        fs::remove(work, ec);
    }

    // True when the engine refuses these bytes with SnapshotFileError; false when it accepts; fails the test on any other outcome.
    bool refused(const std::vector<char>& bytes) {
        spit(work, bytes);
        try {
            (void)engine->load_host(work);
            return false;
        } catch (const SnapshotFileError&) {
            return true;
        } catch (...) {
            CHECK(false); // anything else (bad_alloc, ...) is a bug
            return true;
        }
    }
};

void put32(std::vector<char>& b, std::size_t at, std::uint32_t v) { std::memcpy(b.data() + at, &v, 4); }
void put64(std::vector<char>& b, std::size_t at, std::uint64_t v) { std::memcpy(b.data() + at, &v, 8); }
std::uint64_t get64(const std::vector<char>& b, std::size_t at) {
    std::uint64_t v;
    std::memcpy(&v, b.data() + at, 8);
    return v;
}
void rehash_header(std::vector<char>& b) { put64(b, 160, hash64(b.data(), 160)); }
void rehash_payload(std::vector<char>& b) {
    const std::size_t body = 168;
    const std::size_t end = b.size() - 16;
    put64(b, end, hash64(b.data() + body, end - body, 0));
}

} // namespace

SEPAL_TEST(snap_fuzz, every_truncation_is_refused) {
    Fixture f;
    CHECK(!f.refused(f.good));
    const std::size_t n = f.good.size();
    std::size_t refused = 0;
    for (std::size_t len = 0; len < n; len += (len < 400 || len + 400 > n) ? 1 : 7) {
        refused += f.refused(std::vector<char>(f.good.begin(), f.good.begin() + static_cast<std::ptrdiff_t>(len))) ? 1 : 0;
    }
    std::vector<char> longer = f.good;
    longer.push_back(0);
    CHECK(f.refused(longer));
    std::cout << "  " << refused << " truncations tried on a file of " << n << " bytes, all refused\n";
}

SEPAL_TEST(snap_fuzz, every_flipped_byte_is_refused) {
    Fixture f;
    std::size_t refused = 0;
    for (std::size_t at = 0; at < f.good.size(); ++at) {
        std::vector<char> bad = f.good;
        bad[at] = static_cast<char>(bad[at] ^ 0x01);
        if (f.refused(bad)) ++refused; else { CHECK(false); std::cout << "  accepted a flip at byte " << at << "\n"; }
    }
    // A few multi-byte smashes.
    Rng rng(8);
    for (int i = 0; i < 300; ++i) {
        std::vector<char> bad = f.good;
        const std::size_t at = rng.below(bad.size());
        const std::size_t len = 1 + rng.below(std::min<std::size_t>(64, bad.size() - at));
        for (std::size_t k = 0; k < len; ++k) bad[at + k] = static_cast<char>(rng.below(256));
        if (bad == f.good) continue;
        CHECK(f.refused(bad));
    }
    std::cout << "  " << refused << " single-byte flips of " << f.good.size() << ", all refused\n";
}

SEPAL_TEST(snap_fuzz, lying_headers_with_valid_hashes_are_refused_before_allocating) {
    Fixture f;
    const std::uint64_t extremes[] = {0, 1, 2, 0x7FFFFFFFull, 0xFFFFFFFFull, 0x100000000ull, 0x7FFFFFFFFFFFFFFFull, 0x8000000000000000ull, 0xFFFFFFFFFFFFFFFFull};
    std::size_t tried = 0;
    for (std::size_t at = 8; at < 160; at += 4) {
        for (const std::uint64_t v : extremes) {
            std::vector<char> bad = f.good;
            if (at % 8 == 0) {
                if (get64(bad, at) == v) continue;
                put64(bad, at, v);
            } else {
                if (v > 0xFFFFFFFFull) continue;
                std::uint32_t old;
                std::memcpy(&old, bad.data() + at, 4);
                if (old == static_cast<std::uint32_t>(v)) continue;
                put32(bad, at, static_cast<std::uint32_t>(v));
            }
            rehash_header(bad);
            ++tried;
            if (!f.refused(bad)) {
                CHECK(false);
                std::cout << "  accepted a header with field at " << at << " = " << v << "\n";
            }
        }
    }
    // Lies in the block table, hashes made good again.
    const std::size_t table = 168;
    for (std::size_t entry = 0; entry < 2; ++entry) {
        for (const std::size_t field : {std::size_t{0}, std::size_t{4}, std::size_t{8}, std::size_t{16}}) {
            for (const std::uint64_t v : extremes) {
                std::vector<char> bad = f.good;
                const std::size_t at = table + entry * 24 + field;
                if (field == 0 || field == 4) {
                    if (v > 0xFFFFFFFFull) continue;
                    put32(bad, at, static_cast<std::uint32_t>(v));
                } else {
                    put64(bad, at, v);
                }
                if (bad == f.good) continue;
                if (entry == 0 && field == 8) continue; // the first block's start position is free: a gap in positions is legal
                rehash_payload(bad);
                ++tried;
                if (!f.refused(bad)) {
                    CHECK(false);
                    std::cout << "  accepted a table with entry " << entry << " field " << field << " = " << v << "\n";
                }
            }
        }
    }
    // Swapping the two blocks' entries (order matters: the payload is oldest first) must not slip through either.
    {
        std::vector<char> bad = f.good;
        for (std::size_t k = 0; k < 24; ++k) std::swap(bad[table + k], bad[table + 24 + k]);
        rehash_payload(bad);
        // The entries differ in tokens, so the sizes no longer agree with the payload layout the header describes... or they
        // happen to be equal in size; either way it must not crash. If accepted, the chain is simply the other order.
        (void)f.refused(bad);
    }
    std::cout << "  " << tried << " lying headers and tables, all refused\n";
}

SEPAL_TEST(snap_fuzz, garbage_and_odd_files) {
    Fixture f;
    CHECK(f.refused({}));
    CHECK(f.refused(std::vector<char>(1, 'x')));
    CHECK(f.refused(std::vector<char>(100000, 0)));
    Rng rng(12);
    for (int i = 0; i < 100; ++i) {
        std::vector<char> junk(rng.below(5000));
        for (char& c : junk) c = static_cast<char>(rng.below(256));
        CHECK(f.refused(junk));
    }
    // Right magic, the rest random.
    for (int i = 0; i < 100; ++i) {
        std::vector<char> junk(168 + rng.below(3000));
        for (char& c : junk) c = static_cast<char>(rng.below(256));
        std::memcpy(junk.data(), "SEPALSNP", 8);
        CHECK(f.refused(junk));
    }
    // A valid header + table of a file that is far too short for what they say, hash made good.
    {
        std::vector<char> bad(f.good.begin(), f.good.begin() + 168 + 48);
        CHECK(f.refused(bad));
    }
}

// ---- snap_default: what the engine reserves when no budget is given ----

SEPAL_TEST(snap_default, the_default_pools_hold_pool_snapshots_and_the_next_one_is_refused) {
    for (const bool half_state : {false, true}) {
        EngineOptions o; // all defaults but the storage: no budget, no slot count
        o.half_state = half_state;
        o.max_call_tokens = 4096;
        CudaEngine engine(tiny_test_config(), tiny_weights(31), o);
        const EngineMemory m = engine.memory();
        CHECK_EQ(m.slots_total, std::size_t{64});
        CHECK(m.reserved_bytes() < m.device_free_at_start / 2);
        std::cout << "  " << (half_state ? "binary16" : "float32") << " state: reserved " << m.reserved_bytes() / 1024 << " KiB of " << m.device_free_at_start / (1 << 20)
                  << " MiB free (budget " << m.budget_bytes / 1024 << " KiB)\n";
        Rng rng(5);
        std::vector<DeviceSnapshot> held;
        for (int i = 0; i < 64; ++i) held.push_back(run(engine, {make_row(tokens(rng, 1000), {}, true)})[0].snapshot); // 1000 <= pool_snapshot_tokens
        CHECK_THROWS(run(engine, {make_row(tokens(rng, 1000), {}, true)}), BudgetExceeded);
        held.pop_back();
        CHECK(run(engine, {make_row(tokens(rng, 1000), {}, true)})[0].snapshot.valid());
        // A bigger count asked for explicitly.
        EngineOptions big = o;
        big.pool_snapshots = 200;
        CudaEngine more(tiny_test_config(), tiny_weights(31), big);
        CHECK_EQ(more.memory().slots_total, std::size_t{200});
        // state_slots replaces the count.
        EngineOptions few = o;
        few.state_slots = 7;
        CudaEngine seven(tiny_test_config(), tiny_weights(31), few);
        CHECK_EQ(seven.memory().slots_total, std::size_t{7});
        CHECK(seven.memory().reserved_bytes() < m.reserved_bytes());
    }
}

SEPAL_TEST(snap_default_big, pool_snapshots_zero_takes_most_of_the_free_memory_as_before) {
    if (!environment("SEPAL_BIG_ENGINE_TEST")) {
        std::cout << "SKIPPED: real-model test; this engine takes 85% of the free device memory: set SEPAL_BIG_ENGINE_TEST=1 and hold the GPU lock\n";
        return;
    }
    EngineOptions o;
    o.pool_snapshots = 0;
    o.max_call_tokens = 1024;
    CudaEngine engine(tiny_test_config(), tiny_weights(31), o);
    const EngineMemory m = engine.memory();
    CHECK(m.reserved_bytes() > m.device_free_at_start / 100 * 80);
    CHECK(m.reserved_bytes() <= m.device_free_at_start / 100 * 85);
    std::cout << "  pool_snapshots 0: reserved " << m.reserved_bytes() / (1 << 20) << " MiB of " << m.device_free_at_start / (1 << 20) << " MiB free, " << m.slots_total << " slots\n";
}
