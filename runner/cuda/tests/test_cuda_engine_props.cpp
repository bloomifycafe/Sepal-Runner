// Properties of the GPU engine that do not need a reference: bit-invariance in strict mode, the drift of natural
// mode, repeatability, memory and lifetime of snapshots, and refused requests. Tiny geometry, random weights.

#include <cuda_runtime.h>

#include <map>

#include "engine_test_util.hpp"

using namespace sepal;
using namespace engine_test;
using sepal::cuda::BudgetExceeded;
using sepal::cuda::CudaEngine;
using sepal::cuda::DeviceSnapshot;
using sepal::cuda::EngineOptions;
using sepal::cuda::EngineRow;
using sepal::cuda::EngineRowResult;
using sepal::cuda::ParentRow;
using sepal::cuda::Precision;

namespace {

EngineOptions options_for(Precision precision, std::size_t gemm_rows) {
    EngineOptions o;
    o.precision = precision;
    o.gemm_rows = gemm_rows;
    o.memory_budget_bytes = std::size_t{192} << 20;
    o.max_call_tokens = 4096;
    o.max_call_rows = 256;
    o.state_slots = 128;
    return o;
}

const Weights<float>& weights_of(std::uint64_t seed) {
    static std::map<std::uint64_t, Weights<float>> cache;
    auto it = cache.find(seed);
    if (it == cache.end()) it = cache.emplace(seed, random_weights<float>(tiny_test_config(), seed)).first;
    return it->second;
}

std::vector<EngineRowResult> run(CudaEngine& engine, const std::vector<EngineRow>& rows) {
    return engine.run(std::span<const EngineRow>(rows));
}

std::vector<std::uint32_t> tokens(Rng& rng, std::size_t n) {
    std::vector<std::uint32_t> t(n);
    for (auto& v : t) v = static_cast<std::uint32_t>(rng.below(64));
    return t;
}

EngineRow make_row(std::vector<std::uint32_t> t, std::vector<std::size_t> readouts, bool keep = false) {
    EngineRow r;
    r.token_ids = std::move(t);
    r.readout_offsets = std::move(readouts);
    r.keep_snapshot = keep;
    return r;
}

// Inserts the rows of `subject` (whose ParentRow indices count within `subject`) at position `pos` of `others`.
std::vector<EngineRow> splice(std::vector<EngineRow> others, const std::vector<EngineRow>& subject, std::size_t pos) {
    const std::size_t k = subject.size();
    for (EngineRow& r : others) {
        if (auto* p = std::get_if<ParentRow>(&r.parent); p != nullptr && p->index >= pos) p->index += k;
    }
    std::vector<EngineRow> out(others.begin(), others.begin() + static_cast<std::ptrdiff_t>(pos));
    for (EngineRow r : subject) {
        if (auto* p = std::get_if<ParentRow>(&r.parent)) p->index += pos;
        out.push_back(std::move(r));
    }
    out.insert(out.end(), others.begin() + static_cast<std::ptrdiff_t>(pos), others.end());
    return out;
}

// Rows of other requests to share a call with: a random request, possibly continuing snapshots of `pool`.
std::vector<EngineRow> make_others(Rng& rng, const std::vector<Stored>& pool, std::size_t max_len = 200) {
    FuzzOptions fo;
    fo.max_len = max_len;
    fo.max_groups = 3;
    return make_request(rng, pool, fo).engine;
}

// The same rows in another order (parents keep pointing at their rows). order[i] is the old index of new row i.
std::vector<EngineRow> permute(const std::vector<EngineRow>& rows, const std::vector<std::size_t>& order) {
    std::vector<std::size_t> new_index(rows.size());
    for (std::size_t i = 0; i < order.size(); ++i) new_index[order[i]] = i;
    std::vector<EngineRow> out;
    for (const std::size_t old : order) {
        EngineRow r = rows[old];
        if (auto* p = std::get_if<ParentRow>(&r.parent)) p->index = new_index[p->index];
        out.push_back(std::move(r));
    }
    return out;
}

// A pool of device snapshots from real calls, for `others` to continue.
std::vector<Stored> make_pool(CudaEngine& engine, Rng& rng, std::size_t count) {
    std::vector<Stored> pool;
    std::vector<EngineRow> rows;
    for (std::size_t i = 0; i < count; ++i) rows.push_back(make_row(tokens(rng, 5 + rng.below(60)), {}, true));
    const auto results = run(engine, rows);
    for (std::size_t i = 0; i < count; ++i) {
        Stored s;
        s.device = results[i].snapshot;
        s.end_position = s.device.end_position();
        s.tokens = s.device.tokens();
        pool.push_back(std::move(s));
    }
    return pool;
}

struct Subject {
    std::vector<EngineRow> rows;      // ParentRow indices within the subject
    std::vector<std::size_t> watched; // rows whose hidden states are compared
};

// Three subjects: a plain row; a state with two branches; a row continuing a stored snapshot.
std::vector<Subject> make_subjects(Rng& rng, const DeviceSnapshot& stored) {
    std::vector<Subject> subjects(3);
    subjects[0].rows = {make_row(tokens(rng, 23), {0, 11, 22})};
    subjects[0].watched = {0};
    EngineRow b1 = make_row(tokens(rng, 9), {8});
    b1.parent = ParentRow{0};
    EngineRow b2 = make_row(tokens(rng, 25), {0, 24});
    b2.parent = ParentRow{0};
    subjects[1].rows = {make_row(tokens(rng, 70), {}), b1, b2};
    subjects[1].watched = {1, 2};
    EngineRow c = make_row(tokens(rng, 15), {14, 3});
    c.parent = stored;
    subjects[2].rows = {c};
    subjects[2].watched = {0};
    return subjects;
}

std::vector<std::vector<float>> watched_hidden(const std::vector<EngineRowResult>& results, std::size_t pos, const Subject& s) {
    std::vector<std::vector<float>> out;
    for (const std::size_t w : s.watched) out.push_back(results[pos + w].hidden);
    return out;
}

bool same(const std::vector<std::vector<float>>& a, const std::vector<std::vector<float>>& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (!same_bits(a[i], b[i])) return false;
    }
    return true;
}

double drift(const std::vector<std::vector<float>>& a, const std::vector<std::vector<float>>& b) {
    double worst = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) worst = std::max(worst, max_abs_diff(a[i].data(), b[i].data(), a[i].size()));
    return worst;
}

// Every subject alone, then among random other rows at random places: returns {bit-identical count, trials, worst drift}.
struct Invariance {
    std::size_t trials = 0, identical = 0;
    double worst_drift = 0.0;
};

Invariance measure_invariance(CudaEngine& engine, std::uint64_t seed, std::size_t contexts) {
    Rng rng(seed);
    std::vector<Stored> pool = make_pool(engine, rng, 4);
    // A snapshot for the third subject, made in its own call.
    const auto made = run(engine, {make_row(tokens(rng, 41), {}, true)});
    const std::vector<Subject> subjects = make_subjects(rng, made[0].snapshot);
    Invariance inv;
    for (const Subject& s : subjects) {
        const auto alone = watched_hidden(run(engine, s.rows), 0, s);
        for (std::size_t c = 0; c < contexts; ++c) {
            const std::vector<EngineRow> others = make_others(rng, pool, c % 3 == 0 ? 30 : 200);
            const std::size_t pos = rng.below(others.size() + 1);
            const auto among = run(engine, splice(others, s.rows, pos));
            const auto got = watched_hidden(among, pos, s);
            ++inv.trials;
            if (same(alone, got)) ++inv.identical;
            inv.worst_drift = std::max(inv.worst_drift, drift(alone, got));
            for (const auto& h : got) CHECK(all_finite(h));
        }
    }
    return inv;
}

const char* name_of(Precision p) { return p == Precision::F16 ? "float16" : "float32"; }

} // namespace

SEPAL_TEST(engine_strict, a_row_is_bit_identical_alone_and_among_other_rows) {
    // Different multiply block sizes, including ones whose blocks do not start on 256-byte boundaries.
    for (const Precision precision : {Precision::F32, Precision::F16}) {
        for (const std::size_t t : {std::size_t{8}, std::size_t{16}, std::size_t{24}, std::size_t{32}, std::size_t{64}}) {
            CudaEngine engine(tiny_test_config(), weights_of(31), options_for(precision, t));
            const Invariance inv = measure_invariance(engine, 900 + t, 12);
            std::cout << "  strict " << name_of(precision) << ", gemm_rows " << t << ": " << inv.identical << " of " << inv.trials
                      << " rows bit-identical alone vs among other rows (largest difference " << inv.worst_drift << ")\n";
            CHECK_EQ(inv.identical, inv.trials);
        }
    }
}

SEPAL_TEST(engine_strict, a_branch_from_a_stored_snapshot_equals_the_branch_run_with_its_state) {
    for (const Precision precision : {Precision::F32, Precision::F16}) {
        CudaEngine engine(tiny_test_config(), weights_of(32), options_for(precision, 16));
        Rng rng(77);
        const std::vector<Stored> pool = make_pool(engine, rng, 3);
        std::size_t trials = 0, identical = 0, snapshot_chains = 0;
        for (std::size_t trial = 0; trial < 25; ++trial) {
            const std::vector<std::uint32_t> state = tokens(rng, 20 + rng.below(250));
            const std::size_t nb = 1 + rng.below(8);
            std::vector<std::vector<std::uint32_t>> suffixes;
            for (std::size_t b = 0; b < nb; ++b) suffixes.push_back(tokens(rng, 1 + rng.below(30)));

            // Miss: the state and its branches in one call, among other rows.
            std::vector<EngineRow> family = {make_row(state, {})};
            for (const auto& s : suffixes) {
                EngineRow b = make_row(s, {s.size() - 1});
                b.parent = ParentRow{0};
                family.push_back(b);
            }
            const std::size_t pos = rng.below(5);
            std::vector<EngineRow> others;
            for (std::size_t i = 0; i < pos; ++i) {
                EngineRow o = make_row(tokens(rng, 1 + rng.below(100)), {0});
                if (rng.chance(50)) o.parent = pool[rng.below(pool.size())].device;
                others.push_back(o);
            }
            const auto miss = run(engine, splice(others, family, pos));

            // Hit: the state alone (kept), then the branches continuing it, in another order among other rows.
            const auto stored = run(engine, {make_row(state, {}, true)});
            std::vector<EngineRow> branches;
            for (const auto& s : suffixes) {
                EngineRow b = make_row(s, {s.size() - 1});
                b.parent = stored[0].snapshot;
                branches.push_back(b);
            }
            std::vector<std::size_t> order(nb);
            for (std::size_t i = 0; i < nb; ++i) order[i] = i;
            for (std::size_t i = nb; i > 1; --i) std::swap(order[i - 1], order[rng.below(i)]);
            std::vector<EngineRow> permuted;
            for (const std::size_t o : order) permuted.push_back(branches[o]);
            // The permuted branches come first in the call, then other rows.
            const auto hit = run(engine, splice(make_others(rng, pool, 60), permuted, 0));
            for (std::size_t i = 0; i < nb; ++i) {
                ++trials;
                if (same_bits(miss[pos + 1 + order[i]].hidden, hit[i].hidden)) ++identical;
            }

            // And a branch of a branch: snapshot of a branch, continued later, equals the chain run at once.
            if (trial % 5 == 0) {
                EngineRow mid = make_row(suffixes[0], {}, true);
                mid.parent = stored[0].snapshot;
                const auto kept = run(engine, {mid});
                EngineRow leaf = make_row(tokens(rng, 7), {6});
                leaf.parent = kept[0].snapshot;
                const auto leaf_alone = run(engine, {leaf});
                EngineRow m2 = make_row(suffixes[0], {});
                m2.parent = ParentRow{0};
                EngineRow l2 = leaf;
                l2.parent = ParentRow{1};
                std::vector<EngineRow> chain = {make_row(state, {}), m2, l2};
                const auto at_once = run(engine, chain);
                ++snapshot_chains;
                CHECK(same_bits(leaf_alone[0].hidden, at_once[2].hidden));
            }
        }
        std::cout << "  strict " << name_of(precision) << ": " << identical << " of " << trials << " branches from a stored snapshot are bit-identical to the same branch run with its state in one call ("
                  << snapshot_chains << " three-deep chains checked as well)\n";
        CHECK_EQ(identical, trials);
    }
}

SEPAL_TEST(engine_strict, permuting_the_rows_of_a_call_changes_nothing) {
    for (const Precision precision : {Precision::F32, Precision::F16}) {
        CudaEngine engine(tiny_test_config(), weights_of(33), options_for(precision, 32));
        Rng rng(88);
        const std::vector<Stored> pool = make_pool(engine, rng, 3);
        std::size_t rows_compared = 0, identical = 0, state_probes = 0, state_identical = 0;
        for (std::size_t trial = 0; trial < 20; ++trial) {
            std::vector<EngineRow> rows = make_others(rng, pool);
            for (EngineRow& r : rows) r.keep_snapshot = true; // so the state each row leaves can be probed afterwards
            const auto base = run(engine, rows);
            for (std::size_t round = 0; round < 3; ++round) {
                std::vector<std::size_t> order(rows.size());
                for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
                for (std::size_t i = order.size(); i > 1; --i) std::swap(order[i - 1], order[rng.below(i)]);
                const auto shuffled = run(engine, permute(rows, order));
                for (std::size_t i = 0; i < order.size(); ++i) {
                    ++rows_compared;
                    if (same_bits(base[order[i]].hidden, shuffled[i].hidden)) ++identical;
                    // The snapshot each row left must be the same state: continue both with the same probe row.
                    EngineRow probe = make_row(tokens(rng, 6), {5});
                    probe.parent = base[order[i]].snapshot;
                    EngineRow probe2 = probe;
                    probe2.parent = shuffled[i].snapshot;
                    const auto p1 = run(engine, {probe});
                    const auto p2 = run(engine, {probe2});
                    ++state_probes;
                    if (same_bits(p1[0].hidden, p2[0].hidden)) ++state_identical;
                }
            }
        }
        std::cout << "  strict " << name_of(precision) << ": " << identical << " of " << rows_compared << " rows bit-identical after permuting the call; " << state_identical
                  << " of " << state_probes << " snapshots left behind continue to bit-identical answers\n";
        CHECK_EQ(identical, rows_compared);
        CHECK_EQ(state_identical, state_probes);
    }
}

SEPAL_TEST(engine_strict, repeated_runs_are_bit_identical) {
    for (const std::size_t t : {std::size_t{0}, std::size_t{32}}) {
        for (const Precision precision : {Precision::F32, Precision::F16}) {
            CudaEngine engine(tiny_test_config(), weights_of(34), options_for(precision, t));
            Rng rng(99);
            const std::vector<Stored> pool = make_pool(engine, rng, 3);
            std::size_t compared = 0, identical = 0;
            for (std::size_t trial = 0; trial < 10; ++trial) {
                const std::vector<EngineRow> rows = make_others(rng, pool);
                const auto first = run(engine, rows);
                for (std::size_t rep = 0; rep < 4; ++rep) {
                    const auto again = run(engine, rows);
                    for (std::size_t r = 0; r < rows.size(); ++r) {
                        ++compared;
                        if (same_bits(first[r].hidden, again[r].hidden)) ++identical;
                    }
                }
            }
            std::cout << "  " << (t == 0 ? "natural" : "strict") << " " << name_of(precision) << ": " << identical << " of " << compared << " repeated rows bit-identical\n";
            CHECK_EQ(identical, compared);
        }
    }
}

SEPAL_TEST(engine_natural, drift_with_the_composition_of_the_call) {
    for (const Precision precision : {Precision::F32, Precision::F16}) {
        CudaEngine engine(tiny_test_config(), weights_of(35), options_for(precision, 0));
        const Invariance inv = measure_invariance(engine, 1234, 12);
        std::cout << "  natural " << name_of(precision) << " (tiny geometry): " << inv.identical << " of " << inv.trials
                  << " rows happened to be bit-identical alone vs among other rows; largest difference " << std::scientific << inv.worst_drift << std::fixed << "\n";
        CHECK(inv.worst_drift < 1e-2);
    }
}

// ---- memory and lifetime ----

namespace {

EngineOptions small_budget(std::size_t slots) {
    EngineOptions o;
    o.precision = Precision::F32;
    o.max_call_tokens = 1024;
    o.max_call_rows = 64;
    o.state_slots = slots;
    // The cuBLAS workspace (32 MiB) and the row tables (8 MiB) come first; the rest is for the pools.
    o.memory_budget_bytes = std::size_t{44} << 20;
    return o;
}

std::size_t device_free() {
    std::size_t free_bytes = 0, total = 0;
    cudaMemGetInfo(&free_bytes, &total);
    return free_bytes;
}

} // namespace

SEPAL_TEST(engine_memory, thousands_of_calls_with_snapshots_created_and_released_show_no_growth) {
    CudaEngine engine(tiny_test_config(), weights_of(36), small_budget(24));
    Rng rng(4321);
    std::vector<DeviceSnapshot> ring;
    const auto one_call = [&] {
        std::vector<EngineRow> rows;
        const std::size_t groups = 1 + rng.below(3);
        for (std::size_t g = 0; g < groups; ++g) {
            EngineRow state = make_row(tokens(rng, 10 + rng.below(120)), {}, rng.chance(60));
            if (!ring.empty() && rng.chance(50)) state.parent = ring[rng.below(ring.size())];
            const std::size_t at = rows.size();
            rows.push_back(state);
            for (std::size_t b = 0; b < 1 + rng.below(3); ++b) {
                EngineRow branch = make_row(tokens(rng, 1 + rng.below(20)), {0}, rng.chance(10));
                branch.parent = ParentRow{at};
                rows.push_back(branch);
            }
        }
        const auto results = run(engine, rows);
        for (std::size_t r = 0; r < rows.size(); ++r) {
            if (results[r].snapshot.valid()) ring.push_back(results[r].snapshot);
        }
        while (ring.size() > 6) ring.erase(ring.begin() + static_cast<std::ptrdiff_t>(rng.below(ring.size())));
    };
    for (int i = 0; i < 50; ++i) one_call(); // warm-up: lazily loaded modules and the like
    const std::size_t free_before = device_free();
    const cuda::EngineMemory before = engine.memory();
    engine.reset_peaks();
    std::size_t max_slots = 0, max_kv = 0;
    const int calls = 3000;
    for (int i = 0; i < calls; ++i) {
        one_call();
        const cuda::EngineMemory m = engine.memory();
        max_slots = std::max(max_slots, m.slots_in_use);
        max_kv = std::max(max_kv, m.kv_in_use_bytes);
    }
    const std::size_t free_after = device_free();
    const cuda::EngineMemory during = engine.memory();
    ring.clear();
    const cuda::EngineMemory after = engine.memory();
    std::cout << "  " << calls << " calls with snapshots created and released: device memory free " << free_before << " -> " << free_after
              << " bytes; slots in use at most " << max_slots << " of " << after.slots_total << ", keys and values at most " << max_kv
              << " of " << after.kv_heap_bytes << " bytes; after releasing everything: " << after.slots_in_use << " slots and " << after.kv_in_use_bytes
              << " bytes in use, largest free block " << after.kv_largest_free_bytes << " of " << after.kv_heap_bytes << "\n";
    CHECK_EQ(free_after, free_before);
    CHECK(max_slots <= 6 + 6 + 12); // ring + the call's own rows
    CHECK_EQ(after.slots_in_use, 0u);
    CHECK_EQ(after.kv_in_use_bytes, 0u);
    CHECK_EQ(after.kv_largest_free_bytes, after.kv_heap_bytes); // fully merged: no fragmentation left behind
    CHECK_EQ(before.kv_heap_bytes, during.kv_heap_bytes);
}

SEPAL_TEST(engine_memory, exceeding_the_budget_fails_cleanly_and_the_engine_keeps_working) {
    CudaEngine engine(tiny_test_config(), weights_of(37), small_budget(6));
    Rng rng(555);
    const std::vector<EngineRow> probe = {make_row(tokens(rng, 40), {39, 10})};
    const auto baseline = run(engine, probe);

    // More snapshots than slots in one call.
    std::vector<EngineRow> too_many;
    for (int i = 0; i < 7; ++i) too_many.push_back(make_row(tokens(rng, 5), {0}, true));
    bool threw = false;
    try {
        run(engine, too_many);
    } catch (const BudgetExceeded& e) {
        threw = true;
        std::cout << "  slots: " << e.what() << "\n";
        CHECK(std::string(e.what()).find("state slots") != std::string::npos);
    }
    CHECK(threw);
    CHECK_EQ(engine.memory().slots_in_use, 0u);
    CHECK_EQ(engine.memory().kv_in_use_bytes, 0u);
    CHECK(same_bits(run(engine, probe)[0].hidden, baseline[0].hidden));

    // Too many snapshots held across calls.
    std::vector<DeviceSnapshot> held;
    threw = false;
    try {
        for (int i = 0; i < 7; ++i) held.push_back(run(engine, {make_row(tokens(rng, 5), {}, true)})[0].snapshot);
    } catch (const BudgetExceeded&) {
        threw = true;
    }
    CHECK(threw);
    CHECK_EQ(held.size(), 6u);
    CHECK_EQ(engine.memory().slots_in_use, 6u);
    CHECK(same_bits(run(engine, probe)[0].hidden, baseline[0].hidden)); // a call needing no slot still works
    held.pop_back();
    CHECK(run(engine, {make_row(tokens(rng, 5), {}, true)})[0].snapshot.valid()); // and a freed slot is reusable
    held.clear();

    // Keys and values: a row of 1000 tokens takes 4 layers x 256 bytes per token... until the heap is full.
    CudaEngine big(tiny_test_config(), weights_of(37), [] {
        EngineOptions o = small_budget(64);
        return o;
    }());
    held.clear();
    const std::size_t heap = big.memory().kv_heap_bytes;
    threw = false;
    std::string message;
    try {
        for (std::size_t i = 0; i < 4 * heap / (1000 * 256) + 8; ++i) held.push_back(run(big, {make_row(tokens(rng, 1000), {}, true)})[0].snapshot);
    } catch (const BudgetExceeded& e) {
        threw = true;
        message = e.what();
    }
    CHECK(threw);
    std::cout << "  keys and values: " << held.size() << " snapshots of 1000 tokens fit in a heap of " << heap << " bytes, then: " << message << "\n";
    CHECK(message.find("keys and values") != std::string::npos);
    const std::vector<EngineRow> small = {make_row(tokens(rng, 7), {6})};
    const auto before_release = run(big, small); // 7 tokens still fit in what is left, or throws; either is clean
    (void)before_release;
    CHECK_EQ(big.memory().slots_in_use, held.size());
    held.pop_back();
    held.pop_back();
    CHECK(run(big, {make_row(tokens(rng, 1000), {5}, true)})[0].snapshot.valid()); // released space is reusable
    held.clear();
    CHECK_EQ(big.memory().kv_in_use_bytes, 0u);
    CHECK_EQ(big.memory().kv_largest_free_bytes, big.memory().kv_heap_bytes);
}

SEPAL_TEST(engine_memory, snapshots_outlive_the_handles_of_their_parents_and_the_engine) {
    const Weights<float>& weights = weights_of(38);
    CudaEngine engine(tiny_test_config(), weights, options_for(Precision::F32, 16));
    Rng rng(66);
    const std::vector<std::uint32_t> ta = tokens(rng, 60), tb = tokens(rng, 33), tc = tokens(rng, 21), td = tokens(rng, 9);

    // The chain A -> B -> C -> D in one call...
    EngineRow a = make_row(ta, {}), b = make_row(tb, {}), c = make_row(tc, {}), d = make_row(td, {8, 0});
    b.parent = ParentRow{0};
    c.parent = ParentRow{1};
    d.parent = ParentRow{2};
    const auto whole = run(engine, {a, b, c, d});

    // ...and in four calls, dropping every handle but the last before the next call.
    DeviceSnapshot snap;
    std::uint64_t total = 0;
    for (const auto* t : {&ta, &tb, &tc}) {
        EngineRow r = make_row(*t, {}, true);
        if (snap.valid()) r.parent = snap;
        const auto res = run(engine, {r});
        snap = res[0].snapshot; // the previous handle is released here; the new snapshot keeps its ancestors alive
        total += t->size();
        CHECK_EQ(snap.tokens(), total);
    }
    // Only the snapshot of C is held. Its slot and its keys and values are in use; the ancestors' slots are free.
    CHECK_EQ(engine.memory().slots_in_use, 1u);
    EngineRow last = d;
    last.parent = snap;
    const auto stepped = run(engine, {last});
    CHECK(same_bits(whole[3].hidden, stepped[0].hidden));
    // Run again after more traffic has reused the freed memory.
    for (int i = 0; i < 20; ++i) run(engine, {make_row(tokens(rng, 80), {3}, false)});
    CHECK(same_bits(whole[3].hidden, run(engine, {last})[0].hidden));

    // A snapshot outliving the engine that made it is released without harm.
    DeviceSnapshot orphan;
    {
        CudaEngine temporary(tiny_test_config(), weights, options_for(Precision::F32, 0));
        orphan = run(temporary, {make_row(tokens(rng, 30), {}, true)})[0].snapshot;
        CHECK(orphan.valid());
    }
    CHECK_EQ(orphan.tokens(), 30u);
    orphan = DeviceSnapshot();
    snap = DeviceSnapshot();
    last.parent = std::monostate{}; // the row held a copy of the handle too
    CHECK_EQ(engine.memory().slots_in_use, 0u);
    CHECK_EQ(engine.memory().kv_in_use_bytes, 0u);
}

// ---- refused requests ----

SEPAL_TEST(engine_negative, requests_that_cannot_run_are_refused_and_leave_nothing_behind) {
    const ModelConfig cfg = tiny_test_config();
    CudaEngine engine(cfg, weights_of(39), options_for(Precision::F32, 0));
    Rng rng(11);
    const std::vector<EngineRow> probe = {make_row(tokens(rng, 30), {29})};
    const auto baseline = run(engine, probe);

    const auto refused = [&](const char* what, std::vector<EngineRow> rows) {
        bool threw = false;
        try {
            run(engine, rows);
        } catch (const std::invalid_argument& e) {
            threw = true;
            std::cout << "  refused (" << what << "): " << e.what() << "\n";
        } catch (const std::exception& e) {
            std::cout << "  " << what << ": wrong exception type: " << e.what() << "\n";
        }
        CHECK(threw);
        CHECK_EQ(engine.memory().slots_in_use, 0u);
        CHECK_EQ(engine.memory().kv_in_use_bytes, 0u);
    };

    refused("token id equal to the vocabulary size", {make_row({1, 2, static_cast<std::uint32_t>(cfg.vocab_size)}, {0})});
    refused("token id 0xFFFFFFFF", {make_row({0xFFFFFFFFu}, {0})});
    refused("empty row", {make_row({}, {})});
    refused("empty row among good ones", {make_row({1, 2, 3}, {2}), make_row({}, {}), make_row({4}, {0})});
    refused("readout past the end", {make_row({1, 2, 3}, {3})});
    {
        EngineRow a = make_row({1, 2}, {1}), b = make_row({3, 4}, {1});
        a.parent = ParentRow{1};
        b.parent = ParentRow{0};
        refused("two rows that are each other's parent", {a, b});
    }
    {
        EngineRow a = make_row({1, 2}, {1});
        a.parent = ParentRow{0};
        refused("a row that is its own parent", {a});
    }
    {
        EngineRow a = make_row({1, 2}, {1}), b = make_row({3, 4}, {1}), c = make_row({5}, {0});
        a.parent = ParentRow{2};
        b.parent = ParentRow{0};
        c.parent = ParentRow{1};
        refused("a cycle of three", {a, b, c});
    }
    {
        EngineRow a = make_row({1, 2}, {1});
        a.parent = ParentRow{5};
        refused("a parent row outside the call", {a});
    }
    {
        EngineRow a = make_row({1, 2}, {1});
        a.parent = DeviceSnapshot();
        refused("an empty snapshot handle as parent", {a});
    }
    {
        // A snapshot from another engine of the same geometry, and from an engine of another geometry.
        CudaEngine other(cfg, weights_of(39), options_for(Precision::F32, 0));
        const auto made = run(other, {make_row({1, 2, 3}, {}, true)});
        EngineRow a = make_row({1, 2}, {1});
        a.parent = made[0].snapshot;
        refused("a snapshot of another engine, same geometry", {a});
        ModelConfig other_cfg = cfg;
        other_cfg.hidden_size = 32;
        other_cfg.intermediate_size = 48;
        CudaEngine different(other_cfg, random_weights<float>(other_cfg, 3), options_for(Precision::F32, 0));
        const auto made2 = run(different, {make_row({1, 2, 3}, {}, true)});
        EngineRow b = make_row({1, 2}, {1});
        b.parent = made2[0].snapshot;
        refused("a snapshot of an engine with another geometry", {b});
    }
    {
        std::vector<EngineRow> many(300, make_row({1}, {0}));
        refused("more rows than max_call_rows", many);
        refused("more tokens than max_call_tokens", {make_row(std::vector<std::uint32_t>(5000, 1), {0})});
    }
    CHECK(run(engine, {}).empty());
    // The engine still gives the same answer after all that.
    CHECK(same_bits(run(engine, probe)[0].hidden, baseline[0].hidden));

    // Configuration errors at creation.
    EngineOptions bad = options_for(Precision::F32, 0);
    bad.gemm_rows = 5;
    CHECK_THROWS(CudaEngine(cfg, weights_of(39), bad), std::invalid_argument);
    bad = options_for(Precision::F32, 0);
    bad.memory_budget_bytes = std::size_t{1} << 20;
    CHECK_THROWS(CudaEngine(cfg, weights_of(39), bad), BudgetExceeded);
    bad.memory_budget_bytes = std::size_t{1} << 50;
    CHECK_THROWS(CudaEngine(cfg, weights_of(39), bad), BudgetExceeded);
}
