// Seeded randomised soak of the engine on the tiny geometry: six engines run the same random requests (rows of every length,
// parent rows in any order, stored snapshots, explicit positions, readouts, captures, snapshots kept and released) and are
// checked against each other and against the CPU float32 reference after every call:
//
//   A  float32 natural                         vs the CPU reference            (rounding noise of random weights; bar 8e-3, short rows reach 4e-3)
//   B  float32 strict 8, small pools           vs C, bit for bit (B is refused with BudgetExceeded now and then)
//   C  float32 strict 8, roomy pools, no last-layer pruning, other rows spliced into every call
//   D  float16 natural                         vs A                            (information bar 0.5)
//   E  float16 strict 8, small pools           vs F, bit for bit
//   F  float16 strict 8, roomy, no pruning, other rows spliced in
//
// Suites: soak_short (registered in CTest, SEPAL_SOAK_SECONDS or 40 s) and soak (SEPAL_SOAK_MINUTES, default 20; not in
// CTest). SEPAL_SOAK_SEED sets the seed (default: derived from the clock; always printed first).

#include <chrono>
#include <fstream>
#include <sstream>

#include "attack_util.hpp"
#include "sepal/json.hpp"
#include "sepal/loader.hpp"

using namespace attack;

namespace {

using Clock = std::chrono::steady_clock;

struct RowSpec {
    std::vector<std::uint32_t> tokens;
    int parent_row = -1;     // index within the spec
    int stored = -1;         // index into the store
    std::optional<std::uint64_t> start;
    std::vector<std::size_t> readouts;
    bool keep = false;
    bool capture = false;
};

struct Spec {
    std::vector<RowSpec> rows;
};

// Where row tokens come from: random ids of the tiny vocabulary, or (real model) slices of real text.
const std::vector<std::uint32_t>* g_pool = nullptr;
bool g_real_lengths = false;

std::vector<std::uint32_t> draw(Rng& rng, std::size_t len) {
    if (g_pool == nullptr) return tokens(rng, len);
    const std::size_t at = rng.below(g_pool->size() - len);
    return {g_pool->begin() + static_cast<std::ptrdiff_t>(at), g_pool->begin() + static_cast<std::ptrdiff_t>(at + len)};
}

std::size_t random_len(Rng& rng) {
    if (g_real_lengths) {
        static const std::vector<std::size_t> real_lens = {1, 2, 3, 8, 27, 27, 60, 128, 300, 645, 645};
        return rng.pick(real_lens);
    }
    static const std::vector<std::size_t> lens = {1, 1, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 33, 64, 100, 129, 257};
    if (rng.chance(1)) return 300 + rng.below(700);
    return rng.pick(lens);
}

Spec make_spec(Rng& rng, std::size_t store_size) {
    Spec spec;
    const std::size_t groups = 1 + rng.below(4);
    std::vector<RowSpec> nodes;
    for (std::size_t g = 0; g < groups; ++g) {
        const int stored = store_size != 0 && rng.chance(40) ? static_cast<int>(rng.below(store_size)) : -1;
        const std::size_t shape = rng.below(6);
        const auto node = [&](int parent_row, int stored_parent, bool short_row) {
            RowSpec n;
            const std::size_t len = short_row ? 1 + rng.below(g_real_lengths ? 40 : 12) : random_len(rng);
            n.tokens = draw(rng, len);
            n.parent_row = parent_row;
            n.stored = stored_parent;
            n.keep = rng.chance(35);
            n.capture = rng.chance(4);
            for (std::size_t k = 0, c = rng.below(4); k < c; ++k) n.readouts.push_back(rng.below(len));
            if (rng.chance(12)) n.start = rng.below(rng.chance(30) ? 70000 : 400);
            nodes.push_back(std::move(n));
            return static_cast<int>(nodes.size() - 1);
        };
        switch (shape) {
        case 0: node(-1, stored, false); break;
        case 1: { // state with branches, up to 40
            const int root = node(-1, stored, false);
            for (std::size_t b = 0, n = 1 + rng.below(rng.chance(6) ? 40 : 8); b < n; ++b) node(root, -1, true);
            break;
        }
        case 2: { // chain, depth up to 14
            int prev = node(-1, stored, false);
            for (std::size_t d = 0, n = 1 + rng.below(14); d < n; ++d) prev = node(prev, -1, rng.chance(70));
            break;
        }
        case 3: { // branches of a stored snapshot
            for (std::size_t b = 0, n = 1 + rng.below(6); b < n; ++b) node(-1, stored, true);
            break;
        }
        case 4: { // one-token rows continuing a stored snapshot or each other
            int prev = node(-1, stored, true);
            for (std::size_t d = 0, n = rng.below(5); d < n; ++d) {
                RowSpec& last = nodes.back();
                last.tokens.resize(1 + (last.tokens.size() - 1) % 3);
                for (std::size_t& r : last.readouts) r %= last.tokens.size();
                prev = node(prev, -1, true);
            }
            break;
        }
        default: { // same tokens twice
            const int a = node(-1, stored, false);
            RowSpec copy = nodes[static_cast<std::size_t>(a)];
            nodes.push_back(copy);
            break;
        }
        }
    }
    // Random order of the request: parents may follow their children.
    std::vector<std::size_t> order(nodes.size());
    for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
    if (rng.chance(60)) {
        for (std::size_t i = order.size(); i > 1; --i) std::swap(order[i - 1], order[rng.below(i)]);
    }
    std::vector<std::size_t> position(nodes.size());
    for (std::size_t i = 0; i < order.size(); ++i) position[order[i]] = i;
    spec.rows.resize(nodes.size());
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        RowSpec r = nodes[i];
        if (r.parent_row >= 0) r.parent_row = static_cast<int>(position[static_cast<std::size_t>(r.parent_row)]);
        spec.rows[position[i]] = std::move(r);
    }
    return spec;
}

// The spec as engine rows. `store` holds the snapshots of that engine; nullptr entries (invalid handles) make the spec unusable.
bool instantiate(const Spec& spec, const std::vector<DeviceSnapshot>& store, std::size_t pre, Rng* extra_rng, std::size_t extra_rows, std::vector<EngineRow>& out) {
    out.clear();
    Rng* e = extra_rng;
    for (std::size_t i = 0; i < pre; ++i) out.push_back(make_row(draw(*e, 1 + e->below(60)), {e->below(1)}));
    for (const RowSpec& r : spec.rows) {
        EngineRow row = make_row(r.tokens, r.readouts, r.keep);
        row.capture_layers = r.capture;
        row.start_position = r.start;
        if (r.parent_row >= 0) row.parent = ParentRow{static_cast<std::size_t>(r.parent_row) + pre};
        else if (r.stored >= 0) {
            if (!store[static_cast<std::size_t>(r.stored)].valid()) return false;
            row.parent = store[static_cast<std::size_t>(r.stored)];
        }
        out.push_back(std::move(row));
    }
    for (std::size_t i = 0; i < extra_rows; ++i) {
        std::vector<std::uint32_t> t = draw(*e, 1 + e->below(90));
        out.push_back(make_row(t, {t.size() - 1}));
    }
    return true;
}

struct Counters {
    std::size_t calls = 0, rows = 0, tokens = 0, refused = 0, kept = 0, releases = 0;
    std::size_t checks = 0, failures = 0;
    double ref_seconds = 0, engine_seconds = 0;
    double a_vs_ref = 0, a_vs_b = 0, d_vs_a = 0, e_vs_b = 0;
    std::size_t strict_pairs = 0, strict_equal = 0;
};

struct Member {
    std::string name;
    std::unique_ptr<CudaEngine> engine;
    std::vector<DeviceSnapshot> store;
};

std::uint64_t pick_seed() {
    if (const auto e = environment("SEPAL_SOAK_SEED")) return std::stoull(*e);
    return static_cast<std::uint64_t>(Clock::now().time_since_epoch().count()) & 0xFFFFFFFFull;
}

void soak(double seconds) {
    const std::uint64_t seed = pick_seed();
    std::cout << "  soak seed " << seed << " for " << seconds << " s (SEPAL_SOAK_SEED=" << seed << " reproduces the sequence of requests)\n";
    Rng rng(seed);
    Rng extra(seed ^ 0xABCDEF);

    Knobs ka;                                                       // A
    Knobs kb = ka; kb.gemm_rows = 8; kb.slots = 18; kb.budget_mib = 54; kb.max_tokens = 2048; // B small pools
    Knobs kc = ka; kc.gemm_rows = 8; kc.prune = false; kc.slots = 200; kc.budget_mib = 220; kc.max_rows = 256; // C roomy
    Knobs kd = ka; kd.precision = Precision::F16;                    // D
    Knobs ke = kb; ke.precision = Precision::F16;                    // E
    Knobs kf = kc; kf.precision = Precision::F16;                    // F
    ka.max_tokens = 4096;
    kd.max_tokens = 4096;
    kc.max_tokens = 6000;
    kf.max_tokens = 6000;
    std::vector<Member> m;
    m.push_back({"A f32 natural", make_engine(ka), {}});
    m.push_back({"B f32 strict small", make_engine(kb), {}});
    m.push_back({"C f32 strict roomy", make_engine(kc), {}});
    m.push_back({"D f16 natural", make_engine(kd), {}});
    m.push_back({"E f16 strict small", make_engine(ke), {}});
    m.push_back({"F f16 strict roomy", make_engine(kf), {}});
    World world;
    Counters c;
    const auto start = Clock::now();
    auto next_report = start + std::chrono::seconds(60);
    const std::size_t store_cap = 14;
    std::size_t store_size = 0;

    while (std::chrono::duration<double>(Clock::now() - start).count() < seconds) {
        const Spec spec = make_spec(rng, store_size);
        // Which engines can take this spec (their store entries for it must exist)?
        std::size_t total_tokens = 0;
        for (const RowSpec& r : spec.rows) total_tokens += r.tokens.size();
        const std::size_t pre = rng.below(3), post = rng.below(5);
        std::vector<std::vector<EngineRow>> rows(m.size());
        std::vector<bool> usable(m.size(), true);
        std::vector<std::vector<EngineRowResult>> got(m.size());
        for (std::size_t i = 0; i < m.size(); ++i) {
            const bool roomy = i == 2 || i == 5;
            Rng copy = extra; // all roomy engines see the same other rows; the others see none
            usable[i] = instantiate(spec, m[i].store, roomy ? pre : 0, &copy, roomy ? post : 0, rows[i]);
            if (roomy && total_tokens + 400 > 6000) usable[i] = false;
        }
        extra.next();
        if (total_tokens > 2000) usable[1] = usable[4] = false; // the small-pool engines take 2048 tokens a call
        if (!usable[0] || total_tokens > 4000) {
            // A (or the size limit) cannot take it: drop the spec; the stores stay aligned because nothing was made.
            continue;
        }
        // The pass: every engine in turn; BudgetExceeded is tolerated on the small-pool ones.
        bool any_failed_small[2] = {false, false};
        const auto engines_begin = Clock::now();
        for (std::size_t i = 0; i < m.size(); ++i) {
            if (!usable[i]) continue;
            try {
                got[i] = run(*m[i].engine, rows[i]);
            } catch (const BudgetExceeded&) {
                if (i != 1 && i != 4) {
                    std::cout << "  UNEXPECTED BudgetExceeded in " << m[i].name << "\n";
                    ++c.failures;
                }
                ++c.refused;
                usable[i] = false;
                (i == 1 ? any_failed_small[0] : any_failed_small[1]) = true;
            } catch (const std::exception& e) {
                std::cout << "  EXCEPTION in " << m[i].name << ": " << e.what() << "\n";
                ++c.failures;
                usable[i] = false;
            }
        }
        c.engine_seconds += std::chrono::duration<double>(Clock::now() - engines_begin).count();
        ++c.calls;
        c.rows += spec.rows.size();
        c.tokens += total_tokens;
        const std::size_t base_a = 0;
        // A against the reference (only A's own request; the reference sees A's rows).
        if (usable[0]) {
            const auto ref_begin = Clock::now();
            const auto want = world.reference(rows[0]);
            c.ref_seconds += std::chrono::duration<double>(Clock::now() - ref_begin).count();
            for (std::size_t r = 0; r < rows[0].size(); ++r) {
                const std::size_t count = rows[0][r].readout_offsets.size() * world.hidden();
                ++c.checks;
                if (got[0][r].hidden.size() != count || !all_finite(got[0][r].hidden)) {
                    ++c.failures;
                    continue;
                }
                if (count != 0) {
                    const double d = max_abs_diff(got[0][r].hidden.data(), want[r].hidden.data(), count);
                    c.a_vs_ref = std::max(c.a_vs_ref, d);
                    if (d > 8e-3) {
                        ++c.failures;
                        std::cout << "  MISMATCH A vs reference: " << d << " (row " << r << ", " << rows[0][r].token_ids.size() << " tokens)\n";
                    }
                }
                if (rows[0][r].capture_layers && got[0][r].layers.size() != 8 * rows[0][r].token_ids.size() * world.hidden()) ++c.failures;
            }
            world.adopt(rows[0], got[0], want);
        }
        // Strict pairs: the subject rows are the spec's, found at `pre` (roomy) or 0.
        const auto subject = [&](std::size_t i, std::size_t k) -> const EngineRowResult& {
            const bool roomy = i == 2 || i == 5;
            return got[i][(roomy ? pre : 0) + k];
        };
        for (const auto& [small, roomy] : {std::pair<std::size_t, std::size_t>{1, 2}, std::pair<std::size_t, std::size_t>{4, 5}}) {
            if (!usable[small] || !usable[roomy]) continue;
            for (std::size_t k = 0; k < spec.rows.size(); ++k) {
                ++c.checks;
                ++c.strict_pairs;
                if (same_bits(subject(small, k).hidden, subject(roomy, k).hidden)) {
                    ++c.strict_equal;
                } else {
                    ++c.failures;
                    std::cout << "  STRICT MISMATCH " << m[small].name << " vs " << m[roomy].name << ": row " << k << " (" << spec.rows[k].tokens.size() << " tokens, call " << c.calls << ")\n";
                }
            }
        }
        if (usable[1] && usable[0]) {
            for (std::size_t k = 0; k < spec.rows.size(); ++k) {
                if (subject(1, k).hidden.size() == subject(0, k).hidden.size() && !subject(1, k).hidden.empty()) {
                    const double d = hidden_distance(subject(1, k).hidden, subject(0, k).hidden);
                    c.a_vs_b = std::max(c.a_vs_b, d);
                    ++c.checks;
                    if (d > 8e-3) { ++c.failures; std::cout << "  MISMATCH B vs A: " << d << "\n"; }
                }
            }
        }
        if (usable[3] && usable[0]) {
            for (std::size_t k = 0; k < spec.rows.size(); ++k) {
                if (got[3][k].hidden.empty()) continue;
                const double d = hidden_distance(got[3][k].hidden, got[0][k].hidden);
                c.d_vs_a = std::max(c.d_vs_a, d);
                ++c.checks;
                if (d > 8.0) { ++c.failures; std::cout << "  MISMATCH D vs A: " << d << "\n"; }
            }
        }
        if (usable[4] && usable[1]) {
            for (std::size_t k = 0; k < spec.rows.size(); ++k) {
                if (subject(4, k).hidden.empty()) continue;
                const double d = hidden_distance(subject(4, k).hidden, subject(1, k).hidden);
                c.e_vs_b = std::max(c.e_vs_b, d);
                ++c.checks;
                if (d > 8.0) { ++c.failures; std::cout << "  MISMATCH E vs B: " << d << "\n"; }
            }
        }
        (void)base_a;
        // Store the snapshots of the rows that kept one, at the same indices in every engine (an invalid handle where an engine
        // could not run the call). Then release at random, again at the same indices.
        for (std::size_t k = 0; k < spec.rows.size(); ++k) {
            if (!spec.rows[k].keep) continue;
            for (std::size_t i = 0; i < m.size(); ++i) {
                const bool roomy = i == 2 || i == 5;
                m[i].store.push_back(usable[i] ? got[i][(roomy ? pre : 0) + k].snapshot : DeviceSnapshot());
            }
            ++c.kept;
        }
        // A new snapshot is only usable later if A has it: A is the one every spec needs.
        store_size = m[0].store.size();
        while (store_size > store_cap || (store_size > 3 && rng.chance(8))) {
            const std::size_t at = rng.below(store_size);
            for (Member& mem : m) mem.store.erase(mem.store.begin() + static_cast<std::ptrdiff_t>(at));
            store_size = m[0].store.size();
            ++c.releases;
        }
        if (Clock::now() > next_report) {
            next_report += std::chrono::seconds(60);
            std::cout << "    [" << static_cast<int>(std::chrono::duration<double>(Clock::now() - start).count()) << " s] " << c.calls << " calls, " << c.rows << " rows, " << c.tokens << " tokens, " << c.failures << " failures\n";
        }
    }
    const double elapsed = std::chrono::duration<double>(Clock::now() - start).count();
    std::cout << "  soak done: seed " << seed << ", " << std::fixed << std::setprecision(0) << elapsed << " s, " << c.calls << " calls, " << c.rows << " rows, " << c.tokens << " tokens, " << c.kept
              << " snapshots kept, " << c.releases << " released, " << c.refused << " calls refused with BudgetExceeded by the small engines; time in the six engines " << c.engine_seconds << " s, in the CPU reference " << c.ref_seconds << " s\n"
              << "    float32 natural vs CPU reference: worst " << std::scientific << std::setprecision(2) << c.a_vs_ref << "; float32 strict vs natural: worst " << c.a_vs_b << "; float16 natural vs float32: worst " << c.d_vs_a
              << "; float16 strict vs float32 strict: worst " << c.e_vs_b << std::fixed << "\n"
              << "    strict small vs strict roomy (other rows spliced in, no pruning, other pools): " << c.strict_equal << " of " << c.strict_pairs << " rows bit-identical; " << c.checks << " checks, " << c.failures
              << " failures\n";
    for (Member& mem : m) {
        mem.store.clear();
        const EngineMemory mm = mem.engine->memory();
        CHECK_EQ(mm.slots_in_use, std::size_t{0});
        CHECK_EQ(mm.kv_in_use_bytes, std::size_t{0});
        CHECK_EQ(mm.kv_largest_free_bytes, mm.kv_heap_bytes);
    }
    CHECK_EQ(c.failures, std::size_t{0});
    CHECK(c.calls > 0);
}


// ---- the real model ----
// Two engines on jeff-base run the same random requests built from slices of the alert workload's real text. Phase one: two
// float16 strict engines (small pools; roomy pools with other rows spliced in and no pruning) must agree bit for bit.
// Phase two: float32 natural and float16 natural must stay close, with no answer flip. Needs SEPAL_REAL_MODEL and SEPAL_ALERTS.

struct RealPhase {
    std::size_t calls = 0, rows = 0, tokens_run = 0, refused = 0, compared = 0, equal = 0, flips = 0, hard_flips = 0;
    std::vector<std::string> flip_notes;
    double worst = 0;
};

// `b_fp32_accumulate`: engine b multiplies on cuBLAS with float32 accumulators (the engine's default before MultiplyPath::Auto) instead of
// the default, so that a comparison with the float32 engine measures the float16 operands and nothing else.
RealPhase real_phase(const LoadedModel& model, const DecisionReadout& readout, Precision pa, std::size_t ga, Precision pb, std::size_t gb, bool extras, bool tight, double seconds,
                     std::uint64_t seed, bool b_fp32_accumulate = false) {
    Rng rng(seed), extra(seed ^ 0x5555);
    EngineOptions oa, ob;
    oa.precision = pa;
    oa.gemm_rows = ga;
    oa.max_call_tokens = 4096;
    oa.max_call_rows = 128;
    ob = oa;
    ob.precision = pb;
    ob.gemm_rows = gb;
    if (b_fp32_accumulate) ob.multiply_path = sepal::cuda::MultiplyPath::Cublas;
    if (tight) {
        oa.state_slots = 10;
        oa.memory_budget_bytes = (pa == Precision::F32 ? std::size_t{4800} : std::size_t{2400}) << 20;
    } else {
        oa.state_slots = 40;
        oa.memory_budget_bytes = (pa == Precision::F32 ? std::size_t{5200} : std::size_t{2800}) << 20;
    }
    ob.state_slots = 40;
    ob.memory_budget_bytes = (pb == Precision::F32 ? std::size_t{5200} : std::size_t{2800}) << 20;
    if (extras) ob.prune_last_layer = false;
    if (environment("SEPAL_SOAK_TRACE")) std::cerr << "creating engine a" << std::endl;
    CudaEngine ea(model, oa);
    if (environment("SEPAL_SOAK_TRACE")) std::cerr << "creating engine b" << std::endl;
    CudaEngine eb(model, ob);
    if (environment("SEPAL_SOAK_TRACE")) std::cerr << "engines made" << std::endl;
    RealPhase out;
    std::vector<DeviceSnapshot> sa, sb;
    const auto start = Clock::now();
    while (std::chrono::duration<double>(Clock::now() - start).count() < seconds) {
        const Spec spec = make_spec(rng, std::min(sa.size(), sb.size()));
        std::size_t total = 0;
        for (const RowSpec& r : spec.rows) total += r.tokens.size();
        if (total > 3000) continue;
        const std::size_t pre = extras ? rng.below(3) : 0, post = extras ? rng.below(4) : 0;
        std::vector<EngineRow> ra, rb;
        Rng copy = extra;
        extra.next();
        const bool ok_a = instantiate(spec, sa, 0, &copy, 0, ra);
        const bool ok_b = instantiate(spec, sb, pre, &copy, post, rb);
        if (!ok_a || !ok_b) {
            // a snapshot one engine could not make: keep the stores aligned by dropping every entry invalid in either
            for (std::size_t i = std::min(sa.size(), sb.size()); i-- > 0;) {
                if (!sa[i].valid() || !sb[i].valid()) {
                    sa.erase(sa.begin() + static_cast<std::ptrdiff_t>(i));
                    sb.erase(sb.begin() + static_cast<std::ptrdiff_t>(i));
                }
            }
            continue;
        }
        if (environment("SEPAL_SOAK_TRACE")) {
            std::cerr << "call " << out.calls << ":";
            for (std::size_t k = 0; k < spec.rows.size(); ++k) {
                const RowSpec& r = spec.rows[k];
                std::cerr << " [" << r.tokens.size() << "t parent_row " << r.parent_row << " stored " << r.stored << " start " << (r.start ? std::to_string(*r.start) : "-") << " readouts " << r.readouts.size()
                          << (r.keep ? " keep" : "") << (r.capture ? " capture" : "") << "]";
            }
            std::cerr << " pre " << pre << " post " << post << " stores " << sa.size() << "\n";
        }
        std::vector<EngineRowResult> ga_, gb_;
        bool a_ok = true;
        try {
            ga_ = run(ea, ra);
        } catch (const BudgetExceeded&) {
            a_ok = false;
            ++out.refused;
        }
        if (a_ok) gb_ = run(eb, rb);
        ++out.calls;
        if (a_ok) {
            out.rows += spec.rows.size();
            out.tokens_run += total;
            for (std::size_t k = 0; k < spec.rows.size(); ++k) {
                const EngineRowResult& x = ga_[k];
                const EngineRowResult& y = gb_[pre + k];
                if (x.hidden.empty()) continue;
                ++out.compared;
                if (same_bits(x.hidden, y.hidden)) ++out.equal;
                out.worst = std::max(out.worst, hidden_distance(x.hidden, y.hidden));
                const std::size_t H = x.hidden.size() / spec.rows[k].readouts.size();
                for (std::size_t o = 0; o < spec.rows[k].readouts.size(); ++o) {
                    const std::span<const float> hx(x.hidden.data() + o * H, H), hy(y.hidden.data() + o * H, H);
                    const auto px = readout.probabilities(hx, 2), py = readout.probabilities(hy, 2);
                    if ((px[0] > px[1]) != (py[0] > py[1])) {
                        ++out.flips;
                        if (std::fabs(px[0] - 0.5) > 0.01 && std::fabs(py[0] - 0.5) > 0.01) ++out.hard_flips;
                        std::ostringstream note;
                        note << "p0 " << std::fixed << std::setprecision(5) << px[0] << " vs " << py[0] << " (" << spec.rows[k].tokens.size() << " tokens)";
                        out.flip_notes.push_back(note.str());
                    }
                }
            }
        }
        for (std::size_t k = 0; k < spec.rows.size(); ++k) {
            if (!spec.rows[k].keep) continue;
            sa.push_back(a_ok ? ga_[k].snapshot : DeviceSnapshot());
            sb.push_back(a_ok ? gb_[pre + k].snapshot : DeviceSnapshot());
        }
        while (sa.size() > 8 || (sa.size() > 2 && rng.chance(10))) {
            const std::size_t at = rng.below(sa.size());
            sa.erase(sa.begin() + static_cast<std::ptrdiff_t>(at));
            sb.erase(sb.begin() + static_cast<std::ptrdiff_t>(at));
        }
    }
    return out;
}

} // namespace

SEPAL_TEST(soak_short, random_requests_on_six_engines_checked_against_the_reference_and_each_other) {
    double seconds = 25;
    if (const auto e = environment("SEPAL_SOAK_SECONDS")) seconds = std::stod(*e);
    soak(seconds);
}

SEPAL_TEST(soak, the_long_run) {
    double minutes = 20;
    if (const auto e = environment("SEPAL_SOAK_MINUTES")) minutes = std::stod(*e);
    soak(minutes * 60.0);
}

SEPAL_TEST(real_soak, random_requests_of_real_text_on_the_real_model) {
    if (!environment("SEPAL_REAL_MODEL") || !environment("SEPAL_ALERTS")) {
        std::cout << "SKIPPED: real-model test; set SEPAL_REAL_MODEL and SEPAL_ALERTS\n";
        return;
    }
    double seconds = 120;
    if (const auto e = environment("SEPAL_SOAK_SECONDS")) seconds = std::stod(*e);
    const std::uint64_t seed = pick_seed();
    std::cout << "  real soak seed " << seed << " (" << seconds << " s per phase)\n";
    std::ifstream in(*environment("SEPAL_ALERTS"), std::ios::binary);
    std::stringstream buffer;
    buffer << in.rdbuf();
    std::vector<std::uint32_t> pool;
    const json::Value root = json::parse(buffer.str());
    for (const json::Value& a : root.at("alerts").items()) {
        for (const std::uint64_t id : a.at("state").as_uint_list()) pool.push_back(static_cast<std::uint32_t>(id));
        for (const json::Value& q : a.at("suffixes").items()) {
            for (const std::uint64_t id : q.as_uint_list()) pool.push_back(static_cast<std::uint32_t>(id));
        }
    }
    if (environment("SEPAL_SOAK_TRACE")) std::cerr << "pool " << pool.size() << std::endl;
    g_pool = &pool;
    g_real_lengths = true;
    const LoadedModel model = LoadedModel::open(*environment("SEPAL_REAL_MODEL"));
    const DecisionReadout readout = DecisionReadout::open_directory(*environment("SEPAL_REAL_MODEL"));
    const RealPhase one = real_phase(model, readout, Precision::F16, 128, Precision::F16, 128, true, true, seconds, seed);
    std::cout << "  phase 1, float16 strict 128, tight pools vs roomy pools + other rows + no pruning: " << one.calls << " calls (" << one.refused << " refused by the tight engine), " << one.rows << " rows, "
              << one.tokens_run << " tokens; " << one.equal << " of " << one.compared << " readout rows bit-identical, largest difference " << one.worst << "\n";
    CHECK_EQ(one.equal, one.compared);
    CHECK(one.compared > 0);
    const RealPhase two = real_phase(model, readout, Precision::F32, 0, Precision::F16, 0, false, false, seconds, seed + 1, /*b_fp32_accumulate=*/true);
    std::cout << "  phase 2, float32 natural vs float16 natural (float32 accumulators): " << two.calls << " calls, " << two.rows << " rows, " << two.tokens_run << " tokens; " << two.compared << " readout rows, largest hidden difference "
              << two.worst << ", answer flips " << two.flips << " (" << two.hard_flips << " away from a tie)\n";
    for (const std::string& note : two.flip_notes) std::cout << "    flip: " << note << "\n";
    CHECK(two.worst < 0.1);
    CHECK_EQ(two.hard_flips, std::size_t{0}); // random slices of text often sit near p = 0.5, where a float16 rounding flips the answer
    // The default (cuBLAS with binary16 accumulators in natural mode) on the same kind of requests: farther from float32 in the hidden state, and (measured: largest
    // 0.53 where float32 accumulation gave 0.09, 40 flips of 5,053 read-outs, all within 0.01 of p = 0.5) the same decisions away from a tie.
    const RealPhase three = real_phase(model, readout, Precision::F32, 0, Precision::F16, 0, false, false, seconds, seed + 2);
    std::cout << "  phase 3, float32 natural vs the float16 default (binary16 accumulators): " << three.calls << " calls, " << three.rows << " rows, " << three.tokens_run << " tokens; " << three.compared
              << " readout rows, largest hidden difference " << three.worst << ", answer flips " << three.flips << " (" << three.hard_flips << " away from a tie)\n";
    for (const std::string& note : three.flip_notes) std::cout << "    flip: " << note << "\n";
    CHECK(three.worst < 1.5);
    CHECK_EQ(three.hard_flips, std::size_t{0});
    g_pool = nullptr;
    g_real_lengths = false;
}
