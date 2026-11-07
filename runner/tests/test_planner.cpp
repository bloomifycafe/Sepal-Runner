#include <cstdint>
#include <stdexcept>
#include <string>
#include <variant>

#include "sepal/planner.hpp"
#include "test.hpp"

using namespace sepal;

namespace {

constexpr std::uint64_t kGiB = 1024ull * 1024ull * 1024ull;

ModelConfig preset(const char* name) {
    const auto p = find_preset(name);
    if (!p) throw std::runtime_error(std::string("missing preset ") + name);
    return *p;
}

PrecisionProfile profile(const char* id) {
    const auto p = find_profile(id);
    if (!p) throw std::runtime_error(std::string("missing profile ") + id);
    return *p;
}

// The request the budget matrix uses: a 4,096-token pass of up to 32 rows, and room for one 2,000-token snapshot.
PlanRequest request(const char* preset_name, const char* profile_id, std::uint64_t available) {
    PlanRequest r;
    r.config = preset(preset_name);
    r.profile = profile(profile_id);
    r.available_bytes = available;
    r.token_budget = 4096;
    r.max_rows = 32;
    r.min_snapshots = 1;
    r.min_snapshot_tokens = 2000;
    return r;
}

bool is_plan(const PlanResult& r) { return std::holds_alternative<Plan>(r); }
const Plan& plan_of(const PlanResult& r) { return std::get<Plan>(r); }
const Refusal& refusal_of(const PlanResult& r) { return std::get<Refusal>(r); }

// Bytes to the nearest 0.1 GB (decimal), in tenths.
std::uint64_t tenths_of_gb(std::uint64_t bytes) { return (bytes + 50000000ull) / 100000000ull; }

void check_plan_invariants(const PlanRequest& req, const Plan& p) {
    CHECK_EQ(p.committed_bytes(), req.available_bytes); // every byte is accounted for, none twice
    CHECK_EQ(p.available_bytes, req.available_bytes);
    CHECK_EQ(p.token_budget, req.token_budget);
    CHECK(p.max_token_budget >= req.token_budget);
    CHECK(p.snapshot_arena_bytes >= req.min_snapshots * p.snapshot_bytes_each);
    CHECK_EQ(p.snapshot_bytes_each, snapshot_bytes(req.config, req.profile, req.min_snapshot_tokens));
    CHECK(p.snapshots_held >= req.min_snapshots);
    CHECK_EQ(p.working_bytes,
             working_bytes(req.config, req.profile, req.token_budget, req.max_rows, req.attention_tile));
    CHECK(p.weights_bytes >= trunk_weight_bytes(req.config, req.profile));
}

void check_refusal_invariants(const PlanRequest& req, const Refusal& r) {
    CHECK(r.shortfall_bytes > 0);
    CHECK_EQ(r.available_bytes, req.available_bytes);
    CHECK_EQ(r.required_bytes, r.available_bytes + r.shortfall_bytes);
    CHECK(!r.message.empty());
    CHECK(r.message.find("short by") != std::string::npos); // the shortfall is stated in words too
}

} // namespace

SEPAL_TEST(planner, weight_totals_match_the_design) {
    // docs/RUNNER.md section 3, "Body weights on device, fp16 / int8", in GB to one decimal.
    struct Row {
        const char* name;
        std::uint64_t f16_tenths, i8_tenths;
    };
    const Row rows[] = {{"0.8b", 10, 5}, {"2b", 27, 14}, {"4b", 71, 36}, {"9b", 138, 69}};
    for (const Row& r : rows) {
        const ModelConfig c = preset(r.name);
        const std::uint64_t n = count_params(c).non_embedding();
        CHECK_EQ(trunk_weight_bytes(c, profile("w16")), 2 * n);
        CHECK_EQ(trunk_weight_bytes(c, profile("w8")), n);
        CHECK_EQ(tenths_of_gb(trunk_weight_bytes(c, profile("w16"))), r.f16_tenths);
        CHECK_EQ(tenths_of_gb(trunk_weight_bytes(c, profile("w8"))), r.i8_tenths);

        // The section 11 profiles keep some classes wider, so they sit at or above the uniform figures.
        CHECK(trunk_weight_bytes(c, profile("gpu-f16")) >= trunk_weight_bytes(c, profile("w16")));
        CHECK(trunk_weight_bytes(c, profile("gpu-q8w")) > trunk_weight_bytes(c, profile("w8")));
        CHECK(trunk_weight_bytes(c, profile("gpu-q8w")) < trunk_weight_bytes(c, profile("w16")));
        CHECK_EQ(trunk_weight_bytes(c, profile("cpu-q8")), trunk_weight_bytes(c, profile("gpu-q8w")));
    }
    CHECK(!find_profile("q4").has_value()); // the design has no 4-bit profile
}

SEPAL_TEST(planner, snapshot_bytes_match_the_design) {
    // Section 7: one node for a 600-token state is about 72 MB on 4B and about 28 MB on 0.8B / 2B.
    CHECK_EQ(snapshot_bytes(preset("4b"), profile("gpu-f16"), 600), 600ull * 32768ull + 50331648ull + 2359296ull);
    CHECK_EQ(snapshot_bytes(preset("4b"), profile("gpu-f16"), 600), 72351744ull);
    CHECK_EQ(snapshot_bytes(preset("0.8b"), profile("gpu-f16"), 600), 27574272ull);
    CHECK_EQ(snapshot_bytes(preset("2b"), profile("gpu-f16"), 600), 27574272ull);

    // "47 MB with fp16 state": halve the recurrent state.
    PrecisionProfile f16_state = profile("gpu-f16");
    f16_state.recurrent_state = DType::F16;
    CHECK_EQ(snapshot_bytes(preset("4b"), f16_state, 600), 72351744ull - 25165824ull);
}

SEPAL_TEST(planner, working_bytes_by_hand) {
    // 0.8B, 1,000 tokens, 8 rows, tile 256.
    //   floats per token = 2 x 1024 + max(7168, 6144 + 2048 + 32, 4096 + 1024) + 3584 + 2048 = 15,904
    //   activations 63,616,000   in-pass K/V 12,288,000   score tile 8 x 256 x 1000 x 4 = 8,192,000
    //   row state 8 x (16 x 128 x 128 + 3 x 6144) x 4 = 8,978,432
    const ModelConfig c = preset("0.8b");
    const PrecisionProfile p = profile("w16");
    CHECK_EQ(working_bytes(c, p, 1000, 8, 256), 93074432ull);
    CHECK_EQ(working_bytes(c, p, 0, 0, 256), 0ull);

    // Monotone in tokens and in rows; the score tile stops growing with the tile once tokens exceed it.
    CHECK(working_bytes(c, p, 1001, 8, 256) > working_bytes(c, p, 1000, 8, 256));
    CHECK(working_bytes(c, p, 1000, 9, 256) > working_bytes(c, p, 1000, 8, 256));
    CHECK_EQ(working_bytes(c, p, 100, 8, 256), working_bytes(c, p, 100, 8, 4096));
}

SEPAL_TEST(planner, four_presets_against_8_12_24_gib) {
    // By arithmetic alone, with no reserve: only the 16-bit 9B (13.8 GB of weights) fails to fit 8 and 12 GiB,
    // and in both cases 8-bit weights (6.9 GB) would fit, so the knob named is the weight precision.
    const char* const presets_all[] = {"0.8b", "2b", "4b", "9b"};
    const char* const profiles_all[] = {"w16", "w8"};
    const std::uint64_t budgets[] = {8 * kGiB, 12 * kGiB, 24 * kGiB};

    int plans = 0;
    int refusals = 0;
    for (const char* preset_name : presets_all) {
        for (const char* profile_id : profiles_all) {
            for (const std::uint64_t budget : budgets) {
                const PlanRequest req = request(preset_name, profile_id, budget);
                const PlanResult result = make_plan(req);
                const bool expect_refusal = std::string(preset_name) == "9b" && std::string(profile_id) == "w16" &&
                                            budget < 24 * kGiB;
                CHECK_EQ(is_plan(result), !expect_refusal);
                if (is_plan(result)) {
                    ++plans;
                    check_plan_invariants(req, plan_of(result));
                } else {
                    ++refusals;
                    const Refusal& r = refusal_of(result);
                    check_refusal_invariants(req, r);
                    CHECK(r.knob == Knob::WeightPrecision);
                    CHECK(r.message.find("weight precision") != std::string::npos);
                    // The shortfall is exactly the weights minus the budget.
                    const std::uint64_t weights = trunk_weight_bytes(req.config, req.profile) +
                                                  count_params(req.config).pointer_head * 4;
                    CHECK_EQ(r.shortfall_bytes, weights - budget);
                }
            }
        }
    }
    CHECK_EQ(plans, 22);
    CHECK_EQ(refusals, 2);
}

SEPAL_TEST(planner, more_memory_never_shrinks_a_plan) {
    for (const char* profile_id : {"w16", "w8"}) {
        const PlanResult p12 = make_plan(request("4b", profile_id, 12 * kGiB));
        const PlanResult p24 = make_plan(request("4b", profile_id, 24 * kGiB));
        CHECK(is_plan(p12) && is_plan(p24));
        if (!is_plan(p12) || !is_plan(p24)) continue;
        CHECK_EQ(plan_of(p12).weights_bytes, plan_of(p24).weights_bytes);
        CHECK_EQ(plan_of(p12).working_bytes, plan_of(p24).working_bytes);
        CHECK_EQ(plan_of(p24).snapshot_arena_bytes - plan_of(p12).snapshot_arena_bytes, 12 * kGiB);
        CHECK(plan_of(p24).max_token_budget > plan_of(p12).max_token_budget);
    }
}

SEPAL_TEST(planner, refuses_with_model_size_when_no_precision_fits) {
    for (const char* profile_id : {"w16", "w8"}) {
        const PlanRequest req = request("9b", profile_id, 4 * kGiB);
        const PlanResult result = make_plan(req);
        CHECK(!is_plan(result));
        if (is_plan(result)) continue;
        check_refusal_invariants(req, refusal_of(result));
        CHECK(refusal_of(result).knob == Knob::ModelSize);
        CHECK(refusal_of(result).message.find("model size") != std::string::npos);
    }
}

SEPAL_TEST(planner, refuses_with_token_budget_and_the_suggestion_is_the_limit) {
    PlanRequest req = request("4b", "w16", 8 * kGiB);
    req.token_budget = 65536; // a full-length state does not fit beside 7.1 GB of weights
    const PlanResult result = make_plan(req);
    CHECK(!is_plan(result));
    if (is_plan(result)) return;
    const Refusal& r = refusal_of(result);
    check_refusal_invariants(req, r);
    CHECK(r.knob == Knob::TokenBudget);
    CHECK(r.suggested_value >= 4096);
    CHECK(r.suggested_value < 65536);
    CHECK(r.message.find(std::to_string(r.suggested_value)) != std::string::npos);

    // The suggested budget is granted, and it is the largest that is.
    req.token_budget = r.suggested_value;
    const PlanResult at_limit = make_plan(req);
    CHECK(is_plan(at_limit));
    if (is_plan(at_limit)) {
        check_plan_invariants(req, plan_of(at_limit));
        CHECK_EQ(plan_of(at_limit).max_token_budget, r.suggested_value);
    }
    req.token_budget = r.suggested_value + 1;
    CHECK(!is_plan(make_plan(req)));
}

SEPAL_TEST(planner, refuses_with_snapshot_budget_one_byte_short) {
    PlanRequest req = request("0.8b", "w16", 24 * kGiB);
    const PlanResult roomy = make_plan(req);
    CHECK(is_plan(roomy));
    if (!is_plan(roomy)) return;
    const Plan& base = plan_of(roomy);
    const std::uint64_t exact = base.weights_bytes + base.working_bytes + base.snapshot_bytes_each;

    // Exactly enough: the plan closes with an arena of one snapshot and nothing to spare.
    req.available_bytes = exact;
    const PlanResult tight = make_plan(req);
    CHECK(is_plan(tight));
    if (is_plan(tight)) {
        check_plan_invariants(req, plan_of(tight));
        CHECK_EQ(plan_of(tight).snapshot_arena_bytes, base.snapshot_bytes_each);
        CHECK_EQ(plan_of(tight).snapshots_held, 1u);
        CHECK_EQ(plan_of(tight).max_token_budget, req.token_budget);
    }

    // One byte less: refused, not rounded down.
    req.available_bytes = exact - 1;
    const PlanResult short_by_one = make_plan(req);
    CHECK(!is_plan(short_by_one));
    if (is_plan(short_by_one)) return;
    const Refusal& r = refusal_of(short_by_one);
    check_refusal_invariants(req, r);
    CHECK(r.knob == Knob::SnapshotBudget);
    CHECK_EQ(r.shortfall_bytes, 1u);
    CHECK_EQ(r.suggested_value, 1999u); // one K/V token less fits

    req.min_snapshot_tokens = r.suggested_value;
    CHECK(is_plan(make_plan(req)));
}

SEPAL_TEST(planner, reserve_is_held_back) {
    // 4B at 16 bits fits 8 GiB by arithmetic, but not once 1 GiB is reserved for the driver and the desktop.
    PlanRequest req = request("4b", "w16", 8 * kGiB);
    CHECK(is_plan(make_plan(req)));
    req.reserve_bytes = kGiB;
    const PlanResult reserved = make_plan(req);
    CHECK(!is_plan(reserved));
    if (!is_plan(reserved)) {
        check_refusal_invariants(req, refusal_of(reserved));
        CHECK(refusal_of(reserved).knob == Knob::TokenBudget);
    }

    // With 8-bit weights the same reserve is honoured and stays out of every other part.
    PlanRequest q8 = request("4b", "w8", 8 * kGiB);
    q8.reserve_bytes = kGiB;
    const PlanResult ok = make_plan(q8);
    CHECK(is_plan(ok));
    if (is_plan(ok)) {
        check_plan_invariants(q8, plan_of(ok));
        CHECK_EQ(plan_of(ok).reserve_bytes, kGiB);
    }
}

SEPAL_TEST(planner, embedding_counts_only_when_it_shares_the_budget) {
    PlanRequest req = request("2b", "w16", 24 * kGiB);
    const PlanResult host = make_plan(req);
    req.embedding_on_device = true;
    const PlanResult device = make_plan(req);
    CHECK(is_plan(host) && is_plan(device));
    if (!is_plan(host) || !is_plan(device)) return;
    const std::uint64_t table = 248320ull * 2048ull * 2ull;
    CHECK_EQ(plan_of(host).embedding_bytes, 0u);
    CHECK_EQ(plan_of(host).embedding_host_bytes, table);
    CHECK_EQ(plan_of(device).embedding_bytes, table);
    CHECK_EQ(plan_of(device).embedding_host_bytes, 0u);
    CHECK_EQ(plan_of(host).snapshot_arena_bytes - plan_of(device).snapshot_arena_bytes, table);
    check_plan_invariants(req, plan_of(device));
}

SEPAL_TEST(planner, malformed_requests_throw) {
    PlanRequest req = request("2b", "w16", 12 * kGiB);
    req.token_budget = 0;
    CHECK_THROWS(make_plan(req), std::invalid_argument);

    req = request("2b", "w16", 12 * kGiB);
    req.max_rows = 0;
    CHECK_THROWS(make_plan(req), std::invalid_argument);

    req = request("2b", "w16", 12 * kGiB);
    req.reserve_bytes = req.available_bytes;
    CHECK_THROWS(make_plan(req), std::invalid_argument);

    req = request("2b", "w16", 12 * kGiB);
    req.config.hidden_size = 0;
    CHECK_THROWS(make_plan(req), std::invalid_argument);
}

SEPAL_TEST(planner, pascal_profiles_weights_at_4b) {
    const ModelConfig c = preset("4b");
    const ParamCounts n = count_params(c);
    const PrecisionProfile bf16 = profile("gpu-bf16");
    const PrecisionProfile w8g = profile("gpu-w8g");
    const std::uint64_t bf16_bytes = trunk_weight_bytes(c, bf16);
    const std::uint64_t w8g_bytes = trunk_weight_bytes(c, w8g);

    CHECK(w8g_bytes < bf16_bytes);
    CHECK(bf16_bytes < 2 * w8g_bytes);
    // gpu-w8g: the CUDA library's computed 3.69e9 bytes at 4B (int8 bytes + 3.1% scales + the small tensors), within 2%.
    CHECK_NEAR(static_cast<double>(w8g_bytes), 3.69e9, 0.02 * 3.69e9);
    // gpu-bf16: every matrix at 2 bytes, the small tensors at 4; 6.65 GiB measured on the device.
    CHECK_NEAR(static_cast<double>(bf16_bytes) / (1024.0 * 1024.0 * 1024.0), 6.65, 0.02);

    // Exact by hand. The DeltaNet in_proj_b and in_proj_a (2 x hidden x value heads per layer) stay bfloat16 in gpu-w8g;
    // the other classes are int8. Every input width at 4B is a multiple of 128, so the float32 scales are
    // (int8 values / 128) x 4 bytes.
    const std::uint64_t gates = c.num_deltanet_layers() * 2 * c.hidden_size * c.dn_value_heads;
    const std::uint64_t int8_values = n.mlp + n.mixer_proj + (n.dn_sensitive - gates);
    CHECK_EQ(bf16_bytes, 2 * (n.mlp + n.mixer_proj + n.dn_sensitive) + 4 * n.small);
    CHECK_EQ(w8g_bytes, int8_values + 4 * (int8_values / 128) + 2 * gates + 4 * n.small);
    CHECK_EQ(int8_values % 128, 0u);
    CHECK_EQ(w8g.scale_group, 128u);
    CHECK_EQ(bf16.scale_group, 0u);

    // The scales are the only difference from the same classes without them; the existing profiles count none.
    PrecisionProfile no_scales = w8g;
    no_scales.scale_group = 0;
    CHECK_EQ(w8g_bytes - trunk_weight_bytes(c, no_scales), 4 * (int8_values / 128));
    CHECK_EQ(profile("gpu-q8w").scale_group, 0u);
    CHECK(!profile("gpu-q8w").dn_out_proj.has_value());

    // The other presets keep the order: 8-bit with scales is smaller than bfloat16 and more than half of it.
    for (const char* name : {"0.8b", "2b", "9b"}) {
        const ModelConfig other = preset(name);
        CHECK(trunk_weight_bytes(other, w8g) < trunk_weight_bytes(other, bf16));
        CHECK(trunk_weight_bytes(other, bf16) < 2 * trunk_weight_bytes(other, w8g));
    }
}

SEPAL_TEST(planner, group_scales_round_up_per_row) {
    // The test geometry has input widths that are not multiples of 128, so each row needs ceil(width / 128) scales.
    const ModelConfig c = tiny_test_config();
    const PrecisionProfile w8g = profile("gpu-w8g");
    PrecisionProfile no_scales = w8g;
    no_scales.scale_group = 0;
    const std::uint64_t h = c.hidden_size;
    const std::uint64_t i = c.intermediate_size;
    const auto rows_x_groups = [](std::uint64_t rows, std::uint64_t inputs) { return rows * ((inputs + 127) / 128); };
    const std::uint64_t expected =
        c.num_layers * (2 * rows_x_groups(i, h) + rows_x_groups(h, i)) +
        c.num_attention_layers() * (rows_x_groups(c.attn_in_proj_width(), h) + rows_x_groups(h, c.attn_width())) +
        c.num_deltanet_layers() * (rows_x_groups(c.dn_conv_dim(), h) + rows_x_groups(c.dn_value_dim(), h) +
                                   rows_x_groups(h, c.dn_value_dim()));
    CHECK_EQ(trunk_weight_bytes(c, w8g) - trunk_weight_bytes(c, no_scales), 4 * expected);
    const ParamCounts n = count_params(c);
    const std::uint64_t gates = c.num_deltanet_layers() * 2 * h * c.dn_value_heads;
    CHECK(expected > (n.mlp + n.mixer_proj + (n.dn_sensitive - gates)) / 128); // rounding up gives more than the plain division
}

SEPAL_TEST(planner, pascal_profiles_state_embedding_and_budgets) {
    const ModelConfig c = preset("4b");
    // K/V is float32: 2 x 8 layers x ... = 65,536 bytes a token at 4B against 32,768 in binary16; the state is float32 as before.
    for (const char* id : {"gpu-bf16", "gpu-w8g"}) {
        const PrecisionProfile p = profile(id);
        CHECK(p.kv == DType::F32);
        CHECK(p.recurrent_state == DType::F32 && p.conv_tail == DType::F32 && p.small == DType::F32);
        CHECK_EQ(snapshot_bytes(c, p, 600), 600ull * 65536ull + 50331648ull + 2359296ull);
    }
    CHECK(profile("gpu-bf16").mlp == DType::BF16 && profile("gpu-bf16").embedding == DType::BF16);
    CHECK(profile("gpu-w8g").mlp == DType::I8 && profile("gpu-w8g").mixer_proj == DType::I8);
    CHECK(profile("gpu-w8g").dn_out_proj == DType::I8 && profile("gpu-w8g").dn_sensitive == DType::BF16);

    // The embedding table (bfloat16, 248,320 x 2,560) stays on the host unless the request says otherwise.
    for (const char* id : {"gpu-bf16", "gpu-w8g"}) {
        const PlanRequest req = request("4b", id, 24 * kGiB);
        const PlanResult result = make_plan(req);
        CHECK(is_plan(result));
        if (!is_plan(result)) continue;
        check_plan_invariants(req, plan_of(result));
        CHECK_EQ(plan_of(result).embedding_bytes, 0u);
        CHECK_EQ(plan_of(result).embedding_host_bytes, 248320ull * 2560ull * 2ull);
    }

    // An 8 GiB card with 1 GiB held back (a Pascal desktop card): bfloat16 weights leave too little for the 4,096-token
    // pass, the int8 profile plans. On 11 GiB, bfloat16 plans.
    PlanRequest eight = request("4b", "gpu-bf16", 8 * kGiB);
    eight.reserve_bytes = kGiB;
    const PlanResult bf16_eight = make_plan(eight);
    CHECK(!is_plan(bf16_eight));
    if (!is_plan(bf16_eight)) CHECK(refusal_of(bf16_eight).knob == Knob::TokenBudget);
    eight.profile = profile("gpu-w8g");
    const PlanResult w8g_eight = make_plan(eight);
    CHECK(is_plan(w8g_eight));
    if (is_plan(w8g_eight)) check_plan_invariants(eight, plan_of(w8g_eight));
    PlanRequest eleven = request("4b", "gpu-bf16", 11 * kGiB);
    eleven.reserve_bytes = kGiB;
    const PlanResult bf16_eleven = make_plan(eleven);
    CHECK(is_plan(bf16_eleven));
    if (is_plan(bf16_eleven)) check_plan_invariants(eleven, plan_of(bf16_eleven));
}

SEPAL_TEST(planner, format_bytes_and_knob_names) {
    CHECK_EQ(format_bytes(1000000000ull), "1.00 GB (0.93 GiB)");
    CHECK_EQ(format_bytes(8 * kGiB), "8.59 GB (8.00 GiB)");
    CHECK_EQ(name_of(Knob::WeightPrecision), "weight precision");
    CHECK_EQ(name_of(Knob::ModelSize), "model size");
    CHECK_EQ(name_of(Knob::TokenBudget), "token budget");
    CHECK_EQ(name_of(Knob::SnapshotBudget), "snapshot budget");
}
