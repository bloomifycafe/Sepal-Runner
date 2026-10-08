// sepal-info: print a preset's derived quantities and a memory plan for a given budget.
// Exit status: 0 a plan was granted, 2 the planner refused, 1 usage error.

#include <cstdint>
#include <exception>
#include <iostream>
#include <string>
#include <variant>

#include "sepal/config.hpp"
#include "sepal/planner.hpp"

namespace {

void usage(std::ostream& out) {
    out << "usage: sepal-info <preset> [options]\n"
           "  <preset>               0.8b | 2b | 4b | 9b\n"
           "  --weights <profile>    w16 | w8 | gpu-f16 | gpu-q8w | cpu-q8   (default w16)\n"
           "  --budget-gib <x>       device memory available, in GiB         (default 12)\n"
           "  --reserve-gib <x>      memory held back from the budget         (default 0)\n"
           "  --tokens <n>           token budget of one pass                 (default 4096)\n"
           "  --rows <n>             rows in one pass                         (default 32)\n"
           "  --snapshot-tokens <n>  state tokens per guaranteed snapshot     (default 2000)\n"
           "  --snapshots <n>        guaranteed snapshots                     (default 1)\n"
           "  --embedding-on-device  count the embedding table against the budget (CPU hosts)\n";
}

std::uint64_t gib_to_bytes(double gib) { return static_cast<std::uint64_t>(gib * 1024.0 * 1024.0 * 1024.0); }

std::string with_commas(std::uint64_t value) {
    std::string digits = std::to_string(value);
    for (std::size_t i = digits.size(); i > 3; i -= 3) digits.insert(i - 3, ",");
    return digits;
}

int run(int argc, char** argv) {
    if (argc < 2) {
        usage(std::cerr);
        return 1;
    }
    const std::string preset_name = argv[1];
    if (preset_name == "-h" || preset_name == "--help") {
        usage(std::cout);
        return 0;
    }

    std::string profile_id = "w16";
    double budget_gib = 12.0;
    double reserve_gib = 0.0;
    sepal::PlanRequest request;
    request.token_budget = 4096;
    request.min_snapshot_tokens = 2000;

    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--embedding-on-device") {
            request.embedding_on_device = true;
            continue;
        }
        if (i + 1 >= argc) {
            std::cerr << "sepal-info: option '" << arg << "' is unknown or is missing its value\n";
            return 1;
        }
        const std::string value = argv[++i];
        try {
            if (arg == "--weights") {
                profile_id = value;
            } else if (arg == "--budget-gib") {
                budget_gib = std::stod(value);
            } else if (arg == "--reserve-gib") {
                reserve_gib = std::stod(value);
            } else if (arg == "--tokens") {
                request.token_budget = std::stoull(value);
            } else if (arg == "--rows") {
                request.max_rows = std::stoull(value);
            } else if (arg == "--snapshot-tokens") {
                request.min_snapshot_tokens = std::stoull(value);
            } else if (arg == "--snapshots") {
                request.min_snapshots = std::stoull(value);
            } else {
                std::cerr << "sepal-info: unknown option '" << arg << "'\n";
                return 1;
            }
        } catch (const std::exception&) {
            std::cerr << "sepal-info: '" << value << "' is not a valid value for " << arg << "\n";
            return 1;
        }
    }
    if (!(budget_gib > 0.0) || reserve_gib < 0.0 || budget_gib > 1e6 || reserve_gib > 1e6) {
        std::cerr << "sepal-info: --budget-gib must be positive and --reserve-gib must not be negative\n";
        return 1;
    }

    const auto preset = sepal::find_preset(preset_name);
    if (!preset) {
        std::cerr << "sepal-info: unknown preset '" << preset_name << "' (expected 0.8b, 2b, 4b or 9b)\n";
        return 1;
    }
    const auto profile = sepal::find_profile(profile_id);
    if (!profile) {
        std::cerr << "sepal-info: unknown profile '" << profile_id
                  << "' (expected w16, w8, gpu-f16, gpu-q8w or cpu-q8)\n";
        return 1;
    }

    const sepal::ModelConfig& cfg = *preset;
    const sepal::ParamCounts params = sepal::count_params(cfg);
    using sepal::DType;
    using sepal::format_bytes;

    std::cout << "preset " << cfg.name << "\n"
              << "  layers                     " << cfg.num_layers << " = " << cfg.num_deltanet_layers()
              << " DeltaNet + " << cfg.num_attention_layers() << " full attention\n"
              << "  hidden / MLP               " << cfg.hidden_size << " / " << cfg.intermediate_size << "\n"
              << "  attention heads / KV / dim " << cfg.attn_heads << " / " << cfg.attn_kv_heads << " / "
              << cfg.attn_head_dim << " (" << cfg.attn_rotary_dims() << " dims rotate)\n"
              << "  DeltaNet key / value heads " << cfg.dn_key_heads << " / " << cfg.dn_value_heads << ", dims "
              << cfg.dn_key_head_dim << " / " << cfg.dn_value_head_dim << ", conv " << cfg.dn_conv_dim()
              << " channels x " << cfg.conv_kernel << " taps\n"
              << "derived\n"
              << "  non-embedding parameters   " << with_commas(params.non_embedding()) << "\n"
              << "  embedding parameters       " << with_commas(params.embedding) << "\n"
              << "  FLOPs per token            " << with_commas(sepal::flops_per_token(cfg)) << "\n"
              << "  trunk weights, 16-bit      " << format_bytes(params.non_embedding() * 2) << "\n"
              << "  trunk weights, 8-bit       " << format_bytes(params.non_embedding()) << "\n"
              << "  K/V per state token, f16   " << with_commas(sepal::kv_bytes_per_token(cfg, DType::F16))
              << " bytes\n"
              << "  recurrent state, f32       " << with_commas(sepal::recurrent_state_bytes(cfg, DType::F32))
              << " bytes\n"
              << "  conv tails, f32            " << with_commas(sepal::conv_tail_bytes(cfg, DType::F32))
              << " bytes\n";

    request.config = cfg;
    request.profile = *profile;
    request.available_bytes = gib_to_bytes(budget_gib);
    request.reserve_bytes = gib_to_bytes(reserve_gib);

    std::cout << "request\n"
              << "  profile " << profile->id << ", budget " << format_bytes(request.available_bytes) << ", reserve "
              << format_bytes(request.reserve_bytes) << "\n"
              << "  token budget " << request.token_budget << ", rows " << request.max_rows << ", "
              << request.min_snapshots << " snapshot(s) of " << request.min_snapshot_tokens << " state tokens\n"
              << "  trunk weights under this profile: "
              << format_bytes(sepal::trunk_weight_bytes(cfg, *profile)) << "\n";

    const sepal::PlanResult result = sepal::make_plan(request);
    if (const auto* refusal = std::get_if<sepal::Refusal>(&result)) {
        std::cout << "REFUSED (knob: " << sepal::name_of(refusal->knob) << "): " << refusal->message << "\n";
        return 2;
    }
    const sepal::Plan& plan = std::get<sepal::Plan>(result);
    std::cout << "PLAN (working-buffer figure is an estimate, not a measurement)\n"
              << "  weights                    " << format_bytes(plan.weights_bytes) << "\n";
    if (request.embedding_on_device) {
        std::cout << "  embedding                  " << format_bytes(plan.embedding_bytes) << "\n";
    } else {
        std::cout << "  embedding (host, uncounted) " << format_bytes(plan.embedding_host_bytes) << "\n";
    }
    std::cout << "  working buffers            " << format_bytes(plan.working_bytes) << "\n"
              << "  snapshot arena             " << format_bytes(plan.snapshot_arena_bytes) << " = "
              << plan.snapshots_held << " snapshot(s) of " << request.min_snapshot_tokens << " tokens\n"
              << "  reserve                    " << format_bytes(plan.reserve_bytes) << "\n"
              << "  largest token budget       " << plan.max_token_budget << "\n";
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "sepal-info: " << e.what() << "\n";
        return 1;
    }
}
