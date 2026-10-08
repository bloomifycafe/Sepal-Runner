// sepal-bench: tokens per second of the UNOPTIMISED reference path, as a baseline for later work.
//
// One state row and a number of branches that continue it, in float32, with seeded random weights, on one
// thread. Every matrix multiply goes through the scalar reference GEMM provider. Nothing here is tuned, and
// the figure says nothing about what an optimised engine will reach.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "sepal/config.hpp"
#include "sepal/gemm.hpp"
#include "sepal/ref/executor.hpp"
#include "sepal/weights.hpp"

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
    std::vector<std::string> presets;
    std::size_t state_tokens = 512;
    std::size_t branches = 8;
    std::size_t branch_tokens = 50;
    std::uint64_t vocab = 4096;
    std::uint64_t seed = 1;
};

void usage() {
    std::cout << "usage: sepal-bench [--preset tiny|0.8b|2b|4b|9b]... [--state N] [--branches N] [--branch-tokens N]\n"
                 "                   [--vocab N] [--seed N]\n"
                 "  Runs one state row of --state tokens (default 512) and --branches rows (default 8) of\n"
                 "  --branch-tokens tokens (default 50) that continue it, in float32 with random weights, on one\n"
                 "  thread, and prints tokens per second. Without --preset it runs tiny and 0.8b.\n"
                 "  --vocab (default 4096) is the number of embedding rows allocated for a real geometry: the\n"
                 "  embedding is only looked up, so its size does not change the timing, and the full table of\n"
                 "  248,320 rows would only cost memory. 0 keeps the geometry's own vocabulary.\n";
}

double seconds_since(Clock::time_point start) { return std::chrono::duration<double>(Clock::now() - start).count(); }

std::vector<std::uint32_t> random_tokens(sepal::UnitRandom& rng, std::size_t count, std::uint64_t vocab) {
    std::vector<std::uint32_t> tokens(count);
    for (std::uint32_t& token : tokens) {
        const double unit = (rng.next() / 1.7320508075688772 + 1.0) / 2.0; // back to [0, 1)
        token = static_cast<std::uint32_t>(std::min<double>(unit * static_cast<double>(vocab), static_cast<double>(vocab - 1)));
    }
    return tokens;
}

int run_preset(const std::string& name, const Options& options) {
    sepal::ModelConfig config;
    if (name == "tiny") {
        config = sepal::tiny_test_config();
    } else if (const auto preset = sepal::find_preset(name)) {
        config = *preset;
        if (options.vocab != 0 && options.vocab < config.vocab_size) config.vocab_size = options.vocab;
    } else {
        std::cerr << "sepal-bench: unknown preset '" << name << "'\n";
        return 1;
    }
    const sepal::ParamCounts params = sepal::count_params(config);

    std::cout << "preset " << config.name << ": " << config.num_layers << " layers, hidden " << config.hidden_size << ", "
              << params.non_embedding() << " non-embedding parameters, embedding rows allocated " << config.vocab_size << "\n";

    auto start = Clock::now();
    const sepal::Weights<float> weights = sepal::random_weights<float>(config, options.seed);
    const auto gemm = sepal::make_reference_gemm();
    std::cout << "  random float32 weights built in " << std::fixed << std::setprecision(2) << seconds_since(start) << " s\n";

    sepal::UnitRandom rng(options.seed + 1);
    using Row = sepal::ref::Row<float>;

    Row state;
    state.token_ids = random_tokens(rng, options.state_tokens, config.vocab_size);
    state.readout_offsets = {options.state_tokens - 1};
    start = Clock::now();
    const std::vector<Row> state_request = {state};
    const std::vector<sepal::ref::RowResult<float>> state_result = sepal::ref::run_rows<float>(*gemm, config, weights, state_request);
    const double state_seconds = seconds_since(start);

    std::vector<Row> branches(options.branches);
    for (Row& branch : branches) {
        branch.token_ids = random_tokens(rng, options.branch_tokens, config.vocab_size);
        branch.start_position = options.state_tokens;
        branch.parent = state_result.at(0).snapshot;
        branch.readout_offsets = {options.branch_tokens - 1};
    }
    start = Clock::now();
    const std::vector<sepal::ref::RowResult<float>> branch_results = sepal::ref::run_rows<float>(*gemm, config, weights, branches);
    const double branch_seconds = seconds_since(start);

    // A checksum of the hidden states read out, so the work cannot be skipped, and a guard against a run that
    // went non-finite (random weights are scaled to keep activations bounded, but that is checked, not assumed).
    double checksum = 0.0;
    bool finite = true;
    const auto absorb = [&](const sepal::Tensor& hidden) {
        for (const float value : hidden.span()) {
            checksum += static_cast<double>(value);
            finite = finite && std::isfinite(value);
        }
    };
    absorb(state_result.at(0).hidden);
    for (const auto& result : branch_results) absorb(result.hidden);

    const std::size_t branch_total = options.branches * options.branch_tokens;
    const std::size_t total = options.state_tokens + branch_total;
    const auto rate = [](std::size_t tokens, double seconds) { return seconds > 0.0 ? static_cast<double>(tokens) / seconds : 0.0; };
    std::cout << std::setprecision(3);
    std::cout << "  state row : " << options.state_tokens << " tokens in " << state_seconds << " s = "
              << rate(options.state_tokens, state_seconds) << " tokens/s\n";
    std::cout << "  branches  : " << options.branches << " x " << options.branch_tokens << " = " << branch_total << " tokens in "
              << branch_seconds << " s = " << rate(branch_total, branch_seconds) << " tokens/s\n";
    std::cout << "  total     : " << total << " tokens in " << state_seconds + branch_seconds << " s = "
              << rate(total, state_seconds + branch_seconds) << " tokens/s\n";
    std::cout << "  checksum of the hidden states read out: " << std::setprecision(6) << checksum << (finite ? "" : "  (NOT FINITE)")
              << "\n";
    return finite ? 0 : 3;
}

} // namespace

int main(int argc, char** argv) {
    Options options;
    try {
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            const auto value = [&]() -> std::uint64_t {
                if (i + 1 >= argc) throw std::invalid_argument("missing value after " + arg);
                const std::string text = argv[++i];
                std::size_t used = 0;
                const unsigned long long parsed = std::stoull(text, &used);
                if (used != text.size()) throw std::invalid_argument("not a whole number: " + text);
                return parsed;
            };
            if (arg == "--help" || arg == "-h") {
                usage();
                return 0;
            } else if (arg == "--preset") {
                if (i + 1 >= argc) throw std::invalid_argument("missing value after --preset");
                options.presets.emplace_back(argv[++i]);
            } else if (arg == "--state") {
                options.state_tokens = static_cast<std::size_t>(value());
            } else if (arg == "--branches") {
                options.branches = static_cast<std::size_t>(value());
            } else if (arg == "--branch-tokens") {
                options.branch_tokens = static_cast<std::size_t>(value());
            } else if (arg == "--vocab") {
                options.vocab = value();
            } else if (arg == "--seed") {
                options.seed = value();
            } else {
                throw std::invalid_argument("unknown argument " + arg);
            }
        }
        if (options.state_tokens == 0 || options.branches == 0 || options.branch_tokens == 0) {
            throw std::invalid_argument("--state, --branches and --branch-tokens must be at least 1");
        }
    } catch (const std::exception& e) {
        std::cerr << "sepal-bench: " << e.what() << "\n";
        usage();
        return 1;
    }
    if (options.presets.empty()) options.presets = {"tiny", "0.8b"};

    std::cout << "sepal-bench: UNOPTIMISED REFERENCE PATH. float32, random weights, scalar reference GEMM, 1 thread.\n"
                 "This is a baseline for a deliberately unoptimised path, not a measure of the engine's speed.\n";
    try {
        for (const std::string& preset : options.presets) {
            const int status = run_preset(preset, options);
            if (status != 0) return status;
        }
    } catch (const std::exception& e) {
        std::cerr << "sepal-bench: " << e.what() << "\n";
        return 2;
    }
    return 0;
}
