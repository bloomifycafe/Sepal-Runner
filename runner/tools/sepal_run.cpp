// sepal-run: one forward pass of a real checkpoint on the CPU reference path.
//
//     sepal-run <checkpoint directory> <token file> [--readout <dir or file>] [--options N] [--precision f32|f64]
//
// The token file holds token ids as decimal numbers separated by spaces, commas or new lines (a Python list
// printed with `print(ids)` works too). The pass runs as one row from nothing; the final hidden state at the last
// token (after the final norm) is printed in part, and with a readout the probabilities over the first N options.
// The GPU engine is to take the same arguments.
//
// Exit status: 0 on success, 1 for a usage error, 2 when the checkpoint or the tokens are refused.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include "sepal/gemm.hpp"
#include "sepal/loader.hpp"
#include "sepal/ref/executor.hpp"

namespace {

using Clock = std::chrono::steady_clock;

void usage() {
    std::cout << "usage: sepal-run <checkpoint directory> <token file> [--readout <directory or readout.safetensors>]\n"
                 "                 [--options N] [--temperature T] [--precision f32|f64] [--show N]\n"
                 "  Runs the tokens as one row through the CPU reference path and prints the final hidden state at\n"
                 "  the last token (first --show values, default 8) and, with --readout, the probabilities over\n"
                 "  the first --options options (default 2). A readout directory supplies the temperature through\n"
                 "  its decision_config.json; for a bare file give --temperature.\n";
}

double seconds_since(Clock::time_point start) { return std::chrono::duration<double>(Clock::now() - start).count(); }

std::vector<std::uint32_t> read_tokens(const std::string& path) {
    std::ifstream in(std::filesystem::path(std::u8string(reinterpret_cast<const char8_t*>(path.data()), path.size())));
    if (!in) throw std::runtime_error("cannot read the token file '" + path + "'");
    const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    std::vector<std::uint32_t> tokens;
    std::size_t i = 0;
    while (i < text.size()) {
        if (text[i] >= '0' && text[i] <= '9') {
            std::uint64_t value = 0;
            for (; i < text.size() && text[i] >= '0' && text[i] <= '9'; ++i) {
                value = value * 10 + static_cast<std::uint64_t>(text[i] - '0');
                if (value > 0xFFFFFFFFull) throw std::runtime_error("a token id in the token file is above 2^32");
            }
            tokens.push_back(static_cast<std::uint32_t>(value));
        } else if (text[i] == ' ' || text[i] == ',' || text[i] == '\n' || text[i] == '\r' || text[i] == '\t' || text[i] == '[' ||
                   text[i] == ']') {
            ++i;
        } else {
            throw std::runtime_error(std::string("unexpected character '") + text[i] + "' in the token file");
        }
    }
    if (tokens.empty()) throw std::runtime_error("the token file holds no token ids");
    return tokens;
}

template <class T>
void run(const sepal::LoadedModel& model, const std::vector<std::uint32_t>& tokens, std::size_t show, std::vector<double>& hidden_out) {
    auto start = Clock::now();
    const sepal::Weights<T> weights = model.weights<T>();
    std::cout << "weights converted to " << (sizeof(T) == 4 ? "float32" : "float64") << " in " << std::fixed << std::setprecision(2)
              << seconds_since(start) << " s\n";
    const auto gemm = sepal::make_reference_gemm();
    sepal::ref::Row<T> row;
    row.token_ids = tokens;
    row.readout_offsets = {tokens.size() - 1};
    start = Clock::now();
    const std::vector<sepal::ref::Row<T>> rows = {row};
    const std::vector<sepal::ref::RowResult<T>> result = sepal::ref::run_rows<T>(*gemm, model.config(), weights, rows);
    const double elapsed = seconds_since(start);
    std::cout << "forward pass of " << tokens.size() << " tokens in " << elapsed << " s (" << static_cast<double>(tokens.size()) / elapsed
              << " tokens/s)\n";
    const auto hidden = result[0].hidden.mat().row(0);
    hidden_out.assign(hidden.begin(), hidden.end());
    std::cout << std::setprecision(9) << "final hidden state [" << hidden.size() << "], first " << std::min(show, hidden.size()) << ":";
    for (std::size_t i = 0; i < std::min(show, hidden.size()); ++i) std::cout << " " << static_cast<double>(hidden[i]);
    double sum_sq = 0.0;
    for (const T v : hidden) sum_sq += static_cast<double>(v) * static_cast<double>(v);
    std::cout << "\nfinal hidden state rms " << std::sqrt(sum_sq / static_cast<double>(hidden.size())) << "\n";
}

} // namespace

int main(int argc, char** argv) {
    std::vector<std::string> positional;
    std::string readout_path;
    std::size_t options = 2;
    std::size_t show = 8;
    double temperature = 0.0;
    std::string precision = "f32";
    try {
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            const auto value = [&]() -> std::string {
                if (i + 1 >= argc) throw std::invalid_argument(arg + " needs a value");
                return argv[++i];
            };
            if (arg == "--help" || arg == "-h") {
                usage();
                return 0;
            } else if (arg == "--readout") {
                readout_path = value();
            } else if (arg == "--options") {
                options = static_cast<std::size_t>(std::stoul(value()));
            } else if (arg == "--show") {
                show = static_cast<std::size_t>(std::stoul(value()));
            } else if (arg == "--temperature") {
                temperature = std::stod(value());
            } else if (arg == "--precision") {
                precision = value();
            } else if (!arg.empty() && arg[0] == '-') {
                throw std::invalid_argument("unknown option " + arg);
            } else {
                positional.push_back(arg);
            }
        }
        if (positional.size() != 2 || (precision != "f32" && precision != "f64")) throw std::invalid_argument("expected a checkpoint directory and a token file");
    } catch (const std::exception& error) {
        std::cerr << "sepal-run: " << error.what() << "\n";
        usage();
        return 1;
    }

    try {
        auto start = Clock::now();
        const sepal::LoadedModel model = sepal::LoadedModel::open(positional[0]);
        const sepal::ModelConfig& config = model.config();
        std::cout << "checkpoint " << positional[0] << ": " << config.num_layers << " layers, hidden " << config.hidden_size << ", vocab "
                  << config.vocab_size << ", " << model.raw().size() << " tensors (" << static_cast<double>(model.raw_bytes()) / 1e9
                  << " GB stored), " << model.ignored().size() << " ignored, opened in " << std::fixed << std::setprecision(2)
                  << seconds_since(start) << " s\n";
        std::cout << "closest preset " << model.config_reading().preset << ": "
                  << (model.config_reading().differences.empty() ? "no differences" : std::to_string(model.config_reading().differences.size()) + " difference(s)")
                  << "\n";
        for (const std::string& difference : model.config_reading().differences) std::cout << "  " << difference << "\n";

        const std::vector<std::uint32_t> tokens = read_tokens(positional[1]);
        std::vector<double> hidden;
        if (precision == "f32") {
            run<float>(model, tokens, show, hidden);
        } else {
            run<double>(model, tokens, show, hidden);
        }

        if (!readout_path.empty()) {
            const std::filesystem::path as_path(std::u8string(reinterpret_cast<const char8_t*>(readout_path.data()), readout_path.size()));
            const bool is_directory = std::filesystem::is_directory(as_path);
            if (!is_directory && !(temperature > 0.0)) throw std::invalid_argument("a readout file needs --temperature");
            const sepal::DecisionReadout readout =
                !is_directory ? sepal::DecisionReadout::open(readout_path, temperature)
                : temperature > 0.0 ? sepal::DecisionReadout::open(readout_path + "/readout.safetensors", temperature)
                                    : sepal::DecisionReadout::open_directory(readout_path);
            const std::vector<double> p = readout.probabilities(std::span<const double>(hidden), options);
            std::cout << "readout: " << readout.rows() << " rows, temperature " << std::setprecision(17) << readout.temperature() << "\n";
            std::cout << "probabilities over " << options << " options:";
            for (const double v : p) std::cout << " " << v;
            std::cout << "\n";
        }
    } catch (const std::exception& error) {
        std::cerr << "sepal-run: " << error.what() << "\n";
        return 2;
    }
    return 0;
}
