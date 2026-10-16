// sepal-score: raw logits for a JSONL workload through any backend library's C ABI.
//
//   sepal-score --final <dir> --base <dir> --profile <name> --input <workload.jsonl> --output <logits.jsonl>
//               [--library <path to sepal_runner_*.dll/.so>] [--threads N] [--max-tokens N] [--device N]
//               [--batch N] [--opt key=<json value>]...
//
// Input, one request per line (extra members such as "id" are copied to the output):
//   {"input_ids": [..], "question_types": [..], "instruction_spans": [[b,e],..],
//    "option_offsets": [0, n1, n1+n2, ..], "option_spans": [[b,e],..]}
// (spans may also be flat [b,e,b,e,..]). Output, one line per request, in order:
//   {"index": i, "id": .., "profile": "..", "logits": [[q0 option logits], [q1 option logits], ..]}
// The library defaults to sepal_runner_cpu next to this executable. Only the C ABI is used, so the CUDA library works
// the same way. Exit status: 0 success, 1 usage, 2 failure (the message is printed).

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "sepal/c_api.h"
#include "sepal/json.hpp"

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#else
#  include <dlfcn.h>
#endif

namespace {

struct Api {
    int32_t (*version)() = nullptr;
    const char* (*last_error)() = nullptr;
    int32_t (*open)(const char*, sepal_engine**) = nullptr;
    void (*close)(sepal_engine*) = nullptr;
    int32_t (*info)(sepal_engine*, char*, uint64_t, uint64_t*) = nullptr;
    int32_t (*score)(sepal_engine*, uint64_t, const uint32_t*, const uint64_t*, const uint64_t*, const uint32_t*, const uint32_t*,
                     const uint64_t*, const uint32_t*, float*, double*) = nullptr;
};

void* open_library(const std::string& path) {
#ifdef _WIN32
    return reinterpret_cast<void*>(LoadLibraryA(path.c_str()));
#else
    return dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
}

void* symbol(void* lib, const char* name) {
#ifdef _WIN32
    return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(lib), name));
#else
    return dlsym(lib, name);
#endif
}

template <class F>
void bind(void* lib, const char* name, F& target) {
    void* p = symbol(lib, name);
    if (p == nullptr) throw std::runtime_error(std::string("the library has no symbol ") + name);
    target = reinterpret_cast<F>(p);
}

std::string default_library(const char* argv0) {
    std::string dir(argv0);
    const std::size_t slash = dir.find_last_of("/\\");
    dir = slash == std::string::npos ? std::string() : dir.substr(0, slash + 1);
#ifdef _WIN32
    return dir + "sepal_runner_cpu.dll";
#else
    return dir + "libsepal_runner_cpu.so";
#endif
}

void usage() {
    std::cout << "usage: sepal-score --final <dir> --base <dir> --profile <name> --input <workload.jsonl> --output <logits.jsonl>\n"
                 "                   [--library <path>] [--threads N] [--max-tokens N] [--device N] [--batch N] [--opt key=<json>]...\n";
}

std::string json_string(const std::string& s) {
    std::string out = "\"";
    for (const char c : s) {
        if (c == '"' || c == '\\') {
            out += '\\';
            out += c;
        } else if (c == '\n') {
            out += "\\n";
        } else if (static_cast<unsigned char>(c) < 0x20) {
            out += ' ';
        } else {
            out += c;
        }
    }
    return out + "\"";
}

struct Request {
    std::vector<uint32_t> tokens;
    std::vector<uint32_t> types;
    std::vector<uint32_t> instruction;
    std::vector<uint64_t> option_offsets; // relative to this request's first option
    std::vector<uint32_t> option_spans;
    std::string passthrough;              // ready-made `"id": ...` text or empty
    std::size_t n_options() const { return option_spans.size() / 2; }
};

std::vector<uint32_t> spans_of(const sepal::json::Value& v) {
    std::vector<uint32_t> out;
    for (const sepal::json::Value& item : v.items()) {
        if (item.kind() == sepal::json::Kind::Array) {
            for (const sepal::json::Value& x : item.items()) out.push_back(static_cast<uint32_t>(x.as_uint()));
        } else {
            out.push_back(static_cast<uint32_t>(item.as_uint()));
        }
    }
    return out;
}

Request parse_request(const std::string& line) {
    const sepal::json::Value v = sepal::json::parse(line);
    Request r;
    for (const sepal::json::Value& t : v.at("input_ids").items()) r.tokens.push_back(static_cast<uint32_t>(t.as_uint()));
    for (const sepal::json::Value& t : v.at("question_types").items()) r.types.push_back(static_cast<uint32_t>(t.as_uint()));
    r.instruction = spans_of(v.at("instruction_spans"));
    for (const sepal::json::Value& t : v.at("option_offsets").items()) r.option_offsets.push_back(t.as_uint());
    r.option_spans = spans_of(v.at("option_spans"));
    if (r.instruction.size() != 2 * r.types.size()) throw std::runtime_error("instruction_spans does not match question_types");
    if (r.option_offsets.size() != r.types.size() + 1) throw std::runtime_error("option_offsets must have one entry more than question_types");
    if (r.option_offsets.back() * 2 != r.option_spans.size()) throw std::runtime_error("option_offsets do not match option_spans");
    if (const sepal::json::Value* id = v.find("id")) {
        if (id->kind() == sepal::json::Kind::String) r.passthrough = "\"id\": " + json_string(id->as_string()) + ", ";
        else if (id->kind() == sepal::json::Kind::Number) r.passthrough = "\"id\": " + std::to_string(id->as_uint()) + ", ";
    }
    return r;
}

} // namespace

int main(int argc, char** argv) {
    std::string final_dir, base_dir, profile, input, output, library;
    std::uint64_t threads = 0, max_tokens = 0, device = 0, batch = 1;
    bool have_threads = false, have_max = false, have_device = false;
    std::vector<std::string> opts;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const auto value = [&]() -> std::string {
            if (i + 1 >= argc) {
                usage();
                std::exit(1);
            }
            return argv[++i];
        };
        if (a == "--final") final_dir = value();
        else if (a == "--base") base_dir = value();
        else if (a == "--profile") profile = value();
        else if (a == "--input") input = value();
        else if (a == "--output") output = value();
        else if (a == "--library") library = value();
        else if (a == "--threads") threads = std::stoull(value()), have_threads = true;
        else if (a == "--max-tokens") max_tokens = std::stoull(value()), have_max = true;
        else if (a == "--device") device = std::stoull(value()), have_device = true;
        else if (a == "--batch") batch = std::stoull(value());
        else if (a == "--opt") opts.push_back(value());
        else {
            usage();
            return 1;
        }
    }
    if (final_dir.empty() || base_dir.empty() || profile.empty() || input.empty() || output.empty() || batch == 0) {
        usage();
        return 1;
    }
    try {
        if (library.empty()) library = default_library(argv[0]);
        void* lib = open_library(library);
        if (lib == nullptr) throw std::runtime_error("cannot load the library '" + library + "'");
        Api api;
        bind(lib, "sepal_api_version", api.version);
        bind(lib, "sepal_last_error", api.last_error);
        bind(lib, "sepal_open", api.open);
        bind(lib, "sepal_close", api.close);
        bind(lib, "sepal_info", api.info);
        bind(lib, "sepal_score", api.score);
        if (api.version() != SEPAL_C_API_VERSION) throw std::runtime_error("the library speaks C API version " + std::to_string(api.version()));

        std::string options = "{\"final_dir\": " + json_string(final_dir) + ", \"base_dir\": " + json_string(base_dir) + ", \"profile\": " + json_string(profile);
        if (have_threads) options += ", \"threads\": " + std::to_string(threads);
        if (have_max) options += ", \"max_tokens\": " + std::to_string(max_tokens);
        if (have_device) options += ", \"device\": " + std::to_string(device);
        for (const std::string& o : opts) {
            const std::size_t eq = o.find('=');
            if (eq == std::string::npos) throw std::runtime_error("--opt wants key=<json value>");
            options += ", " + json_string(o.substr(0, eq)) + ": " + o.substr(eq + 1);
        }
        options += "}";

        const auto open_start = std::chrono::steady_clock::now();
        sepal_engine* engine = nullptr;
        if (api.open(options.c_str(), &engine) != 0) throw std::runtime_error(std::string("sepal_open: ") + api.last_error());
        std::cerr << "opened " << profile << " in " << std::chrono::duration<double>(std::chrono::steady_clock::now() - open_start).count() << " s\n";

        std::ifstream in(input, std::ios::binary);
        if (!in) throw std::runtime_error("cannot read '" + input + "'");
        std::ofstream out(output, std::ios::binary);
        if (!out) throw std::runtime_error("cannot write '" + output + "'");

        std::vector<Request> pending;
        std::uint64_t index = 0, written = 0, total_tokens = 0;
        double total_ms = 0;
        const auto flush = [&] {
            if (pending.empty()) return;
            std::vector<uint32_t> tokens, types, instruction, spans;
            std::vector<uint64_t> token_offsets{0}, question_offsets{0}, option_offsets{0};
            std::uint64_t options_total = 0;
            for (const Request& r : pending) {
                tokens.insert(tokens.end(), r.tokens.begin(), r.tokens.end());
                token_offsets.push_back(tokens.size());
                types.insert(types.end(), r.types.begin(), r.types.end());
                question_offsets.push_back(types.size());
                instruction.insert(instruction.end(), r.instruction.begin(), r.instruction.end());
                for (std::size_t q = 0; q < r.types.size(); ++q) option_offsets.push_back(options_total + r.option_offsets[q + 1]);
                spans.insert(spans.end(), r.option_spans.begin(), r.option_spans.end());
                options_total += r.n_options();
            }
            std::vector<float> logits(options_total);
            double times[5] = {0, 0, 0, 0, 0};
            if (api.score(engine, pending.size(), tokens.data(), token_offsets.data(), question_offsets.data(), types.data(), instruction.data(),
                          option_offsets.data(), spans.data(), logits.data(), times) != 0) {
                throw std::runtime_error(std::string("sepal_score (requests ") + std::to_string(written) + ".." +
                                         std::to_string(written + pending.size() - 1) + "): " + api.last_error());
            }
            total_ms += times[2];
            total_tokens += static_cast<std::uint64_t>(times[3]);
            std::size_t at = 0;
            for (const Request& r : pending) {
                out << "{\"index\": " << written++ << ", " << r.passthrough << "\"profile\": " << json_string(profile) << ", \"logits\": [";
                for (std::size_t q = 0; q < r.types.size(); ++q) {
                    out << (q ? ", [" : "[");
                    for (std::uint64_t k = r.option_offsets[q]; k < r.option_offsets[q + 1]; ++k) {
                        char buf[32];
                        std::snprintf(buf, sizeof buf, "%.9g", static_cast<double>(logits[at++]));
                        out << (k > r.option_offsets[q] ? ", " : "") << buf;
                    }
                    out << "]";
                }
                out << "]}\n";
            }
            pending.clear();
        };
        std::string line;
        while (std::getline(in, line)) {
            ++index;
            if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
            try {
                pending.push_back(parse_request(line));
            } catch (const std::exception& e) {
                throw std::runtime_error("line " + std::to_string(index) + " of the workload: " + e.what());
            }
            if (pending.size() >= batch) flush();
        }
        flush();
        out.flush();
        api.close(engine);
        std::cerr << "scored " << written << " request(s), " << total_tokens << " tokens, engine time " << total_ms / 1000.0 << " s\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "sepal-score: " << e.what() << "\n";
        return 2;
    }
}
