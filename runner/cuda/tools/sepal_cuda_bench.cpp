// sepal-cuda-bench: matrix-multiply throughput of the CUDA provider on the layer shapes of the model presets.
//
// It measures the matrix multiplies ONLY. The recurrence, attention, convolution, normalisation and activation
// functions are not on the GPU yet, so every figure is an upper bound on engine throughput, not the engine's
// throughput.
//
//   sepal-cuda-bench [--presets 0.8b,2b,4b] [--tokens 64,256,1024,4096] [--precisions f16,f32]
//                    [--reps 5] [--warmup 2] [--device 0] [--reserve-mib 256]

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <map>
#include <string>
#include <vector>

#include "sepal/config.hpp"
#include "sepal/cuda/cuda_gemm.hpp"

namespace {

using sepal::DType;
using sepal::MatRef;
using sepal::MatView;
using sepal::cuda::CudaGemm;
using sepal::cuda::CudaGemmOptions;
using sepal::cuda::DeviceMatrix;
using sepal::cuda::Precision;
using sepal::cuda::WeightId;

constexpr double kMiB = 1048576.0;
constexpr double kGiB = 1073741824.0;

struct Options {
    std::vector<std::string> presets = {"0.8b", "2b", "4b"};
    std::vector<std::size_t> tokens = {64, 256, 1024, 4096};
    std::vector<Precision> precisions = {Precision::F16, Precision::F32};
    int reps = 5;
    int warmup = 2;
    int device = 0;
    std::size_t reserve_bytes = std::size_t{256} << 20;
};

std::vector<std::string> split(const std::string& s) {
    std::vector<std::string> parts;
    std::size_t begin = 0;
    while (begin <= s.size()) {
        const std::size_t comma = s.find(',', begin);
        const std::size_t end = comma == std::string::npos ? s.size() : comma;
        if (end > begin) parts.push_back(s.substr(begin, end - begin));
        begin = end + 1;
    }
    return parts;
}

[[noreturn]] void usage(const std::string& problem) {
    std::fprintf(stderr,
                 "%s\nusage: sepal-cuda-bench [--presets 0.8b,2b,4b] [--tokens 64,256,1024,4096] "
                 "[--precisions f16,f32] [--reps 5] [--warmup 2] [--device 0] [--reserve-mib 256]\n",
                 problem.c_str());
    std::exit(1);
}

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        if (i + 1 >= argc) usage("missing value for " + flag);
        const std::string value = argv[++i];
        if (flag == "--presets") {
            o.presets = split(value);
        } else if (flag == "--tokens") {
            o.tokens.clear();
            for (const std::string& t : split(value)) o.tokens.push_back(static_cast<std::size_t>(std::stoull(t)));
        } else if (flag == "--precisions") {
            o.precisions.clear();
            for (const std::string& p : split(value)) {
                if (p == "f16") {
                    o.precisions.push_back(Precision::F16);
                } else if (p == "f32") {
                    o.precisions.push_back(Precision::F32);
                } else {
                    usage("unknown precision " + p);
                }
            }
        } else if (flag == "--reps") {
            o.reps = std::stoi(value);
        } else if (flag == "--warmup") {
            o.warmup = std::stoi(value);
        } else if (flag == "--device") {
            o.device = std::stoi(value);
        } else if (flag == "--reserve-mib") {
            o.reserve_bytes = static_cast<std::size_t>(std::stoull(value)) << 20;
        } else {
            usage("unknown option " + flag);
        }
    }
    if (o.reps < 5) usage("--reps must be at least 5");
    if (o.warmup < 1) usage("--warmup must be at least 1");
    if (o.tokens.empty() || o.presets.empty() || o.precisions.empty()) usage("nothing to measure");
    for (const std::size_t t : o.tokens) {
        if (t == 0) usage("--tokens must be positive");
    }
    return o;
}

// One matrix multiply of one layer: Y[T, n] = X[T, k] W[n, k]^T.
struct Projection {
    const char* name;
    std::size_t n;
    std::size_t k;
};

// The GEMMs of layer `layer`, in the order the layer runs them (docs/RUNNER.md section 3): the mixer's
// concatenated in-projection, the mixer's out-projection, the MLP's gate and up as one GEMM, the MLP's down.
std::vector<Projection> layer_projections(const sepal::ModelConfig& c, std::uint64_t layer) {
    const std::size_t h = c.hidden_size, i = c.intermediate_size;
    std::vector<Projection> p;
    if (c.is_full_attention(layer)) {
        p.push_back({"attention in_proj", c.attn_in_proj_width(), h});
        p.push_back({"attention o_proj", h, c.attn_width()});
    } else {
        p.push_back({"deltanet in_proj", c.dn_in_proj_width(), h});
        p.push_back({"deltanet out_proj", h, c.dn_value_dim()});
    }
    p.push_back({"mlp gate+up", 2 * i, h});
    p.push_back({"mlp down", h, i});
    return p;
}

std::vector<float> random_pool(std::size_t count, float scale, std::uint64_t seed) {
    std::vector<float> v(count);
    std::uint64_t s = seed;
    for (float& f : v) {
        s ^= s << 13; // xorshift64
        s ^= s >> 7;
        s ^= s << 17;
        const float u = static_cast<float>(s >> 40) * (1.0f / 16777216.0f);
        f = (2.0f * u - 1.0f) * scale;
    }
    return v;
}

struct Timing {
    double median_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
};

template <typename Fn>
Timing measure(int warmup, int reps, Fn&& pass) {
    for (int i = 0; i < warmup; ++i) pass();
    std::vector<double> ms;
    for (int i = 0; i < reps; ++i) {
        const auto begin = std::chrono::steady_clock::now();
        pass(); // every pass ends by waiting for the device
        const auto end = std::chrono::steady_clock::now();
        ms.push_back(std::chrono::duration<double, std::milli>(end - begin).count());
    }
    std::sort(ms.begin(), ms.end());
    Timing t;
    t.min_ms = ms.front();
    t.max_ms = ms.back();
    const std::size_t mid = ms.size() / 2;
    t.median_ms = ms.size() % 2 == 1 ? ms[mid] : 0.5 * (ms[mid - 1] + ms[mid]);
    return t;
}

const char* precision_name(Precision p) { return p == Precision::F16 ? "fp16" : "fp32"; }

struct Row {
    std::string preset;
    std::string precision;
    std::size_t tokens = 0;
    Timing mlp_device, mlp_host, model;
    double mlp_flops = 0.0;   // per token
    double model_flops = 0.0; // per token
};

void print_rows(const std::vector<Row>& rows, int reps, int warmup) {
    if (rows.empty()) return;
    std::printf("\nRESULTS  (median of %d timed passes after %d warm-up passes; ms per pass and tokens per second)\n",
                reps, warmup);
    std::printf("GEMM ONLY: an upper bound on engine throughput, not the engine's throughput.\n\n");
    std::printf("%-5s %-5s %6s | %10s %11s %7s | %10s %11s | %10s %11s %7s\n", "", "", "", "MLP layer", "MLP layer",
                "", "MLP layer", "MLP layer", "all GEMMs", "all GEMMs", "");
    std::printf("%-5s %-5s %6s | %10s %11s %7s | %10s %11s | %10s %11s %7s\n", "model", "prec", "tokens",
                "device ms", "tokens/s", "TFLOP/s", "host ms", "tokens/s", "device ms", "tokens/s", "TFLOP/s");
    std::printf("------------------------------------------------------------------------------------------------"
                "-----------\n");
    for (const Row& r : rows) {
        const double t = static_cast<double>(r.tokens);
        const auto rate = [t](const Timing& x) { return t / (x.median_ms / 1000.0); };
        std::printf("%-5s %-5s %6zu | %10.3f %11.0f %7.2f | %10.3f %11.0f | %10.2f %11.0f %7.2f\n", r.preset.c_str(),
                    r.precision.c_str(), r.tokens, r.mlp_device.median_ms, rate(r.mlp_device),
                    rate(r.mlp_device) * r.mlp_flops / 1e12, r.mlp_host.median_ms, rate(r.mlp_host),
                    r.model.median_ms, rate(r.model), rate(r.model) * r.model_flops / 1e12);
    }
    std::printf("\nColumns: 'MLP layer device' is the two GEMMs of one MLP (gate+up, down) with operands already on\n"
                "the device. 'MLP layer host' is the same two GEMMs through GemmProvider::gemm_f, which uploads the\n"
                "activations and downloads the float32 result on every call. 'all GEMMs' is every projection of\n"
                "every layer of the model, each with its own weights, operands on the device. TFLOP/s counts two\n"
                "floating-point operations per weight per token.\n");
}

bool run_case(const Options& opt, const sepal::ModelConfig& cfg, Precision precision, std::vector<Row>& rows) {
    const DType compute = sepal::cuda::dtype_of(precision);
    const std::size_t elem = sepal::bytes_of(compute);
    const std::size_t t_max = *std::max_element(opt.tokens.begin(), opt.tokens.end());
    const std::size_t h = cfg.hidden_size, inter = cfg.intermediate_size;

    // Enumerate every projection of every layer.
    std::vector<std::vector<Projection>> layers;
    std::uint64_t weight_elems = 0;
    std::size_t max_weight_elems = 0, max_n = 0;
    std::map<std::size_t, bool> input_widths;
    std::size_t gemm_count = 0;
    for (std::uint64_t l = 0; l < cfg.num_layers; ++l) {
        layers.push_back(layer_projections(cfg, l));
        for (const Projection& p : layers.back()) {
            weight_elems += static_cast<std::uint64_t>(p.n) * p.k;
            max_weight_elems = std::max(max_weight_elems, p.n * p.k);
            max_n = std::max(max_n, p.n);
            input_widths[p.k] = true;
            ++gemm_count;
        }
    }
    const sepal::ParamCounts params = sepal::count_params(cfg);
    const std::uint64_t expected = params.mlp + params.mixer_proj + params.dn_sensitive;

    // The arena this case needs: weights, one input per distinct width, one output, and room for one call.
    const std::size_t pad = 4096; // alignment slack per allocation, generously
    const std::size_t weight_bytes = static_cast<std::size_t>(weight_elems) * elem + gemm_count * pad;
    std::size_t input_bytes = 0;
    for (const auto& [k, unused] : input_widths) input_bytes += t_max * k * elem + pad;
    const std::size_t output_bytes = t_max * max_n * sizeof(float) + pad;
    // Scratch: staging of a float32 host matrix before it is converted, or one host round trip of the MLP layer.
    const std::size_t staging = std::max(max_weight_elems, t_max * inter) * sizeof(float) + pad;
    const std::size_t round_trip = t_max * inter * (sizeof(float) + elem) + t_max * 2 * inter * sizeof(float) + 4 * pad;
    const std::size_t workspace = std::size_t{32} << 20;
    const std::size_t arena = weight_bytes + input_bytes + output_bytes + std::max(staging, round_trip) + workspace +
                              (std::size_t{16} << 20);

    const sepal::cuda::DeviceMemory mem = sepal::cuda::query_device_memory(opt.device);
    std::printf("\n== %s %s: %zu GEMMs in %llu layers, %llu weights (%s the geometry's %llu), %.2f GiB as %s; "
                "arena %.2f GiB; device free %.2f of %.2f GiB\n",
                cfg.name.c_str(), precision_name(precision), gemm_count,
                static_cast<unsigned long long>(cfg.num_layers), static_cast<unsigned long long>(weight_elems),
                weight_elems == expected ? "equal to" : "DIFFERENT FROM", static_cast<unsigned long long>(expected),
                static_cast<double>(weight_elems) * static_cast<double>(elem) / kGiB, precision_name(precision),
                static_cast<double>(arena) / kGiB, static_cast<double>(mem.free_bytes) / kGiB,
                static_cast<double>(mem.total_bytes) / kGiB);
    if (arena + opt.reserve_bytes > mem.free_bytes) {
        std::printf("   SKIPPED: this case needs %.2f GiB (%.2f GiB of weights) plus a reserve of %.0f MiB, and only "
                    "%.2f GiB of device memory is free. Not measured.\n",
                    static_cast<double>(arena) / kGiB,
                    static_cast<double>(weight_elems) * static_cast<double>(elem) / kGiB,
                    static_cast<double>(opt.reserve_bytes) / kMiB, static_cast<double>(mem.free_bytes) / kGiB);
        return false;
    }

    CudaGemm g(CudaGemmOptions{opt.device, precision, arena, workspace});

    // Random weights. The values do not change the time of a dense multiply; every matrix is cut from one pool
    // at a different offset so that the matrices are not copies of each other.
    const std::size_t pool_extra = 1 << 20;
    const std::vector<float> pool = random_pool(max_weight_elems + pool_extra, 0.02f, 0x5EBA1ull);
    std::vector<std::vector<WeightId>> ids(layers.size());
    std::size_t offset = 0;
    for (std::size_t l = 0; l < layers.size(); ++l) {
        for (const Projection& p : layers[l]) {
            offset = (offset + 7919) % pool_extra;
            ids[l].push_back(g.upload_weights(MatRef(pool.data() + offset, DType::F32, p.n, p.k)));
        }
    }

    // Activations: one device matrix per distinct input width, and host copies for the round trip.
    const std::size_t max_k = input_widths.rbegin()->first;
    const std::vector<float> act = random_pool(t_max * max_k, 1.0f, 0xAC71ull);
    std::map<std::size_t, DeviceMatrix> inputs;
    for (const auto& [k, unused] : input_widths) inputs[k] = g.upload_matrix(MatRef(act.data(), DType::F32, t_max, k));
    const DeviceMatrix output = g.alloc_output(t_max, max_n);
    std::vector<float> host_y(t_max * 2 * inter);

    // The MLP of layer 0: its last two projections.
    const std::size_t mlp_first = layers[0].size() - 2;
    const double mlp_flops = 2.0 * 3.0 * static_cast<double>(h) * static_cast<double>(inter);
    const double model_flops = 2.0 * static_cast<double>(weight_elems);

    for (const std::size_t t : opt.tokens) {
        const auto x_view = [&](std::size_t k) {
            DeviceMatrix m = inputs.at(k);
            m.rows = t;
            return m;
        };
        const auto y_view = [&](std::size_t n) { return DeviceMatrix{output.data, DType::F32, t, n}; };

        Row row;
        row.preset = cfg.name;
        row.precision = precision_name(precision);
        row.tokens = t;
        row.mlp_flops = mlp_flops;
        row.model_flops = model_flops;

        row.mlp_device = measure(opt.warmup, opt.reps, [&] {
            for (std::size_t j = mlp_first; j < layers[0].size(); ++j) {
                const Projection& p = layers[0][j];
                g.gemm_device(x_view(p.k), ids[0][j], y_view(p.n), 0.0f);
            }
            g.synchronize();
        });

        row.mlp_host = measure(opt.warmup, opt.reps, [&] {
            for (std::size_t j = mlp_first; j < layers[0].size(); ++j) {
                const Projection& p = layers[0][j];
                g.gemm_f(MatRef(act.data(), DType::F32, t, p.k), ids[0][j], MatView(host_y.data(), t, p.n), 0.0f);
            }
        });

        row.model = measure(opt.warmup, opt.reps, [&] {
            for (std::size_t l = 0; l < layers.size(); ++l) {
                for (std::size_t j = 0; j < layers[l].size(); ++j) {
                    const Projection& p = layers[l][j];
                    g.gemm_device(x_view(p.k), ids[l][j], y_view(p.n), 0.0f);
                }
            }
            g.synchronize();
        });

        // The last GEMM of the pass wrote Y[t, h]: check that real numbers came out.
        std::vector<float> check(t * h);
        g.download(DeviceMatrix{output.data, DType::F32, t, h}, MatView(check.data(), t, h));
        double sum_abs = 0.0;
        bool finite = true;
        for (const float f : check) {
            finite = finite && std::isfinite(f);
            sum_abs += std::fabs(static_cast<double>(f));
        }
        if (!finite || sum_abs == 0.0) {
            std::printf("   ERROR: the output of the last GEMM at %zu tokens is %s\n", t,
                        finite ? "all zero" : "not finite");
            return false;
        }
        std::printf("   %6zu tokens: MLP layer %.3f ms on device (%.3f to %.3f), %.3f ms through the host; "
                    "all GEMMs %.2f ms (%.2f to %.2f)\n",
                    t, row.mlp_device.median_ms, row.mlp_device.min_ms, row.mlp_device.max_ms,
                    row.mlp_host.median_ms, row.model.median_ms, row.model.min_ms, row.model.max_ms);
        std::fflush(stdout);
        rows.push_back(row);
    }

    const sepal::cuda::MemoryStats m = g.memory();
    std::printf("   PEAK DEVICE MEMORY: arena %.3f GiB allocated once, %.3f GiB of it used at peak (%.3f GiB resident "
                "+ %.3f GiB by the largest call); the device as a whole went from %.3f GiB free before the arena to "
                "%.3f GiB free at the lowest, so the provider with cuBLAS took %.3f GiB\n",
                static_cast<double>(m.arena_bytes) / kGiB, static_cast<double>(m.arena_peak_bytes) / kGiB,
                static_cast<double>(m.resident_bytes) / kGiB, static_cast<double>(m.scratch_peak_bytes) / kGiB,
                static_cast<double>(m.device_free_at_start) / kGiB, static_cast<double>(m.device_free_min) / kGiB,
                static_cast<double>(m.device_free_at_start - m.device_free_min) / kGiB);
    return true;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options opt = parse(argc, argv);

        std::printf("sepal-cuda-bench: MATRIX MULTIPLIES ONLY.\n"
                    "The recurrence, attention, convolution, normalisation and activations are not on the GPU yet,\n"
                    "so these figures are an upper bound on engine throughput, not the engine's throughput.\n"
                    "Weights are random; activations are random; outputs are float32.\n");
        {
            const CudaGemm probe(CudaGemmOptions{opt.device, Precision::F32, std::size_t{16} << 20, 0});
            const sepal::cuda::DeviceInfo info = probe.device_info();
            std::printf("device %d: %s, sm_%d%d, %.2f GiB; CUDA runtime %d, driver supports CUDA %d, cuBLAS %d\n",
                        info.device, info.name.c_str(), info.compute_major, info.compute_minor,
                        static_cast<double>(info.total_bytes) / kGiB, info.runtime_version, info.driver_version,
                        info.cublas_version);
        }

        std::vector<Row> rows;
        std::vector<std::string> skipped;
        for (const Precision precision : opt.precisions) {
            for (const std::string& name : opt.presets) {
                const auto cfg = sepal::find_preset(name);
                if (!cfg) usage("unknown preset " + name);
                if (!run_case(opt, *cfg, precision, rows)) {
                    skipped.push_back(name + " " + precision_name(precision));
                }
            }
        }
        print_rows(rows, opt.reps, opt.warmup);
        for (const std::string& s : skipped) std::printf("NOT MEASURED: %s (see the SKIPPED or ERROR line above)\n", s.c_str());
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "sepal-cuda-bench: %s\n", e.what());
        return 2;
    }
}
