// sepal-cuda-deltanet-bench: time per layer of the Gated DeltaNet kernels on the shapes of a pass.
//
//   sepal-cuda-deltanet-bench [--presets 0.8b,2b] [--reps 30] [--device 0]
//
// Shapes: (a) 16 rows x 640 tokens; (b) 1 row x 4096 tokens; (c) 8 rows x 640 tokens in a first wave and 64 rows x 27
// tokens in a second wave (alert-shaped: states plus short branches that continue them, 8 per state); (d) 1 row x 600
// and (e) 1 row x 2048 tokens, the single-row calls of the engine benchmark. Random inputs, the output of the layer is
// binary16 (the input of the out_proj multiply). Each repetition is timed with CUDA events around the launches and
// ends with a wait for the device; the table reports the MINIMUM over the repetitions, because other processes share
// the GPU and the median is noisy.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

#include <cuda_fp16.h>

#include "kernels_deltanet.hpp"
#include "sepal/config.hpp"

namespace {

using namespace sepal::cuda;

struct Options {
    std::vector<std::string> presets = {"0.8b", "2b"};
    int reps = 30;
    int device = 0;
};

struct Shape {
    const char* label;
    std::vector<std::uint32_t> wave1_lengths; // rows of the first wave
    std::vector<std::uint32_t> wave2_lengths; // rows of the second wave (each continues one of the first)
};

struct Timing {
    double scan_ms = 1e30, tail_ms = 1e30, total_ms = 1e30;
};

// Cheap deterministic fill, uploaded to the device.
void fill(float* data, std::size_t count, std::uint32_t seed, float lo, float hi) {
    std::vector<float> host(count);
    for (std::size_t i = 0; i < count; ++i) {
        std::uint32_t x = static_cast<std::uint32_t>(i) * 2654435761u + seed;
        x ^= x >> 15;
        x *= 2246822519u;
        x ^= x >> 13;
        x *= 3266489917u;
        x ^= x >> 16;
        host[i] = lo + (hi - lo) * (static_cast<float>(x >> 8) * (1.0f / 16777216.0f));
    }
    check(cudaMemcpy(data, host.data(), count * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy");
}

// Runs the shape `reps` times with the given lanes (0 = automatic) and returns the minima.
Timing run_shape(const DeltaNetGeometry& g, const Shape& shape, int lanes, int reps, std::size_t* tokens_out) {
    const std::uint32_t rows1 = static_cast<std::uint32_t>(shape.wave1_lengths.size());
    const std::uint32_t rows2 = static_cast<std::uint32_t>(shape.wave2_lengths.size());
    std::vector<RowSpan> spans;
    std::uint32_t total = 0;
    for (const auto* lengths : {&shape.wave1_lengths, &shape.wave2_lengths}) {
        for (const std::uint32_t len : *lengths) {
            spans.push_back({total, len});
            total += len;
        }
    }
    *tokens_out = total;
    // Slots: first-wave row i reads slot i (an initial state) and writes slot rows1 + i. A second-wave row j
    // reads the outgoing slot of first-wave row j / (rows2 / rows1) and writes its own slot.
    std::vector<std::uint32_t> in_slots, out_slots, wave1, wave2;
    for (std::uint32_t i = 0; i < rows1; ++i) {
        in_slots.push_back(i);
        out_slots.push_back(rows1 + i);
        wave1.push_back(i);
    }
    for (std::uint32_t j = 0; j < rows2; ++j) {
        in_slots.push_back(rows1 + j / (rows2 / rows1));
        out_slots.push_back(2 * rows1 + j);
        wave2.push_back(rows1 + j);
    }
    const std::size_t slots = 2 * rows1 + rows2;

    const std::size_t conv_dim = g.conv_dim(), value_dim = g.value_dim(), hv = g.value_heads;
    const std::size_t stride = conv_dim + value_dim + 2 * hv;
    DeviceBuffer<float> proj(static_cast<std::size_t>(total) * stride);
    fill(proj.data(), proj.size(), 1, -2.0f, 2.0f);
    DeviceBuffer<float> conv_w(conv_dim * g.conv_kernel), a_log(hv), dt_bias(hv), norm_w(g.value_head_dim);
    fill(conv_w.data(), conv_w.size(), 2, -0.5f, 0.5f);
    fill(a_log.data(), hv, 3, -3.0f, 0.5f);
    fill(dt_bias.data(), hv, 4, -3.0f, 0.0f);
    fill(norm_w.data(), norm_w.size(), 5, 0.5f, 1.5f);
    DeviceBuffer<float> tails(slots * g.conv_tail_floats()), states(slots * g.recurrent_floats());
    fill(tails.data(), tails.size(), 6, -1.0f, 1.0f);
    fill(states.data(), states.size(), 7, -0.3f, 0.3f);
    DeviceBuffer<__half> out(static_cast<std::size_t>(total) * value_dim);
    check(cudaMemset(out.data(), 0, out.size() * sizeof(__half)), "cudaMemset");
    const DeviceBuffer<RowSpan> d_rows(spans);
    const DeviceBuffer<std::uint32_t> d_in(in_slots), d_out(out_slots), d_w1(wave1);
    const DeviceBuffer<std::uint32_t> d_w2(wave2.empty() ? std::vector<std::uint32_t>(1) : wave2);

    const DeltaNetWeights weights{conv_w.data(), a_log.data(), dt_bias.data(), norm_w.data()};
    const DeltaNetInputs inputs{{proj.data(), stride},
                                {proj.data() + conv_dim, stride},
                                {proj.data() + conv_dim + value_dim, stride},
                                {proj.data() + conv_dim + value_dim + hv, stride}};
    const DeltaNetStates state_arrays{tails.data(), states.data()};
    DeltaNetOptions options;
    options.lanes_per_column = lanes;
    const DeltaNetRows r1{d_rows.data(), d_in.data(), d_out.data(), d_w1.data(), rows1};
    const DeltaNetRows r2{d_rows.data(), d_in.data(), d_out.data(), d_w2.data(), rows2};

    cudaEvent_t e[6];
    for (cudaEvent_t& ev : e) check(cudaEventCreate(&ev), "cudaEventCreate");
    Timing best;
    for (int rep = -2; rep < reps; ++rep) { // two warm-up repetitions
        float scan = 0.0f, tail = 0.0f, whole = 0.0f;
        cudaEvent_t whole_start, whole_stop;
        check(cudaEventCreate(&whole_start), "cudaEventCreate");
        check(cudaEventCreate(&whole_stop), "cudaEventCreate");
        check(cudaEventRecord(whole_start, nullptr), "cudaEventRecord");
        for (const DeltaNetRows* rows : {&r1, &r2}) {
            if (rows->wave_size == 0) continue;
            float ms = 0.0f;
            check(cudaEventRecord(e[0], nullptr), "cudaEventRecord");
            launch_deltanet_scan(g, weights, inputs, *rows, state_arrays, out.data(), value_dim, nullptr, options);
            check(cudaEventRecord(e[1], nullptr), "cudaEventRecord");
            launch_deltanet_conv_tail(g, inputs, *rows, state_arrays, nullptr);
            check(cudaEventRecord(e[2], nullptr), "cudaEventRecord");
            check(cudaEventSynchronize(e[2]), "cudaEventSynchronize");
            check(cudaEventElapsedTime(&ms, e[0], e[1]), "cudaEventElapsedTime");
            scan += ms;
            check(cudaEventElapsedTime(&ms, e[1], e[2]), "cudaEventElapsedTime");
            tail += ms;
        }
        check(cudaEventRecord(whole_stop, nullptr), "cudaEventRecord");
        check(cudaEventSynchronize(whole_stop), "cudaEventSynchronize");
        check(cudaEventElapsedTime(&whole, whole_start, whole_stop), "cudaEventElapsedTime");
        cudaEventDestroy(whole_start);
        cudaEventDestroy(whole_stop);
        if (rep < 0) continue;
        best.scan_ms = std::min(best.scan_ms, static_cast<double>(scan));
        best.tail_ms = std::min(best.tail_ms, static_cast<double>(tail));
        best.total_ms = std::min(best.total_ms, static_cast<double>(whole));
    }
    for (cudaEvent_t& ev : e) cudaEventDestroy(ev);
    return best;
}

std::vector<std::string> split(const std::string& s) {
    std::vector<std::string> parts;
    std::size_t at = 0;
    while (at <= s.size()) {
        const std::size_t comma = s.find(',', at);
        parts.push_back(s.substr(at, comma == std::string::npos ? std::string::npos : comma - at));
        if (comma == std::string::npos) break;
        at = comma + 1;
    }
    return parts;
}

} // namespace

int main(int argc, char** argv) {
    try {
        Options options;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--presets" && i + 1 < argc) {
                options.presets = split(argv[++i]);
            } else if (arg == "--reps" && i + 1 < argc) {
                options.reps = std::atoi(argv[++i]);
            } else if (arg == "--device" && i + 1 < argc) {
                options.device = std::atoi(argv[++i]);
            } else {
                std::fprintf(stderr, "usage: sepal-cuda-deltanet-bench [--presets 0.8b,2b] [--reps 30] [--device 0]\n");
                return 2;
            }
        }
        check(cudaSetDevice(options.device), "cudaSetDevice");
        cudaDeviceProp prop{};
        check(cudaGetDeviceProperties(&prop, options.device), "cudaGetDeviceProperties");
        std::printf("device: %s, %d SMs; minimum of %d repetitions after 2 warm-ups, synchronised; output binary16\n",
                    prop.name, prop.multiProcessorCount, options.reps);

        const std::vector<Shape> shapes = {
            {"(a) 16 x 640", std::vector<std::uint32_t>(16, 640), {}},
            {"(b) 1 x 4096", {4096}, {}},
            {"(c) 8 x 640 + 64 x 27", std::vector<std::uint32_t>(8, 640), std::vector<std::uint32_t>(64, 27)},
            {"(d) 1 x 600", {600}, {}},
            {"(e) 1 x 2048", {2048}, {}},
        };
        for (const std::string& name : options.presets) {
            const auto preset = sepal::find_preset(name);
            if (!preset) {
                std::fprintf(stderr, "unknown preset %s\n", name.c_str());
                return 2;
            }
            DeltaNetGeometry g;
            g.key_heads = static_cast<std::uint32_t>(preset->dn_key_heads);
            g.value_heads = static_cast<std::uint32_t>(preset->dn_value_heads);
            g.key_head_dim = static_cast<std::uint32_t>(preset->dn_key_head_dim);
            g.value_head_dim = static_cast<std::uint32_t>(preset->dn_value_head_dim);
            g.conv_kernel = static_cast<std::uint32_t>(preset->conv_kernel);
            g.rms_norm_eps = preset->rms_norm_eps;
            std::printf("\n%s: %u key heads, %u value heads, dk %u, dv %u, K %u; per layer, ms\n", name.c_str(),
                        g.key_heads, g.value_heads, g.key_head_dim, g.value_head_dim, g.conv_kernel);
            std::printf("%-24s %7s | %9s %9s %9s | %12s\n", "shape", "tokens", "scan", "conv tail", "total", "us / 1k tok");
            for (const Shape& shape : shapes) {
                std::size_t tokens = 0;
                const Timing t = run_shape(g, shape, 0, options.reps, &tokens);
                std::printf("%-24s %7zu | %9.3f %9.3f %9.3f | %12.1f\n", shape.label, tokens, t.scan_ms, t.tail_ms,
                            t.total_ms, 1000.0 * t.total_ms / (static_cast<double>(tokens) / 1000.0) / 1000.0);
            }
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
