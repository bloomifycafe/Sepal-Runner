// sepal-cuda-kernel-bench: time per launch and memory traffic of the elementwise, norm, MLP, embedding and
// readout kernels (src/kernels_elementwise.*) at the layer geometries of the model presets.
//
//   sepal-cuda-kernel-bench [--geoms 1024:3584,2048:6144] [--tokens 1024,4096] [--reps 30] [--warmup 5]
//                           [--readout-rows 256] [--vocab 16384]
//
// A geometry is hidden:intermediate. Each kernel is timed on its own: before every timed launch a 128 MiB buffer
// is written, so the card's L2 cache holds none of the kernel's input, and the launch is bracketed by two events on
// the stream and followed by a wait. The figure is the MINIMUM over the repetitions, because other processes use the
// same GPU and only the minimum is not inflated by them. GB/s is the bytes the kernel must read and write (the
// weight vectors, a few KiB, are left out) divided by that time; it includes the launch's own fixed cost, which is
// what a pass pays too. Other agents may add kernels to this file's table by adding a case to `run_geometry`.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "device.hpp"
#include "kernels_elementwise.hpp"

namespace {

using sepal::cuda::check;
using sepal::cuda::DeviceBuffer;
using sepal::cuda::Elem;

struct Options {
    std::vector<std::pair<std::size_t, std::size_t>> geoms = {{1024, 3584}, {2048, 6144}};
    std::vector<std::size_t> tokens = {1024, 4096};
    int reps = 30;
    int warmup = 5;
    std::size_t readout_rows = 256;
    std::size_t vocab = 16384;
};

std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> parts;
    std::size_t begin = 0;
    while (begin <= s.size()) {
        const std::size_t at = s.find(sep, begin);
        const std::size_t end = at == std::string::npos ? s.size() : at;
        if (end > begin) parts.push_back(s.substr(begin, end - begin));
        begin = end + 1;
    }
    return parts;
}

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const auto value = [&]() -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument("missing value after " + a);
            return argv[++i];
        };
        if (a == "--geoms") {
            o.geoms.clear();
            for (const std::string& g : split(value(), ',')) {
                const std::vector<std::string> hi = split(g, ':');
                if (hi.size() != 2) throw std::invalid_argument("a geometry is hidden:intermediate");
                o.geoms.emplace_back(std::stoul(hi[0]), std::stoul(hi[1]));
            }
        } else if (a == "--tokens") {
            o.tokens.clear();
            for (const std::string& t : split(value(), ',')) o.tokens.push_back(std::stoul(t));
        } else if (a == "--reps") {
            o.reps = std::stoi(value());
        } else if (a == "--warmup") {
            o.warmup = std::stoi(value());
        } else if (a == "--readout-rows") {
            o.readout_rows = std::stoul(value());
        } else if (a == "--vocab") {
            o.vocab = std::stoul(value());
        } else {
            throw std::invalid_argument("unknown option " + a);
        }
    }
    if (o.reps < 20) throw std::invalid_argument("--reps must be at least 20");
    return o;
}

std::vector<float> random_floats(std::size_t count, float scale, std::uint64_t seed) {
    std::vector<float> v(count);
    std::uint64_t s = seed;
    for (float& f : v) {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        f = (static_cast<float>(s >> 40) * (1.0f / 8388608.0f) - 1.0f) * scale;
    }
    return v;
}

class Timer {
public:
    explicit Timer(const Options& o) : options_(o), flush_(std::size_t{128} << 20) {
        // A blocking stream, so that the uploads of DeviceBuffer (default stream) are complete before the first launch.
        check(cudaStreamCreate(&stream_), "cudaStreamCreate");
        check(cudaEventCreate(&start_), "cudaEventCreate");
        check(cudaEventCreate(&stop_), "cudaEventCreate");
    }
    ~Timer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
        cudaStreamDestroy(stream_);
    }
    Timer(const Timer&) = delete;
    Timer& operator=(const Timer&) = delete;

    cudaStream_t stream() const { return stream_; }

    /// Minimum over the repetitions of the time between two events around one launch, in milliseconds.
    double min_ms(const std::function<void()>& launch) {
        for (int i = 0; i < options_.warmup; ++i) launch();
        check(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
        float best = std::numeric_limits<float>::max();
        for (int i = 0; i < options_.reps; ++i) {
            check(cudaMemsetAsync(flush_.data(), i, flush_.size(), stream_), "cudaMemsetAsync");
            check(cudaEventRecord(start_, stream_), "cudaEventRecord");
            launch();
            check(cudaEventRecord(stop_, stream_), "cudaEventRecord");
            check(cudaEventSynchronize(stop_), "cudaEventSynchronize");
            float ms = 0.0f;
            check(cudaEventElapsedTime(&ms, start_, stop_), "cudaEventElapsedTime");
            best = std::min(best, ms);
        }
        return static_cast<double>(best);
    }

private:
    const Options& options_;
    DeviceBuffer<unsigned char> flush_;
    cudaStream_t stream_ = nullptr;
    cudaEvent_t start_ = nullptr, stop_ = nullptr;
};

/// One line of the table, and the sums the per-layer line is made from.
struct Result {
    std::string name;
    double ms;
    double bytes;
};

void print_row(std::size_t hidden, std::size_t inter, std::size_t n, const Result& r) {
    std::printf("%5zu %5zu %5zu  %-30s %9.4f ms  %9.3f MB  %7.1f GB/s\n", hidden, inter, n, r.name.c_str(), r.ms,
                r.bytes / 1e6, r.bytes / (r.ms * 1e-3) / 1e9);
}

void run_geometry(Timer& timer, const Options& o, std::size_t hidden, std::size_t inter, std::size_t n) {
    const cudaStream_t s = timer.stream();
    const DeviceBuffer<float> x(random_floats(n * hidden, 2.0f, 1));
    const DeviceBuffer<float> m(random_floats(n * hidden, 2.0f, 2));
    const DeviceBuffer<float> weight(random_floats(hidden, 0.3f, 3));
    const DeviceBuffer<float> gate_up(random_floats(n * 2 * inter, 4.0f, 4));
    DeviceBuffer<float> r_out(n * hidden);
    DeviceBuffer<float> out32(n * hidden);
    DeviceBuffer<std::uint16_t> out16(n * hidden);
    DeviceBuffer<float> mlp32(n * inter);
    DeviceBuffer<std::uint16_t> mlp16(n * inter);
    const DeviceBuffer<float> table32(random_floats(o.vocab * hidden, 1.0f, 5));
    const DeviceBuffer<std::uint16_t> table16(std::vector<std::uint16_t>(o.vocab * hidden, 0x3C00));
    std::vector<std::uint32_t> host_ids(n);
    for (std::size_t t = 0; t < n; ++t) host_ids[t] = static_cast<std::uint32_t>((t * 2654435761u) % o.vocab);
    const DeviceBuffer<std::uint32_t> ids(host_ids);
    std::vector<std::uint32_t> host_rows(o.readout_rows);
    for (std::size_t j = 0; j < host_rows.size(); ++j) host_rows[j] = static_cast<std::uint32_t>((j * 7919u) % n);
    const DeviceBuffer<std::uint32_t> rows(host_rows);
    DeviceBuffer<float> readout_out(o.readout_rows * hidden);

    const double nh = static_cast<double>(n * hidden);
    const double ni = static_cast<double>(n * inter);
    std::vector<Result> results;
    const auto time = [&](const std::string& name, double bytes, const std::function<void()>& launch) {
        results.push_back({name, timer.min_ms(launch), bytes});
        print_row(hidden, inter, n, results.back());
    };

    time("embed (f16 table)", nh * 2 + nh * 4, [&] {
        sepal::cuda::launch_embed(ids.data(), n, table16.data(), Elem::F16, o.vocab, hidden, out32.data(), hidden, s);
    });
    time("embed (f32 table)", nh * 4 + nh * 4, [&] {
        sepal::cuda::launch_embed(ids.data(), n, table32.data(), Elem::F32, o.vocab, hidden, out32.data(), hidden, s);
    });
    time("rms_norm -> f16", nh * 4 + nh * 2, [&] {
        sepal::cuda::launch_rms_norm(x.data(), hidden, weight.data(), n, hidden, 1e-6f, out16.data(), Elem::F16, hidden, s);
    });
    time("rms_norm -> f32", nh * 4 + nh * 4, [&] {
        sepal::cuda::launch_rms_norm(x.data(), hidden, weight.data(), n, hidden, 1e-6f, out32.data(), Elem::F32, hidden, s);
    });
    time("add_rms_norm -> f16", nh * 8 + nh * 4 + nh * 2, [&] {
        sepal::cuda::launch_add_rms_norm(x.data(), hidden, m.data(), hidden, r_out.data(), hidden, weight.data(), n, hidden,
                                         1e-6f, out16.data(), Elem::F16, hidden, s);
    });
    time("add_rms_norm -> f32", nh * 8 + nh * 4 + nh * 4, [&] {
        sepal::cuda::launch_add_rms_norm(x.data(), hidden, m.data(), hidden, r_out.data(), hidden, weight.data(), n, hidden,
                                         1e-6f, out32.data(), Elem::F32, hidden, s);
    });
    time("add (f32)", nh * 12, [&] {
        sepal::cuda::launch_add(x.data(), hidden, m.data(), hidden, r_out.data(), hidden, n, hidden, s);
    });
    time("swiglu -> f16", ni * 8 + ni * 2, [&] {
        sepal::cuda::launch_swiglu(gate_up.data(), 2 * inter, gate_up.data() + inter, 2 * inter, n, inter, mlp16.data(),
                                   Elem::F16, inter, s);
    });
    time("swiglu -> f32", ni * 8 + ni * 4, [&] {
        sepal::cuda::launch_swiglu(gate_up.data(), 2 * inter, gate_up.data() + inter, 2 * inter, n, inter, mlp32.data(),
                                   Elem::F32, inter, s);
    });
    time("readout_norm (" + std::to_string(o.readout_rows) + " rows)", static_cast<double>(o.readout_rows * hidden) * 8, [&] {
        sepal::cuda::launch_readout_norm(x.data(), hidden, rows.data(), o.readout_rows, weight.data(), hidden, 1e-6f,
                                         readout_out.data(), hidden, s);
    });

    // What one decoder layer spends here outside the multiplies and the mixers' own work: the residual add and norm
    // after the mixer, the residual add and next layer's input norm after the MLP, and the SwiGLU product.
    const auto find = [&](const std::string& name) {
        for (const Result& r : results) {
            if (r.name == name) return r;
        }
        return Result{name, 0.0, 0.0};
    };
    for (const char* suffix : {"f16", "f32"}) {
        const double layer = 2 * find(std::string("add_rms_norm -> ") + suffix).ms + find(std::string("swiglu -> ") + suffix).ms;
        std::printf("%5zu %5zu %5zu  per layer, %s (2 x add_rms_norm + swiglu): %.4f ms; x 24 layers: %.2f ms\n", hidden, inter,
                    n, suffix, layer, 24 * layer);
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options o = parse(argc, argv);
        cudaDeviceProp prop{};
        check(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
        std::printf("%s; minimum of %d repetitions after %d warm-up launches, L2 flushed before each; hidden/inter/N, then\n"
                    "kernel, time per launch, bytes moved, bandwidth\n",
                    prop.name, o.reps, o.warmup);
        Timer timer(o);
        for (const auto& [hidden, inter] : o.geoms) {
            for (const std::size_t n : o.tokens) run_geometry(timer, o, hidden, inter, n);
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "sepal-cuda-kernel-bench: %s\n", e.what());
        return 1;
    }
}
