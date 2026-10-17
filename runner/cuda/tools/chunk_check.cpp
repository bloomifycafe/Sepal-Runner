// Checks and times the chunked DeltaNet kernels (src/kernels_deltanet_chunked.cu) against the scan kernel (src/kernels_deltanet.cu).
//   sepal-cuda-chunk-check [--lengths 1000,64,130,4096] [--seed N] [--state on|off] [--half-state on|off] [--bench REPS] [--input float|half]
// Prints the largest difference of the outputs and of the final states over the rms of the scan kernel's, per row.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_fp16.h>

#include "kernels_deltanet.hpp"
#include "kernels_deltanet_chunked.hpp"

using namespace sepal::cuda;

namespace {

struct Rng {
    std::uint64_t s;
    std::uint64_t next() {
        std::uint64_t z = (s += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    float uniform(float lo, float hi) { return lo + (hi - lo) * (static_cast<float>(next() >> 40) * (1.0f / 16777216.0f)); }
};

std::vector<float> rnd(Rng& r, std::size_t n, float lo, float hi) {
    std::vector<float> v(n);
    for (float& f : v) f = r.uniform(lo, hi);
    return v;
}

template <class T>
std::vector<T> convert(const std::vector<float>& v);
template <>
std::vector<float> convert<float>(const std::vector<float>& v) { return v; }
template <>
std::vector<__half> convert<__half>(const std::vector<float>& v) {
    std::vector<__half> o(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) o[i] = __float2half_rn(v[i]);
    return o;
}

float half_to_float(__half h) { return __half2float(h); }

template <class InT>
int run(const std::vector<std::uint32_t>& lengths, std::uint64_t seed, bool with_state, bool half_state, int bench_reps) {
    DeltaNetGeometry g;
    g.key_heads = 16; g.value_heads = 16; g.key_head_dim = 128; g.value_head_dim = 128; g.conv_kernel = 4; g.rms_norm_eps = 1e-6;
    const std::size_t conv_dim = g.conv_dim(), value_dim = g.value_dim(), hv = g.value_heads;
    Rng rng{seed};
    // a model-like range of decays: some heads forget within a few tokens, some hold for thousands
    std::vector<float> conv_w = rnd(rng, conv_dim * g.conv_kernel, -0.6f, 0.6f), a_log(hv), dt_bias = rnd(rng, hv, -3.0f, 0.0f), norm_w = rnd(rng, 128, 0.5f, 1.5f);
    for (float& f : a_log) f = std::log(rng.uniform(0.005f, 2.0f));
    std::size_t total = 0;
    std::vector<RowSpan> spans;
    for (std::uint32_t l : lengths) { spans.push_back({static_cast<std::uint32_t>(total), l}); total += l; }
    const std::size_t R = lengths.size();
    // inputs sized like the engine's projections: scale of the real activations (the L2 norm makes q, k scale-free)
    std::vector<float> mixed = rnd(rng, total * conv_dim, -2.0f, 2.0f), z = rnd(rng, total * value_dim, -3.0f, 3.0f), b = rnd(rng, total * hv, -3.0f, 3.0f),
                       a = rnd(rng, total * hv, -2.0f, 2.0f);
    // states: slots [0, R) incoming, [R, 2R) scan out, [2R, 3R) chunked out
    const std::size_t slots = 3 * R;
    const std::size_t tail_floats = (g.conv_kernel - 1) * conv_dim, rec_floats = g.recurrent_floats();
    std::vector<float> tails(slots * tail_floats, 0.0f), rec(slots * rec_floats, 0.0f);
    if (with_state) {
        for (std::size_t s = 0; s < R; ++s) {
            for (std::size_t i = 0; i < tail_floats; ++i) tails[s * tail_floats + i] = rng.uniform(-2.0f, 2.0f);
            for (std::size_t i = 0; i < rec_floats; ++i) rec[s * rec_floats + i] = rng.uniform(-0.4f, 0.4f);
        }
    }
    std::vector<std::uint32_t> in_slot(R), out_scan(R), out_chunk(R), wave(R);
    for (std::size_t r = 0; r < R; ++r) { in_slot[r] = with_state ? static_cast<std::uint32_t>(r) : kNoSlot; out_scan[r] = static_cast<std::uint32_t>(R + r); out_chunk[r] = static_cast<std::uint32_t>(2 * R + r); wave[r] = static_cast<std::uint32_t>(r); }

    DeviceBuffer<float> d_conv_w(conv_w), d_a_log(a_log), d_dt(dt_bias), d_norm(norm_w);
    const std::vector<InT> mixed_c = convert<InT>(mixed), z_c = convert<InT>(z), b_c = convert<InT>(b), a_c = convert<InT>(a);
    DeviceBuffer<InT> d_mixed(mixed_c), d_z(z_c), d_b(b_c), d_a(a_c);
    DeviceBuffer<RowSpan> d_spans(spans);
    DeviceBuffer<std::uint32_t> d_in(in_slot), d_out_scan(out_scan), d_out_chunk(out_chunk), d_wave(wave);
    // state storage
    DeviceBuffer<float> d_tails_f(tails), d_rec_f(rec);
    DeviceBuffer<__half> d_tails_h, d_rec_h;
    DeltaNetStates states_scan, states_chunk;
    DeviceBuffer<float> d_tails_f2(tails), d_rec_f2(rec);
    DeviceBuffer<__half> d_tails_h2, d_rec_h2;
    if (half_state) {
        const std::vector<__half> th = convert<__half>(tails), rh = convert<__half>(rec);
        d_tails_h = DeviceBuffer<__half>(th); d_rec_h = DeviceBuffer<__half>(rh);
        d_tails_h2 = DeviceBuffer<__half>(th); d_rec_h2 = DeviceBuffer<__half>(rh);
        states_scan = DeltaNetStates{d_tails_h.data(), d_rec_h.data(), true, nullptr};
        states_chunk = DeltaNetStates{d_tails_h2.data(), d_rec_h2.data(), true, nullptr};
    } else {
        states_scan = DeltaNetStates{d_tails_f.data(), d_rec_f.data(), false, nullptr};
        states_chunk = DeltaNetStates{d_tails_f2.data(), d_rec_f2.data(), false, nullptr};
    }
    DeviceBuffer<__half> d_out_a(total * value_dim), d_out_b(total * value_dim);
    DeltaNetWeights w{d_conv_w.data(), d_a_log.data(), d_dt.data(), d_norm.data()};
    DeltaNetInputsT<InT> in{{d_mixed.data(), conv_dim}, {d_z.data(), value_dim}, {d_b.data(), hv}, {d_a.data(), hv}};

    const DeltaNetRows rows_scan{d_spans.data(), d_in.data(), d_out_scan.data(), d_wave.data(), static_cast<std::uint32_t>(R)};
    const DeltaNetRows rows_chunk{d_spans.data(), d_in.data(), d_out_chunk.data(), d_wave.data(), static_cast<std::uint32_t>(R)};
    const std::vector<std::uint32_t> prefix = deltanet_chunk_prefix(lengths);
    DeviceBuffer<std::uint32_t> d_prefix(prefix);
    DeltaNetChunkPlan plan{d_prefix.data(), prefix.back()};
    const std::size_t ws_bytes = deltanet_chunked_workspace_bytes(g, plan.total_chunks, total);
    DeviceBuffer<char> d_ws(ws_bytes);
    std::printf("rows %zu tokens %zu chunks %u workspace %.1f MiB, inputs %s, state %s\n", R, total, plan.total_chunks, static_cast<double>(ws_bytes) / 1048576.0,
                std::is_same_v<InT, float> ? "float" : "half", half_state ? "half" : "float");

    auto run_scan = [&] { launch_deltanet_wave<__half>(g, w, in, rows_scan, states_scan, d_out_a.data(), value_dim, nullptr); };
    auto run_chunk = [&] { launch_deltanet_chunked<InT>(g, w, in, rows_chunk, plan, states_chunk, d_ws.data(), ws_bytes, total, d_out_b.data(), value_dim, nullptr); };
    run_scan();
    run_chunk();
    check(cudaDeviceSynchronize(), "sync");

    const std::vector<__half> oa = d_out_a.download(), ob = d_out_b.download();
    std::vector<float> ra, rb;
    if (half_state) {
        const std::vector<__half> x = d_rec_h.download(), y = d_rec_h2.download();
        for (__half h : x) ra.push_back(half_to_float(h));
        for (__half h : y) rb.push_back(half_to_float(h));
    } else {
        ra = d_rec_f.download();
        rb = d_rec_f2.download();
    }
    bool bad = false;
    for (std::size_t r = 0; r < R; ++r) {
        double ss = 0, worst = 0, sum_abs = 0;
        for (std::size_t t = 0; t < lengths[r]; ++t) {
            for (std::size_t c = 0; c < value_dim; ++c) {
                const double x = half_to_float(oa[(spans[r].offset + t) * value_dim + c]), y = half_to_float(ob[(spans[r].offset + t) * value_dim + c]);
                ss += x * x;
                worst = std::max(worst, std::fabs(x - y));
                sum_abs += std::fabs(x - y);
            }
        }
        const double rms = std::sqrt(ss / static_cast<double>(lengths[r] * value_dim));
        double sss = 0, sworst = 0;
        for (std::size_t i = 0; i < rec_floats; ++i) {
            const double x = ra[(R + r) * rec_floats + i], y = rb[(2 * R + r) * rec_floats + i];
            sss += x * x;
            sworst = std::max(sworst, std::fabs(x - y));
        }
        const double srms = std::sqrt(sss / static_cast<double>(rec_floats));
        std::printf("  row %zu len %u: out max|diff|/rms %.3e (mean %.3e), state max|diff|/rms %.3e\n", r, lengths[r], worst / rms,
                    sum_abs / static_cast<double>(lengths[r] * value_dim) / rms, sworst / srms);
        if (!(worst / rms < 0.05) || !(sworst / srms < 0.05)) bad = true;
    }
    if (bench_reps > 0) {
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        auto time = [&](auto&& f) {
            float best = 1e30f;
            for (int i = 0; i < bench_reps; ++i) {
                cudaEventRecord(e0);
                f();
                cudaEventRecord(e1);
                cudaEventSynchronize(e1);
                float ms = 0;
                cudaEventElapsedTime(&ms, e0, e1);
                best = std::min(best, ms);
            }
            return best;
        };
        run_scan(); run_chunk();
        const float s = time(run_scan), c = time(run_chunk);
        std::printf("  time per layer (best of %d): scan %.3f ms, chunked %.3f ms, %.2fx\n", bench_reps, s, c, s / c);
    }
    return bad ? 1 : 0;
}

} // namespace

int main(int argc, char** argv) {
    std::vector<std::uint32_t> lengths = {1000, 64, 130, 4096, 7, 63, 65};
    std::uint64_t seed = 1;
    bool with_state = true, half_state = false, half_input = false;
    int bench = 0;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i], v = argv[i + 1];
        if (a == "--lengths") {
            lengths.clear();
            std::stringstream ss(v);
            std::string item;
            while (std::getline(ss, item, ',')) lengths.push_back(static_cast<std::uint32_t>(std::stoul(item)));
        } else if (a == "--seed") seed = std::stoull(v);
        else if (a == "--state") with_state = v == "on";
        else if (a == "--half-state") half_state = v == "on";
        else if (a == "--bench") bench = std::stoi(v);
        else if (a == "--input") half_input = v == "half";
        else { std::cerr << "unknown option " << a << "\n"; return 2; }
    }
    try {
        return half_input ? run<__half>(lengths, seed, with_state, half_state, bench) : run<float>(lengths, seed, with_state, half_state, bench);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 2;
    }
}
