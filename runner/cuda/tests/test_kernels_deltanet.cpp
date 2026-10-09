// Tests of the Gated DeltaNet kernels (src/kernels_deltanet.cu) against the CPU reference in sepal::ref. Needs a
// CUDA device. Built with the shared harness of runner/tests (test.hpp, test_main.cpp).
//
//   sepal_cuda_kernel_deltanet_tests              every suite
//   sepal_cuda_kernel_deltanet_tests <suite>      one of: stages, growth, geometry, continuation, waves,
//                                                 invariance, edge
//
// Lines starting with "[measure]" are the measured differences quoted in the report.
//
// The reference chain is `deltanet_mixer` without its four projections: the same sepal::ref functions in the same
// order, for T = float and for T = double. The inputs (mixed, z, b, a), the weights and the incoming states are
// float32 values given to all three computations.

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <cuda_fp16.h>

#include "kernels_deltanet.hpp"
#include "sepal/config.hpp"
#include "sepal/ref/deltanet.hpp"
#include "test.hpp"

namespace {

using sepal::cuda::check;
using sepal::cuda::DeltaNetGeometry;
using sepal::cuda::DeltaNetInputs;
using sepal::cuda::DeltaNetOptions;
using sepal::cuda::DeltaNetRows;
using sepal::cuda::DeltaNetStates;
using sepal::cuda::DeltaNetWeights;
using sepal::cuda::DeviceBuffer;
using sepal::cuda::kNoSlot;
using sepal::cuda::RowSpan;

// ---- data ----

class Rng {
public:
    explicit Rng(std::uint64_t seed) : state_(seed) {}
    std::uint64_t next() { // splitmix64
        std::uint64_t z = (state_ += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    float uniform(float lo, float hi) {
        const float u = static_cast<float>(next() >> 40) * (1.0f / 16777216.0f);
        return lo + (hi - lo) * u;
    }
    std::uint32_t below(std::uint32_t n) { return static_cast<std::uint32_t>(next() % n); }

private:
    std::uint64_t state_;
};

std::vector<float> random_floats(Rng& rng, std::size_t count, float lo, float hi) {
    std::vector<float> v(count);
    for (float& f : v) f = rng.uniform(lo, hi);
    return v;
}

// One layer's weights and sizes.
struct Problem {
    sepal::ModelConfig cfg;
    DeltaNetGeometry g;
    std::vector<float> conv_w, a_log, dt_bias, norm_w;
    std::size_t conv_dim = 0, value_dim = 0, hv = 0, dk = 0, dv = 0, taps = 0;
};

Problem make_problem(const sepal::ModelConfig& cfg, std::uint64_t seed) {
    Problem p;
    p.cfg = cfg;
    p.g.key_heads = static_cast<std::uint32_t>(cfg.dn_key_heads);
    p.g.value_heads = static_cast<std::uint32_t>(cfg.dn_value_heads);
    p.g.key_head_dim = static_cast<std::uint32_t>(cfg.dn_key_head_dim);
    p.g.value_head_dim = static_cast<std::uint32_t>(cfg.dn_value_head_dim);
    p.g.conv_kernel = static_cast<std::uint32_t>(cfg.conv_kernel);
    p.g.rms_norm_eps = cfg.rms_norm_eps;
    p.conv_dim = p.g.conv_dim();
    p.value_dim = p.g.value_dim();
    p.hv = p.g.value_heads;
    p.dk = p.g.key_head_dim;
    p.dv = p.g.value_head_dim;
    p.taps = p.g.conv_kernel;
    Rng rng(seed);
    p.conv_w = random_floats(rng, p.conv_dim * p.taps, -0.6f, 0.6f);
    // Decays from about 0.25 to 0.9997: some heads forget within a few tokens, some hold for thousands.
    p.a_log.resize(p.hv);
    for (float& f : p.a_log) f = std::log(rng.uniform(0.005f, 2.0f));
    p.dt_bias = random_floats(rng, p.hv, -3.0f, 0.0f);
    p.norm_w = random_floats(rng, p.dv, 0.5f, 1.5f);
    return p;
}

// One row's tokens and the state it starts from.
struct RowData {
    std::size_t len = 0;
    std::vector<float> mixed, z, b, a; // [len, conv_dim], [len, value_dim], [len, Hv], [len, Hv]
    std::vector<float> tail0;          // [K-1, conv_dim]
    std::vector<float> state0;         // [Hv, dk, dv]
};

RowData make_row(const Problem& p, std::size_t len, std::uint64_t seed, bool with_state = true) {
    Rng rng(seed);
    RowData r;
    r.len = len;
    r.mixed = random_floats(rng, len * p.conv_dim, -2.0f, 2.0f);
    r.z = random_floats(rng, len * p.value_dim, -3.0f, 3.0f);
    r.b = random_floats(rng, len * p.hv, -3.0f, 3.0f);
    r.a = random_floats(rng, len * p.hv, -2.0f, 2.0f);
    r.tail0 = with_state ? random_floats(rng, (p.taps - 1) * p.conv_dim, -2.0f, 2.0f)
                         : std::vector<float>((p.taps - 1) * p.conv_dim, 0.0f);
    r.state0 = with_state ? random_floats(rng, p.hv * p.dk * p.dv, -0.4f, 0.4f)
                          : std::vector<float>(p.hv * p.dk * p.dv, 0.0f);
    return r;
}

// A row made of tokens [begin, begin + count) of another row, starting from the given state.
RowData slice_row(const Problem& p, const RowData& whole, std::size_t begin, std::size_t count) {
    RowData r;
    r.len = count;
    r.mixed.assign(whole.mixed.begin() + static_cast<std::ptrdiff_t>(begin * p.conv_dim),
                   whole.mixed.begin() + static_cast<std::ptrdiff_t>((begin + count) * p.conv_dim));
    r.z.assign(whole.z.begin() + static_cast<std::ptrdiff_t>(begin * p.value_dim),
               whole.z.begin() + static_cast<std::ptrdiff_t>((begin + count) * p.value_dim));
    r.b.assign(whole.b.begin() + static_cast<std::ptrdiff_t>(begin * p.hv),
               whole.b.begin() + static_cast<std::ptrdiff_t>((begin + count) * p.hv));
    r.a.assign(whole.a.begin() + static_cast<std::ptrdiff_t>(begin * p.hv),
               whole.a.begin() + static_cast<std::ptrdiff_t>((begin + count) * p.hv));
    r.tail0 = whole.tail0;
    r.state0 = whole.state0;
    return r;
}

// ---- the reference chain ----

template <class T>
std::vector<T> convert(const std::vector<float>& v) {
    return std::vector<T>(v.begin(), v.end());
}

template <class T>
struct RefOut {
    std::vector<T> conv_act; // [len, conv_dim]: q and k normalised (per key head), v
    std::vector<T> reads;    // [len, value_dim]
    std::vector<T> out;      // [len, value_dim]
    std::vector<T> tail;     // [K-1, conv_dim]
    std::vector<T> state;    // [Hv, dk, dv]
};

template <class T>
RefOut<T> reference(const Problem& p, const RowData& r) {
    using namespace sepal::ref;
    const sepal::ModelConfig& cfg = p.cfg;
    const std::size_t len = r.len;
    const std::size_t key_dim = dim(cfg.dn_key_dim());
    const std::size_t value_dim = dim(cfg.dn_value_dim());

    const auto mixed = sepal::BasicTensor<T>::from_values({len, p.conv_dim}, convert<T>(r.mixed));
    const auto kernel = sepal::BasicTensor<T>::from_values({p.conv_dim, p.taps}, convert<T>(p.conv_w));
    const auto tail = sepal::BasicTensor<T>::from_values({p.taps - 1, p.conv_dim}, convert<T>(r.tail0));
    ConvResult<T> conv = causal_conv<T>(mixed.mat(), kernel.mat(), tail.mat());
    for (T& value : conv.outputs.span()) value = silu(value);
    SplitHeads<T> split = split_heads<T>(conv.outputs.mat(), cfg);
    l2_normalise(split.q.as_mat(dim(cfg.dn_key_head_dim)));
    l2_normalise(split.k.as_mat(dim(cfg.dn_key_head_dim)));

    RefOut<T> result;
    result.conv_act.resize(len * p.conv_dim);
    for (std::size_t t = 0; t < len; ++t) {
        T* row = result.conv_act.data() + t * p.conv_dim;
        std::copy(split.q.data() + t * key_dim, split.q.data() + (t + 1) * key_dim, row);
        std::copy(split.k.data() + t * key_dim, split.k.data() + (t + 1) * key_dim, row + key_dim);
        std::copy(split.v.data() + t * value_dim, split.v.data() + (t + 1) * value_dim, row + 2 * key_dim);
    }
    const sepal::BasicTensor<T> q = share_key_heads(split.q, cfg);
    const sepal::BasicTensor<T> k = share_key_heads(split.k, cfg);

    const auto a = sepal::BasicTensor<T>::from_values({len, p.hv}, convert<T>(r.a));
    auto beta = sepal::BasicTensor<T>::from_values({len, p.hv}, convert<T>(r.b));
    const std::vector<T> a_log = convert<T>(p.a_log), dt_bias = convert<T>(p.dt_bias), norm = convert<T>(p.norm_w);
    const sepal::BasicTensor<T> decay = decay_gate<T>(a.mat(), a_log, dt_bias);
    for (T& value : beta.span()) value = sigmoid(value);
    auto state = sepal::BasicTensor<T>::from_values({p.hv, p.dk, p.dv}, convert<T>(r.state0));
    ScanResult<T> scan = delta_rule_scan(q, k, split.v, decay, beta, std::move(state), delta_rule_output_scale(cfg));

    const auto z = sepal::BasicTensor<T>::from_values({len, p.value_dim}, convert<T>(r.z));
    const sepal::BasicTensor<T> normed =
        gated_rms_norm<T>(scan.reads.as_mat(dim(cfg.dn_value_head_dim)), z.as_mat(dim(cfg.dn_value_head_dim)), norm,
                          cfg.rms_norm_eps);
    result.reads.assign(scan.reads.span().begin(), scan.reads.span().end());
    result.out.assign(normed.span().begin(), normed.span().end());
    result.tail.assign(conv.tail.span().begin(), conv.tail.span().end());
    result.state.assign(scan.state.span().begin(), scan.state.span().end());
    return result;
}

// ---- comparison ----

struct Diff {
    double max_abs = 0.0;
    double rel_rms = 0.0; // max_abs / rms of the reference
    double max_rel = 0.0; // elementwise |got - ref| / |ref| over the elements with |ref| >= rms
    double rms = 0.0;
};

template <class T>
Diff compare(const std::vector<float>& got, const std::vector<T>& ref) {
    Diff d;
    if (got.size() != ref.size()) {
        d.max_abs = std::numeric_limits<double>::infinity();
        return d;
    }
    double sum_sq = 0.0;
    for (const T r : ref) sum_sq += static_cast<double>(r) * static_cast<double>(r);
    d.rms = ref.empty() ? 0.0 : std::sqrt(sum_sq / static_cast<double>(ref.size()));
    for (std::size_t i = 0; i < ref.size(); ++i) {
        const double g = static_cast<double>(got[i]);
        const double r = static_cast<double>(ref[i]);
        double diff = std::fabs(g - r);
        if (!std::isfinite(g)) diff = std::numeric_limits<double>::infinity();
        d.max_abs = std::max(d.max_abs, diff);
        if (std::fabs(r) >= d.rms && r != 0.0) d.max_rel = std::max(d.max_rel, diff / std::fabs(r));
    }
    d.rel_rms = d.rms > 0.0 ? d.max_abs / d.rms : d.max_abs;
    return d;
}

bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}

template <class T>
std::vector<float> to_float(const std::vector<T>& v) {
    return std::vector<float>(v.begin(), v.end());
}

// ---- the device side ----

constexpr std::uint32_t kPadBits = 0x7FC0DEADu; // a NaN pattern in every gap between bands

float pad_value() { return std::bit_cast<float>(kPadBits); }

struct Job {
    const RowData* row;
    std::uint32_t in_slot;
    std::uint32_t out_slot;
};

struct RunOptions {
    std::vector<std::vector<std::size_t>> waves; // job indices per launch; empty = one wave of all jobs in order
    int lanes = 0;
    bool half = false;
    bool round_in = false; // the four projections rounded to binary16 and back, still passed as float32
    bool half_in = false;  // the same values passed as binary16 (needs `half`: the output type); implies round_in
};

struct JobResult {
    std::vector<float> out, conv_act, reads;
};

class Rig {
public:
    /// `half_storage`: the slots are binary16 (DeltaNetStates::half_storage); set_slot rounds, get_slot widens.
    Rig(const Problem& p, std::size_t slots, bool half_storage = false)
        : half_storage_(half_storage),
          p_(p),
          conv_w_(p.conv_w),
          a_log_(p.a_log),
          dt_bias_(p.dt_bias),
          norm_w_(p.norm_w),
          tails_(half_storage ? 0 : slots * p.g.conv_tail_floats()),
          states_(half_storage ? 0 : slots * p.g.recurrent_floats()),
          tails_h_(half_storage ? slots * p.g.conv_tail_floats() : 0),
          states_h_(half_storage ? slots * p.g.recurrent_floats() : 0),
          saturations_(1) {
        tails_.zero();
        states_.zero();
        tails_h_.zero();
        states_h_.zero();
        saturations_.zero();
    }

    unsigned saturations() const {
        unsigned n = 0;
        check(cudaMemcpy(&n, saturations_.data(), sizeof n, cudaMemcpyDeviceToHost), "cudaMemcpy");
        return n;
    }

    /// The raw binary16 words of a slot (half storage only): conv tail, recurrent state.
    std::pair<std::vector<std::uint16_t>, std::vector<std::uint16_t>> get_slot_bits(std::uint32_t slot) {
        std::vector<std::uint16_t> tail(p_.g.conv_tail_floats()), state(p_.g.recurrent_floats());
        check(cudaMemcpy(tail.data(), tails_h_.data() + slot * p_.g.conv_tail_floats(), tail.size() * 2, cudaMemcpyDeviceToHost), "cudaMemcpy");
        check(cudaMemcpy(state.data(), states_h_.data() + slot * p_.g.recurrent_floats(), state.size() * 2, cudaMemcpyDeviceToHost), "cudaMemcpy");
        return {std::move(tail), std::move(state)};
    }

    void set_slot(std::uint32_t slot, const std::vector<float>& tail, const std::vector<float>& state) {
        if (half_storage_) {
            std::vector<__half> t(tail.size()), s(state.size());
            for (std::size_t i = 0; i < tail.size(); ++i) t[i] = __float2half_rn(tail[i]);
            for (std::size_t i = 0; i < state.size(); ++i) s[i] = __float2half_rn(state[i]);
            check(cudaMemcpy(tails_h_.data() + slot * p_.g.conv_tail_floats(), t.data(), t.size() * 2, cudaMemcpyHostToDevice), "cudaMemcpy");
            check(cudaMemcpy(states_h_.data() + slot * p_.g.recurrent_floats(), s.data(), s.size() * 2, cudaMemcpyHostToDevice), "cudaMemcpy");
            return;
        }
        check(cudaMemcpy(tails_.data() + slot * p_.g.conv_tail_floats(), tail.data(), tail.size() * sizeof(float),
                         cudaMemcpyHostToDevice),
              "cudaMemcpy");
        check(cudaMemcpy(states_.data() + slot * p_.g.recurrent_floats(), state.data(), state.size() * sizeof(float),
                         cudaMemcpyHostToDevice),
              "cudaMemcpy");
    }

    std::pair<std::vector<float>, std::vector<float>> get_slot(std::uint32_t slot) {
        std::vector<float> tail(p_.g.conv_tail_floats()), state(p_.g.recurrent_floats());
        if (half_storage_) {
            const auto [tb, sb] = get_slot_bits(slot);
            for (std::size_t i = 0; i < tail.size(); ++i) tail[i] = __half2float(std::bit_cast<__half>(tb[i]));
            for (std::size_t i = 0; i < state.size(); ++i) state[i] = __half2float(std::bit_cast<__half>(sb[i]));
            return {std::move(tail), std::move(state)};
        }
        check(cudaMemcpy(tail.data(), tails_.data() + slot * p_.g.conv_tail_floats(), tail.size() * sizeof(float),
                         cudaMemcpyDeviceToHost),
              "cudaMemcpy");
        check(cudaMemcpy(state.data(), states_.data() + slot * p_.g.recurrent_floats(), state.size() * sizeof(float),
                         cudaMemcpyDeviceToHost),
              "cudaMemcpy");
        return {std::move(tail), std::move(state)};
    }

    /// True if the last run left every gap between bands as it found it.
    bool pads_intact() const { return pads_intact_; }

    std::vector<JobResult> run(const std::vector<Job>& jobs, const RunOptions& options = {}) {
        const std::size_t conv_dim = p_.conv_dim, value_dim = p_.value_dim, hv = p_.hv;
        std::size_t total = 0;
        std::vector<RowSpan> spans;
        for (const Job& job : jobs) {
            spans.push_back({static_cast<std::uint32_t>(total), static_cast<std::uint32_t>(job.row->len)});
            total += job.row->len;
        }
        // The four projections as bands of one wider matrix, with a NaN gap between bands.
        const std::size_t z_off = conv_dim + 1;
        const std::size_t b_off = z_off + value_dim + 2;
        const std::size_t a_off = b_off + hv + 1;
        const std::size_t stride = a_off + hv + 3;
        std::vector<float> proj(std::max<std::size_t>(total, 1) * stride, pad_value());
        for (std::size_t j = 0; j < jobs.size(); ++j) {
            const RowData& r = *jobs[j].row;
            for (std::size_t t = 0; t < r.len; ++t) {
                float* row = proj.data() + (spans[j].offset + t) * stride;
                std::copy(r.mixed.begin() + static_cast<std::ptrdiff_t>(t * conv_dim),
                          r.mixed.begin() + static_cast<std::ptrdiff_t>((t + 1) * conv_dim), row);
                std::copy(r.z.begin() + static_cast<std::ptrdiff_t>(t * value_dim),
                          r.z.begin() + static_cast<std::ptrdiff_t>((t + 1) * value_dim), row + z_off);
                std::copy(r.b.begin() + static_cast<std::ptrdiff_t>(t * hv),
                          r.b.begin() + static_cast<std::ptrdiff_t>((t + 1) * hv), row + b_off);
                std::copy(r.a.begin() + static_cast<std::ptrdiff_t>(t * hv),
                          r.a.begin() + static_cast<std::ptrdiff_t>((t + 1) * hv), row + a_off);
            }
        }
        if (options.round_in || options.half_in) {
            for (float& v : proj) v = __half2float(__float2half_rn(v));
        }
        const DeviceBuffer<float> d_proj(options.half_in ? std::vector<float>(1) : proj);
        DeviceBuffer<__half> d_proj_h;
        if (options.half_in) {
            std::vector<__half> bits(proj.size());
            for (std::size_t i = 0; i < proj.size(); ++i) bits[i] = __float2half_rn(proj[i]);
            d_proj_h = DeviceBuffer<__half>(bits);
        }

        std::vector<std::uint32_t> in_slots, out_slots;
        for (const Job& job : jobs) {
            in_slots.push_back(job.in_slot);
            out_slots.push_back(job.out_slot);
        }
        // The kernels never read through a null pointer, but a zero-token pass still needs valid buffers.
        const DeviceBuffer<RowSpan> d_rows(spans.empty() ? std::vector<RowSpan>(1) : spans);
        const DeviceBuffer<std::uint32_t> d_in(in_slots.empty() ? std::vector<std::uint32_t>(1) : in_slots);
        const DeviceBuffer<std::uint32_t> d_out(out_slots.empty() ? std::vector<std::uint32_t>(1) : out_slots);

        const std::size_t out_stride = value_dim + 5, conv_tap_stride = conv_dim + 2, reads_tap_stride = value_dim + 1;
        const std::size_t out_count = std::max<std::size_t>(total, 1) * out_stride;
        DeviceBuffer<float> d_out_f;
        DeviceBuffer<__half> d_out_h;
        if (options.half) {
            d_out_h = DeviceBuffer<__half>(std::vector<__half>(out_count, std::bit_cast<__half>(std::uint16_t{0x7E01})));
        } else {
            d_out_f = DeviceBuffer<float>(std::vector<float>(out_count, pad_value()));
        }
        const DeviceBuffer<float> d_conv_tap(std::vector<float>(std::max<std::size_t>(total, 1) * conv_tap_stride, pad_value()));
        const DeviceBuffer<float> d_reads_tap(std::vector<float>(std::max<std::size_t>(total, 1) * reads_tap_stride, pad_value()));

        std::vector<std::vector<std::size_t>> waves = options.waves;
        if (waves.empty()) {
            waves.emplace_back();
            for (std::size_t j = 0; j < jobs.size(); ++j) waves.back().push_back(j);
        }

        DeltaNetWeights weights{conv_w_.data(), a_log_.data(), dt_bias_.data(), norm_w_.data()};
        DeltaNetInputs inputs{{d_proj.data(), stride},
                              {d_proj.data() + z_off, stride},
                              {d_proj.data() + b_off, stride},
                              {d_proj.data() + a_off, stride}};
        sepal::cuda::DeltaNetInputsHalf inputs_h{{d_proj_h.data(), stride},
                                                 {d_proj_h.data() + z_off, stride},
                                                 {d_proj_h.data() + b_off, stride},
                                                 {d_proj_h.data() + a_off, stride}};
        DeltaNetStates states{tails_.data(), states_.data()};
        if (half_storage_) states = DeltaNetStates{tails_h_.data(), states_h_.data(), true, saturations_.data()};
        DeltaNetOptions launch;
        launch.lanes_per_column = options.lanes;
        launch.taps.conv_act = d_conv_tap.data();
        launch.taps.conv_act_stride = conv_tap_stride;
        launch.taps.reads = d_reads_tap.data();
        launch.taps.reads_stride = reads_tap_stride;

        std::vector<DeviceBuffer<std::uint32_t>> wave_buffers;
        for (const auto& wave : waves) {
            std::vector<std::uint32_t> ids;
            for (const std::size_t j : wave) ids.push_back(static_cast<std::uint32_t>(j));
            wave_buffers.emplace_back(ids.empty() ? std::vector<std::uint32_t>(1) : ids);
            DeltaNetRows rows{d_rows.data(), d_in.data(), d_out.data(), wave_buffers.back().data(),
                              static_cast<std::uint32_t>(wave.size())};
            if (options.half_in) {
                sepal::cuda::launch_deltanet_wave(p_.g, weights, inputs_h, rows, states, d_out_h.data(), out_stride,
                                                  nullptr, launch);
            } else if (options.half) {
                sepal::cuda::launch_deltanet_wave(p_.g, weights, inputs, rows, states, d_out_h.data(), out_stride,
                                                  nullptr, launch);
            } else {
                sepal::cuda::launch_deltanet_wave(p_.g, weights, inputs, rows, states, d_out_f.data(), out_stride,
                                                  nullptr, launch);
            }
        }
        check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

        // Download and cut the bands out of the strided buffers.
        std::vector<float> out_all;
        std::vector<std::uint16_t> out_bits; // the raw binary16 words, for the gap check
        if (options.half) {
            const std::vector<__half> raw = d_out_h.download();
            out_all.resize(raw.size());
            out_bits.resize(raw.size());
            for (std::size_t i = 0; i < raw.size(); ++i) {
                out_all[i] = __half2float(raw[i]);
                out_bits[i] = std::bit_cast<std::uint16_t>(raw[i]);
            }
        } else {
            out_all = d_out_f.download();
        }
        const std::vector<float> conv_all = d_conv_tap.download(), reads_all = d_reads_tap.download();
        pads_intact_ = true;
        const std::uint16_t half_pad = 0x7E01;
        std::vector<JobResult> results(jobs.size());
        for (std::size_t j = 0; j < jobs.size(); ++j) {
            for (std::size_t t = 0; t < jobs[j].row->len; ++t) {
                const std::size_t token = spans[j].offset + t;
                results[j].out.insert(results[j].out.end(), out_all.begin() + static_cast<std::ptrdiff_t>(token * out_stride),
                                      out_all.begin() + static_cast<std::ptrdiff_t>(token * out_stride + value_dim));
                results[j].conv_act.insert(results[j].conv_act.end(),
                                           conv_all.begin() + static_cast<std::ptrdiff_t>(token * conv_tap_stride),
                                           conv_all.begin() + static_cast<std::ptrdiff_t>(token * conv_tap_stride + conv_dim));
                results[j].reads.insert(results[j].reads.end(),
                                        reads_all.begin() + static_cast<std::ptrdiff_t>(token * reads_tap_stride),
                                        reads_all.begin() + static_cast<std::ptrdiff_t>(token * reads_tap_stride + value_dim));
            }
        }
        // The gap columns of every output must still hold the pad pattern.
        for (std::size_t token = 0; token < total; ++token) {
            for (std::size_t c = value_dim; c < out_stride; ++c) {
                if (options.half) {
                    if (out_bits[token * out_stride + c] != half_pad) pads_intact_ = false;
                } else if (std::bit_cast<std::uint32_t>(out_all[token * out_stride + c]) != kPadBits) {
                    pads_intact_ = false;
                }
            }
            for (std::size_t c = conv_dim; c < conv_tap_stride; ++c) {
                if (std::bit_cast<std::uint32_t>(conv_all[token * conv_tap_stride + c]) != kPadBits) pads_intact_ = false;
            }
            for (std::size_t c = value_dim; c < reads_tap_stride; ++c) {
                if (std::bit_cast<std::uint32_t>(reads_all[token * reads_tap_stride + c]) != kPadBits) pads_intact_ = false;
            }
        }
        return results;
    }

private:
    bool half_storage_ = false;
    const Problem& p_;
    DeviceBuffer<float> conv_w_, a_log_, dt_bias_, norm_w_;
    DeviceBuffer<float> tails_, states_;
    DeviceBuffer<__half> tails_h_, states_h_;
    DeviceBuffer<unsigned> saturations_;
    bool pads_intact_ = true;
};

// Runs one row from zero-slot 0 (loaded with its initial state) to slot 1.
struct SingleRun {
    JobResult result;
    std::vector<float> tail, state;
};

SingleRun run_single(const Problem& p, const RowData& row, const RunOptions& options = {}) {
    Rig rig(p, 2);
    rig.set_slot(0, row.tail0, row.state0);
    SingleRun run;
    run.result = rig.run({{&row, 0, 1}}, options)[0];
    auto [tail, state] = rig.get_slot(1);
    run.tail = std::move(tail);
    run.state = std::move(state);
    CHECK(rig.pads_intact());
    return run;
}

// ---- comparing a device run with the reference ----

struct StageDiffs {
    Diff conv_act, reads, out, tail, state;
};

template <class T>
StageDiffs compare_run(const SingleRun& run, const RefOut<T>& ref) {
    return {compare(run.result.conv_act, ref.conv_act), compare(run.result.reads, ref.reads),
            compare(run.result.out, ref.out), compare(run.tail, ref.tail), compare(run.state, ref.state)};
}

void print_stage_diffs(const char* label, const StageDiffs& d) {
    std::printf("[measure] %-34s conv_act %.2e/%.2e  reads %.2e/%.2e  out %.2e/%.2e  tail %.2e  state %.2e/%.2e"
                "   (max_abs / max_rel)\n",
                label, d.conv_act.max_abs, d.conv_act.max_rel, d.reads.max_abs, d.reads.max_rel, d.out.max_abs,
                d.out.max_rel, d.tail.max_abs, d.state.max_abs, d.state.max_rel);
}

// Loose bounds, far above what is measured, for parity of one row with the float32 reference.
void check_parity(const StageDiffs& d, double tol_rel_rms) {
    CHECK(d.conv_act.rel_rms <= tol_rel_rms);
    CHECK(d.reads.rel_rms <= tol_rel_rms);
    CHECK(d.out.rel_rms <= tol_rel_rms);
    CHECK(d.state.rel_rms <= tol_rel_rms);
    CHECK_EQ(d.tail.max_abs, 0.0); // the tail is a copy of inputs: exact
}

sepal::ModelConfig preset(const char* name) {
    const auto found = sepal::find_preset(name);
    CHECK(found.has_value());
    return *found;
}

struct Named {
    const char* name;
    sepal::ModelConfig cfg;
};

} // namespace

// ---- stages: every stage and the whole chain, strides, half output ----

SEPAL_TEST(stages, chain_against_reference) {
    const std::vector<Named> configs = {{"tiny", sepal::tiny_test_config()}, {"0.8b", preset("0.8b")}};
    for (const Named& named : configs) {
        const Problem p = make_problem(named.cfg, 11);
        const std::size_t len = named.cfg.name == "tiny" ? 75 : 200;
        const RowData row = make_row(p, len, 21);
        const RefOut<float> ref_f = reference<float>(p, row);
        const RefOut<double> ref_d = reference<double>(p, row);
        const SingleRun run = run_single(p, row);
        const StageDiffs vs_f = compare_run(run, ref_f);
        const StageDiffs vs_d = compare_run(run, ref_d);
        print_stage_diffs((std::string(named.name) + " device vs float ref").c_str(), vs_f);
        print_stage_diffs((std::string(named.name) + " device vs double ref").c_str(), vs_d);
        check_parity(vs_f, 1e-4);
        check_parity(vs_d, 1e-4);
        // The half output is the float output rounded to binary16.
        RunOptions half;
        half.half = true;
        const SingleRun run_h = run_single(p, row, half);
        bool rounded = run_h.result.out.size() == run.result.out.size();
        for (std::size_t i = 0; rounded && i < run.result.out.size(); ++i) {
            rounded = std::bit_cast<std::uint16_t>(__float2half_rn(run.result.out[i])) ==
                      std::bit_cast<std::uint16_t>(__float2half_rn(run_h.result.out[i])) &&
                      __half2float(__float2half_rn(run.result.out[i])) == run_h.result.out[i];
        }
        CHECK(rounded);
        CHECK(same_bits(run_h.state, run.state));
    }
}

// Projections left in binary16 by the multiply (the engine's cuBLAS binary16 accumulation) are widened as the scan and the conv
// tail read them: the result is bit for bit that of the same values passed as float32.
SEPAL_TEST(stages, binary16_inputs_give_the_bits_of_the_same_values_as_float32) {
    const std::vector<Named> configs = {{"tiny", sepal::tiny_test_config()}, {"0.8b", preset("0.8b")}};
    for (const Named& named : configs) {
        const Problem p = make_problem(named.cfg, 12);
        for (const std::size_t len : {std::size_t{1}, std::size_t{17}, std::size_t{75}, std::size_t{200}}) {
            const RowData row = make_row(p, len, 22);
            RunOptions as_float;
            as_float.half = true;
            as_float.round_in = true;
            RunOptions as_half = as_float;
            as_half.half_in = true;
            const SingleRun a = run_single(p, row, as_float);
            const SingleRun b = run_single(p, row, as_half);
            CHECK(same_bits(a.result.out, b.result.out));
            CHECK(same_bits(a.result.conv_act, b.result.conv_act));
            CHECK(same_bits(a.result.reads, b.result.reads));
            CHECK(same_bits(a.tail, b.tail));
            CHECK(same_bits(a.state, b.state));
            // and the rounding of the inputs is not a no-op, so the test can tell the two apart from the unrounded run
            RunOptions plain;
            plain.half = true;
            CHECK(!same_bits(run_single(p, row, plain).result.out, a.result.out) || len == 1);
        }
    }
}

// The stage-level view of the conv + SiLU + L2 stage alone, from the v channels (conv + SiLU only) and the q, k
// channels (the normalised vectors have unit length up to the epsilon).
SEPAL_TEST(stages, normalised_q_and_k) {
    const Problem p = make_problem(preset("0.8b"), 12);
    const RowData row = make_row(p, 40, 22);
    const SingleRun run = run_single(p, row);
    const std::size_t key_dim = p.g.key_dim();
    double worst = 0.0;
    for (std::size_t t = 0; t < row.len; ++t) {
        for (std::size_t h = 0; h < p.g.key_heads * 2; ++h) {
            double sum_sq = 0.0;
            for (std::size_t i = 0; i < p.dk; ++i) {
                const double x = run.result.conv_act[t * p.conv_dim + h * p.dk + i];
                sum_sq += x * x;
            }
            worst = std::max(worst, std::fabs(sum_sq - 1.0));
        }
    }
    (void)key_dim;
    std::printf("[measure] max | |q|^2 - 1 | over q and k vectors: %.3e\n", worst);
    CHECK(worst < 1e-5);
}

// ---- growth: the scan compounds rounding; measure at 1 .. 4096 tokens ----

SEPAL_TEST(growth, error_by_token_count) {
    const Problem p = make_problem(preset("0.8b"), 31);
    std::printf("[measure] 0.8B geometry, one row, nonzero start; rel = max_abs / rms of the reference\n");
    std::printf("[measure]  tokens | device vs f32 ref: out abs/rel, state abs/rel | device vs f64 ref: out abs/rel, "
                "state abs/rel | f32 ref vs f64 ref: out abs/rel, state abs/rel\n");
    for (const std::size_t tokens : {1u, 16u, 256u, 1024u, 4096u}) {
        const RowData row = make_row(p, tokens, 40 + tokens);
        const RefOut<float> ref_f = reference<float>(p, row);
        const RefOut<double> ref_d = reference<double>(p, row);
        const SingleRun run = run_single(p, row);
        const StageDiffs vs_f = compare_run(run, ref_f);
        const StageDiffs vs_d = compare_run(run, ref_d);
        const Diff ref_out = compare(to_float(ref_f.out), ref_d.out);
        const Diff ref_state = compare(to_float(ref_f.state), ref_d.state);
        const Diff ref_reads = compare(to_float(ref_f.reads), ref_d.reads);
        std::printf("[measure] %6zu | %.2e/%.2e  %.2e/%.2e | %.2e/%.2e  %.2e/%.2e | %.2e/%.2e  %.2e/%.2e"
                    "   (reads: dev-f32 %.2e, dev-f64 %.2e, f32ref-f64 %.2e)\n",
                    tokens, vs_f.out.max_abs, vs_f.out.rel_rms, vs_f.state.max_abs, vs_f.state.rel_rms,
                    vs_d.out.max_abs, vs_d.out.rel_rms, vs_d.state.max_abs, vs_d.state.rel_rms, ref_out.max_abs,
                    ref_out.rel_rms, ref_state.max_abs, ref_state.rel_rms, vs_f.reads.rel_rms, vs_d.reads.rel_rms,
                    ref_reads.rel_rms);
        if (tokens == 1024 || tokens == 4096) {
            CHECK(vs_f.out.rel_rms < 1e-4);
            CHECK(vs_f.reads.rel_rms < 1e-4);
            CHECK(vs_f.state.rel_rms < 1e-4);
            CHECK(vs_f.out.max_rel < 1e-4);
        }
    }
}

// ---- geometry: tiny (axes that coincide in real models differ), 0.8B/2B and 4B sizes ----

SEPAL_TEST(geometry, tiny_many_lengths) {
    const Problem p = make_problem(sepal::tiny_test_config(), 51);
    // dk = 6, dv = 5, 2 key heads, 4 value heads, conv_dim = 2*12 + 20 = 44.
    CHECK_EQ(p.conv_dim, 44u);
    for (const std::size_t len : {1u, 2u, 3u, 4u, 5u, 15u, 16u, 17u, 31u, 33u, 100u}) {
        const RowData row = make_row(p, len, 60 + len);
        const SingleRun run = run_single(p, row);
        const StageDiffs vs_f = compare_run(run, reference<float>(p, row));
        const StageDiffs vs_d = compare_run(run, reference<double>(p, row));
        print_stage_diffs(("tiny len " + std::to_string(len) + " vs f32").c_str(), vs_f);
        check_parity(vs_f, 1e-4);
        check_parity(vs_d, 1e-4);
    }
}

SEPAL_TEST(geometry, real_sizes) {
    const Named configs[] = {{"0.8b", preset("0.8b")}, {"2b", preset("2b")}, {"4b", preset("4b")}};
    for (const Named& named : configs) {
        const Problem p = make_problem(named.cfg, 71);
        // Three rows of different lengths in one wave, to check the geometry with rows side by side.
        const RowData rows[3] = {make_row(p, 150, 81), make_row(p, 37, 82), make_row(p, 1, 83)};
        Rig rig(p, 6);
        for (std::uint32_t i = 0; i < 3; ++i) rig.set_slot(i, rows[i].tail0, rows[i].state0);
        const std::vector<JobResult> results = rig.run({{&rows[0], 0, 3}, {&rows[1], 1, 4}, {&rows[2], 2, 5}});
        CHECK(rig.pads_intact());
        for (std::uint32_t i = 0; i < 3; ++i) {
            SingleRun run;
            run.result = results[i];
            auto [tail, state] = rig.get_slot(3 + i);
            run.tail = std::move(tail);
            run.state = std::move(state);
            const StageDiffs vs_f = compare_run(run, reference<float>(p, rows[i]));
            print_stage_diffs((std::string(named.name) + " row " + std::to_string(i) + " len " +
                               std::to_string(rows[i].len) + " vs f32")
                                  .c_str(),
                              vs_f);
            check_parity(vs_f, 1e-4);
        }
    }
}

// ---- continuation: [0, a) then [a, T) from the outgoing slot equals [0, T), bit for bit ----

namespace {

void continuation_for(const Problem& p, std::size_t total, const std::vector<std::size_t>& splits) {
    const RowData whole = make_row(p, total, 91 + total);
    const SingleRun reference_run = run_single(p, whole);
    for (const std::size_t a : splits) {
        const RowData first = slice_row(p, whole, 0, a);
        const RowData second = slice_row(p, whole, a, total - a);
        Rig rig(p, 3);
        rig.set_slot(0, whole.tail0, whole.state0);
        // Slot 0 -> slot 1 (first part), slot 1 -> slot 2 (second part), as two waves of one pass.
        RunOptions options;
        options.waves = {{0}, {1}};
        const std::vector<JobResult> results = rig.run({{&first, 0, 1}, {&second, 1, 2}}, options);
        CHECK(rig.pads_intact());
        std::vector<float> out = results[0].out, conv_act = results[0].conv_act, reads = results[0].reads;
        out.insert(out.end(), results[1].out.begin(), results[1].out.end());
        conv_act.insert(conv_act.end(), results[1].conv_act.begin(), results[1].conv_act.end());
        reads.insert(reads.end(), results[1].reads.begin(), results[1].reads.end());
        const auto [tail, state] = rig.get_slot(2);
        const auto [mid_tail, mid_state] = rig.get_slot(1);
        // The convolution, the normalisation and the conv tail do not depend on where a row is cut: bit-identical.
        // The recurrence keeps its state scaled by an outstanding decay factor that restarts at the cut, so out, reads
        // and the state agree with the uncut row to rounding only (the cut row is bit-identical to ITSELF in every
        // batch, which the invariance and waves suites check).
        const bool exact = same_bits(conv_act, reference_run.result.conv_act) && same_bits(tail, reference_run.tail);
        const Diff d_out = compare(out, reference_run.result.out);
        const Diff d_reads = compare(reads, reference_run.result.reads);
        const Diff d_state = compare(state, reference_run.state);
        std::printf("[measure] continuation %zu + %zu of %s: vs the uncut row out %.2e reads %.2e state %.2e (max_abs / rms)\n", a,
                    total - a, p.cfg.name.c_str(), d_out.rel_rms, d_reads.rel_rms, d_state.rel_rms);
        CHECK(exact);
        CHECK(d_out.rel_rms < 2e-5);
        CHECK(d_reads.rel_rms < 2e-5);
        CHECK(d_state.rel_rms < 2e-5);
        // The middle slot holds exactly what a row of the first part alone leaves.
        const SingleRun first_alone = run_single(p, first);
        CHECK(same_bits(mid_tail, first_alone.tail));
        CHECK(same_bits(mid_state, first_alone.state));
    }
}

} // namespace

SEPAL_TEST(continuation, tiny) {
    const Problem p = make_problem(sepal::tiny_test_config(), 101);
    continuation_for(p, 70, {1, 2, 3, 4, 5, 15, 16, 17, 32, 35, 69});
    std::printf("[measure] continuation, tiny, 11 split points of 70 tokens: within rounding\n");
}

SEPAL_TEST(continuation, real_0_8b) {
    const Problem p = make_problem(preset("0.8b"), 102);
    continuation_for(p, 90, {1, 2, 3, 4, 17, 45, 89});
    std::printf("[measure] continuation, 0.8B, 7 split points of 90 tokens: within rounding\n");
}

SEPAL_TEST(continuation, real_4b) {
    const Problem p = make_problem(preset("4b"), 103);
    continuation_for(p, 40, {1, 3, 16, 25});
    std::printf("[measure] continuation, 4B, 4 split points of 40 tokens: within rounding\n");
}

// ---- waves: a state row and eight branches that continue it, in one pass ----

namespace {

void waves_for(const Problem& p, std::size_t state_len) {
    const RowData state_row = make_row(p, state_len, 111);
    std::vector<RowData> branches;
    for (std::size_t i = 0; i < 8; ++i) {
        // Branches continue the state row's state, so their own starting state is not used.
        RowData b = make_row(p, 7 + i * 3, 120 + i);
        branches.push_back(std::move(b));
    }
    // One pass: row 0 = the state row, rows 1..8 = branches that read its outgoing slot (second wave).
    Rig pass(p, 10);
    pass.set_slot(0, state_row.tail0, state_row.state0);
    std::vector<Job> jobs = {{&state_row, 0, 1}};
    for (std::uint32_t i = 0; i < 8; ++i) jobs.push_back({&branches[i], 1, 2 + i});
    RunOptions options;
    options.waves = {{0}, {1, 2, 3, 4, 5, 6, 7, 8}};
    const std::vector<JobResult> together = pass.run(jobs, options);
    CHECK(pass.pads_intact());

    // The same branches in a launch of their own, from a state saved from the first run of the state row.
    Rig saved(p, 10);
    saved.set_slot(1, pass.get_slot(1).first, pass.get_slot(1).second);
    std::vector<Job> branch_jobs;
    for (std::uint32_t i = 0; i < 8; ++i) branch_jobs.push_back({&branches[i], 1, 2 + i});
    const std::vector<JobResult> separate = saved.run(branch_jobs);

    bool ok = true;
    for (std::size_t i = 0; i < 8; ++i) {
        ok = ok && same_bits(together[1 + i].out, separate[i].out) &&
             same_bits(together[1 + i].conv_act, separate[i].conv_act) && same_bits(together[1 + i].reads, separate[i].reads);
        const auto a = pass.get_slot(static_cast<std::uint32_t>(2 + i));
        const auto b = saved.get_slot(static_cast<std::uint32_t>(2 + i));
        ok = ok && same_bits(a.first, b.first) && same_bits(a.second, b.second);
    }
    CHECK(ok);
    // The shared parent slot was only read by the branches.
    CHECK(same_bits(pass.get_slot(1).second, saved.get_slot(1).second));

    // And each branch agrees with the float32 reference run from the parent's state.
    const auto [parent_tail, parent_state] = pass.get_slot(1);
    for (std::size_t i = 0; i < 8; ++i) {
        RowData b = branches[i];
        b.tail0 = parent_tail;
        b.state0 = parent_state;
        SingleRun run;
        run.result = together[1 + i];
        auto [tail, state] = pass.get_slot(static_cast<std::uint32_t>(2 + i));
        run.tail = std::move(tail);
        run.state = std::move(state);
        const StageDiffs vs_f = compare_run(run, reference<float>(p, b));
        check_parity(vs_f, 1e-4);
    }
}

} // namespace

SEPAL_TEST(waves, tiny) {
    waves_for(make_problem(sepal::tiny_test_config(), 131), 37);
    std::printf("[measure] waves, tiny: a state row and 8 branches in one pass equal the branches alone, bit for bit\n");
}

SEPAL_TEST(waves, real_0_8b) {
    waves_for(make_problem(preset("0.8b"), 132), 150);
    std::printf("[measure] waves, 0.8B: a state row and 8 branches in one pass equal the branches alone, bit for bit\n");
}

// ---- batch invariance: a row's bits do not depend on the other rows, their lengths, the position, the wave ----

namespace {

void invariance_for(const Problem& p, std::size_t row_len, std::uint64_t seed) {
    const RowData target = make_row(p, row_len, seed);
    const SingleRun alone = run_single(p, target);
    const RunOptions one_lane = [] {
        RunOptions o;
        o.lanes = 1;
        return o;
    }();
    Rng rng(seed + 7);
    for (int trial = 0; trial < 4; ++trial) {
        // Random other rows of random lengths, the target at a random place.
        const std::size_t others = 5 + rng.below(10);
        std::vector<RowData> rows;
        for (std::size_t i = 0; i < others; ++i) rows.push_back(make_row(p, 1 + rng.below(90), seed * 31 + trial * 100 + i));
        const std::size_t at = trial == 0 ? 0 : trial == 1 ? others : rng.below(static_cast<std::uint32_t>(others + 1));
        rows.insert(rows.begin() + static_cast<std::ptrdiff_t>(at), target);

        Rig rig(p, 2 * rows.size());
        std::vector<Job> jobs;
        for (std::uint32_t i = 0; i < rows.size(); ++i) {
            rig.set_slot(2 * i, rows[i].tail0, rows[i].state0);
            jobs.push_back({&rows[i], 2 * i, 2 * i + 1});
        }
        // Run with the default (size-dependent) lane count, and again with each fixed one.
        for (const int lanes : {0, 1, 2, 4, 8}) {
            RunOptions options;
            options.lanes = lanes;
            const std::vector<JobResult> results = rig.run(jobs, options);
            CHECK(rig.pads_intact());
            const auto [tail, state] = rig.get_slot(static_cast<std::uint32_t>(2 * at + 1));
            const bool ok = same_bits(results[at].out, alone.result.out) &&
                            same_bits(results[at].conv_act, alone.result.conv_act) &&
                            same_bits(results[at].reads, alone.result.reads) && same_bits(tail, alone.tail) &&
                            same_bits(state, alone.state);
            if (!ok) {
                std::printf("[measure] invariance FAILED: len %zu trial %d at %zu lanes %d\n", row_len, trial, at, lanes);
            }
            CHECK(ok);
        }
    }
    // The target alone with every fixed lane count.
    for (const int lanes : {1, 2, 4, 8}) {
        RunOptions options;
        options.lanes = lanes;
        const SingleRun run = run_single(p, target, options);
        CHECK(same_bits(run.result.out, alone.result.out));
        CHECK(same_bits(run.state, alone.state));
    }
    (void)one_lane;
}

} // namespace

SEPAL_TEST(invariance, tiny) {
    const Problem p = make_problem(sepal::tiny_test_config(), 141);
    for (const std::size_t len : {1u, 2u, 3u, 17u, 64u}) invariance_for(p, len, 150 + len);
    std::printf("[measure] invariance, tiny: 5 lengths x 4 batches x 5 thread mappings: bit-identical\n");
}

SEPAL_TEST(invariance, real_0_8b) {
    const Problem p = make_problem(preset("0.8b"), 142);
    for (const std::size_t len : {1u, 3u, 40u}) invariance_for(p, len, 160 + len);
    std::printf("[measure] invariance, 0.8B: 3 lengths x 4 batches x 5 thread mappings: bit-identical\n");
}

SEPAL_TEST(invariance, real_4b) {
    const Problem p = make_problem(preset("4b"), 143);
    invariance_for(p, 33, 170);
    std::printf("[measure] invariance, 4B: 1 length x 4 batches x 5 thread mappings: bit-identical\n");
}

// ---- edge cases ----

SEPAL_TEST(edge, short_rows) {
    for (const char* which : {"tiny", "0.8b"}) {
        const Problem p = make_problem(std::string(which) == "tiny" ? sepal::tiny_test_config() : preset("0.8b"), 181);
        // Length 1, and lengths below K - 1 = 3 (the whole conv window still comes from the tail).
        for (const std::size_t len : {1u, 2u, 3u}) {
            const RowData row = make_row(p, len, 190 + len);
            const SingleRun run = run_single(p, row);
            const StageDiffs vs_f = compare_run(run, reference<float>(p, row));
            print_stage_diffs((std::string(which) + " len " + std::to_string(len)).c_str(), vs_f);
            check_parity(vs_f, 1e-4);
        }
    }
}

SEPAL_TEST(edge, zero_inputs) {
    for (const char* which : {"tiny", "0.8b"}) {
        const Problem p = make_problem(std::string(which) == "tiny" ? sepal::tiny_test_config() : preset("0.8b"), 201);
        // All-zero tokens and a zero starting state: nothing is ever written, so everything stays exactly zero.
        RowData zero = make_row(p, 20, 202, /*with_state=*/false);
        for (auto* v : {&zero.mixed, &zero.z, &zero.b, &zero.a}) std::fill(v->begin(), v->end(), 0.0f);
        const SingleRun from_zero = run_single(p, zero);
        auto all_zero = [](const std::vector<float>& v) {
            for (const float f : v) {
                if (f != 0.0f) return false;
            }
            return true;
        };
        CHECK(all_zero(from_zero.result.out));
        CHECK(all_zero(from_zero.result.reads));
        CHECK(all_zero(from_zero.state));
        CHECK(all_zero(from_zero.tail));

        // Zero tokens after a nonzero state: q, k, v are zero (the L2 norm of a zero vector stays zero), so the
        // update adds nothing and the state only decays: S_out = S_in * product of the 20 decays. The reference
        // does the same, so the device must agree with it, and the reads must be zero.
        RowData after = zero;
        const RowData seed = make_row(p, 1, 203);
        after.tail0 = seed.tail0;
        after.state0 = seed.state0;
        const SingleRun run = run_single(p, after);
        const RefOut<float> ref = reference<float>(p, after);
        const Diff state = compare(run.state, ref.state);
        std::printf("[measure] zero tokens after a nonzero state (%s): state max_abs %.2e, rel_rms %.2e\n", which,
                    state.max_abs, state.rel_rms);
        CHECK(state.rel_rms < 1e-4);
        // The nonzero tail leaks into the first K-1 reads through the conv; from then on q is zero, so a read is
        // exactly zero although the state is not.
        const Diff reads = compare(run.result.reads, ref.reads);
        CHECK(reads.max_abs < 1e-5);
        bool exact_zero = true;
        for (std::size_t t = p.taps - 1; t < after.len; ++t) {
            for (std::size_t j = 0; j < p.value_dim; ++j) exact_zero = exact_zero && run.result.reads[t * p.value_dim + j] == 0.0f;
        }
        CHECK(exact_zero);
    }
}

// The recurrence keeps its state scaled by the cumulative decay and folds the scale back in when it falls below 2^-40.
// Decays this strong (down to exactly zero) make that happen every few tokens, in every position of a chunk.
SEPAL_TEST(edge, strong_decay_folds_the_scale_into_the_state) {
    for (const char* which : {"tiny", "0.8b"}) {
        Problem p = make_problem(std::string(which) == "tiny" ? sepal::tiny_test_config() : preset("0.8b"), 221);
        // decay = exp(-exp(a_log) * softplus(a + dt_bias)): from 0.97 for the first head to e^-300 (zero in float32)
        // for the last ones, per token.
        for (std::size_t h = 0; h < p.hv; ++h) {
            const double frac = static_cast<double>(h) / static_cast<double>(std::max<std::size_t>(p.hv - 1, 1));
            p.a_log[h] = static_cast<float>(std::log(0.03 + frac * 200.0));
            p.dt_bias[h] = 0.5f;
        }
        for (const std::size_t len : {1u, 5u, 16u, 17u, 33u, 70u}) {
            const RowData row = make_row(p, len, 222 + len);
            const SingleRun run = run_single(p, row);
            const StageDiffs vs_d = compare_run(run, reference<double>(p, row));
            print_stage_diffs((std::string("strong decay, ") + which + " len " + std::to_string(len)).c_str(), vs_d);
            check_parity(vs_d, 1e-4);
            // and the same row inside a crowd of other rows gives the same bits
            const RowData other = make_row(p, 21, 230 + len);
            Rig rig(p, 4);
            rig.set_slot(0, row.tail0, row.state0);
            rig.set_slot(2, other.tail0, other.state0);
            const std::vector<JobResult> results = rig.run({{&other, 2, 3}, {&row, 0, 1}});
            const auto [tail, state] = rig.get_slot(1);
            CHECK(same_bits(results[1].out, run.result.out));
            CHECK(same_bits(state, run.state));
        }
    }
}

// The convolution window is a compile-time size for the usual 4 taps and a run-time one for the others.
SEPAL_TEST(edge, other_conv_tap_counts) {
    for (const std::uint64_t taps : {1u, 2u, 3u, 5u, 8u}) {
        sepal::ModelConfig cfg = sepal::tiny_test_config();
        cfg.conv_kernel = taps;
        const Problem p = make_problem(cfg, 241 + taps);
        for (const std::size_t len : {1u, 7u, 16u, 20u, 35u}) {
            const RowData row = make_row(p, len, 250 + len);
            const SingleRun run = run_single(p, row);
            const StageDiffs vs_d = compare_run(run, reference<double>(p, row));
            print_stage_diffs(("taps " + std::to_string(taps) + " len " + std::to_string(len)).c_str(), vs_d);
            check_parity(vs_d, 1e-4);
        }
    }
}

SEPAL_TEST(edge, chunk_boundaries_at_real_size) {
    const Problem p = make_problem(preset("0.8b"), 261);
    for (const std::size_t len : {15u, 16u, 17u, 31u, 32u, 33u, 48u, 64u, 65u}) {
        const RowData row = make_row(p, len, 270 + len);
        const SingleRun run = run_single(p, row);
        const StageDiffs vs_d = compare_run(run, reference<double>(p, row));
        print_stage_diffs(("0.8b len " + std::to_string(len)).c_str(), vs_d);
        check_parity(vs_d, 1e-4);
    }
}

SEPAL_TEST(edge, slots) {
    const Problem p = make_problem(sepal::tiny_test_config(), 211);
    const RowData row = make_row(p, 25, 212);
    const SingleRun expected = run_single(p, row);

    // Incoming kNoSlot is the same as a zero slot.
    {
        const RowData fresh = make_row(p, 25, 212, /*with_state=*/false);
        const SingleRun zero_slot = run_single(p, fresh);
        Rig rig(p, 2);
        const std::vector<JobResult> results = rig.run({{&fresh, kNoSlot, 1}});
        const auto [tail, state] = rig.get_slot(1);
        CHECK(same_bits(results[0].out, zero_slot.result.out));
        CHECK(same_bits(state, zero_slot.state));
        CHECK(same_bits(tail, zero_slot.tail));
    }
    // Outgoing kNoSlot: the output is the same and no slot is written.
    {
        Rig rig(p, 3);
        rig.set_slot(0, row.tail0, row.state0);
        rig.set_slot(1, row.tail0, row.state0);
        rig.set_slot(2, row.tail0, row.state0);
        const std::vector<JobResult> results = rig.run({{&row, 0, kNoSlot}});
        CHECK(same_bits(results[0].out, expected.result.out));
        for (std::uint32_t s = 0; s < 3; ++s) {
            const auto [tail, state] = rig.get_slot(s);
            CHECK(same_bits(tail, row.tail0));
            CHECK(same_bits(state, row.state0));
        }
    }
    // One row reads and writes the same slot (continued in place).
    {
        Rig rig(p, 1);
        rig.set_slot(0, row.tail0, row.state0);
        const std::vector<JobResult> results = rig.run({{&row, 0, 0}});
        const auto [tail, state] = rig.get_slot(0);
        CHECK(same_bits(results[0].out, expected.result.out));
        CHECK(same_bits(tail, expected.tail));
        CHECK(same_bits(state, expected.state));
    }
    // A row of length 0 copies its incoming state to its outgoing slot; an empty wave does nothing.
    {
        RowData empty = row;
        empty.len = 0;
        empty.mixed.clear();
        empty.z.clear();
        empty.b.clear();
        empty.a.clear();
        Rig rig(p, 2);
        rig.set_slot(0, row.tail0, row.state0);
        rig.run({{&empty, 0, 1}});
        const auto [tail, state] = rig.get_slot(1);
        CHECK(same_bits(tail, row.tail0));
        CHECK(same_bits(state, row.state0));
        RunOptions none;
        none.waves = {{}};
        rig.run({{&row, 0, 1}}, none);
        const auto [tail2, state2] = rig.get_slot(1);
        CHECK(same_bits(tail2, row.tail0));
    }
    // Rows that share a parent slot read the same bytes whatever the order of the wave.
    {
        Rig rig(p, 5);
        rig.set_slot(0, row.tail0, row.state0);
        const RowData a = make_row(p, 9, 213), b = make_row(p, 21, 214);
        rig.run({{&a, 0, 1}, {&b, 0, 2}});
        const auto s1 = rig.get_slot(1), s2 = rig.get_slot(2);
        rig.run({{&b, 0, 3}, {&a, 0, 4}});
        CHECK(same_bits(s1.second, rig.get_slot(4).second));
        CHECK(same_bits(s2.second, rig.get_slot(3).second));
        CHECK(same_bits(s1.first, rig.get_slot(4).first));
    }
}

SEPAL_TEST(edge, rejects_bad_geometry) {
    DeltaNetGeometry g;
    g.key_heads = 2;
    g.value_heads = 3; // not a multiple
    g.key_head_dim = 4;
    g.value_head_dim = 4;
    g.conv_kernel = 4;
    const DeltaNetRows rows{};
    const DeltaNetInputs inputs{};
    const DeltaNetStates states{};
    CHECK_THROWS(sepal::cuda::launch_deltanet_conv_tail(g, inputs, rows, states, nullptr), std::invalid_argument);
    g.value_heads = 4;
    g.key_head_dim = 129;
    CHECK_THROWS(sepal::cuda::launch_deltanet_conv_tail(g, inputs, rows, states, nullptr), std::invalid_argument);
}

// ---- storage: the state in binary16 ----

// Rounding happens at the store and nowhere else: a row run with binary16 slots from a state that binary16 holds exactly
// gives the very bits the float32 slots give, and what it stores is the float32 result rounded to nearest even.
SEPAL_TEST(storage, half_slots_are_the_float_result_rounded_at_the_store) {
    for (const char* name : {"tiny", "0.8b"}) {
        const sepal::ModelConfig cfg = std::string(name) == "tiny" ? sepal::tiny_test_config() : preset("0.8b");
        const Problem p = make_problem(cfg, 301);
        const std::size_t len = std::string(name) == "tiny" ? 41 : 150;
        // First row from nothing, then a second row continuing the first one's stored state, then a third.
        const RowData r1 = make_row(p, len, 302, false), r2 = make_row(p, len / 2 + 1, 303, false), r3 = make_row(p, 7, 304, false);
        Rig f(p, 4), h(p, 4, true);
        const auto o1f = f.run({{&r1, kNoSlot, 0}});
        const auto o1h = h.run({{&r1, kNoSlot, 0}});
        CHECK(same_bits(o1f[0].out, o1h[0].out)); // the row's own outputs never see the rounding
        const auto [tf, sf] = f.get_slot(0);
        const auto [tb, sb] = h.get_slot_bits(0);
        for (std::size_t i = 0; i < sf.size(); ++i) {
            if (sb[i] != std::bit_cast<std::uint16_t>(__float2half_rn(sf[i]))) { CHECK(false); break; }
        }
        for (std::size_t i = 0; i < tf.size(); ++i) {
            if (tb[i] != std::bit_cast<std::uint16_t>(__float2half_rn(tf[i]))) { CHECK(false); break; }
        }
        // Float32 slots holding the widened values stand in for the binary16 slots.
        const auto [th, sh] = h.get_slot(0);
        f.set_slot(1, th, sh);
        const auto o2f = f.run({{&r2, 1, 2}});
        const auto o2h = h.run({{&r2, 0, 1}});
        CHECK(same_bits(o2f[0].out, o2h[0].out));
        const auto [t2f, s2f] = f.get_slot(2);
        const auto [t2b, s2b] = h.get_slot_bits(1);
        bool all = true;
        for (std::size_t i = 0; i < s2f.size(); ++i) all = all && s2b[i] == std::bit_cast<std::uint16_t>(__float2half_rn(s2f[i]));
        for (std::size_t i = 0; i < t2f.size(); ++i) all = all && t2b[i] == std::bit_cast<std::uint16_t>(__float2half_rn(t2f[i]));
        CHECK(all);
        // In place (read and write the same slot) and a row with an out slot but no in slot both work.
        const auto [t2h, s2h] = h.get_slot(1);
        f.set_slot(3, t2h, s2h);
        const auto o3f = f.run({{&r3, 3, 3}});
        const auto o3h = h.run({{&r3, 1, 1}});
        CHECK(same_bits(o3f[0].out, o3h[0].out));
        CHECK_EQ(h.saturations(), 0u);
        std::printf("[measure] storage %s: binary16 slots = float32 result rounded at the store, outputs equal bit for bit\n", name);
    }
}

// A value beyond the binary16 range is clamped to +-65504, never infinity, and counted; the same input with float32 slots
// really does hold values beyond the range.
SEPAL_TEST(storage, values_beyond_the_range_are_clamped_and_counted) {
    const Problem p = make_problem(sepal::tiny_test_config(), 311);
    RowData big = make_row(p, 24, 312, false);
    for (std::size_t t = 0; t < big.len; ++t) {
        // large positive v channels: SiLU keeps them, the state grows to about beta * v
        for (std::size_t c = 2 * p.g.key_dim(); c < p.conv_dim; ++c) big.mixed[t * p.conv_dim + c] = 4.0e6f;
    }
    Rig f(p, 2), h(p, 2, true);
    f.run({{&big, kNoSlot, 0}});
    h.run({{&big, kNoSlot, 0}});
    const auto [tf, sf] = f.get_slot(0);
    float worst = 0.0f;
    for (const float v : sf) worst = std::max(worst, std::fabs(v));
    CHECK(worst > 65504.0f);
    const auto [tb, sb] = h.get_slot_bits(0);
    float worst_h = 0.0f;
    for (const std::uint16_t b : sb) {
        const float v = __half2float(std::bit_cast<__half>(b));
        CHECK(std::isfinite(v));
        worst_h = std::max(worst_h, std::fabs(v));
    }
    CHECK_EQ(worst_h, 65504.0f);
    CHECK(h.saturations() > 0);
    // The conv tail holds the raw inputs (4e6 here): clamped as well.
    for (const std::uint16_t b : tb) CHECK(std::isfinite(__half2float(std::bit_cast<__half>(b))));
    (void)tf;
    // An in-range input counts nothing.
    Rig h2(p, 2, true);
    const RowData ok = make_row(p, 24, 313, false);
    h2.run({{&ok, kNoSlot, 0}});
    CHECK_EQ(h2.saturations(), 0u);
}
