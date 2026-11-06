// Checks and times the float32 SIMT chunked DeltaNet kernel (src/kernels_deltanet_chunked_f32.cu) against the scan kernel
// (src/kernels_deltanet.cu), which stands for the token-serial recurrence.
//   sepal-cuda-chunk-check-f32 [--geometry 4b|small|odd|all] [--lengths 1,2,3,7,63,64,65,...] [--shape RxL] [--seed N]
//                              [--state on|off] [--half-state on|off] [--pad on|off] [--drop on|off] [--inplace on|off]
//                              [--identity on|off] [--tol 1e-4] [--bench REPS]
// For every geometry and every row of one wave it prints the largest difference of the outputs and of the outgoing recurrent
// states and the rms of the difference, each over the rms of the scan kernel's tensor; the conv tails must be bit-identical.
// Exit status 1 when a relative rms difference is over --tol (default 1e-4). With --identity on (the default) it also checks the
// bit-identity rules of the kernel: two runs of the same wave; the same wave with its rows in the reverse order in the flat
// token list and in the wave; every row alone in a wave of its own. --pad on gives every input and the output a row stride wider
// than the matrix (the padding holds large garbage, which must not be read, and a sentinel in the output, which must stay);
// --drop on runs every third row without an outgoing slot; --inplace on continues every second row in its incoming slot (needs
// --state on). --bench REPS times scan and chunked (best of REPS).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_fp16.h>

#include "kernels_deltanet.hpp"
#include "kernels_deltanet_chunked_f32.hpp"

using namespace sepal::cuda;

namespace {

constexpr float kGarbage = 7.0e30f;   // in the padding of the inputs: a kernel that reads it gives garbage or infinity
constexpr float kSentinel = 12345.0f; // in the padding of the output: a kernel that writes there changes it

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

struct RowData {
    std::vector<float> mixed, z, b, a; // [len, conv_dim], [len, value_dim], [len, value heads] x2
    std::vector<float> tail, rec;      // incoming conv tail [K-1, conv_dim] and state [value heads, dk, dv]
};

struct Options {
    bool with_state = true, half_state = false, pad = false, drop = false, inplace = false;
};

struct Problem {
    DeltaNetGeometry g;
    std::vector<float> conv_w, a_log, dt_bias, norm_w;
    std::vector<std::uint32_t> lengths;
    std::vector<RowData> rows;
    Options opt;
};

Problem make_problem(const DeltaNetGeometry& g, const std::vector<std::uint32_t>& lengths, std::uint64_t seed, const Options& opt) {
    Problem p;
    p.g = g;
    p.lengths = lengths;
    p.opt = opt;
    const std::size_t conv_dim = g.conv_dim(), value_dim = g.value_dim(), hv = g.value_heads;
    Rng rng{seed};
    // a model-like range of decays: some heads forget within a few tokens, some hold for thousands
    p.conv_w = rnd(rng, conv_dim * g.conv_kernel, -0.6f, 0.6f);
    p.dt_bias = rnd(rng, hv, -3.0f, 0.0f);
    p.norm_w = rnd(rng, g.value_head_dim, 0.5f, 1.5f);
    p.a_log.resize(hv);
    for (float& f : p.a_log) f = std::log(rng.uniform(0.005f, 2.0f));
    // each row has its own generator, so a row's data does not depend on the other rows
    for (std::size_t r = 0; r < lengths.size(); ++r) {
        Rng rr{seed * 1000003ull + 7919ull * (r + 1)};
        RowData d;
        const std::size_t len = lengths[r];
        d.mixed = rnd(rr, len * conv_dim, -2.0f, 2.0f);
        d.z = rnd(rr, len * value_dim, -3.0f, 3.0f);
        d.b = rnd(rr, len * hv, -3.0f, 3.0f);
        d.a = rnd(rr, len * hv, -2.0f, 2.0f);
        if (opt.with_state) {
            d.tail = rnd(rr, g.conv_tail_floats(), -2.0f, 2.0f);
            d.rec = rnd(rr, g.recurrent_floats(), -0.4f, 0.4f);
        }
        p.rows.push_back(std::move(d));
    }
    return p;
}

struct RowResult {
    std::vector<float> out, rec, tail; // the outgoing state and conv tail, widened to float32 (empty without an outgoing slot)
    bool out_pad_ok = true;            // the padding of the output rows still holds the sentinel
};

__half to_half(float f) { return __float2half_rn(f); }

enum class Kernel { Scan, Chunked };

// Whether the row at position `i` of a layout has no outgoing slot, and whether it continues in its incoming slot.
bool row_dropped(const Options& o, std::size_t i) { return o.drop && i % 3 == 2; }
bool row_inplace(const Options& o, std::size_t i) { return o.inplace && o.with_state && i % 2 == 1 && !row_dropped(o, i); }

// One flat token layout and wave order of a problem on the device: `order` lists the rows in their order in the flat token
// list, `wave_pos` the positions (in `order`) in the order of the wave.
class Setup {
public:
    Setup(const Problem& p, const std::vector<std::size_t>& order, const std::vector<std::size_t>& wave_pos) : p_(p), order_(order) {
        const DeltaNetGeometry& g = p.g;
        const std::size_t R = order.size();
        const std::size_t conv_dim = g.conv_dim(), value_dim = g.value_dim(), hv = g.value_heads;
        const bool pad = p.opt.pad;
        mixed_stride_ = conv_dim + (pad ? 8 : 0);
        z_stride_ = value_dim + (pad ? 12 : 0);
        b_stride_ = hv + (pad ? 3 : 0);
        a_stride_ = hv + (pad ? 5 : 0);
        out_stride_ = value_dim + (pad ? 4 : 0); // a multiple of 4: the vector stores are used when dv is
        value_dim_ = value_dim;
        std::vector<RowSpan> spans(R);
        std::size_t total = 0;
        for (std::size_t i = 0; i < R; ++i) {
            spans[i] = {static_cast<std::uint32_t>(total), p.lengths[order[i]]};
            total += p.lengths[order[i]];
        }
        total_ = total;
        std::vector<float> mixed(total * mixed_stride_, kGarbage), z(total * z_stride_, kGarbage), b(total * b_stride_, kGarbage),
            a(total * a_stride_, kGarbage);
        for (std::size_t i = 0; i < R; ++i) {
            const RowData& d = p.rows[order[i]];
            for (std::size_t t = 0; t < spans[i].length; ++t) {
                const std::size_t token = spans[i].offset + t;
                std::copy(d.mixed.begin() + static_cast<std::ptrdiff_t>(t * conv_dim), d.mixed.begin() + static_cast<std::ptrdiff_t>((t + 1) * conv_dim),
                          mixed.begin() + static_cast<std::ptrdiff_t>(token * mixed_stride_));
                std::copy(d.z.begin() + static_cast<std::ptrdiff_t>(t * value_dim), d.z.begin() + static_cast<std::ptrdiff_t>((t + 1) * value_dim),
                          z.begin() + static_cast<std::ptrdiff_t>(token * z_stride_));
                std::copy(d.b.begin() + static_cast<std::ptrdiff_t>(t * hv), d.b.begin() + static_cast<std::ptrdiff_t>((t + 1) * hv),
                          b.begin() + static_cast<std::ptrdiff_t>(token * b_stride_));
                std::copy(d.a.begin() + static_cast<std::ptrdiff_t>(t * hv), d.a.begin() + static_cast<std::ptrdiff_t>((t + 1) * hv),
                          a.begin() + static_cast<std::ptrdiff_t>(token * a_stride_));
            }
        }
        // slots [0, R): incoming (by position), [R, 2R): the outgoing slots of the rows that do not continue in place
        tail_floats_ = g.conv_tail_floats();
        rec_floats_ = g.recurrent_floats();
        const std::size_t slots = 2 * R;
        tails_init_.assign(slots * tail_floats_, 0.0f);
        rec_init_.assign(slots * rec_floats_, 0.0f);
        if (p.opt.with_state) {
            for (std::size_t i = 0; i < R; ++i) {
                const RowData& d = p.rows[order[i]];
                std::copy(d.tail.begin(), d.tail.end(), tails_init_.begin() + static_cast<std::ptrdiff_t>(i * tail_floats_));
                std::copy(d.rec.begin(), d.rec.end(), rec_init_.begin() + static_cast<std::ptrdiff_t>(i * rec_floats_));
            }
        }
        std::vector<std::uint32_t> in_slot(R), out_slot(R);
        out_slot_.assign(R, kNoSlot);
        for (std::size_t i = 0; i < R; ++i) {
            in_slot[i] = p.opt.with_state ? static_cast<std::uint32_t>(i) : kNoSlot;
            out_slot[i] = row_dropped(p.opt, i) ? kNoSlot : row_inplace(p.opt, i) ? static_cast<std::uint32_t>(i) : static_cast<std::uint32_t>(R + i);
            out_slot_[i] = out_slot[i];
        }
        std::vector<std::uint32_t> wave, wave_lengths;
        for (std::size_t w : wave_pos) {
            wave.push_back(static_cast<std::uint32_t>(w));
            wave_lengths.push_back(spans[w].length);
        }
        const std::vector<std::uint32_t> prefix = deltanet_chunk_prefix(wave_lengths);

        d_conv_w_ = DeviceBuffer<float>(p.conv_w);
        d_a_log_ = DeviceBuffer<float>(p.a_log);
        d_dt_ = DeviceBuffer<float>(p.dt_bias);
        d_norm_ = DeviceBuffer<float>(p.norm_w);
        d_mixed_ = DeviceBuffer<float>(mixed);
        d_z_ = DeviceBuffer<float>(z);
        d_b_ = DeviceBuffer<float>(b);
        d_a_ = DeviceBuffer<float>(a);
        d_spans_ = DeviceBuffer<RowSpan>(spans);
        d_in_ = DeviceBuffer<std::uint32_t>(in_slot);
        d_out_slot_ = DeviceBuffer<std::uint32_t>(out_slot);
        d_wave_ = DeviceBuffer<std::uint32_t>(wave);
        d_prefix_ = DeviceBuffer<std::uint32_t>(prefix);
        // the scan and the chunked kernel each get their own copy of the states
        for (int k = 0; k < 2; ++k) {
            if (p.opt.half_state) {
                tails_h_[k] = DeviceBuffer<__half>(tails_init_.size());
                rec_h_[k] = DeviceBuffer<__half>(rec_init_.size());
                states_[k] = DeltaNetStates{tails_h_[k].data(), rec_h_[k].data(), true, nullptr};
            } else {
                tails_f_[k] = DeviceBuffer<float>(tails_init_.size());
                rec_f_[k] = DeviceBuffer<float>(rec_init_.size());
                states_[k] = DeltaNetStates{tails_f_[k].data(), rec_f_[k].data(), false, nullptr};
            }
        }
        reset_states();
        const std::vector<float> out_init(total * out_stride_, kSentinel);
        d_out_[0] = DeviceBuffer<float>(out_init);
        d_out_[1] = DeviceBuffer<float>(out_init);
        w_ = DeltaNetWeights{d_conv_w_.data(), d_a_log_.data(), d_dt_.data(), d_norm_.data()};
        in_ = DeltaNetInputs{{d_mixed_.data(), mixed_stride_}, {d_z_.data(), z_stride_}, {d_b_.data(), b_stride_}, {d_a_.data(), a_stride_}};
        rows_ = DeltaNetRows{d_spans_.data(), d_in_.data(), d_out_slot_.data(), d_wave_.data(), static_cast<std::uint32_t>(R)};
        plan_ = DeltaNetChunkPlan{d_prefix_.data(), prefix.back()};
        ws_bytes_ = deltanet_chunked_f32_workspace_bytes(g, plan_.total_chunks, total);
        d_ws_ = DeviceBuffer<char>(ws_bytes_);
        chunks_ = plan_.total_chunks;
        spans_ = spans;
    }

    std::size_t tokens() const { return total_; }
    std::size_t chunks() const { return chunks_; }
    std::size_t workspace_bytes() const { return ws_bytes_; }

    // The states go back to their initial contents (rows continued in place change theirs).
    void reset_states() {
        for (int k = 0; k < 2; ++k) {
            if (p_.opt.half_state) {
                std::vector<__half> th(tails_init_.size()), rh(rec_init_.size());
                for (std::size_t i = 0; i < th.size(); ++i) th[i] = to_half(tails_init_[i]);
                for (std::size_t i = 0; i < rh.size(); ++i) rh[i] = to_half(rec_init_[i]);
                tails_h_[k].upload(th.data(), th.size());
                rec_h_[k].upload(rh.data(), rh.size());
            } else {
                tails_f_[k].upload(tails_init_.data(), tails_init_.size());
                rec_f_[k].upload(rec_init_.data(), rec_init_.size());
            }
        }
    }

    void run(Kernel k) {
        if (k == Kernel::Scan) {
            launch_deltanet_wave<float>(p_.g, w_, in_, rows_, states_[0], d_out_[0].data(), out_stride_, nullptr);
        } else {
            launch_deltanet_chunked_f32(p_.g, w_, in_, rows_, plan_, states_[1], d_ws_.data(), ws_bytes_, total_, d_out_[1].data(), out_stride_, nullptr);
        }
    }

    // Everything the kernel left, per row of the problem (indexed like Problem::rows).
    std::vector<RowResult> results(Kernel k) const {
        const std::size_t R = order_.size();
        const int idx = k == Kernel::Scan ? 0 : 1;
        const std::vector<float> out = d_out_[idx].download();
        std::vector<float> rec, tails;
        if (p_.opt.half_state) {
            for (__half h : rec_h_[idx].download()) rec.push_back(__half2float(h));
            for (__half h : tails_h_[idx].download()) tails.push_back(__half2float(h));
        } else {
            rec = rec_f_[idx].download();
            tails = tails_f_[idx].download();
        }
        std::vector<RowResult> res(p_.rows.size());
        for (std::size_t i = 0; i < R; ++i) {
            RowResult r;
            const std::size_t len = p_.lengths[order_[i]];
            r.out.resize(len * value_dim_);
            for (std::size_t t = 0; t < len; ++t) {
                const std::size_t token = spans_[i].offset + t;
                for (std::size_t c = 0; c < out_stride_; ++c) {
                    const float v = out[token * out_stride_ + c];
                    if (c < value_dim_) r.out[t * value_dim_ + c] = v;
                    else if (v != kSentinel) r.out_pad_ok = false;
                }
            }
            if (out_slot_[i] != kNoSlot) {
                const std::size_t s = out_slot_[i];
                r.rec.assign(rec.begin() + static_cast<std::ptrdiff_t>(s * rec_floats_), rec.begin() + static_cast<std::ptrdiff_t>((s + 1) * rec_floats_));
                r.tail.assign(tails.begin() + static_cast<std::ptrdiff_t>(s * tail_floats_), tails.begin() + static_cast<std::ptrdiff_t>((s + 1) * tail_floats_));
            }
            res[order_[i]] = std::move(r);
        }
        return res;
    }

private:
    const Problem& p_;
    std::vector<std::size_t> order_;
    std::vector<RowSpan> spans_;
    std::vector<std::uint32_t> out_slot_;
    std::vector<float> tails_init_, rec_init_;
    std::size_t mixed_stride_ = 0, z_stride_ = 0, b_stride_ = 0, a_stride_ = 0, out_stride_ = 0, value_dim_ = 0;
    std::size_t tail_floats_ = 0, rec_floats_ = 0, total_ = 0, ws_bytes_ = 0, chunks_ = 0;
    DeviceBuffer<float> d_conv_w_, d_a_log_, d_dt_, d_norm_, d_mixed_, d_z_, d_b_, d_a_;
    DeviceBuffer<RowSpan> d_spans_;
    DeviceBuffer<std::uint32_t> d_in_, d_out_slot_, d_wave_, d_prefix_;
    DeviceBuffer<float> tails_f_[2], rec_f_[2];
    DeviceBuffer<__half> tails_h_[2], rec_h_[2];
    DeltaNetStates states_[2];
    DeviceBuffer<float> d_out_[2];
    DeviceBuffer<char> d_ws_;
    DeltaNetWeights w_;
    DeltaNetInputs in_;
    DeltaNetRows rows_;
    DeltaNetChunkPlan plan_;
};

struct Diff {
    double max_abs = 0, rms_diff = 0, rms_ref = 0;
    double rel_rms() const { return rms_ref > 0 ? rms_diff / rms_ref : 0.0; }
    double max_rel() const { return rms_ref > 0 ? max_abs / rms_ref : 0.0; }
};

Diff compare(const std::vector<float>& ref, const std::vector<float>& got) {
    Diff d;
    if (ref.size() != got.size()) {
        d.max_abs = INFINITY;
        d.rms_diff = INFINITY;
        d.rms_ref = 1.0;
        return d;
    }
    double ss = 0, sd = 0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        const double x = ref[i], y = got[i], e = std::fabs(x - y);
        ss += x * x;
        sd += e * e;
        d.max_abs = std::max(d.max_abs, e);
        if (!(e == e)) d.max_abs = INFINITY; // a NaN anywhere is a failure
    }
    const double n = ref.empty() ? 1.0 : static_cast<double>(ref.size());
    d.rms_ref = std::sqrt(ss / n);
    d.rms_diff = std::sqrt(sd / n);
    if (!(d.rms_diff == d.rms_diff)) d.rms_diff = INFINITY;
    return d;
}

bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}

bool same_row(const RowResult& a, const RowResult& b) { return same_bits(a.out, b.out) && same_bits(a.rec, b.rec) && same_bits(a.tail, b.tail); }

struct Named {
    const char* name;
    DeltaNetGeometry g;
};

std::vector<Named> geometries(const std::string& which) {
    std::vector<Named> all;
    auto add = [&](const char* name, std::uint32_t kh, std::uint32_t vh, std::uint32_t dk, std::uint32_t dv, std::uint32_t taps) {
        DeltaNetGeometry g;
        g.key_heads = kh;
        g.value_heads = vh;
        g.key_head_dim = dk;
        g.value_head_dim = dv;
        g.conv_kernel = taps;
        g.rms_norm_eps = 1e-6;
        all.push_back({name, g});
    };
    if (which == "4b" || which == "all") add("4b (16 key heads, 32 value heads, 128 x 128, 4 taps)", 16, 32, 128, 128, 4);
    if (which == "small" || which == "all") add("small (2 key heads, 4 value heads, 32 x 32, 4 taps)", 2, 4, 32, 32, 4);
    if (which == "odd" || which == "all") add("odd (3 key heads, 6 value heads, 17 x 29, 5 taps)", 3, 6, 17, 29, 5);
    return all;
}

int run_geometry(const Named& geo, const std::vector<std::uint32_t>& lengths, std::uint64_t seed, const Options& opt, bool identity, double tol,
                 int bench_reps) {
    const DeltaNetGeometry& g = geo.g;
    const std::size_t R = lengths.size();
    std::size_t total = 0;
    for (std::uint32_t l : lengths) total += l;
    std::printf("== %s: %zu rows, %zu tokens, incoming state %s, state storage %s%s%s%s\n", geo.name, R, total, opt.with_state ? "yes" : "none",
                opt.half_state ? "binary16" : "float32", opt.pad ? ", padded strides" : "", opt.drop ? ", every third row without outgoing slot" : "",
                opt.inplace && opt.with_state ? ", every second row in place" : "");
    const Problem p = make_problem(g, lengths, seed, opt);
    std::vector<std::size_t> natural(R), reversed(R), wave_natural(R), wave_reversed(R);
    for (std::size_t i = 0; i < R; ++i) {
        natural[i] = i;
        reversed[i] = R - 1 - i;
        wave_natural[i] = i;
        wave_reversed[i] = R - 1 - i;
    }
    bool bad = false;

    std::vector<RowResult> scan, run1;
    {
        Setup s(p, natural, wave_natural);
        std::printf("   %zu chunks, workspace %.1f MiB\n", s.chunks(), static_cast<double>(s.workspace_bytes()) / 1048576.0);
        s.run(Kernel::Scan);
        s.run(Kernel::Chunked);
        check(cudaDeviceSynchronize(), "sync");
        scan = s.results(Kernel::Scan);
        run1 = s.results(Kernel::Chunked);
        if (identity) {
            s.reset_states();
            s.run(Kernel::Chunked);
            check(cudaDeviceSynchronize(), "sync");
            const std::vector<RowResult> again = s.results(Kernel::Chunked);
            for (std::size_t r = 0; r < R; ++r) {
                if (!same_row(run1[r], again[r])) {
                    std::printf("   FAIL: row %zu (len %u) differs between two runs of the same wave\n", r, lengths[r]);
                    bad = true;
                }
            }
        }
        if (bench_reps > 0) {
            cudaEvent_t e0, e1;
            check(cudaEventCreate(&e0), "event");
            check(cudaEventCreate(&e1), "event");
            auto time = [&](Kernel k) {
                float best = 1e30f;
                for (int i = 0; i < bench_reps; ++i) {
                    check(cudaEventRecord(e0), "event");
                    s.run(k);
                    check(cudaEventRecord(e1), "event");
                    check(cudaEventSynchronize(e1), "event");
                    float ms = 0;
                    check(cudaEventElapsedTime(&ms, e0, e1), "event");
                    best = std::min(best, ms);
                }
                return best;
            };
            s.run(Kernel::Scan);
            s.run(Kernel::Chunked);
            check(cudaDeviceSynchronize(), "sync");
            const float ts = time(Kernel::Scan), tc = time(Kernel::Chunked);
            std::printf("   time per layer (best of %d): scan %.3f ms, chunked f32 %.3f ms, %.2fx\n", bench_reps, ts, tc, ts / tc);
            cudaEventDestroy(e0);
            cudaEventDestroy(e1);
        }
    }

    double worst_out = 0, worst_state = 0;
    for (std::size_t r = 0; r < R; ++r) {
        const Diff o = compare(scan[r].out, run1[r].out), s = compare(scan[r].rec, run1[r].rec);
        const bool tails_equal = same_bits(scan[r].tail, run1[r].tail);
        const bool pads_ok = scan[r].out_pad_ok && run1[r].out_pad_ok;
        std::printf("   row %2zu len %5u: out max|d| %.3e (%.3e of rms), rms d/rms %.3e | state max|d| %.3e (%.3e of rms), rms d/rms %.3e%s%s\n", r, lengths[r],
                    o.max_abs, o.max_rel(), o.rel_rms(), s.max_abs, s.max_rel(), s.rel_rms(), tails_equal ? "" : " | CONV TAIL DIFFERS",
                    pads_ok ? "" : " | OUTPUT PADDING WRITTEN");
        worst_out = std::max(worst_out, o.rel_rms());
        worst_state = std::max(worst_state, s.rel_rms());
        if (!(o.rel_rms() <= tol) || !(s.rel_rms() <= tol) || !tails_equal || !pads_ok) {
            std::printf("   FAIL: row %zu (len %u) is over the tolerance %.1e, or its conv tail differs, or the output padding was written\n", r, lengths[r], tol);
            bad = true;
        }
    }
    std::printf("   worst rms d/rms: out %.3e, state %.3e (tolerance %.1e)\n", worst_out, worst_state, tol);

    if (identity) {
        {
            Setup s(p, reversed, wave_reversed);
            s.run(Kernel::Chunked);
            check(cudaDeviceSynchronize(), "sync");
            const std::vector<RowResult> rev = s.results(Kernel::Chunked);
            int diffs = 0;
            for (std::size_t r = 0; r < R; ++r) {
                // a row's slot role (dropped, in place) depends on its position, which the reversal changes: compare the outputs only then
                const bool same_role = row_dropped(opt, r) == row_dropped(opt, R - 1 - r) && row_inplace(opt, r) == row_inplace(opt, R - 1 - r);
                const bool equal = same_role ? same_row(run1[r], rev[r]) : same_bits(run1[r].out, rev[r].out);
                if (!equal) {
                    std::printf("   FAIL: row %zu (len %u) differs when the wave is in the reverse order\n", r, lengths[r]);
                    ++diffs;
                }
            }
            if (diffs == 0) std::printf("   bit-identity: all %zu rows equal in the reversed layout (flat list and wave)\n", R);
            else bad = true;
        }
        int solo_diffs = 0;
        for (std::size_t r = 0; r < R; ++r) {
            if (lengths[r] == 0) continue; // a wave without a token has no inputs to point to
            Setup s(p, {r}, {0});
            s.run(Kernel::Chunked);
            check(cudaDeviceSynchronize(), "sync");
            const std::vector<RowResult> one = s.results(Kernel::Chunked);
            const bool same_role = row_dropped(opt, r) == row_dropped(opt, 0) && row_inplace(opt, r) == row_inplace(opt, 0);
            const bool equal = same_role ? same_row(run1[r], one[r]) : same_bits(run1[r].out, one[r].out);
            if (!equal) {
                std::printf("   FAIL: row %zu (len %u) alone differs from the row in the wave\n", r, lengths[r]);
                ++solo_diffs;
            }
        }
        if (solo_diffs == 0) std::printf("   bit-identity: every row alone equals the row in the wave\n");
        else bad = true;
    }
    return bad ? 1 : 0;
}

} // namespace

int main(int argc, char** argv) {
    std::vector<std::uint32_t> lengths = {1, 2, 3, 7, 63, 64, 65, 127, 128, 129, 1000, 4096};
    std::uint64_t seed = 1;
    Options opt;
    bool identity = true;
    int bench = 0;
    double tol = 1e-4;
    std::string geometry = "all";
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i], v = argv[i + 1];
        if (a == "--lengths") {
            lengths.clear();
            std::stringstream ss(v);
            std::string item;
            while (std::getline(ss, item, ',')) lengths.push_back(static_cast<std::uint32_t>(std::stoul(item)));
        } else if (a == "--shape") { // RxL: R rows of L tokens
            const std::size_t x = v.find('x');
            if (x == std::string::npos) { std::cerr << "--shape wants RxL\n"; return 2; }
            lengths.assign(std::stoul(v.substr(0, x)), static_cast<std::uint32_t>(std::stoul(v.substr(x + 1))));
        } else if (a == "--seed") seed = std::stoull(v);
        else if (a == "--state") opt.with_state = v == "on";
        else if (a == "--half-state") opt.half_state = v == "on";
        else if (a == "--pad") opt.pad = v == "on";
        else if (a == "--drop") opt.drop = v == "on";
        else if (a == "--inplace") opt.inplace = v == "on";
        else if (a == "--identity") identity = v == "on";
        else if (a == "--tol") tol = std::stod(v);
        else if (a == "--bench") bench = std::stoi(v);
        else if (a == "--geometry") geometry = v;
        else { std::cerr << "unknown option " << a << "\n"; return 2; }
    }
    try {
        const std::vector<Named> geos = geometries(geometry);
        if (geos.empty()) { std::cerr << "--geometry is 4b, small, odd or all\n"; return 2; }
        int status = 0;
        for (const Named& geo : geos) {
            if (!deltanet_chunked_f32_supported(geo.g)) { std::cerr << "geometry not supported\n"; return 2; }
            status |= run_geometry(geo, lengths, seed, opt, identity, tol, bench);
        }
        std::printf(status == 0 ? "PASS\n" : "FAIL\n");
        return status;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 2;
    }
}
