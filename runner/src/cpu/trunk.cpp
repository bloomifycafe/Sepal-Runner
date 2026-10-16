#include "sepal/cpu/trunk.hpp"

#include <immintrin.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <deque>
#include <stdexcept>
#include <string>

#include "sepal/cpu/aligned.hpp"
#include "sepal/cpu/attention.hpp"
#include "sepal/cpu/deltanet.hpp"
#include "sepal/cpu/elementwise.hpp"
#include "sepal/cpu/gemm.hpp"
#include "vmath.hpp"

namespace sepal::cpu {

const char* name_of(Profile p) {
    switch (p) {
    case Profile::F32: return "cpu-f32";
    case Profile::BF16: return "cpu-bf16";
    case Profile::Q8: return "cpu-q8";
    }
    return "?";
}

namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

struct Timer {
    double& sink;
    Clock::time_point start = Clock::now();
    explicit Timer(double& s) : sink(s) {}
    ~Timer() { sink += ms_since(start); }
};

std::size_t dimension(std::uint64_t v) { return static_cast<std::size_t>(v); }

// ---- packed model -----------------------------------------------------------------------------------------------

struct DeltaLayer {
    PackedWeight in_qz;  // rows: in_proj_qkv then in_proj_z
    PackedWeight ab;     // rows: in_proj_a then in_proj_b
    PackedWeight out;
    std::vector<float> conv_wt;  // [taps][conv_dim]
    std::vector<float> a_log, dt_bias, norm_w;
};

struct AttentionLayer {
    PackedWeight qkv;  // rows: q_proj (query and gate per head), k_proj, v_proj
    PackedWeight o;
    std::vector<float> q_norm1, k_norm1;  // 1 + stored weight
};

struct Layer {
    bool attention = false;
    std::vector<float> input_norm1, post_norm1;
    PackedWeight gate_up;  // SwiGLU pair
    PackedWeight down;
    DeltaLayer dn;
    AttentionLayer at;
};

std::vector<float> one_plus(const BasicTensor<float>& t) {
    std::vector<float> out(t.size());
    for (std::size_t i = 0; i < out.size(); ++i) out[i] = 1.0f + t[i];
    return out;
}

std::vector<float> copy_of(const BasicTensor<float>& t) { return std::vector<float>(t.data(), t.data() + t.size()); }

std::vector<float> concat(std::initializer_list<const BasicTensor<float>*> parts) {
    std::size_t total = 0;
    for (const auto* p : parts) total += p->size();
    std::vector<float> out;
    out.reserve(total);
    for (const auto* p : parts) out.insert(out.end(), p->data(), p->data() + p->size());
    return out;
}

// ---- per-request state ------------------------------------------------------------------------------------------

struct RequestState {
    std::vector<AlignedBuffer<float>> dn_state;   // per DeltaNet layer: [value heads, dk, dv]
    std::vector<AlignedBuffer<float>> conv_tail;  // per DeltaNet layer: [taps - 1, conv dim]
    std::vector<std::unique_ptr<KvCache>> kv;     // per attention layer
};

struct Segment {
    std::size_t request;
    std::size_t position;  // first position of the chunk
    std::size_t rows;
    std::size_t row0;      // first row in the step buffers
};

// Reusable packed activations keyed by variant; `reset` invalidates them for new data.
class ActsCache {
public:
    void reset() {
        for (Entry& e : entries_) e.valid = false;
    }
    PackedActs& pack(const float* x, std::size_t ld, std::size_t rows, std::size_t k, Variant v, ThreadPool& pool) {
        for (Entry& e : entries_)
            if (e.valid && e.acts.variant == v) return e.acts;
        for (Entry& e : entries_) {
            if (!e.valid) {
                pack_acts(e.acts, x, ld, rows, k, v, pool);
                e.valid = true;
                return e.acts;
            }
        }
        entries_.emplace_back();
        pack_acts(entries_.back().acts, x, ld, rows, k, v, pool);
        entries_.back().valid = true;
        return entries_.back().acts;
    }

private:
    struct Entry {
        PackedActs acts;
        bool valid = false;
    };
    std::deque<Entry> entries_;
};

} // namespace

struct CpuTrunk::Impl {
    ModelConfig cfg;
    TrunkOptions opt;
    Isa isa;
    ThreadPool pool;
    EmbeddingFn embed;
    std::vector<Layer> layers;
    std::vector<float> final_norm1;
    std::uint64_t weight_bytes = 0;
    TrunkTimes times;

    // Derived sizes.
    std::size_t H, I, taps, conv_dim, key_dim, value_dim, hv, hk, dk, dv;
    std::size_t heads, kv_heads, hd, rot_half, qz_width;
    std::size_t n_dn = 0, n_attn = 0;

    // Step workspace.
    AlignedBuffer<float> x, h, qz, ab, conv_out, gate_g, beta_b, core, qbuf, attended, act, cos_tab, sin_tab;
    ActsCache acts_h, acts_c;

    Impl(const ModelConfig& c, const TrunkOptions& o, EmbeddingFn e)
        : cfg(c), opt(o), isa(o.force_isa ? o.isa : best_isa()), pool(o.threads), embed(std::move(e)) {}

    WeightKind kind_for(bool int8_group) const {
        switch (opt.profile) {
        case Profile::F32: return WeightKind::F32;
        case Profile::BF16: return WeightKind::BF16;
        case Profile::Q8: return int8_group ? WeightKind::I8 : WeightKind::BF16;
        }
        return WeightKind::F32;
    }

    void build(const Weights<float>& w);
    void run_step(std::span<const std::span<const std::uint32_t>> requests, std::vector<RequestState>& states,
                  const std::vector<Segment>& segs, std::size_t rows, std::vector<std::vector<float>>& hidden,
                  const LayerObserver* observer);
    void delta_layer(const Layer& L, std::size_t index, std::vector<RequestState>& states, const std::vector<Segment>& segs,
                     std::size_t rows);
    void attention_layer(const Layer& L, std::size_t index, std::vector<RequestState>& states, const std::vector<Segment>& segs,
                         std::size_t rows);
    void mlp(const Layer& L, std::size_t rows);
    std::uint64_t state_bytes(std::size_t tokens) const;
};

void CpuTrunk::Impl::build(const Weights<float>& w) {
    require_valid(cfg);
    H = dimension(cfg.hidden_size);
    I = dimension(cfg.intermediate_size);
    taps = dimension(cfg.conv_kernel);
    conv_dim = dimension(cfg.dn_conv_dim());
    key_dim = dimension(cfg.dn_key_dim());
    value_dim = dimension(cfg.dn_value_dim());
    hv = dimension(cfg.dn_value_heads);
    hk = dimension(cfg.dn_key_heads);
    dk = dimension(cfg.dn_key_head_dim);
    dv = dimension(cfg.dn_value_head_dim);
    heads = dimension(cfg.attn_heads);
    kv_heads = dimension(cfg.attn_kv_heads);
    hd = dimension(cfg.attn_head_dim);
    rot_half = dimension(cfg.attn_rotary_dims()) / 2;
    if (H % 8 != 0 || dk % 8 != 0 || dv % 8 != 0 || hd % 32 != 0 || conv_dim % 8 != 0) {
        throw std::invalid_argument("the CPU backend needs hidden size and DeltaNet head dims multiples of 8 and attention head dim a multiple of 32");
    }
    if (opt.chunk_tokens == 0 || opt.step_tokens < opt.chunk_tokens || opt.delta_chunk > 64) {
        throw std::invalid_argument("TrunkOptions: chunk_tokens must be in [1, step_tokens] and delta_chunk at most 64");
    }
    if (hd > 1024) throw std::invalid_argument("attention head dim above 1024 is not supported");
    qz_width = std::max(conv_dim + value_dim, 2 * heads * hd + 2 * kv_heads * hd);

    const WeightKind k_mlp = kind_for(opt.q8.mlp);
    const WeightKind k_attn = kind_for(opt.q8.attention);
    const WeightKind k_dn_in = kind_for(opt.q8.deltanet_in);
    const WeightKind k_16 = opt.profile == Profile::F32 ? WeightKind::F32 : WeightKind::BF16;

    layers.resize(dimension(cfg.num_layers));
    for (std::size_t i = 0; i < layers.size(); ++i) {
        Layer& L = layers[i];
        L.attention = cfg.is_full_attention(i);
        const Weights<float> lw = w.scope("layers." + std::to_string(i));
        L.input_norm1 = one_plus(lw["input_layernorm.weight"]);
        L.post_norm1 = one_plus(lw["post_attention_layernorm.weight"]);
        const Weights<float> mlp_w = lw.scope("mlp");
        L.gate_up = pack_weight_swiglu(mlp_w["gate_proj.weight"].data(), mlp_w["up_proj.weight"].data(), H, I, H, k_mlp, isa, pool);
        L.down = pack_weight(mlp_w["down_proj.weight"].data(), I, H, I, k_mlp, isa, pool);
        weight_bytes += L.gate_up.bytes() + L.down.bytes();
        if (L.attention) {
            const Weights<float> a = lw.scope("self_attn");
            const std::vector<float> qkv = concat({&a["q_proj.weight"], &a["k_proj.weight"], &a["v_proj.weight"]});
            const std::size_t n = 2 * heads * hd + 2 * kv_heads * hd;
            L.at.qkv = pack_weight(qkv.data(), H, n, H, k_attn, isa, pool);
            L.at.o = pack_weight(a["o_proj.weight"].data(), heads * hd, H, heads * hd, k_attn, isa, pool);
            L.at.q_norm1 = one_plus(a["q_norm.weight"]);
            L.at.k_norm1 = one_plus(a["k_norm.weight"]);
            weight_bytes += L.at.qkv.bytes() + L.at.o.bytes();
            ++n_attn;
        } else {
            const Weights<float> d = lw.scope("linear_attn");
            const std::vector<float> qz_rows = concat({&d["in_proj_qkv.weight"], &d["in_proj_z.weight"]});
            L.dn.in_qz = pack_weight(qz_rows.data(), H, conv_dim + value_dim, H, k_dn_in, isa, pool);
            const std::vector<float> ab_rows = concat({&d["in_proj_a.weight"], &d["in_proj_b.weight"]});
            L.dn.ab = pack_weight(ab_rows.data(), H, 2 * hv, H, k_16, isa, pool);
            L.dn.out = pack_weight(d["out_proj.weight"].data(), value_dim, H, value_dim, k_16, isa, pool);
            const BasicTensor<float>& conv = d["conv1d.weight"];
            L.dn.conv_wt.resize(taps * conv_dim);
            for (std::size_t c = 0; c < conv_dim; ++c)
                for (std::size_t j = 0; j < taps; ++j) L.dn.conv_wt[j * conv_dim + c] = conv[c * taps + j];
            L.dn.a_log = copy_of(d["A_log"]);
            L.dn.dt_bias = copy_of(d["dt_bias"]);
            L.dn.norm_w = copy_of(d["norm.weight"]);
            weight_bytes += L.dn.in_qz.bytes() + L.dn.ab.bytes() + L.dn.out.bytes();
            ++n_dn;
        }
    }
    final_norm1 = one_plus(w["norm.weight"]);

    const std::size_t S = opt.step_tokens;
    x.resize(S * H);
    h.resize(S * H);
    qz.resize(S * qz_width);
    ab.resize(S * 2 * hv);
    conv_out.resize(S * conv_dim);
    gate_g.resize(S * hv);
    beta_b.resize(S * hv);
    core.resize(S * std::max(value_dim, heads * hd));
    qbuf.resize(S * heads * hd);
    attended.resize(S * heads * hd);
    act.resize(S * I);
    cos_tab.resize(S * std::max<std::size_t>(rot_half, 1));
    sin_tab.resize(S * std::max<std::size_t>(rot_half, 1));
}

std::uint64_t CpuTrunk::Impl::state_bytes(std::size_t tokens) const {
    const std::uint64_t dn = n_dn * (static_cast<std::uint64_t>(hv) * dk * dv + (taps - 1) * conv_dim) * 4;
    const std::uint64_t cap = (tokens + attention_key_tile - 1) / attention_key_tile * attention_key_tile;
    return dn + n_attn * 2ull * kv_heads * cap * hd * 4;
}

// ---- layers -----------------------------------------------------------------------------------------------------

void CpuTrunk::Impl::mlp(const Layer& L, std::size_t rows) {
    {
        Timer t(times.other_ms);
        rmsnorm_rows(x.data(), H, h.data(), H, rows, H, L.post_norm1.data(), cfg.rms_norm_eps, pool);
    }
    Timer t(times.gemm_ms);
    acts_h.reset();
    PackedActs& a = acts_h.pack(h.data(), H, rows, H, L.gate_up.variant(), pool);
    gemm_packed(L.gate_up, a, act.data(), I, false, pool);
    acts_c.reset();
    PackedActs& b = acts_c.pack(act.data(), I, rows, I, L.down.variant(), pool);
    gemm_packed(L.down, b, x.data(), H, true, pool);
}

void CpuTrunk::Impl::delta_layer(const Layer& L, std::size_t index, std::vector<RequestState>& states,
                                 const std::vector<Segment>& segs, std::size_t rows) {
    const DeltaLayer& D = L.dn;
    std::size_t dn_index = 0;
    for (std::size_t i = 0; i < index; ++i) dn_index += layers[i].attention ? 0u : 1u;
    const std::size_t ld_qz = qz_width;
    {
        Timer t(times.other_ms);
        rmsnorm_rows(x.data(), H, h.data(), H, rows, H, L.input_norm1.data(), cfg.rms_norm_eps, pool);
    }
    {
        Timer t(times.gemm_ms);
        acts_h.reset();
        PackedActs& a = acts_h.pack(h.data(), H, rows, H, D.in_qz.variant(), pool);
        gemm_packed(D.in_qz, a, qz.data(), ld_qz, false, pool);
        PackedActs& b = acts_h.pack(h.data(), H, rows, H, D.ab.variant(), pool);
        gemm_packed(D.ab, b, ab.data(), 2 * hv, false, pool);
    }
    Timer t(times.other_ms);
    // Gates: g = -exp(A_log) softplus(a + dt_bias) (log decay), beta = sigmoid(b), in double.
    pool.parallel_for((rows + 63) / 64, [&](std::size_t blk, unsigned) {
        const std::size_t end = std::min(rows, (blk + 1) * 64);
        for (std::size_t r = blk * 64; r < end; ++r) {
            for (std::size_t j = 0; j < hv; ++j) {
                const double a = static_cast<double>(ab[r * 2 * hv + j]) + static_cast<double>(D.dt_bias[j]);
                const double softplus = std::max(a, 0.0) + std::log1p(std::exp(-std::fabs(a)));
                gate_g[r * hv + j] = static_cast<float>(-std::exp(static_cast<double>(D.a_log[j])) * softplus);
                const double b = static_cast<double>(ab[r * 2 * hv + hv + j]);
                const double sp = std::max(-b, 0.0) + std::log1p(std::exp(-std::fabs(b)));
                beta_b[r * hv + j] = static_cast<float>(std::exp(-sp));
            }
        }
    });
    for (const Segment& s : segs) {
        RequestState& rs = states[s.request];
        causal_conv_silu(qz.data() + s.row0 * ld_qz, ld_qz, conv_out.data() + s.row0 * conv_dim, conv_dim, s.rows, conv_dim,
                         D.conv_wt.data(), taps, rs.conv_tail[dn_index].data(), pool);
    }
    l2norm_heads(conv_out.data(), conv_dim, rows, hk, dk, 1e-6, pool);
    l2norm_heads(conv_out.data() + key_dim, conv_dim, rows, hk, dk, 1e-6, pool);

    std::vector<DeltaSegment> dsegs;
    dsegs.reserve(segs.size());
    for (const Segment& s : segs) {
        DeltaSegment d;
        d.rows = s.rows;
        d.q = conv_out.data() + s.row0 * conv_dim;
        d.k = d.q + key_dim;
        d.v = d.q + 2 * key_dim;
        d.g = gate_g.data() + s.row0 * hv;
        d.beta = beta_b.data() + s.row0 * hv;
        d.state = states[s.request].dn_state[dn_index].data();
        d.out = core.data() + s.row0 * value_dim;
        dsegs.push_back(d);
    }
    const DeltaGeometry geo{hk, hv, dk, dv};
    const float scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(dk)));
    {
        Timer td(times.delta_ms);
        if (opt.delta_chunk == 0) {
            delta_scan_sequential(geo, dsegs, conv_dim, value_dim, scale, pool);
        } else {
            delta_scan_chunked(geo, dsegs, conv_dim, value_dim, scale, opt.delta_chunk, pool);
        }
    }
    gated_rmsnorm_heads(core.data(), value_dim, qz.data() + conv_dim, ld_qz, core.data(), value_dim, rows, hv, dv,
                        D.norm_w.data(), cfg.rms_norm_eps, pool);
    Timer tg(times.gemm_ms);
    acts_c.reset();
    PackedActs& c = acts_c.pack(core.data(), value_dim, rows, value_dim, D.out.variant(), pool);
    gemm_packed(D.out, c, x.data(), H, true, pool);
}

void CpuTrunk::Impl::attention_layer(const Layer& L, std::size_t index, std::vector<RequestState>& states,
                                     const std::vector<Segment>& segs, std::size_t rows) {
    const AttentionLayer& A = L.at;
    std::size_t attn_index = 0;
    for (std::size_t i = 0; i < index; ++i) attn_index += layers[i].attention ? 1u : 0u;
    const std::size_t ld_qz = qz_width;
    const std::size_t q_width = heads * hd;
    {
        Timer t(times.other_ms);
        rmsnorm_rows(x.data(), H, h.data(), H, rows, H, L.input_norm1.data(), cfg.rms_norm_eps, pool);
    }
    {
        Timer t(times.gemm_ms);
        acts_h.reset();
        PackedActs& a = acts_h.pack(h.data(), H, rows, H, A.qkv.variant(), pool);
        gemm_packed(A.qkv, a, qz.data(), ld_qz, false, pool);
    }
    {
        Timer t(times.other_ms);
        const float q_scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(hd)));
        // Row blocks of 32 positions: a K panel holds up to 32 positions, so blocks never share a cache line.
        struct Block {
            std::size_t seg, first, count;
        };
        std::vector<Block> blocks;
        for (std::size_t si = 0; si < segs.size(); ++si)
            for (std::size_t r = 0; r < segs[si].rows; r += 32) blocks.push_back({si, r, std::min<std::size_t>(32, segs[si].rows - r)});
        pool.parallel_for(blocks.size(), [&](std::size_t bi, unsigned) {
            const Block& b = blocks[bi];
            const Segment& s = segs[b.seg];
            KvCache& cache = *states[s.request].kv[attn_index];
            alignas(32) float ktmp[1024];
            for (std::size_t r = b.first; r < b.first + b.count; ++r) {
                const std::size_t row = s.row0 + r;
                const std::size_t pos = s.position + r;
                float* qz_row = qz.data() + row * ld_qz;
                const float* cs = cos_tab.data() + row * rot_half;
                const float* sn = sin_tab.data() + row * rot_half;
                float* qrow = qbuf.data() + row * q_width;
                for (std::size_t hh = 0; hh < heads; ++hh) {
                    float* dst = qrow + hh * hd;
                    rmsnorm_rows(qz_row + hh * 2 * hd, 2 * hd, dst, hd, 1, hd, A.q_norm1.data(), cfg.rms_norm_eps, pool);
                    for (std::size_t i = 0; i < rot_half; ++i) {
                        const float a = dst[i], bb = dst[rot_half + i];
                        dst[i] = a * cs[i] - bb * sn[i];
                        dst[rot_half + i] = bb * cs[i] + a * sn[i];
                    }
                    for (std::size_t d = 0; d < hd; ++d) dst[d] *= q_scale;
                }
                const float* krow = qz_row + 2 * q_width;
                const float* vrow = krow + kv_heads * hd;
                for (std::size_t g = 0; g < kv_heads; ++g) {
                    rmsnorm_rows(krow + g * hd, hd, ktmp, hd, 1, hd, A.k_norm1.data(), cfg.rms_norm_eps, pool);
                    for (std::size_t i = 0; i < rot_half; ++i) {
                        const float a = ktmp[i], bb = ktmp[rot_half + i];
                        ktmp[i] = a * cs[i] - bb * sn[i];
                        ktmp[rot_half + i] = bb * cs[i] + a * sn[i];
                    }
                    cache.append(pos, g, ktmp, vrow + g * hd);
                }
            }
        });
    }
    {
        Timer t(times.attention_ms);
        std::vector<AttentionSegment> asegs;
        for (const Segment& s : segs) {
            AttentionSegment a;
            a.cache = states[s.request].kv[attn_index].get();
            a.start = s.position;
            a.rows = s.rows;
            a.q = qbuf.data() + s.row0 * q_width;
            a.gate = qz.data() + s.row0 * ld_qz + hd;
            a.out = attended.data() + s.row0 * q_width;
            asegs.push_back(a);
        }
        attention_forward({heads, kv_heads, hd}, isa, asegs, q_width, ld_qz, 2 * hd, q_width, pool);
    }
    Timer t(times.gemm_ms);
    acts_c.reset();
    PackedActs& c = acts_c.pack(attended.data(), q_width, rows, q_width, A.o.variant(), pool);
    gemm_packed(A.o, c, x.data(), H, true, pool);
}

void CpuTrunk::Impl::run_step(std::span<const std::span<const std::uint32_t>> requests, std::vector<RequestState>& states,
                              const std::vector<Segment>& segs, std::size_t rows, std::vector<std::vector<float>>& hidden,
                              const LayerObserver* observer) {
    {
        Timer t(times.other_ms);
        // Embeddings and RoPE tables of every row.
        std::vector<std::size_t> row_segment(rows);
        for (std::size_t si = 0; si < segs.size(); ++si)
            for (std::size_t r = 0; r < segs[si].rows; ++r) row_segment[segs[si].row0 + r] = si;
        const std::size_t half = rot_half;
        std::vector<double> inv_freq(half);
        for (std::size_t i = 0; i < half; ++i)
            inv_freq[i] = std::pow(cfg.rope_theta, -static_cast<double>(2 * i) / static_cast<double>(2 * half));
        pool.parallel_for((rows + 31) / 32, [&](std::size_t blk, unsigned) {
            const std::size_t end = std::min(rows, (blk + 1) * 32);
            for (std::size_t row = blk * 32; row < end; ++row) {
                const Segment& s = segs[row_segment[row]];
                const std::size_t r = row - s.row0;
                embed(requests[s.request][s.position + r], x.data() + row * H);
                const double pos = static_cast<double>(s.position + r);
                for (std::size_t i = 0; i < half; ++i) {
                    const double angle = pos * inv_freq[i];
                    cos_tab[row * half + i] = static_cast<float>(std::cos(angle));
                    sin_tab[row * half + i] = static_cast<float>(std::sin(angle));
                }
            }
        });
    }
    for (std::size_t li = 0; li < layers.size(); ++li) {
        const Layer& L = layers[li];
        if (L.attention) {
            attention_layer(L, li, states, segs, rows);
        } else {
            delta_layer(L, li, states, segs, rows);
        }
        mlp(L, rows);
        if (observer != nullptr) {
            for (const Segment& s : segs) (*observer)(s.request, li, s.position, s.rows, x.data() + s.row0 * H);
        }
    }
    Timer t(times.other_ms);
    for (const Segment& s : segs) {
        rmsnorm_rows(x.data() + s.row0 * H, H, hidden[s.request].data() + s.position * H, H, s.rows, H, final_norm1.data(),
                     cfg.rms_norm_eps, pool);
    }
}

// ---- public interface -------------------------------------------------------------------------------------------

CpuTrunk::CpuTrunk(const ModelConfig& config, const Weights<float>& weights, EmbeddingFn embed, const TrunkOptions& options)
    : impl_(std::make_unique<Impl>(config, options, std::move(embed))) {
    impl_->build(weights);
}

CpuTrunk::~CpuTrunk() = default;

const ModelConfig& CpuTrunk::config() const { return impl_->cfg; }
const TrunkOptions& CpuTrunk::options() const { return impl_->opt; }
Isa CpuTrunk::isa() const { return impl_->isa; }
unsigned CpuTrunk::threads() const { return impl_->pool.size(); }
ThreadPool& CpuTrunk::pool() { return impl_->pool; }
std::uint64_t CpuTrunk::weight_bytes() const { return impl_->weight_bytes; }
const TrunkTimes& CpuTrunk::last_times() const { return impl_->times; }

void CpuTrunk::forward(std::span<const std::span<const std::uint32_t>> requests, std::vector<std::vector<float>>& hidden,
                       const LayerObserver* observer) {
    Impl& m = *impl_;
    for (const auto& r : requests) {
        if (r.empty()) throw std::invalid_argument("a request needs at least one token");
        for (const std::uint32_t t : r)
            if (t >= m.cfg.vocab_size) throw std::invalid_argument("a token id is outside the vocabulary");
    }
    m.times = {};
    const auto start = Clock::now();
    hidden.assign(requests.size(), {});
    std::vector<RequestState> states(requests.size());
    std::vector<std::size_t> done(requests.size(), 0);
    std::vector<std::size_t> active;
    std::size_t next_admit = 0;
    std::uint64_t used = 0;

    for (;;) {
        // Admit requests while their state fits the budget (at least one is always admitted).
        while (next_admit < requests.size()) {
            const std::uint64_t need = m.state_bytes(requests[next_admit].size());
            if (!active.empty() && used + need > m.opt.state_budget_bytes) break;
            RequestState& rs = states[next_admit];
            rs.dn_state.resize(m.n_dn);
            rs.conv_tail.resize(m.n_dn);
            for (std::size_t i = 0; i < m.n_dn; ++i) {
                rs.dn_state[i].resize(m.hv * m.dk * m.dv);
                rs.dn_state[i].zero();
                rs.conv_tail[i].resize((m.taps - 1) * m.conv_dim);
                rs.conv_tail[i].zero();
            }
            for (std::size_t i = 0; i < m.n_attn; ++i) {
                rs.kv.push_back(std::make_unique<KvCache>(m.kv_heads, m.hd, requests[next_admit].size(), m.isa));
            }
            hidden[next_admit].assign(requests[next_admit].size() * m.H, 0.0f);
            used += need;
            active.push_back(next_admit++);
        }
        if (active.empty()) break;
        std::vector<Segment> segs;
        std::size_t rows = 0;
        for (const std::size_t r : active) {
            const std::size_t remaining = requests[r].size() - done[r];
            const std::size_t c = std::min(m.opt.chunk_tokens, remaining);
            if (rows != 0 && rows + c > m.opt.step_tokens) continue;
            segs.push_back({r, done[r], c, rows});
            rows += c;
        }
        m.run_step(requests, states, segs, rows, hidden, observer);
        m.times.tokens += rows;
        ++m.times.steps;
        for (const Segment& s : segs) done[s.request] += s.rows;
        for (std::size_t i = 0; i < active.size();) {
            const std::size_t r = active[i];
            if (done[r] == requests[r].size()) {
                used -= m.state_bytes(requests[r].size());
                states[r] = RequestState{};
                active.erase(active.begin() + static_cast<std::ptrdiff_t>(i));
            } else {
                ++i;
            }
        }
    }
    m.times.total_ms = ms_since(start);
    m.times.other_ms = m.times.total_ms - m.times.gemm_ms - m.times.delta_ms - m.times.attention_ms;
}

} // namespace sepal::cpu
