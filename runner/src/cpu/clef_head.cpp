#include "sepal/cpu/clef_head.hpp"

#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "sepal/cpu/gemm.hpp"
#include "vmath.hpp"

namespace sepal::cpu {

namespace {

constexpr double ln_eps = 1e-5;
constexpr double cosine_eps = 1e-8;
constexpr double normalize_eps = 1e-12;

struct Linear {
    PackedWeight w;
    std::vector<float> bias;  // empty: none
    std::size_t out = 0, in = 0;
};

struct Mha {
    Linear q, k, v, o;
};

struct Ln {
    const float* gamma;
    const float* beta;
};

using Vec = std::vector<float>;

} // namespace

struct CpuClefHead::Impl {
    const ClefHeadWeights& wt;
    ThreadPool& pool;
    Isa isa;
    ClefHeadConfig cfg;
    std::size_t H, W, FF, heads, hd;

    std::map<std::string, Linear, std::less<>> linears;
    std::map<std::string, Mha, std::less<>> mhas;

    Impl(const ClefHeadWeights& w, ThreadPool& p, Isa i) : wt(w), pool(p), isa(i), cfg(w.config) {
        H = cfg.hidden_size;
        W = cfg.width;
        FF = cfg.feedforward;
        heads = cfg.heads;
        hd = W / heads;
        if (W % 8 != 0 || H % 8 != 0) throw std::invalid_argument("the CPU head needs hidden size and width multiples of 8");
        for (const char* name : {"memory_projection", "question_projection", "option_question_projection", "global_projection",
                                 "option_context_projection", "option_lexical_projection"}) {
            add_linear(name, W, H, false);
        }
        for (std::uint32_t l = 0; l < cfg.routing_layers; ++l) {
            const std::string p0 = "evidence_layers." + std::to_string(l);
            add_mha(p0 + ".attention");
            add_linear(p0 + ".feedforward.0", FF, W, true);
            add_linear(p0 + ".feedforward.3", W, FF, true);
        }
        for (std::uint32_t l = 0; l < cfg.layers; ++l) {
            const std::string p0 = "layers." + std::to_string(l);
            add_mha(p0 + ".self_attn");
            add_mha(p0 + ".multihead_attn");
            add_linear(p0 + ".linear1", FF, W, true);
            add_linear(p0 + ".linear2", W, FF, true);
        }
        add_linear("residual_scorer.0", W, 4 * W, true);
        add_linear("residual_scorer.3", 1, W, true);
    }

    Linear make_linear(const float* weight, std::size_t out, std::size_t in, const float* bias) {
        Linear l;
        l.out = out;
        l.in = in;
        l.w = pack_weight(weight, in, out, in, WeightKind::F32, isa, pool);
        if (bias != nullptr) l.bias.assign(bias, bias + out);
        return l;
    }
    void add_linear(const std::string& name, std::size_t out, std::size_t in, bool bias) {
        linears.emplace(name, make_linear(wt.at(name + ".weight").data(), out, in, bias ? wt.at(name + ".bias").data() : nullptr));
    }
    void add_mha(const std::string& name) {
        const float* in_w = wt.at(name + ".in_proj_weight").data();
        const float* in_b = wt.at(name + ".in_proj_bias").data();
        Mha m;
        m.q = make_linear(in_w, W, W, in_b);
        m.k = make_linear(in_w + W * W, W, W, in_b + W);
        m.v = make_linear(in_w + 2 * W * W, W, W, in_b + 2 * W);
        m.o = make_linear(wt.at(name + ".out_proj.weight").data(), W, W, wt.at(name + ".out_proj.bias").data());
        mhas.emplace(name, std::move(m));
    }

    // y [rows, out] = x [rows, in] W^T + b
    Vec apply(const Linear& l, const float* x, std::size_t rows) {
        Vec y(rows * l.out);
        PackedActs a;
        pack_acts(a, x, l.in, rows, l.in, l.w.variant(), pool);
        gemm_packed(l.w, a, y.data(), l.out, false, pool);
        if (!l.bias.empty()) {
            pool.parallel_for((rows + 31) / 32, [&](std::size_t blk, unsigned) {
                const std::size_t end = std::min(rows, (blk + 1) * 32);
                for (std::size_t r = blk * 32; r < end; ++r)
                    for (std::size_t c = 0; c < l.out; ++c) y[r * l.out + c] += l.bias[c];
            });
        }
        return y;
    }

    Vec layer_norm(const float* x, std::size_t rows, std::size_t d, const Ln& ln) {
        Vec y(rows * d);
        pool.parallel_for((rows + 15) / 16, [&](std::size_t blk, unsigned) {
            const std::size_t end = std::min(rows, (blk + 1) * 16);
            for (std::size_t r = blk * 16; r < end; ++r) {
                const float* xr = x + r * d;
                double mean = 0;
                for (std::size_t i = 0; i < d; ++i) mean += xr[i];
                mean /= static_cast<double>(d);
                double var = 0;
                for (std::size_t i = 0; i < d; ++i) var += (xr[i] - mean) * (xr[i] - mean);
                var /= static_cast<double>(d);
                const double inv = 1.0 / std::sqrt(var + ln_eps);
                for (std::size_t i = 0; i < d; ++i) {
                    y[r * d + i] = static_cast<float>((xr[i] - mean) * inv * static_cast<double>(ln.gamma[i]) + static_cast<double>(ln.beta[i]));
                }
            }
        });
        return y;
    }
    Vec layer_norm_named(const std::string& name, const Vec& x, std::size_t rows, std::size_t d) {
        return layer_norm(x.data(), rows, d, {wt.at(name + ".weight").data(), wt.at(name + ".bias").data()});
    }

    static float gelu(float x) {
        return static_cast<float>(0.5 * static_cast<double>(x) * (1.0 + std::erf(static_cast<double>(x) * 0.70710678118654752440)));
    }

    Vec feed_forward(const std::string& first, const std::string& second, const Vec& x, std::size_t rows) {
        Vec hidden = apply(linears.at(first), x.data(), rows);
        pool.parallel_for((hidden.size() + 4095) / 4096, [&](std::size_t blk, unsigned) {
            const std::size_t end = std::min(hidden.size(), (blk + 1) * 4096);
            for (std::size_t i = blk * 4096; i < end; ++i) hidden[i] = gelu(hidden[i]);
        });
        return apply(linears.at(second), hidden.data(), rows);
    }

    // Softmax of every row of [rows, cols] in place.
    void softmax_rows(float* s, std::size_t rows, std::size_t cols, float scale) {
        pool.parallel_for(rows, [&](std::size_t r, unsigned) {
            float* row = s + r * cols;
            float peak = -3.0e38f;
            for (std::size_t j = 0; j < cols; ++j) {
                row[j] *= scale;
                peak = std::max(peak, row[j]);
            }
            const __m256 pk = _mm256_set1_ps(peak);
            __m256 vsum = _mm256_setzero_ps();
            std::size_t j = 0;
            for (; j + 8 <= cols; j += 8) {
                const __m256 e = exp_ps(_mm256_sub_ps(_mm256_loadu_ps(row + j), pk));
                _mm256_storeu_ps(row + j, e);
                vsum = _mm256_add_ps(vsum, e);
            }
            float sum = hsum_ps(vsum);
            for (; j < cols; ++j) {
                row[j] = exp_scalar(row[j] - peak);
                sum += row[j];
            }
            const __m256 inv = _mm256_set1_ps(1.0f / sum);
            for (j = 0; j + 8 <= cols; j += 8) _mm256_storeu_ps(row + j, _mm256_mul_ps(_mm256_loadu_ps(row + j), inv));
            const float invs = 1.0f / sum;
            for (; j < cols; ++j) row[j] *= invs;
        });
    }

    // nn.MultiheadAttention without masks: queries [nq, W] attend over keys/values built from `source` [nk, W].
    Vec multihead(const Mha& m, const float* query, std::size_t nq, const float* source, std::size_t nk) {
        const Vec q = apply(m.q, query, nq);
        const Vec k = apply(m.k, source, nk);
        const Vec v = apply(m.v, source, nk);
        Vec context(nq * W);
        Vec scores(nq * nk);
        Vec vt(hd * nk);
        const float scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(hd)));
        for (std::size_t h = 0; h < heads; ++h) {
            {
                const PackedWeight kw = pack_weight(k.data() + h * hd, W, nk, hd, WeightKind::F32, isa, pool);
                PackedActs qa;
                pack_acts(qa, q.data() + h * hd, W, nq, hd, kw.variant(), pool);
                gemm_packed(kw, qa, scores.data(), nk, false, pool);
            }
            softmax_rows(scores.data(), nq, nk, scale);
            pool.parallel_for(hd, [&](std::size_t d, unsigned) {
                for (std::size_t j = 0; j < nk; ++j) vt[d * nk + j] = v[j * W + h * hd + d];
            });
            const PackedWeight vw = pack_weight(vt.data(), nk, hd, nk, WeightKind::F32, isa, pool);
            PackedActs pa;
            pack_acts(pa, scores.data(), nk, nq, nk, vw.variant(), pool);
            gemm_packed(vw, pa, context.data() + h * hd, W, false, pool);
        }
        return apply(m.o, context.data(), nq);
    }

    static void add_into(Vec& x, const Vec& y) {
        for (std::size_t i = 0; i < x.size(); ++i) x[i] += y[i];
    }

    RequestLogits run(std::span<const float> hidden, const EncodedRequest& request,
                      const std::function<void(std::uint32_t, float*)>& embed);
};

RequestLogits CpuClefHead::Impl::run(std::span<const float> hidden, const EncodedRequest& request,
                                     const std::function<void(std::uint32_t, float*)>& embed) {
    const std::size_t L = request.token_ids.size();
    check_request(request, ~std::uint64_t{0});
    if (hidden.size() != L * H) throw std::invalid_argument("clef head: hidden does not have L x hidden_size values");

    const Vec hn = layer_norm(hidden.data(), L, H, {wt.at("hidden_norm.weight").data(), wt.at("hidden_norm.bias").data()});
    const Vec memory = apply(linears.at("memory_projection"), hn.data(), L);
    const float* global = hn.data() + (L - 1) * H;

    const std::size_t Q = request.questions.size();
    Vec qvec(Q * H);
    std::vector<std::size_t> first_option(Q + 1, 0);
    for (std::size_t q = 0; q < Q; ++q) first_option[q + 1] = first_option[q] + request.questions[q].options.size();
    const std::size_t O = first_option[Q];
    Vec option_ctx(O * H), option_lex(O * H);

    const auto mean_rows = [&](const TokenSpan& span, float* out) {
        std::vector<double> acc(H, 0.0);
        for (std::size_t t = span.begin; t < span.end; ++t)
            for (std::size_t i = 0; i < H; ++i) acc[i] += hn[t * H + i];
        const double count = static_cast<double>(span.end - span.begin);
        for (std::size_t i = 0; i < H; ++i) out[i] = static_cast<float>(acc[i] / count);
    };
    for (std::size_t q = 0; q < Q; ++q) {
        const EncodedQuestion& question = request.questions[q];
        mean_rows(question.instruction, qvec.data() + q * H);
        for (std::size_t k = 0; k < question.options.size(); ++k) {
            const std::size_t o = first_option[q] + k;
            const TokenSpan& span = question.options[k];
            mean_rows(span, option_ctx.data() + o * H);
            std::vector<double> lex(H, 0.0);
            std::vector<float> row(H);
            for (std::uint32_t t = span.begin; t < span.end; ++t) {
                embed(request.token_ids[t], row.data());
                for (std::size_t i = 0; i < H; ++i) lex[i] += row[i];
            }
            for (std::size_t i = 0; i < H; ++i) option_lex[o * H + i] = static_cast<float>(lex[i] / static_cast<double>(span.end - span.begin));
        }
    }

    Vec options = apply(linears.at("option_context_projection"), option_ctx.data(), O);
    {
        const Vec lexical = apply(linears.at("option_lexical_projection"), option_lex.data(), O);
        const Vec question_part = apply(linears.at("option_question_projection"), qvec.data(), Q);
        for (std::size_t q = 0; q < Q; ++q)
            for (std::size_t o = first_option[q]; o < first_option[q + 1]; ++o)
                for (std::size_t i = 0; i < W; ++i) options[o * W + i] += lexical[o * W + i] + question_part[q * W + i];
    }

    for (std::uint32_t l = 0; l < cfg.routing_layers; ++l) {
        const std::string name = "evidence_layers." + std::to_string(l);
        const Vec nq = layer_norm_named(name + ".query_norm", options, O, W);
        const Vec nm = layer_norm_named(name + ".memory_norm", memory, L, W);
        add_into(options, multihead(mhas.at(name + ".attention"), nq.data(), O, nm.data(), L));
        const Vec nf = layer_norm_named(name + ".feedforward_norm", options, O, W);
        add_into(options, feed_forward(name + ".feedforward.0", name + ".feedforward.3", nf, O));
    }

    const Vec base = apply(linears.at("question_projection"), qvec.data(), Q);
    Vec summaries(Q * W, 0.0f);
    for (std::size_t q = 0; q < Q; ++q) {
        const std::size_t n = first_option[q + 1] - first_option[q];
        std::vector<double> weight(n);
        double top = -INFINITY;
        for (std::size_t k = 0; k < n; ++k) {
            double s = 0;
            for (std::size_t i = 0; i < W; ++i) s += static_cast<double>(options[(first_option[q] + k) * W + i]) * base[q * W + i];
            weight[k] = s / std::sqrt(static_cast<double>(W));
            top = std::max(top, weight[k]);
        }
        double sum = 0;
        for (double& v : weight) {
            v = std::exp(v - top);
            sum += v;
        }
        for (std::size_t i = 0; i < W; ++i) {
            double acc = 0;
            for (std::size_t k = 0; k < n; ++k) acc += (weight[k] / sum) * options[(first_option[q] + k) * W + i];
            summaries[q * W + i] = static_cast<float>(acc);
        }
    }
    const Vec summary_normed = layer_norm_named("option_summary_norm", summaries, Q, W);
    const Vec global_part = apply(linears.at("global_projection"), global, 1);
    const float* type_embedding = wt.at("type_embedding.weight").data();
    Vec fields(Q * W);
    for (std::size_t q = 0; q < Q; ++q) {
        const std::size_t type = static_cast<std::size_t>(request.questions[q].type);
        for (std::size_t i = 0; i < W; ++i) fields[q * W + i] = base[q * W + i] + summary_normed[q * W + i] + global_part[i] + type_embedding[type * W + i];
    }

    for (std::uint32_t l = 0; l < cfg.layers; ++l) {
        const std::string name = "layers." + std::to_string(l);
        const Vec n1 = layer_norm_named(name + ".norm1", fields, Q, W);
        add_into(fields, multihead(mhas.at(name + ".self_attn"), n1.data(), Q, n1.data(), Q));
        const Vec n2 = layer_norm_named(name + ".norm2", fields, Q, W);
        add_into(fields, multihead(mhas.at(name + ".multihead_attn"), n2.data(), Q, memory.data(), L));
        const Vec n3 = layer_norm_named(name + ".norm3", fields, Q, W);
        add_into(fields, feed_forward(name + ".linear1", name + ".linear2", n3, Q));
    }
    fields = layer_norm_named("field_norm", fields, Q, W);
    const Vec opt_normed = layer_norm_named("option_norm", options, O, W);

    const double log100 = std::log(100.0);
    const double prior_scale = std::exp(std::min(static_cast<double>(wt.at("prior_logit_scale")[0]), log100));
    const double joint_scale = std::exp(std::min(static_cast<double>(wt.at("joint_logit_scale")[0]), log100));
    const double gate = 1.0 / (1.0 + std::exp(-static_cast<double>(wt.at("residual_gate")[0])));

    // Residual scorer features for every option: [f, o, f*o, |f-o|].
    Vec features(O * 4 * W);
    std::vector<double> priors(O), cosines(O);
    const auto norm_of = [](const float* x, std::size_t d) {
        double s = 0;
        for (std::size_t i = 0; i < d; ++i) s += static_cast<double>(x[i]) * x[i];
        return std::sqrt(s);
    };
    for (std::size_t q = 0; q < Q; ++q) {
        std::vector<double> anchor(H);
        double an = 0;
        for (std::size_t i = 0; i < H; ++i) {
            anchor[i] = static_cast<double>(qvec[q * H + i]) + global[i];
            an += anchor[i] * anchor[i];
        }
        an = std::max(std::sqrt(an), normalize_eps);
        for (std::size_t o = first_option[q]; o < first_option[q + 1]; ++o) {
            const double ln = std::max(norm_of(option_lex.data() + o * H, H), normalize_eps);
            double dot = 0;
            for (std::size_t i = 0; i < H; ++i) dot += (option_lex[o * H + i] / ln) * (anchor[i] / an);
            priors[o] = prior_scale * dot;
            const float* f = fields.data() + q * W;
            const float* op = opt_normed.data() + o * W;
            const double nf = std::max(norm_of(f, W), cosine_eps), no = std::max(norm_of(op, W), cosine_eps);
            double cs = 0;
            for (std::size_t i = 0; i < W; ++i) cs += (f[i] / nf) * (op[i] / no);
            cosines[o] = cs;
            float* feat = features.data() + o * 4 * W;
            for (std::size_t i = 0; i < W; ++i) {
                feat[i] = f[i];
                feat[W + i] = op[i];
                feat[2 * W + i] = f[i] * op[i];
                feat[3 * W + i] = std::fabs(f[i] - op[i]);
            }
        }
    }
    Vec hid = apply(linears.at("residual_scorer.0"), features.data(), O);
    for (float& v : hid) v = gelu(v);
    const Vec residual = apply(linears.at("residual_scorer.3"), hid.data(), O);

    RequestLogits result;
    for (std::size_t q = 0; q < Q; ++q) {
        std::vector<float> logits;
        for (std::size_t o = first_option[q]; o < first_option[q + 1]; ++o) {
            logits.push_back(static_cast<float>(priors[o] + gate * (joint_scale * cosines[o] + static_cast<double>(residual[o]))));
        }
        result.logits.push_back(std::move(logits));
    }
    return result;
}

CpuClefHead::CpuClefHead(const ClefHeadWeights& weights, ThreadPool& pool, Isa isa) : impl_(std::make_unique<Impl>(weights, pool, isa)) {}
CpuClefHead::~CpuClefHead() = default;

RequestLogits CpuClefHead::run(std::span<const float> hidden, const EncodedRequest& request,
                               const std::function<void(std::uint32_t, float*)>& embed) {
    return impl_->run(hidden, request, embed);
}

} // namespace sepal::cpu
