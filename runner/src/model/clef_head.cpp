// The Clef JointSchemaHead, reference implementation (see include/sepal/clef_head.hpp for the structure).
// Float64 accumulation everywhere, single-threaded, written to be read next to joint_schema_model.py.
#include "sepal/clef_head.hpp"

#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>

#include "sepal/json.hpp"
#include "sepal/safetensors.hpp"

namespace sepal {

// ---------------------------------------------------------------------------------------------------------------
// Config and weights
// ---------------------------------------------------------------------------------------------------------------

ClefHeadConfig ClefHeadConfig::parse(std::string_view json_text) {
    json::Value root;
    try {
        root = json::parse(json_text);
    } catch (const std::runtime_error& error) {
        throw std::invalid_argument(std::string("joint_head_config.json: not valid JSON: ") + error.what());
    }
    if (root.kind() != json::Kind::Object) throw std::invalid_argument("joint_head_config.json: not an object");
    ClefHeadConfig cfg;
    static const char* const keys[] = {"hidden_size", "width", "routing_layers", "layers", "heads", "feedforward"};
    std::uint32_t* const fields[] = {&cfg.hidden_size, &cfg.width, &cfg.routing_layers, &cfg.layers, &cfg.heads, &cfg.feedforward};
    for (const std::string& key : root.keys()) {
        if (std::find(std::begin(keys), std::end(keys), key) == std::end(keys)) {
            throw std::invalid_argument("joint_head_config.json: unknown field \"" + key + "\"");
        }
    }
    for (std::size_t i = 0; i < 6; ++i) {
        const json::Value* v = root.find(keys[i]);
        if (v == nullptr) throw std::invalid_argument(std::string("joint_head_config.json: \"") + keys[i] + "\" is missing");
        std::uint64_t n = 0;
        try {
            n = v->as_uint();
        } catch (const std::runtime_error&) {
            throw std::invalid_argument(std::string("joint_head_config.json: \"") + keys[i] + "\" is not a non-negative whole number");
        }
        if (n > 1000000) throw std::invalid_argument(std::string("joint_head_config.json: \"") + keys[i] + "\" is implausibly large");
        *fields[i] = static_cast<std::uint32_t>(n);
    }
    if (cfg.hidden_size == 0 || cfg.width == 0 || cfg.heads == 0 || cfg.feedforward == 0) {
        throw std::invalid_argument("joint_head_config.json: hidden_size, width, heads and feedforward must be positive");
    }
    if (cfg.width % cfg.heads != 0) throw std::invalid_argument("joint_head_config.json: width is not a multiple of heads");
    return cfg;
}

namespace {

using NamedShape = std::pair<std::string, std::vector<std::size_t>>;

void add_ln(std::vector<NamedShape>& out, const std::string& p, std::size_t n) {
    out.push_back({p + ".weight", {n}});
    out.push_back({p + ".bias", {n}});
}
void add_linear(std::vector<NamedShape>& out, const std::string& p, std::size_t o, std::size_t i, bool bias) {
    out.push_back({p + ".weight", {o, i}});
    if (bias) out.push_back({p + ".bias", {o}});
}
void add_mha(std::vector<NamedShape>& out, const std::string& p, std::size_t w) {
    out.push_back({p + ".in_proj_weight", {3 * w, w}});
    out.push_back({p + ".in_proj_bias", {3 * w}});
    add_linear(out, p + ".out_proj", w, w, true);
}

std::vector<NamedShape> head_tensor_shapes(const ClefHeadConfig& c) {
    const std::size_t h = c.hidden_size, w = c.width, f = c.feedforward;
    std::vector<NamedShape> s;
    add_ln(s, "hidden_norm", h);
    add_linear(s, "memory_projection", w, h, false);
    add_linear(s, "question_projection", w, h, false);
    add_linear(s, "option_question_projection", w, h, false);
    add_linear(s, "global_projection", w, h, false);
    add_linear(s, "option_context_projection", w, h, false);
    add_linear(s, "option_lexical_projection", w, h, false);
    s.push_back({"type_embedding.weight", {3, w}});
    for (std::uint32_t i = 0; i < c.routing_layers; ++i) {
        const std::string p = "evidence_layers." + std::to_string(i);
        add_ln(s, p + ".query_norm", w);
        add_ln(s, p + ".memory_norm", w);
        add_mha(s, p + ".attention", w);
        add_ln(s, p + ".feedforward_norm", w);
        add_linear(s, p + ".feedforward.0", f, w, true);
        add_linear(s, p + ".feedforward.3", w, f, true);
    }
    add_ln(s, "option_summary_norm", w);
    for (std::uint32_t i = 0; i < c.layers; ++i) {
        const std::string p = "layers." + std::to_string(i);
        add_mha(s, p + ".self_attn", w);
        add_mha(s, p + ".multihead_attn", w);
        add_linear(s, p + ".linear1", f, w, true);
        add_linear(s, p + ".linear2", w, f, true);
        add_ln(s, p + ".norm1", w);
        add_ln(s, p + ".norm2", w);
        add_ln(s, p + ".norm3", w);
    }
    add_ln(s, "field_norm", w);
    add_ln(s, "option_norm", w);
    add_linear(s, "residual_scorer.0", w, 4 * w, true);
    add_linear(s, "residual_scorer.3", 1, w, true);
    s.push_back({"prior_logit_scale", {}});
    s.push_back({"joint_logit_scale", {}});
    s.push_back({"residual_gate", {}});
    return s;
}

std::string shape_text(const std::vector<std::size_t>& shape) {
    std::string t = "[";
    for (std::size_t i = 0; i < shape.size(); ++i) t += (i ? ", " : "") + std::to_string(shape[i]);
    return t + "]";
}

} // namespace

ClefHeadWeights ClefHeadWeights::open(const std::string& safetensors_path, const ClefHeadConfig& config) {
    const SafetensorsFile file = SafetensorsFile::open(safetensors_path);
    ClefHeadWeights out;
    out.config = config;
    const std::vector<NamedShape> expected = head_tensor_shapes(config);
    std::set<std::string> names;
    for (const auto& [name, shape] : expected) {
        names.insert(name);
        const TensorView* t = file.find(name);
        if (t == nullptr) throw std::runtime_error("head '" + safetensors_path + "': missing tensor " + name);
        if (t->shape != shape) {
            throw std::runtime_error("head tensor " + name + " has shape " + shape_text(t->shape) + ", expected " + shape_text(shape));
        }
        if (t->type != StType::BF16 && t->type != StType::F32 && t->type != StType::F16) {
            throw std::runtime_error("head tensor " + name + " is stored as " + std::string(name_of(t->type)) + ", not BF16, F16 or F32");
        }
        std::vector<float> values(t->elements());
        convert_floating(*t, values.data());
        out.tensors.emplace(name, std::move(values));
        out.shapes.emplace(name, shape);
    }
    for (const TensorView& t : file.tensors()) {
        if (!names.contains(t.name)) throw std::runtime_error("head '" + safetensors_path + "': unexpected tensor " + t.name);
    }
    return out;
}

const std::vector<float>& ClefHeadWeights::at(std::string_view name) const {
    const auto it = tensors.find(name);
    if (it == tensors.end()) throw std::out_of_range("head has no tensor named " + std::string(name));
    return it->second;
}

// ---------------------------------------------------------------------------------------------------------------
// Request check
// ---------------------------------------------------------------------------------------------------------------

void check_request(const EncodedRequest& request, std::uint64_t vocab_size) {
    const std::uint64_t length = request.token_ids.size();
    if (length == 0) throw std::invalid_argument("request has no tokens");
    for (std::size_t i = 0; i < request.token_ids.size(); ++i) {
        if (request.token_ids[i] >= vocab_size) {
            throw std::invalid_argument("token " + std::to_string(i) + " has id " + std::to_string(request.token_ids[i]) +
                                        ", outside the vocabulary of " + std::to_string(vocab_size));
        }
    }
    if (request.questions.empty()) throw std::invalid_argument("request has no questions");
    const auto check_span = [&](const TokenSpan& s, const std::string& what) {
        if (s.end <= s.begin || s.end > length) {
            throw std::invalid_argument(what + " span [" + std::to_string(s.begin) + ", " + std::to_string(s.end) +
                                        ") is empty or outside the " + std::to_string(length) + " tokens");
        }
    };
    for (std::size_t q = 0; q < request.questions.size(); ++q) {
        const EncodedQuestion& question = request.questions[q];
        const std::string qn = "question " + std::to_string(q);
        if (static_cast<std::uint32_t>(question.type) > 2) throw std::invalid_argument(qn + ": unknown type");
        if (question.options.empty()) throw std::invalid_argument(qn + " has no options");
        check_span(question.instruction, qn + " instruction");
        for (std::size_t k = 0; k < question.options.size(); ++k) check_span(question.options[k], qn + " option " + std::to_string(k));
    }
}

// ---------------------------------------------------------------------------------------------------------------
// The reference head
// ---------------------------------------------------------------------------------------------------------------

namespace {

using Vec = std::vector<double>;

constexpr double ln_eps = 1e-5;
constexpr double cosine_eps = 1e-8;
constexpr double normalize_eps = 1e-12;

struct Params {
    const ClefHeadWeights& w;
    const float* get(const std::string& name) const { return w.at(name).data(); }
};

// y[n, out] = x[n, in] W^T + b, W [out, in] row-major (a pointer into a larger matrix selects a block of rows).
Vec linear(const double* x, std::size_t n, std::size_t in, const float* weight, std::size_t out, const float* bias) {
    Vec y(n * out);
    for (std::size_t r = 0; r < n; ++r) {
        const double* xr = x + r * in;
        for (std::size_t o = 0; o < out; ++o) {
            const float* wr = weight + o * in;
            double acc = 0;
            for (std::size_t i = 0; i < in; ++i) acc += xr[i] * static_cast<double>(wr[i]);
            y[r * out + o] = acc + (bias != nullptr ? static_cast<double>(bias[o]) : 0.0);
        }
    }
    return y;
}

// nn.LayerNorm over the last dimension: eps inside the root, biased variance.
Vec layer_norm(const double* x, std::size_t n, std::size_t d, const float* gamma, const float* beta) {
    Vec y(n * d);
    for (std::size_t r = 0; r < n; ++r) {
        const double* xr = x + r * d;
        double mean = 0;
        for (std::size_t i = 0; i < d; ++i) mean += xr[i];
        mean /= static_cast<double>(d);
        double var = 0;
        for (std::size_t i = 0; i < d; ++i) var += (xr[i] - mean) * (xr[i] - mean);
        var /= static_cast<double>(d);
        const double inv = 1.0 / std::sqrt(var + ln_eps);
        for (std::size_t i = 0; i < d; ++i) {
            y[r * d + i] = (xr[i] - mean) * inv * static_cast<double>(gamma[i]) + static_cast<double>(beta[i]);
        }
    }
    return y;
}

double gelu(double x) { return 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0))); }

// nn.MultiheadAttention (batch_first, no mask, no dropout) for one sequence: queries [nq, w] attend over keys and
// values built from `source` [nk, w]. in_proj_weight rows are q, k, v stacked.
Vec multihead_attention(const Params& p, const std::string& prefix, const Vec& query, std::size_t nq, const Vec& source,
                        std::size_t nk, std::size_t width, std::size_t heads) {
    const float* in_w = p.get(prefix + ".in_proj_weight");
    const float* in_b = p.get(prefix + ".in_proj_bias");
    const Vec q = linear(query.data(), nq, width, in_w, width, in_b);
    const Vec k = linear(source.data(), nk, width, in_w + width * width, width, in_b + width);
    const Vec v = linear(source.data(), nk, width, in_w + 2 * width * width, width, in_b + 2 * width);
    const std::size_t hd = width / heads;
    const double scale = 1.0 / std::sqrt(static_cast<double>(hd));
    Vec context(nq * width, 0.0);
    Vec scores(nk);
    for (std::size_t h = 0; h < heads; ++h) {
        for (std::size_t i = 0; i < nq; ++i) {
            double top = -INFINITY;
            for (std::size_t j = 0; j < nk; ++j) {
                double s = 0;
                for (std::size_t d = 0; d < hd; ++d) s += q[i * width + h * hd + d] * k[j * width + h * hd + d];
                scores[j] = s * scale;
                top = std::max(top, scores[j]);
            }
            double sum = 0;
            for (std::size_t j = 0; j < nk; ++j) {
                scores[j] = std::exp(scores[j] - top);
                sum += scores[j];
            }
            for (std::size_t j = 0; j < nk; ++j) {
                const double pj = scores[j] / sum;
                for (std::size_t d = 0; d < hd; ++d) context[i * width + h * hd + d] += pj * v[j * width + h * hd + d];
            }
        }
    }
    return linear(context.data(), nq, width, p.get(prefix + ".out_proj.weight"), width, p.get(prefix + ".out_proj.bias"));
}

Vec layer_norm_named(const Params& p, const std::string& name, const Vec& x, std::size_t n, std::size_t d) {
    return layer_norm(x.data(), n, d, p.get(name + ".weight"), p.get(name + ".bias"));
}

// Linear(w, f) - GELU(erf) - Linear(f, w) with the given parameter names.
Vec feed_forward(const Params& p, const std::string& first, const std::string& second, const Vec& x, std::size_t n, std::size_t width,
                 std::size_t ff) {
    Vec hidden = linear(x.data(), n, width, p.get(first + ".weight"), ff, p.get(first + ".bias"));
    for (double& v : hidden) v = gelu(v);
    return linear(hidden.data(), n, ff, p.get(second + ".weight"), width, p.get(second + ".bias"));
}

void add_into(Vec& x, const Vec& y) {
    for (std::size_t i = 0; i < x.size(); ++i) x[i] += y[i];
}

double norm_of(const double* x, std::size_t d) {
    double s = 0;
    for (std::size_t i = 0; i < d; ++i) s += x[i] * x[i];
    return std::sqrt(s);
}

// F.normalize(x, dim=-1): x / max(||x||, 1e-12)
Vec normalized(const double* x, std::size_t d) {
    const double n = std::max(norm_of(x, d), normalize_eps);
    Vec y(d);
    for (std::size_t i = 0; i < d; ++i) y[i] = x[i] / n;
    return y;
}

// F.cosine_similarity(a, b, dim=-1, eps=1e-8): each vector divided by max(its norm, eps), then dotted.
double cosine(const double* a, const double* b, std::size_t d) {
    const double na = std::max(norm_of(a, d), cosine_eps);
    const double nb = std::max(norm_of(b, d), cosine_eps);
    double s = 0;
    for (std::size_t i = 0; i < d; ++i) s += (a[i] / na) * (b[i] / nb);
    return s;
}

Vec mean_rows(const Vec& rows, std::size_t d, const TokenSpan& span) {
    Vec m(d, 0.0);
    for (std::size_t t = span.begin; t < span.end; ++t) {
        for (std::size_t i = 0; i < d; ++i) m[i] += rows[t * d + i];
    }
    const double count = static_cast<double>(span.end - span.begin);
    for (double& v : m) v /= count;
    return m;
}

} // namespace

RequestLogits clef_head_reference(const ClefHeadWeights& weights, std::span<const float> hidden, const EncodedRequest& request,
                                  EmbeddingRowFn embedding_row, const void* context) {
    const ClefHeadConfig& cfg = weights.config;
    const std::size_t H = cfg.hidden_size, W = cfg.width, FF = cfg.feedforward, heads = cfg.heads;
    const std::size_t L = request.token_ids.size();
    check_request(request, ~std::uint64_t{0});
    if (hidden.size() != L * H) {
        throw std::invalid_argument("clef head: hidden has " + std::to_string(hidden.size()) + " values, expected " +
                                    std::to_string(L) + " x " + std::to_string(H));
    }
    const Params p{weights};

    // h = LayerNorm(hidden) for every position; memory = W_mem h
    const Vec hidden64(hidden.begin(), hidden.end());
    const Vec hn = layer_norm_named(p, "hidden_norm", hidden64, L, H);
    const Vec memory = linear(hn.data(), L, H, p.get("memory_projection.weight"), W, nullptr);
    const double* global = hn.data() + (L - 1) * H;

    const std::size_t Q = request.questions.size();
    Vec qvec_all(Q * H); // question vectors (mean of the instruction span)
    std::vector<std::size_t> first_option(Q + 1, 0);
    Vec option_ctx, option_lex; // [O, H] context means and lexical means
    std::vector<float> row(H);
    for (std::size_t q = 0; q < Q; ++q) {
        const EncodedQuestion& question = request.questions[q];
        const Vec qv = mean_rows(hn, H, question.instruction);
        std::copy(qv.begin(), qv.end(), qvec_all.begin() + static_cast<std::ptrdiff_t>(q * H));
        for (const TokenSpan& span : question.options) {
            const Vec ctx = mean_rows(hn, H, span);
            option_ctx.insert(option_ctx.end(), ctx.begin(), ctx.end());
            Vec lex(H, 0.0);
            for (std::uint32_t t = span.begin; t < span.end; ++t) {
                embedding_row(context, request.token_ids[t], row.data());
                for (std::size_t i = 0; i < H; ++i) lex[i] += static_cast<double>(row[i]);
            }
            for (double& v : lex) v /= static_cast<double>(span.end - span.begin);
            option_lex.insert(option_lex.end(), lex.begin(), lex.end());
        }
        first_option[q + 1] = first_option[q] + question.options.size();
    }
    const std::size_t O = first_option[Q];

    // query[o] = W_ctx ctx + W_lex lex + W_oq q_vec[question of o]
    Vec options = linear(option_ctx.data(), O, H, p.get("option_context_projection.weight"), W, nullptr);
    {
        const Vec lexical = linear(option_lex.data(), O, H, p.get("option_lexical_projection.weight"), W, nullptr);
        const Vec question_part = linear(qvec_all.data(), Q, H, p.get("option_question_projection.weight"), W, nullptr);
        for (std::size_t q = 0; q < Q; ++q) {
            for (std::size_t o = first_option[q]; o < first_option[q + 1]; ++o) {
                for (std::size_t i = 0; i < W; ++i) options[o * W + i] += lexical[o * W + i] + question_part[q * W + i];
            }
        }
    }

    // routing layers: x += MHA(LN_q(x), LN_m(memory), LN_m(memory)); x += FF(LN_ff(x))
    for (std::uint32_t l = 0; l < cfg.routing_layers; ++l) {
        const std::string name = "evidence_layers." + std::to_string(l);
        const Vec nq = layer_norm_named(p, name + ".query_norm", options, O, W);
        const Vec nm = layer_norm_named(p, name + ".memory_norm", memory, L, W);
        add_into(options, multihead_attention(p, name + ".attention", nq, O, nm, L, W, heads));
        const Vec nf = layer_norm_named(p, name + ".feedforward_norm", options, O, W);
        add_into(options, feed_forward(p, name + ".feedforward.0", name + ".feedforward.3", nf, O, W, FF));
    }

    // field[q] = base + LN_sum(summary) + W_g global + type_embedding[type]
    const Vec base = linear(qvec_all.data(), Q, H, p.get("question_projection.weight"), W, nullptr);
    Vec summaries(Q * W, 0.0);
    for (std::size_t q = 0; q < Q; ++q) {
        const std::size_t n = first_option[q + 1] - first_option[q];
        Vec weight(n);
        double top = -INFINITY;
        for (std::size_t k = 0; k < n; ++k) {
            double s = 0;
            for (std::size_t i = 0; i < W; ++i) s += options[(first_option[q] + k) * W + i] * base[q * W + i];
            weight[k] = s / std::sqrt(static_cast<double>(W));
            top = std::max(top, weight[k]);
        }
        double sum = 0;
        for (double& v : weight) {
            v = std::exp(v - top);
            sum += v;
        }
        for (std::size_t k = 0; k < n; ++k) {
            for (std::size_t i = 0; i < W; ++i) summaries[q * W + i] += (weight[k] / sum) * options[(first_option[q] + k) * W + i];
        }
    }
    const Vec summary_normed = layer_norm_named(p, "option_summary_norm", summaries, Q, W);
    const Vec global_part = linear(global, 1, H, p.get("global_projection.weight"), W, nullptr);
    const float* type_embedding = p.get("type_embedding.weight");
    Vec fields(Q * W);
    for (std::size_t q = 0; q < Q; ++q) {
        const std::size_t type = static_cast<std::size_t>(request.questions[q].type);
        for (std::size_t i = 0; i < W; ++i) {
            fields[q * W + i] = base[q * W + i] + summary_normed[q * W + i] + global_part[i] + static_cast<double>(type_embedding[type * W + i]);
        }
    }

    // decoder layers (norm_first): x += SA(LN1 x); x += MHA(LN2 x, memory) with memory NOT normed; x += FF(LN3 x)
    for (std::uint32_t l = 0; l < cfg.layers; ++l) {
        const std::string name = "layers." + std::to_string(l);
        const Vec n1 = layer_norm_named(p, name + ".norm1", fields, Q, W);
        add_into(fields, multihead_attention(p, name + ".self_attn", n1, Q, n1, Q, W, heads));
        const Vec n2 = layer_norm_named(p, name + ".norm2", fields, Q, W);
        add_into(fields, multihead_attention(p, name + ".multihead_attn", n2, Q, memory, L, W, heads));
        const Vec n3 = layer_norm_named(p, name + ".norm3", fields, Q, W);
        add_into(fields, feed_forward(p, name + ".linear1", name + ".linear2", n3, Q, W, FF));
    }
    fields = layer_norm_named(p, "field_norm", fields, Q, W);
    const Vec opt_normed = layer_norm_named(p, "option_norm", options, O, W);

    // logit = s_prior cos(lex, normalize(q_vec + global)) + sigmoid(gate) (s_joint cos(field, opt) + MLP([f, o, f*o, |f-o|]))
    const double log100 = std::log(100.0);
    const double prior_scale = std::exp(std::min(static_cast<double>(p.get("prior_logit_scale")[0]), log100));
    const double joint_scale = std::exp(std::min(static_cast<double>(p.get("joint_logit_scale")[0]), log100));
    const double gate = 1.0 / (1.0 + std::exp(-static_cast<double>(p.get("residual_gate")[0])));

    RequestLogits result;
    for (std::size_t q = 0; q < Q; ++q) {
        Vec sum_vec(H);
        for (std::size_t i = 0; i < H; ++i) sum_vec[i] = qvec_all[q * H + i] + global[i];
        const Vec anchor = normalized(sum_vec.data(), H);
        const std::size_t n = first_option[q + 1] - first_option[q];
        Vec features(n * 4 * W);
        std::vector<double> priors(n), cosines(n);
        for (std::size_t k = 0; k < n; ++k) {
            const std::size_t o = first_option[q] + k;
            const Vec lex_anchor = normalized(option_lex.data() + o * H, H);
            double dot = 0;
            for (std::size_t i = 0; i < H; ++i) dot += lex_anchor[i] * anchor[i];
            priors[k] = prior_scale * dot;
            const double* f = fields.data() + q * W;
            const double* op = opt_normed.data() + o * W;
            cosines[k] = cosine(f, op, W);
            double* feat = features.data() + k * 4 * W;
            for (std::size_t i = 0; i < W; ++i) {
                feat[i] = f[i];
                feat[W + i] = op[i];
                feat[2 * W + i] = f[i] * op[i];
                feat[3 * W + i] = std::fabs(f[i] - op[i]);
            }
        }
        Vec hid = linear(features.data(), n, 4 * W, p.get("residual_scorer.0.weight"), W, p.get("residual_scorer.0.bias"));
        for (double& v : hid) v = gelu(v);
        const Vec residual = linear(hid.data(), n, W, p.get("residual_scorer.3.weight"), 1, p.get("residual_scorer.3.bias"));
        std::vector<float> logits(n);
        for (std::size_t k = 0; k < n; ++k) {
            logits[k] = static_cast<float>(priors[k] + gate * (joint_scale * cosines[k] + residual[k]));
        }
        result.logits.push_back(std::move(logits));
    }
    return result;
}

} // namespace sepal
