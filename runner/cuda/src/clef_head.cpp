#include "sepal/cuda/clef_head_cuda.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <string>
#include <unordered_map>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "clef_head_kernels.hpp"
#include "device.hpp"
#include "sepal/cuda/cuda_gemm.hpp"

namespace sepal::cuda {

namespace {

void check_cuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess) throw CudaError(what, static_cast<long long>(status), cudaGetErrorString(status), "clef head");
}

void check_blas(cublasStatus_t status, const char* what) {
    if (status != CUBLAS_STATUS_SUCCESS) throw CudaError(what, static_cast<long long>(status), cublasGetStatusString(status), "clef head");
}

std::uint32_t u32(std::size_t v) { return static_cast<std::uint32_t>(v); }

/// One multi-head attention's parameters on the device.
struct AttnPtrs {
    const float* in_w = nullptr;  // [3 * width, width]
    const float* in_b = nullptr;  // [3 * width]
    const float* out_w = nullptr; // [width, width]
    const float* out_b = nullptr; // [width]
};
struct NormPtrs {
    const float* w = nullptr;
    const float* b = nullptr;
};
struct FfPtrs {
    const float* w1 = nullptr; // [ff, width]
    const float* b1 = nullptr;
    const float* w2 = nullptr; // [width, ff]
    const float* b2 = nullptr;
};
struct RoutingLayer {
    NormPtrs query_norm, memory_norm, ff_norm;
    AttnPtrs attn;
    FfPtrs ff;
};
struct DecoderLayer {
    NormPtrs norm1, norm2, norm3;
    AttnPtrs self_attn, cross_attn;
    FfPtrs ff;
};

/// One request's place in the batch.
struct Plan {
    std::size_t length = 0;
    std::size_t mem_off = 0;  // tokens
    std::size_t field_off = 0, fields = 0;
    std::size_t option_off = 0, options = 0;
    std::size_t span_q = 0, span_p = 0, span_g = 0; // first span of each kind (global span arrays)
    std::uint32_t min_begin = 0;                    // the first token any span of the request reads
};

} // namespace

struct CudaClefHead::Impl {
    ClefHeadConfig cfg;
    ClefHeadCapacity cap;
    cudaStream_t stream = nullptr;
    cublasHandle_t blas = nullptr;
    std::size_t W = 0, Hd = 0, FF = 0, nh = 0;

    DeviceBuffer<float> weights_mem;
    std::map<std::string, const float*, std::less<>> wp;
    float prior_scale = 0, joint_scale = 0, gate = 0;
    NormPtrs hidden_norm, summary_norm, field_norm, option_norm;
    const float *mem_proj = nullptr, *q_proj = nullptr, *oq_proj = nullptr, *g_proj = nullptr, *ctx_proj = nullptr, *lex_proj = nullptr;
    const float* type_emb = nullptr;
    std::vector<RoutingLayer> routing;
    std::vector<DecoderLayer> decoder;
    const float *scorer_w1 = nullptr, *scorer_b1 = nullptr, *scorer_w2 = nullptr;
    float scorer_b2 = 0;

    // Arenas and scratch
    DeviceBuffer<float> mem, nmem, xscr, mtmp, acc, lex, urows, base, tmpq, glob;
    DeviceBuffer<float> ox, oln, qproj, qk, ctx, att, ao, ffh, f, fln, qkv, features, hid2, cosine, logits, scores;
    DeviceBuffer<std::uint32_t> table;
    DeviceBuffer<ClefSpan> span_table;

    // The batch
    std::vector<const EncodedRequest*> reqs;
    std::vector<Plan> plans;
    std::size_t mem_used = 0, n_fields = 0, n_options = 0, n_spans = 0, n_qspans = 0, n_pspans = 0;
    std::size_t max_options_per_q = 0, max_fields_per_req = 0;
    bool planned = false;
    // table offsets (in uint32 units)
    std::size_t t_q_first = 0, t_q_count = 0, t_q_request = 0, t_q_type = 0, t_opt_question = 0, t_q_req_first = 0, t_q_req_count = 0;
    std::size_t t_tok_begin = 0, t_tok_index = 0, t_opt_req_first = 0;

    Impl(const ClefHeadWeights& weights, const ClefHeadCapacity& capacity, void* stream_) : cfg(weights.config), cap(capacity), stream(static_cast<cudaStream_t>(stream_)) {
        W = cfg.width;
        Hd = cfg.hidden_size;
        FF = cfg.feedforward;
        nh = cfg.heads;
        if (nh == 0 || W % nh != 0) throw std::invalid_argument("clef head: width is not a multiple of the heads");
        upload_weights(weights);
        check_blas(cublasCreate(&blas), "cublasCreate");
        check_blas(cublasSetStream(blas, stream), "cublasSetStream");
        check_blas(cublasSetMathMode(blas, CUBLAS_DEFAULT_MATH), "cublasSetMathMode");
        allocate();
    }
    ~Impl() {
        if (stream != nullptr) cudaStreamSynchronize(stream);
        if (blas != nullptr) cublasDestroy(blas);
    }

    void upload_weights(const ClefHeadWeights& weights) {
        std::size_t total = 0;
        for (const auto& [name, v] : weights.tensors) total += (v.size() + 63) / 64 * 64;
        weights_mem = DeviceBuffer<float>(total);
        std::size_t at = 0;
        std::vector<float> staging;
        for (const auto& [name, v] : weights.tensors) {
            check_cuda(cudaMemcpy(weights_mem.data() + at, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy");
            wp[name] = weights_mem.data() + at;
            at += (v.size() + 63) / 64 * 64;
        }
        auto P = [&](const std::string& n) -> const float* {
            const auto it = wp.find(n);
            if (it == wp.end()) throw std::invalid_argument("clef head: missing weight " + n);
            return it->second;
        };
        auto norm = [&](const std::string& n) { return NormPtrs{P(n + ".weight"), P(n + ".bias")}; };
        auto attn = [&](const std::string& n) { return AttnPtrs{P(n + ".in_proj_weight"), P(n + ".in_proj_bias"), P(n + ".out_proj.weight"), P(n + ".out_proj.bias")}; };
        hidden_norm = norm("hidden_norm");
        summary_norm = norm("option_summary_norm");
        field_norm = norm("field_norm");
        option_norm = norm("option_norm");
        mem_proj = P("memory_projection.weight");
        q_proj = P("question_projection.weight");
        oq_proj = P("option_question_projection.weight");
        g_proj = P("global_projection.weight");
        ctx_proj = P("option_context_projection.weight");
        lex_proj = P("option_lexical_projection.weight");
        type_emb = P("type_embedding.weight");
        for (std::uint32_t i = 0; i < cfg.routing_layers; ++i) {
            const std::string b = "evidence_layers." + std::to_string(i);
            RoutingLayer l;
            l.query_norm = norm(b + ".query_norm");
            l.memory_norm = norm(b + ".memory_norm");
            l.ff_norm = norm(b + ".feedforward_norm");
            l.attn = attn(b + ".attention");
            l.ff = FfPtrs{P(b + ".feedforward.0.weight"), P(b + ".feedforward.0.bias"), P(b + ".feedforward.3.weight"), P(b + ".feedforward.3.bias")};
            routing.push_back(l);
        }
        for (std::uint32_t i = 0; i < cfg.layers; ++i) {
            const std::string b = "layers." + std::to_string(i);
            DecoderLayer l;
            l.norm1 = norm(b + ".norm1");
            l.norm2 = norm(b + ".norm2");
            l.norm3 = norm(b + ".norm3");
            l.self_attn = attn(b + ".self_attn");
            l.cross_attn = attn(b + ".multihead_attn");
            l.ff = FfPtrs{P(b + ".linear1.weight"), P(b + ".linear1.bias"), P(b + ".linear2.weight"), P(b + ".linear2.bias")};
            decoder.push_back(l);
        }
        scorer_w1 = P("residual_scorer.0.weight");
        scorer_b1 = P("residual_scorer.0.bias");
        scorer_w2 = P("residual_scorer.3.weight");
        scorer_b2 = weights.tensors.at("residual_scorer.3.bias").at(0);
        const double limit = std::log(100.0);
        prior_scale = static_cast<float>(std::exp(std::min(static_cast<double>(weights.tensors.at("prior_logit_scale").at(0)), limit)));
        joint_scale = static_cast<float>(std::exp(std::min(static_cast<double>(weights.tensors.at("joint_logit_scale").at(0)), limit)));
        gate = weights.tensors.at("residual_gate").at(0);
    }

    void allocate() {
        const std::size_t S = cap.fields + cap.options + cap.requests;
        mem = DeviceBuffer<float>(cap.memory_tokens * W);
        nmem = DeviceBuffer<float>(cap.memory_tokens * W);
        xscr = DeviceBuffer<float>(cap.call_tokens * Hd);
        mtmp = DeviceBuffer<float>(cap.call_tokens * W);
        acc = DeviceBuffer<float>(S * Hd);
        lex = DeviceBuffer<float>(cap.options * Hd);
        urows = DeviceBuffer<float>(cap.unique_tokens * Hd);
        base = DeviceBuffer<float>(cap.fields * W);
        tmpq = DeviceBuffer<float>(cap.fields * W);
        glob = DeviceBuffer<float>(cap.requests * W);
        const std::size_t Q = std::max(cap.options, cap.fields);
        ox = DeviceBuffer<float>(cap.options * W);
        oln = DeviceBuffer<float>(Q * W);
        qproj = DeviceBuffer<float>(Q * W);
        qk = DeviceBuffer<float>(nh * Q * W);
        ctx = DeviceBuffer<float>(nh * Q * W);
        att = DeviceBuffer<float>(Q * W);
        ao = DeviceBuffer<float>(Q * W);
        ffh = DeviceBuffer<float>(Q * FF);
        f = DeviceBuffer<float>(cap.fields * W);
        fln = DeviceBuffer<float>(cap.fields * W);
        qkv = DeviceBuffer<float>(cap.fields * 3 * W);
        features = DeviceBuffer<float>(cap.options * 4 * W);
        hid2 = DeviceBuffer<float>(cap.options * W);
        cosine = DeviceBuffer<float>(cap.options);
        logits = DeviceBuffer<float>(cap.options);
        scores = DeviceBuffer<float>(cap.score_floats);
        span_table = DeviceBuffer<ClefSpan>(S);
        const std::size_t table_words = 8 * cap.fields + 6 * cap.requests + 2 * cap.options + cap.option_tokens + 16;
        table = DeviceBuffer<std::uint32_t>(table_words);
    }

    // y[n, out] = x[n, in] W[out, in]^T (+ bias)
    void linear(const float* x, std::size_t n, std::size_t in, const float* Wt, std::size_t out, const float* bias, float* y, std::size_t ldy = 0) {
        if (n == 0) return;
        if (ldy == 0) ldy = out;
        float beta = 0.0f;
        if (bias != nullptr) {
            clef_broadcast_bias(y, ldy, n, out, bias, stream);
            beta = 1.0f;
        }
        const float alpha = 1.0f;
        check_blas(cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(out), static_cast<int>(n), static_cast<int>(in), &alpha, Wt,
                               static_cast<int>(in), x, static_cast<int>(in), &beta, y, static_cast<int>(ldy)),
                   "cublasSgemm");
    }

    void layer_norm(const float* x, float* out, std::size_t n, std::size_t width, const NormPtrs& norm) {
        clef_layer_norm(x, width, out, width, n, width, norm.w, norm.b, 1e-5f, stream);
    }

    struct QuerySeg {
        std::size_t off = 0, count = 0, mem_tok = 0, length = 0;
    };

    /// MultiheadAttention of `n` query rows (already layer-normed) over memory `mem_base` (rows of width W), per request segment.
    /// gamma/beta: the affine part of LayerNorm(memory) when `mem_base` holds the normalised memory (routing layers).
    void cross_attention(const float* qin, std::size_t n, const AttnPtrs& a, const float* mem_base, const float* gamma, const float* beta,
                         const std::vector<QuerySeg>& segs, float* out) {
        const float one = 1.0f, zero = 0.0f;
        linear(qin, n, W, a.in_w, W, a.in_b, qproj.data());
        clef_scale(qproj.data(), n * W, 1.0f / std::sqrt(static_cast<float>(W / nh)), stream); // 1/sqrt(head_dim)
        const std::size_t hd = W / nh;
        // qk_h = Q_h Wk_h : [n, W] per head, stored [head][n][W]
        check_blas(cublasSgemmStridedBatched(blas, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(W), static_cast<int>(n), static_cast<int>(hd), &one,
                                             a.in_w + W * W, static_cast<int>(W), static_cast<long long>(hd * W), qproj.data(), static_cast<int>(W),
                                             static_cast<long long>(hd), &zero, qk.data(), static_cast<int>(W), static_cast<long long>(n * W),
                                             static_cast<int>(nh)),
                   "cublasSgemmStridedBatched");
        if (gamma != nullptr) clef_scale_shift_cols(qk.data(), nh * n, W, gamma, nullptr, stream);
        for (const QuerySeg& s : segs) {
            if (s.count == 0) continue;
            const float* m = mem_base + s.mem_tok * W;
            const int L = static_cast<int>(s.length);
            const std::size_t per_query = nh * s.length;
            if (per_query > cap.score_floats) throw std::invalid_argument("clef head: attention scores of one query exceed the scratch");
            const std::size_t step = std::min(s.count, cap.score_floats / per_query);
            for (std::size_t start = 0; start < s.count; start += step) {
                const std::size_t cnt = std::min(step, s.count - start);
                const int c = static_cast<int>(cnt);
                const std::size_t row0 = s.off + start;
                // scores[h] [cnt, L] = qk_h [cnt, W] M^T
                check_blas(cublasSgemmStridedBatched(blas, CUBLAS_OP_T, CUBLAS_OP_N, L, c, static_cast<int>(W), &one, m, static_cast<int>(W), 0,
                                                     qk.data() + row0 * W, static_cast<int>(W), static_cast<long long>(n * W), &zero, scores.data(), L,
                                                     static_cast<long long>(cnt * s.length), static_cast<int>(nh)),
                           "cublasSgemmStridedBatched");
                clef_softmax_rows(scores.data(), nh * cnt, s.length, 1.0f, stream);
                // ctx_h [cnt, W] = P_h [cnt, L] M
                check_blas(cublasSgemmStridedBatched(blas, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(W), c, L, &one, m, static_cast<int>(W), 0,
                                                     scores.data(), L, static_cast<long long>(cnt * s.length), &zero, ctx.data() + row0 * W,
                                                     static_cast<int>(W), static_cast<long long>(n * W), static_cast<int>(nh)),
                           "cublasSgemmStridedBatched");
            }
        }
        if (gamma != nullptr || beta != nullptr) clef_scale_shift_cols(ctx.data(), nh * n, W, gamma, beta, stream);
        // o[:, h*hd ...] = ctx_h Wv_h^T
        check_blas(cublasSgemmStridedBatched(blas, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(hd), static_cast<int>(n), static_cast<int>(W), &one,
                                             a.in_w + 2 * W * W, static_cast<int>(W), static_cast<long long>(hd * W), ctx.data(), static_cast<int>(W),
                                             static_cast<long long>(n * W), &zero, att.data(), static_cast<int>(W), static_cast<long long>(hd),
                                             static_cast<int>(nh)),
                   "cublasSgemmStridedBatched");
        clef_add_bias(att.data(), W, n, W, a.in_b + 2 * W, stream);
        linear(att.data(), n, W, a.out_w, W, a.out_b, out);
    }

    void feed_forward(const float* x_normed, std::size_t n, const FfPtrs& ff, float* out) {
        linear(x_normed, n, W, ff.w1, FF, ff.b1, ffh.data());
        clef_gelu(ffh.data(), n * FF, stream);
        linear(ffh.data(), n, FF, ff.w2, W, ff.b2, out);
    }

    // ---- the public operations ----
    void begin(std::span<const EncodedRequest* const> requests, EmbeddingRowFn embedding_row, const void* context);
    void consume(const float* hidden, std::size_t rows, std::span<const HeadSegment> segments);
    std::vector<RequestLogits> finish();
};

void CudaClefHead::Impl::begin(std::span<const EncodedRequest* const> requests, EmbeddingRowFn embedding_row, const void* context) {
    planned = false;
    const std::size_t R = requests.size();
    if (R == 0 || R > cap.requests) throw std::invalid_argument("clef head: " + std::to_string(R) + " requests (capacity " + std::to_string(cap.requests) + ")");
    std::vector<Plan> np(R);
    std::size_t mem_total = 0, fields = 0, options = 0, option_tokens = 0, max_opts = 0, max_fields = 0, score_need = 0;
    for (std::size_t r = 0; r < R; ++r) {
        const EncodedRequest& req = *requests[r];
        const std::size_t L = req.token_ids.size();
        if (L == 0) throw std::invalid_argument("clef head: request " + std::to_string(r) + " has no tokens");
        if (req.questions.empty()) throw std::invalid_argument("clef head: request " + std::to_string(r) + " has no questions");
        Plan& p = np[r];
        p.length = L;
        p.mem_off = mem_total;
        p.field_off = fields;
        p.fields = req.questions.size();
        p.option_off = options;
        p.min_begin = static_cast<std::uint32_t>(L - 1);
        std::size_t popt = 0;
        for (const EncodedQuestion& q : req.questions) {
            if (q.instruction.begin >= q.instruction.end || q.instruction.end > L) throw std::invalid_argument("clef head: bad instruction span in request " + std::to_string(r));
            if (q.options.empty()) throw std::invalid_argument("clef head: a question without options in request " + std::to_string(r));
            p.min_begin = std::min(p.min_begin, q.instruction.begin);
            for (const TokenSpan& s : q.options) {
                if (s.begin >= s.end || s.end > L) throw std::invalid_argument("clef head: bad option span in request " + std::to_string(r));
                p.min_begin = std::min(p.min_begin, s.begin);
                option_tokens += s.end - s.begin;
            }
            popt += q.options.size();
            max_opts = std::max(max_opts, q.options.size());
        }
        p.options = popt;
        mem_total += L;
        fields += p.fields;
        options += popt;
        max_fields = std::max(max_fields, p.fields);
        score_need = std::max(score_need, nh * L);
    }
    if (mem_total > cap.memory_tokens) throw std::invalid_argument("clef head: " + std::to_string(mem_total) + " memory tokens (capacity " + std::to_string(cap.memory_tokens) + ")");
    if (fields > cap.fields) throw std::invalid_argument("clef head: " + std::to_string(fields) + " fields (capacity " + std::to_string(cap.fields) + ")");
    if (options > cap.options) throw std::invalid_argument("clef head: " + std::to_string(options) + " options (capacity " + std::to_string(cap.options) + ")");
    if (option_tokens > cap.option_tokens) throw std::invalid_argument("clef head: " + std::to_string(option_tokens) + " option tokens (capacity " + std::to_string(cap.option_tokens) + ")");
    if (score_need > cap.score_floats) throw std::invalid_argument("clef head: attention scores of one query need " + std::to_string(score_need) + " floats (capacity " + std::to_string(cap.score_floats) + ")");

    // Spans: [instruction of every field][every option][global of every request], and the tables.
    std::vector<ClefSpan> spans;
    spans.reserve(fields + options + R);
    std::vector<std::uint32_t> q_first, q_count, q_request, q_type, opt_question, req_first, req_count, tok_begin, tok_index;
    std::unordered_map<std::uint32_t, std::uint32_t> unique;
    std::vector<std::uint32_t> unique_tokens;
    std::size_t qi = 0, oi = 0;
    for (std::size_t r = 0; r < R; ++r) {
        np[r].span_q = spans.size();
        for (const EncodedQuestion& q : requests[r]->questions) spans.push_back({q.instruction.begin, q.instruction.end});
    }
    n_qspans = spans.size();
    for (std::size_t r = 0; r < R; ++r) {
        np[r].span_p = spans.size();
        for (const EncodedQuestion& q : requests[r]->questions)
            for (const TokenSpan& s : q.options) spans.push_back({s.begin, s.end});
    }
    n_pspans = spans.size() - n_qspans;
    for (std::size_t r = 0; r < R; ++r) {
        np[r].span_g = spans.size();
        spans.push_back({static_cast<std::uint32_t>(np[r].length - 1), static_cast<std::uint32_t>(np[r].length)});
    }
    tok_begin.push_back(0);
    for (std::size_t r = 0; r < R; ++r) {
        req_first.push_back(u32(np[r].field_off));
        req_count.push_back(u32(np[r].fields));
        for (const EncodedQuestion& q : requests[r]->questions) {
            q_first.push_back(u32(oi));
            q_count.push_back(u32(q.options.size()));
            q_request.push_back(u32(r));
            q_type.push_back(static_cast<std::uint32_t>(q.type));
            if (static_cast<std::uint32_t>(q.type) > 2) throw std::invalid_argument("clef head: bad question type");
            for (const TokenSpan& s : q.options) {
                opt_question.push_back(u32(qi));
                for (std::uint32_t t = s.begin; t < s.end; ++t) {
                    const std::uint32_t tok = requests[r]->token_ids[t];
                    auto [it, fresh] = unique.try_emplace(tok, u32(unique_tokens.size()));
                    if (fresh) unique_tokens.push_back(tok);
                    tok_index.push_back(it->second);
                }
                tok_begin.push_back(u32(tok_index.size()));
                ++oi;
            }
            ++qi;
        }
    }
    if (unique_tokens.size() > cap.unique_tokens) throw std::invalid_argument("clef head: " + std::to_string(unique_tokens.size()) + " distinct option tokens (capacity " + std::to_string(cap.unique_tokens) + ")");

    std::size_t max_per_q = 0;
    for (const std::uint32_t c : q_count) max_per_q = std::max<std::size_t>(max_per_q, c);

    // lexical rows
    std::vector<float> rows(unique_tokens.size() * Hd);
    for (std::size_t u = 0; u < unique_tokens.size(); ++u) embedding_row(context, unique_tokens[u], rows.data() + u * Hd);

    // upload the tables
    std::vector<std::uint32_t> t;
    auto put = [&](const std::vector<std::uint32_t>& v) {
        const std::size_t at = t.size();
        t.insert(t.end(), v.begin(), v.end());
        while (t.size() % 4 != 0) t.push_back(0);
        return at;
    };
    t_q_first = put(q_first);
    t_q_count = put(q_count);
    t_q_request = put(q_request);
    t_q_type = put(q_type);
    t_opt_question = put(opt_question);
    t_q_req_first = put(req_first);
    t_q_req_count = put(req_count);
    t_tok_begin = put(tok_begin);
    t_tok_index = put(tok_index);
    if (t.size() > table.size()) throw std::invalid_argument("clef head: table overflow");
    check_cuda(cudaMemcpyAsync(table.data(), t.data(), t.size() * 4, cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync");
    check_cuda(cudaMemcpyAsync(span_table.data(), spans.data(), spans.size() * sizeof(ClefSpan), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync");
    if (!rows.empty()) check_cuda(cudaMemcpyAsync(urows.data(), rows.data(), rows.size() * 4, cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync");
    check_cuda(cudaMemsetAsync(acc.data(), 0, spans.size() * Hd * 4, stream), "cudaMemsetAsync");
    // pageable host memory is staged before the async copies return, but the vectors above go out of scope: wait now.
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

    reqs.assign(requests.begin(), requests.end());
    plans = std::move(np);
    mem_used = mem_total;
    n_fields = fields;
    n_options = options;
    n_spans = spans.size();
    max_options_per_q = max_per_q;
    max_fields_per_req = max_fields;
    planned = true;
}

void CudaClefHead::Impl::consume(const float* hidden, std::size_t rows, std::span<const HeadSegment> segments) {
    if (!planned) throw std::logic_error("clef head: consume before begin");
    if (rows == 0 || segments.empty()) return;
    if (rows > cap.call_tokens) throw std::invalid_argument("clef head: " + std::to_string(rows) + " rows in one consume (capacity " + std::to_string(cap.call_tokens) + ")");
    for (const HeadSegment& s : segments) {
        if (s.request >= plans.size() || s.first_token + s.count > plans[s.request].length || s.row + s.count > rows)
            throw std::invalid_argument("clef head: bad segment");
    }
    // X = LayerNorm(hidden)
    clef_layer_norm(hidden, Hd, xscr.data(), Hd, rows, Hd, hidden_norm.w, hidden_norm.b, 1e-5f, stream);
    // memory = X W_mem^T  (all rows at once), then each segment goes to its place in the arena
    linear(xscr.data(), rows, Hd, mem_proj, W, nullptr, mtmp.data());
    for (const HeadSegment& s : segments) {
        const Plan& p = plans[s.request];
        check_cuda(cudaMemcpyAsync(mem.data() + (p.mem_off + s.first_token) * W, mtmp.data() + s.row * W, s.count * W * 4, cudaMemcpyDeviceToDevice, stream),
                   "cudaMemcpyAsync");
        const std::uint32_t first = u32(s.first_token), n = u32(s.count);
        const float* x = xscr.data() + s.row * Hd;
        const std::size_t nq = reqs[s.request]->questions.size();
        std::size_t no = 0;
        for (const EncodedQuestion& q : reqs[s.request]->questions) no += q.options.size();
        const std::size_t last = plans[s.request].length;
        if (s.first_token + s.count > plans[s.request].min_begin) {
            clef_span_accumulate(span_table.data(), p.span_q, nq, first, n, x, Hd, acc.data(), Hd, stream);
            clef_span_accumulate(span_table.data(), p.span_p, no, first, n, x, Hd, acc.data(), Hd, stream);
        }
        if (s.first_token + s.count == last) clef_span_accumulate(span_table.data(), p.span_g, 1, first, n, x, Hd, acc.data(), Hd, stream);
    }
}

std::vector<RequestLogits> CudaClefHead::Impl::finish() {
    if (!planned) throw std::logic_error("clef head: finish before begin");
    const std::size_t R = plans.size(), Qn = n_fields, Pn = n_options;
    auto tab = [&](std::size_t off) { return table.data() + off; };
    // Span means: acc rows [fields][options][requests] -> q_vec, ctx, global
    clef_span_finalize(span_table.data(), n_spans, acc.data(), Hd, stream);
    const float* qvec = acc.data();
    const float* ctxv = acc.data() + Qn * Hd;
    const float* globv = acc.data() + (Qn + Pn) * Hd;
    clef_lexical_means(tab(t_tok_begin), tab(t_tok_index), urows.data(), Pn, Hd, lex.data(), stream);

    // normalised memory (no affine part), once for every routing layer
    clef_layer_norm(mem.data(), W, nmem.data(), W, mem_used, W, nullptr, nullptr, 1e-5f, stream);

    // option queries = W_ctx ctx + W_lex lex + W_oq q_vec[question of option]
    {
        const float one = 1.0f;
        linear(ctxv, Pn, Hd, ctx_proj, W, nullptr, ox.data());
        check_blas(cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(W), static_cast<int>(Pn), static_cast<int>(Hd), &one, lex_proj,
                               static_cast<int>(Hd), lex.data(), static_cast<int>(Hd), &one, ox.data(), static_cast<int>(W)),
                   "cublasSgemm");
        linear(qvec, Qn, Hd, oq_proj, W, nullptr, tmpq.data());
        clef_gather_add(ox.data(), tmpq.data(), tab(t_opt_question), Pn, W, stream);
    }
    linear(qvec, Qn, Hd, q_proj, W, nullptr, base.data());
    linear(globv, R, Hd, g_proj, W, nullptr, glob.data());

    std::vector<QuerySeg> opt_segs(R), field_segs(R);
    for (std::size_t r = 0; r < R; ++r) {
        opt_segs[r] = {plans[r].option_off, plans[r].options, plans[r].mem_off, plans[r].length};
        field_segs[r] = {plans[r].field_off, plans[r].fields, plans[r].mem_off, plans[r].length};
    }

    for (const RoutingLayer& l : routing) {
        layer_norm(ox.data(), oln.data(), Pn, W, l.query_norm);
        cross_attention(oln.data(), Pn, l.attn, nmem.data(), l.memory_norm.w, l.memory_norm.b, opt_segs, ao.data());
        clef_add(ox.data(), ao.data(), Pn * W, stream);
        layer_norm(ox.data(), oln.data(), Pn, W, l.ff_norm);
        feed_forward(oln.data(), Pn, l.ff, ao.data());
        clef_add(ox.data(), ao.data(), Pn * W, stream);
    }

    // per-field summary of its routed options
    clef_option_summary(ox.data(), base.data(), tab(t_q_first), tab(t_q_count), Qn, W, tmpq.data(), stream, max_options_per_q);
    layer_norm(tmpq.data(), fln.data(), Qn, W, summary_norm);
    clef_init_fields(base.data(), fln.data(), glob.data(), tab(t_q_request), type_emb, tab(t_q_type), Qn, W, f.data(), stream);

    for (const DecoderLayer& l : decoder) {
        layer_norm(f.data(), fln.data(), Qn, W, l.norm1);
        linear(fln.data(), Qn, W, l.self_attn.in_w, 3 * W, l.self_attn.in_b, qkv.data());
        clef_self_attention(qkv.data(), tab(t_q_request), tab(t_q_req_first), tab(t_q_req_count), Qn, W, nh, att.data(), stream, max_fields_per_req);
        linear(att.data(), Qn, W, l.self_attn.out_w, W, l.self_attn.out_b, ao.data());
        clef_add(f.data(), ao.data(), Qn * W, stream);

        layer_norm(f.data(), fln.data(), Qn, W, l.norm2);
        cross_attention(fln.data(), Qn, l.cross_attn, mem.data(), nullptr, nullptr, field_segs, ao.data());
        clef_add(f.data(), ao.data(), Qn * W, stream);

        layer_norm(f.data(), fln.data(), Qn, W, l.norm3);
        feed_forward(fln.data(), Qn, l.ff, ao.data());
        clef_add(f.data(), ao.data(), Qn * W, stream);
    }
    layer_norm(f.data(), fln.data(), Qn, W, field_norm);
    layer_norm(ox.data(), oln.data(), Pn, W, option_norm);
    clef_scorer_features(fln.data(), oln.data(), tab(t_opt_question), Pn, W, features.data(), cosine.data(), stream);
    linear(features.data(), Pn, 4 * W, scorer_w1, W, scorer_b1, hid2.data());
    clef_gelu(hid2.data(), Pn * W, stream);
    clef_final_logits(lex.data(), qvec, globv, tab(t_opt_question), tab(t_q_request), cosine.data(), hid2.data(), scorer_w2, scorer_b2, Pn, Hd, W,
                      prior_scale, joint_scale, gate, logits.data(), stream);

    std::vector<float> host(Pn);
    check_cuda(cudaMemcpyAsync(host.data(), logits.data(), Pn * 4, cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync");
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

    std::vector<RequestLogits> out(R);
    std::size_t at = 0;
    for (std::size_t r = 0; r < R; ++r) {
        for (const EncodedQuestion& q : reqs[r]->questions) {
            out[r].logits.emplace_back(host.begin() + static_cast<std::ptrdiff_t>(at), host.begin() + static_cast<std::ptrdiff_t>(at + q.options.size()));
            at += q.options.size();
        }
    }
    planned = false;
    return out;
}

// ---- public surface ------------------------------------------------------------------------------------------

std::size_t clef_head_device_bytes(const ClefHeadConfig& cfg, const ClefHeadCapacity& cap) {
    const std::size_t W = cfg.width, Hd = cfg.hidden_size, FF = cfg.feedforward, nh = cfg.heads;
    const std::size_t S = cap.fields + cap.options + cap.requests;
    const std::size_t Q = std::max(cap.options, cap.fields);
    std::size_t floats = 2 * cap.memory_tokens * W + cap.call_tokens * (Hd + W) + S * Hd + cap.options * Hd + cap.unique_tokens * Hd;
    floats += cap.fields * W * 2 + cap.requests * W + cap.options * W + Q * W * 3 + 2 * nh * Q * W + Q * FF;
    floats += cap.fields * W * 2 + cap.fields * 3 * W + cap.options * 4 * W + cap.options * W + 2 * cap.options + cap.score_floats;
    floats += 8 * cap.fields + 6 * cap.requests + 2 * cap.options + cap.option_tokens + 16 + S * 2;
    // weights: about 23 (hidden * width) + layers * (4 attention matrices + 2 feed-forward matrices) with a margin
    const std::size_t weights = (6 * Hd * W + (cfg.routing_layers + cfg.layers) * (4 * W * W + 2 * W * FF) + cfg.layers * 4 * W * W + 4 * W * W) * 1;
    return (floats + weights) * 4 + (std::size_t{64} << 20);
}

CudaClefHead::CudaClefHead(const ClefHeadWeights& weights, const ClefHeadCapacity& capacity, void* stream)
    : impl_(std::make_unique<Impl>(weights, capacity, stream)) {}
CudaClefHead::~CudaClefHead() = default;
const ClefHeadConfig& CudaClefHead::config() const { return impl_->cfg; }
const ClefHeadCapacity& CudaClefHead::capacity() const { return impl_->cap; }
std::size_t CudaClefHead::width() const { return impl_->W; }
void CudaClefHead::begin(std::span<const EncodedRequest* const> requests, EmbeddingRowFn embedding_row, const void* context) {
    impl_->begin(requests, embedding_row, context);
}
void CudaClefHead::consume(const float* hidden, std::size_t rows, std::span<const HeadSegment> segments) { impl_->consume(hidden, rows, segments); }
void CudaClefHead::adopt(std::size_t request, const float* memory, std::size_t count) {
    Impl& m = *impl_;
    if (!m.planned || request >= m.plans.size() || count > m.plans[request].length) throw std::invalid_argument("clef head: bad adopt");
    check_cuda(cudaMemcpyAsync(m.mem.data() + m.plans[request].mem_off * m.W, memory, count * m.W * 4, cudaMemcpyDeviceToDevice, m.stream), "cudaMemcpyAsync");
}
const float* CudaClefHead::memory_rows(std::size_t request) const {
    const Impl& m = *impl_;
    if (!m.planned || request >= m.plans.size()) throw std::invalid_argument("clef head: bad request");
    return m.mem.data() + m.plans[request].mem_off * m.W;
}
std::vector<RequestLogits> CudaClefHead::finish() { return impl_->finish(); }

} // namespace sepal::cuda
