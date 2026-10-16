// Cloudflare's Clef JointSchemaHead (Apache-2.0; RUNNER.md 3.3-3.5), implemented from its published description and
// the Apache-2.0 Clef source (Cloudflare/clef-flash@fde727a joint_schema_model.py). No Unsloth (AGPL) code is copied.
//
// Inputs per request: the final-normed hidden states of EVERY position [L, hidden] (the text model's
// last_hidden_state), the request's spans and types, and output-embedding rows for the option tokens.
// Structure (Clef `JointSchemaHead.forward`):
//   h = LayerNorm(hidden) (eps 1e-5, PyTorch default); memory = W_mem h  [L, width]; global = h[L-1]
//   q_vec[q] = mean h[instruction span]; ctx[o] = mean h[option span]; lex[o] = mean E[option token ids]
//   query[o] = W_ctx ctx + W_lex lex + W_oq q_vec[question of o]
//   routing layers (EvidenceRoutingLayer): x += MHA(LN_q(x), LN_m(memory), LN_m(memory)); x += FF(LN_ff(x)), FF = Linear-GELU(erf)-Linear
//   base[q] = W_q q_vec; summary[q] = sum_k softmax_k(opt_k . base / sqrt(width)) opt_k
//   field[q] = base + LN_sum(summary) + W_g global + type_embedding[type]
//   decoder layers (nn.TransformerDecoderLayer, norm_first, GELU): x += SA(LN1 x) (no mask, over the Q fields);
//                   x += MHA(LN2 x, memory, memory) (memory NOT normed); x += FF(LN3 x)
//   field = LN_field(field); opt = LN_opt(routed opt)
//   logit = s_prior * cos(lex_k, normalize(q_vec + global)) + sigmoid(gate) * (s_joint * cos(field, opt_k) + MLP([f, o, f*o, |f-o|]))
//   s_prior = exp(min(prior_logit_scale, log 100)), s_joint likewise; MLP = Linear(4w, w)-GELU-Linear(w, 1)
// MultiheadAttention: in_proj_weight [3w, w] (q, k, v stacked) + in_proj_bias; out_proj with bias; heads = cfg.heads,
// scores scaled by 1/sqrt(width/heads). cosine_similarity eps 1e-8 (clamped per norm product as PyTorch does);
// F.normalize eps 1e-12.
#pragma once

#include <cstdint>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "sepal/decision.hpp"

namespace sepal {

struct ClefHeadConfig {
    std::uint32_t hidden_size = 0;
    std::uint32_t width = 0;
    std::uint32_t routing_layers = 0;
    std::uint32_t layers = 0;
    std::uint32_t heads = 0;
    std::uint32_t feedforward = 0;
    /// Parses joint_head_config.json text; throws std::invalid_argument on a missing or invalid field.
    static ClefHeadConfig parse(std::string_view json_text);
};

/// Every head parameter in float32 under its state-dict name (e.g. "evidence_layers.0.attention.in_proj_weight",
/// "layers.3.norm2.bias", "prior_logit_scale" with shape {}). `open` checks names and shapes against the config
/// strictly (122 tensors at the 0.8B config).
struct ClefHeadWeights {
    ClefHeadConfig config;
    std::map<std::string, std::vector<float>, std::less<>> tensors;
    std::map<std::string, std::vector<std::size_t>, std::less<>> shapes;
    static ClefHeadWeights open(const std::string& safetensors_path, const ClefHeadConfig& config);
    const std::vector<float>& at(std::string_view name) const;
};

/// Supplies output-embedding row `token` (hidden_size floats) into `out`.
using EmbeddingRowFn = void (*)(const void* context, std::uint32_t token, float* out);

/// The reference head, float64 accumulation throughout, single-threaded: the parity oracle for every fast head.
/// `hidden` is [L, hidden_size] row-major with L == request.token_ids.size().
RequestLogits clef_head_reference(const ClefHeadWeights& weights, std::span<const float> hidden,
                                  const EncodedRequest& request, EmbeddingRowFn embedding_row, const void* context);

/// Checks a request against L and the config (non-empty spans inside [0, L), at least one question and option).
/// Throws std::invalid_argument naming the problem.
void check_request(const EncodedRequest& request, std::uint64_t vocab_size);

} // namespace sepal
