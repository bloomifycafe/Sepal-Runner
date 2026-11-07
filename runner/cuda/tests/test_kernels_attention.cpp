// Tests of the attention kernels (kernels_attention.hpp) against the reference mixer in sepal/ref/attention.hpp.
// Needs a CUDA device.
//
//   sepal_cuda_kernel_attention_tests              every suite
//   sepal_cuda_kernel_attention_tests <suite>      one of: stages, geometry, continuation, shared_context,
//                                                  causality, invariance, positions, rope_exact, tables, split
//
// Four variants of the device path are tested everywhere: F32 keys and values with the plain kernel, F16 with the
// plain kernel, F16 with the tiled tensor-core kernel, and F32 with the tiled float32 kernel of the Pascal profiles. Lines starting with "  [measured]" are the record the
// report quotes. The matrix multiplies that feed the mixer are not under test: the tests fill the q, k and v
// bands of a "multiply output" with random numbers.

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "kernels_attention.hpp"
#include "sepal/config.hpp"
#include "sepal/gemm.hpp"
#include "sepal/ref/attention.hpp"
#include "sepal/ref/primitives.hpp"
#include "sepal/ref/snapshot.hpp"
#include "test.hpp"

namespace {

using sepal::ModelConfig;
using sepal::cuda::TiledKernel;
using sepal::cuda::reset_tiled_kernel;
using sepal::cuda::set_tiled_kernel;
using sepal::cuda::AttentionPath;
using sepal::cuda::AttentionRowDesc;
using sepal::cuda::AttentionRows;
using sepal::cuda::AttentionShape;
using sepal::cuda::AttnElem;
using sepal::cuda::ConstElemMatrix;
using sepal::cuda::DeviceBuffer;
using sepal::cuda::ElemMatrix;
using sepal::cuda::FloatBand;
using sepal::cuda::KvSegment;
using sepal::cuda::RowSpan;

// ---- numbers ----

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

// binary16 on the host: round to nearest, ties to even, like __float2half_rn.
std::uint16_t half_bits(float f) {
    const std::uint32_t x = std::bit_cast<std::uint32_t>(f);
    const std::uint32_t sign = (x >> 16) & 0x8000u;
    if ((x & 0x7FFFFFFFu) > 0x7F800000u) return static_cast<std::uint16_t>(sign | 0x7E00u);
    const int exponent = static_cast<int>((x >> 23) & 0xFFu) - 127 + 15;
    std::uint32_t mantissa = x & 0x7FFFFFu;
    if (exponent >= 31) return static_cast<std::uint16_t>(sign | 0x7C00u);
    if (exponent <= 0) {
        if (exponent < -10) return static_cast<std::uint16_t>(sign);
        mantissa |= 0x800000u;
        const int shift = 14 - exponent;
        std::uint32_t h = mantissa >> shift;
        const std::uint32_t rest = mantissa & ((1u << shift) - 1u);
        const std::uint32_t halfway = 1u << (shift - 1);
        if (rest > halfway || (rest == halfway && (h & 1u))) ++h;
        return static_cast<std::uint16_t>(sign | h);
    }
    std::uint32_t h = (static_cast<std::uint32_t>(exponent) << 10) | (mantissa >> 13);
    const std::uint32_t rest = mantissa & 0x1FFFu;
    if (rest > 0x1000u || (rest == 0x1000u && (h & 1u))) ++h;
    return static_cast<std::uint16_t>(sign | h);
}

float half_value(std::uint16_t h) {
    const std::uint32_t sign = (static_cast<std::uint32_t>(h) & 0x8000u) << 16;
    const std::uint32_t exponent = (h >> 10) & 0x1Fu;
    const std::uint32_t mantissa = h & 0x3FFu;
    if (exponent == 0) {
        const float magnitude = std::ldexp(static_cast<float>(mantissa), -24);
        return sign != 0 ? -magnitude : magnitude;
    }
    if (exponent == 31) return std::bit_cast<float>(sign | 0x7F800000u | (mantissa << 13));
    return std::bit_cast<float>(sign | ((exponent + 112u) << 23) | (mantissa << 13));
}

float round_half(float f) { return half_value(half_bits(f)); }

constexpr double kHalfUnitRoundoff = 4.8828125e-4; // 2^-11

std::size_t elem_size(AttnElem type) { return type == AttnElem::F16 ? 2 : 4; }

// ---- geometry ----

struct Geometry {
    std::string name;
    ModelConfig cfg;
    AttentionShape shape{};
    std::uint32_t qg_width = 0; // 2 * heads * head_dim
    std::uint32_t q_width = 0;  // heads * head_dim
    std::uint32_t kv_width = 0; // kv_heads * head_dim
};

Geometry make_geometry(const ModelConfig& cfg) {
    Geometry g;
    g.name = cfg.name;
    g.cfg = cfg;
    g.shape = {static_cast<std::uint32_t>(cfg.attn_heads), static_cast<std::uint32_t>(cfg.attn_kv_heads),
               static_cast<std::uint32_t>(cfg.attn_head_dim), static_cast<std::uint32_t>(cfg.attn_rotary_dims()),
               cfg.rope_theta, cfg.rms_norm_eps};
    g.qg_width = 2 * g.shape.heads * g.shape.head_dim;
    g.q_width = g.shape.heads * g.shape.head_dim;
    g.kv_width = g.shape.kv_heads * g.shape.head_dim;
    return g;
}

const Geometry& tiny() {
    static const Geometry g = make_geometry(sepal::tiny_test_config());
    return g;
}
const Geometry& preset(const char* name) {
    static std::map<std::string, Geometry> cache;
    auto it = cache.find(name);
    if (it == cache.end()) it = cache.emplace(name, make_geometry(*sepal::find_preset(name))).first;
    return it->second;
}

struct Variant {
    AttnElem kv;
    AttentionPath path;
    const char* name;
    int f32 = -1; // TiledF32: the float32 kernel (F32Kernel) forced for the launch; -1 = the environment's choice
};
// On a card without tensor cores (Pascal) the f16/tiled variant is left out: the others run everywhere. f32/tiled is the first float32
// kernel, f32/split the second (at head_dim 256; other head sizes run the first under either name).
const std::vector<Variant>& variants() {
    static const std::vector<Variant> v = [] {
        std::vector<Variant> all = {{AttnElem::F32, AttentionPath::Reference, "f32/plain"},
                                    {AttnElem::F16, AttentionPath::Reference, "f16/plain"},
                                    {AttnElem::F16, AttentionPath::Tiled, "f16/tiled"},
                                    {AttnElem::F32, AttentionPath::TiledF32, "f32/tiled", 0},
                                    {AttnElem::F32, AttentionPath::TiledF32, "f32/split", 1}};
        if (!sepal::cuda::device_has_tensor_cores()) all.erase(all.begin() + 2);
        return all;
    }();
    return v;
}
// The variants that read binary16 keys and values (f16/plain, and f16/tiled where it runs).
std::vector<const Variant*> f16_variants() {
    std::vector<const Variant*> out;
    for (const Variant& v : variants())
        if (v.kv == AttnElem::F16) out.push_back(&v);
    return out;
}

// ---- random inputs ----

// A pool of logical tokens: the three bands of the multiply output of each, addressed by id, so that a token can
// appear in different rows of different runs with the same numbers.
struct Pool {
    const Geometry* geo;
    std::vector<float> qg, k, v;
    Pool(const Geometry& g, std::size_t count, std::uint64_t seed) : geo(&g) {
        Rng rng(seed);
        qg.resize(count * g.qg_width);
        k.resize(count * g.kv_width);
        v.resize(count * g.kv_width);
        for (float& x : qg) x = rng.uniform(-2.0f, 2.0f);
        for (float& x : k) x = rng.uniform(-2.0f, 2.0f);
        for (float& x : v) x = rng.uniform(-1.0f, 1.0f);
    }
    std::size_t size() const { return k.size() / geo->kv_width; }
};

struct NormWeights {
    std::vector<float> q, k;
};
NormWeights make_weights(const Geometry& g, std::uint64_t seed) {
    Rng rng(seed);
    NormWeights w;
    w.q.resize(g.shape.head_dim);
    w.k.resize(g.shape.head_dim);
    for (float& x : w.q) x = rng.uniform(-0.3f, 0.3f);
    for (float& x : w.k) x = rng.uniform(-0.3f, 0.3f);
    return w;
}

// ---- the plan of a pass ----

// A run of tokens' keys and values. External nodes carry numbers (representable in binary16, so that every variant
// stores them exactly); the others are computed by a row of the pass.
struct Node {
    std::shared_ptr<Node> parent;
    std::uint32_t tokens = 0;
    bool external = false;
    std::vector<float> keys, values; // [tokens, kv_heads, head_dim]
    int producer = -1;               // the row of the current pass that fills it
};
using NodePtr = std::shared_ptr<Node>;

NodePtr external_node(const Geometry& g, std::uint32_t tokens, std::uint64_t seed, NodePtr parent) {
    Rng rng(seed);
    auto n = std::make_shared<Node>();
    n->parent = std::move(parent);
    n->tokens = tokens;
    n->external = true;
    n->keys.resize(static_cast<std::size_t>(tokens) * g.kv_width);
    n->values.resize(n->keys.size());
    for (float& x : n->keys) x = round_half(rng.uniform(-1.7f, 1.7f));
    for (float& x : n->values) x = round_half(rng.uniform(-1.0f, 1.0f));
    return n;
}

struct RowPlan {
    std::vector<std::uint32_t> ids; // logical tokens of the row
    std::uint32_t start = 0;
    NodePtr parent;
    NodePtr node; // the row's own keys and values

    std::uint32_t length() const { return static_cast<std::uint32_t>(ids.size()); }
};

RowPlan make_row(std::vector<std::uint32_t> ids, std::uint32_t start, NodePtr parent) {
    RowPlan r;
    r.ids = std::move(ids);
    r.start = start;
    r.parent = std::move(parent);
    r.node = std::make_shared<Node>();
    r.node->parent = r.parent;
    r.node->tokens = r.length();
    return r;
}

std::vector<std::uint32_t> iota_ids(std::uint32_t first, std::uint32_t count) {
    std::vector<std::uint32_t> ids(count);
    for (std::uint32_t i = 0; i < count; ++i) ids[i] = first + i;
    return ids;
}

std::uint32_t chain_tokens(const NodePtr& n) {
    std::uint32_t total = 0;
    for (const Node* p = n.get(); p != nullptr; p = p->parent.get()) total += p->tokens;
    return total;
}

// ---- running the device path ----

struct RowOut {
    std::vector<float> q;   // [length, q_width]
    std::vector<float> keys, values;
    std::vector<float> out; // [length, q_width]
};
struct PassResult {
    std::vector<RowOut> rows;
    AttentionPath used = AttentionPath::Reference;
    std::vector<unsigned char> out_bytes; // the whole output buffer, to look for stray writes
    std::vector<std::size_t> row_offsets;
};

struct PassOptions {
    std::vector<std::uint32_t> run; // empty: every row
    bool gate_open = false;         // overwrite every gate with 30, for which sigmoid is exactly 1 in float32
    AttnElem out_type_override = AttnElem::F32;
    bool use_out_override = false;
    bool indexed = false; // the segments carry `before` and the kernels find a key by binary search
    bool give_max_keys = true; // fill AttentionRows::max_row_keys (false: the launch reads it back from the device)
};

std::vector<unsigned char> to_bytes(const std::vector<float>& values, AttnElem type) {
    std::vector<unsigned char> bytes(values.size() * elem_size(type));
    if (type == AttnElem::F16) {
        for (std::size_t i = 0; i < values.size(); ++i) {
            const std::uint16_t h = half_bits(values[i]);
            std::memcpy(bytes.data() + 2 * i, &h, 2);
        }
    } else {
        std::memcpy(bytes.data(), values.data(), bytes.size());
    }
    return bytes;
}

std::vector<float> from_bytes(const unsigned char* bytes, std::size_t count, AttnElem type) {
    std::vector<float> values(count);
    for (std::size_t i = 0; i < count; ++i) {
        if (type == AttnElem::F16) {
            std::uint16_t h;
            std::memcpy(&h, bytes + 2 * i, 2);
            values[i] = half_value(h);
        } else {
            std::memcpy(&values[i], bytes + 4 * i, 4);
        }
    }
    return values;
}

PassResult run_device(const Geometry& g, const Pool& pool, const NormWeights& weights, std::vector<RowPlan>& rows,
                      const Variant& variant, const PassOptions& options = {}) {
    const AttnElem kv = variant.kv;
    const AttnElem out_type = options.use_out_override ? options.out_type_override : variant.kv;
    const std::size_t esize = elem_size(kv);
    const std::size_t nrows = rows.size();

    // The flat token list.
    std::vector<RowSpan> spans(nrows);
    std::uint32_t total = 0;
    for (std::size_t r = 0; r < nrows; ++r) {
        spans[r] = {total, rows[r].length()};
        total += rows[r].length();
        rows[r].node->producer = static_cast<int>(r);
    }
    // The multiply output: [q with gate | k | v | 8 columns that nothing may read].
    const std::size_t mult_stride = g.qg_width + 2 * g.kv_width + 8;
    std::vector<float> mult(static_cast<std::size_t>(total) * mult_stride, std::nanf(""));
    for (std::size_t r = 0; r < nrows; ++r) {
        for (std::uint32_t t = 0; t < rows[r].length(); ++t) {
            const std::size_t id = rows[r].ids[t];
            float* dst = mult.data() + (static_cast<std::size_t>(spans[r].offset) + t) * mult_stride;
            std::copy_n(pool.qg.data() + id * g.qg_width, g.qg_width, dst);
            if (options.gate_open) {
                for (std::uint32_t h = 0; h < g.shape.heads; ++h) {
                    std::fill_n(dst + h * 2 * g.shape.head_dim + g.shape.head_dim, g.shape.head_dim, 30.0f);
                }
            }
            std::copy_n(pool.k.data() + id * g.kv_width, g.kv_width, dst + g.qg_width);
            std::copy_n(pool.v.data() + id * g.kv_width, g.kv_width, dst + g.qg_width + g.kv_width);
        }
    }
    DeviceBuffer<float> d_mult(mult);
    DeviceBuffer<float> d_qn(weights.q), d_kn(weights.k);

    // Segments: external nodes are uploaded once per pass; the others are the producing row's buffers.
    std::map<const Node*, std::pair<DeviceBuffer<unsigned char>, DeviceBuffer<unsigned char>>> buffers;
    for (std::size_t r = 0; r < nrows; ++r) {
        const std::size_t bytes = static_cast<std::size_t>(rows[r].length()) * g.kv_width * esize;
        buffers[rows[r].node.get()] = {DeviceBuffer<unsigned char>(bytes), DeviceBuffer<unsigned char>(bytes)};
    }
    std::vector<AttentionRowDesc> descs(nrows);
    std::vector<KvSegment> segments;
    for (std::size_t r = 0; r < nrows; ++r) {
        std::vector<const Node*> chain;
        for (const Node* n = rows[r].node.get(); n != nullptr; n = n->parent.get()) chain.push_back(n);
        std::reverse(chain.begin(), chain.end());
        descs[r] = {static_cast<std::uint32_t>(segments.size()), static_cast<std::uint32_t>(chain.size()), rows[r].start, options.indexed ? 1u : 0u};
        std::uint32_t before = 0;
        for (const Node* n : chain) {
            if (n->external) {
                auto it = buffers.find(n);
                if (it == buffers.end()) {
                    it = buffers
                             .emplace(n, std::make_pair(DeviceBuffer<unsigned char>(to_bytes(n->keys, kv)),
                                                        DeviceBuffer<unsigned char>(to_bytes(n->values, kv))))
                             .first;
                }
            } else if (buffers.find(n) == buffers.end()) {
                throw std::runtime_error("a node computed outside this pass is used as context; externalize it");
            }
            auto& pair = buffers.at(n);
            segments.push_back({pair.first.data(), pair.second.data(), n->tokens, options.indexed ? before : 0u});
            before += n->tokens;
        }
    }
    sepal::cuda::check_attention_rows(spans, descs, segments);

    const std::vector<std::uint32_t> run = options.run; // empty: every row
    std::uint32_t max_len = 0;
    if (run.empty()) {
        for (const RowPlan& row : rows) max_len = std::max(max_len, row.length());
    } else {
        for (std::uint32_t r : run) max_len = std::max(max_len, rows[r].length());
    }
    DeviceBuffer<RowSpan> d_spans(spans);
    DeviceBuffer<AttentionRowDesc> d_descs(descs);
    DeviceBuffer<KvSegment> d_segments(segments);
    DeviceBuffer<std::uint32_t> d_run(run);
    AttentionRows table{d_spans.data(), d_descs.data(), d_segments.data(), run.empty() ? nullptr : d_run.data(),
                        run.empty() ? static_cast<std::uint32_t>(nrows) : static_cast<std::uint32_t>(run.size()), max_len};
    if (options.give_max_keys) {
        std::uint32_t max_keys = 0;
        for (std::size_t r = 0; r < nrows; ++r) {
            if (!run.empty() && std::find(run.begin(), run.end(), static_cast<std::uint32_t>(r)) == run.end()) continue;
            max_keys = std::max(max_keys, chain_tokens(rows[r].parent) + rows[r].length());
        }
        table.max_row_keys = max_keys;
    }

    const std::size_t q_stride = g.q_width + 16;
    const std::size_t out_stride = g.q_width + 8;
    DeviceBuffer<unsigned char> d_q(static_cast<std::size_t>(total) * q_stride * esize);
    DeviceBuffer<unsigned char> d_out(static_cast<std::size_t>(total) * out_stride * elem_size(out_type));
    d_q.zero();
    {
        const std::vector<unsigned char> sentinel(d_out.size(), 0x5A);
        d_out.upload(sentinel.data(), sentinel.size());
    }

    const float* base = d_mult.data();
    const FloatBand q_gate{base, mult_stride};
    const FloatBand k_band{base + g.qg_width, mult_stride};
    const FloatBand v_band{base + g.qg_width + g.kv_width, mult_stride};
    cudaStream_t stream = nullptr;
    sepal::cuda::launch_attention_prepare(g.shape, table, kv, q_gate, k_band, v_band, d_qn.data(), d_kn.data(),
                                          {d_q.data(), q_stride}, stream);
    PassResult result;
    if (variant.f32 >= 0) sepal::cuda::set_f32_kernel(static_cast<sepal::cuda::F32Kernel>(variant.f32));
    result.used = sepal::cuda::launch_attention(g.shape, table, kv, variant.path, {d_q.data(), q_stride}, q_gate,
                                                {d_out.data(), out_stride}, out_type, stream);
    if (variant.f32 >= 0) sepal::cuda::reset_f32_kernel();
    sepal::cuda::check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

    const std::vector<unsigned char> q_bytes = d_q.download();
    result.out_bytes = d_out.download();
    result.rows.resize(nrows);
    for (std::size_t r = 0; r < nrows; ++r) {
        RowOut& row = result.rows[r];
        const std::size_t len = rows[r].length();
        result.row_offsets.push_back(spans[r].offset);
        for (std::size_t t = 0; t < len; ++t) {
            const std::size_t token = spans[r].offset + t;
            const std::vector<float> q = from_bytes(q_bytes.data() + token * q_stride * esize, g.q_width, kv);
            row.q.insert(row.q.end(), q.begin(), q.end());
            const std::vector<float> o = from_bytes(result.out_bytes.data() + token * out_stride * elem_size(out_type),
                                                    g.q_width, out_type);
            row.out.insert(row.out.end(), o.begin(), o.end());
        }
        auto& pair = buffers.at(rows[r].node.get());
        row.keys = from_bytes(pair.first.download().data(), len * g.kv_width, kv);
        row.values = from_bytes(pair.second.download().data(), len * g.kv_width, kv);
    }
    return result;
}

// Makes a node that a row of an earlier pass computed into an external one holding exactly the numbers it stored.
NodePtr externalize(const RowPlan& row, const RowOut& out, NodePtr parent) {
    auto n = std::make_shared<Node>();
    n->parent = std::move(parent);
    n->tokens = row.length();
    n->external = true;
    n->keys = out.keys;
    n->values = out.values;
    return n;
}

// ---- the reference side ----

template <class T>
struct RefRow {
    std::vector<T> q, keys, values, attended, out;
    sepal::ref::AttentionStatePtr<T> state;
};

template <class T>
class RefPass {
public:
    RefPass(const Geometry& g, const Pool& pool, const NormWeights& weights, const std::vector<RowPlan>& rows)
        : g_(g), pool_(pool), rows_(rows), provider_(sepal::make_reference_gemm()) {
        for (float w : weights.q) q_norm_.push_back(static_cast<T>(w));
        for (float w : weights.k) k_norm_.push_back(static_cast<T>(w));
        for (std::size_t r = 0; r < rows_.size(); ++r) by_node_[rows_[r].node.get()] = r;
    }

    const RefRow<T>& row(std::size_t r) {
        auto it = done_.find(r);
        if (it != done_.end()) return it->second;
        return done_.emplace(r, compute(r)).first->second;
    }

private:
    sepal::ref::AttentionStatePtr<T> state_of(const NodePtr& node) {
        if (!node) return std::make_shared<sepal::ref::AttentionState<T>>();
        if (!node->external) return row(by_node_.at(node.get())).state;
        auto it = external_.find(node.get());
        if (it != external_.end()) return it->second;
        auto s = std::make_shared<sepal::ref::AttentionState<T>>();
        if (node->parent) s->parent = state_of(node->parent);
        const std::vector<std::size_t> shape = {node->tokens, g_.shape.kv_heads, g_.shape.head_dim};
        s->keys = sepal::BasicTensor<T>(shape);
        s->values = sepal::BasicTensor<T>(shape);
        for (std::size_t i = 0; i < node->keys.size(); ++i) {
            s->keys[i] = static_cast<T>(node->keys[i]);
            s->values[i] = static_cast<T>(node->values[i]);
        }
        s->tokens = (s->parent ? s->parent->tokens : 0) + node->tokens;
        sepal::ref::AttentionStatePtr<T> ptr = s;
        external_[node.get()] = ptr;
        return ptr;
    }

    RefRow<T> compute(std::size_t r) {
        using namespace sepal::ref;
        const RowPlan& plan = rows_[r];
        const std::size_t tokens = plan.length();
        const std::size_t heads = g_.shape.heads, kv_heads = g_.shape.kv_heads, head_dim = g_.shape.head_dim;
        const AttentionStatePtr<T> parent = state_of(plan.parent);

        sepal::BasicTensor<T> q_gate({tokens, g_.qg_width});
        sepal::BasicTensor<T> k_flat({tokens, g_.kv_width});
        sepal::BasicTensor<T> v({tokens, g_.kv_width});
        std::vector<double> positions(tokens);
        for (std::size_t t = 0; t < tokens; ++t) {
            const std::size_t id = plan.ids[t];
            for (std::size_t c = 0; c < g_.qg_width; ++c) q_gate[t * g_.qg_width + c] = static_cast<T>(pool_.qg[id * g_.qg_width + c]);
            for (std::size_t c = 0; c < g_.kv_width; ++c) {
                k_flat[t * g_.kv_width + c] = static_cast<T>(pool_.k[id * g_.kv_width + c]);
                v[t * g_.kv_width + c] = static_cast<T>(pool_.v[id * g_.kv_width + c]);
            }
            positions[t] = static_cast<double>(plan.start) + static_cast<double>(t);
        }
        const QueryAndGate<T> unpacked = unpack_query_and_gate<T>(q_gate.mat(), g_.cfg);
        const sepal::TensorD angles = rope_angles(positions, g_.cfg);
        sepal::BasicTensor<T> q = rms_norm<T>(unpacked.query.as_mat(head_dim), std::span<const T>(q_norm_), g_.cfg.rms_norm_eps);
        q.reshape({tokens, heads, head_dim});
        apply_rope(q, angles);
        sepal::BasicTensor<T> k = rms_norm<T>(k_flat.as_mat(head_dim), std::span<const T>(k_norm_), g_.cfg.rms_norm_eps);
        k.reshape({tokens, kv_heads, head_dim});
        apply_rope(k, angles);
        v.reshape({tokens, kv_heads, head_dim});

        auto state = std::make_shared<AttentionState<T>>();
        if (parent->tokens != 0) state->parent = parent;
        state->keys = std::move(k);
        state->values = std::move(v);
        state->tokens = parent->tokens + tokens;

        sepal::BasicTensor<T> attended = grouped_query_attention(*provider_, q, *state, parent->tokens, attention_scale(g_.cfg));
        RefRow<T> out;
        out.q.assign(q.span().begin(), q.span().end());
        out.keys.assign(state->keys.span().begin(), state->keys.span().end());
        out.values.assign(state->values.span().begin(), state->values.span().end());
        out.attended.assign(attended.span().begin(), attended.span().end());
        out.out = out.attended;
        for (std::size_t i = 0; i < out.out.size(); ++i) out.out[i] *= sigmoid(unpacked.gate[i]);
        out.state = std::move(state);
        return out;
    }

    const Geometry& g_;
    const Pool& pool_;
    const std::vector<RowPlan>& rows_;
    std::unique_ptr<sepal::GemmProvider> provider_;
    std::vector<T> q_norm_, k_norm_;
    std::map<const Node*, std::size_t> by_node_;
    std::map<std::size_t, RefRow<T>> done_;
    std::map<const Node*, sepal::ref::AttentionStatePtr<T>> external_;
};

// ---- comparing ----

struct Diff {
    double max_abs = 0.0;
    double max_ref = 0.0;
    double rms = 0.0;
    double max_rel = 0.0; // largest elementwise relative difference among elements at least as large as the rms
    double scaled() const { return max_ref > 0.0 ? max_abs / max_ref : 0.0; }
};

template <class A, class B>
Diff compare(const std::vector<A>& value, const std::vector<B>& reference) {
    Diff d;
    if (value.size() != reference.size()) {
        d.max_abs = 1e300;
        return d;
    }
    double sum_sq = 0.0;
    for (std::size_t i = 0; i < value.size(); ++i) {
        const double ref = static_cast<double>(reference[i]);
        const double diff = std::fabs(static_cast<double>(value[i]) - ref);
        if (!(diff <= 1e300)) d.max_abs = 1e300; // NaN or infinity
        d.max_abs = std::max(d.max_abs, diff);
        d.max_ref = std::max(d.max_ref, std::fabs(ref));
        sum_sq += ref * ref;
    }
    d.rms = value.empty() ? 0.0 : std::sqrt(sum_sq / static_cast<double>(value.size()));
    for (std::size_t i = 0; i < value.size(); ++i) {
        const double ref = static_cast<double>(reference[i]);
        if (std::fabs(ref) >= d.rms && ref != 0.0) {
            d.max_rel = std::max(d.max_rel, std::fabs(static_cast<double>(value[i]) - ref) / std::fabs(ref));
        }
    }
    return d;
}

bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

bool all_finite(const std::vector<float>& v) {
    for (float x : v) {
        if (!std::isfinite(x)) return false;
    }
    return true;
}

// Rows of two passes (or two launches) that must agree bit for bit; returns the number of bit-different values.
std::size_t count_different(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return std::max(a.size(), b.size());
    std::size_t n = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0) ++n;
    }
    return n;
}

bool identical_rows(const RowOut& a, const RowOut& b, const char* what, const char* variant) {
    const std::size_t dq = count_different(a.q, b.q);
    const std::size_t dk = count_different(a.keys, b.keys);
    const std::size_t dv = count_different(a.values, b.values);
    const std::size_t dout = count_different(a.out, b.out);
    const bool ok = dq == 0 && dk == 0 && dv == 0 && dout == 0 && all_finite(a.out) && !a.out.empty();
    if (!ok) {
        std::printf("  [DIFFERENT] %s %s: q %zu, keys %zu, values %zu, out %zu values differ\n", what, variant, dq, dk, dv, dout);
    }
    return ok;
}

// The part of a row from token `first` on (or up to `count` tokens), for comparing a row with a longer or shorter one.
RowOut slice(const Geometry& g, const RowOut& row, std::size_t first, std::size_t count) {
    RowOut s;
    const auto cut = [&](const std::vector<float>& v, std::size_t width) {
        return std::vector<float>(v.begin() + static_cast<std::ptrdiff_t>(first * width),
                                  v.begin() + static_cast<std::ptrdiff_t>((first + count) * width));
    };
    s.q = cut(row.q, g.q_width);
    s.out = cut(row.out, g.q_width);
    s.keys = cut(row.keys, g.kv_width);
    s.values = cut(row.values, g.kv_width);
    return s;
}

// Checks one device row against the reference rows (float32 and, when given, float64) and prints the figures.
// `scaled` is the largest absolute difference over the largest magnitude of the quantity; `rel` the largest
// elementwise relative difference among the elements at least as large as the quantity's rms.
void check_against_reference(const char* label, const Variant& variant, const RowOut& dev, const RefRow<float>& ref32,
                             const RefRow<double>* ref64, bool compare_attended, bool verbose) {
    const bool f16 = variant.kv == AttnElem::F16;
    // F32: rounding of the arithmetic only. F16: the stored q, k and v each carry one binary16 rounding; the attention
    // output carries those and the rounding of the probabilities, so it is allowed a few units of 2^-11.
    const double store_limit = f16 ? 2 * kHalfUnitRoundoff : 1e-5;
    const double out_limit = f16 ? 4 * kHalfUnitRoundoff : 1e-5;
    char line[512];
    int used = std::snprintf(line, sizeof line, "  [measured] %-30s %-10s scaled vs f32 ref:", label, variant.name);
    const auto one = [&](const char* stage, const std::vector<float>& value, const std::vector<float>& r32,
                         const std::vector<double>* r64, double stage_limit) {
        const Diff d32 = compare(value, r32);
        CHECK(d32.scaled() <= stage_limit);
        CHECK(all_finite(value));
        if (verbose) {
            std::printf("  [measured] %-30s %-10s %-9s vs f32 ref: max abs %.3e scaled %.3e rel %.3e", label, variant.name,
                        stage, d32.max_abs, d32.scaled(), d32.max_rel);
            if (r64 != nullptr) {
                const Diff d64 = compare(value, *r64);
                const Diff dref = compare(r32, *r64);
                std::printf(" | vs f64 ref: scaled %.3e | f32 ref vs f64 ref: scaled %.3e", d64.scaled(), dref.scaled());
            }
            std::printf("\n");
        } else if (used < static_cast<int>(sizeof line)) {
            used += std::snprintf(line + used, sizeof line - static_cast<std::size_t>(used), " %s %.2e", stage, d32.scaled());
            if (r64 != nullptr) {
                const Diff d64 = compare(value, *r64);
                used += std::snprintf(line + used, sizeof line - static_cast<std::size_t>(used), " (f64 %.2e)", d64.scaled());
            }
        }
    };
    one("q", dev.q, ref32.q, ref64 ? &ref64->q : nullptr, store_limit);
    one("keys", dev.keys, ref32.keys, ref64 ? &ref64->keys : nullptr, store_limit);
    one("values", dev.values, ref32.values, ref64 ? &ref64->values : nullptr, store_limit);
    if (compare_attended) one("attended", dev.out, ref32.attended, ref64 ? &ref64->attended : nullptr, out_limit);
    else one("gated", dev.out, ref32.out, ref64 ? &ref64->out : nullptr, out_limit);
    if (!verbose) std::printf("%s\n", line);
}

// ---- suite: stages ----
// Every stage against the reference functions, on a row with a context, for each geometry and variant. The
// attention output alone is measured with the gate forced open, the gated output with the random gate.

void stages_for(const Geometry& g, std::uint32_t context, std::uint32_t length, std::uint32_t start) {
    const Pool pool(g, length + 8, 11);
    const NormWeights weights = make_weights(g, 12);
    NodePtr ctx = context != 0 ? external_node(g, context, 13, nullptr) : nullptr;
    std::vector<RowPlan> rows;
    rows.push_back(make_row(iota_ids(0, length), start, ctx));
    char label[96];
    std::snprintf(label, sizeof label, "%s ctx %u len %u pos %u", g.name.c_str(), context, length, start);

    RefPass<float> ref32(g, pool, weights, rows);
    RefPass<double> ref64(g, pool, weights, rows);
    for (const Variant& variant : variants()) {
        PassOptions open;
        open.gate_open = true;
        const PassResult attended = run_device(g, pool, weights, rows, variant, open);
        const PassResult gated = run_device(g, pool, weights, rows, variant);
        check_against_reference(label, variant, attended.rows[0], ref32.row(0), &ref64.row(0), true, true);
        // The gated run must not change anything but the gate: same q, keys and values bit for bit.
        CHECK(same_bits(attended.rows[0].q, gated.rows[0].q));
        CHECK(same_bits(attended.rows[0].keys, gated.rows[0].keys));
        const Diff gd = compare(gated.rows[0].out, ref32.row(0).out);
        const Diff gd64 = compare(gated.rows[0].out, ref64.row(0).out);
        std::printf("  [measured] %-30s %-10s %-9s vs f32 ref: scaled %.3e | vs f64 ref: scaled %.3e (random gate)\n", label,
                    variant.name, "gated", gd.scaled(), gd64.scaled());
        CHECK(gd.scaled() <= (variant.kv == AttnElem::F16 ? 4 * kHalfUnitRoundoff : 1e-5));
    }
}

SEPAL_TEST(stages, tiny) {
    stages_for(tiny(), 0, 40, 0);
    stages_for(tiny(), 27, 40, 100);
}
SEPAL_TEST(stages, preset_0_8b) {
    stages_for(preset("0.8b"), 0, 33, 0);
    stages_for(preset("0.8b"), 27, 40, 100);
}
SEPAL_TEST(stages, preset_4b) { stages_for(preset("4b"), 27, 20, 5000); }

// ---- suite: geometry ----
// The whole chain for the geometries and the context lengths of the brief, each variant against the float32 reference
// (and against the float64 one where that is cheap).

void geometry_case(const Geometry& g, std::uint32_t context, std::uint32_t length, std::uint32_t start, bool with_f64) {
    const Pool pool(g, length + 8, 21 + context + length);
    const NormWeights weights = make_weights(g, 22);
    NodePtr ctx = context != 0 ? external_node(g, context, 23 + context, nullptr) : nullptr;
    std::vector<RowPlan> rows;
    rows.push_back(make_row(iota_ids(0, length), start, ctx));
    char label[96];
    std::snprintf(label, sizeof label, "%s ctx %u len %u", g.name.c_str(), context, length);
    RefPass<float> ref32(g, pool, weights, rows);
    std::unique_ptr<RefPass<double>> ref64;
    if (with_f64) ref64 = std::make_unique<RefPass<double>>(g, pool, weights, rows);
    for (const Variant& variant : variants()) {
        const PassResult result = run_device(g, pool, weights, rows, variant);
        check_against_reference(label, variant, result.rows[0], ref32.row(0), ref64 ? &ref64->row(0) : nullptr, false, false);
    }
}

SEPAL_TEST(geometry, tiny) {
    for (std::uint32_t context : {0u, 1u, 27u, 640u}) {
        for (std::uint32_t length : {1u, 27u, 70u}) geometry_case(tiny(), context, length, 3, true);
    }
    geometry_case(tiny(), 4800, 1, 4800, true);
    geometry_case(tiny(), 4800, 27, 4800, true);
}
SEPAL_TEST(geometry, preset_0_8b) {
    const Geometry& g = preset("0.8b");
    for (std::uint32_t context : {0u, 1u, 27u, 640u, 4800u}) {
        for (std::uint32_t length : {1u, 27u}) geometry_case(g, context, length, context, true);
    }
    geometry_case(g, 0, 640, 0, false);
    geometry_case(g, 27, 640, 27, false);
    geometry_case(g, 640, 640, 640, false);
}
SEPAL_TEST(geometry, preset_4b) {
    const Geometry& g = preset("4b");
    for (std::uint32_t context : {0u, 1u, 27u, 640u, 4800u}) {
        for (std::uint32_t length : {1u, 27u}) geometry_case(g, context, length, context, context <= 640);
    }
    geometry_case(g, 0, 640, 0, false);
}

// Large shapes have no host reference (it would take minutes); the three device variants are compared with each other
// instead: the plain kernel on float32 keys and values is the checked reference for the other two.
SEPAL_TEST(geometry, large_device_only) {
    for (const char* name : {"0.8b", "4b"}) {
        const Geometry& g = preset(name);
        const Pool pool(g, 4096 + 700, 31);
        const NormWeights weights = make_weights(g, 32);
        NodePtr ctx = external_node(g, 4800, 33, nullptr);
        std::vector<RowPlan> rows;
        rows.push_back(make_row(iota_ids(0, 4096), 0, nullptr));
        rows.push_back(make_row(iota_ids(4096, 640), 4800, ctx));
        rows.push_back(make_row(iota_ids(4736, 27), 4800, ctx));
        std::vector<PassResult> results;
        for (const Variant& variant : variants()) results.push_back(run_device(g, pool, weights, rows, variant));
        for (std::size_t r = 0; r < rows.size(); ++r) {
            for (std::size_t v = 1; v < results.size(); ++v) {
                const Diff d = compare(results[v].rows[r].out, results[0].rows[r].out);
                std::printf("  [measured] %s large row %zu (len %u ctx %u) %-10s vs f32/plain: max abs %.3e scaled %.3e\n", name, r,
                            rows[r].length(), chain_tokens(rows[r].parent), variants()[v].name, d.max_abs, d.scaled());
                CHECK(d.scaled() <= 4 * kHalfUnitRoundoff);
                CHECK(all_finite(results[v].rows[r].out));
            }
            const Diff d = compare(results[2].rows[r].out, results[1].rows[r].out);
            std::printf("  [measured] %s large row %zu f16/tiled vs f16/plain: max abs %.3e scaled %.3e\n", name, r, d.max_abs,
                        d.scaled());
            CHECK(d.scaled() <= 4 * kHalfUnitRoundoff);
        }
    }
}

// ---- suite: continuation ----
// A row [0, T) alone equals the row split at the cut points, each piece having the pieces before it as context, at
// the tokens both runs share: bit for bit in q, stored keys and values, and the output.

void continuation_case(const Geometry& g, std::uint32_t context, std::uint32_t total, const std::vector<std::uint32_t>& cuts,
                       bool reverse_flat_order) {
    const Pool pool(g, total + 8, 41 + total);
    const NormWeights weights = make_weights(g, 42);
    const NodePtr ctx = context != 0 ? external_node(g, context, 43, nullptr) : nullptr;
    const std::uint32_t start = context + 5;

    std::vector<RowPlan> whole;
    whole.push_back(make_row(iota_ids(0, total), start, ctx));

    std::vector<std::uint32_t> bounds = {0};
    bounds.insert(bounds.end(), cuts.begin(), cuts.end());
    bounds.push_back(total);
    std::vector<RowPlan> pieces;
    for (std::size_t i = 0; i + 1 < bounds.size(); ++i) {
        const NodePtr parent = i == 0 ? ctx : pieces.back().node;
        pieces.push_back(make_row(iota_ids(bounds[i], bounds[i + 1] - bounds[i]), start + bounds[i], parent));
    }
    std::vector<RowPlan> split = pieces;
    if (reverse_flat_order) std::reverse(split.begin(), split.end());

    for (const Variant& variant : variants()) {
        const PassResult one = run_device(g, pool, weights, whole, variant);
        const PassResult many = run_device(g, pool, weights, split, variant);
        for (std::size_t i = 0; i + 1 < bounds.size(); ++i) {
            const std::size_t at = reverse_flat_order ? bounds.size() - 2 - i : i;
            const RowOut expected = slice(g, one.rows[0], bounds[i], bounds[i + 1] - bounds[i]);
            char what[96];
            std::snprintf(what, sizeof what, "%s ctx %u T %u piece %zu [%u,%u)", g.name.c_str(), context, total, i, bounds[i],
                          bounds[i + 1]);
            CHECK(identical_rows(many.rows[at], expected, what, variant.name));
        }
    }
}

SEPAL_TEST(continuation, tiny) {
    const Geometry& g = tiny();
    for (std::uint32_t context : {0u, 1u, 27u, 70u}) {
        for (const std::vector<std::uint32_t>& cuts : std::vector<std::vector<std::uint32_t>>{
                 {1}, {27}, {16}, {32}, {33}, {99}, {27, 64}, {1, 2, 3, 50}}) {
            continuation_case(g, context, 100, cuts, false);
        }
    }
    continuation_case(g, 27, 100, {27, 64}, true);
    continuation_case(g, 0, 100, {1, 2, 3, 50}, true);
}
SEPAL_TEST(continuation, preset_0_8b) {
    const Geometry& g = preset("0.8b");
    for (std::uint32_t context : {0u, 27u, 640u}) {
        continuation_case(g, context, 700, {27}, false);
        continuation_case(g, context, 700, {640}, false);
        continuation_case(g, context, 700, {1, 32, 33, 300}, false);
    }
    continuation_case(g, 27, 700, {27, 100}, true);
}
SEPAL_TEST(continuation, preset_4b) {
    const Geometry& g = preset("4b");
    continuation_case(g, 27, 200, {27}, false);
    continuation_case(g, 0, 200, {1, 100}, false);
}

// ---- suite: shared_context ----
// A state row and eight branch rows that continue it, in one launch, against the same branches one per launch. The
// state row sits in the middle of the flat list and the branches around it.

void shared_case(const Geometry& g, std::uint32_t state_len, std::uint32_t context, bool against_reference) {
    constexpr std::uint32_t kBranches = 8, kBranchLen = 27;
    const Pool pool(g, state_len + kBranches * kBranchLen + 8, 51 + state_len);
    const NormWeights weights = make_weights(g, 52);
    const NodePtr ctx = context != 0 ? external_node(g, context, 53, nullptr) : nullptr;
    const std::uint32_t state_start = context;

    RowPlan state = make_row(iota_ids(0, state_len), state_start, ctx);
    std::vector<RowPlan> branches;
    for (std::uint32_t b = 0; b < kBranches; ++b) {
        // Each branch restarts at the position where the state ended.
        branches.push_back(make_row(iota_ids(state_len + b * kBranchLen, kBranchLen), state_start + state_len, state.node));
    }
    std::vector<RowPlan> together;
    for (std::uint32_t b = 0; b < 3; ++b) together.push_back(branches[b]);
    together.push_back(state);
    for (std::uint32_t b = 3; b < kBranches; ++b) together.push_back(branches[b]);
    const auto index_of_branch = [](std::uint32_t b) { return b < 3 ? b : b + 1; };
    const std::size_t state_index = 3;

    std::unique_ptr<RefPass<float>> ref;
    if (against_reference) ref = std::make_unique<RefPass<float>>(g, pool, weights, together);
    for (const Variant& variant : variants()) {
        const PassResult combined = run_device(g, pool, weights, together, variant);
        const NodePtr state_alone = externalize(state, combined.rows[state_index], ctx);
        std::size_t bit_different_rows = 0;
        for (std::uint32_t b = 0; b < kBranches; ++b) {
            std::vector<RowPlan> alone;
            alone.push_back(make_row(branches[b].ids, branches[b].start, state_alone));
            const PassResult one = run_device(g, pool, weights, alone, variant);
            char what[96];
            std::snprintf(what, sizeof what, "%s state %u ctx %u branch %u", g.name.c_str(), state_len, context, b);
            const bool same = identical_rows(combined.rows[index_of_branch(b)], one.rows[0], what, variant.name);
            CHECK(same);
            if (!same) ++bit_different_rows;
            if (ref && b < 2) {
                check_against_reference(what, variant, combined.rows[index_of_branch(b)], ref->row(index_of_branch(b)), nullptr,
                                        false, false);
            }
        }
        std::printf("  [measured] %s state %u ctx %u %-10s: %u branches in one launch vs one per launch, %zu rows differ in bits\n",
                    g.name.c_str(), state_len, context, variant.name, kBranches, bit_different_rows);
    }
}

SEPAL_TEST(shared_context, tiny) {
    shared_case(tiny(), 130, 0, true);
    shared_case(tiny(), 130, 27, true);
    shared_case(tiny(), 33, 0, true);
}
SEPAL_TEST(shared_context, preset_0_8b) {
    shared_case(preset("0.8b"), 640, 0, false);
    shared_case(preset("0.8b"), 640, 27, true);
}
SEPAL_TEST(shared_context, preset_4b) { shared_case(preset("4b"), 200, 27, false); }

// ---- suite: causality ----

Pool with_replaced(const Pool& pool, const std::vector<std::uint32_t>& ids, std::uint64_t seed) {
    Pool changed = pool;
    const Geometry& g = *pool.geo;
    Rng rng(seed);
    for (std::uint32_t id : ids) {
        for (std::uint32_t c = 0; c < g.qg_width; ++c) changed.qg[static_cast<std::size_t>(id) * g.qg_width + c] = rng.uniform(-2.0f, 2.0f);
        for (std::uint32_t c = 0; c < g.kv_width; ++c) {
            changed.k[static_cast<std::size_t>(id) * g.kv_width + c] = rng.uniform(-2.0f, 2.0f);
            changed.v[static_cast<std::size_t>(id) * g.kv_width + c] = rng.uniform(-1.0f, 1.0f);
        }
    }
    return changed;
}

std::vector<std::uint32_t> id_range(std::uint32_t first, std::uint32_t last) { return iota_ids(first, last - first); }

void causality_case(const Geometry& g, std::uint32_t context, std::uint32_t total) {
    const Pool pool(g, total + 8, 61);
    const NormWeights weights = make_weights(g, 62);
    const NodePtr ctx = context != 0 ? external_node(g, context, 63, nullptr) : nullptr;
    std::vector<RowPlan> rows;
    rows.push_back(make_row(iota_ids(0, total), context, ctx));
    for (const Variant& variant : variants()) {
        const PassResult base = run_device(g, pool, weights, rows, variant);
        for (std::uint32_t t0 : {0u, 1u, 15u, 16u, 31u, 32u, 63u, total / 2, total - 2}) {
            if (t0 + 1 >= total) continue;
            // Every token after t0 is replaced.
            const Pool changed = with_replaced(pool, id_range(t0 + 1, total), 64 + t0);
            const PassResult after = run_device(g, changed, weights, rows, variant);
            char what[96];
            std::snprintf(what, sizeof what, "%s ctx %u T %u change after %u", g.name.c_str(), context, total, t0);
            CHECK(identical_rows(slice(g, after.rows[0], 0, t0 + 1), slice(g, base.rows[0], 0, t0 + 1), what, variant.name));
            // The test is not vacuous: the token after the change does move.
            CHECK(count_different(slice(g, after.rows[0], t0 + 1, 1).out, slice(g, base.rows[0], t0 + 1, 1).out) != 0);
        }
    }
}

// A token outside a row's chain never changes the row: sibling branches, an unrelated row, the rows that continue it.
void chain_case(const Geometry& g) {
    const Pool pool(g, 3000, 71);
    const NormWeights weights = make_weights(g, 72);
    const NodePtr shared = external_node(g, 40, 73, nullptr);
    const NodePtr other = external_node(g, 30, 74, nullptr);
    RowPlan state = make_row(id_range(0, 50), 40, shared);
    RowPlan branch_a = make_row(id_range(100, 127), 90, state.node);
    RowPlan branch_b = make_row(id_range(200, 227), 90, state.node);
    RowPlan sibling_of_shared = make_row(id_range(300, 310), 40, shared);
    RowPlan unrelated = make_row(id_range(400, 450), 30, other);
    std::vector<RowPlan> rows = {branch_a, unrelated, state, sibling_of_shared, branch_b};
    for (const Variant& variant : variants()) {
        const PassResult base = run_device(g, pool, weights, rows, variant);
        // Change one row's tokens at a time; every other row not below it in the chain must keep its bits.
        struct Case {
            const char* name;
            std::size_t changed;                  // index in `rows`
            std::vector<std::size_t> must_stay;   // rows that must not change
        };
        const std::vector<Case> cases = {
            {"sibling branch b", 4, {0, 1, 2, 3}},
            {"unrelated row", 1, {0, 2, 3, 4}},
            {"sibling of the state", 3, {0, 1, 2, 4}},
            {"branch a", 0, {1, 2, 3, 4}},
        };
        for (const Case& c : cases) {
            const Pool changed = with_replaced(pool, rows[c.changed].ids, 75 + c.changed);
            const PassResult after = run_device(g, changed, weights, rows, variant);
            for (std::size_t r : c.must_stay) {
                char what[96];
                std::snprintf(what, sizeof what, "%s change %s, row %zu", g.name.c_str(), c.name, r);
                CHECK(identical_rows(after.rows[r], base.rows[r], what, variant.name));
            }
            CHECK(count_different(after.rows[c.changed].out, base.rows[c.changed].out) != 0);
        }
        // Changing the state does move the branches that continue it, and only those.
        const Pool changed = with_replaced(pool, rows[2].ids, 80);
        const PassResult after = run_device(g, changed, weights, rows, variant);
        CHECK(count_different(after.rows[0].out, base.rows[0].out) != 0);
        CHECK(count_different(after.rows[4].out, base.rows[4].out) != 0);
        CHECK(identical_rows(after.rows[1], base.rows[1], "state changed, unrelated row", variant.name));
        CHECK(identical_rows(after.rows[3], base.rows[3], "state changed, sibling of the state", variant.name));
    }
}

SEPAL_TEST(causality, tiny) {
    causality_case(tiny(), 0, 130);
    causality_case(tiny(), 27, 130);
    chain_case(tiny());
}
SEPAL_TEST(causality, preset_0_8b) {
    causality_case(preset("0.8b"), 27, 300);
    chain_case(preset("0.8b"));
}
SEPAL_TEST(causality, preset_4b) { causality_case(preset("4b"), 0, 100); }

// ---- suite: invariance ----
// A row's output is the same alone and among other random rows of other lengths and contexts, wherever it sits in
// the flat list.

void invariance_case(const Geometry& g, const std::vector<std::uint32_t>& lengths, const std::vector<std::uint32_t>& contexts,
                     std::uint32_t target_length, std::uint32_t target_context, int trials) {
    const Pool pool(g, 5000, 81);
    const NormWeights weights = make_weights(g, 82);
    Rng rng(83 + target_length);
    const NodePtr target_ctx = target_context != 0 ? external_node(g, target_context, 84, nullptr) : nullptr;
    const RowPlan target = make_row(iota_ids(0, target_length), 77, target_ctx);

    for (const Variant& variant : variants()) {
        std::vector<RowPlan> alone = {target};
        const PassResult base = run_device(g, pool, weights, alone, variant);
        for (int trial = 0; trial < trials; ++trial) {
            std::uint32_t next_id = target_length;
            const NodePtr shared = external_node(g, contexts[rng.below(static_cast<std::uint32_t>(contexts.size()))] + 1, 85 + static_cast<std::uint64_t>(trial), nullptr);
            std::vector<RowPlan> others;
            const std::uint32_t count = 2 + rng.below(5);
            for (std::uint32_t i = 0; i < count; ++i) {
                const std::uint32_t length = lengths[rng.below(static_cast<std::uint32_t>(lengths.size()))];
                NodePtr parent;
                switch (rng.below(4)) {
                case 0: break;
                case 1: parent = external_node(g, contexts[rng.below(static_cast<std::uint32_t>(contexts.size()))] + 1, 90 + i, nullptr); break;
                case 2: parent = shared; break;
                default: // continue an earlier row of the same pass
                    if (!others.empty()) parent = others[rng.below(static_cast<std::uint32_t>(others.size()))].node;
                    break;
                }
                others.push_back(make_row(iota_ids(next_id, length), rng.below(70000), parent));
                next_id += length;
            }
            const std::size_t place = trial == 0 ? 0 : trial == 1 ? others.size() : rng.below(static_cast<std::uint32_t>(others.size() + 1));
            std::vector<RowPlan> list = others;
            list.insert(list.begin() + static_cast<std::ptrdiff_t>(place), target);
            const PassResult among = run_device(g, pool, weights, list, variant);
            char what[128];
            std::snprintf(what, sizeof what, "%s T %u ctx %u trial %d among %zu rows at %zu", g.name.c_str(), target_length,
                          target_context, trial, list.size(), place);
            CHECK(identical_rows(among.rows[place], base.rows[0], what, variant.name));
        }
    }
}

SEPAL_TEST(invariance, tiny) {
    invariance_case(tiny(), {1, 2, 5, 16, 27, 33, 64, 130}, {0, 1, 27, 130}, 27, 27, 12);
    invariance_case(tiny(), {1, 2, 5, 16, 27, 33, 64, 130}, {0, 1, 27, 130}, 1, 0, 8);
    invariance_case(tiny(), {1, 2, 5, 16, 27, 33, 64, 130}, {0, 1, 27, 130}, 100, 0, 8);
    invariance_case(tiny(), {1, 2, 5, 16, 27, 33, 64, 130}, {0, 1, 27, 130}, 33, 70, 8);
}
SEPAL_TEST(invariance, preset_0_8b) {
    invariance_case(preset("0.8b"), {1, 5, 27, 64, 200}, {0, 1, 27, 300}, 27, 640, 6);
    invariance_case(preset("0.8b"), {1, 5, 27, 64, 200}, {0, 1, 27, 300}, 130, 0, 6);
}
SEPAL_TEST(invariance, preset_4b) { invariance_case(preset("4b"), {1, 5, 27, 64}, {0, 1, 27, 100}, 27, 100, 4); }

// ---- suite: positions ----

SEPAL_TEST(positions, large_start) {
    for (std::uint32_t start : {60000u, 65536u, 65536u + 8191u}) {
        geometry_case(tiny(), 27, 40, start, true);
        geometry_case(preset("0.8b"), 27, 40, start, true);
    }
}

// A state at a large position and branches that restart where it ended, through the reference chain.
SEPAL_TEST(positions, branch_restarts_at_state_end) {
    for (const Geometry* g : {&tiny(), &preset("0.8b")}) {
        const Pool pool(*g, 200, 101);
        const NormWeights weights = make_weights(*g, 102);
        const NodePtr ctx = external_node(*g, 27, 103, nullptr);
        RowPlan state = make_row(iota_ids(0, 40), 60000, ctx);
        std::vector<RowPlan> rows = {state};
        for (std::uint32_t b = 0; b < 3; ++b) rows.push_back(make_row(iota_ids(40 + 30 * b, 27), 60040, state.node));
        RefPass<float> ref(*g, pool, weights, rows);
        for (const Variant& variant : variants()) {
            const PassResult result = run_device(*g, pool, weights, rows, variant);
            for (std::size_t r = 0; r < rows.size(); ++r) {
                char label[96];
                std::snprintf(label, sizeof label, "%s restart row %zu", g->name.c_str(), r);
                check_against_reference(label, variant, result.rows[r], ref.row(r), nullptr, false, false);
            }
        }
    }
}

// ---- suite: rope_exact ----
// The rotation angle is a float64 product and its cosine and sine are taken in float64. With every query and key
// element +-1 the sum of squares is exact whatever the order, so the normed and rotated vectors can be
// reproduced bit for bit on the host with the reference's own expressions. Checked at every position from 0 to
// 65,536 + 8,192 (and a little beyond), with the F32 and the F16 stores.

SEPAL_TEST(rope_exact, every_position) {
    ModelConfig cfg = *sepal::find_preset("0.8b");
    cfg.attn_heads = 1;
    cfg.attn_kv_heads = 1;
    const Geometry g = make_geometry(cfg);
    const std::uint32_t hd = g.shape.head_dim, pairs = g.shape.rotary_dims / 2;
    constexpr std::uint32_t kRowLen = 2048, kRows = 37, kPerLaunch = 13;
    Pool pool(g, kRowLen, 111);
    Rng signs(112);
    for (std::uint32_t id = 0; id < kRowLen; ++id) {
        for (std::uint32_t c = 0; c < hd; ++c) {
            pool.qg[static_cast<std::size_t>(id) * g.qg_width + c] = (signs.next() & 1) ? 1.0f : -1.0f;
            pool.k[static_cast<std::size_t>(id) * g.kv_width + c] = (signs.next() & 1) ? 1.0f : -1.0f;
        }
    }
    const NormWeights weights = make_weights(g, 113);
    const float eps = static_cast<float>(g.shape.rms_norm_eps);
    const float rms = std::sqrt(static_cast<float>(hd) / static_cast<float>(hd) + eps);

    // Host normalisation of a +-1 vector, then the rotation with the reference's expressions.
    const auto expected = [&](float sign_of, std::uint32_t c, const std::vector<float>& w) { return sign_of / rms * (1.0f + w[c]); };
    std::vector<double> inverse_frequency(pairs);
    for (std::uint32_t i = 0; i < pairs; ++i) {
        inverse_frequency[i] = std::pow(g.shape.rope_theta, -static_cast<double>(2 * i) / static_cast<double>(2 * pairs));
    }

    std::size_t mismatched = 0, compared = 0, mismatched_half = 0;
    for (std::uint32_t first_row = 0; first_row < kRows; first_row += kPerLaunch) {
        std::vector<RowPlan> rows;
        for (std::uint32_t r = first_row; r < std::min(kRows, first_row + kPerLaunch); ++r) {
            rows.push_back(make_row(iota_ids(0, kRowLen), r * kRowLen, nullptr));
        }
        const PassResult f32 = run_device(g, pool, weights, rows, variants()[0]);
        const PassResult f16 = run_device(g, pool, weights, rows, variants()[1]);
        for (std::size_t r = 0; r < rows.size(); ++r) {
            for (std::uint32_t t = 0; t < kRowLen; ++t) {
                const double position = static_cast<double>(rows[r].start) + t;
                const float* sq = pool.qg.data() + static_cast<std::size_t>(t) * g.qg_width;
                const float* sk = pool.k.data() + static_cast<std::size_t>(t) * g.kv_width;
                for (int which = 0; which < 2; ++which) {
                    const float* sign = which == 0 ? sq : sk;
                    const std::vector<float>& w = which == 0 ? weights.q : weights.k;
                    const std::vector<float>& got32 = which == 0 ? f32.rows[r].q : f32.rows[r].keys;
                    const std::vector<float>& got16 = which == 0 ? f16.rows[r].q : f16.rows[r].keys;
                    for (std::uint32_t d = 0; d < hd; ++d) {
                        float value;
                        if (d < 2 * pairs) {
                            const std::uint32_t i = d % pairs;
                            const double angle = position * inverse_frequency[i];
                            const float c = static_cast<float>(std::cos(angle));
                            const float s = static_cast<float>(std::sin(angle));
                            const float first = expected(sign[i], i, w);
                            const float second = expected(sign[pairs + i], pairs + i, w);
                            const volatile float a = first * c;
                            const volatile float b = second * s;
                            const volatile float e = second * c;
                            const volatile float f = first * s;
                            value = d < pairs ? a - b : e + f;
                        } else {
                            value = expected(sign[d], d, w);
                        }
                        const std::size_t at = static_cast<std::size_t>(t) * hd + d;
                        ++compared;
                        if (std::memcmp(&value, &got32[at], 4) != 0) ++mismatched;
                        const float half_expected = round_half(value);
                        if (std::memcmp(&half_expected, &got16[at], 4) != 0) ++mismatched_half;
                    }
                }
            }
        }
    }
    std::printf("  [measured] rope_exact: positions 0 .. %u, %zu q and k elements per store: %zu bit mismatches on the F32 path, "
                "%zu on the F16 path\n",
                kRows * kRowLen - 1, compared, mismatched, mismatched_half);
    CHECK_EQ(mismatched, 0u);
    CHECK_EQ(mismatched_half, 0u);
}

// ---- suite: split ----
// The second float32 kernel (F32Kernel::Split) with chunks small enough that a row of a few hundred keys spans several of them: against
// the float32 reference; bit for bit the same whatever the workspace cap (windows of query tiles), whether the host gives
// max_row_keys, and wherever the row sits among other rows; and its fast exp2 against float64.

const Variant& split_variant() {
    static const Variant v{AttnElem::F32, AttentionPath::TiledF32, "f32/split", 1};
    return v;
}

struct SplitLimits {
    SplitLimits(std::uint32_t chunk, std::size_t cap) { sepal::cuda::set_f32_split_limits(chunk, cap); }
    ~SplitLimits() { sepal::cuda::set_f32_split_limits(0, 0); }
};

SEPAL_TEST(split, chunks_against_reference) {
    const Geometry& g = preset("4b");
    for (const std::uint32_t chunk : {8u, 64u, 200u}) {
        const SplitLimits limits(chunk, 0);
        for (const auto& [context, length] : {std::pair<std::uint32_t, std::uint32_t>{0, 70}, {300, 27}, {130, 1}, {1000, 40}}) {
            const Pool pool(g, length + 8, 401 + context + length);
            const NormWeights weights = make_weights(g, 402);
            NodePtr ctx = context != 0 ? external_node(g, context, 403 + context, nullptr) : nullptr;
            std::vector<RowPlan> rows;
            rows.push_back(make_row(iota_ids(0, length), context, ctx));
            char label[96];
            std::snprintf(label, sizeof label, "4b chunk %u ctx %u len %u", chunk, context, length);
            RefPass<float> ref32(g, pool, weights, rows);
            const PassResult result = run_device(g, pool, weights, rows, split_variant());
            check_against_reference(label, split_variant(), result.rows[0], ref32.row(0), nullptr, false, false);
        }
    }
}

SEPAL_TEST(split, windows_and_max_keys_bit_identical) {
    const Geometry& g = preset("4b");
    const Pool pool(g, 700, 411);
    const NormWeights weights = make_weights(g, 412);
    NodePtr ctx = external_node(g, 333, 413, nullptr);
    std::vector<RowPlan> rows;
    rows.push_back(make_row(iota_ids(0, 300), 333, ctx));
    rows.push_back(make_row(iota_ids(300, 1), 333, ctx));
    rows.push_back(make_row(iota_ids(301, 45), 0, nullptr));
    rows.push_back(make_row(iota_ids(346, 133), 0, nullptr));
    PassResult base;
    {
        const SplitLimits limits(64, 0); // every window at once
        base = run_device(g, pool, weights, rows, split_variant());
    }
    for (const std::size_t cap : {std::size_t{1}, std::size_t{300000}, std::size_t{3000000}}) {
        for (const bool give : {true, false}) {
            const SplitLimits limits(64, cap);
            PassOptions options;
            options.give_max_keys = give;
            const PassResult other = run_device(g, pool, weights, rows, split_variant(), options);
            for (std::size_t r = 0; r < rows.size(); ++r) {
                char what[96];
                std::snprintf(what, sizeof what, "4b row %zu cap %zu max_keys %s", r, cap, give ? "given" : "read back");
                CHECK(identical_rows(other.rows[r], base.rows[r], what, "f32/split"));
            }
        }
    }
}

SEPAL_TEST(split, invariance_small_chunks) {
    const SplitLimits limits(64, 200000);
    const Geometry& g = preset("4b");
    const Pool pool(g, 5000, 421);
    const NormWeights weights = make_weights(g, 422);
    Rng rng(423);
    const NodePtr target_ctx = external_node(g, 250, 424, nullptr);
    const RowPlan target = make_row(iota_ids(0, 90), 77, target_ctx);
    std::vector<RowPlan> alone = {target};
    const PassResult base = run_device(g, pool, weights, alone, split_variant());
    for (int trial = 0; trial < 4; ++trial) {
        std::uint32_t next_id = 90;
        std::vector<RowPlan> others;
        const std::uint32_t count = 2 + rng.below(4);
        for (std::uint32_t i = 0; i < count; ++i) {
            const std::uint32_t length = 1 + rng.below(150);
            NodePtr parent = rng.below(2) ? external_node(g, 1 + rng.below(400), 430 + i + 10 * static_cast<std::uint64_t>(trial), nullptr) : nullptr;
            others.push_back(make_row(iota_ids(next_id, length), rng.below(70000), parent));
            next_id += length;
        }
        const std::size_t place = rng.below(static_cast<std::uint32_t>(others.size() + 1));
        std::vector<RowPlan> list = others;
        list.insert(list.begin() + static_cast<std::ptrdiff_t>(place), target);
        const PassResult among = run_device(g, pool, weights, list, split_variant());
        char what[96];
        std::snprintf(what, sizeof what, "4b trial %d among %zu rows at %zu", trial, list.size(), place);
        CHECK(identical_rows(among.rows[place], base.rows[0], what, "f32/split"));
    }
}

SEPAL_TEST(split, exp2_fast_path_error) {
    // Arguments in [-126, 0] (the softmax takes x - max <= 0; below -126 the result flushes to zero), every 2^-12 and the
    // endpoints: the largest relative error against exp2 in float64.
    std::vector<float> in;
    for (std::int64_t k = -126 * 4096; k <= 0; ++k) in.push_back(static_cast<float>(static_cast<double>(k) / 4096.0));
    for (const float x : {0.0f, -0.0f, -1e-30f, -1e-7f, -0.5f, -125.9999f}) in.push_back(x);
    in.push_back(-std::numeric_limits<float>::infinity());
    DeviceBuffer<float> d_in(in);
    DeviceBuffer<float> d_out(in.size());
    sepal::cuda::attention_ex2_probe(d_in.data(), d_out.data(), static_cast<std::uint32_t>(in.size()), nullptr);
    sepal::cuda::check(cudaStreamSynchronize(nullptr), "cudaStreamSynchronize");
    const std::vector<float> out = d_out.download();
    double worst = 0;
    float worst_x = 0;
    for (std::size_t i = 0; i + 1 < in.size(); ++i) {
        const double exact = std::exp2(static_cast<double>(in[i]));
        const double rel = std::fabs(static_cast<double>(out[i]) - exact) / exact;
        if (rel > worst) {
            worst = rel;
            worst_x = in[i];
        }
    }
    std::printf("  [measured] ex2.approx.ftz.f32 over [-126, 0]: max relative error %.3e (2^%.2f) at x = %.6g\n", worst,
                std::log2(worst), static_cast<double>(worst_x));
    CHECK(worst <= std::ldexp(1.0, -21));
    CHECK(out[in.size() - 1] == 0.0f);            // 2^-inf
    CHECK(out[static_cast<std::size_t>(126 * 4096)] == 1.0f); // 2^0
}

SEPAL_TEST(split, kernels_agree_long) {
    // The default chunk (2,048 keys) over a 4,800-key context: the two float32 kernels against each other and the plain kernel.
    const Geometry& g = preset("4b");
    const Pool pool(g, 700, 441);
    const NormWeights weights = make_weights(g, 442);
    NodePtr ctx = external_node(g, 4800, 443, nullptr);
    std::vector<RowPlan> rows;
    rows.push_back(make_row(iota_ids(0, 640), 4800, ctx));
    rows.push_back(make_row(iota_ids(640, 1), 4800, ctx));
    const PassResult plain = run_device(g, pool, weights, rows, variants()[0]);
    const PassResult first = run_device(g, pool, weights, rows, {AttnElem::F32, AttentionPath::TiledF32, "f32/tiled", 0});
    const PassResult split = run_device(g, pool, weights, rows, split_variant());
    for (std::size_t r = 0; r < rows.size(); ++r) {
        const Diff a = compare(split.rows[r].out, plain.rows[r].out);
        const Diff b = compare(split.rows[r].out, first.rows[r].out);
        std::printf("  [measured] 4b ctx 4800 row %zu f32/split vs f32/plain scaled %.3e, vs f32/tiled scaled %.3e\n", r, a.scaled(),
                    b.scaled());
        CHECK(a.scaled() <= 1e-5);
        CHECK(b.scaled() <= 1e-5);
        CHECK(all_finite(split.rows[r].out));
    }
}

// ---- suite: tables ----

SEPAL_TEST(tables, check_attention_rows) {
    float scratch = 0.0f;
    const KvSegment ok{&scratch, &scratch, 4, 0};
    const std::vector<RowSpan> spans = {{0, 4}};
    const std::vector<AttentionRowDesc> good = {{0, 1, 0, 0}};
    sepal::cuda::check_attention_rows(spans, good, std::vector<KvSegment>{ok});
    CHECK_THROWS(sepal::cuda::check_attention_rows(spans, std::vector<AttentionRowDesc>{{0, 0, 0, 0}}, std::vector<KvSegment>{ok}),
                 std::invalid_argument); // no segments
    CHECK_THROWS(sepal::cuda::check_attention_rows(spans, std::vector<AttentionRowDesc>{{0, 2, 0, 0}}, std::vector<KvSegment>{ok}),
                 std::invalid_argument); // outside the array
    CHECK_THROWS(sepal::cuda::check_attention_rows(spans, good, std::vector<KvSegment>{{&scratch, &scratch, 5, 0}}),
                 std::invalid_argument); // own segment of the wrong size
    CHECK_THROWS(sepal::cuda::check_attention_rows(spans, good, std::vector<KvSegment>{{nullptr, &scratch, 4, 0}}),
                 std::invalid_argument); // null pointer
    CHECK_THROWS(sepal::cuda::check_attention_rows(spans, std::vector<AttentionRowDesc>{{0, 1, 0xFFFFFFFFu, 0}}, std::vector<KvSegment>{ok}),
                 std::invalid_argument); // positions overflow
    // An indexed row must say how many tokens precede each segment.
    const std::vector<AttentionRowDesc> indexed = {{0, 2, 0, 1}};
    sepal::cuda::check_attention_rows(spans, indexed, std::vector<KvSegment>{{&scratch, &scratch, 3, 0}, {&scratch, &scratch, 4, 3}});
    CHECK_THROWS(sepal::cuda::check_attention_rows(spans, indexed, std::vector<KvSegment>{{&scratch, &scratch, 3, 0}, {&scratch, &scratch, 4, 2}}),
                 std::invalid_argument); // the own segment's `before` is wrong
    CHECK_THROWS(sepal::cuda::check_attention_rows(spans, indexed, std::vector<KvSegment>{{&scratch, &scratch, 3, 1}, {&scratch, &scratch, 4, 4}}),
                 std::invalid_argument); // the first segment's `before` is not zero
}

// ---- suite: chains ----
// A row over a long chain of snapshot links: finding a key's segment by binary search (indexed) gives the bits of the scan from
// the first segment, in every kernel.

SEPAL_TEST(chains, indexed_segments_give_the_bits_of_scanned_segments) {
    for (const Geometry* g : {&tiny(), &preset("0.8b")}) {
        for (const std::uint32_t links : {1u, 2u, 7u, 60u, 250u}) {
            const Pool pool(*g, 80, 91 + links);
            const NormWeights weights = make_weights(*g, 92);
            NodePtr chain;
            std::uint32_t context = 0;
            for (std::uint32_t i = 0; i < links; ++i) {
                const std::uint32_t tokens = 1 + (i * 7) % 5; // one-token links among longer ones
                chain = external_node(*g, tokens, 93 + i, chain);
                context += tokens;
            }
            // A long row, a one-token row, and a row of another chain, in one launch.
            std::vector<RowPlan> rows;
            rows.push_back(make_row(iota_ids(0, 37), context, chain));
            rows.push_back(make_row(iota_ids(37, 1), context, chain));
            rows.push_back(make_row(iota_ids(38, 11), 0, nullptr));
            for (const Variant& variant : variants()) {
                const PassResult scanned = run_device(*g, pool, weights, rows, variant);
                PassOptions options;
                options.indexed = true;
                const PassResult binary = run_device(*g, pool, weights, rows, variant, options);
                for (std::size_t r = 0; r < rows.size(); ++r) {
                    char what[96];
                    std::snprintf(what, sizeof what, "%s %u links row %zu", g->name.c_str(), links, r);
                    CHECK(identical_rows(binary.rows[r], scanned.rows[r], what, variant.name));
                }
            }
        }
    }
}

SEPAL_TEST(chains, second_tiled_kernel_gives_the_bits_of_the_first) {
    // v2 (8 warps, 128 query rows, separate key and value buffers) keeps v1's tiles and order of additions: the same bits.
    const Variant tiled{AttnElem::F16, AttentionPath::Tiled, "f16/tiled"};
    if (!sepal::cuda::device_has_tensor_cores()) {
        std::printf("  skipped: no tensor cores on this device (the tiled kernels need compute capability 8.0)\n");
        return;
    }
    for (const Geometry* g : {&preset("0.8b"), &preset("4b")}) {
        const Pool pool(*g, 400, 191);
        const NormWeights weights = make_weights(*g, 192);
        NodePtr chain;
        std::uint32_t context = 0;
        for (std::uint32_t i = 0; i < 5; ++i) {
            const std::uint32_t tokens = 1 + (i * 37) % 70;
            chain = external_node(*g, tokens, 193 + i, chain);
            context += tokens;
        }
        std::vector<RowPlan> rows;
        rows.push_back(make_row(iota_ids(0, 300), context, chain));
        rows.push_back(make_row(iota_ids(300, 1), context, chain));
        rows.push_back(make_row(iota_ids(301, 45), 0, nullptr));
        rows.push_back(make_row(iota_ids(346, 33), 0, nullptr));
        set_tiled_kernel(TiledKernel::First);
        const PassResult first = run_device(*g, pool, weights, rows, tiled);
        set_tiled_kernel(TiledKernel::Second);
        const PassResult second = run_device(*g, pool, weights, rows, tiled);
        reset_tiled_kernel();
        for (std::size_t r = 0; r < rows.size(); ++r) {
            char what[96];
            std::snprintf(what, sizeof what, "%s row %zu", g->name.c_str(), r);
            CHECK(identical_rows(second.rows[r], first.rows[r], what, "v2 vs v1"));
        }
    }
}

SEPAL_TEST(tables, unsupported_geometry_and_paths) {
    // Tiled needs F16 and a supported geometry; Auto falls back to the plain kernel; a bad shape is refused.
    AttentionShape shape{8, 2, 256, 64, 1e7, 1e-6};
    CHECK(sepal::cuda::attention_tiled_supported(shape, AttnElem::F16));
    CHECK(!sepal::cuda::attention_tiled_supported(shape, AttnElem::F32));
    AttentionShape odd_dim{4, 2, 96, 24, 1e7, 1e-6};
    CHECK(!sepal::cuda::attention_tiled_supported(odd_dim, AttnElem::F16));
    AttentionShape group3{6, 2, 64, 16, 1e7, 1e-6};
    CHECK(!sepal::cuda::attention_tiled_supported(group3, AttnElem::F16));

    for (const AttentionShape& s : {odd_dim, group3}) {
        ModelConfig cfg = sepal::tiny_test_config();
        cfg.attn_heads = s.heads;
        cfg.attn_kv_heads = s.kv_heads;
        cfg.attn_head_dim = s.head_dim;
        cfg.partial_rotary_factor = static_cast<double>(s.rotary_dims) / static_cast<double>(s.head_dim);
        const Geometry g = make_geometry(cfg);
        const Pool pool(g, 100, 121);
        const NormWeights weights = make_weights(g, 122);
        const NodePtr ctx = external_node(g, 27, 123, nullptr);
        std::vector<RowPlan> rows = {make_row(iota_ids(0, 40), 27, ctx)};
        RefPass<float> ref(g, pool, weights, rows);
        const Variant autov{AttnElem::F16, AttentionPath::Auto, "f16/auto"};
        const PassResult result = run_device(g, pool, weights, rows, autov);
        CHECK(result.used == AttentionPath::Reference);
        check_against_reference(g.name.c_str(), autov, result.rows[0], ref.row(0), nullptr, false, false);
        const Variant tiled{AttnElem::F16, AttentionPath::Tiled, "f16/tiled"};
        CHECK_THROWS(run_device(g, pool, weights, rows, tiled), std::invalid_argument); // also without tensor cores
    }
    {
        const Geometry& g = tiny();
        const Pool pool(g, 100, 124);
        const NormWeights weights = make_weights(g, 125);
        std::vector<RowPlan> rows = {make_row(iota_ids(0, 40), 0, nullptr)};
        const bool tc = sepal::cuda::device_has_tensor_cores();
        CHECK(run_device(g, pool, weights, rows, {AttnElem::F16, AttentionPath::Auto, "f16/auto"}).used == (tc ? AttentionPath::Tiled : AttentionPath::Reference));
        CHECK(run_device(g, pool, weights, rows, {AttnElem::F32, AttentionPath::Auto, "f32/auto"}).used == AttentionPath::TiledF32);
        CHECK_THROWS(run_device(g, pool, weights, rows, {AttnElem::F32, AttentionPath::Tiled, "f32/tiled"}), std::invalid_argument);
        CHECK_THROWS(run_device(g, pool, weights, rows, {AttnElem::F16, AttentionPath::TiledF32, "f16/tiledf32"}), std::invalid_argument);
    }
    AttentionShape bad{7, 2, 64, 16, 1e7, 1e-6};
    CHECK_THROWS(sepal::cuda::launch_attention_prepare(bad, AttentionRows{}, AttnElem::F32, {}, {}, {}, nullptr, nullptr, {}, nullptr),
                 std::invalid_argument);
}

SEPAL_TEST(tables, subset_zero_length_strides_and_output_type) {
    for (const Geometry* gp : {&tiny(), &preset("0.8b")}) {
        const Geometry& g = *gp;
        const Pool pool(g, 400, 131);
        const NormWeights weights = make_weights(g, 132);
        const NodePtr ctx = external_node(g, 27, 133, nullptr);
        std::vector<RowPlan> rows = {make_row(iota_ids(0, 20), 27, ctx), make_row({}, 0, nullptr), make_row(iota_ids(20, 33), 5, nullptr),
                                     make_row(iota_ids(60, 7), 27, ctx), make_row(iota_ids(70, 70), 0, ctx)};
        for (const Variant& variant : variants()) {
            const PassResult full = run_device(g, pool, weights, rows, variant);
            // Only rows 4 and 0 run: the rest of the output buffer keeps its sentinel, and the rows that ran match.
            PassOptions subset;
            subset.run = {4, 0};
            const PassResult part = run_device(g, pool, weights, rows, variant, subset);
            const std::size_t esize = elem_size(variant.kv);
            const std::size_t stride_bytes = (g.q_width + 8) * esize;
            for (std::size_t r = 0; r < rows.size(); ++r) {
                const bool ran = r == 4 || r == 0;
                for (std::uint32_t t = 0; t < rows[r].length(); ++t) {
                    const unsigned char* line = part.out_bytes.data() + (part.row_offsets[r] + t) * stride_bytes;
                    bool untouched = true;
                    for (std::size_t b = 0; b < stride_bytes; ++b) untouched = untouched && line[b] == 0x5A;
                    CHECK(untouched == !ran);
                }
                if (ran) CHECK(same_bits(part.rows[r].out, full.rows[r].out));
            }
            // The 8 padding columns after each output row are never written.
            bool padding_untouched = true;
            for (std::size_t token = 0; token < full.out_bytes.size() / stride_bytes; ++token) {
                for (std::size_t b = g.q_width * esize; b < stride_bytes; ++b) {
                    padding_untouched = padding_untouched && full.out_bytes[token * stride_bytes + b] == 0x5A;
                }
            }
            CHECK(padding_untouched);
            CHECK(all_finite(full.rows[4].out));
        }
        // A float32 output from float16 keys and values holds exactly what the float16 output rounds.
        for (const Variant* variant : f16_variants()) {
            PassOptions wide;
            wide.use_out_override = true;
            wide.out_type_override = AttnElem::F32;
            const PassResult narrow = run_device(g, pool, weights, rows, *variant);
            const PassResult f32out = run_device(g, pool, weights, rows, *variant, wide);
            bool same = true;
            for (std::size_t r = 0; r < rows.size(); ++r) {
                for (std::size_t i = 0; i < narrow.rows[r].out.size(); ++i) {
                    same = same && narrow.rows[r].out[i] == round_half(f32out.rows[r].out[i]);
                }
            }
            CHECK(same);
        }
    }
}

} // namespace
