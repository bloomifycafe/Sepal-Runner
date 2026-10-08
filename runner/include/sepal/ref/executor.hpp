// The row executor: the engine's request semantics, one row at a time.
// A port of sepal/refengine/executor.py, function for function (that file is adapted from
// jaredpalmer/kev@5e42a7a kev/fused_qwen35.py, Apache-2.0).
//
// A request is a set of rows. A row is a run of tokens that continues a parent: a stored snapshot, another row
// of the same request, or nothing. It reads its parent's snapshot (the DeltaNet states, the attention keys and
// values) and then its own tokens causally, and leaves a snapshot of its own. Rows that share a parent never
// see each other.
//
// The optimised engine will interleave all rows layer by layer (docs/RUNNER.md section 6). This reference
// finishes one row before starting the next, parents first, and every row goes through the same function
// whether its parent was computed a moment ago or loaded from a store. A row's result is therefore a function
// of its own chain and of nothing else, bit for bit.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "sepal/config.hpp"
#include "sepal/gemm.hpp"
#include "sepal/ref/attention.hpp"
#include "sepal/ref/deltanet.hpp"
#include "sepal/ref/primitives.hpp"
#include "sepal/ref/snapshot.hpp"
#include "sepal/tensor.hpp"
#include "sepal/weights.hpp"

namespace sepal::ref {

/// A parent that is another row of the same request, by its index in the request.
struct ParentRow {
    std::size_t index = 0;
};

/// One run of tokens.
///
/// token_ids: at least one token. start_position: the position of the first token (a branch restarts where its
/// state ended). parent: the snapshot or the row this one continues; std::monostate starts from nothing.
/// readout_offsets: offsets into this row's own tokens whose final hidden states the caller wants, in the order
/// wanted.
template <class T>
struct Row {
    std::vector<std::uint32_t> token_ids;
    std::uint64_t start_position = 0;
    std::variant<std::monostate, SnapshotPtr<T>, ParentRow> parent;
    std::vector<std::size_t> readout_offsets;
};

/// hidden: [readout offsets, hidden], after the final norm. snapshot: the state after the row's last token.
template <class T>
struct RowResult {
    BasicTensor<T> hidden;
    SnapshotPtr<T> snapshot;
};

template <class T>
struct LayerResult {
    BasicTensor<T> output; // [T, hidden]
    LayerState<T> state;
};

/// One decoder layer on x [T, hidden], continuing `state`; `weights` is the layer's own scope.
/// `h = norm(x); r = x + mixer(h); y = r + mlp(norm(r))`, the mixer chosen by the kind of state the layer keeps.
template <class T>
LayerResult<T> decoder_layer(const GemmProvider& provider, BasicConstMatView<T> x, std::span<const double> positions,
                             const Weights<T>& weights, const LayerState<T>& state, const ModelConfig& config) {
    const BasicTensor<T> h = rms_norm<T>(x, weights["input_layernorm.weight"].span(), config.rms_norm_eps);
    BasicTensor<T> residual;
    LayerState<T> new_state;
    if (const DeltaNetStatePtr<T>* deltanet = std::get_if<DeltaNetStatePtr<T>>(&state)) {
        DeltaNetResult<T> mixed = deltanet_mixer<T>(provider, h.mat(), weights.scope("linear_attn"), **deltanet, config);
        residual = std::move(mixed.output);
        new_state = std::move(mixed.state);
    } else {
        AttentionResult<T> mixed = attention_mixer<T>(provider, h.mat(), positions, weights.scope("self_attn"),
                                                      std::get<AttentionStatePtr<T>>(state), config);
        residual = std::move(mixed.output);
        new_state = std::move(mixed.state);
    }
    for (std::size_t i = 0; i < residual.size(); ++i) residual[i] = x.data[i] + residual[i];

    const BasicTensor<T> n = rms_norm<T>(residual.mat(), weights["post_attention_layernorm.weight"].span(), config.rms_norm_eps);
    const Weights<T> mlp = weights.scope("mlp");
    BasicTensor<T> output = swiglu_mlp<T>(provider, n.mat(), mlp["gate_proj.weight"], mlp["up_proj.weight"], mlp["down_proj.weight"]);
    for (std::size_t i = 0; i < output.size(); ++i) output[i] = residual[i] + output[i];
    return {std::move(output), std::move(new_state)};
}

template <class T>
struct Continuation {
    BasicTensor<T> last_layer; // [T, hidden], BEFORE the final norm
    SnapshotPtr<T> snapshot;   // after the last token
};

/// The embedding rows of a run of tokens, [T, hidden]. Token ids must already be inside the vocabulary.
template <class T>
BasicTensor<T> embed(const Weights<T>& weights, std::span<const std::uint32_t> token_ids) {
    const BasicConstMatView<T> table = weights[embedding_name].mat();
    BasicTensor<T> x({token_ids.size(), table.cols});
    for (std::size_t t = 0; t < token_ids.size(); ++t) {
        const std::span<const T> row = table.row(token_ids[t]);
        std::copy(row.begin(), row.end(), x.data() + t * table.cols);
    }
    return x;
}

/// Run tokens through the whole stack from `parent`, which is only read.
///
/// Returns the last layer's output BEFORE the final norm and the snapshot after the last token. When
/// `layer_outputs` is given, the output of every layer is appended to it in layer order (the per-layer residual
/// ladder a mismatch with the oracle is located on).
template <class T>
Continuation<T> continue_snapshot(const GemmProvider& provider, const ModelConfig& config, const Weights<T>& weights,
                                  const Snapshot<T>& parent, std::span<const std::uint32_t> token_ids,
                                  std::uint64_t start_position, std::vector<BasicTensor<T>>* layer_outputs = nullptr) {
    std::vector<double> positions(token_ids.size());
    for (std::size_t t = 0; t < positions.size(); ++t) positions[t] = static_cast<double>(start_position) + static_cast<double>(t);

    BasicTensor<T> x = embed(weights, token_ids);
    auto snapshot = std::make_shared<Snapshot<T>>();
    snapshot->n_tokens = parent.n_tokens + token_ids.size();
    for (std::size_t i = 0; i < parent.layers.size(); ++i) {
        LayerResult<T> layer = decoder_layer<T>(provider, x.mat(), positions, weights.scope("layers." + std::to_string(i)),
                                                parent.layers[i], config);
        x = std::move(layer.output);
        snapshot->layers.push_back(std::move(layer.state));
        if (layer_outputs != nullptr) layer_outputs->push_back(x);
    }
    return {std::move(x), std::move(snapshot)};
}

/// Refuse a snapshot of another geometry: its layer kinds must be this model's, in order.
template <class T>
void check_snapshot(const ModelConfig& config, const Snapshot<T>& snapshot) {
    bool ok = snapshot.layers.size() == dim(config.num_layers);
    for (std::size_t i = 0; ok && i < snapshot.layers.size(); ++i) {
        const bool is_attention = std::holds_alternative<AttentionStatePtr<T>>(snapshot.layers[i]);
        const bool has_state = std::visit([](const auto& state) { return state != nullptr; }, snapshot.layers[i]);
        ok = has_state && is_attention == config.is_full_attention(i);
    }
    if (!ok) throw std::invalid_argument("the parent snapshot was not made by a model of this geometry");
}

/// Admission for one row: reject before any compute what could only fail or silently misread later.
template <class T>
void check_row(const ModelConfig& config, const Row<T>& row) {
    if (row.token_ids.empty()) throw std::invalid_argument("a row needs at least one token");
    for (const std::uint32_t token : row.token_ids) {
        if (token >= config.vocab_size) throw std::invalid_argument("a token id is outside the vocabulary");
    }
    for (const std::size_t offset : row.readout_offsets) {
        if (offset >= row.token_ids.size()) throw std::invalid_argument("a readout offset is outside the row");
    }
    if (const SnapshotPtr<T>* snapshot = std::get_if<SnapshotPtr<T>>(&row.parent)) {
        if (*snapshot == nullptr) throw std::invalid_argument("a row's parent snapshot is null");
        check_snapshot(config, **snapshot);
    }
}

/// The rows' indices with every parent row before its children, otherwise in request order.
/// A parent row must be part of the request: the request is the unit that is admitted, scheduled and answered.
/// Parents are given by index here, so a chain of parents that returns to itself is possible to write and is
/// refused.
template <class T>
std::vector<std::size_t> execution_order(std::span<const Row<T>> rows) {
    enum class Mark { None, Placing, Placed };
    std::vector<Mark> marks(rows.size(), Mark::None);
    std::vector<std::size_t> ordered;
    for (std::size_t start = 0; start < rows.size(); ++start) {
        // Walk up to the first ancestor already placed, then place the chain from the top down.
        std::vector<std::size_t> chain;
        for (std::size_t at = start; marks[at] != Mark::Placed;) {
            if (marks[at] == Mark::Placing) throw std::invalid_argument("a row is its own ancestor");
            marks[at] = Mark::Placing;
            chain.push_back(at);
            const ParentRow* parent = std::get_if<ParentRow>(&rows[at].parent);
            if (parent == nullptr) break;
            if (parent->index >= rows.size()) throw std::invalid_argument("a row's parent row is not part of the request");
            at = parent->index;
        }
        for (auto it = chain.rbegin(); it != chain.rend(); ++it) {
            marks[*it] = Mark::Placed;
            ordered.push_back(*it);
        }
    }
    return ordered;
}

/// Run a request. Returns one result per row, in request order.
///
/// Parents run before children. A child continues its parent's snapshot, which is the same value whether the
/// parent ran in this call or its snapshot was kept from an earlier one, so both give identical bits.
template <class T>
std::vector<RowResult<T>> run_rows(const GemmProvider& provider, const ModelConfig& config, const Weights<T>& weights,
                                   std::span<const Row<T>> rows) {
    for (const Row<T>& row : rows) check_row(config, row);
    std::vector<RowResult<T>> results(rows.size());
    const std::size_t hidden_size = dim(config.hidden_size);
    for (const std::size_t index : execution_order(rows)) {
        const Row<T>& row = rows[index];
        SnapshotPtr<T> parent;
        if (std::holds_alternative<std::monostate>(row.parent)) {
            parent = empty_snapshot<T>(config);
        } else if (const SnapshotPtr<T>* snapshot = std::get_if<SnapshotPtr<T>>(&row.parent)) {
            parent = *snapshot;
        } else {
            parent = results[std::get<ParentRow>(row.parent).index].snapshot;
        }
        Continuation<T> run = continue_snapshot<T>(provider, config, weights, *parent, row.token_ids, row.start_position);

        BasicTensor<T> picked({row.readout_offsets.size(), hidden_size});
        for (std::size_t r = 0; r < row.readout_offsets.size(); ++r) {
            const T* source = run.last_layer.data() + row.readout_offsets[r] * hidden_size;
            std::copy(source, source + hidden_size, picked.data() + r * hidden_size);
        }
        results[index].hidden = rms_norm<T>(picked.mat(), weights[final_norm_name].span(), config.rms_norm_eps);
        results[index].snapshot = std::move(run.snapshot);
    }
    return results;
}

} // namespace sepal::ref
