// Snapshots: what a run of tokens leaves behind for the rows that continue it.
// A port of sepal/refengine/snapshot.py.
//
// Per Gated DeltaNet layer that is the recurrent state and the conv tail (the last K-1 conv inputs); per
// attention layer the keys (after their norm and RoPE) and the values of every token so far.
//
// A snapshot is an immutable value. Every state is reached only through a pointer to const, is built once and
// is never written again, so any number of rows can continue one snapshot at the same time and none can see
// another. Children SHARE their parent's state by reference: a child's attention state holds a link to its
// parent's and only its own new keys and values, so nothing a parent computed is copied per child.
#pragma once

#include <cstddef>
#include <memory>
#include <utility>
#include <variant>
#include <vector>

#include "sepal/config.hpp"
#include "sepal/ref/primitives.hpp"
#include "sepal/tensor.hpp"

namespace sepal::ref {

/// recurrent: [value heads, key head dim, value head dim]. conv_tail: [K-1, conv_dim], oldest input first.
template <class T>
struct DeltaNetState {
    BasicTensor<T> recurrent;
    BasicTensor<T> conv_tail;
};

/// The keys and values of every token so far, in token order: the parent's (shared, possibly none) followed by
/// `keys` and `values`, this run's own, each [own tokens, kv heads, head dim].
template <class T>
struct AttentionState {
    std::shared_ptr<const AttentionState> parent;
    BasicTensor<T> keys;
    BasicTensor<T> values;
    std::size_t tokens = 0; // the parent's tokens plus this run's own

    std::size_t own_tokens() const { return tokens - (parent ? parent->tokens : 0); }
};

template <class T>
using DeltaNetStatePtr = std::shared_ptr<const DeltaNetState<T>>;
template <class T>
using AttentionStatePtr = std::shared_ptr<const AttentionState<T>>;
template <class T>
using LayerState = std::variant<DeltaNetStatePtr<T>, AttentionStatePtr<T>>;

/// The state after `n_tokens` tokens: one layer state per decoder layer, in layer order.
template <class T>
struct Snapshot {
    std::size_t n_tokens = 0;
    std::vector<LayerState<T>> layers;
};

template <class T>
using SnapshotPtr = std::shared_ptr<const Snapshot<T>>;

/// The runs of an attention state from the oldest ancestor to the state itself, in token order.
template <class T>
std::vector<const AttentionState<T>*> attention_segments(const AttentionState<T>& state) {
    std::vector<const AttentionState<T>*> chain;
    for (const AttentionState<T>* s = &state; s != nullptr; s = s->parent.get()) chain.push_back(s);
    return std::vector<const AttentionState<T>*>(chain.rbegin(), chain.rend());
}

/// All keys (or all values) of a state as one tensor [tokens, kv heads, head dim]: the oracle's layout, for
/// comparing with it and for storing a snapshot. `member` is &AttentionState<T>::keys or ::values.
template <class T>
BasicTensor<T> gather_attention(const AttentionState<T>& state, BasicTensor<T> AttentionState<T>::*member,
                                const ModelConfig& config) {
    BasicTensor<T> out({state.tokens, dim(config.attn_kv_heads), dim(config.attn_head_dim)});
    std::size_t at = 0;
    for (const AttentionState<T>* segment : attention_segments(state)) {
        const BasicTensor<T>& part = segment->*member;
        std::copy(part.span().begin(), part.span().end(), out.span().begin() + static_cast<std::ptrdiff_t>(at));
        at += part.size();
    }
    return out;
}

/// What a layer holds before any token: a zero recurrent state and conv tail, or no keys and values.
template <class T>
LayerState<T> empty_layer_state(const ModelConfig& config, std::uint64_t layer) {
    if (!config.is_full_attention(layer)) {
        auto state = std::make_shared<DeltaNetState<T>>();
        state->recurrent = BasicTensor<T>({dim(config.dn_value_heads), dim(config.dn_key_head_dim), dim(config.dn_value_head_dim)});
        state->conv_tail = BasicTensor<T>({dim(config.conv_kernel) - 1, dim(config.dn_conv_dim())});
        return DeltaNetStatePtr<T>(std::move(state));
    }
    auto state = std::make_shared<AttentionState<T>>();
    state->keys = BasicTensor<T>({0, dim(config.attn_kv_heads), dim(config.attn_head_dim)});
    state->values = state->keys;
    return AttentionStatePtr<T>(std::move(state));
}

/// The snapshot a row without a parent continues: nothing has been read yet.
template <class T>
SnapshotPtr<T> empty_snapshot(const ModelConfig& config) {
    auto snapshot = std::make_shared<Snapshot<T>>();
    for (std::uint64_t i = 0; i < config.num_layers; ++i) snapshot->layers.push_back(empty_layer_state<T>(config, i));
    return snapshot;
}

} // namespace sepal::ref
