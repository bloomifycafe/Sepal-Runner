// The row executor, snapshots and the two readouts against the oracle, in float64; then the same requests in
// float32 with the difference from the float64 oracle reported.

#include <algorithm>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "golden_fixture.hpp"
#include "sepal/ref/executor.hpp"
#include "sepal/ref/readout.hpp"

using namespace sepal;
using namespace sepal::test;
using namespace sepal::ref;

namespace {

const double tol = oracle_tolerance;

template <class T>
std::vector<RowResult<T>> run(const std::vector<Row<T>>& rows) {
    const Model<T>& m = model<T>();
    return run_rows<T>(*m.gemm, m.config, m.weights, rows);
}

template <class T>
bool same_bits(const BasicTensor<T>& a, const BasicTensor<T>& b) {
    if (a.shape() != b.shape()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) return false;
    }
    return true;
}

// Every tensor of a snapshot, copied out in the oracle's layout.
template <class T>
std::vector<BasicTensor<T>> contents(const Snapshot<T>& snapshot, const ModelConfig& config) {
    std::vector<BasicTensor<T>> out;
    for (const LayerState<T>& layer : snapshot.layers) {
        if (const DeltaNetStatePtr<T>* deltanet = std::get_if<DeltaNetStatePtr<T>>(&layer)) {
            out.push_back((*deltanet)->recurrent);
            out.push_back((*deltanet)->conv_tail);
        } else {
            const AttentionState<T>& attention = *std::get<AttentionStatePtr<T>>(layer);
            out.push_back(gather_attention(attention, &AttentionState<T>::keys, config));
            out.push_back(gather_attention(attention, &AttentionState<T>::values, config));
        }
    }
    return out;
}

// One hidden state [hidden] of the golden request, named in the header as [row name, index among its readouts].
template <class T>
std::span<const T> hidden_at(const std::vector<RowResult<T>>& results, const json::Value& reference) {
    const BasicTensor<T>& hidden = results.at(request_row_index(reference.at(0).as_string())).hidden;
    return hidden.mat().row(static_cast<std::size_t>(reference.at(1).as_uint()));
}

// The oracle's hidden states of the golden request, as results, so a readout can be tested on the oracle's input.
std::vector<RowResult<double>> oracle_results() {
    std::vector<RowResult<double>> results;
    for (const json::Value& entry : golden_case("request").at("rows").items()) results.push_back({named(entry, "hidden"), nullptr});
    return results;
}

template <class T>
BasicTensor<T> stack_rows(const std::vector<RowResult<T>>& results, const json::Value& references, std::size_t width) {
    BasicTensor<T> out({references.items().size(), width});
    for (std::size_t r = 0; r < references.items().size(); ++r) {
        const std::span<const T> row = hidden_at(results, references.at(r));
        std::copy(row.begin(), row.end(), out.data() + r * width);
    }
    return out;
}

// Both readouts of the golden questions from `results`, compared with the oracle's probabilities and logits.
template <class T>
void check_readouts(Worst& worst, const std::vector<RowResult<T>>& results, double tolerance) {
    const Model<T>& m = model<T>();
    const std::size_t hidden = dim(m.config.hidden_size);
    for (const json::Value& question : golden_case("pointer").items()) {
        const BasicTensor<T> options = stack_rows(results, question.at("options"), hidden);
        const double temperature = question.at("temperature").as_number();
        const std::span<const T> decide = hidden_at(results, question.at("decide"));
        EXPECT_CLOSE(worst, "pointer logits", pointer_logits<T>(*m.gemm, m.config, m.weights, decide, options.mat(), temperature),
                     named(question, "logits"), tolerance);
        const std::vector<T> probabilities = pointer_probabilities<T>(*m.gemm, m.config, m.weights, decide, options.mat(), temperature);
        EXPECT_CLOSE(worst, "pointer probabilities", probabilities, named(question, "probabilities"), tolerance);
        double sum = 0.0;
        for (const T p : probabilities) sum += static_cast<double>(p);
        CHECK_NEAR(sum, 1.0, 1e-6);
    }
    for (const json::Value& question : golden_case("label").items()) {
        const BasicTensor<T> rows = label_rows<T>(m.config, m.weights, token_list(question.at("label_token_ids")));
        const double temperature = question.at("temperature").as_number();
        const std::span<const T> slot = hidden_at(results, question.at("slot"));
        EXPECT_CLOSE(worst, "label-row logits", label_row_logits<T>(*m.gemm, slot, rows, temperature), named(question, "logits"), tolerance);
        EXPECT_CLOSE(worst, "label-row probabilities", label_row_probabilities<T>(*m.gemm, slot, rows, temperature),
                     named(question, "probabilities"), tolerance);
    }
}

template <class T>
Row<T> one_pass_row() {
    const json::Value& one = golden_case("one_pass");
    Row<T> row;
    row.token_ids = token_list(one.at("token_ids"));
    row.start_position = one.at("start_position").as_uint();
    row.readout_offsets = index_list(one.at("readout_offsets"));
    return row;
}

} // namespace

// ---- Per-layer hidden states ----

SEPAL_TEST(executor, per_layer_hidden_states_match_the_oracle) {
    const Model<double>& m = model<double>();
    const json::Value& ladder = golden_case("ladder");
    const std::vector<std::uint32_t> tokens = token_list(ladder.at("token_ids"));
    const std::size_t hidden = dim(m.config.hidden_size);

    Worst worst("[f64] per-layer hidden states (embedding, 8 layer outputs, final norm)");
    EXPECT_CLOSE(worst, "embedding", embed<double>(m.weights, tokens), named(ladder, "embedding"), tol);

    std::vector<TensorD> layer_outputs;
    const Continuation<double> run_result = continue_snapshot<double>(*m.gemm, m.config, m.weights, *empty_snapshot<double>(m.config), tokens,
                                                                      ladder.at("start_position").as_uint(), &layer_outputs);
    const TensorD& want = named(ladder, "layer_outputs"); // [layers, T, hidden]
    CHECK_EQ(layer_outputs.size(), want.shape().at(0));
    const std::size_t per_layer = tokens.size() * hidden;
    for (std::size_t i = 0; i < layer_outputs.size(); ++i) {
        Worst layer_worst("[f64]   layer " + std::to_string(i) + (m.config.is_full_attention(i) ? " (attention)" : " (deltanet)"));
        const std::span<const double> expected = want.span().subspan(i * per_layer, per_layer);
        EXPECT_CLOSE(layer_worst, "layer " + std::to_string(i) + " output", std::as_const(layer_outputs[i]).span(), expected, tol);
        worst.see<double>(layer_outputs[i].span(), expected);
    }
    const TensorD final_hidden = rms_norm<double>(run_result.last_layer.mat(), m.weights[final_norm_name].span(), m.config.rms_norm_eps);
    EXPECT_CLOSE(worst, "final hidden states", final_hidden, named(ladder, "final_hidden"), tol);
    CHECK_EQ(run_result.snapshot->n_tokens, tokens.size());
}

// ---- The request ----

SEPAL_TEST(executor, a_request_with_parents_matches_the_oracle) {
    const std::vector<Row<double>> rows = request_rows<double>();
    // The request lists a child before its parent, and a branch of a branch.
    CHECK(std::holds_alternative<ParentRow>(rows.at(0).parent));
    CHECK(std::get<ParentRow>(rows.at(0).parent).index > 0);
    const std::vector<RowResult<double>> results = run(rows);
    Worst worst("[f64] final hidden states at the readout positions, 5 rows");
    const std::vector<json::Value>& entries = golden_case("request").at("rows").items();
    for (std::size_t i = 0; i < entries.size(); ++i) {
        EXPECT_CLOSE(worst, "hidden of row " + entries[i].at("name").as_string(), results.at(i).hidden, named(entries[i], "hidden"), tol);
        CHECK_EQ(results.at(i).hidden.shape().at(0), rows[i].readout_offsets.size());
    }
}

SEPAL_TEST(executor, the_snapshot_of_a_state_row_matches_the_oracle) {
    const Model<double>& m = model<double>();
    const std::vector<RowResult<double>> results = run(request_rows<double>());
    const Snapshot<double>& snapshot = *results.at(request_row_index("state")).snapshot;
    const json::Value& want = golden_case("snapshot");
    CHECK_EQ(snapshot.n_tokens, want.at("n_tokens").as_uint());
    CHECK_EQ(snapshot.layers.size(), want.at("layers").items().size());
    Worst worst("[f64] snapshot contents after the state row (state, conv tail, keys, values of all layers)");
    for (std::size_t i = 0; i < snapshot.layers.size(); ++i) {
        const json::Value& layer = want.at("layers").at(i);
        const std::string name = "layer " + std::to_string(i);
        if (const DeltaNetStatePtr<double>* deltanet = std::get_if<DeltaNetStatePtr<double>>(&snapshot.layers[i])) {
            CHECK_EQ(layer.at("kind").as_string(), std::string("deltanet"));
            CHECK((*deltanet)->recurrent.shape() == named(layer, "recurrent").shape());
            CHECK((*deltanet)->conv_tail.shape() == named(layer, "conv_tail").shape());
            EXPECT_CLOSE(worst, name + " recurrent state", (*deltanet)->recurrent, named(layer, "recurrent"), tol);
            EXPECT_CLOSE(worst, name + " conv tail", (*deltanet)->conv_tail, named(layer, "conv_tail"), tol);
        } else {
            const AttentionState<double>& attention = *std::get<AttentionStatePtr<double>>(snapshot.layers[i]);
            CHECK_EQ(layer.at("kind").as_string(), std::string("attention"));
            CHECK_EQ(attention.tokens, snapshot.n_tokens);
            const TensorD keys = gather_attention(attention, &AttentionState<double>::keys, m.config);
            CHECK(keys.shape() == named(layer, "keys").shape());
            EXPECT_CLOSE(worst, name + " keys", keys, named(layer, "keys"), tol);
            EXPECT_CLOSE(worst, name + " values", gather_attention(attention, &AttentionState<double>::values, m.config), named(layer, "values"), tol);
        }
    }
}

// ---- Continuation ----

SEPAL_TEST(executor, continuing_from_a_snapshot_equals_one_pass) {
    const std::vector<RowResult<double>> chained = run(request_rows<double>());
    const std::vector<RowResult<double>> one = run<double>({one_pass_row<double>()});
    const json::Value& entry = golden_case("one_pass");

    Worst oracle("[f64] one pass over state + branch + leaf, against the oracle");
    EXPECT_CLOSE(oracle, "one-pass hidden states", one.at(0).hidden, named(entry, "hidden"), tol);

    // The same tokens as three rows (state, branch_a from the state's snapshot, leaf_a from branch_a's) give
    // the same hidden states as the single row. Measured here as exactly equal: every kernel of this path
    // treats a token the same whichever row brought it.
    Worst self("[f64] one pass against three chained rows, C++ against C++");
    const std::size_t hidden = one.at(0).hidden.shape().at(1);
    const std::vector<json::Value>& references = entry.at("chained").items();
    CHECK_EQ(references.size(), one.at(0).hidden.shape().at(0));
    for (std::size_t r = 0; r < references.size(); ++r) {
        const std::span<const double> from_chain = hidden_at(chained, references[r]);
        const std::span<const double> from_one = one.at(0).hidden.mat().row(r);
        CHECK_EQ(from_chain.size(), hidden);
        CHECK_EQ(self.see<double>(from_chain, from_one), 0.0);
    }

    // And the snapshot after the chain is the snapshot after the single pass.
    const Model<double>& m = model<double>();
    const std::vector<TensorD> after_chain = contents(*chained.at(request_row_index("leaf_a")).snapshot, m.config);
    const std::vector<TensorD> after_one = contents(*one.at(0).snapshot, m.config);
    CHECK_EQ(after_chain.size(), after_one.size());
    for (std::size_t i = 0; i < after_chain.size(); ++i) CHECK(same_bits(after_chain[i], after_one.at(i)));
}

SEPAL_TEST(executor, a_stored_snapshot_continues_like_the_row_that_made_it) {
    // The parent is the ORACLE's snapshot, loaded from the file as a stored snapshot would be, not a row.
    const SnapshotPtr<double> stored = snapshot_from_golden<double>(golden_case("snapshot"));
    const std::vector<json::Value>& entries = golden_case("request").at("rows").items();
    std::vector<Row<double>> rows = request_rows<double>();
    const std::size_t state_index = request_row_index("state");
    std::vector<Row<double>> branches;
    std::vector<std::size_t> origin;
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const ParentRow* parent = std::get_if<ParentRow>(&rows[i].parent);
        if (parent == nullptr || parent->index != state_index) continue;
        rows[i].parent = stored;
        branches.push_back(rows[i]);
        origin.push_back(i);
    }
    CHECK_EQ(branches.size(), 3u);
    const std::vector<RowResult<double>> results = run(branches);
    Worst worst("[f64] branches continued from a stored snapshot, against the oracle");
    for (std::size_t b = 0; b < branches.size(); ++b) {
        EXPECT_CLOSE(worst, "hidden of a branch from a stored snapshot", results.at(b).hidden, named(entries[origin[b]], "hidden"), tol);
    }
}

// ---- Isolation ----

SEPAL_TEST(executor, branches_together_equal_branches_alone) {
    const Model<double>& m = model<double>();
    const std::vector<Row<double>> rows = request_rows<double>();
    const std::vector<RowResult<double>> together = run(rows);
    const SnapshotPtr<double> state = together.at(request_row_index("state")).snapshot;
    std::size_t compared = 0;
    for (const char* name : {"branch_a", "branch_b", "branch_c"}) {
        Row<double> alone = rows.at(request_row_index(name));
        alone.parent = state;
        const std::vector<RowResult<double>> result = run<double>({alone});
        const RowResult<double>& in_request = together.at(request_row_index(name));
        CHECK(same_bits(result.at(0).hidden, in_request.hidden));
        const std::vector<TensorD> a = contents(*result.at(0).snapshot, m.config);
        const std::vector<TensorD> b = contents(*in_request.snapshot, m.config);
        for (std::size_t i = 0; i < a.size(); ++i) CHECK(same_bits(a[i], b.at(i)));
        ++compared;
    }
    CHECK_EQ(compared, 3u);

    // Request order does not matter either: the rows reversed give the same bits per row.
    std::vector<Row<double>> reversed(rows.rbegin(), rows.rend());
    for (Row<double>& row : reversed) {
        if (ParentRow* parent = std::get_if<ParentRow>(&row.parent)) parent->index = rows.size() - 1 - parent->index;
    }
    const std::vector<RowResult<double>> backwards = run(reversed);
    for (std::size_t i = 0; i < rows.size(); ++i) CHECK(same_bits(backwards.at(rows.size() - 1 - i).hidden, together.at(i).hidden));
}

SEPAL_TEST(executor, a_snapshot_is_unchanged_after_children_run_from_it) {
    const Model<double>& m = model<double>();
    Row<double> state_row = request_rows<double>().at(request_row_index("state"));
    const SnapshotPtr<double> snapshot = run<double>({state_row}).at(0).snapshot;
    const std::vector<TensorD> before = contents(*snapshot, m.config);
    const std::vector<LayerState<double>> layers_before = snapshot->layers;

    std::vector<Row<double>> children;
    for (const char* name : {"branch_a", "branch_b", "branch_c", "branch_a"}) {
        Row<double> child = request_rows<double>().at(request_row_index(name));
        child.parent = snapshot;
        children.push_back(child);
    }
    const std::vector<RowResult<double>> first = run(children);
    const std::vector<RowResult<double>> second = run(children);

    const std::vector<TensorD> after = contents(*snapshot, m.config);
    CHECK_EQ(snapshot->n_tokens, state_row.token_ids.size());
    CHECK_EQ(after.size(), before.size());
    for (std::size_t i = 0; i < before.size(); ++i) CHECK(same_bits(after.at(i), before[i]));
    CHECK(snapshot->layers == layers_before); // the same state objects, not replacements

    // Children share the parent's state by reference: each child's attention state links to the parent's own
    // object and holds only the child's tokens, and no child shares anything of its own with a sibling.
    for (std::size_t i = 0; i < snapshot->layers.size(); ++i) {
        if (!m.config.is_full_attention(i)) continue;
        const AttentionStatePtr<double>& parent = std::get<AttentionStatePtr<double>>(snapshot->layers[i]);
        for (std::size_t c = 0; c < children.size(); ++c) {
            const AttentionStatePtr<double>& child = std::get<AttentionStatePtr<double>>(first.at(c).snapshot->layers.at(i));
            CHECK(child->parent == parent);
            CHECK_EQ(child->keys.shape().at(0), children[c].token_ids.size());
            CHECK_EQ(child->tokens, snapshot->n_tokens + children[c].token_ids.size());
        }
    }
    // Running the same children again from the same snapshot gives the same bits (the first run left no trace),
    // and the two copies of branch_a inside one request agree.
    for (std::size_t c = 0; c < children.size(); ++c) CHECK(same_bits(first.at(c).hidden, second.at(c).hidden));
    CHECK(same_bits(first.at(0).hidden, first.at(3).hidden));
}

// ---- Admission ----

SEPAL_TEST(executor, admission_refuses_what_could_only_fail_later) {
    const Model<double>& m = model<double>();
    const auto refused = [&](const std::vector<Row<double>>& rows) {
        try {
            (void)run_rows<double>(*m.gemm, m.config, m.weights, rows);
        } catch (const std::invalid_argument&) {
            return true;
        }
        return false;
    };
    Row<double> good;
    good.token_ids = {1, 2, 3};
    good.readout_offsets = {2};
    CHECK(!refused({good}));

    Row<double> empty = good;
    empty.token_ids.clear();
    empty.readout_offsets.clear();
    CHECK(refused({empty}));
    Row<double> outside_vocabulary = good;
    outside_vocabulary.token_ids[1] = 64;
    CHECK(refused({outside_vocabulary}));
    Row<double> bad_offset = good;
    bad_offset.readout_offsets = {3};
    CHECK(refused({bad_offset}));
    Row<double> absent_parent = good;
    absent_parent.parent = ParentRow{5};
    CHECK(refused({good, absent_parent}));
    Row<double> own_parent = good;
    own_parent.parent = ParentRow{0};
    CHECK(refused({own_parent}));
    Row<double> first = good;
    Row<double> second = good;
    first.parent = ParentRow{1};
    second.parent = ParentRow{0};
    CHECK(refused({first, second}));
    Row<double> null_snapshot = good;
    null_snapshot.parent = SnapshotPtr<double>();
    CHECK(refused({null_snapshot}));

    // A snapshot of another geometry: its layer kinds are not this model's.
    auto foreign = std::make_shared<Snapshot<double>>(*empty_snapshot<double>(m.config));
    std::swap(foreign->layers[0], foreign->layers[3]);
    Row<double> foreign_parent = good;
    foreign_parent.parent = SnapshotPtr<double>(foreign);
    CHECK(refused({foreign_parent}));
    auto short_snapshot = std::make_shared<Snapshot<double>>(*empty_snapshot<double>(m.config));
    short_snapshot->layers.pop_back();
    Row<double> short_parent = good;
    short_parent.parent = SnapshotPtr<double>(short_snapshot);
    CHECK(refused({short_parent}));
}

SEPAL_TEST(executor, parents_run_before_children_and_request_order_is_kept_otherwise) {
    std::vector<Row<double>> rows(5);
    rows[0].parent = ParentRow{3}; // 0 <- 3 <- 4
    rows[3].parent = ParentRow{4};
    rows[2].parent = ParentRow{1};
    const std::vector<std::size_t> order = execution_order<double>(rows);
    const std::vector<std::size_t> want = {4, 3, 0, 1, 2};
    CHECK(order == want);
}

// ---- Readouts ----

SEPAL_TEST(readout, both_readouts_match_the_oracle_on_the_oracle_hidden_states) {
    Worst worst("[f64] readouts alone, from the oracle's hidden states (pointer and label-row, logits and probabilities)");
    check_readouts<double>(worst, oracle_results(), tol);
}

SEPAL_TEST(readout, both_readouts_match_the_oracle_end_to_end) {
    Worst worst("[f64] readouts end to end, from this engine's hidden states");
    check_readouts<double>(worst, run(request_rows<double>()), tol);
}

SEPAL_TEST(readout, the_pointer_head_applies_its_biases) {
    // With the biases removed the logits move by far more than the tolerance, so a bias-free head cannot pass.
    const Model<double>& m = model<double>();
    std::map<std::string, TensorD, std::less<>> arrays = golden().weights();
    for (const char* name : {"head.q.bias", "head.k.bias"}) {
        for (double& value : arrays.at(name).span()) value = 0.0;
    }
    const Weights<double> bias_free = Weights<double>::from_arrays(m.config, arrays);
    const std::vector<RowResult<double>> results = oracle_results();
    const json::Value& question = golden_case("pointer").at(0);
    const TensorD options = stack_rows(results, question.at("options"), dim(m.config.hidden_size));
    const std::vector<double> logits = pointer_logits<double>(*m.gemm, m.config, bias_free, hidden_at(results, question.at("decide")),
                                                              options.mat(), question.at("temperature").as_number());
    CHECK(max_abs_diff<double>(logits, named(question, "logits").span()) > 1e-4);
}

SEPAL_TEST(readout, bad_questions_are_refused) {
    const Model<double>& m = model<double>();
    const std::vector<double> h(dim(m.config.hidden_size), 0.5);
    const TensorD options({2, dim(m.config.hidden_size)}, 0.25);
    CHECK_THROWS(pointer_logits<double>(*m.gemm, m.config, m.weights, h, options.mat(), 0.0), std::invalid_argument);
    CHECK_THROWS(pointer_logits<double>(*m.gemm, m.config, m.weights, h, options.mat(), -1.0), std::invalid_argument);
    const std::vector<std::uint32_t> twice = {3, 3};
    const std::vector<std::uint32_t> outside = {3, 64};
    const std::vector<std::uint32_t> none;
    CHECK_THROWS(label_rows<double>(m.config, m.weights, twice), std::invalid_argument);
    CHECK_THROWS(label_rows<double>(m.config, m.weights, outside), std::invalid_argument);
    CHECK_THROWS(label_rows<double>(m.config, m.weights, none), std::invalid_argument);
    const std::vector<std::uint32_t> labels = {3, 5};
    const TensorD rows = label_rows<double>(m.config, m.weights, labels);
    CHECK_THROWS(label_row_logits<double>(*m.gemm, h, rows, 0.0), std::invalid_argument);
    // Temperature divides the logits: twice the temperature, half the logits.
    const std::vector<double> at_one = label_row_logits<double>(*m.gemm, h, rows, 1.0);
    const std::vector<double> at_two = label_row_logits<double>(*m.gemm, h, rows, 2.0);
    CHECK_NEAR(at_two.at(0), at_one.at(0) / 2.0, 1e-15);
}

// ---- Float32 ----
//
// The same requests in float32. The differences from the float64 oracle are REPORTED; the bounds asserted
// below are one loose sanity limit, 1e-2, some 250 times the largest value measured on the tiny geometry
// (4e-5; README, "Measured against the oracle"). It is there to catch a broken path, not to state an accuracy.

SEPAL_TEST(float32, the_same_requests_in_float32) {
    const Model<float>& m = model<float>();
    const double sanity = 1e-2;

    // Per-layer ladder.
    const json::Value& ladder = golden_case("ladder");
    const std::vector<std::uint32_t> tokens = token_list(ladder.at("token_ids"));
    std::vector<Tensor> layer_outputs;
    const Continuation<float> ladder_run = continue_snapshot<float>(*m.gemm, m.config, m.weights, *empty_snapshot<float>(m.config), tokens,
                                                                    ladder.at("start_position").as_uint(), &layer_outputs);
    const TensorD& want = named(ladder, "layer_outputs");
    const std::size_t per_layer = tokens.size() * dim(m.config.hidden_size);
    {
        Worst worst("[f32] per-layer hidden states against the float64 oracle (8 layer outputs, final norm)");
        for (std::size_t i = 0; i < layer_outputs.size(); ++i) {
            Worst layer_worst("[f32]   layer " + std::to_string(i));
            const std::span<const double> expected = want.span().subspan(i * per_layer, per_layer);
            EXPECT_CLOSE(layer_worst, "float32 layer output", std::as_const(layer_outputs[i]).span(), expected, sanity);
            worst.see<float>(layer_outputs[i].span(), expected);
        }
        const Tensor final_hidden = rms_norm<float>(ladder_run.last_layer.mat(), m.weights[final_norm_name].span(), m.config.rms_norm_eps);
        EXPECT_CLOSE(worst, "float32 final hidden states", final_hidden, named(ladder, "final_hidden"), sanity);
    }

    // The request, the snapshot, and one pass.
    const std::vector<RowResult<float>> results = run(request_rows<float>());
    {
        Worst worst("[f32] final hidden states at the readout positions, 5 rows");
        const std::vector<json::Value>& entries = golden_case("request").at("rows").items();
        for (std::size_t i = 0; i < entries.size(); ++i) {
            EXPECT_CLOSE(worst, "float32 hidden", results.at(i).hidden, named(entries[i], "hidden"), sanity);
        }
    }
    {
        Worst worst("[f32] snapshot contents after the state row");
        const std::vector<Tensor> got = contents(*results.at(request_row_index("state")).snapshot, m.config);
        const std::vector<json::Value>& layers = golden_case("snapshot").at("layers").items();
        for (std::size_t i = 0; i < layers.size(); ++i) {
            const bool deltanet = layers[i].at("kind").as_string() == "deltanet";
            EXPECT_CLOSE(worst, "float32 snapshot tensor", got.at(2 * i), named(layers[i], deltanet ? "recurrent" : "keys"), sanity);
            EXPECT_CLOSE(worst, "float32 snapshot tensor", got.at(2 * i + 1), named(layers[i], deltanet ? "conv_tail" : "values"), sanity);
        }
    }
    {
        Worst worst("[f32] readout logits and probabilities, both readouts");
        check_readouts<float>(worst, results, sanity);
    }

    // The request invariants hold in float32 as they do in float64: exactly.
    const std::vector<RowResult<float>> one = run<float>({one_pass_row<float>()});
    const std::vector<json::Value>& references = golden_case("one_pass").at("chained").items();
    Worst self("[f32] one pass against three chained rows, C++ against C++");
    for (std::size_t r = 0; r < references.size(); ++r) {
        const std::span<const float> from_one = one.at(0).hidden.mat().row(r);
        std::vector<double> as_double(from_one.begin(), from_one.end());
        CHECK_EQ(self.see<float>(hidden_at(results, references[r]), as_double), 0.0);
    }
}
