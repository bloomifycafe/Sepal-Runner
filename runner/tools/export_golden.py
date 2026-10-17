"""Export golden vectors for the C++ runner from the NumPy float64 reference engine (``sepal_runner.refengine``).

Run from the repository root::

    uv run python runner/tools/export_golden.py

It builds the ``tiny`` preset with seeded random weights, runs a fixed set of requests through the oracle and writes
``runner/tests/golden/tiny.sgv``. The C++ tests read that file and nothing else: the weights, the requests and the
expected results all come from it, so the C++ engine is compared with the oracle and never with itself.

File format (``SEPALGV1``; every integer and float is little-endian)::

    8 bytes   magic, the ASCII text "SEPALGV1"
    u32       H, the length of the header in bytes
    H bytes   the header: one UTF-8 JSON object
    u32       N, the number of arrays
    N times:
        u32       length of the name in bytes
        bytes     the name, UTF-8
        u32       rank R (0 for a scalar)
        R x u64   the shape
        f64 ...   the values in row-major (C) order, prod(shape) of them

The header holds what is not a float array: the geometry (``config``, with the field names of
``sepal_runner.refengine.ModelConfig``), the seed, the oracle's ``UNVERIFIED`` list as it stood when the file was written,
and ``cases``, which gives every test request (token ids, start positions, parents, readout offsets) and the NAME of
each array that holds an expected result. Array names are paths: ``weights/<tensor name>`` for the model, then one
prefix per case. The cases:

``ladder``           one sequence from nothing: the embedding and the output of every decoder layer for every token
                     (the per-layer residual ladder), and the hidden states after the final norm.
``snapshot``         every layer's state after the request's state row: recurrent state and conv tail per DeltaNet
                     layer, keys and values per attention layer.
``steps``            one branch continued from that snapshot, layer by layer: each layer's input, its normed input,
                     its mixer output, its output and its state afterwards. A layer can be tested alone from these.
``deltanet_detail``  the intermediate values inside one DeltaNet mixer of ``steps``.
``attention_detail`` the intermediate values inside one attention mixer of ``steps``.
``request``          rows with parents (a state row, three branches, a branch of a branch; a child is listed before
                     its parent on purpose) and the final hidden states at their readout offsets.
``one_pass``         the state row, one branch and its child as ONE row, read out at the same tokens.
``pointer``          pointer-head questions: logits and probabilities.
``label``            label-row questions: logits and probabilities.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import struct
import sys
from pathlib import Path

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from sepal_runner import refengine  # noqa: E402
from sepal_runner.refengine import attention, deltanet, executor, readout  # noqa: E402
from sepal_runner.refengine.primitives import project, rms_norm, sigmoid, silu  # noqa: E402
from sepal_runner.refengine.snapshot import AttentionState, DeltaNetState  # noqa: E402
from sepal_runner.refengine.weights import EMBEDDING, FINAL_NORM  # noqa: E402

MAGIC = b"SEPALGV1"
PRESET = "tiny"
SEED = 20261009
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "tests" / "golden" / "tiny.sgv"


class Arrays:
    """The named float64 arrays of the file, in the order they were added."""

    def __init__(self) -> None:
        self.items: dict[str, np.ndarray] = {}

    def add(self, name: str, values) -> str:
        if name in self.items:
            raise ValueError(f"array {name} added twice")
        self.items[name] = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
        return name


def write_file(path: Path, header: dict, arrays: Arrays) -> None:
    text = json.dumps(header, indent=1).encode("utf-8")
    with path.open("wb") as out:
        out.write(MAGIC)
        out.write(struct.pack("<I", len(text)))
        out.write(text)
        out.write(struct.pack("<I", len(arrays.items)))
        for name, values in arrays.items.items():
            encoded = name.encode("utf-8")
            out.write(struct.pack("<I", len(encoded)))
            out.write(encoded)
            out.write(struct.pack("<I", values.ndim))
            out.write(struct.pack(f"<{values.ndim}Q", *values.shape))
            out.write(values.tobytes(order="C"))


def add_state(arrays: Arrays, prefix: str, state) -> dict:
    """One layer's state as arrays; returns the header entry naming them."""
    if isinstance(state, DeltaNetState):
        return {"kind": "deltanet",
                "recurrent": arrays.add(f"{prefix}/recurrent", state.recurrent),
                "conv_tail": arrays.add(f"{prefix}/conv_tail", state.conv_tail)}
    assert isinstance(state, AttentionState)
    return {"kind": "attention",
            "keys": arrays.add(f"{prefix}/keys", state.keys),
            "values": arrays.add(f"{prefix}/values", state.values)}


def trace_layers(config, weights, parent, token_ids, start_position):
    """``executor.continue_snapshot`` written out so every layer's input, normed input, mixer output, output and
    state can be kept. Checked below to give exactly what ``continue_snapshot`` gives."""
    positions = start_position + np.arange(len(token_ids), dtype=np.float64)
    x = weights[EMBEDDING][list(token_ids)]
    embedding, steps = x, []
    for i, state in enumerate(parent.layers):
        scope = weights.scope(f"layers.{i}")
        normed = rms_norm(x, scope["input_layernorm.weight"], config.rms_norm_eps)
        if isinstance(state, DeltaNetState):
            mixed, _ = deltanet.deltanet_mixer(normed, scope.scope("linear_attn"), state, config)
        else:
            mixed, _ = attention.attention_mixer(normed, positions, scope.scope("self_attn"), state, config)
        y, new_state = executor.decoder_layer(x, positions, scope, state, config)
        steps.append({"input": x, "normed": normed, "mixer_output": mixed, "output": y, "state": new_state})
        x = y
    reference, _ = executor.continue_snapshot(config, weights, parent, token_ids, start_position)
    if not np.array_equal(x, reference):
        raise AssertionError("the traced layer loop differs from continue_snapshot")
    return embedding, steps, positions


def deltanet_detail(arrays: Arrays, prefix: str, config, weights, x, state) -> dict:
    """The body of ``deltanet.deltanet_mixer`` with every intermediate kept."""
    mixed, z, b, a = project(x, [weights[f"in_proj_{name}.weight"] for name in ("qkv", "z", "b", "a")])
    conv_output, conv_tail = deltanet.causal_conv(mixed, weights["conv1d.weight"][:, 0, :], state.conv_tail)
    q, k, v = deltanet.split_heads(silu(conv_output), config)
    q = deltanet.share_key_heads(deltanet.l2_normalise(q), config)
    k = deltanet.share_key_heads(deltanet.l2_normalise(k), config)
    decay = deltanet.decay_gate(a, weights["A_log"], weights["dt_bias"])
    beta = sigmoid(b)
    reads, recurrent = deltanet.delta_rule_scan(q, k, v, decay, beta, state.recurrent, deltanet.delta_rule_output_scale(config))
    normed = deltanet.gated_rms_norm(reads, z.reshape(reads.shape), weights["norm.weight"], config.rms_norm_eps)
    output = normed.reshape(x.shape[0], config.deltanet_value_dim) @ weights["out_proj.weight"].T
    reference, reference_state = deltanet.deltanet_mixer(x, weights, state, config)
    if not (np.array_equal(output, reference) and np.array_equal(recurrent, reference_state.recurrent)
            and np.array_equal(conv_tail, reference_state.conv_tail)):
        raise AssertionError("the traced DeltaNet mixer differs from deltanet_mixer")
    names = {"mixed": mixed, "z": z, "b": b, "a": a, "conv_output": conv_output, "conv_tail": conv_tail, "q": q, "k": k,
             "v": v, "decay": decay, "beta": beta, "reads": reads, "recurrent": recurrent, "normed": normed, "output": output}
    return {name: arrays.add(f"{prefix}/{name}", values) for name, values in names.items()}


def attention_detail(arrays: Arrays, prefix: str, config, weights, x, positions, state) -> dict:
    """The body of ``attention.attention_mixer`` with every intermediate kept."""
    tokens, eps = x.shape[0], config.rms_norm_eps
    kv_shape = (tokens, config.attention_kv_heads, config.attention_head_dim)
    q_gate, k, v = project(x, [weights[f"{name}_proj.weight"] for name in ("q", "k", "v")])
    q, gate = attention.unpack_query_and_gate(q_gate, config)
    angles = attention.rope_angles(positions, config)
    q = attention.apply_rope(rms_norm(q, weights["q_norm.weight"], eps), angles)
    k = attention.apply_rope(rms_norm(k.reshape(kv_shape), weights["k_norm.weight"], eps), angles)
    keys, values = np.concatenate([state.keys, k]), np.concatenate([state.values, v.reshape(kv_shape)])
    attended = attention.grouped_query_attention(q, keys, values, state.keys.shape[0], attention.attention_scale(config))
    gated = (attended * sigmoid(gate)).reshape(tokens, config.attention_width)
    output = gated @ weights["o_proj.weight"].T
    reference, _ = attention.attention_mixer(x, positions, weights, state, config)
    if not np.array_equal(output, reference):
        raise AssertionError("the traced attention mixer differs from attention_mixer")
    names = {"q_gate": q_gate, "gate": gate, "angles": angles, "q": q, "k": k, "keys": keys, "values": values,
             "attended": attended, "gated": gated, "output": output}
    return {name: arrays.add(f"{prefix}/{name}", values) for name, values in names.items()}


def build(seed: int) -> tuple[dict, Arrays]:
    config = refengine.PRESETS[PRESET]
    weights = refengine.random_weights(config, seed)
    rng = np.random.default_rng(seed + 1)
    arrays = Arrays()
    for name in weights:
        arrays.add(f"weights/{name}", weights[name])

    def tokens(count: int) -> tuple[int, ...]:
        return tuple(int(t) for t in rng.integers(0, config.vocab_size, size=count))

    cases: dict = {}

    # ---- ladder: one sequence from nothing, every layer's output ----
    ladder_tokens = tokens(13)
    embedding, steps, _ = trace_layers(config, weights, refengine.empty_snapshot(config), ladder_tokens, 0)
    cases["ladder"] = {
        "token_ids": list(ladder_tokens), "start_position": 0,
        "embedding": arrays.add("ladder/embedding", embedding),
        "layer_outputs": arrays.add("ladder/layer_outputs", np.stack([step["output"] for step in steps])),
        "final_hidden": arrays.add("ladder/final_hidden",
                                   rms_norm(steps[-1]["output"], weights[FINAL_NORM], config.rms_norm_eps)),
    }

    # ---- the request: a state row, three branches, a branch of a branch ----
    state_tokens, a_tokens, b_tokens, c_tokens, leaf_tokens = tokens(20), tokens(7), tokens(5), tokens(9), tokens(4)
    state = refengine.Row(state_tokens, 0, None, (3, 7, 11, 19))
    branch_a = refengine.Row(a_tokens, 20, state, (2, 6))
    branch_b = refengine.Row(b_tokens, 20, state, (4,))
    branch_c = refengine.Row(c_tokens, 20, state, (0, 8))
    leaf_a = refengine.Row(leaf_tokens, 27, branch_a, (3,))
    named = [("branch_b", branch_b), ("state", state), ("leaf_a", leaf_a), ("branch_a", branch_a), ("branch_c", branch_c)]
    rows = [row for _, row in named]
    results = refengine.run_rows(config, weights, rows)
    index_of = {id(row): i for i, row in enumerate(rows)}
    cases["request"] = {"rows": [
        {"name": name, "token_ids": list(row.token_ids), "start_position": row.start_position,
         "parent": None if row.parent is None else index_of[id(row.parent)],
         "readout_offsets": list(row.readout_offsets),
         "hidden": arrays.add(f"request/{name}/hidden", result.hidden)}
        for (name, row), result in zip(named, results)]}
    result_of = {name: result for (name, _), result in zip(named, results)}

    # ---- the snapshot the state row leaves ----
    snapshot = result_of["state"].snapshot
    cases["snapshot"] = {"row": "state", "n_tokens": snapshot.n_tokens,
                         "layers": [add_state(arrays, f"snapshot/{i}", layer) for i, layer in enumerate(snapshot.layers)]}

    # ---- steps: branch_c continued from that snapshot, layer by layer ----
    _, steps, positions = trace_layers(config, weights, snapshot, c_tokens, 20)
    cases["steps"] = {"token_ids": list(c_tokens), "start_position": 20, "parent": "snapshot", "layers": [
        {"input": arrays.add(f"steps/{i}/input", step["input"]),
         "normed": arrays.add(f"steps/{i}/normed", step["normed"]),
         "mixer_output": arrays.add(f"steps/{i}/mixer_output", step["mixer_output"]),
         "output": arrays.add(f"steps/{i}/output", step["output"]),
         "state": add_state(arrays, f"steps/{i}/state", step["state"])}
        for i, step in enumerate(steps)]}

    kinds = config.layer_kinds
    dn_layer = [i for i, kind in enumerate(kinds) if kind == "deltanet"][1]
    at_layer = [i for i, kind in enumerate(kinds) if kind == "attention"][0]
    cases["deltanet_detail"] = {"layer": dn_layer} | deltanet_detail(
        arrays, "deltanet_detail", config, weights.scope(f"layers.{dn_layer}.linear_attn"),
        steps[dn_layer]["normed"], snapshot.layers[dn_layer])
    cases["attention_detail"] = {"layer": at_layer} | attention_detail(
        arrays, "attention_detail", config, weights.scope(f"layers.{at_layer}.self_attn"),
        steps[at_layer]["normed"], positions, snapshot.layers[at_layer])

    # ---- one pass: state, branch_a and leaf_a as a single row ----
    one_tokens = state_tokens + a_tokens + leaf_tokens
    one_offsets = (3, 7, 11, 19, 22, 26, 30)
    one = refengine.run_rows(config, weights, [refengine.Row(one_tokens, 0, None, one_offsets)])[0]
    cases["one_pass"] = {"token_ids": list(one_tokens), "start_position": 0, "readout_offsets": list(one_offsets),
                         "chained": [["state", 0], ["state", 1], ["state", 2], ["state", 3],
                                     ["branch_a", 0], ["branch_a", 1], ["leaf_a", 0]],
                         "hidden": arrays.add("one_pass/hidden", one.hidden)}

    # ---- readouts ----
    def hidden_at(reference: list) -> np.ndarray:
        name, index = reference
        return result_of[name].hidden[index]

    pointer = []
    option_refs = [["state", 0], ["state", 1], ["state", 2], ["state", 3]]
    for n, (decide_ref, temperature) in enumerate([(["branch_a", 1], 1.0), (["branch_c", 1], 0.5), (["leaf_a", 0], 1.7)]):
        h_decide, h_options = hidden_at(decide_ref), np.stack([hidden_at(ref) for ref in option_refs])
        pointer.append({
            "decide": decide_ref, "options": option_refs, "temperature": temperature,
            "logits": arrays.add(f"pointer/{n}/logits", readout.pointer_logits(config, weights, h_decide, h_options, temperature)),
            "probabilities": arrays.add(f"pointer/{n}/probabilities",
                                        refengine.pointer_probabilities(config, weights, h_decide, h_options, temperature))})
    cases["pointer"] = pointer

    label = []
    for n, (slot_ref, labels, temperature) in enumerate([(["branch_b", 0], [5, 17, 40], 1.0), (["branch_a", 1], [0, 63], 2.0)]):
        h_slot, label_rows = hidden_at(slot_ref), refengine.label_rows(config, weights, labels)
        label.append({
            "slot": slot_ref, "label_token_ids": labels, "temperature": temperature,
            "logits": arrays.add(f"label/{n}/logits", readout.label_row_logits(h_slot, label_rows, temperature)),
            "probabilities": arrays.add(f"label/{n}/probabilities",
                                        refengine.label_row_probabilities(h_slot, label_rows, temperature))})
    cases["label"] = label

    header = {
        "format": "sepal-golden", "version": 1, "generator": "runner/tools/export_golden.py", "oracle": "sepal_runner.refengine",
        "preset": PRESET, "seed": seed, "config": dataclasses.asdict(config),
        "layer_kinds": list(kinds),
        "unverified": [list(item) for item in refengine.UNVERIFIED],
        "weights": list(weights),
        "cases": cases,
    }
    return header, arrays


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    header, arrays = build(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_file(args.output, header, arrays)
    values = sum(a.size for a in arrays.items.values())
    print(f"wrote {args.output} ({args.output.stat().st_size} bytes, {len(arrays.items)} arrays, {values} values)")


if __name__ == "__main__":
    main()
