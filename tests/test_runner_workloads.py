"""The question side of the runner benchmark workloads (pure Python; the file generation needs the Unsloth environment and runs/data)."""
from __future__ import annotations

import random

from sepal.runner import workloads as w


def _wazuh_rows(n=40):
    return [{"line": f"2022-01-15T02:32:{i % 60:02d}Z host{i % 3} sshd[rule {5700 + i % 5} L{3 + i % 6}] Failed password :: from 10.0.0.{i % 9}",
             "rule": str(5700 + i % 5), "level": 3 + i % 6, "desc": "x", "host": f"host{i % 3}", "prog": "sshd", "src": "t"} for i in range(n)]


def test_questions_are_distinct_deterministic_and_have_custom_criteria():
    rows = _wazuh_rows()
    ents = w.entities("wazuh", rows, rows, random.Random(1))
    a = w.build_question_texts("wazuh", rows, ents, random.Random(2), 31, terse=True)
    b = w.build_question_texts("wazuh", rows, ents, random.Random(2), 31, terse=True)
    assert a == b and len({t[0] for t in a}) == 31
    body = w.questions_body(a, random.Random(3), with_tool=True)
    assert len(body) == 32 and body["tool"]["type"] == "choice"
    assert list(body["tool"]["criteria"]) != sorted(body["tool"]["criteria"])          # sent unsorted
    assert all(q["criteria"]["true"] and q["criteria"]["false"] for k, q in body.items() if k != "tool")


def test_shape_list_and_counts():
    names = w.shape_names()
    assert "t16k_q32" in names and "shared_t4k_q8" in names and "t32k_q8" in names and len(names) == 11
    assert w.COUNTS[("1k", 1)] == 64 and w.COUNTS[("4k", 8)] == 32 and w.COUNTS[("16k", 32)] == 12
    assert w.TARGETS["16k"] < w.ORACLE_WINDOW
