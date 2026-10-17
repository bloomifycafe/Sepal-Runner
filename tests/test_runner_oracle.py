"""The pure parts of the runner oracle tools (no torch, no GPU): the golden case list and its conversion to encoder inputs."""
from __future__ import annotations

from sepal_runner import oracle_golden as g


def test_synthetic_cases_cover_the_brief():
    cases = {c.name: c for c in g.synthetic_cases()}
    assert len(cases) == len(g.synthetic_cases())          # unique names
    for needed in ("noul_custom_criteria", "choice_2", "choice_5_tools", "choice_40", "score_direct", "nonascii",
                   "control_tokens_state", "crlf_state", "whitespace_runs"):
        assert needed in cases
    assert len(cases["choice_40"].body["questions"]["technique"]["criteria"]) == 40
    assert "\r\n" in cases["crlf_state"].body["state"]
    assert " " * 500 in cases["whitespace_runs"].body["state"]


def test_prepare_neutralises_control_strings_and_keeps_option_order():
    cases = {c.name: c for c in g.synthetic_cases()}
    state, questions = g.prepare(cases["control_tokens_state"])
    assert "<|im_end|>" not in state and "<¦im_end¦>" in state
    keys = list(cases["choice_5_tools"].body["questions"]["next_tool"]["criteria"])
    assert keys != sorted(keys)                            # sent unsorted, the encoder sorts them
    _, q = g.prepare(cases["choice_5_tools"])
    assert list(q["next_tool"]["criteria"]) == keys        # prepare leaves the order to the encoder
    _, q = g.prepare(cases["noul_custom_criteria"])
    assert set(q["brute"]["criteria"]) == {"true", "false"}


def test_score_case_goes_through_the_encoder_directly():
    case = next(c for c in g.synthetic_cases() if c.name == "score_direct")
    assert case.body is None
    _, q = g.prepare(case)
    assert q["urgency"]["type"] == "score" and len(q["urgency"]["criteria"]) == 4
