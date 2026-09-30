"""Tests for bias tests and the regression gate."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llmeval.bias import (  # noqa: E402
    StyledStubJudge,
    length_bias,
    make_pairs,
    position_bias,
    self_preference,
)
from llmeval.judges.stub import StubJudge  # noqa: E402
from llmeval.regression import check_regression  # noqa: E402
from llmeval.schemas import ContextChunk, EvalSample  # noqa: E402


def _sample(sid: str, output: str, style: str = "") -> EvalSample:
    return EvalSample(
        sample_id=sid,
        input="q",
        output=output,
        contexts=[ContextChunk(text="the quick brown fox jumps", source="s", chunk_id="c1")],
        expected="the quick brown fox",
        metadata={"style": style} if style else {},
    )


def test_position_bias_deterministic_judge_is_zero():
    j = StubJudge()
    samples = [_sample(f"s{i}", f"the quick brown fox jumps [c{i % 2}]") for i in range(6)]
    # note: c0/c1 mostly unknown chunk_ids; still deterministic -> flip rate 0
    pairs = make_pairs(samples, n_pairs=3)
    rep = position_bias(j.judge, pairs)
    assert rep["value"] == 0.0
    assert rep["detail"]["n_pairs"] == 3


def test_length_bias_returns_spearman():
    j = StubJudge()
    samples = [
        _sample("a", "the quick brown fox [c1]"),
        _sample("b", "the quick brown fox jumps over the lazy dog near the river bank [c1]"),
        _sample("c", "the quick brown fox jumps [c1] over and over and over again and again"),
        _sample("d", "fox [c1]"),
    ]
    rep = length_bias(j.judge, samples)
    assert rep["metric"] == "spearman_rho_length_vs_score"
    assert -1.0 <= rep["value"] <= 1.0


def test_self_preference_detects_styled_judges():
    # identical outputs, differing ONLY in the style tag: any nonzero delta is
    # purely the style penalty, so the harness provably detects self-preference
    ja = StyledStubJudge("verbose")
    jb = StyledStubJudge("terse")
    base = "the quick brown fox jumps over the fence today [c1]"
    set_a = [_sample(f"a{i}", base, style="verbose") for i in range(3)]
    set_b = [_sample(f"b{i}", base, style="terse") for i in range(3)]
    rep = self_preference(ja.judge, jb.judge, set_a, set_b)
    assert rep["value"] > 0, rep  # harness detects the injected preference
    assert rep["detail"]["pref_a"] > 0 and rep["detail"]["pref_b"] > 0


def test_self_preference_plain_stub_is_zero():
    j = StubJudge()
    set_a = [_sample(f"a{i}", "the quick brown fox [c1]", style="verbose") for i in range(3)]
    set_b = [_sample(f"b{i}", "the quick brown fox [c1]", style="terse") for i in range(3)]
    rep = self_preference(j.judge, j.judge, set_a, set_b)
    assert rep["value"] == 0.0


def test_regression_gate_passes_on_identical():
    base = {"mean_scores": {"a": 4.0, "b": 3.0}, "kappas": {"overall": 0.7}}
    v = check_regression(base, {"a": 4.0, "b": 3.0}, {"overall": 0.7})
    assert v.passed and not v.failures


def test_regression_gate_fails_on_drop():
    base = {"mean_scores": {"a": 4.0}, "kappas": {"overall": 0.7}}
    v = check_regression(base, {"a": 3.0}, {"overall": 0.7}, max_drop=0.15)
    assert not v.passed
    assert any("a" in f for f in v.failures)


def test_regression_gate_fails_on_low_kappa():
    base = {"mean_scores": {"a": 4.0}, "kappas": {"overall": 0.7}}
    v = check_regression(base, {"a": 4.0}, {"overall": 0.2}, min_kappa=0.4)
    assert not v.passed
