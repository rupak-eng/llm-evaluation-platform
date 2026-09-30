"""Tests for metric computations (pure functions, deterministic)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llmeval import metrics as M  # noqa: E402
from llmeval.rubric import conciseness_score, scale_1_5  # noqa: E402


def test_tokenize_basic():
    assert M.tokenize("Revenue was $4.82 billion!") == ["revenue", "was", "4", "82", "billion"]


def test_sentences_protects_decimals():
    sents = M.sentences("Revenue was $4.82 billion in 2024. It grew 18%.")
    assert len(sents) == 2
    assert "4.82" in sents[0]


def test_find_citations_attaches_to_preceding_sentence():
    out = "Elena Marsh has been CEO since 2019. [c1] She was COO before. [c1]"
    cites = M.find_citations(out)
    assert cites == [("c1", "Elena Marsh has been CEO since 2019."), ("c1", "She was COO before.")]


def test_find_citations_inline():
    out = "Revenue was $4.82 billion [c1] in 2024."
    cites = M.find_citations(out)
    assert len(cites) == 1 and cites[0][0] == "c1"


def test_overlap_fraction():
    assert M.overlap_fraction("the cat sat", "the cat sat on the mat") == 1.0
    assert abs(M.overlap_fraction("the dog ran", "the cat sat") - 1 / 3) < 1e-9
    assert M.overlap_fraction("", "anything") == 0.0


def test_answer_f1_identical():
    assert M.answer_f1("hello world", "hello world") == 1.0


def test_answer_f1_disjoint():
    assert M.answer_f1("aaa bbb", "ccc ddd") == 0.0


def test_scale_1_5_boundaries():
    assert scale_1_5(0.95) == 5
    assert scale_1_5(0.80) == 4
    assert scale_1_5(0.60) == 3
    assert scale_1_5(0.40) == 2
    assert scale_1_5(0.39) == 1
    assert scale_1_5(0.0) == 1


def test_conciseness_exact():
    s, m = conciseness_score(100, 100)
    assert s == 5 and m["ratio"] == 1.0
    s, _ = conciseness_score(600, 100)
    assert s == 1
    s, _ = conciseness_score(250, 100)
    assert s == 3
