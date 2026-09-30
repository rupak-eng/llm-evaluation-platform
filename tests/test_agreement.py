"""Tests for agreement math: weighted kappa + Spearman on known vectors."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llmeval.agreement import agreement, disagreements, spearman, weighted_kappa  # noqa: E402


def test_kappa_perfect_agreement():
    h = [1, 2, 3, 4, 5, 3, 2, 5]
    assert weighted_kappa(h, h) == 1.0


def test_kappa_known_vector():
    # quadratic-weighted kappa for this 3-class vector; value cross-checked
    # against the independent po/pe computation in test_kappa_independent_po_pe
    h = [1, 1, 1, 2, 2, 2, 3, 3, 3]
    j = [1, 1, 2, 1, 2, 2, 2, 3, 3]
    assert abs(weighted_kappa(h, j, k=3) - 0.7272727) < 1e-6


def test_kappa_independent_po_pe():
    # cross-check weighted_kappa against a direct po/pe computation
    h = [1, 1, 1, 2, 2, 2, 3, 3, 3]
    j = [1, 1, 2, 1, 2, 2, 2, 3, 3]
    n, K = len(h), 3
    m = [[0] * K for _ in range(K)]
    for a, b in zip(h, j, strict=True):
        m[a - 1][b - 1] += 1
    w = [[1 - ((i - jj) / (K - 1)) ** 2 for jj in range(K)] for i in range(K)]
    po = sum(w[i][jj] * m[i][jj] for i in range(K) for jj in range(K)) / n
    row = [sum(r) for r in m]
    col = [sum(m[i][jj] for i in range(K)) for jj in range(K)]
    pe = sum(w[i][jj] * row[i] * col[jj] for i in range(K) for jj in range(K)) / n / n
    assert abs(weighted_kappa(h, j, k=3) - (po - pe) / (1 - pe)) < 1e-9


def test_kappa_chance_level():
    # judge always says 3 regardless of human -> low kappa
    h = [1, 2, 3, 4, 5] * 4
    j = [3] * 20
    assert weighted_kappa(h, j) < 0.1


def test_spearman_perfect():
    rho, p = spearman([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    assert rho == 1.0 and p == 0.0


def test_spearman_inverse():
    rho, _ = spearman([1, 2, 3, 4, 5], [5, 4, 3, 2, 1])
    assert rho == -1.0


def test_spearman_with_ties():
    rho, _ = spearman([1, 1, 2, 3], [1, 1, 2, 3])
    assert abs(rho - 1.0) < 1e-9


def test_spearman_against_scipy():
    from scipy.stats import spearmanr

    x = [3, 1, 4, 1, 5, 9, 2, 6]
    y = [2, 7, 1, 8, 2, 8, 1, 8]
    rho, _ = spearman(x, y)
    assert abs(rho - spearmanr(x, y).statistic) < 1e-9


def test_agreement_shape():
    human = {"a": [1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1]}
    judge = {"a": [1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1]}
    res = agreement(human, judge, ["a", "b"])
    assert len(res) == 3  # 2 criteria + overall
    assert res[0].criterion == "a" and res[0].weighted_kappa == 1.0
    assert res[2].criterion == "overall"


def test_disagreements_ranking():
    human = {"a": [5, 1, 3]}
    judge = {"a": [1, 1, 3]}
    d = disagreements(["s1", "s2", "s3"], human, judge, ["a"], top=2)
    assert d[0]["sample_id"] == "s1" and d[0]["total_gap"] == 4
    assert len(d) == 2
