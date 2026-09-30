"""Agreement measurement: weighted Cohen's kappa + Spearman correlation.

Compares human labels against judge scores on the same 1-5 ordinal scale,
per criterion and overall (mean across criteria). Uses quadratic weights for
kappa (a 1-vs-5 disagreement is worse than 4-vs-5), which is the standard
choice for ordinal scales.

Implemented by hand (no sklearn dependency) and unit-tested against known
vectors, including the textbook kappa example.
"""

from __future__ import annotations

import math

from .schemas import AgreementResult


def confusion(human: list[int], judge: list[int], k: int = 5) -> list[list[int]]:
    m = [[0] * k for _ in range(k)]
    for h, j in zip(human, judge, strict=True):
        m[h - 1][j - 1] += 1
    return m


def weighted_kappa(human: list[int], judge: list[int], k: int = 5) -> float:
    """Quadratic-weighted Cohen's kappa for ordinal 1..k ratings."""
    if len(human) != len(judge) or not human:
        raise ValueError("need non-empty paired ratings")
    m = confusion(human, judge, k)
    n = len(human)
    # quadratic weights: w[i][j] = 1 - ((i-j)/(k-1))^2
    w = [[1 - ((i - j) / (k - 1)) ** 2 for j in range(k)] for i in range(k)]
    row = [sum(r) for r in m]
    col = [sum(m[i][j] for i in range(k)) for j in range(k)]
    po = sum(w[i][j] * m[i][j] for i in range(k) for j in range(k)) / n
    pe = sum(w[i][j] * row[i] * col[j] for i in range(k) for j in range(k)) / (n * n)
    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)


def _ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for t in range(i, j + 1):
            ranks[order[t]] = avg
        i = j + 1
    return ranks


def spearman(x: list[float], y: list[float]) -> tuple[float, float | None]:
    """Spearman rho with tie correction; returns (rho, two-sided p-value)."""
    if len(x) != len(y) or len(x) < 3:
        raise ValueError("need >=3 paired values")
    rx, ry = _ranks([float(v) for v in x]), _ranks([float(v) for v in y])
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if dx == 0 or dy == 0:
        return 0.0, None
    rho = num / (dx * dy)
    rho = max(-1.0, min(1.0, rho))
    if abs(abs(rho) - 1.0) < 1e-12:  # snap float dust to exact +-1
        rho = math.copysign(1.0, rho)
    if abs(rho) >= 1.0:
        return rho, 0.0
    t = rho * math.sqrt((n - 2) / max(1e-12, 1 - rho * rho))
    p = 2 * (1 - _t_cdf(abs(t), n - 2))
    return rho, p


def _t_cdf(t: float, df: int) -> float:
    # regularized incomplete beta via math.betainc is unavailable; use normal
    # approximation for df>=30, else Simpson integration of the t density.
    if df >= 30:
        return 0.5 * (1 + math.erf(t / math.sqrt(2)))

    def f(x: float) -> float:
        return (
            math.gamma((df + 1) / 2)
            / (math.sqrt(df * math.pi) * math.gamma(df / 2))
            * (1 + x * x / df) ** (-(df + 1) / 2)
        )

    n = 2000
    a, b = 0.0, t
    h = (b - a) / n
    s = f(a) + f(b) + 4 * sum(f(a + (2 * i - 1) * h) for i in range(1, n // 2 + 1))
    s += 2 * sum(f(a + 2 * i * h) for i in range(1, n // 2))
    return 0.5 + s * h / 3


def agreement(
    human: dict[str, list[int]], judge: dict[str, list[int]], criteria: list[str]
) -> list[AgreementResult]:
    """Per-criterion + overall agreement. Dicts map criterion -> aligned score lists."""
    results: list[AgreementResult] = []
    for c in criteria:
        h, j = human[c], judge[c]
        rho, p = spearman([float(v) for v in h], [float(v) for v in j])
        results.append(
            AgreementResult(
                criterion=c,
                n=len(h),
                weighted_kappa=weighted_kappa(h, j),
                spearman_rho=rho,
                spearman_p=p,
                mean_human=sum(h) / len(h),
                mean_judge=sum(j) / len(j),
            )
        )
    # overall: mean score across criteria per item
    ho = [
        sum(human[c][i] for c in criteria) / len(criteria) for i in range(len(human[criteria[0]]))
    ]
    jo = [
        sum(judge[c][i] for c in criteria) / len(criteria) for i in range(len(judge[criteria[0]]))
    ]
    rho, p = spearman(ho, jo)
    # kappa needs integers: round the per-item means to the 1-5 scale
    hk = [max(1, min(5, round(v))) for v in ho]
    jk = [max(1, min(5, round(v))) for v in jo]
    results.append(
        AgreementResult(
            criterion="overall",
            n=len(ho),
            weighted_kappa=weighted_kappa(hk, jk),
            spearman_rho=rho,
            spearman_p=p,
            mean_human=sum(ho) / len(ho),
            mean_judge=sum(jo) / len(jo),
        )
    )
    return results


def disagreements(
    sample_ids: list[str],
    human: dict[str, list[int]],
    judge: dict[str, list[int]],
    criteria: list[str],
    top: int = 5,
) -> list[dict]:
    """Worst human-vs-judge disagreements, ranked by total absolute gap."""
    rows = []
    for i, sid in enumerate(sample_ids):
        gap = sum(abs(human[c][i] - judge[c][i]) for c in criteria)
        rows.append(
            {
                "sample_id": sid,
                "total_gap": gap,
                "human": {c: human[c][i] for c in criteria},
                "judge": {c: judge[c][i] for c in criteria},
            }
        )
    rows.sort(key=lambda r: -r["total_gap"])
    return rows[:top]
