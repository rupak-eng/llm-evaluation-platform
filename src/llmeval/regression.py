"""Regression detection: compare a new eval run against a committed baseline.

The baseline is a JSON file (bench/results/baseline.json) produced by a real
run. The gate fails when:
  - any criterion mean drops by more than `max_drop` (relative), or
  - overall weighted kappa drops below `min_kappa`.
Used by the CI workflow and the local `make ci-local` gate.
"""

from __future__ import annotations

import json
from pathlib import Path

from .schemas import RegressionVerdict


def load_baseline(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def check_regression(
    baseline: dict,
    current_mean_scores: dict[str, float],
    current_kappas: dict[str, float],
    max_drop: float = 0.15,
    min_kappa: float = 0.40,
) -> RegressionVerdict:
    base_means: dict[str, float] = baseline.get("mean_scores", {})
    base_kappas: dict[str, float] = baseline.get("kappas", {})
    deltas: dict[str, float] = {}
    kappa_deltas: dict[str, float] = {}
    failures: list[str] = []

    for crit, base in base_means.items():
        cur = current_mean_scores.get(crit)
        if cur is None:
            failures.append(f"missing criterion in current run: {crit}")
            continue
        rel = (cur - base) / base if base else (0.0 if cur == 0 else float("-inf"))
        deltas[crit] = round(rel, 4)
        if rel < -max_drop:
            failures.append(
                f"{crit}: dropped {rel:.1%} (baseline {base:.3f} -> current {cur:.3f}), "
                f"threshold -{max_drop:.0%}"
            )

    for crit, base in base_kappas.items():
        cur = current_kappas.get(crit)
        if cur is None:
            continue
        kappa_deltas[crit] = round(cur - base, 4)

    overall_kappa = current_kappas.get("overall")
    if overall_kappa is not None and overall_kappa < min_kappa:
        failures.append(f"overall weighted kappa {overall_kappa:.3f} below minimum {min_kappa:.2f}")

    return RegressionVerdict(
        passed=not failures,
        deltas=deltas,
        kappa_deltas=kappa_deltas,
        threshold_max_drop=max_drop,
        threshold_min_kappa=min_kappa,
        failures=failures,
    )
