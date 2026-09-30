"""CI regression gate (also run by GitHub Actions).

Steps:
  1. Run the eval harness (stub judge, calibration set) -> current run.
  2. Compute agreement (weighted kappa + Spearman) vs committed human labels.
  3. Compare against bench/results/baseline.json via check_regression().
  4. Exit 1 (fail the build) if any criterion drops beyond REGRESSION_MAX_DROP
     or overall kappa falls below REGRESSION_MIN_KAPPA.

Usage: make ci-local   (or: python bench/scripts/ci_gate.py)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from llmeval.config import settings  # noqa: E402
from llmeval.regression import check_regression, load_baseline  # noqa: E402
from llmeval.runner import run_eval  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    baseline = load_baseline(ROOT / "bench" / "results" / "baseline.json")
    result = run_eval("calibration", "stub", run_id="ci-gate")
    verdict = check_regression(
        baseline,
        result["mean_scores"],
        result["kappas"],
        max_drop=settings.regression_max_drop,
        min_kappa=settings.regression_min_kappa,
    )
    report = {
        "run_id": result["run_id"],
        "mean_scores": result["mean_scores"],
        "kappas": result["kappas"],
        "verdict": verdict.model_dump(),
    }
    dest = ROOT / "bench" / "results" / "ci-gate.json"
    dest.write_text(json.dumps(report, indent=2))
    print(f"report -> {dest}")
    for crit, delta in verdict.deltas.items():
        print(f"  {crit:20s} delta={delta:+.1%}")
    print(f"  overall kappa: {result['kappas'].get('overall')}")
    if verdict.passed:
        print("CI GATE: PASS")
        return 0
    print("CI GATE: FAIL")
    for f_ in verdict.failures:
        print("  -", f_)
    return 1


if __name__ == "__main__":
    sys.exit(main())
