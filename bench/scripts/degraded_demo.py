"""Degraded-variant demo: proves the regression gate catches real quality drops.

Builds a degraded copy of the calibration set with all citations stripped
from the outputs, runs the stub judge over it, and feeds the result through
check_regression() against the committed baseline.

Expected outcome: FAIL with a large negative delta on citation_precision /
citation_recall. Exits 2 on FAIL (0 on PASS) so CI demos can assert the
nonzero exit. See demo/degraded-gate.log for a captured run.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from llmeval.agreement import agreement  # noqa: E402
from llmeval.config import settings  # noqa: E402
from llmeval.judges.stub import StubJudge  # noqa: E402
from llmeval.regression import check_regression, load_baseline  # noqa: E402
from llmeval.rubric import CRITERIA  # noqa: E402
from llmeval.schemas import EvalSample  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    samples = []
    with open(ROOT / "adapters" / "fixtures" / "calibration.jsonl") as f:
        for line in f:
            if line.strip():
                samples.append(EvalSample.model_validate(json.loads(line)))
    labels = {}
    with open(ROOT / "human_label" / "seed_labels.jsonl") as f:
        for line in f:
            if line.strip():
                lb = json.loads(line)
                labels[lb["sample_id"]] = lb["scores"]

    # degraded variant: strip every citation marker from every output
    degraded = []
    for s in samples:
        d = s.model_copy(deep=True)
        d.output = re.sub(r"\[[^\]]+\]", "", d.output).strip()
        d.sample_id = s.sample_id + "-degraded"
        degraded.append(d)

    judge = StubJudge()
    outputs = [judge.judge(d) for d in degraded]
    mean_scores = {
        c: round(sum(o.by_criterion()[c].score for o in outputs) / len(outputs), 3)
        for c in CRITERIA
    }
    human = {c: [labels[s.sample_id][c] for s in samples] for c in CRITERIA}
    js = {c: [o.by_criterion()[c].score for o in outputs] for c in CRITERIA}
    kappas = {a.criterion: round(a.weighted_kappa, 3) for a in agreement(human, js, CRITERIA)}
    kappas["overall"] = round(
        sum(a.weighted_kappa for a in agreement(human, js, CRITERIA)) / len(CRITERIA), 3
    )

    baseline = load_baseline(ROOT / "bench" / "results" / "baseline.json")
    verdict = check_regression(
        baseline,
        mean_scores,
        kappas,
        max_drop=settings.regression_max_drop,
        min_kappa=settings.regression_min_kappa,
    )
    print("degraded mean scores:", json.dumps(mean_scores, indent=2))
    print("verdict passed:", verdict.passed)
    for crit, delta in verdict.deltas.items():
        print(f"  {crit:20s} delta={delta:+.1%}")
    for f_ in verdict.failures:
        print("  FAIL:", f_)
    dest = ROOT / "demo" / "degraded-verdict.json"
    dest.write_text(
        json.dumps(
            {
                "mean_scores": mean_scores,
                "kappas": kappas,
                "verdict": verdict.model_dump(),
            },
            indent=2,
        )
    )
    print(f"verdict -> {dest}")
    if verdict.passed:
        print("UNEXPECTED: degraded variant passed the gate")
        return 0
    print("GATE CORRECTLY REJECTED the degraded variant")
    return 2


if __name__ == "__main__":
    sys.exit(main())
