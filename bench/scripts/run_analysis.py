"""Full analysis: disagreements, self-preference, regression-gate demo.

Writes bench/results/analysis.json. All numbers are measured, not invented.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from llmeval.agreement import disagreements  # noqa: E402
from llmeval.bias import StyledStubJudge, self_preference  # noqa: E402
from llmeval.config import settings  # noqa: E402
from llmeval.judges.stub import StubJudge  # noqa: E402
from llmeval.regression import check_regression  # noqa: E402
from llmeval.rubric import CRITERIA  # noqa: E402
from llmeval.schemas import EvalSample  # noqa: E402
from llmeval.storage.db import Store  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


class CitationStrippingJudge(StubJudge):
    """Degraded variant: identical to stub but citations are stripped before
    judging — simulates a system change that drops citation rendering."""

    name = "stub-no-citations"

    def judge(self, sample: EvalSample) -> "JudgeOutput":  # type: ignore[override]
        from llmeval import metrics as M

        degraded = sample.model_copy(deep=True)
        degraded.output = M.strip_citations(degraded.output)
        out = super().judge(degraded)
        out.judge_name = self.name
        return out


def main() -> None:
    store = Store(settings.database_url)
    labels = {lb["sample_id"]: lb["scores"] for lb in store.get_human_labels()}
    samples = [EvalSample.model_validate(p) for p in store.get_samples("calibration")]
    judge = StubJudge()
    outputs = {o.sample_id: o for o in (judge.judge(s) for s in samples)}

    common = [s.sample_id for s in samples if s.sample_id in labels]
    human = {c: [labels[s][c] for s in common] for c in CRITERIA}
    js = {c: [outputs[s].by_criterion()[c].score for s in common] for c in CRITERIA}
    dis = disagreements(common, human, js, CRITERIA, top=5)

    # self-preference: identical outputs, style tag differs only
    base = next(s for s in samples if s.sample_id == "rev-good")
    set_a, set_b = [], []
    for i, s in enumerate(samples[:12]):
        a = s.model_copy(deep=True)
        b = s.model_copy(deep=True)
        a.metadata["style"] = "verbose"
        b.metadata["style"] = "terse"
        a.sample_id, b.sample_id = f"{s.sample_id}-v", f"{s.sample_id}-t"
        set_a.append(a)
        set_b.append(b)
    ja, jb = StyledStubJudge("verbose"), StyledStubJudge("terse")
    self_pref = self_preference(ja.judge, jb.judge, set_a, set_b)
    # control: plain stub judges show zero preference on the same sets
    plain = StubJudge()
    control = self_preference(plain.judge, plain.judge, set_a, set_b)

    # regression gate demo: baseline run vs degraded variant
    baseline = json.load(open(ROOT / "bench" / "results" / "baseline-stub-001.json"))
    degraded_out = [CitationStrippingJudge().judge(s) for s in samples]
    cur_means = {
        c: round(sum(o.by_criterion()[c].score for o in degraded_out) / len(degraded_out), 3)
        for c in CRITERIA
    }
    verdict = check_regression(
        {"mean_scores": baseline["mean_scores"], "kappas": baseline["kappas"]},
        cur_means,
        baseline["kappas"],
        max_drop=settings.regression_max_drop,
        min_kappa=settings.regression_min_kappa,
    )

    analysis = {
        "disagreements_top5": dis,
        "self_preference": self_pref,
        "self_preference_control_plain_stub": control,
        "regression_demo": {
            "baseline_means": baseline["mean_scores"],
            "degraded_means": cur_means,
            "verdict": verdict.model_dump(),
        },
    }
    dest = ROOT / "bench" / "results" / "analysis.json"
    dest.write_text(json.dumps(analysis, indent=2))
    print(f"wrote {dest}")
    print("self_preference (styled):", round(self_pref["value"], 3), self_pref["detail"])
    print("self_preference (control):", round(control["value"], 3))
    print("regression demo passed:", verdict.passed)
    for f_ in verdict.failures:
        print("  FAIL:", f_)


if __name__ == "__main__":
    main()
