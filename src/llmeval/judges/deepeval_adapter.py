"""DeepEval adapter: the production-judge path. Lazy import — deepeval is NOT
a hard dependency of the core pipeline.

Runs one DeepEval ``GEval`` per rubric criterion (each with its own
evaluation_steps so the model reasons criterion-by-criterion), then maps the
0-10 GEval score to the anchored 1-5 rubric scale and returns a
``JudgeOutput`` that feeds the same agreement / bias / regression analysis.

Install:  pip install -r requirements-optional.txt
Use:      JUDGE_PROVIDER=deepeval JUDGE_MODEL=gpt-4o-mini make eval

NOTE: exercised in CI only when the optional deps are installed and
JUDGE_API_KEY is present; the numbers reported in this repo's README come
from the deterministic stub judge unless stated otherwise.
"""

from __future__ import annotations

import time

from ..schemas import CriterionScore, EvalSample, JudgeOutput

CRITERION_GUIDANCE = {
    "citation_precision": (
        "Check every [chunk_id] citation in the actual output. "
        "A citation is valid only if the chunk_id exists in the retrieval context "
        "AND the cited claim is supported by that chunk."
    ),
    "citation_recall": (
        "List the key claims in the expected output. Count how many appear in the actual output "
        "with a valid citation."
    ),
    "faithfulness": (
        "Check every factual sentence of the actual output against the retrieval context. "
        "Flag any claim not supported by the context."
    ),
    "answer_f1": (
        "Compare the actual output to the expected output for factual coverage. "
        "Penalize missing key facts and added unsupported facts."
    ),
    "conciseness": (
        "Judge whether the actual output answers without padding, repetition, or irrelevant detail."
    ),
}


def _to_1_5(score_0_10: float) -> int:
    # exact linear map of DeepEval's 0-10 to the anchored 1-5 scale
    return max(1, min(5, int(round(1 + (score_0_10 / 10) * 4))))


def judge_with_deepeval(sample: EvalSample, model: str = "gpt-4o-mini") -> JudgeOutput:
    from deepeval import assert_test  # noqa: F401  (validates import early)
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCase

    t0 = time.perf_counter()
    test_case = LLMTestCase(
        input=sample.input,
        actual_output=sample.output,
        expected_output=sample.expected,
        retrieval_context=[f"[{c.chunk_id}] {c.text}" for c in sample.contexts],
    )
    scores: list[CriterionScore] = []
    errors: list[str] = []
    for criterion, guidance in CRITERION_GUIDANCE.items():
        metric = GEval(
            name=criterion,
            criteria=guidance,
            evaluation_steps=[
                "Read the actual output and the retrieval context.",
                guidance,
                "Give a score from 0 (total failure) to 10 (perfect) with reasons.",
            ],
            model=model,
        )
        try:
            metric.measure(test_case)
            raw = float(metric.score or 0.0)
            scores.append(
                CriterionScore(
                    criterion=criterion,
                    score=_to_1_5(raw),
                    rationale=str(metric.reason or "")[:500],
                    measured={"deepeval_score_0_10": raw},
                )
            )
        except Exception as e:  # keep the run alive; record the failure
            errors.append(f"{criterion}: {type(e).__name__}: {e}")
    return JudgeOutput(
        sample_id=sample.sample_id,
        judge_name=f"deepeval:{model}",
        scores=scores,
        errors=errors,
        latency_s=time.perf_counter() - t0,
    )
