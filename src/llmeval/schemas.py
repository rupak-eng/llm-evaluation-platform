"""Pydantic schemas for the evaluation platform.

The ecosystem contract (shared with the sibling portfolio projects
``knowledge-graph-rag`` and ``multi-agent-research-assistant``) is the
``EvalSample`` JSON object::

    {
      "input": "user question",
      "output": "system answer, with citations as [chunk_id]",
      "contexts": [{"text": "...", "source": "...", "chunk_id": "c1"}],
      "expected": "reference answer / key claims",
      "metadata": {"system": "kg-rag", "run_id": "..."}
    }
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class ContextChunk(BaseModel):
    text: str
    source: str = ""
    chunk_id: str


class EvalSample(BaseModel):
    """One item to evaluate. Ingested natively by the harness."""

    sample_id: str
    input: str
    output: str
    contexts: list[ContextChunk] = Field(default_factory=list)
    expected: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class CriterionScore(BaseModel):
    criterion: str
    score: int = Field(ge=1, le=5)
    rationale: str
    measured: dict[str, Any] = Field(default_factory=dict)


class JudgeOutput(BaseModel):
    """Structured judge output: per-criterion scores + rationale + citations checked."""

    sample_id: str
    judge_name: str
    scores: list[CriterionScore]
    citations_checked: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    latency_s: float = 0.0

    def by_criterion(self) -> dict[str, CriterionScore]:
        return {s.criterion: s for s in self.scores}


class HumanLabel(BaseModel):
    sample_id: str
    labeler: str
    scores: dict[str, int]
    note: str = ""
    labeled_at: datetime = Field(default_factory=datetime.utcnow)


class AgreementResult(BaseModel):
    criterion: str
    n: int
    weighted_kappa: float
    spearman_rho: float
    spearman_p: float | None = None
    mean_human: float
    mean_judge: float


class BiasReport(BaseModel):
    test: str
    metric: str
    value: float
    detail: dict[str, Any] = Field(default_factory=dict)
    passed: bool | None = None


class RegressionVerdict(BaseModel):
    passed: bool
    deltas: dict[str, float]
    kappa_deltas: dict[str, float]
    threshold_max_drop: float
    threshold_min_kappa: float
    failures: list[str] = Field(default_factory=list)


class EvalRunRecord(BaseModel):
    run_id: str
    judge_name: str
    dataset: str
    n_samples: int
    created_at: datetime = Field(default_factory=datetime.utcnow)
    mean_scores: dict[str, float] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)


JudgeKind = Literal["stub", "openai_compatible", "deepeval"]
