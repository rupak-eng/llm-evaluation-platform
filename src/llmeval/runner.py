"""Eval runner: judges a dataset, stores the run, writes JSON artifacts.

Usage:
    python -m llmeval.runner --dataset calibration --judge stub
    python -m llmeval.runner --dataset calibration --judge stub --run-id ci-123

Produces bench/results/<run_id>.json and persists the run to the DB.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from pathlib import Path

from .agreement import agreement
from .bias import length_bias, make_pairs, position_bias
from .config import settings
from .judges.stub import StubJudge
from .rubric import CRITERIA
from .schemas import EvalSample
from .storage.db import Store

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("llmeval.runner")

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_dataset(name: str) -> list[EvalSample]:
    path = REPO_ROOT / "adapters" / "fixtures" / f"{name}.jsonl"
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(EvalSample.model_validate(json.loads(line)))
    return samples


def make_judge(kind: str, model: str | None = None):
    if kind == "stub":
        return StubJudge()
    if kind == "groq":
        from .judges.groq import GroqJudge

        return GroqJudge(model=model or "openai/gpt-oss-120b")
    if kind == "openai_compat":
        from .judges.openai_compat import OpenAICompatJudge

        return OpenAICompatJudge()
    if kind == "deepeval":
        from .judges import deepeval_adapter

        class _D:
            name = f"deepeval:{settings.judge_model}"

            def judge(self, sample: EvalSample):
                return deepeval_adapter.judge_with_deepeval(sample, settings.judge_model)

        return _D()
    raise ValueError(f"unknown judge: {kind}")


def run_eval(dataset: str, judge_kind: str, run_id: str | None = None) -> dict:
    judge = make_judge(judge_kind)
    samples = load_dataset(dataset)
    if not samples:
        raise RuntimeError(f"dataset '{dataset}' is empty")
    run_id = run_id or f"{judge.name}-{uuid.uuid4().hex[:8]}"
    log.info("judging %d samples with %s (run %s)", len(samples), judge.name, run_id)

    outputs = [judge.judge(s) for s in samples]

    # mean scores per criterion
    mean_scores = {
        c: round(sum(o.by_criterion()[c].score for o in outputs) / len(outputs), 3)
        for c in CRITERIA
    }

    # agreement vs human labels (if present in DB)
    store = Store(settings.database_url)
    labels = store.get_human_labels()
    label_map = {lb["sample_id"]: lb["scores"] for lb in labels}
    common = [s.sample_id for s in samples if s.sample_id in label_map]
    agreements = []
    if common:
        human = {c: [label_map[sid][c] for sid in common] for c in CRITERIA}
        js = {c: [] for c in CRITERIA}
        id_by_sid = {o.sample_id: o for o in outputs}
        for sid in common:
            for c in CRITERIA:
                js[c].append(id_by_sid[sid].by_criterion()[c].score)
        agreements = [a.model_dump() for a in agreement(human, js, CRITERIA)]

    # bias tests
    biases = [
        position_bias(judge.judge, make_pairs(samples, seed=settings.eval_seed)),
        length_bias(judge.judge, samples),
    ]

    # persist
    score_rows = [
        {
            "run_id": run_id,
            "sample_id": o.sample_id,
            "criterion": s.criterion,
            "score": s.score,
            "rationale": s.rationale,
            "measured": s.measured,
        }
        for o in outputs
        for s in o.scores
    ]
    bias_rows = [
        {
            "run_id": run_id,
            "test": b["test"],
            "metric": b["metric"],
            "value": b["value"],
            "detail": b["detail"],
        }
        for b in biases
    ]
    store.save_run(
        {
            "run_id": run_id,
            "judge_name": judge.name,
            "dataset": dataset,
            "n_samples": len(samples),
            "mean_scores": mean_scores,
            "config": {"judge_kind": judge_kind, "seed": settings.eval_seed},
        },
        score_rows,
        bias_rows,
    )

    result = {
        "run_id": run_id,
        "judge": judge.name,
        "dataset": dataset,
        "n_samples": len(samples),
        "n_human_labeled": len(common),
        "mean_scores": mean_scores,
        "kappas": {a["criterion"]: round(a["weighted_kappa"], 3) for a in agreements},
        "agreements": agreements,
        "bias": biases,
        "errors": [e for o in outputs for e in o.errors],
    }
    out_path = REPO_ROOT / "bench" / "results" / f"{run_id}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    log.info("wrote %s", out_path)
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the eval harness")
    ap.add_argument("--dataset", default="calibration")
    ap.add_argument("--judge", default=settings.judge_provider)
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args(argv)
    try:
        result = run_eval(args.dataset, args.judge, args.run_id)
    except Exception:
        log.exception("eval run failed")
        return 1
    print(
        json.dumps({k: result[k] for k in ("run_id", "judge", "mean_scores", "kappas")}, indent=2)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
