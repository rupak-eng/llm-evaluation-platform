"""FastAPI service: health + eval runs + labels. Thin wrapper over the harness."""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .config import settings
from .runner import run_eval
from .schemas import HumanLabel
from .storage.db import Store

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("llmeval.api")

app = FastAPI(title="llm-evaluation-platform", version="0.1.0")


def _store() -> Store:
    return Store(settings.database_url)


@app.get("/health")
def health() -> dict:
    store = _store()
    n_runs = len(store.list_runs())
    return {
        "status": "ok",
        "database": "postgres" if settings.is_postgres else "sqlite",
        "eval_runs": n_runs,
        "judge_provider": settings.judge_provider,
    }


class RunRequest(BaseModel):
    dataset: str = "calibration"
    judge: str = "stub"


@app.post("/runs")
def create_run(req: RunRequest) -> dict:
    try:
        return run_eval(req.dataset, req.judge)
    except Exception as e:
        log.exception("run failed")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/runs")
def list_runs() -> list[dict]:
    return _store().list_runs()


@app.get("/runs/{run_id}")
def get_run(run_id: str) -> dict:
    store = _store()
    runs = [r for r in store.list_runs() if r["run_id"] == run_id]
    if not runs:
        raise HTTPException(status_code=404, detail="run not found")
    return {
        **runs[0],
        "scores": store.get_run_scores(run_id),
        "bias": store.get_bias_reports(run_id),
    }


@app.post("/labels")
def add_label(label: HumanLabel) -> dict:
    _store().add_human_label(label.sample_id, label.labeler, label.scores, label.note)
    return {"ok": True}


@app.get("/labels")
def list_labels() -> list[dict]:
    return _store().get_human_labels()
