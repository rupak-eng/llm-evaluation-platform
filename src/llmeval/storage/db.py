"""Eval history storage: Postgres in production, SQLite locally/for tests.

Schema (SQLAlchemy Core + ORM):
  samples       — ingested EvalSample JSON
  human_labels  — human scores per sample (labeler, timestamp)
  eval_runs     — one row per judge run over a dataset
  judge_scores  — per (run, sample, criterion) judge score
  bias_reports  — bias test results per run
Tables are created with checkfirst=True (safe on fresh DBs).
"""
from __future__ import annotations

import json
from datetime import datetime

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    select,
)
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


class Sample(Base):
    __tablename__ = "samples"
    sample_id = Column(String, primary_key=True)
    dataset = Column(String, index=True, default="calibration")
    payload = Column(JSON, nullable=False)


class HumanLabelRow(Base):
    __tablename__ = "human_labels"
    id = Column(Integer, primary_key=True, autoincrement=True)
    sample_id = Column(String, index=True, nullable=False)
    labeler = Column(String, nullable=False)
    scores = Column(JSON, nullable=False)
    note = Column(Text, default="")
    labeled_at = Column(DateTime, default=datetime.utcnow)


class EvalRun(Base):
    __tablename__ = "eval_runs"
    run_id = Column(String, primary_key=True)
    judge_name = Column(String, nullable=False)
    dataset = Column(String, nullable=False)
    n_samples = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    mean_scores = Column(JSON, default=dict)
    config = Column(JSON, default=dict)


class JudgeScore(Base):
    __tablename__ = "judge_scores"
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String, ForeignKey("eval_runs.run_id"), index=True, nullable=False)
    sample_id = Column(String, index=True, nullable=False)
    criterion = Column(String, nullable=False)
    score = Column(Integer, nullable=False)
    rationale = Column(Text, default="")
    measured = Column(JSON, default=dict)


class BiasReportRow(Base):
    __tablename__ = "bias_reports"
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String, ForeignKey("eval_runs.run_id"), index=True, nullable=False)
    test = Column(String, nullable=False)
    metric = Column(String, nullable=False)
    value = Column(Float, nullable=False)
    detail = Column(JSON, default=dict)


class Store:
    def __init__(self, database_url: str):
        connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
        self.engine = create_engine(database_url, connect_args=connect_args)
        Base.metadata.create_all(self.engine, checkfirst=True)
        self.Session = sessionmaker(bind=self.engine)

    # -- samples --
    def upsert_samples(self, dataset: str, payloads: list[dict]) -> int:
        with self.Session() as s:
            n = 0
            for p in payloads:
                row = s.get(Sample, p["sample_id"])
                if row:
                    row.payload = p
                    row.dataset = dataset
                else:
                    s.add(Sample(sample_id=p["sample_id"], dataset=dataset, payload=p))
                n += 1
            s.commit()
            return n

    def get_samples(self, dataset: str) -> list[dict]:
        with self.Session() as s:
            rows = s.execute(select(Sample).where(Sample.dataset == dataset)).scalars().all()
            return [r.payload for r in rows]

    # -- human labels --
    def add_human_label(self, sample_id: str, labeler: str, scores: dict, note: str = "") -> None:
        with self.Session() as s:
            s.add(HumanLabelRow(sample_id=sample_id, labeler=labeler, scores=scores, note=note))
            s.commit()

    def get_human_labels(self) -> list[dict]:
        with self.Session() as s:
            rows = s.execute(select(HumanLabelRow)).scalars().all()
            return [
                {
                    "sample_id": r.sample_id,
                    "labeler": r.labeler,
                    "scores": r.scores,
                    "note": r.note,
                    "labeled_at": (r.labeled_at or datetime.utcnow()).isoformat(),
                }
                for r in rows
            ]

    # -- runs --
    def save_run(self, run: dict, scores: list[dict], bias: list[dict] | None = None) -> None:
        with self.Session() as s:
            s.add(EvalRun(**run))
            for sc in scores:
                s.add(JudgeScore(**sc))
            for b in bias or []:
                s.add(BiasReportRow(**b))
            s.commit()

    def list_runs(self) -> list[dict]:
        with self.Session() as s:
            rows = s.execute(select(EvalRun).order_by(EvalRun.created_at.desc())).scalars().all()
            return [
                {
                    "run_id": r.run_id,
                    "judge_name": r.judge_name,
                    "dataset": r.dataset,
                    "n_samples": r.n_samples,
                    "created_at": (r.created_at or datetime.utcnow()).isoformat(),
                    "mean_scores": r.mean_scores or {},
                    "config": r.config or {},
                }
                for r in rows
            ]

    def get_run_scores(self, run_id: str) -> list[dict]:
        with self.Session() as s:
            rows = (
                s.execute(select(JudgeScore).where(JudgeScore.run_id == run_id)).scalars().all()
            )
            return [
                {
                    "sample_id": r.sample_id,
                    "criterion": r.criterion,
                    "score": r.score,
                    "rationale": r.rationale,
                    "measured": r.measured or {},
                }
                for r in rows
            ]

    def get_bias_reports(self, run_id: str) -> list[dict]:
        with self.Session() as s:
            rows = (
                s.execute(select(BiasReportRow).where(BiasReportRow.run_id == run_id))
                .scalars()
                .all()
            )
            return [
                {"test": r.test, "metric": r.metric, "value": r.value, "detail": r.detail or {}}
                for r in rows
            ]
