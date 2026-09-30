"""Tests for storage (SQLite), the stub judge, and the API."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from llmeval.judges.stub import StubJudge  # noqa: E402
from llmeval.rubric import CRITERIA  # noqa: E402
from llmeval.schemas import ContextChunk, EvalSample  # noqa: E402
from llmeval.storage.db import Store  # noqa: E402


def _sample() -> EvalSample:
    return EvalSample(
        sample_id="t1",
        input="What is revenue?",
        output="Revenue was $4.82 billion in fiscal 2024. [c1]",
        contexts=[
            ContextChunk(
                text="Revenue was $4.82 billion in fiscal 2024, up 18% year over year.",
                source="10-K",
                chunk_id="c1",
            )
        ],
        expected="Revenue was $4.82 billion in fiscal 2024.",
    )


def test_stub_judge_deterministic_and_complete():
    j = StubJudge()
    a, b = j.judge(_sample()), j.judge(_sample())
    assert [s.criterion for s in a.scores] == CRITERIA
    # latency_s is nondeterministic; compare everything else
    da, db = a.model_dump(), b.model_dump()
    da.pop("latency_s"), db.pop("latency_s")
    assert da == db
    assert a.by_criterion()["citation_precision"].score == 5


def test_stub_judge_catches_fake_citation():
    s = _sample()
    s.output = "The moon is made of cheese. [nope]"
    out = StubJudge().judge(s)
    assert out.by_criterion()["citation_precision"].score == 1
    assert out.by_criterion()["faithfulness"].score == 1


def test_store_roundtrip(tmp_path):
    store = Store(f"sqlite:///{tmp_path}/t.db")
    store.upsert_samples("calibration", [_sample().model_dump(mode="json")])
    assert len(store.get_samples("calibration")) == 1
    store.add_human_label("t1", "me", {"citation_precision": 5}, "ok")
    labels = store.get_human_labels()
    assert labels[0]["scores"]["citation_precision"] == 5
    store.save_run(
        {
            "run_id": "r1",
            "judge_name": "stub",
            "dataset": "calibration",
            "n_samples": 1,
            "mean_scores": {"a": 4.0},
            "config": {},
        },
        [
            {
                "run_id": "r1",
                "sample_id": "t1",
                "criterion": "a",
                "score": 4,
                "rationale": "x",
                "measured": {},
            }
        ],
    )
    runs = store.list_runs()
    assert runs[0]["run_id"] == "r1"
    assert store.get_run_scores("r1")[0]["score"] == 4


def test_api_health_and_runs(tmp_path, monkeypatch):
    import llmeval.api as api_mod
    import llmeval.config as cfg_mod

    monkeypatch.setattr(cfg_mod.settings, "database_url", f"sqlite:///{tmp_path}/api.db")
    client = TestClient(api_mod.app)
    h = client.get("/health")
    assert h.status_code == 200
    assert h.json()["status"] == "ok"
    assert client.get("/runs").status_code == 200
    r = client.post(
        "/labels",
        json={
            "sample_id": "t1",
            "labeler": "me",
            "scores": {"citation_precision": 5},
        },
    )
    assert r.status_code == 200
