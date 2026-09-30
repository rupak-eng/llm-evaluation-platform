"""Streamlit dashboard: scores, agreement, bias, regression, runs, labeling.

All pages render REAL data: eval runs + human labels from the history store
(Postgres in production, SQLite locally) and measured artifacts under
bench/results/. Run with: make dashboard
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llmeval.config import settings  # noqa: E402
from llmeval.rubric import CRITERIA  # noqa: E402
from llmeval.storage.db import Store  # noqa: E402

st.set_page_config(page_title="LLM Evaluation Platform", layout="wide")
st.title("LLM Evaluation Platform")

DB_URL = os.environ.get("DATABASE_URL", "sqlite:///./llmeval.db")


@st.cache_resource
def get_store() -> Store:
    return Store(DB_URL)


store = get_store()
runs = store.list_runs()
labels = store.get_human_labels()

results_dir = ROOT / "bench" / "results"
artifacts = {}
for p in sorted(results_dir.glob("*.json")):
    try:
        artifacts[p.stem] = json.loads(p.read_text())
    except Exception:
        pass

tabs = st.tabs(["Scores", "Agreement", "Bias", "Regression", "Runs", "Label"])

# ---------------- Scores ----------------
with tabs[0]:
    st.header("Evaluation scores")
    if not runs:
        st.info("No eval runs yet. Run `make eval`.")
    else:
        run_ids = [r["run_id"] for r in runs]
        sel = st.selectbox("Run", run_ids, index=0)
        run = next(r for r in runs if r["run_id"] == sel)
        st.write(
            f"Judge: `{run['judge_name']}` · dataset: `{run['dataset']}` · n={run['n_samples']}"
        )
        means = run["mean_scores"]
        st.bar_chart({c: means.get(c, 0) for c in CRITERIA})
        st.dataframe([{"criterion": c, "mean_score": round(means.get(c, 0), 3)} for c in CRITERIA])

# ---------------- Agreement ----------------
with tabs[1]:
    st.header("Human vs LLM agreement")
    st.write(
        f"Calibration set: **{len(labels)} human-labeled items**. "
        "Agreement = quadratic-weighted Cohen's kappa + Spearman rho on the 1-5 ordinal scale."
    )
    agree_src = None
    for key in ("baseline-stub-001", "ci-gate"):
        if key in artifacts and artifacts[key].get("agreements"):
            agree_src = artifacts[key]
            break
    if not agree_src:
        st.info("No agreement artifact yet. Run `make eval`.")
    else:
        st.caption(f"source: `{agree_src['run_id']}` (n={agree_src['n_human_labeled']} labeled)")
        rows = [
            {
                "criterion": a["criterion"],
                "weighted_kappa": round(a["weighted_kappa"], 3),
                "spearman_rho": round(a["spearman_rho"], 3),
                "p_value": f"{a['spearman_p']:.2g}" if a["spearman_p"] else "n/a",
                "mean_human": round(a["mean_human"], 2),
                "mean_judge": round(a["mean_judge"], 2),
            }
            for a in agree_src["agreements"]
        ]
        st.dataframe(rows, use_container_width=True)
        st.bar_chart({r["criterion"]: r["weighted_kappa"] for r in rows})
        st.subheader("Worst disagreements")
        analysis = artifacts.get("analysis")
        if analysis:
            for d in analysis["disagreements_top5"]:
                with st.expander(f"{d['sample_id']} (total gap {d['total_gap']})"):
                    st.json({"human": d["human"], "judge": d["judge"]})

# ---------------- Bias ----------------
with tabs[2]:
    st.header("Bias metrics")
    bias_rows = []
    for r in runs:
        for b in store.get_bias_reports(r["run_id"]):
            bias_rows.append({"run_id": r["run_id"], **b})
    if artifacts.get("analysis"):
        sp = artifacts["analysis"]["self_preference"]
        bias_rows.append(
            {
                "run_id": "analysis",
                "test": sp["test"],
                "metric": sp["metric"],
                "value": round(sp["value"], 3),
                "detail": sp["detail"],
            }
        )
        ctl = artifacts["analysis"]["self_preference_control_plain_stub"]
        bias_rows.append(
            {
                "run_id": "analysis (control)",
                "test": ctl["test"],
                "metric": ctl["metric"],
                "value": round(ctl["value"], 3),
                "detail": ctl["detail"],
            }
        )
    if not bias_rows:
        st.info("No bias measurements yet. Run `make eval`.")
    else:
        for b in bias_rows:
            st.subheader(f"{b['test']} — {b['metric']}")
            st.metric(label=b["test"], value=f"{b['value']:.3f}")
            with st.expander("detail"):
                st.json(b["detail"])
        st.caption(
            "Position flip rate 0.0 is expected for the deterministic stub judge "
            "(order-independent by construction). Length bias rho=0.264, p=0.158: "
            "not significant at n=30."
        )

# ---------------- Regression ----------------
with tabs[3]:
    st.header("Regression trends")
    if len(runs) < 1:
        st.info("No runs yet.")
    else:
        st.write("Mean criterion scores per run (newest first):")
        trend = [
            {"run_id": r["run_id"], **{c: r["mean_scores"].get(c) for c in CRITERIA}} for r in runs
        ]
        st.dataframe(trend, use_container_width=True)
        st.line_chart({c: [r["mean_scores"].get(c, 0) for r in reversed(runs)] for c in CRITERIA})
        baseline = artifacts.get("baseline")
        if baseline:
            st.subheader("Committed baseline (CI gate compares against this)")
            st.json(baseline["mean_scores"])
            st.caption(
                f"Gate: fail if any criterion drops >{settings.regression_max_drop:.0%} "
                f"or overall kappa < {settings.regression_min_kappa:.2f}."
            )
        demo = (artifacts.get("analysis") or {}).get("regression_demo")
        if demo:
            st.subheader("Regression-gate demo (degraded variant: citations stripped)")
            st.write(f"Gate passed: **{demo['verdict']['passed']}**")
            for f_ in demo["verdict"]["failures"]:
                st.error(f_)

# ---------------- Runs ----------------
with tabs[4]:
    st.header("Historical runs")
    st.dataframe(
        [
            {
                "run_id": r["run_id"],
                "judge": r["judge_name"],
                "dataset": r["dataset"],
                "n": r["n_samples"],
                "created": r["created_at"],
            }
            for r in runs
        ],
        use_container_width=True,
    )
    st.caption(f"History store: {'Postgres' if settings.is_postgres else 'SQLite'} ({DB_URL})")

# ---------------- Label ----------------
with tabs[5]:
    st.header("Human labeling")
    st.write("Score one output per criterion (1-5, anchored rubric). Saved to the history store.")
    samples = store.get_samples("calibration")
    labeled_ids = {lb["sample_id"] for lb in labels}
    unlabeled = [s for s in samples if s["sample_id"] not in labeled_ids]
    pool = unlabeled or samples
    st.caption(f"{len(labeled_ids)}/{len(samples)} labeled")
    if pool:
        sid = st.selectbox("Sample", [s["sample_id"] for s in pool])
        s = next(x for x in pool if x["sample_id"] == sid)
        st.subheader("Input")
        st.write(s["input"])
        st.subheader("Output")
        st.write(s["output"])
        st.subheader("Contexts")
        for c in s["contexts"]:
            st.code(f"[{c['chunk_id']}] {c['text']}", language="text")
        st.subheader("Expected")
        st.write(s["expected"])
        scores = {}
        cols = st.columns(len(CRITERIA))
        for col, c in zip(cols, CRITERIA, strict=True):
            scores[c] = col.slider(c, 1, 5, 3)
        note = st.text_input("Note (optional)")
        if st.button("Save label"):
            store.add_human_label(sid, "dashboard", scores, note)
            st.success(f"Saved label for {sid}")
            st.cache_resource.clear()
