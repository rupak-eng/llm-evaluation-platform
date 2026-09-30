"""Groq real-judge benchmark: full protocol vs the hand-labeled calibration set.

Runs GroqJudge over all 30 calibration samples, computes weighted kappa +
Spearman vs human labels, and records full provenance:
  provider, exact model id, UTC timestamp, dataset size, per-call latency,
  token totals (incl. reasoning_tokens), cost estimate.

Cost estimate uses Groq's published per-1M-token rates as of 2026-09-30
(verified against multiple public sources; re-check console.groq.com before
budgeting — rates change):
  openai/gpt-oss-120b : $0.15 in / $0.60 out
  openai/gpt-oss-20b  : $0.075 in / $0.30 out
  qwen/qwen3.8-27b    : $0.80 in / $4.00 out (preview model)

Usage:
  python bench/scripts/run_groq_benchmark.py --model openai/gpt-oss-120b
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from llmeval.agreement import agreement  # noqa: E402
from llmeval.judges.groq import GroqJudge  # noqa: E402
from llmeval.rubric import CRITERIA  # noqa: E402
from llmeval.schemas import CriterionScore, EvalSample, JudgeOutput  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

# per-1M-token USD, Groq published pricing 2026-09-30 (estimate — verify before budgeting)
PRICING = {
    "openai/gpt-oss-120b": (0.15, 0.60),
    "openai/gpt-oss-20b": (0.075, 0.30),
    "qwen/qwen3.8-27b": (0.80, 4.00),
}


def load_calibration() -> tuple[list[EvalSample], dict[str, dict]]:
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
    return samples, labels


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="judge only first N samples (smoke)")
    ap.add_argument(
        "--resume", action="store_true", help="skip samples already clean in the existing raw JSONL"
    )
    args = ap.parse_args()

    samples, labels = load_calibration()
    if args.limit:
        samples = samples[: args.limit]
    judge = GroqJudge(model=args.model)
    started = datetime.now(UTC)
    wall0 = time.perf_counter()

    slug = args.model.replace("/", "-")
    raw_dest = ROOT / "bench" / "results" / f"groq-{slug}-raw.jsonl"
    already_clean: dict[str, dict] = {}
    if args.resume and raw_dest.exists():
        for line in raw_dest.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if not r["errors"] and len(r["scores"]) == len(CRITERIA):
                    already_clean[r["sample_id"]] = r
        samples = [s for s in samples if s.sample_id not in already_clean]
        print(f"resume: {len(already_clean)} already clean, {len(samples)} to judge")

    outputs = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for out in ex.map(judge.judge, samples):
            outputs.append(out)
            n_ok = sum(1 for o in outputs if not o.errors)
            print(f"  {len(outputs)}/{len(samples)} judged ({n_ok} clean)", flush=True)

    # merge resumed clean rows back in so the artifact covers the full set
    for sid, r in already_clean.items():
        outputs.append(
            JudgeOutput(
                sample_id=sid,
                judge_name=r["judge_name"],
                scores=[CriterionScore.model_validate(s) for s in r["scores"]],
                errors=r["errors"],
                latency_s=0.0,
                usage=r["usage"],
            )
        )

    failed = [o.sample_id for o in outputs if o.errors or len(o.scores) != len(CRITERIA)]
    clean = [o for o in outputs if o.sample_id not in failed]
    by_id = {o.sample_id: o for o in outputs}
    # incremental raw save: never lose judged data if aggregation fails
    # (append mode so --resume merges with prior rows; dedupe by sample_id)
    seen: dict[str, dict] = {}
    if raw_dest.exists() and not args.resume:
        raw_dest.unlink()  # fresh run replaces stale rows
    if raw_dest.exists():
        for line in raw_dest.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                seen[r["sample_id"]] = r
    with open(raw_dest, "w") as f:
        for o in outputs:
            seen[o.sample_id] = {
                "sample_id": o.sample_id,
                "judge_name": o.judge_name,
                "scores": [s.model_dump() for s in o.scores],
                "errors": o.errors,
                "usage": o.usage,
            }
        for r in seen.values():
            f.write(json.dumps(r) + "\n")
    print(f"wrote raw {raw_dest} ({len(clean)}/{len(outputs)} clean)")
    common = [
        s.sample_id for s in samples if s.sample_id in labels and not by_id[s.sample_id].errors
    ]
    human = {c: [labels[s][c] for s in common] for c in CRITERIA}
    js = {c: [by_id[s].by_criterion()[c].score for s in common] for c in CRITERIA}
    agreements = [a.model_dump() for a in agreement(human, js, CRITERIA)]

    tot_prompt = sum((o.usage.get("prompt_tokens") or 0) for o in outputs)
    tot_completion = sum((o.usage.get("completion_tokens") or 0) for o in outputs)
    tot_reasoning = sum((o.usage.get("reasoning_tokens") or 0) for o in outputs)
    latencies = [o.latency_s for o in outputs]
    rate_in, rate_out = PRICING.get(args.model, (None, None))
    cost_usd = None
    if rate_in is not None:
        cost_usd = round(tot_prompt / 1e6 * rate_in + tot_completion / 1e6 * rate_out, 4)

    result = {
        "provenance": {
            "provider": "groq",
            "model": args.model,
            "model_version_note": "Groq exposes the model id only; no separate version string",
            "dataset": "calibration",
            "n_samples": len(samples),
            "n_agreement_pairs": len(common),
            "started_utc": started.isoformat(),
            "finished_utc": datetime.now(UTC).isoformat(),
            "label": f"Real Provider: Groq {args.model}",
        },
        "mean_scores": {
            c: round(sum(o.by_criterion()[c].score for o in clean) / len(clean), 3)
            for c in CRITERIA
        }
        if clean
        else {},
        "kappas": {a["criterion"]: round(a["weighted_kappa"], 3) for a in agreements},
        "agreements": agreements,
        "tokens": {
            "prompt": tot_prompt,
            "completion": tot_completion,
            "reasoning": tot_reasoning,
            "total": tot_prompt + tot_completion,
        },
        "latency_s": {
            "mean": round(sum(latencies) / len(latencies), 2),
            "max": round(max(latencies), 2),
            "total_wall": round(time.perf_counter() - wall0, 1),
        },
        "cost_usd_estimate": cost_usd,
        "cost_note": (
            "Estimate from Groq published per-1M rates as of 2026-09-30 "
            f"({rate_in}/{rate_out}); verify at console.groq.com before budgeting."
            if cost_usd is not None
            else "no pricing on file for this model"
        ),
        "failed_samples": [
            {"sample_id": o.sample_id, "errors": o.errors} for o in outputs if o.sample_id in failed
        ],
        "n_clean": len(clean),
    }
    dest = ROOT / "bench" / "results" / f"groq-{slug}.json"
    dest.write_text(json.dumps(result, indent=2))
    print(f"wrote {dest}")
    print("kappas:", result["kappas"])
    print(f"tokens: {result['tokens']} cost_est=${cost_usd}")
    print(f"failed: {len(failed)}")
    return 0 if not failed else 2


if __name__ == "__main__":
    sys.exit(main())
