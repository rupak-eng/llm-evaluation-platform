"""Position-bias probe with a real Groq judge.

Runs pairwise comparisons in both orders and measures the flip rate:
fraction of pairs whose verdict changes when A/B are swapped. Writes
bench/results/position-bias-groq.json.

Usage: python bench/scripts/run_position_bias.py --model openai/gpt-oss-120b --pairs 8
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from llmeval.bias import make_pairs, position_bias  # noqa: E402
from llmeval.judges.groq import GroqJudge  # noqa: E402
from llmeval.schemas import EvalSample  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--pairs", type=int, default=8)
    args = ap.parse_args()

    samples = []
    with open(ROOT / "adapters" / "fixtures" / "calibration.jsonl") as f:
        for line in f:
            if line.strip():
                samples.append(EvalSample.model_validate(json.loads(line)))
    judge = GroqJudge(model=args.model)
    pairs = make_pairs(samples, seed=42, n_pairs=args.pairs)
    report = position_bias(judge.judge, pairs)
    report["provenance"] = {
        "provider": "groq",
        "model": args.model,
        "measured_utc": datetime.now(UTC).isoformat(),
    }
    dest = ROOT / "bench" / "results" / "position-bias-groq.json"
    dest.write_text(json.dumps(report, indent=2))
    d = report["detail"]
    print(f"flip_rate={report['value']:.3f} ({d['n_flips']}/{d['n_pairs']})")
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
