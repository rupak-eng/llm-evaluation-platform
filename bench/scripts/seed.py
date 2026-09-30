"""Seed script: loads the calibration fixtures + human labels into the DB.

Usage: make seed   (or: python bench/scripts/seed.py)
Idempotent: upserts samples, appends labels only if none exist for the dataset.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from llmeval.config import settings  # noqa: E402
from llmeval.schemas import EvalSample  # noqa: E402
from llmeval.storage.db import Store  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    store = Store(settings.database_url)
    samples: list[dict] = []
    with open(ROOT / "adapters" / "fixtures" / "calibration.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                s = EvalSample.model_validate(json.loads(line))
                samples.append(s.model_dump(mode="json"))
    n = store.upsert_samples("calibration", samples)
    print(f"upserted {n} samples (db: {'postgres' if settings.is_postgres else 'sqlite'})")

    existing = {lb["sample_id"] for lb in store.get_human_labels()}
    added = 0
    with open(ROOT / "human_label" / "seed_labels.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            lb = json.loads(line)
            if lb["sample_id"] in existing:
                continue
            store.add_human_label(lb["sample_id"], lb["labeler"], lb["scores"], lb.get("note", ""))
            added += 1
    print(f"added {added} human labels")


if __name__ == "__main__":
    main()
