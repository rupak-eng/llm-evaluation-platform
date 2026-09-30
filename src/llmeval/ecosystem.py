"""Ecosystem adapter: ingest the shared sample schema natively.

Sibling projects (knowledge-graph-rag, multi-agent-research-assistant) emit
``eval/sample_outputs.jsonl`` with one JSON object per line::

    {"input": ..., "output": ..., "contexts": [{"text","source","chunk_id"}],
     "expected": ..., "metadata": {...}}

This adapter validates each line into ``EvalSample`` (assigning a stable
``sample_id`` when absent) and writes a harness-ready ``.jsonl`` fixture.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..schemas import EvalSample


def _stable_id(obj: dict) -> str:
    h = hashlib.sha256((obj.get("input", "") + obj.get("output", "")).encode()).hexdigest()[:12]
    return f"eco-{h}"


def ingest_ecosystem_jsonl(src: str | Path, dataset: str) -> list[EvalSample]:
    samples: list[EvalSample] = []
    with open(src) as f:
        for _, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            obj.setdefault("sample_id", _stable_id(obj))
            obj.setdefault("metadata", {})["dataset"] = dataset
            samples.append(EvalSample.model_validate(obj))
    return samples


def write_fixture(samples: list[EvalSample], dest: str | Path) -> None:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w") as f:
        for s in samples:
            f.write(s.model_dump_json() + "\n")
