"""Ecosystem adapters: how sibling repos plug into llm-evaluation-platform.

Each sibling repo emits eval records in the shared native JSONL schema:
  {"input","output","contexts":[{"text","source","chunk_id"}],"expected","metadata"}

This module documents the per-repo mapping (field renames) and ships
ready-made example records under adapters/examples/ so any repo can be
evaluated with:
  python -m llmeval.runner --dataset <name> --judge stub --run-id <id>
after converting with: python adapters/to_native.py <repo> in.jsonl out.jsonl
"""

from __future__ import annotations

# Field mapping per sibling repo: native field -> repo's field name.
REPO_FIELD_MAP = {
    # RAG-chatbot: chat API logs {"query","answer","retrieved_chunks":[{"content","doc","id"}]}
    "rag-chatbot": {
        "input": "query",
        "output": "answer",
        "contexts": "retrieved_chunks",
        "text": "content",
        "source": "doc",
        "chunk_id": "id",
        "expected": None,  # chat logs have no gold answer; falls back to ""
    },
    # ai-cost-doctor reports: question / report_markdown / evidence[]
    "ai-cost-doctor": {
        "input": "question",
        "output": "report_markdown",
        "contexts": "evidence",
        "text": "snippet",
        "source": "source",
        "chunk_id": "ref",
        "expected": None,
    },
    # llm-research-agent runs: task / final_summary / sources[]
    "llm-research-agent": {
        "input": "task",
        "output": "final_summary",
        "contexts": "sources",
        "text": "excerpt",
        "source": "url",
        "chunk_id": "source_id",
        "expected": None,
    },
}


def to_native(repo: str, record: dict) -> dict:
    """Convert one repo-specific record to the shared native eval schema."""
    m = REPO_FIELD_MAP[repo]
    contexts = [
        {"text": c[m["text"]], "source": c[m["source"]], "chunk_id": str(c[m["chunk_id"]])}
        for c in record.get(m["contexts"], [])
    ]
    return {
        "sample_id": record.get("id") or record.get("sample_id") or "unknown",
        "input": record.get(m["input"], ""),
        "output": record.get(m["output"], ""),
        "contexts": contexts,
        "expected": record.get(m["expected"]) if m["expected"] else "",
        "metadata": {"source_repo": repo},
    }
