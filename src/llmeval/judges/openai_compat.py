"""OpenAI-compatible LLM judge (production path). Lazy: nothing imported at module load.

Uses the rubric text as the system prompt and asks the model to reason first,
then emit structured JSON matching ``JudgeOutput``. Requires
``JUDGE_BASE_URL`` / ``JUDGE_API_KEY`` / ``JUDGE_MODEL`` env vars.
"""
from __future__ import annotations

import json
import os
import time

from ..rubric import CRITERIA, RUBRIC_TEXT
from ..schemas import EvalSample, JudgeOutput


def _client():
    from openai import OpenAI  # lazy: optional dependency

    return OpenAI(base_url=os.environ["JUDGE_BASE_URL"], api_key=os.environ["JUDGE_API_KEY"])


PROMPT_TEMPLATE = """You are an evaluation judge. Score the OUTPUT below against the rubric.
Reason about each criterion FIRST, then give the score. Return ONLY valid JSON.

RUBRIC:
{rubric}

INPUT:
{input}

CONTEXTS (chunk_id -> text):
{contexts}

EXPECTED ANSWER:
{expected}

OUTPUT TO JUDGE:
{output}

Return JSON exactly in this shape:
{{"scores": [{{"criterion": "<one of: {criteria}>", "score": <1-5 int>, "rationale": "<why>"}}],
  "citations_checked": [{{"chunk_id": "...", "valid": true/false, "reason": "..."}}]}}"""


class OpenAICompatJudge:
    """LLM-as-judge via any OpenAI-compatible API."""

    name = "openai_compat"

    def __init__(self, model: str | None = None):
        self.model = model or os.environ.get("JUDGE_MODEL", "")

    def judge(self, sample: EvalSample) -> JudgeOutput:
        t0 = time.perf_counter()
        client = _client()
        contexts = "\n".join(f"[{c.chunk_id}] {c.text}" for c in sample.contexts)
        prompt = PROMPT_TEMPLATE.format(
            rubric=RUBRIC_TEXT,
            criteria=", ".join(CRITERIA),
            input=sample.input,
            contexts=contexts,
            expected=sample.expected,
            output=sample.output,
        )
        resp = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        data = json.loads(resp.choices[0].message.content or "{}")
        out = JudgeOutput(
            sample_id=sample.sample_id,
            judge_name=f"{self.name}:{self.model}",
            scores=data.get("scores", []),
            citations_checked=data.get("citations_checked", []),
            latency_s=time.perf_counter() - t0,
        )
        # validate all criteria present
        missing = set(CRITERIA) - set(out.by_criterion())
        if missing:
            out.errors.append(f"judge omitted criteria: {sorted(missing)}")
        return out
