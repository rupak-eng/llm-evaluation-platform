"""Groq LLM judge (production judge path). In-repo provider, stdlib only.

Auth: GROQ_API_KEY env var (transient) first, else the Secure Vault surrogate
for ``custom.groq`` applied via ``add_surrogate_to_request``. The key is kept
in memory only — never logged, printed, or persisted.

Verified-live gotchas (2026-09-30):
- api.groq.com sits behind Cloudflare: requests MUST carry a browser-like
  User-Agent or they are rejected with error 1010.
- gpt-oss models return ``reasoning_tokens`` in usage and can emit terse
  content, so the JSON parser is defensive (fences stripped, brace matching,
  per-criterion coercion with errors recorded, never a crash).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import urllib.request

from ..rubric import CRITERIA, RUBRIC_TEXT
from ..schemas import CriterionScore, EvalSample, JudgeOutput

log = logging.getLogger("llmeval.judges.groq")

BASE_URL = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
PRIMARY_MODEL = "openai/gpt-oss-120b"
CROSS_MODELS = ["openai/gpt-oss-20b", "qwen/qwen3.8-27b"]

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

SYSTEM_PROMPT = (
    "You are an evaluation judge for RAG answers. Score the OUTPUT against the "
    "rubric below. First write a short 'rationale' per criterion, then the score. "
    "A citation [chunk_id] is valid only if the chunk_id appears in CONTEXTS and "
    "the cited claim is actually supported by that chunk's text. "
    "Return ONLY a JSON object with this exact shape:\n"
    '{"rationales": {"<criterion>": "<1-3 sentences>"}, '
    '"scores": [{"criterion": "<criterion>", "score": <1-5 integer>, '
    '"citations": ["<chunk_id>", ...]}], '
    '"citations_checked": [{"chunk_id": "<id>", "valid": true, "reason": "<why>"}]}'
)


def _user_prompt(sample: EvalSample) -> str:
    contexts = "\n".join(f"[{c.chunk_id}] {c.text}" for c in sample.contexts)
    return (
        f"RUBRIC:\n{RUBRIC_TEXT}\n\n"
        f"CRITERIA (score each): {', '.join(CRITERIA)}\n\n"
        f"INPUT:\n{sample.input}\n\n"
        f"CONTEXTS:\n{contexts}\n\n"
        f"EXPECTED ANSWER:\n{sample.expected}\n\n"
        f"OUTPUT TO JUDGE:\n{sample.output}\n"
    )


def _resolve_api_key() -> str | None:
    """GROQ_API_KEY env first; else the vault surrogate. Never logged."""
    env_key = os.environ.get("GROQ_API_KEY")
    if env_key:
        return env_key
    sys.path.insert(0, "/opt/hatch/skills/skill-creator/bin")
    try:
        from dynamic_credentials import dynamic_credential_entry

        return str(dynamic_credential_entry("custom.groq")["surrogate"]).strip()
    except Exception as e:  # ImportError | DynamicCredentialError
        log.warning("no Groq credential available: %s", type(e).__name__)
        return None


def _extract_json(text: str) -> dict:
    """Defensive JSON extraction for terse model output."""
    t = text.strip()
    if t.startswith("```"):
        # strip markdown fences
        lines = t.splitlines()
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        t = "\n".join(lines).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    # brace matching fallback: largest {...} span
    start, end = t.find("{"), t.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(t[start : end + 1])
    raise ValueError("no JSON object found in judge response")


def _coerce_scores(data: dict, sample_id: str) -> tuple[list[CriterionScore], list[str]]:
    scores: list[CriterionScore] = []
    errors: list[str] = []
    rationales = data.get("rationales") or {}
    items = data.get("scores") or data.get("criteria") or []
    for item in items:
        if not isinstance(item, dict):
            continue
        crit = str(item.get("criterion", item.get("name", ""))).strip().lower()
        crit = crit.replace("-", "_").replace(" ", "_")
        aliases = {
            "citation_precison": "citation_precision",  # common model typo
            "answer_token_f1": "answer_f1",
            "token_f1": "answer_f1",
            "faithfullness": "faithfulness",
        }
        crit = aliases.get(crit, crit)
        if crit not in CRITERIA:
            errors.append(f"unknown criterion in judge response: {crit!r}")
            continue
        try:
            score = int(item.get("score"))
        except (TypeError, ValueError):
            errors.append(f"non-integer score for {crit}")
            continue
        score = max(1, min(5, score))
        scores.append(
            CriterionScore(
                criterion=crit,
                score=score,
                rationale=str(rationales.get(crit, item.get("rationale", "")))[:500],
                measured={"citations": item.get("citations", [])},
            )
        )
    missing = [c for c in CRITERIA if c not in {s.criterion for s in scores}]
    if missing:
        errors.append(f"judge omitted criteria: {missing}")
    return scores, errors


class GroqJudge:
    """LLM-as-judge via Groq's OpenAI-compatible API. Structured Pydantic out."""

    name = "groq"

    def __init__(self, model: str = PRIMARY_MODEL, timeout: int = 120, max_retries: int = 6):
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self._api_key: str | None = None  # resolved lazily, memory only

    @property
    def judge_name(self) -> str:
        return f"groq:{self.model}"

    def _key(self) -> str:
        if not self._api_key:
            key = _resolve_api_key()
            if not key:
                raise RuntimeError(
                    "no Groq credential: set GROQ_API_KEY or submit the custom.groq vault entry"
                )
            self._api_key = key
        return self._api_key

    def _post(self, payload: dict) -> dict:
        # NOTE: we intentionally do NOT use the vault's add_surrogate_to_request
        # here because the key may come from the transient env var; when it
        # comes from the vault we already hold the surrogate in memory and
        # attach it as a Bearer header ourselves. Either way it is never logged.
        body = json.dumps(payload).encode()
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(
                f"{BASE_URL}/chat/completions",
                data=body,
                headers={"Content-Type": "application/json", "User-Agent": USER_AGENT},
                method="POST",
            )
            req.add_header("Authorization", "Bearer " + self._key())
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode())
            except Exception as e:  # HTTPError (429/5xx) or URLError/timeout
                last_err = e
                code = getattr(e, "code", None)
                if code in (400, 401, 403, 404):
                    raise  # auth/shape errors are not retryable
                # honor Retry-After on 429; otherwise exponential backoff capped at 60s
                wait = 2**attempt
                try:
                    ra = e.headers.get("Retry-After") if hasattr(e, "headers") else None
                    if ra:
                        wait = max(wait, int(float(ra)))
                except Exception:
                    pass
                wait = min(wait, 60)
                log.warning(
                    "groq call failed (attempt %d/%d, code=%s): %s; retrying in %ds",
                    attempt + 1,
                    self.max_retries + 1,
                    code,
                    type(e).__name__,
                    wait,
                )
                time.sleep(wait)
        raise RuntimeError(f"groq call failed after retries: {last_err}")

    def judge(self, sample: EvalSample) -> JudgeOutput:
        t0 = time.perf_counter()
        errors: list[str] = []
        usage: dict = {"model": self.model, "provider": "groq"}
        try:
            resp = self._post(
                {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": _user_prompt(sample)},
                    ],
                    "temperature": 0,
                    "response_format": {"type": "json_object"},
                }
            )
            msg = resp["choices"][0]["message"]
            content = msg.get("content") or ""
            if not content and "reasoning_content" in msg:
                errors.append("empty content; reasoning_content present but unused")
            data = _extract_json(content)
            scores, perr = _coerce_scores(data, sample.sample_id)
            errors.extend(perr)
            citations_checked = data.get("citations_checked", [])
            u = resp.get("usage") or {}
            usage.update(
                {
                    "prompt_tokens": u.get("prompt_tokens"),
                    "completion_tokens": u.get("completion_tokens"),
                    "reasoning_tokens": (u.get("completion_tokens_details") or {}).get(
                        "reasoning_tokens"
                    ),
                    "total_tokens": u.get("total_tokens"),
                }
            )
        except Exception as e:
            errors.append(f"{type(e).__name__}: {str(e)[:200]}")
            scores, citations_checked = [], []
            log.error("groq judge failed for %s: %s", sample.sample_id, type(e).__name__)
        latency = time.perf_counter() - t0
        usage["latency_s"] = round(latency, 2)
        return JudgeOutput(
            sample_id=sample.sample_id,
            judge_name=self.judge_name,
            scores=scores,
            citations_checked=citations_checked if isinstance(citations_checked, list) else [],
            errors=errors,
            latency_s=latency,
            usage=usage,
        )
