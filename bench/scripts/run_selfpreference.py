"""Real self-preference test with Groq judges.

Protocol:
  1. GENERATE: for 8 calibration inputs, generate a grounded answer with
     openai/gpt-oss-120b and with qwen/qwen3.8-27b (same contexts, citation
     prompt). Generator identity is NOT shown to the judges.
  2. JUDGE: openai/gpt-oss-120b, openai/gpt-oss-20b, qwen/qwen3.8-27b each
     score all 16 outputs blind against the rubric.
  3. MEASURE: for each judge, mean score on 120b-generated vs qwen-generated
     outputs. Self-preference delta = own-family mean minus other-family mean
     (gpt-oss-120b and gpt-oss-20b count as one family).

Writes bench/results/selfpreference-groq.json with per-judge tables.
All provenance (model ids, UTC timestamps, tokens) recorded.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from llmeval.judges.groq import BASE_URL, USER_AGENT, GroqJudge, _resolve_api_key  # noqa: E402
from llmeval.schemas import EvalSample  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
GENERATORS = ["openai/gpt-oss-120b", "qwen/qwen3.8-27b"]
JUDGES = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]
TOPICS = ["rev", "margin", "seg", "debt", "rd", "capex", "legal", "div"]

GEN_SYSTEM = (
    "Answer the user's question using ONLY the facts in CONTEXTS. "
    "Every factual claim must end with a citation [chunk_id] matching the chunk "
    "it came from. If the contexts lack the answer, say so plainly. Be concise."
)


def generate_answer(model: str, sample: EvalSample) -> dict:
    contexts = "\n".join(f"[{c.chunk_id}] {c.text}" for c in sample.contexts)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": GEN_SYSTEM},
            {
                "role": "user",
                "content": f"CONTEXTS:\n{contexts}\n\nQUESTION:\n{sample.input}",
            },
        ],
        "temperature": 0.2,
    }
    t0 = time.perf_counter()
    req = urllib.request.Request(
        f"{BASE_URL}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "User-Agent": USER_AGENT},
        method="POST",
    )
    key = _resolve_api_key()
    assert key, "no Groq credential"
    req.add_header("Authorization", "Bearer " + key)
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode())
    latency = time.perf_counter() - t0
    return {
        "text": data["choices"][0]["message"]["content"] or "",
        "usage": data.get("usage") or {},
        "latency_s": round(latency, 2),
    }


def main() -> int:
    samples = []
    with open(ROOT / "adapters" / "fixtures" / "calibration.jsonl") as f:
        for line in f:
            if line.strip():
                s = EvalSample.model_validate(json.loads(line))
                if s.sample_id.split("-")[0] in TOPICS and s.sample_id.endswith("-good"):
                    samples.append(s)
    samples = samples[:8]
    assert len(samples) == 8, f"expected 8 generator inputs, got {len(samples)}"

    started = datetime.now(UTC)
    # 1. generate
    gen_outputs: list[dict] = []

    def _gen(args):
        model, s = args
        r = generate_answer(model, s)
        return {
            "generator": model,
            "sample_id": f"gen-{s.sample_id}-{model.split('/')[-1]}",
            "input": s.input,
            "contexts": [c.model_dump() for c in s.contexts],
            "expected": s.expected,
            "output": r["text"],
            "gen_usage": r["usage"],
            "gen_latency_s": r["latency_s"],
        }

    with ThreadPoolExecutor(max_workers=1) as ex:
        jobs = [(m, s) for m in GENERATORS for s in samples]
        for i, g in enumerate(ex.map(_gen, jobs)):
            gen_outputs.append(g)
            print(f"  generated {i + 1}/{len(jobs)} ({g['generator']})", flush=True)
            time.sleep(5)  # be gentle: qwen preview tier is tightly rate-limited

    # 2. judge blind (judges never see the generator field)
    judge_samples = [
        EvalSample(
            sample_id=g["sample_id"],
            input=g["input"],
            output=g["output"],
            contexts=g["contexts"],
            expected=g["expected"],
            metadata={},
        )
        for g in gen_outputs
    ]
    results: dict[str, dict] = {}
    for jm in JUDGES:
        judge = GroqJudge(model=jm)
        with ThreadPoolExecutor(max_workers=1) as ex:
            outs = list(ex.map(judge.judge, judge_samples))
        per_gen: dict[str, list[float]] = {}
        for g, o in zip(gen_outputs, outs, strict=True):
            mean = sum(x.score for x in o.scores) / len(o.scores) if o.scores else 0.0
            per_gen.setdefault(g["generator"], []).append(mean)
        family = "gpt-oss" if "gpt-oss" in jm else "qwen"
        own = [
            m
            for gen, ms in per_gen.items()
            if (("gpt-oss" in gen) == (family == "gpt-oss"))
            for m in ms
        ]
        other = [
            m
            for gen, ms in per_gen.items()
            if (("gpt-oss" in gen) != (family == "gpt-oss"))
            for m in ms
        ]
        results[jm] = {
            "family": family,
            "mean_on_120b_outputs": round(
                sum(per_gen[GENERATORS[0]]) / len(per_gen[GENERATORS[0]]), 3
            ),
            "mean_on_qwen_outputs": round(
                sum(per_gen[GENERATORS[1]]) / len(per_gen[GENERATORS[1]]), 3
            ),
            "self_preference_delta": round(sum(own) / len(own) - sum(other) / len(other), 3),
            "n_judged": len(outs),
            "judge_errors": sum(1 for o in outs if o.errors),
        }
        print(f"judge {jm}: delta={results[jm]['self_preference_delta']}", flush=True)

    report = {
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(UTC).isoformat(),
        "generators": GENERATORS,
        "judges": JUDGES,
        "n_inputs": len(samples),
        "note": (
            "Positive self_preference_delta means the judge scored outputs from its "
            "own model family higher. Judges were blind to generator identity."
        ),
        "per_judge": results,
        "generated_outputs": [
            {k: g[k] for k in ("generator", "sample_id", "output")} for g in gen_outputs
        ],
    }
    dest = ROOT / "bench" / "results" / "selfpreference-groq.json"
    dest.write_text(json.dumps(report, indent=2))
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
