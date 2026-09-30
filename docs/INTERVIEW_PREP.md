# Interview Prep — LLM Evaluation Platform

Every answer below is grounded in this repo's actual code and measured
results. If an interviewer probes deeper, the file/artifact to open is named.

## The one-minute pitch

"I built an LLM-as-judge evaluation platform: a five-criterion rubric on a
1–5 ordinal scale, a deterministic rule-based judge for CI, a real Groq
LLM judge emitting structured Pydantic scores, a 30-item hand-labeled
calibration set, agreement metrics (quadratic-weighted Cohen's kappa,
Spearman), bias probes (position, length, self-preference), a regression
gate that fails CI on quality drops, eval history in Postgres, a Streamlit
dashboard with a human-labeling UI, and adapters so sibling RAG projects can
plug in. The deterministic judge agrees with my hand labels at overall
κ=0.907; the Groq gpt-oss-120b judge at κ=0.984."

## Questions you can now answer cold

**Why LLM-as-judge instead of just BLEU/ROUGE?**
Lexical overlap can't check citation validity or faithfulness — our
biggest human-vs-judge disagreement (`div-broken`) was a semantically
contradictory dividend claim the token-overlap judge scored as supported
because lexical overlap exceeded 50%. The rubric's citation_precision /
citation_recall criteria exist precisely because RAG answers live or die on
whether citations point at supporting chunks. See `bench/results/analysis.json`
top_disagreements.

**How do you know the judge is any good?**
Agreement against 30 hand labels: quadratic-weighted kappa per criterion
plus Spearman. Weighted kappa penalizes a 1-vs-5 disagreement more than a
4-vs-5, which matches how much we care. Implementation is hand-rolled in
`src/llmeval/agreement.py` and cross-checked against independent
observed/expected arithmetic in tests.

**Deterministic vs LLM judge trade-off?**
The stub judge is exact, free, and runs in CI in ~2s for 30 samples
(κ=0.907 vs hand labels). The Groq judge (κ=0.84) reasons about semantics
but costs tokens (~1.9k/sample, ~$0.0013/sample at gpt-oss-120b pricing),
takes ~3s/sample, and hit 429 rate limits until I serialized requests.
Production pattern: stub in CI on every commit, LLM judge nightly or on
release candidates.

**How did you handle the Groq API's sharp edges?**
Three, all verified live: (1) api.groq.com is behind Cloudflare — requests
without a browser User-Agent get error 1010; (2) the key serves
openai/gpt-oss-120b, openai/gpt-oss-20b, qwen/qwen3.8-27b — NOT the
llama-3.3-70b-versatile the docs suggested; (3) gpt-oss returns
`reasoning_tokens` in usage and terse content, so the parser strips
markdown fences, brace-matches, normalizes criterion names, and records
errors instead of crashing. See `src/llmeval/judges/groq.py`.

**What bias did you actually measure?**
Position flip rate 0.0 on the deterministic judge (expected — it's
order-independent), length-vs-score Spearman 0.264 (p=0.158, not
significant), and a styled-stub self-preference delta of 0.317 vs 0.000 for
the plain control. The real self-preference test: generate answers with
gpt-oss-120b and qwen3.8-27b, have all three judges score them blind, and
compare own-family vs other-family means. See
`bench/results/selfpreference-groq.json`.

**Tell me about a real bug you found.**
Postgres caught what SQLite masked: `Store.save_run` inserted the parent
`eval_runs` row and child `bias_reports` rows in one session. Without
`relationship()`s, SQLAlchemy's unit of work can't see the FK dependency
and falls back to alphabetical mapper ordering — "BiasReportRow" <
"EvalRun" — so the child INSERT fired first and Postgres raised
ForeignKeyViolation. SQLite never enforces FKs, so it passed there. Fix:
explicit `s.flush()` after adding the parent, with a comment explaining
why. The `judge_scores` table only worked by luck ("EvalRun" < "JudgeScore"
alphabetically).

**How does the regression gate work?**
`bench/results/baseline.json` is the committed baseline. `check_regression`
fails the run if any criterion mean drops more than 15% relative or
overall kappa falls below 0.40. Proven with a degraded variant (citations
stripped): citation_precision −68.8%, citation_recall −62.0%, gate exits 2.
Captured in `demo/degraded-gate.log`.

**Why Pydantic for judge output?**
The judge contract is `JudgeOutput`: per-criterion score + rationale +
measured intermediates + citations checked + errors + latency + token
usage. Structured output makes disagreement analysis and bias probes
computable instead of eyeballed. Malformed LLM responses coerce/clamp and
record errors rather than crashing the run.

## Numbers to have on hand (all from committed artifacts)

| Judge | Overall κ | Worst criterion κ |
|---|---|---|
| Deterministic stub | 0.907 | answer_f1 0.674 |
| Groq openai/gpt-oss-120b | 0.984 | conciseness 0.640 |

- Calibration: 30 items, 10 scenarios × good/mediocre/broken, fictional Meridian Dynamics corpus.
- Groq cost: ~2.1k tokens/sample (incl. ~1k reasoning), $0.0274 for 30 samples (~$0.001/sample at $0.15/$0.60 per 1M).
- Tests: 30 pytest, all green; ruff clean.

## If they ask "what would you do with more time?"

1. Replace the token-overlap faithfulness heuristic with an NLI model or
   the LLM judge's own entailment check (the `div-broken` failure).
2. Calibrate the 1–5 scale with item-response theory once the label set
   grows past ~200 items and multiple labelers.
3. Add a second human labeler and report inter-annotator agreement as the
   ceiling the judge is chasing.
4. DeepEval adapter is written but unexecuted — run it as a third judge
   for triangulation.
