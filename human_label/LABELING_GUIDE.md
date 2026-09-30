# Labeling guide (how the calibration set was scored)

**Provenance (read this first).** Labeler: `manual-rubric-pass-v1` — these
30 reference labels were produced by the engineering agent on 2026-09-30 by
manually applying the rubric in `src/llmeval/rubric.py` to each item in full
(input, output, all context chunks, expected answer), scored 1–5 per
criterion. They are **not independent human annotation**; treat them as
carefully constructed reference labels for calibrating judges, not as
ground truth from a separate annotator. Labels were written from judgment,
NOT copied from the stub judge's output.

Key judgment calls (the interesting ones):

- **Lexical support != semantic support.** The stub judge counts a citation
  valid at >=50% token overlap. `div-broken` ("The dividend was suspended...")
  shares 54% of its tokens with the chunk yet says the opposite of it. Human
  label: citation_precision = 1. This is the project's core honesty finding:
  token-overlap citation checking catches missing support but not
  contradiction; it motivates the LLM-judge production path.
- **Paraphrase vs coverage.** `capex-mediocre` covers the expected claim in
  different words ("mainly the Dresden fab starting production in H2 2026").
  The judge's 60%-overlap coverage check misses it (recall=1); a human sees
  the claim is covered (recall=5).
- **Uncited-but-true claims.** `legal-mediocre`, `debt-mediocre`, `rev-mediocre`
  get faithfulness 5/2/3 (claims are true per the chunks) but
  citation_precision=1 and citation_recall=1 — the rubric deliberately
  punishes missing citations even when the facts are right.
- **Wrong-chunk citations.** `cust-mediocre` cites the revenue chunk for a
  customer-concentration claim. The numbers are right, the citation is wrong:
  precision=1, faithfulness=5. Citation validity and factuality are separate
  axes — that separation is intentional.
- **answer_f1 (human)** is a holistic correctness-vs-expected judgment, not a
  recomputation of token F1. `margin-mediocre` gets 4: all numbers right, one
  invented elaboration.
- **conciseness (human)** judges padding/repetition, roughly aligned with the
  output/expected length ratio but penalizing fluff the ratio misses
  (`seg-mediocre` human 3 vs length-ratio 4).

Format (`human_label/seed_labels.jsonl`, one JSON object per line):

    {"sample_id": "rev-good", "labeler": "Rupak Jee Kashyap",
     "scores": {"citation_precision": 5, "citation_recall": 5, "faithfulness": 5,
                "answer_f1": 5, "conciseness": 5},
     "note": "fully cited, correct, concise", "labeled_at": "2026-09-30T..."}
