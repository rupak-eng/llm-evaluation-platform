"""The evaluation rubric: 5 measurable criteria on an anchored 1-5 ordinal scale.

Every criterion has an EXACT computation — no vibes. The stub judge in
``judges/stub.py`` implements these computations deterministically, and the
hand-labeling guide in ``human_label/LABELING_GUIDE.md`` uses the same anchor
definitions so human and judge scores are comparable.

Citation syntax (shared with the ecosystem fixtures): a citation is a bracket
token ``[chunk_id]`` that must match a ``chunk_id`` in the sample's contexts.
The sentence carrying the citation is the "claim"; it is checked against the
cited chunk's text.
"""

from __future__ import annotations

CRITERIA = [
    "citation_precision",
    "citation_recall",
    "faithfulness",
    "answer_f1",
    "conciseness",
]


def scale_1_5(fraction: float) -> int:
    """Map a 0..1 fraction to an anchored 1-5 ordinal score."""
    if fraction >= 0.95:
        return 5
    if fraction >= 0.80:
        return 4
    if fraction >= 0.60:
        return 3
    if fraction >= 0.40:
        return 2
    return 1


def conciseness_score(output_words: int, expected_words: int) -> tuple[int, dict]:
    """Exact conciseness: penalize outputs much longer than the reference.

    ratio = output_words / max(expected_words, 1). Scores:
      r <= 1.5 -> 5 ; <= 2.0 -> 4 ; <= 3.0 -> 3 ; <= 5.0 -> 2 ; else 1.
    """
    ratio = output_words / max(expected_words, 1)
    if ratio <= 1.5:
        s = 5
    elif ratio <= 2.0:
        s = 4
    elif ratio <= 3.0:
        s = 3
    elif ratio <= 5.0:
        s = 2
    else:
        s = 1
    return s, {
        "output_words": output_words,
        "expected_words": expected_words,
        "ratio": round(ratio, 3),
    }


RUBRIC_TEXT = """
# Evaluation rubric (anchored 1-5, higher is better)

## 1. citation_precision
Of the citations in the output, what fraction are *valid*?
Valid = the [chunk_id] exists in the provided contexts AND the sentence
carrying it shares >=50% token overlap with the cited chunk text (the claim
is actually supported by what was cited).
Score: 5 (>=0.95) 4 (>=0.80) 3 (>=0.60) 2 (>=0.40) 1 (<0.40).
An output with factual claims and zero citations scores 1.

## 2. citation_recall
Of the key claims in the expected answer, what fraction appear in the output
*with a valid citation*? Claim matching is case-insensitive token overlap:
an expected claim counts as covered if some output sentence shares >=60% of
the claim's content tokens.
Score thresholds as above.

## 3. faithfulness
Of the output's factual sentences, what fraction are supported by at least
one context chunk (>=50% token overlap with that chunk)? Sentences that are
pure boilerplate ("Here is the answer:") are excluded.
Score thresholds as above.

## 4. answer_f1
Token-level F1 of the output against the expected answer (citations stripped).
Score: 5 (>=0.80) 4 (>=0.60) 3 (>=0.40) 2 (>=0.20) 1 (<0.20).

## 5. conciseness
Does the output answer without padding? Exact: ratio = output words /
expected words; 5 (<=1.5) 4 (<=2.0) 3 (<=3.0) 2 (<=5.0) 1 (>5.0).
"""
