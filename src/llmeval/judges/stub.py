"""Deterministic rule-based judge implementing the rubric exactly.

No LLM calls, no randomness: given the same sample it always returns the
same scores. This is a *feature* for this platform — it makes the agreement
numbers, bias tests, and regression gate real measurements of the pipeline
rather than noise from an unseeded model. The production path swaps in
``openai_compat.py`` / ``deepeval_adapter.py`` behind the same interface.

Scoring computations (see rubric.py for anchor definitions):
- citation_precision: fraction of citations that are valid
  (chunk_id exists AND claim sentence shares >=50% token overlap with chunk)
- citation_recall: fraction of expected claims covered by a validly-cited
  output sentence (>=60% token overlap of the claim)
- faithfulness: fraction of non-boilerplate output sentences supported by
  some context chunk (>=50% token overlap)
- answer_f1: token F1 vs expected, mapped to 1-5
- conciseness: exact length-ratio scoring
"""

from __future__ import annotations

import time

from .. import metrics as M
from ..rubric import CRITERIA, conciseness_score, scale_1_5
from ..schemas import CriterionScore, EvalSample, JudgeOutput

CLAIM_SUPPORT_THRESHOLD = 0.50
EXPECTED_COVERAGE_THRESHOLD = 0.60


def _expected_claims(expected: str) -> list[str]:
    return [s for s in M.sentences(expected) if not M.is_boilerplate(s)]


class StubJudge:
    """Rule-based judge. Deterministic."""

    name = "stub"

    def judge(self, sample: EvalSample) -> JudgeOutput:
        t0 = time.perf_counter()
        errors: list[str] = []
        chunks = {c.chunk_id: c.text for c in sample.contexts}
        citations = M.find_citations(sample.output)
        citations_checked: list[dict] = []

        # --- citation precision ---
        valid = 0
        for chunk_id, claim in citations:
            chunk_text = chunks.get(chunk_id)
            if chunk_text is None:
                citations_checked.append(
                    {"chunk_id": chunk_id, "valid": False, "reason": "unknown chunk_id"}
                )
                continue
            support = M.overlap_fraction(claim, chunk_text)
            ok = support >= CLAIM_SUPPORT_THRESHOLD
            valid += int(ok)
            citations_checked.append(
                {
                    "chunk_id": chunk_id,
                    "valid": ok,
                    "support": round(support, 3),
                    "reason": None if ok else "claim not supported by cited chunk",
                }
            )
        precision_frac = valid / len(citations) if citations else 0.0
        factual_sents = [s for s in M.sentences(sample.output) if not M.is_boilerplate(s)]
        if factual_sents and not citations:
            precision_note = "factual claims present but no citations"
        else:
            precision_note = f"{valid}/{len(citations)} citations valid"

        # --- citation recall ---
        claims = _expected_claims(sample.expected)
        covered = 0
        valid_cited_claims = [
            claim
            for (cid, claim), chk in zip(citations, citations_checked, strict=True)
            if chk["valid"]
        ]
        for exp_claim in claims:
            if any(
                M.overlap_fraction(exp_claim, cited) >= EXPECTED_COVERAGE_THRESHOLD
                for cited in valid_cited_claims
            ):
                covered += 1
        recall_frac = covered / len(claims) if claims else 1.0

        # --- faithfulness ---
        supported = 0
        for sent in factual_sents:
            if any(
                M.overlap_fraction(sent, c.text) >= CLAIM_SUPPORT_THRESHOLD for c in sample.contexts
            ):
                supported += 1
        faith_frac = supported / len(factual_sents) if factual_sents else 1.0

        # --- answer F1 ---
        f1 = M.answer_f1(sample.output, sample.expected)

        # --- conciseness ---
        out_words = len(M.tokenize(M.strip_citations(sample.output)))
        exp_words = len(M.tokenize(sample.expected))

        scores = [
            CriterionScore(
                criterion="citation_precision",
                score=scale_1_5(precision_frac),
                rationale=precision_note,
                measured={"fraction": round(precision_frac, 3), "n_citations": len(citations)},
            ),
            CriterionScore(
                criterion="citation_recall",
                score=scale_1_5(recall_frac),
                rationale=f"{covered}/{len(claims)} expected claims covered with valid citations",
                measured={"fraction": round(recall_frac, 3), "n_expected_claims": len(claims)},
            ),
            CriterionScore(
                criterion="faithfulness",
                score=scale_1_5(faith_frac),
                rationale=(
                    f"{supported}/{len(factual_sents)} factual sentences supported by contexts"
                ),
                measured={"fraction": round(faith_frac, 3)},
            ),
            CriterionScore(
                criterion="answer_f1",
                score=scale_1_5(f1) if f1 < 0.95 else 5,
                rationale=f"token F1 vs expected = {f1:.3f}",
                measured={"f1": round(f1, 3)},
            ),
        ]
        conc_s, conc_m = conciseness_score(out_words, exp_words)
        scores.append(
            CriterionScore(
                criterion="conciseness",
                score=conc_s,
                rationale=f"output/expected word ratio = {conc_m['ratio']}",
                measured=conc_m,
            )
        )
        assert [s.criterion for s in scores] == CRITERIA
        return JudgeOutput(
            sample_id=sample.sample_id,
            judge_name=self.name,
            scores=scores,
            citations_checked=citations_checked,
            errors=errors,
            latency_s=time.perf_counter() - t0,
        )
