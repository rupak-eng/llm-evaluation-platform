"""Bias tests for the judge. All three are exact measurements:

1. position_bias — pairwise comparison run in both orders; the flip rate is
   the fraction of pairs whose verdict changes when A/B are swapped.
2. length_bias — Spearman correlation between output length (words) and the
   judge's overall score on the calibration set. Nonzero correlation with
   quality held uncontrolled is reported as-is (see README for the caveat).
3. self_preference — two stub judges with different "identities" score two
   output sets written in each identity's style; the preference delta is
   the mean score difference. With deterministic stub judges this is an
   exact pipeline test; with LLM judges it measures real self-preference.
"""

from __future__ import annotations

import copy
import random
from collections.abc import Callable

from . import metrics as M
from .agreement import spearman
from .rubric import CRITERIA
from .schemas import EvalSample, JudgeOutput


def pairwise_verdict(
    judge: Callable[[EvalSample], JudgeOutput], a: EvalSample, b: EvalSample
) -> str:
    """Which output wins: 'a', 'b', or 'tie', by mean criterion score."""
    sa = judge(a)
    sb = judge(b)
    ma = sum(s.score for s in sa.scores) / len(sa.scores)
    mb = sum(s.score for s in sb.scores) / len(sb.scores)
    if abs(ma - mb) < 1e-9:
        return "tie"
    return "a" if ma > mb else "b"


def position_bias(
    judge: Callable[[EvalSample], JudgeOutput], pairs: list[tuple[EvalSample, EvalSample]]
) -> dict:
    """Flip rate when candidate order is swapped. Deterministic judges score 0
    by construction (order-independent); LLM judges reveal real position bias."""
    flips = 0
    details = []
    for a, b in pairs:
        v1 = pairwise_verdict(judge, a, b)
        # swap: present b first, a second
        a2, b2 = copy.deepcopy(b), copy.deepcopy(a)
        v2 = pairwise_verdict(judge, a2, b2)
        # map v2 back to original labeling: in swapped run "a" means original b
        v2_mapped = {"a": "b", "b": "a", "tie": "tie"}[v2]
        flipped = v1 != v2_mapped
        flips += int(flipped)
        details.append(
            {
                "a": a.sample_id,
                "b": b.sample_id,
                "verdict_order1": v1,
                "verdict_order2_mapped": v2_mapped,
                "flipped": flipped,
            }
        )
    return {
        "test": "position_bias",
        "metric": "flip_rate",
        "value": flips / len(pairs) if pairs else 0.0,
        "detail": {"n_pairs": len(pairs), "n_flips": flips, "pairs": details},
    }


def length_bias(judge: Callable[[EvalSample], JudgeOutput], samples: list[EvalSample]) -> dict:
    """Spearman rho between output word count and mean judge score."""
    lengths, scores = [], []
    for s in samples:
        out = judge(s)
        mean_score = sum(x.score for x in out.scores) / len(out.scores)
        lengths.append(float(len(M.tokenize(M.strip_citations(s.output)))))
        scores.append(mean_score)
    rho, p = spearman(lengths, scores)
    return {
        "test": "length_bias",
        "metric": "spearman_rho_length_vs_score",
        "value": rho,
        "detail": {"n": len(samples), "p_value": p},
    }


class StyledStubJudge:
    """Stub judge with an identity label and a style penalty.

    Used for the self-preference pipeline test: judge A penalizes outputs
    tagged with style B and vice versa, so a nonzero preference delta proves
    the test harness can detect self-preference when it exists.
    """

    def __init__(self, identity: str, penalty: float = 0.6):
        from .judges.stub import StubJudge

        self.identity = identity
        self.penalty = penalty
        self._base = StubJudge()
        self.name = f"styled-stub:{identity}"

    def judge(self, sample: EvalSample) -> JudgeOutput:
        out = self._base.judge(sample)
        style = (sample.metadata or {}).get("style", "")
        if style and style != self.identity:
            # penalize the "other" style: drop each criterion by 1 (floor 1)
            for s in out.scores:
                if s.criterion in ("conciseness", "answer_f1"):
                    s.score = max(1, s.score - 1)
                    s.rationale += f" [style penalty: {self.identity} dis prefers {style}]"
        return out


def self_preference(
    judge_a: Callable[[EvalSample], JudgeOutput],
    judge_b: Callable[[EvalSample], JudgeOutput],
    set_a: list[EvalSample],
    set_b: list[EvalSample],
) -> dict:
    """Each judge scores both sets; preference = (own-style mean) - (other mean).

    Positive delta for a judge means it scores its own style higher.
    """

    def mean(judge, samples):
        tot, n = 0, 0
        for s in samples:
            out = judge(s)
            tot += sum(x.score for x in out.scores) / len(out.scores)
            n += 1
        return tot / n if n else 0.0

    pref_a = mean(judge_a, set_a) - mean(judge_a, set_b)
    pref_b = mean(judge_b, set_b) - mean(judge_b, set_a)
    return {
        "test": "self_preference",
        "metric": "mean_preference_delta",
        "value": (pref_a + pref_b) / 2,
        "detail": {
            "judge_a": getattr(judge_a, "name", "?"),
            "judge_b": getattr(judge_b, "name", "?"),
            "pref_a": round(pref_a, 3),
            "pref_b": round(pref_b, 3),
            "n_a": len(set_a),
            "n_b": len(set_b),
        },
    }


def make_pairs(
    samples: list[EvalSample], seed: int = 42, n_pairs: int = 10
) -> list[tuple[EvalSample, EvalSample]]:
    rng = random.Random(seed)
    idx = list(range(len(samples)))
    rng.shuffle(idx)
    pairs = []
    for i in range(0, min(len(idx) - 1, n_pairs * 2), 2):
        pairs.append((samples[idx[i]], samples[idx[i + 1]]))
    return pairs


CRITERIA_EXPORT = CRITERIA
