"""Exact, deterministic metric computations used by the stub judge.

All functions are pure and unit-tested. Tokenisation is deliberately simple
(lowercase alphanumeric tokens) so the computations are inspectable and
reproducible — the point of the stub judge is a *measured* pipeline, not SOTA
semantics. The LLM judge path (judges/openai_compat.py) uses the same
criterion definitions with model reasoning.
"""
from __future__ import annotations

import re

TOKEN_RE = re.compile(r"[a-z0-9]+")
CITATION_RE = re.compile(r"\[([A-Za-z0-9_\-./]+)\]")
SENTENCE_RE = re.compile(r"[^.!?]+[.!?]")
_DEC_DOT = "\ue000"  # placeholder so "4.82" is not treated as a sentence boundary


def _protect_decimals(text: str) -> str:
    return re.sub(r"(?<=\d)\.(?=\d)", _DEC_DOT, text)


def _restore_decimals(text: str) -> str:
    return text.replace(_DEC_DOT, ".")


def _raw_sentences(text: str) -> list[str]:
    """Split into sentences without stripping citations; decimals protected.

    A citation marker written after the sentence terminator ("... 2019. [c1]")
    is pulled inside the terminator ("... 2019[c1].") so it stays attached to
    the claim it was meant to support.
    """
    text = _protect_decimals(text).strip()
    text = re.sub(r"([.!?])\s*(\[[A-Za-z0-9_\-./]+\])", r"\2\1", text)
    if not text:
        return []
    parts = SENTENCE_RE.findall(text)
    rest = SENTENCE_RE.sub("", text).strip()
    out = [_restore_decimals(p).strip() for p in parts if p.strip()]
    if rest:
        out.append(_restore_decimals(rest).strip())
    return out

BOILERPLATE_PATTERNS = (
    "here is",
    "here's",
    "in summary",
    "to summarize",
    "based on the provided",
    "according to the provided",
)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def token_set(text: str) -> set[str]:
    return set(tokenize(text))


def strip_citations(text: str) -> str:
    return CITATION_RE.sub("", text)


def sentences(text: str) -> list[str]:
    return [s for s in (strip_citations(s).strip() for s in _raw_sentences(text)) if s]


def is_boilerplate(sentence: str) -> bool:
    low = sentence.lower()
    return any(p in low for p in BOILERPLATE_PATTERNS) or len(tokenize(sentence)) < 4


def overlap_fraction(claim: str, evidence: str) -> float:
    """Fraction of the claim's tokens that appear in the evidence."""
    ct = token_set(claim)
    if not ct:
        return 0.0
    et = token_set(evidence)
    return len(ct & et) / len(ct)


def find_citations(output: str) -> list[tuple[str, str]]:
    """Return (chunk_id, claim_sentence) for each citation marker.

    The claim is the sentence carrying the marker, with citations stripped.
    A marker standing alone (e.g. "... 2019. [chunk] Next ...") attaches to
    the nearest preceding non-empty sentence — the claim it was meant to cite.
    """
    found: list[tuple[str, str]] = []
    pending: list[str] = []  # markers seen before any claim text
    last_claim = ""
    for rs in _raw_sentences(output):
        markers = CITATION_RE.findall(rs)
        claim = strip_citations(rs).strip()
        if claim:
            last_claim = claim
            for m in pending:
                found.append((m, last_claim))
            pending = []
        for m in markers:
            if claim:
                found.append((m, claim))
            elif last_claim:
                found.append((m, last_claim))
            else:
                pending.append(m)
    return found


def answer_f1(output: str, expected: str) -> float:
    """Token-level F1 of output vs expected (citations stripped)."""
    out = tokenize(strip_citations(output))
    exp = tokenize(expected)
    if not out or not exp:
        return 0.0
    from collections import Counter

    co, ce = Counter(out), Counter(exp)
    overlap = sum((co & ce).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(out)
    recall = overlap / len(exp)
    return 2 * precision * recall / (precision + recall)
