"""Build the hand-calibration dataset: 10 scenarios x 3 quality variants = 30 items.

Corpus: "Meridian Dynamics Inc." — a fictional semiconductor company whose
FY2024 10-K-style disclosures are written to mirror real SEC filing language
(revenue, margins, segments, debt, R&D, capex, legal, dividend). The scenarios
match the question shapes the sibling knowledge-graph-rag project answers.

Quality tiers are DELIBERATE:
  good     — faithful, every claim cited with a valid chunk_id, concise
  mediocre — right facts, but missing citations / one wrong citation / verbose
  broken   — hallucinated numbers, citations to nonexistent chunks, contradictions

The human labels in human_label/seed_labels.jsonl were scored by the repo
author against the rubric (not copied from the judge) — see LABELING_GUIDE.md.
"""

from __future__ import annotations

import json
from pathlib import Path

CH = {
    "rev": "Meridian Dynamics reported revenue of $4.82 billion for fiscal 2024, up 18% year over year from $4.09 billion in fiscal 2023.",
    "margin": "Gross margin was 61.4% in fiscal 2024, compared with 58.9% in fiscal 2023, driven by a richer mix of data-center accelerators and lower memory costs.",
    "ceo": "Elena Marsh has served as Chief Executive Officer since March 2019. Prior to Meridian, she was COO of Halcyon Semiconductors.",
    "cust": "Two customers each accounted for more than 10% of revenue in fiscal 2024: Northwind Cloud (14%) and Vectra Systems (11%). No other customer exceeded 8%.",
    "seg": "By segment, Data Center revenue grew 34% to $2.91 billion, Client Computing grew 6% to $1.32 billion, and Embedded declined 4% to $0.59 billion in fiscal 2024.",
    "debt": "Long-term debt stood at $1.20 billion at year end, with $400 million maturing in 2027 and $800 million in 2030. Interest coverage was 11.2x.",
    "rd": "Research and development expense was $1.05 billion in fiscal 2024 (21.8% of revenue), up from $920 million in 2023 and $810 million in 2022.",
    "capex": "Capital expenditures totaled $620 million in fiscal 2024, primarily for the new Dresden fabrication facility, which is expected to begin production in the second half of 2026.",
    "legal": "The patent dispute with Corvus Labs was settled in October 2024 for $45 million with no admission of liability. No material litigation remained pending at year end.",
    "div": "The board raised the quarterly dividend to $0.28 per share in November 2024, the third consecutive annual increase, reflecting confidence in sustained cash generation.",
}

SID = {
    "rev": "md-fy24-revenue",
    "margin": "md-fy24-margin",
    "ceo": "md-fy24-leadership",
    "cust": "md-fy24-customers",
    "seg": "md-fy24-segments",
    "debt": "md-fy24-debt",
    "rd": "md-fy24-rd",
    "capex": "md-fy24-capex",
    "legal": "md-fy24-legal",
    "div": "md-fy24-dividend",
}


def ctx(*keys: str) -> list[dict]:
    return [
        {"text": CH[k], "source": "Meridian Dynamics FY2024 10-K (fictional)", "chunk_id": SID[k]}
        for k in keys
    ]


def item(
    sample_id: str,
    topic: str,
    input_q: str,
    keys: tuple[str, ...],
    expected: str,
    output: str,
    tier: str,
) -> dict:
    return {
        "sample_id": sample_id,
        "input": input_q,
        "output": output,
        "contexts": ctx(*keys),
        "expected": expected,
        "metadata": {"system": "fixture", "topic": topic, "tier": tier},
    }


ITEMS: list[dict] = []

# 1. revenue
exp = "Revenue was $4.82 billion in fiscal 2024. This was up 18% from $4.09 billion in fiscal 2023."
ITEMS += [
    item(
        "rev-good",
        "revenue",
        "What was Meridian's revenue in FY2024 and how did it change?",
        ("rev",),
        exp,
        "Meridian Dynamics reported revenue of $4.82 billion for fiscal 2024, up 18% year over year from $4.09 billion in fiscal 2023. [md-fy24-revenue]",
        "good",
    ),
    item(
        "rev-mediocre",
        "revenue",
        "What was Meridian's revenue in FY2024 and how did it change?",
        ("rev",),
        exp,
        "Revenue came in at $4.82 billion for fiscal 2024, which represents very strong 18% growth versus the prior year's $4.09 billion, continuing the company's impressive multi-year expansion trajectory across all of its end markets and geographies. [md-fy24-revenue]",
        "mediocre",
    ),
    item(
        "rev-broken",
        "revenue",
        "What was Meridian's revenue in FY2024 and how did it change?",
        ("rev",),
        exp,
        "Meridian posted record revenue of $5.60 billion in fiscal 2024, up 31% year over year. [md-fy24-earnings-call] Management guided to $7 billion for next year. [md-fy24-guidance]",
        "broken",
    ),
]

# 2. margin
exp = "Gross margin was 61.4% in fiscal 2024, up from 58.9% in fiscal 2023. The improvement was driven by data-center accelerator mix and lower memory costs."
ITEMS += [
    item(
        "margin-good",
        "margin",
        "What drove the change in gross margin?",
        ("margin",),
        exp,
        "Gross margin was 61.4% in fiscal 2024, compared with 58.9% in fiscal 2023, driven by a richer mix of data-center accelerators and lower memory costs. [md-fy24-margin]",
        "good",
    ),
    item(
        "margin-mediocre",
        "margin",
        "What drove the change in gross margin?",
        ("margin",),
        exp,
        "Gross margin expanded to 61.4% from 58.9% last year. [md-fy24-margin] The main drivers were the shift toward higher-margin data-center accelerators, lower memory input costs, and improved factory utilization across the Dresden and Austin sites.",
        "mediocre",
    ),
    item(
        "margin-broken",
        "margin",
        "What drove the change in gross margin?",
        ("margin",),
        exp,
        "Gross margin collapsed to 41.2% as memory costs surged and the company took a $200 million inventory writedown. [md-fy24-margin]",
        "broken",
    ),
]

# 3. CEO
exp = "Elena Marsh has been CEO since March 2019. She was previously COO of Halcyon Semiconductors."
ITEMS += [
    item(
        "ceo-good",
        "ceo",
        "Who is the CEO and how long have they been in the role?",
        ("ceo",),
        exp,
        "Elena Marsh has served as Chief Executive Officer since March 2019. [md-fy24-leadership] Prior to Meridian, she was COO of Halcyon Semiconductors. [md-fy24-leadership]",
        "good",
    ),
    item(
        "ceo-mediocre",
        "ceo",
        "Who is the CEO and how long have they been in the role?",
        ("ceo",),
        exp,
        "Elena Marsh has served as Chief Executive Officer since March 2019. Prior to Meridian, she was COO of Halcyon Semiconductors. [md-fy24-leadership]",
        "mediocre",
    ),
    item(
        "ceo-broken",
        "ceo",
        "Who is the CEO and how long have they been in the role?",
        ("ceo",),
        exp,
        "David Okafor became CEO in January 2024 after serving as CFO. [md-fy24-leadership]",
        "broken",
    ),
]

# 4. customer concentration
exp = "Northwind Cloud (14%) and Vectra Systems (11%) each exceeded 10% of revenue in fiscal 2024. No other customer exceeded 8%."
ITEMS += [
    item(
        "cust-good",
        "customers",
        "Is revenue concentrated in a few customers?",
        ("cust",),
        exp,
        "Yes. Two customers each accounted for more than 10% of revenue in fiscal 2024: Northwind Cloud (14%) and Vectra Systems (11%). [md-fy24-customers] No other customer exceeded 8%. [md-fy24-customers]",
        "good",
    ),
    item(
        "cust-mediocre",
        "customers",
        "Is revenue concentrated in a few customers?",
        ("cust",),
        exp,
        "Revenue is concentrated: Northwind Cloud was 14% and Vectra Systems 11% of fiscal 2024 revenue. [md-fy24-revenue] No other customer exceeded 8%.",
        "mediocre",
    ),
    item(
        "cust-broken",
        "customers",
        "Is revenue concentrated in a few customers?",
        ("cust",),
        exp,
        "No, the customer base is highly diversified: the largest customer was under 5% of revenue. [md-fy24-customers]",
        "broken",
    ),
]

# 5. segments (comparison)
exp = "Data Center grew fastest at 34% to $2.91 billion. Client Computing grew 6% to $1.32 billion. Embedded declined 4% to $0.59 billion."
ITEMS += [
    item(
        "seg-good",
        "segments",
        "Which segment grew fastest in FY2024?",
        ("seg",),
        exp,
        "Data Center grew fastest, up 34% to $2.91 billion. [md-fy24-segments] Client Computing grew 6% to $1.32 billion, and Embedded declined 4% to $0.59 billion. [md-fy24-segments]",
        "good",
    ),
    item(
        "seg-mediocre",
        "segments",
        "Which segment grew fastest in FY2024?",
        ("seg",),
        exp,
        "Data Center was the fastest grower at 34% growth to $2.91B, reflecting continued AI infrastructure demand, hyperscaler capex cycles, strong enterprise refresh activity, and share gains in networking silicon. [md-fy24-segments] Client Computing grew 6% to $1.32B while Embedded fell 4% to $0.59B. [md-fy24-segments]",
        "mediocre",
    ),
    item(
        "seg-broken",
        "segments",
        "Which segment grew fastest in FY2024?",
        ("seg",),
        exp,
        "Embedded grew fastest at 22% as automotive demand surged. [md-fy24-segments] Data Center was flat. [md-fy24-segments]",
        "broken",
    ),
]

# 6. debt (two-hop: maturity + coverage)
exp = "Long-term debt was $1.20 billion: $400 million maturing in 2027 and $800 million in 2030. Interest coverage was 11.2x."
ITEMS += [
    item(
        "debt-good",
        "debt",
        "Summarize the debt maturity profile and coverage.",
        ("debt",),
        exp,
        "Long-term debt stood at $1.20 billion at year end, with $400 million maturing in 2027 and $800 million in 2030. [md-fy24-debt] Interest coverage was 11.2x. [md-fy24-debt]",
        "good",
    ),
    item(
        "debt-mediocre",
        "debt",
        "Summarize the debt maturity profile and coverage.",
        ("debt",),
        exp,
        "Long-term debt stood at $1.20 billion at year end, with $400 million maturing in 2027 and $800 million in 2030. Interest coverage was a comfortable 11.2x, well above covenant thresholds.",
        "mediocre",
    ),
    item(
        "debt-broken",
        "debt",
        "Summarize the debt maturity profile and coverage.",
        ("debt",),
        exp,
        "The company carries $3.4 billion of debt with $1.1 billion due next year, and interest coverage has fallen to 2.1x, raising refinancing risk. [md-fy24-debt-note]",
        "broken",
    ),
]

# 7. R&D trend (aggregation over years)
exp = "R&D was $1.05 billion in 2024 (21.8% of revenue), up from $920 million in 2023 and $810 million in 2022."
ITEMS += [
    item(
        "rd-good",
        "rd",
        "How has R&D spending trended?",
        ("rd",),
        exp,
        "Research and development expense was $1.05 billion in fiscal 2024 (21.8% of revenue), up from $920 million in 2023 and $810 million in 2022. [md-fy24-rd]",
        "good",
    ),
    item(
        "rd-mediocre",
        "rd",
        "How has R&D spending trended?",
        ("rd",),
        exp,
        "R&D rose to $1.05 billion in fiscal 2024, or 21.8% of revenue, from $920 million in 2023. [md-fy24-rd] The 2022 figure was $810 million, showing consistent investment growth.",
        "mediocre",
    ),
    item(
        "rd-broken",
        "rd",
        "How has R&D spending trended?",
        ("rd",),
        exp,
        "R&D spending was cut 30% to $700 million as the company shifted to a fabless cost model. [md-fy24-rd]",
        "broken",
    ),
]

# 8. capex / fab
exp = "Capex was $620 million in fiscal 2024, mostly for the new Dresden fab, expected to begin production in H2 2026."
ITEMS += [
    item(
        "capex-good",
        "capex",
        "What is the company spending capex on?",
        ("capex",),
        exp,
        "Capital expenditures totaled $620 million in fiscal 2024, primarily for the new Dresden fabrication facility, which is expected to begin production in the second half of 2026. [md-fy24-capex]",
        "good",
    ),
    item(
        "capex-mediocre",
        "capex",
        "What is the company spending capex on?",
        ("capex",),
        exp,
        "Capex was $620 million, mainly the Dresden fab starting production in H2 2026. [md-fy24-capex] This continues a multi-year investment program that also upgraded test and packaging capacity in Austin and Penang, positioning the company for the next product cycle.",
        "mediocre",
    ),
    item(
        "capex-broken",
        "capex",
        "What is the company spending capex on?",
        ("capex",),
        exp,
        "The company slashed capex to $150 million and cancelled the Dresden fab project. [md-fy24-capex]",
        "broken",
    ),
]

# 9. legal
exp = "The Corvus Labs patent dispute was settled in October 2024 for $45 million with no admission of liability. No material litigation remained pending."
ITEMS += [
    item(
        "legal-good",
        "legal",
        "What is the status of the Corvus Labs litigation?",
        ("legal",),
        exp,
        "The patent dispute with Corvus Labs was settled in October 2024 for $45 million with no admission of liability. [md-fy24-legal] No material litigation remained pending at year end. [md-fy24-legal]",
        "good",
    ),
    item(
        "legal-mediocre",
        "legal",
        "What is the status of the Corvus Labs litigation?",
        ("legal",),
        exp,
        "The patent dispute with Corvus Labs was settled in October 2024 for $45 million with no admission of liability. No material litigation remained pending at year end.",
        "mediocre",
    ),
    item(
        "legal-broken",
        "legal",
        "What is the status of the Corvus Labs litigation?",
        ("legal",),
        exp,
        "Meridian lost the Corvus Labs case and was ordered to pay $380 million in damages; an appeal is pending. [md-fy24-legal]",
        "broken",
    ),
]

# 10. dividend
exp = "The quarterly dividend was raised to $0.28 per share in November 2024, the third consecutive annual increase."
ITEMS += [
    item(
        "div-good",
        "dividend",
        "Did the dividend change?",
        ("div",),
        exp,
        "The board raised the quarterly dividend to $0.28 per share in November 2024, the third consecutive annual increase, reflecting confidence in sustained cash generation. [md-fy24-dividend]",
        "good",
    ),
    item(
        "div-mediocre",
        "dividend",
        "Did the dividend change?",
        ("div",),
        exp,
        "Yes, the quarterly dividend went up to $0.28 per share in November 2024. [md-fy24-dividend] It was the third straight annual increase, and the payout ratio remains conservative at under 30% of free cash flow.",
        "mediocre",
    ),
    item(
        "div-broken",
        "dividend",
        "Did the dividend change?",
        ("div",),
        exp,
        "The dividend was suspended in November 2024 to preserve cash for the Dresden fab. [md-fy24-dividend]",
        "broken",
    ),
]


def main() -> None:
    dest = Path(__file__).resolve().parents[2] / "adapters" / "fixtures" / "calibration.jsonl"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w") as f:
        for it in ITEMS:
            f.write(json.dumps(it) + "\n")
    print(f"wrote {len(ITEMS)} items -> {dest}")


if __name__ == "__main__":
    main()
