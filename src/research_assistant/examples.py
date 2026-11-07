"""Example research questions shipped with the demo."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Example:
    id: str
    question: str
    hint: str
    needs_web: bool = False


EXAMPLES: list[Example] = [
    Example(
        id="gpai-obligations",
        question=(
            "What transparency and documentation obligations does the EU AI Act place on providers "
            "of general-purpose AI models?"
        ),
        hint="Answerable entirely from the local EU AI Act (Articles 53 and 55, Annexes XI-XII).",
    ),
    Example(
        id="rmf-vs-act-risk",
        question=(
            "How does the EU AI Act's risk-based approach compare with the risk management approach "
            "in the NIST AI Risk Management Framework?"
        ),
        hint="Cross-document synthesis over the EU AI Act and NIST AI RMF 1.0.",
    ),
    Example(
        id="genai-risks",
        question=(
            "What risks are specific to generative AI according to NIST, and which EU AI Act "
            "provisions are most relevant to them?"
        ),
        hint="Combines the NIST Generative AI Profile with the EU AI Act's GPAI provisions.",
    ),
    Example(
        id="act-timeline",
        question="When do the EU AI Act's obligations for general-purpose AI models start to apply?",
        hint="The local text gives application dates; live web search can confirm current guidance.",
        needs_web=True,
    ),
]

EXAMPLES_BY_ID = {e.id: e for e in EXAMPLES}
