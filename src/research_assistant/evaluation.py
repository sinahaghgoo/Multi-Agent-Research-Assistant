"""Evaluation of a produced :class:`Report` against a grounded expectation set.

Checks are split into two honestly-labelled groups:

* **Automated, deterministic** (no model judgment): citation validity, source recall,
  expected-finding coverage (keyword matching), and a lexical *grounding proxy* that
  measures word overlap between a finding and its cited evidence. The grounding proxy is
  a weak signal, not an entailment check.
* **Manual** (human judgment): whether a sampled claim is actually supported by its cited
  evidence. :func:`manual_review_rows` produces the rows a human fills in. The same model
  scoring its own output is *not* treated as independent validation.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .retrieval import tokenize
from .schemas import Report


@dataclass(frozen=True)
class ExpectedFinding:
    label: str
    all_of: tuple[str, ...] = ()
    any_of: tuple[str, ...] = ()

    def matches(self, text: str) -> bool:
        low = text.lower()
        if any(term.lower() not in low for term in self.all_of):
            return False
        if self.any_of and not any(term.lower() in low for term in self.any_of):
            return False
        return bool(self.all_of or self.any_of)


@dataclass(frozen=True)
class EvalQuestion:
    id: str
    question: str
    needs_web: bool
    expected_sources: tuple[str, ...]
    expected_locations: tuple[str, ...]
    expected_findings: tuple[ExpectedFinding, ...]


def load_eval_questions(path: Path) -> list[EvalQuestion]:
    data = json.loads(path.read_text(encoding="utf-8"))
    questions = []
    for q in data["questions"]:
        questions.append(
            EvalQuestion(
                id=q["id"],
                question=q["question"],
                needs_web=q.get("needs_web", False),
                expected_sources=tuple(q.get("expected_sources", [])),
                expected_locations=tuple(q.get("expected_locations", [])),
                expected_findings=tuple(
                    ExpectedFinding(f["label"], tuple(f.get("all_of", [])), tuple(f.get("any_of", [])))
                    for f in q.get("expected_findings", [])
                ),
            )
        )
    return questions


def _grounding(statement: str, evidence_text: str) -> float:
    """Fraction of a finding's content words that also appear in its cited evidence text.

    A deterministic lexical proxy for grounding. High overlap suggests the claim's terms
    come from the evidence; it does NOT prove the evidence entails the claim.
    """
    claim_terms = set(tokenize(statement))
    if not claim_terms:
        return 0.0
    evidence_terms = set(tokenize(evidence_text))
    return len(claim_terms & evidence_terms) / len(claim_terms)


@dataclass
class EvalResult:
    question_id: str
    mode: str
    status: str
    # Automated, deterministic
    citation_validity: float
    answer_has_citations: bool
    invalid_citation_ids: list[str]
    source_recall: float
    missing_sources: list[str]
    location_recall: float
    finding_coverage: float
    matched_findings: list[str]
    missing_findings: list[str]
    mean_grounding_proxy: float
    # Operational
    duration_s: float
    llm_calls: int
    tool_calls: int
    n_evidence: int
    n_sources: int
    stop_reason: str
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_report(report: Report, expected: EvalQuestion) -> EvalResult:
    cited_source_ids = {e.source_id for e in report.evidence if e.id in _cited_ids(report)}
    missing_sources = [s for s in expected.expected_sources if s not in cited_source_ids]
    source_recall = (
        1.0
        if not expected.expected_sources
        else (len(expected.expected_sources) - len(missing_sources)) / len(expected.expected_sources)
    )

    evidence_locations = " ".join(f"{e.source_id} {e.location}" for e in report.evidence).lower()
    matched_locations = [loc for loc in expected.expected_locations if loc.lower() in evidence_locations]
    location_recall = (
        1.0 if not expected.expected_locations else len(matched_locations) / len(expected.expected_locations)
    )

    report_text = _report_text(report)
    matched, missing = [], []
    for finding in expected.expected_findings:
        (matched if finding.matches(report_text) else missing).append(finding.label)
    coverage = len(matched) / len(expected.expected_findings) if expected.expected_findings else 0.0

    groundings = _finding_groundings(report)
    mean_grounding = sum(groundings) / len(groundings) if groundings else 0.0

    c = report.citation_check
    return EvalResult(
        question_id=expected.id,
        mode=report.run.mode,
        status=report.run.status,
        citation_validity=c.validity_rate,
        answer_has_citations=c.answer_has_citations,
        invalid_citation_ids=c.invalid_ids,
        source_recall=source_recall,
        missing_sources=missing_sources,
        location_recall=location_recall,
        finding_coverage=coverage,
        matched_findings=matched,
        missing_findings=missing,
        mean_grounding_proxy=round(mean_grounding, 3),
        duration_s=report.run.duration_s,
        llm_calls=report.run.llm_calls,
        tool_calls=report.run.tool_calls,
        n_evidence=len(report.evidence),
        n_sources=len(report.sources),
        stop_reason=report.run.stop_reason,
        errors=report.run.errors,
    )


def _cited_ids(report: Report) -> set[str]:
    ids: set[str] = set()
    for f in report.key_findings + report.conflicts:
        ids.update(f.evidence_ids)
    return ids


def _report_text(report: Report) -> str:
    parts = [report.answer]
    parts += [f.statement for f in report.key_findings + report.conflicts]
    return "\n".join(parts)


def _finding_groundings(report: Report) -> list[float]:
    quotes = {e.id: e.quote for e in report.evidence}
    scores = []
    for f in report.key_findings + report.conflicts:
        if f.evidence_ids:
            text = " ".join(quotes.get(i, "") for i in f.evidence_ids)
            scores.append(_grounding(f.statement, text))
    return scores


@dataclass
class ManualReviewRow:
    question_id: str
    finding: str
    evidence_ids: list[str]
    evidence_quotes: str
    grounding_proxy: float
    supported: str = ""  # filled by a human: "yes" / "partial" / "no"


def manual_review_rows(report: Report, question_id: str) -> list[ManualReviewRow]:
    """Rows for a human to mark claim support. This is the independent-of-model check."""
    quotes = {e.id: (e.location, e.quote) for e in report.evidence}
    rows = []
    for f in report.key_findings + report.conflicts:
        if not f.evidence_ids:
            continue
        joined = " | ".join(
            f"[{i}] {quotes.get(i, ('', ''))[0]}: {quotes.get(i, ('', ''))[1][:300]}" for i in f.evidence_ids
        )
        rows.append(
            ManualReviewRow(
                question_id=question_id,
                finding=f.statement,
                evidence_ids=list(f.evidence_ids),
                evidence_quotes=joined,
                grounding_proxy=round(_grounding(f.statement, re.sub(r"\[E\d+\][^:]*:", "", joined)), 3),
            )
        )
    return rows


def aggregate(results: list[EvalResult]) -> dict:
    if not results:
        return {}

    def mean(attr: str) -> float:
        return round(sum(getattr(r, attr) for r in results) / len(results), 3)

    return {
        "n": len(results),
        "completion_rate": round(sum(r.status == "completed" for r in results) / len(results), 3),
        "mean_citation_validity": mean("citation_validity"),
        "answer_cited_rate": round(sum(r.answer_has_citations for r in results) / len(results), 3),
        "mean_source_recall": mean("source_recall"),
        "mean_finding_coverage": mean("finding_coverage"),
        "mean_grounding_proxy": mean("mean_grounding_proxy"),
        "mean_duration_s": mean("duration_s"),
        "mean_llm_calls": mean("llm_calls"),
        "mean_tool_calls": mean("tool_calls"),
    }
