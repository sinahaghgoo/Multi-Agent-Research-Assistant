"""Deterministic post-processing of the writer's draft: citation validity checks.

Validity means "the cited ID refers to evidence recorded in this run". Whether the evidence
actually supports the statement is a separate question that this module does not answer.
"""

from __future__ import annotations

import re

from .schemas import CheckedFinding, CitationCheck, DraftFinding, Evidence

_INLINE = re.compile(r"\[([^\[\]]*\bE\d+\b[^\[\]]*)\]")
_ID = re.compile(r"\bE\d+\b", re.IGNORECASE)


def normalise_ids(raw_ids: list[str]) -> list[str]:
    ids: list[str] = []
    for raw in raw_ids:
        for match in _ID.findall(str(raw)):
            ident = match.upper()
            if ident not in ids:
                ids.append(ident)
    return ids


def inline_citations(text: str) -> list[str]:
    ids: list[str] = []
    for group in _INLINE.findall(text):
        ids.extend(i for i in normalise_ids([group]) if i not in ids)
    return ids


def check_findings(
    findings: list[DraftFinding], evidence: list[Evidence]
) -> list[CheckedFinding]:
    known = {e.id for e in evidence}
    checked = []
    for f in findings:
        ids = normalise_ids(f.evidence_ids) + [
            i for i in inline_citations(f.statement) if i not in f.evidence_ids
        ]
        valid = [i for i in dict.fromkeys(ids) if i in known]
        invalid = [i for i in dict.fromkeys(ids) if i not in known]
        checked.append(
            CheckedFinding(
                statement=f.statement.strip(),
                evidence_ids=valid,
                invalid_evidence_ids=invalid,
                cited=bool(valid),
            )
        )
    return checked


def citation_check(
    answer: str, findings: list[CheckedFinding], evidence: list[Evidence]
) -> CitationCheck:
    known = {e.id for e in evidence}
    answer_ids = inline_citations(answer)
    all_ids = list(answer_ids)
    invalid: list[str] = [i for i in answer_ids if i not in known]
    for f in findings:
        all_ids.extend(f.evidence_ids + f.invalid_evidence_ids)
        invalid.extend(f.invalid_evidence_ids)
    return CitationCheck(
        total_citations=len(all_ids),
        valid_citations=sum(1 for i in all_ids if i in known),
        invalid_ids=sorted(set(invalid), key=lambda s: int(s[1:])),
        uncited_findings=sum(1 for f in findings if not f.cited),
        answer_has_citations=any(i in known for i in answer_ids),
    )


def strip_invalid_inline(text: str, evidence: list[Evidence]) -> str:
    """Mark inline citations that do not resolve, so the rendered report cannot mislead."""
    known = {e.id for e in evidence}

    def repl(match: re.Match) -> str:
        ids = normalise_ids([match.group(1)])
        parts = [i if i in known else f"{i}?" for i in ids]
        return "[" + ", ".join(parts) + "]"

    return _INLINE.sub(repl, text)
