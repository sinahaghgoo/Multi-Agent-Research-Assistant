from __future__ import annotations

from research_assistant.report import report_to_markdown
from research_assistant.schemas import (
    CheckedFinding,
    CitationCheck,
    DraftFinding,
    Evidence,
    Report,
    RunInfo,
    Source,
)
from research_assistant.validation import (
    check_findings,
    citation_check,
    inline_citations,
    normalise_ids,
    strip_invalid_inline,
)


def _evidence(ids):
    return [
        Evidence(
            id=i,
            passage_id=f"p:{i}",
            source_id="doc-a",
            location="Art 1",
            quote="text",
            note="n",
            recorded_by="a",
        )
        for i in ids
    ]


def test_normalise_ids_dedupes_and_uppercases():
    assert normalise_ids(["e1", "E1", "E2, E3"]) == ["E1", "E2", "E3"]


def test_inline_citations_extracts_bracketed_ids():
    assert inline_citations("A claim [E1] and another [E2, E3].") == ["E1", "E2", "E3"]
    assert inline_citations("No citations here.") == []


def test_check_findings_separates_valid_and_invalid():
    evidence = _evidence(["E1", "E2"])
    findings = [
        DraftFinding(statement="valid", evidence_ids=["E1"]),
        DraftFinding(statement="hallucinated", evidence_ids=["E9"]),
        DraftFinding(statement="mixed [E2]", evidence_ids=["E2", "E7"]),
    ]
    checked = check_findings(findings, evidence)
    assert checked[0].evidence_ids == ["E1"] and checked[0].cited
    assert checked[1].evidence_ids == [] and not checked[1].cited
    assert checked[1].invalid_evidence_ids == ["E9"]
    assert checked[2].evidence_ids == ["E2"] and checked[2].invalid_evidence_ids == ["E7"]


def test_citation_check_counts_validity():
    evidence = _evidence(["E1", "E2"])
    findings = check_findings(
        [
            DraftFinding(statement="a", evidence_ids=["E1"]),
            DraftFinding(statement="b", evidence_ids=["E9"]),
        ],
        evidence,
    )
    check = citation_check("Answer with [E1] and [E9].", findings, evidence)
    assert check.total_citations == 4  # E1, E9 inline + E1, E9 in findings
    assert check.valid_citations == 2
    assert "E9" in check.invalid_ids
    assert check.uncited_findings == 1
    assert check.answer_has_citations


def test_strip_invalid_inline_marks_unknown_ids():
    evidence = _evidence(["E1"])
    out = strip_invalid_inline("Supported [E1] but not [E5].", evidence)
    assert "[E1]" in out
    assert "[E5?]" in out


def test_report_to_markdown_contains_sections():
    report = Report(
        question="Q?",
        answer="Answer [E1].",
        key_findings=[CheckedFinding(statement="Finding", evidence_ids=["E1"], cited=True)],
        conflicts=[],
        limitations=["A limitation."],
        open_questions=["An open question?"],
        evidence=_evidence(["E1"]),
        sources=[Source(id="doc-a", kind="local", title="Example Reg", published="2024-07-12")],
        citation_check=CitationCheck(
            total_citations=2,
            valid_citations=2,
            invalid_ids=[],
            uncited_findings=0,
            answer_has_citations=True,
        ),
        run=RunInfo(
            mode="multi_agent",
            model="qwen3.5:9b",
            embedding_model="qwen3-embedding:0.6b",
            web_enabled=False,
            started_at="2026-11-07T10:00:00",
            duration_s=12.0,
            llm_calls=5,
            tool_calls=3,
            status="completed",
            stop_reason="supervisor_finished",
        ),
    )
    md = report_to_markdown(report)
    for heading in [
        "## Answer",
        "## Key findings",
        "## Evidence",
        "## Sources",
        "## Citation check",
    ]:
        assert heading in md
    assert "[E1]" in md
    assert "Example Reg" in md
