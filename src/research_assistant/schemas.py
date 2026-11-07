"""Data records that flow through the workflow.

Identifiers are the backbone of source tracking:

* ``Source.id``   – ``eu-ai-act`` for local documents, ``web-1``, ``web-2`` ... for fetched pages.
* ``Passage.id``  – ``<source_id>:<section>:<n>``, e.g. ``eu-ai-act:art-51:1``. Stable for the
  local corpus because passages are committed to the repository.
* ``Evidence.id`` – ``E1``, ``E2`` ... assigned in the order the analyst records evidence.

Evidence quotes are always copied from the stored passage text, never from model output.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Source(BaseModel):
    id: str
    kind: Literal["local", "web"]
    title: str
    url: str | None = None
    publisher: str | None = None
    published: str | None = None
    retrieved: str | None = None
    license: str | None = None
    flags: list[str] = Field(default_factory=list)


class Passage(BaseModel):
    id: str
    source_id: str
    location: str
    text: str


class Evidence(BaseModel):
    id: str
    passage_id: str
    source_id: str
    location: str
    quote: str
    note: str = Field(description="The analyst model's note on why the passage matters (model-written).")
    recorded_by: str


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str = ""


class Decision(BaseModel):
    """One supervisor routing decision, as chosen by the model or imposed by a guard."""

    turn: int
    action: Literal["web_research", "document_analysis", "finish"]
    task: str = ""
    rationale: str = ""
    decided_by: Literal["model", "guard", "fallback"] = "model"


class WorkerReport(BaseModel):
    agent: str
    task: str
    summary: str
    tool_calls: int
    stop_reason: str
    new_source_ids: list[str] = Field(default_factory=list)
    new_evidence_ids: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Report schema. ``ReportDraft`` is what the writer model must return (JSON schema
# constrained decoding); ``Report`` is the validated, deterministic result.
# ---------------------------------------------------------------------------


class DraftFinding(BaseModel):
    statement: str = Field(description="One specific claim that answers part of the question.")
    evidence_ids: list[str] = Field(
        description="IDs (e.g. 'E3') of evidence items whose text directly supports the statement."
    )


class ReportDraft(BaseModel):
    answer: str = Field(
        description="A direct answer in 2-4 sentences. Cite evidence inline like [E1] or [E2, E5]."
    )
    key_findings: list[DraftFinding] = Field(description="3-6 key findings, each with evidence IDs.")
    conflicts: list[DraftFinding] = Field(
        default_factory=list,
        description="Points where evidence items disagree or may be outdated. Empty if none.",
    )
    limitations: list[str] = Field(
        default_factory=list,
        description="What the collected evidence does not establish or cover.",
    )
    open_questions: list[str] = Field(
        default_factory=list, description="Follow-up questions the evidence leaves unanswered."
    )


class CheckedFinding(BaseModel):
    statement: str
    evidence_ids: list[str]
    invalid_evidence_ids: list[str] = Field(default_factory=list)
    cited: bool


class CitationCheck(BaseModel):
    """Automated citation *validity* check. It does not verify that evidence supports claims."""

    total_citations: int
    valid_citations: int
    invalid_ids: list[str]
    uncited_findings: int
    answer_has_citations: bool

    @property
    def validity_rate(self) -> float:
        return self.valid_citations / self.total_citations if self.total_citations else 0.0


class RunInfo(BaseModel):
    mode: Literal["multi_agent", "single_agent"]
    model: str
    embedding_model: str | None
    web_enabled: bool
    started_at: str
    duration_s: float
    llm_calls: int
    tool_calls: int
    status: Literal["completed", "partial", "failed"]
    stop_reason: str
    errors: list[str] = Field(default_factory=list)


class Report(BaseModel):
    question: str
    answer: str
    key_findings: list[CheckedFinding]
    conflicts: list[CheckedFinding]
    limitations: list[str]
    open_questions: list[str]
    evidence: list[Evidence]
    sources: list[Source]
    citation_check: CitationCheck
    run: RunInfo
