"""LangGraph workflow: a supervisor routes between two worker agents, then a writer and a
deterministic finaliser produce the report.

    START -> supervisor -(web_research)-------> web_researcher ----> supervisor
                        -(document_analysis)--> document_analyst --> supervisor
                        -(finish)-------------> writer -> finalize -> END

Model decisions: which step to take next and the task text (supervisor tool call); which tools
to call with which arguments (workers); the report wording and which evidence IDs to cite
(writer). Deterministic code: routing on the chosen tool, all budgets and guards, tool
execution, ID assignment, evidence quotes, citation validation and Markdown rendering.
"""

from __future__ import annotations

import operator
import time
from datetime import UTC, datetime
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field

from . import prompts
from .agents import LLMError, call_llm, run_tool_loop
from .schemas import (
    Decision,
    DraftFinding,
    Evidence,
    Report,
    ReportDraft,
    RunInfo,
    Source,
    WorkerReport,
)
from .text import truncate
from .tools import ToolSession, Workspace, analysis_tools, web_tools
from .validation import check_findings, citation_check, strip_invalid_inline

MAX_WEB_ROUNDS = 2
MAX_EMPTY_ANALYSIS_RETRIES = 2  # force re-analysis if a round records no evidence


def merge_dicts(left: dict, right: dict) -> dict:
    return {**left, **right}


class ResearchState(TypedDict, total=False):
    question: str
    mode: Literal["multi_agent", "single_agent"]
    started_at: str
    # Evidence records (canonical)
    sources: Annotated[dict[str, Source], merge_dicts]
    evidence: Annotated[list[Evidence], operator.add]
    allowed_urls: Annotated[list[str], operator.add]
    seen_passage_ids: Annotated[list[str], operator.add]
    fetch_count: int
    # Control flow
    turn: int
    web_rounds: int
    analysis_rounds: int
    pending_web_sources: list[str]
    decisions: Annotated[list[Decision], operator.add]
    worker_reports: Annotated[list[WorkerReport], operator.add]
    next_action: str
    next_task: str
    stop_reason: str
    errors: Annotated[list[str], operator.add]
    # Output
    draft: ReportDraft | None
    writer_status: str
    report: Report | None


# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------


class delegate_web_research(BaseModel):  # noqa: N801 - tool names are snake_case
    """Ask the web researcher to find and fetch public web sources."""

    task: str = Field(description="What to look for on the web and why.")


class delegate_document_analysis(BaseModel):  # noqa: N801
    """Ask the document analyst to search the documents and record evidence passages."""

    task: str = Field(description="What evidence to look for, and in which sources if relevant.")


class finish_research(BaseModel):  # noqa: N801
    """Stop researching and write the report from the recorded evidence."""

    reason: str = Field(
        description="Why the recorded evidence is sufficient, or why more steps would not help."
    )


_TOOL_TO_ACTION = {
    "delegate_web_research": "web_research",
    "delegate_document_analysis": "document_analysis",
    "finish_research": "finish",
}


def _web_available(state: ResearchState, ws: Workspace) -> bool:
    return ws.search is not None and state.get("web_rounds", 0) < MAX_WEB_ROUNDS


def render_status(state: ResearchState, ws: Workspace) -> str:
    s = ws.settings
    turn = state.get("turn", 0) + 1
    lines = [
        f"Research question: {state['question']}",
        "",
        f"Turn {turn} of {s.max_supervisor_turns}.",
    ]
    if ws.search is None:
        lines.append("Web research: disabled for this run.")
    else:
        lines.append(
            f"Web research: {'available' if _web_available(state, ws) else 'budget used up'} "
            f"({state.get('web_rounds', 0)} of {MAX_WEB_ROUNDS} rounds used)."
        )
    sources = state.get("sources", {})
    local = [
        f"{sid} ({ws.index.source_titles.get(sid, src.title)})"
        for sid, src in sources.items()
        if src.kind == "local"
    ]
    web = [f"{sid}: {src.title} <{src.url}>" for sid, src in sources.items() if src.kind == "web"]
    lines.append("Local documents: " + "; ".join(local))
    lines.append("Fetched web sources: " + ("; ".join(web) if web else "none"))
    pending = state.get("pending_web_sources", [])
    if pending:
        lines.append(f"Fetched but not yet analysed: {', '.join(pending)}")
    evidence = state.get("evidence", [])
    lines.append(f"\nRecorded evidence ({len(evidence)} items):")
    for e in evidence:
        lines.append(f"- {e.id} [{e.source_id}, {e.location}] {truncate(e.note, 160)}")
    if not evidence:
        lines.append("- none yet")
    reports = state.get("worker_reports", [])
    if reports:
        lines.append("\nPrevious steps:")
        for i, r in enumerate(reports, start=1):
            lines.append(
                f"{i}. {r.agent} — task: {truncate(r.task, 140)}\n   result: {truncate(r.summary, 300)}"
            )
    lines.append("\nCall exactly one tool to choose the next step.")
    return "\n".join(lines)


def _decision(state: ResearchState, action: str, task: str, rationale: str, by: str) -> dict[str, Any]:
    turn = state.get("turn", 0) + 1
    return {
        "decisions": [Decision(turn=turn, action=action, task=task, rationale=rationale, decided_by=by)],
        "next_action": action,
        "next_task": task,
        "turn": turn,
    }


def supervisor_node(state: ResearchState, runtime: Runtime[Workspace]) -> dict[str, Any]:
    ws = runtime.context
    s = ws.settings
    tracer = ws.tracer
    question = state["question"]
    evidence = state.get("evidence", [])
    pending = state.get("pending_web_sources", [])

    def guard(action: str, task: str, reason: str, stop_reason: str | None = None) -> dict[str, Any]:
        tracer.emit("supervisor", "guard", action, summary=reason)
        update = _decision(state, action, task, reason, "guard")
        if stop_reason:
            update["stop_reason"] = stop_reason
        return update

    # Hard stopping conditions, checked before asking the model.
    if ws.time_left() < s.writer_reserve_s:
        return guard("finish", "", "time budget nearly used; reserving time for the writer", "deadline")
    if state.get("turn", 0) >= s.max_supervisor_turns:
        return guard("finish", "", f"reached {s.max_supervisor_turns} supervisor turns", "turn_limit")
    if len(evidence) >= s.max_evidence:
        return guard("finish", "", "evidence limit reached", "evidence_limit")

    tools: list[type[BaseModel]] = [delegate_document_analysis, finish_research]
    if _web_available(state, ws):
        tools.insert(0, delegate_web_research)
    bound = ws.llm.bind_tools(tools)
    messages: list[BaseMessage] = [
        SystemMessage(prompts.SUPERVISOR),
        HumanMessage(render_status(state, ws)),
    ]

    call = None
    for attempt in range(2):
        try:
            ai = call_llm(bound, messages, tracer, "supervisor")
        except LLMError as exc:
            tracer.emit("supervisor", "error", "llm_unavailable", summary=str(exc))
            break
        valid = [c for c in ai.tool_calls if c["name"] in _TOOL_TO_ACTION]
        if valid:
            call = valid[0]
            if len(ai.tool_calls) > 1:
                tracer.emit(
                    "supervisor",
                    "guard",
                    "extra_tool_calls_ignored",
                    summary=f"{len(ai.tool_calls)} calls; used the first",
                )
            break
        messages += [ai, HumanMessage("You must call exactly one of the provided tools.")]
        tracer.emit("supervisor", "guard", "no_tool_call", summary=f"attempt {attempt + 1}")

    if call is None:  # deterministic fallback policy
        if state.get("analysis_rounds", 0) == 0 or pending:
            return guard(
                "document_analysis",
                question,
                "no valid decision from model; default to document analysis",
            )
        return guard(
            "finish",
            "",
            "no valid decision from model; finishing with recorded evidence",
            "fallback",
        )

    action = _TOOL_TO_ACTION[call["name"]]
    task = str(call["args"].get("task") or call["args"].get("reason") or "").strip()

    # Guards that override model choices which would waste the budget or lose evidence.
    if action == "finish" and pending:
        return guard(
            "document_analysis",
            f"Analyse the newly fetched web sources {', '.join(pending)} for: {question}",
            "web sources were fetched but not analysed yet",
        )
    if action == "finish" and not evidence and state.get("analysis_rounds", 0) < MAX_EMPTY_ANALYSIS_RETRIES:
        return guard(
            "document_analysis",
            f"No evidence has been recorded yet, so the question cannot be answered. Search the "
            f"documents for: {question}. You MUST call record_evidence on the most relevant "
            f"passages — searching alone does not record evidence.",
            "no evidence recorded yet; retrying document analysis",
        )

    tracer.emit("supervisor", "decision", action, summary=truncate(task, 200))
    update = _decision(state, action, task if action != "finish" else "", task, "model")
    if action == "finish":
        update["stop_reason"] = "supervisor_finished"
    return update


def route_from_supervisor(state: ResearchState) -> str:
    return {"web_research": "web_researcher", "document_analysis": "document_analyst"}.get(
        state.get("next_action", "finish"), "writer"
    )


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------


def _worker_deadline(ws: Workspace) -> float:
    return time.monotonic() + max(0.0, ws.time_left() - ws.settings.writer_reserve_s)


def web_researcher_node(state: ResearchState, runtime: Runtime[Workspace]) -> dict[str, Any]:
    ws = runtime.context
    session = ToolSession(ws, state, "web_researcher")
    task = state.get("next_task") or state["question"]
    fetched = [f"{sid}: {s.title}" for sid, s in state.get("sources", {}).items() if s.kind == "web"]
    user = (
        f"Research question: {state['question']}\nYour task: {task}\n"
        f"Already fetched: {'; '.join(fetched) if fetched else 'nothing yet'}"
    )
    ws.tracer.emit("web_researcher", "node", "start", summary=truncate(task, 200))
    result = run_tool_loop(
        ws.llm,
        web_tools(session),
        prompts.WEB_RESEARCHER,
        user,
        agent="web_researcher",
        tracer=ws.tracer,
        max_tool_calls=ws.settings.max_worker_tool_calls,
        deadline=_worker_deadline(ws),
    )
    report = WorkerReport(
        agent="web_researcher",
        task=task,
        summary=result.summary,
        tool_calls=result.tool_calls,
        stop_reason=result.stop_reason,
        new_source_ids=list(session.new_sources),
        errors=session.errors + result.errors,
    )
    update = session.state_update()
    update.update(
        worker_reports=[report],
        web_rounds=state.get("web_rounds", 0) + 1,
        pending_web_sources=state.get("pending_web_sources", []) + list(session.new_sources),
        errors=result.errors,
    )
    return update


def document_analyst_node(state: ResearchState, runtime: Runtime[Workspace]) -> dict[str, Any]:
    ws = runtime.context
    session = ToolSession(ws, state, "document_analyst")
    task = state.get("next_task") or state["question"]
    recorded = [f"{e.id} ({e.passage_id})" for e in state.get("evidence", [])]
    source_ids = ", ".join(state.get("sources", {}))
    user = (
        f"Research question: {state['question']}\nYour task: {task}\n"
        f"Available source_ids: {source_ids}\n"
        f"Evidence already recorded: {', '.join(recorded) if recorded else 'none'}"
    )
    ws.tracer.emit("document_analyst", "node", "start", summary=truncate(task, 200))
    result = run_tool_loop(
        ws.llm,
        analysis_tools(session),
        prompts.DOCUMENT_ANALYST,
        user,
        agent="document_analyst",
        tracer=ws.tracer,
        max_tool_calls=ws.settings.max_worker_tool_calls,
        deadline=_worker_deadline(ws),
    )
    report = WorkerReport(
        agent="document_analyst",
        task=task,
        summary=result.summary,
        tool_calls=result.tool_calls,
        stop_reason=result.stop_reason,
        new_evidence_ids=[e.id for e in session.new_evidence],
        errors=session.errors + result.errors,
    )
    update = session.state_update()
    update.update(
        worker_reports=[report],
        analysis_rounds=state.get("analysis_rounds", 0) + 1,
        pending_web_sources=[],
        errors=result.errors,
    )
    return update


def single_agent_node(state: ResearchState, runtime: Runtime[Workspace]) -> dict[str, Any]:
    """Baseline: one agent with all four tools and a comparable total tool budget."""
    ws = runtime.context
    session = ToolSession(ws, state, "single_agent")
    tools = analysis_tools(session) + (web_tools(session) if ws.search is not None else [])
    user = (
        f"Research question: {state['question']}\n"
        f"Available source_ids: {', '.join(state.get('sources', {}))}\n"
        f"Web tools: {'available' if ws.search is not None else 'disabled for this run'}"
    )
    ws.tracer.emit("single_agent", "node", "start", summary=truncate(state["question"], 200))
    result = run_tool_loop(
        ws.llm,
        tools,
        prompts.SINGLE_AGENT,
        user,
        agent="single_agent",
        tracer=ws.tracer,
        max_tool_calls=ws.settings.max_single_agent_tool_calls,
        deadline=_worker_deadline(ws),
    )
    report = WorkerReport(
        agent="single_agent",
        task=state["question"],
        summary=result.summary,
        tool_calls=result.tool_calls,
        stop_reason=result.stop_reason,
        new_source_ids=list(session.new_sources),
        new_evidence_ids=[e.id for e in session.new_evidence],
        errors=session.errors + result.errors,
    )
    update = session.state_update()
    stop = {"answered": "agent_finished"}.get(result.stop_reason, result.stop_reason)
    update.update(worker_reports=[report], stop_reason=stop, errors=result.errors)
    return update


# ---------------------------------------------------------------------------
# Writer and finaliser
# ---------------------------------------------------------------------------


def render_evidence(evidence: list[Evidence], sources: dict[str, Source], ws: Workspace) -> str:
    blocks = []
    for e in evidence:
        src = sources.get(e.source_id)
        title = ws.index.source_titles.get(e.source_id, src.title if src else e.source_id)
        date = f"; published {src.published}" if src and src.published else ""
        date += f"; retrieved {src.retrieved}" if src and src.kind == "web" and src.retrieved else ""
        blocks.append(
            f"[{e.id}] {title} ({e.source_id}), {e.location}{date}\n"
            f"Analyst note: {e.note}\nText: {truncate(e.quote, 1200)}"
        )
    return "\n\n".join(blocks)


def fallback_draft(evidence: list[Evidence]) -> ReportDraft:
    """Used when the writer model fails: list evidence notes without model synthesis."""
    return ReportDraft(
        answer="The report writer failed, so no synthesised answer is available. The recorded "
        "evidence and the analyst's notes are listed below.",
        key_findings=[DraftFinding(statement=e.note, evidence_ids=[e.id]) for e in evidence[:8]],
        limitations=["Findings are the document analyst's notes, not a synthesised answer."],
    )


def writer_node(state: ResearchState, runtime: Runtime[Workspace]) -> dict[str, Any]:
    ws = runtime.context
    evidence = state.get("evidence", [])
    ws.tracer.emit("writer", "node", "start", summary=f"{len(evidence)} evidence items")
    if not evidence:
        return {"draft": None, "writer_status": "no_evidence"}
    user = (
        f"Research question: {state['question']}\n\nEvidence:\n\n"
        f"{render_evidence(evidence, state.get('sources', {}), ws)}"
    )
    messages = [SystemMessage(prompts.WRITER), HumanMessage(user)]
    # Try the configured method first, then the alternates: some Ollama models reject
    # grammar-constrained formats (HTTP 501) but support tool-calling-based structuring.
    methods = _dedupe([ws.settings.structured_output_method, "function_calling", "json_schema"])
    last_exc: Exception | None = None
    for method in methods:
        try:
            structured = ws.llm.with_structured_output(ReportDraft, method=method)
            draft = call_llm(structured, messages, ws.tracer, "writer", retries=0)
            return {"draft": draft, "writer_status": "ok"}
        except Exception as exc:  # LLMError, ResponseError (501), schema validation
            last_exc = exc
            ws.tracer.emit("writer", "error", "writer_method_failed", summary=f"{method}: {str(exc)[:120]}")
    ws.tracer.emit("writer", "error", "writer_failed", summary=str(last_exc)[:200])
    return {
        "draft": fallback_draft(evidence),
        "writer_status": "fallback",
        "errors": [f"writer: {last_exc}"],
    }


def _dedupe(items: list[str]) -> list[str]:
    seen: list[str] = []
    for item in items:
        if item not in seen:
            seen.append(item)
    return seen


def _deterministic_limitations(state: ResearchState, ws: Workspace, cited_sources: set[str]) -> list[str]:
    notes: list[str] = []
    if ws.search is None:
        notes.append("Web search was disabled for this run; only the local document corpus was consulted.")
    if not any(sid.startswith("web-") for sid in cited_sources):
        notes.append(
            "All cited evidence comes from the local corpus, which reflects the document versions "
            "listed under Sources (e.g. the EU AI Act as published in July 2024). Later amendments, "
            "guidance or implementation changes are not covered."
        )
    reason = state.get("stop_reason", "")
    if reason in {"deadline", "turn_limit", "tool_budget", "llm_error", "evidence_limit"}:
        notes.append(f"Research stopped early because of a run limit ({reason}).")
    for sid, src in state.get("sources", {}).items():
        if "possible_prompt_injection" in src.flags:
            notes.append(
                f"Source {sid} contains text resembling instructions to an AI; it was treated as data only."
            )
    if state.get("writer_status") == "fallback":
        notes.append("The writer model failed; findings are unsynthesised analyst notes.")
    return notes


def finalize_node(state: ResearchState, runtime: Runtime[Workspace]) -> dict[str, Any]:
    ws = runtime.context
    evidence = state.get("evidence", [])
    sources = state.get("sources", {})
    draft = state.get("draft")
    if draft is None:
        draft = ReportDraft(
            answer="No evidence was recorded, so the question cannot be answered from the available sources.",
            key_findings=[],
            limitations=["No evidence was recorded during the run."],
        )
    findings = check_findings(draft.key_findings, evidence)
    conflicts = check_findings(draft.conflicts, evidence)
    check = citation_check(draft.answer, findings + conflicts, evidence)

    cited_ids = {i for f in findings + conflicts for i in f.evidence_ids}
    cited_sources = {e.source_id for e in evidence if e.id in cited_ids}
    used_sources = {e.source_id for e in evidence} | {sid for sid, s in sources.items() if s.kind == "web"}
    errors = list(state.get("errors", []))

    if not evidence:
        status = "failed"
    elif state.get("writer_status") != "ok" or state.get("stop_reason") in {
        "deadline",
        "llm_error",
    }:
        status = "partial"
    else:
        status = "completed"

    s = ws.settings
    report = Report(
        question=state["question"],
        answer=strip_invalid_inline(draft.answer, evidence),
        key_findings=findings,
        conflicts=conflicts,
        limitations=[*draft.limitations, *_deterministic_limitations(state, ws, cited_sources)],
        open_questions=draft.open_questions,
        evidence=evidence,
        sources=[sources[sid] for sid in sources if sid in used_sources],
        citation_check=check,
        run=RunInfo(
            mode=state.get("mode", "multi_agent"),
            model=s.chat_model,
            embedding_model=s.embedding_model if ws.index.dense_available else None,
            web_enabled=ws.search is not None,
            started_at=state.get("started_at", datetime.now(UTC).isoformat(timespec="seconds")),
            duration_s=round(time.monotonic() - ws.tracer.started, 1),
            llm_calls=ws.tracer.llm_calls,
            tool_calls=ws.tracer.tool_calls,
            status=status,
            stop_reason=state.get("stop_reason", "") or "completed",
            errors=errors + ws.index.warnings,
        ),
    )
    ws.tracer.emit(
        "finalize",
        "node",
        "citation_check",
        summary=f"{check.valid_citations}/{check.total_citations} citations valid; "
        f"{check.uncited_findings} uncited findings",
    )
    return {"report": report}


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_multi_agent_graph():
    g = StateGraph(ResearchState, context_schema=Workspace)
    g.add_node("supervisor", supervisor_node)
    g.add_node("web_researcher", web_researcher_node)
    g.add_node("document_analyst", document_analyst_node)
    g.add_node("writer", writer_node)
    g.add_node("finalize", finalize_node)
    g.add_edge(START, "supervisor")
    g.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "web_researcher": "web_researcher",
            "document_analyst": "document_analyst",
            "writer": "writer",
        },
    )
    g.add_edge("web_researcher", "supervisor")
    g.add_edge("document_analyst", "supervisor")
    g.add_edge("writer", "finalize")
    g.add_edge("finalize", END)
    return g.compile()


def build_single_agent_graph():
    g = StateGraph(ResearchState, context_schema=Workspace)
    g.add_node("single_agent", single_agent_node)
    g.add_node("writer", writer_node)
    g.add_node("finalize", finalize_node)
    g.add_edge(START, "single_agent")
    g.add_edge("single_agent", "writer")
    g.add_edge("writer", "finalize")
    g.add_edge("finalize", END)
    return g.compile()
