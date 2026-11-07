from __future__ import annotations

from datetime import UTC, datetime

from research_assistant.graph import (
    build_multi_agent_graph,
    build_single_agent_graph,
    supervisor_node,
)
from research_assistant.schemas import Source

from .conftest import make_workspace
from .fakes import ScriptedChatModel, StubSearch, ToolCall, ai, draft_json


def _initial_state(mini_sources, mode="multi_agent", question="What documentation must providers keep?"):
    return {
        "question": question,
        "mode": mode,
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "sources": dict(mini_sources),
        "fetch_count": 0,
        "turn": 0,
        "web_rounds": 0,
        "analysis_rounds": 0,
        "pending_web_sources": [],
    }


def test_multi_agent_run_terminates_with_valid_report(settings, mini_index, mini_sources):
    model = ScriptedChatModel(
        responses=[
            ai(tool_calls=[ToolCall("delegate_document_analysis", {"task": "Find documentation duties"})]),
            ai(
                tool_calls=[
                    ToolCall(
                        "search_documents",
                        {"query": "technical documentation obligations providers"},
                    )
                ]
            ),
            ai(
                tool_calls=[
                    ToolCall(
                        "record_evidence",
                        {"passage_id": "doc-a:art-1:1", "note": "Documentation duty."},
                    )
                ]
            ),
            ai(content="Recorded the documentation obligation."),
            ai(tool_calls=[ToolCall("finish_research", {"reason": "evidence covers the question"})]),
        ],
        structured_payload=draft_json(
            "Providers must keep technical documentation [E1].",
            [("Providers draw up technical documentation", ["E1"])],
            limitations=["Only one document consulted."],
        ),
    )
    ws = make_workspace(settings, llm=model, index=mini_index, search=None)
    final = build_multi_agent_graph().invoke(
        _initial_state(mini_sources), context=ws, config={"recursion_limit": 60}
    )
    report = final["report"]

    assert report.run.status == "completed"
    assert len(report.evidence) == 1
    assert report.evidence[0].id == "E1"
    # Citation is valid and the finding cites recorded evidence.
    assert report.citation_check.valid_citations == report.citation_check.total_citations
    assert report.citation_check.total_citations >= 1
    assert report.key_findings[0].evidence_ids == ["E1"]
    assert report.run.stop_reason == "supervisor_finished"


def test_supervisor_turn_limit_forces_finish(settings, mini_index, mini_sources):
    model = ScriptedChatModel(
        responses=[ai(tool_calls=[ToolCall("delegate_document_analysis", {"task": "x"})])]
    )
    ws = make_workspace(settings, llm=model, index=mini_index, search=None)
    state = _initial_state(mini_sources)
    state["turn"] = settings.max_supervisor_turns  # already at the limit
    update = supervisor_node(state, _runtime(ws))
    assert update["next_action"] == "finish"
    assert update["stop_reason"] == "turn_limit"
    assert update["decisions"][0].decided_by == "guard"


def test_supervisor_guard_forces_analysis_of_unanalysed_web_sources(settings, mini_index, mini_sources):
    # Model wants to finish, but a fetched web source has not been analysed yet.
    model = ScriptedChatModel(responses=[ai(tool_calls=[ToolCall("finish_research", {"reason": "done"})])])
    ws = make_workspace(settings, llm=model, index=mini_index, search=StubSearch([]))
    state = _initial_state(mini_sources)
    state["turn"] = 1
    state["analysis_rounds"] = 0
    state["web_rounds"] = 1
    state["pending_web_sources"] = ["web-1"]
    state["sources"]["web-1"] = Source(id="web-1", kind="web", title="page", url="https://e.org")
    update = supervisor_node(state, _runtime(ws))
    assert update["next_action"] == "document_analysis"
    assert update["decisions"][0].decided_by == "guard"


def test_supervisor_cannot_finish_before_any_analysis(settings, mini_index, mini_sources):
    model = ScriptedChatModel(responses=[ai(tool_calls=[ToolCall("finish_research", {"reason": "done"})])])
    ws = make_workspace(settings, llm=model, index=mini_index, search=None)
    state = _initial_state(mini_sources)
    state["turn"] = 1
    update = supervisor_node(state, _runtime(ws))
    assert update["next_action"] == "document_analysis"


def test_supervisor_retries_analysis_when_no_evidence_recorded(settings, mini_index, mini_sources):
    # An analysis round ran but recorded nothing; finishing would produce an empty report.
    model = ScriptedChatModel(responses=[ai(tool_calls=[ToolCall("finish_research", {"reason": "covered"})])])
    ws = make_workspace(settings, llm=model, index=mini_index, search=None)
    state = _initial_state(mini_sources)
    state["turn"] = 2
    state["analysis_rounds"] = 1  # already analysed once, but no evidence
    update = supervisor_node(state, _runtime(ws))
    assert update["next_action"] == "document_analysis"
    assert update["decisions"][0].decided_by == "guard"


def test_supervisor_allows_empty_finish_after_retry_budget(settings, mini_index, mini_sources):
    # After the retry budget is spent, a 0-evidence finish is allowed (graceful failure).
    model = ScriptedChatModel(
        responses=[ai(tool_calls=[ToolCall("finish_research", {"reason": "nothing found"})])]
    )
    ws = make_workspace(settings, llm=model, index=mini_index, search=None)
    state = _initial_state(mini_sources)
    state["turn"] = 3
    state["analysis_rounds"] = 2  # retry budget spent
    update = supervisor_node(state, _runtime(ws))
    assert update["next_action"] == "finish"


def test_supervisor_falls_back_when_model_gives_no_tool_call(settings, mini_index, mini_sources):
    model = ScriptedChatModel(responses=[ai(content="I think we should stop."), ai(content="still no tool")])
    ws = make_workspace(settings, llm=model, index=mini_index, search=None)
    state = _initial_state(mini_sources)
    state["turn"] = 1
    state["analysis_rounds"] = 1  # analysis already happened -> fallback can finish
    update = supervisor_node(state, _runtime(ws))
    assert update["next_action"] == "finish"
    assert update["decisions"][0].decided_by in {"guard", "fallback"}


def test_single_agent_run_produces_report(settings, mini_index, mini_sources):
    model = ScriptedChatModel(
        responses=[
            ai(tool_calls=[ToolCall("search_documents", {"query": "technical documentation obligations"})]),
            ai(
                tool_calls=[ToolCall("record_evidence", {"passage_id": "doc-a:art-1:1", "note": "Doc duty."})]
            ),
            ai(content="Found the documentation obligation."),
        ],
        structured_payload=draft_json("Providers keep documentation [E1].", [("Keep documentation", ["E1"])]),
    )
    ws = make_workspace(settings, llm=model, index=mini_index, search=None)
    final = build_single_agent_graph().invoke(
        _initial_state(mini_sources, mode="single_agent"),
        context=ws,
        config={"recursion_limit": 40},
    )
    report = final["report"]
    assert report.run.mode == "single_agent"
    assert len(report.evidence) == 1
    assert report.run.status == "completed"


class _Runtime:
    def __init__(self, ctx):
        self.context = ctx


def _runtime(ws):
    return _Runtime(ws)
