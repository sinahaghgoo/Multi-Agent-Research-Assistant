from __future__ import annotations

import time

from research_assistant.agents import run_tool_loop
from research_assistant.tools import ToolSession, analysis_tools

from .conftest import make_workspace
from .fakes import ScriptedChatModel, ToolCall, ai


def _session(settings, mini_index, mini_sources, search=None):
    ws = make_workspace(settings, llm=None, index=mini_index, search=search)
    state = {
        "sources": mini_sources,
        "evidence": [],
        "allowed_urls": [],
        "seen_passage_ids": [],
        "fetch_count": 0,
    }
    return ws, ToolSession(ws, state, "analyst")


def test_loop_stops_when_model_answers(settings, mini_index, mini_sources):
    model = ScriptedChatModel(
        responses=[
            ai(tool_calls=[ToolCall("search_documents", {"query": "documentation"})]),
            ai(tool_calls=[ToolCall("record_evidence", {"passage_id": "doc-a:art-1:1", "note": "duty"})]),
            ai(content="Recorded one passage on documentation duties."),
        ]
    )
    ws, session = _session(settings, mini_index, mini_sources)
    result = run_tool_loop(
        ws.llm or model,
        analysis_tools(session),
        "sys",
        "task",
        agent="analyst",
        tracer=ws.tracer,
        max_tool_calls=6,
        deadline=time.monotonic() + 30,
    )
    assert result.stop_reason == "answered"
    assert result.tool_calls == 2
    assert len(session.new_evidence) == 1


def test_loop_enforces_tool_budget(settings, mini_index, mini_sources):
    # Model keeps searching forever; the budget must cut it off.
    responses = [ai(tool_calls=[ToolCall("search_documents", {"query": "x"})]) for _ in range(10)]
    responses.append(ai(content="summary after budget"))
    model = ScriptedChatModel(responses=responses)
    ws, session = _session(settings, mini_index, mini_sources)
    result = run_tool_loop(
        model,
        analysis_tools(session),
        "sys",
        "task",
        agent="analyst",
        tracer=ws.tracer,
        max_tool_calls=3,
        deadline=time.monotonic() + 30,
    )
    assert result.stop_reason == "tool_budget"
    assert result.tool_calls == 3


def test_loop_respects_deadline(settings, mini_index, mini_sources):
    model = ScriptedChatModel(responses=[ai(tool_calls=[ToolCall("search_documents", {"query": "x"})])])
    ws, session = _session(settings, mini_index, mini_sources)
    result = run_tool_loop(
        model,
        analysis_tools(session),
        "sys",
        "task",
        agent="analyst",
        tracer=ws.tracer,
        max_tool_calls=6,
        deadline=time.monotonic() - 1,
    )
    assert result.stop_reason == "deadline"
    assert result.tool_calls == 0


def test_loop_reports_unknown_tool_without_crashing(settings, mini_index, mini_sources):
    model = ScriptedChatModel(
        responses=[
            ai(tool_calls=[ToolCall("nonexistent_tool", {"foo": "bar"})]),
            ai(content="done"),
        ]
    )
    ws, session = _session(settings, mini_index, mini_sources)
    result = run_tool_loop(
        model,
        analysis_tools(session),
        "sys",
        "task",
        agent="analyst",
        tracer=ws.tracer,
        max_tool_calls=6,
        deadline=time.monotonic() + 30,
    )
    assert result.stop_reason == "answered"
    assert any("unknown tool" in e for e in result.errors)
