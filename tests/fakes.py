"""Scripted fake chat model and search provider for deterministic tests.

The fake model returns pre-programmed tool calls / content based on a queue of "turns",
so the graph, tool loops, guards and validation can be tested without Ollama.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage

from research_assistant.schemas import ReportDraft
from research_assistant.web import SearchResult


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]


def ai(content: str = "", tool_calls: Sequence[ToolCall] = ()) -> AIMessage:
    calls = [
        {"name": c.name, "args": c.args, "id": f"call_{i}", "type": "tool_call"}
        for i, c in enumerate(tool_calls)
    ]
    return AIMessage(content=content, tool_calls=calls)


class ScriptedChatModel(FakeMessagesListChatModel):
    """Replays a fixed list of AIMessages; supports bind_tools and structured output."""

    structured_payload: dict | None = None

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001
        return self

    def with_structured_output(self, schema, **kwargs):  # noqa: ANN001
        payload = self.structured_payload or {
            "answer": "Draft answer citing [E1].",
            "key_findings": [{"statement": "A finding.", "evidence_ids": ["E1"]}],
            "conflicts": [],
            "limitations": ["A limitation."],
            "open_questions": [],
        }

        class _Structured:
            def invoke(_self, _messages: list[BaseMessage]):  # noqa: ANN001
                return ReportDraft.model_validate(payload)

        return _Structured()


def draft_json(answer: str, findings: list[tuple[str, list[str]]], **extra) -> dict:
    return {
        "answer": answer,
        "key_findings": [{"statement": s, "evidence_ids": ids} for s, ids in findings],
        "conflicts": extra.get("conflicts", []),
        "limitations": extra.get("limitations", []),
        "open_questions": extra.get("open_questions", []),
    }


class StubSearch:
    name = "stub"

    def __init__(self, results: list[SearchResult]) -> None:
        self._results = results

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        return self._results[:max_results]
