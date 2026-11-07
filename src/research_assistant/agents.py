"""Bounded tool-calling loop shared by the worker agents and the single-agent baseline."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool

from .trace import Tracer


class LLMError(RuntimeError):
    """The model server failed twice in a row (timeout, connection error, server error)."""


def call_llm(
    runnable: Runnable, messages: list[BaseMessage], tracer: Tracer, agent: str, retries: int = 1
) -> object:
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        started = time.monotonic()
        try:
            result = runnable.invoke(messages)
        except Exception as exc:  # httpx/ollama errors, schema validation errors
            last_exc = exc
            tracer.emit(
                agent,
                "error",
                "llm_error",
                duration_s=time.monotonic() - started,
                summary=f"attempt {attempt + 1}: {type(exc).__name__}: {str(exc)[:200]}",
            )
            continue
        usage = getattr(result, "usage_metadata", None) or {}
        tracer.emit(
            agent,
            "llm",
            "llm_call",
            duration_s=time.monotonic() - started,
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
        )
        return result
    raise LLMError(f"{type(last_exc).__name__}: {last_exc}")


@dataclass
class LoopResult:
    summary: str
    tool_calls: int
    stop_reason: str
    errors: list[str] = field(default_factory=list)


def _fmt_args(args: dict) -> str:
    return json.dumps(args, ensure_ascii=False)[:200]


def run_tool_loop(
    llm: BaseChatModel,
    tools: list[BaseTool],
    system_prompt: str,
    user_prompt: str,
    *,
    agent: str,
    tracer: Tracer,
    max_tool_calls: int,
    deadline: float,
) -> LoopResult:
    """Let the model call tools until it answers in text, the budget is spent, or time runs out.

    The model chooses which tool to call and with which arguments; this loop deterministically
    executes the calls, converts exceptions into error messages the model can react to, and
    enforces the tool-call budget and the deadline.
    """
    bound = llm.bind_tools(tools)
    by_name = {t.name: t for t in tools}
    messages: list[BaseMessage] = [SystemMessage(system_prompt), HumanMessage(user_prompt)]
    calls = 0
    errors: list[str] = []

    while True:
        if time.monotonic() > deadline:
            tracer.emit(agent, "guard", "deadline", summary="run deadline reached inside worker")
            return LoopResult("Stopped: run deadline reached.", calls, "deadline", errors)
        try:
            ai = call_llm(bound, messages, tracer, agent)
        except LLMError as exc:
            errors.append(str(exc))
            return LoopResult(f"Stopped: model error ({exc}).", calls, "llm_error", errors)
        assert isinstance(ai, AIMessage)
        messages.append(ai)

        if not ai.tool_calls:
            text = ai.content if isinstance(ai.content, str) else str(ai.content)
            return LoopResult(text.strip() or "(no summary)", calls, "answered", errors)

        for call in ai.tool_calls:
            if calls >= max_tool_calls:
                content = "Tool budget exhausted. Do not call more tools; reply with your summary."
            elif call["name"] not in by_name:
                content = f"ERROR: unknown tool {call['name']!r}. Available: {', '.join(by_name)}"
                errors.append(content)
            else:
                calls += 1
                tracer.emit(agent, "tool_call", call["name"], args=_fmt_args(call["args"]))
                started = time.monotonic()
                try:
                    content = str(by_name[call["name"]].invoke(call["args"]))
                except Exception as exc:  # argument validation errors, unexpected tool failures
                    content = f"ERROR: {type(exc).__name__}: {str(exc)[:300]}"
                    errors.append(content)
                    tracer.emit(
                        agent,
                        "error",
                        call["name"],
                        duration_s=time.monotonic() - started,
                        summary=content,
                    )
            messages.append(ToolMessage(content, tool_call_id=call["id"], name=call["name"]))

        if calls >= max_tool_calls:
            # One last call without tools so the agent can summarise what it did.
            messages.append(HumanMessage("Tool budget reached. Summarise your results now."))
            try:
                final = call_llm(llm, messages, tracer, agent)
                text = final.content if isinstance(final.content, str) else str(final.content)
            except LLMError as exc:
                errors.append(str(exc))
                text = "Stopped: tool budget reached (summary unavailable)."
            tracer.emit(agent, "guard", "tool_budget", summary=f"{calls} tool calls used")
            return LoopResult(text.strip(), calls, "tool_budget", errors)
