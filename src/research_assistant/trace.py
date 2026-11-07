"""Execution trace: an append-only list of events with an optional live listener."""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, Field

EventKind = Literal["node", "decision", "llm", "tool_call", "tool_result", "guard", "error"]


class TraceEvent(BaseModel):
    seq: int
    t: float = Field(description="Seconds since the run started.")
    agent: str
    kind: EventKind
    name: str
    detail: dict[str, Any] = Field(default_factory=dict)
    duration_s: float | None = None

    def one_line(self) -> str:
        bits = [f"{self.t:6.1f}s", f"[{self.agent}]", self.kind, self.name]
        summary = self.detail.get("summary") or self.detail.get("args") or ""
        if summary:
            bits.append(str(summary)[:140])
        if self.duration_s is not None:
            bits.append(f"({self.duration_s:.1f}s)")
        return " ".join(bits)


class Tracer:
    def __init__(self, listener: Callable[[TraceEvent], None] | None = None) -> None:
        self.started = time.monotonic()
        self.events: list[TraceEvent] = []
        self.listener = listener
        self.llm_calls = 0
        self.tool_calls = 0

    def emit(
        self,
        agent: str,
        kind: EventKind,
        name: str,
        duration_s: float | None = None,
        **detail: Any,
    ) -> TraceEvent:
        if kind == "llm":
            self.llm_calls += 1
        elif kind == "tool_call":
            self.tool_calls += 1
        event = TraceEvent(
            seq=len(self.events) + 1,
            t=round(time.monotonic() - self.started, 2),
            agent=agent,
            kind=kind,
            name=name,
            detail=detail,
            duration_s=round(duration_s, 2) if duration_s is not None else None,
        )
        self.events.append(event)
        if self.listener is not None:
            # A UI callback must never break the run.
            with contextlib.suppress(Exception):
                self.listener(event)
        return event
