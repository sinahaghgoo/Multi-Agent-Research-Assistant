from __future__ import annotations

import time

import pytest

from research_assistant.config import Settings
from research_assistant.retrieval import PassageIndex
from research_assistant.schemas import Passage, Source
from research_assistant.tools import Workspace
from research_assistant.trace import Tracer


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        cache_dir=tmp_path / "cache",
        runs_dir=tmp_path / "runs",
        max_supervisor_turns=4,
        max_worker_tool_calls=4,
        max_single_agent_tool_calls=8,
        run_timeout_s=60,
        writer_reserve_s=5,
    )


@pytest.fixture
def mini_passages() -> list[Passage]:
    return [
        Passage(
            id="doc-a:art-1:1",
            source_id="doc-a",
            location="Article 1",
            text="Providers of general-purpose AI models shall draw up technical documentation "
            "and keep it up to date for the AI Office.",
        ),
        Passage(
            id="doc-a:art-2:1",
            source_id="doc-a",
            location="Article 2",
            text="This Regulation applies across all sectors without prejudice to data protection law.",
        ),
        Passage(
            id="doc-b:p1:1",
            source_id="doc-b",
            location="PDF page 1",
            text="The framework organises risk management into govern, map, measure and manage functions.",
        ),
    ]


@pytest.fixture
def mini_sources() -> dict[str, Source]:
    return {
        "doc-a": Source(id="doc-a", kind="local", title="Example Regulation", published="2024-07-12"),
        "doc-b": Source(id="doc-b", kind="local", title="Example Framework", published="2023-01-26"),
    }


@pytest.fixture
def mini_index(mini_passages, mini_sources) -> PassageIndex:
    titles = {sid: s.title for sid, s in mini_sources.items()}
    return PassageIndex(mini_passages, titles, embedder=None, cache_dir=None)


def make_workspace(settings, llm, index, search, deadline_s: float = 60.0) -> Workspace:
    return Workspace(
        settings=settings,
        llm=llm,
        index=index,
        search=search,
        tracer=Tracer(),
        deadline=time.monotonic() + deadline_s,
    )
