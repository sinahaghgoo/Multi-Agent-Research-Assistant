"""High-level entry point: assemble dependencies, run a graph, return a Report + trace."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from langchain_core.language_models import BaseChatModel

from .config import Settings
from .corpus import load_passages, local_sources
from .graph import build_multi_agent_graph, build_single_agent_graph
from .models import make_chat_model, make_embedder, require_models
from .retrieval import Embedder, PassageIndex
from .schemas import Report
from .tools import Workspace
from .trace import TraceEvent, Tracer
from .web import DDGSSearch, SearchProvider

Mode = Literal["multi_agent", "single_agent"]


@dataclass
class RunOutput:
    report: Report
    events: list[TraceEvent]


class ResearchAssistant:
    """Loads the corpus and builds an index once, then answers many questions."""

    def __init__(
        self,
        settings: Settings,
        *,
        llm: BaseChatModel | None = None,
        embedder: Embedder | None | Literal["auto"] = "auto",
        search: SearchProvider | None | Literal["auto"] = "auto",
        check_models: bool = True,
    ) -> None:
        self.settings = settings
        want_embeddings = embedder == "auto" or isinstance(embedder, Embedder)
        if check_models and llm is None:
            require_models(settings, need_embeddings=want_embeddings and embedder == "auto")

        self.llm = llm or make_chat_model(settings)
        if embedder == "auto":
            try:
                embedder = make_embedder(settings)
            except Exception:
                embedder = None
        self.search: SearchProvider | None = (
            (DDGSSearch(settings.search_backend, settings.search_region, settings.http_timeout_s)
             if settings.web_enabled else None)
            if search == "auto" else search
        )

        sources = local_sources(settings.corpus_dir)
        passages = load_passages(settings.corpus_dir)
        self.index = PassageIndex(
            passages,
            {sid: s.title for sid, s in sources.items()},
            embedder=embedder if isinstance(embedder, Embedder) else None,
            cache_dir=settings.cache_dir,
        )
        self._local_sources = sources
        self._multi = build_multi_agent_graph()
        self._single = build_single_agent_graph()

    def index_summary(self) -> str:
        mode = "hybrid (BM25 + embeddings)" if self.index.dense_available else "BM25 only"
        return f"{len(self.index.passages)} passages from {len(self._local_sources)} documents; {mode}"

    def run(
        self,
        question: str,
        *,
        mode: Mode = "multi_agent",
        on_event: Callable[[TraceEvent], None] | None = None,
    ) -> RunOutput:
        tracer = Tracer(listener=on_event)
        # Re-seed index to the committed corpus so web pages from a previous run don't leak in.
        self.index.source_titles = {sid: s.title for sid, s in self._local_sources.items()}
        ws = Workspace(
            settings=self.settings,
            llm=self.llm,
            index=self.index,
            search=self.search,
            tracer=tracer,
            deadline=time.monotonic() + self.settings.run_timeout_s,
        )
        state = {
            "question": question.strip(),
            "mode": mode,
            "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "sources": dict(self._local_sources),
            "fetch_count": 0,
            "turn": 0,
            "web_rounds": 0,
            "analysis_rounds": 0,
            "pending_web_sources": [],
        }
        graph = self._multi if mode == "multi_agent" else self._single
        tracer.emit(mode, "node", "run_start", summary=question)
        final = graph.invoke(state, context=ws, config={"recursion_limit": 60})
        report = final["report"]
        tracer.emit(mode, "node", "run_end", summary=f"status={report.run.status}")
        return RunOutput(report=report, events=tracer.events)


def build_assistant(settings: Settings | None = None, **kwargs) -> ResearchAssistant:
    return ResearchAssistant(settings or Settings.from_env(), **kwargs)
