"""Runtime configuration, read from environment variables (optionally via a .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_bool(name: str, default: bool) -> bool:
    return _env(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # Models (Ollama)
    ollama_base_url: str = "http://localhost:11434"
    chat_model: str = "qwen3.5:9b"
    embedding_model: str = "qwen3-embedding:0.6b"
    temperature: float = 0.2
    seed: int = 7
    num_ctx: int = 16384
    llm_timeout_s: float = 240.0

    # Web access
    web_enabled: bool = True
    search_backend: str = "auto"
    search_region: str = "us-en"
    http_timeout_s: float = 20.0
    max_download_bytes: int = 6_000_000

    # Budgets and stopping conditions
    max_supervisor_turns: int = 5
    max_worker_tool_calls: int = 6
    max_single_agent_tool_calls: int = 14
    max_fetches_per_run: int = 4
    max_evidence: int = 16
    search_top_k: int = 5
    run_timeout_s: float = 900.0
    writer_reserve_s: float = 120.0  # time kept aside at the end for the report writer

    # Paths
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    cache_dir: Path = field(default_factory=lambda: PROJECT_ROOT / ".cache")
    runs_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "runs")

    @property
    def corpus_dir(self) -> Path:
        return self.data_dir / "corpus"

    def with_overrides(self, **kwargs: object) -> Settings:
        clean = {k: v for k, v in kwargs.items() if v is not None}
        return replace(self, **clean)

    @classmethod
    def from_env(cls, env_file: Path | None = None) -> Settings:
        load_dotenv(env_file or PROJECT_ROOT / ".env", override=False)
        d = cls()
        return cls(
            ollama_base_url=_env("RA_OLLAMA_BASE_URL", d.ollama_base_url),
            chat_model=_env("RA_CHAT_MODEL", d.chat_model),
            embedding_model=_env("RA_EMBEDDING_MODEL", d.embedding_model),
            temperature=float(_env("RA_TEMPERATURE", str(d.temperature))),
            seed=int(_env("RA_SEED", str(d.seed))),
            num_ctx=int(_env("RA_NUM_CTX", str(d.num_ctx))),
            llm_timeout_s=float(_env("RA_LLM_TIMEOUT_S", str(d.llm_timeout_s))),
            web_enabled=_env_bool("RA_WEB_ENABLED", d.web_enabled),
            search_backend=_env("RA_SEARCH_BACKEND", d.search_backend),
            search_region=_env("RA_SEARCH_REGION", d.search_region),
            http_timeout_s=float(_env("RA_HTTP_TIMEOUT_S", str(d.http_timeout_s))),
            max_supervisor_turns=int(_env("RA_MAX_SUPERVISOR_TURNS", str(d.max_supervisor_turns))),
            max_worker_tool_calls=int(_env("RA_MAX_WORKER_TOOL_CALLS", str(d.max_worker_tool_calls))),
            max_single_agent_tool_calls=int(
                _env("RA_MAX_SINGLE_AGENT_TOOL_CALLS", str(d.max_single_agent_tool_calls))
            ),
            max_fetches_per_run=int(_env("RA_MAX_FETCHES_PER_RUN", str(d.max_fetches_per_run))),
            run_timeout_s=float(_env("RA_RUN_TIMEOUT_S", str(d.run_timeout_s))),
            runs_dir=Path(_env("RA_RUNS_DIR", str(d.runs_dir))),
            cache_dir=Path(_env("RA_CACHE_DIR", str(d.cache_dir))),
        )
