"""Model construction and a lightweight Ollama health check."""

from __future__ import annotations

from dataclasses import dataclass

import httpx
from langchain_ollama import ChatOllama

from .config import Settings
from .retrieval import OllamaEmbedder


def make_chat_model(settings: Settings) -> ChatOllama:
    """A deterministic-as-possible ChatOllama with thinking disabled for tool-calling speed."""
    return ChatOllama(
        model=settings.chat_model,
        base_url=settings.ollama_base_url,
        temperature=settings.temperature,
        num_ctx=settings.num_ctx,
        seed=settings.seed,
        reasoning=False,  # disable thinking tokens: faster, and tool loops don't need them
        client_kwargs={"timeout": settings.llm_timeout_s},
    )


def make_embedder(settings: Settings) -> OllamaEmbedder:
    return OllamaEmbedder(model=settings.embedding_model, base_url=settings.ollama_base_url)


@dataclass
class OllamaStatus:
    reachable: bool
    models: list[str]
    detail: str = ""

    def has(self, model: str) -> bool:
        base = model.split(":")[0]
        return any(m == model or m.split(":")[0] == base for m in self.models)


def check_ollama(settings: Settings) -> OllamaStatus:
    try:
        response = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=5.0)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        return OllamaStatus(False, [], f"{type(exc).__name__}: {exc}")
    models = [m["name"] for m in response.json().get("models", [])]
    return OllamaStatus(True, models)


def require_models(settings: Settings, *, need_embeddings: bool) -> OllamaStatus:
    status = check_ollama(settings)
    if not status.reachable:
        raise RuntimeError(
            f"Ollama is not reachable at {settings.ollama_base_url} ({status.detail}). "
            "Start it with `ollama serve`."
        )
    missing = []
    if not status.has(settings.chat_model):
        missing.append(settings.chat_model)
    if need_embeddings and not status.has(settings.embedding_model):
        missing.append(settings.embedding_model)
    if missing:
        raise RuntimeError(
            "Missing Ollama models: "
            + ", ".join(missing)
            + ". Pull them with: "
            + "; ".join(f"ollama pull {m}" for m in missing)
        )
    return status
