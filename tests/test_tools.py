from __future__ import annotations

from research_assistant.tools import ToolSession
from research_assistant.web import FetchedDocument, SearchResult

from .conftest import make_workspace
from .fakes import StubSearch


def _state(sources):
    return {
        "sources": sources,
        "evidence": [],
        "allowed_urls": [],
        "seen_passage_ids": [],
        "fetch_count": 0,
    }


def test_record_evidence_rejects_unknown_passage(settings, mini_index, mini_sources):
    ws = make_workspace(settings, llm=None, index=mini_index, search=None)
    session = ToolSession(ws, _state(mini_sources), "analyst")
    out = session.record_evidence("nope:1", "note")
    assert out.startswith("ERROR")
    assert not session.new_evidence


def test_record_evidence_requires_prior_retrieval(settings, mini_index, mini_sources):
    ws = make_workspace(settings, llm=None, index=mini_index, search=None)
    session = ToolSession(ws, _state(mini_sources), "analyst")
    # Passage exists but was never returned by search_documents in this session.
    out = session.record_evidence("doc-a:art-1:1", "note")
    assert "not been retrieved" in out
    assert not session.new_evidence


def test_search_then_record_evidence_succeeds_and_copies_quote(settings, mini_index, mini_sources):
    ws = make_workspace(settings, llm=None, index=mini_index, search=None)
    session = ToolSession(ws, _state(mini_sources), "analyst")
    session.search_documents("technical documentation obligations")
    out = session.record_evidence("doc-a:art-1:1", "States the documentation duty.")
    assert out.startswith("Recorded E1")
    assert len(session.new_evidence) == 1
    ev = session.new_evidence[0]
    # The quote is copied verbatim from the stored passage, never model-authored.
    assert ev.quote == mini_index.get("doc-a:art-1:1").text
    assert ev.source_id == "doc-a"


def test_record_evidence_deduplicates(settings, mini_index, mini_sources):
    ws = make_workspace(settings, llm=None, index=mini_index, search=None)
    session = ToolSession(ws, _state(mini_sources), "analyst")
    session.search_documents("documentation")
    session.record_evidence("doc-a:art-1:1", "first")
    out = session.record_evidence("doc-a:art-1:1", "second")
    assert "Already recorded" in out
    assert len(session.new_evidence) == 1


def test_fetch_page_rejects_url_not_from_search(settings, mini_index, mini_sources):
    ws = make_workspace(settings, llm=None, index=mini_index, search=StubSearch([]))
    session = ToolSession(ws, _state(mini_sources), "web")
    out = session.fetch_page("https://example.com/evil")
    assert out.startswith("ERROR")
    assert "not returned by web_search" in out


def test_web_search_disabled_returns_error(settings, mini_index, mini_sources):
    ws = make_workspace(settings, llm=None, index=mini_index, search=None)
    session = ToolSession(ws, _state(mini_sources), "web")
    out = session.web_search("anything")
    assert out.startswith("ERROR")
    assert "disabled" in out


def test_web_search_then_fetch_adds_source_and_passages(settings, mini_index, mini_sources):
    result = SearchResult(title="AI Office guidance", url="https://europa.eu/guide", snippet="...")
    ws = make_workspace(settings, llm=None, index=mini_index, search=StubSearch([result]))
    doc = FetchedDocument(
        url="https://europa.eu/guide",
        final_url="https://europa.eu/guide",
        title="AI Office guidance",
        text="The AI Office publishes guidance on the obligations for providers. " * 20,
        content_type="text/html",
        retrieved="2026-11-07",
    )
    ws.fetcher = lambda url: doc
    session = ToolSession(ws, _state(mini_sources), "web")
    session.web_search("AI Office guidance")  # authorises the URL
    out = session.fetch_page("https://europa.eu/guide")
    assert "Fetched web-1" in out
    assert "web-1" in session.new_sources
    # The fetched page's passages are searchable and recordable afterwards.
    session.search_documents("AI Office obligations", source_id="web-1")
    assert any(pid.startswith("web-1:") for pid in session.new_seen)


def test_fetch_budget_is_enforced(settings, mini_index, mini_sources):
    settings = settings.with_overrides(max_fetches_per_run=1)
    r1 = SearchResult(title="a", url="https://a.example.org/x", snippet="")
    r2 = SearchResult(title="b", url="https://b.example.org/y", snippet="")
    ws = make_workspace(settings, llm=None, index=mini_index, search=StubSearch([r1, r2]))
    ws.fetcher = lambda url: FetchedDocument(
        url=url,
        final_url=url,
        title="t",
        text="word " * 60,
        content_type="text/html",
        retrieved="2026-11-07",
    )
    session = ToolSession(ws, _state(mini_sources), "web")
    session.web_search("a")
    session.web_search("b")
    assert "Fetched web-1" in session.fetch_page("https://a.example.org/x")
    out = session.fetch_page("https://b.example.org/y")
    assert out.startswith("ERROR") and "budget" in out


def test_fetched_passages_are_wrapped_as_untrusted(settings, mini_index, mini_sources):
    result = SearchResult(title="page", url="https://news.example.com/a", snippet="")
    ws = make_workspace(settings, llm=None, index=mini_index, search=StubSearch([result]))
    ws.fetcher = lambda url: FetchedDocument(
        url=url,
        final_url=url,
        title="page",
        text="Breaking developments about AI model governance and documentation. " * 15,
        content_type="text/html",
        retrieved="2026-11-07",
    )
    session = ToolSession(ws, _state(mini_sources), "web")
    session.web_search("page")
    session.fetch_page("https://news.example.com/a")
    out = session.search_documents("AI model governance", source_id="web-1")
    assert "<untrusted_content" in out
