"""Web search and page retrieval.

Retrieved content is untrusted. This module only reads public http(s) resources:
it blocks private/loopback addresses (also after redirects), caps download size,
and flags text that looks like an instruction aimed at the model.
"""

from __future__ import annotations

import io
import ipaddress
import re
import socket
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx

from .schemas import SearchResult
from .text import normalize

USER_AGENT = "Mozilla/5.0 (compatible; research-assistant/0.1; local research demo)"
_ALLOWED_TYPES = ("text/html", "application/xhtml+xml", "application/pdf", "text/plain")
_INJECTION_PATTERNS = re.compile(
    r"ignore (all |any )?(previous|prior|above) instructions|disregard (the |all )?(previous|above)"
    r"|you are now (a|an|the)\b|system prompt|new instructions:|do not tell the user",
    re.IGNORECASE,
)


class WebError(RuntimeError):
    """A recoverable web failure that is reported to the model as a tool error."""


class SearchProvider(Protocol):
    name: str

    def search(self, query: str, max_results: int) -> list[SearchResult]: ...


class DDGSSearch:
    """Keyless metasearch via the `ddgs` library (unofficial; may be rate limited)."""

    def __init__(self, backend: str = "auto", region: str = "us-en", timeout_s: float = 15) -> None:
        self.name = f"ddgs:{backend}"
        self.backend, self.region, self.timeout_s = backend, region, timeout_s

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        from ddgs import DDGS

        try:
            rows = DDGS(timeout=int(self.timeout_s)).text(
                query, region=self.region, max_results=max_results, backend=self.backend
            )
        except Exception as exc:
            raise WebError(f"search failed: {type(exc).__name__}: {exc}") from exc
        results = []
        for row in rows or []:
            url = row.get("href") or row.get("url")
            if url:
                results.append(
                    SearchResult(title=row.get("title", ""), url=url, snippet=row.get("body", ""))
                )
        return results[:max_results]


class StaticSearch:
    """Deterministic provider for tests and recorded demos."""

    def __init__(self, results: dict[str, list[SearchResult]], name: str = "static") -> None:
        self.name, self.results = name, results

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        for key, rows in self.results.items():
            if key.lower() in query.lower():
                return rows[:max_results]
        return []


@dataclass
class FetchedDocument:
    url: str
    final_url: str
    title: str
    text: str
    content_type: str
    retrieved: str
    flags: list[str] = field(default_factory=list)
    pages: list[str] | None = None  # PDF pages, if the document was a PDF


def normalize_url(url: str) -> str:
    parts = urlsplit(url.strip())
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path or "/", parts.query, ""))


def _check_public_host(url: str) -> None:
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"}:
        raise WebError(f"blocked URL scheme {parts.scheme!r}; only http(s) is allowed")
    host = parts.hostname
    if not host:
        raise WebError("URL has no host")
    try:
        infos = socket.getaddrinfo(host, parts.port or (443 if parts.scheme == "https" else 80))
    except socket.gaierror as exc:
        raise WebError(f"could not resolve host {host!r}") from exc
    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        if not address.is_global:
            raise WebError(f"blocked non-public address {address} for host {host!r}")


def _extract_html(raw: bytes, url: str) -> tuple[str, str]:
    import trafilatura

    html = raw.decode("utf-8", errors="replace")
    text = trafilatura.extract(html, url=url, include_comments=False, include_tables=True) or ""
    meta = trafilatura.extract_metadata(html, default_url=url)
    title = (meta.title if meta and meta.title else "") or url
    return title, text


def _extract_pdf(raw: bytes, max_pages: int) -> tuple[str, list[str]]:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(raw))
    title = (reader.metadata.title if reader.metadata and reader.metadata.title else "") or ""
    pages = [normalize(p.extract_text() or "") for p in reader.pages[:max_pages]]
    return title, pages


def fetch_document(
    url: str,
    *,
    timeout_s: float = 20.0,
    max_bytes: int = 6_000_000,
    max_redirects: int = 5,
    max_pdf_pages: int = 80,
    client: httpx.Client | None = None,
) -> FetchedDocument:
    """Fetch a public page or PDF and return its main text. Raises WebError on failure."""
    own_client = client is None
    client = client or httpx.Client(timeout=timeout_s, headers={"User-Agent": USER_AGENT})
    current = url
    try:
        for _ in range(max_redirects + 1):
            _check_public_host(current)
            with client.stream("GET", current, follow_redirects=False) as response:
                if response.is_redirect:
                    location = response.headers.get("location")
                    if not location:
                        raise WebError("redirect without location header")
                    current = urljoin(current, location)
                    continue
                if response.status_code >= 400:
                    raise WebError(f"HTTP {response.status_code} for {current}")
                content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
                if content_type and not content_type.startswith(_ALLOWED_TYPES):
                    raise WebError(f"unsupported content type {content_type!r}")
                chunks, size = [], 0
                for chunk in response.iter_bytes():
                    size += len(chunk)
                    if size > max_bytes:
                        raise WebError(f"document larger than {max_bytes} bytes")
                    chunks.append(chunk)
                raw = b"".join(chunks)
                break
        else:
            raise WebError("too many redirects")
    except httpx.TimeoutException as exc:
        raise WebError(f"timed out fetching {current}") from exc
    except httpx.HTTPError as exc:
        raise WebError(f"HTTP error fetching {current}: {type(exc).__name__}") from exc
    finally:
        if own_client:
            client.close()

    retrieved = datetime.now(UTC).date().isoformat()
    is_pdf = content_type == "application/pdf" or raw[:5] == b"%PDF-"
    pages = None
    if is_pdf:
        title, pages = _extract_pdf(raw, max_pdf_pages)
        text = "\n".join(pages)
        title = title or current.rsplit("/", 1)[-1]
    else:
        title, text = _extract_html(raw, current)
        text = normalize(text)
    if len(text.split()) < 30:
        raise WebError("no readable main text could be extracted")

    flags = []
    if _INJECTION_PATTERNS.search(text):
        flags.append("possible_prompt_injection")
    return FetchedDocument(
        url=url,
        final_url=current,
        title=normalize(title).replace("\n", " ")[:200],
        text=text,
        content_type=content_type or ("application/pdf" if is_pdf else "text/html"),
        retrieved=retrieved,
        flags=flags,
        pages=pages,
    )
