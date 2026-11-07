"""Local document corpus: download, parse into citable passages, and load.

The EU AI Act is parsed from the EUR-Lex HTML so passages carry legal locations
(Article / Recital / Annex). NIST PDFs are parsed page by page ("PDF page N").
"""

from __future__ import annotations

import hashlib
import io
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import httpx
from lxml import html as lxml_html
from pypdf import PdfReader

from .schemas import Passage, Source
from .text import chunk_lines, normalize

PASSAGES_FILE = "passages.jsonl"
_LABEL_ONLY = re.compile(r"^(\(?[0-9a-zivx]{1,5}[.)]|—|-|•)$", re.IGNORECASE)
_PAGE_NUMBER = re.compile(r"^(Page )?\d{1,3}$")


@dataclass(frozen=True)
class Section:
    key: str  # used in passage ids, e.g. "art-5"
    location: str  # human-readable, e.g. "Article 5 (Prohibited AI practices)"
    lines: list[str]


def load_manifest(corpus_dir: Path) -> dict:
    return json.loads((corpus_dir / "manifest.json").read_text(encoding="utf-8"))


def local_sources(corpus_dir: Path) -> dict[str, Source]:
    sources = {}
    for entry in load_manifest(corpus_dir)["sources"]:
        sources[entry["id"]] = Source(
            id=entry["id"],
            kind="local",
            title=entry["title"],
            url=entry.get("canonical_url") or entry["url"],
            publisher=entry.get("publisher"),
            published=entry.get("published"),
            retrieved=entry.get("retrieved"),
            license=entry.get("license"),
        )
    return sources


def load_passages(corpus_dir: Path) -> list[Passage]:
    path = corpus_dir / PASSAGES_FILE
    with path.open(encoding="utf-8") as fh:
        return [Passage.model_validate_json(line) for line in fh if line.strip()]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _merge_labels(texts: list[str]) -> list[str]:
    """Join list labels such as '(a)' or '1.' with the text that follows them."""
    lines: list[str] = []
    pending = ""
    for text in texts:
        if _LABEL_ONLY.match(text):
            pending = f"{pending} {text}".strip()
            continue
        lines.append(f"{pending} {text}".strip() if pending else text)
        pending = ""
    if pending:
        lines.append(pending)
    return lines


def _p_texts(element) -> list[str]:
    texts = []
    for p in element.iter("p"):
        text = normalize(p.text_content())
        if text:
            texts.append(text.replace("\n", " "))
    return texts


def parse_eurlex_html(raw: bytes) -> Iterator[Section]:
    tree = lxml_html.fromstring(raw)
    for div in tree.xpath('//div[@class="eli-subdivision"][starts-with(@id, "rct_")]'):
        number = div.get("id").split("_", 1)[1]
        lines = _merge_labels(_p_texts(div))
        yield Section(f"rct-{number}", f"Recital {number}", lines)

    for div in tree.xpath('//div[@class="eli-subdivision"][starts-with(@id, "art_")]'):
        art_id = div.get("id")
        if not re.fullmatch(r"art_\d+", art_id):
            continue
        number = art_id.split("_", 1)[1]
        heading = div.xpath('.//p[@class="oj-sti-art"]')
        title = normalize(heading[0].text_content()) if heading else ""
        texts = [
            t for t in _p_texts(div) if t != f"Article {number}" and t != title
        ]
        location = f"Article {number}" + (f" ({title})" if title else "")
        yield Section(f"art-{number}", location, _merge_labels(texts))

    for div in tree.xpath('//div[@class="eli-container"][starts-with(@id, "anx_")]'):
        number = div.get("id").split("_", 1)[1]
        titles = [normalize(p.text_content()) for p in div.xpath('./p[@class="oj-doc-ti"]')]
        subtitle = titles[1] if len(titles) > 1 else ""
        texts = [t for t in _p_texts(div) if t not in titles]
        location = f"Annex {number}" + (f" ({subtitle})" if subtitle else "")
        yield Section(f"anx-{number}", location, _merge_labels(texts))


def parse_pdf(raw: bytes, header_lines: tuple[str, ...] = ()) -> Iterator[Section]:
    reader = PdfReader(io.BytesIO(raw))
    for index, page in enumerate(reader.pages, start=1):
        text = normalize(page.extract_text() or "")
        lines = [
            line
            for line in text.splitlines()
            if not _PAGE_NUMBER.match(line) and line not in header_lines
        ]
        if sum(len(line.split()) for line in lines) < 25:
            continue  # cover pages, blank pages
        yield Section(f"p{index}", f"PDF page {index}", lines)


def sections_to_passages(source_id: str, sections: Iterator[Section]) -> list[Passage]:
    passages = []
    for section in sections:
        for n, chunk in enumerate(chunk_lines(section.lines), start=1):
            passages.append(
                Passage(
                    id=f"{source_id}:{section.key}:{n}",
                    source_id=source_id,
                    location=section.location,
                    text=chunk,
                )
            )
    return passages


# ---------------------------------------------------------------------------
# Preparation (download + parse + write passages.jsonl)
# ---------------------------------------------------------------------------

_PDF_HEADERS = {
    "nist-ai-rmf": ("NIST AI 100-1 AI RMF 1.0",),
    "nist-ai-600-1": ("NIST AI 600-1",),
}


def download_raw(entry: dict, raw_dir: Path, timeout_s: float = 60.0) -> tuple[bytes, bool]:
    """Return raw bytes (downloading if needed) and whether the checksum matches the manifest."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / entry["raw_file"]
    if not path.exists():
        headers = {"User-Agent": "research-assistant-corpus/0.1 (+local research demo)"}
        response = httpx.get(entry["url"], headers=headers, timeout=timeout_s, follow_redirects=True)
        response.raise_for_status()
        path.write_bytes(response.content)
    raw = path.read_bytes()
    return raw, hashlib.sha256(raw).hexdigest() == entry.get("sha256")


def prepare_corpus(corpus_dir: Path) -> dict[str, int]:
    manifest = load_manifest(corpus_dir)
    raw_dir = corpus_dir / "raw"
    all_passages: list[Passage] = []
    counts: dict[str, int] = {}
    for entry in manifest["sources"]:
        raw, checksum_ok = download_raw(entry, raw_dir)
        if not checksum_ok:
            print(
                f"warning: {entry['id']} checksum differs from manifest; the upstream file "
                "may have changed and passage ids may not match the committed evaluation set."
            )
        if entry["format"] == "eurlex_html":
            sections = parse_eurlex_html(raw)
        elif entry["format"] == "pdf":
            sections = parse_pdf(raw, _PDF_HEADERS.get(entry["id"], ()))
        else:
            raise ValueError(f"unknown format {entry['format']!r}")
        passages = sections_to_passages(entry["id"], sections)
        counts[entry["id"]] = len(passages)
        all_passages.extend(passages)

    ids = [p.id for p in all_passages]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate passage ids produced while parsing the corpus")
    with (corpus_dir / PASSAGES_FILE).open("w", encoding="utf-8") as fh:
        for passage in all_passages:
            fh.write(passage.model_dump_json() + "\n")
    return counts
