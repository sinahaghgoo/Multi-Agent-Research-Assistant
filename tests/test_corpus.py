from __future__ import annotations

import pytest

from research_assistant.config import Settings
from research_assistant.corpus import load_passages, local_sources
from research_assistant.text import chunk_lines, normalize

CORPUS = Settings().corpus_dir
has_corpus = (CORPUS / "passages.jsonl").exists()
requires_corpus = pytest.mark.skipif(not has_corpus, reason="corpus not prepared")


def test_normalize_fixes_ligatures_and_hyphen_breaks():
    assert "efficient" in normalize("eﬃcient")  # ﬃ ligature
    assert "documentation" in normalize("documen-\ntation")


def test_chunk_lines_respects_max_words():
    lines = [f"word{i} " * 50 for i in range(10)]
    chunks = chunk_lines(lines, max_words=120)
    assert all(len(c.split()) <= 120 for c in chunks)
    assert len(chunks) > 1


def test_chunk_lines_merges_short_tail():
    chunks = chunk_lines(["a " * 200, "tiny tail"], max_words=220, min_words=40)
    assert len(chunks) == 1


@requires_corpus
def test_corpus_passage_ids_are_unique_and_located():
    passages = load_passages(CORPUS)
    ids = [p.id for p in passages]
    assert len(ids) == len(set(ids))
    assert len(passages) > 500
    assert all(p.location for p in passages)
    assert all(p.text.strip() for p in passages)


@requires_corpus
def test_corpus_sources_match_passages():
    passages = load_passages(CORPUS)
    sources = local_sources(CORPUS)
    assert {p.source_id for p in passages} <= set(sources)
    # The EU AI Act must contribute Article-level passages.
    assert any(p.id.startswith("eu-ai-act:art-53") for p in passages)
