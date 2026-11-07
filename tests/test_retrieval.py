from __future__ import annotations

from research_assistant.retrieval import BM25, PassageIndex, tokenize


def test_tokenize_drops_stopwords_keeps_numbers():
    tokens = tokenize("What are the obligations under Article 53?")
    assert "obligations" in tokens
    assert "53" in tokens
    assert "the" not in tokens and "are" not in tokens


def test_bm25_ranks_matching_document_first():
    bm25 = BM25()
    bm25.add(
        [
            tokenize("technical documentation for AI models"),
            tokenize("unrelated content about cooking"),
        ]
    )
    scores = bm25.scores(tokenize("technical documentation"))
    assert scores[0] > scores[1]


def test_index_search_finds_relevant_passage(mini_index):
    hits = mini_index.search("technical documentation obligations providers", k=2)
    assert hits
    assert hits[0][0].id == "doc-a:art-1:1"


def test_index_search_can_filter_by_source(mini_index):
    hits = mini_index.search("risk management functions", k=3, source_ids={"doc-b"})
    assert all(p.source_id == "doc-b" for p, _ in hits)


def test_index_get_returns_passage_or_none(mini_index):
    assert mini_index.get("doc-a:art-1:1") is not None
    assert mini_index.get("missing") is None


def test_index_degrades_to_bm25_when_embedder_fails(mini_passages, mini_sources):
    class BrokenEmbedder:
        name = "broken"

        def embed_documents(self, texts):
            raise RuntimeError("server down")

        def embed_query(self, text):
            raise RuntimeError("server down")

    titles = {sid: s.title for sid, s in mini_sources.items()}
    index = PassageIndex(mini_passages, titles, embedder=BrokenEmbedder(), cache_dir=None)
    assert not index.dense_available
    assert index.warnings
    assert index.search("documentation", k=1)  # still works via BM25
