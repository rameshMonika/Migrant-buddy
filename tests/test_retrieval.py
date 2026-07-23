import numpy as np
import pytest

from migrantbuddy.indexing import Chunk
from migrantbuddy.retrieval.service import (
    Retriever,
    RetrievalResult,
    build_bm25_index,
    reciprocal_rank_fusion,
    tokenize,
)


class FakeModel:
    """Stands in for SentenceTransformer -- deterministic, non-zero vectors
    so tests don't need the real BGE-M3 / SEA-LION-E5 models loaded.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name

    def encode(self, texts, convert_to_numpy: bool = True, prompt_name: str | None = None):
        if isinstance(texts, str):
            return self._vector(texts)
        return np.array([self._vector(t) for t in texts])

    @staticmethod
    def _vector(text: str) -> np.ndarray:
        h = sum(ord(c) for c in text) % 97 + 1
        return np.array([float(h), float(len(text) % 13 + 1)])


class FakeCollection:
    """Stands in for a chromadb.Collection -- ignores the actual query
    embedding and just returns ids in a fixed order with increasing
    distance, so tests can check the distance-to-score conversion without
    a real Chroma index.
    """

    def __init__(self, ids_in_order: list[str]):
        self.ids_in_order = ids_in_order

    def query(self, query_embeddings, n_results: int):
        ids = self.ids_in_order[:n_results]
        distances = [float(i) for i in range(len(ids))]
        return {"ids": [ids], "distances": [distances]}


def make_chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id="doc-1",
        url="https://example.com/doc-1",
        heading_path="Section",
        text=text,
        token_count=len(text) // 4,
    )


CHUNKS = [
    make_chunk("c1", "overtime pay rules and calculation"),
    make_chunk("c2", "salary payment schedule and timing"),
    make_chunk("c3", "medical insurance coverage requirements"),
]


def test_tokenize_lowercases_and_splits_on_word_boundaries():
    assert tokenize("Overtime Pay!") == ["overtime", "pay"]


def test_build_bm25_index_scores_lexically_matching_chunk_highest():
    bm25 = build_bm25_index(CHUNKS)

    scores = bm25.get_scores(tokenize("overtime pay"))

    assert scores[0] > scores[1]
    assert scores[0] > scores[2]


def test_reciprocal_rank_fusion_favors_items_ranked_highly_in_both_lists():
    list_a = [RetrievalResult("c1", 0.9), RetrievalResult("c2", 0.5)]
    list_b = [RetrievalResult("c2", 5.0), RetrievalResult("c1", 1.0)]

    fused = reciprocal_rank_fusion(list_a, list_b)

    # c1 is #1 in list_a and #2 in list_b; c2 is #2 in list_a and #1 in
    # list_b -- symmetric ranks, so RRF should score them equally.
    assert {r.chunk_id for r in fused} == {"c1", "c2"}
    assert fused[0].score == pytest.approx(fused[1].score)


def test_reciprocal_rank_fusion_ignores_raw_score_only_rank_matters():
    list_a = [RetrievalResult("c1", 1000.0), RetrievalResult("c2", 0.001)]

    fused = reciprocal_rank_fusion(list_a)

    assert fused[0].chunk_id == "c1"


@pytest.fixture
def retriever(monkeypatch: pytest.MonkeyPatch) -> Retriever:
    monkeypatch.setattr("migrantbuddy.retrieval.service.SentenceTransformer", FakeModel)
    collection = FakeCollection(["c1", "c2", "c3"])
    return Retriever(CHUNKS, collection, embedding_model_name="fake", reranker_model_name="fake")


def test_dense_converts_chroma_distance_to_higher_is_better_score(retriever: Retriever):
    results = retriever.dense("overtime pay", top_k=3)

    assert [r.chunk_id for r in results] == ["c1", "c2", "c3"]
    # distance 0 -> score 0; distance 1 -> score -1; higher score = closer.
    assert results[0].score > results[1].score > results[2].score


def test_bm25_search_ranks_lexically_matching_chunk_first(retriever: Retriever):
    results = retriever.bm25_search("overtime pay", top_k=3)

    assert results[0].chunk_id == "c1"


def test_hybrid_combines_dense_and_bm25(retriever: Retriever):
    results = retriever.hybrid("overtime pay", top_k=3)

    assert {r.chunk_id for r in results} == {"c1", "c2", "c3"}
    assert results[0].chunk_id == "c1"  # top-ranked in both dense (fake) and bm25


def test_hybrid_rerank_returns_top_k_results(retriever: Retriever):
    results = retriever.hybrid_rerank("overtime pay", top_k=2)

    assert len(results) == 2


def test_rerank_orders_by_cosine_similarity_to_query(monkeypatch: pytest.MonkeyPatch):
    vectors = {
        "query": np.array([1.0, 0.0]),
        "match": np.array([1.0, 0.0]),  # same direction as query -> similarity 1.0
        "orthogonal": np.array([0.0, 1.0]),  # perpendicular -> similarity 0.0
        "opposite": np.array([-1.0, 0.0]),  # opposite direction -> similarity -1.0
    }

    class ControlledModel:
        def __init__(self, model_name: str):
            pass

        def encode(self, texts, convert_to_numpy=True, prompt_name=None):
            if isinstance(texts, str):
                return vectors[texts]
            return np.array([vectors[t] for t in texts])

    monkeypatch.setattr("migrantbuddy.retrieval.service.SentenceTransformer", ControlledModel)
    chunks = [
        make_chunk("c-match", "match"),
        make_chunk("c-opposite", "opposite"),
        make_chunk("c-orthogonal", "orthogonal"),
    ]
    collection = FakeCollection([c.chunk_id for c in chunks])
    retriever = Retriever(chunks, collection, embedding_model_name="fake", reranker_model_name="fake")
    candidates = [RetrievalResult("c-opposite", 0.0), RetrievalResult("c-orthogonal", 0.0), RetrievalResult("c-match", 0.0)]

    results = retriever.rerank("query", candidates, top_k=3)

    assert [r.chunk_id for r in results] == ["c-match", "c-orthogonal", "c-opposite"]
