"""Hybrid retrieval: dense (Chroma + BGE-M3), BM25, reciprocal-rank-fusion
hybrid, and SEA-LION-E5 rerank. Extracted from notebooks/04_retrieval.ipynb
(dense/BM25/hybrid) and notebooks/05_generation.ipynb's reranker (the settled
SEA-LION choice from 04b_retrieval_sealion_rerank.ipynb -- see CLAUDE.md).

A single `Retriever` loads the embedding model, reranker model, and BM25
index once and serves many queries -- the notebooks rebuild all of this as
module-level globals per run, which doesn't fit a long-lived API server.
"""

import re
from dataclasses import dataclass
from typing import Sequence

import chromadb
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from migrantbuddy.config import EMBEDDING_MODEL_NAME, RERANKER_MODEL_NAME
from migrantbuddy.indexing import Chunk
from migrantbuddy.observability import observe


@dataclass
class RetrievalResult:
    chunk_id: str
    score: float  # higher is always better, across every method here


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def build_bm25_index(chunks: Sequence[Chunk]) -> BM25Okapi:
    return BM25Okapi([tokenize(chunk.text) for chunk in chunks])


def reciprocal_rank_fusion(*ranked_lists: Sequence[RetrievalResult], k: int = 60) -> list[RetrievalResult]:
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, result in enumerate(ranked, start=1):
            scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + 1.0 / (k + rank)

    fused = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [RetrievalResult(chunk_id=chunk_id, score=score) for chunk_id, score in fused]


class Retriever:
    def __init__(
        self,
        chunks: Sequence[Chunk],
        collection: chromadb.Collection,
        *,
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
        reranker_model_name: str = RERANKER_MODEL_NAME,
    ):
        self.chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        self.collection = collection
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.reranker_model = SentenceTransformer(reranker_model_name)
        self.bm25 = build_bm25_index(chunks)

    @observe()
    def dense(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)
        results = self.collection.query(query_embeddings=[query_embedding.tolist()], n_results=top_k)
        # Chroma returns distance (lower is better) -- negate so every method
        # in this class shares a "higher score is better" convention.
        return [
            RetrievalResult(chunk_id=chunk_id, score=-distance)
            for chunk_id, distance in zip(results["ids"][0], results["distances"][0])
        ]

    @observe()
    def bm25_search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        chunk_ids = list(self.chunks_by_id)
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(zip(chunk_ids, scores), key=lambda item: item[1], reverse=True)[:top_k]
        return [RetrievalResult(chunk_id=chunk_id, score=float(score)) for chunk_id, score in ranked]

    @observe()
    def hybrid(self, query: str, top_k: int = 5, *, candidate_k: int = 10) -> list[RetrievalResult]:
        dense_results = self.dense(query, top_k=candidate_k)
        bm25_results = self.bm25_search(query, top_k=candidate_k)
        return reciprocal_rank_fusion(dense_results, bm25_results)[:top_k]

    @observe()
    def rerank(self, query: str, candidates: Sequence[RetrievalResult], top_k: int = 5) -> list[RetrievalResult]:
        """SEA-LION-E5 reranking -- a bi-encoder (cosine similarity), not a
        cross-encoder like a typical reranker. "STS" is the only named
        prompt documented on the model card; used for both query and
        passage since this only needs a symmetric similarity score, not
        asymmetric retrieval.
        """
        query_embedding = self.reranker_model.encode(query, convert_to_numpy=True, prompt_name="STS")
        candidate_texts = [self.chunks_by_id[c.chunk_id].text for c in candidates]
        candidate_embeddings = self.reranker_model.encode(
            candidate_texts, convert_to_numpy=True, prompt_name="STS"
        )
        similarities = candidate_embeddings @ query_embedding / (
            np.linalg.norm(candidate_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        reranked = sorted(zip(candidates, similarities), key=lambda item: item[1], reverse=True)
        return [
            RetrievalResult(chunk_id=candidate.chunk_id, score=float(score))
            for candidate, score in reranked[:top_k]
        ]

    @observe()
    def hybrid_rerank(self, query: str, top_k: int = 5, *, candidate_k: int = 10) -> list[RetrievalResult]:
        candidates = self.hybrid(query, top_k=candidate_k)
        return self.rerank(query, candidates, top_k=top_k)
