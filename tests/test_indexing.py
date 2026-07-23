import json
from pathlib import Path

import numpy as np
import pytest

from migrantbuddy.indexing.chunking import Chunk, chunk_document
from migrantbuddy.indexing.service import (
    build_chroma_collection,
    embed_chunks,
    index_documents,
    save_chunks,
    save_embeddings,
    validate_embeddings,
)
from migrantbuddy.ingestion import IngestedDocument


class FakeEmbeddingModel:
    """Stands in for SentenceTransformer -- returns a deterministic vector per
    text so tests don't need to download/load the real ~2GB BGE-M3 model.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name

    def encode(self, texts: list[str], convert_to_numpy: bool = True) -> np.ndarray:
        # Deterministic, distinct-per-text vector: hash of text -> fixed seed.
        return np.array([[float(len(text) % 7), float(i)] for i, text in enumerate(texts)])


def make_document(text: str, *, document_id: str = "doc-1") -> IngestedDocument:
    return IngestedDocument(
        document_id=document_id,
        url=f"https://example.com/{document_id}",
        title="Doc Title",
        authority="MOM",
        category="salary",
        content_type="official_guidance",
        language="en",
        source_type="html",
        fetched_at="2026-01-01T00:00:00+00:00",
        text=text,
    )


def make_chunk(chunk_id: str = "doc-1::chunk-0", text: str = "some chunk text") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id="doc-1",
        url="https://example.com/doc-1",
        heading_path="Section",
        text=text,
        token_count=len(text) // 4,
    )


def test_embed_chunks_returns_one_vector_per_chunk(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("migrantbuddy.indexing.service.SentenceTransformer", FakeEmbeddingModel)
    chunks = [make_chunk("c1", "alpha"), make_chunk("c2", "beta")]

    embeddings = embed_chunks(chunks, model_name="fake-model")

    assert embeddings.shape[0] == 2


def test_validate_embeddings_passes_for_healthy_array():
    embeddings = np.random.rand(3, 1024)

    validate_embeddings(embeddings, expected_dim=1024)  # should not raise


def test_validate_embeddings_rejects_wrong_dimension():
    embeddings = np.random.rand(3, 512)

    with pytest.raises(ValueError, match="Unexpected embedding dimension"):
        validate_embeddings(embeddings, expected_dim=1024)


def test_validate_embeddings_rejects_nans():
    embeddings = np.random.rand(3, 1024)
    embeddings[0, 0] = np.nan

    with pytest.raises(ValueError, match="Found NaNs"):
        validate_embeddings(embeddings, expected_dim=1024)


def test_save_chunks_round_trips_through_json(tmp_path: Path):
    chunks = [make_chunk("c1", "alpha"), make_chunk("c2", "beta")]
    output_path = tmp_path / "chunks.json"

    result_path = save_chunks(chunks, output_path)

    assert result_path == output_path
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert [c["chunk_id"] for c in saved] == ["c1", "c2"]


def test_save_embeddings_writes_npy_and_chunk_ids(tmp_path: Path):
    chunks = [make_chunk("c1", "alpha"), make_chunk("c2", "beta")]
    embeddings = np.array([[1.0, 2.0], [3.0, 4.0]])
    embeddings_path = tmp_path / "embeddings.npy"
    chunk_ids_path = tmp_path / "chunk_ids.json"

    save_embeddings(chunks, embeddings, embeddings_path, chunk_ids_path)

    loaded = np.load(embeddings_path)
    assert np.array_equal(loaded, embeddings)
    assert json.loads(chunk_ids_path.read_text(encoding="utf-8")) == ["c1", "c2"]


def test_build_chroma_collection_indexes_all_chunks(tmp_path: Path):
    chunks = [make_chunk("c1", "alpha"), make_chunk("c2", "beta")]
    embeddings = np.array([[1.0, 2.0], [3.0, 4.0]])

    collection = build_chroma_collection(chunks, embeddings, chroma_dir=tmp_path, collection_name="test_collection")

    assert collection.count() == 2


def test_build_chroma_collection_rebuilds_fresh_on_second_call(tmp_path: Path):
    chunks_v1 = [make_chunk("c1", "alpha"), make_chunk("c2", "beta"), make_chunk("c3", "gamma")]
    embeddings_v1 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    build_chroma_collection(chunks_v1, embeddings_v1, chroma_dir=tmp_path, collection_name="test_collection")

    chunks_v2 = [make_chunk("c1", "alpha")]
    embeddings_v2 = np.array([[1.0, 2.0]])
    collection = build_chroma_collection(chunks_v2, embeddings_v2, chroma_dir=tmp_path, collection_name="test_collection")

    # Rebuilt fresh -- stale entries from v1 (c2, c3) must not linger.
    assert collection.count() == 1


def test_index_documents_runs_chunk_embed_and_index_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("migrantbuddy.indexing.service.SentenceTransformer", FakeEmbeddingModel)
    document = make_document("## Section\n\nSome body text about salary rules.")

    result = index_documents([document], chroma_dir=tmp_path, collection_name="test_collection")

    assert len(result.chunks) > 0
    assert result.embeddings.shape[0] == len(result.chunks)
    assert result.collection.count() == len(result.chunks)
