"""Embed chunks and build the Chroma index. Extracted from
notebooks/03_embedding.ipynb (embedding) and notebooks/04_retrieval.ipynb's
Step 2 (Chroma collection build) -- indexing owns building the searchable
index; retrieval (not yet extracted) owns querying it.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

from migrantbuddy.config import CHROMA_COLLECTION_NAME, CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL_NAME
from migrantbuddy.indexing.chunking import Chunk, chunk_documents
from migrantbuddy.ingestion import IngestedDocument


def embed_chunks(chunks: Sequence[Chunk], *, model_name: str = EMBEDDING_MODEL_NAME) -> np.ndarray:
    model = SentenceTransformer(model_name)
    texts = [chunk.text for chunk in chunks]
    return model.encode(texts, convert_to_numpy=True)


def validate_embeddings(embeddings: np.ndarray, *, expected_dim: int = 1024) -> None:
    if embeddings.shape[1] != expected_dim:
        raise ValueError(f"Unexpected embedding dimension: {embeddings.shape[1]} (expected {expected_dim})")
    if np.isnan(embeddings).any():
        raise ValueError("Found NaNs in embeddings")


def save_chunks(chunks: Sequence[Chunk], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps([asdict(chunk) for chunk in chunks], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_path


def save_embeddings(
    chunks: Sequence[Chunk], embeddings: np.ndarray, embeddings_path: Path, chunk_ids_path: Path
) -> tuple[Path, Path]:
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_path, embeddings)
    chunk_ids_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_ids_path.write_text(
        json.dumps([chunk.chunk_id for chunk in chunks], indent=2),
        encoding="utf-8",
    )
    return embeddings_path, chunk_ids_path


def build_chroma_collection(
    chunks: Sequence[Chunk],
    embeddings: np.ndarray,
    *,
    chroma_dir: Path,
    collection_name: str = CHROMA_COLLECTION_NAME,
) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(chroma_dir))

    # Rebuild fresh each run rather than appending -- collection.add() isn't
    # idempotent, and re-running with a changed corpus (e.g. more/fewer
    # source pages) would either error on duplicate IDs or leave stale
    # entries from a previous corpus version.
    if collection_name in [c.name for c in client.list_collections()]:
        client.delete_collection(name=collection_name)
    collection = client.create_collection(name=collection_name)

    collection.add(
        ids=[chunk.chunk_id for chunk in chunks],
        embeddings=embeddings.tolist(),
        documents=[chunk.text for chunk in chunks],
        metadatas=[
            {
                "document_id": chunk.document_id,
                "url": chunk.url,
                "heading_path": chunk.heading_path,
            }
            for chunk in chunks
        ],
    )
    return collection


@dataclass(eq=False)  # embeddings is a numpy array -- default __eq__ would raise on comparison
class IndexResult:
    chunks: list[Chunk]
    embeddings: np.ndarray
    collection: chromadb.Collection


def index_documents(
    documents: Sequence[IngestedDocument],
    *,
    chroma_dir: Path,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    embedding_model_name: str = EMBEDDING_MODEL_NAME,
    collection_name: str = CHROMA_COLLECTION_NAME,
) -> IndexResult:
    """Chunk -> embed -> build Chroma collection, end to end."""
    chunks = chunk_documents(documents, chunk_size=chunk_size, overlap=overlap)
    embeddings = embed_chunks(chunks, model_name=embedding_model_name)
    collection = build_chroma_collection(
        chunks, embeddings, chroma_dir=chroma_dir, collection_name=collection_name
    )
    return IndexResult(chunks=chunks, embeddings=embeddings, collection=collection)
