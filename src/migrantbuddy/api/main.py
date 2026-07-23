"""FastAPI app factory. Loads the already-built chunks + Chroma collection
at startup (this notebook pipeline's job, not this module's -- see
indexing.service.index_documents) and wires up a ConversationService once,
reused across every request. Conversation state itself (message history,
running summary) lives in the graph's checkpointer, not in this service --
see rag/graph.py.
"""

import json

import chromadb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from migrantbuddy.api.rate_limit import RateLimiter
from migrantbuddy.api.routes import router
from migrantbuddy.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DIR,
    FRONTEND_ORIGIN,
    PROCESSED_DIR,
    RATE_LIMIT_ENABLED,
)
from migrantbuddy.indexing.chunking import Chunk
from migrantbuddy.rag import ConversationService
from migrantbuddy.retrieval import Retriever


def load_chunks() -> list[Chunk]:
    chunks_path = PROCESSED_DIR / "chunks.json"
    raw_chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    return [Chunk(**item) for item in raw_chunks]


def create_app() -> FastAPI:
    app = FastAPI(title="migrantBuddy")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[FRONTEND_ORIGIN],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    chunks = load_chunks()
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(name=CHROMA_COLLECTION_NAME)
    retriever = Retriever(chunks, collection)
    app.state.conversation_service = ConversationService(retriever)
    app.state.rate_limiter = RateLimiter() if RATE_LIMIT_ENABLED else None

    app.include_router(router)
    return app


app = create_app()
