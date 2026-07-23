import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from migrantbuddy.api.routes import get_conversation_service, get_rate_limiter, router
from migrantbuddy.rag import ConversationResult


class FakeConversationService:
    def __init__(self, answer_text: str = "the answer", sources: list[str] | None = None):
        self.answer_text = answer_text
        self.sources = sources or []
        self.last_thread_id = None
        self.last_message = None

    def answer(self, thread_id: str, message: str) -> ConversationResult:
        self.last_thread_id = thread_id
        self.last_message = message
        return ConversationResult(thread_id=thread_id, message=message, answer=self.answer_text, sources=self.sources)


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)
    fake_service = FakeConversationService(sources=["https://example.com/a"])
    app.dependency_overrides[get_conversation_service] = lambda: fake_service
    return TestClient(app), fake_service


def test_health_returns_200_ok(client):
    test_client, _ = client

    response = test_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_returns_200_with_answer_and_sources(client):
    test_client, _ = client

    response = test_client.post(
        "/chat", json={"message": "How much overtime pay am I entitled to?", "thread_id": "thread-1"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["message"] == "How much overtime pay am I entitled to?"
    assert body["thread_id"] == "thread-1"
    assert body["answer"] == "the answer"
    assert body["sources"] == ["https://example.com/a"]


def test_chat_passes_thread_id_and_message_to_service(client):
    test_client, fake_service = client

    test_client.post("/chat", json={"message": "query", "thread_id": "thread-abc"})

    assert fake_service.last_thread_id == "thread-abc"
    assert fake_service.last_message == "query"


def test_chat_rejects_missing_message(client):
    test_client, _ = client

    response = test_client.post("/chat", json={"thread_id": "thread-1"})

    assert response.status_code == 422


def test_chat_rejects_missing_thread_id(client):
    test_client, _ = client

    response = test_client.post("/chat", json={"message": "query"})

    assert response.status_code == 422


# --- rate limiting ---


class FakeRateLimiter:
    def __init__(self, allowed: bool):
        self.allowed = allowed
        self.last_key = None

    def is_allowed(self, key: str) -> bool:
        self.last_key = key
        return self.allowed


def test_chat_succeeds_when_no_rate_limiter_configured(client):
    # The default `client` fixture never sets get_rate_limiter, so it
    # should resolve to None (disabled) and behave exactly as before.
    test_client, _ = client

    response = test_client.post("/chat", json={"message": "query", "thread_id": "thread-1"})

    assert response.status_code == 200


def test_chat_returns_429_when_rate_limiter_denies():
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_conversation_service] = lambda: FakeConversationService()
    app.dependency_overrides[get_rate_limiter] = lambda: FakeRateLimiter(allowed=False)
    test_client = TestClient(app)

    response = test_client.post("/chat", json={"message": "query", "thread_id": "thread-1"})

    assert response.status_code == 429


def test_chat_succeeds_when_rate_limiter_allows():
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_conversation_service] = lambda: FakeConversationService()
    app.dependency_overrides[get_rate_limiter] = lambda: FakeRateLimiter(allowed=True)
    test_client = TestClient(app)

    response = test_client.post("/chat", json={"message": "query", "thread_id": "thread-1"})

    assert response.status_code == 200
