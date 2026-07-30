import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from migrantbuddy.api.routes import get_conversation_service, get_rate_limiter, router
from migrantbuddy.rag.service import PreparedTurn


def parse_sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for raw_event in text.split("\n\n"):
        if not raw_event.strip():
            continue
        event_name = None
        data = None
        for line in raw_event.splitlines():
            if line.startswith("event: "):
                event_name = line[len("event: ") :]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: ") :])
        if event_name is not None:
            events.append((event_name, data))
    return events


class FakeConversationService:
    def __init__(self, sources: list[str] | None = None):
        self.sources = sources or []
        self.last_thread_id = None
        self.last_message = None
        self.finalized_prepared = None
        self.finalized_answer = None

    def prepare_turn(self, thread_id: str, message: str) -> PreparedTurn:
        self.last_thread_id = thread_id
        self.last_message = message
        return PreparedTurn(
            config={"configurable": {"thread_id": thread_id}},
            system_prompt="system prompt",
            user_prompt=message,
            sources=self.sources,
        )

    def finalize_turn(self, prepared: PreparedTurn, answer: str) -> None:
        self.finalized_prepared = prepared
        self.finalized_answer = answer


def fake_stream_answer_yielding(*chunks: str):
    def fake(system_prompt, user_prompt, result_info=None):
        if result_info is not None:
            result_info["truncated"] = False
        yield from chunks

    return fake


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch):
    app = FastAPI()
    app.include_router(router)
    fake_service = FakeConversationService(sources=["https://example.com/a"])
    app.dependency_overrides[get_conversation_service] = lambda: fake_service
    # Default stream -- individual tests override this via the same
    # monkeypatch instance when they need different chunks/truncation.
    monkeypatch.setattr(
        "migrantbuddy.api.routes.stream_answer", fake_stream_answer_yielding("the answer")
    )
    return TestClient(app), fake_service


def test_health_returns_200_ok(client):
    test_client, _ = client

    response = test_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_streams_sources_then_tokens_then_done(client, monkeypatch: pytest.MonkeyPatch):
    test_client, _ = client
    monkeypatch.setattr(
        "migrantbuddy.api.routes.stream_answer", fake_stream_answer_yielding("Hello", " world")
    )

    response = test_client.post(
        "/chat",
        json={"message": "How much overtime pay am I entitled to?", "thread_id": "thread-1"},
    )

    assert response.status_code == 200
    events = parse_sse(response.text)
    assert events[0] == ("sources", {"sources": ["https://example.com/a"]})
    assert events[1] == ("token", {"text": "Hello"})
    assert events[2] == ("token", {"text": " world"})
    assert events[-1][0] == "done"


def test_chat_passes_thread_id_and_message_to_prepare_turn(client):
    test_client, fake_service = client

    test_client.post("/chat", json={"message": "query", "thread_id": "thread-abc"})

    assert fake_service.last_thread_id == "thread-abc"
    assert fake_service.last_message == "query"


def test_chat_finalizes_turn_with_the_concatenated_answer(client, monkeypatch: pytest.MonkeyPatch):
    test_client, fake_service = client
    monkeypatch.setattr(
        "migrantbuddy.api.routes.stream_answer", fake_stream_answer_yielding("Hello", " world")
    )

    test_client.post("/chat", json={"message": "query", "thread_id": "thread-1"})

    assert fake_service.finalized_answer == "Hello world"


def test_chat_trims_the_saved_answer_but_not_the_streamed_tokens_when_truncated(
    client, monkeypatch: pytest.MonkeyPatch
):
    # The accepted tradeoff from real token streaming: already-streamed
    # tokens can't be un-shown, so only the *saved* (finalize_turn) copy
    # gets trimmed -- the live token events still show the raw, untrimmed
    # text exactly as generated.
    test_client, fake_service = client

    def fake_stream_answer(system_prompt, user_prompt, result_info=None):
        if result_info is not None:
            result_info["truncated"] = True
        yield "First complete sentence. Second cut off mid-wo"

    monkeypatch.setattr("migrantbuddy.api.routes.stream_answer", fake_stream_answer)

    response = test_client.post("/chat", json={"message": "query", "thread_id": "thread-1"})

    events = parse_sse(response.text)
    token_text = "".join(data["text"] for name, data in events if name == "token")
    assert token_text == "First complete sentence. Second cut off mid-wo"
    assert fake_service.finalized_answer == "First complete sentence."


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


def test_chat_returns_429_when_rate_limiter_denies(monkeypatch: pytest.MonkeyPatch):
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_conversation_service] = lambda: FakeConversationService()
    app.dependency_overrides[get_rate_limiter] = lambda: FakeRateLimiter(allowed=False)
    monkeypatch.setattr(
        "migrantbuddy.api.routes.stream_answer", fake_stream_answer_yielding("answer")
    )
    test_client = TestClient(app)

    response = test_client.post("/chat", json={"message": "query", "thread_id": "thread-1"})

    assert response.status_code == 429


def test_chat_succeeds_when_rate_limiter_allows(monkeypatch: pytest.MonkeyPatch):
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_conversation_service] = lambda: FakeConversationService()
    app.dependency_overrides[get_rate_limiter] = lambda: FakeRateLimiter(allowed=True)
    monkeypatch.setattr(
        "migrantbuddy.api.routes.stream_answer", fake_stream_answer_yielding("answer")
    )
    test_client = TestClient(app)

    response = test_client.post("/chat", json={"message": "query", "thread_id": "thread-1"})

    assert response.status_code == 200
