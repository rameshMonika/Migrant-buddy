import pytest
from pydantic import ValidationError

from migrantbuddy.api.schemas import ChatRequest


def test_chat_request_requires_message():
    with pytest.raises(ValidationError):
        ChatRequest(thread_id="thread-1")


def test_chat_request_requires_thread_id():
    with pytest.raises(ValidationError):
        ChatRequest(message="How much overtime pay am I entitled to?")


def test_chat_request_accepts_message_and_thread_id():
    request = ChatRequest(message="How much overtime pay am I entitled to?", thread_id="thread-1")

    assert request.message == "How much overtime pay am I entitled to?"
    assert request.thread_id == "thread-1"
