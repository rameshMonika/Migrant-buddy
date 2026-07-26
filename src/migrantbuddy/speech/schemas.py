"""Message shape sent over /transcribe/ws -- one per finalized segment."""

from pydantic import BaseModel


class TranscriptMessage(BaseModel):
    text: str
    final: bool
