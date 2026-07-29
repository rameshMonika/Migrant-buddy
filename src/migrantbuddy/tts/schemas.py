"""Request/response shapes for the tts service's two endpoints -- minting a
LiveAvatar session token and generating one sentence's ElevenLabs audio for
the frontend to feed into that session via `repeatAudio()`.
"""

from pydantic import BaseModel


class SessionStartResponse(BaseModel):
    session_token: str
    session_id: str


class SpeakRequest(BaseModel):
    text: str
    previous_text: str | None = None


class SpeakResponse(BaseModel):
    audio_base64: str
