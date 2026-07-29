"""Thin HTTP client for LiveAvatar's session-token endpoint
(docs.liveavatar.com, part of the HeyGen ecosystem). Mirrors
elevenlabs_client.py's shape -- one stateless function per call.

Minting a session token is the only step that needs our secret API key --
everything after that (connecting the WebRTC session, calling
`repeatAudio()`) happens client-side using the short-lived token this
returns, never the API key itself. Confirmed against LiveAvatar's own
reference integration (apps/demo/app/api/start-lite-session/route.ts in
github.com/heygen-com/liveavatar-web-sdk).
"""

import requests

from migrantbuddy.config import (
    LIVEAVATAR_API_KEY,
    LIVEAVATAR_API_URL,
    LIVEAVATAR_AVATAR_ID,
    LIVEAVATAR_IS_SANDBOX,
)


def create_session_token(
    *,
    avatar_id: str = LIVEAVATAR_AVATAR_ID,
    is_sandbox: bool = LIVEAVATAR_IS_SANDBOX,
    api_key: str = LIVEAVATAR_API_KEY,
    api_url: str = LIVEAVATAR_API_URL,
    timeout: float = 30.0,
) -> dict:
    """Returns {"session_token": ..., "session_id": ...} for a LITE-mode
    session -- tts/routes.py's /session/start hands this straight to the
    frontend, which uses session_token to construct a LiveAvatarSession.
    """
    response = requests.post(
        f"{api_url}/v1/sessions/token",
        headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
        json={"mode": "LITE", "avatar_id": avatar_id, "is_sandbox": is_sandbox},
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()["data"]
    return {"session_token": data["session_token"], "session_id": data["session_id"]}
