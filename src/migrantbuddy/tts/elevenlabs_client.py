"""Thin HTTP client for ElevenLabs' text-to-speech API. Kept as a plain
function, not a class -- each call is a single stateless HTTP request,
mirroring generation/ollama_client.py's/vllm_client.py's shape.

Uses the plain (non-streaming) `/with-timestamps` endpoint, not the
streaming one -- LiveAvatar's LITE mode (tts/liveavatar_client.py) wants one
whole utterance's audio per `repeatAudio()` call, not a stream of chunks to
reassemble, and this matches LiveAvatar's own reference integration exactly
(apps/demo/app/api/elevenlabs-text-to-speech/route.ts in
github.com/heygen-com/liveavatar-web-sdk).

Requests raw PCM (`pcm_24000`: 16-bit signed little-endian mono, 24kHz) via
`output_format` -- LiveAvatar's LITE-mode audio ingest is hardcoded to
24kHz (see `splitPcm24kStringToChunks` in that same SDK), so this isn't a
free choice; it must match.
"""

import requests

from migrantbuddy.config import ELEVENLABS_API_KEY, ELEVENLABS_MODEL_ID, ELEVENLABS_VOICE_ID

PCM_OUTPUT_FORMAT = "pcm_24000"
PCM_SAMPLE_RATE = 24000


def generate_speech(
    text: str,
    *,
    voice_id: str = ELEVENLABS_VOICE_ID,
    model_id: str = ELEVENLABS_MODEL_ID,
    api_key: str = ELEVENLABS_API_KEY,
    previous_text: str | None = None,
    timeout: float = 60.0,
) -> str:
    """Returns the base64-encoded PCM_OUTPUT_FORMAT audio for `text` --
    tts/routes.py calls this once per completed sentence received from the
    frontend, passing the prior sentence as `previous_text` for prosody
    continuity, then hands the result straight to LiveAvatar's
    `session.repeatAudio()` on the frontend.
    """
    body = {"text": text, "model_id": model_id}
    if previous_text:
        body["previous_text"] = previous_text

    response = requests.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/with-timestamps",
        params={"output_format": PCM_OUTPUT_FORMAT},
        headers={"xi-api-key": api_key, "Content-Type": "application/json"},
        json=body,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()["audio_base64"]
