# Three independently runnable images (rag/speech/tts) from one shared
# dependency layer -- mirrors the RAG/Whisper/TTS service split in
# src/migrantbuddy/api vs src/migrantbuddy/speech vs src/migrantbuddy/tts.
# Build with `--target rag`, `--target speech`, or `--target tts`.
#
# `pip install -e .` (editable) is deliberate, not a shortcut: config.py
# computes PROJECT_ROOT as Path(__file__).resolve().parents[2], which only
# resolves to /app if migrantbuddy still lives under /app/src/migrantbuddy
# (as it does here) rather than being copied into site-packages by a normal
# `pip install .`. This matches the local venv's editable install exactly,
# so DATA_DIR/CHROMA_DIR resolve the same way in and out of Docker.

FROM python:3.12-slim AS base

WORKDIR /app

COPY pyproject.toml ./
COPY src/ ./src/

RUN pip install --no-cache-dir -e .

FROM base AS rag

EXPOSE 8000
CMD ["uvicorn", "migrantbuddy.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM base AS speech

# faster-whisper shells out to ffmpeg to decode whatever audio format the
# browser's MediaRecorder sends -- not a pip package, a system dependency.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

EXPOSE 8002
CMD ["uvicorn", "migrantbuddy.speech.main:app", "--host", "0.0.0.0", "--port", "8002"]

FROM base AS tts

# No local model to load (ElevenLabs/LiveAvatar are remote APIs) -- no extra
# system deps needed beyond the shared base layer.
EXPOSE 8003
CMD ["uvicorn", "migrantbuddy.tts.main:app", "--host", "0.0.0.0", "--port", "8003"]
