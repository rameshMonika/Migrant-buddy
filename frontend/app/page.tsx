"use client";

import { LiveAvatarSession, SessionEvent } from "@heygen/liveavatar-web-sdk";
import { useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import LiveAvatar from "./components/LiveAvatar";

type Message = {
  role: "user" | "assistant";
  content: string;
  sources?: string[];
};

type TranscriptMessage = {
  text: string;
  final: boolean;
};

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";
// Separate services from API_BASE_URL -- the RAG service (chat), the
// Whisper service (speech-to-text), and the TTS service (voice output) run
// as independent processes, so the frontend talks to each directly rather
// than one proxying the other.
const WHISPER_WS_URL = process.env.NEXT_PUBLIC_WHISPER_WS_URL ?? "ws://localhost:8002/transcribe/ws";
const TTS_API_BASE_URL = process.env.NEXT_PUBLIC_TTS_API_BASE_URL ?? "http://localhost:8003";

// A sentence is "complete" once it ends in one of these -- mirrors the
// sentence-terminator set generation/truncation.py uses server-side for
// the same reason (covers the scripts SEA-LION actually answers in).
const SENTENCE_END_CHARS = new Set([".", "!", "?", "。", "!", "?", "؟", "၊", "။"]);

function endsSentence(char: string): boolean {
  return SENTENCE_END_CHARS.has(char);
}

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // One thread_id per page load == one conversation. The backend's
  // checkpointer uses this to load/save this conversation's history and
  // running summary server-side -- the client only ever sends the latest
  // message, never the full history.
  const [threadId] = useState(() => crypto.randomUUID());
  const [isRecording, setIsRecording] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const webSocketRef = useRef<WebSocket | null>(null);

  // Text-to-speech + LiveAvatar (docs.liveavatar.com, LITE mode) -- LiveAvatar
  // renders the lip-synced video server-side from audio we generate via
  // ElevenLabs and push in via `repeatAudio()`; see tts/routes.py.
  const [isMuted, setIsMuted] = useState(false);
  const [isAvatarReady, setIsAvatarReady] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const liveAvatarSessionRef = useRef<LiveAvatarSession | null>(null);
  const liveAvatarSessionPromiseRef = useRef<Promise<void> | null>(null);
  // Text not yet forming a complete sentence -- flushed to TTS once it
  // ends in sentence-terminating punctuation, rather than waiting for the
  // whole answer (same reasoning as token-streaming the text itself).
  const pendingSentenceRef = useRef("");
  // The last complete sentence sent, passed to ElevenLabs as previous_text
  // for prosody continuity across separate per-sentence audio clips.
  const previousSentenceRef = useRef("");

  // Lazily creates and starts the LiveAvatar session on the first message of
  // the conversation -- kept idempotent (a single in-flight promise) since
  // sendMessage can call this on every turn. Not started eagerly on page
  // load: minting a session consumes LiveAvatar credits/quota even if the
  // visitor never sends a message.
  function ensureLiveAvatarSession(): Promise<void> {
    if (liveAvatarSessionPromiseRef.current) {
      return liveAvatarSessionPromiseRef.current;
    }

    const promise = (async () => {
      const response = await fetch(`${TTS_API_BASE_URL}/session/start`, { method: "POST" });
      if (!response.ok) {
        throw new Error(`Failed to start avatar session: ${response.status}`);
      }
      const { session_token: sessionToken } = await response.json();

      const session = new LiveAvatarSession(sessionToken, { voiceChat: false });
      session.on(SessionEvent.SESSION_STREAM_READY, () => {
        if (videoRef.current) {
          session.attach(videoRef.current);
          // Autoplay-with-audio can still be blocked by the browser even
          // though this session was kicked off inside a user gesture (the
          // WebRTC handshake itself takes long enough that some browsers'
          // user-activation window has since expired) -- explicit play()
          // with a caught rejection is a defensive fallback, not the
          // primary mechanism.
          void videoRef.current.play().catch(() => {});
        }
        setIsAvatarReady(true);
      });
      liveAvatarSessionRef.current = session;

      await session.start();
    })();

    liveAvatarSessionPromiseRef.current = promise;
    return promise;
  }

  async function sendSentenceToSpeak(sentence: string) {
    const trimmed = sentence.trim();
    if (!trimmed || isMuted) {
      return;
    }
    const textForThisSentence = trimmed;
    const previousText = previousSentenceRef.current;
    previousSentenceRef.current = textForThisSentence;

    try {
      await ensureLiveAvatarSession();
      const response = await fetch(`${TTS_API_BASE_URL}/speak`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: textForThisSentence,
          previous_text: previousText || undefined,
        }),
      });
      if (!response.ok) {
        throw new Error(`Failed to generate speech: ${response.status}`);
      }
      const { audio_base64: audioBase64 } = await response.json();
      liveAvatarSessionRef.current?.repeatAudio(audioBase64);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Voice output failed.");
    }
  }

  async function startRecording() {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const socket = new WebSocket(WHISPER_WS_URL);
      webSocketRef.current = socket;

      socket.onmessage = (event) => {
        const data: TranscriptMessage = JSON.parse(event.data);
        // Populate the input box rather than auto-sending -- transcription
        // errors are common, and getting a question wrong matters more here
        // (employment/legal rights) than in a casual chat app.
        setInput((previous) => (previous ? `${previous} ${data.text}` : data.text));
      };

      socket.onerror = () => {
        // Without this, a socket that errors instead of closing cleanly
        // left isTranscribing stuck true forever -- permanently disabling
        // the input/mic/Send buttons, since onclose (the only other place
        // that resets it) doesn't reliably fire in every error case.
        setError("Lost connection to the transcription service.");
        setIsTranscribing(false);
      };

      socket.onclose = () => {
        // The server closes the socket itself once it's replied to "stop"
        // with whatever was left in the buffer -- that's the signal that
        // transcription is done, not a fixed timeout on the client.
        setIsTranscribing(false);
      };

      await new Promise<void>((resolve, reject) => {
        socket.addEventListener("open", () => resolve(), { once: true });
        socket.addEventListener("error", () => reject(new Error("Couldn't reach the transcription service.")), {
          once: true,
        });
      });

      const mediaRecorder = new MediaRecorder(stream);

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0 && socket.readyState === WebSocket.OPEN) {
          socket.send(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        // Stop the mic indicator/permission light -- getUserMedia's stream
        // stays "live" until every track is explicitly stopped.
        stream.getTracks().forEach((track) => track.stop());
        setIsTranscribing(true);
        if (socket.readyState === WebSocket.OPEN) {
          socket.send("stop");
        } else {
          setIsTranscribing(false);
        }
      };

      mediaRecorderRef.current = mediaRecorder;
      // A timeslice streams chunks live as they're recorded instead of
      // buffering everything until stop().
      mediaRecorder.start(250);
      setIsRecording(true);
    } catch {
      setError("Couldn't access the microphone -- check your browser's permission settings.");
    }
  }

  function stopRecording() {
    mediaRecorderRef.current?.stop();
    setIsRecording(false);
  }

  function updateLastMessage(update: (message: Message) => Message) {
    setMessages((previous) => {
      const updated = [...previous];
      updated[updated.length - 1] = update(updated[updated.length - 1]);
      return updated;
    });
  }

  async function sendMessage() {
    const message = input.trim();
    if (!message || isLoading) {
      return;
    }

    // Kicked off (not awaited) inside this user-gesture-triggered handler --
    // starting the LiveAvatar session later, after the SSE round trip, would
    // push the WebRTC handshake even further from the gesture that's meant
    // to unlock autoplay-with-audio (see ensureLiveAvatarSession's comment).
    void ensureLiveAvatarSession();

    setMessages((previous) => [
      ...previous,
      { role: "user", content: message },
      // Empty placeholder, filled in as `token` events stream in below --
      // this is what makes the answer appear progressively instead of all
      // at once.
      { role: "assistant", content: "" },
    ]);
    setInput("");
    setIsLoading(true);
    setError(null);
    // Fresh sentence-buffering state for this turn -- leftovers from a
    // prior answer must not bleed into this one's TTS/previous_text.
    pendingSentenceRef.current = "";
    previousSentenceRef.current = "";

    try {
      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, thread_id: threadId }),
      });

      if (!response.ok || !response.body) {
        throw new Error(`Request failed: ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          break;
        }
        buffer += decoder.decode(value, { stream: true });

        // SSE events are separated by a blank line; each event is an
        // "event: <name>" line followed by a "data: <json>" line.
        let boundary = buffer.indexOf("\n\n");
        while (boundary !== -1) {
          const rawEvent = buffer.slice(0, boundary);
          buffer = buffer.slice(boundary + 2);

          const eventName = rawEvent.match(/^event: (.+)$/m)?.[1];
          const rawData = rawEvent.match(/^data: (.+)$/m)?.[1];

          if (eventName && rawData) {
            const data = JSON.parse(rawData);
            if (eventName === "sources") {
              updateLastMessage((last) => ({ ...last, sources: data.sources }));
            } else if (eventName === "token") {
              updateLastMessage((last) => ({ ...last, content: last.content + data.text }));

              // Flush a complete sentence to TTS as soon as it's ready,
              // rather than waiting for the whole answer -- same
              // reasoning as token-streaming the text itself.
              pendingSentenceRef.current += data.text;
              let sentenceEnd = -1;
              for (let i = 0; i < pendingSentenceRef.current.length; i++) {
                if (endsSentence(pendingSentenceRef.current[i])) {
                  sentenceEnd = i;
                }
              }
              if (sentenceEnd !== -1) {
                void sendSentenceToSpeak(pendingSentenceRef.current.slice(0, sentenceEnd + 1));
                pendingSentenceRef.current = pendingSentenceRef.current.slice(sentenceEnd + 1);
              }
            } else if (eventName === "done" && pendingSentenceRef.current.trim()) {
              // Flush whatever's left even without terminal punctuation --
              // otherwise a short final clause never gets spoken at all.
              void sendSentenceToSpeak(pendingSentenceRef.current);
              pendingSentenceRef.current = "";
            }
          }

          boundary = buffer.indexOf("\n\n");
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <main style={{ maxWidth: 640, margin: "0 auto", padding: 24 }}>
      <h1>migrantBuddy</h1>
      <p>
        Ask a question about Singapore employment rules (salary, hours, work permits,
        medical insurance).
      </p>

      <div style={{ display: "flex", justifyContent: "center", marginBottom: 16 }}>
        <LiveAvatar videoRef={videoRef} isReady={isAvatarReady} />
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 12, marginBottom: 16 }}>
        {messages.map((message, index) => (
          <div key={index} style={{ textAlign: message.role === "user" ? "right" : "left" }}>
            <div
              className={message.role === "assistant" ? "markdown-content" : undefined}
              style={{
                display: "inline-block",
                padding: "8px 12px",
                borderRadius: 8,
                background: message.role === "user" ? "#0366d6" : "#f0f0f0",
                color: message.role === "user" ? "#fff" : "#000",
                maxWidth: "80%",
                textAlign: "left",
              }}
            >
              {message.role === "assistant" ? (
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
              ) : (
                message.content
              )}
            </div>
            {message.sources && message.sources.length > 0 && (
              <div style={{ fontSize: 12, color: "#666", marginTop: 4 }}>
                Sources:{" "}
                {message.sources.map((source, sourceIndex) => (
                  <a
                    key={sourceIndex}
                    href={source}
                    target="_blank"
                    rel="noreferrer"
                    style={{ marginRight: 8 }}
                  >
                    [{sourceIndex + 1}]
                  </a>
                ))}
              </div>
            )}
          </div>
        ))}
        {isLoading && <div style={{ color: "#666" }}>Thinking…</div>}
        {error && <div style={{ color: "red" }}>{error}</div>}
      </div>

      <form
        onSubmit={(event) => {
          event.preventDefault();
          void sendMessage();
        }}
        style={{ display: "flex", gap: 8 }}
      >
        <input
          value={input}
          onChange={(event) => setInput(event.target.value)}
          placeholder={isTranscribing ? "Transcribing…" : "e.g. How much overtime pay am I entitled to?"}
          disabled={isTranscribing}
          style={{ flex: 1, padding: 8 }}
        />
        <button
          type="button"
          onClick={isRecording ? stopRecording : startRecording}
          disabled={isLoading || isTranscribing}
          title={isRecording ? "Stop recording" : "Record a question"}
          style={{
            background: isRecording ? "#d73a49" : undefined,
            color: isRecording ? "#fff" : undefined,
          }}
        >
          {isRecording ? "⏹" : "🎤"}
        </button>
        <button
          type="button"
          onClick={() => setIsMuted((previous) => !previous)}
          title={isMuted ? "Unmute voice output" : "Mute voice output"}
        >
          {isMuted ? "🔇" : "🔊"}
        </button>
        <button type="submit" disabled={isLoading || isTranscribing}>
          Send
        </button>
      </form>
    </main>
  );
}
