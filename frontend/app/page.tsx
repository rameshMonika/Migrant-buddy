"use client";

import { LiveAvatarSession, SessionEvent } from "@heygen/liveavatar-web-sdk";
import { useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import LiveAvatar from "./components/LiveAvatar";
import { MicIcon, SendIcon, SpeakerIcon, SpeakerMutedIcon, StopIcon } from "./components/icons";

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

// Voice + lip-sync (LiveAvatar/HeyGen + ElevenLabs) -- re-enabled now that
// the earlier generation-quality bugs are fixed. Flip back to false to
// disable the whole feature again without touching the implementation.
const AVATAR_TTS_ENABLED = true;

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
  // Safety net for the "stop" round trip below -- if the server's close
  // frame is ever dropped, delayed, or swallowed by a proxy (onclose is the
  // only other place isTranscribing resets), this fires anyway so the
  // input/mic/Send buttons never end up stuck disabled indefinitely.
  const transcribeTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  function clearTranscribeTimeout() {
    if (transcribeTimeoutRef.current !== null) {
      clearTimeout(transcribeTimeoutRef.current);
      transcribeTimeoutRef.current = null;
    }
  }

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
    if (!AVATAR_TTS_ENABLED) {
      return;
    }
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
        clearTranscribeTimeout();
        setError("Lost connection to the transcription service.");
        setIsTranscribing(false);
      };

      socket.onclose = () => {
        // The server closes the socket itself once it's replied to "stop"
        // with whatever was left in the buffer -- that's the normal signal
        // that transcription is done. transcribeTimeoutRef (started when
        // "stop" is sent, below) is the fallback for when this never fires.
        clearTranscribeTimeout();
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
          // Server processing (final Whisper flush) plus network round
          // trip should take well under this -- if it hasn't closed by
          // then, force the buttons back on rather than leave the user
          // stuck unable to send or record again.
          clearTranscribeTimeout();
          transcribeTimeoutRef.current = setTimeout(() => {
            transcribeTimeoutRef.current = null;
            setIsTranscribing(false);
          }, 8000);
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
    if (AVATAR_TTS_ENABLED) {
      void ensureLiveAvatarSession();
    }

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
    <main className="app-shell">
      <div className="app-header">
        <h1>migrantBuddy</h1>
        <p>
          Ask a question about Singapore employment rules (salary, hours, work permits,
          medical insurance).
        </p>
      </div>

      <div className="layout-grid">
        {AVATAR_TTS_ENABLED && (
          <div className="avatar-panel">
            <LiveAvatar videoRef={videoRef} isReady={isAvatarReady} />
          </div>
        )}

        <div className="chat-panel">
          <div className="messages-scroll">
            {messages.map((message, index) => (
              <div key={index} className={`message-row from-${message.role}`}>
                <div>
                  <div
                    className={`bubble bubble-${message.role} ${
                      message.role === "assistant" ? "markdown-content" : ""
                    }`}
                  >
                    {message.role === "assistant" ? (
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
                    ) : (
                      message.content
                    )}
                  </div>
                  {message.sources && message.sources.length > 0 && (
                    <div className="sources-row">
                      Sources:
                      {message.sources.map((source, sourceIndex) => (
                        <a
                          key={sourceIndex}
                          href={source}
                          target="_blank"
                          rel="noreferrer"
                          className="source-pill"
                        >
                          {sourceIndex + 1}
                        </a>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            ))}
            {isLoading && <div className="status-line">Thinking…</div>}
            {error && <div className="status-line is-error">{error}</div>}
          </div>

          <form
            onSubmit={(event) => {
              event.preventDefault();
              void sendMessage();
            }}
            className="composer"
          >
            <button
              type="button"
              className={`icon-button ${isRecording ? "is-active" : ""}`}
              onClick={isRecording ? stopRecording : startRecording}
              disabled={isLoading || isTranscribing}
              title={isRecording ? "Stop recording" : "Record a question"}
            >
              {isRecording ? <StopIcon /> : <MicIcon />}
            </button>
            <input
              className="composer-input"
              value={input}
              onChange={(event) => setInput(event.target.value)}
              placeholder={isTranscribing ? "Transcribing…" : "e.g. How much overtime pay am I entitled to?"}
              disabled={isTranscribing}
            />
            {AVATAR_TTS_ENABLED && (
              <button
                type="button"
                className="icon-button"
                onClick={() => setIsMuted((previous) => !previous)}
                title={isMuted ? "Unmute voice output" : "Mute voice output"}
              >
                {isMuted ? <SpeakerMutedIcon /> : <SpeakerIcon />}
              </button>
            )}
            <button
              type="submit"
              className="send-button"
              disabled={isLoading || isTranscribing}
              title="Send"
            >
              <SendIcon />
            </button>
          </form>
        </div>
      </div>
    </main>
  );
}
