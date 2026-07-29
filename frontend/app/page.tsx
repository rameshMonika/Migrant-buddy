"use client";

import { useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

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
// Separate service from API_BASE_URL -- the RAG service (chat) and the
// Whisper service (speech-to-text) run as two independent processes, so the
// frontend talks to each directly rather than one proxying the other.
const WHISPER_WS_URL = process.env.NEXT_PUBLIC_WHISPER_WS_URL ?? "ws://localhost:8002/transcribe/ws";

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
            }
            // "done" needs no handling -- the loop ends when the stream closes.
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
        <button type="submit" disabled={isLoading || isTranscribing}>
          Send
        </button>
      </form>
    </main>
  );
}
