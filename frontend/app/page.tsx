"use client";

import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

type Message = {
  role: "user" | "assistant";
  content: string;
  sources?: string[];
};

type ChatResponse = {
  thread_id: string;
  message: string;
  answer: string;
  sources: string[];
};

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

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

  async function sendMessage() {
    const message = input.trim();
    if (!message || isLoading) {
      return;
    }

    setMessages((previous) => [...previous, { role: "user", content: message }]);
    setInput("");
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, thread_id: threadId }),
      });

      if (!response.ok) {
        throw new Error(`Request failed: ${response.status}`);
      }

      const data: ChatResponse = await response.json();
      setMessages((previous) => [
        ...previous,
        { role: "assistant", content: data.answer, sources: data.sources },
      ]);
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
          placeholder="e.g. How much overtime pay am I entitled to?"
          style={{ flex: 1, padding: 8 }}
        />
        <button type="submit" disabled={isLoading}>
          Send
        </button>
      </form>
    </main>
  );
}
