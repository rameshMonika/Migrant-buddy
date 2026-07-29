"use client";

import { RefObject } from "react";

// Thin wrapper around the <video> element LiveAvatar streams into (see
// page.tsx's ensureLiveAvatarSession -- `session.attach(videoRef.current)`
// is called once `session.stream_ready` fires). Replaces the old
// placeholder SVG Avatar -- LiveAvatar renders the actual lip-synced video
// server-side, nothing to draw here ourselves.
export default function LiveAvatar({
  videoRef,
  isReady,
}: {
  videoRef: RefObject<HTMLVideoElement | null>;
  isReady: boolean;
}) {
  return (
    <div style={{ position: "relative", width: 320, height: 320 }}>
      <video
        ref={videoRef}
        autoPlay
        playsInline
        style={{ width: "100%", height: "100%", borderRadius: 8, backgroundColor: "#222" }}
      />
      {!isReady && (
        <div
          style={{
            position: "absolute",
            inset: 0,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            color: "#ccc",
            fontSize: 14,
          }}
        >
          Connecting…
        </div>
      )}
    </div>
  );
}
