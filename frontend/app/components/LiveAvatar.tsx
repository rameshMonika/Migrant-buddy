"use client";

import { RefObject } from "react";
import { VideoIcon } from "./icons";

// Thin wrapper around the <video> element LiveAvatar streams into (see
// page.tsx's ensureLiveAvatarSession -- `session.attach(videoRef.current)`
// is called once `session.stream_ready` fires). Replaces the old
// placeholder SVG Avatar -- LiveAvatar renders the actual lip-synced video
// server-side, nothing to draw here ourselves.
//
// Sized by its parent (globals.css's .avatar-panel), not a fixed pixel
// size, so the left-column layout in page.tsx can make it as large as the
// available space allows.
export default function LiveAvatar({
  videoRef,
  isReady,
}: {
  videoRef: RefObject<HTMLVideoElement | null>;
  isReady: boolean;
}) {
  return (
    <div className="avatar-video-frame">
      <video ref={videoRef} autoPlay playsInline />
      {!isReady && (
        <div className="avatar-placeholder">
          <div className="avatar-placeholder-icon">
            <VideoIcon />
          </div>
          <span className="sr-only">Connecting to avatar…</span>
        </div>
      )}
    </div>
  );
}
