#!/usr/bin/env python3
"""Smoke sample for ``rkv.VideoWriter``.

Reads one BGR frame from a synthetic source and writes it to a local
MP4 (or any extension PyAV understands) using the MPP hardware
encoder. Codec choices:

* ``mp4v`` (default) — PyAV's built-in MPEG-4 Part 2 software
  encoder, works on any machine.
* ``h264`` — MPP hardware only (requires Rockchip SoC); raises
  ``RuntimeError`` on platforms where MPP isn't available.
* ``mjpeg`` — PyAV's built-in MJPEG software encoder.

Run::

    python samples/video_writer.py /tmp/out.mp4
    python samples/video_writer.py /tmp/out.mp4 h264
    python samples/video_writer.py /tmp/out.mkv mjpeg 30

If the destination uses the ``rkv+ws://`` scheme, the writer pushes
encoded JPEG packets straight to a pyrkaiweb ingest endpoint
instead. See ``app/services/ingest.py`` for the server side.
"""
from __future__ import annotations

import sys
import time

import numpy as np

from rkv import VideoWriter


def synthetic_frame(i: int, w: int = 640, h: int = 360) -> np.ndarray:
    """Generate a moving colour-bar frame so the output isn't a black square."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    bar_w = w // 8
    for k in range(8):
        x0 = k * bar_w
        frame[:, x0 : x0 + bar_w, 0] = (k * 32 + i * 5) & 0xFF
        frame[:, x0 : x0 + bar_w, 1] = (k * 64 + i * 7) & 0xFF
        frame[:, x0 : x0 + bar_w, 2] = (k * 96 + i * 11) & 0xFF
    # Scrolling white tick so the encoded output clearly changes frame-to-frame.
    tick_x = (i * 8) % w
    frame[max(0, h - 6): h, tick_x : tick_x + 12, :] = 255
    return frame


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: video_writer.py <path> [codec=mp4v] [fps=30] [n_frames=60]")
        return 2

    path = argv[1]
    codec = argv[2] if len(argv) > 2 else "mp4v"
    fps = int(argv[3]) if len(argv) > 3 else 30
    n = int(argv[4]) if len(argv) > 4 else 60

    print(f"opening VideoWriter -> {path!r} ({codec}, {fps}fps)")
    t0 = time.monotonic()
    out = VideoWriter(
        path,
        fps=fps,
        frame_size=(640, 360),
        codec=codec,
        profile="baseline",
        gop=30,
        bitrate=2_000_000,
    )
    print(f"  open ok in {(time.monotonic() - t0) * 1000:.1f} ms")
    print(
        f"  codec={out.codec} backend={out.backend} "
        f"profile={out.profile} isOpened={out.isOpened}"
    )

    total_ms = 0.0
    for i in range(n):
        frame = synthetic_frame(i)
        t = time.monotonic()
        out.write(frame)
        elapsed_ms = (time.monotonic() - t) * 1000
        total_ms += elapsed_ms
        if i % 10 == 0 or i == n - 1:
            print(f"  frame {i:3d}/{n}: {elapsed_ms:6.2f} ms")

    out.release()
    print(f"done; avg frame time = {total_ms / n:.2f} ms ({n / (total_ms / 1000):.0f} fps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))