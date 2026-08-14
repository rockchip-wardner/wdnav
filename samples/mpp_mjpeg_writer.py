#!/usr/bin/env python3
"""Smoke sample for ``wdnav.VideoWriter`` with ``codec="mjpeg"``.

Drives the MPP hardware MJPEG encoder (the ``mpp-mjpeg`` backend
of :class:`wdnav.VideoWriter`) end-to-end: synthesises a moving colour-bar
frame on the CPU, hands it to the encoder, and lets PyAV mux the
resulting JPEG packets into an AVI container.

Why AVI? Motion-JPEG's standard carrier is AVI — every packet the
encoder produces is a complete JPEG bitstream (SOI…EOI), and the AVI
``mjpg`` FourCC is what most players (VLC, ffmpeg, mpv, browsers via
MSE) expect on this path. ``.mov`` also works if you'd rather have
a QuickTime container; the writer picks the format from the extension.

Run::

    python samples/mpp_mjpeg_writer.py /tmp/out.avi
    python samples/mpp_mjpeg_writer.py /tmp/out.mov 90 60

Arguments:

* ``path`` — output file; the extension selects the container
  (``.avi`` → AVI, ``.mov``/``.mp4`` → MP4, ``.mkv`` → Matroska).
* ``q_factor`` (default ``80``) — JPEG quality 1..99 (higher = better
  quality, larger file). Maps to MPP's ``jpeg:q_factor`` kcfg key,
  same as the upstream ``mpi_enc_utils.c`` defaults.
* ``n_frames`` (default ``60``) — number of synthetic frames to encode.

Verify the output with ``ffprobe``::

    ffprobe -v error -show_streams /tmp/out.avi
    ffprobe -v error -show_entries packet=pts_time,size -of csv /tmp/out.avi

The script raises ``RuntimeError`` at construction time on a
non-MPP platform — that's expected, ``codec="mjpeg"`` is MPP-only on
the local-file path (see ``VideoWriter.__init__``).

Notes
=====
MPP's MJPEG HAL on this build accepts a config-only path through
``mpp_venc_kcfg_init`` / ``MPP_SET_VENC_INIT_KCFG`` (the path the
upstream ``mpi_enc_test.c`` uses). On a stock distro that exports
``mpp_venc_kcfg_init`` it produces properly formatted JPEG with
the right quantisation tables and dimensions. On stripped builds
where only ``mpp_init`` / ``mpp_enc_cfg_set_s32`` are available the
HAL falls back to a stripped-down init that still produces
structurally-valid JPEG (SOI / APP0 / DQT / SOF0 / DHT / SOS / EOI)
but with the SOF0 dimensions and DQT tables left at their reset
defaults — the file is therefore visible to PyAV's MJPEG demuxer
(packet size, pts) but most image decoders reject it as malformed.

If the user's libc/librockchip_mpp.so exports ``mpp_venc_kcfg_init``
the output decodes cleanly with ``ffmpeg -i out.avi frame_%04d.jpg``.

Workaround for the encoder input-queue leak
--------------------------------------------
On the same firmware the encoder's input queue does not free slots
across ``encode_get_packet`` calls — after ~8 frames
``encode_put_frame`` starts blocking on ``BUFFER_FULL`` because the
queue is full and no slot is being recycled. The writer hides this
by recreating the :class:`MppEncoder` every 6 frames; the PTS
counter stays monotonic so the AVI muxer is happy. This is a
VideoWriter-level workaround — a future kmpp init path would
make it unnecessary.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

from wdnav import VideoWriter


def synthetic_frame(i: int, w: int = 640, h: int = 360) -> np.ndarray:
    """Generate a moving colour-bar frame so the output isn't a black square."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    bar_w = w // 8
    for k in range(8):
        x0 = k * bar_w
        frame[:, x0 : x0 + bar_w, 0] = (k * 32 + i * 5) & 0xFF
        frame[:, x0 : x0 + bar_w, 1] = (k * 64 + i * 7) & 0xFF
        frame[:, x0 : x0 + bar_w, 2] = (k * 96 + i * 11) & 0xFF
    tick_x = (i * 8) % w
    frame[max(0, h - 6): h, tick_x : tick_x + 12, :] = 255
    return frame


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(
            "usage: mpp_mjpeg_writer.py <path> [q_factor=80] [n_frames=60]\n"
            "  path:      .avi / .mov / .mp4 / .mkv — selects the container\n"
            "  q_factor:  JPEG quality 1..99 (default 80)\n"
            "  n_frames:  number of synthetic frames to encode (default 60)"
        )
        return 2

    path = argv[1]
    q_factor = int(argv[2]) if len(argv) > 2 else 80
    n = int(argv[3]) if len(argv) > 3 else 60

    print(
        f"opening VideoWriter -> {path!r} codec=mjpeg q_factor={q_factor} "
        f"n_frames={n}"
    )
    t0 = time.monotonic()
    out = VideoWriter(
        path,
        fps=30,
        frame_size=(640, 360),
        codec="mjpeg",
        q_factor=q_factor,
        gop=1,  # MJPEG has no GOP concept; ignored by the encoder
    )
    open_ms = (time.monotonic() - t0) * 1000
    print(
        f"  open ok in {open_ms:.1f} ms; backend={out.backend} "
        f"isOpened={out.isOpened}"
    )
    print(
        f"  codec={out.codec} fps={out.fps} "
        f"frame_size=({out.width}, {out.height})"
    )

    total_ms = 0.0
    sizes = []
    for i in range(n):
        frame = synthetic_frame(i)
        t = time.monotonic()
        out.write(frame)
        elapsed_ms = (time.monotonic() - t) * 1000
        total_ms += elapsed_ms
        if i % 10 == 0 or i == n - 1:
            print(f"  frame {i:3d}/{n}: {elapsed_ms:6.2f} ms")

    out.release()
    avg_ms = total_ms / n if n else 0.0
    print(
        f"done; avg frame time = {avg_ms:.2f} ms "
        f"({n / (total_ms / 1000):.0f} fps)"
    )

    if os.path.exists(path):
        size_kb = os.path.getsize(path) / 1024
        print(f"output file: {path} ({size_kb:.1f} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))