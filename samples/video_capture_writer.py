#!/usr/bin/env python3
"""End-to-end smoke test: ``VideoCapture`` → ``VideoWriter``.

Wires up the two halves of the public API so a single command exercises
both directions of the pipeline — VPU decode (MPP) + RGA colour
conversion on the read side, MPP hardware encoding (or PyAV's
(MPP hardware-only) on the write side. Useful as a sanity
check that both halves still work after refactors, and as a starting
point for transcode / passthrough pipelines.

Usage::

    python samples/video_capture_writer.py bus.mp4 /tmp/out.mp4
    python samples/video_capture_writer.py rtsp://host/stream /tmp/out.mp4
    python samples/video_capture_writer.py                 # synthetic source
    python samples/video_capture_writer.py bus.mp4 /tmp/out.mp4 --read0

Pass ``--read0`` to use :meth:`VideoCapture.read0` — the zero-copy
read path that returns a :class:`wdnav.mpp.DecodedFrame` whose ``fd`` is
the MPP-decoded DMA-BUF. The frame is then handed straight to
:meth:`VideoWriter.write` via the same ``frame0`` shape the public API
expects; ``VideoWriter.write`` keeps the zero-copy semantics by
asking :meth:`DecodedFrame.numpy` for the BGR view only when an actual
ndarray is needed (the RGA colour-conversion still happens, but no
extra Python-side copy is inserted between read and write).

If ``input_path`` does not exist (or is omitted) the demo falls back
to a synthetic colour-bar source so the script still runs end-to-end
on a dev machine without test media. ``--max-frames`` caps the run so
live streams don't loop forever.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

from wdnav import VideoCapture, VideoWriter


def _synthetic_frame(i: int, w: int, h: int) -> np.ndarray:
    """Moving colour-bar frame so the encoded output clearly changes
    frame-to-frame. Mirrors the generator used by ``video_writer.py``.
    """
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Decode frames with wdnav.VideoCapture and re-encode them "
            "with wdnav.VideoWriter. Verifies the capture → writer "
            "pipeline works end-to-end."
        ),
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="bus.mp4",
        help=(
            "Input source: a local file path or an rtsp:// URL. "
            "Default: %(default)s. Falls back to a synthetic source "
            "if the file is missing."
        ),
    )
    parser.add_argument(
        "output",
        nargs="?",
        default="/tmp/rkv_capture_writer_out.mp4",
        help="Output MP4 path. Default: %(default)s.",
    )
    parser.add_argument(
        "--codec",
        default="mp4v",
        choices=("h264", "mjpeg", "mp4v"),
        help="Output codec (default %(default)s).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=0,
        help=(
            "Output frame rate (0 = inherit from the input via "
            "VideoCapture.fps, falling back to 30 if unknown)."
        ),
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        metavar="N",
        help="Stop after N frames (0 = no cap).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=0,
        help="Resize to WIDTH before encoding (0 = keep source size).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=0,
        help="Resize to HEIGHT before encoding (0 = keep source size).",
    )
    parser.add_argument(
        "--read0",
        action="store_true",
        help=(
            "Use VideoCapture.read0() instead of read(). The decoded "
            "frame is handed to VideoWriter.write() as a DecodedFrame "
            "so the DMA-BUF fd flows through the pipeline end-to-end "
            "before RGA converts it to BGR for the encoder."
        ),
    )
    return parser


def _is_real_source(path: str) -> bool:
    """Treat anything with a ``://`` scheme as live (always real). For
    plain paths require the file to exist on disk — otherwise the
    synthetic fallback kicks in so the demo always runs."""
    if "://" in path:
        return True
    return Path(path).is_file()


def main(argv: list[str]) -> int:
    args = _build_parser().parse_args(argv[1:])

    use_synthetic = not _is_real_source(args.input)
    if use_synthetic:
        width = args.width or 640
        height = args.height or 360
        fps = args.fps or 30
        max_frames = args.max_frames or 60
        print(
            f"input {args.input!r} not found — using synthetic "
            f"{width}x{height}@{fps}fps source, {max_frames} frames"
        )
    else:
        cap_probe = VideoCapture(args.input, timeout=10.0)
        if not cap_probe.isOpened():
            print(f"Error: cannot open {args.input!r}")
            return 1
        width = args.width or cap_probe.width
        height = args.height or cap_probe.height
        fps = args.fps or int(round(cap_probe.fps)) or 30
        max_frames = args.max_frames
        print(
            f"input {args.input!r}: {cap_probe.width}x{cap_probe.height} "
            f"@ {cap_probe.fps:.2f}fps → encode {width}x{height}@{fps}fps"
        )
        cap_probe.release()

    # Make sure the parent directory of the output exists; PyAV
    # refuses to mux into a file in a non-existent directory.
    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"opening VideoWriter -> {args.output!r} ({args.codec}, {fps}fps)")
    t0 = time.monotonic()
    out = VideoWriter(
        args.output,
        fps=fps,
        frame_size=(width, height),
        codec=args.codec,
        profile="baseline",
        gop=30,
        bitrate=2_000_000,
    )
    open_ms = (time.monotonic() - t0) * 1000
    print(
        f"  open ok in {open_ms:.1f} ms; backend={out.backend} "
        f"codec={out.codec} profile={out.profile} isOpened={out.isOpened}"
    )
    if not out.isOpened:
        return 1

    n_written = 0
    total_read_ms = 0.0
    total_write_ms = 0.0
    if use_synthetic:
        for i in range(max_frames):
            t = time.monotonic()
            frame = _synthetic_frame(i, width, height)
            total_read_ms += (time.monotonic() - t) * 1000

            t = time.monotonic()
            out.write(frame)
            total_write_ms += (time.monotonic() - t) * 1000

            n_written += 1
            if n_written % 10 == 0 or n_written == max_frames:
                print(f"  frame {n_written:4d}/{max_frames}")
    else:
        with VideoCapture(args.input, timeout=10.0) as cap:
            if not cap.isOpened():
                print(f"Error: cannot reopen {args.input!r}")
                out.release()
                return 1

            try:
                while True:
                    if max_frames and cap.frame_pos >= max_frames:
                        break

                    t = time.monotonic()
                    if args.read0:
                        # Zero-copy path: hand the DecodedFrame
                        # straight to the writer; numpy() inside
                        # DecodedFrame does the BGR conversion via RGA
                        # only when actually requested.
                        ret, frame0 = cap.read0(width or None, height or None)
                        frame_to_write = frame0
                    else:
                        # Eager path: BGR ndarray returned from read().
                        ret, frame0 = cap.read(width or None, height or None)
                        frame_to_write = frame0
                    total_read_ms += (time.monotonic() - t) * 1000
                    if not ret:
                        print("  end of stream")
                        break
                    if frame_to_write is None:
                        continue

                    t = time.monotonic()
                    out.write(frame_to_write)
                    total_write_ms += (time.monotonic() - t) * 1000

                    n_written += 1
                    if n_written % 30 == 0:
                        path_label = "read0→write0" if args.read0 else "read→write"
                        print(
                            f"  frame {n_written:4d} [{path_label}] "
                            f"(read {total_read_ms / n_written:.2f} ms, "
                            f"write {total_write_ms / n_written:.2f} ms)"
                        )
            finally:
                cap.release()

    out.release()

    out_size = os.path.getsize(args.output) if os.path.isfile(args.output) else 0
    avg_read = total_read_ms / max(n_written, 1)
    avg_write = total_write_ms / max(n_written, 1)
    print(
        f"done; wrote {n_written} frames to {args.output!r} "
        f"({out_size} bytes)\n"
        f"  avg read  = {avg_read:6.2f} ms/frame\n"
        f"  avg write = {avg_write:6.2f} ms/frame\n"
        f"  total     = {(total_read_ms + total_write_ms):.1f} ms"
    )
    return 0 if n_written > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))