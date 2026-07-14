"""4-way H.264 / H.265 hardware decode with 2x2 (田-grid) display.

Each of the four input files gets its own decode thread and its own
``rkv.VideoCapture`` (so its own ``MppDecoder`` and VPU core). The
main thread composites the latest frame from each stream into a
single 2x2 image and shows it via ``cv2.imshow`` (or prints a stats
line when cv2 is unavailable).

Per-call ``read(w, h)`` does the colour-convert + resize to the grid
cell in one RGA2 hardware pass, so the host CPU only touches the
composite (a small 2x2 image). The grid layout:

    +-----------+-----------+
    |  slot 0   |  slot 1   |
    |  (TL)     |  (TR)     |
    +-----------+-----------+
    |  slot 2   |  slot 3   |
    |  (BL)     |  (BR)     |
    +-----------+-----------+

Each slot prints its own per-second frame count so you can see
relative throughput per stream (different files will hit the VPU
differently depending on resolution, bitrate, and B-frame use).

Usage::

    /usr/bin/python3 samples/multi_decoder.py a.mp4 b.mp4 c.mp4 d.mp4
    /usr/bin/python3 samples/multi_decoder.py bus.mp4 bus.mp4 bus.mp4 bus.mp4
    DISPLAY=0 /usr/bin/python3 samples/multi_decoder.py a.mp4 b.mp4 c.mp4 d.mp4
    /usr/bin/python3 samples/multi_decoder.py a.mp4 b.mp4 c.mp4 d.mp4 --width 1920 --height 1080

Press ``q`` in the display window to stop. The decode threads are
daemons and exit when their source EOFs (or the main thread tears
the window down).

Note on concurrency: ``rkv.mpp`` guards every C call with a process-
wide ``_mpp_lock``, so the four Python threads serialise through
that lock at the FFI boundary. The VPU cores themselves can run
parallel decode, but the Python wrapper exposes a single safe
context at a time. This sample still shows the API for fanning out
to multiple decoders — for true parallel VPU utilisation, run
multiple processes (one per stream) instead of threads.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

from rkv import VideoCapture


# Number of streams in the grid. Locked to 4 because the sample is
# named after the 2x2 (田) layout; tweaking this would need a layout
# helper instead of the (col, row) = (slot%2, slot//2) shortcut.
_NUM_STREAMS = 4


class Stream:
    """One decode pipeline + its slot in the 2x2 grid."""

    def __init__(self, path: str, slot: int, cell_size: Tuple[int, int]) -> None:
        self.path = path
        self.slot = slot
        self.cell_w, self.cell_h = cell_size
        self.cap: Optional[VideoCapture] = None
        # Latest frame, resized to cell size, BGR. None until the
        # first frame is decoded; protected by ``self.lock`` for
        # the rare case where the main thread composites mid-write.
        self.latest: Optional[np.ndarray] = None
        self.lock = threading.Lock()
        self.eof = False
        self.frames = 0
        self.error: Optional[str] = None
        # Coarse timing — main thread logs aggregate throughput
        # every ``--stats-interval`` seconds.
        self.t0 = time.monotonic()

    def open(self) -> None:
        try:
            self.cap = VideoCapture(self.path, timeout=10.0)
        except (FileNotFoundError, RuntimeError) as e:
            self.error = f"open failed: {e}"
            self.eof = True
            return
        if not self.cap.isOpened():
            self.error = "isOpened() = False"
            self.eof = True
            return
        c = self.cap
        print(
            f"[slot {self.slot}] {self.path}: "
            f"{c.width}x{c.height} @ {c.fps:.2f} fps",
            flush=True,
        )

    def run(self) -> None:
        """Decode loop. Runs in its own thread; exits on EOF or open error.

        Paces itself to the source's reported framerate using an
        *absolute* timeline (t0 + N * frame_period) rather than
        sleeping 1/fps after each frame. The per-iteration sleep
        approach drifts cumulatively over hundreds of frames; the
        absolute one stays accurate for the whole stream. Falls
        back to 25 fps when the demuxer doesn't report a rate
        (VFR MP4, raw camera streams, etc.).
        """
        if self.cap is None:
            return
        cell_w, cell_h = self.cell_w, self.cell_h
        fps = self.cap.fps if self.cap.fps and self.cap.fps > 0 else 25.0
        frame_period = 1.0 / fps
        t0 = time.monotonic()
        target_index = 0
        try:
            while not self.eof:
                # Per-call (w, h) drives RGA2 to do colour-convert
                # + resize in one hardware pass. No host CPU work
                # until the BGR ndarray is returned.
                ret, frame = self.cap.read(cell_w, cell_h)
                if not ret:
                    self.eof = True
                    break
                with self.lock:
                    # ``.copy()`` so the composite thread doesn't
                    # race with the next read() call (which would
                    # overwrite the underlying _bgr_buf view).
                    self.latest = frame.copy() if frame is not None else None
                self.frames += 1

                # Sleep until frame ``target_index`` is due on the
                # absolute timeline. The VPU decodes much faster
                # than the source fps (often 200+ fps for a 24 fps
                # H.264 stream), so this is where most of the wall
                # time goes. Capped sleep keeps the daemon
                # responsive when the user hits 'q'.
                target_index += 1
                target_time = t0 + target_index * frame_period
                remaining = target_time - time.monotonic()
                if remaining > 0:
                    time.sleep(min(remaining, 0.05))
        except Exception as e:  # noqa: BLE001 — surface to the cell, keep going
            self.error = f"decode error: {e}"
        finally:
            try:
                self.cap.release()
            except Exception:
                pass


def _slot_xy(slot: int, cell_w: int, cell_h: int) -> Tuple[int, int]:
    """Top-left (x, y) of ``slot`` in the grid image."""
    col, row = slot % 2, slot // 2
    return col * cell_w, row * cell_h


def _render_cell(stream: Stream, cell_w: int, cell_h: int) -> np.ndarray:
    """Return a (cell_h, cell_w, 3) BGR ndarray for one grid cell."""
    if stream.error is not None:
        cell = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
        cv2.putText(
            cell, stream.error, (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA,
        )
        return cell
    with stream.lock:
        frame = stream.latest
    if frame is None:
        return np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
    if frame.shape[1] != cell_w or frame.shape[0] != cell_h:
        return cv2.resize(frame, (cell_w, cell_h))
    return frame


def main(args: argparse.Namespace) -> int:
    paths: List[str] = list(args.paths)
    if len(paths) != _NUM_STREAMS:
        print(
            f"Error: need exactly {_NUM_STREAMS} paths for the 田 grid, "
            f"got {len(paths)}",
            file=sys.stderr,
        )
        return 1

    # Grid must be even in both dimensions so 2x2 cells are integer-sized.
    grid_w, grid_h = args.width, args.height
    if grid_w % 2 or grid_h % 2:
        print("Error: --width and --height must be even", file=sys.stderr)
        return 1
    cell_w, cell_h = grid_w // 2, grid_h // 2

    streams = [Stream(p, i, (cell_w, cell_h)) for i, p in enumerate(paths)]
    for s in streams:
        s.open()

    threads = [
        threading.Thread(
            target=s.run,
            daemon=True,
            name=f"decoder-{s.slot}",
        )
        for s in streams
    ]
    for t in threads:
        t.start()

    cv2.namedWindow("rkv multi-decoder", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("rkv multi-decoder", grid_w, grid_h)

    grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    last_stats = time.monotonic()
    try:
        while True:
            for s in streams:
                cell = _render_cell(s, cell_w, cell_h)
                x, y = _slot_xy(s.slot, cell_w, cell_h)
                grid[y : y + cell_h, x : x + cell_w] = cell

            cv2.imshow("rkv multi-decoder", grid)
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                break

            now = time.monotonic()
            if now - last_stats >= args.stats_interval:
                parts = []
                for s in streams:
                    elapsed = max(now - s.t0, 1e-6)
                    fps = s.frames / elapsed
                    tag = f"err" if s.error else ("eof" if s.eof else "live")
                    parts.append(f"ch{s.slot}={s.frames}f@{fps:.1f}fps[{tag}]")
                print("  " + "  ".join(parts), flush=True)
                last_stats = now

            if all(s.eof or s.error for s in streams):
                cv2.waitKey(0)
                break
    finally:
        cv2.destroyAllWindows()
        for t in threads:
            t.join(timeout=1.0)

    # Final tally.
    for s in streams:
        elapsed = max(time.monotonic() - s.t0, 1e-6)
        print(
            f"[slot {s.slot}] {s.path}: "
            f"{s.frames} frames in {elapsed:.1f}s "
            f"({s.frames / elapsed:.1f} fps)"
            + (f" — {s.error}" if s.error else ""),
            flush=True,
        )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Decode 4 video files in parallel and display them in a "
            "2x2 (田) grid. Use the same path four times to bench a "
            "single file under multi-stream load."
        ),
    )
    p.add_argument(
        "paths",
        nargs=_NUM_STREAMS,
        metavar="PATH",
        help=(
            f"Exactly {_NUM_STREAMS} input files. Each opens its own "
            "VideoCapture / MppDecoder context."
        ),
    )
    p.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Grid width in pixels (must be even, default %(default)s).",
    )
    p.add_argument(
        "--height",
        type=int,
        default=720,
        help="Grid height in pixels (must be even, default %(default)s).",
    )
    p.add_argument(
        "--stats-interval",
        type=float,
        default=5.0,
        help="Seconds between per-stream fps prints (default %(default)s).",
    )
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args(sys.argv[1:])
    raise SystemExit(main(args))
