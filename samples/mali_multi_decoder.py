"""2x2 hardware decode + GPU composition via wdnav.SurfaceFlinger.

Four input files, each on its own decode thread + its own
:class:`wdnav.VideoCapture` + its own :class:`wdnav.Surface` on a 2x2 grid in
a single :class:`wdnav.SurfaceFlinger`. The main thread just calls
``sf.read()`` once per loop iteration; the per-frame composition is GPU.

Same per-stream pacing as :file:`multi_decoder.py`, but the grid layer
is composited on the GPU instead of being pieced together with cv2 + numpy.

Each per-call ``cap.read(w, h)`` drives RGA2 to do colour-convert + resize
to the cell size in one hardware pass, so the host CPU only writes a
pre-sized BGR ndarray into each Surface's software buffer. The compositor
then uploads + blends four layers on the GPU every frame.

    python samples/mali_multi_decoder.py a.mp4 b.mp4 c.mp4 d.mp4
    python samples/mali_multi_decoder.py bus.mp4 bus.mp4 bus.mp4 bus.mp4
    python samples/mali_multi_decoder.py a.mp4 b.mp4 c.mp4 d.mp4 --width 1920 --height 1080

Press ``q`` in the display window to stop.
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

import wdnav


_NUM_STREAMS = 4


class Stream:
    """One decode pipeline + its slot in the 2x2 grid."""

    def __init__(
        self,
        path: str,
        slot: int,
        surface: "wdnav.Surface",
        cell_size: Tuple[int, int],
    ) -> None:
        self.path = path
        self.slot = slot
        self.surface = surface
        self.cell_w, self.cell_h = cell_size
        self.cap: Optional[wdnav.VideoCapture] = None
        self.eof = False
        self.frames = 0
        self.t0 = time.monotonic()
        self.error: Optional[str] = None

    def open(self) -> None:
        try:
            self.cap = wdnav.VideoCapture(self.path, timeout=10.0)
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

        Paces itself to the source's reported framerate using an absolute
        timeline (``t0 + N * frame_period``) rather than sleeping 1/fps
        after each frame. Falls back to 25 fps when the demuxer doesn't
        report a rate (VFR MP4, raw camera streams, etc.).
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
                ret, frame = self.cap.read(cell_w, cell_h)
                if not ret or frame is None:
                    self.eof = True
                    break
                with self.surface._lock:
                    self.surface.latest_ndarray[:] = frame
                self.surface.post(pts=self.frames)
                self.frames += 1

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


def main(args: argparse.Namespace) -> int:
    paths: List[str] = list(args.paths)
    if len(paths) != _NUM_STREAMS:
        print(
            f"Error: need exactly {_NUM_STREAMS} paths for the 2x2 grid, "
            f"got {len(paths)}",
            file=sys.stderr,
        )
        return 1

    grid_w, grid_h = args.width, args.height
    if grid_w % 2 or grid_h % 2:
        print("Error: --width and --height must be even", file=sys.stderr)
        return 1
    cell_w, cell_h = grid_w // 2, grid_h // 2

    sf = wdnav.SurfaceFlinger(w=grid_w, h=grid_h)
    surfaces = [
        sf.create_surface(
            (i % 2) * cell_w, (i // 2) * cell_h,
            cell_w, cell_h, zpos=0, fmt="BGR888",
        )
        for i in range(_NUM_STREAMS)
    ]

    streams = [
        Stream(p, i, surfaces[i], (cell_w, cell_h))
        for i, p in enumerate(paths)
    ]
    for s in streams:
        s.open()

    threads = [
        threading.Thread(target=s.run, daemon=True, name=f"decoder-{s.slot}")
        for s in streams
    ]
    for t in threads:
        t.start()

    cv2.namedWindow("wdnav multi-decoder", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("wdnav multi-decoder", grid_w, grid_h)

    last_stats = time.monotonic()
    try:
        while True:
            img = sf.read(timeout=5.0)
            cv2.imshow("wdnav multi-decoder", img)
            if cv2.waitKey(30) & 0xFF == ord("q"):
                break

            now = time.monotonic()
            if now - last_stats >= args.stats_interval:
                parts = []
                for s in streams:
                    elapsed = max(now - s.t0, 1e-6)
                    fps = s.frames / elapsed
                    tag = "err" if s.error else ("eof" if s.eof else "live")
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
        sf.close()

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
            "Decode 4 video files in parallel and composite them on the GPU "
            "via wdnav.SurfaceFlinger."
        ),
    )
    p.add_argument(
        "paths",
        nargs=_NUM_STREAMS,
        metavar="PATH",
        help=(
            f"Exactly {_NUM_STREAMS} input files. Each opens its own "
            "VideoCapture / MppDecoder / Surface."
        ),
    )
    p.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Compositor output width (must be even, default %(default)s).",
    )
    p.add_argument(
        "--height",
        type=int,
        default=720,
        help="Compositor output height (must be even, default %(default)s).",
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
