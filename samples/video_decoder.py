"""Minimal smoke test for rkv.VideoCapture.

Run with the system Python (which has apt-installed python3-av):

    /usr/bin/python3 samples/test_video_decoder.py bus_fight.mp4
    /usr/bin/python3 samples/test_video_decoder.py rtsp://user:pass@host:554/stream
    DISPLAY=0 /usr/bin/python3 samples/test_video_decoder.py bus_fight.mp4   # headless

A local file path or an rtsp:// URL is passed positionally. For
non-terminating sources (RTSP, live cameras), cap the run with
``--max-frames``; press ``q`` in the display window to stop early.
"""

from __future__ import annotations

import argparse
import sys

import cv2

from rkv import VideoCapture


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Open a local file or a network stream with rkv.VideoCapture and display / count decoded frames."),
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="bus.mp4",
        help=(
            "Input source: a local file path or an rtsp:// URL. "
            "Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Stop after N frames (0 = no cap). Useful for live streams "
            "that never reach EOF."
        ),
    )
    return parser


def main(args) -> int:
    cap = VideoCapture(args.path, timeout=10.0)

    if not cap.isOpened():
        print(f"Error: cannot open {args.path}")
        return 1

    print(f"opened {args.path}: {cap.width}x{cap.height} @ {cap.fps:.2f} fps")

    cv2.namedWindow("Video Player", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Video Player", min(cap.width, 1280), min(cap.height, 720))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Video Player", frame)
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

        if args.max_frames and cap.frame_pos >= args.max_frames:
            break

    print(f"played {cap.frame_pos} frames")
    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    args = _build_parser().parse_args(sys.argv[1:])
    raise SystemExit(main(args))