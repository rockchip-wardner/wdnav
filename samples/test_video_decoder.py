"""Minimal smoke test for rkv.VideoCapture.

Run with the system Python (which has apt-installed python3-av):

    /usr/bin/python3 samples/test_video_decoder.py bus_fight.mp4
    /usr/bin/python3 samples/test_video_decoder.py rtsp://user:pass@host:554/stream
    DISPLAY=0 /usr/bin/python3 samples/test_video_decoder.py bus_fight.mp4   # headless

The third argument, if present, is a frame-count cap (useful for
non-terminating sources like RTSP).
"""

from __future__ import annotations

import os
import sys

import cv2

from rkv import VideoCapture


def main(argv: list[str]) -> int:
    video_path = argv[1] if len(argv) > 1 else "bus.mp4"
    cap_opts = {"rtsp_transport": "tcp"} if video_path.startswith("rtsp://") else None
    cap = VideoCapture(video_path, options=cap_opts, timeout=10.0)

    if not cap.isOpened():
        print(f"Error: cannot open {video_path}")
        return 1

    print(f"opened {video_path}: {cap.width}x{cap.height} @ {cap.fps:.2f} fps")
    window_open = os.environ.get("DISPLAY", "1") != "0"
    max_frames = int(argv[2]) if len(argv) > 2 else 0

    if window_open and cap.width:
        cv2.namedWindow("Video Player", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Video Player", min(cap.width, 1280), min(cap.height, 720))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if window_open:
            cv2.imshow("Video Player", frame)
            if cv2.waitKey(25) & 0xFF == ord("q"):
                break
        elif cap.frame_pos % 30 == 0:
            print(f"  frame {cap.frame_pos}")

        if max_frames and cap.frame_pos >= max_frames:
            break

    print(f"played {cap.frame_pos} frames")
    cap.release()
    if window_open:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))