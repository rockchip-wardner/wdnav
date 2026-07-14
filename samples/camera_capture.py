"""Local-camera capture smoke test.

Demonstrates two complementary APIs:

1. ``MediaDevice`` walks the Media Controller graph at ``/dev/mediaX``
   so you can see what sensors / ISPs / capture nodes are wired up.
2. ``VideoCapture`` opens a V4L2 capture node at ``/dev/videoX`` and
   pulls BGR frames. Without an active sensor attached the open will
   fail — that's a hardware issue, not the API; the sample prints
   whatever frames it can get before bailing out.

Usage::

    /usr/bin/python3 samples/test_camera_capture.py             # /dev/media0, /dev/video0
    /usr/bin/python3 samples/test_camera_capture.py /dev/media1  # pick a different media dev
    DISPLAY=0 /usr/bin/python3 samples/test_camera_capture.py    # headless

The optional second argument is a frame-count cap; pass ``0`` (default)
to read until the source EOFs or release() is called.
"""

from __future__ import annotations

import os
import sys

try:
    import cv2
    _has_cv2 = True
except ImportError:
    cv2 = None
    _has_cv2 = False

from rkv import MediaDevice, VideoCapture


def print_topology(dev: MediaDevice) -> None:
    """Pretty-print the entities + their first few links."""
    ents = dev.entities()
    print(f"  device: {dev.path}")
    print(f"  {len(ents)} entities:")

    by_name = {e.name: e for e in ents}
    for ent in ents:
        dev_path = ent.device_path or "-"
        print(
            f"    [{ent.id:3d}] {ent.name:30s} type={ent.type:25s} dev={dev_path}"
        )

    links = dev.links()
    print(f"  {len(links)} links:")
    for link in links[:8]:
        marker = "*" if link.enabled else " "
        print(
            f"    {marker} {link.source_entity}:{link.source_pad}"
            f" -> {link.sink_entity}:{link.sink_pad}"
        )
    if len(links) > 8:
        print(f"    ... ({len(links) - 8} more)")

    print("  capture nodes (entities with /dev/videoX paths):")
    for ent in dev.capture_nodes():
        is_subdev = "subdev" in (ent.device_path or "")
        suffix = "  [subdev, configure via media-ctl]" if is_subdev else ""
        print(f"    {ent.device_path}  ({ent.name}){suffix}")


def main(argv: list[str]) -> int:
    media_path = argv[1] if len(argv) > 1 else "/dev/media0"
    max_frames = int(argv[2]) if len(argv) > 2 else 0

    # ---- Phase 1: inspect the media topology ----------------------------
    print(f"== {media_path} topology ==")
    try:
        dev = MediaDevice(media_path)
    except FileNotFoundError as e:
        print(f"  cannot probe media device: {e}")
        return 1

    print_topology(dev)

    # Pick the first V4L2 capture node (/dev/videoX), skipping subdevs
    # (/dev/v4l-subdevX) which aren't directly capturable.
    capture_nodes = [
        e for e in dev.capture_nodes()
        if e.device_path and e.device_path.startswith("/dev/video")
    ]
    if not capture_nodes:
        print(f"  no /dev/videoX capture nodes on {media_path}; nothing to capture")
        return 1

    capture_path = capture_nodes[0].device_path
    print(f"\n== opening capture node {capture_path} ==")

    window_open = _has_cv2 and os.environ.get("DISPLAY", "1") != "0"

    try:
        cap = VideoCapture(capture_path)
    except OSError as e:
        # Most common cause: sensor isn't powered / no source connected,
        # so the kernel refuses to open the capture node.
        print(f"  cannot open {capture_path}: {e}")
        print("  (this usually means no sensor is wired up — the kernel")
        print("   won't activate the CSI/ISP until it sees a streaming source)")
        return 1

    if not cap.isOpened():
        print(f"  cannot open {capture_path} (no active sensor? "
              f"check media-ctl -p and dmesg)")
        return 1

    print(f"  opened {capture_path}: {cap.width}x{cap.height} @ {cap.fps:.2f} fps")

    if window_open and cap.width:
        cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Camera", min(cap.width, 1280), min(cap.height, 720))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if window_open:
            cv2.imshow("Camera", frame)
            if cv2.waitKey(25) & 0xFF == ord("q"):
                break
        elif cap.frame_pos % 30 == 0:
            print(f"  frame {cap.frame_pos}")

        if max_frames and cap.frame_pos >= max_frames:
            break

    print(f"  played {cap.frame_pos} frames")
    cap.release()
    if window_open:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))