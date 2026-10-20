"""HDMI-IN (rk_hdmirx) capture via wdnav.v4l2.

Verifies that the bundled V4L2 helpers can open RK3588's HDMI-RX capture
node and stream a few frames. Run on the device, with the rockchip user
(a member of the ``video`` group) or as root:

    cd .venv/src/wdnav
    python -m samples.hdmiin_capture

Default settings match what the kernel actually negotiates on the stock
board: 1920x1080 BGR3, 4 mmap'd buffers. Override with CLI flags.

Exit code:
  0 — captured at least one frame
  1 — device not found, or open() failed
  2 — captured zero frames in the timeout (signal/format problem)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

# Allow `python -m samples.hdmiin_capture` from .venv/src/wdnav
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))


def _find_hdmiin_device() -> str:
    """Return the /dev/videoN path that hosts the rk_hdmirx driver.

    Falls back to ``/dev/video0`` if the sysfs scan doesn't find it
    (the stock board always has HDMI-IN on video0).
    """
    v4l2_root = Path("/sys/class/video4linux")
    if not v4l2_root.is_dir():
        return "/dev/video0"
    for entry in sorted(v4l2_root.iterdir()):
        try:
            name = (entry / "name").read_text().strip()
        except OSError:
            continue
        if "hdmirx" in name.lower() or "hdmi_rx" in name.lower():
            return "/dev/" + entry.name
    return "/dev/video0"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--device", default=None,
                   help="Override V4L2 device path (default: auto-detect rk_hdmirx).")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--pixfmt", default="bgr3",
                   help="Pixel format FourCC. rk_hdmirx's native FourCC "
                        "is BGR3 (\"bgr3\"); asking for nv12 / nv21 / yuyv "
                        "fails S_FMT with EINVAL on this kernel. Pass "
                        "--pixfmt=nv12 if your board actually negotiates "
                        "another format and you want to force the attempt.")
    p.add_argument("--buffers", type=int, default=4,
                   help="Number of mmap'd capture buffers (≥ 2).")
    p.add_argument("--frames", type=int, default=10,
                   help="Number of frames to capture before exiting.")
    p.add_argument("--save-jpeg", default=None,
                   help="If set, save the first frame as a JPEG here (via cv2/av).")
    args = p.parse_args()

    device = args.device or _find_hdmiin_device()
    if not Path(device).exists():
        print(f"FATAL: {device} not found", file=sys.stderr)
        return 1

    print(f"Opening {device} ({args.width}x{args.height} {args.pixfmt}, "
          f"buffers={args.buffers})")
    from wdnav.v4l2 import V4L2Capture  # imported lazily so the test can
                                        # fail with a useful message before
                                        # the (heavy) mpp import chain

    try:
        cap = V4L2Capture(device,
                          width=args.width, height=args.height,
                          pixel_format=args.pixfmt,
                          num_buffers=args.buffers)
    except PermissionError as e:
        print(f"FATAL: cannot open {device}: {e}\n"
              "       make sure you are root or in the 'video' group.",
              file=sys.stderr)
        return 1
    except OSError as e:
        print(f"FATAL: {e}", file=sys.stderr)
        return 1

    print(f"driver={cap.driver!r} card={cap.card!r} "
          f"actual={cap.width}x{cap.height} fmt={cap.pixel_format!r} "
          f"planes={cap.num_planes} mplane={cap.is_multiplanar}")

    cap.start()
    t0 = time.time()
    captured = 0
    saved = False
    try:
        for i in range(args.frames):
            buf = cap.dequeue(timeout_s=2.0)
            if buf is None:
                print(f"  frame {i}: timeout (no buffer within 2s)")
                continue
            captured += 1
            # Show per-frame luma mean — sanity check that we got real
            # pixel data and not all-zero garbage.
            luma = buf.plane_arrays[0]
            mean = float(luma.mean())
            seq = buf.sequence
            ts = buf.timestamp
            print(f"  frame {i}: idx={buf.index} seq={seq} ts={ts:.3f} "
                  f"luma_mean={mean:.1f} bytes_per_plane="
                  f"{[len(p) for p in buf.plane_arrays]}")
            if (not saved) and args.save_jpeg:
                # Bypass cv2 / libjpeg — wdnav is the only allowed path
                # for image I/O. We hand the first plane (luma) to av
                # as a grayscale frame so we don't need cv2 just for
                # smoke-test JPEG output. If av is missing, skip.
                try:
                    import av
                    container = av.open(args.save_jpeg, "w")
                    stream = container.add_stream("mjpeg", rate=1)
                    stream.width = luma.shape[1]
                    stream.height = luma.shape[0]
                    stream.pix_fmt = "gray"
                    frame = av.VideoFrame(0, format="gray")
                    frame = av.VideoFrame.from_ndarray(luma, format="gray")
                    for packet in stream.encode(frame):
                        container.mux(packet)
                    container.close()
                    print(f"  saved {args.save_jpeg} (grayscale, {luma.shape})")
                    saved = True
                except Exception as e:
                    print(f"  save_jpeg skipped: {e!r}")
    finally:
        cap.close()
    elapsed = time.time() - t0
    fps = captured / elapsed if elapsed > 0 else 0.0
    print(f"Captured {captured}/{args.frames} frames in {elapsed:.2f}s "
          f"({fps:.1f} fps)")
    return 0 if captured > 0 else 2


if __name__ == "__main__":
    sys.exit(main())
