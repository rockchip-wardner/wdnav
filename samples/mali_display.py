"""Single-stream demo: wdnav.VideoCapture -> wdnav.SurfaceComposer -> cv2.imshow.

Demonstrates the compositor with two stacked surfaces:

* the **video** plane is filled from the decoded BGR frame produced by
  the VPU + RGA pipeline (``cap.read(w, h)`` returns an OpenCV-compatible
  BGR ndarray, which we copy into :attr:`Surface.latest_ndarray` and
  hand to the worker thread via :meth:`Surface.post`). This is the
  software path; the zero-copy DMA-BUF path is also wired up — see
  :meth:`VideoCapture.read0` followed by :meth:`Surface.post_fd` —
  but requires a libmali build that actually accepts general dma-buf
  fds through ``EGL_EXT_image_dma_buf_import``. Some X11/wayland libmali
  builds advertise the extension without honouring it; this sample uses
  the BGR path because it works on every build.
* the **OSD** plane is rendered via :attr:`Surface.latest_ndarray` and
  :meth:`Surface.post` so the same path is exercised in software.

Run:

    python samples/mali_display.py /path/to/video.mp4
    python samples/mali_display.py rtsp://user:pass@host/stream

A local file path or an ``rtsp://`` URL is accepted positionally. For
non-terminating sources (RTSP, live cameras) cap the run with ``--max-frames``;
press ``q`` in the display window to stop early.
"""
from __future__ import annotations

import argparse
import sys

import cv2

import wdnav


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Open a local file or a network stream with wdnav.VideoCapture, "
            "feed it into a wdnav.SurfaceComposer, and show the composited result."
        ),
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="bus.mp4",
        help="Input source: a local file path or an rtsp:// URL. Default: %(default)s.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Compositor output width (default %(default)s).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Compositor output height (default %(default)s).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        metavar="N",
        help="Stop after N frames (0 = no cap). Useful for live streams.",
    )
    parser.add_argument(
        "--zero-copy",
        action="store_true",
        help=(
            "Use the DMA-BUF zero-copy path: ``cap.read0()`` returns the "
            "VPU-decoded ``DecodedFrame`` and we post its fd straight into "
            "the Surface via :meth:`Surface.post_fd`. Requires a libmali "
            "build whose EGL_EXT_image_dma_buf_import actually accepts "
            "general dma-buf fds."
        ),
    )
    return parser


def main(args) -> int:
    cap = wdnav.VideoCapture(args.path, timeout=10.0)
    if not cap.isOpened():
        print(f"Error: cannot open {args.path}")
        return 1
    print(f"opened {args.path}: {cap.width}x{cap.height} @ {cap.fps:.2f} fps")

    sf = wdnav.SurfaceComposer(w=args.width, h=args.height)
    if args.zero_copy:
        plan1 = sf.create_surface(0, 0, args.width, args.height, zpos=0, fmt="NV12")
    else:
        plan1 = sf.create_surface(0, 0, args.width, args.height, zpos=0, fmt="BGR888")
    osd_h = max(48, args.height // 8)
    plan2 = sf.create_surface(0, 0, args.width, osd_h, zpos=1, fmt="BGR888")

    cv2.namedWindow("wdnav SurfaceComposer", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("wdnav SurfaceComposer", min(args.width, 1280), min(args.height, 720))

    while True:
        if args.zero_copy:
            ret, frame0 = cap.read0()
            if ret and frame0 is not None and frame0.fd >= 0:
                stride = frame0.hor_stride or frame0.width
                plan1.post_fd(frame0.fd, pts=int(frame0.pts or 0), stride=stride)
        else:
            ret, frame = cap.read(args.width, args.height)
            if ret and frame is not None:
                plan1.latest_ndarray[:] = frame
                plan1.post(pts=cap.frame_pos)
        if not ret:
            break

        osd = plan2.latest_ndarray
        osd[:] = (32, 32, 32)
        cv2.putText(
            osd,
            f"wdnav SurfaceComposer x Mali-G610  frame {cap.frame_pos}",
            (16, max(20, osd_h - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        plan2.post(pts=cap.frame_pos)

        img = sf.read()
        cv2.imshow("wdnav SurfaceComposer", img)
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break
        if args.max_frames and cap.frame_pos >= args.max_frames:
            break

    print(f"played {cap.frame_pos} frames")
    cap.release()
    sf.close()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    args = _build_parser().parse_args(sys.argv[1:])
    raise SystemExit(main(args))
