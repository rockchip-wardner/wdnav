"""Pure-GL smoke test for wdnav.SurfaceComposer.

Verifies the surfaceless EGL context + FBO + PBO + shader path end-to-end
without involving VideoCapture. Useful for debugging compositor issues in
isolation from the decoder.

    python samples/mali_surfaceless_smoke.py [out.png]

Writes a 640x480 PNG of a synthetic gradient composited through a single
BGR888 Surface. Also exercises ``wdnav.DmaHeap`` to allocate a DMA-BUF
fd and shows how to wrap it with ``wdnav.mali.create_image_dma_buf``;
the actual compositing stays on the BGR surface because that path is
guaranteed to work regardless of whether libmali accepts general
DMA-BUF fds.
"""
from __future__ import annotations

import sys

import cv2
import numpy as np

import wdnav


def main() -> int:
    out_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/rkv_surfaceless_smoke.png"
    if wdnav.SurfaceComposer is None:
        print("libmali not loadable; SurfaceComposer is None", file=sys.stderr)
        return 1

    if wdnav.DmaHeap is not None:
        print(f"available dma-heaps: {wdnav.list_heaps()}")
        heap = wdnav.DmaHeap("system")
        buf = heap.alloc(640 * 480 * 3 // 2)  # NV12-sized buffer
        print(f"DmaHeap alloc: fd={buf.fd}, {buf.size}B")
        buf.close()
    else:
        print("/dev/dma_heap not present on this system", file=sys.stderr)

    sf = wdnav.SurfaceComposer(w=640, h=480)
    s = sf.create_surface(0, 0, 640, 480, zpos=0, fmt="BGR888")

    buf = s.latest_ndarray
    yy, xx = np.mgrid[0:480, 0:640]
    buf[..., 0] = xx & 0xFF
    buf[..., 1] = yy & 0xFF
    buf[..., 2] = ((xx + yy) >> 1) & 0xFF

    s.post(pts=1)
    img = sf.read(timeout=5.0)
    if img.shape != (480, 640, 3):
        print(f"FAIL: unexpected shape {img.shape}", file=sys.stderr)
        sf.close()
        return 1

    cv2.imwrite(out_path, img)
    print(f"OK: composited frame written to {out_path}")
    sf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
