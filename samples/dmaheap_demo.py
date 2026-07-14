"""Standalone demo of rkv.DmaHeap — allocate DMA-BUF fds and use them.

The DMA-BUF heap allocator is the canonical way to obtain fd-backed
buffers that can be handed across hardware engines (VPU, RGA, EGL via
``EGL_EXT_image_dma_buf_import``). This sample allocates a buffer,
writes a checkerboard into it via mmap, and reports what we got back.

The interesting follow-on is to wrap the fd with
``rkv.mali.create_image_dma_buf`` (matching the layout in
``/userdata/rbs/frameworks/glengine/buffer_gl_image.cpp``) so the
SurfaceFlinger worker can composite it on the GPU.

    python samples/dmaheap_demo.py [heap-name] [size]
"""
from __future__ import annotations

import mmap
import sys

import rkv


def main() -> int:
    if rkv.DmaHeap is None:
        print("/dev/dma_heap not present on this system", file=sys.stderr)
        return 1

    heaps = rkv.list_heaps()
    if not heaps:
        print("no dma-heap devices available", file=sys.stderr)
        return 1
    print(f"available dma-heaps: {heaps}")

    name = sys.argv[1] if len(sys.argv) > 1 else "system"
    size = int(sys.argv[2]) if len(sys.argv) > 2 else (1920 * 1088 * 3 // 2)

    with rkv.DmaHeap(name) as heap:
        with heap.alloc(size) as buf:
            print(f"heap={name}  size={buf.size}  fd={buf.fd}")

            try:
                mm = buf.mmap()
            except OSError as e:
                print(f"mmap unavailable on this heap: {e}", file=sys.stderr)
                return 0

            half = (size // 2) // mmap.PAGESIZE * mmap.PAGESIZE
            for offset in range(0, half, mmap.PAGESIZE):
                page = (offset // mmap.PAGESIZE) & 0xFF
                mm[offset:offset + mmap.PAGESIZE] = bytes([page]) * mmap.PAGESIZE
            print("wrote checkerboard to first half")
            mm.close()

            try:
                buf.sync_read()
                print("dma_buf sync READ ok")
            except OSError as e:
                print(f"dma_buf sync failed: {e}", file=sys.stderr)

    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
