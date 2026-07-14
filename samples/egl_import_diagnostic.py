"""eglCreateImageKHR diagnostic — mirror of buffer_gl_image.cpp.

This sample runs every ``eglCreateImageKHR(EGL_LINUX_DMA_BUF_EXT, ...)``
attribute layout that the rbs glengine uses in
``/userdata/rbs/frameworks/glengine/buffer_gl_image.cpp``, with a
DMA-BUF fd allocated via ``/dev/dma_heap/system`` (modern kernel) or
``/dev/dri/card0`` (legacy Rockchip kernel). If every format returns
``EGL_BAD_PARAMETER (0x3009)``, the libmali build on this machine
fundamentally does not honour ``EGL_EXT_image_dma_buf_import`` for
fds the user can actually allocate — the SurfaceFlinger compositor
must fall back to the BGR software upload path.

Run on any target board to triage whether DMA-BUF zero-copy is usable:

    python samples/egl_import_diagnostic.py

The reference platform rbs/glengine runs on (Rockchip 5.x kernel +
their libmali build) reports ``OK`` for every format. The X11/wayland
libmali 1.9 build on a stock Debian 12 install reports ``FAIL`` for
every format.
"""
from __future__ import annotations

import ctypes
import fcntl
import os
import struct
import sys


def make_dma_buf(size: int) -> int:
    """Allocate via ``/dev/dma_heap/system`` (the modern path)."""
    fd = os.open("/dev/dma_heap/system", os.O_RDWR | os.O_CLOEXEC)
    buf = bytearray(24)
    struct.pack_into("<QIIQ", buf, 0, size, 0, 0, 0)
    fcntl.ioctl(fd, 0xC0184800, buf)
    new_fd = struct.unpack_from("<I", buf, 8)[0]
    os.close(fd)
    return new_fd


def _bind(name, args, ret):
    fn = getattr(lib, name)
    fn.argtypes = args
    fn.restype = ret
    return fn


lib = ctypes.CDLL("libmali.so", mode=ctypes.RTLD_GLOBAL)

eglGetDisplay = _bind("eglGetDisplay", [ctypes.c_void_p], ctypes.c_void_p)
eglInitialize = _bind(
    "eglInitialize",
    [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32)],
    ctypes.c_int,
)
eglTerminate = _bind("eglTerminate", [ctypes.c_void_p], ctypes.c_int)
eglChooseConfig = _bind(
    "eglChooseConfig",
    [ctypes.c_void_p,
     ctypes.POINTER(ctypes.c_int),
     ctypes.POINTER(ctypes.c_void_p),
     ctypes.c_int,
     ctypes.POINTER(ctypes.c_int)],
    ctypes.c_int,
)
eglCreateContext = _bind(
    "eglCreateContext",
    [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)],
    ctypes.c_void_p,
)
eglMakeCurrent = _bind(
    "eglMakeCurrent",
    [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p],
    ctypes.c_int,
)
eglCreateImageKHR = _bind(
    "eglCreateImageKHR",
    [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
     ctypes.POINTER(ctypes.c_int)],
    ctypes.c_void_p,
)
eglDestroyImageKHR = _bind("eglDestroyImageKHR", [ctypes.c_void_p, ctypes.c_void_p], ctypes.c_int)
eglGetError = _bind("eglGetError", [], ctypes.c_int)


EGL_DEFAULT_DISPLAY = 0
EGL_NONE = 0x3038
EGL_WIDTH = 0x3056
EGL_HEIGHT = 0x3050
EGL_LINUX_DRM_FOURCC_EXT = 0x327D
EGL_DMA_BUF_PLANE0_FD_EXT = 0x3272
EGL_DMA_BUF_PLANE0_OFFSET_EXT = 0x3273
EGL_DMA_BUF_PLANE0_PITCH_EXT = 0x3274
EGL_DMA_BUF_PLANE1_FD_EXT = 0x3275
EGL_DMA_BUF_PLANE1_OFFSET_EXT = 0x3276
EGL_DMA_BUF_PLANE1_PITCH_EXT = 0x3277
EGL_LINUX_DMA_BUF_EXT = 0x3278
EGL_RED_SIZE = 0x3024
EGL_GREEN_SIZE = 0x3023
EGL_BLUE_SIZE = 0x3022
EGL_ALPHA_SIZE = 0x3021
EGL_RENDERABLE_TYPE = 0x3040
EGL_OPENGL_ES3_BIT = 0x0040

DRM_FORMAT_NV12 = 0x3231564E
DRM_FORMAT_ABGR8888 = 0x34324142
DRM_FORMAT_BGR888 = 0x33524742
DRM_FORMAT_RGB888 = 0x34324752
DRM_FORMAT_R8 = 0x20203852
DRM_FORMAT_RG88 = 0x38385247
DRM_FORMAT_ARGB8888 = 0x34325241
DRM_FORMAT_XRGB8888 = 0x34325258


def run_attempt(dpy, fourcc, w, h, planes, label):
    """Run a single ``eglCreateImageKHR`` call, exactly mirroring the
    per-format attribute layout in ``buffer_gl_image.cpp``."""
    attribs_list = [
        EGL_WIDTH, w,
        EGL_HEIGHT, h,
        EGL_LINUX_DRM_FOURCC_EXT, fourcc,
    ]
    for i, plane in enumerate(planes):
        offset, pitch, plane_fd = plane
        if i == 0:
            attribs_list += [
                EGL_DMA_BUF_PLANE0_FD_EXT, plane_fd,
                EGL_DMA_BUF_PLANE0_OFFSET_EXT, offset,
                EGL_DMA_BUF_PLANE0_PITCH_EXT, pitch,
            ]
        elif i == 1:
            attribs_list += [
                EGL_DMA_BUF_PLANE1_FD_EXT, plane_fd,
                EGL_DMA_BUF_PLANE1_OFFSET_EXT, offset,
                EGL_DMA_BUF_PLANE1_PITCH_EXT, pitch,
            ]
    attribs_list.append(EGL_NONE)
    arr = (ctypes.c_int * len(attribs_list))(*attribs_list)
    img = eglCreateImageKHR(
        dpy, 0, EGL_LINUX_DMA_BUF_EXT, None,
        ctypes.cast(arr, ctypes.POINTER(ctypes.c_int)),
    )
    err = eglGetError()
    if img:
        eglDestroyImageKHR(dpy, img)
        print(f"  {label:35s}: OK 0x{img:x}")
    else:
        print(f"  {label:35s}: FAIL EGL error 0x{err:x}")


def main() -> int:
    dpy = eglGetDisplay(EGL_DEFAULT_DISPLAY)
    eglInitialize(dpy, None, None)
    configs = (ctypes.c_void_p * 16)()
    num = ctypes.c_int()
    attr = (ctypes.c_int * 11)(
        EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8,
        EGL_ALPHA_SIZE, 0, EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT,
        EGL_NONE,
    )
    eglChooseConfig(
        dpy, ctypes.cast(attr, ctypes.POINTER(ctypes.c_int)),
        configs, 16, ctypes.byref(num),
    )
    ctx = eglCreateContext(
        dpy, configs[0], 0,
        ctypes.cast(
            (ctypes.c_int * 3)(0x3098, 3, EGL_NONE),
            ctypes.POINTER(ctypes.c_int),
        ),
    )
    eglMakeCurrent(dpy, 0, 0, ctx)

    fd = make_dma_buf(64 * 64 * 4)
    print(f"dma-buf fd={fd} (size 16384, allocated via /dev/dma_heap/system)")
    print()
    print("=== Per-format eglCreateImageKHR — attribute layout from")
    print("    /userdata/rbs/frameworks/glengine/buffer_gl_image.cpp ===")
    run_attempt(dpy, DRM_FORMAT_ABGR8888, 64, 64, [(0, 64 * 4, fd)],
                "ABGR8888 [case IMAGE_FMT_RGBX8888]")
    run_attempt(dpy, DRM_FORMAT_ARGB8888, 64, 64, [(0, 64 * 4, fd)], "ARGB8888")
    run_attempt(dpy, DRM_FORMAT_BGR888, 64, 64, [(0, 64 * 3, fd)],
                "BGR888 [case IMAGE_FMT_RGB888]")
    run_attempt(dpy, DRM_FORMAT_RGB888, 64, 64, [(0, 64 * 3, fd)],
                "RGB888 [case IMAGE_FMT_BGR888]")
    run_attempt(dpy, DRM_FORMAT_NV12, 64, 64, [(0, 64, fd), (64 * 64, 64, fd)],
                "NV12 2-plane [case NV12/NV16]")
    run_attempt(dpy, DRM_FORMAT_R8, 64, 64, [(0, 64, fd)], "R8 [case IMAGE_FMT_Y8]")
    run_attempt(dpy, DRM_FORMAT_RG88, 64, 64, [(0, 128, fd)], "RG88 [case IMAGE_FMT_RG88]")
    run_attempt(dpy, DRM_FORMAT_XRGB8888, 64, 64, [(0, 64 * 4, fd)], "XRGB8888")

    os.close(fd)
    eglTerminate(dpy)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
