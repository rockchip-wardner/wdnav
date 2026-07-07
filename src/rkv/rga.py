"""ctypes wrapper for Rockchip's librga.so (RGA2 2D graphics accelerator).

This module exposes the C im2d API surface that ships in
``/usr/include/rga/`` (the C wrappers ``*_t`` — ``imcvtcolor_t``,
``imresize_t``, ``imcopy_t`` …) so the Python side can do hardware
color-space conversion and scaling without going through the C++ im2d
header. The hardware engine handles YUV ↔ RGB, scaling, cropping and
rotation in a single pass; the alternative (numpy + ``cv2.cvtColor``)
eats ~3-5x the CPU on a 1080p stream.

Library load order:

    1. ``rkv/_native/librga.so.2``     (vendored copy inside the installed wheel)
    2. ``<repo>/native/librga.so.2``   (vendored copy in the source tree)
    3. ``/usr/lib/aarch64-linux-gnu/librga.so.2``  (system install)
    4. ``ctypes.util.find_library("rga")``         (fallback)

Typical usage (NV12 → BGR + optional resize for a decoded frame):

    import rkv.rga as rga

    src = rga.wrap_nv12(ptr=frame.ptr, w=frame.width, h=frame.height,
                         hor_stride=frame.hor_stride)
    dst = rga.alloc_bgr(w=out_w, h=out_h)
    rga.cvtcolor(src, dst)        # or rga.resize(src, dst) for scale-only
    bgr = rga.as_ndarray(dst)     # zero-copy view of the dst buffer
"""

from __future__ import annotations

import ctypes
import os
import threading
from ctypes import (
    POINTER,
    Structure,
    c_char_p,
    c_int,
    c_int32,
    c_size_t,
    c_uint8,
    c_uint16,
    c_uint32,
    c_uint64,
    c_void_p,
)
from typing import Optional

# ---------------------------------------------------------------------------
# Library loading
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))

_LIBS_TO_TRY = (
    # 1) installed wheel: rkv/_native/librga.so.2
    os.path.join(_HERE, "_native", "librga.so.2"),
    # 2) editable / source tree: <repo>/native/librga.so.2
    #    src/rkv/rga.py -> ../../../native/librga.so.2
    os.path.normpath(os.path.join(_HERE, "..", "..", "native", "librga.so.2")),
    # 3) system install
    "/usr/lib/aarch64-linux-gnu/librga.so.2",
)


def _load_library():
    last_err: Optional[OSError] = None
    for path in _LIBS_TO_TRY:
        if os.path.exists(path):
            try:
                return ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            except OSError as e:
                last_err = e
                continue
    try:
        return ctypes.CDLL("rga", mode=ctypes.RTLD_GLOBAL)
    except OSError as e:
        raise OSError(
            "Could not load librga.so.2. Tried: "
            + ", ".join(_LIBS_TO_TRY)
            + ". Last error: "
            + repr(last_err or e)
        )


_lib = _load_library()
# RGA is thread-safe per-context but the legacy initialisation isn't;
# keep a single lock around the C API to keep things simple.
_rga_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Surface formats (subset of rga.h's _Rga_SURF_FORMAT — enough for video)
# ---------------------------------------------------------------------------
#
# Each constant is `(RK_FORMAT_XXX) << 8`, matching the C enum. We only
# carry the ones we need: NV12 / NV21 / I420 for H.264/HEVC output and
# the OpenCV-compatible BGR/BGRX/RGB/RGBA family for the host.

class RgaFormat:
    # YUV — 2 plane (semi-planar, "SP")
    YCbCr_420_SP = 0x0A << 8   # NV12: YYYY...UV...
    YCrCb_420_SP = 0x0E << 8   # NV21: YYYY...VU...
    YCbCr_422_SP = 0x08 << 8
    YCrCb_422_SP = 0x0C << 8
    YCbCr_420_P  = 0x0B << 8   # I420: YYYY...U...V...
    YCrCb_420_P  = 0x0F << 8
    YV12_420_P   = 0x0B << 8  # alias for I420

    # RGB / BGR — packed
    RGB_565    = 0x04 << 8
    RGB_888    = 0x02 << 8
    BGR_888    = 0x07 << 8
    RGBA_8888  = 0x00 << 8
    BGRA_8888  = 0x03 << 8
    RGBX_8888  = 0x01 << 8
    BGRX_8888  = 0x16 << 8


# BPP lookup, used for buffer-size calculation when RGA allocates
# destination storage. Values match the C ``get_bpp_from_format`` helper
# in RgaUtils.h.
_BPP = {
    RgaFormat.YCbCr_420_SP: 1.5,
    RgaFormat.YCrCb_420_SP: 1.5,
    RgaFormat.YCbCr_422_SP: 2,
    RgaFormat.YCrCb_422_SP: 2,
    RgaFormat.YCbCr_420_P:  1.5,
    RgaFormat.YCrCb_420_P:  1.5,
    RgaFormat.RGB_565:      2,
    RgaFormat.RGB_888:      3,
    RgaFormat.BGR_888:      3,
    RgaFormat.RGBA_8888:    4,
    RgaFormat.BGRA_8888:    4,
    RgaFormat.RGBX_8888:    4,
    RgaFormat.BGRX_8888:    4,
}


def bytes_per_pixel(fmt: int) -> float:
    """BPP for a format, as a float. NV12/I420 = 1.5, RGB = 3, RGBA = 4."""
    try:
        return _BPP[fmt]
    except KeyError as e:
        raise ValueError(f"unknown RGA format: 0x{fmt:x}") from e


# ---------------------------------------------------------------------------
# C structs (im2d_type.h)
# ---------------------------------------------------------------------------


class im_rect(Structure):
    _fields_ = [
        ("x", c_int),
        ("y", c_int),
        ("width", c_int),
        ("height", c_int),
    ]


class im_handle_param(Structure):
    _fields_ = [
        ("width", c_uint32),
        ("height", c_uint32),
        ("format", c_uint32),
    ]


class rga_buffer_t(Structure):
    """Mirrors ``rga_buffer_t`` in ``im2d_type.h``.

    We only populate the fields that the C im2d API actually reads when
    it receives a ``rga_buffer_t`` by value: ``vir_addr`` / ``fd`` /
    ``width`` / ``height`` / ``wstride`` / ``hstride`` / ``format``.
    Everything else stays zero-initialised.
    """
    _fields_ = [
        ("vir_addr", c_void_p),
        ("phy_addr", c_void_p),
        ("fd", c_int),
        ("width", c_int),
        ("height", c_int),
        ("wstride", c_int),
        ("hstride", c_int),
        ("format", c_int),
        ("color_space_mode", c_int),
        ("global_alpha", c_int),
        ("rd_mode", c_int),
        ("color", c_int),
        ("colorkey_range_max", c_int),
        ("colorkey_range_min", c_int),
        ("nn", c_uint8 * 24),     # im_nn_t — opaque, only matters for NN ops
        ("rop_code", c_int),
        ("handle", c_void_p),
    ]


# Mirror the IM_STATUS enum
IM_STATUS_NOERROR = 2
IM_STATUS_SUCCESS = 1
IM_STATUS_NOT_SUPPORTED = -1
IM_STATUS_FAILED = -2
IM_STATUS_OUT_OF_MEMORY = -3
IM_STATUS_INVALID_PARAM = -4
IM_STATUS_ILLEGAL_PARAM = -5
IM_STATUS_ERROR_VERSION = -6
IM_STATUS_NO_SESSION = -7

# sync mode
IM_SYNC = 1 << 19

# color-space mode
IM_COLOR_SPACE_DEFAULT = 0

# interpolation modes
INTER_NEAREST = 0
INTER_LINEAR = 1
INTER_CUBIC = 2


# ---------------------------------------------------------------------------
# C function bindings
# ---------------------------------------------------------------------------

# Buffer import / release
_lib.importbuffer_virtualaddr.argtypes = [c_void_p, POINTER(im_handle_param)]
_lib.importbuffer_virtualaddr.restype = c_void_p  # rga_buffer_handle_t

_lib.wrapbuffer_virtualaddr_t.argtypes = [
    c_void_p, c_int, c_int, c_int, c_int, c_int,
]
_lib.wrapbuffer_virtualaddr_t.restype = rga_buffer_t

_lib.wrapbuffer_fd_t.argtypes = [
    c_int, c_int, c_int, c_int, c_int, c_int,
]
_lib.wrapbuffer_fd_t.restype = rga_buffer_t

_lib.wrapbuffer_handle_t.argtypes = [
    c_void_p, c_int, c_int, c_int, c_int, c_int,
]
_lib.wrapbuffer_handle_t.restype = rga_buffer_t

_lib.importbuffer_fd.argtypes = [c_int, POINTER(im_handle_param)]
_lib.importbuffer_fd.restype = c_void_p  # rga_buffer_handle_t

_lib.releasebuffer_handle.argtypes = [c_void_p]
_lib.releasebuffer_handle.restype = c_int

# Operations (the *_t C variants — no default args, no name mangling)
_lib.imcopy_t.argtypes = [rga_buffer_t, rga_buffer_t, c_int]
_lib.imcopy_t.restype = c_int

_lib.imcvtcolor_t.argtypes = [
    rga_buffer_t, rga_buffer_t, c_int, c_int, c_int, c_int,
]
_lib.imcvtcolor_t.restype = c_int

_lib.imresize_t.argtypes = [
    rga_buffer_t, rga_buffer_t, c_double_p, c_double_p, c_int, c_int,
] if False else None  # placeholder, set below
_lib.imresize_t.argtypes = [
    rga_buffer_t, rga_buffer_t, ctypes.c_double, ctypes.c_double, c_int, c_int,
]
_lib.imresize_t.restype = c_int

_lib.imcheck_t.argtypes = [
    rga_buffer_t, rga_buffer_t, rga_buffer_t,
    im_rect, im_rect, im_rect, c_int,
]
_lib.imcheck_t.restype = c_int

# imcheck takes its rga_buffer_t args by VALUE, not pointer. im2d_common.h:
#   IM_C_API IM_STATUS imcheck_t(const rga_buffer_t src, const rga_buffer_t dst,
#                                const rga_buffer_t pat, ...);
# ctypes passes small structs in registers / by value, but for safety we
# leave it as-is and let ctypes use the natural calling convention.

_lib.c_RkRgaInit.argtypes = []
_lib.c_RkRgaInit.restype = c_int

_lib.c_RkRgaDeInit.argtypes = []
_lib.c_RkRgaDeInit.restype = None

_lib.c_RkRgaFlush.argtypes = []
_lib.c_RkRgaFlush.restype = c_int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check(ret: int, where: str) -> None:
    """Raise if RGA returns a negative / error status."""
    if ret < 0 or ret not in (IM_STATUS_SUCCESS, IM_STATUS_NOERROR):
        raise RuntimeError(f"RGA call {where!r} failed: status={ret}")


def init() -> None:
    """Idempotent RGA initialisation. The im2d context is auto-allocated
    on first use, so this is only needed for the legacy C API."""
    with _rga_lock:
        _lib.c_RkRgaInit()


def shutdown() -> None:
    """Release the RGA context. Mostly useful in tests; production
    callers can let the OS reap the kernel state on process exit."""
    with _rga_lock:
        _lib.c_RkRgaDeInit()


# ---------------------------------------------------------------------------
# Buffer wrappers — keep these alive as long as you need the underlying
# rga_buffer_t struct (which holds the address / handle the C API reads).
# ---------------------------------------------------------------------------


class Image:
    """An RGA-side image handle.

    Holds the ``rga_buffer_t`` struct that the C im2d functions consume
    *by value*. When you pass an ``Image`` to ``cvtcolor`` / ``resize``,
    a copy of the struct is made — so you can let the Image go out of
    scope once the call returns. The underlying memory (the RGA handle
    or the wrapped virtual address) outlives the Image only if you
    imported it via :func:`import_virtual`; raw wrapping of an external
    pointer is the caller's responsibility.
    """

    __slots__ = ("_buf", "_handle", "_owned_buf")

    def __init__(self, buf: rga_buffer_t, handle: Optional[int] = None,
                 owned_buf: Optional["AllocatedBuffer"] = None) -> None:
        # _buf is the value that goes into C. _handle is kept alive so
        # releasebuffer_handle is only called when we own it.
        self._buf = buf
        self._handle = handle
        self._owned_buf = owned_buf

    @property
    def raw(self) -> rga_buffer_t:
        return self._buf

    @property
    def width(self) -> int:
        return self._buf.width

    @property
    def height(self) -> int:
        return self._buf.height

    @property
    def format(self) -> int:
        return self._buf.format

    def __del__(self) -> None:
        try:
            if self._handle:
                _lib.releasebuffer_handle(self._handle)
        except Exception:
            pass


class AllocatedBuffer:
    """A buffer RGA allocated for us (via ``c_RkRgaGetAllocBuffer``).

    Owns the virtual pointer; the RGA handle is released on destruction.
    Call ``as_ndarray(...)`` to get a zero-copy numpy view of the pixels.
    """

    __slots__ = ("_ptr", "_size", "_handle", "_shape", "_dtype")

    def __init__(self, ptr: int, size: int, handle: int,
                 width: int, height: int, fmt: int) -> None:
        self._ptr = ptr
        self._size = size
        self._handle = handle
        # RGB/BGR: 3 bytes/px; RGBA: 4 bytes/px; NV12: 1.5 bytes/px (Y then UV).
        bpp = bytes_per_pixel(fmt)
        if bpp == int(bpp):
            # packed RGB
            self._dtype = ctypes.c_uint8 * int(size)
            self._shape = (height, width, int(bpp))
        else:
            # planar YUV (NV12 / I420): we expose a 1D buffer; the caller
            # is expected to know the layout.
            self._dtype = ctypes.c_uint8 * size
            self._shape = (size,)
        self._handle = handle

    @property
    def ptr(self) -> int:
        return self._ptr

    @property
    def size(self) -> int:
        return self._size

    def as_ndarray(self):
        """View the buffer as a numpy array (zero copy)."""
        import numpy as np
        buf = (self._dtype).from_address(self._ptr)
        return np.frombuffer(buf, dtype=np.uint8).reshape(*self._shape)

    def __del__(self) -> None:
        try:
            if self._handle:
                _lib.releasebuffer_handle(self._handle)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Image construction
# ---------------------------------------------------------------------------


def wrap_virtual(
    vir_addr: int,
    width: int,
    height: int,
    format: int,
    hstride: int = 0,
    wstride: int = 0,
) -> Image:
    """Wrap an existing virtual-address buffer as an RGA image.

    Use this for buffers you already own (numpy arrays, MPP frame
    pointers, etc.) — RGA reads from the address in place. No copy.
    """
    if hstride == 0:
        hstride = height
    if wstride == 0:
        wstride = width
    with _rga_lock:
        buf = _lib.wrapbuffer_virtualaddr_t(
            ctypes.c_void_p(vir_addr),
            width, height, wstride, hstride, format,
        )
    return Image(buf)


def wrap_fd(
    fd: int,
    width: int,
    height: int,
    format: int,
    hstride: int = 0,
    wstride: int = 0,
) -> Image:
    """Wrap a DMA-BUF fd as an RGA image.

    Zero-copy path for MPP → RGA: both engines speak DMA, so the YUV
    plane lives in a single CMA heap allocation throughout the
    decode → colour-convert → display chain. Avoids the CPU-side
    cache-coherency dance that ``wrap_virtual`` would otherwise need.
    """
    if hstride == 0:
        hstride = height
    if wstride == 0:
        wstride = width
    with _rga_lock:
        buf = _lib.wrapbuffer_fd_t(
            fd, width, height, wstride, hstride, format,
        )
    return Image(buf)


def wrap_nv12(
    ptr: int,
    width: int,
    height: int,
    hor_stride: int = 0,
    ver_stride: int = 0,
) -> Image:
    """Convenience wrapper for NV12 (MPP's default decoder output).

    Equivalent to ``wrap_virtual(ptr, width, height, RK_FORMAT_YCbCr_420_SP,
    wstride=hor_stride, hstride=ver_stride)``. Prefer
    :func:`wrap_nv12_fd` when the source is an MPP frame, since
    RGA + VPU both speak DMA-BUF and you save a kernel crossing.
    """
    return wrap_virtual(
        ptr, width, height, RgaFormat.YCbCr_420_SP,
        wstride=hor_stride or width, hstride=ver_stride or height,
    )


def wrap_nv12_fd(
    fd: int,
    width: int,
    height: int,
    hor_stride: int = 0,
    ver_stride: int = 0,
) -> Image:
    """Wrap a DMA-BUF fd holding an NV12 image (MPP's default output).

    The fd is the same one returned by ``MppDecoder``'s
    ``DecodedFrame.fd``; passing it through means the YUV plane stays
    in the same CMA allocation for decode → RGA → display, with no
    CPU cache-management overhead.
    """
    return wrap_fd(
        fd, width, height, RgaFormat.YCbCr_420_SP,
        wstride=hor_stride or width, hstride=ver_stride or height,
    )


def import_virtual(vir_addr: int, width: int, height: int, format: int) -> Image:
    """Import a virtual-address buffer into RGA's handle table.

    RGA's handle table lets the hardware use cached buffer metadata for
    better performance than wrapping the address every call. Use this for
    long-lived buffers; for one-shot conversions, :func:`wrap_virtual` is
    fine.
    """
    param = im_handle_param(width=width, height=height, format=format)
    with _rga_lock:
        handle = _lib.importbuffer_virtualaddr(ctypes.c_void_p(vir_addr), ctypes.byref(param))
    buf = _lib.wrapbuffer_handle_t(
        handle, width, height, width, height, format,
    )
    return Image(buf, handle=handle)


def import_fd(fd: int, width: int, height: int, format: int) -> Image:
    """Import a DMA-BUF fd into RGA's handle table.

    For long-lived DMA buffers, this avoids the per-call handle setup
    that ``wrap_fd`` would otherwise do. The handle is released when
    the returned :class:`Image` goes out of scope.
    """
    param = im_handle_param(width=width, height=height, format=format)
    with _rga_lock:
        handle = _lib.importbuffer_fd(fd, ctypes.byref(param))
    buf = _lib.wrapbuffer_handle_t(
        handle, width, height, width, height, format,
    )
    return Image(buf, handle=handle)


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------


def cvtcolor(src: Image, dst: Image, sync: bool = True) -> None:
    """Convert ``src`` to ``dst``'s pixel format. NV12 → BGR888 is the
    common case for video decode."""
    with _rga_lock:
        ret = _lib.imcvtcolor_t(
            src.raw, dst.raw,
            src.format, dst.format,
            IM_COLOR_SPACE_DEFAULT,
            IM_SYNC if sync else 0,
        )
    _check(ret, "imcvtcolor_t")


def resize(src: Image, dst: Image, interpolation: int = INTER_LINEAR,
           sync: bool = True) -> None:
    """Scale ``src`` to ``dst``'s dimensions. The destination rect drives
    the output size; if you want explicit scale factors, compute the
    destination size first and call this."""
    with _rga_lock:
        ret = _lib.imresize_t(
            src.raw, dst.raw,
            0.0, 0.0,
            interpolation,
            IM_SYNC if sync else 0,
        )
    _check(ret, "imresize_t")


def copy(src: Image, dst: Image, sync: bool = True) -> None:
    """Raw RGA copy — useful when src/dst have the same format and you
    just want a hardware DMA."""
    with _rga_lock:
        ret = _lib.imcopy_t(src.raw, dst.raw, IM_SYNC if sync else 0)
    _check(ret, "imcopy_t")


def flush() -> None:
    """Block until all queued RGA operations complete."""
    with _rga_lock:
        _lib.c_RkRgaFlush()


__all__ = [
    "RgaFormat",
    "Image",
    "rga_buffer_t",
    "im_rect",
    "im_handle_param",
    "bytes_per_pixel",
    "wrap_virtual",
    "wrap_nv12",
    "wrap_fd",
    "wrap_nv12_fd",
    "import_virtual",
    "import_fd",
    "cvtcolor",
    "resize",
    "copy",
    "flush",
    "init",
    "shutdown",
]