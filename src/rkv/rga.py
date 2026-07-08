"""ctypes wrapper for Rockchip's im2d C API (librga.so).

This module binds the **im2d C entry points** (``imcvtcolor_t``,
``imresize_t``, ``importbuffer_*``, ``wrapbuffer_handle_t``,
``querystring``, ``imcheck``) that ship in ``/usr/include/rga/`` and
are documented under ``im2d.h`` / ``im2d_single.h`` / ``im2d_common.h``.

Why im2d (not the older ``c_RkRga*`` API):
  - **Auto-dispatches between RGA2 and RGA3** at runtime. The im2d
    context (``g_im2d_context``) probes the kernel at load time and
    routes every call to whichever core is available. RGA2 (legacy
    VOP, RK356x and earlier) and RGA3 (the new multi-core engine on
    RK3588 / RK3568 / RK3562 etc.) are picked transparently — a
    wheel built for RK356x still works unmodified on RK3588.

  - **Handle-based buffer management.** Importing a buffer (VA, fd
    or physical) returns a stable ``rga_buffer_handle_t``; the
    library caches its stride / format / size and reuses the mapping
    across calls. The old direct-``wrapbuffer_virtualaddr_t`` path
    bypasses that cache and can miss driver-side stride alignment
    (which is exactly the ``status=-1`` we kept hitting on this
    RGA2 build for some 1366x768 sources).

  - **``querystring``** reports the active backend, supported
    formats, and resolution limits at runtime, so we can fail
    gracefully when the underlying RGA can't do a particular
    conversion.

The C++ im2d.hpp wrappers above this layer (overloaded ``imcvtcolor``
etc., ``importbuffer_fd(int, int)`` 2-arg) are convenient for C++
but add name-mangling to a C linkage. We use the C variants throughout.

Typical use (NV12 fd from MPP → BGR + resize on hardware):

    import rkv.rga as rga

    src = rga.import_nv12_fd(frame.fd, frame.width, frame.height)
    dst = rga.import_bgr_va(dst_addr, out_w, out_h)
    rga.cvtcolor(src, dst)        # colour convert
    rga.resize(src, dst)          # or scale in a single RGA pass
    rga.flush()
"""

from __future__ import annotations

import ctypes
import os
import threading
from ctypes import (
    POINTER,
    Structure,
    Union,
    c_char_p,
    c_double,
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
            "Could not load librga.so.2 (im2d). Tried: "
            + ", ".join(_LIBS_TO_TRY)
            + ". Last error: "
            + repr(last_err or e)
        )


_lib = _load_library()
# RGA's im2d context is per-process; serialise all calls into a single
# lock to keep multi-threading simple. The kernel side does the
# real work in parallel on the hardware cores. We use RLock (not Lock)
# because the public functions (import_*, cvtcolor, ...) call into
# internal helpers that also acquire the lock; a non-reentrant Lock
# would deadlock the first nested call.
_rga_lock = threading.RLock()


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

    # Luma-only — write just the Y plane, no chroma
    YCbCr_400    = 0x15 << 8   # single-channel grayscale

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
    RgaFormat.YCbCr_400:    1,    # luma-only, 1 byte/pixel
    RgaFormat.RGB_565:      2,
    RgaFormat.RGB_888:      3,
    RgaFormat.BGR_888:      3,
    RgaFormat.RGBA_8888:    4,
    RgaFormat.BGRA_8888:    4,
    RgaFormat.RGBX_8888:    4,
    RgaFormat.BGRX_8888:    4,
}


# Reverse lookup: format code -> symbolic name. Built once at
# import from the public RgaFormat class so users (and warning
# messages) see "BGR_888" instead of "0x700". Built from
# vars() so any new constants added to RgaFormat are picked
# up automatically. dunder / private attributes are filtered
# out. Aliased formats (e.g. YV12_420_P == YCbCr_420_P) collapse
# to whichever name comes first in vars() ordering; that's
# fine for human-readable diagnostics.
_FORMAT_NAMES: dict = {
    v: k for k, v in vars(RgaFormat).items()
    if not k.startswith("_") and isinstance(v, int)
}


def format_name(fmt: int) -> str:
    """Return the symbolic name for a RgaFormat code (e.g. ``"BGR_888"``).

    Falls back to ``"0x<hex>"`` for unknown codes — useful when a
    future rga.h revision adds a format we haven't taught the
    wrapper about yet, or when a flag-packed value (e.g. stride
    bits in the low byte) doesn't match a constant exactly.
    """
    return _FORMAT_NAMES.get(fmt, f"0x{fmt:x}")


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


class im_nn_t(Structure):
    """The opaque im_nn_t; only matters for the quantize op."""

    _fields_ = [
        ("scale_r", c_int),
        ("scale_g", c_int),
        ("scale_b", c_int),
        ("offset_r", c_int),
        ("offset_g", c_int),
        ("offset_b", c_int),
    ]


class im_handle_param(Structure):
    """``im_handle_param_t`` — buffer geometry passed to importbuffer_*.

    The im2d context uses width / height / format to size the cached
    mapping and the stride cache, so we must set all three before
    calling import.
    """

    _fields_ = [
        ("width", c_uint32),
        ("height", c_uint32),
        ("format", c_uint32),
    ]


class rga_buffer_t(Structure):
    """``rga_buffer_t`` from im2d_type.h.

    The C struct has an anonymous union at the end (alpha0/alpha1 vs
    global_alpha); ctypes doesn't model that cleanly so we lay it out
    explicitly as ``global_alpha`` (an int covering both cases — the
    alpha-bit mode is only used in obscure OSD code paths).

    The trailing fields ``colorkey_range``, ``nn`` and ``rop_code``
    are kept as opaque byte arrays so the struct size matches the
    C layout exactly. Substituting real ``im_colorkey_range`` /
    ``im_nn_t`` structs here changes the ctypes-computed size and
    leaves the trailing fields uninitialised when the buffer is
    passed by value to ``imcvtcolor_t`` — the im2d context then
    reads junk for the handle and fails to dispatch.
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
        ("nn", c_uint8 * 24),           # im_nn_t (24 bytes) — opaque
        ("rop_code", c_int),
        ("handle", c_void_p),            # rga_buffer_handle_t (uint32_t on
                                        # disk, but the C function reads it as
                                        # a 32-bit value at the right offset
                                        # regardless of how ctypes labels the
                                        # field).
    ]


# Mirror the IM_INFORMATION enum
_RGA_VENDOR = 0
_RGA_VERSION = 1
_RGA_MAX_INPUT = 2
_RGA_MAX_OUTPUT = 3
_RGA_BYTE_STRIDE = 4
_RGA_SCALE_LIMIT = 5
_RGA_INPUT_FORMAT = 6
_RGA_OUTPUT_FORMAT = 7
_RGA_FEATURE = 8
_RGA_EXPECTED = 9
_RGA_ALL = 10

# Mirror the IM_STATUS enum (im2d_type.h)
IM_STATUS_NOERROR = 2
IM_STATUS_SUCCESS = 1
IM_STATUS_NOT_SUPPORTED = -1
IM_STATUS_FAILED = -2
IM_STATUS_OUT_OF_MEMORY = -3
IM_STATUS_INVALID_PARAM = -4
IM_STATUS_ILLEGAL_PARAM = -5
IM_STATUS_ERROR_VERSION = -6
IM_STATUS_NO_SESSION = -7

# sync mode flag
IM_SYNC = 1 << 19

# color-space mode
IM_COLOR_SPACE_DEFAULT = 0

# interpolation modes
INTER_NEAREST = 0
INTER_LINEAR = 1
INTER_CUBIC = 2


# ---------------------------------------------------------------------------
# C function bindings (im2d C ABI — names match the headers exactly)
# ---------------------------------------------------------------------------

# rga_buffer_handle_t is uint32_t; we already declared rga_buffer_t.handle
# as c_uint32. For function signatures that pass it as an opaque
# argument we can use c_void_p (ctypes passes the integer through).

# ---- Buffer import / release (im2d_buffer.h) ----
_lib.importbuffer_virtualaddr.argtypes = [c_void_p, POINTER(im_handle_param)]
_lib.importbuffer_virtualaddr.restype = c_uint32   # rga_buffer_handle_t

_lib.importbuffer_fd.argtypes = [c_int, POINTER(im_handle_param)]
_lib.importbuffer_fd.restype = c_uint32

_lib.importbuffer_physicaladdr.argtypes = [c_uint64, POINTER(im_handle_param)]
_lib.importbuffer_physicaladdr.restype = c_uint32

_lib.releasebuffer_handle.argtypes = [c_uint32]
_lib.releasebuffer_handle.restype = c_int

# ---- Buffer wrapping (im2d_buffer.h) ----
# wrapbuffer_handle_t(handle, w, h, wstride, hstride, format) -> rga_buffer_t
_lib.wrapbuffer_handle_t.argtypes = [
    c_uint32, c_int, c_int, c_int, c_int, c_int,
]
_lib.wrapbuffer_handle_t.restype = rga_buffer_t

# wrapbuffer_virtualaddr_t / wrapbuffer_fd_t also exported by the library,
# but per the im2d docs the handle-based path is the recommended one.
_lib.wrapbuffer_virtualaddr_t.argtypes = [
    c_void_p, c_int, c_int, c_int, c_int, c_int,
]
_lib.wrapbuffer_virtualaddr_t.restype = rga_buffer_t

_lib.wrapbuffer_fd_t.argtypes = [
    c_int, c_int, c_int, c_int, c_int, c_int,
]
_lib.wrapbuffer_fd_t.restype = rga_buffer_t

# ---- Operations (im2d_single.h) — ``*_t`` C entry points ----
_lib.imcopy_t.argtypes = [rga_buffer_t, rga_buffer_t, c_int]
_lib.imcopy_t.restype = c_int

_lib.imcvtcolor_t.argtypes = [
    rga_buffer_t, rga_buffer_t, c_int, c_int, c_int, c_int,
]
_lib.imcvtcolor_t.restype = c_int

_lib.imresize_t.argtypes = [
    rga_buffer_t, rga_buffer_t, c_double, c_double, c_int, c_int,
]
_lib.imresize_t.restype = c_int

# ---- imcheck / querystring / imStrError (im2d_common.h) ----
# imcheck_t(src, dst, pat, src_rect, dst_rect, pat_rect, mode_usage) -> IM_STATUS
_lib.imcheck_t.argtypes = [
    rga_buffer_t, rga_buffer_t, rga_buffer_t,
    im_rect, im_rect, im_rect, c_int,
]
_lib.imcheck_t.restype = c_int

_lib.querystring.argtypes = [c_int]
_lib.querystring.restype = c_char_p

_lib.imStrError_t.argtypes = [c_int]
_lib.imStrError_t.restype = c_char_p

# Legacy c_RkRgaFlush still exported by librga for old code paths; we
# use it because im2d doesn't expose a global flush via the C ABI.
_lib.c_RkRgaFlush.argtypes = []
_lib.c_RkRgaFlush.restype = c_int


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

def _query_property(name: int) -> str:
    """Call ``querystring(name)`` and return the result as a Python str.

    Returns ``"<unavailable>"`` if the library can't answer (older
    librga.so without querystring would fail with a symbol error; we
    catch it and degrade gracefully).
    """
    try:
        with _rga_lock:
            ptr = _lib.querystring(name)
    except AttributeError:
        return "<querystring unavailable>"
    if not ptr:
        return ""
    return ctypes.string_at(ptr).decode("ascii", "replace")


def vendor() -> str:
    """The RGA vendor string — e.g. ``"Rockchip"``."""
    return _query_property(_RGA_VENDOR)


def version() -> str:
    """The RGA version string — distinguishes RGA2 (legacy) from RGA3."""
    return _query_property(_RGA_VERSION)


def max_input_resolution() -> str:
    """Maximum input resolution the active backend can handle."""
    return _query_property(_RGA_MAX_INPUT)


def max_output_resolution() -> str:
    return _query_property(_RGA_MAX_OUTPUT)


def input_formats() -> str:
    """List of supported input pixel formats."""
    return _query_property(_RGA_INPUT_FORMAT)


def output_formats() -> str:
    return _query_property(_RGA_OUTPUT_FORMAT)


def imstrerror(status: int) -> str:
    """Decode an ``IM_STATUS_*`` integer into a human message."""
    try:
        with _rga_lock:
            ptr = _lib.imStrError_t(status)
    except AttributeError:
        return f"im2d status {status}"
    if not ptr:
        return f"im2d status {status}"
    return ctypes.string_at(ptr).decode("ascii", "replace")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check(ret: int, where: str) -> None:
    """Raise RuntimeError on any non-success im2d return.

    IM_STATUS_SUCCESS (1) and IM_STATUS_NOERROR (2) are both accepted;
    any other value is a failure with a human message from imStrError.
    """
    if ret in (IM_STATUS_SUCCESS, IM_STATUS_NOERROR):
        return
    msg = imstrerror(ret)
    raise RuntimeError(f"im2d {where!r} failed: status={ret} {msg}")


def _make_handle_param(width: int, height: int, format: int) -> im_handle_param:
    """Build a populated im_handle_param_t for an import call.

    Width and height are in pixels; ``format`` is one of the
    :class:`RgaFormat` constants. The library uses these to size the
    cached mapping and pre-validate the import.
    """
    p = im_handle_param()
    p.width = c_uint32(width)
    p.height = c_uint32(height)
    p.format = c_uint32(format)
    return p


def _wrap_handle(
    handle: int,
    width: int,
    height: int,
    format: int,
    *,
    wstride: int = 0,
    hstride: int = 0,
) -> rga_buffer_t:
    """``wrapbuffer_handle_t`` with sensible defaults.

    Caller must hold ``_rga_lock`` (no re-acquire). All public
    functions that call this already hold the lock; if you're adding
    a new call site, wrap it.
    """
    if wstride == 0:
        wstride = width
    if hstride == 0:
        hstride = height
    return _lib.wrapbuffer_handle_t(
        c_uint32(handle), width, height, wstride, hstride, format,
    )


# ---------------------------------------------------------------------------
# Buffer object — the im2d-recommended lifecycle
# ---------------------------------------------------------------------------


class Buffer:
    """A handle-backed im2d buffer.

    The :class:`Buffer` owns an im2d ``rga_buffer_handle_t`` allocated
    by ``importbuffer_*`` and releases it on destruction. Operations
    like :func:`cvtcolor` / :func:`resize` take the underlying
    ``rga_buffer_t`` by value (cheap to copy), so a single :class:`Buffer`
    can be reused for many calls.

    Use :attr:`buf` when calling im2d operations. ``buf`` is a fresh
    ``rga_buffer_t`` on every access — fill it via :meth:`descriptor`
    if you need explicit stride / rect overrides.
    """

    __slots__ = ("_handle", "_format", "_w", "_h", "_buf", "_refcount_held")

    def __init__(
        self,
        handle: int,
        width: int,
        height: int,
        format: int,
        *,
        wstride: int = 0,
        hstride: int = 0,
    ) -> None:
        self._handle = handle
        self._format = format
        self._w = width
        self._h = height
        self._buf = _wrap_handle(handle, width, height, format,
                                wstride=wstride, hstride=hstride)
        # Cached width/height are used by ``descriptor`` to re-emit
        # ``rga_buffer_t`` structs with custom rect / stride.
        self._refcount_held = True

    @property
    def handle(self) -> int:
        """The raw im2d handle. ``0`` if already released."""
        return self._handle

    @property
    def format(self) -> int:
        return self._format

    @property
    def width(self) -> int:
        return self._w

    @property
    def height(self) -> int:
        return self._h

    @property
    def buf(self) -> rga_buffer_t:
        """The ``rga_buffer_t`` value to pass into im2d operations."""
        return self._buf

    def descriptor(
        self,
        *,
        x: int = 0,
        y: int = 0,
        w: Optional[int] = None,
        h: Optional[int] = None,
        wstride: Optional[int] = None,
        hstride: Optional[int] = None,
    ) -> rga_buffer_t:
        """Build a ``rga_buffer_t`` with custom rect / stride for one call.

        Use this when an operation needs to act on a sub-region or a
        different stride than what the handle was imported with.
        """
        b = rga_buffer_t()
        b.fd = self._buf.fd
        b.vir_addr = self._buf.vir_addr
        b.phy_addr = self._buf.phy_addr
        b.width = w if w is not None else self._w
        b.height = h if h is not None else self._h
        b.wstride = wstride if wstride is not None else b.width
        b.hstride = hstride if hstride is not None else b.height
        b.format = self._format
        b.color_space_mode = IM_COLOR_SPACE_DEFAULT
        return b

    def release(self) -> None:
        """Release the underlying im2d handle. Idempotent."""
        if self._handle == 0:
            return
        h = self._handle
        self._handle = 0
        with _rga_lock:
            _lib.releasebuffer_handle(c_uint32(h))

    def __del__(self) -> None:
        try:
            self.release()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Public buffer-import API
# ---------------------------------------------------------------------------


def _do_import(op, width: int, height: int, format: int,
               wstride: int = 0, hstride: int = 0) -> Buffer:
    """Call one of the importbuffer_* entry points, returning a Buffer.

    ``op`` is the bound C function (``importbuffer_virtualaddr``,
    ``importbuffer_fd``, ...). Returns ``None``-ish on failure by
    raising a descriptive RuntimeError.

    Note: callers must hold ``_rga_lock``; this helper doesn't take
    it again (which would deadlock with the non-reentrant Lock).
    """
    param = _make_handle_param(width, height, format)
    handle = op(ctypes.byref(param))
    if handle == 0:
        raise RuntimeError(
            f"im2d importbuffer failed: handle=0 ({op.__name__}, "
            f"w={width} h={height} fmt=0x{format:x})"
        )
    return Buffer(handle, width, height, format, wstride=wstride, hstride=hstride)


def import_va(vir_addr: int, width: int, height: int, format: int) -> Buffer:
    """Import a virtual-address buffer into im2d.

    The buffer is registered in the im2d handle table; subsequent
    :func:`cvtcolor` / :func:`resize` calls reuse the cached mapping.
    The handle is released automatically when the returned
    :class:`Buffer` is garbage-collected.
    """
    with _rga_lock:
        return _do_import(
            lambda p: _lib.importbuffer_virtualaddr(c_void_p(vir_addr), p),
            width, height, format,
        )


def import_fd(fd: int, width: int, height: int, format: int) -> Buffer:
    """Import a DMA-BUF fd into im2d.

    Zero-copy path for VPU → RGA: the kernel hands both engines the
    same DMA, so the YUV plane stays in a single CMA allocation
    throughout decode → colour-convert → display. Avoids the CPU
    cache-coherency dance that :func:`import_va` would otherwise
    need.
    """
    with _rga_lock:
        return _do_import(
            lambda p: _lib.importbuffer_fd(c_int(fd), p),
            width, height, format,
        )


def import_pa(pa: int, width: int, height: int, format: int) -> Buffer:
    """Import a physical-address buffer into im2d."""
    with _rga_lock:
        return _do_import(
            lambda p: _lib.importbuffer_physicaladdr(c_uint64(pa), p),
            width, height, format,
        )


# Convenience wrappers for the two formats we use most:

def import_nv12_va(ptr: int, width: int, height: int,
                   hor_stride: int = 0, ver_stride: int = 0) -> Buffer:
    """Import a virtual-address NV12 buffer (MPP's virtual fallback)."""
    return import_va(ptr, width, height, RgaFormat.YCbCr_420_SP,
                     wstride=hor_stride or width,
                     hstride=ver_stride or height)


def import_nv12_fd(fd: int, width: int, height: int,
                   hor_stride: int = 0, ver_stride: int = 0) -> Buffer:
    """Import a DMA-BUF fd holding an NV12 image (MPP's default output)."""
    return import_fd(fd, width, height, RgaFormat.YCbCr_420_SP,
                     wstride=hor_stride or width,
                     hstride=ver_stride or height)


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------


def imcheck(
    src: rga_buffer_t,
    dst: rga_buffer_t,
    *,
    usage: int = 0,
    pat: Optional[rga_buffer_t] = None,
    src_rect: im_rect = im_rect(),
    dst_rect: im_rect = im_rect(),
) -> None:
    """Pre-flight check: raise if im2d cannot do this operation.

    Use this before :func:`cvtcolor` / :func:`resize` to surface a
    meaningful error before doing the (expensive) real work. The
    ``usage`` flag matches ``IM_*`` flags if you want to be explicit
    (default 0 means "auto-pick").
    """
    if pat is None:
        pat = rga_buffer_t()
    pat_ref = ctypes.byref(pat) if pat else ctypes.c_void_p(0)
    # imcheck_t wants values, but im2d's signature is by-value
    # through the ctypes call. Pass the structs directly.
    with _rga_lock:
        ret = _lib.imcheck_t(
            src, dst, pat,
            src_rect, dst_rect, im_rect(),
            usage,
        )
    _check(ret, "imcheck")


def cvtcolor(src: rga_buffer_t, dst: rga_buffer_t, sync: bool = True) -> None:
    """Convert ``src`` to ``dst``'s pixel format. NV12 → BGR888 is the
    common case for video decode."""
    with _rga_lock:
        ret = _lib.imcvtcolor_t(
            src, dst,
            int(src.format), int(dst.format),
            IM_COLOR_SPACE_DEFAULT,
            IM_SYNC if sync else 0,
        )
    _check(ret, "imcvtcolor")


def resize(
    src: rga_buffer_t,
    dst: rga_buffer_t,
    interpolation: int = INTER_LINEAR,
    sync: bool = True,
) -> None:
    """Scale ``src`` to ``dst``'s dimensions. The destination size drives
    the output."""
    with _rga_lock:
        ret = _lib.imresize_t(
            src, dst, 0.0, 0.0, interpolation,
            IM_SYNC if sync else 0,
        )
    _check(ret, "imresize")


def copy(src: rga_buffer_t, dst: rga_buffer_t, sync: bool = True) -> None:
    """Raw RGA copy — same-format hardware DMA."""
    with _rga_lock:
        ret = _lib.imcopy_t(src, dst, IM_SYNC if sync else 0)
    _check(ret, "imcopy")


def flush() -> None:
    """Block until all queued RGA operations complete."""
    with _rga_lock:
        _lib.c_RkRgaFlush()


# ---------------------------------------------------------------------------
# Legacy direct-wrap API (no handle)
# ---------------------------------------------------------------------------
#
# These mirror the older ``wrapbuffer_*_t`` entry points. They work
# without an import step but bypass the im2d context's stride cache
# and don't get the auto RGA2/RGA3 dispatch benefits. Prefer
# :func:`import_va` / :func:`import_fd` for new code.

def wrap_virtual(
    vir_addr: int,
    width: int,
    height: int,
    format: int,
    hstride: int = 0,
    wstride: int = 0,
) -> rga_buffer_t:
    if hstride == 0:
        hstride = height
    if wstride == 0:
        wstride = width
    with _rga_lock:
        return _lib.wrapbuffer_virtualaddr_t(
            c_void_p(vir_addr), width, height, wstride, hstride, format,
        )


def wrap_fd(
    fd: int,
    width: int,
    height: int,
    format: int,
    hstride: int = 0,
    wstride: int = 0,
) -> rga_buffer_t:
    if hstride == 0:
        hstride = height
    if wstride == 0:
        wstride = width
    with _rga_lock:
        return _lib.wrapbuffer_fd_t(
            fd, width, height, wstride, hstride, format,
        )


def wrap_nv12(ptr: int, width: int, height: int,
              hor_stride: int = 0, ver_stride: int = 0) -> rga_buffer_t:
    """Wrap a virtual-address NV12 image (no handle, no import).

    Equivalent to ``wrap_virtual(ptr, w, h, RK_FORMAT_YCbCr_420_SP,
    wstride=hor_stride or w, hstride=ver_stride or h)``.
    """
    return wrap_virtual(
        ptr, width, height, RgaFormat.YCbCr_420_SP,
        hstride=ver_stride or height,
        wstride=hor_stride or width,
    )


def wrap_nv12_fd(fd: int, width: int, height: int,
                 hor_stride: int = 0, ver_stride: int = 0) -> rga_buffer_t:
    return wrap_fd(
        fd, width, height, RgaFormat.YCbCr_420_SP,
        hstride=ver_stride or height,
        wstride=hor_stride or width,
    )


# ---------------------------------------------------------------------------
# Backend self-test (called once at import time)
# ---------------------------------------------------------------------------


def _backend_self_test() -> str:
    """Return a one-line summary of the active RGA backend."""
    try:
        v = version()
        vendor_str = vendor()
    except Exception as e:
        return f"<im2d unavailable: {e}>"
    return f"im2d {v} ({vendor_str})"


_BACKEND = _backend_self_test()


def backend_info() -> str:
    """Return a human-readable summary of the active RGA backend."""
    return _BACKEND


__all__ = [
    "RgaFormat",
    "rga_buffer_t",
    "im_rect",
    "im_handle_param",
    "im_nn_t",
    "Buffer",
    "bytes_per_pixel",
    "format_name",
    "vendor",
    "version",
    "max_input_resolution",
    "max_output_resolution",
    "input_formats",
    "output_formats",
    "imstrerror",
    "imcheck",
    "import_va",
    "import_fd",
    "import_pa",
    "import_nv12_va",
    "import_nv12_fd",
    "cvtcolor",
    "resize",
    "copy",
    "flush",
    "wrap_virtual",
    "wrap_fd",
    "wrap_nv12",
    "wrap_nv12_fd",
    "backend_info",
    "IM_STATUS_SUCCESS",
    "IM_STATUS_NOERROR",
    "IM_COLOR_SPACE_DEFAULT",
    "IM_SYNC",
    "INTER_LINEAR",
]