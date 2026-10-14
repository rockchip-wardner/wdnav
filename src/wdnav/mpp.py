"""Python wrapper for the Rockchip MPP hardware codec via the
``libshim_video.so`` C shim.

This module is now a thin glue layer between Python callers
(``wdnav.video.VideoCapture``, ``wdnav.encoder.VideoWriter``) and the
high-level ``rkv_enc_*`` / ``rkv_dec_*`` C wrappers in
``src/wdnav/cshim_video_encoder.c`` / ``src/wdnav/cshim_video_decoder.c``.
Those wrappers hide all the MppPacket / MppFrame / MppBuffer handle
plumbing — what used to be hundreds of lines of ctypes calls in this
module is now a handful of function calls into the shim.

Public surface kept for backwards compatibility with existing callers:

* Coding / pixel-format / rate-control enum constants
  (``MppCodingType``, ``MppFrameFormat``, ``MppEncRcMode``, …) and the
  ``CODING_BY_NAME`` mapping PyAV streams and FFmpeg codec names to
  MPP coding type codes.
* ``DecodedFrame`` — the result of :meth:`MppDecoder.recv_frame`
  (with its ``.numpy()`` helper that converts via RGA2).
* ``EncodedPacket`` — the result of :meth:`MppEncoder.encode` /
  :meth:`encode_external_fd`.
* ``MppDecoder`` and ``MppEncoder`` classes — high-level Python
  wrappers around the C shim's opaque encoder / decoder handles.
* Annex B / avcC / hvcC helpers (``avcc_to_annexb``,
  ``parse_avcc_extradata``, ``parse_hvcc_extradata``) — pure-Python,
  no shim needed.

The shim itself is loaded from ``<this-package>/lib/libshim_video.so``
(the canonical path) and falls back to ``ctypes.CDLL("libshim_video")``
if the .so was dropped somewhere custom.
"""

from __future__ import annotations

import ctypes
import os
import threading
from ctypes import (
    POINTER,
    Structure,
    byref,
    c_char_p,
    c_int,
    c_int16,
    c_int32,
    c_int64,
    c_size_t,
    c_uint8,
    c_uint16,
    c_uint32,
    c_uint64,
    c_void_p,
)
from dataclasses import dataclass
from typing import Optional, Tuple

# ---------------------------------------------------------------------------
# Library loading
# ---------------------------------------------------------------------------

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_LIBS_TO_TRY = (
    os.path.join(_THIS_DIR, "lib", "libshim_video.so"),
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
        return ctypes.CDLL("libshim_video", mode=ctypes.RTLD_GLOBAL)
    except OSError as e:
        raise OSError(
            "Could not load libshim_video.so from the package lib/ "
            "directory or via ldconfig. The package ships "
            "libshim_video.so (compiled from "
            "src/wdnav/cshim_video_encoder.c / cshim_video_decoder.c "
            "at wheel-build time); if it's missing, the wheel was "
            "built without the C shim step — see _build_rkv.py at "
            "the project root. Install librockchip-mpp system-wide "
            "(e.g. apt install librockchip-mpp1) so the shim can "
            "dlopen librockchip_mpp.so at runtime. Last error: "
            + repr(last_err or e)
        )


_lib = _load_library()

# A single lock guards all shim calls. The shim is not documented as
# thread-safe across separate contexts, but more importantly the
# upstream MPP context itself isn't either — and we currently only own
# one decoder / encoder at a time per process.
_mpp_lock = threading.Lock()


def _check(ret: int, where: str, err_buf: Optional[ctypes.Array] = None) -> None:
    """Raise if a shim call returns a negative status. ``err_buf``, if
    provided, is a 512-byte ctypes string buffer the shim fills with a
    human-readable error message.
    """
    if ret < 0:
        msg = ""
        if err_buf is not None and err_buf.value:
            try:
                msg = err_buf.value.decode("utf-8", "replace")
            except Exception:
                msg = ""
        raise RuntimeError(
            f"{where} failed: ret={ret}" + (f" — {msg}" if msg else "")
        )


# ---------------------------------------------------------------------------
# Constant mirrors — match the values the C shim uses internally
# ---------------------------------------------------------------------------

RK_U8 = c_uint8
RK_U16 = c_uint16
RK_U32 = c_uint32
RK_S16 = c_int16
RK_S32 = c_int32
RK_S64 = c_int64


class MppCtxType:
    DEC = 0
    ENC = 1
    ISP = 2
    BUTT = 3


class MppCodingType:
    UNUSED = 0
    AUTO_DETECT = 1
    MPEG2 = 2
    H263 = 3
    MPEG4 = 4
    WMV = 5
    RV = 6
    AVC = 7          # H.264
    MJPEG = 8
    VP8 = 9
    VP9 = 10
    # HEVC is an explicit assignment upstream; see cshim_video_encoder.c.
    HEVC = 0x01000004
    AVSPLUS = 0x01000005
    AVS = 0x01000006
    AVS2 = 0x01000007
    AV1 = 0x01000008


class MppFrameFormat:
    YUV420SP = 0          # NV12
    YUV420SP_10BIT = 1
    YUV422SP = 2
    YUV420P = 4
    YUV420SP_VU = 5       # NV21
    YUV422P = 6
    YUV400 = 12
    RGB888 = (1 << 16) + 6
    BGR888 = (1 << 16) + 7
    BGRA = (1 << 16) + 12
    RGBA = (1 << 16) + 13


class MppBufferType:
    NORMAL = 0
    ION = 1
    EXT_DMA = 2
    DRM = 3
    DMA_HEAP = 4
    FLAG_CONTIG = 0x00010000
    FLAG_CACHABLE = 0x00020000
    FLAG_WC = 0x00040000
    FLAG_SECURE = 0x00080000
    FLAG_KMAP = 0x00100000
    FLAG_DMA32 = 0x00200000


class MppBufferMode:
    INTERNAL = 0
    EXTERNAL = 1


# Rate-control modes — positional enum (RcMode_e in mpp_rc_api.h:
# VBR=0, CBR=1, FIXQP=2, AVBR=3). Matches what cshim_video_encoder.c.
class MppEncRcMode:
    VBR = 0
    CBR = 1
    FIXQP = 2
    AVBR = 3
    SMTRC = 4


# H.264 profile_idc codes — match rkv_enc_create's profile arg.
class MppEncProfile:
    BASELINE = 66
    MAIN = 77
    HIGH = 100


# H.264 profile name → profile_idc code. Mirrors _H264_PROFILE_BY_NAME
# in encoder.py / mpp.py.
_H264_PROFILE_BY_NAME = {
    "baseline": MppEncProfile.BASELINE,
    "main": MppEncProfile.MAIN,
    "high": MppEncProfile.HIGH,
}


# Map a handful of FFmpeg codec names (PyAV / avcodec) to MppCodingType
# values. Extend as needed.
CODING_BY_NAME = {
    "h264": MppCodingType.AVC,
    "hevc": MppCodingType.HEVC,
    "h265": MppCodingType.HEVC,
    "vp9": MppCodingType.VP9,
    "vp8": MppCodingType.VP8,
    "av1": MppCodingType.AV1,
    "mpeg2video": MppCodingType.MPEG2,
    "mpeg4": MppCodingType.MPEG4,
    "mjpeg": MppCodingType.MJPEG,
}


# Map a handful of profile names (Python-side ergonomics) to the
# profile_idc byte the shim expects.
def _resolve_h264_profile(name: str) -> int:
    code = _H264_PROFILE_BY_NAME.get(name.lower())
    if code is None:
        raise ValueError(
            f"unsupported H.264 profile {name!r}; expected one of "
            f"{sorted(_H264_PROFILE_BY_NAME)}"
        )
    return code


# ---------------------------------------------------------------------------
# C struct mirrors — must match the layouts in cshim_video_*.c exactly.
# ---------------------------------------------------------------------------


class _RkvEncPacket(Structure):
    """Mirror of ``rkv_enc_packet_t`` in cshim_video_encoder.c."""
    _fields_ = [
        ("data", POINTER(c_uint8)),  # void* widened to RK_U8*
        ("length", c_uint32),
        ("flag", c_uint32),
        ("pts", c_int64),
    ]


class _RkvDecFrame(Structure):
    """Mirror of ``rkv_dec_frame_t`` in cshim_video_decoder.c."""
    _fields_ = [
        ("width", c_int32),
        ("height", c_int32),
        ("hor_stride", c_int32),
        ("ver_stride", c_int32),
        ("fmt", c_int32),
        ("pts", c_int64),
        ("ptr", c_int64),         # void* widened to RK_S64
        ("fd", c_int32),
        ("buf_size", c_uint32),
        ("is_info_change", c_uint32),
        ("discard", c_uint32),
        ("errinfo", c_uint32),
        ("eos", c_uint32),
    ]


# ---------------------------------------------------------------------------
# Function bindings — point at the rkv_enc_* / rkv_dec_* exports.
# ---------------------------------------------------------------------------

# rkv_enc_create(int coding_type, int width, int height,
#                int hor_stride, int ver_stride,
#                int fps, int bitrate, int gop,
#                int profile, int rc_mode, int input_format, int q_factor,
#                char *err_buf, int err_buf_len,
#                rkv_encoder_t **out_enc);
_lib.rkv_enc_create.argtypes = [
    c_int, c_int, c_int, c_int, c_int, c_int, c_int,
    c_int, c_int, c_int, c_int, c_int,
    c_char_p, c_int, POINTER(c_void_p),
]
_lib.rkv_enc_create.restype = c_int

# rkv_enc_get_extradata(rkv_encoder_t *enc,
#                       const RK_U8 **out_data, RK_U32 *out_size);
_lib.rkv_enc_get_extradata.argtypes = [
    c_void_p, POINTER(POINTER(c_uint8)), POINTER(c_uint32),
]
_lib.rkv_enc_get_extradata.restype = c_int

# rkv_enc_put_frame(rkv_encoder_t *enc,
#                   const void *pixels, size_t pixels_size,
#                   int width, int height, int hor_stride, int ver_stride,
#                   int input_format, int64_t pts,
#                   char *err_buf, int err_buf_len);
_lib.rkv_enc_put_frame.argtypes = [
    c_void_p, c_void_p, c_size_t,
    c_int, c_int, c_int, c_int,
    c_int, c_int64,
    c_char_p, c_int,
]
_lib.rkv_enc_put_frame.restype = c_int

# rkv_enc_put_frame_fd(...);
_lib.rkv_enc_put_frame_fd.argtypes = [
    c_void_p, c_int, c_size_t,
    c_int, c_int, c_int, c_int,
    c_int, c_int64,
    c_char_p, c_int,
]
_lib.rkv_enc_put_frame_fd.restype = c_int

# rkv_enc_get_packet(rkv_encoder_t *enc, rkv_enc_packet_t *out,
#                    char *err_buf, int err_buf_len);
_lib.rkv_enc_get_packet.argtypes = [c_void_p, POINTER(_RkvEncPacket),
                                     c_char_p, c_int]
_lib.rkv_enc_get_packet.restype = c_int

# rkv_enc_flush(rkv_encoder_t *enc, rkv_enc_packet_t *out,
#               char *err_buf, int err_buf_len);
_lib.rkv_enc_flush.argtypes = [c_void_p, POINTER(_RkvEncPacket),
                                c_char_p, c_int]
_lib.rkv_enc_flush.restype = c_int

_lib.rkv_enc_force_keyframe.argtypes = [c_void_p]
_lib.rkv_enc_force_keyframe.restype = c_int

_lib.rkv_enc_destroy.argtypes = [c_void_p]
_lib.rkv_enc_destroy.restype = None

# rkv_dec_create(int coding_type, int split_parse, int extra_buf_count,
#                int output_format,
#                char *err_buf, int err_buf_len,
#                rkv_decoder_t **out_dec);
_lib.rkv_dec_create.argtypes = [
    c_int, c_int, c_int, c_int,
    c_char_p, c_int, POINTER(c_void_p),
]
_lib.rkv_dec_create.restype = c_int

# rkv_dec_put_packet(rkv_decoder_t *dec,
#                    const void *data, size_t length,
#                    int64_t pts, int eos,
#                    char *err_buf, int err_buf_len);
_lib.rkv_dec_put_packet.argtypes = [
    c_void_p, c_void_p, c_size_t,
    c_int64, c_int,
    c_char_p, c_int,
]
_lib.rkv_dec_put_packet.restype = c_int

# rkv_dec_get_frame(rkv_decoder_t *dec, int timeout_ms,
#                   rkv_dec_frame_t *out,
#                   char *err_buf, int err_buf_len);
_lib.rkv_dec_get_frame.argtypes = [
    c_void_p, c_int, POINTER(_RkvDecFrame),
    c_char_p, c_int,
]
_lib.rkv_dec_get_frame.restype = c_int

# rkv_dec_release_frame(rkv_decoder_t *dec, char *err_buf, int err_buf_len);
_lib.rkv_dec_release_frame.argtypes = [c_void_p, c_char_p, c_int]
_lib.rkv_dec_release_frame.restype = c_int

# rkv_dec_reset(rkv_decoder_t *dec, char *err_buf, int err_buf_len);
_lib.rkv_dec_reset.argtypes = [c_void_p, c_char_p, c_int]
_lib.rkv_dec_reset.restype = c_int

_lib.rkv_dec_destroy.argtypes = [c_void_p]
_lib.rkv_dec_destroy.restype = None


# ---------------------------------------------------------------------------
# Format / conversion helpers — shared between DecodedFrame.numpy and the
# eager VideoCapture.read path. Putting them here keeps the mapping
# definitions (MPP-format → RGA-format and the user-facing format names)
# next to the MPP decoder that emits ``fmt`` codes.
# ---------------------------------------------------------------------------


_MPP_TO_RGA_FMT = {
    0: "YCbCr_420_SP",
    4: "YCbCr_420_P",
    5: "YCrCb_420_SP",
}
_MPP_TO_RGA_NAME = {0: "NV12", 4: "I420", 5: "NV21"}


_OUTPUT_FORMAT_BY_NAME = {
    # 8-bit packed pixel formats
    "BGR":     ("BGR_888",      3,   "HWC"),
    "BGR888":  ("BGR_888",      3,   "HWC"),
    "RGB":     ("RGB_888",      3,   "HWC"),
    "RGB888":  ("RGB_888",      3,   "HWC"),
    # 8-bit packed pixel + alpha
    "BGRA":    ("BGRA_8888",    4,   "HWC"),
    "BGRA8888": ("BGRA_8888",   4,   "HWC"),
    "RGBA":    ("RGBA_8888",    4,   "HWC"),
    "RGBA8888": ("RGBA_8888",   4,   "HWC"),
    # 8-bit YUV 4:2:0 semi-planar (RGA writes full Y + interleaved UV)
    "NV12":    ("YCbCr_420_SP", 1.5, "YUV"),
    "NV21":    ("YCrCb_420_SP", 1.5, "YUV"),
    # Luma-only via RGA's native YUV400 format (single channel, 1 BPP).
    "GRAY":    ("YCbCr_400",    1,   "GRAY"),
    "GREY":    ("YCbCr_400",    1,   "GRAY"),
    "Y":       ("YCbCr_400",    1,   "GRAY"),
}


def _resolve_output_format(fmt: Optional[str]) -> Tuple[int, float, str]:
    """Return ``(rga_format, bytes_per_pixel, layout)`` for ``fmt``."""
    from . import rga

    if not fmt:
        return rga.RgaFormat.BGR_888, 3, "HWC"
    try:
        rga_attr, bpp, layout = _OUTPUT_FORMAT_BY_NAME[fmt.upper()]
    except KeyError as e:
        raise ValueError(
            f"unknown output format {fmt!r}; supported: "
            f"{sorted(_OUTPUT_FORMAT_BY_NAME)}"
        ) from e
    return getattr(rga.RgaFormat, rga_attr), bpp, layout


def _rga_input_format(fmt: int) -> int:
    """Resolve an MPP ``fmt`` code to its RGA input format constant."""
    from . import rga

    try:
        attr = _MPP_TO_RGA_FMT[fmt]
    except KeyError as e:
        raise RuntimeError(
            f"MPP frame format {fmt} is not supported by RGA input; "
            f"supported: {sorted(_MPP_TO_RGA_NAME.values())}"
        ) from e
    return getattr(rga.RgaFormat, attr)


def _as_ndarray(buf_storage, width: int, height: int, bpp: float, layout: str):
    """Reshape an RGA output buffer into the right ndarray view."""
    import numpy as np

    raw = np.frombuffer(buf_storage, dtype=np.uint8)
    if layout == "YUV":
        return raw.reshape(height * 3 // 2, width)
    if layout == "GRAY":
        return raw[: height * width].reshape(height, width)
    channels = int(round(bpp))
    return raw.reshape(height, width, channels)


# ---------------------------------------------------------------------------
# Data classes — stable public surface for video.py / encoder.py.
# ---------------------------------------------------------------------------


@dataclass
class EncodedPacket:
    """One encoded access unit returned by :meth:`MppEncoder.encode`."""

    data: bytes
    pts: int
    is_keyframe: bool


@dataclass
class DecodedFrame:
    """A decoded frame returned by :meth:`MppDecoder.recv_frame`."""

    width: int
    height: int
    hor_stride: int
    ver_stride: int
    fmt: int
    pts: int
    ptr: int           # pointer into the MppBuffer's storage; 0 if DMA-backed only
    fd: int            # DMA-BUF fd; -1 if not available
    buf_size: int
    raw: "MppFrame._RawRef" = None  # opaque handle held by the decoder
    _dmabuf: "DmaBuf | None" = None  # owned DMA-BUF (set by crop())

    out_w: Optional[int] = None
    out_h: Optional[int] = None
    out_fmt: Optional[str] = None

    def numpy(self, w=None, h=None, fmt=None):
        """Materialise this frame into a contiguous ndarray via RGA2."""
        # If read0() cached the BGR before releasing the DMA-BUF, return
        # that cached ndarray instead of re-reading the (possibly reused
        # by now) DMA-BUF. See video.py:read0() for the rationale.
        cached = getattr(self, "_cached_bgr", None)
        if cached is not None:
            return cached

        import ctypes

        from . import rga

        import numpy as np

        target_w = w if w else self.out_w
        target_h = h if h else self.out_h
        target_fmt_name = fmt if fmt else self.out_fmt

        input_fmt = _rga_input_format(self.fmt)
        target_fmt, bpp, layout = _resolve_output_format(target_fmt_name)
        target_w = target_w or self.width
        target_h = target_h or self.height

        hor_stride = self.hor_stride or self.width
        ver_stride = self.ver_stride or self.height

        if self.fd is not None and self.fd >= 0:
            src = rga.wrap_fd(
                self.fd, self.width, self.height, input_fmt,
                wstride=hor_stride, hstride=ver_stride,
            )
        else:
            src = rga.wrap_virtual(
                self.ptr, self.width, self.height, input_fmt,
                wstride=hor_stride, hstride=ver_stride,
            )

        rga_w = (target_w + 15) & ~15
        bpp_target = rga.bytes_per_pixel(target_fmt)
        size = -(-int(rga_w * target_h * bpp_target) // 1)
        buf_storage = (ctypes.c_uint8 * size)()
        dst = rga.wrap_virtual(
            ctypes.addressof(buf_storage), rga_w, target_h, target_fmt,
        )
        try:
            if (target_w, target_h) == (self.width, self.height) or layout != "HWC":
                rga.cvtcolor(src, dst)
            else:
                rga.resize(src, dst)
            rga.flush()
        except RuntimeError as e:
            raise RuntimeError(
                f"RGA cannot convert {_MPP_TO_RGA_NAME.get(self.fmt, self.fmt)} "
                f"{self.width}x{self.height} to "
                f"{rga.format_name(target_fmt)} {target_w}x{target_h}: {e}"
            ) from e

        out = _as_ndarray(buf_storage, rga_w, target_h, bpp, layout)
        if rga_w != target_w:
            out = out[:, :target_w]
        return out

    def crop(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
    ) -> "DecodedFrame":
        """Crop this frame via RGA in normalised [0, 1] coordinates.

        Returns a new :class:`DecodedFrame` holding a freshly allocated
        DMA-BUF (``fd >= 0``) that owns the cropped pixel data, so it can
        be handed straight to :meth:`wdnav.encoder.VideoWriter.write` for
        zero-copy ``encode_external_fd``, or chained with
        :meth:`numpy` (``fmt="BGR"``) for an ndarray. The source pixel
        format is preserved — for colour conversion chain ``.crop(...)``
        with :meth:`numpy`.

        Parameters
        ----------
        x, y, w, h:
            Normalised top-left and size, all in ``[0, 1]``. ``w`` and
            ``h`` must be strictly positive. The pixel-space region is
            computed as ``round(coord * dimension)`` and then clamped so
            it never exceeds the source frame.
        """
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0
                and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
            raise ValueError(
                f"x, y, w, h must be in [0, 1], got ({x}, {y}, {w}, {h})"
            )
        if w == 0.0 or h == 0.0:
            raise ValueError("w and h must be > 0")

        # Deferred import so mpp.py stays import-safe on dev boxes that
        # lack /dev/dma_heap (only clip() actually needs the allocator).
        from .dmaheap import DmaHeap

        src_w, src_h = self.width, self.height
        crop_x = int(round(x * src_w))
        crop_y = int(round(y * src_h))
        crop_w = min(int(round(w * src_w)), src_w - crop_x)
        crop_h = min(int(round(h * src_h)), src_h - crop_y)
        if crop_w <= 0 or crop_h <= 0:
            raise ValueError("clip region is empty after rounding")

        from . import rga

        input_fmt = _rga_input_format(self.fmt)
        hor_stride = self.hor_stride or self.width
        ver_stride = self.ver_stride or self.height
        if self.fd is not None and self.fd >= 0:
            src = rga.wrap_fd(
                self.fd, src_w, src_h, input_fmt,
                wstride=hor_stride, hstride=ver_stride,
            )
        else:
            src = rga.wrap_virtual(
                self.ptr, src_w, src_h, input_fmt,
                wstride=hor_stride, hstride=ver_stride,
            )

        rga_w = (crop_w + 15) & ~15
        # MPP output formats (NV12/NV21/I420) all share the 1.5 bpp
        # semi-planar / planar layout; ceil-div to bytes is crop_w*crop_h*3/2.
        alloc_size = rga_w * crop_h * 3 // 2

        heap = DmaHeap()
        try:
            dmabuf = heap.alloc(alloc_size)
        finally:
            heap.close()

        dst = rga.wrap_fd(
            dmabuf.fd, rga_w, crop_h, input_fmt,
            wstride=rga_w, hstride=crop_h,
        )
        rect = rga.im_rect(crop_x, crop_y, crop_w, crop_h)
        try:
            rga.imcrop(src, dst, rect)
            rga.flush()
        except RuntimeError as e:
            dmabuf.close()
            raise RuntimeError(
                f"RGA crop failed: src {_MPP_TO_RGA_NAME.get(self.fmt, self.fmt)} "
                f"{src_w}x{src_h} -> {crop_w}x{crop_h} @ ({crop_x}, {crop_y}): {e}"
            ) from e

        new = DecodedFrame(
            width=crop_w,
            height=crop_h,
            hor_stride=rga_w,
            ver_stride=crop_h,
            fmt=self.fmt,
            pts=self.pts,
            ptr=0,
            fd=dmabuf.fd,
            buf_size=alloc_size,
            raw=None,
            out_w=None,
            out_h=None,
            out_fmt=None,
        )
        # Hold dmabuf alive on the new frame so the fd stays valid for
        # the lifetime of the returned DecodedFrame. dataclass doesn't
        # declare this attribute, so use object.__setattr__ to bypass
        # the field machinery.
        object.__setattr__(new, "_dmabuf", dmabuf)
        return new


# Packet flag bit 4 marks intra / key frames per cshim_video_encoder.c.
_PACKET_FLAG_INTRA = 1 << 4


# ---------------------------------------------------------------------------
# MppEncoder / MppDecoder — thin Python wrappers around the shim's
# opaque encoder / decoder handles.
# ---------------------------------------------------------------------------


class MppEncoder:
    """High-level Python wrapper around ``rkv_enc_*`` in libshim_video.so.

    Mirrors the historical :class:`MppEncoder` API: ``encode(bgr_ndarray)``
    returns an :class:`EncodedPacket`; ``encode_external_fd(fd, ...)``
    does the same zero-copy from a caller-owned DMA-BUF. ``extradata``
    is the SPS / PPS / VPS blob (Annex B) the shim pulled out of MPP
    at init time.
    """

    # input_format → (bpp, channels, layout). Layout drives the shape
    # validation in :meth:`encode`.
    _FORMAT_LAYOUT = {
        MppFrameFormat.YUV420SP: (1.5, 1, "YUV"),
        MppFrameFormat.YUV420SP_10BIT: (3.0, 1, "YUV"),
        MppFrameFormat.YUV422SP: (2.0, 1, "YUV"),
        MppFrameFormat.YUV420P: (1.5, 1, "YUV"),
        MppFrameFormat.YUV420SP_VU: (1.5, 1, "YUV"),
        MppFrameFormat.YUV400: (1.0, 1, "YUV"),
        MppFrameFormat.RGB888: (3.0, 3, "HWC"),
        MppFrameFormat.BGR888: (3.0, 3, "HWC"),
        MppFrameFormat.RGBA: (4.0, 4, "HWC"),
        MppFrameFormat.BGRA: (4.0, 4, "HWC"),
    }
    # Packed pixel formats (BGR/RGB-family, 3 or 4 channels) that
    # :meth:`encode` converts to NV12 before handing the buffer to
    # the shim. The shim/encoder sees NV12 either way, so its input
    # buffer (``w*h*3`` for BGR/RGB-mode init) is just oversized —
    # 1.5 bytes/pixel NV12 still fits inside.
    _PACKED_PIXEL_FORMATS = {
        MppFrameFormat.RGB888,
        MppFrameFormat.BGR888,
        MppFrameFormat.RGBA,
        MppFrameFormat.BGRA,
    }
    _YUV_HEIGHT_MULTIPLIER = {
        MppFrameFormat.YUV420SP: 1.5,
        MppFrameFormat.YUV420SP_10BIT: 1.5,
        MppFrameFormat.YUV422SP: 2.0,
        MppFrameFormat.YUV420P: 1.5,
        MppFrameFormat.YUV420SP_VU: 1.5,
        MppFrameFormat.YUV400: 1.0,
    }

    def __init__(
        self,
        coding_type: int = MppCodingType.AVC,
        *,
        width: int,
        height: int,
        hor_stride: int = 0,
        ver_stride: int = 0,
        fps: int = 30,
        bitrate: int = 2_000_000,
        gop: int = 30,
        profile: str = "baseline",
        input_format: int = MppFrameFormat.BGR888,
        rc_mode: str = "cbr",
        q_factor: int = 80,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")
        if fps <= 0:
            raise ValueError("fps must be positive")
        if coding_type == MppCodingType.AVC:
            profile_idc = _resolve_h264_profile(profile)
        else:
            profile_idc = 0
        rc_mode_id = {
            "vbr": MppEncRcMode.VBR,
            "cbr": MppEncRcMode.CBR,
            "fixqp": MppEncRcMode.FIXQP,
            "avbr": MppEncRcMode.AVBR,
        }.get(rc_mode.lower())
        if rc_mode_id is None:
            raise ValueError(f"unsupported rc_mode: {rc_mode!r}")
        if input_format not in self._FORMAT_LAYOUT:
            raise ValueError(
                f"unsupported input_format {input_format!r}; expected one of "
                f"{sorted(self._FORMAT_LAYOUT)}"
            )
        if coding_type == MppCodingType.MJPEG:
            if not 1 <= q_factor <= 99:
                raise ValueError(
                    f"q_factor must be in 1..99 for MJPEG, got {q_factor!r}"
                )
            rc_mode_id = MppEncRcMode.FIXQP
        elif q_factor != 80:
            raise ValueError(
                "q_factor only applies to MJPEG encoding; ignored otherwise"
            )

        self._coding = coding_type
        self._width = int(width)
        self._height = int(height)
        # Decoder output buffers frequently pad each row past ``width``
        # (e.g. HEVC on RK3588 reports 720x1280 frames with
        # hor_stride=768). The encoder's ``prep:hor_stride`` is locked
        # at init, so we forward the actual strides here. ``0`` means
        # "no padding", which falls back to ``width`` / ``height`` in
        # the shim. The eager-BGR :meth:`encode` path requires
        # ``hor_stride == width`` (numpy arrays are contiguous);
        # callers that need a non-zero padding must use the zero-copy
        # :meth:`encode_external_fd` path exclusively.
        self._hor_stride = int(hor_stride) if hor_stride > 0 else 0
        self._ver_stride = int(ver_stride) if ver_stride > 0 else 0
        self._fps = int(fps)
        self._bitrate = int(bitrate)
        self._gop = int(gop)
        self._profile_name = profile
        self._profile_idc = profile_idc
        self._rc_mode_name = rc_mode
        self._input_format = input_format
        self._q_factor = int(q_factor)

        # The shim / MPP HAL is configured at init time via
        # ``prep:format``. If we leave it at BGR888 but feed NV12
        # frames from :meth:`encode`, the HAL applies its own
        # BGR→YUV conversion on top — and on this SoC that path
        # treats packed BGR bytes as if they were planar Y/UV,
        # producing the classic "left/centre/right stripes"
        # garbage. So whenever the caller hands in a packed-pixel
        # format (BGR / RGB / RGBA / BGRA), we route through the
        # RGA conversion in :meth:`encode` and tell the shim the
        # format is NV12 end-to-end. For everything else (NV12,
        # NV21, I420, …) we pass the caller's format straight
        # through.
        if input_format in self._PACKED_PIXEL_FORMATS:
            self._shim_input_format = int(MppFrameFormat.YUV420SP)
        else:
            self._shim_input_format = int(input_format)

        self._handle: Optional[c_void_p] = None
        self._extradata: bytes = b""
        self._closed = False
        self._flushed = False

        err_buf = ctypes.create_string_buffer(512)
        handle = c_void_p()
        with _mpp_lock:
            ret = _lib.rkv_enc_create(
                int(coding_type), int(width), int(height),
                self._hor_stride, self._ver_stride,
                int(fps),
                int(bitrate), int(gop),
                int(profile_idc), int(rc_mode_id),
                self._shim_input_format, int(q_factor),
                err_buf, 512,
                byref(handle),
            )
        if ret != 0:
            raise RuntimeError(
                f"MppEncoder: rkv_enc_create failed: ret={ret} — "
                f"{err_buf.value.decode('utf-8', 'replace')}"
            )
        self._handle = handle

        # Pull SPS/PPS / VPS out of the encoder into a Python bytes.
        ed_ptr = POINTER(c_uint8)()
        ed_size = c_uint32(0)
        with _mpp_lock:
            ret = _lib.rkv_enc_get_extradata(self._handle, byref(ed_ptr), byref(ed_size))
        if ret == 0 and ed_size:
            self._extradata = ctypes.string_at(ed_ptr, ed_size.value)

    @property
    def width(self) -> int: return self._width
    @property
    def height(self) -> int: return self._height
    @property
    def hor_stride(self) -> int:
        """Configured ``prep:hor_stride`` (``0`` ⇒ same as ``width``)."""
        return self._hor_stride or self._width
    @property
    def ver_stride(self) -> int:
        """Configured ``prep:ver_stride`` (``0`` ⇒ same as ``height``)."""
        return self._ver_stride or self._height
    @property
    def fps(self) -> int: return self._fps
    @property
    def bitrate(self) -> int: return self._bitrate
    @property
    def gop(self) -> int: return self._gop
    @property
    def profile(self) -> str: return self._profile_name
    @property
    def rc_mode(self) -> str: return self._rc_mode_name
    @property
    def q_factor(self) -> int: return self._q_factor
    @property
    def coding_type(self) -> int: return self._coding
    @property
    def extradata(self) -> bytes: return self._extradata

    def force_keyframe(self) -> None:
        if self._closed or self._flushed:
            raise RuntimeError("MppEncoder is closed")
        with _mpp_lock:
            ret = _lib.rkv_enc_force_keyframe(self._handle)
        if ret != 0:
            raise RuntimeError(f"rkv_enc_force_keyframe failed: ret={ret}")

    def __enter__(self): return self
    def __exit__(self, *exc): self.close()

    def close(self) -> None:
        if self._closed or self._handle is None:
            self._closed = True
            return
        self._closed = True
        with _mpp_lock:
            _lib.rkv_enc_destroy(self._handle)
        self._handle = None

    def encode(self, frame) -> EncodedPacket:
        """Submit one ndarray and block until the encoded packet returns."""
        import numpy as np

        if self._closed:
            raise RuntimeError("MppEncoder is closed")
        if self._flushed:
            raise RuntimeError("MppEncoder is already flushed")
        if not isinstance(frame, np.ndarray):
            raise TypeError("frame must be a numpy.ndarray")

        _, channels, layout = self._FORMAT_LAYOUT[self._input_format]
        if layout == "YUV":
            expected_shape = (
                int(self._height * self._YUV_HEIGHT_MULTIPLIER[self._input_format]),
                self._width,
            )
            expected_dtype = (
                np.uint16 if self._input_format == MppFrameFormat.YUV420SP_10BIT
                else np.uint8
            )
        else:
            expected_shape = (self._height, self._width, channels)
            expected_dtype = np.uint8

        if (
            frame.ndim != len(expected_shape)
            or frame.shape[0] != expected_shape[0]
            or frame.shape[1] != expected_shape[1]
            or frame.dtype != expected_dtype
        ):
            raise ValueError(
                f"frame must be {expected_shape} {np.dtype(expected_dtype).name}, "
                f"input_format {self._input_format}, got shape={frame.shape} "
                f"dtype={frame.dtype.name}"
            )
        if not frame.flags.c_contiguous:
            frame = np.ascontiguousarray(frame)

        # When the caller passed a packed-pixel format (BGR / RGB /
        # RGBA / BGRA), we go through RGA → NV12 (or the numpy
        # fallback) and feed the shim NV12 directly. The shim was
        # already configured with ``prep:format = NV12`` at init
        # time (see :meth:`__init__`), so the per-frame format arg
        # here is redundant — but passing it keeps the on-frame
        # ``mpp_frame_set_fmt`` in sync with the buffer layout
        # even if a future refactor moves init-time prep config.
        if self._input_format in self._PACKED_PIXEL_FORMATS:
            shim_input = _bgr_to_nv12_np(frame, self._input_format)
            shim_size = shim_input.nbytes
            shim_buf = shim_input.ctypes.data
        else:
            shim_input = frame
            shim_size = frame.nbytes
            shim_buf = frame.ctypes.data
        shim_format = self._shim_input_format

        err_buf = ctypes.create_string_buffer(512)
        with _mpp_lock:
            ret = _lib.rkv_enc_put_frame(
                self._handle,
                ctypes.cast(shim_buf, c_void_p),
                shim_size,
                self._width, self._height,
                self._width, self._height,
                shim_format, 0,
                err_buf, 512,
            )
        if ret != 0:
            raise RuntimeError(
                f"rkv_enc_put_frame failed: ret={ret} — "
                f"{err_buf.value.decode('utf-8', 'replace')}"
            )

        return self._drain_packet()

    def encode_external_fd(
        self,
        fd: int,
        width: int,
        height: int,
        hor_stride: int,
        ver_stride: int,
        input_format: int = MppFrameFormat.YUV420SP,
        pts: int = 0,
    ) -> EncodedPacket:
        """Encode one externally-owned DMA-BUF frame, zero-copy."""
        if self._closed:
            raise RuntimeError("MppEncoder is closed")
        if self._flushed:
            raise RuntimeError("MppEncoder is already flushed")
        if fd is None or fd < 0:
            raise ValueError(f"invalid fd {fd!r}")
        if width <= 0 or height <= 0 or hor_stride <= 0 or ver_stride <= 0:
            raise ValueError(
                f"width/height/hor_stride/ver_stride must all be positive; "
                f"got {width}x{height} {hor_stride}x{ver_stride}"
            )
        err_buf = ctypes.create_string_buffer(512)
        with _mpp_lock:
            ret = _lib.rkv_enc_put_frame_fd(
                self._handle,
                int(fd), hor_stride * ver_stride * 3 // 2,
                int(width), int(height), int(hor_stride), int(ver_stride),
                int(input_format), int(pts),
                err_buf, 512,
            )
        if ret != 0:
            raise RuntimeError(
                f"rkv_enc_put_frame_fd failed: ret={ret} — "
                f"{err_buf.value.decode('utf-8', 'replace')}"
            )
        return self._drain_packet()

    def _drain_packet(self) -> EncodedPacket:
        """Pull the packet the shim produced for the most recent put."""
        err_buf = ctypes.create_string_buffer(512)
        out = _RkvEncPacket()
        with _mpp_lock:
            ret = _lib.rkv_enc_get_packet(self._handle, byref(out),
                                          err_buf, 512)
        if ret == 1:
            # shim returned "no packet yet" — shouldn't happen for
            # synchronous encode, but treat as empty data.
            return EncodedPacket(data=b"", pts=0, is_keyframe=False)
        if ret < 0:
            raise RuntimeError(
                f"rkv_enc_get_packet failed: ret={ret} — "
                f"{err_buf.value.decode('utf-8', 'replace')}"
            )
        data = b""
        if out.length and out.data:
            data = ctypes.string_at(out.data, out.length)
        return EncodedPacket(
            data=data,
            pts=int(out.pts),
            is_keyframe=bool(out.flag & _PACKET_FLAG_INTRA),
        )

    def flush(self) -> Optional[EncodedPacket]:
        """Drive an EOS through the encoder and return the trailing packet."""
        if self._closed:
            raise RuntimeError("MppEncoder is closed")
        if self._flushed:
            return None
        self._flushed = True
        err_buf = ctypes.create_string_buffer(512)
        out = _RkvEncPacket()
        with _mpp_lock:
            ret = _lib.rkv_enc_flush(self._handle, byref(out), err_buf, 512)
        if ret == 1:
            return None  # EOS drain returned nothing
        if ret < 0:
            raise RuntimeError(
                f"rkv_enc_flush failed: ret={ret} — "
                f"{err_buf.value.decode('utf-8', 'replace')}"
            )
        data = b""
        if out.length and out.data:
            data = ctypes.string_at(out.data, out.length)
        return EncodedPacket(
            data=data,
            pts=int(out.pts),
            is_keyframe=bool(out.flag & _PACKET_FLAG_INTRA),
        )


# Sentinel type for the decoder's held frame field on DecodedFrame.raw.
# We use a sentinel since the shim holds the actual MppFrame internally
# and the Python side doesn't need to introspect it.
class _RawFrameRef:
    __slots__ = ("handle",)


class MppDecoder:
    """High-level Python wrapper around ``rkv_dec_*`` in libshim_video.so.

    Calling sequence:

    * :meth:`send_packet` enqueues one access unit; the shim retries on
      BUFFER_FULL with bounded exponential backoff internally.
    * :meth:`recv_frame` returns ``None`` if no frame is ready, or a
      :class:`DecodedFrame` whose ``fd`` / ``ptr`` describe the
      decoded buffer. Hand it back via :meth:`release_frame`.
    """

    def __init__(
        self,
        coding_type: int = MppCodingType.AVC,
        split_parse: bool = True,
        extra_buf_count: int = 24,
        output_format: int = MppFrameFormat.YUV420SP,
    ) -> None:
        self._coding_type = coding_type
        self._extra_buf_count = extra_buf_count
        self._handle: Optional[c_void_p] = None
        self._closed = False

        err_buf = ctypes.create_string_buffer(512)
        handle = c_void_p()
        with _mpp_lock:
            ret = _lib.rkv_dec_create(
                int(coding_type), int(bool(split_parse)),
                int(extra_buf_count), int(output_format),
                err_buf, 512, byref(handle),
            )
        if ret != 0:
            raise RuntimeError(
                f"MppDecoder: rkv_dec_create failed: ret={ret} — "
                f"{err_buf.value.decode('utf-8', 'replace')}"
            )
        self._handle = handle

    def __enter__(self): return self
    def __exit__(self, *exc): self.close()

    def close(self) -> None:
        if self._closed or self._handle is None:
            self._closed = True
            return
        with _mpp_lock:
            # 先排空再 destroy。读完 N 帧就 release() 是常态，此时硬件解码
            # 任务手上还挂着 in-flight 的 MppBuffer，直接 destroy 会让
            # buf_group 的 deinit 撞上 ref_count != 0 而放弃归还，DMA-BUF fd
            # 就永久漏在进程里。送空 EOS packet 让 VPU 把 pipeline 吐干净，
            # 连续 5 次取不到才算排空，最多试 3 轮。
            # 全程 best-effort：drain 失败不该阻止 destroy，否则把「资源
            # 泄漏」升级成「close 抛异常」。
            err_buf = ctypes.create_string_buffer(512)
            out = _RkvDecFrame()
            for _ in range(3):
                if _lib.rkv_dec_put_packet(
                    self._handle, None, 0, 0, 1, err_buf, 512,
                ) != 0:
                    break
                misses = 0
                while misses < 5:
                    if _lib.rkv_dec_get_frame(
                        self._handle, 30, byref(out), err_buf, 512,
                    ) != 1:
                        misses += 1
                        continue
                    _lib.rkv_dec_release_frame(self._handle, err_buf, 512)
                    misses = 0
                if misses >= 5:
                    break  # 已排空，不必再送 EOS
            _lib.rkv_dec_destroy(self._handle)
        self._handle = None
        self._closed = True

    def send_packet(self, data: bytes, pts: int = 0, eos: bool = False) -> None:
        """Submit one packet to the decoder."""
        if self._closed:
            raise RuntimeError("MppDecoder is closed")
        err_buf = ctypes.create_string_buffer(512)
        with _mpp_lock:
            ret = _lib.rkv_dec_put_packet(
                self._handle,
                ctypes.cast(
                    ctypes.c_char_p(bytes(data)) if data else None,
                    c_void_p,
                ),
                len(data) if data else 0,
                int(pts), int(bool(eos)),
                err_buf, 512,
            )
        if ret != 0:
            raise RuntimeError(
                f"rkv_dec_put_packet failed: ret={ret} — "
                f"{err_buf.value.decode('utf-8', 'replace')}"
            )

    def recv_frame(self, timeout_ms: int = 30) -> Optional[DecodedFrame]:
        """Try to pull one decoded frame out of the decoder.

        Returns ``None`` if no frame is ready within *timeout_ms*.
        """
        if self._closed:
            raise RuntimeError("MppDecoder is closed")
        err_buf = ctypes.create_string_buffer(512)
        out = _RkvDecFrame()
        with _mpp_lock:
            ret = _lib.rkv_dec_get_frame(self._handle, int(timeout_ms),
                                          byref(out), err_buf, 512)
        if ret == 0:
            return None
        if ret < 0:
            raise RuntimeError(
                f"rkv_dec_get_frame failed: ret={ret} — "
                f"{err_buf.value.decode('utf-8', 'replace')}"
            )
        # ret == 1: a real frame is in ``out``.
        return DecodedFrame(
            width=int(out.width),
            height=int(out.height),
            hor_stride=int(out.hor_stride),
            ver_stride=int(out.ver_stride),
            fmt=int(out.fmt),
            pts=int(out.pts),
            ptr=int(out.ptr) or 0,
            fd=int(out.fd) if out.fd is not None else -1,
            buf_size=int(out.buf_size),
            raw=None,
        )

    def release_frame(self, frame: DecodedFrame) -> None:
        """Return a frame received via :meth:`recv_frame` to MPP's pool."""
        if self._closed:
            return
        err_buf = ctypes.create_string_buffer(512)
        with _mpp_lock:
            _lib.rkv_dec_release_frame(self._handle, err_buf, 512)

    def reset(self) -> None:
        if self._closed:
            raise RuntimeError("MppDecoder is closed")
        err_buf = ctypes.create_string_buffer(512)
        with _mpp_lock:
            ret = _lib.rkv_dec_reset(self._handle, err_buf, 512)
        if ret != 0:
            raise RuntimeError(
                f"rkv_dec_reset failed: ret={ret} — "
                f"{err_buf.value.decode('utf-8', 'replace')}"
            )


# ---------------------------------------------------------------------------
# Backwards-compat: MppDecoder.MppFrame._RawRef is referenced from
# DecodedFrame.raw type stubs. Provide a real type so isinstance() works.
# ---------------------------------------------------------------------------

MppDecoder._RawRef = _RawFrameRef  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Annex B / avcC / hvcC helpers — pure-Python, used by callers feeding
# compressed streams into the decoder (or for avcC box construction).
# ---------------------------------------------------------------------------


def avcc_to_annexb(data: bytes) -> bytes:
    """Convert a 4-byte-length-prefixed H.264/HEVC sample to Annex B.

    MP4 stores H.264/HEVC samples in "AVCC" / "HVCC" format — a sequence
    of NAL units, each prefixed with a big-endian 4-byte length. MPP's
    ``split_parse`` looks for the classic ``00 00 00 01`` start code,
    so we have to translate. Returns ``data`` unchanged when it
    already starts with a start code.
    """
    if len(data) >= 4 and data[:4] == b"\x00\x00\x00\x01":
        return data
    if len(data) >= 3 and data[:3] == b"\x00\x00\x01":
        return data

    out = bytearray()
    i = 0
    n = len(data)
    while i + 4 <= n:
        nal_len = int.from_bytes(data[i : i + 4], "big")
        i += 4
        if nal_len <= 0 or i + nal_len > n:
            out.extend(data[i - 4 :])
            break
        out.extend(b"\x00\x00\x00\x01")
        out.extend(data[i : i + nal_len])
        i += nal_len
    return bytes(out)


def parse_avcc_extradata(extradata: bytes) -> bytes:
    """Return SPS+PPS as an Annex B byte stream from an avcC configuration.

    MP4 stores H.264 configuration in the avcC sample-entry box, not in
    the media samples themselves. We parse the avcC and emit
    ``00 00 00 01``-prefixed SPS followed by PPS so callers can prepend
    it to the first packet before handing it to ``MppDecoder.send_packet``.
    """
    if not extradata or len(extradata) < 8 or extradata[0] != 1:
        return b""
    n_sps = extradata[5] & 0x1F
    i = 6
    out = bytearray()
    for _ in range(n_sps):
        if i + 2 > len(extradata):
            break
        sps_len = int.from_bytes(extradata[i : i + 2], "big")
        i += 2
        if i + sps_len > len(extradata):
            break
        out.extend(b"\x00\x00\x00\x01")
        out.extend(extradata[i : i + sps_len])
        i += sps_len
    if i + 1 > len(extradata):
        return bytes(out)
    n_pps = extradata[i]
    i += 1
    for _ in range(n_pps):
        if i + 2 > len(extradata):
            break
        pps_len = int.from_bytes(extradata[i : i + 2], "big")
        i += 2
        if i + pps_len > len(extradata):
            break
        out.extend(b"\x00\x00\x00\x01")
        out.extend(extradata[i : i + pps_len])
        i += pps_len
    return bytes(out)


_HVC_NAL_VPS = 32
_HVC_NAL_SPS = 33
_HVC_NAL_PPS = 34


def parse_hvcc_extradata(extradata: bytes) -> bytes:
    """Return VPS+SPS+PPS as an Annex B byte stream from an hvcC configuration."""
    if not extradata or len(extradata) < 23 or extradata[0] != 1:
        return b""
    i = 22
    num_arrays = extradata[i]
    i += 1
    out = bytearray()
    for _ in range(num_arrays):
        if i + 3 > len(extradata):
            break
        nal_type = extradata[i] & 0x3F
        i += 1
        num_nalus = int.from_bytes(extradata[i : i + 2], "big")
        i += 2
        if nal_type not in (_HVC_NAL_VPS, _HVC_NAL_SPS, _HVC_NAL_PPS):
            for _inner in range(num_nalus):
                if i + 2 > len(extradata):
                    break
                nal_len = int.from_bytes(extradata[i : i + 2], "big")
                i += 2 + nal_len
            continue
        for _ in range(num_nalus):
            if i + 2 > len(extradata):
                break
            nal_len = int.from_bytes(extradata[i : i + 2], "big")
            i += 2
            if i + nal_len > len(extradata):
                break
            out.extend(b"\x00\x00\x00\x01")
            out.extend(extradata[i : i + nal_len])
            i += nal_len
    return bytes(out)


def nv12_to_bgr(ptr_addr: int, width: int, height: int,
                hor_stride: int, ver_stride: int):
    """Convert an NV12 frame at *ptr_addr* into a (H, W, 3) BGR ndarray.

    Pure-Python convenience helper. Returns ``None`` when numpy isn't
    available. The returned array borrows memory from the decoder
    buffer; the caller must ``release_frame()`` only after they're
    done with the array.
    """
    try:
        import numpy as np
    except ImportError:
        return None

    y_size = hor_stride * ver_stride
    total = y_size + (hor_stride * ver_stride) // 2
    buf = (ctypes.c_uint8 * total).from_address(ptr_addr)
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(ver_stride * 3 // 2, hor_stride)
    return _nv12_to_bgr_np(arr[: height * 3 // 2, :width])


def _nv12_to_bgr_np(nv12):
    """Vectorised NV12 → BGR888 conversion."""
    import numpy as np

    h, w = nv12.shape[0] * 2 // 3, nv12.shape[1]
    y = nv12[:h, :].astype(np.int16)
    uv = nv12[h : h + h // 2, :].reshape(h // 2, w // 2, 2)
    u = uv[..., 0].astype(np.int16) - 128
    v = uv[..., 1].astype(np.int16) - 128

    u_up = np.repeat(np.repeat(u, 2, axis=0), 2, axis=1)
    v_up = np.repeat(np.repeat(v, 2, axis=0), 2, axis=1)
    b = np.clip(y + 1.772 * u_up, 0, 255).astype(np.uint8)
    g = np.clip(y - 0.34414 * u_up - 0.71414 * v_up, 0, 255).astype(np.uint8)
    r = np.clip(y + 1.402 * v_up, 0, 255).astype(np.uint8)
    return np.stack([b, g, r], axis=-1)


def _bgr_to_nv12_np(bgr, src_format: int):
    """Convert a packed-pixel ndarray (BGR/RGB/RGBA/BGRA) to NV12 via RGA.

    Used by :meth:`MppEncoder.encode` when the caller's
    ``input_format`` is BGR/RGB-family: the firmware's on-chip
    BGR→YUV path was producing garbage on this SoC (decoded output
    showed the three channel planes split into left/centre/right
    strips instead of merged chroma), so we route the conversion
    through RGA2 and feed the encoder NV12 directly. The encoder's
    input buffer is sized for BGR (``w*h*3``) and so always has
    room for the NV12 payload (``w*h*3/2``).

    ``src_format`` is the ``MppFrameFormat`` the caller is feeding in
    (BGR888 / RGB888 / BGRA / RGBA) — drives the RGA source format
    code so the channel order matches the caller's memory layout.

    Both ``height`` and ``width`` must be even — NV12 chroma is
    4:2:0 subsampled. :meth:`MppEncoder.encode` rejects odd
    dimensions upstream so the firmware doesn't see a mismatch
    between the reported frame geometry and the NV12 payload.

    Falls back to a numpy BT.601 conversion if RGA fails (driver
    missing, formats RGA doesn't support, etc.) so the encoder
    still works on dev boxes without librga.
    """
    import ctypes

    import numpy as np

    from . import rga

    h, w = bgr.shape[0], bgr.shape[1]
    if h % 2 or w % 2:
        raise ValueError(
            f"BGR→NV12 conversion requires even width/height, got {w}x{h}"
        )

    rga_src_fmt = {
        MppFrameFormat.BGR888: rga.RgaFormat.BGR_888,
        MppFrameFormat.RGB888: rga.RgaFormat.RGB_888,
        MppFrameFormat.BGRA:   rga.RgaFormat.BGRA_8888,
        MppFrameFormat.RGBA:   rga.RgaFormat.RGBA_8888,
    }[src_format]
    if bgr.ndim != 3 or bgr.shape[2] not in (3, 4):
        raise ValueError(
            f"expected 3- or 4-channel packed-pixel ndarray, got shape {bgr.shape}"
        )

    # RGA wants a contig, aligned buffer. Allocate via ``ctypes`` so we
    # can pass its virtual address to ``wrap_virtual`` without copying.
    nv12_size = h * w * 3 // 2
    dst_storage = (ctypes.c_uint8 * nv12_size)()

    try:
        src_buf = rga.wrap_virtual(bgr.ctypes.data, w, h, rga_src_fmt)
        dst_buf = rga.wrap_virtual(
            ctypes.addressof(dst_storage), w, h, rga.RgaFormat.YCbCr_420_SP
        )
        rga.cvtcolor(src_buf, dst_buf, sync=True)
        rga.flush()
    except Exception:
        # librga missing / format unsupported — fall back to numpy.
        return _bgr_to_nv12_np_fallback(bgr)

    return np.frombuffer(dst_storage, dtype=np.uint8).reshape(h * 3 // 2, w)


def _bgr_to_nv12_np_fallback(bgr):
    """Pure-numpy BT.601 limited-range BGR→NV12. Used only when RGA fails.

    Kept as a separate function so the RGA fast path isn't burdened
    with the fallback's computation overhead on every frame, and so
    future RGA fixes can be benchmarked against the baseline.
    """
    import numpy as np

    h, w = bgr.shape[0], bgr.shape[1]
    if bgr.shape[2] == 4:
        bgr = bgr[:, :, :3]
    b = bgr[..., 0].astype(np.int16)
    g = bgr[..., 1].astype(np.int16)
    r = bgr[..., 2].astype(np.int16)

    # BT.601 limited range, integer fixed-point to match the firmware's
    # coefficients and avoid float rounding noise. Coefficients are
    # multiplied by 256 then right-shifted; +128 rounds to nearest.
    y = (( 66 * r + 129 * g +  25 * b + 128) >> 8) + 16
    u = ((-38 * r -  74 * g + 112 * b + 128) >> 8) + 128
    v = ((112 * r -  94 * g -  18 * b + 128) >> 8) + 128

    y = np.clip(y, 16, 235).astype(np.uint8)
    u = np.clip(u, 16, 240).astype(np.uint8)
    v = np.clip(v, 16, 240).astype(np.uint8)

    u_sub = u[::2, ::2]
    v_sub = v[::2, ::2]
    uv = np.empty((h // 2, w // 2, 2), dtype=np.uint8)
    uv[..., 0] = u_sub
    uv[..., 1] = v_sub

    nv12 = np.empty((h * 3 // 2, w), dtype=np.uint8)
    nv12[:h] = y
    nv12[h:] = uv.reshape(h // 2, w)
    return nv12


def mpp_buffer_sync_cpu_read(buf):
    """No-op stub — kept for backwards-compat; the shim manages
    cache-coherency internally for the buffer paths it owns."""

def mpp_buffer_sync_cpu_write(buf):
    """No-op stub — kept for backwards-compat; the shim manages
    cache-coherency internally for the buffer paths it owns."""


__all__ = [
    "CODING_BY_NAME",
    "DecodedFrame",
    "EncodedPacket",
    "MppBufferMode",
    "MppBufferType",
    "MppCodingType",
    "MppCtxType",
    "MppDecoder",
    "MppEncoder",
    "MppEncProfile",
    "MppEncRcMode",
    "MppFrameFormat",
    "avcc_to_annexb",
    "parse_avcc_extradata",
    "parse_hvcc_extradata",
    "nv12_to_bgr",
    "mpp_buffer_sync_cpu_read",
    "mpp_buffer_sync_cpu_write",
]  # noqa: E501