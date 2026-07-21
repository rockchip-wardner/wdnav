"""ctypes wrapper for Rockchip's librockchip_mpp.so (MPP) hardware video decoder.

This module exposes a thin Python class around the C ``MppApi`` decoder
interface used by ``mpi_dec_test.c``. It is intentionally minimal — only the
symbols exercised by the simple-decoder loop are bound:

    mpp_create / mpp_init / mpp_destroy / mpp_reset
    MppPacket init / deinit / set_data / set_size / set_pos / set_length / set_pts / set_eos
    MppFrame  init / deinit / get_*
    MppBufferGroup get / put + MppBuffer get / put
    mpi->decode_put_packet / decode_get_frame / control / reset
    mpp_dec_cfg_init / deinit / set_u32 + MPP_DEC_GET_CFG / SET_CFG

Anything encoder/ISP related is not bound — call sites for that path should
go straight to ``_lib`` if needed.

The library is **not** bundled with this package. It must be installed
system-wide (apt / dnf / prebuilt deb). Loading order:

    1. ``/usr/lib/aarch64-linux-gnu/librockchip_mpp.so.0``   (Debian multiarch)
    2. ``ctypes.CDLL("rockchip_mpp", mode=RTLD_GLOBAL)``     (ldconfig fallback)

If neither succeeds, :func:`_load_library` raises :class:`OSError` and
:mod:`rkv` ``import``'s :class:`VideoCapture` becomes ``None`` (see
:mod:`rkv.__init__`).
"""

from __future__ import annotations

import ctypes
import os
import threading
from ctypes import (
    CFUNCTYPE,
    POINTER,
    Structure,
    byref,
    c_char_p,
    c_int,
    c_int32,
    c_int64,
    c_size_t,
    c_uint32,
    c_uint64,
    c_void_p,
)
from dataclasses import dataclass, field
from typing import Optional, Tuple

# ---------------------------------------------------------------------------
# Library loading
# ---------------------------------------------------------------------------

_LIBS_TO_TRY = (
    # 系统安装（Debian multiarch 路径）。如果用户把 .so 装到了别的地方，
    # 由下面 CTYPES.CDLL("rockchip_mpp", ...) 通过 ldconfig 兜底。
    "/usr/lib/aarch64-linux-gnu/librockchip_mpp.so.0",
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
    # 兜底：让 dlopen 走 LD_LIBRARY_PATH / /etc/ld.so.cache。
    try:
        return ctypes.CDLL("rockchip_mpp", mode=ctypes.RTLD_GLOBAL)
    except OSError as e:
        raise OSError(
            "Could not load librockchip_mpp.so.0 from system paths. "
            "Install librockchip-mpp system-wide (e.g. apt install "
            "librockchip-mpp1) or place librockchip_mpp.so.0 under "
            "/usr/lib/aarch64-linux-gnu/ and run `ldconfig`. "
            "Last error: " + repr(last_err or e)
        )


_lib = _load_library()

# A single lock guards all MPP calls.  The C library is not documented as
# thread-safe across separate contexts and we currently only own one context
# at a time, but it is cheap insurance against accidental concurrent use.
_mpp_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Primitive types (mirror rk_type.h / mpp_err.h)
# ---------------------------------------------------------------------------

RK_U32 = c_uint32
RK_S32 = c_int32
RK_U64 = c_uint64
RK_S64 = c_int64
MPP_RET = c_int32  # signed so negative error codes work
MppCtx = c_void_p
MppPacket = c_void_p
MppFrame = c_void_p
MppBuffer = c_void_p
MppBufferGroup = c_void_p
MppParam = c_void_p


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
    HEVC = 13        # H.265
    AVS = 18
    AVS2 = 19
    AV1 = 20


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
    RGBA = (1 << 16) + 13
    BGRA = (1 << 16) + 12


class MppBufferType:
    NORMAL = 0
    ION = 1
    EXT_DMA = 2
    DRM = 3
    DMA_HEAP = 4
    # Flag bits in the upper 16 bits — match rk_mpp_buffer.h
    # ``MPP_BUFFER_FLAGS_*`` constants. Combine with type via ``|``:
    # ``MppBufferType.DRM | MppBufferType.FLAG_CACHABLE``.
    FLAG_CONTIG = 0x00010000
    FLAG_CACHABLE = 0x00020000
    FLAG_WC = 0x00040000
    FLAG_SECURE = 0x00080000
    FLAG_KMAP = 0x00100000
    FLAG_DMA32 = 0x00200000


class MppBufferMode:
    INTERNAL = 0
    EXTERNAL = 1


# Subset of MpiCmd we actually use (see rk_mpi_cmd.h). Values come straight
# from the header so they stay compatible with future header revisions.
# Command codes are reconstructed from the bit layout documented in
# rk_mpi_cmd.h so we don't depend on the enum ordering. Each segment has
# an explicit base value (via MPP_FLAG_OR) and the enumerators that follow
# it are just `base + n`:
#
#   bit 20-23 module id   (CODEC = 0x00300000)
#   bit 16-19 ctx id      (DEC    = 0x00010000)
#   bit  8-15 cfg id      (QUERY  = 0x00000100 / CFG = 0x00000200)
#   bit  0- 7 command id  (1-based)
#
# So:
#   MPP_DEC_CMD_BASE   = 0x00310000
#   MPP_DEC_SET_OUTPUT_FORMAT = 0x00310000 + 10 = 0x0031000A
#   CMD_DEC_CMD_CFG     = 0x00310200
#   MPP_DEC_SET_CFG     = 0x00310201
#   MPP_DEC_GET_CFG     = 0x00310202
class MpiCmd:
    DEC_SET_FRAME_INFO = 0x00310001
    DEC_SET_EXT_BUF_GROUP = 0x00310002
    DEC_SET_INFO_CHANGE_READY = 0x00310003
    DEC_SET_OUTPUT_FORMAT = 0x0031000A
    DEC_SET_CFG = 0x00310201
    DEC_GET_CFG = 0x00310202

    # MPP_ENC_* command codes reconstructed from rk_mpi_cmd.h. The bit
    # layout is:
    #   bit 20-23 module id   (CODEC = 0x00300000)
    #   bit 16-19 ctx id      (DEC    = 0x00010000, ENC = 0x00020000)
    #   bit  8-15 cfg id      (QUERY  = 0x00000100, CFG  = 0x00000200)
    #   bit  0- 7 command id  (1-based, auto-incremented from CMD_*_BASE)
    #
    # MPP_ENC_CMD_BASE = 0x00320000, then enum auto-increments:
    #   0x01 SET_CFG  0x02 GET_CFG  0x03 SET_PREP_CFG ...
    #   0x07 SET_CODEC_CFG  0x08 GET_CODEC_CFG
    #   0x09 SET_IDR_FRAME
    #   0x0A SET_OSD_LEGACY_0 ... 0x0C SET_OSD_LEGACY_2
    #   0x0D GET_HDR_SYNC
    # MPP_ENC_CFG_MISC (separate block) holds SET_HEADER_MODE / GET_HEADER_MODE.
    # We ask for ``MPP_ENC_HEADER_MODE_EACH_IDR`` after ``MPP_ENC_SET_CFG``
    # so every IDR access unit that comes out of the encoder carries
    # its own SPS/PPS NALs in the bitstream — a downstream
    # ``VideoDecoder`` can then start decoding from any random
    # keyframe without having first received the separate avcC
    # extradata packet that the default mode only emits once at
    # stream start.
    ENC_SET_CFG = 0x00320201
    ENC_GET_CFG = 0x00320202
    ENC_SET_IDR_FRAME = 0x00320009
    ENC_GET_HDR_SYNC = 0x0032000D
    # Activate a rate-control implementation. Without this call the rc ctx
    # has no implementation bound, so per-frame paths trip
    # ``fps_*_denom >= 1`` asserts on Rockchip firmware for H.264 / H.265
    # right after ``MPP_ENC_SET_CFG``.
    ENC_SET_RC_API_CURRENT = 0x00320205
    # ``MPP_ENC_SET_HEADER_MODE`` lives in the MISC sub-block
    # (``MPP_ENC_CFG_MISC`` = CMD_MODULE_CODEC | CMD_CTX_ID_ENC |
    # CMD_ENC_CFG_MISC = 0x00328000); the first member
    # (``MPP_ENC_SET_HEADER_MODE``) is the base + 1.
    ENC_SET_HEADER_MODE = 0x00328001


# Header (SPS/PPS/VPS) emission mode for MPP_ENC_SET_HEADER_MODE.
# Values match rk_venc_cmd.h / MppEncHeaderMode. ``EACH_IDR`` is what
# rkv.VideoWriter asks for when emitting H.264 over the WS sink: every
# keyframe then carries its own SPS/PPS, so a downstream decoder
# (WebCodecs ``VideoDecoder``, etc.) can start decoding from any
# random IDR without first having to wait for the separate avcC
# extradata packet. ``NO_HEADER`` (bit 1 in the rkv enum, value 1
# historically) isn't part of the public C enum; the previous rkv
# definition was wrong and is kept here only because the source
# value ``0`` corresponds to ``MPP_ENC_HEADER_MODE_DEFAULT``.
class MppEncHeaderMode:
    DEFAULT = 0     # MPP_ENC_HEADER_MODE_DEFAULT
    EACH_IDR = 1    # MPP_ENC_HEADER_MODE_EACH_IDR — see rk_venc_cmd.h:661


# ---------------------------------------------------------------------------
# MppBufferInfo (mpp_buffer.h) — used when reading pointer / fd out of MppBuffer
# ---------------------------------------------------------------------------


class MppBufferInfo(Structure):
    _fields_ = [
        ("type", c_uint32),
        ("size", c_size_t),
        ("ptr", c_void_p),
        ("hnd", c_void_p),
        ("fd", c_int),
        ("index", c_int),
    ]


# ---------------------------------------------------------------------------
# MppApi (rk_mpi.h) — function-pointer table returned by mpp_create()
# ---------------------------------------------------------------------------
#
# The struct layout (offsets are critical — ctypes will hand back garbage if
# the order doesn't match the C definition):
#
#   RK_U32 size, version;
#   MPP_RET (*decode)(ctx, packet, *frame);
#   MPP_RET (*decode_put_packet)(ctx, packet);
#   MPP_RET (*decode_get_frame)(ctx, *frame);
#   MPP_RET (*encode)(ctx, frame, *packet);
#   MPP_RET (*encode_put_frame)(ctx, frame);
#   MPP_RET (*encode_get_packet)(ctx, *packet);
#   MPP_RET (*isp)(ctx, dst, src);
#   MPP_RET (*isp_put_frame)(ctx, frame);
#   MPP_RET (*isp_get_frame)(ctx, *frame);
#   MPP_RET (*poll)(ctx, type, timeout);
#   MPP_RET (*dequeue)(ctx, type, *task);
#   MPP_RET (*enqueue)(ctx, type, task);
#   MPP_RET (*reset)(ctx);
#   MPP_RET (*control)(ctx, cmd, param);
#   RK_U32 reserv[16];


_C_DECODE = CFUNCTYPE(MPP_RET, MppCtx, MppPacket, POINTER(MppFrame))
_C_DECODE_PUT_PACKET = CFUNCTYPE(MPP_RET, MppCtx, MppPacket)
_C_DECODE_GET_FRAME = CFUNCTYPE(MPP_RET, MppCtx, POINTER(MppFrame))
_C_ENCODE = CFUNCTYPE(MPP_RET, MppCtx, MppFrame, POINTER(MppPacket))
_C_ENCODE_PUT_FRAME = CFUNCTYPE(MPP_RET, MppCtx, MppFrame)
_C_ENCODE_GET_PACKET = CFUNCTYPE(MPP_RET, MppCtx, POINTER(MppPacket))
_C_ISP = CFUNCTYPE(MPP_RET, MppCtx, MppFrame, MppFrame)
_C_ISP_PUT_FRAME = CFUNCTYPE(MPP_RET, MppCtx, MppFrame)
_C_ISP_GET_FRAME = CFUNCTYPE(MPP_RET, MppCtx, POINTER(MppFrame))
_C_POLL = CFUNCTYPE(MPP_RET, MppCtx, c_int, c_int)
_C_DEQUEUE = CFUNCTYPE(MPP_RET, MppCtx, c_int, c_void_p)
_C_ENQUEUE = CFUNCTYPE(MPP_RET, MppCtx, c_int, c_void_p)
_C_RESET = CFUNCTYPE(MPP_RET, MppCtx)
_C_CONTROL = CFUNCTYPE(MPP_RET, MppCtx, c_int, MppParam)


class RcApiBrief(Structure):
    """Mirrors ``RcApiBrief_t`` from mpp_rc_api.h. Pass by reference to
    ``MPP_ENC_SET_RC_API_CURRENT`` to bind a named rc implementation
    ("default" / "smart") to the encoder — without this the rc ctx has
    ``api = NULL`` and per-frame paths trip ``fps_*_denom >= 1``
    asserts on Rockchip firmware.
    """
    _fields_ = [
        ("name", c_char_p),
        ("type", c_int32),
    ]


class MppApi(Structure):
    _fields_ = [
        ("size", RK_U32),
        ("version", RK_U32),
        ("decode", _C_DECODE),
        ("decode_put_packet", _C_DECODE_PUT_PACKET),
        ("decode_get_frame", _C_DECODE_GET_FRAME),
        ("encode", _C_ENCODE),
        ("encode_put_frame", _C_ENCODE_PUT_FRAME),
        ("encode_get_packet", _C_ENCODE_GET_PACKET),
        ("isp", _C_ISP),
        ("isp_put_frame", _C_ISP_PUT_FRAME),
        ("isp_get_frame", _C_ISP_GET_FRAME),
        ("poll", _C_POLL),
        ("dequeue", _C_DEQUEUE),
        ("enqueue", _C_ENQUEUE),
        ("reset", _C_RESET),
        ("control", _C_CONTROL),
        ("reserv", RK_U32 * 16),
    ]


# ---------------------------------------------------------------------------
# Top-level MppApi* getters / setters
# ---------------------------------------------------------------------------

_lib.mpp_create.argtypes = [POINTER(MppCtx), POINTER(POINTER(MppApi))]
_lib.mpp_create.restype = MPP_RET

_lib.mpp_init.argtypes = [MppCtx, c_int, c_int]
_lib.mpp_init.restype = MPP_RET

_lib.mpp_destroy.argtypes = [MppCtx]
_lib.mpp_destroy.restype = MPP_RET

_lib.mpp_check_support_format.argtypes = [c_int, c_int]
_lib.mpp_check_support_format.restype = MPP_RET


# MppPacket
_lib.mpp_packet_init.argtypes = [POINTER(MppPacket), c_void_p, c_size_t]
_lib.mpp_packet_init.restype = MPP_RET

_lib.mpp_packet_init_with_buffer.argtypes = [POINTER(MppPacket), MppBuffer]
_lib.mpp_packet_init_with_buffer.restype = MPP_RET

_lib.mpp_packet_deinit.argtypes = [POINTER(MppPacket)]
_lib.mpp_packet_deinit.restype = MPP_RET

_lib.mpp_packet_set_data.argtypes = [MppPacket, c_void_p]
_lib.mpp_packet_set_data.restype = None

_lib.mpp_packet_set_size.argtypes = [MppPacket, c_size_t]
_lib.mpp_packet_set_size.restype = None

_lib.mpp_packet_set_pos.argtypes = [MppPacket, c_void_p]
_lib.mpp_packet_set_pos.restype = None

_lib.mpp_packet_set_length.argtypes = [MppPacket, c_size_t]
_lib.mpp_packet_set_length.restype = None

_lib.mpp_packet_set_pts.argtypes = [MppPacket, RK_S64]
_lib.mpp_packet_set_pts.restype = None

_lib.mpp_packet_set_eos.argtypes = [MppPacket]
_lib.mpp_packet_set_eos.restype = MPP_RET

_lib.mpp_packet_get_pts.argtypes = [MppPacket]
_lib.mpp_packet_get_pts.restype = RK_S64

# Read-side accessors used by the encoder wrapper (read out encoded NALs).
_lib.mpp_packet_get_pos.argtypes = [MppPacket]
_lib.mpp_packet_get_pos.restype = c_void_p

_lib.mpp_packet_get_length.argtypes = [MppPacket]
_lib.mpp_packet_get_length.restype = c_size_t

_lib.mpp_packet_get_flag.argtypes = [MppPacket]
_lib.mpp_packet_get_flag.restype = RK_U32

# Packet flag bits from rk_mpp_packet.h / mpp_packet.h. We only need the
# intra-frame marker here; the other bits (REF / DISPLAY / CORRUPT / END)
# aren't surfaced to Python callers.
MPP_PACKET_FLAG_INTRA = 1 << 4   # bit 4 of the packet flag word
# (the previous definition had this as ``1 << 0`` which silently classified
# every IDR as a delta, so the WS sink never sent a ``PKT_TYPE_KEY`` and
# the WebCodecs decoder stayed forever in pre-IDR state → black screen.)

_lib.mpp_packet_get_eos.argtypes = [MppPacket]
_lib.mpp_packet_get_eos.restype = RK_U32


# MppMeta (key/value bag attached to a frame). The encoder uses it to
# carry the *output* packet pre-allocated by the caller — without that
# binding the MJPEG encoder's HAL falls back to asking the buffer group
# for one, which on this firmware reports ``size 0`` and trips the
# ``mpp_enc_check_pkt_buf`` assertion. The H.264 path can also be
# driven via meta, but historically our wrapper has used the
# direct ``encode_get_packet`` API and that still works for H.264.
MppMeta = c_void_p

MppMetaKey = c_int32
KEY_OUTPUT_PACKET = (ord('o') << 24) | (ord('p') << 16) | (ord('k') << 8) | ord('t')

_lib.mpp_meta_get_with_tag.argtypes = [
    POINTER(MppMeta), c_char_p, c_char_p,
]
_lib.mpp_meta_get_with_tag.restype = MPP_RET
_lib.mpp_meta_put.argtypes = [MppMeta]
_lib.mpp_meta_put.restype = MPP_RET
_lib.mpp_meta_set_packet.argtypes = [MppMeta, MppMetaKey, MppPacket]
_lib.mpp_meta_set_packet.restype = MPP_RET
_lib.mpp_meta_get_packet.argtypes = [MppMeta, MppMetaKey, POINTER(MppPacket)]
_lib.mpp_meta_get_packet.restype = MPP_RET

# MppFrame
_lib.mpp_frame_init.argtypes = [POINTER(MppFrame)]
_lib.mpp_frame_init.restype = MPP_RET

_lib.mpp_frame_deinit.argtypes = [POINTER(MppFrame)]
_lib.mpp_frame_deinit.restype = MPP_RET

for _name in (
    "get_width", "get_height", "get_hor_stride", "get_ver_stride",
    "get_buf_size", "get_fmt", "get_pts", "get_dts", "get_poc",
    "get_errinfo", "get_discard", "get_eos", "get_info_change",
    "get_buffer",
):
    _fn = getattr(_lib, f"mpp_frame_{_name}")
    _fn.argtypes = [MppFrame]
    _fn.restype = RK_U32 if _name.startswith("get_buf_size") is False else c_size_t
# The above loop is a bit awkward because buf_size returns size_t and get_buffer
# returns MppBuffer. Fix the few that don't follow the RK_U32 rule:
_lib.mpp_frame_get_buf_size.restype = c_size_t
_lib.mpp_frame_get_pts.restype = RK_S64
_lib.mpp_frame_get_dts.restype = RK_S64
_lib.mpp_frame_get_buffer.restype = MppBuffer


# Encoder-side frame setters. We only bind what the encoder driver
# actually consumes (size / format / buffer / pts / dts).
for _name in (
    "set_width", "set_height",
    "set_hor_stride", "set_ver_stride",
    "set_fmt", "set_pts", "set_dts",
    "set_buffer", "set_buf_size",
    "set_eos",
):
    _fn = getattr(_lib, f"mpp_frame_{_name}")
    _fn.argtypes = [MppFrame, RK_U32]
    if _name == "set_buffer":
        _fn.argtypes = [MppFrame, MppBuffer]
    elif _name == "set_pts" or _name == "set_dts":
        _fn.argtypes = [MppFrame, RK_S64]
    _fn.restype = None if _name != "set_eos" else MPP_RET

# MppFrame ↔ MppMeta attachment. The encoder reads KEY_OUTPUT_PACKET
# off the frame's meta when ``encode_put_frame`` is dispatched, so
# getting the bindings for the setter / getter is required to drive
# the meta-based encode path (used by the MJPEG encoder).
_lib.mpp_frame_set_meta.argtypes = [MppFrame, MppMeta]
_lib.mpp_frame_set_meta.restype = None
_lib.mpp_frame_get_meta.argtypes = [MppFrame]
_lib.mpp_frame_get_meta.restype = MppMeta


# MppBuffer
_lib.mpp_buffer_get_with_tag.argtypes = [
    MppBufferGroup, POINTER(MppBuffer), c_size_t, c_char_p, c_char_p,
]
_lib.mpp_buffer_get_with_tag.restype = MPP_RET

# Single-export ``mpp_buffer_get_with_tag`` is the only ``*_get`` callable
# from the .so — the public ``mpp_buffer_get`` name is a macro that
# calls ``mpp_buffer_get_with_tag(group, buf, size, MODULE_TAG, __FUNCTION__)``
# at the C level, so we drive the latter directly. ``mpp_buffer_put`` is
# similarly macro-only; the underlying export is
# ``mpp_buffer_put_with_caller``.

_lib.mpp_buffer_put_with_caller.argtypes = [MppBuffer, c_char_p]
_lib.mpp_buffer_put_with_caller.restype = MPP_RET

_lib.mpp_buffer_get_ptr_with_caller.argtypes = [MppBuffer, c_char_p]
_lib.mpp_buffer_get_ptr_with_caller.restype = c_void_p

_lib.mpp_buffer_get_size_with_caller.argtypes = [MppBuffer, c_char_p]
_lib.mpp_buffer_get_size_with_caller.restype = c_size_t

_lib.mpp_buffer_get_fd_with_caller.argtypes = [MppBuffer, c_char_p]
_lib.mpp_buffer_get_fd_with_caller.restype = c_int

# External-DMA-BUF import. ``mpp_buffer_import_with_tag`` wraps a
# caller-owned fd (e.g. the DMA-BUF handed back by the MPP decoder)
# as an MppBuffer so the encoder can read from it without copying
# through a userspace malloc buffer. The buffer's reference count
# stays at 1; the caller is responsible for keeping the underlying
# DMA-BUF alive (the ``MppBuffer`` handle inherits its lifetime from
# the file descriptor).
_lib.mpp_buffer_import_with_tag.argtypes = [
    MppBufferGroup, POINTER(MppBufferInfo), POINTER(MppBuffer),
    c_char_p, c_char_p,
]
_lib.mpp_buffer_import_with_tag.restype = MPP_RET

# Cache-coherency ops: required if a CPU wants to read or write a DMA
# buffer (the VPU writes through DMA, which doesn't invalidate the CPU
# data cache). The fd path (used by RGA) doesn't need these.
_lib.mpp_buffer_sync_begin_f.argtypes = [MppBuffer, c_int, c_char_p]
_lib.mpp_buffer_sync_begin_f.restype = MPP_RET

_lib.mpp_buffer_sync_end_f.argtypes = [MppBuffer, c_int, c_char_p]
_lib.mpp_buffer_sync_end_f.restype = MPP_RET


def mpp_buffer_sync_cpu_read(buf: MppBuffer) -> None:
    """Invalidate the CPU cache so the next CPU read sees DMA writes.

    Required before reading an MppBuffer's virtual address from the CPU
    when the VPU / RGA just finished writing it via DMA. The fd-based
    path (RGA) does *not* need this — DMA-BUF passing is cache-coherent.
    """
    _check(_lib.mpp_buffer_sync_begin_f(buf, 1, b"rkv-mpp"), "mpp_buffer_sync_begin (cpu read)")


def mpp_buffer_sync_cpu_write(buf: MppBuffer) -> None:
    """Flush the CPU cache so DMA reads see the CPU's writes."""
    _check(_lib.mpp_buffer_sync_end_f(buf, 0, b"rkv-mpp"), "mpp_buffer_sync_end (cpu write)")


# MppBufferGroup
_lib.mpp_buffer_group_get.argtypes = [
    POINTER(MppBufferGroup), c_int, c_int, c_char_p, c_char_p,
]
_lib.mpp_buffer_group_get.restype = MPP_RET

_lib.mpp_buffer_group_put.argtypes = [MppBufferGroup]
_lib.mpp_buffer_group_put.restype = MPP_RET

_lib.mpp_buffer_group_usage.argtypes = [MppBufferGroup]
_lib.mpp_buffer_group_usage.restype = c_size_t

if hasattr(_lib, "mpp_buffer_group_limit_config"):
    _lib.mpp_buffer_group_limit_config.argtypes = [MppBufferGroup, c_size_t, c_int]
    _lib.mpp_buffer_group_limit_config.restype = MPP_RET


# MppDecCfg
_lib.mpp_dec_cfg_init.argtypes = [POINTER(c_void_p)]
_lib.mpp_dec_cfg_init.restype = MPP_RET

_lib.mpp_dec_cfg_deinit.argtypes = [c_void_p]
_lib.mpp_dec_cfg_deinit.restype = MPP_RET

_lib.mpp_dec_cfg_set_u32.argtypes = [c_void_p, c_char_p, RK_U32]
_lib.mpp_dec_cfg_set_u32.restype = MPP_RET

_lib.mpp_dec_cfg_get_u32.argtypes = [c_void_p, c_char_p, POINTER(RK_U32)]
_lib.mpp_dec_cfg_get_u32.restype = MPP_RET


# ---------------------------------------------------------------------------
# MppEncCfg — generic key/value configuration store for the encoder. Used
# via ``MPP_ENC_SET_CFG`` / ``MPP_ENC_GET_CFG``. Same calling shape as
# MppDecCfg but the keys are different (e.g. ``prep:width``, ``rc:gop``).
# ---------------------------------------------------------------------------

_lib.mpp_enc_cfg_init.argtypes = [POINTER(c_void_p)]
_lib.mpp_enc_cfg_init.restype = MPP_RET

_lib.mpp_enc_cfg_deinit.argtypes = [c_void_p]
_lib.mpp_enc_cfg_deinit.restype = MPP_RET

for _enc_fn in ("set_u8", "set_u16", "set_u32", "set_u64"):
    _f = getattr(_lib, f"mpp_enc_cfg_{_enc_fn}")
    _f.argtypes = [c_void_p, c_char_p, c_uint32]
    _f.restype = MPP_RET
for _enc_fn in ("set_s8", "set_s16", "set_s32", "set_s64"):
    _f = getattr(_lib, f"mpp_enc_cfg_{_enc_fn}")
    _f.argtypes = [c_void_p, c_char_p, c_int32]
    _f.restype = MPP_RET
for _enc_fn in ("set_st", "set_ptr"):
    _f = getattr(_lib, f"mpp_enc_cfg_{_enc_fn}")
    _f.argtypes = [c_void_p, c_char_p, c_void_p]
    _f.restype = MPP_RET

for _enc_fn in ("get_u8", "get_u16", "get_u32", "get_u64"):
    _f = getattr(_lib, f"mpp_enc_cfg_{_enc_fn}")
    _f.argtypes = [c_void_p, c_char_p, POINTER(c_uint32)]
    _f.restype = MPP_RET
for _enc_fn in ("get_s8", "get_s16", "get_s32", "get_s64"):
    _f = getattr(_lib, f"mpp_enc_cfg_{_enc_fn}")
    _f.argtypes = [c_void_p, c_char_p, POINTER(c_int32)]
    _f.restype = MPP_RET
for _enc_fn in ("get_st", "get_ptr"):
    _f = getattr(_lib, f"mpp_enc_cfg_{_enc_fn}")
    _f.argtypes = [c_void_p, c_char_p, c_void_p]
    _f.restype = MPP_RET


# ---------------------------------------------------------------------------
# High-level helpers
# ---------------------------------------------------------------------------


def _check(ret: int, where: str) -> None:
    """Raise if a MPP call returns a negative status."""
    if ret < 0:
        # MPP_ERR_* values are negative numbers; expose them in the message.
        raise RuntimeError(f"MPP call {where!r} failed: ret={ret}")


# Map a handful of FFmpeg codec names (as reported by PyAV / avcodec) to
# MppCodingType values. Extend as needed.
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


# ---------------------------------------------------------------------------
# Format / conversion helpers — shared between ``DecodedFrame.numpy`` and the
# eager ``VideoCapture.read`` path. Putting them here keeps the mapping
# definitions (MPP-format → RGA-format and the user-facing format names)
# next to the MPP decoder that emits ``fmt`` codes.
# ---------------------------------------------------------------------------


# MPP frame-format codes we accept on the RGA input side. Maps directly
# onto the rga.h surface-format enum (left-shifted by 8): 0 = NV12,
# 4 = I420, 5 = NV21.
_MPP_TO_RGA_FMT = {
    0: "YCbCr_420_SP",
    4: "YCbCr_420_P",
    5: "YCrCb_420_SP",
}
_MPP_TO_RGA_NAME = {0: "NV12", 4: "I420", 5: "NV21"}


# Map the short names users pass to ``numpy(fmt=...)`` /
# ``VideoCapture.read(fmt=...)`` to the RgaFormat constant plus
# buffer-layout metadata. ``layout`` is one of:
#   ``"HWC"``  — packed pixel (BGR/RGB/BGRA/RGBA); shape ``(H, W, C)``.
#   ``"YUV"``  — NV12/NV21 full buffer; shape ``(H * 3 // 2, W)`` (Y plane
#                + interleaved UV at offset H*W).
#   ``"GRAY"`` — single-channel luma from a NV12 RGA pass; shape ``(H, W)``.
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
    """Return ``(rga_format, bytes_per_pixel, layout)`` for ``fmt``.

    ``None`` or empty string defaults to BGR_888 packed pixels. The
    RGA format code is resolved at call time (lazy) because importing
    ``rkv.rga`` at module-load time would force the rga ctypes layer
    to load before the user has had a chance to mock it for tests.
    """
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


def _as_ndarray(
    buf_storage,
    width: int,
    height: int,
    bpp: float,
    layout: str,
) -> "np.ndarray":
    """Reshape an RGA output buffer into the right ndarray view.

    ``layout`` is one of ``"HWC"`` (packed pixels), ``"YUV"`` (full
    NV12 / NV21 plane), or ``"GRAY"`` (single-channel luma extracted
    from the Y plane of a NV12 RGA pass). For 1.5-BPP layouts we
    expose the full buffer; for 1-BPP we slice off the Y plane (the
    first H*W bytes of the NV12 layout) so the returned ndarray is
    exactly ``(H, W)``.
    """
    import numpy as np

    raw = np.frombuffer(buf_storage, dtype=np.uint8)
    if layout == "YUV":
        return raw.reshape(height * 3 // 2, width)
    if layout == "GRAY":
        return raw[: height * width].reshape(height, width)
    channels = int(round(bpp))
    return raw.reshape(height, width, channels)


@dataclass
class DecodedFrame:
    """A decoded frame returned by :class:`MppDecoder`.

    ``fd`` is the DMA-BUF file descriptor of the underlying MppBuffer
    and is the preferred handle to hand to other hardware engines (RGA,
    display controller, encoder …). It avoids the CPU cache-coherency
    dance that ``ptr`` would require.

    ``ptr`` is still populated for callers who want to feed the frame
    into a CPU-side pipeline (OpenCV, numpy ops, etc.). If
    the CPU is going to read it, call :func:`mpp_buffer_sync_cpu_read`
    *before* touching the memory, and :func:`mpp_buffer_sync_cpu_write`
    *after* any write — the wrapper does that automatically for the
    built-in code paths.

    Call :meth:`numpy` to materialise the frame into a contiguous ndarray
    via RGA2 — that gives the same result as the eager
    :meth:`rkv.VideoCapture.read` path without touching the CPU between
    MPP and RGA.
    """

    width: int
    height: int
    hor_stride: int
    ver_stride: int
    fmt: int
    pts: int
    ptr: int           # pointer into the MppBuffer's storage (Y plane for NV12)
    fd: int            # DMA-BUF fd of the MppBuffer; -1 if not available
    buf_size: int      # size of the MppBuffer in bytes
    raw: "MppFrame"    # underlying MppFrame handle (c_void_p), for advanced users

    # Per-call output overrides set by :meth:`rkv.VideoCapture.read0` and
    # consumed as defaults by :meth:`numpy`. ``None`` means "use the
    # source's native size / BGR". Calling :meth:`numpy` with explicit
    # overrides always wins.
    out_w: Optional[int] = None
    out_h: Optional[int] = None
    out_fmt: Optional[str] = None

    def numpy(
        self,
        w: Optional[int] = None,
        h: Optional[int] = None,
        fmt: Optional[str] = None,
    ) -> "np.ndarray":
        """Materialise this frame into a contiguous ndarray via RGA2.

        Mirrors :meth:`rkv.VideoCapture.read` so callers can write
        either::

            ret, frame = cap.read(w, h, fmt)

        or::

            ret, frame0 = cap.read0(w, h, fmt)
            frame = frame0.numpy()

        with the same ndarray result. ``w`` / ``h`` resize via RGA in
        the same hardware pass as the colour conversion; ``fmt``
        selects the output pixel format (case-insensitive):

        * ``"BGR"`` (default), ``"RGB"``, ``"BGRA"``, ``"RGBA"`` —
          packed pixels, shape ``(H, W, C)``.
        * ``"NV12"``, ``"NV21"`` — full YUV420 buffer, shape
          ``(H * 3 // 2, W)``. The interleaved UV plane sits at offset
          ``H * W``.
        * ``"GRAY"`` / ``"GREY"`` / ``"Y"`` — single-channel luma, shape
          ``(H, W)``.

        Args explicitly passed here override the ``out_w`` / ``out_h`` /
        ``out_fmt`` stashed by :meth:`rkv.VideoCapture.read0`; either of
        them being ``None`` falls back to the stashed value, which in
        turn defaults to ``None`` (= source size, BGR).
        """
        import ctypes

        from . import rga

        # Lazy numpy import — keeps mpp.py usable in environments without
        # numpy (rare, but the ctypes layer is the primary dependency).
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

        # RGA wants a destination stride aligned to 16 bytes; the visible
        # region may be smaller than the allocated buffer, so we trim
        # it back after the conversion.
        rga_w = (target_w + 15) & ~15
        bpp_target = rga.bytes_per_pixel(target_fmt)
        size = -(-int(rga_w * target_h * bpp_target) // 1)  # ceil
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


class MppDecoder:
    """Thin ctypes wrapper around MPP's hardware video decoder.

    Mirrors the flow of ``mpi_dec_test.c``: create context, init as decoder,
    configure split_parse, then ``decode()`` packet-by-packet and drain frames.

    Parameters
    ----------
    coding_type:
        One of :class:`MppCodingType` (e.g. ``MppCodingType.AVC`` for H.264).
    split_parse:
        Whether to enable MPP's internal frame splitter. ``True`` lets MPP
        find frame boundaries inside arbitrary packet chunks — keep it on
        unless you know your demuxer already produces frame-aligned packets.
    extra_buf_count:
        Number of frame buffers to allocate when the decoder reports its
        first info-change. 24 matches the upstream test's default.
    output_format:
        Optional forced output format (``MppFrameFormat.YUV420SP`` by default,
        which is NV12 — most decoders natively output this).
    """

    def __init__(
        self,
        coding_type: int = MppCodingType.AVC,
        split_parse: bool = True,
        extra_buf_count: int = 24,
        output_format: int = MppFrameFormat.YUV420SP,
    ) -> None:
        with _mpp_lock:
            self._ctx = MppCtx()
            self._api = POINTER(MppApi)()
            _check(_lib.mpp_create(byref(self._ctx), byref(self._api)), "mpp_create")
            _check(_lib.mpp_init(self._ctx, MppCtxType.DEC, coding_type), "mpp_init")

            # split_parse configuration goes through MppDecCfg, same as
            # mpi_dec_test.c lines 518-541.
            cfg = c_void_p()
            _check(_lib.mpp_dec_cfg_init(byref(cfg)), "mpp_dec_cfg_init")
            try:
                _check(
                    self._api.contents.control(self._ctx, MpiCmd.DEC_GET_CFG, cfg),
                    "MPP_DEC_GET_CFG",
                )
                _check(
                    _lib.mpp_dec_cfg_set_u32(
                        cfg, b"base:split_parse", 1 if split_parse else 0
                    ),
                    "mpp_dec_cfg_set_u32 split_parse",
                )
                _check(
                    self._api.contents.control(self._ctx, MpiCmd.DEC_SET_CFG, cfg),
                    "MPP_DEC_SET_CFG",
                )
            finally:
                _lib.mpp_dec_cfg_deinit(cfg)

            # Force the decoder's output format. The MPP default is to give
            # us compressed-domain output (FBC / tile) which the caller has
            # to convert; asking for NV12 keeps the host code simple.
            # The C reference's "simple" decoder path (H.264/HEVC/VP9) does
            # not set this, so it matches the same code path the SDK used to
            # ship against.

        self._coding_type = coding_type
        self._extra_buf_count = extra_buf_count
        self._output_format = output_format
        self._buf_group: Optional[MppBufferGroup] = None
        # Lazily-initialised MppPacket (created on first send_packet) plus a
        # scratch buffer the packet points into. The buffer must outlive
        # decode_put_packet until the corresponding frame is consumed; we
        # reuse one big buffer for every send_packet because callers always
        # drain pending frames between sends.
        self._packet: MppPacket = MppPacket()
        self._packet_buf: Optional[ctypes.Array] = None
        self._packet_buf_size: int = 0
        self._closed = False

    # -- context manager ----------------------------------------------------

    def __enter__(self) -> "MppDecoder":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with _mpp_lock:
            if self._packet:
                _lib.mpp_packet_deinit(byref(self._packet))
                self._packet = MppPacket()
            if self._buf_group:
                _lib.mpp_buffer_group_put(self._buf_group)
                self._buf_group = None
            if self._ctx:
                _lib.mpp_destroy(self._ctx)
                self._ctx = MppCtx()

    # -- packet feeding -----------------------------------------------------

    def _ensure_packet_buf(self, size: int) -> ctypes.Array:
        """Make sure ``self._packet_buf`` is at least *size* bytes."""
        # Always allocate at least 1 MiB so we don't reallocate on every
        # small packet.
        target = max(size, 1 << 20)
        if self._packet_buf is None or self._packet_buf_size < target:
            # Tear down the previous packet so the new buffer can be
            # attached to a fresh one.
            if self._packet:
                _lib.mpp_packet_deinit(byref(self._packet))
                self._packet = MppPacket()
            buf = (ctypes.c_uint8 * target)()
            # Initialise the packet with NULL/0 (matches mpi_dec_test.c's
            # mpp_packet_init(&packet, NULL, 0)). The scratch buffer is
            # owned separately — send_packet() plugs it in via
            # mpp_packet_set_data each call.
            _check(
                _lib.mpp_packet_init(byref(self._packet), None, 0),
                "mpp_packet_init",
            )
            self._packet_buf = buf
            self._packet_buf_size = target
        return self._packet_buf

    def send_packet(
        self,
        data: bytes,
        pts: int = 0,
        eos: bool = False,
    ) -> None:
        """Copy *data* into the decoder's input packet and submit it.

        ``pts`` is in the demuxer's timescale (whatever you want — it just
        gets echoed back on :attr:`DecodedFrame.pts`). ``eos`` flags the
        packet as end-of-stream, which triggers a final flush of pending
        frames on the next :meth:`recv_frame` calls.

        Retries automatically on ``MPP_ERR_BUFFER_FULL``: when the decoder's
        packet queue is saturated we drain every available frame and try
        again. Bounded by ``max_retries`` (default 32) so a pathological
        caller can't get stuck here forever.
        """
        if self._closed:
            raise RuntimeError("MppDecoder is closed")
        size = len(data)
        with _mpp_lock:
            buf = self._ensure_packet_buf(size)
            ctypes.memmove(buf, data, size)
            _lib.mpp_packet_set_data(self._packet, buf)
            _lib.mpp_packet_set_size(self._packet, size)
            _lib.mpp_packet_set_pos(self._packet, buf)
            _lib.mpp_packet_set_length(self._packet, size)
            _lib.mpp_packet_set_pts(self._packet, pts)
            if eos:
                _check(_lib.mpp_packet_set_eos(self._packet), "mpp_packet_set_eos")
            ret = self._api.contents.decode_put_packet(self._ctx, self._packet)
            # -1012 = MPP_ERR_BUFFER_FULL. The hardware can't accept more
            # input until at least one frame is pulled by the caller.
            # The C reference's outer loop msleep()s for 1ms between
            # packets and retries. We do the same — *not* drain frames
            # here, because the info-change frame that tells us to set
            # up an external buffer group arrives here too, and stealing
            # it would leave the decoder stuck. The caller's recv_frame
            # loop is what makes progress. We sleep up to ~5s total
            # before raising — long enough to let the VPU's internal
            # thread catch up on a slow cold-start, short enough that
            # a genuinely wedged decoder still surfaces as a clear
            # error.
            import time as _t
            max_retries = 100
            attempts = 0
            backoff_ms = 1
            while ret == -1012 and attempts < max_retries:
                attempts += 1
                _t.sleep(backoff_ms / 1000.0)
                backoff_ms = min(backoff_ms * 2, 50)
                ret = self._api.contents.decode_put_packet(self._ctx, self._packet)
            _check(ret, "decode_put_packet")

    def recv_frame(self, timeout_ms: int = 30) -> Optional[DecodedFrame]:
        """Try to pull one decoded frame out of the decoder.

        Returns ``None`` if no frame is ready within *timeout_ms*. To match
        the C reference loop exactly, call this in a tight loop until it
        returns a frame after each :meth:`send_packet`.
        """
        if self._closed:
            raise RuntimeError("MppDecoder is closed")
        frame = MppFrame()
        with _mpp_lock:
            ret = self._api.contents.decode_get_frame(self._ctx, byref(frame))
            if ret == -8:  # MPP_ERR_TIMEOUT
                return None
            if ret != 0:
                _check(ret, "decode_get_frame")
            if not frame:
                # Successful call, no frame ready yet — caller should retry
                # after submitting more packets (matches the C reference's
                # `if (frame)` check after decode_get_frame returns).
                return None

            # On the first frame the decoder reports an info-change; allocate
            # an external buffer group sized to its needs (matches
            # mpi_dec_test.c lines 130-178).
            if _lib.mpp_frame_get_info_change(frame):
                width = _lib.mpp_frame_get_width(frame)
                height = _lib.mpp_frame_get_height(frame)
                hor_stride = _lib.mpp_frame_get_hor_stride(frame)
                ver_stride = _lib.mpp_frame_get_ver_stride(frame)
                buf_size = _lib.mpp_frame_get_buf_size(frame)

                if self._buf_group:
                    _lib.mpp_buffer_group_put(self._buf_group)
                grp = MppBufferGroup()
                _check(
                    _lib.mpp_buffer_group_get(
                        byref(grp),
                        MppBufferType.DMA_HEAP,
                        MppBufferMode.INTERNAL,
                        b"rkv-mpp",
                        b"info_change",
                    ),
                    "mpp_buffer_group_get",
                )
                # Limit the group so it can only hand out buffers of this size,
                # then prime the allocator with extra_buf_count entries. The
                # upstream test uses the same trick via dec_buf_mgr_setup.
                if hasattr(_lib, "mpp_buffer_group_limit_config"):
                    _check(
                        _lib.mpp_buffer_group_limit_config(
                            grp, buf_size, self._extra_buf_count
                        ),
                        "mpp_buffer_group_limit_config",
                    )
                for _ in range(self._extra_buf_count):
                    buf = MppBuffer()
                    _check(
                        _lib.mpp_buffer_get_with_tag(grp, byref(buf), buf_size, b"rkv-mpp", b"prime"),
                        "mpp_buffer_get prime",
                    )
                    _lib.mpp_buffer_put_with_caller(buf, b"prime")
                _check(
                    self._api.contents.control(
                        self._ctx, MpiCmd.DEC_SET_EXT_BUF_GROUP, grp
                    ),
                    "MPP_DEC_SET_EXT_BUF_GROUP",
                )
                self._buf_group = grp

                _check(
                    self._api.contents.control(
                        self._ctx, MpiCmd.DEC_SET_INFO_CHANGE_READY, None
                    ),
                    "MPP_DEC_SET_INFO_CHANGE_READY",
                )
                _lib.mpp_frame_deinit(byref(frame))
                return None

            # Drop frames the VPU itself doesn't want surfaced:
            # - discard: reference / B-frames the decoder skipped
            # - errinfo: frames that hit a decode error
            # Both come through mpp_frame_get_buffer with a NULL
            # MppBuffer; without this filter we'd fall into the
            # mpp_buffer_get_ptr / _fd calls below and the C side
            # would log "invalid NULL input from rkv-mpp" warnings.
            # Matches mpi_dec_test.c's discard/errinfo handling.
            if _lib.mpp_frame_get_discard(frame) or _lib.mpp_frame_get_errinfo(frame):
                _lib.mpp_frame_deinit(byref(frame))
                return None

            buf = _lib.mpp_frame_get_buffer(frame)
            # Defensive: even after the discard / errinfo filter, some
            # internal frames (e.g. placeholders near EOS) come back
            # with no buffer attached. mpp_buffer_get_ptr_with_caller
            # / _fd both log an error on NULL — bail out cleanly
            # instead of spamming the log.
            if not buf:
                _lib.mpp_frame_deinit(byref(frame))
                return None
            ptr = _lib.mpp_buffer_get_ptr_with_caller(buf, b"rkv-mpp")
            # mpp_buffer_get_fd returns -1 when the buffer has no fd
            # (e.g. a NORMAL malloc-backed buffer, not a DMA one). The
            # info_change path we set up uses DMA_HEAP so this normally
            # returns a real fd, but be defensive either way.
            fd = _lib.mpp_buffer_get_fd_with_caller(buf, b"rkv-mpp")
            out = DecodedFrame(
                width=_lib.mpp_frame_get_width(frame),
                height=_lib.mpp_frame_get_height(frame),
                hor_stride=_lib.mpp_frame_get_hor_stride(frame),
                ver_stride=_lib.mpp_frame_get_ver_stride(frame),
                fmt=_lib.mpp_frame_get_fmt(frame),
                pts=_lib.mpp_frame_get_pts(frame),
                ptr=ptr or 0,
                fd=fd if fd is not None else -1,
                buf_size=_lib.mpp_frame_get_buf_size(frame),
                raw=frame,
            )
        return out

    def release_frame(self, frame: DecodedFrame) -> None:
        """Return a frame received via :meth:`recv_frame` to MPP.

        The frame's MppBuffer reference count is dropped here; without this
        the internal buffer group will eventually refuse new frames.
        """
        if frame.raw:
            with _mpp_lock:
                _lib.mpp_frame_deinit(byref(frame.raw))
            frame.raw = MppFrame()  # type: ignore[assignment]

    def reset(self) -> None:
        with _mpp_lock:
            if self._ctx:
                _check(self._api.contents.reset(self._ctx), "mpi->reset")


# ---------------------------------------------------------------------------
# MppEncoder — hardware H.264 / HEVC encoder wrapper.
#
# Mirrors the structure of the upstream ``mpi_enc_test.c`` reference. Each
# ``encode(bgr)`` call wraps the input BGR ndarray in an MppFrame, hands it
# to MPP via ``encode_put_frame``, and yields back the encoded NAL packet(s)
# via ``encode_get_packet``. The raw NAL bytes are produced in Annex B
# format (start-code prefixed) which downstream container muxers can
# convert to avcC if needed.
# ---------------------------------------------------------------------------


# Rate-control modes from rk_venc_rc.h (MppEncRcMode_e). The C enum
# is positional, starting at VBR=0:
#   MPP_ENC_RC_MODE_VBR   = 0
#   MPP_ENC_RC_MODE_CBR   = 1
#   MPP_ENC_RC_MODE_FIXQP = 2
#   MPP_ENC_RC_MODE_AVBR  = 3
#   MPP_ENC_RC_MODE_SMTRC = 4
# Sending the wrong value here makes the rc preset bail out with
# fps_in_denom / fps_out_denom assertions and abort() the process.
class MppEncRcMode:
    VBR = 0       # variable bitrate
    CBR = 1       # constant bitrate
    FIXQP = 2     # constant QP (no rate control)
    AVBR = 3      # adaptive VBR
    SMTRC = 4     # smart rc


# Profile codes from rk_venc_cmd.h / avcC's ``profile_idc`` byte.
_H264_PROFILE_BY_NAME = {
    "baseline": 66,
    "main": 77,
    "high": 100,
}


@dataclass
class EncodedPacket:
    """One encoded access unit returned by :meth:`MppEncoder.encode`.

    ``data`` is the raw NAL bytes in Annex B format (one or more
    ``00 00 00 01``-prefixed NAL units — MPP typically packs the slice
    data after the slice header). ``is_keyframe`` is ``True`` for I/P-
    frames emitted after a GOP boundary or explicit ``force_idr``; both
    mp4 muxer and pyrkaiweb WS sink use it to mark keyframe entries
    so consumers can seek into them.
    """

    data: bytes
    pts: int
    is_keyframe: bool


class MppEncoder:
    """Thin ctypes wrapper around MPP's hardware video encoder.

    Mirrors the ``MppDecoder`` API: ``__init__`` configures the encoder
    via the kcfg-based ``MPP_ENC_SET_CFG`` path (matches the upstream
    ``mpi_enc_test.c`` reference's newer key-value config interface
    rather than the legacy ``MppEnc*Cfg`` structs). ``encode(bgr)``
    takes a ``(H, W, 3)`` uint8 BGR ndarray — the encoder's input
    format is fixed at :data:`MppFrameFormat.BGR888`, which avoids a
    CPU-side NV12 conversion step before each encode call.

    Parameters
    ----------
    coding_type:
        :data:`MppCodingType.AVC` (H.264, default), :data:`MppCodingType.HEVC`,
        or :data:`MppCodingType.MJPEG` (Motion JPEG). H.264 is what most
        downstream tools (MP4 muxers, browsers, pyrkaiweb ingest) expect;
        MJPEG is a hardware path for the ``codec="mjpeg"`` branch of
        :class:`rkv.VideoWriter` — each frame is encoded as an
        independent JPEG (no inter-frame coding, no SPS/PPS), which
        makes it ideal for stills capture and AVI streaming.
    width, height:
        Frame dimensions in pixels. The encoder allocates input /
        output buffers sized for these on the first frame.
    fps:
        Output frame rate (Hz). Encoded into both the rate-control config
        (``rc:fps_*``) and the ``video stream`` time base PyAV sees on
        mux.
    bitrate:
        Target bitrate in bps. Surfaces as both ``rc:bps_target`` and
        the upper bound ``rc:bps_max`` so CBR keeps a tight cap.
        Ignored when ``coding_type == MppCodingType.MJPEG``.
    gop:
        GOP length in frames. ``1`` = all-IDR (lowest latency, biggest
        file); ``30`` ≈ 1 second of keyframes at 30 fps. MJPEG ignores
        this — every frame is a keyframe.
    profile:
        H.264 only. ``"baseline"``, ``"main"``, or ``"high"``. HEVC's /
        MJPEG's profile argument is currently ignored.
    input_format:
        Pixel format of the ndarray handed to :meth:`encode`. One of
        the planar YUV / packed RGB formats MPP's venc accepts:

        * :data:`MppFrameFormat.YUV420SP` — NV12, shape
          ``(H * 3 // 2, W)``, ``uint8``. Matches the MPP decoder's
          default output — the zero-copy
          ``cap.read0().writer.write()`` path lands here. Also the
          most common input for MJPEG.
        * :data:`MppFrameFormat.YUV420P` — I420, same shape.
        * :data:`MppFrameFormat.YUV420SP_VU` — NV21, same shape.
        * :data:`MppFrameFormat.RGB888` / :data:`MppFrameFormat.BGR888` —
          shape ``(H, W, 3)``, ``uint8``.
        * :data:`MppFrameFormat.RGBA` / :data:`MppFrameFormat.BGRA` —
          shape ``(H, W, 4)``, ``uint8``.

        The encoder will validate shape / dtype at :meth:`encode` time
        and raise ``ValueError`` on mismatch — it does not perform any
        BGR↔YUV conversion.
    rc_mode:
        ``"cbr"`` (default), ``"vbr"``, ``"fixqp"``, or ``"avbr"``.
        Passed straight through to ``rc:mode`` — MPP validates against
        the platform's encoder capabilities. For MJPEG this is forced
        to ``"fixqp"`` since MJPEG has no rate control beyond the JPEG
        quantisation tables.
    q_factor:
        JPEG quality factor in 1..99, default ``80``. Higher = better
        quality, larger file. Only takes effect when
        ``coding_type == MppCodingType.MJPEG``; passing a non-default
        value with H.264 / HEVC raises ``ValueError``.
    """

    # input_format → (bpp, channels, layout) — used to size the
    # scratch buffer and to validate the ndarray shape in
    # :meth:`encode`. Layout is ``"HWC"`` for packed pixels,
    # ``"YUV"`` for planar/semi-planar.
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
        if coding_type == MppCodingType.AVC and profile not in _H264_PROFILE_BY_NAME:
            raise ValueError(
                f"unsupported H.264 profile {profile!r}; expected one of "
                f"{sorted(_H264_PROFILE_BY_NAME)}"
            )
        if rc_mode not in ("cbr", "vbr", "fixqp", "avbr"):
            raise ValueError(f"unsupported rc_mode: {rc_mode!r}")
        if input_format not in self._FORMAT_LAYOUT:
            raise ValueError(
                f"unsupported input_format {input_format!r}; expected one of "
                f"{sorted(self._FORMAT_LAYOUT)}"
            )
        if coding_type == MppCodingType.MJPEG:
            # MJPEG ignores ``profile`` / ``bitrate`` / ``rc_mode`` —
            # it has no inter-frame coding so there's no rate control
            # in the H.264 sense; quality is controlled solely by the
            # JPEG ``q_factor`` (1..99). Force ``fixqp`` to match the
            # upstream ``mpi_enc_utils.c`` default and don't write the
            # rc:bps_* / rc:fps_* keys — they only confuse the JPEG rc
            # implementation.
            if not 1 <= q_factor <= 99:
                raise ValueError(
                    f"q_factor must be in 1..99 for MJPEG, got {q_factor!r}"
                )
            rc_mode = "fixqp"
            profile = ""
        elif q_factor != 80:
            raise ValueError(
                "q_factor only applies to MJPEG encoding; ignored otherwise"
            )

        self._coding = coding_type
        self._width = int(width)
        self._height = int(height)
        self._fps = int(fps)
        self._bitrate = int(bitrate)
        self._gop = int(gop)
        self._profile_name = profile
        self._profile_idc = _H264_PROFILE_BY_NAME.get(profile, 66) if coding_type == MppCodingType.AVC else 0
        self._rc_mode_name = rc_mode
        self._q_factor = int(q_factor)
        self._input_format = input_format
        # ``True`` once ``self._packet`` has been bound to a backing
        # buffer; the MJPEG path keeps it bound across calls (so it
        # sits inside the encoder's meta and is updated in place),
        # while the H.264 path rebinds every :meth:`encode` call.
        self._packet_bound = False
        self._hor_stride = self._width
        self._ver_stride = self._height
        # Bytes per frame in the configured input format. Planar YUV
        # uses height × stride × bpp; packed pixels use the channel count.
        if input_format in self._YUV_HEIGHT_MULTIPLIER:
            self._frame_buf_size = int(
                self._hor_stride * self._ver_stride * self._YUV_HEIGHT_MULTIPLIER[input_format]
            )
        else:
            self._frame_buf_size = int(self._hor_stride * self._ver_stride * 3)

        # MPP encoder state.
        self._ctx: Optional[MppCtx] = None
        self._api: Optional[POINTER(MppApi)] = None
        self._buf_group: Optional[MppBufferGroup] = None
        self._frm_buf: Optional[MppBuffer] = None
        # A dedicated MppBuffer used as the scratch storage for the
        # output packet that backs ``MPP_ENC_GET_HDR_SYNC`` (and is
        # reused for every :meth:`encode` output packet). MPP writes
        # encoded NAL bytes into this buffer via the MppPacket's
        # ``pos``/``length`` fields.
        self._pkt_buf: Optional[MppBuffer] = None
        # Raw SPS/PPS / VPS bytes returned by MPP_ENC_GET_HDR_SYNC. Annex B
        # format (start-code prefixed). Downstream code can either pass
        # these through as-is (for live ingest / rkv+ws://) or convert to
        # an avcC box (for MP4 muxing).
        self._extradata: bytes = b""
        # Lazily-created MppFrame scratch + output packet scratch.
        self._frame: MppFrame = MppFrame()
        self._packet: MppPacket = MppPacket()
        self._pts: int = 0
        self._force_idr: bool = False
        self._closed = False
        self._flushed = False

        # Configure the encoder in the locked section — mpp_create,
        # mpp_init, and MPP_ENC_SET_CFG all touch global encoder state.
        with _mpp_lock:
            self._ctx = MppCtx()
            self._api = POINTER(MppApi)()
            _check(_lib.mpp_create(byref(self._ctx), byref(self._api)), "mpp_create")
            _check(
                _lib.mpp_init(self._ctx, MppCtxType.ENC, self._coding),
                "mpp_init",
            )

            # Set the encoder config via the key-value interface. Missing
            # keys are silently ignored at the MPP level, so we set only
            # what we care about.
            cfg = c_void_p()
            _check(_lib.mpp_enc_cfg_init(byref(cfg)), "mpp_enc_cfg_init")
            try:
                _check(
                    self._api.contents.control(
                        self._ctx, MpiCmd.ENC_GET_CFG, cfg
                    ),
                    "MPP_ENC_GET_CFG",
                )
                _cfg = _lib.mpp_enc_cfg_set_s32
                _cfg(cfg, b"prep:width", self._width)
                _cfg(cfg, b"prep:height", self._height)
                _cfg(cfg, b"prep:hor_stride", self._hor_stride)
                _cfg(cfg, b"prep:ver_stride", self._ver_stride)
                _cfg(cfg, b"prep:format", int(self._input_format))
                _cfg(cfg, b"prep:range", 2)  # 2 = full-range; matches CV output
                if coding_type == MppCodingType.AVC:
                    _cfg(cfg, b"h264:profile", self._profile_idc)
                    _cfg(cfg, b"h264:level", 41)  # 4.1 covers 1080p@30
                if coding_type == MppCodingType.MJPEG:
                    # MJPEG has no rate control in the H.264 sense;
                    # ``q_factor`` (1..99) controls the JPEG quantisation
                    # tables. ``qf_max`` / ``qf_min`` bound the rate-control
                    # model's quant range; the upstream ``mpi_enc_utils.c``
                    # defaults are 99 / 1. Keep them so callers can override
                    # via ``q_factor`` only and still get sane behaviour.
                    _cfg(cfg, b"jpeg:q_factor", self._q_factor)
                    _cfg(cfg, b"jpeg:qf_max", 99)
                    _cfg(cfg, b"jpeg:qf_min", 1)
                _cfg(cfg, b"rc:mode", getattr(MppEncRcMode, rc_mode.upper()))
                # Even in FIXQP mode the JPEG rc model wants a non-zero
                # bit budget so it can derive quantisation tables. The
                # upstream ``mpi_enc_utils.c`` sets ``bps_target = w*h*fps*1.5``
                # (a sanity 8-bit RGB multiplier) for FIXQP and leaves
                # bps_min / bps_max at zero — match that. Without
                # bps_target the JPEG HAL falls back to all-ones
                # quantisation tables, which produces a syntactically
                # valid but visually-broken JPEG that downstream
                # decoders reject.
                if coding_type == MppCodingType.MJPEG:
                    # ``w*h*fps/8`` matches the upstream
                    # ``mpi_enc_utils.c`` default for FIXQP MJPEG (a
                    # 0.125-byte-per-pixel budget) and gives the JPEG
                    # rc model the right input to derive quantisation
                    # tables. Higher values push qtables toward 1
                    # (worse quality, smaller file).
                    fixqp_bps = max(
                        self._width * self._height * self._fps // 8,
                        100_000,
                    )
                    _cfg(cfg, b"rc:bps_target", fixqp_bps)
                    _cfg(cfg, b"rc:bps_max", 0)
                    _cfg(cfg, b"rc:bps_min", 0)
                else:
                    _cfg(cfg, b"rc:bps_target", self._bitrate)
                    _cfg(cfg, b"rc:bps_max", self._bitrate)
                    _cfg(cfg, b"rc:bps_min", self._bitrate // 4)
                _cfg(cfg, b"rc:fps_in_num", self._fps)
                _cfg(cfg, b"rc:fps_in_denom", 1)
                _cfg(cfg, b"rc:fps_in_flex", 0)
                _cfg(cfg, b"rc:fps_out_num", self._fps)
                _cfg(cfg, b"rc:fps_out_denom", 1)
                _cfg(cfg, b"rc:fps_out_flex", 0)
                _cfg(cfg, b"rc:gop", self._gop)
                # Drop-frame knobs and rc tuning. Drop-mode is disabled by default;
                # super_mode / max_reenc_times are forced off too. These
                # are belt-and-braces — the critical missing call is
                # ``MPP_ENC_SET_RC_API_CURRENT`` below; without it the rc
                # ctx has no implementation bound and the per-frame paths
                # assert fps_*_denom even when our kcfg values are correct.
                _cfg(cfg, b"rc:drop_mode", 0)        # MPP_ENC_RC_DROP_FRM_DISABLED
                _cfg(cfg, b"rc:max_reenc_times", 0)
                _cfg(cfg, b"rc:super_mode", 0)
                _cfg(cfg, b"rc:drop_thd", 20)
                _cfg(cfg, b"rc:drop_gap", 1)
                _check(
                    self._api.contents.control(
                        self._ctx, MpiCmd.ENC_SET_CFG, cfg
                    ),
                    "MPP_ENC_SET_CFG",
                )

                # Bind an rc implementation for AVC / HEVC / MJPEG. Without
                # this the rc context has no implementation bound and
                # per-frame paths assert ``fps_*_denom >= 1`` even when
                # our kcfg is correct. ``name = b"default"`` selects the
                # stock RC; ``smart`` is opt-in via cmd-line flag. For
                # MJPEG the jpege RC implementation is the one that
                # actually reads ``jpeg:q_factor`` to drive the quant
                # tables — without this call the encoder produces garbage
                # / refuses frames.
                if coding_type in (
                    MppCodingType.AVC, MppCodingType.HEVC, MppCodingType.MJPEG
                ):
                    name_buf = b"default\0"
                    brief = RcApiBrief()
                    brief.name = c_char_p(name_buf)
                    brief.type = c_int32(coding_type)
                    _check(
                        self._api.contents.control(
                            self._ctx, MpiCmd.ENC_SET_RC_API_CURRENT,
                            byref(brief),
                        ),
                        "MPP_ENC_SET_RC_API_CURRENT",
                    )

                # Switch header emission to ``MPP_ENC_HEADER_MODE_EACH_IDR``
                # so every IDR access unit arrives with SPS + PPS NALs
                # prepended to its bitstream payload. Default mode only
                # attaches SPS/PPS to the very first frame and shoves
                # them into a separate avcC extradata packet; with the
                # default, a downstream decoder that joined mid-stream
                # never sees the SPS/PPS until the next IDR (which
                # might be gop_len frames away) and renders black until
                # then. EACH_IDR makes every keyframe self-contained.
                # MJPEG has no SPS/PPS, so the call is a no-op there but
                # we still skip it for clarity.
                if coding_type in (MppCodingType.AVC, MppCodingType.HEVC):
                    hdr_mode = c_int32(1)        # MPP_ENC_HEADER_MODE_EACH_IDR
                    _check(
                        self._api.contents.control(
                            self._ctx, MpiCmd.ENC_SET_HEADER_MODE,
                            byref(hdr_mode),
                        ),
                        "MPP_ENC_SET_HEADER_MODE",
                    )
            finally:
                _lib.mpp_enc_cfg_deinit(cfg)

            # Allocate one input frame buffer (reused per encode). DRM+CACHABLE
            # buffers are what the upstream ``mpi_enc_test.c`` uses and
            # what the VPU expects on RK3588 / RK3568 / etc.; the kernel
            # ``drm`` driver backs them with contiguous CMA memory that
            # the encoder's DMA path can consume without a CPU-side
            # bounce buffer. NORMAL buffers go through ``allocator_std``
            # which recent MPP versions mark simulation-only.
            self._buf_group = MppBufferGroup()
            _check(
                _lib.mpp_buffer_group_get(
                    byref(self._buf_group),
                    MppBufferType.DRM | MppBufferType.FLAG_CACHABLE,
                    MppBufferMode.INTERNAL,
                    b"rkv-mpp",
                    b"rkv-encoder-frm",
                ),
                "mpp_buffer_group_get",
            )
            self._frm_buf = MppBuffer()
            _check(
                _lib.mpp_buffer_get_with_tag(
                    self._buf_group,
                    byref(self._frm_buf),
                    self._frame_buf_size,
                    b"rkv-mpp",
                    b"rkv-encoder-frm",
                ),
                "mpp_buffer_get_with_tag frame buffer",
            )
            # Output NAL storage. Sized to comfortably hold one coded
            # access unit at the configured bitrate (peak NAL ~ 64 KiB
            # for a 1080p frame at 60 Mbps — be generous).
            self._pkt_buf = MppBuffer()
            _check(
                _lib.mpp_buffer_get_with_tag(
                    self._buf_group,
                    byref(self._pkt_buf),
                    256 * 1024,
                    b"rkv-mpp",
                    b"rkv-encoder-pkt",
                ),
                "mpp_buffer_get_with_tag packet buffer",
            )

            # Pull SPS/PPS / VPS out of the encoder so callers can stash
            # them as the codec's extradata. MPP writes the whole NAL
            # blob into the packet — Annex B start codes included.
            # The packet must be backed by an explicit MppBuffer; the
            # bare ``mpp_packet_init(NULL, 0)`` form segfaults on this
            # MPP version, so we reuse the encoder's output scratch
            # buffer.
            tmp_pkt = MppPacket()
            _check(
                _lib.mpp_packet_init_with_buffer(byref(tmp_pkt), self._pkt_buf),
                "mpp_packet_init_with_buffer (HDR_SYNC)",
            )
            try:
                _lib.mpp_packet_set_length(tmp_pkt, 0)
                ret = self._api.contents.control(
                    self._ctx, MpiCmd.ENC_GET_HDR_SYNC, tmp_pkt
                )
                if coding_type == MppCodingType.MJPEG:
                    # MJPEG has no SPS/PPS / VPS — every frame is a
                    # self-contained JPEG. HDR_SYNC returns an error on
                    # this path (no header to fetch), so swallow it
                    # rather than raising. ``self._extradata`` stays
                    # ``b""`` which is what callers should expect.
                    pass
                elif ret != 0:
                    raise RuntimeError(
                        f"MPP_ENC_GET_HDR_SYNC failed: ret={ret}"
                    )
                else:
                    pos = _lib.mpp_packet_get_pos(tmp_pkt)
                    length = _lib.mpp_packet_get_length(tmp_pkt)
                    if pos and length:
                        self._extradata = ctypes.string_at(pos, length)
            finally:
                _lib.mpp_packet_deinit(byref(tmp_pkt))

    # -- public properties ---------------------------------------------------

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def fps(self) -> int:
        return self._fps

    @property
    def bitrate(self) -> int:
        return self._bitrate

    @property
    def gop(self) -> int:
        return self._gop

    @property
    def profile(self) -> str:
        return self._profile_name

    @property
    def rc_mode(self) -> str:
        return self._rc_mode_name

    @property
    def q_factor(self) -> int:
        """JPEG quality factor (1..99) used by the MJPEG encoder.

        Higher == better quality, larger file. Ignored when ``coding_type``
        is H.264 / HEVC — those derive quantisation from ``rc_mode``.
        """
        return self._q_factor

    @property
    def coding_type(self) -> int:
        return self._coding

    @property
    def extradata(self) -> bytes:
        """SPS / PPS / VPS blob in Annex B format (``00 00 00 01`` start codes).

        MP4 muxers normally want this as an avcC box; downstream
        :class:`VideoWriter` does that conversion via the
        ``_annexb_to_avcC`` helper.
        """
        return self._extradata

    def force_keyframe(self) -> None:
        """Make the next ``encode()`` call an IDR frame (keyframe)."""
        self._force_idr = True

    # -- context manager ----------------------------------------------------

    def __enter__(self) -> "MppEncoder":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with _mpp_lock:
            if self._packet_bound and self._packet:
                # The MJPEG path keeps one packet handle bound to
                # ``self._pkt_buf`` across ``encode`` calls — release
                # it before the buffer itself goes away.
                _lib.mpp_packet_deinit(byref(self._packet))
                self._packet = MppPacket()
                self._packet_bound = False
            if self._frm_buf:
                _lib.mpp_buffer_put_with_caller(self._frm_buf, b"rkv-mpp")
                self._frm_buf = MppBuffer()
            if self._pkt_buf:
                _lib.mpp_buffer_put_with_caller(self._pkt_buf, b"rkv-mpp")
                self._pkt_buf = MppBuffer()
            if self._buf_group:
                _lib.mpp_buffer_group_put(self._buf_group)
                self._buf_group = MppBufferGroup()
            if self._ctx:
                _lib.mpp_destroy(self._ctx)
                self._ctx = MppCtx()

    # -- encode entry point -------------------------------------------------

    def encode(self, frame: "np.ndarray") -> EncodedPacket:
        """Submit one ndarray to the encoder and block until the
        matching encoded packet is available.

        The expected shape depends on ``input_format``:

        * NV12 / NV21 / I420 — ``(H * 3 // 2, W)`` ``uint8`` (Y plane
          + UV plane, either interleaved or stacked).
        * RGB / BGR — ``(H, W, 3)`` ``uint8``.
        * RGBA / BGRA — ``(H, W, 4)`` ``uint8``.

        The dtype / layout is validated; an asynchronous MPP error
        (typically BUFFER_FULL on the input queue) raises ``RuntimeError``.

        The encoder is single-threaded and synchronous: one ``encode``
        call in, one :class:`EncodedPacket` out. If you want to keep
        the encoder busy across multiple frame producers, serialise
        ``encode`` calls yourself with a lock.
        """
        import numpy as np

        if self._closed:
            raise RuntimeError("MppEncoder is closed")
        if self._flushed:
            raise RuntimeError("MppEncoder is already flushed")

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
                f"frame must be {expected_shape} {np.dtype(expected_dtype).name}, input_format "
                f"{self._input_format}, got shape={frame.shape} dtype={frame.dtype.name}"
            )
        if not frame.flags.c_contiguous:
            frame = np.ascontiguousarray(frame)

        with _mpp_lock:
            # Optionally request an IDR for this frame.
            if self._force_idr:
                _check(
                    self._api.contents.control(
                        self._ctx, MpiCmd.ENC_SET_IDR_FRAME, None
                    ),
                    "MPP_ENC_SET_IDR_FRAME",
                )
                self._force_idr = False

            # Wrap the BGR data in an MppFrame that points at our
            # pre-allocated NORMAL buffer.
            buf_ptr = _lib.mpp_buffer_get_ptr_with_caller(
                self._frm_buf, b"rkv-mpp"
            )
            if not buf_ptr:
                raise RuntimeError("MppEncoder: failed to get buffer pointer")
            ctypes.memmove(buf_ptr, frame.ctypes.data, self._frame_buf_size)
            # CPU wrote to the buffer — flush so DMA can read it
            # coherently. NORMAL buffers are CPU-malloc-backed, so this
            # is a plain cache flush (no DMA cache to worry about).
            try:
                mpp_buffer_sync_cpu_write(self._frm_buf)
            except RuntimeError:
                pass  # some lib versions emit "not implemented" here

            if self._coding == MppCodingType.MJPEG:
                # MJPEG needs the output packet pre-attached to the
                # frame via MppMeta (KEY_OUTPUT_PACKET) — without it
                # the JPEG HAL falls back to asking the buffer group
                # for a packet, which reports size 0 on this firmware
                # and trips ``mpp_enc_check_pkt_buf``. The H.264 path
                # doesn't need this; the direct ``encode_get_packet``
                # API works for it.
                data, pts_out, is_keyframe = self._encode_put_get_mjpeg()
            else:
                data, pts_out, is_keyframe = self._encode_put_get_h264()

            self._pts += 1

        return EncodedPacket(data=data, pts=pts_out, is_keyframe=is_keyframe)

    def _encode_put_get_h264(self):
        """H.264 encode path: put frame, get packet directly.

        Returns ``(data, pts, is_keyframe)`` so the caller can wrap
        them in an :class:`EncodedPacket` once outside the lock.
        """
        _lib.mpp_frame_init(byref(self._frame))
        try:
            _lib.mpp_frame_set_width(self._frame, self._width)
            _lib.mpp_frame_set_height(self._frame, self._height)
            _lib.mpp_frame_set_hor_stride(self._frame, self._hor_stride)
            _lib.mpp_frame_set_ver_stride(self._frame, self._ver_stride)
            _lib.mpp_frame_set_fmt(self._frame, int(self._input_format))
            _lib.mpp_frame_set_pts(self._frame, self._pts)
            _lib.mpp_frame_set_buffer(self._frame, self._frm_buf)
            ret = self._api.contents.encode_put_frame(self._ctx, self._frame)
            if ret != 0:
                raise RuntimeError(f"encode_put_frame failed: ret={ret}")
        finally:
            _lib.mpp_frame_deinit(byref(self._frame))
            self._frame = MppFrame()

        _check(
            _lib.mpp_packet_init_with_buffer(byref(self._packet), self._pkt_buf),
            "mpp_packet_init_with_buffer (output)",
        )
        self._packet_bound = True
        try:
            _lib.mpp_packet_set_length(self._packet, 0)
            ret = self._api.contents.encode_get_packet(
                self._ctx, byref(self._packet)
            )
            if ret != 0:
                raise RuntimeError(f"encode_get_packet failed: ret={ret}")
            pos = _lib.mpp_packet_get_pos(self._packet)
            length = _lib.mpp_packet_get_length(self._packet)
            flag = _lib.mpp_packet_get_flag(self._packet)
            data = ctypes.string_at(pos, length) if length else b""
            is_keyframe = bool(flag & MPP_PACKET_FLAG_INTRA)
            pts_out = _lib.mpp_packet_get_pts(self._packet)
        finally:
            _lib.mpp_packet_deinit(byref(self._packet))
            self._packet = MppPacket()
            self._packet_bound = False
        return data, pts_out, is_keyframe

    def _encode_put_get_mjpeg(self):
        """MJPEG encode path: pre-attach output packet via MppMeta.

        The JPEG HAL refuses to fall back to asking the buffer group
        for an output packet (it asserts in
        ``mpp_enc_check_pkt_buf``), so the caller has to provide one
        upfront — see :func:`encode` for the rationale.

        The packet is bound to ``self._pkt_buf`` once (the first time
        this method runs) and then reused across calls; binding it
        every call would thrash MPP's internal refcounts and emit
        ``put_meta invalid negative ref_count -1`` warnings. Returns
        the same ``(data, pts, is_keyframe)`` tuple as the H.264
        helper.
        """
        if not self._packet_bound:
            _check(
                _lib.mpp_packet_init_with_buffer(
                    byref(self._packet), self._pkt_buf
                ),
                "mpp_packet_init_with_buffer (mjpeg output)",
            )
            self._packet_bound = True

        meta = MppMeta()
        ret = _lib.mpp_meta_get_with_tag(
            byref(meta), b"rkv-mpp", b"rkv-encoder-mjpeg"
        )
        if ret != 0 or not meta:
            raise RuntimeError(f"mpp_meta_get failed: ret={ret}")
        try:
            _lib.mpp_packet_set_length(self._packet, 0)
            _check(
                _lib.mpp_meta_set_packet(
                    meta, KEY_OUTPUT_PACKET, self._packet
                ),
                "mpp_meta_set_packet (KEY_OUTPUT_PACKET)",
            )
            _lib.mpp_frame_init(byref(self._frame))
            try:
                _lib.mpp_frame_set_width(self._frame, self._width)
                _lib.mpp_frame_set_height(self._frame, self._height)
                _lib.mpp_frame_set_hor_stride(self._frame, self._hor_stride)
                _lib.mpp_frame_set_ver_stride(self._frame, self._ver_stride)
                _lib.mpp_frame_set_fmt(self._frame, int(self._input_format))
                _lib.mpp_frame_set_pts(self._frame, self._pts)
                _lib.mpp_frame_set_buffer(self._frame, self._frm_buf)
                _lib.mpp_frame_set_meta(self._frame, meta)
                ret = self._api.contents.encode_put_frame(
                    self._ctx, self._frame
                )
                if ret != 0:
                    raise RuntimeError(
                        f"encode_put_frame (mjpeg) failed: ret={ret}"
                    )
            finally:
                _lib.mpp_frame_deinit(byref(self._frame))
                self._frame = MppFrame()

            # The same packet handle's ``pos``/``length`` fields get
            # updated by the encoder — that's what upstream
            # ``mpi_enc_test.c`` does too. ``mpp_meta_put`` releases
            # its borrow on the packet; the underlying buffer
            # (``self._pkt_buf``) and the packet itself stay alive
            # until :meth:`close` deinits them.
            pos = _lib.mpp_packet_get_pos(self._packet)
            length = _lib.mpp_packet_get_length(self._packet)
            flag = _lib.mpp_packet_get_flag(self._packet)
            data = ctypes.string_at(pos, length) if length else b""
            is_keyframe = bool(flag & MPP_PACKET_FLAG_INTRA)
            pts_out = _lib.mpp_packet_get_pts(self._packet)
        finally:
            _lib.mpp_meta_put(meta)

        return data, pts_out, is_keyframe

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
        """Encode one externally-owned DMA-BUF frame, **zero-copy**.

        Imports the given ``fd`` as an :class:`MppBuffer` via
        ``mpp_buffer_import_with_tag`` so MPP can read from the
        caller-owned DMA-BUF directly — no CPU-side memmove, no
        malloc scratch buffer. The caller is responsible for keeping
        the DMA-BUF alive until encoding completes; the
        :class:`MppBuffer` returned by the import takes a reference
        on the underlying fd, so as long as the imported buffer
        isn't ``put``-back too early, the encoder sees valid data.

        The ``width`` / ``height`` / ``hor_stride`` / ``ver_stride``
        / ``input_format`` arguments describe the buffer's layout
        (the DMA-BUF might be larger than the visible frame, so
        strides are mandatory). ``input_format`` defaults to NV12
        (``MppFrameFormat.YUV420SP``) — the output format of the
        MPP decoder, which is what the typical ``read0() →
        write(frame0) → encode_external_fd`` pipeline produces.

        The encoder's *output* format is H.264 by default; the
        input format and the output format are independent.
        """
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

        with _mpp_lock:
            # Optionally request an IDR for this frame.
            if self._force_idr:
                _check(
                    self._api.contents.control(
                        self._ctx, MpiCmd.ENC_SET_IDR_FRAME, None
                    ),
                    "MPP_ENC_SET_IDR_FRAME",
                )
                self._force_idr = False

            # Import the external DMA-BUF fd as an MppBuffer. We pass
            # ``group = NULL`` so the imported buffer attaches to the
            # encoder's default internal group rather than the
            # encoder's pre-allocated NORMAL one. This lets MPP hand
            # the same DMA-BUF to the encoder via the kernel's
            # ``dma_buf`` path without any CPU round-trip.
            info = MppBufferInfo()
            info.fd = fd
            info.size = hor_stride * ver_stride * 3 // 2  # NV12 worst case
            info.type = MppBufferType.DMA_HEAP
            ext_buf = MppBuffer()
            _check(
                _lib.mpp_buffer_import_with_tag(
                    None, byref(info), byref(ext_buf),
                    b"rkv-mpp", b"rkv-encoder-ext",
                ),
                "mpp_buffer_import_with_tag",
            )
            try:
                _lib.mpp_frame_init(byref(self._frame))
                try:
                    _lib.mpp_frame_set_width(self._frame, width)
                    _lib.mpp_frame_set_height(self._frame, height)
                    _lib.mpp_frame_set_hor_stride(self._frame, hor_stride)
                    _lib.mpp_frame_set_ver_stride(self._frame, ver_stride)
                    _lib.mpp_frame_set_fmt(self._frame, int(input_format))
                    _lib.mpp_frame_set_pts(self._frame, pts)
                    _lib.mpp_frame_set_buffer(self._frame, ext_buf)
                    ret = self._api.contents.encode_put_frame(
                        self._ctx, self._frame
                    )
                    if ret != 0:
                        raise RuntimeError(
                            f"encode_put_frame (external) failed: ret={ret}"
                        )
                finally:
                    _lib.mpp_frame_deinit(byref(self._frame))
                    self._frame = MppFrame()

                # Drain the encoded packet (same path as the eager
                # ``encode`` method).
                _check(
                    _lib.mpp_packet_init_with_buffer(
                        byref(self._packet), self._pkt_buf
                    ),
                    "mpp_packet_init_with_buffer (external output)",
                )
                try:
                    _lib.mpp_packet_set_length(self._packet, 0)
                    ret = self._api.contents.encode_get_packet(
                        self._ctx, byref(self._packet)
                    )
                    if ret != 0:
                        raise RuntimeError(
                            f"encode_get_packet (external) failed: ret={ret}"
                        )
                    pos = _lib.mpp_packet_get_pos(self._packet)
                    length = _lib.mpp_packet_get_length(self._packet)
                    flag = _lib.mpp_packet_get_flag(self._packet)
                    data = ctypes.string_at(pos, length) if length else b""
                    is_keyframe = bool(flag & MPP_PACKET_FLAG_INTRA)
                    pts_out = _lib.mpp_packet_get_pts(self._packet)
                finally:
                    _lib.mpp_packet_deinit(byref(self._packet))
                    self._packet = MppPacket()
            finally:
                # Release the imported MppBuffer. The underlying DMA-BUF
                # is unaffected (the kernel keeps a ref via the fd).
                _lib.mpp_buffer_put_with_caller(ext_buf, b"rkv-mpp")

        return EncodedPacket(data=data, pts=pts_out, is_keyframe=is_keyframe)

    def flush(self) -> Optional[EncodedPacket]:
        """Send an EOS frame to the encoder and pull the final packet.

        Returns the last :class:`EncodedPacket` produced before EOS, or
        ``None`` if the encoder had nothing queued. After this returns
        the encoder cannot encode any more frames.
        """
        if self._closed:
            raise RuntimeError("MppEncoder is closed")
        if self._flushed:
            return None
        self._flushed = True
        with _mpp_lock:
            _lib.mpp_frame_init(byref(self._frame))
            try:
                _lib.mpp_frame_set_eos(self._frame)
                ret = self._api.contents.encode_put_frame(self._ctx, self._frame)
                if ret != 0:
                    raise RuntimeError(f"encode_put_frame (EOS) failed: ret={ret}")
            finally:
                _lib.mpp_frame_deinit(byref(self._frame))
                self._frame = MppFrame()

            ret = self._api.contents.encode_get_packet(
                self._ctx, byref(self._packet)
            )
            if ret != 0:
                return None
            try:
                pos = _lib.mpp_packet_get_pos(self._packet)
                length = _lib.mpp_packet_get_length(self._packet)
                flag = _lib.mpp_packet_get_flag(self._packet)
                data = ctypes.string_at(pos, length) if length else b""
                is_keyframe = bool(flag & MPP_PACKET_FLAG_INTRA)
                pts_out = _lib.mpp_packet_get_pts(self._packet)
            finally:
                _lib.mpp_packet_deinit(byref(self._packet))
                self._packet = MppPacket()
            return EncodedPacket(
                data=data, pts=pts_out, is_keyframe=is_keyframe
            )


# ---------------------------------------------------------------------------
# YUV (NV12) → BGR helper — purely for convenience in test/demo code.
# ---------------------------------------------------------------------------


def nv12_to_bgr(ptr_addr: int, width: int, height: int, hor_stride: int, ver_stride: int):
    """Convert an NV12 frame at *ptr_addr* into a (H, W, 3) BGR ndarray.

    Returns ``None`` if numpy is unavailable — callers that don't need a
    numpy view can use :class:`DecodedFrame` directly. The function does not
    copy on success; the returned array borrows memory from the decoder
    buffer, so the caller must ``release_frame()`` only after they're done
    with the array.
    """
    try:
        import numpy as np
    except ImportError:
        return None

    # NV12 layout: Y plane of stride*ver_stride bytes, then interleaved UV
    # plane of stride*ver_stride/2 bytes. Only the active area (width x
    # height) is meaningful — the rest is decoder padding.
    y_size = hor_stride * ver_stride
    total = y_size + (hor_stride * ver_stride) // 2
    buf = (ctypes.c_uint8 * total).from_address(ptr_addr)
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(ver_stride * 3 // 2, hor_stride)
    bgr_full = _nv12_to_bgr_np(arr[: height * 3 // 2, :width])
    return bgr_full


def _nv12_to_bgr_np(nv12):
    """Vectorised NV12 → BGR888 conversion.

    Lifted from the standard ITU-R BT.601 limited-range matrix. Operates on
    a (H*3/2, W) uint8 view; returns (H, W, 3) uint8.
    """
    import numpy as np

    h, w = nv12.shape[0] * 2 // 3, nv12.shape[1]
    y = nv12[:h, :].astype(np.int16)
    uv = nv12[h : h + h // 2, :].reshape(h // 2, w // 2, 2)
    u = uv[..., 0].astype(np.int16) - 128
    v = uv[..., 1].astype(np.int16) - 128

    # BT.601 limited range (matches what MPP emits for H.264):
    u_up = np.repeat(np.repeat(u, 2, axis=0), 2, axis=1)
    v_up = np.repeat(np.repeat(v, 2, axis=0), 2, axis=1)
    b = np.clip(y + 1.772 * u_up, 0, 255).astype(np.uint8)
    g = np.clip(y - 0.34414 * u_up - 0.71414 * v_up, 0, 255).astype(np.uint8)
    r = np.clip(y + 1.402 * v_up, 0, 255).astype(np.uint8)
    return np.stack([b, g, r], axis=-1)


# ---------------------------------------------------------------------------
# Annex B helpers — needed by anything that feeds raw compressed samples
# into MPP (PyAV, FFmpeg, etc.).
# ---------------------------------------------------------------------------


def avcc_to_annexb(data: bytes) -> bytes:
    """Convert a 4-byte-length-prefixed H.264/HEVC sample to Annex B.

    MP4 stores H.264/HEVC samples in "AVCC" / "HVCC" format — a sequence of
    NAL units, each prefixed with a big-endian 4-byte length. MPP's
    ``split_parse`` looks for the classic ``00 00 00 01`` start code, so we
    have to translate. Returns ``data`` unchanged when it already starts
    with a start code (some demuxers / streams are Annex B natively).
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
            # malformed — bail out and return whatever we have so MPP's
            # parser at least has something to look at.
            out.extend(data[i - 4 :])
            break
        out.extend(b"\x00\x00\x00\x01")
        out.extend(data[i : i + nal_len])
        i += nal_len
    return bytes(out)


def parse_avcc_extradata(extradata: bytes) -> bytes:
    """Return SPS+PPS as an Annex B byte stream from an avcC configuration.

    MP4 stores H.264 configuration in the avcC sample-entry box, not in
    the media samples themselves. MPP needs the SPS/PPS injected via the
    input stream (or the equivalent of what the upstream test does by
    feeding a raw .264 file). We parse the avcC and emit ``00 00 00 01``
    prefixed SPS followed by PPS so callers can prepend it to the first
    packet.
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


__all__ = [
    "MppApi",
    "MppBufferInfo",
    "MppBufferMode",
    "MppBufferType",
    "MppCtxType",
    "MppCodingType",
    "MppDecoder",
    "MppEncoder",
    "MppEncHeaderMode",
    "MppEncRcMode",
    "MppFrameFormat",
    "MpiCmd",
    "CODING_BY_NAME",
    "DecodedFrame",
    "EncodedPacket",
    "avcc_to_annexb",
    "parse_avcc_extradata",
    "nv12_to_bgr",
    "mpp_buffer_sync_cpu_read",
    "mpp_buffer_sync_cpu_write",
]