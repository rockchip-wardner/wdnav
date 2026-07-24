"""VideoWriter — MPP hardware encoder sink (no cv2, no libx264).

Replaces the original cv2.VideoWriter / cv2.imencode based sink. The
encoded byte stream comes from one backend:

  * **MPP** (hardware) — :class:`rkv.mpp.MppEncoder`. The encoder
    runs on the Rockchip VPU and accepts BGR / NV12 / I420 input.
    When the input is a :class:`rkv.mpp.DecodedFrame` (zero-copy
    DMA-BUF straight from :meth:`VideoCapture.read0`) the encoder
    imports the decoder's fd as an :class:`MppBuffer` via
    ``mpp_buffer_import_with_tag`` and feeds it straight to the
    VPU — **no RGA, no Python-side memmove, no numpy intermediate**
    between decode and encode. For BGR ndarray input: one
    ``memmove`` + on-chip BGR→YUV. PyAV's only role is muxing the
    encoded NALs into the requested container (MP4 / MKV / …).

  H.264 is MPP-only — there is no software fallback. Picking
  ``codec="h264"`` on a box where :class:`rkv.mpp.MppEncoder` can't
  initialise (no Rockchip SoC, or an MPP build that rejects the
  requested width / profile) raises ``RuntimeError`` at construction
  time. The non-H.264 codecs (``mp4v`` for local files,
  ``mjpeg`` for ``rkv+ws://``) still go through PyAV's own
  built-in encoders — those don't need libx264.

The ``rkv+ws://`` sink supports two wire formats:

  * ``codec="mjpeg"`` (default) — PyAV's software MJPEG encoder,
    one packet per frame. JPEG is the simplest wire format that
    pyrkaiweb's ingest handler accepts.
  * ``codec="h264"`` — :class:`rkv.mpp.MppEncoder` for VPU H.264,
    laid out as avcC header + 4-byte length-prefixed NAL access
    units (MSE-friendly). Backend selection: ``"ws-h264-mpp"`` on
    Rockchip, ``RuntimeError`` if MPP isn't available.

Public API mirrors OpenCV's ``VideoWriter`` so existing call sites
work unchanged:

    out = VideoWriter("/tmp/out.mp4", fps=30, frame_size=(640, 360),
                      codec="mp4v", gop=30, bitrate=2_000_000)
    out.write(frame)                # BGR (H, W, 3) uint8 ndarray
    out.write(frame0)              # DecodedFrame with .fd (zero-copy)

    out = VideoWriter("rkv+ws://host:9999/source", fps=30,
                      frame_size=(640, 360),
                      codec="h264", gop=30, bitrate=2_000_000)
    # → sends avcC header + length-prefixed H.264 NAL units over WS
    out.write(frame)                # BGR ndarray
    out.write(frame0)              # DecodedFrame (zero-copy DMA-BUF → MPP)
    out.release()

Copy budget by path
===================

``read0() → write(frame0)``:

  | path                    | CPU copies | hw passes |
  |-------------------------|-----------:|-----------:|
  | MPP zero-copy (DMA-BUF) |          0 |  VPU only  |
  | MPP eager BGR           |          1 |  VPU only  |
  | PyAV mp4v / mjpeg      |          1 |  PyAV      |

The MPP zero-copy path is the headline: only the VPU and the encoder
DMA engine touch the pixel data, the CPU never sees it. The single
remaining copy on the eager-BGR branch is the
``memmove`` that pulls the numpy array into MPP's input buffer;
on the zero-copy branch even that is gone because the encoder
accepts NV12 directly without a BGR round-trip.
"""
from __future__ import annotations

import fractions
import os
import socket
import struct
import threading
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    import av
    _AV_IMPORT_ERROR: Optional[Exception] = None
except Exception as e:  # pragma: no cover - import-time fallback only
    av = None  # type: ignore
    _AV_IMPORT_ERROR = e


# Wire-format byte prefix shared with pyrkaiweb's ingest handler:
#   0x00  delta frame
#   0x01  key frame
#   0x02  extradata (avcC for H.264) — sent once
#   0xFF  EOS marker — sent on release()
PKT_TYPE_DELTA = 0x00
PKT_TYPE_KEY = 0x01
PKT_TYPE_HEADER = 0x02
PKT_TYPE_EOS = 0xFF

# Default JPEG quality for the rkv+ws:// sink.
_WS_JPEG_QUALITY = 75

# MPP's MJPEG encoder on the ``cfg-kcfg`` init path (the path we have
# to use because ``mpp_venc_kcfg_init`` isn't exported in this
# librockchip_mpp.so build) accumulates internal state across
# ``encode()`` calls and the input queue fills up after roughly
# ``_MPP_MJPEG_RECYCLE_EVERY`` frames — ``encode_put_frame`` then
# blocks indefinitely on ``BUFFER_FULL``. Recreating the encoder
# clears the queue. The value 6 leaves a 2-frame margin under the
# observed deadlock threshold.
_MPP_MJPEG_RECYCLE_EVERY = 6

# H.264 profile codes for avcC configurationRecord:
#   66  baseline
#   77  main
#   100 high
_H264_PROFILE_BY_NAME = {
    "baseline": 66,
    "main": 77,
    "high": 100,
}


def _import_av():
    if av is None:
        raise RuntimeError(
            "PyAV is required for VideoWriter but couldn't be imported. "
            f"Original error: {_AV_IMPORT_ERROR}"
        )
    return av


@dataclass
class _EncodedChunk:
    data: bytes
    pts_us: int
    is_keyframe: bool


# ---------------------------------------------------------------------------
# Annex B → avcC / length-prefixed NAL helpers
# ---------------------------------------------------------------------------


def _parse_annexb_nals(data: bytes):
    """Yield (start_offset, end_offset, nal_type) for each NAL unit.

    Skips both 3-byte (``00 00 01``) and 4-byte (``00 00 00 01``) start
    codes. Honours H.264's emulation-prevention bytes (``00 00 03``)
    so we never split inside a NAL.
    """
    nals = []
    i = 0
    n = len(data)
    while i < n:
        if (
            i + 3 < n
            and data[i] == 0
            and data[i + 1] == 0
            and data[i + 2] == 0
            and data[i + 3] == 1
        ):
            start = i + 4
        elif i + 2 < n and data[i] == 0 and data[i + 1] == 0 and data[i + 2] == 1:
            start = i + 3
        else:
            i += 1
            continue
        j = start
        end = n
        while j + 2 < n:
            if data[j] == 0 and data[j + 1] == 0:
                if data[j + 2] == 1:
                    end = j
                    break
                if j + 3 < n and data[j + 2] == 0 and data[j + 3] == 1:
                    end = j
                    break
            j += 1
        nal_type = data[start] & 0x1F
        nals.append((start, end, nal_type))
        i = end
    return nals


def _strip_annexb(data: bytes) -> Tuple[bytes, bytes]:
    """Convert an Annex B H.264 bitstream into MP4-friendly form.

    Returns ``(length_prefixed_payload, avcC_extradata)``. The payload
    is the input with each NAL prefixed by a 4-byte big-endian
    length — the format PyAV's MP4 muxer needs to interleave access
    units across containers. ``avcC_extradata`` is the avcC
    configuration box built from the SPS / PPS NAL units; assign it
    to ``stream.codec_context.extradata`` so the muxer writes it as
    the ``avc1`` atom at the head of the file.

    Returns ``(data, b'')`` when no SPS / PPS is found — i.e. the
    caller passed non-header NALs only. The MP4 muxer will then need
    SPS / PPS to be supplied separately (or the stream stays
    unplayable).
    """
    nals = _parse_annexb_nals(data)
    if not nals:
        return data, b""
    sps_list, pps_list = [], []
    payload = bytearray()
    for start, end, nal_type in nals:
        nal_bytes = data[start:end]
        payload += len(nal_bytes).to_bytes(4, "big")
        payload += nal_bytes
        if nal_type == 7:  # SPS
            sps_list.append(nal_bytes)
        elif nal_type == 8:  # PPS
            pps_list.append(nal_bytes)
    if not sps_list:
        return bytes(payload), b""
    sps = sps_list[0]
    avcc = bytearray()
    avcc.append(1)  # configurationVersion
    avcc.append(sps[1])  # profile_idc
    avcc.append(sps[2])  # profile constraint flags
    avcc.append(sps[3])  # level_idc
    avcc.append(0xFF)  # lengthSizeMinusOne = 3 (4-byte NAL lengths)
    avcc.append(0xE0 | len(sps_list))  # reserved + numOfSequenceParameterSets
    for s in sps_list:
        avcc += len(s).to_bytes(2, "big") + s
    avcc.append(len(pps_list))
    for p in pps_list:
        avcc += len(p).to_bytes(2, "big") + p
    return bytes(payload), bytes(avcc)


# ---------------------------------------------------------------------------
# VideoWriter
# ---------------------------------------------------------------------------


class VideoWriter:
    """Hardware/software video encoder sink (see module docstring).

    Parameters
    ----------
    path:
        Output sink. Either a local file path (extension determines
        container — PyAV handles MP4/MKV/AVI/etc.) or an
        ``rkv+ws://host:port/source_id`` URL pointing at a pyrkaiweb
        ingest endpoint.
    fps:
        Output frame rate (Hz).
    frame_size:
        ``(width, height)``. Optional — when omitted (or passed as
        ``(0, 0)``) the encoder backend is initialised lazily on the
        first :meth:`write` call using the size of the first frame.
        A :class:`rkv.mpp.DecodedFrame` contributes ``width`` /
        ``height``; a BGR ndarray contributes ``(frame.shape[1],
        frame.shape[0])``. The width and height of every subsequent
        frame must match the first one.
    codec:
        ``"h264"`` (default, MPP hardware only — requires a
        Rockchip SoC; raises on non-MPP platforms), ``"mjpeg"``
        (Motion JPEG in AVI, PyAV software), ``"mp4v"`` (MPEG-4
        Part 2 in MP4, PyAV software). ``"h264"`` selects the 
        H.264 wire format described in :mod:`rkv.encoder`'s module 
        docstring.
    profile:
        H.264 profile: ``"baseline"`` (default, no B-frames, lowest
        latency), ``"main"``, ``"high"``. Recorded; the backend
        decides whether to honour it.
    bitrate:
        Target bitrate in bits/second. Recorded.
    gop:
        GOP length in frames. ``1`` = all-IDR. Recorded.
    bframes:
        Number of B-frames between P-frames. ``0`` (default).
    rc_mode:
        ``"cbr"`` (default), ``"vbr"``, ``"fixqp"``, ``"avbr"``. Recorded.
    """

    CODEC_H264 = "h264"
    CODEC_H264_RAW = "h264_raw"
    CODEC_HEVC_RAW = "hevc_raw"
    CODEC_MJPEG = "mjpeg"
    CODEC_JPEG = "jpeg"
    CODEC_MP4V = "mp4v"

    PROFILE_BASELINE = "baseline"
    PROFILE_MAIN = "main"
    PROFILE_HIGH = "high"

    RC_VBR = "vbr"
    RC_CBR = "cbr"
    RC_FIXQP = "fixqp"
    RC_AVBR = "avbr"

    def __init__(
        self,
        path: str,
        fps: int = 30,
        frame_size: Tuple[int, int] = (0, 0),
        *,
        codec: str = "h264",
        profile: str = "baseline",
        bitrate: int = 2_000_000,
        gop: int = 30,
        bframes: int = 0,
        rc_mode: str = "cbr",
        q_factor: int = 80,
    ) -> None:
        if codec not in (
            self.CODEC_H264,
            self.CODEC_H264_RAW,
            self.CODEC_HEVC_RAW,
            self.CODEC_MJPEG,
            self.CODEC_JPEG,
            self.CODEC_MP4V,
        ):
            raise ValueError(f"unsupported codec: {codec!r}")
        if codec in (self.CODEC_H264, self.CODEC_H264_RAW) and profile not in _H264_PROFILE_BY_NAME:
            raise ValueError(f"unsupported profile: {profile!r}")
        if rc_mode not in (self.RC_VBR, self.RC_CBR, self.RC_FIXQP, self.RC_AVBR):
            raise ValueError(f"unsupported rc_mode: {rc_mode!r}")
        if codec in (self.CODEC_MJPEG, self.CODEC_JPEG) and not 1 <= q_factor <= 99:
            raise ValueError(
                f"q_factor must be in 1..99 for codec={codec!r}, got {q_factor!r}"
            )
        if gop < 1:
            raise ValueError("gop must be >= 1")

        self._path = path
        self._fps = fps
        self._width, self._height = frame_size
        # ``_pending_init`` is True when the caller didn't pass a
        # positive ``frame_size`` — the encoder backend (MPP / PyAV)
        # and the output container are created lazily on the first
        # :meth:`write` call so the size can be read from the first
        # frame. ``write`` clears the flag once init succeeds.
        self._pending_init = self._width <= 0 or self._height <= 0
        self._codec = codec
        self._profile_name = profile
        self._bitrate = bitrate
        self._gop = gop
        self._bframes = bframes
        self._rc_mode = rc_mode
        self._q_factor = q_factor

        self._is_open = False
        self._ws_sink: Optional["_WebSocketSink"] = None
        self._container = None
        self._stream = None
        self._ws_pts_us = 0
        self._backend: str = ""
        self._raw_file = None
        self._raw_extradata: bytes = b""
        self._raw_extradata_written: bool = False
        self._raw_single_frame_written: bool = False

        if path.startswith("rkv+ws://"):
            # Open the WS sink immediately so writes can flow as soon
            # as the encoder backend is up. The H.264 encoder + avcC
            # header send are deferred to the first ``write()`` when
            # frame_size is unknown at construction time.
            self._ws_sink = _WebSocketSink(path)
            if codec == self.CODEC_H264 and not self._pending_init:
                # H.264 WS path: MPP-only, raises on non-MPP platforms.
                self._init_ws_h264()
            elif codec != self.CODEC_H264:
                # Default JPEG wire format — no encoder init needed.
                self._backend = "ws-jpeg"
            self._is_open = True
            return

        # Make sure the parent directory exists — PyAV refuses to open
        # a file in a non-existent directory with a confusing error.
        out_dir = os.path.dirname(os.path.abspath(path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        # No explicit frame_size at construction time — defer backend
        # init until the first ``write()`` so we can read the size
        # from the first frame. The local file isn't opened yet, but
        # the writer is considered "open" so callers can still
        # ``write`` / ``release`` it normally.
        if self._pending_init:
            self._is_open = True
            return

        # H.264 is MPP-only — there is no software fallback.
        # MJPEG on the local-file path also goes through MPP — the
        # project's design principle is that *all* encoding is MPP, and
        # PyAV is only used for demux/mux. Each MJPEG packet is a
        # complete JPEG with SOI/EOI markers, which matches the layout
        # AVI's MJPEG stream expects.
        if codec in (self.CODEC_H264_RAW, self.CODEC_HEVC_RAW):
            try:
                self._init_mpp_raw_video(path)
            except Exception as e:
                name = "H.264" if codec == self.CODEC_H264_RAW else "HEVC"
                raise RuntimeError(
                    f"rkv.VideoWriter: raw {name} encoding requires MPP "
                    f"hardware encoder but initialization failed: {e}."
                ) from e
            self._backend = (
                "mpp-h264-raw"
                if codec == self.CODEC_H264_RAW
                else "mpp-hevc-raw"
            )
            self._is_open = True
            return

        if codec == self.CODEC_JPEG:
            try:
                self._init_mpp_raw_jpeg(path)
            except Exception as e:
                raise RuntimeError(
                    f"rkv.VideoWriter: raw JPEG encoding requires MPP "
                    f"hardware encoder but initialization failed: {e}."
                ) from e
            self._backend = "mpp-jpeg"
            self._is_open = True
            return

        if codec == self.CODEC_H264:
            try:
                self._init_mpp(path)
            except Exception as e:
                raise RuntimeError(
                    f"rkv.VideoWriter: H.264 local-file encoding requires "
                    f"MPP hardware encoder but MppEncoder.init failed: "
                    f"{e}. Use codec='mp4v' for a PyAV software path, or "
                    f"run on Rockchip hardware."
                ) from e
            self._backend = "mpp"
            self._is_open = True
            return

        if codec == self.CODEC_MJPEG:
            try:
                self._init_mpp_mjpeg(path, q_factor=q_factor)
            except Exception as e:
                raise RuntimeError(
                    f"rkv.VideoWriter: MJPEG local-file encoding requires "
                    f"MPP hardware JPEG encoder but MppEncoder.init failed: "
                    f"{e}. Use codec='mp4v' for an MPEG-4 Part 2 software "
                    f"path on non-MPP platforms."
                ) from e
            self._backend = "mpp-mjpeg"
            self._is_open = True
            return

        self._init_pyav(path)
        self._backend = "pyav"
        self._is_open = True

    def _init_from_frame(self, frame) -> None:
        """Initialise the deferred backend using the size of the first frame.

        Called from :meth:`write` when the writer was constructed
        without a positive ``frame_size``. Pulls ``width`` /
        ``height`` from a :class:`rkv.mpp.DecodedFrame` or, for a
        BGR ndarray, derives them from ``frame.shape`` (``(W, H) =
        (frame.shape[1], frame.shape[0])``). Stores the resolved
        size on ``self`` and routes through the same backend setup
        paths as the eager-init branches above.

        Raises ``ValueError`` when the frame is neither a
        :class:`DecodedFrame` nor a numpy ndarray with a usable
        shape, and ``RuntimeError`` when the chosen backend refuses
        to initialise (e.g. H.264 on a non-MPP platform).
        """
        if hasattr(frame, "fd"):
            w, h = int(frame.width), int(frame.height)
        else:
            try:
                h, w = int(frame.shape[0]), int(frame.shape[1])
            except (AttributeError, IndexError, TypeError) as e:
                raise ValueError(
                    f"cannot infer frame_size from frame of type "
                    f"{type(frame).__name__}; pass frame_size=(W, H) "
                    f"to VideoWriter().__init__"
                ) from e
        if w <= 0 or h <= 0:
            raise ValueError(
                f"first frame has non-positive size ({w}, {h}); "
                f"pass frame_size=(W, H) to VideoWriter().__init__"
            )
        self._width, self._height = w, h

        path = self._path
        codec = self._codec

        if path.startswith("rkv+ws://"):
            # WS sink was already opened in ``__init__`` — only the
            # encoder backend (and the avcC header send for H.264)
            # needs to come up now that we know the size.
            if codec == self.CODEC_H264:
                self._init_ws_h264()
            else:
                self._backend = "ws-jpeg"
            return

        # Local-file path: create the parent directory before opening
        # the output container (PyAV refuses a non-existent directory
        # with a confusing error).
        out_dir = os.path.dirname(os.path.abspath(path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        if codec in (self.CODEC_H264_RAW, self.CODEC_HEVC_RAW):
            try:
                self._init_mpp_raw_video(path)
            except Exception as e:
                name = "H.264" if codec == self.CODEC_H264_RAW else "HEVC"
                raise RuntimeError(
                    f"rkv.VideoWriter: raw {name} encoding requires MPP "
                    f"hardware encoder but initialization failed: {e}."
                ) from e
            self._backend = (
                "mpp-h264-raw"
                if codec == self.CODEC_H264_RAW
                else "mpp-hevc-raw"
            )
            return

        if codec == self.CODEC_JPEG:
            try:
                self._init_mpp_raw_jpeg(path)
            except Exception as e:
                raise RuntimeError(
                    f"rkv.VideoWriter: raw JPEG encoding requires MPP "
                    f"hardware encoder but initialization failed: {e}."
                ) from e
            self._backend = "mpp-jpeg"
            return

        if codec == self.CODEC_H264:
            try:
                self._init_mpp(path)
            except Exception as e:
                raise RuntimeError(
                    f"rkv.VideoWriter: H.264 local-file encoding requires "
                    f"MPP hardware encoder but MppEncoder.init failed: "
                    f"{e}. Use codec='mp4v' for a PyAV software path, or "
                    f"run on Rockchip hardware."
                ) from e
            self._backend = "mpp"
            return

        if codec == self.CODEC_MJPEG:
            try:
                self._init_mpp_mjpeg(path, q_factor=self._q_factor)
            except Exception as e:
                raise RuntimeError(
                    f"rkv.VideoWriter: MJPEG local-file encoding requires "
                    f"MPP hardware JPEG encoder but MppEncoder.init failed: "
                    f"{e}. Use codec='mp4v' for an MPEG-4 Part 2 software "
                    f"path on non-MPP platforms."
                ) from e
            self._backend = "mpp-mjpeg"
            return

        self._init_pyav(path)
        self._backend = "pyav"

    # -- properties ---------------------------------------------------------

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
    def codec(self) -> str:
        return self._codec

    @property
    def profile(self) -> str:
        return self._profile_name

    @property
    def backend(self) -> str:
        """Which encoder path the writer ended up using (``"mpp"`` /
        ``"pyav"`` / ``"ws-jpeg"``). Useful for diagnostics."""
        return self._backend

    @property
    def isOpened(self) -> bool:
        return self._is_open

    # -- backends -----------------------------------------------------------

    def _init_mpp(self, path: str) -> None:
        """Open the file for MPP-encoded output. PyAV handles muxing.

        Sets up:

        * the MPP hardware encoder (H.264 via the VPU),
        * a PyAV container chosen from the file extension — the same
          convention :meth:`_init_mpp_mjpeg` and :meth:`_init_pyav`
          use. ``.mp4`` / ``.mov`` map to ``format="mp4"``; ``.mkv``
          maps to ``format="matroska"``. Unrecognised extensions
          fall back to MP4 (matching the historical default).
        * an avcC extradata captured from the encoder's
          ``MPP_ENC_GET_HDR_SYNC`` so PyAV writes a valid MP4 with
          the ``avc1`` atom at the head. Skipped for matroska (it
          stores SPS / PPS as a CodecState side-band).
        """
        from .mpp import MppEncoder, MppEncRcMode

        rc_mode_map = {
            self.RC_CBR: MppEncRcMode.CBR,
            self.RC_VBR: MppEncRcMode.VBR,
            self.RC_FIXQP: MppEncRcMode.FIXQP,
            self.RC_AVBR: MppEncRcMode.AVBR,
        }
        self._encoder = MppEncoder(
            width=self._width,
            height=self._height,
            fps=self._fps,
            bitrate=self._bitrate,
            gop=self._gop,
            profile=self._profile_name,
            rc_mode=self._rc_mode.lower(),
        )
        self._rc_mode_id = rc_mode_map.get(self._rc_mode, MppEncRcMode.CBR)
        # Frame counter — PyAV expects packet PTS in *time-base ticks*,
        # not microseconds. With ``time_base = 1 / fps`` (seconds), a
        # PTS of 0 / 1 / 2 / … maps directly to the frame index and
        # what FFmpeg's muxer (MP4, AVI, …) writes into the index.
        self._pts = 0

        av = _import_av()
        ext = os.path.splitext(path)[1].lstrip(".").lower()
        container_format = {
            "mp4": "mp4",
            "mov": "mp4",
            "mkv": "matroska",
        }.get(ext, "mp4")
        # ``_send_mpp_packet`` reads this to decide which NAL container
        # the muxer expects.
        self._container_format = container_format
        # MPP 的 Annex-B SPS/PPS 头。在 ``_send_mpp_packet`` 里会被转成
        # length-prefixed 格式前置到首 packet，让 muxer 的 ``extract_extradata``
        # BSF 抽到正确的 SPS/PPS。
        self._mpp_matroska_header = self._encoder.extradata or b""
        self._container = av.open(path, "w", format=container_format)
        # ``add_mux_stream`` skips creating a :class:`CodecContext`
        # — pyAV's mux path runs no encoder, only the BSF pipeline. We
        # pass width/height/pix_fmt/time_base as kwargs (they end up on
        # the stream's ``codecpar``); the codec itself is ``h264`` so
        # both mp4 and matroska pick the right muxer defaults.
        self._stream = self._container.add_mux_stream(
            "h264",
            rate=self._fps,
            width=self._width,
            height=self._height,
            pix_fmt="yuv420p",
            time_base=fractions.Fraction(1, self._fps),
        )

    def _init_mpp_mjpeg(self, path: str, *, q_factor: int = 80) -> None:
        """Open the file for MPP-encoded MJPEG output.

        Sets up:

        * the MPP hardware MJPEG encoder (one independent JPEG per frame),
        * a PyAV container whose format is taken from the file extension
          (``.avi`` → ``avi``, ``.mov`` → ``mp4``). AVI is the canonical
          container for MJPEG streams; PyAV also accepts ``mov``. The
          stream itself is added with the ``mjpeg`` codec name and a
          ``yuvj420p`` pix_fmt so the muxer writes the correct
          ``AVIh`` / ``mjpg`` headers and a sane FourCC.

        No SPS / PPS / VPS to pull — every encoded packet is a self-
        contained JPEG. The encoder's ``_init_mpp`` extradata handling
        isn't needed here.

        ``input_format`` defaults to :data:`MppFrameFormat.BGR888` so
        the eager-BGR write path (``encode(bgr_ndarray)``) works
        without an extra BGR→NV12 step in Python; the zero-copy
        decoder-fd path can still override it via
        :meth:`MppEncoder.encode_external_fd` with ``YUV420SP``.
        """
        from .mpp import MppEncoder, MppCodingType, MppFrameFormat

        self._encoder_q_factor = q_factor
        self._mpp_mjpeg_input_format = MppFrameFormat.BGR888
        self._make_mpp_mjpeg_encoder()

        av = _import_av()
        ext = os.path.splitext(path)[1].lstrip(".").lower()
        container_format = {
            "avi": "avi",
            "mov": "mp4",
            "mp4": "mp4",
            "mkv": "matroska",
        }.get(ext)
        if container_format:
            self._container = av.open(path, "w", format=container_format)
        else:
            # No extension match → fall back to AVI (the canonical MJPEG
            # container); FFmpeg's mjpeg muxer is reliable there.
            self._container = av.open(path, "w", format="avi")
        self._stream = self._container.add_stream("mjpeg", rate=self._fps)
        self._stream.width = self._width
        self._stream.height = self._height
        self._stream.pix_fmt = "yuvj420p"
        # PyAV ≥ 12 rejects raw tuples for time_base; ``fractions.Fraction``
        # carries the ``.numerator`` / ``.denominator`` attributes its
        # Cython setter reads. Setting only the codec_context is enough
        # — ``stream.time_base`` auto-derives for mux.
        self._stream.codec_context.time_base = fractions.Fraction(1, self._fps)
        # Per-MppEncoder frame counter — incremented in ``write`` to
        # decide when to call :meth:`_recycle_mpp_mjpeg_encoder` on
        # the next frame. The recycle keeps the encoder's internal
        # input queue from filling up on builds that don't reset it
        # between calls. See :meth:`_make_mpp_mjpeg_encoder` for the
        # full rationale.
        self._mpp_mjpeg_frame_count = 0
        # PTS counter that survives encoder recycles — when a new
        # encoder takes over, its internal ``_pts`` resets to 0 but
        # the value we hand to PyAV keeps going up so the AVI muxer
        # doesn't reject non-monotonic PTS.
        self._mpp_mjpeg_pts = 0

    def _make_mpp_mjpeg_encoder(self) -> None:
        """(Re)create the underlying :class:`MppEncoder` for MJPEG.

        Called once from :meth:`_init_mpp_mjpeg` and again from
        :meth:`_recycle_mpp_mjpeg_encoder` every
        ``_MPP_MJPEG_RECYCLE_EVERY`` frames. The recycle is a
        workaround for MPP firmware that doesn't free input queue
        slots between ``encode_get_packet`` calls — after ~8 frames
        the queue fills up and ``encode_put_frame`` blocks
        indefinitely on what should be a fast operation. Recreating
        the encoder clears the queue; callers don't see the swap
        because the new encoder produces the same MJPEG format.
        """
        from .mpp import MppEncoder, MppCodingType

        if getattr(self, "_encoder", None) is not None:
            try:
                self._encoder.close()
            except Exception:
                pass
            self._encoder = None
        self._encoder = MppEncoder(
            MppCodingType.MJPEG,
            width=self._width,
            height=self._height,
            fps=self._fps,
            gop=self._gop,
            input_format=self._mpp_mjpeg_input_format,
            rc_mode="fixqp",
            q_factor=self._encoder_q_factor,
        )
        # The encoder's own ``_pts`` resets to 0; we keep
        # ``_mpp_mjpeg_pts`` monotonic so the PyAV packet PTS keeps
        # going up across recycles. See :meth:`_send_mpp_mjpeg_packet`.
        self._pts = 0

    def _recycle_mpp_mjpeg_encoder(self) -> None:
        """Tear down and re-create the MJPEG encoder (workaround for
        the firmware's input-queue leak — see :meth:`_make_mpp_mjpeg_encoder`).
        """
        self._make_mpp_mjpeg_encoder()

    def _init_mpp_raw_video(self, path: str) -> None:
        from .mpp import MppCodingType, MppEncoder

        coding = (
            MppCodingType.AVC
            if self._codec == self.CODEC_H264_RAW
            else MppCodingType.HEVC
        )
        self._encoder = MppEncoder(
            coding_type=coding,
            width=self._width,
            height=self._height,
            fps=self._fps,
            bitrate=self._bitrate,
            gop=self._gop,
            profile=self._profile_name,
            rc_mode=self._rc_mode.lower(),
        )
        self._raw_file = open(path, "wb")
        self._raw_extradata = self._encoder.extradata or b""
        self._raw_extradata_written = False
        self._pts = 0

    def _init_mpp_raw_jpeg(self, path: str) -> None:
        from .mpp import MppCodingType, MppEncoder, MppFrameFormat

        self._encoder = MppEncoder(
            coding_type=MppCodingType.MJPEG,
            width=self._width,
            height=self._height,
            fps=max(self._fps, 1),
            gop=self._gop,
            input_format=MppFrameFormat.BGR888,
            rc_mode="fixqp",
            q_factor=self._q_factor,
        )
        self._raw_file = open(path, "wb")
        self._pts = 0
        self._raw_single_frame_written = False

    def _init_pyav(self, path: str) -> None:
        """Open the file using PyAV's built-in encoders.

        Supports only ``mp4v`` (``mpeg4``). H.264 is MPP-only (see
        :meth:`_init_mpp`); MJPEG is also MPP-only on the local-file
        path (see :meth:`_init_mpp_mjpeg`). PyAV's software encoder is
        reserved for codecs MPP doesn't have an accelerator for.
        """
        av = _import_av()
        codec_map = {
            self.CODEC_MP4V: ("mpeg4", "yuv420p"),
        }
        if self._codec in (self.CODEC_H264, self.CODEC_MJPEG):
            raise RuntimeError(
                f"VideoWriter._init_pyav called for codec={self._codec!r} — "
                f"that path should be routed through _init_mpp / "
                f"_init_mpp_mjpeg. This is a bug."
            )
        if self._codec not in codec_map:
            raise ValueError(f"unsupported codec for PyAV path: {self._codec!r}")
        codec_name, pix_fmt = codec_map[self._codec]
        # Choose container format from the file extension; default to
        # MP4 if PyAV can't autodetect.
        ext = os.path.splitext(path)[1].lstrip(".").lower()
        container_format = {
            "mp4": "mp4",
            "mov": "mp4",
            "mkv": "matroska",
            "avi": "avi",
        }.get(ext)

        if container_format:
            self._container = av.open(path, "w", format=container_format)
        else:
            self._container = av.open(path, "w")
        self._stream = self._container.add_stream(codec_name, rate=self._fps)
        self._stream.width = self._width
        self._stream.height = self._height
        self._stream.pix_fmt = pix_fmt
        if self._bitrate:
            self._stream.bit_rate = self._bitrate
        self._stream.options = {"g": str(self._gop)}
        # For PyAV to encode a BGR ndarray we need to convert to the
        # stream's pix_fmt. The write path handles that.
        self._encoder = None

    def _init_ws_h264(self) -> None:
        """Initialize an H.264 stream over the WebSocket sink.

        Wire format (matches pyrkaiweb's ingest handler):

        * One ``PKT_TYPE_HEADER`` packet first, carrying the
          avcC configuration record built from the encoder's
          SPS / PPS NAL units.
        * Each subsequent access unit sent as ``PKT_TYPE_KEY`` or
          ``PKT_TYPE_DELTA`` with the H.264 access unit's bytes
          **length-prefixed** (4-byte big-endian length followed by
          the NAL) — that's the layout MSE / ``html5 video`` expects.
        * ``PKT_TYPE_EOS`` on :meth:`release`.

        H.264 on the WS path is MPP-only — there is no software
        fallback. If :class:`rkv.mpp.MppEncoder` can't initialise
        here (no Rockchip SoC, or an MPP build that rejects the
        requested width / profile), we raise ``RuntimeError`` so
        the caller knows to fall back to a JPEG sink or a
        Rockchip build instead of silently sending garbage.
        """
        from .mpp import MppEncoder

        try:
            self._encoder = MppEncoder(
                width=self._width,
                height=self._height,
                fps=self._fps,
                bitrate=self._bitrate,
                gop=self._gop,
                profile=self._profile_name,
                # H.264 over the WS sink is fixed to CBR — the "default"
                # rc preset we bind below doesn't reliably produce IDR
                # frames under VBR on this firmware, which leaves the
                # browser decoder stuck waiting for its first keyframe
                # (manifesting as a black screen). CBR's narrower rate
                # range keeps the preset happy. ``mjpeg`` skips MPP
                # entirely (PyAV software path) and uses FIXQP for
                # deterministic per-frame quality.
                rc_mode="fixqp" if self._codec == self.CODEC_MJPEG else "cbr",
            )
        except Exception as e:
            raise RuntimeError(
                f"rkv.VideoWriter: H.264 over rkv+ws:// requires MPP "
                f"hardware encoder but MppEncoder.init failed: {e}. "
                f"Use codec='mjpeg' instead, or run on Rockchip hardware."
            ) from e
        self._backend = "ws-h264-mpp"
        # Send avcC header up front so the receiver can build its
        # MediaSource buffer without waiting for the first keyframe.
        self._send_ws_h264_header(self._encoder.extradata)

    def _send_ws_h264_header(self, annex_b_extradata: bytes) -> None:
        """Send the avcC configuration record as ``PKT_TYPE_HEADER``.

        ``annex_b_extradata`` is the SPS/PPS blob the encoder
        exposed in Annex B form (start-code prefixed). ``_strip_annexb``
        does both the conversion to length-prefixed and the avcC
        build in one pass.
        """
        if not annex_b_extradata:
            return
        _, avcc = _strip_annexb(annex_b_extradata)
        if avcc and self._ws_sink is not None:
            self._ws_sink.send_packet(PKT_TYPE_HEADER, avcc, pts_us=0)

    # -- write --------------------------------------------------------------

    def write(self, frame) -> None:
        """Submit one BGR ndarray (or zero-copy ``DecodedFrame``) frame.

        When the writer was constructed without an explicit
        ``frame_size``, the first call to ``write`` also initialises
        the encoder backend using the size of ``frame``.

        ``frame`` is either:

        * a ``(H, W, 3)`` ``uint8`` BGR ndarray — OpenCV-compatible.
          The frame is copied into the encoder's internal buffer
          (1 CPU memmove) and the encoder handles BGR→YUV on-chip.
        * a :class:`rkv.mpp.DecodedFrame` (zero-copy DMA-BUF). The
          path depends on the codec:

          - **H.264 (MP4 / WS)** — the DMA-BUF fd is imported as an
            :class:`MppBuffer` via
            :meth:`MppEncoder.encode_external_fd` and handed straight
            to the encoder's VPU. **No CPU-side copy between decode
            and encode** — the encoded NALs are written by MPP into
            its own internal scratch buffer that the muxer then
            drains. Falls back to the BGR-then-encoder path if the
            encoder can't be configured for the source format.
          - **MJPEG** — the MJPEG HAL aborts on externally-imported
            DMA-BUF (``mpp_enc_check_pkt_buf`` assertion on this
            firmware), so the zero-copy path is skipped. The
            DecodedFrame is converted to NV12 via RGA in one pass
            (:meth:`rkv.mpp.DecodedFrame.numpy` with ``fmt="NV12"``)
            and fed to a YUV420SP-configured MJPEG encoder. The
            encoder is rebuilt on the first frame if it was
            initialised for BGR. This is still one RGA pass, no
            BGR round-trip.

        The PyAV software backend (``mp4v`` / ``mjpeg``) always
        operates on a BGR ndarray; a ``DecodedFrame`` input is
        converted via :meth:`numpy` (one RGA DMA copy) before
        reaching the encoder. There is no software backend for
        ``h264`` — that path requires MPP.
        """
        if not self.isOpened:
            raise RuntimeError("VideoWriter is closed")

        if self._pending_init:
            # First-frame-derived size: pick width / height off this
            # frame and bring the backend up. Subsequent writes skip
            # this branch. Failure here raises — the writer stays
            # open-but-uninitialised so the next ``write`` can retry.
            self._init_from_frame(frame)
            self._pending_init = False

        if self._ws_sink is not None:
            if self._backend in ("ws-h264-mpp", "ws-h264-pyav"):
                self._send_ws_h264(frame)
            else:
                self._send_ws_jpeg(frame)
            return

        is_decoded_frame = hasattr(frame, "fd")

        # Zero-copy path: input is a DecodedFrame (DMA-BUF fd),
        # encoder backend is the MPP H.264 path. We import the
        # decoder's DMA-BUF fd as an MppBuffer and feed it straight
        # to the VPU — no RGA, no memmove, no Python-side allocation
        # for pixel data. The MJPEG HAL is excluded: importing an
        # external DMA-BUF trips ``mpp_enc_check_pkt_buf`` and the
        # process aborts before the try/except in Python can catch
        # anything. MJPEG takes the NV12 short path below instead.
        if (
            self._encoder is not None
            and is_decoded_frame
            and getattr(frame, "fd", -1) >= 0
            and self._backend
            not in ("mpp-mjpeg", "mpp-jpeg")
        ):
            try:
                from .mpp import MppFrameFormat

                pkt = self._encoder.encode_external_fd(
                    fd=int(frame.fd),
                    width=int(frame.width),
                    height=int(frame.height),
                    hor_stride=int(frame.hor_stride or frame.width),
                    ver_stride=int(frame.ver_stride or frame.height),
                    input_format=MppFrameFormat.YUV420SP,
                    pts=int(frame.pts or self._pts),
                )
                if self._backend == "mpp-h264-raw":
                    self._send_mpp_raw_video_packet(pkt)
                elif self._backend == "mpp-hevc-raw":
                    self._send_mpp_raw_video_packet(pkt)
                else:
                    self._send_mpp_packet(pkt)
                return
            except RuntimeError as e:
                # Zero-copy path failed (e.g. format mismatch on a
                # platform that doesn't accept NV12 in the encoder).
                # Fall through to the eager BGR path so the writer
                # still produces a valid output.
                print(
                    f"rkv.encoder: zero-copy encode_external_fd failed "
                    f"({e}); falling back to eager BGR path",
                    flush=True,
                )

        if self._encoder is not None:
            # MJPEG NV12 short path: take the DecodedFrame's NV12
            # directly via RGA, switch the encoder to YUV420SP on
            # first use. Skips the BGR round-trip that
            # ``_decoded_frame_to_bgr`` would otherwise insert
            # (BGR→NV12 on the encoder side and NV12→BGR on the RGA
            # side). The encoder is rebuilt whenever the desired
            # input format doesn't match — covering both
            # frame0→BGR and BGR→frame0 transitions in the same
            # stream — and the periodic ``_recycle_mpp_mjpeg_encoder``
            # below picks up the new ``_mpp_mjpeg_input_format``
            # automatically.
            if self._backend == "mpp-mjpeg":
                from .mpp import MppFrameFormat as _MJPEG_FMT

                desired_format = (
                    _MJPEG_FMT.YUV420SP if is_decoded_frame else _MJPEG_FMT.BGR888
                )
                if self._mpp_mjpeg_input_format != desired_format:
                    self._mpp_mjpeg_input_format = desired_format
                    self._make_mpp_mjpeg_encoder()
                if is_decoded_frame:
                    pixels = frame.numpy(fmt="NV12")
                else:
                    pixels = frame
            elif self._backend == "mpp-jpeg":
                pixels = (
                    self._decoded_frame_to_bgr(frame)
                    if is_decoded_frame
                    else frame
                )
            else:
                pixels = (
                    self._decoded_frame_to_bgr(frame)
                    if is_decoded_frame
                    else frame
                )
            # On the MJPEG backend, recycle the encoder every
            # ``_MPP_MJPEG_RECYCLE_EVERY`` frames to clear the
            # firmware's input-queue leak — see the rationale in
            # :meth:`_make_mpp_mjpeg_encoder`. We recycle *after*
            # the previous frame has been encoded and muxed, so the
            # call site is mid-stream and the PTS counter restarts at
            # 0 for the new encoder. PyAV's AVI / MP4 muxer only
            # cares about monotonically-increasing PTS, so we patch
            # it up to keep going from where we left off.
            if (
                self._backend == "mpp-mjpeg"
                and self._mpp_mjpeg_frame_count > 0
                and self._mpp_mjpeg_frame_count % _MPP_MJPEG_RECYCLE_EVERY == 0
            ):
                self._recycle_mpp_mjpeg_encoder()
            pkt = self._encoder.encode(pixels)
            if self._backend == "mpp-mjpeg":
                # MJPEG packets are complete JPEG bitstreams (SOI…EOI)
                # with no start codes to strip and no NAL structure.
                self._send_mpp_mjpeg_packet(pkt)
                self._mpp_mjpeg_frame_count += 1
            elif self._backend in ("mpp-h264-raw", "mpp-hevc-raw"):
                self._send_mpp_raw_video_packet(pkt)
            elif self._backend == "mpp-jpeg":
                self._send_mpp_raw_jpeg_packet(pkt)
            else:
                self._send_mpp_packet(pkt)
            return

        # PyAV software path. Always operate on a BGR ndarray.
        bgr = self._decoded_frame_to_bgr(frame) if is_decoded_frame else frame
        if not bgr.flags.c_contiguous:
            bgr = np.ascontiguousarray(bgr)
        # av.VideoFrame.from_ndarray accepts rgb24/bgr24/etc; we always
        # feed bgr24 because that's what VideoCapture produces. PyAV
        # handles the BGR→YUV conversion inside stream.encode().
        av_frame = _import_av().VideoFrame.from_ndarray(bgr, format="bgr24")
        for pkt in self._stream.encode(av_frame):
            self._container.mux(pkt)

    def release(self) -> None:
        if not self._is_open:
            return
        self._is_open = False
        try:
            if self._ws_sink is not None:
                try:
                    self._ws_sink.send_packet(PKT_TYPE_EOS, b"", pts_us=0)
                except Exception:
                    pass
                self._ws_sink.close()
                self._ws_sink = None
            if self._stream is not None and self._container is not None:
                # Flush any packets the encoder is still holding.
                try:
                    for pkt in self._stream.encode():
                        self._container.mux(pkt)
                except Exception:
                    pass
            if self._container is not None:
                try:
                    self._container.close()
                except Exception:
                    pass
                self._container = None
                self._stream = None
            if self._encoder is not None:
                # MPP MJPEG encoder may have one packet still queued
                # when the user calls release(); flush it through the
                # container before tearing down the encoder. ``flush``
                # is a no-op for H.264 callers too because the encoder
                # already returned every frame synchronously from
                # ``encode()``.
                if self._backend == "mpp-mjpeg":
                    try:
                        tail = self._encoder.flush()
                        if tail is not None and tail.data:
                            self._send_mpp_mjpeg_packet(tail)
                    except Exception:
                        pass
                elif self._backend in ("mpp-h264-raw", "mpp-hevc-raw"):
                    try:
                        tail = self._encoder.flush()
                        if tail is not None and tail.data:
                            self._send_mpp_raw_video_packet(tail)
                    except Exception:
                        pass
                try:
                    self._encoder.close()
                except Exception:
                    pass
                self._encoder = None
            if self._raw_file is not None:
                try:
                    self._raw_file.flush()
                except Exception:
                    pass
                try:
                    self._raw_file.close()
                except Exception:
                    pass
                self._raw_file = None
        except Exception:
            pass

    # -- context manager ----------------------------------------------------

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.release()

    # -- internals ---------------------------------------------------------

    def _send_mpp_raw_video_packet(self, pkt) -> None:
        if not pkt.data or self._raw_file is None:
            return
        nals = _parse_annexb_nals(pkt.data)
        if not nals:
            raise RuntimeError("MPP returned a non-Annex-B video packet")
        if not self._raw_extradata_written:
            if self._codec == self.CODEC_H264_RAW:
                nal_types = {nal_type for _, _, nal_type in nals}
                has_header = {7, 8}.issubset(nal_types)
            else:
                nal_types = {
                    (pkt.data[start] >> 1) & 0x3F
                    for start, _, _ in nals
                }
                has_header = {32, 33, 34}.issubset(nal_types)
            if self._raw_extradata and not has_header:
                self._raw_file.write(self._raw_extradata)
            self._raw_extradata_written = True
        self._raw_file.write(pkt.data)
        self._pts += 1

    def _send_mpp_raw_jpeg_packet(self, pkt) -> None:
        if not pkt.data or self._raw_file is None:
            return
        start = pkt.data.find(b"\xff\xd8")
        end = pkt.data.rfind(b"\xff\xd9")
        if start < 0 or end < start:
            raise RuntimeError("MPP returned an invalid JPEG packet")
        self._raw_file.seek(0)
        self._raw_file.truncate()
        self._raw_file.write(pkt.data[start:end + 2])
        self._raw_file.flush()
        self._raw_single_frame_written = True
        self._pts += 1

    def _send_mpp_packet(self, pkt) -> None:
        """Mux one :class:`EncodedPacket` from MPP into the PyAV container.

        PyAV 主分支 PR #2305 路径：mp4 / matroska muxer 走
        ``extract_extradata`` BSF，需要 **length-prefixed** NALs（4 字节
        big-endian 长度 + payload）。把 MPP 输出的 Annex-B 转 length-prefixed
        后，第一帧额外 prepend 一次 SPS/PPS，让 BSF 能抽到正确 SPS/PPS
        写到 codecpar.extradata；后续帧不再 prepend。
        """
        if not pkt.data:
            return
        payload, _ = _strip_annexb(pkt.data)
        if self._mpp_matroska_header:
            _, sps_pps = _strip_annexb(self._mpp_matroska_header)
            payload = sps_pps + payload
            self._mpp_matroska_header = b""
        av = _import_av()
        packet = av.Packet(payload)
        # PTS / DTS are frame indices in ``time_base`` ticks (which is
        # ``1 / fps`` seconds on this stream). One frame per tick.
        packet.pts = self._pts
        packet.dts = self._pts
        packet.duration = 1
        packet.time_base = self._stream.time_base
        packet.stream = self._stream
        self._container.mux(packet)
        self._pts += 1

    def _send_mpp_mjpeg_packet(self, pkt) -> None:
        """Mux one MJPEG :class:`EncodedPacket` from MPP into the PyAV container.

        MJPEG packets are complete JPEG bitstreams (each starts with
        ``FF D8`` SOI and ends with ``FF D9`` EOI), so the only thing
        we have to do is hand them to PyAV with the right timing. There's
        no NAL unit / start-code processing, no avcC extradata to attach
        (the JPEG headers themselves travel inside the packet).

        Time-base is ``1 / fps`` (seconds); PTS / DTS are the frame
        index (0, 1, 2, …) so the AVI muxer writes a sane
        ``PTS × time_base`` and accepts more than one frame per
        stream. PyAV's AVI muxer is strict — without an explicit
        ``time_base`` + ``duration`` it raises ``EINVAL`` on the
        second packet; without ``is_keyframe=True`` it never
        populates the index.
        """
        if not pkt.data:
            return
        av = _import_av()
        packet = av.Packet(pkt.data)
        # PTS / DTS are monotonic frame counters across encoder
        # recycles — PyAV's AVI muxer rejects non-monotonic PTS with
        # ``EINVAL``. ``self._pts`` (the per-encoder counter) may
        # reset on recycle; ``self._mpp_mjpeg_pts`` keeps going up
        # across the whole writer lifetime.
        packet.pts = self._mpp_mjpeg_pts
        packet.dts = self._mpp_mjpeg_pts
        packet.time_base = self._stream.codec_context.time_base
        packet.duration = 1
        packet.stream = self._stream
        # AVI/MJPEG streams expect every packet to be a key frame —
        # setting ``is_keyframe=True`` makes the muxer write ``K`` flags
        # in the AVI index so seek / scrub still works on the output.
        packet.is_keyframe = True
        self._container.mux(packet)
        self._pts += 1
        self._mpp_mjpeg_pts += 1

    def _send_ws_jpeg(self, frame) -> None:
        """WS path: software MJPEG encode via PyAV, push as one packet."""
        bgr = self._decoded_frame_to_bgr(frame) if hasattr(frame, "fd") else frame
        av = _import_av()
        # Software MJPEG encode. Single-frame-in-single-frame-out; no
        # GOP concept so every frame is a keyframe for the wire.
        codec = av.CodecContext.create("mjpeg", "w")
        codec.width = self._width
        codec.height = self._height
        codec.pix_fmt = "yuvj420p"
        codec.options = {"q": str(_WS_JPEG_QUALITY)}
        codec.open()
        av_frame = av.VideoFrame.from_ndarray(bgr, format="bgr24").reformat(
            format="yuvj420p"
        )
        data = bytes(codec.encode(av_frame)[0])
        if not data:
            return
        pts_us = time.monotonic_ns() // 1000
        self._ws_sink.send_packet(PKT_TYPE_KEY, data, pts_us=pts_us)

    def _send_ws_h264(self, frame) -> None:
        """WS path: encode one BGR / ``DecodedFrame`` to H.264, push to WS.

        Wire layout:

        * ``PKT_TYPE_HEADER`` once with the avcC configuration
          record (sent at init time by :meth:`_init_ws_h264`).
        * One ``PKT_TYPE_KEY`` / ``PKT_TYPE_DELTA`` per access unit
          with the H.264 bytes laid out as **4-byte
          length-prefixed NAL units** — the layout MSE / ``html5
          video`` expects. MPP emits Annex B (start-code
          prefixed); :func:`_strip_annexb` rewrites the payload in
          one pass.

        MPP-only — see :meth:`_init_ws_h264` for why the PyAV
        software fallback was removed.
        """
        if self._ws_sink is None or self._encoder is None:
            return
        is_decoded_frame = hasattr(frame, "fd")

        pkt = self._mpp_encode_ws(frame, is_decoded_frame)
        if pkt is None:
            return
        self._ws_pts_us += int(1_000_000 / max(self._fps, 1))
        payload, _ = _strip_annexb(pkt.data)
        if not payload:
            return
        # ``pkt.is_keyframe`` comes from the MPP packet flag, but on the
        # firmware shipped with this build the flag is consistently
        # zero (the H.264 HAL only sets the EOS / EXTRA_DATA bits; intra
        # lives in the H.264 NAL type). Parse the access unit we just
        # length-prefixed: an IDR NAL (type 5) is a key frame, a
        # trailing-slice NAL (type 1) is a delta. Anything else
        # (SPS/PPS/AUD filler) inherits the previous classification so
        # a key SPS that's queued just before its IDR slice stays on
        # the same type as the slice that follows.
        pkt_type = self._classify_nal(payload) or (
            PKT_TYPE_KEY if pkt.is_keyframe else PKT_TYPE_DELTA
        )
        self._ws_sink.send_packet(
            pkt_type, payload, pts_us=self._ws_pts_us
        )

    @staticmethod
    def _classify_nal(length_prefixed: bytes) -> int:
        """Return ``PKT_TYPE_KEY`` / ``PKT_TYPE_DELTA`` / ``0`` for a
        4-byte-length-prefixed H.264 access unit. ``0`` means "no
        slice header in here" (pure SPS / PPS / filler), in which case
        the caller should fall back to the previous classification."""
        off = 0
        while off + 4 <= len(length_prefixed):
            nal_len = int.from_bytes(
                length_prefixed[off:off + 4], "big"
            )
            if nal_len <= 0 or off + 4 + nal_len > len(length_prefixed):
                return 0
            nal = length_prefixed[off + 4:off + 4 + nal_len]
            off += 4 + nal_len
            if not nal:
                continue
            nal_type = nal[0] & 0x1F
            # 5 = Coded slice of IDR picture (key frame)
            # 7 = SPS, 8 = PPS, 9 = AUD, etc. — leave to caller
            if nal_type == 5:
                return PKT_TYPE_KEY
            if nal_type in (1, 2, 3, 4):
                return PKT_TYPE_DELTA
        return 0

    def _mpp_encode_ws(self, frame, is_decoded_frame: bool):
        """Submit one frame to the MPP encoder for the WS path.

        Returns the :class:`EncodedPacket` (or ``None`` on failure).
        Tries the zero-copy DMA-BUF path first when the input is a
        :class:`DecodedFrame`; falls back to the eager BGR path if
        the platform's encoder can't accept the import directly.
        """
        assert self._encoder is not None
        if (
            is_decoded_frame
            and getattr(frame, "fd", -1) >= 0
        ):
            try:
                from .mpp import MppFrameFormat

                return self._encoder.encode_external_fd(
                    fd=int(frame.fd),
                    width=int(frame.width),
                    height=int(frame.height),
                    hor_stride=int(frame.hor_stride or frame.width),
                    ver_stride=int(frame.ver_stride or frame.height),
                    input_format=MppFrameFormat.YUV420SP,
                    pts=self._ws_pts_us,
                )
            except RuntimeError as e:
                print(
                    f"rkv.encoder: WS zero-copy encode_external_fd failed "
                    f"({e}); falling back to eager BGR path",
                    flush=True,
                )
        bgr = self._decoded_frame_to_bgr(frame) if is_decoded_frame else frame
        return self._encoder.encode(bgr)

    def _decoded_frame_to_bgr(self, frame) -> np.ndarray:
        """Best-effort ``DecodedFrame`` → BGR ndarray (zero-copy path).

        Calls :meth:`rkv.mpp.DecodedFrame.numpy` with BGR as the
        requested format. Any failure here surfaces a clear error so
        the caller knows the zero-copy path didn't take.
        """
        if not hasattr(frame, "numpy"):
            return frame
        return frame.numpy(fmt="BGR")


# ---------------------------------------------------------------------------
# WebSocket sink
# ---------------------------------------------------------------------------


class _WebSocketSink:
    """Persistent WebSocket client to a pyrkaiweb ingest endpoint.

    Stdlib socket + a hand-rolled RFC 6455 handshake / frame writer so
    rkv doesn't need to pull in ``websockets`` or ``aiohttp``. Per-frame
    cost is small (one mask + a few header bytes) so the lack of TLS
    acceleration isn't critical; ``wss://`` is a one-line addition if
    needed later (``ssl.wrap_socket`` around ``socket.create_connection``).
    """

    _HANDSHAKE_TIMEOUT_S = 5.0
    _WRITE_LOCK = threading.Lock()

    def __init__(self, url: str) -> None:
        parsed = urllib.parse.urlparse(url)
        # Accept both ``rkv+ws://host:port/path`` and bare
        # ``host:port/path`` (no scheme) forms.
        accepted_scheme = parsed.scheme in ("", "rkv+ws", "rkv+http", "rkv+tcp")
        if not accepted_scheme or not parsed.hostname:
            raise ValueError(
                f"rkv+ws:// URL must be of the form rkv+ws://host:port/path, got {url!r}"
            )
        host = parsed.hostname
        port = parsed.port or 80
        path = parsed.path or "/"
        if parsed.query:
            path += "?" + parsed.query

        self._host = host
        self._port = port
        self._path = path
        self._sock: Optional[socket.socket] = None
        self._connected = False
        self._connect()

    def _connect(self) -> None:
        s = socket.create_connection(
            (self._host, self._port), timeout=self._HANDSHAKE_TIMEOUT_S
        )
        s.settimeout(self._HANDSHAKE_TIMEOUT_S)
        handshake = (
            f"GET {self._path} HTTP/1.1\r\n"
            f"Host: {self._host}:{self._port}\r\n"
            f"Upgrade: websocket\r\n"
            f"Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
            f"Sec-WebSocket-Version: 13\r\n"
            f"\r\n"
        ).encode("ascii")
        try:
            s.sendall(handshake)
        except OSError:
            s.close()
            raise

        buf = bytearray()
        deadline = time.monotonic() + self._HANDSHAKE_TIMEOUT_S
        try:
            while b"\r\n\r\n" not in buf:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    s.close()
                    raise RuntimeError("WS handshake timeout")
                s.settimeout(remaining)
                chunk = s.recv(4096)
                if not chunk:
                    s.close()
                    raise RuntimeError("WS handshake closed before completion")
                buf.extend(chunk)
                if len(buf) > 8192:
                    s.close()
                    raise RuntimeError("WS handshake response too large")
        except (OSError, RuntimeError):
            try:
                s.close()
            except:  # noqa: E722
                pass
            raise
        head = bytes(buf).split(b"\r\n\r\n", 1)[0].decode("iso-8859-1", errors="replace")
        first = head.split("\r\n", 1)[0]
        if "101" not in first:
            try:
                s.close()
            except:  # noqa: E722
                pass
            raise RuntimeError(f"WS handshake failed: {first!r}")

        s.settimeout(None)
        self._sock = s
        self._connected = True

    def send_packet(self, type_byte: int, data: bytes, pts_us: int) -> None:
        if not self._connected or self._sock is None:
            raise RuntimeError("WS sink not connected")
        payload = struct.pack("<BQ", type_byte, pts_us) + data
        n = len(payload)
        mask = struct.pack(">I", time.monotonic_ns() & 0xFFFFFFFF)
        if n < 126:
            header = bytes([0x82, 0x80 | n])  # FIN + binary, masked
        elif n < 65536:
            header = bytes([0x82, 0x80 | 126]) + struct.pack(">H", n)
        else:
            header = bytes([0x82, 0x80 | 127]) + struct.pack(">Q", n)
        masked = bytes(b ^ mask[i & 3] for i, b in enumerate(payload))

        with self._WRITE_LOCK:
            try:
                self._sock.sendall(header + mask + masked)
            except OSError:
                self._connected = False
                try:
                    self._sock.close()
                except Exception:
                    pass
                self._connect()
                self._sock.sendall(header + mask + masked)
                self._connected = True

    def close(self) -> None:
        if self._sock is not None:
            try:
                with self._WRITE_LOCK:
                    self._sock.sendall(bytes([0x88, 0x80]) + b"\x00\x00\x00\x00")
            except Exception:
                pass
            try:
                self._sock.close()
            except Exception:
                pass
        self._sock = None
        self._connected = False