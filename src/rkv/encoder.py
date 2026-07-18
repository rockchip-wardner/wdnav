"""VideoWriter — hardware/software video encoder sink (no cv2).

Replaces the original cv2.VideoWriter / cv2.imencode based sink. Local
files go through PyAV for muxing; the encoded byte stream comes from
one of two backends:

  * **MPP** (hardware) — :class:`rkv.mpp.MppEncoder`. When the input
    is a :class:`rkv.mpp.DecodedFrame` (zero-copy DMA-BUF straight
    from :meth:`VideoCapture.read0`) the encoder imports the
    decoder's fd as an :class:`MppBuffer` via
    ``mpp_buffer_import_with_tag`` and feeds it straight to the VPU
    — **no RGA, no Python-side memmove, no numpy intermediate**
    between decode and encode. Fallback for BGR ndarray input:
    ``mpp_buffer_get_ptr`` + ``memmove`` (1 CPU copy) + on-chip
    BGR→YUV. PyAV only muxes the encoded NALs into the requested
    container (MP4 / MKV / …).
  * **PyAV** (software) — :mod:`av.codec.Codec` ``libx264`` (or
    ``mjpeg``). Used as a fallback when the MPP encoder can't be
    initialised (e.g. development box without a Rockchip SoC, or
    platform builds where :class:`rkv.mpp.MppEncoder` raises on the
    first frame). Always operates on BGR ndarray; 1 RGA DMA copy
    from the decoder's DMA-BUF (via ``DecodedFrame.numpy``) + 1
    libx264 encode.

The ``rkv+ws://`` sink always uses PyAV's software MJPEG encoder —
JPEG packets are the simplest wire format that pyrkaiweb's ingest
handler accepts. Pushing raw H.264 NALs down the same socket would
need a separate protocol change on the server side.

Public API mirrors OpenCV's ``VideoWriter`` so existing call sites
work unchanged:

    out = VideoWriter("/tmp/out.mp4", fps=30, frame_size=(640, 360),
                      codec="mp4v", gop=30, bitrate=2_000_000)
    out.write(frame)                # BGR (H, W, 3) uint8 ndarray
    out.write(frame0)              # DecodedFrame with .fd (zero-copy)
    out.release()

Copy budget by path
===================

``read0() → write(frame0)``:

  | path                    | CPU copies | DMA copies | hw passes |
  |-------------------------|-----------:|------------:|-----------:|
  | MPP zero-copy (DMA-BUF) |          0 |           1 |    NVPU+RGA|
  | MPP eager BGR           |          1 |           1 |    NVPU+RGA|
  | PyAV libx264            |          1 |           1 |    libx264 |

The MPP zero-copy path is the headline: only the VPU and the encoder
DMA engine touch the pixel data, the CPU never sees it. The single
remaining DMA copy is the RGA colour-conversion / resize that's done
by ``DecodedFrame.numpy`` (on demand) — eliminated entirely on the
zero-copy branch because the encoder accepts NV12 directly without a
BGR round-trip.
"""
from __future__ import annotations

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
        ``(width, height)`` — required.
    codec:
        ``"h264"`` (default, hardware MPP on Rockchip, software
        libx264 elsewhere), ``"mjpeg"`` (Motion JPEG in AVI),
        ``"mp4v"`` (MPEG-4 Part 2 in MP4, software only). The WS sink
        always uses software MJPEG regardless of this argument.
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
    CODEC_MJPEG = "mjpeg"
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
    ) -> None:
        if frame_size[0] <= 0 or frame_size[1] <= 0:
            raise ValueError("frame_size must be positive (W, H)")
        if codec not in (self.CODEC_H264, self.CODEC_MJPEG, self.CODEC_MP4V):
            raise ValueError(f"unsupported codec: {codec!r}")
        if codec == self.CODEC_H264 and profile not in _H264_PROFILE_BY_NAME:
            raise ValueError(f"unsupported profile: {profile!r}")
        if rc_mode not in (self.RC_VBR, self.RC_CBR, self.RC_FIXQP, self.RC_AVBR):
            raise ValueError(f"unsupported rc_mode: {rc_mode!r}")
        if gop < 1:
            raise ValueError("gop must be >= 1")

        self._path = path
        self._fps = fps
        self._width, self._height = frame_size
        self._codec = codec
        self._profile_name = profile
        self._bitrate = bitrate
        self._gop = gop
        self._bframes = bframes
        self._rc_mode = rc_mode

        self._is_open = False
        self._ws_sink: Optional["_WebSocketSink"] = None
        self._container = None
        self._stream = None
        self._backend: str = ""  # "mpp", "pyav", "ws-jpeg"

        if path.startswith("rkv+ws://"):
            self._ws_sink = _WebSocketSink(path)
            self._backend = "ws-jpeg"
            self._is_open = True
            return

        # Make sure the parent directory exists — PyAV refuses to open
        # a file in a non-existent directory with a confusing error.
        out_dir = os.path.dirname(os.path.abspath(path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        # Try MPP hardware encoder first; fall back to PyAV software
        # if it can't initialise on this platform / build.
        if codec == self.CODEC_H264:
            try:
                self._init_mpp(path)
                self._backend = "mpp"
                self._is_open = True
                return
            except Exception as e:
                # Surface the failure for diagnostics but keep going so
                # the user gets a working writer on non-target boxes.
                print(
                    f"rkv.encoder: MPP hardware encoder unavailable ({e}); "
                    f"falling back to PyAV libx264",
                    flush=True,
                )

        self._init_pyav(path)
        self._backend = "pyav"
        self._is_open = True

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
        * a PyAV MP4 container for muxing,
        * an avcC extradata captured from the encoder's
          ``MPP_ENC_GET_HDR_SYNC`` so PyAV writes a valid MP4 with
          the ``avc1`` atom at the head.
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
        self._pts_us = 0  # microseconds; PyAV time_base = 1/fps seconds

        av = _import_av()
        self._container = av.open(path, "w", format="mp4")
        self._stream = self._container.add_stream("h264", rate=self._fps)
        self._stream.width = self._width
        self._stream.height = self._height
        self._stream.pix_fmt = "yuv420p"
        self._stream.time_base = self._stream.codec_context.time_base = (
            1,
            self._fps,
        )
        # SPS/PPS extradata from MPP — store on the codec context so
        # the muxer writes the avc1 atom at the head of the file.
        extradata_annexb = self._encoder.extradata
        if extradata_annexb:
            _, avcc = _strip_annexb(extradata_annexb)
            if avcc:
                self._stream.codec_context.extradata = avcc

    def _init_pyav(self, path: str) -> None:
        """Open the file using a PyAV software encoder.

        Picks ``libx264`` for H.264, ``mjpeg`` for MJPEG, ``mpeg4``
        for MP4V. Output is always BGR in / H.264 out (re-encoded to
        yuv420p inside PyAV).
        """
        av = _import_av()
        codec_map = {
            self.CODEC_H264: ("libx264", "yuv420p"),
            self.CODEC_MJPEG: ("mjpeg", "yuvj420p"),
            self.CODEC_MP4V: ("mpeg4", "yuv420p"),
        }
        codec_name, pix_fmt = codec_map[self._codec]
        # Choose container format from the file extension; default to
        # MP4 if PyAV can't autodetect.
        ext = os.path.splitext(path)[1].lstrip(".").lower()
        container_format = {
            "mp4": "mp4",
            "mov": "mp4",
            "mkv": "matroska",
            "avi": "avi",
            "h264": "h264",
        }.get(ext)

        kwargs = {}
        if self._codec == self.CODEC_H264:
            x264_opts = {
                "preset": "ultrafast",
                "tune": "zerolatency",
                "profile": self._profile_name,
                "x264-params": f"keyint={self._gop}:min-keyint={self._gop}:scenecut=0",
                "bf": str(self._bframes),
                "b": str(self._bitrate),
            }
            kwargs["options"] = {k: v for k, v in x264_opts.items() if v}

        if container_format:
            self._container = av.open(path, "w", format=container_format, **kwargs)
        else:
            self._container = av.open(path, "w", **kwargs)
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

    # -- write --------------------------------------------------------------

    def write(self, frame) -> None:
        """Submit one BGR ndarray (or zero-copy ``DecodedFrame``) frame.

        ``frame`` is either:

        * a ``(H, W, 3)`` ``uint8`` BGR ndarray — OpenCV-compatible.
          The frame is copied into the encoder's internal buffer
          (1 CPU memmove) and the encoder handles BGR→YUV on-chip.
        * a :class:`rkv.mpp.DecodedFrame` (zero-copy DMA-BUF).
          When the MPP hardware backend is active, the DMA-BUF fd
          is imported as an :class:`MppBuffer` via
          :meth:`MppEncoder.encode_external_fd` and handed straight
          to the encoder's VPU. **No CPU-side copy between decode
          and encode** — the encoded NALs are written by MPP into
          its own internal scratch buffer that the muxer then
          drains. Falls back to the BGR-then-encoder path if the
          encoder can't be configured for the source format.

        The PyAV software backend always operates on a BGR ndarray
        (the underlying ``libx264`` doesn't take a DMA-BUF); a
        ``DecodedFrame`` input is converted via :meth:`numpy` (one
        RGA DMA copy) before reaching the encoder.
        """
        if not self.isOpened:
            raise RuntimeError("VideoWriter is closed")

        if self._ws_sink is not None:
            self._send_ws_jpeg(frame)
            return

        # Zero-copy path: input is a DecodedFrame (DMA-BUF fd),
        # encoder backend is the MPP hardware path. We import the
        # decoder's DMA-BUF fd as an MppBuffer and feed it straight
        # to the VPU — no RGA, no memmove, no Python-side allocation
        # for pixel data.
        is_decoded_frame = hasattr(frame, "fd")
        if (
            self._encoder is not None
            and is_decoded_frame
            and getattr(frame, "fd", -1) >= 0
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
                    pts=int(frame.pts or self._pts_us),
                )
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
            # MPP hardware path, eager BGR.
            bgr = self._decoded_frame_to_bgr(frame) if is_decoded_frame else frame
            pkt = self._encoder.encode(bgr)
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
                try:
                    self._encoder.close()
                except Exception:
                    pass
                self._encoder = None
        except Exception:
            pass

    # -- context manager ----------------------------------------------------

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.release()

    # -- internals ---------------------------------------------------------

    def _send_mpp_packet(self, pkt) -> None:
        """Mux one :class:`EncodedPacket` from MPP into the PyAV container."""
        if not pkt.data:
            return
        # MPP emits Annex B NAL units; PyAV's MP4 muxer wants 4-byte
        # length-prefixed form plus the avcC atom (already set in
        # _init_mpp). Strip the start codes and rebuild a packet.
        payload, _ = _strip_annexb(pkt.data)
        av = _import_av()
        packet = av.Packet(payload)
        # Time-base: PyAV writes the packet's pts into the MP4 with
        # stream.codec_context.time_base. Use 1/fps seconds so PTS = frame
        # index works out.
        packet.pts = self._pts_us
        packet.dts = self._pts_us
        packet.time_base = self._stream.codec_context.time_base
        packet.stream = self._stream
        self._container.mux(packet)
        # Use the same per-second tick; one frame per PTS tick.
        self._pts_us += int(1_000_000 / max(self._fps, 1))

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