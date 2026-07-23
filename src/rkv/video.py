"""VideoCapture — a hardware-accelerated wrapper around the MPP decoder and
V4L2 capture pipeline.

Designed to mirror the OpenCV ``VideoCapture`` calling convention so existing
OpenCV-style pipelines can be retargeted at the Rockchip VPU with a one-line
import swap. The heavy lifting (PyAV demux, AVCC→Annex B, SPS/PPS injection,
NV12/YUYV → BGR via RGA2) lives in this module; the underlying ctypes
binding to ``librockchip_mpp.so`` is ``rkv.mpp``.

Every pixel-format conversion runs through RGA2 (``librockchip_mpp`` cannot
do colour conversion on its own). If the requested format / size combo is
not accepted by the active RGA backend, ``VideoCapture.read`` raises
``RuntimeError`` rather than silently degrading.

Typical usage:

    import numpy as np
    from rkv import VideoCapture

    cap = VideoCapture("bus.mp4")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # frame is a (H, W, 3) uint8 ndarray — feed to OpenCV / PIL / your
        # own pipeline as needed.
    cap.release()

The class is also a context manager, so a ``with`` block guarantees the
decoder is torn down on exit.
"""

from __future__ import annotations

import ctypes
import re
from pathlib import Path
from typing import Optional, Tuple

import av
import numpy as np

# Note: ``rkv.v4l2`` (the upstream ``v4l2`` package) is **only** needed when
# opening a ``/dev/video*`` capture node. Importing it eagerly forces every
# VideoCapture() call — including the common file/URL path — to pay the cost
# of loading it, and triggers spurious import errors on environments where
# the kernel-headers shim is broken (notably the Python 3.11 build, where
# ``v4l2.py`` uses ``range(1, 9) + [0x80]`` for enum assignment). We defer the
# import to ``_init_v4l2`` so file decoding never touches it.
v4l2 = None  # populated lazily on first /dev/video* open

from . import rga
from .mpp import (
    CODING_BY_NAME,
    DecodedFrame,
    MppCodingType,
    MppDecoder,
    _as_ndarray,
    _resolve_output_format,
    avcc_to_annexb,
    parse_avcc_extradata,
    parse_hvcc_extradata,
)


# Buffer-full retry inside send_packet is short (~5 s) so the wrapper
# surfaces a stuck VPU quickly. The reader in :meth:`VideoCapture.read`
# picks up the slack by draining hard on every BUFFER_FULL before
# retrying — the same pattern ``mpi_dec_test.c`` uses with its
# msleep(1)-bounded outer loop.
_BUFFER_FULL_TOKEN = "ret=-1012"
_DRAIN_RETRIES = 400
_DRAIN_TIMEOUT_MS = 5
# V4L2 fourcc → RGA input format. Only the planar YUV formats (NV12,
# NV21, I420) overlap with the MPP decoder's MPP_TO_RGA map; YUYV/UYVY
# here are V4L2-only and go through the V4L2-specific path.
_V4L2_INPUT_TO_RGA = {
    "NV12": rga.RgaFormat.YCbCr_420_SP,
    "NV21": rga.RgaFormat.YCrCb_420_SP,
    "YUYV": rga.RgaFormat.YUYV_422,
    "UYVY": rga.RgaFormat.UYVY_422,
    "YUV420": rga.RgaFormat.YCbCr_420_P,
}
# V4L2 fourcc → MPP ``fmt`` code, used when we wrap a V4L2 buffer into a
# ``DecodedFrame`` for the unified ``read0()`` API. YUYV/UYVY do not
# have an MPP equivalent and are NOT supported by ``read0()`` — callers
# must use ``read()`` for those sensors.
_V4L2_TO_MPP_FMT = {
    "NV12": 0,
    "NV21": 5,
    "YUV420": 4,
}


def _demuxer_options_for(path_str: str) -> dict:
    """Pick ffmpeg demuxer options based on the source's URL scheme.

    Centralises protocol-specific tuning so the caller never has to
    know about ffmpeg's ``-option value`` plumbing. Add new schemes
    here as the need arises; local files and ``/dev/videoX`` return
    an empty dict (no ffmpeg-level overrides required).
    """
    if path_str.startswith("rtsp://"):
        # ffmpeg's RTSP default is UDP; on lossy networks this drops
        # frames silently. TCP is the only safe choice for live ingest.
        return {"rtsp_transport": "tcp"}
    return {}


# FFmpeg's ``av_dump_format`` produces one line per stream; for the first video
# stream it reads ``streams[i]->codecpar->codec_id`` and looks the codec name up
# via ``avcodec_descriptor_get`` — neither of which requires the actual decoder
# to be registered with libavcodec. We use that string as a fallback when
# ``stream.codec_context`` is ``None`` (PyAV only populates a CodecContext after
# ``avcodec_find_decoder`` succeeds, which fails for codecs missing from the
# slim build of FFmpeg that the rkv samples bundle — see AGENTS.md).
_VIDEO_STREAM_RE = re.compile(
    r"\bVideo:\s+(?P<codec>[A-Za-z0-9_]+)[^\n]*?"
    r"(?P<width>\d{2,5})x(?P<height>\d{2,5}),"
)


def _video_stream_info_from_dump(container) -> Tuple[Optional[str], int, int]:
    """Recover codec name + width/height from ``av_dump_format`` output.

    Only used as a fallback when ``stream.codec_context`` is ``None``.
    Returns ``(None, 0, 0)`` if the dump is empty or contains no ``Video:``
    line, so the caller can surface a clear error.
    """
    prev_level = av.logging.get_level()
    try:
        # ``av_dump_format`` writes via the FFmpeg log callback, which PyAV
        # captures only when the log level is verbose enough.
        av.logging.set_level(av.logging.VERBOSE)
        dump = container.dumps_format()
    finally:
        av.logging.set_level(prev_level)
    m = _VIDEO_STREAM_RE.search(dump)
    if not m:
        return None, 0, 0
    return m.group("codec"), int(m.group("width")), int(m.group("height"))


def _read_codec_config_box(path_str: str, kinds: Tuple[str, ...]) -> bytes:
    """Pull the H.264 ``avcC`` / H.265 ``hvcC`` configuration record straight
    out of the MP4 (or HEIF/MOV) container.

    PyAV exposes the same data through ``stream.codec_context.extradata``, but
    only when the corresponding FFmpeg decoder has been registered; the slim
    build shipped with the rkv samples deliberately omits decoders beyond H.264
    (see AGENTS.md). Walking the ISO BMFF box tree ourselves keeps the path
    demuxer-only and free of any avcodec dependency.

    ``kinds`` is the list of box four-character codes to look for inside the
    first video track's ``stsd`` sample-entry (``avcC``, ``hvcC``). Returns
    ``b""`` when nothing is found — callers treat that as "no SPS/PPS to
    inject", which is fine for streaming protocols that carry them in-band.
    """
    with open(path_str, "rb") as f:
        data = f.read()
    # Container format boxes that can carry ``moov``. We scan all top-level
    # boxes because ``mdat`` can sit before or after ``moov`` (and ``moof``
    # appears in fragmented MP4, which we don't bother handling — those files
    # already supply SPS/PPS per fragment).
    moov_offsets: list[int] = []
    i = 0
    n = len(data)
    while i + 8 <= n:
        size = int.from_bytes(data[i : i + 4], "big")
        tag = data[i + 4 : i + 8]
        if size < 8 or i + size > n:
            break
        if tag == b"moov":
            moov_offsets.append(i + 8)
        elif tag == b"moof":
            # Fragmented MP4 — bail out; SPS/PPS travel with each ``mdat``.
            return b""
        i += size

    # Iterate every moov we found; in practice there's one, but we don't want
    # a malformed file to silently return empty.
    for moov_start in moov_offsets:
        config_box = _walk_for_codec_config(data, moov_start, kinds)
        if config_box:
            return config_box
    return b""


def _walk_for_codec_config(
    data: bytes, start: int, kinds: Tuple[str, ...]
) -> bytes:
    """Depth-limited iso-bmff walker looking for the first video track's
    sample-entry and the requested configuration box inside it.

    The walker doesn't try to be a generic ISO BMFF parser — it knows the
    handful of box types required to reach ``avcC``/``hvcC`` from ``moov`` and
    treats any non-box-shaped area (non-container box or stray fixed fields
    of an AVC/HEVC sample entry) as bytes it can step past. ``stsd`` skips
    its 8-byte version+flags+entry-count prelude; VisualSampleEntry sample
    entries (``avc1``/``hvc1``) skip their 78-byte fixed fields before we
    look for child boxes.
    """
    kinds_b = tuple(k.encode("ascii") for k in kinds)
    container_boxes = (
        b"trak", b"mdia", b"minf", b"stbl", b"stsd",
        b"hvc1", b"hev1", b"avc1", b"avc3", b"av01",
    )
    _VSE_FIXED_LEN = 78
    _STSD_HEADER_LEN = 8  # 1B version + 3B flags + 4B entry_count

    def _walk(offset: int, end: int, depth: int) -> bytes:
        i = offset
        while i + 8 <= end:
            sz = int.from_bytes(data[i : i + 4], "big")
            tag = data[i + 4 : i + 8]
            if sz < 8 or i + sz > end or sz > end - i + 8:
                # Stray bytes that aren't a valid box (e.g. inside
                # VisualSampleEntry compressed-name padding): advance and
                # keep scanning rather than giving up on this whole branch.
                i += 4
                continue
            inner_end = i + sz
            if tag in kinds_b:
                return data[i + 8 : inner_end]
            if tag in container_boxes and depth < 16:
                child_start = i + 8
                if tag == b"stsd":
                    child_start = i + 8 + _STSD_HEADER_LEN
                elif tag in (b"hvc1", b"hev1", b"avc1", b"avc3", b"av01"):
                    child_start = i + 8 + _VSE_FIXED_LEN
                child = _walk(child_start, inner_end, depth + 1)
                if child:
                    return child
            i = inner_end
        return b""

    return _walk(start, len(data), 0)


class VideoCapture:
    """OpenCV-style interface to the MPP hardware decoder / V4L2 camera.

    Iterates an MP4 (or any container PyAV can read) and returns BGR
    frames as ``numpy.ndarray`` shaped ``(H, W, 3)``. Also supports
    local cameras via V4L2 capture nodes (``/dev/videoX``); the
    output is the same BGR ndarray, but the conversion is done by
    ffmpeg / libswscale instead of the VPU.

    Parameters
    ----------
    path:
        Path to the input media file, a network URL (``rtsp://``,
        ``http://``, ``tcp://`` …), or a V4L2 capture node
        (``/dev/video0`` etc.). PyAV's container layer handles the
        protocol; the codec must be one of the formats MPP knows how to
        decode (H.264, HEVC, VP9, AV1, …) — for raw-camera streams
        the "codec" is just the pixel format (NV12 / YUYV / MJPEG),
        which ffmpeg demuxes as ``rawvideo``. Protocol-specific demuxer
        options (e.g. ``rtsp_transport=tcp`` for RTSP) are applied
        automatically and are not exposed on this API.
    timeout:
        Seconds to wait for the container to open *and* for each
        subsequent read. Accepts a float (single value) or a
        ``(open_timeout, read_timeout)`` tuple. Useful for network streams
        that might be unreachable; ``None`` lets PyAV block indefinitely.
    output_size:
        Optional ``(width, height)`` to scale every decoded frame to.
        Resize and pixel-format conversion run in hardware through RGA2;
        unsupported format / size combinations raise ``RuntimeError``.
    v4l2_input_format:
        V4L2-only. Pixel format to request from the sensor / ISP —
        e.g. ``"nv12"``, ``"yuyv422"``, ``"mjpeg"``. ``None`` lets the
        driver pick (usually whatever is currently configured).
    v4l2_size:
        V4L2-only. ``(width, height)`` to request. ``None`` keeps the
        driver's current setting.
    v4l2_framerate:
        V4L2-only. Frames per second to request. ``None`` keeps the
        driver's current setting.

    Notes
    -----
    Media Controller paths (``/dev/mediaX``) are **not** opened directly
    — use :class:`rkv.media.MediaDevice` to inspect / configure the
    pipeline first, then pass the resulting capture node to this class.
    """

    def __init__(
        self,
        path: str | Path,
        timeout: Optional[float] = 10.0,
        output_size: Optional[Tuple[int, int]] = None,
        *,
        v4l2_input_format: Optional[str] = None,
        v4l2_size: Optional[Tuple[int, int]] = None,
        v4l2_framerate: Optional[int] = None,
    ) -> None:
        path_str = str(path)

        # Route based on path prefix. /dev/videoX is V4L2 capture;
        # /dev/mediaX is topology — refuse and point at MediaDevice.
        if path_str.startswith("/dev/media"):
            raise RuntimeError(
                f"{path_str}: this is a Media Controller device, not a "
                f"capture node. Use rkv.media.MediaDevice to inspect / "
                f"configure the pipeline, then open the resulting "
                f"/dev/videoX with VideoCapture."
            )
        if path_str.startswith("/dev/video"):
            self._init_v4l2(
                path_str,
                input_format=v4l2_input_format,
                size=v4l2_size,
                framerate=v4l2_framerate,
                output_size=output_size,
            )
            return

        # Default: file / URL via PyAV container + MPP hardware decoder.
        # str(path) lets URLs through unmodified; Path.is_file() is only
        # used to give a friendlier error for the local-file case.
        local = Path(path)
        if "://" not in path_str and not local.is_file():
            raise FileNotFoundError(f"VideoCapture: no such file {local}")

        kwargs = {}
        options = _demuxer_options_for(path_str)
        if options:
            kwargs["options"] = options
        if timeout is not None:
            kwargs["timeout"] = timeout

        self._container = av.open(path_str, **kwargs)
        if not self._container.streams.video:
            self._container.close()
            raise RuntimeError(f"{path_str}: no video stream found")
        self._stream = self._container.streams.video[0]
        ctx = self._stream.codec_context
        if ctx is not None:
            codec_name = ctx.name
            width, height = ctx.width or 0, ctx.height or 0
        else:
            # Slim FFmpeg build (e.g. ``--enable-decoder=h264`` only) leaves
            # ``codec_context`` unset for codecs whose decoder isn't registered.
            # Demuxing still works — we recover the codec name (and width /
            # height) from the FFmpeg ``av_dump_format`` output, which reads
            # ``codecpar->codec_id`` and the descriptor table, neither of
            # which require the decoder to exist.
            codec_name, width, height = _video_stream_info_from_dump(self._container)
        coding = CODING_BY_NAME.get(codec_name)
        if coding is None:
            self._container.close()
            raise RuntimeError(
                f"{path_str}: codec {codec_name!r} is not supported by MPP. "
                f"Known codecs: {sorted(CODING_BY_NAME)}"
            )

        # Properties surfaced to the caller (OpenCV-compatible names).
        self.width: int = width
        self.height: int = height
        self.fps: float = float(self._stream.average_rate or 0)
        self.frame_pos: int = 0   # frames handed back to the caller so far

        # Live streams (RTSP, HTTP, etc.) never EOF. Detect by looking at
        # the protocol: anything with "://" is treated as live so we keep
        # the demuxer open until the caller calls release(). V4L2 nodes
        # are also "live" — there's no file EOF, only release() stops them.
        self._is_live: bool = "://" in path_str or path_str.startswith("/dev/video")
        self._path_str: str = path_str

        # Hardware-assisted output path. RGA2 lives on the same SOC as the
        # VPU, so feeding it the MPP frame's virtual address is zero-copy.
        # Hardware-assisted output path. RGA2 lives on the same SOC as the
        # VPU, so feeding it the MPP frame's DMA-BUF fd is zero-copy. The
        # conversion buffer is owned by :meth:`DecodedFrame.numpy` (one
        # allocation per call is cheap relative to the RGA pass itself).
        self._output_size: Optional[Tuple[int, int]] = output_size

        self._decoder = MppDecoder(coding_type=coding, split_parse=False)
        # H.264/HEVC: prepend SPS/PPS from the avcC/hvcC configuration
        # box to the first sample so the decoder knows how to parse it.
        # ``CODING_BY_NAME`` always hands back a normalised ffmpeg name
        # (``h264`` / ``hevc``); the extra aliases are kept for callers
        # that set ``codec_name`` directly on a custom-built container.
        if codec_name in ("h264", "avc1"):
            config_box = _read_codec_config_box(
                path_str, ("avcC",)
            )
            self._sps_pps = parse_avcc_extradata(config_box)
        elif codec_name in ("hevc", "h265", "hvc1", "hev1"):
            config_box = _read_codec_config_box(
                path_str, ("hvcC",)
            )
            self._sps_pps = parse_hvcc_extradata(config_box)
        else:
            self._sps_pps = b""
        self._sent_extradata = not self._sps_pps
        self._codec_name = codec_name
        self._is_v4l2: bool = False
        self._opened = True
        self._eof = False

    # -- V4L2 init ----------------------------------------------------------

    def _init_v4l2(
        self,
        path: str,
        *,
        input_format: Optional[str],
        size: Optional[Tuple[int, int]],
        framerate: Optional[int],
        output_size: Optional[Tuple[int, int]],
    ) -> None:
        """Open a /dev/videoX node via rkv.v4l2.V4L2Capture.

        ``v4l2_input_format`` is a FourCC string (e.g. ``"nv12"``,
        ``"yuyv"``). ``v4l2_size`` and ``v4l2_framerate`` default to
        ``None`` (= keep the driver's current setting).
        """
        global v4l2  # noqa: PLW0603 — module-level lazy import binding
        if v4l2 is None:
            from . import v4l2 as _v4l2_mod
            v4l2 = _v4l2_mod
        if not input_format:
            input_format = "nv12"   # most ISP outputs default to NV12
        if not size:
            size = (640, 480)
        # framerate isn't used by V4L2Capture (driver keeps what it has);
        # recorded here only for the public property.
        self._v4l2_input_format = input_format
        self._v4l2_size = size
        self._v4l2_framerate = framerate

        self._v4l2_cap = v4l2.V4L2Capture(
            path,
            width=size[0],
            height=size[1],
            pixel_format=input_format,
            num_buffers=4,
        )

        self.width: int = self._v4l2_cap.width
        self.height: int = self._v4l2_cap.height
        # V4L2 doesn't surface framerate via QUERYCAP; preserve user input.
        self.fps: float = float(framerate or 0)
        self.frame_pos: int = 0
        self._is_live: bool = True
        self._path_str: str = path

        # The V4L2 buffer queue yields raw NV12/YUYV/etc.; RGA performs
        # resize and colour conversion from the exported DMA-BUF.
        self._output_size: Optional[Tuple[int, int]] = output_size
        self._decoder = None
        self._container = None
        self._codec_name: Optional[str] = None
        self._sps_pps = b""
        self._sent_extradata = True
        self._is_v4l2: bool = True
        self._opened = True
        self._eof = False
        # Try to start the capture queue. STREAMON fails with EPERM when
        # no sensor is feeding the pipeline (e.g. on a board where the
        # sensor module is physically disconnected). Surface a clear
        # error in that case so callers don't see a bare errno.
        try:
            self._v4l2_cap.start()
        except OSError as e:
            self._opened = False
            try:
                self._v4l2_cap.close()
            except Exception:
                pass
            self._v4l2_cap = None
            raise RuntimeError(
                f"{path}: cannot start streaming ({e.strerror or e}). "
                f"Is a sensor connected and the ISP pipeline configured?"
            ) from e

    # -- OpenCV-style API ---------------------------------------------------

    def isOpened(self) -> bool:
        return self._opened

    def read(
        self,
        w: Optional[int] = None,
        h: Optional[int] = None,
        fmt: Optional[str] = None,
    ) -> Tuple[bool, Optional[np.ndarray]]:
        """Pull the next decoded frame from the source.

        For local files this returns ``(True, bgr)`` for every frame and
        ``(False, None)`` once the file is exhausted. For live streams
        (RTSP, HTTP, V4L2 cameras) it loops forever; the caller has to
        call :meth:`release` to stop.

        Per-call conversion via RGA2:

        - ``w`` / ``h``: when set, the returned ndarray has shape
          ``(h, w, 3)`` (or ``(h, w, 4)`` for BGRA/RGBA). Resize runs
          on RGA2 in the same hardware pass as the color conversion —
          ~free relative to the VPU/MIPI rate.
        - ``fmt``: pixel format name. Case-insensitive. Currently
          supported: ``"BGR"`` (default), ``"RGB"``, ``"BGRA"``,
          ``"RGBA"``, ``"NV12"``, ``"NV21"``, ``"GRAY"`` / ``"GREY"`` /
          ``"Y"``.

        When **all three** are ``None`` (the default), the frame comes
        back at the source's native size and BGR — i.e. the
        constructor-time ``output_size`` is honoured but otherwise no
        per-call work happens.

        This is a thin wrapper around :meth:`read0` followed by
        :meth:`DecodedFrame.numpy` — see those for the lower-level
        zero-copy API.
        """
        ret, frame0 = self.read0(w, h, fmt)
        if not ret:
            return False, None
        return True, frame0.numpy()

    def read0(
        self,
        w: Optional[int] = None,
        h: Optional[int] = None,
        fmt: Optional[str] = None,
    ) -> Tuple[bool, Optional[DecodedFrame]]:
        """Pull the next raw decoded frame, without doing any RGA conversion.

        Returns a :class:`rkv.mpp.DecodedFrame` whose ``fd`` is the
        DMA-BUF file descriptor of the underlying MPP buffer (or
        ``plane_fds[0]`` for V4L2 sources). Hand it to RGA, the display
        controller, or the encoder for hardware processing without
        touching the CPU cache. Or call :meth:`DecodedFrame.numpy` to
        materialise the frame into a contiguous ndarray via RGA2.

        ``w`` / ``h`` / ``fmt`` are *not* consumed here — they are
        stashed on the returned frame and used as defaults by
        :meth:`DecodedFrame.numpy`. That makes ``cap.read0(w, h,
        fmt).numpy()`` equivalent to ``cap.read(w, h, fmt)``.

        V4L2 sources only support the planar YUV formats (NV12 / NV21 /
        I420). For YUYV / UYVY sensors use :meth:`read` directly — its
        V4L2 path runs the RGA conversion without going through
        :meth:`DecodedFrame.numpy`.
        """
        if not self._opened or self._eof:
            return False, None

        out_w = w if w else None
        out_h = h if h else None
        out_fmt = (fmt.upper() if isinstance(fmt, str) else None) if fmt else None
        # When the caller passed no overrides at all, fall back to the
        # constructor-time output_size for backward compat.
        if out_w is None and out_h is None and out_fmt is None:
            if self._output_size is not None:
                out_w, out_h = self._output_size

        if getattr(self, "_is_v4l2", False):
            return self._read0_v4l2(out_w, out_h, out_fmt)

        for raw_packet in self._container.demux(self._stream):
            data = bytes(raw_packet)
            if not data:
                continue
            if not self._send(
                data, pts=int(raw_packet.pts or raw_packet.dts or 0),
                out_w=out_w, out_h=out_h, out_fmt=out_fmt,
            ):
                return False, None

            frame = self._drain_one_raw(out_w=out_w, out_h=out_h, out_fmt=out_fmt)
            if frame is not None:
                # Return the MPP MppFrame handle to MPP's internal pool
                # immediately — the fd / ptr / size fields on the frame
                # remain valid (the DMA-BUF is owned by the kernel-side
                # buffer group), so the caller can still feed them to
                # RGA / encoder / display. Without this release the
                # MPP buffer pool (24 entries by default) fills up and
                # the next send_packet starts raising BUFFER_FULL.
                self._decoder.release_frame(frame)
                frame.out_w, frame.out_h, frame.out_fmt = out_w, out_h, out_fmt
                self.frame_pos += 1
                return True, frame

        if self._is_live:
            import time as _t
            _t.sleep(0.005)
            return self.read0(w, h, fmt)

        self._decoder.send_packet(b"", pts=0, eos=True)
        idle = 0
        while idle < 10:
            frame = self._recv_raw(timeout_ms=50)
            if frame is not None:
                self._decoder.release_frame(frame)
                frame.out_w, frame.out_h, frame.out_fmt = out_w, out_h, out_fmt
                self.frame_pos += 1
                return True, frame
            idle += 1

        self._eof = True
        return False, None

    def release(self) -> None:
        if not self._opened:
            return
        self._opened = False
        try:
            if self._decoder is not None:
                self._decoder.close()
        finally:
            self._decoder = None  # type: ignore[assignment]
            if self._container is not None:
                self._container.close()
            self._container = None  # type: ignore[assignment]
            if getattr(self, "_v4l2_cap", None) is not None:
                try:
                    self._v4l2_cap.close()
                except Exception:
                    pass
                self._v4l2_cap = None

    # -- V4L2 read path -----------------------------------------------------

    def _read0_v4l2(
        self,
        out_w: Optional[int] = None,
        out_h: Optional[int] = None,
        out_fmt: Optional[str] = None,
    ) -> Tuple[bool, Optional[DecodedFrame]]:
        """V4L2 variant of :meth:`read0`.

        Wraps the dequeued :class:`rkv.v4l2.V4L2BufferInfo` into a
        :class:`DecodedFrame` so callers can use the uniform
        ``cap.read0(...).numpy()`` pattern. Only planar YUV formats
        (NV12 / NV21 / I420) are supported; YUYV / UYVY sensors must use
        :meth:`read` directly.
        """
        cap = self._v4l2_cap
        if cap is None:
            return False, None
        pixel_format = cap.pixel_format.upper()
        if pixel_format not in _V4L2_TO_MPP_FMT:
            raise NotImplementedError(
                f"read0() does not support V4L2 pixel format {pixel_format!r}; "
                f"supported: {sorted(_V4L2_TO_MPP_FMT)}. Use read() instead."
            )
        buf = cap.dequeue(timeout_s=2.0)
        if buf is None:
            return False, None
        try:
            stride = buf.plane_arrays[0].shape[1] if buf.plane_arrays else buf.width
            frame = DecodedFrame(
                width=buf.width,
                height=buf.height,
                hor_stride=stride,
                ver_stride=buf.height,
                fmt=_V4L2_TO_MPP_FMT[pixel_format],
                pts=int(buf.timestamp * 1e6) if buf.timestamp else 0,
                ptr=0,
                fd=buf.plane_fds[0] if buf.plane_fds else -1,
                buf_size=stride * buf.height * 3 // 2,
                raw=None,  # V4L2 frames don't carry an MppFrame handle
                out_w=out_w,
                out_h=out_h,
                out_fmt=out_fmt,
            )
        finally:
            cap.requeue(buf)
        self.frame_pos += 1
        return True, frame

    def _read_v4l2(
        self,
        out_w: Optional[int] = None,
        out_h: Optional[int] = None,
        out_fmt: Optional[str] = None,
    ) -> Tuple[bool, Optional[np.ndarray]]:
        """Pull one decoded frame from a V4L2 capture node.

        Loops forever for live sources (cameras have no EOF); the caller
        must call :meth:`release` to stop. The per-call ``out_w`` /
        ``out_h`` / ``out_fmt`` overrides are forwarded to RGA so the
        caller can ask for a different size or pixel format frame-by-frame.
        Raises ``RuntimeError`` if RGA cannot service the request.

        Used by :meth:`read` for V4L2 sources — it has its own
        conversion path so YUYV / UYVY sensors are still supported
        (those formats don't fit :meth:`DecodedFrame.numpy`'s MPP
        decoder input set).
        """
        cap = self._v4l2_cap
        if cap is None:
            return False, None
        buf = cap.dequeue(timeout_s=2.0)
        if buf is None:
            return False, None
        try:
            bgr = self._v4l2_to_bgr(buf, out_w, out_h, out_fmt)
        finally:
            cap.requeue(buf)
        if bgr is not None:
            self.frame_pos += 1
            return True, bgr
        return False, None

    def _v4l2_to_bgr(
        self,
        buf: "v4l2.V4L2BufferInfo",
        out_w: Optional[int] = None,
        out_h: Optional[int] = None,
        out_fmt: Optional[str] = None,
    ) -> np.ndarray:
        """Convert a V4L2 raw buffer into a contiguous ndarray via RGA2.

        Supported V4L2 source formats: NV12, NV21, I420, YUYV, UYVY.
        Anything else raises. Resize and per-call ``out_fmt`` /
        ``out_w`` / ``out_h`` overrides are honoured; on RGA rejection
        the exception propagates.
        """
        fmt = buf.pixel_format.upper()
        if fmt not in _V4L2_INPUT_TO_RGA:
            raise RuntimeError(
                f"V4L2 pixel format {fmt!r} is not supported by RGA; "
                f"supported: {sorted(_V4L2_INPUT_TO_RGA)}"
            )
        target_w = out_w if out_w else (self._output_size[0] if self._output_size else buf.width)
        target_h = out_h if out_h else (self._output_size[1] if self._output_size else buf.height)
        target_fmt, bpp, layout = _resolve_output_format(out_fmt)
        input_fmt = _V4L2_INPUT_TO_RGA[fmt]

        if buf.plane_fds and buf.plane_fds[0] >= 0:
            src = rga.wrap_fd(
                buf.plane_fds[0], buf.width, buf.height, input_fmt,
                wstride=buf.width, hstride=buf.height,
            )
        elif buf.plane_arrays:
            if len(buf.plane_arrays) == 1:
                src_array = np.ascontiguousarray(buf.plane_arrays[0])
            else:
                src_array = np.concatenate(
                    [np.asarray(plane).reshape(-1) for plane in buf.plane_arrays]
                )
            src = rga.wrap_virtual(
                int(src_array.ctypes.data), buf.width, buf.height, input_fmt,
                wstride=buf.width, hstride=buf.height,
            )
        else:
            raise RuntimeError("V4L2 buffer has no DMA-BUF fd or virtual-address plane")

        rga_w = (target_w + 15) & ~15
        bpp_target = rga.bytes_per_pixel(target_fmt)
        size = -(-int(rga_w * target_h * bpp_target) // 1)   # ceil
        buf_storage = (ctypes.c_uint8 * size)()
        dst = rga.wrap_virtual(
            ctypes.addressof(buf_storage), rga_w, target_h, target_fmt
        )
        try:
            if (target_w, target_h) == (buf.width, buf.height):
                rga.cvtcolor(src, dst)
            else:
                rga.resize(src, dst)
            rga.flush()
        except RuntimeError as e:
            raise RuntimeError(
                f"RGA cannot convert V4L2 {buf.pixel_format.upper()} "
                f"{buf.width}x{buf.height} to {rga.format_name(target_fmt)} "
                f"{target_w}x{target_h}: {e}"
            ) from e

        out = _as_ndarray(buf_storage, rga_w, target_h, bpp, layout)
        if rga_w != target_w:
            out = out[:, :target_w]
        return out

    # -- context manager / destructor --------------------------------------

    def __enter__(self) -> "VideoCapture":
        return self

    def __exit__(self, *_exc) -> None:
        self.release()

    def __del__(self) -> None:
        try:
            self.release()
        except Exception:
            pass

    # -- internals ----------------------------------------------------------

    def _send(
        self,
        data: bytes,
        pts: int,
        out_w: Optional[int] = None,
        out_h: Optional[int] = None,
        out_fmt: Optional[str] = None,
    ) -> bool:
        """Submit one packet to MPP, draining on BUFFER_FULL.

        Returns ``False`` if the stream is exhausted. The ``out_*``
        args are forwarded to :meth:`_drain_one_raw` so frames we
        surface from this BUFFER_FULL retry path pick up the per-call
        overrides the caller asked for.
        """
        if self._codec_name in ("h264", "hevc", "h265"):
            annexb = avcc_to_annexb(data)
            if self._sps_pps and not self._sent_extradata:
                annexb = self._sps_pps + annexb
                self._sent_extradata = True
        else:
            annexb = data

        while True:
            try:
                self._decoder.send_packet(annexb, pts=pts, eos=False)
                return True
            except RuntimeError as e:
                if _BUFFER_FULL_TOKEN not in str(e):
                    raise
                # VPU's packet queue is saturated; pull a frame and try
                # again. If no frame is ready yet (cold start), give the
                # hardware a brief moment to catch up.
                frame = self._drain_one_raw(
                    timeout_ms=_DRAIN_TIMEOUT_MS,
                    out_w=out_w, out_h=out_h, out_fmt=out_fmt,
                )
                if frame is not None:
                    # We have to come back to the caller with this frame
                    # before retrying the send — but we're inside _send.
                    # Stash it via _prefetch so read0() picks it up next.
                    # We also record the (w, h, fmt) it was produced
                    # with; a later read0() with different overrides
                    # invalidates the cache.
                    self._prefetched = frame
                    self._prefetched_args = (out_w, out_h, out_fmt)
                    return True
                import time as _t
                _t.sleep(0.005)

    def _drain_one_raw(
        self,
        timeout_ms: int = _DRAIN_TIMEOUT_MS,
        out_w: Optional[int] = None,
        out_h: Optional[int] = None,
        out_fmt: Optional[str] = None,
    ) -> Optional[DecodedFrame]:
        """Pull at most one raw frame out of the decoder.

        Returns ``None`` if no frame is ready within *timeout_ms*. The
        BUFFER_FULL retry path stashes frames in ``self._prefetched``
        along with the (out_w, out_h, out_fmt) they were produced
        with; if the next call comes in with different overrides, the
        cache is invalidated so we don't surface a frame whose RGA
        conversion args were computed against stale (w, h, fmt).
        """
        if hasattr(self, "_prefetched") and self._prefetched is not None:
            if getattr(self, "_prefetched_args", None) == (out_w, out_h, out_fmt):
                out, self._prefetched = self._prefetched, None
                self._prefetched_args = None
                return out
            self._prefetched = None
            self._prefetched_args = None
        return self._recv_raw(timeout_ms=timeout_ms)

    def _recv_raw(
        self,
        timeout_ms: int,
    ) -> Optional[DecodedFrame]:
        """Run ``recv_frame`` once and return the raw ``DecodedFrame``.

        ``release_frame`` is *not* called here — the caller (either the
        demux loop in :meth:`read0` or the BUFFER_FULL retry path in
        :meth:`_send`) owns the frame's lifetime. The caller can either
        pass the frame back to MPP via ``release_frame(frame)`` or hand
        it to RGA / an encoder for hardware processing.
        """
        return self._decoder.recv_frame(timeout_ms=timeout_ms)


__all__ = ["VideoCapture"]