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
    avcc_to_annexb,
    parse_avcc_extradata,
)


# Buffer-full retry inside send_packet is short (~5 s) so the wrapper
# surfaces a stuck VPU quickly. The reader in :meth:`VideoCapture.read`
# picks up the slack by draining hard on every BUFFER_FULL before
# retrying — the same pattern ``mpi_dec_test.c`` uses with its
# msleep(1)-bounded outer loop.
_BUFFER_FULL_TOKEN = "ret=-1012"
_DRAIN_RETRIES = 400
_DRAIN_TIMEOUT_MS = 5
# MPP frame-format codes we accept. Maps directly onto the rga.h
# surface-format enum (left-shifted by 8): 0 = NV12, 4 = I420, 5 = NV21.
_MPP_TO_RGA_FMT = {
    0: rga.RgaFormat.YCbCr_420_SP,
    4: rga.RgaFormat.YCbCr_420_P,
    5: rga.RgaFormat.YCrCb_420_SP,
}
_MPP_TO_RGA_NAME = {0: "NV12", 4: "I420", 5: "NV21"}
_V4L2_INPUT_TO_RGA = {
    "NV12": rga.RgaFormat.YCbCr_420_SP,
    "NV21": rga.RgaFormat.YCrCb_420_SP,
    "YUYV": rga.RgaFormat.YUYV_422,
    "UYVY": rga.RgaFormat.UYVY_422,
    "YUV420": rga.RgaFormat.YCbCr_420_P,
}


# Map the short names users pass to ``read(fmt=...)`` to the corresponding
# RgaFormat constants plus buffer-layout metadata. Each entry is
# ``(rga_format, bytes_per_pixel, layout)`` where ``layout`` is one of:
#   ``"HWC"``  — packed pixel (BGR/RGB/BGRA/RGBA); shape ``(H, W, C)``.
#   ``"YUV"``  — NV12/NV21 full buffer; shape ``(H * 3 // 2, W)`` (Y plane
#                + interleaved UV at offset H*W).
#   ``"GRAY"`` — single-channel luma from a NV12 RGA pass; shape ``(H, W)``.
# RGA writes directly into the destination buffer in this format, so for
# example ``fmt="RGB"`` swaps R and B in one hardware pass.
_OUTPUT_FORMAT_BY_NAME = {
    # 8-bit packed pixel formats
    "BGR":    (rga.RgaFormat.BGR_888,     3,   "HWC"),
    "BGR888": (rga.RgaFormat.BGR_888,     3,   "HWC"),
    "RGB":    (rga.RgaFormat.RGB_888,     3,   "HWC"),
    "RGB888": (rga.RgaFormat.RGB_888,     3,   "HWC"),
    # 8-bit packed pixel + alpha
    "BGRA":      (rga.RgaFormat.BGRA_8888, 4, "HWC"),
    "BGRA8888": (rga.RgaFormat.BGRA_8888, 4, "HWC"),
    "RGBA":      (rga.RgaFormat.RGBA_8888, 4, "HWC"),
    "RGBA8888": (rga.RgaFormat.RGBA_8888, 4, "HWC"),
    # 8-bit YUV 4:2:0 semi-planar (RGA writes full Y + interleaved UV)
    "NV12":      (rga.RgaFormat.YCbCr_420_SP, 1.5, "YUV"),
    "NV21":      (rga.RgaFormat.YCrCb_420_SP, 1.5, "YUV"),
    # Luma-only via RGA's native YUV400 format (single channel, 1 BPP).
    "GRAY":      (rga.RgaFormat.YCbCr_400, 1,   "GRAY"),
    "GREY":      (rga.RgaFormat.YCbCr_400, 1,   "GRAY"),
    "Y":         (rga.RgaFormat.YCbCr_400, 1,   "GRAY"),
}


def _resolve_output_format(fmt: Optional[str]) -> Tuple[int, float, str]:
    """Return ``(rga_format, bytes_per_pixel, layout)`` for ``fmt``.

    ``None`` or empty string defaults to BGR_888 packed pixels.
    """
    if not fmt:
        return rga.RgaFormat.BGR_888, 3, "HWC"
    try:
        return _OUTPUT_FORMAT_BY_NAME[fmt.upper()]
    except KeyError as e:
        raise ValueError(
            f"unknown output format {fmt!r}; supported: "
            f"{sorted(_OUTPUT_FORMAT_BY_NAME)}"
        ) from e


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


def _as_ndarray(
    buf_storage: "ctypes.Array",
    width: int,
    height: int,
    bpp: float,
    layout: str,
) -> np.ndarray:
    """Reshape an RGA output buffer into the right ndarray view.

    ``layout`` is one of ``"HWC"`` (packed pixels), ``"YUV"`` (full NV12 /
    NV21 plane), or ``"GRAY"`` (single-channel luma extracted from the
    Y plane of a NV12 RGA pass). For 1.5-BPP layouts we expose the full
    buffer; for 1-BPP we slice off the Y plane (the first H*W bytes of
    the NV12 layout) so the returned ndarray is exactly ``(H, W)``.
    """
    raw = np.frombuffer(buf_storage, dtype=np.uint8)
    if layout == "YUV":
        # Full NV12 layout: Y plane at [0, H*W), interleaved UV at
        # [H*W, H*W*3/2). Return the full (H*3/2, W) view so callers
        # can hand it to another YUV codec / encoder.
        return raw.reshape(height * 3 // 2, width)
    if layout == "GRAY":
        # NV12 was written internally; only the Y plane is the user-
        # visible result. The first H*W bytes are exactly the Y plane
        # (stride == width in our wrap_virtual call).
        return raw[: height * width].reshape(height, width)
    # HWC: packed RGB-family. Channels = bytes per pixel (3 or 4).
    channels = int(round(bpp))
    return raw.reshape(height, width, channels)


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
        codec_name = self._stream.codec_context.name
        coding = CODING_BY_NAME.get(codec_name)
        if coding is None:
            self._container.close()
            raise RuntimeError(
                f"{path_str}: codec {codec_name!r} is not supported by MPP. "
                f"Known codecs: {sorted(CODING_BY_NAME)}"
            )

        # Properties surfaced to the caller (OpenCV-compatible names).
        self.width: int = self._stream.codec_context.width or 0
        self.height: int = self._stream.codec_context.height or 0
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
        # The destination BGR buffer is allocated once and grown lazily.
        self._output_size: Optional[Tuple[int, int]] = output_size
        # Cached RGA dst buffer (BGR888). Keyed on (w, h); rebuilt when
        # the effective output dimensions change.
        self._bgr_buf_key: Optional[Tuple[int, int]] = None
        self._bgr_buf: Optional[ctypes.Array] = None  # ctypes c_uint8 array

        self._decoder = MppDecoder(coding_type=coding, split_parse=False)
        # H.264/HEVC: prepend SPS/PPS from the avcC/hvcC configuration
        # box to the first sample so the decoder knows how to parse it.
        self._sps_pps = (
            parse_avcc_extradata(self._stream.codec_context.extradata or b"")
            if codec_name in ("h264", "hevc", "h265")
            else b""
        )
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
          ``"RGBA"``. Anything else falls back to BGR.

        When **all three** are ``None`` (the default), the frame comes
        back at the source's native size and BGR — i.e. the
        constructor-time ``output_size`` is honoured but otherwise no
        per-call work happens.
        """
        if not self._opened or self._eof:
            return False, None

        # Resolve per-call overrides. An empty string is treated like None
        # so callers can pass through variables without an explicit None check.
        out_w = w if w else None
        out_h = h if h else None
        out_fmt = (fmt.upper() if isinstance(fmt, str) else None) if fmt else None
        # When the caller passed no overrides at all, fall back to the
        # constructor-time output_size for backward compat.
        if out_w is None and out_h is None and out_fmt is None:
            if self._output_size is not None:
                out_w, out_h = self._output_size

        # V4L2 cameras skip MPP entirely — the VPU has nothing to do on
        # raw pixels, and the V4L2 demuxer yields decoded frames directly.
        if getattr(self, "_is_v4l2", False):
            return self._read_v4l2(out_w, out_h, out_fmt)

        # Phase 1: feed more packets from PyAV and surface any frame
        # produced along the way. We yield the *first* decoded frame we
        # see — the caller drives pacing by calling read() again.
        for raw_packet in self._container.demux(self._stream):
            data = bytes(raw_packet)
            if not data:
                continue
            if not self._send(data, pts=int(raw_packet.pts or raw_packet.dts or 0),
                              out_w=out_w, out_h=out_h, out_fmt=out_fmt):
                return False, None

            frame = self._drain_one(out_w=out_w, out_h=out_h, out_fmt=out_fmt)
            if frame is not None:
                return True, frame

        # Phase 2: file EOF. Signal end-of-stream so MPP flushes any
        # frames it queued, then drain whatever comes back. For live
        # streams we skip this — there is no EOF.
        if self._is_live:
            # No frame in the queue? Wait briefly for the next packet
            # to arrive over the network, then loop back to demux().
            import time as _t
            _t.sleep(0.005)
            return self.read(w, h, fmt)

        self._decoder.send_packet(b"", pts=0, eos=True)
        idle = 0
        while idle < 10:
            frame = self._recv_bgr(timeout_ms=50, out_w=out_w, out_h=out_h, out_fmt=out_fmt)
            if frame is not None:
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
        return self._v4l2_rga_to_bgr(buf, out_w, out_h, out_fmt)

    # -- RGA zero-copy helpers (shared with the MPP path) --------------------

    def _v4l2_rga_to_bgr(
        self,
        buf: "v4l2.V4L2BufferInfo",
        out_w: Optional[int] = None,
        out_h: Optional[int] = None,
        out_fmt: Optional[str] = None,
    ) -> np.ndarray:
        """Convert a V4L2 DMA-BUF into the requested output through RGA2."""
        target_w = out_w if out_w else (self._output_size[0] if self._output_size else buf.width)
        target_h = out_h if out_h else (self._output_size[1] if self._output_size else buf.height)
        target_fmt, bpp, layout = _resolve_output_format(out_fmt)
        input_fmt = _V4L2_INPUT_TO_RGA[buf.pixel_format.upper()]

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
        buf_storage = self._ensure_bgr_buffer(rga_w, target_h, fmt=target_fmt)
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

        Returns ``False`` if the stream is exhausted (read() should stop).
        The ``out_*`` args are forwarded to :meth:`_drain_one` so frames
        we surface from this BUFFER_FULL retry path use the per-call
        resolution the caller asked for.
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
                frame = self._drain_one(timeout_ms=_DRAIN_TIMEOUT_MS,
                                       out_w=out_w, out_h=out_h, out_fmt=out_fmt)
                if frame is not None:
                    # We have to come back to the caller with this frame
                    # before retrying the send — but we're inside _send.
                    # Stash it via _prefetch so read() picks it up next.
                    # We also record the (w, h, fmt) it was produced
                    # with; a later read() with different overrides
                    # invalidates the cache.
                    self._prefetched = frame
                    self._prefetched_args = (out_w, out_h, out_fmt)
                    return True
                import time as _t
                _t.sleep(0.005)

    def _drain_one(
        self,
        timeout_ms: int = _DRAIN_TIMEOUT_MS,
        out_w: Optional[int] = None,
        out_h: Optional[int] = None,
        out_fmt: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        """Pull at most one frame out of the decoder, returning BGR or None.

        The BUFFER_FULL retry path stashes the frame it produces in
        ``self._prefetched`` along with the (out_w, out_h, out_fmt) it
        was produced with. If the next ``read()`` calls with different
        overrides, the cache is invalidated so we don't hand back a
        BGR frame when the caller asked for GRAY.
        """
        if hasattr(self, "_prefetched") and self._prefetched is not None:
            if getattr(self, "_prefetched_args", None) == (out_w, out_h, out_fmt):
                out, self._prefetched = self._prefetched, None
                self._prefetched_args = None
                return out
            # Cache was produced for a different (w, h, fmt) — drop it
            # and re-convert this time.
            self._prefetched = None
            self._prefetched_args = None
        return self._recv_bgr(
            timeout_ms=timeout_ms,
            out_w=out_w, out_h=out_h, out_fmt=out_fmt,
        )

    def _recv_bgr(
        self,
        timeout_ms: int,
        out_w: Optional[int] = None,
        out_h: Optional[int] = None,
        out_fmt: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        """Run recv_frame once, convert via RGA, release the MPP frame.

        ``out_w`` / ``out_h`` / ``out_fmt`` override the constructor-time
        settings for this single frame. When all three are ``None`` the
        default BGR-at-source-size behaviour applies. Raises
        ``RuntimeError`` if RGA cannot service the requested format.
        """
        frame = self._decoder.recv_frame(timeout_ms=timeout_ms)
        if frame is None:
            return None
        try:
            bgr = self._rga_to_bgr(frame, out_w, out_h, out_fmt)
        finally:
            self._decoder.release_frame(frame)
        if bgr is not None:
            self.frame_pos += 1
        return bgr

    def _ensure_bgr_buffer(
        self,
        width: int,
        height: int,
        fmt: int = rga.RgaFormat.BGR_888,
    ) -> ctypes.Array:
        """Allocate (or reuse) the RGA dst buffer for ``(width, height, fmt)``.

        Buffer size = ``w * h * bytes_per_pixel(fmt)`` rounded up to whole
        bytes. BGR_888 / RGB_888 need 3 bytes per pixel; BGRA_8888 /
        RGBA_8888 need 4; NV12 / NV21 / GRAY need 1.5 (we allocate the
        full NV12 layout and slice off the Y plane for GRAY). The cache
        key includes the format so swapping between BGR and RGB doesn't
        reuse a wrongly-sized buffer.
        """
        key = (width, height, fmt)
        if getattr(self, "_bgr_buf_key", None) == key and self._bgr_buf is not None:
            return self._bgr_buf
        bpp = rga.bytes_per_pixel(fmt)
        # NV12 / NV21 are 1.5 BPP; round up so the chroma plane has its
        # full half-row of bytes (RGA writes the interleaved UV plane
        # at offset H*W).
        size = -(-int(width * height * bpp) // 1)   # ceil(w*h*bpp)
        self._bgr_buf = (ctypes.c_uint8 * size)()
        self._bgr_buf_key = key
        return self._bgr_buf

    def _rga_to_bgr(
        self,
        frame: DecodedFrame,
        out_w: Optional[int] = None,
        out_h: Optional[int] = None,
        out_fmt: Optional[str] = None,
    ) -> np.ndarray:
        """Convert an MPP frame directly to the requested output through RGA2."""
        if frame.fmt not in _MPP_TO_RGA_FMT:
            raise RuntimeError(
                f"MPP frame format {frame.fmt} is not supported by RGA input; "
                f"supported: {sorted(_MPP_TO_RGA_NAME.values())}"
            )
        target_w = out_w if out_w else (self._output_size[0] if self._output_size else frame.width)
        target_h = out_h if out_h else (self._output_size[1] if self._output_size else frame.height)
        target_fmt, bpp, layout = _resolve_output_format(out_fmt)
        try:
            return self._rga_nv12_to(
                frame, target_w, target_h, target_fmt, bpp, layout,
            )
        except RuntimeError as e:
            raise RuntimeError(
                f"RGA cannot convert MPP {_MPP_TO_RGA_NAME[frame.fmt]} "
                f"{frame.width}x{frame.height} to {rga.format_name(target_fmt)} "
                f"{target_w}x{target_h}: {e}"
            ) from e

    def _rga_nv12_to(
        self,
        frame: DecodedFrame,
        target_w: int,
        target_h: int,
        target_fmt: int,
        target_bpp: float,
        target_layout: str,
    ) -> np.ndarray:
        """RGA-only: NV12(NV21/I420) → ``target_fmt`` at ``target_w×H``.

        Raises ``RuntimeError`` if RGA rejects the call.
        """
        hor_stride = frame.hor_stride or frame.width
        ver_stride = frame.ver_stride or frame.height
        input_fmt = _MPP_TO_RGA_FMT[frame.fmt]

        if frame.fd is not None and frame.fd >= 0:
            src = rga.wrap_fd(
                frame.fd, frame.width, frame.height, input_fmt,
                wstride=hor_stride, hstride=ver_stride,
            )
        else:
            src = rga.wrap_virtual(
                frame.ptr, frame.width, frame.height, input_fmt,
                wstride=hor_stride, hstride=ver_stride,
            )

        rga_w = (target_w + 15) & ~15

        buf = self._ensure_bgr_buffer(rga_w, target_h, fmt=target_fmt)
        dst = rga.wrap_virtual(
            ctypes.addressof(buf), rga_w, target_h, target_fmt,
        )
        if (target_w, target_h) == (frame.width, frame.height) or target_layout != "HWC":
            rga.cvtcolor(src, dst)
        else:
            rga.resize(src, dst)
        rga.flush()
        out = _as_ndarray(buf, rga_w, target_h, target_bpp, target_layout)
        if rga_w != target_w:
            out = out[:, :target_w]
        return out


__all__ = ["VideoCapture"]