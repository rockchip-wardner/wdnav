"""VideoCapture — a thin OpenCV-style wrapper around the MPP hardware decoder.

Designed to mirror ``cv2.VideoCapture`` so existing OpenCV pipelines can be
retargeted at the Rockchip VPU with a one-line import swap. The heavy
lifting (PyAV demux, AVCC→Annex B, SPS/PPS injection, NV12→BGR) lives in
this module; the underlying ctypes binding to ``librockchip_mpp.so`` is
``rkv.mpp``.

Typical usage:

    import cv2
    from rkv import VideoCapture

    cap = VideoCapture("bus.mp4")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("video", frame)
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break
    cap.release()

The class is also a context manager, so a ``with`` block guarantees the
decoder is torn down on exit.
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Optional, Tuple

import av
import cv2
import numpy as np

from . import v4l2 as _v4l2_mod

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
# Same set as above — the formats _nv12_to_bgr() knows how to cvtColor.
# (Was referenced but never defined in the upstream source; restoring it
# here so the cv2 fallback path actually runs.)
_FRAME_BGR_FORMATS = frozenset(_MPP_TO_RGA_FMT.keys())


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
        For file sources the resize is done in hardware by the 2D RGA
        engine (RGA2) on the same SOC as the VPU; for V4L2 sources it's
        done by libswscale (still single-pass, software). ``None`` keeps
        the native frame size.
    use_rga:
        When ``True`` (default), hardware-color-convert NV12 → BGR via
        RGA2 (file sources only). Set ``False`` to fall back to
        ``cv2.cvtColor`` — useful when RGA isn't available, or when
        reproducing a bug that doesn't appear in the RGA path. Ignored
        for V4L2 sources (they always go through libswscale).
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
        use_rga: bool = True,
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
        self._use_rga: bool = use_rga
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
        if not input_format:
            input_format = "nv12"   # most ISP outputs default to NV12
        if not size:
            size = (640, 480)
        # framerate isn't used by V4L2Capture (driver keeps what it has);
        # recorded here only for the public property.
        self._v4l2_input_format = input_format
        self._v4l2_size = size
        self._v4l2_framerate = framerate

        self._v4l2_cap = _v4l2_mod.V4L2Capture(
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

        # The V4L2 buffer queue yields raw NV12/YUYV/etc. — we do the
        # BGR conversion in _v4l2_to_bgr via cv2.
        self._output_size: Optional[Tuple[int, int]] = output_size
        self._use_rga: bool = False
        self._decoder = None
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
        ``out_h`` / ``out_fmt`` overrides are forwarded to the conversion
        path (RGA when available, cv2 otherwise) so the caller can ask
        for a different size or pixel format frame-by-frame.
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
        buf: "_v4l2_mod.V4L2BufferInfo",
        out_w: Optional[int] = None,
        out_h: Optional[int] = None,
        out_fmt: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        """Convert a V4L2 raw buffer into a contiguous BGR ndarray.

        Two paths, picked per call:

        - **RGA zero-copy** (default for NV12/NV21 with a valid DMA-BUF fd):
          ``rga.wrap_nv12_fd`` + ``rga.cvtcolor`` / ``rga.resize`` against the
          destination mmap buffer. Honours per-call ``out_w``/``out_h``/``out_fmt``
          (resize + pixel format conversion in one hardware pass). Falls
          back to cv2 if RGA rejects the configuration.
        - **cv2 fallback** (for YUYV / UYVY / formats RGA doesn't understand,
          or when the driver didn't give us an EXPBUF fd). Uses the
          constructor's ``output_size`` for resize if set; per-call
          ``out_w``/``out_h`` are also honoured.

        In both paths the returned ndarray borrows the destination buffer
        memory, so callers who need to keep a frame past the next read()
        should ``.copy()`` it.
        """
        fmt = buf.pixel_format.upper()

        # -------- RGA fast path: NV12 / NV21 with DMA-BUF fd --------
        if (
            self._use_rga
            and fmt in ("NV12", "NV21")
            and buf.plane_fds
            and buf.plane_fds[0] >= 0
            and len(buf.plane_arrays) >= 2
        ):
            bgr = self._v4l2_rga_to_bgr(buf, out_w, out_h, out_fmt)
            if bgr is not None:
                return bgr
            # RGA rejected the configuration — fall through to cv2.

        # -------- cv2 fallback --------
        if fmt == "NV12":
            if len(buf.plane_arrays) != 2:
                return None
            y = buf.plane_arrays[0].reshape(buf.height, buf.width)
            uv = buf.plane_arrays[1].reshape(buf.height // 2, buf.width)
            h2 = buf.height * 3 // 2
            yuv = np.empty(h2 * buf.width, dtype=np.uint8)
            yuv[: buf.height * buf.width] = y.reshape(-1)
            yuv[buf.height * buf.width:] = uv.reshape(-1)
            yuv = yuv.reshape(h2, buf.width)
        elif fmt == "NV21":
            if len(buf.plane_arrays) != 2:
                return None
            # NV21 = same as NV12 but UV order swapped in the chroma plane.
            # cv2.COLOR_YUV2BGR_NV21 handles the swap for us if we pass the
            # same stacked buffer shape.
            y = buf.plane_arrays[0].reshape(buf.height, buf.width)
            uv = buf.plane_arrays[1].reshape(buf.height // 2, buf.width)
            h2 = buf.height * 3 // 2
            yuv = np.empty(h2 * buf.width, dtype=np.uint8)
            yuv[: buf.height * buf.width] = y.reshape(-1)
            yuv[buf.height * buf.width:] = uv.reshape(-1)
            yuv = yuv.reshape(h2, buf.width)
        elif fmt == "YUYV":
            raw = buf.plane_arrays[0]
            yuv = raw.reshape(buf.height, buf.width, 2)[..., 0]
            return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_YUYV)
        elif fmt == "UYVY":
            raw = buf.plane_arrays[0]
            yuv = raw.reshape(buf.height, buf.width, 2)[..., 1]
            return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_UYVY)
        else:
            return None

        if fmt in ("NV12", "NV21"):
            bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12 if fmt == "NV12" else cv2.COLOR_YUV2BGR_NV21)

        # Honour the same override chain as the RGA path.
        target_w = out_w if out_w else (self._output_size[0] if self._output_size else buf.width)
        target_h = out_h if out_h else (self._output_size[1] if self._output_size else buf.height)
        if (target_w, target_h) != (buf.width, buf.height):
            bgr = cv2.resize(bgr, (target_w, target_h))

        # cv2 can produce any 8-bit pixel format via cvtColor, but we
        # only natively return BGR / RGB shapes. Other formats would
        # need extra channels (BGRA/RGBA) — not implemented in the
        # fallback path; caller should rely on RGA for those.
        return bgr

    # -- RGA zero-copy helpers (shared with the MPP path) --------------------

    def _v4l2_rga_to_bgr(
        self,
        buf: "_v4l2_mod.V4L2BufferInfo",
        out_w: Optional[int] = None,
        out_h: Optional[int] = None,
        out_fmt: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        """Hardware NV12/NV21 → output via RGA2 from a V4L2 DMA-BUF fd.

        Mirrors ``_rga_to_bgr`` (the MPP path) but uses ``buf.plane_fds[0]``
        (the DMA-BUF fd exported via VIDIOC_EXPBUF) instead of
        ``frame.fd``. Honours per-call overrides in priority order:
        ``out_w`` / ``out_h`` → constructor's ``output_size`` → source size.
        Same chain applies to ``out_fmt`` (default BGR_888).

        Resize and pixel-format conversion happen in the same hardware
        pass when both ``out_w/out_h`` and ``out_fmt`` are provided.
        """
        from . import rga

        if not buf.plane_fds or buf.plane_fds[0] < 0:
            return None

        target_w = out_w if out_w else (self._output_size[0] if self._output_size else buf.width)
        target_h = out_h if out_h else (self._output_size[1] if self._output_size else buf.height)
        rga_fmt, bpp, layout = _resolve_output_format(out_fmt)
        # Same hardware-strategy as the MPP path: HWC dsts target BGR_888
        # via RGA, then a cv2 step handles the channel swap / alpha.
        if layout == "HWC":
            hw_fmt = rga.RgaFormat.BGR_888
            hw_bpp = 3
            hw_layout = "HWC"
        else:
            hw_fmt = rga_fmt
            hw_bpp = bpp
            hw_layout = layout
        try:
            src = rga.wrap_nv12_fd(
                buf.plane_fds[0], buf.width, buf.height,
                hor_stride=buf.width, ver_stride=buf.height,
            )
        except RuntimeError:
            return None

        # Same 16-aligned dst-stride dance as the MPP path: RGA2
        # rejects dst.wstride values that aren't multiples of 16
        # for RGB-family output. Round up, allocate the buffer
        # with the aligned width, and slice the result back to
        # target_w before returning.
        rga_w = (target_w + 15) & ~15

        buf_storage = self._ensure_bgr_buffer(rga_w, target_h, fmt=hw_fmt)
        dst = rga.wrap_virtual(
            ctypes.addressof(buf_storage), rga_w, target_h, hw_fmt
        )
        try:
            if (target_w, target_h) == (buf.width, buf.height) or hw_layout != "HWC":
                rga.cvtcolor(src, dst)
            else:
                rga.resize(src, dst)
            rga.flush()
        except RuntimeError:
            return None

        if layout == "HWC" and rga_fmt != rga.RgaFormat.BGR_888:
            bgr_view = np.frombuffer(buf_storage, dtype=np.uint8).reshape(
                target_h, rga_w, 3
            )[:, :target_w]
            if rga_fmt == rga.RgaFormat.RGB_888:
                return cv2.cvtColor(bgr_view, cv2.COLOR_BGR2RGB)
            if rga_fmt == rga.RgaFormat.BGRA_8888:
                return cv2.cvtColor(bgr_view, cv2.COLOR_BGR2BGRA)
            if rga_fmt == rga.RgaFormat.RGBA_8888:
                return cv2.cvtColor(bgr_view, cv2.COLOR_BGR2RGBA)

        out = _as_ndarray(buf_storage, rga_w, target_h, hw_bpp, hw_layout)
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
        """Run recv_frame once, convert NV12/NV21/I420 → output, release.

        ``out_w`` / ``out_h`` / ``out_fmt`` override the constructor-time
        settings for this single frame. When all three are ``None`` we
        behave as before (BGR at the source size).
        """
        frame = self._decoder.recv_frame(timeout_ms=timeout_ms)
        if frame is None:
            return None
        try:
            if self._use_rga and frame.fmt in _MPP_TO_RGA_FMT:
                bgr = self._rga_to_bgr(frame, out_w, out_h, out_fmt)
            else:
                bgr = _nv12_to_bgr(frame)
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
    ) -> Optional[np.ndarray]:
        """Convert ``frame`` to output via RGA2. Honors per-call overrides.

        Strategy (in order of preference):
          1. RGA writes the requested target format directly (NV12 /
             NV21 / GRAY). Works on most RGA cores for BGR/RGB; tested
             case-by-case below.
          2. RGA writes BGR_888 (always supported, cheapest HWC) and
             a single cv2.cvtColor post-step does the channel swap
             (BGR→RGB) / alpha insertion (BGR→BGRA/RGBA).
          3. If RGA fails outright, fall back to the cv2 NV12→BGR path
             and apply the same cv2 post-step.

        For YUV / GRAY targets with RGA unavailable, we re-use the
        cv2 BGR result and convert via cv2 (BGR→NV12 / BGR→GRAY).
        Slower but always works.
        """
        # Apply precedence: per-call > constructor > source size
        target_w = out_w if out_w else (self._output_size[0] if self._output_size else frame.width)
        target_h = out_h if out_h else (self._output_size[1] if self._output_size else frame.height)
        rga_fmt, bpp, layout = _resolve_output_format(out_fmt)

        # Strategy: try RGA all the way; fall back to cv2 only when RGA
        # raises. RGA2 doesn't support every NV12→X conversion
        # (e.g. NV12→BGRA fails on this machine's RGA2), so the
        # exception path is real. On RGA3 (RK3588+) every conversion
        # succeeds and the cv2 path is never taken.

        # Step 1: try RGA NV12 → target format directly. RGA2_Enhance
        # and RGA3 both support every HWC family (BGR_888, RGB_888,
        # BGRA_8888, RGBA_8888), so on most callers this step
        # produces the answer with no post-processing. rga.output_
        # formats()'s printed list omits BGR_888 / BGRA_8888 even
        # though the cores handle them, so don't trust that string
        # as a capability check — try and see.
        bgr_ndarray: Optional[np.ndarray] = None
        rga_native_format = False
        try:
            bgr_ndarray = self._rga_nv12_to(
                frame, target_w, target_h, rga_fmt, bpp, layout,
            )
            rga_native_format = True
        except RuntimeError:
            bgr_ndarray = None

        # Step 2: try RGA NV12 → BGR_888. Step 1 failed (usually
        # because the dst format is exotic — NV12 → RGBA on cores
        # where the alpha path isn't wired, or a stride that the
        # core's allocator can't satisfy). BGR_888 is the cheapest
        # universally-supported HWC format and what the post-
        # process helpers expect. _rga_nv12_to handles the 16-aligned
        # dst-width requirement.
        if bgr_ndarray is None:
            try:
                bgr_ndarray = self._rga_nv12_to(
                    frame, target_w, target_h, rga.RgaFormat.BGR_888, 3, "HWC",
                )
            except RuntimeError:
                bgr_ndarray = None

        # Step 3: if RGA is completely unavailable, fall back to cv2.
        if bgr_ndarray is None:
            # Both RGA attempts failed; surface that to the operator.
            # On RGA2 this happens for rare combinations (e.g. NV12
            # → GRAY at odd widths, or sizes beyond the core's
            # maximum output). The cv2 path is correct but ~5-10x
            # slower than the RGA path, so make it visible.
            import sys
            print(
                f"[rkv] RGA failed for NV12→{rga.format_name(rga_fmt)} at "
                f"{target_w}x{target_h}; falling back to cv2",
                file=sys.stderr, flush=True,
            )
            bgr_ndarray = _nv12_to_bgr(frame)
            if bgr_ndarray is None:
                return None
            # Resize via cv2 if needed.
            if (bgr_ndarray.shape[1], bgr_ndarray.shape[0]) != (target_w, target_h):
                bgr_ndarray = cv2.resize(
                    bgr_ndarray, (target_w, target_h),
                )

        # If RGA already produced the requested target format, we
        # already have the answer. No post-process needed.
        if rga_native_format and (rga_fmt, layout) in (
            (rga.RgaFormat.BGR_888, "HWC"),
            (rga.RgaFormat.RGB_888, "HWC"),
            (rga.RgaFormat.YCbCr_420_SP, "YUV"),
            (rga.RgaFormat.YCrCb_420_SP, "YUV"),
            (rga.RgaFormat.YCbCr_400, "GRAY"),
        ):
            return bgr_ndarray

        # RGA gave us BGR (Step 1 with a BGR-family rga_fmt, or Step
        # 2's BGR_888 fallback, or the cv2 path). Convert to the
        # requested target format via cv2. The BGR→X cvtColor codes
        # / _bgr_to_nv12 / _bgr_to_nv21 helpers all expect BGR
        # input — by the time we reach here, that's what we have.
        if rga_fmt == rga.RgaFormat.BGR_888:
            return bgr_ndarray
        if rga_fmt == rga.RgaFormat.RGB_888:
            return cv2.cvtColor(bgr_ndarray, cv2.COLOR_BGR2RGB)
        if rga_fmt == rga.RgaFormat.BGRA_8888:
            return cv2.cvtColor(bgr_ndarray, cv2.COLOR_BGR2BGRA)
        if rga_fmt == rga.RgaFormat.RGBA_8888:
            return cv2.cvtColor(bgr_ndarray, cv2.COLOR_BGR2RGBA)
        if rga_fmt == rga.RgaFormat.YCbCr_420_SP:
            return _bgr_to_nv12(bgr_ndarray)
        if rga_fmt == rga.RgaFormat.YCrCb_420_SP:
            return _bgr_to_nv21(bgr_ndarray)
        if rga_fmt == rga.RgaFormat.YCbCr_400:
            return cv2.cvtColor(bgr_ndarray, cv2.COLOR_BGR2GRAY)
        return bgr_ndarray

    def _rga_nv12_to(
        self,
        frame: DecodedFrame,
        target_w: int,
        target_h: int,
        target_fmt: int,
        target_bpp: float,
        target_layout: str,
    ) -> Optional[np.ndarray]:
        """RGA-only: NV12(NV21/I420) → ``target_fmt`` at ``target_w×H``.

        Returns the ndarray view, or raises RuntimeError if RGA rejects
        the call. Caller decides what to do on failure.
        """
        hor_stride = frame.hor_stride or frame.width
        ver_stride = frame.ver_stride or frame.height

        src = None
        if frame.fd is not None and frame.fd >= 0:
            try:
                src = rga.wrap_nv12_fd(
                    frame.fd, frame.width, frame.height,
                    hor_stride=hor_stride, ver_stride=ver_stride,
                )
            except RuntimeError:
                src = None
        if src is None:
            try:
                src = rga.wrap_nv12(
                    frame.ptr, frame.width, frame.height,
                    hor_stride=hor_stride, ver_stride=ver_stride,
                )
            except RuntimeError:
                raise

        # RGA2 (and several RGA3 firmwares) require the dst width /
        # stride to be 16-aligned for RGB-family output formats —
        # imcheck rejects with "rgb888 width stride should be 16
        # aligned" otherwise (verified on the RGA2_Enhance + RGA3
        # core in im2d v1.10.4). The source side is fine: MPP
        # already hands us hor_stride=1376 for a 1366-wide H.264
        # frame, which is 16-aligned. Round the dst up to a
        # 16-multiple, allocate the buffer with that aligned width,
        # and slice the result back to the caller's target_w before
        # returning.
        rga_w = (target_w + 15) & ~15

        buf = self._ensure_bgr_buffer(rga_w, target_h, fmt=target_fmt)
        dst = rga.wrap_virtual(
            ctypes.addressof(buf), rga_w, target_h, target_fmt,
        )
        # Same-size uses cvtcolor; otherwise resize. YUV / GRAY dsts
        # always use cvtcolor because imresize_t rejects YCbCr_400.
        if (target_w, target_h) == (frame.width, frame.height) or target_layout != "HWC":
            rga.cvtcolor(src, dst)
        else:
            rga.resize(src, dst)
        rga.flush()
        # Strip the right-edge padding columns (rga_w - target_w of
        # them) so the returned ndarray is exactly the requested
        # resolution. _as_ndarray reshapes the underlying storage
        # using rga_w, so the column slice is what trims the view.
        out = _as_ndarray(buf, rga_w, target_h, target_bpp, target_layout)
        if rga_w != target_w:
            out = out[:, :target_w]
        return out


def _bgr_to_nv12(bgr: np.ndarray) -> np.ndarray:
    """Convert a BGR ndarray to NV12 layout (Y plane + interleaved UV).

    Output shape: ``(H * 3 // 2, W)`` so callers can hand it straight
    to another YUV codec / encoder. Returns a fresh contiguous ndarray.
    """
    h, w = bgr.shape[:2]
    yuv_i420 = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)
    # cv2's I420 buffer: Y plane fills the first h rows; the next
    # h/2 rows hold U (left w/2 cols) and V (right w/2 cols) side by
    # side. NV12 wants the Y plane then interleaved UV (cols U,V,U,V).
    y = yuv_i420[:h, :]
    uv_split = yuv_i420[h:, :]
    u = uv_split[:, :w // 2]
    v = uv_split[:, w // 2:]
    flat = np.empty(h * w + (h // 2) * w, dtype=np.uint8)
    flat[:h * w] = y.reshape(-1)
    uv = flat[h * w:].reshape(h // 2, w)
    uv[:, 0::2] = u
    uv[:, 1::2] = v
    return flat.reshape(h * 3 // 2, w)


def _bgr_to_nv21(bgr: np.ndarray) -> np.ndarray:
    """Convert a BGR ndarray to NV21 layout (Y plane + interleaved VU)."""
    h, w = bgr.shape[:2]
    yuv_i420 = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)
    y = yuv_i420[:h, :]
    uv_split = yuv_i420[h:, :]
    v = uv_split[:, :w // 2]
    u = uv_split[:, w // 2:]
    flat = np.empty(h * w + (h // 2) * w, dtype=np.uint8)
    flat[:h * w] = y.reshape(-1)
    vu = flat[h * w:].reshape(h // 2, w)
    vu[:, 0::2] = v   # NV21 = V then U
    vu[:, 1::2] = u
    return flat.reshape(h * 3 // 2, w)


def _nv12_to_bgr(frame: DecodedFrame) -> Optional[np.ndarray]:
    """Convert an MPP frame's YUV buffer into a contiguous BGR ndarray.

    Supports NV12 (most common), NV21 and I420 — anything else is
    dropped (returning ``None``). The output ndarray is a fresh
    allocation owned by OpenCV; the source MPP buffer is read in-place
    and can be released as soon as this function returns.

    No intermediate copy of the YUV data is needed: ``cv2.cvtColor``
    handles the non-contiguous YUV slice produced by stride padding
    directly, and the frame is released only after cv2 returns, so the
    source memory stays valid for the duration of the conversion.
    """
    if frame.ptr == 0 or frame.buf_size == 0:
        return None
    if frame.fmt not in _FRAME_BGR_FORMATS:
        return None

    w, h = frame.width, frame.height
    hor_stride, ver_stride = frame.hor_stride, frame.ver_stride
    total = hor_stride * ver_stride * 3 // 2
    if total > frame.buf_size:
        total = frame.buf_size

    raw = np.ctypeslib.as_array(
        (ctypes.c_uint8 * total).from_address(frame.ptr)
    ).reshape(ver_stride * 3 // 2, hor_stride)

    # cv2's NV12/NV21/I420 decoders want a (h*1.5, w) buffer; the rest
    # of the MppBuffer is decoder padding. Pass the (possibly
    # non-contiguous) slice directly — cv2 handles strides internally
    # and the output is always a fresh contiguous BGR allocation.
    yuv = raw[: h * 3 // 2, :w]
    if frame.fmt == 0:
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
    if frame.fmt == 5:
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV21)
    if frame.fmt == 4:
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
    return None


__all__ = ["VideoCapture"]