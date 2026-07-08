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
        which ffmpeg demuxes as ``rawvideo``.
    options:
        Optional dict of demuxer options forwarded to ffmpeg. For RTSP
        you typically want ``{"rtsp_transport": "tcp"}`` so the video
        stream isn't lost to UDP jitter. Ignored if the protocol doesn't
        accept it.
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
        options: Optional[dict] = None,
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

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Pull the next decoded frame from the source.

        For local files this returns ``(True, bgr)`` for every frame and
        ``(False, None)`` once the file is exhausted. For live streams
        (RTSP, HTTP, V4L2 cameras) it loops forever; the caller has to
        call :meth:`release` to stop.
        """
        if not self._opened or self._eof:
            return False, None

        # V4L2 cameras skip MPP entirely — the VPU has nothing to do on
        # raw pixels, and the V4L2 demuxer yields decoded frames directly.
        if getattr(self, "_is_v4l2", False):
            return self._read_v4l2()

        # Phase 1: feed more packets from PyAV and surface any frame
        # produced along the way. We yield the *first* decoded frame we
        # see — the caller drives pacing by calling read() again.
        for raw_packet in self._container.demux(self._stream):
            data = bytes(raw_packet)
            if not data:
                continue
            if not self._send(data, pts=int(raw_packet.pts or raw_packet.dts or 0)):
                return False, None

            frame = self._drain_one()
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
            return self.read()

        self._decoder.send_packet(b"", pts=0, eos=True)
        idle = 0
        while idle < 10:
            frame = self._recv_bgr(timeout_ms=50)
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

    def _read_v4l2(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Pull one decoded frame from a V4L2 capture node.

        Loops forever for live sources (cameras have no EOF); the caller
        must call :meth:`release` to stop. Each V4L2 buffer is converted
        to BGR via cv2 (single-pass software path) before returning.
        """
        cap = self._v4l2_cap
        if cap is None:
            return False, None
        buf = cap.dequeue(timeout_s=2.0)
        if buf is None:
            return False, None
        try:
            bgr = self._v4l2_to_bgr(buf)
        finally:
            cap.requeue(buf)
        if bgr is not None:
            self.frame_pos += 1
            return True, bgr
        return False, None

    def _v4l2_to_bgr(self, buf: "_v4l2_mod.V4L2BufferInfo") -> Optional[np.ndarray]:
        """Convert a V4L2 raw buffer into a contiguous BGR ndarray.

        Two paths, picked per call:

        - **RGA zero-copy** (default for NV12/NV21 with a valid DMA-BUF fd):
          ``rga.wrap_nv12_fd`` + ``rga.cvtcolor`` / ``rga.resize`` against the
          destination BGR888 mmap buffer. Honours ``output_size`` via the
          RGA scaler. Falls back to cv2 if RGA rejects the configuration
          (odd stride, unusual colour space, etc.).
        - **cv2 fallback** (for YUYV / UYVY / formats RGA doesn't understand,
          or when the driver didn't give us an EXPBUF fd).

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
            bgr = self._v4l2_rga_to_bgr(buf)
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
        if self._output_size is not None:
            bgr = cv2.resize(
                bgr, (self._output_size[0], self._output_size[1])
            )
        return bgr

    # -- RGA zero-copy helpers (shared with the MPP path) --------------------

    def _ensure_bgr_buffer(self, width: int, height: int) -> "ctypes.Array":
        """Allocate (or reuse) the RGA dst BGR buffer for ``(width, height)``."""
        # Reuse the existing attr if already set up by the MPP path.
        # If not, lazy-init here.
        if not hasattr(self, "_bgr_buf_key") or self._bgr_buf_key != (width, height):
            size = width * height * 3   # BGR888 = 3 bytes/pixel
            self._bgr_buf = (ctypes.c_uint8 * size)()
            self._bgr_buf_key = (width, height)
        return self._bgr_buf

    def _v4l2_rga_to_bgr(self, buf: "_v4l2_mod.V4L2BufferInfo") -> Optional[np.ndarray]:
        """Hardware NV12/NV21 → BGR via RGA2 from a V4L2 DMA-BUF fd.

        Mirrors ``_rga_to_bgr`` (the MPP path) but uses ``buf.plane_fds[0]``
        (the DMA-BUF fd exported via VIDIOC_EXPBUF) instead of
        ``frame.fd``. Honours ``self._output_size``: if set, dispatches to
        ``rga.resize`` instead of ``rga.cvtcolor`` so the scaling happens
        in the same hardware pass.
        """
        from . import rga

        if buf.plane_fds[0] < 0:
            return None

        out_w, out_h = self._output_size or (buf.width, buf.height)
        # Single-planar capture → mmap buffer is at offset 0; stride = width.
        # Multiplanar NV12: plane 0 (Y) has stride=width, plane 1 (UV)
        # stride=width. RGA's wrap_nv12_fd uses these strides directly.
        try:
            src = rga.wrap_nv12_fd(
                buf.plane_fds[0], buf.width, buf.height,
                hor_stride=buf.width, ver_stride=buf.height,
            )
        except RuntimeError:
            return None

        buf_storage = self._ensure_bgr_buffer(out_w, out_h)
        dst = rga.wrap_virtual(
            ctypes.addressof(buf_storage), out_w, out_h, rga.RgaFormat.BGR_888
        )
        try:
            if (out_w, out_h) == (buf.width, buf.height):
                rga.cvtcolor(src, dst)
            else:
                rga.resize(src, dst)
            rga.flush()
        except RuntimeError:
            return None

        return np.frombuffer(buf_storage, dtype=np.uint8).reshape(out_h, out_w, 3)

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

    def _send(self, data: bytes, pts: int) -> bool:
        """Submit one packet to MPP, draining on BUFFER_FULL.

        Returns ``False`` if the stream is exhausted (read() should stop).
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
                frame = self._drain_one(timeout_ms=_DRAIN_TIMEOUT_MS)
                if frame is not None:
                    # We have to come back to the caller with this frame
                    # before retrying the send — but we're inside _send.
                    # Stash it via _prefetch so read() picks it up next.
                    self._prefetched = frame
                    return True
                import time as _t
                _t.sleep(0.005)

    def _drain_one(
        self, timeout_ms: int = _DRAIN_TIMEOUT_MS
    ) -> Optional[np.ndarray]:
        """Pull at most one frame out of the decoder, returning BGR or None."""
        if hasattr(self, "_prefetched") and self._prefetched is not None:
            out, self._prefetched = self._prefetched, None
            return out
        return self._recv_bgr(timeout_ms=timeout_ms)

    def _recv_bgr(self, timeout_ms: int) -> Optional[np.ndarray]:
        """Run recv_frame once, convert NV12/NV21/I420 → BGR, release."""
        frame = self._decoder.recv_frame(timeout_ms=timeout_ms)
        if frame is None:
            return None
        try:
            if self._use_rga and frame.fmt in _MPP_TO_RGA_FMT:
                bgr = self._rga_to_bgr(frame)
            else:
                bgr = _nv12_to_bgr(frame)
        finally:
            self._decoder.release_frame(frame)
        if bgr is not None:
            self.frame_pos += 1
        return bgr

    def _ensure_bgr_buffer(self, width: int, height: int) -> ctypes.Array:
        """Allocate (or reuse) the RGA dst BGR buffer for ``(width, height)``."""
        key = (width, height)
        if self._bgr_buf_key == key and self._bgr_buf is not None:
            return self._bgr_buf
        size = width * height * 3   # BGR888 = 3 bytes/pixel
        self._bgr_buf = (ctypes.c_uint8 * size)()
        self._bgr_buf_key = key
        return self._bgr_buf

    def _rga_to_bgr(self, frame: DecodedFrame) -> Optional[np.ndarray]:
        """Convert ``frame`` to BGR via RGA2. Honors ``self._output_size``.

        Zero-copy on the input side: we prefer passing the MPP frame's
        DMA-BUF fd to RGA (``wrap_nv12_fd``) so both engines stay in
        DMA land. The virtual-address fallback is used only when MPP
        couldn't give us an fd (rare; e.g. a NORMAL alloc buffer).

        The output is a fresh numpy view on a ctypes-owned buffer that
        we re-use across frames of the same size, so steady-state
        allocations stay zero.
        """
        src_fmt = _MPP_TO_RGA_FMT[frame.fmt]
        out_w, out_h = self._output_size or (frame.width, frame.height)
        hor_stride = frame.hor_stride or frame.width
        ver_stride = frame.ver_stride or frame.height

        # Try the fd path first (true zero-copy, no cache management).
        # Fall back to the virtual-address path if MPP didn't give us
        # an fd (it can happen with NORMAL alloc buffers, but the
        # default DMA_HEAP allocator should always provide one).
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
                return _nv12_to_bgr(frame)

        buf = self._ensure_bgr_buffer(out_w, out_h)
        dst = rga.wrap_virtual(
            ctypes.addressof(buf), out_w, out_h, rga.RgaFormat.BGR_888,
        )
        try:
            if (out_w, out_h) == (frame.width, frame.height):
                rga.cvtcolor(src, dst)
            else:
                rga.resize(src, dst)
            rga.flush()
        except RuntimeError:
            # RGA rejected the configuration (e.g. odd stride, unusual
            # colour space). Fall back to the slow cv2 path so the caller
            # still gets a frame.
            return _nv12_to_bgr(frame)

        # numpy view onto the ctypes buffer. The buffer is reused across
        # frames of the same size, so callers who want to hold a frame
        # past the next read() should .copy() it themselves. The view is
        # valid until either release() or the next size change.
        bgr = np.frombuffer(buf, dtype=np.uint8).reshape(out_h, out_w, 3)
        return bgr


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