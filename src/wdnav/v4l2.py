"""V4L2 capture via the python3-v4l2 PyPI bindings (``pip install v4l2``).

This is a thin wrapper layer on top of the third-party ``v4l2`` PyPI
package — that package supplies the ioctl numbers, the ``v4l2_fourcc``
helper, and most of the single-planar C structs we need. The
``v4l2`` 0.2 release is from 2014 and is **broken on Python 3.11**
(it uses ``range(...) + [...]`` which is a Python 2 idiom); a one-line
sed patch on site-packages fixes the import, see ``samples/test_camera_capture.py``
for the patch.

We extend the package's coverage in two ways the upstream doesn't:

1. **Multiplanar V4L2 devices** (V4L2_CAP_VIDEO_CAPTURE_MPLANE) like
   Rockchip's ``/dev/video11`` rkisp_mainpath, which is what most
   embedded camera pipelines actually expose. The package's
   ``v4l2_buffer`` union only models offset/userptr — no planes.
2. **Buffer-queue management**: request N buffers via VIDIOC_REQBUFS,
   mmap each, run VIDIOC_QBUF/DQBUF, and re-queue automatically. Just
   ``start(); while True: dequeue()``.

Typical usage::

    from wdnav.v4l2 import V4L2Capture

    cap = V4L2Capture("/dev/video11", width=1920, height=1080,
                       pixel_format="nv12", num_buffers=4)
    cap.start()
    try:
        for _ in range(30):
            buf = cap.dequeue(timeout_s=2.0)
            if buf is None:
                break
            # buf.plane_arrays[0] is an (H, W) uint8 view of plane 0 (Y)
            # For NV12 the second plane (UV) is buf.plane_arrays[1]
            # Hand the buffer to RGA via the fd (rga.wrap_fd(...)) — opencv
            # is no longer used; RGA is the only conversion backend.
            cap.requeue(buf)
    finally:
        cap.close()
"""

from __future__ import annotations

import ctypes
import fcntl
import mmap
import os
import select
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

import v4l2 as _v4l2


# ---------------------------------------------------------------------------
# Constants the upstream `v4l2` package is missing
# ---------------------------------------------------------------------------

# /usr/include/linux/videodev2.h
V4L2_CAP_VIDEO_CAPTURE_MPLANE = 0x1000
# The upstream `v4l2` package predates the *_MPLANE types and uses
# `range(1, 9) + [0x80]` to define V4L2_BUF_TYPE_*, so it stops at 8.
# Modern kernel defines VIDEO_CAPTURE_MPLANE = 9.
V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE = 9

# field value used to request the driver pick one; many drivers only
# support _NONE and reject _ANY
V4L2_FIELD_ANY = _v4l2.V4L2_FIELD_ANY
V4L2_FIELD_NONE = _v4l2.V4L2_FIELD_NONE
V4L2_MEMORY_MMAP = _v4l2.V4L2_MEMORY_MMAP

# Pixel format FourCCs we accept (string -> integer FourCC).
_PIXEL_FORMATS = {
    "nv12": _v4l2.V4L2_PIX_FMT_NV12,
    "nv21": _v4l2.V4L2_PIX_FMT_NV21,
    "yuv420": _v4l2.V4L2_PIX_FMT_YUV420,    # I420
    "yuyv": _v4l2.V4L2_PIX_FMT_YUYV,
    "uyvy": _v4l2.V4L2_PIX_FMT_UYVY,
    "bgr24": _v4l2.V4L2_PIX_FMT_BGR24,
    "rgb24": _v4l2.V4L2_PIX_FMT_RGB24,
    "mjpeg": _v4l2.V4L2_PIX_FMT_MJPEG,
}


def fourcc_to_str(fmt: int) -> str:
    """Decode a V4L2 FourCC int back to a 4-char string (LE)."""
    # V4L2 FourCCs are little-endian; bytes = [a,b,c,d] -> int = a | b<<8 | c<<16 | d<<24
    return "".join(chr((fmt >> (8 * i)) & 0xFF) for i in range(4))


def str_to_fourcc(s: str) -> int:
    """Pack a 4-char string into a V4L2 FourCC int (LE). Case-insensitive."""
    if len(s) != 4:
        raise ValueError("FourCC must be exactly 4 characters")
    # V4L2 FourCCs are conventionally uppercase; "nv12" should work the same
    # as "NV12". The kernel drivers validate exact character case and reject
    # lowercase, so normalize here.
    return _v4l2.v4l2_fourcc(*s.upper())


# ---------------------------------------------------------------------------
# Structs (extended for multiplanar; v4l2 package is incomplete here)
# ---------------------------------------------------------------------------


class v4l2_plane_pix_format(ctypes.Structure):
    """<linux/videodev2.h> struct v4l2_plane_pix_format."""

    _fields_ = [
        ("sizeimage", ctypes.c_uint),
        ("bytesperline", ctypes.c_uint),
        ("reserved", ctypes.c_uint16 * 6),
    ]


class v4l2_pix_format_mplane(ctypes.Structure):
    """<linux/videodev2.h> struct v4l2_pix_format_mplane."""

    _fields_ = [
        ("width", ctypes.c_uint),
        ("height", ctypes.c_uint),
        ("pixelformat", ctypes.c_uint),
        ("field", ctypes.c_uint),
        ("colorspace", ctypes.c_uint),
        ("plane_fmt", v4l2_plane_pix_format * 8),   # VIDEO_MAX_PLANES
        ("num_planes", ctypes.c_ubyte),
        ("flags", ctypes.c_ubyte),
        ("ycbcr_enc", ctypes.c_ubyte),              # union with hsv_enc
        ("quantization", ctypes.c_ubyte),
        ("xfer_func", ctypes.c_ubyte),
        ("reserved", ctypes.c_ubyte * 7),
    ]


class v4l2_format(ctypes.Structure):
    """<linux/videodev2.h> struct v4l2_format.

    The package's ``v4l2_format`` struct only defines pix/win/vbi/sliced/raw_data
    in its union — no pix_mp. We redefine the full union locally so multiplanar
    devices work.

    Two layout tricks needed to match the kernel's 208-byte v4l2_format on
    this 6.1 kernel:

    1. **4-byte pad after `type`** — the kernel aligns the `fmt` union to 8
       bytes (because v4l2_window contains 8-byte pointers), so ``type``
       sits at offset 0 and ``fmt`` starts at offset 8, not 4. Without
       this pad every field reads 4 bytes too early.
    2. **raw_data[204] in the union** — keeps the union 204 bytes wide so
       the total struct is 208, matching the size encoded in the
       ioctl numbers (``VIDIOC_G_FMT`` = ``0xc0d05604``, size field = 0xd0
       = 208). The kernel validates the size and returns EINVAL otherwise.
    """

    class _u(ctypes.Union):
        _fields_ = [
            ("pix", _v4l2.v4l2_pix_format),
            ("pix_mp", v4l2_pix_format_mplane),
            ("raw_data", ctypes.c_ubyte * 204),
        ]

    _anonymous_ = ("u",)
    _fields_ = [
        ("type", ctypes.c_uint),
        ("_pad", ctypes.c_uint),
        ("u", _u),
    ]


class v4l2_plane(ctypes.Structure):
    """<linux/videodev2.h> struct v4l2_plane (multiplanar buffer plane)."""

    _fields_ = [
        ("bytesused", ctypes.c_uint),
        ("length", ctypes.c_uint),
        ("mem_offset", ctypes.c_uint),                # union member
        ("userptr", ctypes.c_ulong),
        ("fd", ctypes.c_int),
        ("data_offset", ctypes.c_uint),
        ("reserved", ctypes.c_uint * 11),
    ]


class v4l2_buffer_mp(ctypes.Structure):
    """``v4l2_buffer`` for multiplanar capture.

    We define this locally because the upstream package's ``v4l2_buffer`` only
    has offset/userptr in its union. Layout matches the kernel header up to
    the planes union member.
    """

    class _u(ctypes.Union):
        _fields_ = [
            ("offset", ctypes.c_uint),
            ("userptr", ctypes.c_ulong),
            ("planes", ctypes.c_void_p),               # v4l2_plane* filled by kernel
            ("fd", ctypes.c_int),
        ]

    _anonymous_ = ("u",)
    _fields_ = [
        ("index", ctypes.c_uint),
        ("type", ctypes.c_uint),
        ("bytesused", ctypes.c_uint),
        ("flags", ctypes.c_uint),
        ("field", ctypes.c_uint),
        ("timestamp", _v4l2.timeval),
        ("timecode", _v4l2.v4l2_timecode),
        ("sequence", ctypes.c_uint),
        ("memory", ctypes.c_uint),
        ("u", _u),
        ("length", ctypes.c_uint),
        ("reserved2", ctypes.c_uint),
        ("request_fd", ctypes.c_int),
    ]


# ---------------------------------------------------------------------------
# EXPBUF — export each mmap'd buffer as a DMA-BUF fd
# ---------------------------------------------------------------------------


class v4l2_exportbuffer(ctypes.Structure):
    """<linux/videodev2.h> struct v4l2_exportbuffer.

    For multiplanar capture, call this once per plane (with ``plane`` set
    to the plane index 0..N-1); each call returns the DMA-BUF fd for that
    plane. For single-planar capture, just leave ``plane`` at 0.
    """

    _fields_ = [
        ("type", ctypes.c_uint),
        ("index", ctypes.c_uint),
        ("plane", ctypes.c_uint),
        ("flags", ctypes.c_uint),
        ("fd", ctypes.c_int),
        ("reserved", ctypes.c_uint * 11),
    ]


# Compute VIDIOC_EXPBUF ourselves — the upstream package predates this ioctl
# (0.2 from 2014 doesn't define it). nr=16, _IOWR('V', 16, 68) on this kernel.
_VIDIOC_EXPBUF_NR = 16
_VIDIOC_EXPBUF_SIZE = ctypes.sizeof(v4l2_exportbuffer)
VIDIOC_EXPBUF = (
    (3 << 30)                  # _IOC_READ | _IOC_WRITE
    | (_VIDIOC_EXPBUF_SIZE << 16)
    | (ord("V") << 8)
    | _VIDIOC_EXPBUF_NR
)


# ---------------------------------------------------------------------------
# Buffer info returned to callers
# ---------------------------------------------------------------------------


@dataclass
class V4L2BufferInfo:
    """One dequeued frame from the V4L2 capture queue.

    ``plane_arrays`` is a tuple of ndarrays — one per plane — each a
    ``(height, stride)`` uint8 view onto the mmap'd kernel buffer (NO copy).
    The array is **valid until requeue()**; copy if you need to hold it past
    the next DQBUF. ``plane_bytes_used`` is the kernel-reported byte count
    per plane (often > width for NV12 / YUYV where stride padding exists).

    ``plane_fds`` is the matching tuple of **DMA-BUF file descriptors** — one
    per plane. These are exported once during :class:`V4L2Capture` setup
    (via ``VIDIOC_EXPBUF``) and reused for every DQBUF on the same buffer
    index. Hand them to ``wdnav.rga.wrap_fd`` / ``wrap_nv12_fd`` for hardware
    color conversion / resize without going through CPU cache.

    Single-planar buffers always have a single-element ``plane_fds`` tuple.
    """

    index: int
    plane_arrays: Tuple[np.ndarray, ...]
    plane_bytes_used: Tuple[int, ...]
    plane_fds: Tuple[int, ...] = ()            # DMA-BUF fds (one per plane)
    width: int = 0
    height: int = 0
    pixel_format: str = ""
    timestamp: float = 0.0
    sequence: int = 0


# ---------------------------------------------------------------------------
# V4L2 capture
# ---------------------------------------------------------------------------


class V4L2Capture:
    """Open ``/dev/videoX``, negotiate format, mmap a buffer queue, capture frames.

    Supports both **single-planar** (V4L2_CAP_VIDEO_CAPTURE) and **multiplanar**
    (V4L2_CAP_VIDEO_CAPTURE_MPLANE) capture — matches automatically based
    on what the device reports.

    Parameters
    ----------
    path:
        Path to the capture node, e.g. ``"/dev/video11"``.
    width, height:
        Requested resolution. After ``VIDIOC_S_FMT`` the actual width/height
        may differ (some sensors round up to a multiple of 16); read them
        back via the .width / .height properties.
    pixel_format:
        FourCC string. Common values: ``"nv12"``, ``"nv21"``, ``"yuyv"``,
        ``"uyvy"``, ``"bgr24"``, ``"mjpeg"``, ``"yuv420"``.
    num_buffers:
        Number of mmap'd capture buffers (≥ 2). 4 is a good default.
    """

    def __init__(
        self,
        path: str,
        *,
        width: int,
        height: int,
        pixel_format: str,
        num_buffers: int = 4,
    ) -> None:
        self._path = path
        # O_RDWR + O_NONBLOCK so select() works in dequeue()
        self._fd = os.open(path, os.O_RDWR | os.O_NONBLOCK)
        try:
            cap = _ioctl_capture(self._fd, _v4l2.VIDIOC_QUERYCAP, _v4l2.v4l2_capability())
            driver = cap.driver.decode("ascii", "replace").rstrip("\x00")
            card = cap.card.decode("ascii", "replace").rstrip("\x00")
            self._driver = driver
            self._card = card
            self._caps = cap.capabilities
            self._is_multiplanar = bool(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE_MPLANE)
            self._buf_type = (
                V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE
                if self._is_multiplanar
                else _v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
            )
            self._memory = V4L2_MEMORY_MMAP

            fmt_int = str_to_fourcc(pixel_format)
            self._fmt = _negotiate_format(
                self._fd, self._is_multiplanar, width, height, fmt_int
            )
            self._pixel_format_str = pixel_format

            # Allocate + mmap buffers
            req = _v4l2.v4l2_requestbuffers()
            req.count = num_buffers
            req.type = self._buf_type
            req.memory = self._memory
            _ioctl_capture(self._fd, _v4l2.VIDIOC_REQBUFS, req)
            actual_count = req.count
            if actual_count < 2:
                raise RuntimeError(
                    "VIDIOC_REQBUFS returned count={0}, need at least 2".format(actual_count)
                )

            self._buffers: List[_PlaneBuf] = []
            self._no_export = False
            for i in range(actual_count):
                self._buffers.append(self._map_buffer(i))
            self._num_buffers = len(self._buffers)
        except Exception:
            try:
                os.close(self._fd)
            except OSError:
                pass
            raise

    # -- introspection -----------------------------------------------------

    @property
    def path(self) -> str:
        return self._path

    @property
    def driver(self) -> str:
        return self._driver

    @property
    def card(self) -> str:
        return self._card

    @property
    def is_multiplanar(self) -> bool:
        return self._is_multiplanar

    @property
    def width(self) -> int:
        if self._is_multiplanar:
            return self._fmt.pix_mp.width
        return self._fmt.pix.width

    @property
    def height(self) -> int:
        if self._is_multiplanar:
            return self._fmt.pix_mp.height
        return self._fmt.pix.height

    @property
    def pixel_format(self) -> str:
        if self._is_multiplanar:
            return fourcc_to_str(self._fmt.pix_mp.pixelformat)
        return fourcc_to_str(self._fmt.pix.pixelformat)

    @property
    def num_planes(self) -> int:
        if self._is_multiplanar:
            return self._fmt.pix_mp.num_planes
        return 1

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        """VIDIOC_STREAMON."""
        t = ctypes.c_uint(self._buf_type)
        _ioctl_capture(self._fd, _v4l2.VIDIOC_STREAMON, t)
        # Queue every buffer so the driver can start filling them.
        for i in range(self._num_buffers):
            self._enqueue_index(i)

    def stop(self) -> None:
        """VIDIOC_STREAMOFF. Idempotent."""
        t = ctypes.c_uint(self._buf_type)
        # Best-effort — some drivers return EINVAL if not streaming.
        try:
            _ioctl_capture(self._fd, _v4l2.VIDIOC_STREAMOFF, t)
        except OSError:
            pass

    def close(self) -> None:
        """STREAMOFF, munmap, close. Safe to call multiple times."""
        if self._fd < 0:
            return
        try:
            self.stop()
        finally:
            for pb in self._buffers:
                pb.close()
            self._buffers.clear()
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = -1

    # -- context manager ---------------------------------------------------

    def __enter__(self) -> "V4L2Capture":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    # -- dequeue / requeue -------------------------------------------------

    def dequeue(self, timeout_s: float = 1.0) -> Optional[V4L2BufferInfo]:
        """Block up to ``timeout_s`` for a captured frame, return its info.

        Returns ``None`` on timeout. The returned :class:`V4L2BufferInfo`'s
        numpy views are valid until you call :meth:`requeue` for that index.
        """
        # Wait for the FD to signal "at least one buffer dequeueable".
        if self._fd < 0:
            return None
        try:
            r, _, _ = select.select([self._fd], [], [], timeout_s)
        except (OSError, ValueError):
            return None
        if not r:
            return None

        if self._is_multiplanar:
            return self._dequeue_mp()
        return self._dequeue_sp()

    def requeue(self, buf: V4L2BufferInfo) -> None:
        """Re-enqueue a dequeued buffer for the next capture cycle."""
        self._enqueue_index(buf.index)

    # -- internals ---------------------------------------------------------

    def _map_buffer(self, index: int) -> "_PlaneBuf":
        """VIDIOC_QUERYBUF + mmap + VIDIOC_EXPBUF for a single buffer."""
        if self._is_multiplanar:
            buf = v4l2_buffer_mp()
            buf.type = self._buf_type
            buf.memory = self._memory
            buf.index = index
            buf.length = 8  # VIDEO_MAX_PLANES
            planes_arr = (v4l2_plane * 8)()
            buf.u.planes = ctypes.cast(
                ctypes.pointer(planes_arr), ctypes.c_void_p
            ).value
            _ioctl_capture(self._fd, _v4l2.VIDIOC_QUERYBUF, buf)

            # Collect (mem_offset, length) per plane
            planes = []
            for p in range(buf.length):
                pl = planes_arr[p]
                if pl.length > 0:
                    planes.append((pl.mem_offset, pl.length))
            pb = _PlaneBuf(index, planes, self._fd, self._is_multiplanar, planes_arr)
            # Export a DMA-BUF fd per non-empty plane (RGA can read these
            # directly without touching the CPU cache).
            for p in range(len(planes)):
                pb._fds.append(self._export_buffer(index, plane=p))
            return pb
        else:
            buf = _v4l2.v4l2_buffer()
            buf.type = self._buf_type
            buf.memory = self._memory
            buf.index = index
            _ioctl_capture(self._fd, _v4l2.VIDIOC_QUERYBUF, buf)
            pb = _PlaneBuf(
                index,
                [(buf.m.offset, buf.length)],
                self._fd,
                self._is_multiplanar,
                None,
            )
            pb._fds.append(self._export_buffer(index, plane=0))
            return pb

    def _export_buffer(self, index: int, plane: int = 0) -> int:
        """VIDIOC_EXPBUF → return the DMA-BUF fd for ``plane`` of buffer ``index``.

        VIDIOC_EXPBUF gives us a dup'd reference; the caller (us) must
        ``os.close(fd)`` when done. Reusing the same buffer index for
        every frame means we can cache the fd for the lifetime of the
        queue and skip this syscall on the hot path.
        """
        eb = v4l2_exportbuffer()
        eb.type = self._buf_type
        eb.index = index
        eb.plane = plane
        try:
            _ioctl_capture(self._fd, VIDIOC_EXPBUF, eb)
        except OSError as e:
            # Some drivers don't support EXPBUF (e.g. virtual loopback);
            # return -1 so callers fall back to the mmap'd virtual address.
            self._no_export = True
            return -1
        if eb.fd < 0:
            return -1
        return eb.fd

    def _dequeue_sp(self) -> Optional[V4L2BufferInfo]:
        buf = _v4l2.v4l2_buffer()
        buf.type = self._buf_type
        buf.memory = self._memory
        try:
            _ioctl_capture(self._fd, _v4l2.VIDIOC_DQBUF, buf)
        except OSError as e:
            if e.errno in (11, 35):  # EAGAIN / EAGAIN-on-BSD
                return None
            raise
        pb = self._buffers[buf.index]
        # bytesused lives at offset 2 in the same struct (uint32 #3) — but
        # easier to reach via the field name where upstream defines it.
        bytes_used = buf.bytesused
        plane_views = pb.numpy_views_from_bytes_used(bytes_used, self.width, self.height)
        ts = buf.timestamp.secs + buf.timestamp.usecs * 1e-6
        return V4L2BufferInfo(
            index=buf.index,
            plane_arrays=plane_views,
            plane_bytes_used=(bytes_used,),
            plane_fds=tuple(pb._fds),
            width=self.width,
            height=self.height,
            pixel_format=self._pixel_format_str,
            timestamp=ts,
            sequence=buf.sequence,
        )

    def _dequeue_mp(self) -> Optional[V4L2BufferInfo]:
        buf = v4l2_buffer_mp()
        buf.type = self._buf_type
        buf.memory = self._memory
        planes_arr = (v4l2_plane * 8)()
        buf.u.planes = ctypes.cast(
            ctypes.pointer(planes_arr), ctypes.c_void_p
        ).value
        buf.length = 8
        try:
            _ioctl_capture(self._fd, _v4l2.VIDIOC_DQBUF, buf)
        except OSError as e:
            if e.errno in (11, 35):
                return None
            raise
        pb = self._buffers[buf.index]
        # bytesused is per-plane for multiplanar; buf.length was set on the
        # reqbufs call but may be 8 here — use the actual plane count from
        # the negotiated format (num_planes) instead.
        np_planes = self._fmt.pix_mp.num_planes
        used = tuple(planes_arr[p].bytesused for p in range(np_planes))
        plane_views = pb.numpy_views_from_bytes_used_per_plane(
            used, self.width, self.height
        )
        ts = buf.timestamp.secs + buf.timestamp.usecs * 1e-6
        return V4L2BufferInfo(
            index=buf.index,
            plane_arrays=plane_views,
            plane_bytes_used=used,
            plane_fds=tuple(pb._fds),
            width=self.width,
            height=self.height,
            pixel_format=self._pixel_format_str,
            timestamp=ts,
            sequence=buf.sequence,
        )

    def _enqueue_index(self, index: int) -> None:
        if self._is_multiplanar:
            buf = v4l2_buffer_mp()
            buf.type = self._buf_type
            buf.memory = self._memory
            buf.index = index
            planes_arr = (v4l2_plane * 8)()
            buf.u.planes = ctypes.cast(
                ctypes.pointer(planes_arr), ctypes.c_void_p
            ).value
            buf.length = 8
            _ioctl_capture(self._fd, _v4l2.VIDIOC_QBUF, buf)
        else:
            buf = _v4l2.v4l2_buffer()
            buf.type = self._buf_type
            buf.memory = self._memory
            buf.index = index
            _ioctl_capture(self._fd, _v4l2.VIDIOC_QBUF, buf)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class _PlaneBuf:
    """One mmap'd capture buffer (single or multiplane).

    Each plane owns (a) an mmap of its memory and (b) a DMA-BUF fd exported
    via VIDIOC_EXPBUF. Both are kept in sync for the buffer's lifetime;
    ``close()`` releases them in the right order (close fd before mmap).
    """

    __slots__ = (
        "index",
        "_plane_specs",     # list of (mem_offset, length)
        "_mmaps",           # list of mmap.mmap
        "_fds",             # list of DMA-BUF fds (one per plane)
        "_fd",
        "_is_mp",
        "_planes_arr",
    )

    def __init__(self, index, plane_specs, fd, is_mp, planes_arr):
        self.index = index
        self._fd = fd
        self._plane_specs = plane_specs
        self._is_mp = is_mp
        self._planes_arr = planes_arr
        self._mmaps = []
        self._fds = []
        for off, length in plane_specs:
            if length == 0:
                self._mmaps.append(None)
                continue
            self._mmaps.append(
                mmap.mmap(fd, length, offset=off)
            )

    def close(self):
        # Close DMABUF fds first (the kernel owns the underlying buffer;
        # after the fd is closed the mmap becomes invalid).
        for fd in self._fds:
            if fd is not None and fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass
        self._fds.clear()
        for m in self._mmaps:
            if m is not None:
                try:
                    m.close()
                except Exception:
                    pass
        self._mmaps.clear()

    def numpy_views_from_bytes_used(
        self, total_bytes_used: int, width: int, height: int
    ) -> Tuple[np.ndarray, ...]:
        # Single-planar: one full-frame view at the full mmap size.
        m = self._mmaps[0]
        if m is None:
            return ()
        size = total_bytes_used if total_bytes_used > 0 else m.size()
        return (np.frombuffer(m, dtype=np.uint8, count=size),)

    def numpy_views_from_bytes_used_per_plane(
        self, used_per_plane, width, height
    ) -> Tuple[np.ndarray, ...]:
        out = []
        for m, used in zip(self._mmaps, used_per_plane):
            if m is None:
                continue
            size = used if used > 0 else len(m)
            out.append(np.frombuffer(m, dtype=np.uint8, count=size))
        return tuple(out)


def _ioctl_capture(fd, request: int, arg):
    """Call ioctl. Returns the populated struct (mutated in place)."""
    fcntl.ioctl(fd, request, arg)
    return arg


def _negotiate_format(fd, is_mp: bool, width: int, height: int, fmt_int: int) -> v4l2_format:
    """VIDIOC_S_FMT and return the *negotiated* format back."""
    fmt = v4l2_format()
    fmt.type = (
        V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE if is_mp
        else _v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
    )
    if is_mp:
        fmt.pix_mp.width = width
        fmt.pix_mp.height = height
        fmt.pix_mp.pixelformat = fmt_int
        fmt.pix_mp.field = V4L2_FIELD_NONE
    else:
        fmt.pix.width = width
        fmt.pix.height = height
        fmt.pix.pixelformat = fmt_int
        fmt.pix.field = V4L2_FIELD_NONE
    _ioctl_capture(fd, _v4l2.VIDIOC_S_FMT, fmt)
    # fmt is now populated with the driver's actual width/height/bytesperline.
    return fmt


__all__ = [
    "V4L2Capture",
    "V4L2BufferInfo",
    "v4l2_exportbuffer",
    "VIDIOC_EXPBUF",
    "fourcc_to_str",
    "str_to_fourcc",
    "V4L2_CAP_VIDEO_CAPTURE_MPLANE",
    "V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE",
]  # type: ignore[list-struct]  -- @formatter:off