"""DMA-BUF allocators for hardware-engine interop.

Two allocation paths are exposed, both producing ``DmaBuf`` instances that
own a Linux DMA-BUF file descriptor:

* :class:`DmaHeap` — modern Linux kernel ``/dev/dma_heap/<name>``
  allocator. Available on any reasonably modern (5.6+) kernel with
  ``CONFIG_DMABUF_HEAPS`` enabled. Uses one ioctl, returns a fresh fd,
  no DRM dependency.
* :class:`WdnavDmaBuf` — legacy Rockchip-specific path, mirrored on
  ``/userdata/rbs/core/buffers/allocator_drm.cpp``: ``DRM_IOCTL_ROCKCHIP_GEM_CREATE``
  on ``/dev/dri/card0`` followed by ``drmPrimeHandleToFD``. Use this
  only when running against an older Rockchip kernel that does not
  expose DMA-HEAP nodes. Newer kernels (≥5.6) typically return
  ``ENOTTY`` for ``DRM_IOCTL_ROCKCHIP_GEM_CREATE`` because the ioctl
  was retired in favour of the standard DMA-HEAP allocator.

Both paths produce a real ``/dmabuf:N`` fd. Whether that fd is then
accepted by the libmali ``eglCreateImageKHR(EGL_LINUX_DMA_BUF_EXT, ...)``
path is a separate concern — see :mod:`wdnav.mali` and the reference
``/userdata/rbs/frameworks/glengine/buffer_gl_image.cpp`` for the
attribute layout.

Typical use as a producer-side buffer for the GPU compositor:

    import wdnav.dmaheap as dh
    with dh.DmaHeap("system") as heap:
        with heap.alloc(1920 * 1080 * 3 // 2) as buf:
            # ... mmap / fill / or wrap in EGLImage via wdnav.mali ...
            pass
"""

from __future__ import annotations

import ctypes
import fcntl
import os
import struct
from typing import Optional


_HEAP_DIR = "/dev/dma_heap"
_HEAP_IOCTL_ALLOC = 0xC0184800


def list_heaps() -> list[str]:
    """Return the names of all ``/dev/dma_heap/*`` devices present on the system."""
    try:
        return sorted(os.listdir(_HEAP_DIR))
    except FileNotFoundError:
        return []


class DmaHeapError(RuntimeError):
    """Raised on any failure to allocate from a DMA-BUF heap."""


class DmaBuf:
    """A DMA-BUF file descriptor returned by :meth:`DmaHeap.alloc`.

    Owns the fd; closing the object closes the fd (and thus releases the
    underlying buffer to the heap allocator). Most user code should use
    this object directly without worrying about the underlying integer
    fd; it can be passed to :meth:`wdnav.mali.create_image_dma_buf` via the
    :attr:`fd` property.
    """

    __slots__ = ("_fd", "_size", "_closed")

    def __init__(self, fd: int, size: int) -> None:
        self._fd = fd
        self._size = size
        self._closed = False

    @property
    def fd(self) -> int:
        """The DMA-BUF file descriptor, valid until :meth:`close`."""
        if self._closed:
            raise ValueError("DmaBuf already closed")
        return self._fd

    @property
    def size(self) -> int:
        """Allocation size in bytes, as passed to :meth:`DmaHeap.alloc`."""
        return self._size

    def fileno(self) -> int:
        """Same as :attr:`fd`. Allows ``DmaBuf`` to interoperate with code
        that expects a fileno-returning object (``os.read``, selectors, ...).
        """
        return self.fd

    def mmap(self, length: Optional[int] = None, prot: int = 0x1 | 0x2, flags: int = 0x1) -> int:
        """Memory-map the buffer into the current process. Returns the
        mmap'd address as an integer.

        ``prot`` defaults to ``PROT_READ | PROT_WRITE``; ``flags`` defaults
        to ``MAP_SHARED``. Pass ``length=None`` to map the entire buffer.
        """
        import mmap
        if length is None:
            length = self._size
        return mmap.mmap(self._fd, length, prot=prot, flags=flags)

    def sync_read(self) -> None:
        """Invalidate the CPU cache so subsequent CPU reads see GPU writes."""
        _dma_buf_sync(self._fd, begin=True, read=True, write=False)

    def sync_write(self) -> None:
        """Flush the CPU cache so the GPU sees CPU writes to the buffer."""
        _dma_buf_sync(self._fd, begin=True, read=False, write=True)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            os.close(self._fd)
        except OSError:
            pass

    def __enter__(self) -> "DmaBuf":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        state = "closed" if self._closed else f"fd={self._fd}, {self._size}B"
        return f"<DmaBuf {state}>"


class DmaHeap:
    """Allocator bound to a single ``/dev/dma_heap/<name>`` device.

    The kernel's DMA-BUF heap interface is a tiny ioctl that takes a
    length and returns a freshly-created DMA-BUF file descriptor. The fd
    is reference-counted: closing all references releases the buffer.
    """

    __slots__ = ("_name", "_path", "_fd")

    def __init__(self, name: str = "system") -> None:
        self._name = name
        self._path = os.path.join(_HEAP_DIR, name)
        if not os.path.exists(self._path):
            raise DmaHeapError(
                f"dma-heap device {self._path!r} not present; "
                f"available heaps: {list_heaps()}"
            )
        self._fd: Optional[int] = None

    def _open(self) -> int:
        if self._fd is None:
            self._fd = os.open(self._path, os.O_RDWR | os.O_CLOEXEC)
        return self._fd

    @property
    def name(self) -> str:
        return self._name

    @property
    def path(self) -> str:
        return self._path

    def alloc(self, size: int, fd_flags: int = 0, heap_flags: int = 0) -> DmaBuf:
        """Allocate ``size`` bytes and return the resulting :class:`DmaBuf`.

        ``fd_flags`` and ``heap_flags`` are passed straight through to the
        kernel; both default to zero (plain read/write). Common values for
        ``fd_flags`` include ``O_CLOEXEC`` / ``O_RDONLY``; ``heap_flags``
        are vendor-defined (e.g., Rockchip-specific protection hints).
        """
        if size <= 0:
            raise ValueError(f"DmaHeap.alloc: size must be positive, got {size}")
        ctrl_fd = self._open()
        # struct dma_heap_allocation_data on aarch64:
        #   uint64_t len;        /* input  */
        #   uint32_t fd;         /* output */
        #   uint32_t fd_flags;   /* input  */
        #   uint64_t heap_flags; /* input  */
        # total: 24 bytes (8 + 4 + 4 + 8, no padding on LP64)
        buf = bytearray(24)
        struct.pack_into(
            "<QIIQ", buf, 0,
            size, 0, fd_flags, heap_flags,
        )
        try:
            fcntl.ioctl(ctrl_fd, _HEAP_IOCTL_ALLOC, buf)
        except OSError as e:
            raise DmaHeapError(
                f"dma-heap {self._name!r} alloc({size}) failed: {e}"
            ) from e
        new_fd = struct.unpack_from("<I", buf, 8)[0]
        return DmaBuf(new_fd, size)

    def close(self) -> None:
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None

    def __enter__(self) -> "DmaHeap":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        return f"<DmaHeap {self._name!r} at {self._path!r}>"


def _dma_buf_sync(fd: int, begin: bool, read: bool, write: bool) -> None:
    """DMA_BUF_IOCTL_SYNC wrapper. Calls begin (DMA_BUF_SYNC_START) when
    ``begin=True``, end (DMA_BUF_SYNC_END) when ``begin=False``. ``read``
    and ``write`` set DMA_BUF_SYNC_READ/WRITE flags.
    """
    DMA_BUF_SYNC_READ = 1 << 0
    DMA_BUF_SYNC_WRITE = 1 << 1
    DMA_BUF_SYNC_START = 0 << 2
    DMA_BUF_SYNC_END = 1 << 2
    flags = 0
    if read:
        flags |= DMA_BUF_SYNC_READ
    if write:
        flags |= DMA_BUF_SYNC_WRITE
    if begin:
        flags |= DMA_BUF_SYNC_START
    else:
        flags |= DMA_BUF_SYNC_END
    DMA_BUF_BASE = 0x25
    ioctl = (3 << 30) | (DMA_BUF_BASE << 8) | 0
    sync_data = ctypes.c_uint64(flags)
    fcntl.ioctl(fd, ioctl, sync_data)


_RK_GEM_CREATE_IOCTL = 0xC0105240
_RK_GEM_MAP_OFFSET_IOCTL = 0xC0105241
_PRIME_HANDLE_TO_FD_IOCTL = 0xC00C522D
_RK_BO_CONTIG = 1 << 0
_RK_BO_CACHABLE = 1 << 1
_RK_BO_WC = 1 << 2


class WdnavDmaBufError(DmaHeapError):
    """Raised when the Rockchip DRM GEM path cannot satisfy an allocation."""


class WdnavDmaBuf:
    """Legacy Rockchip-specific DMA-BUF allocator.

    Mirrors ``/userdata/rbs/core/buffers/allocator_drm.cpp``: opens
    ``/dev/dri/card0``, runs ``DRM_IOCTL_ROCKCHIP_GEM_CREATE`` to get a
    Rockchip GEM handle, then ``drmPrimeHandleToFD`` to export that
    handle as a DMA-BUF fd. The fd is what ``eglCreateImageKHR`` (in
    ``buffer_gl_image.cpp``) consumes.

    Use this only when ``/dev/dma_heap/`` is unavailable (older
    Rockchip kernels). Newer kernels typically return ``ENOTTY`` for
    ``DRM_IOCTL_ROCKCHIP_GEM_CREATE``; in that case use :class:`DmaHeap`
    instead. Both paths produce identical ``/dmabuf:N`` fds.

    The constructor intentionally does not probe whether the
    Rockchip-specific ioctl is supported on the running kernel;
    :meth:`alloc` will raise :class:`WdnavDmaBufError` on first use if
    the kernel rejects the ioctl. ``alloc_contig`` and
    ``alloc_wc`` helpers pre-set the matching ``ROCKCHIP_BO_*`` flags.
    """

    __slots__ = ("_card", "_size", "_handle", "_fd", "_closed")

    def __init__(self, card_path: str = "/dev/dri/card0") -> None:
        self._card = os.open(card_path, os.O_RDWR | os.O_CLOEXEC)
        self._size = 0
        self._handle = 0
        self._fd = -1
        self._closed = False

    def _alloc(self, size: int, flags: int) -> "DmaBuf":
        if size <= 0:
            raise ValueError(f"size must be positive, got {size}")
        buf = bytearray(16)
        ctypes.c_uint64.from_buffer(buf, 0).value = size
        ctypes.c_uint32.from_buffer(buf, 8).value = flags
        try:
            fcntl.ioctl(self._card, _RK_GEM_CREATE_IOCTL, buf)
        except OSError as e:
            raise WdnavDmaBufError(
                f"DRM_IOCTL_ROCKCHIP_GEM_CREATE failed (errno={e.errno}, "
                f"{e.strerror}); kernel may not support this ioctl — "
                f"try DmaHeap instead"
            ) from e
        handle = ctypes.c_uint32.from_buffer(buf, 12).value
        fd_buf = bytearray(12)
        ctypes.c_uint32.from_buffer(fd_buf, 0).value = handle
        ctypes.c_uint32.from_buffer(fd_buf, 4).value = 0
        try:
            fcntl.ioctl(self._card, _PRIME_HANDLE_TO_FD_IOCTL, fd_buf)
        except OSError as e:
            try:
                _rk_gem_close(self._card, handle)
            except OSError:
                pass
            raise WdnavDmaBufError(
                f"drmPrimeHandleToFD failed (errno={e.errno}, {e.strerror})"
            ) from e
        fd = ctypes.c_int32.from_buffer(fd_buf, 8).value
        self._size = size
        self._handle = handle
        self._fd = fd
        return DmaBuf(fd, size)

    def alloc(self, size: int) -> "DmaBuf":
        """Allocate a non-contiguous, non-cacheable GEM buffer."""
        return self._alloc(size, 0)

    def alloc_contig(self, size: int) -> "DmaBuf":
        """Allocate a CMA-contiguous buffer (``ROCKCHIP_BO_CONTIG``).

        Mirrors ``DrmAllocator::DrmAllocator(true /* useCMA */)`` in
        ``allocator_drm.cpp``. Required for buffers that the VPU or ISP
        will consume as physical addresses.
        """
        return self._alloc(size, _RK_BO_CONTIG)

    def alloc_wc(self, size: int) -> "DmaBuf":
        """Allocate a write-combining, non-cacheable buffer
        (``ROCKCHIP_BO_WC``). Useful for GPU-rendered surfaces that need
        streaming writes without cache flush overhead.
        """
        return self._alloc(size, _RK_BO_WC)

    def alloc_cacheable(self, size: int) -> "DmaBuf":
        """Allocate a cacheable buffer (``ROCKCHIP_BO_CACHABLE``). Needs
        explicit ``DmaBuf.sync_read`` / ``sync_write`` calls around CPU
        access (same pattern as :meth:`DrmAllocator::lock` /
        ``unlock``).
        """
        return self._alloc(size, _RK_BO_CACHABLE)

    @property
    def card_path(self) -> str:
        return f"/proc/self/fd/{self._card}"

    @property
    def handle(self) -> int:
        """The Rockchip GEM handle backing the last allocation. ``0`` if
        no allocation is currently live or already closed."""
        return self._handle

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._fd >= 0:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = -1
        if self._handle > 0:
            try:
                _rk_gem_close(self._card, self._handle)
            except OSError:
                pass
            self._handle = 0
        try:
            os.close(self._card)
        except OSError:
            pass

    def __enter__(self) -> "WdnavDmaBuf":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        state = "closed" if self._closed else (
            f"fd={self._fd}, handle=0x{self._handle:x}, {self._size}B"
            if self._fd >= 0 else "ready"
        )
        return f"<WdnavDmaBuf {state}>"


def _rk_gem_close(card_fd: int, handle: int) -> None:
    """Mirror of ``DRM_IOCTL_GEM_CLOSE`` on the GEM handle."""
    buf = bytearray(8)
    ctypes.c_uint32.from_buffer(buf, 0).value = handle
    DRM_IOCTL_GEM_CLOSE = 0xC010640E
    fcntl.ioctl(card_fd, DRM_IOCTL_GEM_CLOSE, buf)


__all__ = [
    "DmaHeap",
    "DmaBuf",
    "DmaHeapError",
    "WdnavDmaBuf",
    "WdnavDmaBufError",
    "list_heaps",
]
