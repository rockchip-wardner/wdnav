"""Offscreen GLES compositor for rkv.

Implements an Android SurfaceFlinger-style architecture on top of libmali:

* Each :class:`Surface` is a triple-buffered BufferQueue keyed by either a
  producer-side DMA-BUF fd (zero-copy import via
  ``EGL_EXT_image_dma_buf_import``) or a CPU-side BGR ndarray (software fill
  that the worker uploads on the next composite).
* A single worker thread owns the EGL context and runs a composite loop:
  draw textured quads sorted by zpos, ``glFinish`` to fence the GPU, then
  ``glReadPixels`` into a PBO that maps to a numpy view. Producer threads
  never touch EGL/GL; they only call :meth:`Surface.post_fd` or
  :meth:`Surface.post`.
* :meth:`SurfaceFlinger.read` blocks on a Python ``threading.Event`` set by
  the worker after each frame and returns a numpy ndarray view over the
  current PBO mapping (zero-copy).

The worker tries a PBuffer surface first (matching the rbs glengine layout
in ``frameworks/glengine/glbase_egl.cpp``) and falls back to
``EGL_KHR_surfaceless_context`` when the active libmali build rejects
PBuffer creation. Both paths give the same FBO + readback behaviour; only
the EGL display binding differs.
"""

from __future__ import annotations

import ctypes
import os
import queue
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import mali as _m


@dataclass
class _BufferSlot:
    """One triple-buffer slot on a Surface.

    Producer-side state (set by ``post_fd`` / ``post``, consumed by worker):
    ``pending_fd`` (DMA-BUF fd from hardware producer), ``pending_pts``,
    ``pending_stride`` (DMA-BUF row pitch in pixels — required for the
    ``EGL_EXT_image_dma_buf_import`` path), ``pending_cpu`` (numpy
    ndarray to upload, or None).

    GL state (allocated lazily by the worker the first time the slot is
    observed, never touched by producer threads): ``gl_tex_nv`` for NV12
    layers (a ``GL_TEXTURE_EXTERNAL_OES`` target so the Mali driver can
    sample the Y/UV planes in one shot), ``gl_tex`` for BGR layers, plus
    the EGLImage handle currently bound to that texture.
    """

    pending_fd: int = -1
    pending_pts: int = 0
    pending_stride: int = 0
    pending_cpu: Optional[np.ndarray] = None
    pts: int = 0

    gl_tex_nv: int = 0
    gl_tex: int = 0
    image: int = 0
    allocated: bool = False


_NV12_FMT = "NV12"
_BGR_FMT = "BGR888"
_FB_BPP = 4
_QUAD_FLOATS = (
    -1.0, -1.0,   0.0, 0.0,
     1.0, -1.0,   1.0, 0.0,
    -1.0,  1.0,   0.0, 1.0,
     1.0,  1.0,   1.0, 1.0,
)


def _rect_pixels(x: int, y: int, w: int, h: int, screen_w: int, screen_h: int) -> tuple[float, float, float, float]:
    """Return ``(center_x, center_y, half_w, half_h)`` in NDC.

    ``(x, y)`` is the screen-space top-left of the rect; NDC Y is flipped
    (GL origin is at bottom-left, screen origin at top-left).
    """
    half_w = w / screen_w
    half_h = h / screen_h
    center_x = (x + w / 2.0) / screen_w * 2.0 - 1.0
    center_y = 1.0 - (y + h / 2.0) / screen_h * 2.0
    return center_x, center_y, half_w, half_h


def _yuv_nv_image(
    dpy: int, fd: int, width: int, height: int, stride: int,
) -> int:
    """Mirror the NV12 branch of ``constructEglImage`` in rbs'
    ``buffer_gl_image.cpp``: one EGLImage, two planes, NV12 fourcc.

    Returns the EGLImage handle. The shader samples this image as a
    single ``samplerExternalOES``; the GPU driver handles the chroma
    upsampling in hardware. The ``stride`` argument matters: MPP/RGA
    allocate buffers with ``hor_stride`` bytes per row that may exceed
    the visible ``width`` (typically rounded up to 16 / 64 for the
    hardware). Passing ``width`` as the plane pitch therefore misaligns
    the UV plane; the reference uses the actual stride.
    """
    attribs = (ctypes.c_int * 25)(
        _m.EGL_WIDTH, width,
        _m.EGL_HEIGHT, height,
        _m.EGL_LINUX_DRM_FOURCC_EXT, _m.DRMFormat.NV12,
        _m.EGL_DMA_BUF_PLANE0_FD_EXT, fd,
        _m.EGL_DMA_BUF_PLANE0_OFFSET_EXT, 0,
        _m.EGL_DMA_BUF_PLANE0_PITCH_EXT, stride,
        _m.EGL_DMA_BUF_PLANE1_FD_EXT, fd,
        _m.EGL_DMA_BUF_PLANE1_OFFSET_EXT, stride * height,
        _m.EGL_DMA_BUF_PLANE1_PITCH_EXT, stride,
        _m.EGL_NONE,
    )
    image = _m._lib.eglCreateImageKHR(
        dpy, _m.EGL_NO_CONTEXT, _m.EGL_LINUX_DMA_BUF_EXT, None,
        ctypes.cast(attribs, _m.POINTER(_m.c_int)),
    )
    if not image:
        err = _m._lib.eglGetError()
        raise RuntimeError(
            f"eglCreateImageKHR(NV12 {width}x{height} stride={stride}, fd={fd}) "
            f"failed: EGL error 0x{err:x}"
        )
    return image


class Surface:
    """A single composited layer.

    Owns three :class:`_BufferSlot` instances and a CPU-side BGR ndarray
    accessible via :attr:`latest_ndarray`. Producers push a frame with
    :meth:`post_fd` (hardware zero-copy) or :meth:`post` (software upload).
    """

    __slots__ = (
        "id",
        "x",
        "y",
        "w",
        "h",
        "zpos",
        "fmt",
        "_default_stride",
        "_slots",
        "_free",
        "_ready",
        "_cpu_buf",
        "_lock",
        "_owner",
    )

    def __init__(
        self,
        sf: "SurfaceFlinger",
        id: int,
        x: int,
        y: int,
        w: int,
        h: int,
        zpos: int,
        fmt: str,
    ) -> None:
        self.id = id
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.zpos = zpos
        self.fmt = fmt
        self._default_stride = w
        self._slots = [_BufferSlot() for _ in range(3)]
        self._free: queue.SimpleQueue = queue.SimpleQueue()
        self._ready: queue.SimpleQueue = queue.SimpleQueue()
        for s in self._slots:
            self._free.put(s)
        self._cpu_buf = np.zeros((h, w, 3), dtype=np.uint8)
        self._lock = threading.Lock()

    @property
    def latest_ndarray(self) -> np.ndarray:
        """Direct view over a CPU-side BGR buffer. Fill it then call
        :meth:`post` to make the worker upload + composite it.
        """
        return self._cpu_buf

    def post_fd(
        self,
        fd: int,
        pts: int = 0,
        *,
        stride: Optional[int] = None,
    ) -> None:
        """Hand a DMA-BUF fd to this surface.

        Blocks if all three slots are full (Android BufferQueue semantics).
        ``stride`` is the buffer's row pitch in pixels — set this whenever
        the producer's stride may differ from the surface's ``w`` (which
        is common for MPP/RGA output buffers). When ``None`` (the
        default), ``w`` is used as the stride, matching the simpler
        case where the producer allocates with width-padded buffers.
        """
        if fd < 0:
            raise ValueError(f"Surface.post_fd: invalid fd {fd}")
        slot = self._free.get()
        slot.pending_fd = fd
        slot.pending_pts = pts
        slot.pending_stride = stride if stride and stride > 0 else self._default_stride
        slot.pending_cpu = None
        self._ready.put(slot)
        if self._owner is not None:
            self._owner.notify()

    def post(self, pts: int = 0) -> None:
        """Submit the current :attr:`latest_ndarray` as a new frame.

        Snapshots the CPU buffer at call time so the producer can
        immediately start filling it again for the next frame.
        """
        slot = self._free.get()
        slot.pending_fd = -1
        slot.pending_cpu = self._cpu_buf.copy()
        slot.pending_pts = pts
        slot.pending_stride = 0
        self._ready.put(slot)
        if self._owner is not None:
            self._owner.notify()

    def _owner_set(self, sf: "SurfaceFlinger") -> None:
        self._owner = sf


class SurfaceFlinger:
    """Offscreen GLES compositor.

    The constructor starts a worker thread that owns the EGL context,
    compiles the shaders, and allocates the FBO + PBO + VBO. ``create_surface``
    is callable from any thread; ``read`` blocks on the worker's frame-ready
    event; ``close`` is idempotent.

    ``egl_display`` lets callers provide a pre-initialised EGL display
    handle. The default (``None``) walks ``eglGetDisplay(EGL_DEFAULT_DISPLAY)``
    + ``eglInitialize`` — the same path the rbs glengine uses for an
    X11/wayland compositor. Pass a different display (e.g. one obtained
    via :func:`rkv.mali.get_display` with ``EGL_PLATFORM_GBM_KHR`` and a
    gbm_device pointer) when you need dma-buf import to work on a build
    that doesn't support it on the default display.
    """

    def __init__(
        self,
        w: int,
        h: int,
        *,
        egl_display: Optional[ctypes.c_void_p] = None,
    ) -> None:
        if w <= 0 or h <= 0:
            raise ValueError(f"SurfaceFlinger: invalid size {w}x{h}")
        self.w = w
        self.h = h
        self._surfaces: dict[int, Surface] = {}
        self._next_surface_id = 0
        self._surfaces_lock = threading.Lock()

        self._wakeup = threading.Event()
        self._frame_ready = threading.Event()
        self._stop = threading.Event()
        self._pbo_view: Optional[np.ndarray] = None
        self._frame_count = 0
        self._error: Optional[BaseException] = None
        self._external_display = egl_display

        self._thread = threading.Thread(
            target=self._worker_main,
            name="rkv-gl-worker",
            daemon=True,
        )
        self._ready_event = threading.Event()
        self._thread.start()
        if not self._ready_event.wait(timeout=10.0):
            raise RuntimeError("SurfaceFlinger: worker thread did not become ready within 10s")
        if self._error is not None:
            raise self._error

    def notify(self) -> None:
        self._wakeup.set()

    def create_surface(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        *,
        zpos: int = 0,
        fmt: str = "NV12",
    ) -> Surface:
        """Allocate a new composited layer."""
        if fmt not in (_NV12_FMT, _BGR_FMT):
            raise ValueError(
                f"SurfaceFlinger.create_surface: unsupported fmt {fmt!r}; "
                f"expected one of {_NV12_FMT!r}, {_BGR_FMT!r}"
            )
        if x < 0 or y < 0 or w <= 0 or h <= 0:
            raise ValueError(f"SurfaceFlinger.create_surface: bad rect {x},{y} {w}x{h}")
        if x + w > self.w or y + h > self.h:
            raise ValueError(
                f"SurfaceFlinger.create_surface: rect {x},{y} {w}x{h} "
                f"exceeds frame {self.w}x{self.h}"
            )
        with self._surfaces_lock:
            sid = self._next_surface_id
            self._next_surface_id += 1
            surf = Surface(self, sid, x, y, w, h, zpos, fmt)
            surf._owner_set(self)
            self._surfaces[sid] = surf
        self.notify()
        return surf

    def read(self, *, timeout: Optional[float] = None) -> np.ndarray:
        """Block until the worker has produced a new frame, then return a
        ``(H, W, 3)`` BGR ``numpy.ndarray`` view over the PBO mapping.
        """
        if not self._frame_ready.wait(timeout=timeout):
            raise TimeoutError("SurfaceFlinger.read: timed out waiting for frame")
        self._frame_ready.clear()
        if self._error is not None:
            raise RuntimeError(
                f"SurfaceFlinger worker crashed: {self._error}"
            ) from self._error
        if self._pbo_view is None:
            raise RuntimeError("SurfaceFlinger.read: no frame available")
        return self._pbo_view

    def close(self) -> None:
        if self._stop.is_set():
            return
        self._stop.set()
        self._wakeup.set()
        self._thread.join(timeout=2.0)
        if self._thread.is_alive():
            raise RuntimeError("SurfaceFlinger: worker did not stop within 2s")

    def _cleanup_egl(self) -> None:
        if getattr(self, "_owns_surface", False) and self._surface:
            try:
                _m.destroy_surface(self._dpy, self._surface)
            except Exception:
                pass
            self._surface = ctypes.c_void_p(0)
        if getattr(self, "_owns_display", False):
            try:
                _m.terminate(self._dpy)
            except Exception:
                pass

    def __enter__(self) -> "SurfaceFlinger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _worker_main(self) -> None:
        try:
            self._init_egl()
            self._init_shaders()
            self._init_pbos()
            self._init_fullscreen_quad()
            self._ready_event.set()
            self._composite_loop()
        except BaseException as e:
            self._error = e
            try:
                self._ready_event.set()
            except Exception:
                pass
            self._frame_ready.set()
        finally:
            try:
                self._cleanup_egl()
            except Exception:
                pass

    def _init_egl(self) -> None:
        """Initialise the EGL display, context, and (best-effort) PBuffer.

        Mirrors ``EGLNativeGBM::initDisplay`` + ``GLBaseEGL::initSurface``
        in ``frameworks/glengine/glbase_egl.cpp``: a platform display
        (here ``EGL_DEFAULT_DISPLAY``; production code on an RK board
        might also pass ``eglGetPlatformDisplayEXT(EGL_PLATFORM_GBM_KHR,
        gbm_device, NULL)``), an ES3 context, then either a PBuffer
        surface that the FBO can attach to or a surfaceless context.
        PBuffer creation failure is non-fatal; we fall back to
        ``EGL_KHR_surfaceless_context`` which is what this libmali
        build is restricted to in practice.
        """
        if self._external_display is not None:
            self._dpy = ctypes.c_void_p(self._external_display.value)
            self._owns_display = False
        else:
            self._dpy = _m.get_default_display()
            self._owns_display = True
            _m.initialize(self._dpy)
        configs = _m.choose_config(self._dpy, [
            (_m.EGL_RED_SIZE, 8),
            (_m.EGL_GREEN_SIZE, 8),
            (_m.EGL_BLUE_SIZE, 8),
            (_m.EGL_ALPHA_SIZE, 0),
            (_m.EGL_RENDERABLE_TYPE, _m.EGL_OPENGL_ES3_BIT),
        ])
        if not configs:
            raise RuntimeError("SurfaceFlinger: no ES3 config found")
        self._cfg = configs[0]
        self._ctx = _m.create_context(self._dpy, self._cfg, client_version=3)

        self._surface = ctypes.c_void_p(0)
        try:
            self._surface = _m.create_pbuffer_surface(
                self._dpy, self._cfg, self.w, self.h,
            )
            self._owns_surface = True
        except RuntimeError:
            self._surface = ctypes.c_void_p(0)
            self._owns_surface = False
        try:
            _m.make_current(
                self._dpy, self._surface, self._surface, self._ctx,
            )
        except RuntimeError:
            _m.make_current_surfaceless(self._dpy, self._ctx)

        rb_id = (ctypes.c_uint32 * 1)()
        _m.glGenRenderbuffers(1, rb_id)
        self._rb = rb_id[0]
        _m.glBindRenderbuffer(_m.GL_RENDERBUFFER, self._rb)
        _m.glRenderbufferStorage(_m.GL_RENDERBUFFER, _m.GL_RGBA8, self.w, self.h)

        fbo_id = (ctypes.c_uint32 * 1)()
        _m.glGenFramebuffers(1, fbo_id)
        self._fbo = fbo_id[0]
        _m.glBindFramebuffer(_m.GL_FRAMEBUFFER, self._fbo)
        _m.glFramebufferRenderbuffer(
            _m.GL_FRAMEBUFFER, _m.GL_COLOR_ATTACHMENT0,
            _m.GL_RENDERBUFFER, self._rb,
        )
        status = _m.glCheckFramebufferStatus(_m.GL_FRAMEBUFFER)
        if status != _m.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(
                f"SurfaceFlinger: FBO incomplete, status=0x{status:x}"
            )
        _m.glBindFramebuffer(_m.GL_FRAMEBUFFER, 0)

    def _init_shaders(self) -> None:
        self._nv12_prog = _m._NV12Program()
        self._rgb_prog = _m._RGBProgram()
        self._nv12_uniforms = _m.nv12_uniforms()

    def _init_pbos(self) -> None:
        size = self.w * self.h * _FB_BPP
        ids = (ctypes.c_uint32 * 2)()
        _m.glGenBuffers(2, ids)
        self._pbo = (ids[0], ids[1])
        self._pbo_size = size
        self._pbo_idx = 0
        for pid in self._pbo:
            _m.glBindBuffer(_m.GL_PIXEL_PACK_BUFFER, pid)
            _m.glBufferData(_m.GL_PIXEL_PACK_BUFFER, size, None, _m.GL_STREAM_READ)
        _m.glBindBuffer(_m.GL_PIXEL_PACK_BUFFER, 0)
        self._pbo_mapped = [False, False]
        self._pbo_ptrs = [0, 0]

    def _init_fullscreen_quad(self) -> None:
        vbo_id = (ctypes.c_uint32 * 1)()
        _m.glGenBuffers(1, vbo_id)
        self._vbo = vbo_id[0]
        arr = (ctypes.c_float * len(_QUAD_FLOATS))(*_QUAD_FLOATS)
        _m.glBindBuffer(_m.GL_ARRAY_BUFFER, self._vbo)
        _m.glBufferData(
            _m.GL_ARRAY_BUFFER, ctypes.sizeof(arr), ctypes.cast(arr, ctypes.c_void_p),
            _m.GL_STATIC_DRAW,
        )
        _m.glPixelStorei(_m.GL_PACK_ALIGNMENT, 1)

    def _composite_loop(self) -> None:
        _m.glDisable(_m.GL_DEPTH_TEST)
        _m.glDisable(_m.GL_BLEND)
        while not self._stop.is_set():
            self._wakeup.wait(timeout=0.05)
            self._wakeup.clear()
            if self._stop.is_set():
                break
            surfaces = self._snapshot_surfaces()
            if not surfaces:
                continue
            drawn = self._composite_one(surfaces)
            if not drawn:
                continue
            _m.glFlush()
            _m.glFinish()
            self._swap_pbo_and_signal()

    def _snapshot_surfaces(self) -> list[Surface]:
        with self._surfaces_lock:
            return list(self._surfaces.values())

    def _composite_one(self, surfaces: list[Surface]) -> bool:
        _m.glBindFramebuffer(_m.GL_FRAMEBUFFER, self._fbo)
        _m.glViewport(0, 0, self.w, self.h)
        _m.glClearColor(0.0, 0.0, 0.0, 1.0)
        _m.glClear(_m.GL_COLOR_BUFFER_BIT)
        drawn = False
        for surf in sorted(surfaces, key=lambda s: s.zpos):
            try:
                slot = surf._ready.get_nowait()
            except queue.Empty:
                continue
            self._alloc_slot(slot, surf)
            self._draw_surface(surf, slot)
            slot.pending_fd = -1
            slot.pending_cpu = None
            slot.pts = slot.pending_pts
            drawn = True
            surf._free.put(slot)
        return drawn

    def _alloc_slot(self, slot: _BufferSlot, surf: Surface) -> None:
        if slot.allocated:
            return
        if surf.fmt == _NV12_FMT:
            ids = (ctypes.c_uint32 * 1)()
            _m.glGenTextures(1, ids)
            slot.gl_tex_nv = ids[0]
            _m.glBindTexture(_m.GL_TEXTURE_EXTERNAL_OES, slot.gl_tex_nv)
            _m.glTexParameteri(_m.GL_TEXTURE_EXTERNAL_OES, _m.GL_TEXTURE_MIN_FILTER, _m.GL_LINEAR)
            _m.glTexParameteri(_m.GL_TEXTURE_EXTERNAL_OES, _m.GL_TEXTURE_MAG_FILTER, _m.GL_LINEAR)
            _m.glTexParameteri(_m.GL_TEXTURE_EXTERNAL_OES, _m.GL_TEXTURE_WRAP_S, _m.GL_CLAMP_TO_EDGE)
            _m.glTexParameteri(_m.GL_TEXTURE_EXTERNAL_OES, _m.GL_TEXTURE_WRAP_T, _m.GL_CLAMP_TO_EDGE)
        else:
            ids = (ctypes.c_uint32 * 1)()
            _m.glGenTextures(1, ids)
            slot.gl_tex = ids[0]
            _m.glBindTexture(_m.GL_TEXTURE_2D, slot.gl_tex)
            _m.glTexParameteri(_m.GL_TEXTURE_2D, _m.GL_TEXTURE_MIN_FILTER, _m.GL_LINEAR)
            _m.glTexParameteri(_m.GL_TEXTURE_2D, _m.GL_TEXTURE_MAG_FILTER, _m.GL_LINEAR)
            _m.glTexParameteri(_m.GL_TEXTURE_2D, _m.GL_TEXTURE_WRAP_S, _m.GL_CLAMP_TO_EDGE)
            _m.glTexParameteri(_m.GL_TEXTURE_2D, _m.GL_TEXTURE_WRAP_T, _m.GL_CLAMP_TO_EDGE)
            _m.glTexImage2D(
                _m.GL_TEXTURE_2D, 0, _m.GL_RGB8, surf.w, surf.h, 0,
                _m.GL_RGB, _m.GL_UNSIGNED_BYTE, None,
            )
        slot.allocated = True

    def _draw_surface(self, surf: Surface, slot: _BufferSlot) -> None:
        ndc = _rect_pixels(surf.x, surf.y, surf.w, surf.h, self.w, self.h)
        if surf.fmt == _NV12_FMT:
            if slot.pending_fd >= 0:
                if slot.image:
                    _m.destroy_image(self._dpy, slot.image)
                stride = slot.pending_stride or surf.w
                slot.image = _yuv_nv_image(
                    self._dpy, slot.pending_fd,
                    surf.w, surf.h, stride,
                )
            _m.glActiveTexture(_m.GL_TEXTURE0)
            _m.glBindTexture(_m.GL_TEXTURE_EXTERNAL_OES, slot.gl_tex_nv)
            _m.glEGLImageTargetTexture2DOES(_m.GL_TEXTURE_EXTERNAL_OES, slot.image)
            self._use_program(self._nv12_prog)
            _m.glUniform1i(self._nv12_prog.loc_tex, 0)
            self._set_nv12_uniforms(self._nv12_prog)
        else:
            assert slot.pending_cpu is not None, "BGR slot must carry pending_cpu"
            _m.glActiveTexture(_m.GL_TEXTURE0)
            _m.glBindTexture(_m.GL_TEXTURE_2D, slot.gl_tex)
            _m.glPixelStorei(_m.GL_UNPACK_ALIGNMENT, 1)
            _m.glTexSubImage2D(
                _m.GL_TEXTURE_2D, 0, 0, 0, surf.w, surf.h,
                _m.GL_RGB, _m.GL_UNSIGNED_BYTE,
                slot.pending_cpu.ctypes.data_as(ctypes.c_void_p),
            )
            self._use_program(self._rgb_prog)
            _m.glUniform1i(self._rgb_prog.loc_tex, 0)
        _m.glUniform4f(
            self._active_prog.loc_rect,
            ndc[0], ndc[1], ndc[2], ndc[3],
        )
        self._draw_quad(self._active_prog)

    def _use_program(self, prog) -> None:
        _m.glUseProgram(prog.handle)
        self._active_prog = prog

    def _set_nv12_uniforms(self, prog) -> None:
        u = self._nv12_uniforms
        for loc, key in (
            (prog.loc_yuv_offset, "yuvOffset"),
            (prog.loc_yuv_scale, "yuvScale"),
            (prog.loc_yuv2r, "yuv2r"),
            (prog.loc_yuv2g, "yuv2g"),
            (prog.loc_yuv2b, "yuv2b"),
        ):
            vec = u[key]
            _m.glUniform3f(loc, vec[0], vec[1], vec[2])

    def _draw_quad(self, prog) -> None:
        _m.glBindBuffer(_m.GL_ARRAY_BUFFER, self._vbo)
        stride = 4 * ctypes.sizeof(ctypes.c_float)
        _m.glEnableVertexAttribArray(prog.a_pos)
        _m.glVertexAttribPointer(
            prog.a_pos, 2, _m.GL_FLOAT, _m.GL_FALSE, stride, None,
        )
        offset = 2 * ctypes.sizeof(ctypes.c_float)
        _m.glEnableVertexAttribArray(prog.a_uv)
        _m.glVertexAttribPointer(
            prog.a_uv, 2, _m.GL_FLOAT, _m.GL_FALSE, stride,
            ctypes.c_void_p(offset),
        )
        _m.glDrawArrays(_m.GL_TRIANGLE_STRIP, 0, 4)
        _m.glDisableVertexAttribArray(prog.a_pos)
        _m.glDisableVertexAttribArray(prog.a_uv)

    def _swap_pbo_and_signal(self) -> None:
        prev_idx = self._pbo_idx
        next_idx = 1 - prev_idx
        if self._pbo_mapped[prev_idx]:
            _m.glBindBuffer(_m.GL_PIXEL_PACK_BUFFER, self._pbo[prev_idx])
            _m.glUnmapBuffer(_m.GL_PIXEL_PACK_BUFFER)
            self._pbo_mapped[prev_idx] = False
        _m.glBindBuffer(_m.GL_PIXEL_PACK_BUFFER, self._pbo[next_idx])
        _m.glReadPixels(
            0, 0, self.w, self.h,
            _m.GL_RGBA, _m.GL_UNSIGNED_BYTE, None,
        )
        ptr = _m.glMapBufferRange(
            _m.GL_PIXEL_PACK_BUFFER, 0, self._pbo_size, _m.GL_MAP_READ_BIT,
        )
        if not ptr:
            raise RuntimeError("SurfaceFlinger: glMapBufferRange returned 0")
        self._pbo_ptrs[next_idx] = ptr
        self._pbo_mapped[next_idx] = True
        _m.glBindBuffer(_m.GL_PIXEL_PACK_BUFFER, 0)
        self._pbo_idx = next_idx
        cbuf = (ctypes.c_uint8 * self._pbo_size).from_address(ptr)
        raw = np.frombuffer(cbuf, dtype=np.uint8).reshape(self.h, self.w, 4)
        self._pbo_view = raw[:, :, [2, 1, 0]].copy()
        self._frame_count += 1
        self._frame_ready.set()


__all__ = ["SurfaceFlinger", "Surface"]
