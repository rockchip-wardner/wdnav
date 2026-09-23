"""Offscreen GLES compositor for wdnav.

Built on top of :mod:`wdnav.mali`, which wraps the libshim_mali.so C shim
emitted from ``src/wdnav/cshim_mali.c``. The shim owns every EGL/GLES
state-machine call; this module only holds the Python-side threading,
queueing, and per-frame draw logic.

* Each :class:`Surface` is a single-buffered BufferQueue keyed by either a
  producer-side DMA-BUF fd (zero-copy import via
  ``EGL_EXT_image_dma_buf_import``) or a CPU-side BGR ndarray.
* A single worker thread owns the :class:`~wdnav.mali.MaliContext` and
  runs a composite loop: draw textured quads sorted by zpos, fence the
  GPU, and mmap the DMA-BUF-backed FBO so the consumer reads straight
  from GPU memory — no glReadPixels involved.
* :meth:`SurfaceComposer.read` blocks on a Python event set by the worker
  after each frame and returns a zero-copy ndarray view over the
  DMA-BUF mmap.
"""

from __future__ import annotations

import ctypes
import mmap
import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np

from . import dmaheap
from . import mali as _m


_NV12_FMT = "NV12"
_BGR_FMT = "BGR888"

# Standard BT.601 limited-range NV12 → BGR coefficients. Same constants
# the original mali.py exposed via BT601_LIMITED; composer.py is
# the only consumer so they're local now.
_BT601_OFFSET = (16.0, 128.0, 128.0)
_BT601_YUV2R = (1.164, 0.0, 1.596)
_BT601_YUV2G = (1.164, -0.392, -0.813)
_BT601_YUV2B = (1.164, 2.017, 0.0)


@dataclass
class _BufferSlot:
    """One triple-buffer slot on a Surface.

    Producer-side state (set by ``post_fd`` / ``post``, consumed by worker):
    ``pending_fd`` (DMA-BUF fd from hardware producer), ``pending_pts``,
    ``pending_stride`` (DMA-BUF row pitch in pixels), ``pending_cpu``
    (numpy ndarray to upload, or None).

    GL state (allocated lazily by the worker the first time the slot is
    observed, never touched by producer threads): ``tex`` is the
    :class:`~wdnav.mali.MaliTexture` backing the slot — for NV12 layers
    it's a stable external texture that gets re-bound to a fresh EGLImage
    each frame; for BGR layers it's a one-shot GL_TEXTURE_2D that we
    upload CPU pixels into.
    """

    pending_fd: int = -1
    pending_pts: int = 0
    pending_stride: int = 0
    pending_cpu: Optional[np.ndarray] = None
    pts: int = 0

    tex: Optional[_m.MaliTexture] = None
    allocated: bool = False


def _rect_pixels(x: int, y: int, w: int, h: int,
                 screen_w: int, screen_h: int) -> tuple[float, float, float, float]:
    """Return ``(center_x, center_y, half_w, half_h)`` in NDC.

    ``(x, y)`` is the screen-space top-left of the rect; NDC Y is flipped
    (GL origin is at bottom-left, screen origin at top-left).
    """
    half_w = w / screen_w
    half_h = h / screen_h
    cx = (x + w / 2.0) / screen_w * 2.0 - 1.0
    cy = 1.0 - (y + h / 2.0) / screen_h * 2.0
    return cx, cy, half_w, half_h


class Surface:
    """A single composited layer."""

    __slots__ = (
        "id", "x", "y", "w", "h", "zpos", "fmt", "_default_stride",
        "_slots", "_free", "_ready", "_cpu_buf", "_lock", "_owner",
    )

    def __init__(self, sf: "SurfaceComposer", id: int, x: int, y: int,
                 w: int, h: int, zpos: int, fmt: str) -> None:
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

    def post_fd(self, fd: int, pts: int = 0,
                *, stride: Optional[int] = None) -> None:
        """Hand a DMA-BUF fd to this surface.

        Blocks if all three slots are full (Android BufferQueue semantics).
        ``stride`` is the buffer's row pitch in pixels — set this whenever
        the producer's stride may differ from the surface's ``w`` (which
        is common for MPP/RGA output buffers).
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

    def _owner_set(self, sf: "SurfaceComposer") -> None:
        self._owner = sf


# ---------------------------------------------------------------------------
# Typer — Surface subclass for CPU-side text rendering via MaliText
# ---------------------------------------------------------------------------
#
# Typer holds a ``wdnav.MaliText`` handle (a thin wrapper over the
# ``rkm_text_*`` c-shim functions that drive freetype). All glyph
# rasterisation + alpha-blend blits happen in C; this Python class only
# carries string / size / colour configuration.
#
# Font discovery goes through fontconfig (fc-match) when available, so a
# ``font`` argument can name a system font (``"sans"``, ``"mono"``,
# ``"DejaVu Serif"``, ``"wqy-zenhei"``, ...) OR a .ttf / .otf / .ttc path.


def _fc_resolve(spec):
    """Resolve a font spec to a concrete .ttf / .otf / .ttc path.

    Rules (in order):
      * ``None`` / ``""`` / ``"sans"``  → ``fc-match sans``
      * any other string that's an existing file path → return as-is
      * any other string → ``fc-match -f "%{file}\\n" <spec>`` (lets users
        pass canonical family names like ``"DejaVu Serif"``,
        ``"wqy-zenhei"``, or aliases like ``"mono"``).

    Raises:
        RuntimeError: fc-match missing on PATH, or spec doesn't resolve.
    """
    if spec is None or spec == "" or spec == "sans":
        spec_arg = "sans"
    elif isinstance(spec, os.PathLike):
        return os.fspath(spec)
    elif os.path.isfile(spec):
        return spec
    else:
        spec_arg = str(spec)

    try:
        out = subprocess.run(
            ["fc-match", "-f", "%{file}\\n", spec_arg],
            capture_output=True, text=True, timeout=2.0, check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        raise RuntimeError(
            f"Typer: fc-match failed ({e}); install fontconfig or pass "
            f"a font file path directly"
        ) from e

    path = out.stdout.strip()
    if out.returncode != 0 or not path or not os.path.isfile(path):
        raise RuntimeError(
            f"Typer: fc-match could not resolve {spec_arg!r} to a font file "
            f"(rc={out.returncode}, path={path!r})"
        )
    return path


class Typer(Surface):
    """Surface subclass dedicated to CPU-side text rendering.

    Typer extends :class:`Surface` with a small ``wdnav.MaliText`` handle
    that renders strings onto :attr:`latest_ndarray` via the c-shim's
    freetype path. Use it for OSDs, captions, debug labels, etc.

    Usage::

        typer = sf.create_typer(0, 0, 640, 96)        # 640x96 OSD strip
        typer.text("wdnav SurfaceComposer", xy=(8, 8))  # default font
        typer.text("frame 42", xy=(8, 32), size=18,
                   color=(0, 255, 255))                # cyan BGR
        typer.post(pts=1)

    Notes:
      * Internally BGR888 (the underlying :class:`Surface` only supports
        BGR for CPU buffers). ``color`` and ``bg_color`` follow the
        OpenCV / BGR convention.
      * ``font`` accepts a path (``os.PathLike`` / str to a .ttf / .otf
        / .ttc file), a fontconfig family / pattern string (``"sans"``,
        ``"mono"``, ``"DejaVu Serif"``, ``"wqy-zenhei"``, ...), or
        ``None`` for ``"sans"``.
      * Rendering goes through freetype via the c-shim; CJK / TTF /
        OpenType / TTC are all supported as long as the chosen font
        covers the codepoints in the string. ``wqy-zenhei`` on Debian /
        Ubuntu ships with Latin + CJK glyphs in face_index 0.
      * ``bg_color`` is a BGR tuple drawn behind the text cluster; use
        ``None`` for transparent (default).
    """

    __slots__ = ("_text", "_color", "_bg_color")

    def __init__(self, sf, id, x, y, w, h, zpos, *,
                 font=None, size=24,
                 color=(255, 255, 255),
                 bg_color=None):
        # Typer only makes sense on BGR surfaces; force the format here.
        super().__init__(sf, id, x, y, w, h, zpos, _BGR_FMT)
        font_path = _fc_resolve(font)
        self._text = _m.MaliText(font_path, int(size))
        self._color = tuple(int(c) for c in color)
        self._bg_color = (tuple(int(c) for c in bg_color)
                          if bg_color is not None else None)

    @property
    def font_path(self) -> str:
        """Resolved path to the .ttf / .otf / .ttc file backing this Typer."""
        return self._text.font_path

    def textsize(self, s, *, size=None, font=None):
        """Return the (ink_w, line_h) :meth:`text` would draw at ``size``.

        ``font`` and ``size`` are scoped to this measurement only — they
        don't change the Typer's persistent font/size.
        """
        if not s:
            return (0, 0)
        size_px = int(size) if size is not None else self._text.size_px
        if font is None and size_px == self._text.size_px:
            return self._text.measure(s)
        new_path = (_fc_resolve(font) if font is not None
                    else self._text.font_path)
        probe = _m.MaliText(new_path, size_px)
        try:
            return probe.measure(s)
        finally:
            probe.close()

    def text(self, s, xy, *, size=None, color=None, font=None,
             bg_color=None):
        """Render ``s`` at pixel coordinate ``xy`` (baseline-left).

        ``xy`` follows freetype's pen convention: the glyph cluster is
        drawn ABOVE the baseline at ``y``, with the leftmost pixel at
        ``x``. This matches how most OSD libraries (and cv2.putText)
        treat baselines.

        Returns the (drawn_w, drawn_h) actually inked in pixels. Empty
        strings return (0, 0).
        """
        if not s:
            return (0, 0)
        c = self._color if color is None else tuple(int(v) for v in color)
        bg = self._bg_color if bg_color is None else (
            tuple(int(v) for v in bg_color) if bg_color is not None else None)

        canvas = self.latest_ndarray
        H, W = canvas.shape[:2]
        x0, y0 = int(xy[0]), int(xy[1])

        size_px = int(size) if size is not None else self._text.size_px
        if font is None and size_px == self._text.size_px:
            text = self._text
        else:
            new_path = (_fc_resolve(font) if font is not None
                        else self._text.font_path)
            text = _m.MaliText(new_path, size_px)
        try:
            ink_w, ink_h = text.measure(s)
            if bg is not None:
                x_bg0 = max(x0, 0)
                y_bg0 = max(y0 - ink_h, 0)
                x_bg1 = min(x0 + ink_w, W)
                y_bg1 = min(y0, H)
                if x_bg1 > x_bg0 and y_bg1 > y_bg0:
                    canvas[y_bg0:y_bg1, x_bg0:x_bg1] = bg
            return text.render(s, x0, y0, color_bgr=c,
                               bgr_buf=canvas, buf_w=W, buf_h=H)
        finally:
            if text is not self._text:
                text.close()


class SurfaceComposer:
    """Offscreen GLES compositor.

    The constructor starts a worker thread that owns the
    :class:`~wdnav.mali.MaliContext`, compiles the NV12 + BGR fragment
    programs, and allocates the FBO + PBO + VBO. ``create_surface`` is
    callable from any thread; ``read`` blocks on the worker's frame-ready
    event; ``close`` is idempotent.

    All EGL / GLES detail lives in the C shim — this class only schedules
    per-surface draw calls and packages the readback pointer into a numpy
    view.
    """

    def __init__(self, w: int, h: int) -> None:
        if w <= 0 or h <= 0:
            raise ValueError(f"SurfaceComposer: invalid size {w}x{h}")
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

        self._thread = threading.Thread(
            target=self._worker_main,
            name="wdnav-gl-worker",
            daemon=True,
        )
        self._ready_event = threading.Event()
        self._thread.start()
        if not self._ready_event.wait(timeout=10.0):
            raise RuntimeError(
                "SurfaceComposer: worker thread did not become ready within 10s"
            )
        if self._error is not None:
            raise self._error

    def notify(self) -> None:
        self._wakeup.set()

    def create_surface(self, x: int, y: int, w: int, h: int, *,
                       zpos: int = 0, fmt: str = "NV12") -> Surface:
        """Allocate a new composited layer."""
        if fmt not in (_NV12_FMT, _BGR_FMT):
            raise ValueError(
                f"SurfaceComposer.create_surface: unsupported fmt {fmt!r}; "
                f"expected one of {_NV12_FMT!r}, {_BGR_FMT!r}"
            )
        if x < 0 or y < 0 or w <= 0 or h <= 0:
            raise ValueError(f"SurfaceComposer.create_surface: bad rect {x},{y} {w}x{h}")
        if x + w > self.w or y + h > self.h:
            raise ValueError(
                f"SurfaceComposer.create_surface: rect {x},{y} {w}x{h} "
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

    def create_typer(self, x: int, y: int, w: int, h: int, *,
                     zpos: int = 0,
                     font: Union[str, os.PathLike, None] = None,
                     size: int = 24,
                     color: Tuple[int, int, int] = (255, 255, 255),
                     bg_color: Optional[Tuple[int, int, int]] = None) -> "Typer":
        """Allocate a new :class:`Typer` (Surface subclass) for text overlay.

        ``font`` may be:
          * ``None`` / ``""`` / ``"sans"`` — fontconfig default sans
          * any string — fontconfig family/pattern (e.g. ``"mono"``,
            ``"DejaVu Serif"``, ``"wqy-zenhei"``)
          * any path (str / :class:`os.PathLike`) — explicit .ttf / .otf / .ttc

        ``color`` / ``bg_color`` are BGR triples (OpenCV convention).
        ``bg_color=None`` (default) means no background fill.

        Failure modes:
          * ``fontconfig`` not installed → :class:`RuntimeError`
          * font not resolvable → :class:`RuntimeError`
        """
        # Surface-rect validation (reuse create_surface's path).
        if x < 0 or y < 0 or w <= 0 or h <= 0:
            raise ValueError(f"SurfaceComposer.create_typer: bad rect {x},{y} {w}x{h}")
        if x + w > self.w or y + h > self.h:
            raise ValueError(
                f"SurfaceComposer.create_typer: rect {x},{y} {w}x{h} "
                f"exceeds frame {self.w}x{self.h}"
            )
        with self._surfaces_lock:
            sid = self._next_surface_id
            self._next_surface_id += 1
            typer = Typer(
                self, sid, x, y, w, h, zpos,
                font=font, size=size, color=color, bg_color=bg_color,
            )
            typer._owner_set(self)
            self._surfaces[sid] = typer
        self.notify()
        return typer

    def read(self, *, timeout: Optional[float] = None) -> np.ndarray:
        """Block until the worker has produced a new frame, then return
        a ``(H, W, 3)`` BGR ``numpy.ndarray`` view over the PBO mapping.
        """
        if not self._frame_ready.wait(timeout=timeout):
            raise TimeoutError("SurfaceComposer.read: timed out waiting for frame")
        self._frame_ready.clear()
        if self._error is not None:
            raise RuntimeError(
                f"SurfaceComposer worker crashed: {self._error}"
            ) from self._error
        if self._pbo_view is None:
            raise RuntimeError("SurfaceComposer.read: no frame available")
        return self._pbo_view

    def close(self) -> None:
        if self._stop.is_set():
            return
        self._stop.set()
        self._wakeup.set()
        self._thread.join(timeout=2.0)
        if self._thread.is_alive():
            raise RuntimeError("SurfaceComposer: worker did not stop within 2s")

    def __enter__(self) -> "SurfaceComposer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Worker thread
    # ------------------------------------------------------------------

    def _worker_main(self) -> None:
        try:
            ctx = _m.MaliContext(self.w, self.h)
            stride_bytes = self.w * 4  # RGBA8

            # Double-buffer the FBO backing so the GPU never overwrites
            # the buffer the consumer is still reading. With multiple
            # producer threads (e.g. mali_multi_decoder.py — 4 parallel
            # VideoCapture threads each writing into its own Surface
            # while the worker composites + readback), the single-buf
            # path shows visible tearing. Two DMA-BUFs cost an extra
            # ``stride * h`` bytes (~1.2 MB at 640x480 RGBA) which is
            # negligible vs. the GPU/CPU win.
            heap = dmaheap.DmaHeap("system")
            buf_a = heap.alloc(stride_bytes * self.h)
            buf_b = heap.alloc(stride_bytes * self.h)
            target = _m.MaliTarget(ctx, buf_a.fd,
                                   self.w, self.h, stride_bytes,
                                   fd_b=buf_b.fd)
            quad_vbo = _m.MaliQuadVbo(ctx)
            nv12_prog, rgb_prog = self._build_programs(ctx)

            self._ctx = ctx
            self._target = target
            self._bufs = (buf_a, buf_b)
            # Per-fd mmap cache. ``swap()`` returns one of the two fds;
            # we keep an mmap for each so we never re-mmap on hot path.
            # Stale entries (fd never returned by swap) are GC'd lazily
            # by capping the dict size.
            self._mmaps: dict[int, mmap.mmap] = {}
            self._last_used_fd: Optional[int] = None
            self._quad_vbo = quad_vbo
            self._nv12_prog = nv12_prog
            self._rgb_prog = rgb_prog

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
            for name in ("_target", "_mmaps", "_bufs", "_quad_vbo",
                         "_nv12_prog", "_rgb_prog", "_ctx"):
                obj = getattr(self, name, None)
                if obj is None:
                    continue
                if isinstance(obj, dict):
                    # mmap cache {fd: mmap.mmap}
                    for m in obj.values():
                        try: m.close()
                        except Exception: pass
                elif isinstance(obj, list):
                    for m in obj:
                        if m is not None:
                            try: m.close()
                            except Exception: pass
                elif isinstance(obj, tuple):
                    for m in obj:
                        try: m.close()
                        except Exception: pass
                else:
                    try:
                        obj.close()
                    except Exception:
                        pass

    def _build_programs(self, ctx: _m.MaliContext):
        vs = (
            b"#version 300 es\n"
            b"in vec2 aPos;\n"
            b"in vec2 aUV;\n"
            b"uniform vec4 uRect;\n"
            b"out vec2 vUV;\n"
            b"void main() {\n"
            b"  vec2 pos = aPos * uRect.zw + uRect.xy;\n"
            b"  vUV = aUV;\n"
            b"  gl_Position = vec4(pos, 0.0, 1.0);\n"
            b"}\n"
        )
        nv12_fs = (
            b"#version 300 es\n"
            b"#extension GL_OES_EGL_image_external_essl3 : require\n"
            b"precision highp float;\n"
            b"in vec2 vUV;\n"
            b"uniform samplerExternalOES texNV;\n"
            b"uniform vec3 yuvOffset;\n"
            b"uniform vec3 yuvScale;\n"
            b"uniform vec3 yuv2r;\n"
            b"uniform vec3 yuv2g;\n"
            b"uniform vec3 yuv2b;\n"
            b"out vec4 fragColor;\n"
            b"void main() {\n"
            b"  vec4 c = texture(texNV, vUV);\n"
            b"  float y = c.r;\n"
            b"  float u = c.g;\n"
            b"  float v = c.b;\n"
            b"  float yv = (y * 255.0 - yuvOffset.x) * yuvScale.x;\n"
            b"  float uv = (u * 255.0 - yuvOffset.y) * yuvScale.y;\n"
            b"  float vv = (v * 255.0 - yuvOffset.z) * yuvScale.z;\n"
            b"  float r = dot(vec3(yv, uv, vv), yuv2r);\n"
            b"  float g = dot(vec3(yv, uv, vv), yuv2g);\n"
            b"  float b = dot(vec3(yv, uv, vv), yuv2b);\n"
            b"  fragColor = vec4(clamp(b, 0.0, 255.0) / 255.0,\n"
            b"                     clamp(g, 0.0, 255.0) / 255.0,\n"
            b"                     clamp(r, 0.0, 255.0) / 255.0,\n"
            b"                     1.0);\n"
            b"}\n"
        )
        rgb_fs = (
            b"#version 300 es\n"
            b"precision highp float;\n"
            b"in vec2 vUV;\n"
            b"uniform sampler2D tex;\n"
            b"out vec4 fragColor;\n"
            b"void main() {\n"
            b"  vec3 c = texture(tex, vUV).rgb;\n"
            b"  fragColor = vec4(c.b, c.g, c.r, 1.0);\n"
            b"}\n"
        )

        nv12 = _m.MaliProgram(
            ctx, vs, nv12_fs,
            uniform_names=("texNV", "yuvOffset", "yuvScale",
                           "yuv2r", "yuv2g", "yuv2b", "uRect"),
            attrib_names=("aPos", "aUV"),
        )
        rgb = _m.MaliProgram(
            ctx, vs, rgb_fs,
            uniform_names=("tex", "uRect"),
            attrib_names=("aPos", "aUV"),
        )
        return nv12, rgb

    def _composite_loop(self) -> None:
        target = self._target
        quad_vbo = self._quad_vbo

        # Cap the worker's composite rate at ~30 Hz to absorb producer
        # notify bursts. With ``mali_multi_decoder.py`` 4 producers at
        # 30 fps, each ``post()`` fires ``notify()`` and wakes the
        # worker. Without this cap the worker would run 4 cycles per
        # 33 ms (one per producer post), each cycle draining only ONE
        # slot per surface — yielding 4 partial frames per producer
        # frame: cycle 1 draws only surface 0, cycle 2 only surface 1,
        # etc. The visible symptom is "the 4 cells aren't aligned" /
        # tearing.
        #
        # Capping at one composite per 30 ms forces the worker to wait
        # until all 4 producers have notified (within ~1 ms of each
        # other), then drain every surface's ``_ready`` queue in one
        # cycle. Result: each composite shows all 4 surfaces' latest
        # content from the same logical moment.
        #
        # The 0.030 s interval matches the slowest expected producer
        # fps (30). Faster producers will queue up 2-3 slots per surface
        # which ``_composite_one`` discards (drain-all keep-latest).
        _MIN_COMPOSITE_INTERVAL = 0.030
        last_composite_t = 0.0

        while not self._stop.is_set():
            self._wakeup.wait(timeout=0.05)
            self._wakeup.clear()
            if self._stop.is_set():
                break

            now = time.monotonic()
            if now - last_composite_t < _MIN_COMPOSITE_INTERVAL:
                # Producer notify came in mid-tick. The previous cycle
                # is too recent to composite again — wait for the next
                # wakeup. The 50 ms wait timeout will eventually let
                # the loop through if producer pacing collapses.
                continue

            surfaces = self._snapshot_surfaces()
            if not surfaces:
                continue
            drawn = self._composite_one(surfaces, quad_vbo)
            if not drawn:
                continue
            self._swap_dmabuf_and_signal(target)
            last_composite_t = time.monotonic()

    def _snapshot_surfaces(self) -> list[Surface]:
        with self._surfaces_lock:
            return list(self._surfaces.values())

    def _composite_one(self, surfaces: list[Surface],
                       quad_vbo: _m.MaliQuadVbo) -> bool:
        """Composite each surface with its LATEST ready slot.

        We do NOT clear the FBO here. With multiple producer threads
        (``mali_multi_decoder.py`` — 4 parallel ``VideoCapture``
        threads each posting into its own Surface), the producers drift
        apart by a few ms. If we cleared, any surface whose ``_ready``
        queue happened to be empty on this composite cycle would render
        as black for that frame — the dreaded "3 of 4 cells are black"
        pattern. Instead, surfaces with nothing new leave their FBO
        region untouched and the previous content stays (lag of ~1
        ping-pong cycle at worst).

        Each surface's ``_ready`` queue may hold 0..3 slots
        (triple-buffered). We drain all of them, keep the one with the
        highest pts, and discard the rest. This eliminates bufferbloat
        (worker never accumulates stale frames) and — paired with the
        30 Hz cap in ``_composite_loop`` — eliminates the per-cycle
        partial-drain that produced tearing (each cycle now grabs the
        latest from every surface in one shot, so all 4 surfaces are
        within a few ms of each other).

        First-frame cost: the DMA-BUFs start zeroed. Until each
        surface has been drawn once, it'll appear black in its region.
        After warmup (≤ a few frames) every region has been touched
        at least once and the no-clear behaviour kicks in.
        """
        self._target.bind()
        drawn = False
        for surf in sorted(surfaces, key=lambda s: s.zpos):
            # Drain all ready slots; keep the one with the highest pts.
            latest: Optional[_BufferSlot] = None
            others: list[_BufferSlot] = []
            while True:
                try:
                    slot = surf._ready.get_nowait()
                except queue.Empty:
                    break
                if latest is None or slot.pending_pts > latest.pending_pts:
                    if latest is not None:
                        others.append(latest)
                    latest = slot
                else:
                    others.append(slot)
            # Return older / newer-but-stale slots to the free pool.
            for s in others:
                s.pending_fd = -1
                s.pending_cpu = None
                surf._free.put(s)

            if latest is None:
                # No new content for this surface on this cycle — its
                # FBO region retains whatever was last drawn (no-clear
                # policy).
                continue

            self._alloc_slot(surf, latest)
            self._draw_surface(surf, latest, quad_vbo)
            latest.pending_fd = -1
            latest.pending_cpu = None
            latest.pts = latest.pending_pts
            drawn = True
            surf._free.put(latest)
        return drawn

    def _alloc_slot(self, surf: Surface, slot: _BufferSlot) -> None:
        if slot.allocated:
            return
        if surf.fmt == _NV12_FMT:
            # Stable texture; per-frame rebind to the latest EGLImage.
            slot.tex = _m.MaliTexture.nv12(self._ctx, surf.w, surf.h)
        else:
            slot.tex = _m.MaliTexture.rgb(self._ctx, surf.w, surf.h)
        slot.allocated = True

    def _draw_surface(self, surf: Surface, slot: _BufferSlot,
                      quad_vbo: _m.MaliQuadVbo) -> None:
        ndc = _rect_pixels(surf.x, surf.y, surf.w, surf.h, self.w, self.h)

        if surf.fmt == _NV12_FMT:
            assert slot.pending_fd >= 0, "NV12 slot must carry pending_fd"
            assert slot.tex is not None
            slot.tex.rebind_fd(slot.pending_fd, surf.w, surf.h,
                               slot.pending_stride)
            _m.draw_textured_quad(
                self._nv12_prog, slot.tex,
                sampler="texNV",
                vec3_uniforms=[
                    ("yuvOffset", *_BT601_OFFSET),
                    ("yuvScale", *_BT601_YUV2R),  # identity: (Y×1.164, …)
                    ("yuv2r", *_BT601_YUV2R),
                    ("yuv2g", *_BT601_YUV2G),
                    ("yuv2b", *_BT601_YUV2B),
                ],
                rect_uniform=("uRect", ndc[0], ndc[1], ndc[2], ndc[3]),
                vbo=quad_vbo,
            )
        else:
            assert slot.pending_cpu is not None, "BGR slot must carry pending_cpu"
            assert slot.tex is not None
            slot.tex.upload_rgb(slot.pending_cpu, 0)
            _m.draw_textured_quad(
                self._rgb_prog, slot.tex,
                sampler="tex",
                rect_uniform=("uRect", ndc[0], ndc[1], ndc[2], ndc[3]),
                vbo=quad_vbo,
            )

    def _swap_dmabuf_and_signal(self, target: _m.MaliTarget) -> None:
        """Flush GPU, sync the DMA-BUF for CPU read, mmap it
        (no glReadPixels), build the BGR numpy view, and signal.

        In double-buffer mode the shim returns a different fd on each
        call (the one the GPU just finished writing). We keep a
        ``{fd: mmap}`` cache so the hot path doesn't re-mmap. Stale
        entries (fd never returned by swap) get evicted lazily when
        the cache exceeds 4 entries.

        The shim flushes the GPU and DMA_BUF_SYNCs the backing the GPU
        just wrote into; we read the result lazily via mmap, with
        cache invalidation handled by DMA_BUF_SYNC_START in the shim.
        """
        fd = target.swap()
        if fd < 0:
            raise RuntimeError("SurfaceComposer: target.swap returned -1")

        size = self.w * self.h * 4  # RGBA8888

        # Lazily mmap the fd (per-fd cache for double-buffer mode).
        mm = self._mmaps.get(fd)
        if mm is None:
            # PROT_READ because the CPU only reads; MAP_SHARED so DMA-BUF
            # coherency is maintained with the GPU side.
            mm = mmap.mmap(
                fd, size, prot=mmap.PROT_READ, flags=mmap.MAP_SHARED,
            )
            self._mmaps[fd] = mm
            # Evict stale entries if cache is growing unboundedly.
            # We expect at most 2 fds (single or double buffer mode);
            # anything beyond is a leak somewhere.
            if len(self._mmaps) > 4:
                stale = [k for k in self._mmaps if k != fd]
                for k in stale:
                    try: self._mmaps[k].close()
                    except Exception: pass
                    del self._mmaps[k]
        self._last_used_fd = fd

        raw = np.frombuffer(mm, dtype=np.uint8
                            ).reshape(self.h, self.w, 4)
        # libmali-valhall-g610-g24p0 1.9's RGBA8888 FBO writes the
        # framebuffer to the consumer DMA-BUF with **alpha in byte 0**
        # (verified empirically against /tmp/test_render_big.c + the
        # surfaceflinger smoke tests). Layout per pixel is therefore
        # ``[A, B, G, R]``. Pick bytes [1, 2, 3] to get BGR (drop A).
        self._pbo_view = raw[:, :, [1, 2, 3]].copy()
        self._frame_count += 1
        self._frame_ready.set()


__all__ = ["SurfaceComposer", "Surface", "Typer"]