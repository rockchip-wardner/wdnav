"""ctypes wrapper for libshim_mali.so — high-level offscreen GL compositor.

This module is the Python-side glue for the C shim in
:mod:`wdnav.cshim_mali` (``libshim_mali.so``). The shim exposes opaque
handles for the offscreen rendering primitives
(:class:`MaliContext`, :class:`MaliTarget`, :class:`MaliProgram`,
:class:`MaliTexture`, :class:`MaliPbo`, :class:`MaliQuadVbo`); this
module:

  * loads the .so from ``<this-package>/lib/libshim_mali.so`` (with a
    fallback to ``libshim_mali`` via ``ldconfig``);
  * sets ``argtypes`` / ``restype`` for every shim entry point;
  * mirrors the EGL/GL_*/DRM fourcc constants composer.py uses
    (kept for the docstring references in other modules — SurfaceComposer
    itself no longer touches them);
  * exposes the high-level Python classes that wrap each handle.

The previous version of this module was a 845-line ctypes wrapper that
drove ``libmali.so`` directly. It is replaced by this thin shim layer to
match the ``cshim_video_*.c`` pattern used by :mod:`wdnav.mpp`.
"""

from __future__ import annotations

import ctypes
import os
import threading
from ctypes import (
    POINTER,
    Structure,
    c_char_p,
    c_int,
    c_int32,
    c_void_p,
)
from typing import Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Library loading
# ---------------------------------------------------------------------------

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_LIB_PATH = os.path.join(_THIS_DIR, "lib", "libshim_mali.so")


def _load_library():
    last_err: Optional[OSError] = None
    if os.path.exists(_LIB_PATH):
        try:
            return ctypes.CDLL(_LIB_PATH, mode=ctypes.RTLD_GLOBAL)
        except OSError as e:
            last_err = e
    try:
        return ctypes.CDLL("libshim_mali", mode=ctypes.RTLD_GLOBAL)
    except OSError as e:
        raise OSError(
            "Could not load libshim_mali.so from " + _LIB_PATH + " or via "
            "ldconfig. The C shim is built from "
            "src/wdnav/cshim_mali.c by _build_wdnav.py during wheel "
            "install; if the .so is missing the wheel was built without "
            "the Mali shim step. Install libmali system-wide (e.g. "
            "apt install libmali-valhall-g610-g24p0-x11-wayland-gbm) "
            "so the shim can dlopen libmali.so at runtime. Last error: "
            + repr(last_err or e)
        )


_lib = _load_library()

# Serialise every shim call. The shim is single-threaded by design and
# composer.py only ever calls it from one worker thread; the lock
# here is defensive so that if a future caller invokes MaliContext /
# MaliTarget methods from multiple threads they don't race on the GL
# state machine.
_mali_lock = threading.Lock()


# ---------------------------------------------------------------------------
# err_buf + uniform-vec3 helpers
# ---------------------------------------------------------------------------

_ERR_BUF_SIZE = 512


def _make_err_buf():
    """Return a fresh ctypes string buffer the C shim fills on failure."""
    return ctypes.create_string_buffer(_ERR_BUF_SIZE)


def _check(ret: int, where: str, err_buf=None) -> None:
    """Raise RuntimeError if the shim returned a negative status."""
    if ret < 0:
        msg = ""
        if err_buf is not None and err_buf.value:
            try:
                msg = err_buf.value.decode("utf-8", "replace")
            except Exception:
                msg = "<undecodable err_buf>"
        raise RuntimeError(
            f"{where} failed: {msg}" if msg else f"{where} failed"
        )


class _UniformVec3(Structure):
    """Mirror of ``rkm_uniform_vec3_t`` in cshim_mali.c."""
    _fields_ = [
        ("name", c_char_p),
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("z", ctypes.c_float),
    ]


# ---------------------------------------------------------------------------
# argtypes / restype for every shim entry point
# ---------------------------------------------------------------------------

_lib.rkm_ctx_create.argtypes = [
    c_int, c_int, POINTER(c_void_p), c_char_p, c_int,
]
_lib.rkm_ctx_create.restype = c_int

_lib.rkm_ctx_make_current.argtypes = [c_void_p]
_lib.rkm_ctx_make_current.restype = c_int

_lib.rkm_ctx_is_pbuffer.argtypes = [c_void_p]
_lib.rkm_ctx_is_pbuffer.restype = c_int

_lib.rkm_ctx_release.argtypes = [c_void_p]
_lib.rkm_ctx_release.restype = c_int

_lib.rkm_ctx_destroy.argtypes = [c_void_p]
_lib.rkm_ctx_destroy.restype = None

_lib.rkm_target_create.argtypes = [
    c_void_p, c_int, c_int, c_int, c_int, c_int, c_int,
    POINTER(c_void_p), c_char_p, c_int,
]
_lib.rkm_target_create.restype = c_int

_lib.rkm_target_bind.argtypes = [c_void_p]
_lib.rkm_target_bind.restype = c_int

_lib.rkm_target_clear.argtypes = [
    c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
]
_lib.rkm_target_clear.restype = c_int

_lib.rkm_target_swap.argtypes = [
    c_void_p, POINTER(c_int), c_char_p, c_int,
]
_lib.rkm_target_swap.restype = c_int

_lib.rkm_target_destroy.argtypes = [c_void_p]
_lib.rkm_target_destroy.restype = None

_lib.rkm_program_create.argtypes = [
    c_void_p,
    c_char_p, c_int,        # vs_src, vs_len
    c_char_p, c_int,        # fs_src, fs_len
    POINTER(c_char_p), c_int,   # uniform names + count
    POINTER(c_char_p), c_int,   # attrib names + count
    POINTER(c_void_p),
    c_char_p, c_int,
]
_lib.rkm_program_create.restype = c_int

_lib.rkm_program_uniform_loc.argtypes = [c_void_p, c_char_p]
_lib.rkm_program_uniform_loc.restype = c_int

_lib.rkm_program_attrib_loc.argtypes = [c_void_p, c_char_p]
_lib.rkm_program_attrib_loc.restype = c_int

_lib.rkm_program_destroy.argtypes = [c_void_p]
_lib.rkm_program_destroy.restype = None

_lib.rkm_texture_create_nv12.argtypes = [
    c_void_p, c_int, c_int, POINTER(c_void_p), c_char_p, c_int,
]
_lib.rkm_texture_create_nv12.restype = c_int

_lib.rkm_texture_rebind_nv12.argtypes = [
    c_void_p, c_int, c_int, c_int, c_int, c_char_p, c_int,
]
_lib.rkm_texture_rebind_nv12.restype = c_int

_lib.rkm_texture_create_rgb.argtypes = [
    c_void_p, c_int, c_int, POINTER(c_void_p), c_char_p, c_int,
]
_lib.rkm_texture_create_rgb.restype = c_int

_lib.rkm_texture_upload_rgb.argtypes = [c_void_p, c_void_p, c_int]
_lib.rkm_texture_upload_rgb.restype = c_int

_lib.rkm_texture_destroy.argtypes = [c_void_p]
_lib.rkm_texture_destroy.restype = None

# MaliText (freetype-backed CPU text rasteriser; lives in cshim_mali.c).
_lib.rkm_text_create.argtypes = [
    c_char_p, c_int, POINTER(c_void_p), c_char_p, c_int,
]
_lib.rkm_text_create.restype = c_int

_lib.rkm_text_set_size.argtypes = [c_void_p, c_int, c_char_p, c_int]
_lib.rkm_text_set_size.restype = c_int

_lib.rkm_text_destroy.argtypes = [c_void_p]
_lib.rkm_text_destroy.restype = None

_lib.rkm_text_render.argtypes = [
    c_void_p, c_char_p,
    c_int, c_int,
    c_int, c_int, c_int,
    c_void_p, c_int, c_int,
    POINTER(c_int), POINTER(c_int),
    c_char_p, c_int,
]
_lib.rkm_text_render.restype = c_int

_lib.rkm_text_measure.argtypes = [
    c_void_p, c_char_p,
    POINTER(c_int), POINTER(c_int),
    c_char_p, c_int,
]
_lib.rkm_text_measure.restype = c_int

_lib.rkm_quad_vbo_create.argtypes = [c_void_p, POINTER(c_void_p), c_char_p, c_int]
_lib.rkm_quad_vbo_create.restype = c_int

_lib.rkm_quad_vbo_destroy.argtypes = [c_void_p]
_lib.rkm_quad_vbo_destroy.restype = None

_lib.rkm_draw_textured_quad.argtypes = [
    c_void_p, c_void_p, c_char_p,
    POINTER(_UniformVec3), c_int,
    c_char_p,
    ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
    c_void_p,
    c_char_p, c_int,
]
_lib.rkm_draw_textured_quad.restype = c_int


# ---------------------------------------------------------------------------
# Constants — mirror of cshim_mali.c. composer.py itself no longer
# reads these; they remain here so module docstrings and external code
# that referenced e.g. ``wdnav.mali.EGL_LINUX_DRM_FOURCC_EXT`` still
# resolve.
# ---------------------------------------------------------------------------

EGL_NONE = 0x3038
EGL_NO_CONTEXT = 0
EGL_NO_DISPLAY = 0
EGL_NO_SURFACE = 0
EGL_DEFAULT_DISPLAY = 0
EGL_LINUX_DMA_BUF_EXT = 0x3278
EGL_LINUX_DRM_FOURCC_EXT = 0x327D
EGL_DMA_BUF_PLANE0_FD_EXT = 0x3272
EGL_DMA_BUF_PLANE0_OFFSET_EXT = 0x3273
EGL_DMA_BUF_PLANE0_PITCH_EXT = 0x3274
EGL_DMA_BUF_PLANE1_FD_EXT = 0x3275
EGL_DMA_BUF_PLANE1_OFFSET_EXT = 0x3276
EGL_DMA_BUF_PLANE1_PITCH_EXT = 0x3277
EGL_WIDTH = 0x3056
EGL_HEIGHT = 0x3050
EGL_RED_SIZE = 0x3024
EGL_GREEN_SIZE = 0x3023
EGL_BLUE_SIZE = 0x3022
EGL_ALPHA_SIZE = 0x3021
EGL_RENDERABLE_TYPE = 0x3040
EGL_OPENGL_ES3_BIT = 0x0040

GL_FALSE = 0
GL_TRUE = 1
GL_DEPTH_TEST = 0x0B71
GL_BLEND = 0x0BE2
GL_COLOR_BUFFER_BIT = 0x00004000
GL_TEXTURE_2D = 0x0DE1
GL_TEXTURE_EXTERNAL_OES = 0x8D65
GL_TEXTURE0 = 0x84C0
GL_LINEAR = 0x2601
GL_CLAMP_TO_EDGE = 0x812F
GL_RGB8 = 0x8051
GL_RGB = 0x1907
GL_RGBA = 0x1908
GL_RGBA8 = 0x8058
GL_UNSIGNED_BYTE = 0x1401
GL_FLOAT = 0x1406
GL_TRIANGLE_STRIP = 0x0005


class DRMFormat:
    """DRM fourcc pixel format codes used by ``EGL_EXT_image_dma_buf_import``.

    Values come straight from ``/usr/include/libdrm/drm_fourcc.h`` —
    ``fourcc_code(c0, c1, c2, c3) = (c0) | (c1<<8) | (c2<<16) | (c3<<24)``.
    """

    ARGB8888 = 0x34325241  # 'A','R','2','4' */
    XRGB8888 = 0x34325258  # 'X','R','2','4' */
    RGBA8888 = 0x41424752  # 'R','G','B','A' */
    RGB888   = 0x33424752  # 'R','G','B','3' */
    BGR888   = 0x33524742  # 'B','G','R','3' */
    NV12     = 0x3231564E  # 'N','V','1','2' */
    NV21     = 0x3132564E  # 'N','V','2','1' */


# ---------------------------------------------------------------------------
# Handle wrappers
# ---------------------------------------------------------------------------

class _OwnedHandle:
    """Mixin for a ctypes handle that owns a C-side resource.

    Subclasses must call ``super().__init__(handle, destroy)`` once they
    have a non-null ``c_void_p`` from the shim. ``close()`` is idempotent
    and the object also releases on ``__del__`` so a forgotten handle
    doesn't leak on the C side.
    """

    __slots__ = ("_handle", "_destroy")

    def __init__(self, handle: c_void_p, destroy) -> None:
        self._handle = handle
        self._destroy = destroy

    @property
    def handle(self) -> c_void_p:
        return self._handle

    def close(self) -> None:
        if self._handle:
            with _mali_lock:
                self._destroy(self._handle)
            self._handle = c_void_p(0)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class MaliContext(_OwnedHandle):
    """Owns the EGL display / context / PBuffer for the offscreen compositor.

    Construct with the frame size. ``close()`` terminates the EGL
    display. The :class:`MaliTarget` / :class:`MaliProgram` /
    :class:`MaliTexture` / :class:`MaliPbo` / :class:`MaliQuadVbo`
    objects created against a context become invalid after the context
    is closed (the C shim doesn't track ownership so calling their
    methods after close is undefined behaviour).
    """

    def __init__(self, width: int, height: int) -> None:
        h = c_void_p(0)
        err = _make_err_buf()
        with _mali_lock:
            ret = _lib.rkm_ctx_create(
                c_int(width), c_int(height),
                ctypes.byref(h), err, c_int(_ERR_BUF_SIZE),
            )
        _check(ret, "rkm_ctx_create", err)
        super().__init__(h, _lib.rkm_ctx_destroy)

    def make_current(self) -> None:
        with _mali_lock:
            _check(_lib.rkm_ctx_make_current(self.handle), "rkm_ctx_make_current")

    @property
    def is_pbuffer(self) -> bool:
        """True if the shim successfully created an EGL PBuffer surface,
        False if it fell back to ``EGL_KHR_surfaceless_context``.

        On libmali builds that advertise PBuffer-capable configs (most
        Rockchip BSP builds before 1.9) this returns True; on builds
        that reject PBuffer creation (libmali-valhall-g610-g24p0 1.9
        on Debian bookworm arm64 is one), it returns False and the
        compositor runs through the surfaceless fallback path. The
        composite + readback surface is identical either way.
        """
        with _mali_lock:
            return bool(_lib.rkm_ctx_is_pbuffer(self.handle))

    @property
    def is_pbuffer(self) -> bool:
        """True if the shim successfully created an EGL PBuffer surface,
        False if it fell back to ``EGL_KHR_surfaceless_context``.

        On libmali builds that advertise PBuffer-capable configs (most
        Rockchip BSP builds before 1.9) this returns True; on builds
        that reject PBuffer creation (libmali-valhall-g610-g24p0 1.9
        on Debian bookworm arm64 is one), it returns False and the
        compositor runs through the surfaceless fallback path. The
        composite + readback surface is identical either way.
        """
        with _mali_lock:
            return bool(_lib.rkm_ctx_is_pbuffer(self.handle))

    def release(self) -> None:
        with _mali_lock:
            _check(_lib.rkm_ctx_release(self.handle), "rkm_ctx_release")


class MaliTarget(_OwnedHandle):
    """Offscreen render target backed by one or two DMA-BUF fds.

    The FBO color renderbuffer is bound to a DMA-BUF via an EGLImage
    (``EGL_EXT_image_dma_buf_import`` + ``glEGLImageTargetRenderbufferStorageOES``),
    so the GPU writes land directly in mmap'd consumer memory — no
    ``glReadPixels`` involved. ``swap`` flushes the GPU, syncs the
    DMA-BUF for CPU read, and returns its fd so the caller can mmap it.

    Single-buffer mode (default, ``enable_double_buf=False``):
      The shim has a single backing; ``swap`` returns the same fd every
      time. If the consumer can't keep up with the worker (e.g. a slow
      ``cv2.imshow`` loop, or 4 surface producers writing concurrently),
      the GPU's next frame may overwrite the consumer's read buffer →
      visible tearing.

    Double-buffer mode (``enable_double_buf=True``):
      The shim ping-pongs between two backings A and B. ``swap`` returns
      whichever one the GPU *just finished writing* — the OTHER one is
      now bound to the FBO for the GPU's next draw, so the GPU never
      overwrites the buffer the consumer is reading. The returned fd
      therefore alternates between two values. **Callers must cache the
      mmap per-fd** (or rebuild it on every swap); the old ``mmap on
      first use, reuse forever`` pattern is unsafe in this mode.

    ``fd`` (and ``fd_b`` in double-buffer mode) are caller-owned DMA-BUF
    file descriptors of size ``>= stride_bytes * height`` (RGBA8888
    layout — 4 bytes per pixel). The shim does not close them;
    ``mali_close``-equivalent is the caller's responsibility (typically
    the wrapping :class:`wdnav.DmaBuf` instances).
    """

    def __init__(self, ctx: MaliContext,
                 fd: int, width: int, height: int, stride_bytes: int,
                 *,
                 fd_b: Optional[int] = None) -> None:
        enable_db = fd_b is not None
        fd_b_arg = int(fd_b) if enable_db else -1
        h = c_void_p(0)
        err = _make_err_buf()
        with _mali_lock:
            ret = _lib.rkm_target_create(
                ctx.handle,
                c_int(int(fd)),
                c_int(fd_b_arg),
                c_int(int(width)), c_int(int(height)), c_int(int(stride_bytes)),
                c_int(1 if enable_db else 0),
                ctypes.byref(h), err, c_int(_ERR_BUF_SIZE),
            )
        _check(ret, "rkm_target_create", err)
        super().__init__(h, _lib.rkm_target_destroy)
        self._width = width
        self._height = height
        self._stride_bytes = stride_bytes
        self._fds = (int(fd), int(fd_b_arg)) if enable_db else (int(fd), -1)
        self._double_buf = enable_db

    @property
    def double_buf(self) -> bool:
        """True if this target was created with two DMA-BUF backings."""
        return self._double_buf

    @property
    def fds(self) -> tuple:
        """Tuple of DMA-BUF fds the target was created with. Length 1
        in single-buffer mode, length 2 in double-buffer mode. The
        order matches the indices the shim ping-pongs between on
        successive ``swap()`` calls.
        """
        return self._fds

    def bind(self) -> None:
        with _mali_lock:
            _check(_lib.rkm_target_bind(self.handle), "rkm_target_bind")

    def clear(self, r: float = 0.0, g: float = 0.0,
              b: float = 0.0, a: float = 1.0) -> None:
        with _mali_lock:
            _check(
                _lib.rkm_target_clear(self.handle,
                                      ctypes.c_float(r), ctypes.c_float(g),
                                      ctypes.c_float(b), ctypes.c_float(a)),
                "rkm_target_clear",
            )

    def swap(self) -> int:
        """Flush GPU, DMA_BUF_SYNC the backing for CPU read, and
        return its fd so the caller can ``mmap`` it.

        In double-buffer mode the returned fd alternates between the
        two backings, so the caller must keep a ``{fd: mmap}`` cache.
        """
        out = c_int(-1)
        err = _make_err_buf()
        with _mali_lock:
            _check(
                _lib.rkm_target_swap(self.handle, ctypes.byref(out),
                                      err, c_int(_ERR_BUF_SIZE)),
                "rkm_target_swap", err,
            )
        return out.value


class MaliProgram(_OwnedHandle):
    """Compiled + linked shader program with a uniform / attrib cache."""

    def __init__(self, ctx: MaliContext,
                 vs_src: bytes, fs_src: bytes,
                 uniform_names: Sequence[str] = (),
                 attrib_names: Sequence[str] = ()) -> None:
        vs_buf = ctypes.create_string_buffer(vs_src)
        fs_buf = ctypes.create_string_buffer(fs_src)
        u_arr = (c_char_p * len(uniform_names))(
            *[n.encode("ascii") for n in uniform_names]
        ) if uniform_names else (c_char_p * 0)()
        a_arr = (c_char_p * len(attrib_names))(
            *[n.encode("ascii") for n in attrib_names]
        ) if attrib_names else (c_char_p * 0)()

        h = c_void_p(0)
        err = _make_err_buf()
        with _mali_lock:
            ret = _lib.rkm_program_create(
                ctx.handle,
                vs_buf, c_int(len(vs_src)),
                fs_buf, c_int(len(fs_src)),
                u_arr, c_int(len(uniform_names)),
                a_arr, c_int(len(attrib_names)),
                ctypes.byref(h),
                err, c_int(_ERR_BUF_SIZE),
            )
        _check(ret, "rkm_program_create", err)
        super().__init__(h, _lib.rkm_program_destroy)

    def uniform_loc(self, name: str) -> int:
        with _mali_lock:
            return _lib.rkm_program_uniform_loc(self.handle, name.encode("ascii"))

    def attrib_loc(self, name: str) -> int:
        with _mali_lock:
            return _lib.rkm_program_attrib_loc(self.handle, name.encode("ascii"))


class MaliTexture(_OwnedHandle):
    """Texture handle; NV12 (external + EGLImage) or RGB (2D + CPU upload)."""

    def __init__(self) -> None:
        raise TypeError(
            "use MaliTexture.nv12(...) / MaliTexture.nv12_with_fd(...) / "
            "MaliTexture.rgb(...) — the ctor is private"
        )

    @classmethod
    def _wrap(cls, handle: c_void_p) -> "MaliTexture":
        obj = cls.__new__(cls)
        obj._handle = handle
        obj._destroy = _lib.rkm_texture_destroy
        return obj

    @classmethod
    def nv12(cls, ctx: MaliContext, width: int, height: int) -> "MaliTexture":
        """Create an empty NV12 GL_TEXTURE_EXTERNAL_OES texture. Pair
        with :meth:`rebind_fd` once you have a producer fd."""
        h = c_void_p(0)
        err = _make_err_buf()
        with _mali_lock:
            ret = _lib.rkm_texture_create_nv12(
                ctx.handle, c_int(width), c_int(height),
                ctypes.byref(h), err, c_int(_ERR_BUF_SIZE),
            )
        _check(ret, "rkm_texture_create_nv12", err)
        return cls._wrap(h)

    def rebind_fd(self, fd: int, width: int, height: int, stride: int) -> None:
        """Wrap the latest DMA-BUF fd as an EGLImage and bind it to this
        texture. Destroys any previously-bound image."""
        err = _make_err_buf()
        with _mali_lock:
            _check(
                _lib.rkm_texture_rebind_nv12(
                    self.handle, c_int(fd), c_int(width), c_int(height),
                    c_int(stride), err, c_int(_ERR_BUF_SIZE),
                ),
                "rkm_texture_rebind_nv12", err,
            )

    @classmethod
    def rgb(cls, ctx: MaliContext, width: int, height: int) -> "MaliTexture":
        """Create an empty GL_TEXTURE_2D / GL_RGB8 texture."""
        h = c_void_p(0)
        err = _make_err_buf()
        with _mali_lock:
            ret = _lib.rkm_texture_create_rgb(
                ctx.handle, c_int(width), c_int(height),
                ctypes.byref(h), err, c_int(_ERR_BUF_SIZE),
            )
        _check(ret, "rkm_texture_create_rgb", err)
        return cls._wrap(h)

    def upload_rgb(self, pixels, pitch: int = 0) -> None:
        """Upload a packed (or row-pitched) RGB8 ndarray / buffer."""
        if hasattr(pixels, "ctypes"):
            ptr = pixels.ctypes.data_as(c_void_p)
        else:
            ptr = c_void_p(int(pixels))
        with _mali_lock:
            _check(
                _lib.rkm_texture_upload_rgb(self.handle, ptr, c_int(pitch)),
                "rkm_texture_upload_rgb",
            )


class MaliQuadVbo(_OwnedHandle):
    """Fullscreen quad VBO (TRIANGLE_STRIP, 4 verts, position+UV)."""

    def __init__(self, ctx: MaliContext) -> None:
        h = c_void_p(0)
        err = _make_err_buf()
        with _mali_lock:
            ret = _lib.rkm_quad_vbo_create(ctx.handle, ctypes.byref(h),
                                            err, c_int(_ERR_BUF_SIZE))
        _check(ret, "rkm_quad_vbo_create", err)
        super().__init__(h, _lib.rkm_quad_vbo_destroy)


def draw_textured_quad(prog: MaliProgram,
                       tex: MaliTexture,
                       *,
                       sampler: Optional[str] = None,
                       vec3_uniforms=(),
                       rect_uniform: Optional[Tuple[str, float, float, float, float]] = None,
                       vbo: MaliQuadVbo) -> None:
    """Bind ``prog`` + ``tex``, set uniforms, draw a textured quad.

    ``vec3_uniforms`` is an iterable of ``(name, x, y, z)`` tuples;
    ``rect_uniform`` is a ``(name, cx, cy, hw, hh)`` tuple. The shim
    silently skips uniforms not present in the program.
    """
    arr_t = _UniformVec3 * max(1, len(vec3_uniforms))
    arr = arr_t()
    n_uniforms = 0
    for i, item in enumerate(vec3_uniforms):
        if len(item) != 4:
            raise ValueError(
                f"vec3_uniforms[{i}] must be (name, x, y, z); got {item!r}"
            )
        name, x, y, z = item
        arr[i].name = ctypes.c_char_p(name.encode("ascii"))
        arr[i].x = ctypes.c_float(x)
        arr[i].y = ctypes.c_float(y)
        arr[i].z = ctypes.c_float(z)
        n_uniforms = i + 1

    if rect_uniform is None:
        rect_name = c_char_p()
        rect_cx = rect_cy = rect_hw = rect_hh = 0.0
    else:
        if len(rect_uniform) != 5:
            raise ValueError(
                f"rect_uniform must be (name, cx, cy, hw, hh); got {rect_uniform!r}"
            )
        rect_name = c_char_p(rect_uniform[0].encode("ascii"))
        _, rect_cx, rect_cy, rect_hw, rect_hh = rect_uniform

    sampler_c = c_char_p(sampler.encode("ascii")) if sampler else c_char_p()

    err = _make_err_buf()
    with _mali_lock:
        ret = _lib.rkm_draw_textured_quad(
            prog.handle, tex.handle, sampler_c,
            arr, c_int(n_uniforms),
            rect_name,
            ctypes.c_float(rect_cx), ctypes.c_float(rect_cy),
            ctypes.c_float(rect_hw), ctypes.c_float(rect_hh),
            vbo.handle,
            err, c_int(_ERR_BUF_SIZE),
        )
    _check(ret, "rkm_draw_textured_quad", err)


class MaliText(_OwnedHandle):
    """CPU-side text rasteriser (freetype ctypes via the shim).

    Wraps the ``rkm_text_*`` c-shim functions. Owns a freetype FT_Face
    handle (loaded from a .ttf / .otf / .ttc file) plus a small per-
    (codepoint, size) glyph bitmap cache. Use ``render`` to blit a UTF-8
    string with alpha blending into a caller-owned BGR888 ``uint8``
    buffer (typically the ``Surface.latest_ndarray``).

    The shim does the actual glyph rasterisation; this class is a thin
    wrapper that owns the lifecycle and forwards utf8 + buffer + color.
    """

    @classmethod
    def clip_from_string(cls, s: str) -> bytes:
        """Return ``s`` as a NUL-terminated UTF-8 ``c_char_p``-compatible
        bytes object. Always includes the trailing NUL.
        """
        return s.encode("utf-8") + b"\x00"

    def __init__(self, font_path: str, size_px: int) -> None:
        if not font_path:
            raise ValueError("MaliText: font_path must be a non-empty string")
        if size_px <= 0:
            raise ValueError(f"MaliText: size_px must be > 0, got {size_px}")

        h = c_void_p(0)
        err = _make_err_buf()
        # ctypes passes c_char_p from a bytes object directly.
        with _mali_lock:
            ret = _lib.rkm_text_create(
                font_path.encode("utf-8"),
                c_int(int(size_px)),
                ctypes.byref(h),
                err, c_int(_ERR_BUF_SIZE),
            )
        _check(ret, "rkm_text_create", err)
        super().__init__(h, _lib.rkm_text_destroy)
        self._size_px = int(size_px)
        self._font_path = font_path

    @property
    def font_path(self) -> str:
        """Path to the .ttf / .otf / .ttc file backing this text handle."""
        return self._font_path

    @property
    def size_px(self) -> int:
        """Current freetype point size in pixels."""
        return self._size_px

    def set_size(self, size_px: int) -> None:
        """Resize the text handle (frees the glyph cache internally).

        Subsequent calls to :meth:`render` / :meth:`measure` will use
        the new size. Cheap when ``size_px`` matches the current size.
        """
        if size_px <= 0:
            raise ValueError(f"MaliText.set_size: size_px must be > 0, got {size_px}")
        if size_px == self._size_px:
            return
        err = _make_err_buf()
        with _mali_lock:
            ret = _lib.rkm_text_set_size(
                self.handle, c_int(int(size_px)),
                err, c_int(_ERR_BUF_SIZE),
            )
        _check(ret, "rkm_text_set_size", err)
        self._size_px = int(size_px)

    def render(self, s: str, x: int, y: int, *,
               color_bgr: Tuple[int, int, int],
               bgr_buf, buf_w: int, buf_h: int) -> Tuple[int, int]:
        """Render ``s`` with alpha blending onto a BGR888 uint8 buffer.

        ``x``, ``y`` is the baseline-left of the rendered text (matches
        freetype's pen convention). ``color_bgr`` is a (B, G, R) triple.

        Returns the (drawn_w, drawn_h) actually inked in pixels. Empty
        strings return (0, 0).
        """
        if not s:
            return (0, 0)
        if bgr_buf is None or buf_w <= 0 or buf_h <= 0:
            raise ValueError(
                f"MaliText.render: invalid buffer buf_w={buf_w} buf_h={buf_h}"
            )
        cb, cg, cr = (int(c) for c in color_bgr)
        utf8 = MaliText.clip_from_string(s)
        out_w = c_int(0)
        out_h = c_int(0)
        err = _make_err_buf()
        # bgr_buf may be a numpy ndarray; ctypes can read its data pointer.
        if hasattr(bgr_buf, "ctypes"):
            ptr = bgr_buf.ctypes.data_as(c_void_p)
        else:
            ptr = c_void_p(int(bgr_buf))
        with _mali_lock:
            ret = _lib.rkm_text_render(
                self.handle, utf8,
                c_int(int(x)), c_int(int(y)),
                c_int(cb), c_int(cg), c_int(cr),
                ptr, c_int(int(buf_w)), c_int(int(buf_h)),
                ctypes.byref(out_w), ctypes.byref(out_h),
                err, c_int(_ERR_BUF_SIZE),
            )
        _check(ret, "rkm_text_render", err)
        return (out_w.value, out_h.value)

    def measure(self, s: str) -> Tuple[int, int]:
        """Return the (ink_w, line_h) the current size would render.

        ``ink_w`` is the width of the visible glyph cluster (so callers
        can right-align, etc.); ``line_h`` is the font line height
        (ascent + descent + leading) at the current size, useful for
        vertical centring multi-line text.
        """
        if not s:
            return (0, 0)
        utf8 = MaliText.clip_from_string(s)
        out_w = c_int(0)
        out_h = c_int(0)
        err = _make_err_buf()
        with _mali_lock:
            ret = _lib.rkm_text_measure(
                self.handle, utf8,
                ctypes.byref(out_w), ctypes.byref(out_h),
                err, c_int(_ERR_BUF_SIZE),
            )
        _check(ret, "rkm_text_measure", err)
        return (out_w.value, out_h.value)


__all__ = [
    "MaliContext",
    "MaliTarget",
    "MaliProgram",
    "MaliTexture",
    "MaliQuadVbo",
    "MaliText",
    "draw_textured_quad",
    "DRMFormat",
]