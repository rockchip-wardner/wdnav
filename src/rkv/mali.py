"""ctypes wrapper for the Mali GLES driver (``libmali.so``).

This module exposes the subset of EGL / GLES / DRM-fourcc functions needed
by :mod:`rkv.surfaceflinger` to run an offscreen GL compositor on Rockchip:

* ``EGLDisplay`` / ``EGLContext`` lifecycle on a surfaceless ES 3.x context.
* ``EGL_EXT_image_dma_buf_import`` to wrap a producer-side DMA-BUF fd as a
  zero-copy ``EGLImage`` (the consumer side never touches the pixels).
* ``glEGLImageTargetTexture2DOES`` / ``glEGLImageTargetRenderbufferStorageOES``
  for binding those images into the GL pipeline.
* ``glFenceSync`` / ``glClientWaitSync`` for GPU-side frame barriers.
* DRM fourcc constants used by the dma_buf import attribute list.

The library is **not** bundled with this package. It must be installed
system-wide; if neither ``/usr/lib/aarch64-linux-gnu/libmali.so`` nor an
ldconfig-registered ``mali`` resolves, the import of :mod:`rkv`
``SurfaceFlinger`` falls back to ``None`` (see :mod:`rkv.__init__`).

Typical use (offscreen composition):

    import rkv.mali as mali
    dpy = mali.get_default_display()
    mali.initialize(dpy)
    ctx = mali.create_context(dpy, cfg, client_version=3)
    mali.make_current_surfaceless(dpy, ctx)
    tex = mali.create_texture_2d_from_dma_buf(dpy, fd, w, h, DRMFormat.NV12, ...)
"""

from __future__ import annotations

import ctypes
import os
import threading
from ctypes import (
    POINTER,
    c_char_p,
    c_int,
    c_int32,
    c_int64,
    c_size_t,
    c_uint8,
    c_uint32,
    c_uint64,
    c_void_p,
)
from typing import Optional, Tuple


_LIBS_TO_TRY = (
    "/usr/lib/aarch64-linux-gnu/libmali.so",
)


def _load_library():
    last_err: Optional[OSError] = None
    for path in _LIBS_TO_TRY:
        if os.path.exists(path):
            try:
                return ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            except OSError as e:
                last_err = e
                continue
    try:
        return ctypes.CDLL("mali", mode=ctypes.RTLD_GLOBAL)
    except OSError as e:
        raise OSError(
            "Could not load libmali.so from system paths. Install the "
            "Rockchip Mali GLES driver (libmali-valhall-g610-g24p0-*) "
            "system-wide, or place libmali.so under "
            "/usr/lib/aarch64-linux-gnu/ and run `ldconfig`. "
            "Last error: " + repr(last_err or e)
        )


_lib = _load_library()


_mali_lock = threading.RLock()


class DRMFormat:
    """DRM fourcc pixel format codes used by ``EGL_EXT_image_dma_buf_import``.

    Values match ``drm_fourcc.h`` on aarch64 Linux. Only the formats rkv's
    compositor actually consumes are listed.
    """

    ARGB8888 = 0x34325241
    XRGB8888 = 0x34325258
    RGBA8888 = 0x34324152
    RGB888 = 0x34324752
    BGR888 = 0x33524742

    NV12 = 0x3231564E
    NV21 = 0x3132564E

    R8 = 0x20203852
    GR88 = 0x38385247
    RG88 = 0x38384752


_BPP_BY_DRM = {
    DRMFormat.ARGB8888: 4,
    DRMFormat.XRGB8888: 4,
    DRMFormat.RGBA8888: 4,
    DRMFormat.RGB888: 3,
    DRMFormat.BGR888: 3,
    DRMFormat.NV12: 1.5,
    DRMFormat.NV21: 1.5,
    DRMFormat.R8: 1,
    DRMFormat.GR88: 2,
    DRMFormat.RG88: 2,
}


def bytes_per_pixel_drm(fmt: int) -> float:
    try:
        return _BPP_BY_DRM[fmt]
    except KeyError as e:
        raise ValueError(f"unknown DRM fourcc: 0x{fmt:x}") from e


EGL_DEFAULT_DISPLAY = 0
EGL_NO_DISPLAY = 0
EGL_NO_CONTEXT = 0
EGL_NO_SURFACE = 0
EGL_FALSE = 0
EGL_TRUE = 1
EGL_NONE = 0x3038
EGL_OPENGL_ES_API = 0x3140
EGL_OPENGL_API = 0x3142
EGL_NO_RESET_NOTIFICATION_KHR = 0x3101
EGL_CONTEXT_LOST = 0x300E
EGL_CONTEXT_CLIENT_VERSION = 0x3098
EGL_PLATFORM_SURFACELESS_MESA = 0x31DD
EGL_PLATFORM_DEVICE_EXT = 0x313F
EGL_PLATFORM_GBM_KHR = 0x31D7
EGL_PLATFORM_WAYLAND_KHR = 0x31D8
EGL_PLATFORM_X11_KHR = 0x31D9
EGL_PBUFFER_BIT = 0x0001
EGL_PIXMAP_BIT = 0x0002
EGL_WINDOW_BIT = 0x0004
EGL_VG_COLORSPACE_LINEAR_BIT = 0x0020
EGL_VG_ALPHA_FORMAT_PRE_BIT = 0x0040
EGL_MULTISAMPLE_RESOLVE_BOX_BIT = 0x0200
EGL_SWAP_BEHAVIOR_PRESERVED_BIT = 0x0400
EGL_BIND_TO_TEXTURE_RGB = 0x303D
EGL_BIND_TO_TEXTURE_RGBA = 0x303E
EGL_TEXTURE_FORMAT = 0x3080
EGL_TEXTURE_2D = 0x305B
EGL_TEXTURE_RGB = 0x305D
EGL_TEXTURE_RGBA = 0x305E
EGL_TEXTURE_TARGET = 0x3081
EGL_MIPMAP_TEXTURE = 0x3082
EGL_NO_TEXTURE = 0x305C
EGL_BACK_BUFFER = 0x3084
EGL_LINUX_DMA_BUF_EXT = 0x3278
EGL_LINUX_DRM_FOURCC_EXT = 0x327D
EGL_DMA_BUF_PLANE0_FD_EXT = 0x3272
EGL_DMA_BUF_PLANE0_OFFSET_EXT = 0x3273
EGL_DMA_BUF_PLANE0_PITCH_EXT = 0x3274
EGL_DMA_BUF_PLANE1_FD_EXT = 0x3275
EGL_DMA_BUF_PLANE1_OFFSET_EXT = 0x3276
EGL_DMA_BUF_PLANE1_PITCH_EXT = 0x3277
EGL_DMA_BUF_PLANE2_FD_EXT = 0x3279
EGL_DMA_BUF_PLANE2_OFFSET_EXT = 0x327A
EGL_DMA_BUF_PLANE2_PITCH_EXT = 0x327B
EGL_WIDTH = 0x3056
EGL_HEIGHT = 0x3050
EGL_RED_SIZE = 0x3024
EGL_GREEN_SIZE = 0x3023
EGL_BLUE_SIZE = 0x3022
EGL_ALPHA_SIZE = 0x3021
EGL_DEPTH_SIZE = 0x3025
EGL_STENCIL_SIZE = 0x3026
EGL_RENDERABLE_TYPE = 0x3040
EGL_SURFACE_TYPE = 0x303D
EGL_PBUFFER_BIT = 0x0001
EGL_OPENGL_ES3_BIT = 0x0040
EGL_OPENGL_ES2_BIT = 0x0004
EGL_VENDOR = 0x3054
EGL_VERSION = 0x3053
EGL_CLIENT_APIS = 0x3052
EGL_EXTENSIONS = 0x3055
EGL_SYNC_FENCE_KHR = 0x3116
EGL_SYNC_NEW_FRAME_NV = 0x313E
EGL_SYNC_NATIVE_FENCE_ANDROID = 0x3144
EGL_NO_SYNC = 0
EGL_SYNC_PRIOR_COMMANDS_COMPLETE_KHR = 0x310E
EGL_SYNC_CONDITION_KHR = 0x310F
EGL_SYNC_STATUS_KHR = 0x3111
EGL_SIGNALED_KHR = 0x3112
EGL_UNSIGNALED_KHR = 0x3113
EGL_FOREVER_KHR = 0xFFFFFFFFFFFFFFFF
EGL_SYNC_TYPE_KHR = 0x310D
EGL_CONDITION_SATISFIED_KHR = 0x3110

GL_DEPTH_BUFFER_BIT = 0x00000100
GL_STENCIL_BUFFER_BIT = 0x00000400
GL_COLOR_BUFFER_BIT = 0x00004000
GL_FALSE = 0
GL_TRUE = 1
GL_ZERO = 0
GL_ONE = 1
GL_SRC_ALPHA = 0x0302
GL_ONE_MINUS_SRC_ALPHA = 0x0303
GL_FRONT = 0x0404
GL_BACK = 0x0405
GL_FRONT_AND_BACK = 0x0408
GL_TEXTURE_2D = 0x0DE1
GL_TEXTURE_EXTERNAL_OES = 0x8D65
GL_TEXTURE0 = 0x84C0
GL_TEXTURE1 = 0x84C1
GL_TEXTURE2 = 0x84C2
GL_TEXTURE3 = 0x84C3
GL_TEXTURE_MIN_FILTER = 0x2801
GL_TEXTURE_MAG_FILTER = 0x2800
GL_TEXTURE_WRAP_S = 0x2802
GL_TEXTURE_WRAP_T = 0x2803
GL_LINEAR = 0x2601
GL_NEAREST = 0x2600
GL_CLAMP_TO_EDGE = 0x812F
GL_REPEAT = 0x2901
GL_RGBA = 0x1908
GL_RGB = 0x1907
GL_RGBA8 = 0x8058
GL_RGB8 = 0x8051
GL_R8 = 0x8229
GL_RG = 0x8227
GL_RG8 = 0x822B
GL_RED = 0x1903
GL_UNSIGNED_BYTE = 0x1401
GL_FLOAT = 0x1406
GL_HALF_FLOAT = 0x140B
GL_ARRAY_BUFFER = 0x8892
GL_ELEMENT_ARRAY_BUFFER = 0x8893
GL_PIXEL_PACK_BUFFER = 0x88EB
GL_PIXEL_UNPACK_BUFFER = 0x88EC
GL_STREAM_DRAW = 0x88E0
GL_STREAM_READ = 0x88E1
GL_STREAM_COPY = 0x88E2
GL_STATIC_DRAW = 0x88E4
GL_STATIC_READ = 0x88E5
GL_STATIC_COPY = 0x88E6
GL_DYNAMIC_DRAW = 0x88E8
GL_DYNAMIC_READ = 0x88E9
GL_DYNAMIC_COPY = 0x88EA
GL_MAP_READ_BIT = 0x0002
GL_MAP_WRITE_BIT = 0x0001
GL_MAP_INVALIDATE_RANGE_BIT = 0x0004
GL_MAP_INVALIDATE_BUFFER_BIT = 0x0008
GL_WRITE_ONLY = 0x88B9
GL_READ_ONLY = 0x88B8
GL_READ_WRITE = 0x88BA
GL_FRAMEBUFFER = 0x8D40
GL_RENDERBUFFER = 0x8D41
GL_COLOR_ATTACHMENT0 = 0x8CE0
GL_DEPTH_ATTACHMENT = 0x8D00
GL_STENCIL_ATTACHMENT = 0x8D01
GL_FRAMEBUFFER_COMPLETE = 0x8CD5
GL_TRIANGLES = 0x0004
GL_TRIANGLE_STRIP = 0x0005
GL_TRIANGLE_FAN = 0x0006
GL_VERTEX_SHADER = 0x8B31
GL_FRAGMENT_SHADER = 0x8B30
GL_COMPILE_STATUS = 0x8B81
GL_LINK_STATUS = 0x8B82
GL_INFO_LOG_LENGTH = 0x8B84
GL_ACTIVE_UNIFORMS = 0x8B86
GL_NO_ERROR = 0
GL_VERSION = 0x1F02
GL_VENDOR = 0x1F01
GL_RENDERER = 0x1F00
GL_SHADING_LANGUAGE_VERSION = 0x8B8C
GL_SYNC_GPU_COMMANDS_COMPLETE = 0x9117
GL_SYNC_FLUSH_COMMANDS_BIT = 0x00000001
GL_UNPACK_ALIGNMENT = 0x0CF5
GL_PACK_ALIGNMENT = 0x0D05
GL_BLEND = 0x0BE2
GL_SCISSOR_TEST = 0x0C11
GL_DEPTH_TEST = 0x0B71
GL_STENCIL_TEST = 0x0B90
GL_CULL_FACE = 0x0B44
GL_DITHER = 0x0BD0
GL_LINE_SMOOTH = 0x0B20

_NV_SHADER_SOURCE = b"""#version 300 es
#extension GL_OES_EGL_image_external_essl3 : require
precision highp float;
in vec2 vUV;
uniform samplerExternalOES texNV;
uniform vec3 yuvOffset;
uniform vec3 yuvScale;
uniform vec3 yuv2r;
uniform vec3 yuv2g;
uniform vec3 yuv2b;
out vec4 fragColor;
void main() {
    vec4 c = texture(texNV, vUV);
    float y = c.r;
    float u = c.g;
    float v = c.b;
    float yv = (y * 255.0 - yuvOffset.x) * yuvScale.x;
    float uv = (u * 255.0 - yuvOffset.y) * yuvScale.y;
    float vv = (v * 255.0 - yuvOffset.z) * yuvScale.z;
    float r = dot(vec3(yv, uv, vv), yuv2r);
    float g = dot(vec3(yv, uv, vv), yuv2g);
    float b = dot(vec3(yv, uv, vv), yuv2b);
    fragColor = vec4(clamp(b, 0.0, 255.0) / 255.0,
                     clamp(g, 0.0, 255.0) / 255.0,
                     clamp(r, 0.0, 255.0) / 255.0,
                     1.0);
}
"""

_RGB_SHADER_SOURCE = b"""#version 300 es
precision highp float;
in vec2 vUV;
uniform sampler2D tex;
out vec4 fragColor;
void main() {
    vec3 c = texture(tex, vUV).rgb;
    fragColor = vec4(c.b, c.g, c.r, 1.0);
}
"""

_VERTEX_SHADER_SOURCE = b"""#version 300 es
in vec2 aPos;
in vec2 aUV;
uniform vec4 uRect;
out vec2 vUV;
void main() {
    vec2 pos = aPos * uRect.zw + uRect.xy;
    vUV = aUV;
    gl_Position = vec4(pos, 0.0, 1.0);
}
"""


def _egl_bind(fn, *args):
    with _mali_lock:
        return fn(*args)


def get_default_display() -> c_void_p:
    return _lib.eglGetDisplay(EGL_DEFAULT_DISPLAY)


def get_display(platform: int, native: c_void_p) -> c_void_p:
    if hasattr(_lib, "eglGetPlatformDisplay"):
        return _lib.eglGetPlatformDisplay(platform, native, None)
    if hasattr(_lib, "eglGetPlatformDisplayEXT"):
        return _lib.eglGetPlatformDisplayEXT(platform, native, None)
    raise AttributeError("eglGetPlatformDisplay is unavailable in this libmali build")


def initialize(dpy: c_void_p) -> Tuple[int, int]:
    major = c_int32()
    minor = c_int32()
    ok = _lib.eglInitialize(dpy, ctypes.byref(major), ctypes.byref(minor))
    if not ok:
        raise RuntimeError("eglInitialize failed")
    return int(major.value), int(minor.value)


def terminate(dpy: c_void_p) -> None:
    _lib.eglTerminate(dpy)


def query_string(dpy: c_void_p, name: int) -> str:
    ptr = _lib.eglQueryString(dpy, name)
    if not ptr:
        return ""
    return ctypes.string_at(ptr).decode("ascii", "replace")


def bind_api(api: int) -> bool:
    """Set the current rendering API (ES or GL).

    Returns True if the binding was accepted, False otherwise. Some libmali
    builds return False from ``eglBindAPI`` but still have ES as the default
    after ``eglInitialize``; callers should fall back to a probe of
    :func:`query_api` rather than treating False as fatal.
    """
    return bool(_lib.eglBindAPI(api))


def query_api() -> int:
    """Return the current rendering API (e.g. :data:`EGL_OPENGL_ES_API`)."""
    return int(_lib.eglQueryAPI())


def choose_config(dpy: c_void_p, attribs: list[tuple[int, int]]) -> list[c_void_p]:
    """Return all EGLConfigs matching ``attribs``.

    ``attribs`` is a list of (EGL_xxx, value) pairs; the function appends
    ``EGL_NONE`` automatically.
    """
    flat = [v for pair in attribs for v in pair]
    flat.append(EGL_NONE)
    arr = (ctypes.c_int * len(flat))(*flat)
    cfg_arr = (ctypes.c_void_p * 32)()
    num = c_int()
    ok = _lib.eglChooseConfig(
        dpy, ctypes.cast(arr, POINTER(c_int)), cfg_arr, 32, ctypes.byref(num)
    )
    if not ok:
        raise RuntimeError("eglChooseConfig failed")
    return [cfg_arr[i] for i in range(num.value)]


def create_context(
    dpy: c_void_p,
    cfg: c_void_p,
    share_ctx: c_void_p = EGL_NO_CONTEXT,
    client_version: int = 3,
) -> c_void_p:
    """Create an ES 3.x context. ``share_ctx`` lets multiple contexts
    share textures/programs (not currently used by rkv)."""
    attribs = (ctypes.c_int * 5)(
        EGL_CONTEXT_CLIENT_VERSION, client_version,
        EGL_NONE, EGL_NONE, 0,
    )
    ctx = _lib.eglCreateContext(dpy, cfg, share_ctx, attribs)
    if not ctx:
        raise RuntimeError("eglCreateContext failed")
    return ctx


def destroy_context(dpy: c_void_p, ctx: c_void_p) -> None:
    if ctx:
        _lib.eglDestroyContext(dpy, ctx)


def make_current_surfaceless(dpy: c_void_p, ctx: c_void_p) -> None:
    ok = _lib.eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx)
    if not ok:
        raise RuntimeError("eglMakeCurrent(surfaceless) failed")


def release_current() -> None:
    _lib.eglMakeCurrent(EGL_NO_DISPLAY, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT)


def create_image_dma_buf(
    dpy: c_void_p,
    fd: int,
    width: int,
    height: int,
    fourcc: int,
    planes: list[tuple[int, int]],
) -> c_void_p:
    """Wrap a DMA-BUF fd as an EGLImage.

    ``planes`` is a list of (offset, pitch) per plane. For NV12 the producer
    hands a single fd covering both planes; we pass two entries with the same
    fd but different offsets so the driver knows about the UV plane.
    """
    attribs: list[int] = [
        EGL_WIDTH, width,
        EGL_HEIGHT, height,
        EGL_LINUX_DRM_FOURCC_EXT, fourcc,
    ]
    for i, (offset, pitch) in enumerate(planes):
        if i == 0:
            attribs += [EGL_DMA_BUF_PLANE0_FD_EXT, fd,
                        EGL_DMA_BUF_PLANE0_OFFSET_EXT, offset,
                        EGL_DMA_BUF_PLANE0_PITCH_EXT, pitch]
        elif i == 1:
            attribs += [EGL_DMA_BUF_PLANE1_FD_EXT, fd,
                        EGL_DMA_BUF_PLANE1_OFFSET_EXT, offset,
                        EGL_DMA_BUF_PLANE1_PITCH_EXT, pitch]
        elif i == 2:
            attribs += [EGL_DMA_BUF_PLANE2_FD_EXT, fd,
                        EGL_DMA_BUF_PLANE2_OFFSET_EXT, offset,
                        EGL_DMA_BUF_PLANE2_PITCH_EXT, pitch]
    attribs.append(EGL_NONE)
    arr = (ctypes.c_int * len(attribs))(*attribs)
    image = _lib.eglCreateImageKHR(
        dpy, EGL_NO_CONTEXT, EGL_LINUX_DMA_BUF_EXT, None,
        ctypes.cast(arr, POINTER(c_int)),
    )
    if not image:
        err = _lib.eglGetError()
        raise RuntimeError(
            f"eglCreateImageKHR(DRM_FOURCC=0x{fourcc:x}, {width}x{height}, "
            f"planes={planes}) failed: EGL error 0x{err:x}"
        )
    return image


def destroy_image(dpy: c_void_p, image: c_void_p) -> None:
    if image:
        _lib.eglDestroyImageKHR(dpy, image)


def get_proc_address(name: str) -> c_void_p:
    return _lib.eglGetProcAddress(name.encode("ascii"))


def _has_proc(name: str) -> bool:
    try:
        addr = get_proc_address(name)
        return bool(addr)
    except AttributeError:
        return False


def create_pbuffer_surface(dpy: c_void_p, cfg: c_void_p, width: int, height: int) -> c_void_p:
    """Create an EGL PBuffer surface (the offscreen-render-target of choice
    when ``EGL_KHR_surfaceless_context`` is unavailable or when the user
    wants a stable context binding).

    Some libmali builds advertise ``EGL_PBUFFER_BIT`` in their configs but
    reject ``eglCreatePbufferSurface`` anyway — in that case the function
    returns ``EGL_NO_SURFACE``. Callers should fall back to surfaceless
    via :func:`make_current_surfaceless`.
    """
    attribs = (ctypes.c_int64 * 5)(EGL_WIDTH, width, EGL_HEIGHT, height, EGL_NONE)
    surface = _lib.eglCreatePbufferSurface(dpy, cfg, attribs)
    if not surface:
        raise RuntimeError(
            f"eglCreatePbufferSurface({width}x{height}) failed: EGL error 0x{_lib.eglGetError():x}"
        )
    return surface


def destroy_surface(dpy: c_void_p, surface: c_void_p) -> None:
    if surface:
        _lib.eglDestroySurface(dpy, surface)


def make_current(dpy: c_void_p, draw: c_void_p, read: c_void_p, ctx: c_void_p) -> None:
    if not _lib.eglMakeCurrent(dpy, draw, read, ctx):
        raise RuntimeError(
            f"eglMakeCurrent failed: EGL error 0x{_lib.eglGetError():x}"
        )


_lib.eglGetDisplay.argtypes = [c_void_p]
_lib.eglGetDisplay.restype = c_void_p
_lib.eglInitialize.argtypes = [c_void_p, POINTER(c_int32), POINTER(c_int32)]
_lib.eglInitialize.restype = c_int
_lib.eglTerminate.argtypes = [c_void_p]
_lib.eglTerminate.restype = c_int
_lib.eglQueryString.argtypes = [c_void_p, c_int]
_lib.eglQueryString.restype = c_char_p
_lib.eglBindAPI.argtypes = [c_int]
_lib.eglBindAPI.restype = c_int
_lib.eglQueryAPI.argtypes = []
_lib.eglQueryAPI.restype = c_int
_lib.eglChooseConfig.argtypes = [c_void_p, POINTER(c_int), POINTER(c_void_p), c_int, POINTER(c_int)]
_lib.eglChooseConfig.restype = c_int
_lib.eglGetConfigAttrib.argtypes = [c_void_p, c_void_p, c_int, POINTER(c_int)]
_lib.eglGetConfigAttrib.restype = c_int
_lib.eglCreatePbufferSurface.argtypes = [c_void_p, c_void_p, POINTER(c_int64)]
_lib.eglCreatePbufferSurface.restype = c_void_p
_lib.eglDestroySurface.argtypes = [c_void_p, c_void_p]
_lib.eglDestroySurface.restype = c_int
_lib.eglCreateContext.argtypes = [c_void_p, c_void_p, c_void_p, POINTER(c_int)]
_lib.eglCreateContext.restype = c_void_p
_lib.eglDestroyContext.argtypes = [c_void_p, c_void_p]
_lib.eglDestroyContext.restype = c_int
_lib.eglMakeCurrent.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p]
_lib.eglMakeCurrent.restype = c_int
_lib.eglCreateImageKHR.argtypes = [c_void_p, c_void_p, c_int, c_void_p, POINTER(c_int)]
_lib.eglCreateImageKHR.restype = c_void_p
_lib.eglDestroyImageKHR.argtypes = [c_void_p, c_void_p]
_lib.eglDestroyImageKHR.restype = c_int
_lib.eglGetError.argtypes = []
_lib.eglGetError.restype = c_int
_lib.eglGetProcAddress.argtypes = [c_char_p]
_lib.eglGetProcAddress.restype = c_void_p


def _bind_gl(name, argtypes, restype):
    fn = getattr(_lib, name, None)
    if fn is None:
        raise AttributeError(f"{name} not exported by libmali")
    fn.argtypes = argtypes
    fn.restype = restype
    return fn


glCreateShader = _bind_gl("glCreateShader", [c_int], c_int)
glShaderSource = _bind_gl("glShaderSource", [c_int, c_int, POINTER(c_char_p), POINTER(c_int)], None)
glCompileShader = _bind_gl("glCompileShader", [c_int], None)
glGetShaderiv = _bind_gl("glGetShaderiv", [c_int, c_int, POINTER(c_int)], None)
glGetShaderInfoLog = _bind_gl("glGetShaderInfoLog", [c_int, c_int, POINTER(c_int), c_char_p], None)
glCreateProgram = _bind_gl("glCreateProgram", [], c_int)
glAttachShader = _bind_gl("glAttachShader", [c_int, c_int], None)
glLinkProgram = _bind_gl("glLinkProgram", [c_int], None)
glGetProgramiv = _bind_gl("glGetProgramiv", [c_int, c_int, POINTER(c_int)], None)
glUseProgram = _bind_gl("glUseProgram", [c_int], None)
glDeleteProgram = _bind_gl("glDeleteProgram", [c_int], None)
glDeleteShader = _bind_gl("glDeleteShader", [c_int], None)
glGetUniformLocation = _bind_gl("glGetUniformLocation", [c_int, c_char_p], c_int)
glGetAttribLocation = _bind_gl("glGetAttribLocation", [c_int, c_char_p], c_int)
glUniform1i = _bind_gl("glUniform1i", [c_int, c_int], None)
glUniform4f = _bind_gl("glUniform4f", [c_int, c_float := ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float], None)
glUniform3f = _bind_gl("glUniform3f", [c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float], None)
glUniform3fv = _bind_gl("glUniform3fv", [c_int, c_int, ctypes.c_void_p], None)
glUniformMatrix3fv = _bind_gl("glUniformMatrix3fv", [c_int, c_int, c_uint8, ctypes.c_void_p], None)
glEnableVertexAttribArray = _bind_gl("glEnableVertexAttribArray", [c_int], None)
glDisableVertexAttribArray = _bind_gl("glDisableVertexAttribArray", [c_int], None)
glVertexAttribPointer = _bind_gl(
    "glVertexAttribPointer",
    [c_int, c_int, c_int, c_uint8, c_int, ctypes.c_void_p],
    None,
)
glGenBuffers = _bind_gl("glGenBuffers", [c_int, POINTER(c_uint32)], None)
glBindBuffer = _bind_gl("glBindBuffer", [c_int, c_uint32], None)
glBufferData = _bind_gl(
    "glBufferData",
    [c_int, c_size_t, ctypes.c_void_p, c_int],
    None,
)
glDeleteBuffers = _bind_gl("glDeleteBuffers", [c_int, POINTER(c_uint32)], None)
glMapBufferRange = _bind_gl(
    "glMapBufferRange",
    [c_int, c_int, c_size_t, c_uint32],
    ctypes.c_void_p,
)
glUnmapBuffer = _bind_gl("glUnmapBuffer", [c_int], c_uint8)
glGenTextures = _bind_gl("glGenTextures", [c_int, POINTER(c_uint32)], None)
glDeleteTextures = _bind_gl("glDeleteTextures", [c_int, POINTER(c_uint32)], None)
glBindTexture = _bind_gl("glBindTexture", [c_int, c_uint32], None)
glActiveTexture = _bind_gl("glActiveTexture", [c_int], None)
glTexImage2D = _bind_gl(
    "glTexImage2D",
    [c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int, ctypes.c_void_p],
    None,
)
glTexSubImage2D = _bind_gl(
    "glTexSubImage2D",
    [c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int, ctypes.c_void_p],
    None,
)
glTexParameteri = _bind_gl("glTexParameteri", [c_int, c_int, c_int], None)
glGenFramebuffers = _bind_gl("glGenFramebuffers", [c_int, POINTER(c_uint32)], None)
glDeleteFramebuffers = _bind_gl("glDeleteFramebuffers", [c_int, POINTER(c_uint32)], None)
glBindFramebuffer = _bind_gl("glBindFramebuffer", [c_int, c_uint32], None)
glGenRenderbuffers = _bind_gl("glGenRenderbuffers", [c_int, POINTER(c_uint32)], None)
glDeleteRenderbuffers = _bind_gl("glDeleteRenderbuffers", [c_int, POINTER(c_uint32)], None)
glBindRenderbuffer = _bind_gl("glBindRenderbuffer", [c_int, c_uint32], None)
glRenderbufferStorage = _bind_gl(
    "glRenderbufferStorage",
    [c_int, c_int, c_int, c_int],
    None,
)
glFramebufferRenderbuffer = _bind_gl(
    "glFramebufferRenderbuffer",
    [c_int, c_int, c_int, c_uint32],
    None,
)
glCheckFramebufferStatus = _bind_gl("glCheckFramebufferStatus", [c_int], c_int)
glViewport = _bind_gl("glViewport", [c_int, c_int, c_int, c_int], None)
glClearColor = _bind_gl("glClearColor", [ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float], None)
glClear = _bind_gl("glClear", [c_uint32], None)
glDrawArrays = _bind_gl("glDrawArrays", [c_int, c_int, c_int], None)
glBlendFunc = _bind_gl("glBlendFunc", [c_int, c_int], None)
glEnable = _bind_gl("glEnable", [c_int], None)
glDisable = _bind_gl("glDisable", [c_int], None)
glReadPixels = _bind_gl(
    "glReadPixels",
    [c_int, c_int, c_int, c_int, c_int, c_int, ctypes.c_void_p],
    None,
)
glPixelStorei = _bind_gl("glPixelStorei", [c_int, c_int], None)
glGetError = _bind_gl("glGetError", [], c_int)
glFinish = _bind_gl("glFinish", [], None)
glFlush = _bind_gl("glFlush", [], None)
glFenceSync = _bind_gl("glFenceSync", [c_int, c_int], ctypes.c_void_p)
glClientWaitSync = _bind_gl("glClientWaitSync", [ctypes.c_void_p, c_uint32, c_uint64], c_int)
glDeleteSync = _bind_gl("glDeleteSync", [ctypes.c_void_p], None)
glGetString = _bind_gl("glGetString", [c_int], c_char_p)
glEGLImageTargetTexture2DOES = _bind_gl(
    "glEGLImageTargetTexture2DOES",
    [c_int, c_void_p],
    None,
)
glEGLImageTargetRenderbufferStorageOES = _bind_gl(
    "glEGLImageTargetRenderbufferStorageOES",
    [c_int, c_void_p],
    None,
)


def compile_shader(source: bytes, kind: int, label: str = "shader") -> int:
    """Compile a single GLSL shader stage. ``source`` is raw UTF-8/ASCII."""
    sh = glCreateShader(kind)
    if not sh:
        raise RuntimeError(f"{label}: glCreateShader returned 0")
    src_arr = (ctypes.c_char_p * 1)(source)
    length = (ctypes.c_int * 1)(len(source))
    glShaderSource(sh, 1, src_arr, length)
    glCompileShader(sh)
    status = c_int()
    glGetShaderiv(sh, GL_COMPILE_STATUS, ctypes.byref(status))
    if status.value != GL_TRUE:
        log_len = c_int()
        glGetShaderiv(sh, GL_INFO_LOG_LENGTH, ctypes.byref(log_len))
        if log_len.value > 1:
            buf = ctypes.create_string_buffer(log_len.value)
            glGetShaderInfoLog(sh, log_len.value, None, buf)
            msg = buf.value.decode(errors="replace")
        else:
            msg = "<no log>"
        glDeleteShader(sh)
        raise RuntimeError(f"{label}: compile failed: {msg}")
    return sh


def link_program(vs: int, fs: int, label: str = "program") -> int:
    """Link a vertex+fragment pair into a usable GL program."""
    prog = glCreateProgram()
    if not prog:
        raise RuntimeError(f"{label}: glCreateProgram returned 0")
    glAttachShader(prog, vs)
    glAttachShader(prog, fs)
    glLinkProgram(prog)
    status = c_int()
    glGetProgramiv(prog, GL_LINK_STATUS, ctypes.byref(status))
    if status.value != GL_TRUE:
        log_len = c_int()
        glGetProgramiv(prog, GL_INFO_LOG_LENGTH, ctypes.byref(log_len))
        if log_len.value > 1:
            buf = ctypes.create_string_buffer(log_len.value)
            glGetShaderInfoLog(prog, log_len.value, None, buf)
            msg = buf.value.decode(errors="replace")
        else:
            msg = "<no log>"
        glDeleteProgram(prog)
        raise RuntimeError(f"{label}: link failed: {msg}")
    return prog


def get_gl_error(where: str) -> None:
    err = glGetError()
    if err != GL_NO_ERROR:
        raise RuntimeError(f"GL error 0x{err:x} at {where}")


class _NV12Program:
    """Compiled NV12 -> BGR quad program with cached uniform locations.

    Uses ``GL_OES_EGL_image_external`` so the Mali driver can sample the
    NV12 dma-buf image directly (handles the chroma upsampling in
    hardware). The fragment shader still has to do the YUV -> RGB
    conversion explicitly.
    """

    def __init__(self) -> None:
        vs = compile_shader(_VERTEX_SHADER_SOURCE, GL_VERTEX_SHADER, "nv12.vs")
        fs = compile_shader(_NV_SHADER_SOURCE, GL_FRAGMENT_SHADER, "nv12.fs")
        self.handle = link_program(vs, fs, "nv12.program")
        glDeleteShader(vs)
        glDeleteShader(fs)
        self.loc_rect = glGetUniformLocation(self.handle, b"uRect")
        self.loc_tex = glGetUniformLocation(self.handle, b"texNV")
        self.loc_yuv_offset = glGetUniformLocation(self.handle, b"yuvOffset")
        self.loc_yuv_scale = glGetUniformLocation(self.handle, b"yuvScale")
        self.loc_yuv2r = glGetUniformLocation(self.handle, b"yuv2r")
        self.loc_yuv2g = glGetUniformLocation(self.handle, b"yuv2g")
        self.loc_yuv2b = glGetUniformLocation(self.handle, b"yuv2b")
        self.a_pos = glGetAttribLocation(self.handle, b"aPos")
        self.a_uv = glGetAttribLocation(self.handle, b"aUV")

    def release(self) -> None:
        glDeleteProgram(self.handle)


class _RGBProgram:
    def __init__(self) -> None:
        vs = compile_shader(_VERTEX_SHADER_SOURCE, GL_VERTEX_SHADER, "rgb.vs")
        fs = compile_shader(_RGB_SHADER_SOURCE, GL_FRAGMENT_SHADER, "rgb.fs")
        self.handle = link_program(vs, fs, "rgb.program")
        glDeleteShader(vs)
        glDeleteShader(fs)
        self.loc_rect = glGetUniformLocation(self.handle, b"uRect")
        self.loc_tex = glGetUniformLocation(self.handle, b"tex")
        self.a_pos = glGetAttribLocation(self.handle, b"aPos")
        self.a_uv = glGetAttribLocation(self.handle, b"aUV")

    def release(self) -> None:
        glDeleteProgram(self.handle)


BT601_LIMITED = (
    (16.0, 128.0, 128.0),
    (1.164, 0.0, 1.596),
    (1.164, -0.392, -0.813),
    (1.164, 2.017, 0.0),
)


def nv12_uniforms(scale: float = 1.0) -> dict[str, tuple[float, float, float]]:
    """Return the standard NV12 -> BGR uniforms for BT.601 limited range.

    ``scale`` is multiplied into the Y/U/V pre-scaling so callers can apply
    brightness gain (e.g., for blending).
    """
    return {
        "yuvOffset": (BT601_LIMITED[0][0], BT601_LIMITED[0][1], BT601_LIMITED[0][2]),
        "yuvScale": (scale * BT601_LIMITED[1][0],
                     scale * BT601_LIMITED[1][1],
                     scale * BT601_LIMITED[1][2]),
        "yuv2r": BT601_LIMITED[1],
        "yuv2g": BT601_LIMITED[2],
        "yuv2b": BT601_LIMITED[3],
    }


def check_ext(dpy: c_void_p, name: str) -> bool:
    return name in query_string(dpy, EGL_EXTENSIONS).split()


__all__ = [
    "DRMFormat",
    "bytes_per_pixel_drm",
    "get_default_display",
    "get_display",
    "initialize",
    "terminate",
    "query_string",
    "bind_api",
    "choose_config",
    "create_context",
    "destroy_context",
    "make_current_surfaceless",
    "make_current",
    "release_current",
    "create_pbuffer_surface",
    "destroy_surface",
    "create_image_dma_buf",
    "destroy_image",
    "get_proc_address",
    "check_ext",
    "compile_shader",
    "link_program",
    "get_gl_error",
    "nv12_uniforms",
    "BT601_LIMITED",
    "_NV12Program",
    "_RGBProgram",
]
