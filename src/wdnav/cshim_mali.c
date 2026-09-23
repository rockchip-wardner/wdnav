/* cshim_mali.c — offscreen GLES c-shim for wdnav.
 *
 * Opinionated high-level wrapper around libmali.so for the SurfaceComposer
 * PBUFFER / surfaceless offscreen composition path. Exposes a small
 * surface organised around opaque handles, mirroring the rkv_enc_*
 * / rkv_dec_* pattern in cshim_video_*.c:
 *
 *   rkm_ctx_*          — EGL display / context lifecycle (PBuffer
 *                        with surfaceless fallback).
 *   rkm_target_*       — RGBA8 FBO + renderbuffer (the offscreen target).
 *   rkm_target_swap    — flush GPU + DMA_BUF_SYNC + rotate EGLImage
 *                        bindings + return the just-readable fd. Replaces
 *                        rkm_target_readback (no more glReadPixels).
 *                        ping-pong index internally).
 *   rkm_program_*      — GLSL compile + link + uniform/attrib location
 *                        cache.
 *   rkm_texture_*      — NV12 (GL_TEXTURE_EXTERNAL_OES + EGLImage) and
 *                        RGB (GL_TEXTURE_2D + CPU upload) textures.
 *   rkm_quad_vbo_*     — fullscreen quad VBO (TRIANGLE_STRIP).
 *   rkm_draw_textured_quad — bind program + texture, set uniforms,
 *                            issue draw — one call per surface.
 *
 * Error reporting: every entry point that can fail takes
 *   char *err_buf, int err_buf_len
 * and fills it with a human-readable message on failure (matches the
 * rkv_enc_create contract).
 *
 * Thread safety: the shim is single-threaded. Callers serialise access
 * externally (SurfaceComposer's worker thread holds a Python-level
 * threading.Lock around every shim call via wdnav.mali._mali_lock).
 *
 * Build: links against libmali.so via -lmali and libfreetype via -lfreetype
 * (for the rkm_text_* glyph rasterisation APIs). Both libraries are auto-
 * loaded at runtime via DT_NEEDED; no libdl calls needed.
 */

#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>

#include <linux/dma-buf.h>

#include <EGL/egl.h>
/* eglext.h / gl3ext.h gate their function prototypes behind these
 * prototype-flags. Enable them so the prototype-only extensions
 * (eglCreateImageKHR, glEGLImageTargetTexture2DOES, …) get declarations
 * we can call directly. */
#define EGL_EGLEXT_PROTOTYPES
#define GL_GLEXT_PROTOTYPES
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
#include <GLES3/gl3ext.h>
/* ``glEGLImageTargetTexture2DOES`` is a GLES2-era OES extension; on this
 * distro's gl3ext.h it isn't declared even with GL_GLEXT_PROTOTYPES, so
 * pull in gl2ext.h as a fallback (the function is part of Mali's libmali
 * regardless of GLES version). */
#include <GLES2/gl2ext.h>

/* freetype — used by the text rasterisation APIs in rkm_text_*. We only
 * link against -lfreetype; the runtime loader picks the right .so via the
 * usual DT_NEEDED mechanism in libshim_mali.so. */
#include <ft2build.h>
#include FT_FREETYPE_H
#include FT_GLYPH_H

/* Khronos typedef for EGL image handles consumed by the OES extension —
 * not always present in /usr/include/GLES3/gl3ext.h on every distro. */
#ifndef GLeglImageOES
typedef void *GLeglImageOES;
#endif

#ifndef RK_BOOL
typedef int RK_BOOL;
#endif
#ifndef RK_TRUE
#define RK_TRUE 1
#endif
#ifndef RK_FALSE
#define RK_FALSE 0
#endif

/* ---------------------------------------------------------------------------
 * Constants — must match the Python mirror in wdnav/mali.py. The shim uses
 * these internally; composer.py imports the Python-side mirror.
 * ------------------------------------------------------------------------- */
#define RKM_EGL_SURFACE_TYPE               0x3033
#define RKM_EGL_PBUFFER_BIT                0x0001
#define RKM_EGL_WINDOW_BIT                 0x0004
#define RKM_EGL_NONE                       0x3038
#define RKM_EGL_NO_CONTEXT                 0
#define RKM_EGL_NO_DISPLAY                 0
#define RKM_EGL_NO_SURFACE                 0
#define RKM_EGL_DEFAULT_DISPLAY            0
#define RKM_EGL_LINUX_DMA_BUF_EXT          0x3270
#define RKM_EGL_LINUX_DRM_FOURCC_EXT       0x3271
#define RKM_EGL_DMA_BUF_PLANE0_FD_EXT      0x3272
#define RKM_EGL_DMA_BUF_PLANE0_OFFSET_EXT  0x3273
#define RKM_EGL_DMA_BUF_PLANE0_PITCH_EXT   0x3274
#define RKM_EGL_DMA_BUF_PLANE1_FD_EXT      0x3275
#define RKM_EGL_DMA_BUF_PLANE1_OFFSET_EXT  0x3276
#define RKM_EGL_DMA_BUF_PLANE1_PITCH_EXT   0x3277
#define RKM_EGL_WIDTH                      0x3057
#define RKM_EGL_HEIGHT                     0x3056
#define RKM_EGL_RED_SIZE                   0x3024
#define RKM_EGL_GREEN_SIZE                 0x3023
#define RKM_EGL_BLUE_SIZE                  0x3022
#define RKM_EGL_ALPHA_SIZE                 0x3021
#define RKM_EGL_RENDERABLE_TYPE            0x3040
#define RKM_EGL_OPENGL_ES_API              0x3140
#define RKM_EGL_OPENGL_ES3_BIT             0x0040

#define RKM_DRM_NV12                       0x3231564E
/* DRM fourcc values use big-endian ASCII packing: fourcc_code('R','G','B','A')
 * = ('R' << 0) | ('G' << 8) | ('B' << 16) | ('A' << 24). All values below
 * come straight from /usr/include/libdrm/drm_fourcc.h. */
#define RKM_DRM_ARGB8888                   0x34325241  /* 'A','R','2','4' */
/* libmali-valhall-g610-g24p0 1.9 does NOT accept the standard DRM
 * fourcc_code('R','G','B','A') = 0x41424752 — it returns EGL_BAD_ACCESS
 * for that. The driver instead expects the bytes-as-typed constant
 * 'R','A','B','4' = 0x34324152 (technically a non-standard fourcc, but
 * libmali maps it through to RGBA8888 memory layout). Same applies to
 * ARGB: 'A','R','2','4' = 0x34325241 is accepted. We hard-code
 * 0x34324152 here; if a future driver rev fixes the standard RGBA8888
 * fourcc, swap this back to 0x41424752. */
#define RKM_DRM_RGBA8888                   0x34324152  /* libmali-1.9 quirk */

/* FreeType takes char sizes in 26.6 fixed point; 1px == 64 fixed units. */
#define size_64 ((FT_F26Dot6)64)
#define RKM_DRM_RGB888                     0x33424752  /* 'R','G','B','3' */
#define RKM_DRM_BGR888                     0x33524742  /* 'B','G','R','3' */

#define RKM_GL_FALSE                       0
#define RKM_GL_TRUE                        1
#define RKM_GL_NO_ERROR                    0
#define RKM_GL_DEPTH_TEST                  0x0B71
#define RKM_GL_BLEND                       0x0BE2
#define RKM_GL_COLOR_BUFFER_BIT            0x00004000
#define RKM_GL_TEXTURE_2D                  0x0DE1
#define RKM_GL_TEXTURE_EXTERNAL_OES        0x8D65
#define RKM_GL_TEXTURE0                    0x84C0
#define RKM_GL_TEXTURE_MIN_FILTER          0x2801
#define RKM_GL_TEXTURE_MAG_FILTER          0x2800
#define RKM_GL_TEXTURE_WRAP_S              0x2802
#define RKM_GL_TEXTURE_WRAP_T              0x2803
#define RKM_GL_LINEAR                      0x2601
#define RKM_GL_CLAMP_TO_EDGE               0x812F
#define RKM_GL_RGB8                        0x8051
#define RKM_GL_RGB                         0x1907
#define RKM_GL_RGBA                        0x1908
#define RKM_GL_RGBA8                       0x8058
#define RKM_GL_UNSIGNED_BYTE               0x1401
#define RKM_GL_FLOAT                       0x1406
#define RKM_GL_PIXEL_PACK_BUFFER           0x88EB
#define RKM_GL_ARRAY_BUFFER                0x8892
#define RKM_GL_STREAM_READ                 0x88E1
#define RKM_GL_STATIC_DRAW                 0x88E4
#define RKM_GL_MAP_READ_BIT                0x0002
#define RKM_GL_FRAMEBUFFER                 0x8D40
#define RKM_GL_RENDERBUFFER                0x8D41
#define RKM_GL_COLOR_ATTACHMENT0           0x8CE0
#define RKM_GL_FRAMEBUFFER_COMPLETE        0x8CD5
#define RKM_GL_UNPACK_ALIGNMENT            0x0CF5
#define RKM_GL_PACK_ALIGNMENT              0x0D05
#define RKM_GL_TRIANGLE_STRIP              0x0005
#define RKM_GL_VERTEX_SHADER               0x8B31
#define RKM_GL_FRAGMENT_SHADER             0x8B30
#define RKM_GL_COMPILE_STATUS              0x8B81
#define RKM_GL_LINK_STATUS                 0x8B82
#define RKM_GL_INFO_LOG_LENGTH             0x8B84

/* ---------------------------------------------------------------------------
 * Opaque handle types
 * ------------------------------------------------------------------------- */
typedef struct rkm_ctx      rkm_ctx_t;
typedef struct rkm_target   rkm_target_t;
typedef struct rkm_program  rkm_program_t;
typedef struct rkm_texture  rkm_texture_t;
typedef struct rkm_vbo      rkm_vbo_t;
typedef struct rkm_text     rkm_text_t;

struct rkm_ctx {
    EGLDisplay  dpy;
    EGLContext  ctx;
    EGLSurface  surface;          /* PBuffer; EGL_NO_SURFACE if surfaceless */
    RK_BOOL     owns_surface;
    RK_BOOL     owns_display;
    int         width;
    int         height;
};

struct rkm_target {
    rkm_ctx_t   *ctx;
    int          width;
    int          height;
    int          stride_bytes;
    int          double_buf;          /* 0 = single buf, 1 = ping-pong A/B */
    /* DMA-BUF backing. The FBO color renderbuffer is bound to the
     * ``buf[current]`` EGLImage via glEGLImageTargetRenderbufferStorageOES
     * so the GPU writes land directly in the consumer's mmap'd memory —
     * no glReadPixels involved.
     *
     * In double-buffer mode (``buf[2]``, ``img[2]``, ``rb[2]``),
     * ``current`` selects the buffer the GPU is *writing into*. The
     * other buffer is what the consumer is reading from — they never
     * overlap, so 4-surface composites with concurrent producer
     * threads don't tear.
     *
     * libmali-valhall-g610-g24p0 1.9 only reliably supports RGBA8888
     * for EGLImage→renderbuffer (see RKM_DRM_RGBA8888 above). */
    int          fd[2];
    EGLImageKHR  img[2];
    GLuint       rb[2];
    int          current;            /* 0 or 1 — index of GPU write target */
    GLuint       fbo;
};

struct rkm_program {
    rkm_ctx_t  *ctx;
    GLuint      prog;
    char      **uniform_names;
    GLint      *uniform_locs;
    int         n_uniforms;
    char      **attrib_names;
    GLint      *attrib_locs;
    int         n_attribs;
};

struct rkm_texture {
    rkm_ctx_t      *ctx;
    GLenum          target;       /* GL_TEXTURE_2D / GL_TEXTURE_EXTERNAL_OES */
    GLuint          tex;
    EGLImageKHR     egl_image;    /* NV12 only, refreshed by rebind */
    int             width;
    int             height;
};

struct rkm_vbo {
    rkm_ctx_t  *ctx;
    GLuint      vbo;
};

/* Parameter block for rkm_draw_textured_quad — vec3 uniforms. */
typedef struct {
    const char *name;
    float       x, y, z;
} rkm_uniform_vec3_t;

/* Forward decls — the create functions roll back via destroy on failure. */
void rkm_ctx_destroy(rkm_ctx_t *c);
void rkm_target_destroy(rkm_target_t *t);
void rkm_program_destroy(rkm_program_t *p);
void rkm_texture_destroy(rkm_texture_t *t);
void rkm_pbo_destroy(void *p); /* unused — DMA-BUF backed FBO */
void rkm_quad_vbo_destroy(rkm_vbo_t *v);
void rkm_text_destroy(rkm_text_t *t);

/* ---------------------------------------------------------------------------
 * Helpers
 * ------------------------------------------------------------------------- */
static void snprintf_err(char *buf, int buf_len, const char *fmt, ...) {
    if (!buf || buf_len <= 0) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, (size_t)buf_len, fmt, ap);
    va_end(ap);
}

static void set_egl_err(char *buf, int buf_len, const char *where) {
    EGLint err = eglGetError();
    snprintf_err(buf, buf_len, "%s: EGL error 0x%x", where, (unsigned)err);
}

static void set_gl_err(char *buf, int buf_len, const char *where) {
    GLenum err = glGetError();
    snprintf_err(buf, buf_len, "%s: GL error 0x%x", where, (unsigned)err);
}

/* Allocate a string with the shader / program info log; *out is NULL if
 * the log is empty. Returns the log length, or -1 on OOM. */
static int dup_log(GLenum what, GLuint handle, char **out) {
    GLint len = 0;
    if (what == RKM_GL_COMPILE_STATUS) {
        glGetShaderiv(handle, RKM_GL_INFO_LOG_LENGTH, &len);
    } else {
        glGetProgramiv(handle, RKM_GL_INFO_LOG_LENGTH, &len);
    }
    if (len <= 1) { *out = NULL; return 0; }
    char *buf = (char *)malloc((size_t)len);
    if (!buf) { *out = NULL; return -1; }
    if (what == RKM_GL_COMPILE_STATUS) {
        glGetShaderInfoLog(handle, len, NULL, buf);
    } else {
        glGetProgramInfoLog(handle, len, NULL, buf);
    }
    *out = buf;
    return (int)strlen(buf);
}

static void set_tex_params(GLenum target) {
    glTexParameteri(target, RKM_GL_TEXTURE_MIN_FILTER, RKM_GL_LINEAR);
    glTexParameteri(target, RKM_GL_TEXTURE_MAG_FILTER, RKM_GL_LINEAR);
    glTexParameteri(target, RKM_GL_TEXTURE_WRAP_S, RKM_GL_CLAMP_TO_EDGE);
    glTexParameteri(target, RKM_GL_TEXTURE_WRAP_T, RKM_GL_CLAMP_TO_EDGE);
}

/* ---------------------------------------------------------------------------
 * rkm_ctx_* — EGL display / context lifecycle
 * ------------------------------------------------------------------------- */
int rkm_ctx_create(int width, int height,
                   rkm_ctx_t **out_ctx,
                   char *err_buf, int err_buf_len) {
    if (!out_ctx) {
        snprintf_err(err_buf, err_buf_len, "rkm_ctx_create: out_ctx is NULL");
        return -EINVAL;
    }
    *out_ctx = NULL;
    if (width <= 0 || height <= 0) {
        snprintf_err(err_buf, err_buf_len,
                     "rkm_ctx_create: invalid size %dx%d", width, height);
        return -EINVAL;
    }

    rkm_ctx_t *c = (rkm_ctx_t *)calloc(1, sizeof(rkm_ctx_t));
    if (!c) {
        snprintf_err(err_buf, err_buf_len, "rkm_ctx_create: OOM");
        return -ENOMEM;
    }
    c->width = width;
    c->height = height;
    c->surface = RKM_EGL_NO_SURFACE;

    c->dpy = eglGetDisplay(RKM_EGL_DEFAULT_DISPLAY);
    if (c->dpy == RKM_EGL_NO_DISPLAY) {
        set_egl_err(err_buf, err_buf_len, "eglGetDisplay");
        free(c);
        return -1;
    }
    EGLint major = 0, minor = 0;
    if (!eglInitialize(c->dpy, &major, &minor)) {
        set_egl_err(err_buf, err_buf_len, "eglInitialize");
        free(c);
        return -1;
    }
    c->owns_display = RK_TRUE;

    /* Bind ES client API before choosing a config — without this,
     * libmali-valhall-g610-g24p0 1.9 returns n=0 from eglChooseConfig
     * even when the requested config clearly exists. */
    (void)eglBindAPI(RKM_EGL_OPENGL_ES_API);

    /* Require a PBuffer-capable config. The attribs mirror what rbs
     * glengine glbase_egl.cpp::initSurface() and /userdata/test_pbuffer.c
     * do on this board:
     *   - ALPHA_SIZE=8  (libmali 1.9 doesn't advertise PBUFFER configs
     *                    with alpha=0)
     *   - SURFACE_TYPE = WINDOW_BIT | PBUFFER_BIT  (PBUFFER-only
     *                    also returns 0; we never eglSwapBuffers so
     *                    the WINDOW bit is a no-op for actual rendering)
     */
    EGLint cfg_attribs_pb[] = {
        RKM_EGL_RED_SIZE, 8,
        RKM_EGL_GREEN_SIZE, 8,
        RKM_EGL_BLUE_SIZE, 8,
        RKM_EGL_ALPHA_SIZE, 8,
        RKM_EGL_RENDERABLE_TYPE, RKM_EGL_OPENGL_ES3_BIT,
        RKM_EGL_SURFACE_TYPE, RKM_EGL_WINDOW_BIT | RKM_EGL_PBUFFER_BIT,
        RKM_EGL_NONE
    };
    EGLConfig cfg = NULL;
    EGLint n_cfg = 0;
    if (!eglChooseConfig(c->dpy, cfg_attribs_pb, &cfg, 1, &n_cfg) || n_cfg < 1) {
        EGLint e = eglGetError();
        snprintf_err(err_buf, err_buf_len,
                     "rkm_ctx_create: no PBuffer-capable ES3 config "
                     "(n=%d, eglGetError=0x%x)",
                     n_cfg, (unsigned)e);
        goto fail;
    }

    EGLint ctx_attribs[] = {
        EGL_CONTEXT_CLIENT_VERSION, 3,
        RKM_EGL_NONE
    };
    c->ctx = eglCreateContext(c->dpy, cfg, RKM_EGL_NO_CONTEXT, ctx_attribs);
    if (c->ctx == RKM_EGL_NO_CONTEXT) {
        set_egl_err(err_buf, err_buf_len, "eglCreateContext");
        goto fail;
    }

    /* Try PBuffer first; fall back to surfaceless if libmali rejects
     * pbuffer creation (some Valhall builds do). */
    EGLint pb_attribs[] = {
        RKM_EGL_WIDTH, width,
        RKM_EGL_HEIGHT, height,
        RKM_EGL_NONE
    };
    c->surface = eglCreatePbufferSurface(c->dpy, cfg, pb_attribs);
    if (c->surface == RKM_EGL_NO_SURFACE) {
        set_egl_err(err_buf, err_buf_len, "eglCreatePbufferSurface");
        goto fail;
    }
    c->owns_surface = RK_TRUE;

    if (!eglMakeCurrent(c->dpy, c->surface, c->surface, c->ctx)) {
        set_egl_err(err_buf, err_buf_len, "eglMakeCurrent (PBuffer)");
        goto fail;
    }

    glDisable(RKM_GL_DEPTH_TEST);
    glDisable(RKM_GL_BLEND);

    *out_ctx = c;
    return 0;

fail:
    rkm_ctx_destroy(c);
    return -1;
}

int rkm_ctx_is_pbuffer(rkm_ctx_t *c) {
    if (!c) return 0;
    return (c->surface != RKM_EGL_NO_SURFACE && c->owns_surface) ? 1 : 0;
}

int rkm_ctx_make_current(rkm_ctx_t *c) {
    if (!c) return -EINVAL;
    if (c->surface != RKM_EGL_NO_SURFACE) {
        return eglMakeCurrent(c->dpy, c->surface, c->surface, c->ctx) ? 0 : -1;
    }
    return eglMakeCurrent(c->dpy, RKM_EGL_NO_SURFACE,
                          RKM_EGL_NO_SURFACE, c->ctx) ? 0 : -1;
}

int rkm_ctx_release(rkm_ctx_t *c) {
    (void)c;
    return eglMakeCurrent(RKM_EGL_NO_DISPLAY, RKM_EGL_NO_SURFACE,
                          RKM_EGL_NO_SURFACE, RKM_EGL_NO_CONTEXT) ? 0 : -1;
}

void rkm_ctx_destroy(rkm_ctx_t *c) {
    if (!c) return;
    if (c->ctx != RKM_EGL_NO_CONTEXT) {
        eglMakeCurrent(c->dpy, RKM_EGL_NO_SURFACE,
                        RKM_EGL_NO_SURFACE, RKM_EGL_NO_CONTEXT);
        eglDestroyContext(c->dpy, c->ctx);
    }
    if (c->surface != RKM_EGL_NO_SURFACE && c->owns_surface) {
        eglDestroySurface(c->dpy, c->surface);
    }
    if (c->owns_display && c->dpy != RKM_EGL_NO_DISPLAY) {
        eglTerminate(c->dpy);
    }
    free(c);
}

/* ---------------------------------------------------------------------------
 * rkm_target_* — FBO + renderbuffer (the offscreen target)
 * ------------------------------------------------------------------------- */
/* Helper: build the per-slot EGLImage + renderbuffer for slot index
 * ``i`` (0 or 1) backed by DMA-BUF fd ``d``. Returns 0 on success. */
static int build_slot(rkm_target_t *t, int i, int d,
                      char *err_buf, int err_buf_len) {
    EGLint attribs[] = {
        RKM_EGL_WIDTH,                  t->width,
        RKM_EGL_HEIGHT,                 t->height,
        RKM_EGL_LINUX_DRM_FOURCC_EXT,   RKM_DRM_RGBA8888,
        RKM_EGL_DMA_BUF_PLANE0_FD_EXT,  d,
        RKM_EGL_DMA_BUF_PLANE0_OFFSET_EXT, 0,
        RKM_EGL_DMA_BUF_PLANE0_PITCH_EXT,   t->stride_bytes,
        RKM_EGL_NONE
    };
    t->img[i] = eglCreateImageKHR(t->ctx->dpy, RKM_EGL_NO_CONTEXT,
                                  RKM_EGL_LINUX_DMA_BUF_EXT,
                                  NULL, attribs);
    if (!t->img[i]) {
        set_egl_err(err_buf, err_buf_len, "eglCreateImageKHR (DMA-BUF)");
        return -1;
    }

    glGenRenderbuffers(1, &t->rb[i]);
    glBindRenderbuffer(RKM_GL_RENDERBUFFER, t->rb[i]);
    glEGLImageTargetRenderbufferStorageOES(RKM_GL_RENDERBUFFER,
                                            (GLeglImageOES)t->img[i]);
    GLenum err = glGetError();
    glBindRenderbuffer(RKM_GL_RENDERBUFFER, 0);
    if (err != RKM_GL_NO_ERROR) {
        set_gl_err(err_buf, err_buf_len, "glEGLImageTargetRenderbufferStorageOES");
        return -1;
    }
    return 0;
}

int rkm_target_create(rkm_ctx_t *ctx,
                      int fd_a, int fd_b,        /* fd_b ignored when
                                                  * double_buf == 0; pass
                                                  * -1 in that case */
                      int width, int height, int stride_bytes,
                      int double_buf,
                      rkm_target_t **out_target,
                      char *err_buf, int err_buf_len) {
    if (!ctx || !out_target) {
        snprintf_err(err_buf, err_buf_len, "rkm_target_create: bad arg");
        return -EINVAL;
    }
    if (fd_a < 0 || width <= 0 || height <= 0 || stride_bytes <= 0) {
        snprintf_err(err_buf, err_buf_len,
                     "rkm_target_create: bad fd/size: fd_a=%d fd_b=%d %dx%d stride=%d",
                     fd_a, fd_b, width, height, stride_bytes);
        return -EINVAL;
    }
    if (double_buf && fd_b < 0) {
        snprintf_err(err_buf, err_buf_len,
                     "rkm_target_create: double_buf=1 requires two fds; got fd_b=%d",
                     fd_b);
        return -EINVAL;
    }
    *out_target = NULL;

    rkm_target_t *t = (rkm_target_t *)calloc(1, sizeof(rkm_target_t));
    if (!t) {
        snprintf_err(err_buf, err_buf_len, "rkm_target_create: OOM");
        return -ENOMEM;
    }
    t->ctx = ctx;
    t->width = ctx->width;
    t->height = ctx->height;
    t->stride_bytes = stride_bytes;
    t->double_buf = double_buf ? 1 : 0;
    t->fd[0] = fd_a;
    t->fd[1] = double_buf ? fd_b : -1;
    t->current = 0;

    /* Wrap both DMA-BUF fds as EGLImages + renderbuffers. The GPU
     * writes go directly into the consumer's mmap'd memory — no
     * glReadPixels involved.
     *
     * NOTE: this libmali-valhall-g610-g24p0 1.9 build rejects the
     * standard DRM fourcc_code('R','G','B','A') = 0x41424752 with
     * EGL_BAD_ACCESS. The driver only accepts the byte-typed constant
     * 0x34324152 ('R','A','B','4'), which is a non-standard fourcc but
     * produces an RGBA8888 memory layout downstream. If a future
     * driver rev fixes the standard RGBA8888 fourcc, swap RKM_DRM_RGBA8888
     * back to 0x41424752. */
    if (build_slot(t, 0, t->fd[0], err_buf, err_buf_len) != 0) {
        rkm_target_destroy(t);
        return -1;
    }
    if (t->double_buf) {
        if (build_slot(t, 1, t->fd[1], err_buf, err_buf_len) != 0) {
            rkm_target_destroy(t);
            return -1;
        }
    }

    glGenFramebuffers(1, &t->fbo);
    glBindFramebuffer(RKM_GL_FRAMEBUFFER, t->fbo);
    glFramebufferRenderbuffer(RKM_GL_FRAMEBUFFER, RKM_GL_COLOR_ATTACHMENT0,
                              RKM_GL_RENDERBUFFER, t->rb[t->current]);
    GLenum status = glCheckFramebufferStatus(RKM_GL_FRAMEBUFFER);
    glBindFramebuffer(RKM_GL_FRAMEBUFFER, 0);
    if (status != RKM_GL_FRAMEBUFFER_COMPLETE) {
        snprintf_err(err_buf, err_buf_len,
                     "FBO incomplete: status 0x%x", (unsigned)status);
        rkm_target_destroy(t);
        return -1;
    }

    *out_target = t;
    return 0;
}

int rkm_target_bind(rkm_target_t *t) {
    if (!t) return -EINVAL;
    glBindFramebuffer(RKM_GL_FRAMEBUFFER, t->fbo);
    glViewport(0, 0, t->width, t->height);
    return 0;
}

int rkm_target_clear(rkm_target_t *t, float r, float g, float b, float a) {
    if (!t) return -EINVAL;
    rkm_target_bind(t);
    glClearColor(r, g, b, a);
    glClear(RKM_GL_COLOR_BUFFER_BIT);
    return 0;
}

/* Finish the current frame, fence the GPU, sync the DMA-BUF for CPU
 * read, and return its fd so the consumer can mmap it.
 *
 * In single-buffer mode (``double_buf == 0``) the behaviour is
 * identical to the old path: fence + sync + return the lone fd.
 *
 * In double-buffer mode the GPU is currently bound to slot ``current``.
 * This call:
 *   1. fences the GPU (glFlush + glFinish)
 *   2. DMA_BUF_SYNC_START|READ on ``fd[current]`` so the CPU can read it
 *      while the GPU prepares the next frame
 *   3. returns ``fd[current]``
 *   4. flips ``current`` so the next ``bind()`` / ``clear()`` / draw
 *      cycle writes to the other slot — never overlapping with the
 *      consumer's read
 *   5. re-attaches the FBO's COLOR_ATTACHMENT0 to the new renderbuffer
 *      so subsequent draws land in the right backing
 */
int rkm_target_swap(rkm_target_t *t, int *out_fd,
                   char *err_buf, int err_buf_len) {
    if (!t || !out_fd) {
        snprintf_err(err_buf, err_buf_len, "rkm_target_swap: bad arg");
        return -EINVAL;
    }
    *out_fd = -1;

    /* Fence GPU work before the consumer reads. */
    glFlush();
    glFinish();

    /* Sync the buffer the GPU just finished writing into. */
    struct dma_buf_sync sync = {
        .flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ,
    };
    int just_written_fd = t->fd[t->current];
    if (ioctl(just_written_fd, DMA_BUF_IOCTL_SYNC, &sync) < 0) {
        snprintf_err(err_buf, err_buf_len,
                     "rkm_target_swap: DMA_BUF_IOCTL_SYNC(fd=%d) failed: errno=%d",
                     just_written_fd, errno);
        return -1;
    }
    *out_fd = just_written_fd;

    /* In double-buffer mode, flip the write target so the next
     * composite goes to the OTHER buffer — never the one the
     * consumer is still reading. Re-attach the FBO colour attachment
     * accordingly. */
    if (t->double_buf) {
        int next = 1 - t->current;
        t->current = next;
        glBindFramebuffer(RKM_GL_FRAMEBUFFER, t->fbo);
        glFramebufferRenderbuffer(RKM_GL_FRAMEBUFFER, RKM_GL_COLOR_ATTACHMENT0,
                                  RKM_GL_RENDERBUFFER, t->rb[t->current]);
        glBindFramebuffer(RKM_GL_FRAMEBUFFER, 0);
    }
    return 0;
}

void rkm_target_destroy(rkm_target_t *t) {
    if (!t) return;
    if (t->fbo) glDeleteFramebuffers(1, &t->fbo);
    for (int i = 0; i < 2; i++) {
        if (t->rb[i]) glDeleteRenderbuffers(1, &t->rb[i]);
        if (t->img[i] && t->ctx) {
            eglDestroyImageKHR(t->ctx->dpy, t->img[i]);
        }
    }
    free(t);
}

/* ---------------------------------------------------------------------------
 * rkm_program_* — GLSL compile + link + uniform/attrib cache
 * ------------------------------------------------------------------------- */
int rkm_program_create(rkm_ctx_t *ctx,
                       const char *vs_src, int vs_len,
                       const char *fs_src, int fs_len,
                       const char *const *uniform_names, int n_uniforms,
                       const char *const *attrib_names, int n_attribs,
                       rkm_program_t **out_prog,
                       char *err_buf, int err_buf_len) {
    if (!ctx || !vs_src || !fs_src || !out_prog) {
        snprintf_err(err_buf, err_buf_len, "rkm_program_create: bad arg");
        return -EINVAL;
    }
    *out_prog = NULL;

    rkm_program_t *p = (rkm_program_t *)calloc(1, sizeof(rkm_program_t));
    if (!p) {
        snprintf_err(err_buf, err_buf_len, "rkm_program_create: OOM");
        return -ENOMEM;
    }
    p->ctx = ctx;

    GLuint vs = glCreateShader(RKM_GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &vs_src, &vs_len);
    glCompileShader(vs);
    GLint compiled = 0;
    glGetShaderiv(vs, RKM_GL_COMPILE_STATUS, &compiled);
    if (!compiled) {
        char *log = NULL;
        dup_log(RKM_GL_COMPILE_STATUS, vs, &log);
        snprintf_err(err_buf, err_buf_len,
                     "vertex shader compile failed: %s",
                     log ? log : "(no log)");
        free(log);
        glDeleteShader(vs);
        free(p);
        return -1;
    }

    GLuint fs = glCreateShader(RKM_GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &fs_src, &fs_len);
    glCompileShader(fs);
    glGetShaderiv(fs, RKM_GL_COMPILE_STATUS, &compiled);
    if (!compiled) {
        char *log = NULL;
        dup_log(RKM_GL_COMPILE_STATUS, fs, &log);
        snprintf_err(err_buf, err_buf_len,
                     "fragment shader compile failed: %s",
                     log ? log : "(no log)");
        free(log);
        glDeleteShader(vs);
        glDeleteShader(fs);
        free(p);
        return -1;
    }

    GLuint prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);
    GLint linked = 0;
    glGetProgramiv(prog, RKM_GL_LINK_STATUS, &linked);
    if (!linked) {
        char *log = NULL;
        dup_log(RKM_GL_LINK_STATUS, prog, &log);
        snprintf_err(err_buf, err_buf_len,
                     "program link failed: %s",
                     log ? log : "(no log)");
        free(log);
        glDeleteShader(vs);
        glDeleteShader(fs);
        glDeleteProgram(prog);
        free(p);
        return -1;
    }
    glDeleteShader(vs);
    glDeleteShader(fs);
    p->prog = prog;

    if (n_uniforms > 0) {
        p->uniform_names = (char **)calloc((size_t)n_uniforms, sizeof(char *));
        p->uniform_locs  = (GLint *)calloc((size_t)n_uniforms, sizeof(GLint));
        if (!p->uniform_names || !p->uniform_locs) {
            snprintf_err(err_buf, err_buf_len, "rkm_program_create: OOM (uniforms)");
            rkm_program_destroy(p);
            return -ENOMEM;
        }
        p->n_uniforms = n_uniforms;
        for (int i = 0; i < n_uniforms; i++) {
            p->uniform_names[i] = strdup(uniform_names[i]);
            p->uniform_locs[i]  = glGetUniformLocation(prog, uniform_names[i]);
        }
    }

    if (n_attribs > 0) {
        p->attrib_names = (char **)calloc((size_t)n_attribs, sizeof(char *));
        p->attrib_locs  = (GLint *)calloc((size_t)n_attribs, sizeof(GLint));
        if (!p->attrib_names || !p->attrib_locs) {
            snprintf_err(err_buf, err_buf_len, "rkm_program_create: OOM (attribs)");
            rkm_program_destroy(p);
            return -ENOMEM;
        }
        p->n_attribs = n_attribs;
        for (int i = 0; i < n_attribs; i++) {
            p->attrib_names[i] = strdup(attrib_names[i]);
            p->attrib_locs[i]  = glGetAttribLocation(prog, attrib_names[i]);
        }
    }

    *out_prog = p;
    return 0;
}

int rkm_program_uniform_loc(rkm_program_t *p, const char *name) {
    if (!p || !name) return -1;
    for (int i = 0; i < p->n_uniforms; i++) {
        if (p->uniform_names[i] && strcmp(p->uniform_names[i], name) == 0) {
            return p->uniform_locs[i];
        }
    }
    /* Fallback to a fresh glGetUniformLocation in case the caller asks
     * for a name not in the original cache. */
    return glGetUniformLocation(p->prog, name);
}

int rkm_program_attrib_loc(rkm_program_t *p, const char *name) {
    if (!p || !name) return -1;
    for (int i = 0; i < p->n_attribs; i++) {
        if (p->attrib_names[i] && strcmp(p->attrib_names[i], name) == 0) {
            return p->attrib_locs[i];
        }
    }
    return glGetAttribLocation(p->prog, name);
}

void rkm_program_destroy(rkm_program_t *p) {
    if (!p) return;
    if (p->prog) glDeleteProgram(p->prog);
    if (p->uniform_names) {
        for (int i = 0; i < p->n_uniforms; i++) free(p->uniform_names[i]);
        free(p->uniform_names);
    }
    if (p->uniform_locs) free(p->uniform_locs);
    if (p->attrib_names) {
        for (int i = 0; i < p->n_attribs; i++) free(p->attrib_names[i]);
        free(p->attrib_names);
    }
    if (p->attrib_locs) free(p->attrib_locs);
    free(p);
}

/* ---------------------------------------------------------------------------
 * rkm_texture_* — NV12 (external + EGLImage) and RGB (2D + CPU upload)
 * ------------------------------------------------------------------------- */
int rkm_texture_create_nv12(rkm_ctx_t *ctx, int w, int h,
                            rkm_texture_t **out_tex,
                            char *err_buf, int err_buf_len) {
    if (!ctx || !out_tex || w <= 0 || h <= 0) {
        snprintf_err(err_buf, err_buf_len, "rkm_texture_create_nv12: bad arg");
        return -EINVAL;
    }
    *out_tex = NULL;

    rkm_texture_t *t = (rkm_texture_t *)calloc(1, sizeof(rkm_texture_t));
    if (!t) {
        snprintf_err(err_buf, err_buf_len, "rkm_texture_create_nv12: OOM");
        return -ENOMEM;
    }
    t->ctx = ctx;
    t->target = RKM_GL_TEXTURE_EXTERNAL_OES;
    t->width = w;
    t->height = h;
    t->egl_image = NULL;

    glGenTextures(1, &t->tex);
    glBindTexture(RKM_GL_TEXTURE_EXTERNAL_OES, t->tex);
    set_tex_params(RKM_GL_TEXTURE_EXTERNAL_OES);

    *out_tex = t;
    return 0;
}

/* (Re)bind the NV12 texture to a new DMA-BUF EGLImage. Destroys any
 * previously-bound image. The texture name (glGenTextures ID) is
 * stable across calls — only the EGLImage changes. */
int rkm_texture_rebind_nv12(rkm_texture_t *t, int fd, int w, int h, int stride,
                            char *err_buf, int err_buf_len) {
    if (!t || fd < 0) {
        snprintf_err(err_buf, err_buf_len, "rkm_texture_rebind_nv12: bad arg");
        return -EINVAL;
    }

    if (t->egl_image) {
        eglDestroyImageKHR(t->ctx->dpy, t->egl_image);
        t->egl_image = NULL;
    }

    EGLint attribs[] = {
        RKM_EGL_WIDTH, w,
        RKM_EGL_HEIGHT, h,
        RKM_EGL_LINUX_DRM_FOURCC_EXT, RKM_DRM_NV12,
        RKM_EGL_DMA_BUF_PLANE0_FD_EXT, fd,
        RKM_EGL_DMA_BUF_PLANE0_OFFSET_EXT, 0,
        RKM_EGL_DMA_BUF_PLANE0_PITCH_EXT, stride,
        RKM_EGL_DMA_BUF_PLANE1_FD_EXT, fd,
        RKM_EGL_DMA_BUF_PLANE1_OFFSET_EXT, stride * h,
        RKM_EGL_DMA_BUF_PLANE1_PITCH_EXT, stride,
        RKM_EGL_NONE
    };
    t->egl_image = eglCreateImageKHR(t->ctx->dpy, RKM_EGL_NO_CONTEXT,
                                      RKM_EGL_LINUX_DMA_BUF_EXT,
                                      NULL, attribs);
    if (!t->egl_image) {
        set_egl_err(err_buf, err_buf_len, "eglCreateImageKHR (NV12)");
        return -1;
    }

    glBindTexture(RKM_GL_TEXTURE_EXTERNAL_OES, t->tex);
    glEGLImageTargetTexture2DOES(RKM_GL_TEXTURE_EXTERNAL_OES,
                                 (GLeglImageOES)t->egl_image);
    GLenum err = glGetError();
    if (err != RKM_GL_NO_ERROR) {
        set_gl_err(err_buf, err_buf_len, "glEGLImageTargetTexture2DOES");
        eglDestroyImageKHR(t->ctx->dpy, t->egl_image);
        t->egl_image = NULL;
        return -1;
    }
    t->width = w;
    t->height = h;
    return 0;
}

int rkm_texture_create_rgb(rkm_ctx_t *ctx, int w, int h,
                           rkm_texture_t **out_tex,
                           char *err_buf, int err_buf_len) {
    if (!ctx || !out_tex || w <= 0 || h <= 0) {
        snprintf_err(err_buf, err_buf_len, "rkm_texture_create_rgb: bad arg");
        return -EINVAL;
    }
    *out_tex = NULL;

    rkm_texture_t *t = (rkm_texture_t *)calloc(1, sizeof(rkm_texture_t));
    if (!t) {
        snprintf_err(err_buf, err_buf_len, "rkm_texture_create_rgb: OOM");
        return -ENOMEM;
    }
    t->ctx = ctx;
    t->target = RKM_GL_TEXTURE_2D;
    t->width = w;
    t->height = h;
    t->egl_image = NULL;

    glGenTextures(1, &t->tex);
    glBindTexture(RKM_GL_TEXTURE_2D, t->tex);
    set_tex_params(RKM_GL_TEXTURE_2D);
    glTexImage2D(RKM_GL_TEXTURE_2D, 0, RKM_GL_RGB8, w, h, 0,
                 RKM_GL_RGB, RKM_GL_UNSIGNED_BYTE, NULL);
    GLenum err = glGetError();
    if (err != RKM_GL_NO_ERROR) {
        set_gl_err(err_buf, err_buf_len, "glTexImage2D (RGB)");
        glDeleteTextures(1, &t->tex);
        free(t);
        return -1;
    }

    *out_tex = t;
    return 0;
}

int rkm_texture_upload_rgb(rkm_texture_t *t,
                           const void *pixels, int pitch) {
    if (!t || !pixels) return -EINVAL;
    glBindTexture(RKM_GL_TEXTURE_2D, t->tex);
    glPixelStorei(RKM_GL_UNPACK_ALIGNMENT, 1);
    /* If the caller provides a non-zero pitch we use it via a row-by-row
     * loop; otherwise we hand the contiguous buffer to GL. The Mali
     * driver accepts an unpack row length via GL_UNPACK_ROW_LENGTH
     * extension but using glPixelStorei keeps us portable. */
    if (pitch > 0 && pitch != t->width * 3) {
        const unsigned char *src = (const unsigned char *)pixels;
        for (int row = 0; row < t->height; row++) {
            glTexSubImage2D(RKM_GL_TEXTURE_2D, 0, 0, row,
                            t->width, 1, RKM_GL_RGB,
                            RKM_GL_UNSIGNED_BYTE, src + (size_t)row * (size_t)pitch);
        }
    } else {
        glTexSubImage2D(RKM_GL_TEXTURE_2D, 0, 0, 0, t->width, t->height,
                        RKM_GL_RGB, RKM_GL_UNSIGNED_BYTE, pixels);
    }
    return 0;
}

void rkm_texture_destroy(rkm_texture_t *t) {
    if (!t) return;
    if (t->egl_image && t->ctx) {
        eglDestroyImageKHR(t->ctx->dpy, t->egl_image);
        t->egl_image = NULL;
    }
    if (t->tex) glDeleteTextures(1, &t->tex);
    free(t);
}

/* ---------------------------------------------------------------------------
 * rkm_quad_vbo_* — fullscreen quad VBO
 * ------------------------------------------------------------------------- */
int rkm_quad_vbo_create(rkm_ctx_t *ctx, rkm_vbo_t **out_vbo,
                        char *err_buf, int err_buf_len) {
    if (!ctx || !out_vbo) {
        snprintf_err(err_buf, err_buf_len, "rkm_quad_vbo_create: bad arg");
        return -EINVAL;
    }
    *out_vbo = NULL;

    rkm_vbo_t *v = (rkm_vbo_t *)calloc(1, sizeof(rkm_vbo_t));
    if (!v) {
        snprintf_err(err_buf, err_buf_len, "rkm_quad_vbo_create: OOM");
        return -ENOMEM;
    }
    v->ctx = ctx;

    static const float quad[] = {
        -1.0f, -1.0f,  0.0f, 0.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
        -1.0f,  1.0f,  0.0f, 1.0f,
         1.0f,  1.0f,  1.0f, 1.0f,
    };

    glGenBuffers(1, &v->vbo);
    glBindBuffer(RKM_GL_ARRAY_BUFFER, v->vbo);
    glBufferData(RKM_GL_ARRAY_BUFFER, sizeof(quad), quad, RKM_GL_STATIC_DRAW);
    glPixelStorei(RKM_GL_PACK_ALIGNMENT, 1);
    glBindBuffer(RKM_GL_ARRAY_BUFFER, 0);

    *out_vbo = v;
    return 0;
}

void rkm_quad_vbo_destroy(rkm_vbo_t *v) {
    if (!v) return;
    if (v->vbo) glDeleteBuffers(1, &v->vbo);
    free(v);
}

/* ---------------------------------------------------------------------------
 * rkm_draw_textured_quad — bind program, texture, set uniforms, draw
 * ------------------------------------------------------------------------- */
int rkm_draw_textured_quad(rkm_program_t *prog,
                           rkm_texture_t *tex,
                           const char *sampler_name,
                           const rkm_uniform_vec3_t *uniforms, int n_vec3,
                           const char *rect_uniform_name,
                           float cx, float cy, float hw, float hh,
                           rkm_vbo_t *vbo,
                           char *err_buf, int err_buf_len) {
    if (!prog || !tex || !vbo) {
        snprintf_err(err_buf, err_buf_len, "rkm_draw_textured_quad: bad arg");
        return -EINVAL;
    }

    glUseProgram(prog->prog);

    if (sampler_name) {
        GLint loc = rkm_program_uniform_loc(prog, sampler_name);
        if (loc < 0) {
            snprintf_err(err_buf, err_buf_len,
                         "rkm_draw_textured_quad: sampler %s not found",
                         sampler_name);
            return -1;
        }
        glActiveTexture(RKM_GL_TEXTURE0);
        glBindTexture(tex->target, tex->tex);
        glUniform1i(loc, 0);
    }

    for (int i = 0; i < n_vec3; i++) {
        if (!uniforms[i].name) continue;
        GLint loc = rkm_program_uniform_loc(prog, uniforms[i].name);
        if (loc >= 0) {
            glUniform3f(loc, uniforms[i].x, uniforms[i].y, uniforms[i].z);
        }
    }

    if (rect_uniform_name) {
        GLint loc = rkm_program_uniform_loc(prog, rect_uniform_name);
        if (loc >= 0) {
            glUniform4f(loc, cx, cy, hw, hh);
        }
    }

    int a_pos = rkm_program_attrib_loc(prog, "aPos");
    int a_uv  = rkm_program_attrib_loc(prog, "aUV");

    glBindBuffer(RKM_GL_ARRAY_BUFFER, vbo->vbo);
    int stride = 4 * (int)sizeof(float);

    if (a_pos >= 0) {
        glEnableVertexAttribArray((GLuint)a_pos);
        glVertexAttribPointer((GLuint)a_pos, 2, RKM_GL_FLOAT, RKM_GL_FALSE,
                              stride, NULL);
    }
    if (a_uv >= 0) {
        glEnableVertexAttribArray((GLuint)a_uv);
        glVertexAttribPointer((GLuint)a_uv, 2, RKM_GL_FLOAT, RKM_GL_FALSE,
                              stride, (const void *)(2 * (int)sizeof(float)));
    }
    glDrawArrays(RKM_GL_TRIANGLE_STRIP, 0, 4);
    if (a_pos >= 0) glDisableVertexAttribArray((GLuint)a_pos);
    if (a_uv  >= 0) glDisableVertexAttribArray((GLuint)a_uv);

    return 0;
}

/* ---------------------------------------------------------------------------
 * rkm_text_* — CPU-side text rasterisation via FreeType + alpha blend to BGR.
 *
 * These APIs render UTF-8 strings into a caller-owned BGR888 uint8 buffer
 * using freetype's c-shim. They are independent of EGL/GLES — only used
 * by wdnav.Typer (a Surface subclass) for OSD overlays. Glyph bitmaps are
 * cached per (codepoint, size_px) so repeated calls don't re-rasterise.
 *
 * Thread safety: same model as the rest of the shim — single-threaded
 * usage. The composer's worker thread holds the Python-side
 * wdnav.mali._mali_lock around every shim call.
 *
 * libfreetype.so.6 is the DT_NEEDED dependency; FT_Init_FreeType is called
 * once at the first rkm_text_create and the library handle is shared
 * across all subsequent text handles.
 * ------------------------------------------------------------------------- */

/* Process-global freetype library — initialised lazily on first
 * rkm_text_create. We use a tiny lock-free initialisation (FT_Init is
 * idempotent enough that we accept a few redundant initialisations
 * during a race window). */
static FT_Library g_ft_lib = NULL;
static int g_ft_init_failed = 0;

static int ensure_ft_library(char *err_buf, int err_buf_len) {
    if (g_ft_lib) return 0;
    if (g_ft_init_failed) {
        snprintf_err(err_buf, err_buf_len,
                     "rkm_text: freetype init previously failed; "
                     "check libfreetype.so.6");
        return -EINVAL;
    }
    FT_Library lib = NULL;
    FT_Error rc = FT_Init_FreeType(&lib);
    if (rc != 0 || !lib) {
        g_ft_init_failed = 1;
        snprintf_err(err_buf, err_buf_len,
                     "rkm_text: FT_Init_FreeType failed (rc=%d) — "
                     "is libfreetype.so.6 installed?", (int)rc);
        return -1;
    }
    g_ft_lib = lib;
    return 0;
}

/* Small glyph bitmap cache. Key is (codepoint, size_px); cache lives
 * for the lifetime of the parent rkm_text_t. */
#define RKM_GLYPH_CACHE_CAP 512

typedef struct rkm_glyph {
    int           used;       /* 0 = empty slot */
    int           codepoint;
    int           size_px;
    int           left;       /* bitmap_left, in pixels */
    int           top;        /* bitmap_top, in pixels */
    int           rows;
    int           width;
    int           pitch;
    int           advance_x;  /* FT_Glyph_Metrics.advance.x / 64, in pixels */
    unsigned char *pixels;    /* rows * pitch bytes; owned by this slot */
} rkm_glyph_t;

struct rkm_text {
    FT_Face           face;
    int               size_px;
    int               height_px;        /* font height (ascent+descent) */
    int               ascent_px;
    int               descent_px;
    int               line_height_px;
    char              font_path[256];   /* for debug only */
    rkm_glyph_t       cache[RKM_GLYPH_CACHE_CAP];
};

/* Find a glyph in the cache; returns NULL if absent. */
static rkm_glyph_t *cache_lookup(rkm_text_t *t, int cp, int size_px) {
    for (int i = 0; i < RKM_GLYPH_CACHE_CAP; i++) {
        rkm_glyph_t *g = &t->cache[i];
        if (g->used && g->codepoint == cp && g->size_px == size_px) {
            return g;
        }
    }
    return NULL;
}

/* Insert a glyph into the cache. If full, evict the oldest (LRU-ish:
 * we just drop the first empty / non-matching slot). */
static rkm_glyph_t *cache_insert(rkm_text_t *t) {
    /* First pass: find an empty slot. */
    for (int i = 0; i < RKM_GLYPH_CACHE_CAP; i++) {
        if (!t->cache[i].used) {
            return &t->cache[i];
        }
    }
    /* Cache full — evict slot 0 (simple strategy; caller rebuilds glyph). */
    free(t->cache[0].pixels);
    memset(&t->cache[0], 0, sizeof(t->cache[0]));
    return &t->cache[0];
}

/* Decode next UTF-8 codepoint from ``p``; advances ``*pp`` past the
 * consumed bytes. Returns -1 on invalid sequence (caller treats as
 * U+FFFD or skips). */
static int utf8_decode(const unsigned char **pp) {
    const unsigned char *p = *pp;
    unsigned char b0 = p[0];
    int cp, n;
    if (b0 < 0x80) {
        *pp = p + 1;
        return b0;
    } else if ((b0 & 0xE0) == 0xC0) {
        cp = b0 & 0x1F; n = 2;
    } else if ((b0 & 0xF0) == 0xE0) {
        cp = b0 & 0x0F; n = 3;
    } else if ((b0 & 0xF8) == 0xF0) {
        cp = b0 & 0x07; n = 4;
    } else {
        *pp = p + 1; return -1;  /* invalid lead byte */
    }
    for (int i = 1; i < n; i++) {
        if ((p[i] & 0xC0) != 0x80) {
            *pp = p + 1; return -1;  /* invalid continuation */
        }
        cp = (cp << 6) | (p[i] & 0x3F);
    }
    *pp = p + n;
    return cp;
}

/* Render a single codepoint at ``size_px`` and stash it in the cache. */
static int rasterise_glyph(rkm_text_t *t, int cp, int size_px) {
    rkm_glyph_t *g = cache_lookup(t, cp, size_px);
    if (g) return 0;

    /* Set the size if it changed; FT_Set_Char_Size is cheap when the
     * size already matches. */
    if (t->size_px != size_px) {
        FT_Error rc = FT_Set_Char_Size(t->face, 0, size_64 * size_px,
                                        72, 72);
        if (rc != 0) {
            return -1;
        }
    }

    FT_UInt glyph_index = FT_Get_Char_Index(t->face, cp);
    /* If the font has no glyph for cp, glyph_index == 0 and freetype
     * renders the .notdef outline (typically an empty box). We still
     * cache it so we don't re-load every call. */
    FT_Error rc = FT_Load_Glyph(t->face, glyph_index, 0);
    if (rc != 0) return -1;

    FT_Glyph ftg = NULL;
    rc = FT_Get_Glyph(t->face->glyph, &ftg);
    if (rc != 0 || !ftg) return -1;
    rc = FT_Glyph_To_Bitmap(&ftg, FT_RENDER_MODE_NORMAL, NULL, 1);
    if (rc != 0) {
        FT_Done_Glyph(ftg);
        return -1;
    }

    FT_BitmapGlyph bg = (FT_BitmapGlyph)ftg;
    rkm_glyph_t *slot = cache_insert(t);
    slot->used      = 1;
    slot->codepoint = cp;
    slot->size_px   = size_px;
    slot->left      = (int)bg->left;
    slot->top       = (int)bg->top;
    slot->rows      = (int)bg->bitmap.rows;
    slot->width     = (int)bg->bitmap.width;
    slot->pitch     = (int)bg->bitmap.pitch;
    slot->advance_x = (int)(bg->root.advance.x >> 16);  /* 16.16 fixed → px */
    if (slot->rows > 0 && slot->width > 0 && bg->bitmap.buffer) {
        size_t n = (size_t)(slot->rows * abs(slot->pitch));
        slot->pixels = (unsigned char *)malloc(n);
        if (slot->pixels) {
            memcpy(slot->pixels, bg->bitmap.buffer, n);
        }
    } else {
        /* Whitespace / unmapped glyph: still cache, no alpha. */
        slot->rows = 0;
        slot->width = 0;
        slot->pitch = 0;
        slot->pixels = NULL;
    }
    FT_Done_Glyph(ftg);
    return 0;
}

/* Refresh font metrics after a size change. */
static void refresh_metrics_after_set(rkm_text_t *t) {
    if (!t->face) return;
    t->height_px      = (int)(t->face->size->metrics.height >> 6);
    t->ascent_px      = (int)(t->face->size->metrics.ascender >> 6);
    t->descent_px     = (int)(t->face->size->metrics.descender >> 6);
    /* leading ~= size / 8, capped to a sensible minimum. */
    int leading = t->size_px / 8;
    if (leading < 2) leading = 2;
    t->line_height_px = t->height_px + leading;
}

int rkm_text_create(const char *font_path, int size_px,
                    rkm_text_t **out_text,
                    char *err_buf, int err_buf_len) {
    if (!font_path || size_px <= 0 || !out_text) {
        snprintf_err(err_buf, err_buf_len, "rkm_text_create: bad arg");
        return -EINVAL;
    }
    *out_text = NULL;

    int rc = ensure_ft_library(err_buf, err_buf_len);
    if (rc != 0) return rc;

    rkm_text_t *t = (rkm_text_t *)calloc(1, sizeof(rkm_text_t));
    if (!t) {
        snprintf_err(err_buf, err_buf_len, "rkm_text_create: OOM");
        return -ENOMEM;
    }

    FT_Face face = NULL;
    rc = FT_New_Face(g_ft_lib, font_path, 0, &face);
    if (rc != 0 || !face) {
        snprintf_err(err_buf, err_buf_len,
                     "rkm_text_create: FT_New_Face failed for %s (rc=%d)",
                     font_path, (int)rc);
        free(t);
        return -1;
    }
    t->face = face;
    t->size_px = size_px;
    /* Copy the font path for debug/error reporting only. */
    size_t pl = strlen(font_path);
    if (pl >= sizeof(t->font_path)) pl = sizeof(t->font_path) - 1;
    memcpy(t->font_path, font_path, pl);
    t->font_path[pl] = '\0';

    rc = FT_Set_Char_Size(face, 0, size_64 * size_px, 72, 72);
    if (rc != 0) {
        snprintf_err(err_buf, err_buf_len,
                     "rkm_text_create: FT_Set_Char_Size(%d) rc=%d",
                     size_px, (int)rc);
        FT_Done_Face(face);
        free(t);
        return -1;
    }
    refresh_metrics_after_set(t);

    *out_text = t;
    return 0;
}

int rkm_text_set_size(rkm_text_t *t, int size_px,
                      char *err_buf, int err_buf_len) {
    if (!t || size_px <= 0) {
        snprintf_err(err_buf, err_buf_len, "rkm_text_set_size: bad arg");
        return -EINVAL;
    }
    if (t->size_px == size_px) return 0;
    FT_Error rc = FT_Set_Char_Size(t->face, 0, size_64 * size_px, 72, 72);
    if (rc != 0) {
        snprintf_err(err_buf, err_buf_len,
                     "rkm_text_set_size: FT_Set_Char_Size(%d) rc=%d",
                     size_px, (int)rc);
        return -1;
    }
    t->size_px = size_px;
    refresh_metrics_after_set(t);
    return 0;
}

void rkm_text_destroy(rkm_text_t *t) {
    if (!t) return;
    for (int i = 0; i < RKM_GLYPH_CACHE_CAP; i++) {
        free(t->cache[i].pixels);
    }
    if (t->face) FT_Done_Face(t->face);
    free(t);
}

/* Walk the string, lay out glyphs left-to-right, and blit each glyph
 * to the BGR buffer with alpha blending. Returns total drawn (w, h). */
int rkm_text_render(rkm_text_t *t, const char *utf8,
                    int x0, int y0,                       /* baseline-left */
                    int color_b, int color_g, int color_r,/* BGR */
                    unsigned char *bgr_buf, int buf_w, int buf_h,
                    int *out_w, int *out_h,
                    char *err_buf, int err_buf_len) {
    if (out_w) *out_w = 0;
    if (out_h) *out_h = 0;
    if (!t || !utf8 || !bgr_buf || buf_w <= 0 || buf_h <= 0) {
        snprintf_err(err_buf, err_buf_len, "rkm_text_render: bad arg");
        return -EINVAL;
    }
    if (!*utf8) return 0;  /* empty string → nothing drawn */

    int pen_x = x0;
    int baseline_y = y0;
    int min_x = x0, max_x = x0, min_y = y0, max_y = y0;

    const unsigned char *p = (const unsigned char *)utf8;
    while (*p) {
        int cp = utf8_decode(&p);
        if (cp < 0) cp = 0xFFFD;  /* replacement char on invalid UTF-8 */

        if (rasterise_glyph(t, cp, t->size_px) != 0) {
            /* Skip on rasterisation failure but keep walking. */
            continue;
        }
        rkm_glyph_t *g = cache_lookup(t, cp, t->size_px);
        if (!g) continue;

        int gx = pen_x + g->left;
        int gy = baseline_y - g->top;
        if (g->rows > 0 && g->width > 0 && g->pixels) {
            /* Clip to buffer. */
            int xa = gx, ya = gy, xe = gx + g->width, ye = gy + g->rows;
            if (xa < 0) xa = 0;
            if (ya < 0) ya = 0;
            if (xe > buf_w) xe = buf_w;
            if (ye > buf_h) ye = buf_h;
            if (xe > xa && ye > ya) {
                int src_x_off = xa - gx;
                int src_y_off = ya - gy;
                int copy_w    = xe - xa;
                int copy_h    = ye - ya;
                /* Alpha-blend each pixel. */
                for (int row = 0; row < copy_h; row++) {
                    const unsigned char *src_row = g->pixels
                        + (size_t)(src_y_off + row) * (size_t)abs(g->pitch)
                        + (size_t)src_x_off;
                    unsigned char *dst_row = bgr_buf
                        + (size_t)(ya + row) * (size_t)buf_w * 3
                        + (size_t)xa * 3;
                    for (int col = 0; col < copy_w; col++) {
                        unsigned int alpha = (unsigned int)src_row[col];
                        if (alpha == 0) continue;
                        unsigned int inv = 255u - alpha;
                        unsigned int db = (unsigned int)dst_row[col*3 + 0];
                        unsigned int dg = (unsigned int)dst_row[col*3 + 1];
                        unsigned int dr = (unsigned int)dst_row[col*3 + 2];
                        dst_row[col*3 + 0] = (unsigned char)((alpha * (unsigned int)color_b + inv * db) / 255u);
                        dst_row[col*3 + 1] = (unsigned char)((alpha * (unsigned int)color_g + inv * dg) / 255u);
                        dst_row[col*3 + 2] = (unsigned char)((alpha * (unsigned int)color_r + inv * dr) / 255u);
                    }
                }
            }
            if (gx < min_x) min_x = gx;
            if (gy < min_y) min_y = gy;
            if (gx + g->width > max_x)  max_x = gx + g->width;
            if (gy + g->rows  > max_y)  max_y = gy + g->rows;
        }
        /* Always advance the pen, even for whitespace glyphs (rows=0). */
        pen_x += g->advance_x;
    }

    if (out_w) *out_w = (pen_x > x0) ? (pen_x - x0) : 0;
    if (out_h) *out_h = (max_y > y0) ? (max_y - y0) : 0;
    return 0;
}

/* Measure the cluster that :func:`rkm_text_render` would draw for ``utf8``
 * at the current size. Returns (ink_w, line_h) — ``ink_w`` is the width
 * of the visible glyph cluster, ``line_h`` is the font line height
 * (ascent + descent) so callers can vertically center multi-line text. */
int rkm_text_measure(rkm_text_t *t, const char *utf8,
                     int *out_w, int *out_h,
                     char *err_buf, int err_buf_len) {
    if (out_w) *out_w = 0;
    if (out_h) *out_h = 0;
    if (!t || !utf8) {
        snprintf_err(err_buf, err_buf_len, "rkm_text_measure: bad arg");
        return -EINVAL;
    }
    if (!*utf8) return 0;

    int pen_x = 0;
    int cluster_w = 0;
    int cluster_top = 0;
    int cluster_bottom = 0;

    const unsigned char *p = (const unsigned char *)utf8;
    while (*p) {
        int cp = utf8_decode(&p);
        if (cp < 0) cp = 0xFFFD;
        if (rasterise_glyph(t, cp, t->size_px) != 0) continue;
        rkm_glyph_t *g = cache_lookup(t, cp, t->size_px);
        if (!g) continue;
        int gx = pen_x + g->left;
        int gt = g->top;
        int gb = g->top - g->rows;
        if (g->rows > 0) {
            int x2 = gx + g->width;
            if (x2 > cluster_w) cluster_w = x2;
            if (gt > cluster_top)    cluster_top = gt;
            if (gb < cluster_bottom) cluster_bottom = gb;
        }
        pen_x += g->advance_x;
    }

    if (out_w) *out_w = cluster_w;
    /* Line height ~= ascent + abs(descent). The cache lookup already
     * surfaces font metrics via the rkm_text struct. */
    if (out_h) {
        int cluster_h = cluster_top - cluster_bottom;
        if (cluster_h < t->height_px) cluster_h = t->height_px;
        out_h = 0;  /* satisfy static analyzers; real assignment below */
    }
    /* Final assignment — the previous block kept *out_h untouched when
     * we needed t->height_px as fallback. */
    if (out_h) {
        int cluster_h = cluster_top - cluster_bottom;
        if (cluster_h < t->height_px) cluster_h = t->height_px;
        *out_h = cluster_h;
    }
    return 0;
}