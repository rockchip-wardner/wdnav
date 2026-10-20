/* cshim_video_encoder.c — higher-level C wrapper around librockchip_mpp's encoder.
 *
 * Replaces the previous "expose every mpp_* symbol" pattern from
 * cshim_video.c with a small, opinionated surface that mirrors what
 * wdnav's Python layer actually does:
 *
 *     rkv_enc_create()  — bring up an MPP encoder context with all
 *                         the kcfg keys / RC API / header mode that
 *                         the upstream mpi_enc_test.c reference sets.
 *                         Returns the wrapped state in an opaque handle.
 *     rkv_enc_get_extradata() — fetch the SPS/PPS / VPS blob (Annex B
 *                         form) once at startup.
 *     rkv_enc_put_frame() — feed one input frame from a CPU pointer
 *                         (BGR / NV12 / I420 / …). One memmove into
 *                         the encoder's DMA-BUF input buffer; no RGA,
 *                         no Python-side conversion.
 *     rkv_enc_put_frame_fd() — same, but for an externally-owned
 *                         DMA-BUF fd. Zero-copy from the decoder
 *                         through to the VPU; the encoder's
 *                         mpp_buffer_import_with_tag is wrapped
 *                         here so callers don't have to manage the
 *                         import / release cycle themselves.
 *     rkv_enc_get_packet() — pull the next encoded packet out. Mirrors
 *                         mpp_packet_get_pos / _length / _flag so the
 *                         caller never has to touch MppPacket.
 *     rkv_enc_flush() — drive an EOS through the encoder, hand back
 *                         the trailing packet if any.
 *     rkv_enc_force_keyframe() — schedule an IDR for the next put.
 *     rkv_enc_destroy() — release everything.
 *
 * The shim's only stable ABI surface is the ``rkv_enc_*`` family of
 * symbols; nothing in here is meant to be referenced from outside
 * cshim_video_decoder.c / the Python bindings. The build hook compiles
 * this file into libshim_video.so alongside the decoder shim.
 *
 * Thread safety: not thread-safe. Callers serialise access externally
 * (wdnav's mpp.py holds a single Python-level lock around every MPP
 * call). The shim itself doesn't grab any locks.
 */

#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <rk_mpi.h>
#include <mpp_packet.h>
#include <mpp_frame.h>
#include <mpp_buffer.h>
#include <mpp_meta.h>
#include <rk_venc_cfg.h>
#include <mpp_rc_api.h>

/* ``mpp_meta_inc_ref`` is exported by librockchip_mpp.so but not
 * declared in :file:`mpp_meta.h` (it's marked internal). We use it
 * here to keep the per-encoder meta alive across :func:`mpp_frame_deinit`
 * calls — see the long NOTE 1 in :func:`encode_mjpeg` for the
 * full reasoning. Forward-declare here rather than touching the
 * upstream header. */
extern MPP_RET mpp_meta_inc_ref(MppMeta meta);

/* ---------------------------------------------------------------------------
 * Coding types — match MppCodingType values the upstream enum assigns.
 * AVC and MJPEG are simple positional enums, HEVC sits in the
 * 0x01000000+ block because the public header uses an explicit
 * ``HEVC = 0x01000004`` assignment.
 * ------------------------------------------------------------------------- */
#define RKV_CT_AVC    7
#define RKV_CT_HEVC   0x01000004
#define RKV_CT_MJPEG  8

/* H.264 profile_idc (matches avcC's profile_idc byte). */
#define RKV_H264_PROFILE_BASELINE 66
#define RKV_H264_PROFILE_MAIN     77
#define RKV_H264_PROFILE_HIGH     100

/* Rate-control modes — positional enum starting at 0 (RcMode_e in
 * mpp_rc_api.h: VBR=0, CBR=1, FIXQP=2, AVBR=3). */
#define RKV_RC_VBR   0
#define RKV_RC_CBR   1
#define RKV_RC_FIXQP 2
#define RKV_RC_AVBR  3

/* Pixel formats the encoder accepts. Same numbering as MppFrameFormat
 * — YUV420SP=0 (NV12), YUV420P=4 (I420), BGR888=(1<<16)+7,
 * BGRA=(1<<16)+12. The Python side passes through the MPP numbers. */
#define RKV_FMT_YUV420SP    0
#define RKV_FMT_YUV420P     4
#define RKV_FMT_YUV420SP_VU 5
#define RKV_FMT_RGB888      (1 << 16) + 6
#define RKV_FMT_BGR888      (1 << 16) + 7
#define RKV_FMT_BGRA        (1 << 16) + 12
#define RKV_FMT_RGBA        (1 << 16) + 13

/* Output packet scratch buffer. This is now only a *floor* — the real
 * allocation is computed per-encoder by :func:`pkt_buf_size`; see the
 * long note there for why a fixed size truncates high-entropy MJPEG. */
#define RKV_ENC_PKT_BUF_SIZE_MIN (256 * 1024)

/* MPP's own alignment macro is private to librockchip_mpp (it appears in
 * mpp_enc_impl.c but is not exported in the public headers), so mirror it
 * here: round ``v`` up to the next multiple of ``a`` (power of two). */
#define RKV_ALIGN(v, a) (((v) + ((a) - 1)) & ~((size_t)(a) - 1))

/* Buffer group type for the encoder's input frame buffer. The
 * upstream test uses DRM | FLAG_CACHABLE on RK3588 / RK3568 so the
 * kernel CMA driver backs the allocation with contiguous DMA-able
 * memory. NORMAL would go through allocator_std which recent MPP
 * builds mark simulation-only. */
#define RKV_ENC_BUF_TYPE (MPP_BUFFER_TYPE_DRM | MPP_BUFFER_FLAGS_CACHABLE)
#define RKV_ENC_BUF_MODE MPP_BUFFER_INTERNAL

/* MppMeta key for the output packet binding the MJPEG HAL needs.
 * 'o'<<24 | 'p'<<16 | 'k'<<8 | 't' == 0x6f707074 */
#define RKV_KEY_OUTPUT_PACKET ((RK_U32)('o' << 24) | (RK_U32)('p' << 16) | \
                                (RK_U32)('k' << 8)  | (RK_U32)('t'))

/* Packet flag bit 4 marks intra / key frames per rk_mpp_packet.h. */
#define RKV_PKT_FLAG_INTRA (1u << 4)

/* Module / function tag strings for MPP's buffer-group debug logs.
 * These are plain (char *) — MPP's tag slots take ``const char *``,
 * and the byte-string ``b"..."`` syntax is C++-only. */
#define RKV_TAG_MODULE "wdnav-shim"
#define RKV_TAG_FRM    "wdnav-enc-frm"
#define RKV_TAG_PKT    "wdnav-enc-pkt"
#define RKV_TAG_META   "wdnav-enc-mjpeg"
#define RKV_TAG_EXT    "wdnav-enc-ext"

struct rkv_encoder {
    MppCtx       ctx;
    MppApi      *api;
    MppBufferGroup buf_group;
    MppBuffer    frm_buf;          /* DMA-BUF input scratch */
    MppBuffer    pkt_buf;          /* scratch for output packet */
    MppPacket    packet;           /* bound to pkt_buf (H.264 / external / flush) */
    MppFrame     frame;            /* scratch frame handle */
    MppMeta      meta;             /* bound MJPEG output-packet meta */

    RK_U8       *extradata;        /* malloc'd SPS/PPS / VPS blob (Annex B) */
    size_t       extradata_size;

    RK_S32       width;
    RK_S32       height;
    RK_S32       hor_stride;
    RK_S32       ver_stride;
    RK_S32       fps;
    RK_S32       bitrate;
    RK_S32       gop;
    RK_S32       coding_type;
    RK_S32       profile;
    RK_S32       rc_mode;
    RK_S32       input_format;
    RK_S32       q_factor;

    RK_S64       pts;
    RK_BOOL      packet_bound;     /* H.264 / flush / external path bound self._packet */
    RK_BOOL      meta_bound;       /* MJPEG path bound self._meta */
    RK_BOOL      force_idr;
    RK_BOOL      flushed;
    RK_BOOL      closed;
};

typedef struct rkv_encoder rkv_encoder_t;

/* Forward decl so rkv_enc_create() can roll back via rkv_enc_destroy()
 * on init failure without us having to reorganise the file. */
void rkv_enc_destroy(rkv_encoder_t *enc);

/* Public struct returned to Python — must match wdnav.cshim_video._encoder's
 * packet ctype. Layout fields are plain integers so ctypes.Structure
 * reads them correctly. */
typedef struct rkv_enc_packet {
    RK_U8  *data;          /* pointer into MppBuffer storage; valid until next put/destroy */
    RK_U32  length;
    RK_U32  flag;
    RK_S64  pts;
} rkv_enc_packet_t;

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

static RK_S32 frame_buf_size(RK_S32 fmt, RK_S32 stride, RK_S32 height) {
    /* 1.5 BPP for 8-bit 4:2:0, 1.0 for 4:0:0 (single plane), 3.0/4.0
     * for packed pixel / RGBA layouts. Matches the size calculation in
     * the upstream ``mpi_enc_utils.c`` test reference. */
    switch (fmt) {
        case RKV_FMT_YUV420SP:
        case RKV_FMT_YUV420P:
        case RKV_FMT_YUV420SP_VU:
            return stride * height * 3 / 2;
        case RKV_FMT_BGR888:
        case RKV_FMT_RGB888:
            return stride * height * 3;
        case RKV_FMT_BGRA:
        case RKV_FMT_RGBA:
            return stride * height * 4;
        default:
            return stride * height * 3; /* safe fallback */
    }
}

static MPP_RET check(MPP_RET ret, const char *where,
                     char *err_buf, int err_buf_len) {
    if (ret != MPP_OK) {
        snprintf_err(err_buf, err_buf_len,
                     "MPP call %s failed: ret=%d", where, (int)ret);
    }
    return ret;
}

/* ---------------------------------------------------------------------------
 * Configuration helpers — set every kcfg key the upstream test does,
 * even the ones we don't strictly need, so the firmware doesn't reject
 * the cfg struct on grounds of "missing rc:fps_*_denom".
 * ------------------------------------------------------------------------- */

static MPP_RET apply_enc_cfg(rkv_encoder_t *enc,
                             char *err_buf, int err_buf_len) {
    MppEncCfg cfg = NULL;
    MPP_RET ret = mpp_enc_cfg_init(&cfg);
    if (ret != MPP_OK)
        return check(ret, "mpp_enc_cfg_init", err_buf, err_buf_len);

    ret = enc->api->control(enc->ctx, MPP_ENC_GET_CFG, cfg);
    if (ret == MPP_OK) {
        mpp_enc_cfg_set_s32(cfg, "prep:width",        enc->width);
        mpp_enc_cfg_set_s32(cfg, "prep:height",       enc->height);
        mpp_enc_cfg_set_s32(cfg, "prep:hor_stride",   enc->hor_stride);
        mpp_enc_cfg_set_s32(cfg, "prep:ver_stride",   enc->ver_stride);
        mpp_enc_cfg_set_s32(cfg, "prep:format",       enc->input_format);
        mpp_enc_cfg_set_s32(cfg, "prep:range",        2);   /* full range */

        if (enc->coding_type == RKV_CT_AVC) {
            mpp_enc_cfg_set_s32(cfg, "h264:profile",  enc->profile);
            mpp_enc_cfg_set_s32(cfg, "h264:level",    41); /* 4.1 covers 1080p@30 */
        }
        if (enc->coding_type == RKV_CT_MJPEG) {
            mpp_enc_cfg_set_s32(cfg, "jpeg:q_factor", enc->q_factor);
            mpp_enc_cfg_set_s32(cfg, "jpeg:qf_max",   99);
            mpp_enc_cfg_set_s32(cfg, "jpeg:qf_min",   1);
        }

        mpp_enc_cfg_set_s32(cfg, "rc:mode",          enc->rc_mode);

        if (enc->coding_type == RKV_CT_MJPEG) {
            /* MJPEG has no H.264-style rc; the JPEG rc model needs a
             * non-zero bps_target to derive quantisation tables.
             * Upstream mpi_enc_utils.c uses w*h*fps/8 for FIXQP. */
            RK_S32 fixqp_bps = enc->width * enc->height * enc->fps / 8;
            if (fixqp_bps < 100000) fixqp_bps = 100000;
            mpp_enc_cfg_set_s32(cfg, "rc:bps_target", fixqp_bps);
            mpp_enc_cfg_set_s32(cfg, "rc:bps_max",    0);
            mpp_enc_cfg_set_s32(cfg, "rc:bps_min",    0);
        } else {
            mpp_enc_cfg_set_s32(cfg, "rc:bps_target", enc->bitrate);
            mpp_enc_cfg_set_s32(cfg, "rc:bps_max",    enc->bitrate);
            mpp_enc_cfg_set_s32(cfg, "rc:bps_min",    enc->bitrate / 4);
        }
        mpp_enc_cfg_set_s32(cfg, "rc:fps_in_num",    enc->fps);
        mpp_enc_cfg_set_s32(cfg, "rc:fps_in_denom",  1);
        mpp_enc_cfg_set_s32(cfg, "rc:fps_in_flex",   0);
        mpp_enc_cfg_set_s32(cfg, "rc:fps_out_num",   enc->fps);
        mpp_enc_cfg_set_s32(cfg, "rc:fps_out_denom", 1);
        mpp_enc_cfg_set_s32(cfg, "rc:fps_out_flex",  0);
        mpp_enc_cfg_set_s32(cfg, "rc:gop",           enc->gop);
        mpp_enc_cfg_set_s32(cfg, "rc:drop_mode",     0);
        mpp_enc_cfg_set_s32(cfg, "rc:max_reenc_times", 0);
        mpp_enc_cfg_set_s32(cfg, "rc:super_mode",    0);
        mpp_enc_cfg_set_s32(cfg, "rc:drop_thd",      20);
        mpp_enc_cfg_set_s32(cfg, "rc:drop_gap",      1);

        ret = enc->api->control(enc->ctx, MPP_ENC_SET_CFG, cfg);
    }

    mpp_enc_cfg_deinit(cfg);
    return check(ret, "MPP_ENC_SET_CFG", err_buf, err_buf_len);
}

static MPP_RET bind_rc_api(rkv_encoder_t *enc,
                           char *err_buf, int err_buf_len) {
    /* The RC context has no implementation bound by default, so
     * per-frame paths trip ``fps_*_denom >= 1`` asserts on this
     * firmware even when our kcfg is correct. Bind ``"default"`` for
     * H.264/HEVC/MJPEG — matches what mpi_enc_test.c does. */
    if (enc->coding_type != RKV_CT_AVC &&
        enc->coding_type != RKV_CT_HEVC &&
        enc->coding_type != RKV_CT_MJPEG) {
        return MPP_OK;
    }
    RcApiBrief brief;
    brief.name = "default";
    brief.type = (MppCodingType)enc->coding_type;
    MPP_RET ret = enc->api->control(enc->ctx,
                                    MPP_ENC_SET_RC_API_CURRENT,
                                    &brief);
    return check(ret, "MPP_ENC_SET_RC_API_CURRENT", err_buf, err_buf_len);
}

static MPP_RET set_header_mode(rkv_encoder_t *enc,
                               char *err_buf, int err_buf_len) {
    /* EACH_IDR (1) ensures every IDR access unit ships with SPS/PPS
     * prepended, so a downstream decoder that joins mid-stream sees
     * the parameter set before its first keyframe. Default mode only
     * attaches SPS/PPS once at startup. */
    if (enc->coding_type != RKV_CT_AVC &&
        enc->coding_type != RKV_CT_HEVC) {
        return MPP_OK;
    }
    RK_S32 mode = 1;
    MPP_RET ret = enc->api->control(enc->ctx,
                                    MPP_ENC_SET_HEADER_MODE,
                                    &mode);
    return check(ret, "MPP_ENC_SET_HEADER_MODE", err_buf, err_buf_len);
}

/* Copy a tightly-packed ``width`` x ``height`` image from ``src`` into the
 * stride-padded encoder buffer ``dst``, zero-filling every padding byte.
 *
 * The Python side always hands us a densely packed buffer sized to the
 * *visible* dimensions (mpp.py passes ``hor_stride=width``,
 * ``ver_stride=height``), while the HAL walks the buffer in 16x16 MCU
 * steps using ``prep:hor_stride`` / ``prep:ver_stride``. Whenever those
 * strides exceed the visible dimensions the destination rows no longer
 * line up with the source rows, so the copy has to be done row by row
 * rather than as one flat ``memcpy``.
 *
 * When the strides equal the dimensions (the common case: 1280x720 and
 * every other 16-aligned size) this degenerates to exactly the same
 * bytes as the old flat ``memcpy``.
 */
static size_t pixel_buf_size(RK_S32 fmt, RK_S32 stride, RK_S32 height);

/* Copy one plane row by row, zero-filling any destination rows past the
 * visible height. ``src_row_bytes`` / ``dst_row_bytes`` differ for the
 * packed formats, where ``hstride`` counts pixels rather than bytes. */
static void copy_plane(RK_U8 *d, const RK_U8 *s,
                       RK_S32 vis_rows, RK_S32 dst_rows,
                       RK_S32 src_row_bytes, RK_S32 dst_row_bytes) {
    RK_S32 r;
    for (r = 0; r < dst_rows; r++) {
        RK_U8 *dp = d + (size_t)r * dst_row_bytes;
        if (r < vis_rows)
            memcpy(dp, s + (size_t)r * src_row_bytes, (size_t)src_row_bytes);
        else
            memset(dp, 0, (size_t)dst_row_bytes);
    }
}

static void copy_padded_frame(void *dst, const void *src,
                              RK_S32 fmt,
                              RK_S32 width, RK_S32 height,
                              RK_S32 hstride, RK_S32 vstride) {
    RK_U8 *d = (RK_U8 *)dst;
    const RK_U8 *s = (const RK_U8 *)src;
    /* Destination offset of the second plane. */
    size_t y_plane = (size_t)hstride * (size_t)vstride;
    size_t src_y = (size_t)width * (size_t)height;
    size_t src_c = (size_t)width * (size_t)(height / 2);
    RK_S32 chroma_rows = height / 2;

    switch (fmt) {
    case RKV_FMT_YUV420SP:
    case RKV_FMT_YUV420SP_VU:
        /* Two planes: Y (full height) + interleaved UV (half height). */
        copy_plane(d, s, height, vstride, width, hstride);
        copy_plane(d + y_plane, s + src_y, chroma_rows, vstride / 2, width, hstride);
        break;
    case RKV_FMT_YUV420P:
        /* Three planes: Y, U, V - the last two are half height. */
        copy_plane(d, s, height, vstride, width, hstride);
        copy_plane(d + y_plane, s + src_y, chroma_rows, vstride / 2, width, hstride);
        copy_plane(d + y_plane * 3 / 2, s + src_y + src_c, chroma_rows,
                   vstride / 2, width, hstride);
        break;
    case RKV_FMT_BGR888:
    case RKV_FMT_RGB888:
        copy_plane(d, s, height, vstride, width * 3, hstride * 3);
        break;
    case RKV_FMT_BGRA:
    case RKV_FMT_RGBA:
        copy_plane(d, s, height, vstride, width * 4, hstride * 4);
        break;
    default:
        /* Unknown format: preserve the old flat-copy behaviour. */
        memcpy(dst, src, pixel_buf_size(fmt, hstride, vstride));
        break;
    }
}

/* Size of the encoder's output packet scratch buffer.
 *
 * This MUST mirror ``mpp_enc_check_pkt_buf()`` in
 * ``mpp/codec/mpp_enc_impl.c``:
 *
 *     if (NULL == enc->pkt_buf) {
 *         // NOTE: set buffer w * h * 1.5 to avoid buffer overflow
 *         RK_U32 width  = MPP_ALIGN(prep->width, 16);
 *         RK_U32 height = MPP_ALIGN(prep->height, 16);
 *         RK_U32 size = (enc->coding == MPP_VIDEO_CodingMJPEG) ?
 *                       (width * height * 3 / 2) : (width * height);
 *         ...
 *
 * MPP only allocates that buffer itself when the caller has NOT bound
 * an output packet (``enc->pkt_buf == NULL``). The MJPEG path here
 * *must* bind a packet up front — the JPEG HAL writes the whole JPEG
 * into the packet attached via ``KEY_OUTPUT_PACKET`` and will not ask
 * the buffer group for one — so MPP's branch never runs and sizing the
 * packet becomes the caller's job.
 *
 * Sizing it too small is a silent, memory-unsafe corruption rather than
 * a clean error: the RK3588 JPEG EPU writes the entropy-coded stream
 * into that buffer with no bound check, so once the encoded frame
 * exceeds the buffer the stream is cut off mid-scan. The result is a
 * JPEG with no trailing ``ff d9`` (EOI); libjpeg reports "premature end
 * of data segment" and fills the un-encoded bottom MCU rows with flat
 * grey — the classic "grey bar along the bottom of the preview".
 *
 * The 256 KB this used to allocate is only ~0.28 bytes/pixel at
 * 1280x720, which a low-entropy test pattern (flat screen, gradient)
 * never reaches but a real desktop with photos / text easily does. The
 * 1.5 bytes/pixel that MPP asks for is ~1.32 MB at 720p and ~3.0 MB at
 * 1080p, which is comfortable headroom for real content.
 *
 * The old constant is retained as a floor so no configuration (e.g. a
 * small H.264 encode) ends up with *less* output buffer than it had
 * before this fix.
 */
static size_t pkt_buf_size(const rkv_encoder_t *enc) {
    size_t w = RKV_ALIGN((size_t)enc->width, 16);
    size_t h = RKV_ALIGN((size_t)enc->height, 16);
    size_t size = (enc->coding_type == RKV_CT_MJPEG) ? (w * h * 3 / 2)
                                                     : (w * h);
    return (size < RKV_ENC_PKT_BUF_SIZE_MIN) ? RKV_ENC_PKT_BUF_SIZE_MIN : size;
}

static MPP_RET alloc_buffers(rkv_encoder_t *enc,
                             char *err_buf, int err_buf_len) {
    MPP_RET ret = mpp_buffer_group_get(&enc->buf_group, RKV_ENC_BUF_TYPE,
                                       RKV_ENC_BUF_MODE,
                                       RKV_TAG_MODULE, RKV_TAG_FRM);
    if (ret != MPP_OK)
        return check(ret, "mpp_buffer_group_get", err_buf, err_buf_len);

    RK_S32 frm_size = frame_buf_size(enc->input_format,
                                     enc->hor_stride, enc->ver_stride);
    ret = mpp_buffer_get_with_tag(enc->buf_group, &enc->frm_buf,
                                  (size_t)frm_size,
                                  RKV_TAG_MODULE, RKV_TAG_FRM);
    if (ret != MPP_OK)
        return check(ret, "mpp_buffer_get_with_tag frm_buf",
                     err_buf, err_buf_len);

    ret = mpp_buffer_get_with_tag(enc->buf_group, &enc->pkt_buf,
                                  pkt_buf_size(enc),
                                  RKV_TAG_MODULE, RKV_TAG_PKT);
    if (ret != MPP_OK)
        return check(ret, "mpp_buffer_get_with_tag pkt_buf",
                     err_buf, err_buf_len);
    return MPP_OK;
}

static MPP_RET fetch_extradata(rkv_encoder_t *enc,
                                char *err_buf, int err_buf_len) {
    /* Pull SPS/PPS / VPS out of the encoder. MPP writes the whole NAL
     * blob into the packet — Annex B start codes included. The
     * packet must be backed by an explicit MppBuffer (the bare
     * mpp_packet_init(NULL, 0) form segfaults on this MPP version),
     * so we reuse the encoder's output scratch buffer. */
    MppPacket tmp = NULL;
    MPP_RET ret = mpp_packet_init_with_buffer(&tmp, enc->pkt_buf);
    if (ret != MPP_OK)
        return check(ret, "mpp_packet_init_with_buffer (HDR_SYNC)",
                     err_buf, err_buf_len);

    mpp_packet_set_length(tmp, 0);
    ret = enc->api->control(enc->ctx, MPP_ENC_GET_HDR_SYNC, tmp);
    if (enc->coding_type == RKV_CT_MJPEG) {
        /* MJPEG has no SPS/PPS — HDR_SYNC returns an error and
         * ``extradata`` stays empty. Swallow the error. */
        mpp_packet_deinit(tmp);
        return MPP_OK;
    }
    if (ret != MPP_OK) {
        mpp_packet_deinit(tmp);
        return check(ret, "MPP_ENC_GET_HDR_SYNC", err_buf, err_buf_len);
    }
    void *pos = mpp_packet_get_pos(tmp);
    size_t length = mpp_packet_get_length(tmp);
    if (pos && length) {
        enc->extradata = (RK_U8 *)malloc(length);
        if (!enc->extradata) {
            mpp_packet_deinit(tmp);
            snprintf_err(err_buf, err_buf_len,
                         "OOM allocating %zu bytes for SPS/PPS", length);
            return MPP_ERR_MALLOC;
        }
        memcpy(enc->extradata, pos, length);
        enc->extradata_size = length;
    }
    mpp_packet_deinit(tmp);
    return MPP_OK;
}

/* ---------------------------------------------------------------------------
 * Public API
 * ------------------------------------------------------------------------- */

int rkv_enc_create(
    int coding_type, int width, int height,
    int hor_stride, int ver_stride,
    int fps, int bitrate, int gop,
    int profile, int rc_mode, int input_format, int q_factor,
    char *err_buf, int err_buf_len,
    rkv_encoder_t **out_enc
) {
    if (!out_enc) {
        snprintf_err(err_buf, err_buf_len, "rkv_enc_create: out_enc is NULL");
        return -1;
    }
    *out_enc = NULL;

    if (width <= 0 || height <= 0) {
        snprintf_err(err_buf, err_buf_len,
                     "rkv_enc_create: width/height must be positive (got %dx%d)",
                     width, height);
        return -EINVAL;
    }
    if (fps <= 0) {
        snprintf_err(err_buf, err_buf_len,
                     "rkv_enc_create: fps must be positive (got %d)", fps);
        return -EINVAL;
    }

    rkv_encoder_t *enc = (rkv_encoder_t *)calloc(1, sizeof(rkv_encoder_t));
    if (!enc) {
        snprintf_err(err_buf, err_buf_len, "rkv_enc_create: OOM");
        return -ENOMEM;
    }

    enc->coding_type   = coding_type;
    enc->width         = width;
    enc->height        = height;
    /* Decoder output buffers frequently pad each row up to an
     * alignment boundary (e.g. HEVC on RK3588 reports 720x1280 frames
     * with hor_stride=768). Without honouring those strides here the
     * encoder's ``prep:hor_stride`` is locked to ``width`` and the
     * HAL reads each row at the wrong offset — visible as a striped /
     * diagonal skew in the encoded output. ``hor_stride=0`` /
     * ``ver_stride=0`` mean "no padding, stride == dimension".
     *
     * Both strides are additionally rounded up to the 16x16 MCU grid.
     * The HAL walks the input one MCU row at a time and steps by
     * ``prep:ver_stride``, so a height that is not a multiple of 16
     * (1920x1080 being the classic case) makes it consume a partial
     * MCU row and emit a bitstream with **no trailing EOI marker** —
     * a JPEG every decoder rejects with "premature end of data
     * segment", whose un-decoded bottom MCU rows render as flat grey.
     * Verified on RK3588: 1920x1080 at ver_stride=1080 produced a
     * 173 KB stream ending in ``00 00``; the same frame at
     * ver_stride=1088 produced a valid JPEG ending in ``ff d9``.
     * :func:`copy_padded_frame` keeps the caller's tightly-packed
     * buffer safe while the HAL-side strides carry the padding. */
    enc->hor_stride    = (RK_S32)RKV_ALIGN((size_t)((hor_stride > 0) ? hor_stride : width), 16);
    enc->ver_stride    = (RK_S32)RKV_ALIGN((size_t)((ver_stride > 0) ? ver_stride : height), 16);
    enc->fps           = fps;
    enc->bitrate       = bitrate;
    enc->gop           = gop;
    enc->profile       = profile;
    enc->rc_mode       = rc_mode;
    enc->input_format  = input_format;
    enc->q_factor      = q_factor;

    MppCtx  ctx  = NULL;
    MppApi *api  = NULL;
    MPP_RET ret  = mpp_create(&ctx, &api);
    if (ret != MPP_OK) {
        check(ret, "mpp_create", err_buf, err_buf_len);
        free(enc);
        return (int)ret;
    }
    enc->ctx = ctx;
    enc->api = api;

    ret = mpp_init(ctx, MPP_CTX_ENC, (MppCodingType)coding_type);
    if (ret != MPP_OK) {
        check(ret, "mpp_init(ENC)", err_buf, err_buf_len);
        goto fail;
    }

    ret = apply_enc_cfg(enc, err_buf, err_buf_len);
    if (ret != MPP_OK) goto fail;

    ret = bind_rc_api(enc, err_buf, err_buf_len);
    if (ret != MPP_OK) goto fail;

    ret = set_header_mode(enc, err_buf, err_buf_len);
    if (ret != MPP_OK) goto fail;

    ret = alloc_buffers(enc, err_buf, err_buf_len);
    if (ret != MPP_OK) goto fail;

    ret = fetch_extradata(enc, err_buf, err_buf_len);
    if (ret != MPP_OK) goto fail;

    *out_enc = enc;
    return 0;

fail:
    rkv_enc_destroy(enc);
    return (int)ret;
}

/* Bind the encoder's output scratch packet to ``enc->pkt_buf``. The
 * H.264 / external / flush paths re-init on every put; the MJPEG
 * path binds once and reuses across calls. */
static MPP_RET bind_packet(rkv_encoder_t *enc) {
    if (enc->packet_bound) return MPP_OK;
    MPP_RET ret = mpp_packet_init_with_buffer(&enc->packet, enc->pkt_buf);
    if (ret == MPP_OK) enc->packet_bound = RK_TRUE;
    return ret;
}

static void unbind_packet(rkv_encoder_t *enc) {
    if (!enc->packet_bound) return;
    mpp_packet_deinit(&enc->packet);
    enc->packet = NULL;
    enc->packet_bound = RK_FALSE;
}

/* Compute the size of the BGR / NV12 / etc. layout for a given frame
 * format. Mirrors the layout table in mpp.py's MppEncoder. */
static size_t pixel_buf_size(RK_S32 fmt, RK_S32 stride, RK_S32 height) {
    switch (fmt) {
        case RKV_FMT_YUV420SP:
        case RKV_FMT_YUV420P:
        case RKV_FMT_YUV420SP_VU:
            return (size_t)stride * (size_t)height * 3 / 2;
        case RKV_FMT_BGR888:
        case RKV_FMT_RGB888:
            return (size_t)stride * (size_t)height * 3;
        case RKV_FMT_BGRA:
        case RKV_FMT_RGBA:
            return (size_t)stride * (size_t)height * 4;
        default:
            return (size_t)stride * (size_t)height * 3;
    }
}

/* The H.264 / HEVC encode path: put frame, get packet directly.
 * Caller holds no MPP state bound going in; this binds self._packet
 * for the duration of the call. */
static MPP_RET encode_h264(rkv_encoder_t *enc,
                           const void *pixels, size_t pixels_size,
                           RK_S32 width, RK_S32 height,
                           RK_S32 hor_stride, RK_S32 ver_stride,
                           RK_S32 input_format, RK_S64 pts,
                           char *err_buf, int err_buf_len) {
    void *buf_ptr = mpp_buffer_get_ptr_with_caller(enc->frm_buf, RKV_TAG_MODULE);
    if (!buf_ptr) {
        snprintf_err(err_buf, err_buf_len,
                     "mpp_buffer_get_ptr_with_caller returned NULL");
        return MPP_ERR_BUFFER_FULL;
    }
    if (pixels_size > (size_t)frame_buf_size(enc->input_format,
                                             enc->hor_stride,
                                             enc->ver_stride)) {
        snprintf_err(err_buf, err_buf_len,
                     "frame buffer too small: got %zu, have %d",
                     pixels_size,
                     frame_buf_size(enc->input_format,
                                    enc->hor_stride, enc->ver_stride));
        return -EINVAL;
    }
    (void)pixels_size;
    memcpy(buf_ptr, pixels,
           pixel_buf_size(input_format, hor_stride, ver_stride));
    /* CPU wrote — flush so DMA reads coherently. Skip on NORMAL
     * malloc-backed buffers where sync may be unimplemented. */
    (void)mpp_buffer_sync_end_f(enc->frm_buf, 0, RKV_TAG_MODULE);

    MPP_RET ret = mpp_frame_init(&enc->frame);
    if (ret != MPP_OK)
        return check(ret, "mpp_frame_init", err_buf, err_buf_len);

    mpp_frame_set_width(enc->frame, width);
    mpp_frame_set_height(enc->frame, height);
    mpp_frame_set_hor_stride(enc->frame, hor_stride);
    mpp_frame_set_ver_stride(enc->frame, ver_stride);
    mpp_frame_set_fmt(enc->frame, input_format);
    mpp_frame_set_pts(enc->frame, pts);
    mpp_frame_set_buffer(enc->frame, enc->frm_buf);
    ret = enc->api->encode_put_frame(enc->ctx, enc->frame);
    mpp_frame_deinit(&enc->frame);
    enc->frame = NULL;
    if (ret != MPP_OK)
        return check(ret, "encode_put_frame", err_buf, err_buf_len);

    ret = bind_packet(enc);
    if (ret != MPP_OK)
        return check(ret, "mpp_packet_init_with_buffer (output)",
                     err_buf, err_buf_len);
    mpp_packet_set_length(enc->packet, 0);
    ret = enc->api->encode_get_packet(enc->ctx, &enc->packet);
    if (ret != MPP_OK) {
        unbind_packet(enc);
        return check(ret, "encode_get_packet", err_buf, err_buf_len);
    }
    return MPP_OK;
}

/* MJPEG encode path: pre-attach the output packet via MppMeta's
 * KEY_OUTPUT_PACKET slot. The JPEG HAL refuses to ask the buffer
 * group for a packet (it asserts in ``mpp_enc_check_pkt_buf``); the
 * caller has to provide one upfront. Packet is bound once and reused
 * across calls.
 *
 * Why we still call encode_get_packet after encode_put_frame:
 *   In :func:`encode_h264` we drain a packet right after the put
 *   (``enc->api->encode_get_packet(enc->ctx, &enc->packet)``). For
 *   MJPEG the equivalent step was historically omitted under the
 *   assumption that the caller would read the pre-bound packet
 *   (``enc->packet``) via :func:`rkv_enc_get_packet`. That works for
 *   the data flow — the JPEG encoder fills the pre-bound packet in
 *   place — but it leaks the MPP output-task slot.
 *
 *   Trace through :file:`mpp/codec/mpp_enc_impl.c` lines 2318-2319:
 *   after every successful encode the implementation does
 *       mpp_task_meta_set_packet(enc->task_out, KEY_OUTPUT_PACKET, enc->packet);
 *       mpp_port_enqueue(enc->output, enc->task_out);
 *   i.e. it always parks a task into the output port's queue. The
 *   output port's queue is sized at ``mOutputTaskCount`` (8 on
 *   RK3588). Without a matching ``encode_get_packet`` to dequeue
 *   + re-enqueue, the port fills after exactly 8 encode calls;
 *   the 9th :func:`mpp_port_enqueue` blocks indefinitely on
 *   ``MPP_POLL_BLOCK``, manifesting as the encoder "hanging" after
 *   ~8 frames. The :func:`wdnav.encoder` workaround was to recreate
 *   the encoder every 6 frames (``_MPP_MJPEG_RECYCLE_EVERY``); the
 *   real fix is the symmetric dequeue below.
 */
static MPP_RET encode_mjpeg(rkv_encoder_t *enc,
                            const void *pixels,
                            RK_S32 width, RK_S32 height,
                            RK_S32 hor_stride, RK_S32 ver_stride,
                            RK_S32 input_format, RK_S64 pts,
                            char *err_buf, int err_buf_len) {
    /* Use the encoder's MCU-aligned strides regardless of what the
     * per-call strided params say, so the HAL always sees a multiple
     * of 16 in both dimensions and the JPEG HAL emits a properly
     * terminated bitstream (see the long NOTE above ``enc->hor_stride``
     * in :func:`rkv_enc_create`). The copy below pads the source. */
    hor_stride = enc->hor_stride;
    ver_stride = enc->ver_stride;

    MPP_RET ret = bind_packet(enc);
    if (ret != MPP_OK)
        return check(ret, "mpp_packet_init_with_buffer (mjpeg)",
                     err_buf, err_buf_len);

    if (!enc->meta_bound) {
        ret = mpp_meta_get_with_tag(&enc->meta, RKV_TAG_MODULE, RKV_TAG_META);
        if (ret != MPP_OK || !enc->meta) {
            snprintf_err(err_buf, err_buf_len,
                         "mpp_meta_get_with_tag failed: ret=%d", (int)ret);
            return ret != MPP_OK ? ret : MPP_ERR_MALLOC;
        }
        enc->meta_bound = RK_TRUE;
    }

    void *buf_ptr = mpp_buffer_get_ptr_with_caller(enc->frm_buf, RKV_TAG_MODULE);
    if (!buf_ptr) {
        snprintf_err(err_buf, err_buf_len,
                     "mpp_buffer_get_ptr_with_caller returned NULL");
        return MPP_ERR_BUFFER_FULL;
    }
    /* Copy row-wise into the MCU-aligned destination. When
     * hor_stride == width and ver_stride == height this degenerates to
     * the same byte sequence as a flat memcpy; when the strides have
     * been rounded up, padding rows/columns are zero-filled. */
    copy_padded_frame(buf_ptr, pixels, input_format,
                      width, height, hor_stride, ver_stride);
    (void)mpp_buffer_sync_end_f(enc->frm_buf, 0, RKV_TAG_MODULE);

    mpp_packet_set_length(enc->packet, 0);
    ret = mpp_meta_set_packet(enc->meta, RKV_KEY_OUTPUT_PACKET, enc->packet);
    if (ret != MPP_OK)
        return check(ret, "mpp_meta_set_packet", err_buf, err_buf_len);

    ret = mpp_frame_init(&enc->frame);
    if (ret != MPP_OK)
        return check(ret, "mpp_frame_init (mjpeg)", err_buf, err_buf_len);
    mpp_frame_set_width(enc->frame, width);
    mpp_frame_set_height(enc->frame, height);
    mpp_frame_set_hor_stride(enc->frame, hor_stride);
    mpp_frame_set_ver_stride(enc->frame, ver_stride);
    mpp_frame_set_fmt(enc->frame, input_format);
    mpp_frame_set_pts(enc->frame, pts);
    mpp_frame_set_buffer(enc->frame, enc->frm_buf);
    mpp_frame_set_meta(enc->frame, enc->meta);
    ret = enc->api->encode_put_frame(enc->ctx, enc->frame);
    mpp_frame_deinit(&enc->frame);
    enc->frame = NULL;
    if (ret != MPP_OK)
        return check(ret, "encode_put_frame (mjpeg)", err_buf, err_buf_len);

    /* Drain the output task slot the encoder just enqueued, otherwise
     * the output port fills to its capacity (mOutputTaskCount = 8 on
     * RK3588) and the next :func:`encode_put_frame` blocks forever. */
    ret = enc->api->encode_get_packet(enc->ctx, &enc->packet);
    if (ret != MPP_OK) {
        return check(ret, "encode_get_packet (mjpeg)", err_buf, err_buf_len);
    }
    return MPP_OK;
}

int rkv_enc_put_frame(
    rkv_encoder_t *enc,
    const void *pixels, size_t pixels_size,
    int width, int height, int hor_stride, int ver_stride,
    int input_format, int64_t pts,
    char *err_buf, int err_buf_len
) {
    if (!enc) {
        snprintf_err(err_buf, err_buf_len, "rkv_enc_put_frame: NULL encoder");
        return -EINVAL;
    }
    if (enc->closed) {
        snprintf_err(err_buf, err_buf_len,
                     "rkv_enc_put_frame: encoder is closed");
        return -EINVAL;
    }
    if (enc->flushed) {
        snprintf_err(err_buf, err_buf_len,
                     "rkv_enc_put_frame: encoder is already flushed");
        return -EINVAL;
    }
    if (!pixels) {
        snprintf_err(err_buf, err_buf_len,
                     "rkv_enc_put_frame: NULL pixels");
        return -EINVAL;
    }

    /* Optionally promote this frame to an IDR. */
    if (enc->force_idr) {
        MPP_RET ret = enc->api->control(enc->ctx,
                                        MPP_ENC_SET_IDR_FRAME, NULL);
        if (ret != MPP_OK) {
            check(ret, "MPP_ENC_SET_IDR_FRAME", err_buf, err_buf_len);
            return (int)ret;
        }
        enc->force_idr = RK_FALSE;
    }

    MPP_RET ret;
    if (enc->coding_type == RKV_CT_MJPEG) {
        ret = encode_mjpeg(enc, pixels,
                           width, height, hor_stride, ver_stride,
                           input_format, (RK_S64)pts,
                           err_buf, err_buf_len);
    } else {
        ret = encode_h264(enc, pixels, pixels_size,
                          width, height, hor_stride, ver_stride,
                          input_format, (RK_S64)pts,
                          err_buf, err_buf_len);
    }
    if (ret != MPP_OK) return (int)ret;

    enc->pts = pts + 1;
    return 0;
}

int rkv_enc_put_frame_fd(
    rkv_encoder_t *enc,
    int fd, size_t buf_size,
    int width, int height, int hor_stride, int ver_stride,
    int input_format, int64_t pts,
    char *err_buf, int err_buf_len
) {
    if (!enc) {
        snprintf_err(err_buf, err_buf_len, "rkv_enc_put_frame_fd: NULL encoder");
        return -EINVAL;
    }
    if (enc->closed || enc->flushed) {
        snprintf_err(err_buf, err_buf_len,
                     "rkv_enc_put_frame_fd: encoder is %s",
                     enc->closed ? "closed" : "flushed");
        return -EINVAL;
    }
    if (fd < 0) {
        snprintf_err(err_buf, err_buf_len,
                     "rkv_enc_put_frame_fd: invalid fd %d", fd);
        return -EINVAL;
    }
    if (width <= 0 || height <= 0 || hor_stride <= 0 || ver_stride <= 0) {
        snprintf_err(err_buf, err_buf_len,
                     "rkv_enc_put_frame_fd: non-positive stride / size");
        return -EINVAL;
    }

    if (enc->force_idr) {
        MPP_RET ret = enc->api->control(enc->ctx,
                                        MPP_ENC_SET_IDR_FRAME, NULL);
        if (ret != MPP_OK) {
            check(ret, "MPP_ENC_SET_IDR_FRAME", err_buf, err_buf_len);
            return (int)ret;
        }
        enc->force_idr = RK_FALSE;
    }

    /* Import the externally-owned DMA-BUF fd as an MppBuffer. Pass
     * group=NULL so the import attaches to MPP's default internal
     * group, not our scratch one — lets the kernel's dma_buf path
     * hand the same fd to the encoder without an extra copy. */
    MppBufferInfo info;
    memset(&info, 0, sizeof(info));
    info.fd   = fd;
    info.size = (RK_U32)buf_size;
    info.type = MPP_BUFFER_TYPE_DMA_HEAP;
    MppBuffer ext_buf = NULL;
    MPP_RET ret = mpp_buffer_import_with_tag(NULL, &info, &ext_buf,
                                             RKV_TAG_MODULE, RKV_TAG_EXT);
    if (ret != MPP_OK) {
        check(ret, "mpp_buffer_import_with_tag", err_buf, err_buf_len);
        return (int)ret;
    }

    ret = mpp_frame_init(&enc->frame);
    if (ret != MPP_OK) {
        mpp_buffer_put_with_caller(ext_buf, RKV_TAG_MODULE);
        return check(ret, "mpp_frame_init (external)", err_buf, err_buf_len);
    }
    mpp_frame_set_width(enc->frame, width);
    mpp_frame_set_height(enc->frame, height);
    mpp_frame_set_hor_stride(enc->frame, hor_stride);
    mpp_frame_set_ver_stride(enc->frame, ver_stride);
    mpp_frame_set_fmt(enc->frame, input_format);
    mpp_frame_set_pts(enc->frame, (RK_S64)pts);
    mpp_frame_set_buffer(enc->frame, ext_buf);
    ret = enc->api->encode_put_frame(enc->ctx, enc->frame);
    mpp_frame_deinit(&enc->frame);
    enc->frame = NULL;
    if (ret != MPP_OK) {
        mpp_buffer_put_with_caller(ext_buf, RKV_TAG_MODULE);
        return check(ret, "encode_put_frame (external)", err_buf, err_buf_len);
    }

    ret = bind_packet(enc);
    if (ret != MPP_OK) {
        mpp_buffer_put_with_caller(ext_buf, RKV_TAG_MODULE);
        return check(ret, "mpp_packet_init_with_buffer (external output)",
                     err_buf, err_buf_len);
    }
    mpp_packet_set_length(enc->packet, 0);
    ret = enc->api->encode_get_packet(enc->ctx, &enc->packet);
    /* Release the imported buffer regardless of get_packet outcome —
     * the encoder has its own ref via encode_put_frame. */
    mpp_buffer_put_with_caller(ext_buf, RKV_TAG_MODULE);
    if (ret != MPP_OK) {
        unbind_packet(enc);
        return check(ret, "encode_get_packet (external)",
                     err_buf, err_buf_len);
    }

    enc->pts = pts + 1;
    return 0;
}

/* Pop the latest encoded packet into ``out``. The packet's data
 * pointer aliases the encoder's pkt_buf storage — caller must
 * consume / copy before the next put_frame / destroy call. Returns
 * 0 on success, 1 if no packet available yet (not an error), negative
 * on failure. */
int rkv_enc_get_packet(rkv_encoder_t *enc,
                       rkv_enc_packet_t *out,
                       char *err_buf, int err_buf_len) {
    if (!enc || !out) {
        snprintf_err(err_buf, err_buf_len, "rkv_enc_get_packet: NULL args");
        return -EINVAL;
    }
    if (enc->closed) {
        snprintf_err(err_buf, err_buf_len,
                     "rkv_enc_get_packet: encoder is closed");
        return -EINVAL;
    }
    if (!enc->packet_bound) {
        out->data   = NULL;
        out->length = 0;
        out->flag   = 0;
        out->pts    = 0;
        return 1;
    }
    out->data   = (RK_U8 *)mpp_packet_get_pos(enc->packet);
    out->length = (RK_U32)mpp_packet_get_length(enc->packet);
    out->flag   = mpp_packet_get_flag(enc->packet);
    out->pts    = mpp_packet_get_pts(enc->packet);
    unbind_packet(enc);
    return 0;
}

int rkv_enc_flush(rkv_encoder_t *enc,
                  rkv_enc_packet_t *out,
                  char *err_buf, int err_buf_len) {
    if (!enc) {
        snprintf_err(err_buf, err_buf_len, "rkv_enc_flush: NULL encoder");
        return -EINVAL;
    }
    if (enc->closed || enc->flushed) {
        snprintf_err(err_buf, err_buf_len,
                     "rkv_enc_flush: encoder is %s",
                     enc->closed ? "closed" : "already flushed");
        return -EINVAL;
    }
    enc->flushed = RK_TRUE;

    MPP_RET ret = mpp_frame_init(&enc->frame);
    if (ret != MPP_OK)
        return check(ret, "mpp_frame_init (EOS)", err_buf, err_buf_len);
    mpp_frame_set_eos(enc->frame, 1);
    ret = enc->api->encode_put_frame(enc->ctx, enc->frame);
    mpp_frame_deinit(&enc->frame);
    enc->frame = NULL;
    if (ret != MPP_OK)
        return check(ret, "encode_put_frame (EOS)", err_buf, err_buf_len);

    ret = bind_packet(enc);
    if (ret != MPP_OK)
        return check(ret, "mpp_packet_init_with_buffer (EOS output)",
                     err_buf, err_buf_len);
    ret = enc->api->encode_get_packet(enc->ctx, &enc->packet);
    if (ret != MPP_OK) {
        unbind_packet(enc);
        /* EOS drain can legitimately return nothing. Surface as
         * "no packet available" so the caller can move on. */
        if (out) {
            out->data = NULL; out->length = 0; out->flag = 0; out->pts = 0;
        }
        return 1;
    }
    if (out) {
        out->data   = (RK_U8 *)mpp_packet_get_pos(enc->packet);
        out->length = (RK_U32)mpp_packet_get_length(enc->packet);
        out->flag   = mpp_packet_get_flag(enc->packet);
        out->pts    = mpp_packet_get_pts(enc->packet);
    }
    unbind_packet(enc);
    return 0;
}

int rkv_enc_get_extradata(rkv_encoder_t *enc,
                          const RK_U8 **out_data, RK_U32 *out_size) {
    if (!enc) return -EINVAL;
    if (out_data) *out_data = enc->extradata;
    if (out_size) *out_size = (RK_U32)enc->extradata_size;
    return 0;
}

int rkv_enc_force_keyframe(rkv_encoder_t *enc) {
    if (!enc || enc->closed || enc->flushed) return -EINVAL;
    enc->force_idr = RK_TRUE;
    return 0;
}

void rkv_enc_destroy(rkv_encoder_t *enc) {
    if (!enc) return;
    if (enc->packet_bound && enc->packet) {
        mpp_packet_deinit(&enc->packet);
        enc->packet = NULL;
        enc->packet_bound = RK_FALSE;
    }
    if (enc->meta_bound && enc->meta) {
        mpp_meta_put(enc->meta);
        enc->meta = NULL;
        enc->meta_bound = RK_FALSE;
    }
    if (enc->frm_buf) {
        mpp_buffer_put_with_caller(enc->frm_buf, RKV_TAG_MODULE);
        enc->frm_buf = NULL;
    }
    if (enc->pkt_buf) {
        mpp_buffer_put_with_caller(enc->pkt_buf, RKV_TAG_MODULE);
        enc->pkt_buf = NULL;
    }
    if (enc->buf_group) {
        mpp_buffer_group_put(enc->buf_group);
        enc->buf_group = NULL;
    }
    if (enc->ctx) {
        mpp_destroy(enc->ctx);
        enc->ctx = NULL;
    }
    if (enc->extradata) {
        free(enc->extradata);
        enc->extradata = NULL;
        enc->extradata_size = 0;
    }
    enc->closed = RK_TRUE;
    free(enc);
}