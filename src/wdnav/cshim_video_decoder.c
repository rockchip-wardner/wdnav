/* cshim_video_decoder.c — higher-level C wrapper around librockchip_mpp's decoder.
 *
 * Companion to cshim_video_encoder.c. Same design principle: hide
 * every MppPacket / MppFrame / MppBuffer handle behind a small,
 * opinionated surface so the Python side never has to call mpp_*
 * directly. The decoder shim manages:
 *
 *   - the MPP context and its API table,
 *   - the scratch MppPacket + its 1 MiB backing buffer used to feed
 *     packets to decode_put_packet (reused across calls),
 *   - the external buffer group that's allocated on the first
 *     info-change frame and reported back via rkv_dec_get_frame,
 *   - the lifecycle of MppFrame handles the caller has handed back.
 *
 * Public API:
 *
 *   rkv_dec_create()         — bring up an MPP decoder context with
 *                              split_parse and friends configured.
 *   rkv_dec_put_packet()     — feed one input packet (already-Annex-B
 *                              H.264 / HEVC bytes from a demuxer).
 *   rkv_dec_get_frame()      — pull the next decoded frame. Returns
 *                              width / height / stride / fmt / pts /
 *                              ptr / fd / buf_size and a set of flags
 *                              that the caller can dispatch on
 *                              (``is_info_change``, ``discard``,
 *                              ``errinfo``).
 *   rkv_dec_release_frame()  — return the most recently decoded
 *                              MppFrame to MPP's pool.
 *   rkv_dec_reset()          — clear decoder state (mpi->reset).
 *   rkv_dec_destroy()        — release everything.
 *
 * Thread safety: same as the encoder shim — single-threaded by
 * design; callers serialise access externally.
 */

#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <rk_mpi.h>
#include <mpp_packet.h>
#include <mpp_frame.h>
#include <mpp_buffer.h>
#include <rk_vdec_cfg.h>

/* Coding-type values the Python layer passes through. AVC and the
 * HEVC id come from rk_type.h's MppCodingType — the upstream enum
 * explicitly assigns HEVC = 0x01000004. */
#define RKV_CT_AVC    7
#define RKV_CT_HEVC   0x01000004
#define RKV_CT_VP9    10
#define RKV_CT_AV1    0x01000008

/* Pixel formats the decoder emits. Most decoders settle on NV12
 * (YUV420SP=0) when MPP_DEC_SET_OUTPUT_FORMAT=0 is asked for; the
 * Python side forces this so callers get a uniform layout. */
#define RKV_FMT_YUV420SP 0
#define RKV_FMT_YUV420P  4
#define RKV_FMT_YUV420SP_VU 5

/* Buffer group type for the decoder's external frame pool. The
 * upstream mpi_dec_test.c uses DMA_HEAP | INTERNAL on RK3588 — the
 * kernel CMA heap backs the allocation with contiguous DMA-able
 * memory. NORMAL would go through allocator_std which recent MPP
 * builds mark simulation-only. */
#define RKV_DEC_BUF_TYPE     MPP_BUFFER_TYPE_DMA_HEAP
#define RKV_DEC_BUF_MODE     MPP_BUFFER_INTERNAL

/* 1 MiB scratch packet buffer — large enough for a typical H.264
 * access unit (Annex B SPS/PPS prepended). Reused across calls;
 * resizes only on grow. */
#define RKV_DEC_PACKET_BUF_MIN (1u << 20)

/* Buffer-full retry budget. The VPU's internal queue saturates
 * occasionally on cold start or when callers feed packets faster
 * than they drain frames. The upstream test sleeps 1 ms between
 * retries; we do the same with exponential backoff capped at 50 ms
 * for up to 5 s total. */
#define RKV_DEC_RETRY_MAX      100
#define RKV_DEC_RETRY_BACKOFF_MS 1

/* Module / function tag strings for MPP's buffer-group debug logs.
 * These are plain (char *) — MPP's tag slots take ``const char *``,
 * and the byte-string ``b"..."`` syntax is C++-only. */
#define RKV_TAG_MODULE  "wdnav-shim"
#define RKV_TAG_PACKET  "wdnav-dec-pkt"
#define RKV_TAG_INFO    "wdnav-dec-info"

struct rkv_decoder {
    MppCtx       ctx;
    MppApi      *api;
    MppBufferGroup buf_group;     /* external pool, allocated on info-change */

    MppPacket    packet;          /* scratch packet */
    RK_U8       *packet_buf;      /* backing memory for self._packet */
    size_t       packet_buf_size; /* current allocation */

    MppFrame     held_frame;      /* last frame handed to caller */
    RK_BOOL      held_valid;

    RK_S32       coding_type;
    RK_S32       split_parse;
    RK_S32       extra_buf_count;
    RK_S32       output_format;

    RK_BOOL      closed;
};

typedef struct rkv_decoder rkv_decoder_t;

/* Forward decl so rkv_dec_create() can roll back via rkv_dec_destroy()
 * on init failure without us having to reorganise the file. */
void rkv_dec_destroy(rkv_decoder_t *dec);

/* Public struct returned to Python. Plain integer fields so ctypes
 * reads it as a Structure without further translation. ``fd`` is -1
 * when the underlying buffer isn't DMA-backed (the upstream
 * mpi_dec_test.c deals with this by branching on the fd being -1). */
typedef struct rkv_dec_frame {
    RK_S32  width;
    RK_S32  height;
    RK_S32  hor_stride;
    RK_S32  ver_stride;
    RK_S32  fmt;
    RK_S64  pts;
    RK_S64  ptr;            /* void* widened to RK_S64 */
    RK_S32  fd;
    RK_U32  buf_size;
    RK_U32  is_info_change;
    RK_U32  discard;
    RK_U32  errinfo;
    RK_U32  eos;
} rkv_dec_frame_t;

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

static MPP_RET check(MPP_RET ret, const char *where,
                     char *err_buf, int err_buf_len) {
    if (ret != MPP_OK) {
        snprintf_err(err_buf, err_buf_len,
                     "MPP call %s failed: ret=%d", where, (int)ret);
    }
    return ret;
}

/* Reallocate the scratch packet buffer to be at least ``size`` bytes
 * and rebind ``dec->packet`` to point at the new backing. */
static MPP_RET ensure_packet_buf(rkv_decoder_t *dec, size_t size,
                                 char *err_buf, int err_buf_len) {
    size_t target = size;
    if (target < RKV_DEC_PACKET_BUF_MIN) target = RKV_DEC_PACKET_BUF_MIN;
    if (dec->packet_buf && dec->packet_buf_size >= target) {
        return MPP_OK;
    }
    /* Tear down the old packet first so its buffer ref is gone. */
    if (dec->packet) {
        mpp_packet_deinit(&dec->packet);
        dec->packet = NULL;
    }
    RK_U8 *buf = (RK_U8 *)malloc(target);
    if (!buf) {
        snprintf_err(err_buf, err_buf_len,
                     "ensure_packet_buf: OOM allocating %zu bytes", target);
        return MPP_ERR_MALLOC;
    }
    MPP_RET ret = mpp_packet_init(&dec->packet, NULL, 0);
    if (ret != MPP_OK) {
        free(buf);
        return check(ret, "mpp_packet_init", err_buf, err_buf_len);
    }
    dec->packet_buf      = buf;
    dec->packet_buf_size = target;
    return MPP_OK;
}

/* Called when ``decode_get_frame`` reports an info-change. Allocate
 * the external buffer group, prime it with ``extra_buf_count``
 * buffers of the reported size, then attach it to the decoder and
 * acknowledge the change so the decoder will start emitting frames.
 *
 * Returns MPP_OK on success; on error, the decoder is left in its
 * current state (caller will surface the error). */
static MPP_RET handle_info_change(rkv_decoder_t *dec, MppFrame frame,
                                  char *err_buf, int err_buf_len) {
    RK_U32 width        = mpp_frame_get_width(frame);
    RK_U32 height       = mpp_frame_get_height(frame);
    RK_U32 hor_stride   = mpp_frame_get_hor_stride(frame);
    RK_U32 ver_stride   = mpp_frame_get_ver_stride(frame);
    size_t buf_size     = mpp_frame_get_buf_size(frame);

    /* Drop any pre-existing group. The MPP spec says callers have to
     * release the previous one before re-attaching. */
    if (dec->buf_group) {
        mpp_buffer_group_put(dec->buf_group);
        dec->buf_group = NULL;
    }
    MppBufferGroup grp = NULL;
    MPP_RET ret = mpp_buffer_group_get(&grp,
                                       RKV_DEC_BUF_TYPE,
                                       RKV_DEC_BUF_MODE,
                                       RKV_TAG_MODULE, RKV_TAG_INFO);
    if (ret != MPP_OK)
        return check(ret, "mpp_buffer_group_get (info-change)",
                     err_buf, err_buf_len);

    /* Limit the group to the reported buffer size, then prime with
     * ``extra_buf_count`` buffers — matches mpi_dec_test.c. */
    ret = mpp_buffer_group_limit_config(grp, buf_size,
                                        (RK_S32)dec->extra_buf_count);
    if (ret != MPP_OK) {
        mpp_buffer_group_put(grp);
        return check(ret, "mpp_buffer_group_limit_config",
                     err_buf, err_buf_len);
    }
    for (RK_S32 i = 0; i < dec->extra_buf_count; i++) {
        MppBuffer buf = NULL;
        ret = mpp_buffer_get_with_tag(grp, &buf, buf_size,
                                      RKV_TAG_MODULE, RKV_TAG_INFO);
        if (ret != MPP_OK) {
            mpp_buffer_group_put(grp);
            return check(ret, "mpp_buffer_get prime", err_buf, err_buf_len);
        }
        mpp_buffer_put_with_caller(buf, RKV_TAG_MODULE);
    }

    ret = dec->api->control(dec->ctx, MPP_DEC_SET_EXT_BUF_GROUP, grp);
    if (ret != MPP_OK) {
        mpp_buffer_group_put(grp);
        return check(ret, "MPP_DEC_SET_EXT_BUF_GROUP", err_buf, err_buf_len);
    }
    dec->buf_group = grp;

    ret = dec->api->control(dec->ctx, MPP_DEC_SET_INFO_CHANGE_READY, NULL);
    if (ret != MPP_OK)
        return check(ret, "MPP_DEC_SET_INFO_CHANGE_READY",
                     err_buf, err_buf_len);
    (void)width; (void)height; (void)hor_stride; (void)ver_stride;
    return MPP_OK;
}

/* ---------------------------------------------------------------------------
 * Public API
 * ------------------------------------------------------------------------- */

int rkv_dec_create(
    int coding_type, int split_parse, int extra_buf_count,
    int output_format,
    char *err_buf, int err_buf_len,
    rkv_decoder_t **out_dec
) {
    if (!out_dec) {
        snprintf_err(err_buf, err_buf_len, "rkv_dec_create: out_dec is NULL");
        return -1;
    }
    *out_dec = NULL;

    rkv_decoder_t *dec = (rkv_decoder_t *)calloc(1, sizeof(rkv_decoder_t));
    if (!dec) {
        snprintf_err(err_buf, err_buf_len, "rkv_dec_create: OOM");
        return -ENOMEM;
    }

    dec->coding_type    = coding_type;
    dec->split_parse    = split_parse ? 1 : 0;
    dec->extra_buf_count = extra_buf_count > 0 ? extra_buf_count : 24;
    dec->output_format  = output_format;

    MppCtx  ctx = NULL;
    MppApi *api = NULL;
    MPP_RET ret = mpp_create(&ctx, &api);
    if (ret != MPP_OK) {
        check(ret, "mpp_create", err_buf, err_buf_len);
        free(dec);
        return (int)ret;
    }
    dec->ctx = ctx;
    dec->api = api;

    ret = mpp_init(ctx, MPP_CTX_DEC, (MppCodingType)coding_type);
    if (ret != MPP_OK) {
        check(ret, "mpp_init(DEC)", err_buf, err_buf_len);
        goto fail;
    }

    /* Configure split_parse. Without this flag the input parser
     * expects frame-aligned packets; demuxers rarely produce those,
     * so turn it on by default. */
    MppDecCfg cfg = NULL;
    ret = mpp_dec_cfg_init(&cfg);
    if (ret != MPP_OK) {
        check(ret, "mpp_dec_cfg_init", err_buf, err_buf_len);
        goto fail;
    }
    ret = dec->api->control(dec->ctx, MPP_DEC_GET_CFG, cfg);
    if (ret == MPP_OK) {
        mpp_dec_cfg_set_u32(cfg, "base:split_parse",
                            (RK_U32)dec->split_parse);
        ret = dec->api->control(dec->ctx, MPP_DEC_SET_CFG, cfg);
    }
    mpp_dec_cfg_deinit(cfg);
    if (ret != MPP_OK) {
        check(ret, "MPP_DEC_SET_CFG", err_buf, err_buf_len);
        goto fail;
    }

    /* Force the decoder's output format. Default MPP behaviour gives
     * back compressed-domain output (FBC / tile) which would force
     * the caller to do a colour conversion; asking for NV12 keeps
     * the host code simple. */
    ret = dec->api->control(dec->ctx, MPP_DEC_SET_OUTPUT_FORMAT,
                            (MppParam)(intptr_t)dec->output_format);
    if (ret != MPP_OK) {
        check(ret, "MPP_DEC_SET_OUTPUT_FORMAT", err_buf, err_buf_len);
        goto fail;
    }

    /* Pre-allocate the scratch packet buffer so the first put_packet
     * doesn't have to malloc under the lock. */
    ret = ensure_packet_buf(dec, RKV_DEC_PACKET_BUF_MIN,
                            err_buf, err_buf_len);
    if (ret != MPP_OK) goto fail;

    *out_dec = dec;
    return 0;

fail:
    rkv_dec_destroy(dec);
    return (int)ret;
}

/* Feed one packet to the decoder. Retries automatically on
 * MPP_ERR_BUFFER_FULL (-1012) so callers don't have to drain frames
 * synchronously. Returns 0 on success, -1012 on retry exhaustion,
 * negative MPP error code otherwise. */
int rkv_dec_put_packet(rkv_decoder_t *dec,
                       const void *data, size_t length,
                       int64_t pts, int eos,
                       char *err_buf, int err_buf_len) {
    if (!dec) {
        snprintf_err(err_buf, err_buf_len, "rkv_dec_put_packet: NULL decoder");
        return -EINVAL;
    }
    if (dec->closed) {
        snprintf_err(err_buf, err_buf_len,
                     "rkv_dec_put_packet: decoder is closed");
        return -EINVAL;
    }
    if (!data && length > 0) {
        snprintf_err(err_buf, err_buf_len,
                     "rkv_dec_put_packet: NULL data with non-zero length");
        return -EINVAL;
    }

    MPP_RET ret = ensure_packet_buf(dec, length,
                                    err_buf, err_buf_len);
    if (ret != MPP_OK) return (int)ret;

    if (length > 0) {
        memcpy(dec->packet_buf, data, length);
        mpp_packet_set_data(dec->packet, dec->packet_buf);
        mpp_packet_set_size(dec->packet, length);
        mpp_packet_set_pos(dec->packet, dec->packet_buf);
        mpp_packet_set_length(dec->packet, length);
    } else {
        mpp_packet_set_data(dec->packet, NULL);
        mpp_packet_set_size(dec->packet, 0);
        mpp_packet_set_pos(dec->packet, NULL);
        mpp_packet_set_length(dec->packet, 0);
    }
    mpp_packet_set_pts(dec->packet, (RK_S64)pts);
    if (eos) {
        ret = mpp_packet_set_eos(dec->packet);
        if (ret != MPP_OK) {
            check(ret, "mpp_packet_set_eos", err_buf, err_buf_len);
            return (int)ret;
        }
    }

    /* Bounded exponential backoff — same shape as mpi_dec_test.c's
     * outer msleep(1) loop. */
    RK_S32 backoff_ms = RKV_DEC_RETRY_BACKOFF_MS;
    RK_S32 attempts   = 0;
    ret = dec->api->decode_put_packet(dec->ctx, dec->packet);
    while (ret == MPP_ERR_BUFFER_FULL && attempts < RKV_DEC_RETRY_MAX) {
        attempts++;
        /* best-effort sleep; we don't have usleep but nanosleep is
         * always available on Linux. */
        struct timespec ts;
        ts.tv_sec  = backoff_ms / 1000;
        ts.tv_nsec = (long)(backoff_ms % 1000) * 1000000L;
        nanosleep(&ts, NULL);
        if (backoff_ms < 50) backoff_ms *= 2;
        ret = dec->api->decode_put_packet(dec->ctx, dec->packet);
    }
    if (ret != MPP_OK) {
        check(ret, "decode_put_packet", err_buf, err_buf_len);
        return (int)ret;
    }
    return 0;
}

/* Pull the next decoded frame. Returns:
 *    1  — a real frame was written into ``out``;
 *    0  — no frame available (caller should retry);
 *   -1  — error (message in err_buf).
 *
 * The frame's MppFrame handle is held internally until the caller
 * calls rkv_dec_release_frame(); the decoded data in the MppBuffer
 * stays valid in the meantime (the buffer group keeps a ref). */
int rkv_dec_get_frame(rkv_decoder_t *dec, int timeout_ms,
                      rkv_dec_frame_t *out,
                      char *err_buf, int err_buf_len) {
    if (!dec || !out) {
        snprintf_err(err_buf, err_buf_len, "rkv_dec_get_frame: NULL args");
        return -1;
    }
    if (dec->closed) {
        snprintf_err(err_buf, err_buf_len,
                     "rkv_dec_get_frame: decoder is closed");
        return -1;
    }
    if (dec->held_valid) {
        snprintf_err(err_buf, err_buf_len,
                     "rkv_dec_get_frame: previous frame not released");
        return -1;
    }

    MppFrame frame = NULL;
    MPP_RET ret = dec->api->decode_get_frame(dec->ctx, &frame);
    (void)timeout_ms; /* the upstream API doesn't honour a timeout
                         parameter; we keep the slot so the Python
                         signature can match without churn. */
    if (ret == MPP_ERR_TIMEOUT) {
        memset(out, 0, sizeof(*out));
        return 0;
    }
    if (ret != MPP_OK) {
        check(ret, "decode_get_frame", err_buf, err_buf_len);
        return (int)ret;
    }
    if (!frame) {
        memset(out, 0, sizeof(*out));
        return 0;
    }

    /* Info-change: hand off to the buffer-group allocator, ack the
     * decoder, then return "no frame yet" to the caller. */
    if (mpp_frame_get_info_change(frame)) {
        ret = handle_info_change(dec, frame, err_buf, err_buf_len);
        mpp_frame_deinit(&frame);
        if (ret != MPP_OK) return (int)ret;
        memset(out, 0, sizeof(*out));
        return 0;
    }

    RK_U32 discard = mpp_frame_get_discard(frame);
    RK_U32 errinfo = mpp_frame_get_errinfo(frame);
    if (discard || errinfo) {
        mpp_frame_deinit(&frame);
        memset(out, 0, sizeof(*out));
        return 0;
    }

    MppBuffer buf = mpp_frame_get_buffer(frame);
    if (!buf) {
        /* Defensive: some internal frames come back with no buffer
         * (e.g. placeholders near EOS). Bail without spamming the
         * log. */
        mpp_frame_deinit(&frame);
        memset(out, 0, sizeof(*out));
        return 0;
    }

    void *ptr = mpp_buffer_get_ptr_with_caller(buf, RKV_TAG_MODULE);
    RK_S32 fd  = mpp_buffer_get_fd_with_caller(buf, RKV_TAG_MODULE);
    if (fd < 0) fd = -1;

    out->width        = (RK_S32)mpp_frame_get_width(frame);
    out->height       = (RK_S32)mpp_frame_get_height(frame);
    out->hor_stride   = (RK_S32)mpp_frame_get_hor_stride(frame);
    out->ver_stride   = (RK_S32)mpp_frame_get_ver_stride(frame);
    out->fmt          = (RK_S32)mpp_frame_get_fmt(frame);
    out->pts          = mpp_frame_get_pts(frame);
    out->ptr          = (RK_S64)(intptr_t)ptr;
    out->fd           = fd;
    out->buf_size     = (RK_U32)mpp_frame_get_buf_size(frame);
    out->is_info_change = 0;
    out->discard        = 0;
    out->errinfo        = 0;
    out->eos            = mpp_frame_get_eos(frame);

    /* Hang on to the frame so rkv_dec_release_frame can drop the
     * MppFrame ref. ``ptr`` / ``fd`` keep pointing at valid memory
     * because the MppBuffer's group ref outlives the MppFrame. */
    dec->held_frame = frame;
    dec->held_valid = RK_TRUE;
    return 1;
}

int rkv_dec_release_frame(rkv_decoder_t *dec,
                          char *err_buf, int err_buf_len) {
    if (!dec) {
        snprintf_err(err_buf, err_buf_len,
                     "rkv_dec_release_frame: NULL decoder");
        return -EINVAL;
    }
    if (!dec->held_valid) {
        /* Idempotent — releasing without a held frame is a no-op.
         * Matches mpp.py's historical behaviour. */
        return 0;
    }
    mpp_frame_deinit(&dec->held_frame);
    dec->held_frame = NULL;
    dec->held_valid = RK_FALSE;
    return 0;
}

int rkv_dec_reset(rkv_decoder_t *dec,
                  char *err_buf, int err_buf_len) {
    if (!dec) {
        snprintf_err(err_buf, err_buf_len, "rkv_dec_reset: NULL decoder");
        return -EINVAL;
    }
    if (dec->closed) {
        snprintf_err(err_buf, err_buf_len,
                     "rkv_dec_reset: decoder is closed");
        return -EINVAL;
    }
    MPP_RET ret = dec->api->reset(dec->ctx);
    return check(ret, "mpi->reset", err_buf, err_buf_len);
}

void rkv_dec_destroy(rkv_decoder_t *dec) {
    if (!dec) return;
    if (dec->held_valid && dec->held_frame) {
        mpp_frame_deinit(&dec->held_frame);
        dec->held_frame = NULL;
        dec->held_valid = RK_FALSE;
    }
    if (dec->packet) {
        mpp_packet_deinit(&dec->packet);
        dec->packet = NULL;
    }
    if (dec->packet_buf) {
        free(dec->packet_buf);
        dec->packet_buf = NULL;
        dec->packet_buf_size = 0;
    }
    if (dec->buf_group) {
        mpp_buffer_group_put(dec->buf_group);
        dec->buf_group = NULL;
    }
    if (dec->ctx) {
        mpp_destroy(dec->ctx);
        dec->ctx = NULL;
    }
    dec->closed = RK_TRUE;
    free(dec);
}