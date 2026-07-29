# -*- coding: utf-8 -*-
"""类型 stub —— 给 IDE / mypy 看，运行时不会被 import。"""

from typing import Any, Optional

import numpy as np

__version__: str


class DecodedFrame:
    """A raw decoded frame returned by :meth:`VideoCapture.read0`.

    Holds the DMA-BUF fd (or virtual address) of the underlying MPP /
    V4L2 buffer; the actual pixel data sits in kernel-side memory and
    only needs to be touched when :meth:`numpy` materialises it into a
    contiguous ndarray via RGA2.
    """

    width: int
    height: int
    hor_stride: int
    ver_stride: int
    fmt: int
    pts: int
    ptr: int
    fd: int
    buf_size: int
    raw: Any
    out_w: Optional[int]
    out_h: Optional[int]
    out_fmt: Optional[str]
    def numpy(
        self,
        w: Optional[int] = ...,
        h: Optional[int] = ...,
        fmt: Optional[str] = ...,
    ) -> np.ndarray: ...
    def crop(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
    ) -> "DecodedFrame": ...


class EncodedPacket:
    """A single encoded access unit returned by ``MppEncoder.encode``."""

    data: bytes
    pts: int
    is_keyframe: bool


class MppEncoder:
    def __init__(
        self,
        coding_type: int = ...,
        *,
        width: int = ...,
        height: int = ...,
        fps: int = ...,
        bitrate: int = ...,
        gop: int = ...,
        profile: str = ...,
        input_format: int = ...,
        rc_mode: str = ...,
    ) -> None: ...
    width: int
    height: int
    fps: int
    bitrate: int
    gop: int
    profile: str
    rc_mode: str
    extradata: bytes
    def force_keyframe(self) -> None: ...
    def encode(self, frame_bgr: np.ndarray) -> EncodedPacket: ...
    def encode_external_fd(
        self,
        fd: int,
        width: int,
        height: int,
        hor_stride: int,
        ver_stride: int,
        input_format: int = ...,
        pts: int = ...,
    ) -> EncodedPacket: ...
    def flush(self) -> Optional[EncodedPacket]: ...
    def close(self) -> None: ...


class VideoCapture:
    def __init__(
        self,
        path: str,
        timeout: Optional[float] = ...,
        output_size: Optional[tuple[int, int]] = ...,
        *,
        v4l2_input_format: Optional[str] = ...,
        v4l2_size: Optional[tuple[int, int]] = ...,
        v4l2_framerate: Optional[int] = ...,
    ) -> None: ...
    width: int
    height: int
    fps: float
    frame_pos: int
    def isOpened(self) -> bool: ...
    def read(
        self,
        w: Optional[int] = ...,
        h: Optional[int] = ...,
        fmt: Optional[str] = ...,
    ) -> tuple[bool, Optional[np.ndarray]]: ...
    def read0(
        self,
        w: Optional[int] = ...,
        h: Optional[int] = ...,
        fmt: Optional[str] = ...,
    ) -> tuple[bool, Optional[DecodedFrame]]: ...
    def release(self) -> None: ...
    def __enter__(self) -> "VideoCapture": ...
    def __exit__(self, *exc) -> None: ...


class VideoWriter:
    CODEC_H264: str
    CODEC_H264_RAW: str
    CODEC_HEVC_RAW: str
    CODEC_MJPEG: str
    CODEC_JPEG: str

    PROFILE_BASELINE: str
    PROFILE_MAIN: str
    PROFILE_HIGH: str

    RC_VBR: str
    RC_CBR: str
    RC_FIXQP: str
    RC_AVBR: str

    width: int
    height: int
    fps: int
    bitrate: int
    gop: int
    codec: str
    profile: str
    isOpened: bool

    def __init__(
        self,
        path: str,
        fps: int = ...,
        frame_size: tuple[int, int] = ...,
        *,
        codec: str | None = ...,
        profile: str = ...,
        bitrate: int = ...,
        gop: int = ...,
        bframes: int = ...,
        rc_mode: str = ...,
        q_factor: int = ...,
    ) -> None: ...
    def write(self, frame: np.ndarray | "DecodedFrame") -> None: ...
    backend: str
    def release(self) -> None: ...
    def __enter__(self) -> "VideoWriter": ...
    def __exit__(self, *exc) -> None: ...


class SurfaceFlinger:
    def __init__(
        self,
        w: int,
        h: int,
        *,
        egl_display: Optional[int] = ...,
    ) -> None: ...
    w: int
    h: int
    def create_surface(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        *,
        zpos: int = ...,
        fmt: str = ...,
    ) -> "Surface": ...
    def read(self, *, timeout: Optional[float] = ...) -> np.ndarray: ...
    def close(self) -> None: ...
    def __enter__(self) -> "SurfaceFlinger": ...
    def __exit__(self, *exc) -> None: ...


class Surface:
    id: int
    x: int
    y: int
    w: int
    h: int
    zpos: int
    fmt: str
    def post_fd(
        self, fd: int, pts: int = ..., *, stride: Optional[int] = ...,
    ) -> None: ...
    def post(self, pts: int = ...) -> None: ...
    @property
    def latest_ndarray(self) -> np.ndarray: ...


class DmaHeap:
    name: str
    path: str
    def __init__(self, name: str = ...) -> None: ...
    def alloc(self, size: int, fd_flags: int = ..., heap_flags: int = ...) -> "DmaBuf": ...
    def close(self) -> None: ...
    def __enter__(self) -> "DmaHeap": ...
    def __exit__(self, *exc) -> None: ...


class DmaBuf:
    size: int
    @property
    def fd(self) -> int: ...
    def fileno(self) -> int: ...
    def mmap(self, length: Optional[int] = ...) -> int: ...
    def sync_read(self) -> None: ...
    def sync_write(self) -> None: ...
    def close(self) -> None: ...
    def __enter__(self) -> "DmaBuf": ...
    def __exit__(self, *exc) -> None: ...


class DmaHeapError(RuntimeError): ...


class RkDmaBuf:
    handle: int
    card_path: str
    def __init__(self, card_path: str = ...) -> None: ...
    def alloc(self, size: int) -> "DmaBuf": ...
    def alloc_contig(self, size: int) -> "DmaBuf": ...
    def alloc_wc(self, size: int) -> "DmaBuf": ...
    def alloc_cacheable(self, size: int) -> "DmaBuf": ...
    def close(self) -> None: ...
    def __enter__(self) -> "RkDmaBuf": ...
    def __exit__(self, *exc) -> None: ...


class RkDmaBufError(DmaHeapError): ...


def list_heaps() -> list[str]: ...


class MediaDevice: ...
class MediaEntity: ...
class MediaLink: ...
class MediaPad: ...
