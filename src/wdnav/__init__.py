"""wdnav —— Python 入口。

公开 API 全部从 ``mali`` / ``SurfaceComposer`` / ``video`` / ``media`` /
``dmaheap`` 透传过来；用户 ``import wdnav`` 后能直接拿到 ``VideoCapture``、
``MediaDevice``、``SurfaceComposer``、``Surface``、``DmaHeap``、
``WdnavDmaBuf``、版本号等。
"""
from __future__ import absolute_import, division, print_function

from ._version import __version__

# ``VideoCapture`` 拉起时会 import mpp / rga，进而 ctypes 加载 librockchip_mpp
# 和 librga —— 在非 Rockchip 平台会抛 OSError。这里用条件导入兜底，保证
# ``import wdnav`` 在开发机上不会崩。
try:
    from .video import VideoCapture
except Exception:  # pragma: no cover - platform-specific load failure
    VideoCapture = None

# ``MediaDevice`` 需要 ``media-ctl`` 在 PATH 上；缺工具时同样设为 None。
try:
    from .media import (
        MediaDevice,
        MediaEntity,
        MediaLink,
        MediaPad,
    )
except Exception:  # pragma: no cover - missing media-ctl or parse failure
    MediaDevice = None  # type: ignore[assignment]
    MediaEntity = None  # type: ignore[assignment]
    MediaLink = None    # type: ignore[assignment]
    MediaPad = None     # type: ignore[assignment]

# ``SurfaceComposer`` 需要 ``libmali.so`` 和一个可用的 EGL display。
# 在没装 Mali 驱动或没有可绑定 display 的环境上，相应的 import 会抛
# ``OSError`` 或 ``RuntimeError``；这里同样兜底为 None，保持
# ``import wdnav`` 在开发机上不崩。
try:
    from .composer import SurfaceComposer, Surface, Typer
except Exception:  # pragma: no cover - libmali.so missing or EGL init failed
    SurfaceComposer = None  # type: ignore[assignment]
    Surface = None   # type: ignore[assignment]
    Typer = None     # type: ignore[assignment]

# ``DmaHeap`` 需要内核启用 ``CONFIG_DMABUF_HEAPS``，并且至少有
# ``/dev/dma_heap/<name>`` 节点；没有 dma-heap 设备的开发机 / 容器环境
# 上同样置为 None。``WdnavDmaBuf`` 走 rbs glengine 同款的 Rockchip GEM ioctl
# 路径 (参见 ``/userdata/rbs/core/buffers/allocator_drm.cpp``); 它的 import
# 不依赖 dma-heap 节点存在 — ``WdnavDmaBuf().alloc()`` 在老内核失败时抛
# ``WdnavDmaBufError``, 而不是模块 import 时崩。
try:
    from .dmaheap import (
        DmaBuf,
        DmaHeap,
        DmaHeapError,
        WdnavDmaBuf,
        WdnavDmaBufError,
        list_heaps,
    )
except Exception:  # pragma: no cover - /dev/dma_heap not present
    DmaHeap = None        # type: ignore[assignment]
    DmaBuf = None         # type: ignore[assignment]
    DmaHeapError = None   # type: ignore[assignment]
    WdnavDmaBuf = None       # type: ignore[assignment]
    WdnavDmaBufError = None  # type: ignore[assignment]
    list_heaps = None     # type: ignore[assignment]

# ``VideoWriter`` uses ``av`` (PyAV) for muxing and MPP for
# encoding — every codec (H.264 / HEVC / MJPEG / JPEG) goes
# through the Rockchip VPU, no software fallback. See
# :mod:`wdnav.encoder` for the full story.
# Imported lazily so a missing PyAV on a dev box surfaces as a
# clear ``VideoWriter = None`` rather than an ``import wdnav`` failure.
try:
    from .encoder import VideoWriter
except Exception:  # pragma: no cover - PyAV missing or partial install
    VideoWriter = None  # type: ignore[assignment]

__all__ = [
    "__version__",
    "VideoCapture",
    "VideoWriter",
    "MediaDevice",
    "MediaEntity",
    "MediaLink",
    "MediaPad",
    "SurfaceComposer",
    "Surface",
    "Typer",
    "DmaHeap",
    "DmaBuf",
    "DmaHeapError",
    "WdnavDmaBuf",
    "WdnavDmaBufError",
    "list_heaps",
]
