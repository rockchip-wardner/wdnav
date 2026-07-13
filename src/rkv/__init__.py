# -*- coding: utf-8 -*-
"""rkv —— Python 入口。

公开 API 全部从 ``video`` / ``media`` 透传过来；用户 ``import rkv`` 后能直接拿到
``VideoCapture``、``MediaDevice``、版本号等。
"""
from __future__ import absolute_import, division, print_function

from ._version import __version__

# ``VideoCapture`` 拉起时会 import mpp / rga，进而 ctypes 加载 librockchip_mpp
# 和 librga —— 在非 Rockchip 平台会抛 OSError。这里用条件导入兜底，保证
# ``import rkv`` 在开发机上不会崩。
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

__all__ = [
    "__version__",
    "VideoCapture",
    "MediaDevice",
    "MediaEntity",
    "MediaLink",
    "MediaPad",
]