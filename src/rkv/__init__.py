# -*- coding: utf-8 -*-
"""rkv —— Python 入口。

公开 API 全部从 ``core`` / ``video`` 透传过来；用户 ``import rkv`` 后能直接拿到
``Rkv``、``VideoCapture``、异常类、版本号等。
"""
from __future__ import absolute_import, division, print_function

from ._version import __version__
from .core import (
    Backend,
    KeyNotFound,
    NativeBackend,
    PythonBackend,
    Rkv,
    RkvError,
    make_backend,
)

# ``VideoCapture`` 拉起时会 import mpp / rga，进而 ctypes 加载 librockchip_mpp
# 和 librga —— 在非 Rockchip 平台会抛 OSError。这里用条件导入兜底，确保 KV 路径
# 在任何平台都能 ``import rkv`` 成功。
try:
    from .video import VideoCapture
except Exception:  # pragma: no cover - platform-specific load failure
    VideoCapture = None

__all__ = [
    "__version__",
    "Backend",
    "KeyNotFound",
    "NativeBackend",
    "PythonBackend",
    "Rkv",
    "RkvError",
    "make_backend",
    "VideoCapture",
]