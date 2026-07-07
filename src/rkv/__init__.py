# -*- coding: utf-8 -*-
"""rkv —— Python 入口。

公开 API 全部从 ``core`` 透传过来；用户 ``import rkv`` 后能直接拿到
``Rkv``、异常类、版本号等。
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

__all__ = [
    "__version__",
    "Backend",
    "KeyNotFound",
    "NativeBackend",
    "PythonBackend",
    "Rkv",
    "RkvError",
    "make_backend",
]
