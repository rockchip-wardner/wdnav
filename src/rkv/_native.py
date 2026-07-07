# -*- coding: utf-8 -*-
"""加载 native/ 下的本地共享库。

把 native/librkv.so 拷贝/安装到 src/rkv/_native/librkv.so 之后，
本模块通过 ctypes 把它们加载进来。优先尝试已经打包进 wheel 的
``rkv._native.librkv``；找不到时回退到开发目录下的 native/。
"""
from __future__ import absolute_import, division, print_function

import ctypes
import os
import sys


_LIB_CANDIDATES = (
    # 安装后的 wheel：rkv/_native/librkv.so
    "librkv.so",
    "librkv.so.1",
    # macOS（按需）
    "librkv.dylib",
    # Windows（按需）
    "librkv.dll",
)


def _candidate_paths():
    """收集可能的 .so 路径：先 wheel 内、再源码 native/。"""
    paths = []

    # 1) rkv/_native/（wheel 安装后的位置）
    here = os.path.dirname(os.path.abspath(__file__))
    bundled = os.path.join(here, "_native")
    for name in _LIB_CANDIDATES:
        paths.append(os.path.join(bundled, name))

    # 2) 源码 native/（开发模式 + editable install）
    #    src/rkv/_native.py -> ../../../native/librkv.so
    repo_native = os.path.normpath(
        os.path.join(here, "..", "..", "native")
    )
    for name in _LIB_CANDIDATES:
        paths.append(os.path.join(repo_native, name))

    return paths


def load_library():
    """加载并返回 ctypes 库对象；找不到时抛 RuntimeError。"""
    last_err = None
    for path in _candidate_paths():
        if os.path.exists(path):
            try:
                return ctypes.CDLL(path)
            except OSError as exc:
                last_err = exc
                continue
    msg = "librkv native library not found. Tried:\n  - "
    msg += "\n  - ".join(_candidate_paths())
    if last_err is not None:
        msg += "\nLast error: {0}".format(last_err)
    raise RuntimeError(msg)


# 懒加载：真正用到时才打开 .so，避免 import rkv 时就因缺库而崩溃
_native = None


def get_native():
    """获取已加载的 native 库，第一次调用时初始化。"""
    global _native
    if _native is None:
        _native = load_library()
    return _native


def is_available():
    """检查 native 库是否可用，不抛异常。"""
    try:
        load_library()
        return True
    except Exception:
        return False
