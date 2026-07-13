# -*- coding: utf-8 -*-
"""加载 librkv.so（KV 子系统的 native 后端）。

本模块**不**随包发布 ``librkv.so``；调用方必须在系统层面安装它
（apt / dnf / 预编译 deb 等），并保证 ldconfig 能找到。本模块只通过
``ctypes.util.find_library`` + ``dlopen`` 在系统路径里查找；找不到
就直接抛 :class:`RuntimeError`，由 ``make_backend()`` 退回到
``PythonBackend``，由 ``Rkv`` / ``NativeBackend`` 在实际构造时再
把错误信息交给用户。
"""
from __future__ import absolute_import, division, print_function

import ctypes
import ctypes.util


def load_library():
    """通过 ldconfig 加载 librkv.so；找不到时抛 RuntimeError。"""
    # find_library 会按 ldconfig / soname 规则做匹配。版本化的 soname
    # 先尝试（ldconfig 可能只注册了 librkv.so.1），回落到无版本号。
    for soname in ("librkv.so.1", "librkv.so", "librkv"):
        path = ctypes.util.find_library(soname)
        if path:
            try:
                return ctypes.CDLL(path)
            except OSError:
                continue
    # 最后一搏：让 dlopen 直接拿短名，依靠 LD_LIBRARY_PATH / /etc/ld.so.cache
    try:
        return ctypes.CDLL("rkv")
    except OSError as exc:
        raise RuntimeError(
            "librkv native library not found via ldconfig. "
            "Install librkv system-wide (apt install ...) and run "
            "`ldconfig`, or set LD_LIBRARY_PATH to the directory containing "
            "librkv.so. Tried sonames: librkv.so.1, librkv.so, librkv. "
            "Last error: " + repr(exc)
        )


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
