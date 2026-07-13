# -*- coding: utf-8 -*-
"""rkv 的纯 Python 主逻辑。

这里实现一个最小可用的 key-value store 抽象，对上层 API 屏蔽
``native`` 库是否存在 —— 没 .so 时退化为纯 Python 后端。
"""
from __future__ import absolute_import, division, print_function

from . import _native


class RkvError(Exception):
    """所有 rkv 异常的基类。"""


class KeyNotFound(RkvError, KeyError):
    """请求的 key 不存在。"""


class Backend(object):
    """后端接口基类。具体实现见 :class:`PythonBackend` / :class:`NativeBackend`。"""

    def get(self, key):
        raise NotImplementedError

    def set(self, key, value):
        raise NotImplementedError

    def delete(self, key):
        raise NotImplementedError

    def keys(self):
        raise NotImplementedError

    def close(self):
        pass


class PythonBackend(Backend):
    """纯 Python 字典后端，作为 native 不可用时的 fallback。"""

    def __init__(self):
        self._data = {}

    def get(self, key):
        if key not in self._data:
            raise KeyNotFound(key)
        return self._data[key]

    def set(self, key, value):
        self._data[key] = value

    def delete(self, key):
        if key not in self._data:
            raise KeyNotFound(key)
        del self._data[key]

    def keys(self):
        return list(self._data.keys())


class NativeBackend(Backend):
    """通过 ctypes 调用系统安装的 librkv.so（必须由用户在系统层面装好）。"""

    def __init__(self):
        self._lib = _native.get_native()
        # 这里按你的 native 库签名来声明函数原型。
        # 下面是占位示例，按需调整。
        try:
            self._lib.rkv_get.restype = ctypes.c_char_p
            self._lib.rkv_get.argtypes = [ctypes.c_char_p]

            self._lib.rkv_set.restype = ctypes.c_int
            self._lib.rkv_set.argtypes = [ctypes.c_char_p, ctypes.c_char_p]

            self._lib.rkv_delete.restype = ctypes.c_int
            self._lib.rkv_delete.argtypes = [ctypes.c_char_p]
        except AttributeError as exc:
            raise RkvError(
                "native library does not export expected symbols: {0}".format(exc)
            )

    def get(self, key):
        result = self._lib.rkv_get(key.encode("utf-8"))
        if result is None:
            raise KeyNotFound(key)
        return result.decode("utf-8")

    def set(self, key, value):
        rc = self._lib.rkv_set(key.encode("utf-8"), value.encode("utf-8"))
        if rc != 0:
            raise RkvError("rkv_set failed with rc={0}".format(rc))

    def delete(self, key):
        rc = self._lib.rkv_delete(key.encode("utf-8"))
        if rc != 0:
            raise KeyNotFound(key)

    def keys(self):
        # native 库如果支持迭代，这里调用相应函数；否则由上层 dict 缓存兜底
        raise NotImplementedError("native backend does not support keys() yet")


def make_backend(prefer="native"):
    """根据可用性选择后端。``prefer`` 为 ``"native"`` 或 ``"python"``。"""
    if prefer == "native" and _native.is_available():
        try:
            return NativeBackend()
        except Exception:
            # native 加载成功但符号不匹配，降级
            pass
    return PythonBackend()


class Rkv(object):
    """用户面 KV 客户端。

    用法::

        with Rkv.open() as db:
            db.set("k", "v")
            assert db.get("k") == "v"
    """

    def __init__(self, backend=None):
        self._backend = backend if backend is not None else make_backend()
        self._closed = False

    @classmethod
    def open(cls, prefer="native"):
        return cls(backend=make_backend(prefer=prefer))

    # ---- KV 操作 -------------------------------------------------------
    def get(self, key):
        self._check_open()
        return self._backend.get(key)

    def set(self, key, value):
        self._check_open()
        self._backend.set(key, value)

    def delete(self, key):
        self._check_open()
        self._backend.delete(key)

    def keys(self):
        self._check_open()
        return self._backend.keys()

    # ---- 上下文管理 ---------------------------------------------------
    def close(self):
        if not self._closed:
            self._backend.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def _check_open(self):
        if self._closed:
            raise RkvError("Rkv client is closed")