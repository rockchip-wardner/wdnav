# -*- coding: utf-8 -*-
"""冒烟测试：保证 rkv 包能 import 并且 PythonBackend 可以正常工作。

不依赖 .so —— 即使 native 库缺失，这个测试也应当通过。
"""
from __future__ import absolute_import, division, print_function

import rkv
from rkv import (
    Backend,
    KeyNotFound,
    PythonBackend,
    Rkv,
    RkvError,
    make_backend,
)


def test_version_is_string():
    assert isinstance(rkv.__version__, str)
    assert rkv.__version__  # 非空


def test_public_api_exports():
    for name in ("Backend", "KeyNotFound", "PythonBackend", "Rkv",
                 "RkvError", "make_backend", "__version__"):
        assert hasattr(rkv, name), "rkv.{0} missing".format(name)


def test_python_backend_basic():
    b = PythonBackend()
    b.set("a", "1")
    b.set("b", "2")
    assert b.get("a") == "1"
    assert sorted(b.keys()) == ["a", "b"]
    b.delete("a")
    assert "a" not in b.keys()


def test_python_backend_missing_key():
    b = PythonBackend()
    try:
        b.get("nope")
    except KeyNotFound:
        return
    assert False, "expected KeyNotFound"


def test_rkv_context_manager_uses_python_backend():
    # 没有 .so 时会自动 fallback 到 PythonBackend
    db = Rkv.open(prefer="python")
    try:
        db.set("hello", "world")
        assert db.get("hello") == "world"
    finally:
        db.close()


def test_rkv_closed_raises():
    db = Rkv.open(prefer="python")
    db.close()
    try:
        db.set("x", "y")
    except RkvError:
        return
    assert False, "expected RkvError after close"


def test_make_backend_returns_a_backend():
    backend = make_backend(prefer="python")
    assert isinstance(backend, Backend)
