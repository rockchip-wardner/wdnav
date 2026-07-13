# -*- coding: utf-8 -*-
"""冒烟测试：保证 rkv 包能 import，公开符号存在，version 是非空字符串。

VideoCapture / MediaDevice 在非 Rockchip 平台会被 ``__init__`` 的 try/except
设成 ``None``（没有 librockchip_mpp / librga / media-ctl）。本测试不依赖
native 库，也不会尝试实际加载——它在开发机 / CI / Rockchip 目标机上行为一致。
"""
from __future__ import absolute_import, division, print_function

import rkv


def test_version_is_string():
    assert isinstance(rkv.__version__, str)
    assert rkv.__version__


def test_public_api_exports():
    for name in ("__version__", "VideoCapture", "MediaDevice",
                 "MediaEntity", "MediaLink", "MediaPad"):
        assert hasattr(rkv, name), "rkv.{0} missing".format(name)


def test_conditional_symbols_are_either_class_or_none():
    # ``VideoCapture`` 和 ``MediaDevice`` 在 native 缺失 / media-ctl 缺失时
    # 是 None，否则是真类。两者都属预期。
    assert rkv.VideoCapture is None or isinstance(rkv.VideoCapture, type)
    assert rkv.MediaDevice is None or isinstance(rkv.MediaDevice, type)