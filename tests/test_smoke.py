# -*- coding: utf-8 -*-
"""冒烟测试：保证 wdnav 包能 import，公开符号存在，version 是非空字符串。

VideoCapture / MediaDevice / SurfaceFlinger / DmaHeap 在非 Rockchip / 无
libmali / 无 dma-heap 设备的环境上会被 ``__init__`` 的 try/except 设成
``None``。本测试不依赖 native 库，也不会尝试实际加载——它在开发机 / CI
/ Rockchip 目标机上行为一致。
"""
from __future__ import absolute_import, division, print_function

import wdnav


def test_version_is_string():
    assert isinstance(wdnav.__version__, str)
    assert wdnav.__version__


def test_public_api_exports():
    for name in (
        "__version__",
        "VideoCapture",
        "VideoWriter",
        "MediaDevice",
        "MediaEntity",
        "MediaLink",
        "MediaPad",
        "SurfaceFlinger",
        "Surface",
        "DmaHeap",
        "DmaBuf",
        "DmaHeapError",
        "list_heaps",
    ):
        assert hasattr(wdnav, name), "wdnav.{0} missing".format(name)


def test_conditional_symbols_are_either_class_or_none():
    # ``VideoCapture``、``MediaDevice``、``SurfaceFlinger``、``Surface``、
    # ``DmaHeap``、``DmaBuf`` 在 native 缺失时是 None，否则是真类。两者都
    # 属预期。``VideoWriter`` 依赖 cv2，缺 cv2 时同样退化为 None。
    assert wdnav.VideoCapture is None or isinstance(wdnav.VideoCapture, type)
    assert wdnav.VideoWriter is None or isinstance(wdnav.VideoWriter, type)
    assert wdnav.MediaDevice is None or isinstance(wdnav.MediaDevice, type)
    assert wdnav.SurfaceFlinger is None or isinstance(wdnav.SurfaceFlinger, type)
    assert wdnav.Surface is None or isinstance(wdnav.Surface, type)
    assert wdnav.DmaHeap is None or isinstance(wdnav.DmaHeap, type)
    assert wdnav.DmaBuf is None or isinstance(wdnav.DmaBuf, type)


def test_surfaceflinger_module_imports():
    # mali / surfaceflinger / video / dmaheap 模块在纯 import 不创建
    # EGL context / 不分配 DMA-BUF，所以可以无条件 import；这一项确保
    # conditional import 至少触达源码。
    import wdnav.mali  # noqa: F401
    import wdnav.surfaceflinger  # noqa: F401
    import wdnav.video  # noqa: F401
    import wdnav.dmaheap  # noqa: F401


def test_dmaheap_basic():
    """If a dma-heap device is present, do a small alloc to validate the
    ctypes binding. Otherwise skip — the dev machine / CI doesn't have
    one and we don't want to fail there.
    """
    if wdnav.DmaHeap is None:
        return
    heaps = wdnav.list_heaps()
    if not heaps:
        return
    with wdnav.DmaHeap(heaps[0]) as h:
        with h.alloc(4096) as b:
            assert b.fd >= 0
            assert b.size == 4096
