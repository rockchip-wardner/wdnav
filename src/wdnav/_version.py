# -*- coding: utf-8 -*-
"""版本号单一来源。

打包时 setuptools 会读取这里的 __version__，运行时用户也可以
``python -c "import wdnav; print(wdnav.__version__)"`` 直接拿到。
"""
from __future__ import absolute_import, division, print_function

__version__ = "0.5.1"