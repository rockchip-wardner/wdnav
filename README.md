# pyrkv

> 纯 Python 实现的 rockchip 硬件编解码和硬件2D加速器封装包，封装了libmpp和librga的python ctypes实现，
> 使用pyav和python-v4l2来实现前端协议和文件格式解析以及demux，后端对接vpu与rga，兼容cv2接口，可直接使用cv2做后续算法与显示。

## 项目结构

```
pyrkv/
├── pyproject.toml          # 构建配置（setuptools）
├── MANIFEST.in             # sdist 打包规则
├── README.md
├── LICENSE
├── src/
│   └── rkv/                # 实际可被 import 的包
│       ├── __init__.py     # 公开 API
│       ├── _version.py
│       ├── mpp.py          # librockchip_mpp.so 系统加载器
│       ├── rga.py          # librga.so 系统加载器
│       ├── video.py        # 视频解码上层（依赖 mpp + rga）
│       ├── media.py        # media-ctl 解析（依赖 v4l2）
│       ├── v4l2.py         # v4l2 helpers
│       ├── rkv.pyi         # 类型 stub
│       └── py.typed        # PEP 561 类型标记
├── samples/                # 用法示例
│   └── README.md
├── tests/                  # pytest 测试
│   └── test_smoke.py
└── docs/
```

> **native 库不在仓库里** —— `librockchip_mpp.so.0` / `librga.so.2`
> 必须由用户在系统层面安装（apt / dnf / 预编译 deb）。`rkv` 通过
> `ctypes.util.find_library` + `dlopen` 在系统路径里查找它们；
> 找不到时分别抛 `RuntimeError` / `OSError`，由 `__init__.py` 的 try/except
> 兜底，把 `VideoCapture` / `MediaDevice` 设成 `None`，保证 `import rkv`
> 在开发机上不会崩。

## 安装

开发模式安装（可编辑）：

```bash
pip install -e .
```

构建 wheel：

```bash
pip install build
python -m build
# 产物在 dist/ 目录下
```

## 使用

```python
import rkv
cap = rkv.VideoCapture(path)  # 这里path可以是本地文件也可以是网络流，也可以是v4l2设备
ret, frame = cap.read()  # 读取一帧，与cv2的用法是一致的，返回的格式也是一致的
ret, frame = cap.read(w, h, fmt)  # 这是cv2没有的接口，使用RGA对输出的格式分辨率进行的指定转换 

# 其他详细示例见 samples/ 目录
```

## 兼容性

- 目标兼容 Python 3.x 全系列（`requires-python = ">=3.0"`）
- 平台：当前主要面向 Linux（.so）

## 开发

```bash
# 运行测试
pip install -e ".[dev]"
pytest

# 发布到 PyPI（需先配置凭据）
twine upload dist/*
```
