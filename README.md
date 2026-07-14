# pyrkv

> 纯 Python 实现的 rockchip 硬件编解码和硬件2D加速器封装包，封装了libmpp和librga的python ctypes实现，
> 使用pyav和python-v4l2来实现前端协议和文件格式解析以及demux，后端对接vpu与rga。
> `VideoCapture.read()` 返回与 OpenCV 兼容的 `(H, W, 3)` ndarray，可直接交给 PIL /
> OpenCV / numpy 等下游算法使用；像素格式转换与缩放全部走 RGA2，库本体不再依赖 opencv-python。

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
cap = rkv.VideoCapture(path)  # path 可以是本地文件、网络流或 /dev/videoX
ret, frame = cap.read()  # 与 cv2 用法一致；BGR (H, W, 3) ndarray
ret, frame = cap.read(w, h, fmt)  # 这是 cv2 没有的接口，使用 RGA 进行指定分辨率/像素格式的转换

# 其他详细示例见 samples/ 目录
```

`samples/*.py` 默认使用 cv2 做显示；要跑无显示版本，设 `DISPLAY=0`，或在没装
`opencv-python` 的环境直接跑——脚本会自动降级为每 30 帧打印一次的 headless 循环。

## 兼容性

- 目标兼容 Python 3.10+（`requires-python = ">=3.10"`）
- 平台：当前主要面向 Linux（.so）

## 开发

```bash
# 运行测试
pip install -e ".[dev]"
pytest

# 发布到 PyPI（需先配置凭据）
twine upload dist/*
```
