# pyrkv

> 纯 Python 实现的 rockchip 硬件编解码和硬件2D加速器封装包，封装了libmpp、librga、libmali的python ctypes实现，
> 使用pyav和python-v4l2来实现前端协议、握手和文件格式解析以及demux，后端对接vpu与rga进行编解码。


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

## 外部so库的依赖

**native 库不在仓库里** —— librockchip_mpp.so.0,librga.so.2,libmali.so等
必须由用户在系统层面安装（apt / dnf / 预编译 deb），一般放在/usr/lib/等系统目录下。
`rkv` 通过 `ctypes.util.find_library` + `dlopen` 在系统路径里查找它们；

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

### 硬件解码
```python
import rkv
cap = rkv.VideoCapture(path)  # path 可以是本地文件、网络流或V4l2设备
ret, frame = cap.read()  # 与 cv2 用法一致；BGR (H, W, 3) ndarray
ret, frame = cap.read(w, h, fmt)  # 这是 cv2 没有的接口，使用 RGA 进行指定分辨率/像素格式的转换
ret, frame0 = cap.read0(...) # 这是0拷贝接口，获取到的是dmabuffer的fd等内容，可以对接传递给
```

### 硬件编码
```python
import rkv
out = rkv.VideoWriter(path, fourcc, fps, (width, height), gop, ...)  # path 可以是本地文件、网络流、websocket等
out.write(frame)  # 兼容cv2的numpy格式frame，一般来说格式是BGR
out.write(frame0)  # 0拷贝接口，传入dmabuffer结构体
```

* 其他详细示例见 samples/ 目录
`samples/*.py` 直接 `import cv2` 做窗口显示，跑前 `pip install opencv-python`。

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
