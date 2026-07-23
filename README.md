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

**PyAV 版本要求（重要）**

`rkv.encoder.VideoWriter` 强依赖 PyAV 主分支合并的 [PR #2305（commit `5b48f2d`，"Extract codec extradata in add_mux_stream for Matroska"）](https://github.com/PyAV-Org/PyAV/pull/2305)。该 PR 引入了 `Container.add_mux_stream()` 和 matroska muxer 自动跑的 `extract_extradata` BSF，用于从 MPP 预编码 packet 抽取 SPS/PPS 写到 avcC。

PyPI 上最新 wheel `av==18.0.0` **不包含此修复**，装上后 `VideoWriter` 写出的 MP4 / MKV 默认会被容器填 1280×720 High profile 的 avcC，decoder 拒绝解码（无任何报错，但视频解码 0 帧）。

`pyproject.toml` 里把 Linux 下的 `av` 依赖下界标到 `11.0.dev0` 来强制要求含此修复的版本（实际上 PyPI 上没有 ≥19 wheel，pip 会回退到 18.0.0；因此**必须显式装主分支**）：

```bash
# 先装主分支的 PyAV（Linux 唯一可行路径）
pip install av @ https://github.com/PyAV-Org/PyAV/archive/main.tar.gz

# 再装 rkv 本身
pip install -e .
```

主分支 PyAV 编译依赖系统 FFmpeg 头文件（`libavcodec-dev`、`libavformat-dev`、`libavdevice-dev`、`libavfilter-dev`、`libavutil-dev`、`libswscale-dev`、`libswresample-dev`），FFmpeg ≥ 6.0。Debian sid arm64 上有现成 deb 包：

```bash
# 一次性拉齐所有 dev 依赖（也可以用 apt 安装）
for f in ffmpeg libavcodec-dev libavdevice-dev libavfilter-dev libavformat-dev libavutil-dev libpostproc-dev libswscale-dev libswresample-dev; do
  apt-get download $f
done
```

非 Linux 平台维持旧的 `av>=10.0` 约束（VideoCapture 路径不依赖 PR #2305）。

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
