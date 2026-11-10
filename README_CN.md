[English](README.md) | [简体中文](README_CN.md)

# wdnav

> Rockchip 硬件编解码和图形加速器 python 封装包

## 项目设计

* 兼容cv的使用习惯，但又兼顾了0拷贝实现
* 封装rga/gpu/vpu硬件加速单元
* 可用于绝大多数Rockchip芯片

## 项目依赖

### 外部 so 库的依赖

**native 库不在仓库里** —— `librockchip_mpp.so.0`、`librga.so.2`、`libmali.so`
等必须由用户在系统层面安装（apt / dnf / 预编译 deb），一般放在 `/usr/lib/` 等系统
目录下。`wdnav` 通过 `ctypes.util.find_library` + `dlopen` 在系统路径里查找它们，
或在构建期链进 C shim。

| 库 | 用于 | Debian / Ubuntu 包 |
|---|---|---|
| `librockchip_mpp.so.0` | `VideoCapture` / `VideoWriter` 硬件解码 + 编码 | `librockchip-mpp1` |
| `librga.so.2` | `VideoCapture.read(w, h)` 里的 RGA2 色彩转换 / 缩放 | `librga2` |
| `libmali.so`（`libmali-valhall-g610-g24p0-x11-wayland-gbm`） | `SurfaceComposer` / `Typer` 离屏 GLES —— 由 `libshim_mali.so` 的 DT_NEEDED 拉入 | `libmali-valhall-g610-g24p0-x11-wayland-gbm` |
| `libfreetype.so.6` | `Typer`（Surface 子类）文字栅格化 —— c-shim 会调 `FT_Init_FreeType` 等（见 `rkm_text_*`） | `libfreetype-dev`（同时提供 `libfreetype.so.6` 与 `/usr/include/freetype2/{ft2build.h,freetype/}`） |
| `libfontconfig` + `fc-match` 命令行 | `Typer(font=...)` 的字体族 / pattern 解析（如 `"sans"`、`"wqy-zenhei"`、`"DejaVu Serif"`） | `fontconfig` |
| `libv4l2.so.0` | `MediaDevice` / `v4l2.py` 辅助函数（仅当你要打开 `/dev/video*`） | `libv4l2-dev` |

`libmali.so` 与 `libfreetype.so.6` 在构建期链进 `libshim_mali.so`
（`-lmali -lfreetype`）；加载器通过系统路径或 `.so` 里记录的 `-Wl,-rpath` 解析它们。
构建期必须带头文件 —— `libfreetype-dev` 提供 `/usr/include/freetype2/ft2build.h`，
Khronos EGL/GLES 头文件（`libecg-dev`、`libgles2-mesa-dev`）提供 `/usr/include/EGL/`、
`/usr/include/GLES3/`。

Typer demo 里的中文字体（`"wqy-zenhei"`）需装 `fonts-wqy-zenhei`。

### 外部 whl 的依赖

* **PyAV** —— 用于音视频 demux / mux。demux 之后把裸流交给 MPP 做硬件编解码。

PyAV 编译依赖系统 FFmpeg 头文件（`libavcodec-dev`、`libavformat-dev`、
`libavdevice-dev`、`libavfilter-dev`、`libavutil-dev`、`libswscale-dev`、
`libswresample-dev`），FFmpeg ≥ 6.0。Debian sid arm64 上有现成 deb 包：

```bash
# 一次性拉齐所有 dev 依赖（也可以用 apt 安装）
for f in ffmpeg libavcodec-dev libavdevice-dev libavfilter-dev libavformat-dev libavutil-dev libpostproc-dev libswscale-dev libswresample-dev; do
  apt-get download $f
done
```

### 关于 PyAV 版本

`VideoWriter` 依赖 PyAV 的一个补丁（PR #2305，于 2026-06-14 合入 main 分支），
它修了 `Container.add_mux_stream` 以及 matroska muxer 的 `extract_extradata` BSF。
在 PyAV 发布 ≥19 版本之前，**请直接装 main 分支的 PyAV**：

```bash
pip install av @ https://github.com/PyAV-Org/PyAV/archive/main.tar.gz
```

如果继续用 PyPI 上的 18.0.0，`VideoWriter` 会写默认的 1280×720 High avcC，
下游 decoder 将无法解码。

## 从源码编译

```bash
python -m build
```

## 接口设计简述

### 硬件解码

```python
import wdnav
cap = wdnav.VideoCapture(path)  # path 可以是本地文件、网络流或V4l2设备，注意V4l2设备可能会因为兼容问题需要代码调测
ret, frame = cap.read()  # 与 cv2 用法一致；BGR (H, W, 3) ndarray
ret, frame = cap.read(w, h, fmt)  # 这是 cv2 没有的接口，使用 RGA 进行指定分辨率/像素格式的转换
ret, frame0 = cap.read0(...) # 这是0拷贝接口，获取到的是dmabuffer，可以0拷贝传递给其他组件，默认参数将会直接获取原始解码数据
frame0.numpy(w, h, fmt) # 这样dmabuffer就可以转回numpy形式虚拟地址，同 cap.read(w, h, fmt)
```

### 硬件编码

```python
import wdnav
out = wdnav.VideoWriter(path, fourcc, fps, (width, height), gop, ...)
                       # path 可以是本地文件、网络流、websocket等
out.write(frame)  # 兼容cv2的numpy格式frame，注意他的格式固定为BGR
out.write(frame0)  # 0拷贝接口，传入dmabuffer结构体，图片格式在frame0数据结构中声明
```

### 图像合成

（TBD）

### 其他详细示例见 samples 目录

`samples/*.py` 直接 `import cv2` 做窗口显示，跑前 `pip install opencv-python`。

## 兼容性

- 目标兼容 Python 3.10+（`requires-python = ">=3.10"`）
- 平台：当前主要面向 Linux（.so）

## License

Apache-2.0。见 [LICENSE](LICENSE)。
