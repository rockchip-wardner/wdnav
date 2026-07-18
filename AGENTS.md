# AGENTS.md

Quick reference for OpenCode sessions in this repo. The project is a Rockchip
MPP / RGA / V4L2 / DmaHeap / GPU Python binding (import name `rkv`, distribution name
`pyrkv`); 


## Repo map (one-line each)

- `src/rkv/__init__.py` — public API surface; conditional `VideoCapture` /
  `VideoWriter` / `MediaDevice` / `SurfaceFlinger` / `DmaHeap` import so
  `import rkv` never hard-fails on a dev box missing native libs.
- `src/rkv/_version.py` — single source of truth for `__version__`.
- `src/rkv/video.py` — `VideoCapture` (MPP decoder + RGA colour-conversion,
  `read()` / `read0()` zero-copy DMA-BUF handle).
- `src/rkv/mpp.py` — ctypes bindings to `librockchip_mpp`; `MppEncoder`,
  `MppDecoder`, `EncodedPacket`, `DecodedFrame` (and the format /
  helper maps they share).
- `src/rkv/rga.py` — ctypes bindings to `librga`; colour-conversion
  + resize helpers used by `DecodedFrame.numpy`.
- `src/rkv/encoder.py` — `VideoWriter` (MPP hardware H.264 + PyAV
  mp4v / mjpeg muxing; no cv2, no libx264 fallback). Also hosts the
  `rkv+ws://` WebSocket sink (JPEG + H.264 via MPP).
- `src/rkv/surfaceflinger.py` — `SurfaceFlinger` / `Surface` (EGL +
  Mali GLES compositor for displaying decoded frames).
- `src/rkv/mali.py` — Mali GPU / EGL helpers backing `SurfaceFlinger`.
- `src/rkv/media.py` — `media-ctl` topology parser used by the
  camera-capture sample.
- `src/rkv/v4l2.py` — V4L2 capture helpers (the upstream
  `python-v4l2` shim, loaded lazily only when `/dev/video*` is opened).
- `src/rkv/dmaheap.py` — `DmaHeap` / `RkDmaBuf` DMA-BUF allocators
  (`/dev/dma_heap/*` and Rockchip GEM ioctl respectively).
- `src/rkv/rkv.pyi` + `py.typed` — type stubs (PEP 561).
- `samples/` — runnable examples, no `test_` prefix on filenames.
- `tests/` — `test_smoke.py` only, import-only, never requires `.so`.


## 接口设计

### 设计原则

- 兼容cv2,但不重复调用cv2，rkv不对cv2做任何依赖
- 使用pyav做握手和demux，不使用pyav和libx264做任何编解码操作，编解码由mpp来完成
- 不做任何软解和软拷贝的fallback操作，如果mpp、rga、mali库都完不成的操作就直接报错

### 视频解码

```python
import rkv
cap = rkv.VideoCapture(path)  # path 可以是本地文件、网络流或V4l2设备
ret, frame = cap.read()  # 与 cv2 用法一致；BGR (H, W, 3) ndarray
ret, frame = cap.read(w, h, fmt)  # 这是 cv2 没有的接口，使用 RGA 进行指定分辨率/像素格式的转换
ret, frame0 = cap.read0(w, h, fmt) # 这是0拷贝接口，获取到的是dmabuffer的fd等内容，可以对接传递给
# cap.read0(...).numpy()得到的结果就是cap.read(...) #
```

### 视频编码

```python
import rkv
out = rkv.VideoWriter(path, fourcc, fps, (width, height), gop, ...)  # path 可以是本地文件、网络流、websocket等
out.write(frame)  # 兼容cv2的numpy格式frame，一般来说格式是BGR
out.write(frame0)  # 0拷贝接口，传入dmabuffer结构体
```

## 重要提示

- 用中文回答
- 不要私自提交git commit
