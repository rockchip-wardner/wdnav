# AGENTS.md

Quick reference for OpenCode sessions in this repo. The project is a Rockchip
MPP / RGA / V4L2 / DmaHeap / GPU Python binding (import name `rkv`, distribution name
`pyrkv`); 


## Repo map (one-line each)

- `src/rkv/__init__.py` — public API surface; conditional VideoCapture /
  MediaDevice import.
- `src/rkv/_version.py` — single source of truth for `__version__`.
- `src/rkv/video.py`, `mpp.py`, `rga.py` — Rockchip MPP + RGA2 decoder.
  Only loaded via `import rkv.video` (or implicitly through VideoCapture).
- `src/rkv/media.py`, `v4l2.py` — `media-ctl` parser and v4l2 helpers used
  by the camera capture sample.
- `src/rkv/rkv.pyi` + `py.typed` — type stubs (PEP 561).
- `samples/` — runnable examples, no `test_` prefix on filenames.
- `tests/` — `test_smoke.py` only, import-only, never requires `.so`.


## 接口设计

### 视频解码

```python
import rkv
cap = rkv.VideoCapture(path)  # path 可以是本地文件、网络流或V4l2设备
ret, frame = cap.read()  # 与 cv2 用法一致；BGR (H, W, 3) ndarray
ret, frame = cap.read(w, h, fmt)  # 这是 cv2 没有的接口，使用 RGA 进行指定分辨率/像素格式的转换
ret, frame0 = cap.read0(...) # 这是0拷贝接口，获取到的是dmabuffer的fd等内容，可以对接传递给
```

### 视频编码

```python
import rkv
out = rkv.VideoWriter(path, fourcc, fps, (width, height), gop, ...)  # path 可以是本地文件、网络流、websocket等
out.write(frame)  # 兼容cv2的numpy格式frame，一般来说格式是BGR
out.write(frame0)  # 0拷贝接口，传入dmabuffer结构体
```

