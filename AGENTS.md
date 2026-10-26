# AGENTS.md

Quick reference for OpenCode sessions in this repo. The project is a Rockchip
MPP / RGA / V4L2 / DmaHeap / GPU Python binding (import name `wdnav`, distribution name
`wdnav`); 


## Repo map (one-line each)

- `src/wdnav/__init__.py` — public API surface; conditional `VideoCapture` /
  `VideoWriter` / `MediaDevice` / `SurfaceComposer` / `DmaHeap` import so
  `import wdnav` never hard-fails on a dev box missing native libs.
- `src/wdnav/_version.py` — single source of truth for `__version__`.
- `src/wdnav/video.py` — `VideoCapture` (MPP decoder + RGA colour-conversion,
  `read()` / `read0()` zero-copy DMA-BUF handle).
- `src/wdnav/mpp.py` — ctypes bindings to `librockchip_mpp`; `MppEncoder`,
  `MppDecoder`, `EncodedPacket`, `DecodedFrame` (and the format /
  helper maps they share).
- `src/wdnav/rga.py` — ctypes bindings to `librga`; colour-conversion
  + resize helpers used by `DecodedFrame.numpy`.
- `src/wdnav/encoder.py` — `VideoWriter` (MPP hardware H.264 + PyAV
  mp4v / mjpeg muxing; no cv2, no libx264 fallback). Also hosts the
  `wdnav+ws://` WebSocket sink (JPEG + H.264 via MPP).
- `src/wdnav/composer.py` — `SurfaceComposer` / `Surface` / `Typer`
  (EGL + Mali GLES compositor for displaying decoded frames;
  `Typer` extends `Surface` with freetype text rasterisation via
  `wdnav.MaliText` and fontconfig `fc-match` font resolution).
- `src/wdnav/cshim_mali.c` — C shim compiled to `libshim_mali.so`;
  thin wrapper over `libmali.so` (EGL/GLES PBuffer path + DMA-BUF
  FBO) and `libfreetype.so.6` (text rasterisation). Linked at build
  time by `_build_wdnav.py` (`-lmali -lfreetype -I/usr/include/freetype2`).
- `src/wdnav/mali.py` — Python ctypes wrapper over `libshim_mali.so`
  (no direct `libmali` calls).
- `src/wdnav/media.py` — `media-ctl` topology parser used by the
  camera-capture sample.
- `src/wdnav/v4l2.py` — V4L2 capture helpers (the upstream
  `python-v4l2` shim, loaded lazily only when `/dev/video*` is opened).
- `src/wdnav/dmaheap.py` — `DmaHeap` / `WdnavDmaBuf` DMA-BUF allocators
  (`/dev/dma_heap/*` and Rockchip GEM ioctl respectively).
- `src/wdnav/wdnav.pyi` + `py.typed` — type stubs (PEP 561).
- `samples/` — runnable examples, no `test_` prefix on filenames.
- `tests/` — `test_smoke.py` only, import-only, never requires `.so`.


## 接口设计

### 设计原则

- 兼容cv2,但不重复调用cv2，wdnav不对cv2做任何依赖
- 使用pyav做握手和demux，不使用pyav和libx264做任何编解码操作，编解码由mpp来完成
- 不做任何软解和软拷贝的fallback操作，不做任何兜底操作，如果mpp、rga、mali库都完不成的操作就直接报错，不要用pyav来做jpeg、264、265等软件编解码
- 在封装和调试mpp和rga库时候，仅参考他们的sample和test源码以及公共头文件部分，不要去阅读任何mpp和rga实现的源码
  mpp只允许阅读他的test、utils、inc目录，其他目录禁止阅读
- 除非要给cv2使用，否则不建议用read接口或者保存numpy数组，这样会造成下次硬件使用时候只能从虚拟地址获取内容，效率很低，建议用read0接口
- 除非要给cv2使用，否则不是用BGR格式保存，硬件编解码对NV12更加友好
- 编码除jpeg用FIXQP模式外，其余都用CBR

### 修改守则

改动某个功能的时候，不影响其它功能；如果需要改动到公共功能，需要自动
跑真实测试每个关联的功能项是否都正常。

- 改动只触及目标功能及其直接依赖，不顺手 refactor 无关代码。
- 公共功能改动（`mpp.py` / `rga.py` / `encoder.py` 等被多个调用方
  共享的模块、native `.so` 的 ctypes 绑定）必须跑所有关联功能的真实
  测试（`samples/` 里用该模块的样例、`tests/test_smoke.py`、调用方
  集成路径），不允许"测试跳过"或"代码改对了就行"。
- 失败 → 修到全通过再交。

### 视频解码

```python
import wdnav
cap = wdnav.VideoCapture(path)  # path 可以是本地文件、网络流或V4l2设备
ret, frame = cap.read(w, h, fmt)  # 这是cv2接口的扩展，使用 RGA 进行指定分辨率/像素格式的转换，如果参数都不给，默认与cv2一样fmt=BGR
ret, frame0 = cap.read0(w, h, fmt) # 这是0拷贝接口，获取到的是dmabuffer的fd等内容，可以对接传递给
# cap.read0(...).numpy()得到的结果就是cap.read(...) #
```

### 视频编码

```python
import wdnav
out = wdnav.VideoWriter(path, fourcc, fps, (width, height), gop, ...)  # path 可以是本地文件、网络流、websocket等
out.write(frame)  # 兼容cv2的numpy格式frame，一般来说格式是BGR
out.write(frame0)  # 0拷贝接口，传入dmabuffer结构体
```

## 重要提示

- 用中文回答
- 除非我要求，否则不要自动提交git commit
- 由于视频编码出现python调用mpp和c调用mpp得到的sps pps不一致问题，所以我们封装一层高级的c shim接口给python调用

## V4L2 采集：G_FMT 兜底与"撒谎的四cc"

`src/wdnav/v4l2.py` 的 `_negotiate_format()` 带 **G_FMT 兜底**：

1. 先按调用方请求的格式 S_FMT；
2. 失败（EINVAL）就 `VIDIOC_G_FMT`（ioctl 号 `0xC0D05604`，上游 `v4l2`
   包没导出，这里写死）问驱动"当前是什么格式"；
3. 用驱动报的格式再 S_FMT 一次，成功就用它。

**为什么必须有**：有些驱动只认自己那一种格式。本机 rk_hdmirx（RK3588
HDMI-RX）只接受 `type=V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE + 'NV24'`，
bgr3 / nv12 / rgb3 / yuyv / uyvy 一律
`OSError: [Errno 22] Invalid argument` —— 没有兜底就永远打不开设备。

**配套：驱动报的四cc可能是撒谎的。** rk_hdmirx 报 'NV24'（V4L2 规范里
是 12bit YUV420SP），实际写进缓冲区的是 **8bit YUV422SP**（Y 平面 W*H +
交错 UV 平面 W*H，共 2*W*H；缓冲区声明的 sizeimage 是 W*H*3，尾部 W*H
字节是驱动没写的空洞）。按 fourcc 猜布局 → 满屏竖条纹。

所以对外暴露了 :func:`frame_layout` / :attr:`V4L2Capture.frame_layout`：
把"驱动报了什么"翻译成"数据实际长什么样"（payload: bgr24 / yuv422sp /
yuv420sp + 每像素字节数 + 有效字节数）。已知会撒谎的驱动登记在
`_DRIVER_PAYLOAD_OVERRIDES` 里。

**调用方约定**：`V4L2Capture.pixel_format` 报的是**协商结果**，不是请求值
（G_FMT 兜底后两者可能不同）；`V4L2BufferInfo.pixel_format` 同理。

## 编译策略 — `libshim_video.so` (v0.3.1 unconditional rebuild)

`src/wdnav/_build_wdnav.py::build_py.run()` 在每次 `pip install` /
`pip install -e` 时**无条件** `os.unlink(_OUTPUT_SO)` 然后 `compile_shim()`,
不再走 mtime 比较的 fast path。代价是两段 `.c` 的 gcc 编译 ~5s。

**为什么改**:旧的 mtime 检查 (`os.path.getmtime(_OUTPUT_SO) < src_mtime`)
只比 C 源 (`cshim_video_encoder.c` / `cshim_video_decoder.c`),
但 `import wdnav` 时 `mpp.py` 等会在 module 顶部执行
`_lib.rkv_xxx.argtypes = [...]`(module-level ctypes 绑定)。
如果新 commit 在 `mpp.py` / `rga.py` / `video.py` / `encoder.py`
里加了一个 `_lib.rkv_new_feature.argtypes = [...]` 的绑定而
`cshim_video_*.c` 没改,mtime 不动、hook 跳过 rebuild,`import wdnav`
会立刻抛 `AttributeError: .../libshim_video.so: undefined symbol: rkv_new_feature`
—— 跟 wdndds 0.6.1 修掉的 `wdndds_topic_writer_create` 是同一个 trap
(只是入口不同:wdndds 是 `Hub()`,wdnav 是 `import wdnav`)。

**显式跳过**(不删、不重编):
- CLI: `--skip-wdnav-video-build`
- Env: `SKIP_WDNAV_VIDEO_BUILD=1`
适用于 CI 预编 `libshim_video.so` 的生产镜像;本机 dev box 不应跳过。

**未变**:
- `_C_SOURCES` 仍是那两个 `.c` 文件
- 输出路径仍是 `src/wdnav/lib/libshim_video.so`
- `compile_shim()` 本身的 cmake-style 注释 / 输出格式 / cross-build env vars (`WDNAV_MPP_INC` / `WDNAV_MPP_LIB`) 全部保留
