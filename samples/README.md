# samples/

用来放 wdnav 的用法示例。每个示例建议是一个**可独立运行**的 `.py` 文件，
方便用户 `python samples/<name>.py` 直接看到效果。

## 添加新示例的约定

- 顶部加一段 docstring 说明这个示例在演示什么
- 不要依赖 pytest 之类的测试框架 —— 直接 `if __name__ == "__main__":` 跑
- 避免硬编码绝对路径，使用相对路径
- 输出尽量用 `print()` 而不是 logging，方便用户直接看到结果

## 示例索引

| 文件 | 演示内容 |
| --- | --- |
| `video_decoder.py` | 单路 H.264/H.265 硬件解码 + cv2 显示 |
| `video_writer.py` | 合成 BGR 帧 → MPP 硬编 (H.264) / PyAV mpeg4 (mp4v) / ``wdnav+ws://`` 写 MP4 |
| `video_capture_writer.py` | ``VideoCapture`` 读 → ``VideoWriter`` 写的端到端转码样例 |
| `multi_decoder.py` | 4 路并行解码 + cv2 拼成 2×2 田字格 |
| `camera_capture.py` | `/dev/videoX` 摄像头捕获 |
| `mali_surfaceless_smoke.py` | SurfaceFlinger 纯 GL 烟雾测试, 不依赖 VideoCapture |
| `mali_display.py` | 单路 video 走 SurfaceFlinger 合成 (BGR888 path) |
| `mali_multi_decoder.py` | 4 路 video 在 SurfaceFlinger 上做 2×2 网格 GPU 合成 |
| `dmaheap_demo.py` | `/dev/dma_heap/` 分配, mmap, sync_read 演示 |
| `egl_import_diagnostic.py` | 用 rbs/glengine 同款属性表跑 eglCreateImageKHR, 判定 libmali 是否支持 DMA-BUF 零拷贝 |

> **依赖说明**：核心库 `wdnav` 现在**完全不再依赖 cv2** —— 编解码走 MPP+PyAV，
> 显示用的 sample 仍然 `import cv2` 做窗口（``opencv-python`` 是可选依赖，
> 不影响库本身）。`video_writer.py` / `video_capture_writer.py` 是
> 不需要 cv2 的；其他几个 sample 需要 `pip install opencv-python` 才能跑。
