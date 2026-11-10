[English](README.md) | [简体中文](README_CN.md)

# wdnav

> Python bindings for Rockchip hardware video codec and graphics accelerator.

## Project Design

* Compatible with `cv2` usage, while also supporting zero-copy paths
* Wraps RGA / GPU / VPU hardware acceleration units
* Works on the vast majority of Rockchip SoCs

## Dependencies

### Native `.so` libraries

**Native libraries are not vendored in this repo** and must be installed at the
system level (apt / dnf / prebuilt deb), typically under `/usr/lib/` and
similar system dirs. `wdnav` resolves them via `ctypes.util.find_library` +
`dlopen` on the system path, or links them into the C shims at build time.

| Library | Required for | Debian / Ubuntu package |
|---|---|---|
| `librockchip_mpp.so.0` | `VideoCapture` / `VideoWriter` HW decode + encode | `librockchip-mpp1` |
| `librga.so.2` | RGA2 colour-convert / resize in `VideoCapture.read(w, h)` | `librga2` |
| `libmali.so` (`libmali-valhall-g610-g24p0-x11-wayland-gbm`) | `SurfaceComposer` / `Typer` offscreen GLES — pulled in by `libshim_mali.so`'s DT_NEEDED | `libmali-valhall-g610-g24p0-x11-wayland-gbm` |
| `libfreetype.so.6` | `Typer` (Surface subclass) text rasterisation — c-shim calls `FT_Init_FreeType` etc. from `rkm_text_*` | `libfreetype-dev` (provides both `libfreetype.so.6` + `/usr/include/freetype2/{ft2build.h,freetype/}`) |
| `libfontconfig` + `fc-match` CLI | `Typer(font=...)` font-family / pattern resolution (e.g. `"sans"`, `"wqy-zenhei"`, `"DejaVu Serif"`) | `fontconfig` |
| `libv4l2.so.0` | `MediaDevice` / `v4l2.py` helpers (only if you open `/dev/video*`) | `libv4l2-dev` |

`libmali.so` and `libfreetype.so.6` are linked into `libshim_mali.so` at build
time (`-lmali -lfreetype`); the loader resolves them through the system path
or via the `-Wl,-rpath` recorded in the .so. Headers must be present at
build time — `libfreetype-dev` for `/usr/include/freetype2/ft2build.h`,
Khronos EGL/GLES headers (`libecg-dev`, `libgles2-mesa-dev`) for
`/usr/include/EGL/`, `/usr/include/GLES3/`.

For the CJK font in the Typer demo (`"wqy-zenhei"`) install `fonts-wqy-zenhei`.

### External wheels

* **PyAV** — used for audio/video demux / mux. After demux, the raw stream is handed
  off to MPP for hardware encode / decode.

PyAV requires the system FFmpeg headers (`libavcodec-dev`, `libavformat-dev`,
`libavdevice-dev`, `libavfilter-dev`, `libavutil-dev`, `libswscale-dev`,
`libswresample-dev`), FFmpeg ≥ 6.0. On Debian sid arm64 the deb packages are
available directly:

```bash
# Pull all dev deps at once (apt install also works)
for f in ffmpeg libavcodec-dev libavdevice-dev libavfilter-dev libavformat-dev libavutil-dev libpostproc-dev libswscale-dev libswresample-dev; do
  apt-get download $f
done
```

### Note on PyAV version

`VideoWriter` depends on a PyAV patch (PR #2305, merged 2026-06-14 into PyAV main)
that fixes `Container.add_mux_stream` and the matroska muxer `extract_extradata` BSF.
Until PyAV publishes a wheel ≥19 on PyPI, **install PyAV from main**:

```bash
pip install av @ https://github.com/PyAV-Org/PyAV/archive/main.tar.gz
```

If you stick with the PyPI 18.0.0 wheel, `VideoWriter` will emit a default
1280×720 High avcC and downstream decoders will refuse to decode.

## Building from Source

```bash
python -m build
```

## API Overview

### Hardware Decode

```python
import wdnav

cap = wdnav.VideoCapture(path)  # path can be a local file, a network stream, or a V4L2 device.
                                # Note: V4L2 devices may need code-level tuning for compatibility.

ret, frame = cap.read()  # Same as cv2; BGR (H, W, 3) ndarray

ret, frame = cap.read(w, h, fmt)   # cv2 does not offer this: uses RGA for arbitrary
                                    # resolution / pixel-format conversion

ret, frame0 = cap.read0(...)        # Zero-copy: returns a dmabuffer handle that can be
                                    # passed to other components without a CPU-side copy.
                                    # With default args you get the raw decoded data.

frame0.numpy(w, h, fmt)             # Convert the dmabuffer back to a numpy array,
                                    # equivalent to cap.read(w, h, fmt)
```

### Hardware Encode

```python
import wdnav

out = wdnav.VideoWriter(path, fourcc, fps, (width, height), gop, ...)
                       # path can be a local file, a network stream, a websocket, etc.

out.write(frame)   # cv2-compatible numpy frame; format is fixed to BGR

out.write(frame0)  # Zero-copy: takes a dmabuffer struct; pixel format is declared
                   # inside the frame0 data structure
```

### Compositing

(TBD)

### More examples

See the `samples/` directory. `samples/*.py` imports `cv2` directly for window display,
so install `opencv-python` first.

## Compatibility

- Target: Python 3.10+ (`requires-python = ">=3.10"`)
- Platform: currently Linux (`.so`)

## License

Apache-2.0. See [LICENSE](LICENSE).
