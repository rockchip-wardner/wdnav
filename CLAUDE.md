# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目定位

`pyrkv` —— 混合型 Python 包，跨两个子系统：

1. **纯 Python KV 客户端**（`Rkv` / `Backend`），设计上永不依赖 native 库。
2. **Rockchip 视频解码**（`VideoCapture`）：通过 ctypes 调 `librockchip_mpp.so` 和 `librga.so`，在 Rockchip 平台上走硬件 VPU + RGA2 加速。

distribution 名 `pyrkv`，**导入名是 `rkv`**（不要搞反）。License: Apache-2.0。

## 常用命令

> 这个系统是 Debian 的 PEP 668 严格环境，**不能用 `pip install` 直接装到全局**。所有命令都用仓库自带的 `.venv/`：

```bash
# 可编辑安装（含 dev 依赖：pytest / build / twine，加上运行期 av / cv2 / numpy）
.venv/bin/pip install -e ".[dev]"

# 跑全部测试（只测 KV 路径，不依赖 mpp / rga .so）
.venv/bin/pytest

# 跑单个测试文件 / 函数 / 关键字匹配
.venv/bin/pytest tests/test_smoke.py
.venv/bin/pytest tests/test_smoke.py::test_rkv_context_manager_uses_python_backend
.venv/bin/pytest -k closed_raises

# 构建 sdist + wheel（产物在 dist/）
.venv/bin/python -m build

# 检查 wheel 里是否带上了 video 模块的 .so
unzip -l dist/pyrkv-0.1.0-py3-none-any.whl | grep -E '(librga|librockchip|_native)'

# 端到端跑 video 解码 sample（headless：DISPLAY=0）
DISPLAY=0 .venv/bin/python samples/test_video_decoder.py /path/to/video.mp4 60

# 临时构建到独立目录（避免污染 dist/）
.venv/bin/python -m build --outdir /tmp/pyrkv-build
```

仓库没有配置 lint（ruff/flake8/black），不要假设它们存在。

## 架构要点

### KV 子系统（`src/rkv/core.py`）

- `Backend` —— 抽象基类（`get/set/delete/keys/close`）
- `PythonBackend` —— 纯 `dict` 后端
- `NativeBackend` —— 通过 ctypes 调 `librkv.so`（**占位假设**）
- `make_backend(prefer="native")` —— **自动降级**：native 缺失时返回 `PythonBackend`，**不抛异常**。`prefer="python"` 可强制走 Python 后端
- `Rkv` —— 用户面 KV 客户端，带状态门面 + 上下文管理器

降级逻辑意味着：**`import rkv` 永远不会失败**。`Rkv` 在没 .so 时也能用，退化成字典后端。写新代码不要破坏这个性质。

### Video 子系统（`src/rkv/{video,mpp,rga}.py`）

- `rkv.video.VideoCapture` —— OpenCV 风格的硬件解码接口，签名镜像 `cv2.VideoCapture`
- `rkv.mpp` —— `librockchip_mpp.so` 的 ctypes 封装（MppApi、MppDecoder、Annex B helpers）
- `rkv.rga` —— `librga.so` 的 ctypes 封装（cvtcolor / resize / zero-copy fd/nv12 wrapping）

`VideoCapture` 在 `__init__.py` 里通过 try/except 条件导入，**只会在 `import rkv.video` 时才 ctypes 加载 .so**——所以纯 KV 用户在非 Rockchip 平台 `import rkv` 也不会因为缺库而崩溃，只是 `rkv.VideoCapture is None`。

### Native 库加载路径

`mpp.py` 和 `rga.py` 各自的 `_LIBS_TO_TRY` 按这个顺序找：

1. wheel 安装后的 `rkv/_native/<lib>.so`（由 `pyproject.toml` 的 `package-data` 打包）
2. 源码 `native/<lib>.so`（开发模式 fallback）
3. `/usr/lib/aarch64-linux-gnu/<lib>.so`（系统装）
4. `ctypes.util.find_library(...)`（兜底）

`src/rkv/_native/*.so` **是符号链接**，指向 `native/` 下的真实文件——避免双份存储。修改 .so 时只动 `native/` 即可。

`native/librga.so.2` 和 `native/librockchip_mpp.so.0` 都是**入库文件**（`.gitignore` 用 `!native/*.so.*` 和 `!src/rkv/_native/*.so.*` 双白名单放行）。`native/build/`、`native/target/` 这些子目录才是被忽略的构建产物。

### Video 子系统的 native 符号契约

- `librockchip_mpp.so` 导出 `mpp_*` / `MppApi.*` 接口——`mpp.py` 里全部按 `rk_mpi.h` 的字段顺序绑定了
- `librga.so` 导出 `c_RkRga*` / `im2d_*` 接口——`rga.py` 绑定的是 im2d 的 C 风格 wrapper
- 实际硬件支持依 SOC 而异（RK3588 / RK3568 / RK3562 等 codec 表有差异），遇到不支持的格式 `VideoCapture.__init__` 会抛 `RuntimeError`

### 修过的一个 upstream bug

`src/rkv/video.py` 里 `_FRAME_BGR_FORMATS` 是从 rksdk 移植时补的——upstream 引用了但从未定义，导致 `cv2.cvtColor` 兜底路径 `NameError`。移植版里定义成 `frozenset(_MPP_TO_RGA_FMT.keys()) = {0, 4, 5}`（NV12 / I420 / NV21）。改格式支持时同步更新。

## 兼容性约束

- `requires-python = ">=3.10"`，因为 video 子系统用了 `X | None`（PEP 604，3.10+）和 `list[str]`（PEP 585，3.9+ 运行时、3.10+ 解析时）。
- KV 子系统（`core.py` / `_native.py` / `_version.py` / `__init__.py` / `tests/`）首行仍是 `from __future__ import absolute_import, division, print_function`，语法兼容 3.x；但因为整个 wheel 现在 `>=3.10`，没必要再在 KV 路径上额外约束。
- video 子系统（`mpp.py` / `rga.py` / `video.py`）使用 `from __future__ import annotations` 和现代类型注解，**不要**手动降级 PEP 604/585。
- 类型信息放在 `src/rkv/rkv.pyi`（stub）和 `src/rkv/py.typed`（PEP 561 marker）里。

## 测试约定

`tests/test_smoke.py` 只测 KV 路径（`PythonBackend` + `Rkv.open(prefer="python")`），**不依赖 mpp / rga .so**。新增测试同样要保证不依赖 native 库，否则 CI 在没 .so 的环境下会挂。

video 子系统目前没有自动测试，靠 `samples/test_video_decoder.py` 端到端验证。CI 要加 video 测试时必须把 `librockchip_mpp` / `librga` 的可用性当 precondition。

## 修改时的检查清单

- 改 `core.py` 的 `NativeBackend` 符号声明 → 同步 `rkv.pyi`
- 改 `mpp.py` / `rga.py` 的 ctypes 绑定 → 同步 `rkv.pyi`（如果公共签名变了）
- 改 `pyproject.toml` 的 `package-data` → 重新 `.venv/bin/python -m build` 并 `unzip -l dist/*.whl` 确认 .so 进了 `rkv/_native/`
- 改 `MANIFEST.in` → 用 `.venv/bin/python -m build --sdist` 单独验证 sdist
- 改 `.gitignore` 的 `*.so` 白名单 → `git check-ignore -v <file>` 双重确认
- 改公开 API → 更新 `src/rkv/__init__.py` 的 `__all__`
- 改版本号 → 改 `src/rkv/_version.py` 的 `__version__`，并对应改 `pyproject.toml` 的 `version`
- 加新 .so → 拷到 `native/`、在 `src/rkv/_native/` 建符号链接、更新 `_LIBS_TO_TRY` 列表、更新 `.gitignore` 和 `MANIFEST.in`、更新 `pyproject.toml` 的 `package-data` glob