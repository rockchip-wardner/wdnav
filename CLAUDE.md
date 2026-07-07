# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目定位

`pyrkv` —— 混合型 Python 包：纯 Python 实现 + 捆绑本地 `.so` + 第三方 Python 依赖。
distribution 名 `pyrkv`，**导入名是 `rkv`**（不要搞反）。License: Apache-2.0。

## 常用命令

> 这个系统是 Debian 的 PEP 668 严格环境，**不能用 `pip install` 直接装到全局**。所有命令都用仓库自带的 `.venv/`：

```bash
# 可编辑安装（含 dev 依赖：pytest / build / twine）
.venv/bin/pip install -e ".[dev]"

# 跑全部测试
.venv/bin/pytest

# 跑单个测试文件
.venv/bin/pytest tests/test_smoke.py

# 跑单个测试函数
.venv/bin/pytest tests/test_smoke.py::test_rkv_context_manager_uses_python_backend

# 跑单个测试 by 关键字匹配名字
.venv/bin/pytest -k closed_raises

# 构建 sdist + wheel（产物在 dist/）
.venv/bin/python -m build

# 检查 wheel 里是否带上了 .so
unzip -l dist/pyrkv-0.1.0-py3-none-any.whl | grep -E '(librkv|_native)'

# 临时构建到独立目录（避免污染 dist/）
.venv/bin/python -m build --outdir /tmp/pyrkv-build
```

仓库没有配置 lint（ruff/flake8/black），不要假设它们存在。如果用户要求"格式化代码"，用系统自带的 `python3 -m py_compile` 做语法检查即可。

## 架构要点

### Backend 抽象 + 自动降级

`src/rkv/core.py` 是核心，定义了：

- `Backend` —— 抽象基类（`get/set/delete/keys/close`）
- `PythonBackend` —— 纯 `dict` 后端
- `NativeBackend` —— 通过 `ctypes` 调 `librkv.so`
- `make_backend(prefer="native")` —— **自动降级**：native 缺失时返回 `PythonBackend`，**不抛异常**。`prefer="python"` 可强制走 Python 后端。
- `Rkv` —— 用户面的 KV 客户端，是带状态门面（`_closed` 标志、上下文管理器）

降级逻辑意味着：**`import rkv` 永远不会失败**。`rkv.Rkv.open()` 在没 .so 时也能用，只是退化成字典后端。写新代码时不要破坏这个性质。

### Native 库的加载路径

`src/rkv/_native.py` 的查找顺序（**懒加载**，第一次 `get_native()` 时才 `CDLL()`）：

1. wheel 安装后的 `rkv/_native/librkv.so`（由 `pyproject.toml` 的 `package-data` 打包）
2. 源码 `native/librkv.so`（开发模式 fallback）

所以 `native/librkv.so` 是**合法源码文件**而不是构建产物 —— `.gitignore` 用 `!native/*.so` 白名单把它放行。`native/build/`、`native/target/` 这些子目录才是被忽略的构建产物。

### Native 符号契约

`core.py` 的 `NativeBackend.__init__` 假设 `librkv.so` 导出 `rkv_get` / `rkv_set` / `rkv_delete` 三个函数（参数为 `c_char_p`）。**这是占位假设** —— 真实 native 库的实际符号签名需要按需调整那段 ctypes 声明。改完后同步更新 `rkv.pyi` 的类型 stub。

### 把 .so 打进 wheel 的双轨配置

`pyproject.toml` 里同时配了两段，缺一不可：

- `tool.setuptools.package-data` —— 把 `src/rkv/_native/*.so` 作为 `rkv` 包数据装进 wheel（运行时查找路径 #1）
- `tool.setuptools.data-files` —— 把 `native/*.so` 作为额外数据文件装到 wheel 的 `rkv/native/` 路径

两边覆盖同一批文件（一个是 `src/...` 视角，一个是 `native/` 视角），改完要重新 `python -m build` 才会生效。

## 兼容性约束

- `requires-python = ">=3.0"`，目标是 **Python 3.x 全系列**。
- 所有 `.py` 文件首行有 `from __future__ import absolute_import, division, print_function`。
- **不要使用**：f-string（3.6+）、`typing` 模块注解（3.5+ 才稳定）、walrus `:=`、match/case、结构化模式匹配。
- 类型信息放在 `src/rkv/rkv.pyi`（stub）和 `src/rkv/py.typed`（PEP 561 marker）里，不要塞进运行时代码。

## 测试约定

`tests/test_smoke.py` 只测纯 Python 路径（`PythonBackend` + `Rkv.open(prefer="python")`），**不依赖 .so**。
新加测试时同样要保证不依赖 native 库，否则 CI 在没 .so 的环境下会挂。

## 修改时的检查清单

- 改 `core.py` 的 `NativeBackend` 符号声明 → 同步 `rkv.pyi`
- 改 `pyproject.toml` 的 `package-data`/`data-files` → 重新 `python -m build` 并 `unzip -l` 确认文件进了 wheel
- 改 `MANIFEST.in` → 用 `python -m build --sdist` 单独验证 sdist
- 改公开 API → 更新 `src/rkv/__init__.py` 的 `__all__` 和 `rkv.pyi`
- 改版本号 → 改 `src/rkv/_version.py` 的 `__version__`，并对应改 `pyproject.toml` 的 `version`
