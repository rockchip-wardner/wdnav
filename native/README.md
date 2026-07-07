# native/

把预编译的本地共享库放在本目录下，构建 wheel 时会一并打包进去。

## 当前约定

| 平台  | 文件名              |
| ----- | ------------------- |
| Linux | `librkv.so`         |
| macOS | `librkv.dylib`      |
| Win   | `librkv.dll`        |

加载逻辑在 [`src/rkv/_native.py`](../src/rkv/_native.py)。它会按以下顺序查找：

1. wheel 安装后的 `rkv/_native/librkv.so`（由 `pyproject.toml` 的 `package-data` 段提供）
2. 源码 `native/librkv.so`（开发模式 / editable install）

## 放置步骤

1. 把 `.so` 拷到本目录：

   ```bash
   cp /path/to/built/librkv.so ./
   ```

2. 重新构建 wheel：

   ```bash
   python -m build
   ```

3. 检查产物里是否带了 .so：

   ```bash
   unzip -l dist/pyrkv-*.whl | grep librkv
   ```

## 占位说明

目前 `native/` 是空的 —— `_native.load_library()` 找不到文件时会抛 `RuntimeError`，
`make_backend()` 自动 fallback 到 `PythonBackend`（纯字典后端），
所以**没有 .so 也能正常 `import rkv` 并使用基础 KV 功能**。
