# pyrkv

> Python 版的 rkv 工具包 —— 包含纯 Python 实现、捆绑本地 .so、以及对第三方 Python 库的依赖。

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
│       ├── core.py         # 纯 Python 主逻辑
│       ├── _native.py      # 加载 native/ 下的 .so
│       ├── rkv.pyi         # 类型 stub
│       └── py.typed        # PEP 561 类型标记
├── native/                 # 预编译的 .so 放在这里
│   └── README.md
├── samples/                # 用法示例
│   └── README.md
├── tests/                  # pytest 测试
│   └── test_smoke.py
└── docs/
```

## 安装

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

```python
import rkv

# 详细示例见 samples/ 目录
```

## 兼容性

- 目标兼容 Python 3.x 全系列（`requires-python = ">=3.0"`）
- 平台：当前主要面向 Linux（.so）

## 开发

```bash
# 运行测试
pip install -e ".[dev]"
pytest

# 发布到 PyPI（需先配置凭据）
twine upload dist/*
```
