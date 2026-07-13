# samples/

用来放 rkv 的用法示例。每个示例建议是一个**可独立运行**的 `.py` 文件，
方便用户 `python samples/<name>.py` 直接看到效果。

## 添加新示例的约定

- 顶部加一段 docstring 说明这个示例在演示什么
- 不要依赖 pytest 之类的测试框架 —— 直接 `if __name__ == "__main__":` 跑
- 避免硬编码绝对路径，使用相对路径
- 输出尽量用 `print()` 而不是 logging，方便用户直接看结果
