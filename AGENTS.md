# AGENTS.md

Quick reference for OpenCode sessions in this repo. The project is a Rockchip
MPP / RGA2 / V4L2 Python binding (import name `rkv`, distribution name
`pyrkv`); for deeper architecture see `CLAUDE.md` (currently not checked in
— fall back to `README.md` and source).

## TL;DR commands

System is Debian + PEP 668: do **not** run global `pip install`. Always use the
in-tree `.venv/`. As shipped, `.venv/` only has `build` / `pip` / `setuptools`
— **the project itself and dev tools are not pre-installed**, so step 1 below
is mandatory before any test or import runs.

```bash
# 1. one-time setup (installs rkv + pytest/build/twine + av/numpy/v4l2)
.venv/bin/pip install -e ".[dev]"

# 2. tests (import-only smoke, never needs librockchip_mpp / librga)
.venv/bin/pytest
.venv/bin/pytest tests/test_smoke.py::test_version_is_string

# 3. rebuild sdist + wheel — wheel should NOT contain any .so
.venv/bin/python -m build
unzip -l dist/pyrkv-0.1.1-py3-none-any.whl | grep -E '\.so' || echo "OK: no .so files bundled"

# 4. end-to-end video (headless): sample renamed, now samples/video_decoder.py
DISPLAY=0 .venv/bin/python samples/video_decoder.py /path/to/video.mp4
```

Do **not** invoke `ruff` / `flake8` / `black` / `mypy` — none are configured.

## Gotchas an agent would otherwise miss

- **Distribution name ≠ import name.** PyPI package is `pyrkv`; runtime
  import is `rkv`. Don't write `import pyrkv`.
- **`import rkv` must keep working without .so files.** `VideoCapture` and
  `MediaDevice` are imported under `try/except` in `src/rkv/__init__.py` and
  become `None` when native libs / `media-ctl` are missing — any change that
  turns that into an unconditional import or eager ctypes call will break
  `import rkv` on non-Rockchip / developer machines. Preserve this fallback.
- **`import rkv.video` no longer needs opencv-python.** All pixel-format
  conversion goes through RGA (`src/rkv/video.py:_v4l2_rga_to_bgr`,
  `_rga_to_bgr`, `_rga_nv12_to`). When RGA cannot accept a format / size
  combination the helper raises `RuntimeError`; there is no software
  fallback. Adding one would re-introduce the cv2 dependency.
- **No `.so` files in the repo.** `librockchip_mpp.so.0` / `librga.so.2` are
  loaded **only** from the system (Debian multiarch path first, then
  `ctypes.util.find_library` / `dlopen` via ldconfig). If the library isn't
  installed system-wide, the loader raises `RuntimeError` / `OSError` with a
  message naming the missing soname. Don't reintroduce vendored `.so` —
  `.gitignore` no longer whitelists any such path.
- **`requires-python = ">=3.10"`.** The video subsystem (`mpp.py` / `rga.py`
  / `video.py` / `media.py` / `v4l2.py`) uses PEP 604 unions and PEP 585
  generics; do not "downgrade" to `Optional[X]` / `List[X]`.
- **Version lives in two places** that must move together:
  `src/rkv/_version.py::__version__` and `pyproject.toml`'s `version`.
  Public stubs (`src/rkv/rkv.pyi`) currently only carry `__version__: str`;
  expand them when you add new public symbols.
- **No video auto-tests.** `tests/test_smoke.py` only verifies `import rkv`
  and the conditional `VideoCapture` / `MediaDevice` fallbacks — never
  imports `mpp` / `rga` / `video`. Video verification is
  `samples/video_decoder.py`, `samples/multi_decoder.py`,
  `samples/camera_capture.py` — they require a Rockchip target with the
  .so files actually loadable.
- **No CI / pre-commit / lint config in the repo.** Verify locally with the
  commands above; there is no GitHub Actions workflow to lean on.

## When you change things, also update

| You changed…                          | Then update…                                                      |
| ------------------------------------- | ----------------------------------------------------------------- |
| Public API symbols in `__init__.py`   | `src/rkv/rkv.pyi`                                                |
| `mpp.py` / `rga.py` ctypes signatures | `src/rkv/rkv.pyi` (if visible publicly) + `_LIBS_TO_TRY` lists    |
| `pyproject.toml` `package-data` glob  | Rebuild wheel; `unzip -l dist/*.whl` to confirm no `.so` ships   |
| `MANIFEST.in`                         | `.venv/bin/python -m build --sdist` to verify sdist contents      |
| `.gitignore` (added/removed entries)  | `git check-ignore -v <path>` on the affected files                |
| Public API symbols                    | `__all__` in `src/rkv/__init__.py`                                |
| Version                               | `src/rkv/_version.py` **and** `pyproject.toml` `version`          |
| Added a new native dep                | Add a system install hint to the loader's `OSError` message; **do not** vendor the `.so` (re-introduce `native/` would break the no-bundle invariant) |

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
