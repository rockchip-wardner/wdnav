# AGENTS.md

Quick reference for OpenCode sessions in this repo. For deeper architecture
(see video subsystem / KV backend design / native-symbol contracts / per-edit
checklists), read `CLAUDE.md` next to this file — it stays the source of truth.

## TL;DR commands

System is Debian + PEP 668: do **not** run global `pip install`. Always use the
in-tree `.venv/`. As shipped, `.venv/` only has `build` / `pip` / `setuptools`
— **the project itself and dev tools are not pre-installed**, so step 1 below
is mandatory before any test or import runs.

```bash
# 1. one-time setup (installs rkv + pytest/build/twine + av/cv2/numpy)
.venv/bin/pip install -e ".[dev]"

# 2. tests (KV-only smoke, never needs librockchip_mpp / librga)
.venv/bin/pytest
.venv/bin/pytest tests/test_smoke.py::test_rkv_context_manager_uses_python_backend
.venv/bin/pytest -k closed_raises

# 3. rebuild sdist + wheel — wheel should NOT contain any .so
.venv/bin/python -m build
unzip -l dist/pyrkv-0.1.0-py3-none-any.whl | grep -E '\.so' || echo "OK: no .so files bundled"

# 4. end-to-end video (headless): sample renamed, now samples/video_decoder.py
DISPLAY=0 .venv/bin/python samples/video_decoder.py /path/to/video.mp4
```

Do **not** invoke `ruff` / `flake8` / `black` / `mypy` — none are configured.

## Gotchas an agent would otherwise miss

- **Distribution name ≠ import name.** PyPI package is `pyrkv`; runtime
  import is `rkv`. Don't write `import pyrkv`.
- **`import rkv` must keep working without .so files.** `VideoCapture` and
  `MediaDevice` are imported under `try/except` in `src/rkv/__init__.py:23-40`
  and become `None` when native libs are missing — any change that turns that
  into an unconditional import or eager ctypes call will break `import rkv`
  on non-Rockchip / developer machines. Preserve this fallback.
- **No `.so` files in the repo.** `librkv.so` / `librockchip_mpp.so.0` /
  `librga.so.2` are loaded **only** from the system (Debian multiarch path
  first, then `ctypes.util.find_library` / `dlopen` via ldconfig). If the
  library isn't installed system-wide, the loader raises `RuntimeError` /
  `OSError` with a message naming the missing soname. Don't reintroduce
  vendored `.so` — `.gitignore` no longer whitelists any such path.
- **`requires-python = ">=3.10"`.** The video subsystem (`mpp.py` / `rga.py`
  / `video.py` / `media.py` / `v4l2.py`) uses PEP 604 unions and PEP 585
  generics; do not "downgrade" to `Optional[X]` / `List[X]`.
- **Version lives in two places** that must move together:
  `src/rkv/_version.py::__version__` and `pyproject.toml`'s `version`. Same
  for stubs: changes to public signatures in `core.py` / `mpp.py` / `rga.py`
  need a matching update in `src/rkv/rkv.pyi`.
- **No video auto-tests.** `tests/` only covers the KV path on purpose
  (CI environments without librockchip_mpp / librga would fail otherwise).
  Video verification is `samples/video_decoder.py`,
  `samples/multi_decoder.py`, `samples/camera_capture.py` — they require a
  Rockchip target with the .so files actually loadable.
- **No CI / pre-commit / lint config in the repo.** Verify locally with the
  commands above; there is no GitHub Actions workflow to lean on.

## When you change things, also update

| You changed…                          | Then update…                                                      |
| ------------------------------------- | ----------------------------------------------------------------- |
| `core.py` public surface              | `src/rkv/rkv.pyi`                                                |
| `mpp.py` / `rga.py` ctypes signatures | `src/rkv/rkv.pyi` (if visible publicly) + `_LIBS_TO_TRY` lists    |
| `pyproject.toml` `package-data` glob  | Rebuild wheel; `unzip -l dist/*.whl` to confirm no `.so` ships   |
| `MANIFEST.in`                         | `.venv/bin/python -m build --sdist` to verify sdist contents      |
| `.gitignore` (added/removed entries)  | `git check-ignore -v <path>` on the affected files                |
| Public API symbols                    | `__all__` in `src/rkv/__init__.py`                                |
| Version                               | `src/rkv/_version.py` **and** `pyproject.toml` `version`          |
| Added a new native dep                | Add a system install hint to the loader's `OSError` message; **do not** vendor the `.so` (re-introduce `native/` would break the no-bundle invariant) |

## Repo map (one-line each)

- `src/rkv/core.py` — `Backend` / `PythonBackend` / `NativeBackend` /
  `Rkv` / `make_backend` (auto-fallback). Pure Python entry point.
- `src/rkv/__init__.py` — public API surface; conditional VideoCapture /
  MediaDevice import.
- `src/rkv/video.py`, `mpp.py`, `rga.py` — Rockchip MPP + RGA2 decoder.
  Only loaded via `import rkv.video` (or implicitly through VideoCapture).
- `src/rkv/media.py`, `v4l2.py` — `media-ctl` parser and v4l2 helpers used
  by the camera capture sample.
- `src/rkv/_native.py` — `librkv.so` (KV native backend) system loader.
- `src/rkv/rkv.pyi` + `py.typed` — type stubs (PEP 561).
- `samples/` — runnable examples, no `test_` prefix on filenames.
- `tests/` — `test_smoke.py` only, KV path, never requires `.so`.
