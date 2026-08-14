"""setuptools build hook — compile the libshim_video.so C shim before wheel packaging.

The package ships two C source files (``src/wdnav/cshim_video_encoder.c``
and ``src/wdnav/cshim_video_decoder.c``) that wrap ``librockchip_mpp.so``
behind a small, opinionated surface (``rkv_enc_*`` / ``rkv_dec_*``).
The shim has to be compiled for the wheel's target architecture
before setuptools can package it, so this module provides a custom
:class:`build_py` subclass that:

  1. Locates the Rockchip MPP public headers (default
     ``/usr/include/rockchip``; override with ``WDNAV_MPP_INC`` env var);
  2. Invokes ``cc -shared -fPIC`` on both ``cshim_video_encoder.c``
     and ``cshim_video_decoder.c`` with that include path on the
     command line, linking them together into ``libshim_video.so``;
  3. Drops the resulting ``libshim_video.so`` into ``src/wdnav/lib/``
     so setuptools' regular package-data machinery picks it up and
     bundles it in the wheel;
  4. Then defers to the standard ``build_py`` so Python source
     packaging proceeds unchanged.

Compile failures are surfaced as a clear :class:`RuntimeError` listing the
underlying gcc error — silent failures here would mean a wheel that
imports fine but fails the first time anyone tries to use MPP.

Why not setuptools.distutils ``Extension`` / ``build_ext``?
----------------------------------------------------------
``Extension`` would work for the compile step itself, but it ties the
output path to the build_lib/build_platlib tree under ``build/`` and
requires an extra ``build_clib`` / ``build_ext`` orchestration to
copy the result back into the source tree's ``src/wdnav/lib/`` where
``package_data`` knows to look. A direct ``subprocess.run`` keeps
the build path trivially auditable and side-steps the
``build_py``/``build_ext`` ordering issue (mpp.py reads from
``src/wdnav/lib/libshim_video.so`` at import time, so the .so MUST
exist in the source tree before ``build_py`` copies the package out).

Why is the hook a top-level module, not nested under ``wdnav/``?
--------------------------------------------------------------
setuptools' ``cmdclass`` discovery looks up the class by import path
at the start of the build — *before* ``wdnav`` has been installed.
Putting the hook under ``wdnav._build`` would force a chicken-and-egg
import (wdnav imports mpp imports _lib, _build imports wdnav). A
top-level module next to ``pyproject.toml`` is reachable as soon as
setuptools adds the project root to ``sys.path``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from typing import List, Optional, Sequence

from setuptools.command.build_py import build_py as _build_py


# ---------------------------------------------------------------------------
# Compile configuration
# ---------------------------------------------------------------------------

# Canonical location of the Rockchip MPP public headers on this
# machine. Override at build time via ``WDNAV_MPP_INC=/path/to/inc`` for
# cross-builds / CI runners where the headers live elsewhere.
_DEFAULT_MPP_INC = "/usr/include/rockchip"

# Canonical location of ``librockchip_mpp.so`` corresponding to the
# headers above. Decoupled from ``_DEFAULT_MPP_INC`` because the Rockchip
# upstream keeps headers under ``/usr/include/rockchip`` while Debian's
# multiarch layout drops the shared library under
# ``/usr/lib/aarch64-linux-gnu``. Override with ``WDNAV_MPP_LIB``.
_DEFAULT_MPP_LIB = "/usr/lib/aarch64-linux-gnu"

# C source / output layout — relative to the project root. ``build_py``
# runs from the project root, but we resolve everything through
# ``os.path.dirname(__file__)`` so the hook still works when invoked
# from a different working directory (e.g. ``pip install .``).
#
# The hook lives at ``src/wdnav/_build_rkv.py``, so ``__file__``'s
# directory is the package directory ``src/wdnav/``; the project
# root is one level up.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
_C_SOURCES = (
    os.path.join(_THIS_DIR, "cshim_video_encoder.c"),
    os.path.join(_THIS_DIR, "cshim_video_decoder.c"),
)
_OUTPUT_DIR = os.path.join(_THIS_DIR, "lib")
_OUTPUT_SO = os.path.join(_OUTPUT_DIR, "libshim_video.so")


def _resolve_mpp_inc() -> str:
    """Pick the Rockchip MPP public-header directory.

    Resolution order:

      1. ``WDNAV_MPP_INC`` environment variable (CI / cross-build knob);
      2. ``/usr/include/rockchip`` — the canonical Debian install path
         for the Rockchip public headers;
      3. The empty string — at which point we skip ``-I`` and let gcc
         search its default path. The shim will still build (because
         it doesn't actually call any MPP symbol, only takes its
         address for ``__typeof__``), but the resulting ``.so`` may
         carry the wrong prototypes if the default headers disagree
         with the system ``librockchip_mpp.so``.

    The directory is *not* validated here — gcc itself will error out
    with a clear ``fatal error: rk_mpi.h: No such file`` if the path
    is bogus.
    """
    return os.environ.get("WDNAV_MPP_INC", _DEFAULT_MPP_INC)


def _resolve_cc() -> str:
    """Pick the C compiler to use.

    Honours ``CC`` (the standard autoconf / make / distutils knob) so
    a CI runner can swap in ``aarch64-linux-gnu-gcc`` for cross-builds
    without touching this file. Defaults to ``cc`` (whatever the
    system aliases to — gcc on Debian, clang on macOS, etc.).
    """
    return os.environ.get("CC", "cc")


def _build_command(cc: str, mpp_inc: str) -> List[str]:
    """Assemble the gcc command line.

    Flags:

      * ``-shared -fPIC`` — produce a position-independent shared
        library (the only thing ctypes.CDLL can dlopen).
      * ``-O2 -Wall -Wextra`` — reasonable warnings + speed. The shim
        is tiny, so ``-O2`` is fine; ``-Werror`` is deliberately
        *off* so a future gcc version with new warnings doesn't break
        the wheel build.
      * ``-I<mpp_inc>`` — adds the Rockchip MPP public headers to the
        include search path.
      * ``-lrockchip_mpp`` — link against ``librockchip_mpp.so``
        directly. The shim calls into the upstream library's
        exported symbols (mpp_create / mpp_init / mpp_buffer_* /
        mpp_packet_* / …), so the upstream library needs to be
        present at link time and at run time. The build still
        works on a host without librockchip_mpp installed because
        gcc's ``-lrockchip_mpp`` only resolves the symbols the shim
        actually references (and produces a clear
        ``cannot find -lrockchip_mpp`` on a dev box that lacks the
        upstream library — instead of producing a silently-broken
        wheel that only fails at first use).
      * ``-Wl,-rpath,<mpp_lib_dir>`` — adds the runtime search path
        so the resulting ``libshim_video.so`` finds the upstream
        ``librockchip_mpp.so`` on hosts that don't have it in
        ``ldconfig``. ``mpp_lib_dir`` defaults to
        ``/usr/lib/aarch64-linux-gnu`` (Debian multiarch) and can be
        overridden with ``WDNAV_MPP_LIB``.

    Both encoder and decoder C sources are listed in the same gcc
    invocation so gcc links them into a single ``.so``. Order
    matters: ``-o`` must come after the sources so gcc parses them
    for ``-I`` first.
    """
    mpp_lib = os.environ.get("WDNAV_MPP_LIB", _DEFAULT_MPP_LIB)
    cmd: List[str] = [
        cc,
        "-shared",
        "-fPIC",
        "-O2",
        "-Wall",
        "-Wextra",
    ]
    if mpp_inc:
        cmd += [f"-I{mpp_inc}"]
    cmd += list(_C_SOURCES)
    cmd += [
        "-o", _OUTPUT_SO,
        "-lrockchip_mpp",
    ]
    if os.path.isdir(mpp_lib):
        cmd += [f"-Wl,-rpath,{mpp_lib}"]
    return cmd


def compile_shim(verbose: bool = False) -> None:
    """Compile ``cshim_video_encoder.c`` and ``cshim_video_decoder.c``
    into ``src/wdnav/lib/libshim_video.so``.

    Skipped silently when ``SKIP_WDNAV_VIDEO_BUILD=1`` is set in the
    environment — useful for readthedocs / sdist-only builds where the
    C toolchain isn't available and we just want to ship Python
    sources. A warning is printed to stderr in that case so a
    developer running ``pip install .`` by hand notices the gap.

    Raises :class:`RuntimeError` if the compile fails. The error
    message includes the full gcc command line and its stderr so the
    cause is obvious without re-running the build.
    """
    if os.environ.get("SKIP_WDNAV_VIDEO_BUILD") == "1":
        sys.stderr.write(
            "build_rkv: SKIP_WDNAV_VIDEO_BUILD=1, leaving "
            f"{_OUTPUT_SO} as-is (shim may be stale)\n"
        )
        return

    cc = _resolve_cc()
    if shutil.which(cc) is None:
        raise RuntimeError(
            f"build_rkv: C compiler {cc!r} not found on PATH; install "
            f"gcc or set CC=/path/to/cc"
        )

    mpp_inc = _resolve_mpp_inc()
    if mpp_inc and not os.path.isdir(mpp_inc):
        # Don't hard-fail here — the user may have set WDNAV_MPP_INC to
        # a path that doesn't exist on this machine, but the system
        # /usr/include/rockchip tree might still satisfy the includes.
        # Drop the -I flag and let gcc try its default search path.
        sys.stderr.write(
            f"build_rkv: warning: {mpp_inc} not found, falling back "
            f"to the compiler's default header search path\n"
        )
        mpp_inc = ""

    cmd = _build_command(cc, mpp_inc)
    if verbose:
        sys.stderr.write(
            "build_rkv: compiling C shim: " + " ".join(cmd) + "\n"
        )

    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as e:
        raise RuntimeError(
            f"build_rkv: failed to invoke C compiler {cc!r}: {e}"
        ) from e

    if proc.returncode != 0:
        raise RuntimeError(
            "build_rkv: libshim_video.so failed to compile.\n"
            f"  command: {' '.join(cmd)}\n"
            f"  gcc stderr:\n{proc.stderr.decode('utf-8', 'replace')}"
        )

    if not os.path.exists(_OUTPUT_SO):
        # gcc returned 0 but the .so isn't where we expected. Shouldn't
        # happen, but report a clean error instead of letting the
        # later ``package_data`` step silently miss it.
        raise RuntimeError(
            f"build_rkv: compile reported success but {_OUTPUT_SO} "
            f"is missing; check {cc}'s -o handling"
        )


# ---------------------------------------------------------------------------
# setuptools cmdclass integration
# ---------------------------------------------------------------------------


class build_py(_build_py):
    """Custom ``build_py`` that compiles the C shim before copying Python.

    We override :meth:`run` (rather than :meth:`build_package_data` or
    one of the more granular hooks) because:

      * :meth:`build_package_data` only fires for the data files
        already declared in ``package_data``; we need the .so on disk
        before that step walks the source tree.
      * :meth:`run` is the documented extension point for "I need to
        do something extra between finalizing options and copying
        Python sources", which is exactly our case.

    The compile is a no-op if the .so is already up to date *and* the
    source tree is older than the .so — a small touch that keeps
    ``pip install -e .`` fast on iterative development. The touch
    check is intentionally simple (``mtime`` only): the shim is a
    pair of small files, so a finer-grained hash is overkill.
    """

    user_options = _build_py.user_options + [
        ("skip-wdnav-video-build", None,
         "Skip the C shim compile (sets SKIP_WDNAV_VIDEO_BUILD=1)."),
    ]

    def initialize_options(self) -> None:
        super().initialize_options()
        self.skip_rkv_video_build: bool = False

    def finalize_options(self) -> None:
        super().finalize_options()
        if self.skip_rkv_video_build:
            os.environ["SKIP_WDNAV_VIDEO_BUILD"] = "1"

    def run(self) -> None:
        # Decide whether to skip the compile: an explicit opt-in, or
        # ``SKIP_WDNAV_VIDEO_BUILD`` already set in the environment (e.g.
        # by a CI step that pre-built the .so), wins. Otherwise
        # always recompile when the .so is missing or any of the
        # source files are newer than the .so.
        if not self.skip_rkv_video_build and os.environ.get("SKIP_WDNAV_VIDEO_BUILD") != "1":
            src_mtime = max(os.path.getmtime(p) for p in _C_SOURCES)
            needs_rebuild = (
                not os.path.exists(_OUTPUT_SO)
                or os.path.getmtime(_OUTPUT_SO) < src_mtime
            )
            if needs_rebuild:
                compile_shim(verbose=self.verbose)
            elif self.verbose:
                sys.stderr.write(
                    f"build_rkv: {_OUTPUT_SO} is up to date, "
                    f"skipping compile\n"
                )

        # Defer to the standard build_py so Python sources, package
        # data, and the freshly compiled .so all get copied into
        # build_lib and (later) into the wheel.
        super().run()
