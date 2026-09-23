"""Text-overlay demo for wdnav.Typer (Surface subclass).

Renders a synthetic 640x480 BGR frame and overlays several text rows on
top using :class:`wdnav.Typer`, demonstrating:

* default font (fontconfig ``sans`` → wqy-zenhei on this board),
* explicit fontconfig family name (``"mono"``),
* explicit .ttf / .otf file path (resolved via fc-match for portability),
* Chinese / English mix with size + colour overrides.

Output goes to ``/tmp/out_typer.png`` by default. The PNG is written by
the bundled ``_png.py`` (no cv2 / PIL dependency).

    python samples/typer_demo.py [out.png]
"""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np

import wdnav

from _png import write_png_bgr


def main() -> int:
    out_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/out_typer.png"
    if wdnav.SurfaceComposer is None:
        print("libmali not loadable; SurfaceComposer is None", file=sys.stderr)
        return 1
    if wdnav.Typer is None:
        print("Typer unavailable", file=sys.stderr)
        return 1

    sf = wdnav.SurfaceComposer(w=640, h=480)

    # Background plane — flat dark gray (zpos=0). We use a flat color
    # because the pre-existing BGR/RGB byte-order mismatch in
    # ``rkm_texture_upload_rgb`` swaps R/B on the GPU readback path;
    # on a flat dark-gray surface that's invisible, and the Typer text
    # shows up cleanly against it.
    bg = sf.create_surface(0, 0, 640, 480, zpos=0, fmt="BGR888")
    bg.latest_ndarray[:] = (40, 40, 40)
    bg.post(pts=0)

    # OSD strip on top (zpos=1) — uses Typer (freetype via c-shim).
    typer = sf.create_typer(
        0, 0, 640, 480, zpos=1,
        font="sans",            # fontconfig default
        size=28,
        color=(255, 255, 255),  # white BGR
    )
    print(f"typer.font_path = {typer.font_path}")
    print(f"sans size 28 bbox of 'wdnav' = {typer.textsize('wdnav')}")

    # Layout: a few lines, baseline at descending y offsets. freetype
    # draws ABOVE the baseline, so size=N text needs y >= N.
    typer.text("wdnav SurfaceComposer — Mali-G610 PBuffer + DMA-BUF FBO",
               xy=(12, 28))
    typer.text("Typer: Surface subclass, freetype via c-shim (rkm_text)",
               xy=(12, 64), color=(0, 255, 255), size=22)

    # fontconfig alias for monospace
    typer.text("mono: ./samples/typer_demo.py /tmp/out.png",
               xy=(12, 100), font="mono", color=(0, 0, 0),
               bg_color=(255, 255, 255), size=18)

    # CJK (freetype + system wqy-zenhei handles these on this board).
    typer.text("中文: 字体光栅化走 freetype + c-shim",
               xy=(12, 134), size=22, color=(255, 255, 200))

    # Big colored text
    typer.text("BIG 42", xy=(12, 188), size=42, color=(0, 140, 255))

    # Demonstrate passing an explicit font file (resolved through fc-match
    # so the demo runs on any distro without hard-coding paths). DejaVu
    # Sans Bold is shipped with the Debian/Ubuntu ``fonts-dejavu`` package
    # — the fc-match call returns nothing on boards without it, and we
    # silently fall back to the system sans in that case.
    dejavu = subprocess.run(
        ["fc-match", "-f", "%{file}\n", "DejaVu Sans:style=Bold"],
        capture_output=True, text=True, check=False,
    ).stdout.strip()
    if dejavu and os.path.isfile(dejavu):
        typer.text(f"explicit path: {os.path.basename(dejavu)}",
                   xy=(12, 252), font=dejavu, size=22,
                   color=(0, 255, 0))
    else:
        typer.text("(DejaVu not on this system — skipping explicit path)",
                   xy=(12, 252), size=20, color=(180, 180, 180))

    # Yellow line near the bottom of the OSD strip
    typer.text("frame #0   pts=0",
               xy=(12, 290), size=18, color=(0, 255, 255))

    typer.post(pts=0)

    # Give the worker a moment to drain both posts before reading.
    import time
    time.sleep(0.3)

    img = sf.read(timeout=5.0)
    if img.shape != (480, 640, 3):
        print(f"FAIL: unexpected shape {img.shape}", file=sys.stderr)
        sf.close()
        return 1

    # Sanity check: white text was rendered onto the dark-gray bg, so
    # we expect bright pixels at the text row positions.
    # Background was 40 (flat gray); text rows have pixels well above 200.
    text_band = img[20:300, :]
    bright = ((text_band[..., 0] > 200) & (text_band[..., 1] > 200) &
              (text_band[..., 2] > 200)).sum()
    print(f"bright pixel count in text band: {bright}")
    if bright < 100:
        print("FAIL: text band looks empty — Typer probably did not draw",
              file=sys.stderr)
        sf.close()
        return 1

    write_png_bgr(out_path, img)
    print(f"OK: composited frame with Typer overlay written to {out_path}")
    sf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())