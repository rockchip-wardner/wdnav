"""Minimal PNG writer using only stdlib + numpy (no cv2/PIL)."""
import struct
import zlib
import numpy as np


def write_png_bgr(path: str, img_bgr: np.ndarray) -> None:
    if img_bgr.dtype != np.uint8:
        raise ValueError(f"expected uint8, got {img_bgr.dtype}")
    if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3) BGR, got {img_bgr.shape}")
    H, W = img_bgr.shape[:2]
    # PNG expects RGB rows; our array is BGR — swap channels.
    rgb = img_bgr[..., ::-1].copy()
    # Add filter-byte 0 (None) to each row.
    raw = np.empty((H, 1 + W * 3), dtype=np.uint8)
    raw[:, 0] = 0  # filter type 0 = None
    raw[:, 1:] = rgb.reshape(H, W * 3)

    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    png = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0)  # 8-bit RGB
    idat = zlib.compress(raw.tobytes(), 6)
    with open(path, "wb") as f:
        f.write(png)
        f.write(chunk(b"IHDR", ihdr))
        f.write(chunk(b"IDAT", idat))
        f.write(chunk(b"IEND", b""))
