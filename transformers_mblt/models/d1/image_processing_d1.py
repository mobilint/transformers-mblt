"""Host-side picture preprocessing for the D1 vision MXQ (Pillow and NumPy only).

A picture is letterboxed into one ``image_size`` square: scaled with bicubic resampling so that it fits while keeping
its aspect ratio, and centred on black. Its RGB values are normalized to ``[-1, 1]`` (mean 0.5, std 0.5) and cut into
``patch_size`` patches in row-major order, each flattened as ``(row, column, channel)``. The vision MXQ maps the
``(1, patches, patch_size**2 * 3)`` array to one feature row per image token.

This replaces upstream's tiling processor (up to ten 512 tiles plus a thumbnail) with the single fixed-size input the
vision MXQ was compiled for: every picture becomes the same number of image tokens.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

_HALF = np.float32(127.5)


def load_image(image: Any) -> Any:
    """``image`` as an RGB ``PIL.Image``: a ``PIL.Image``, a file path, or an ``(height, width[, 3])`` uint8 array."""
    from PIL import Image

    if isinstance(image, (str, os.PathLike)):
        with Image.open(image) as opened:
            return opened.convert("RGB")
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8 or image.ndim not in (2, 3):
            raise TypeError(f"an image array must be uint8 (height, width[, channels]), got {image.dtype} {image.shape}")
        return Image.fromarray(image).convert("RGB")
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    raise TypeError(f"an image is a PIL.Image, a file path or a uint8 array, got {type(image).__name__}")


def letterbox(image: Any, size: int) -> Any:
    """``image`` scaled to fit a ``size`` square (bicubic, aspect kept) and centred on black."""
    from PIL import Image, ImageOps

    fitted = ImageOps.contain(image.convert("RGB"), (size, size), method=Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(fitted, ((size - fitted.width) // 2, (size - fitted.height) // 2))
    return canvas


def patchify(image: Any, size: int, patch: int) -> np.ndarray:
    """The vision MXQ input of one picture: ``(1, (size // patch) ** 2, patch * patch * 3)`` float32 in ``[-1, 1]``."""
    if size % patch:
        raise ValueError(f"image size {size} is not a multiple of the patch size {patch}")
    rgb = np.asarray(letterbox(load_image(image), size), dtype=np.float32)
    normalized = (rgb - _HALF) / _HALF
    grid = size // patch
    patches = normalized.reshape(grid, patch, grid, patch, 3).transpose(0, 2, 1, 3, 4)
    return np.ascontiguousarray(patches.reshape(1, grid * grid, patch * patch * 3))
