"""Make the built-in synthetic test image: a smooth colour gradient with fine grain.

It is where a sharpener's or denoiser's invented texture, tile seams and mottle show
most, and it needs nobody's photograph. At its default size and seed it is the
`synth.png` of the 2026-10-30 sharpen investigation, pixel for pixel.
"""

__author__ = "Joseph Reagle"
__copyright__ = "Copyright (C) 2026 Joseph Reagle"
__license__ = "MIT"

from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from opai_bench.config import data_dir

SIZE = (3000, 2000)


def gradient(width: int = SIZE[0], height: int = SIZE[1], seed: int = 1) -> np.ndarray:
    """Return an 8-bit RGB gradient with grain of about 1.5 levels, slightly soft.

    >>> image = gradient(64, 48)
    >>> image.shape, image.dtype.name
    ((48, 64, 3), 'uint8')
    >>> bool((gradient(64, 48) == image).all())  # deterministic
    True
    """
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    base = np.stack(
        [
            90 + 80 * x / width + 20 * y / height,
            120 + 60 * y / height,
            170 - 40 * x / width,
        ],
        -1,
    )
    grain = rng.normal(0, 2.0, (height, width, 3))
    image = base + ndimage.gaussian_filter(grain, (0.7, 0.7, 0)) * 1.5
    return np.clip(np.round(image), 0, 255).astype(np.uint8)


def ensure(width: int = SIZE[0], height: int = SIZE[1]) -> Path:
    """Return the gradient as a PNG under the data folder, made the first time."""
    path = data_dir() / "synthetic" / f"gradient-{width}x{height}.png"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(gradient(width, height)).save(path)
    return path
