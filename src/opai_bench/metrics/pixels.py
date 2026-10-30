"""Score two renderings of one photograph against each other, pixel by pixel.

FP16 against FP32, one model against another or against the input: overall, and split
into flat and detailed areas. Images are 8-bit RGB arrays; differences are in 8-bit
levels (0 to 255) over all three channels, SSIM on luma.
"""

__author__ = "Joseph Reagle"
__copyright__ = "Copyright (C) 2026 Joseph Reagle"
__license__ = "MIT"

from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

LUMA = np.array([0.299, 0.587, 0.114])
# Flat: where the input's smoothed gradient (Sobel of the luma at sigma 2, in levels)
# is under this over a whole 15-px neighbourhood --- a sky, a wall, a backdrop. The
# measure of the 2026-10-30 sharpen investigation (`an2.py`).
FLAT_GRADIENT = 8.0
FLAT_WINDOW = 15
# Strong edges, for the edge gains of the quality tier: smoothed gradient above this.
EDGE_GRADIENT = 60.0
# A PSNR for identical images, so means and limits stay finite.
PSNR_IDENTICAL = 100.0


def load(path: str | Path) -> np.ndarray:
    """Return an image file as 8-bit RGB."""
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def luma(image: np.ndarray) -> np.ndarray:
    """Return the Rec. 601 luma of an RGB image, as float64 levels.

    >>> float(luma(np.full((1, 1, 3), 200, np.uint8))[0, 0])
    200.0
    """
    return image.astype(np.float64) @ LUMA


def gradient(plane: np.ndarray) -> np.ndarray:
    """Return the Sobel gradient magnitude of a plane smoothed at sigma 2."""
    smooth = ndimage.gaussian_filter(plane, 2)
    return np.hypot(ndimage.sobel(smooth, 0), ndimage.sobel(smooth, 1))


def flat_mask(plane: np.ndarray) -> np.ndarray:
    """Return where a luma plane is flat (see `FLAT_GRADIENT`).

    >>> plane = np.zeros((40, 80)); plane[:, 40:] = 100
    >>> mask = flat_mask(plane)
    >>> bool(mask[20, 5]), bool(mask[20, 40])
    (True, False)
    """
    flat = (gradient(plane) < FLAT_GRADIENT).astype(float)
    return np.asarray(ndimage.uniform_filter(flat, FLAT_WINDOW) > 0.99)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """Return the peak signal-to-noise ratio in dB of two 8-bit images.

    `PSNR_IDENTICAL` where they are equal.

    >>> a = np.zeros((4, 4, 3), np.uint8); b = a.copy(); b[0, 0, 0] = 16
    >>> round(psnr(a, b), 2), psnr(a, a)
    (40.86, 100.0)
    """
    difference = a.astype(np.float64) - b.astype(np.float64)
    error = float(np.mean(difference**2)) if a.size else 0.0
    if error == 0:
        return PSNR_IDENTICAL
    return min(PSNR_IDENTICAL, float(10 * np.log10(255.0**2 / error)))


def ssim_map(a: np.ndarray, b: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """Return the SSIM index at each pixel of two planes in 8-bit levels.

    Wang, Bovik, Sheikh and Simoncelli (2004), with their Gaussian window (sigma 1.5)
    and constants (K1 0.01, K2 0.03, L 255).

    >>> rng = np.random.default_rng(0)
    >>> a = rng.uniform(0, 255, (32, 32))
    >>> float(ssim_map(a, a).mean())
    1.0
    >>> float(ssim_map(a, 255 - a).mean()) < 0
    True
    """
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2

    def blur(plane: np.ndarray) -> np.ndarray:
        return ndimage.gaussian_filter(plane, sigma, truncate=3.5)

    mean_a, mean_b = blur(a), blur(b)
    var_a = blur(a * a) - mean_a**2
    var_b = blur(b * b) - mean_b**2
    covariance = blur(a * b) - mean_a * mean_b
    numerator = (2 * mean_a * mean_b + c1) * (2 * covariance + c2)
    return numerator / ((mean_a**2 + mean_b**2 + c1) * (var_a + var_b + c2))


def compare(reference: np.ndarray, test: np.ndarray, flat: np.ndarray) -> dict:
    """Score `test` against `reference` (same size), overall, flat and detailed.

    The largest and mean absolute difference, PSNR and SSIM, over the whole photograph,
    over the `flat` mask (`_flat`) and over the rest (`_detail`); and the flat share.

    >>> reference = np.full((32, 32, 3), 100, np.uint8)
    >>> test = reference.copy(); test[0, 0] = 110
    >>> flat = np.ones((32, 32), bool); flat[:, 16:] = False
    >>> scores = compare(reference, test, flat)
    >>> scores["max_abs"], scores["max_abs_detail"], scores["flat_share"]
    (10.0, 0.0, 0.5)

    Rounding noise of one level, as FP16 against FP32 often is, scores high:

    >>> from opai_bench.synthetic import gradient
    >>> image = gradient(400, 300)
    >>> noise = np.random.default_rng(0).integers(-1, 2, image.shape)
    >>> noisy = np.clip(image.astype(int) + noise, 0, 255).astype(np.uint8)
    >>> scores = compare(image, noisy, flat_mask(luma(image)))
    >>> scores["max_abs"], scores["psnr"] > 45, scores["ssim"] > 0.95
    (1.0, True, True)
    """
    if reference.shape != test.shape:
        raise ValueError(f"sizes differ: {reference.shape} against {test.shape}")
    difference = np.abs(reference.astype(np.int16) - test.astype(np.int16))
    structure = ssim_map(luma(reference), luma(test))
    scores = {"flat_share": float(flat.mean())}
    masks = (("", np.ones(flat.shape, bool)), ("_flat", flat), ("_detail", ~flat))
    for suffix, mask in masks:
        if not mask.any():
            continue
        scores |= {
            f"max_abs{suffix}": float(difference[mask].max()),
            f"mean_abs{suffix}": float(difference[mask].mean()),
            f"psnr{suffix}": psnr(reference[mask], test[mask]),
            f"ssim{suffix}": float(structure[mask].mean()),
        }
    return scores


def resized(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Return `image` resized to `size` (width, height) with Lanczos.

    The input an upscale is compared against.
    """
    if (image.shape[1], image.shape[0]) == size:
        return image
    resample = Image.Resampling.LANCZOS
    return np.asarray(Image.fromarray(image).resize(size, resample))
