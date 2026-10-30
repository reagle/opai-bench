"""Measure the sharpen/denoise quality tier of an output against its input.

What a tiled filter does to a photograph beyond its intended effect. Ported from the
ad-hoc scripts of the 2026-10-30 sharpen investigation (`an2.py`, `bands.py`,
`coarse.py`), which found Petersburg's invented mottle and the tile seams.

- **Seams** (`seam`, `seam_null`, `seam_excess`, `seam_p90`, `tile_sd`): the step in
  the residual (output minus input luma, over flat pixels) across each internal seam
  of the model's tile grid, against the same step half a stride away, mid-tile, where
  no seam is. A seam shows as `seam` well above `seam_null`; `tile_sd` is the spread
  of the tiles' mean residual (a tile that drifts brighter).
- **Mottle** (`mottle`): the residual low-passed at sigma 3 px, its 1st to 99th
  percentile spread over flat pixels, in levels: invented low-frequency texture. About
  one level is the edge of visibility on a smooth gradient.
- **Bands** (`band_rms_*`, `band_gain_*`, `band_corr_*` over flat pixels;
  `edge_gain_*` over strong edges): the image split by wavelength (2--4, 4--8, 8--16,
  16--32, 32--64 px and over 64; 2 px is the Nyquist limit, so there is no finer
  band). Per band, the residual's RMS, the output's RMS over the input's (gain), and
  the residual's correlation with the input's own band: amplified detail correlates,
  invented detail does not. `edge_gain_2_8` is what sharpening does at edges (above 1
  sharpens, below 1 softens).
- **Invention** (`inv_max`, `inv_p90`, `inv_median`): per tile of the grid, the RMS of
  the change finer than a radius-16 box applied twice, over the input's own detail at
  that scale plus one 8-bit level (the app's `filter::invention`). The app's guard
  judges the raw model output with it; this sees the final photograph, after the
  anchor and the blend.
"""

__author__ = "Joseph Reagle"
__copyright__ = "Copyright (C) 2026 Joseph Reagle"
__license__ = "MIT"

import itertools
import math

import numpy as np
import scipy.fft
from scipy import ndimage

from opai_bench.metrics.pixels import EDGE_GRADIENT, flat_mask, gradient, luma

BAND_EDGES = (2, 4, 8, 16, 32, 64, math.inf)
EDGE_BANDS = ((2, 4), (4, 8), (8, 16), (16, 32), (2, 8))
INVENTION_RADIUS = 16
INVENTION_FLOOR = 1.0 / 255.0
# The flat mask is eroded this far for the band measures, so an edge's halo does not
# leak into them.
BAND_EROSION = 16
# The strips either side of a seam that are compared, and the margin kept from the
# tile's top and bottom (its overlaps with the tiles above and below).
STRIP = 16
MARGIN = 24
# A tile's middle, away from its overlaps, for its mean residual.
INSET = 40


def band_name(low: float, high: float) -> str:
    """Return a band's name in a measure's key.

    >>> band_name(2, 4), band_name(64, math.inf)
    ('2_4', '64_up')
    """
    return f"{low:g}_{'up' if math.isinf(high) else f'{high:g}'}"


def tile_offsets(length: int, size: int, overlap: int) -> list[int]:
    """Return where the tiles of a grid start along one side (the app's `run_tiled`).

    Every `size - overlap`, the last moved back to end at the edge.

    >>> tile_offsets(600, 256, 16)
    [0, 240, 344]
    >>> tile_offsets(200, 256, 16)
    [0]
    """
    if length <= size:
        return [0]
    stride, last, out, position = size - overlap, length - size, [], 0
    while position < length:
        start = min(position, last)
        if out and out[-1] == start:
            break
        out.append(start)
        if start + size >= length:
            break
        position += stride
    return out


def masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    """Return the mean of `values` where `mask` is set, or NaN over too few pixels."""
    weight = mask.sum()
    return float((values * mask).sum() / weight) if weight > 30 else math.nan


def seam_steps(
    residual: np.ndarray,
    mask: np.ndarray,
    grid: tuple[list[int], list[int]],
    size: int,
    shift: int = 0,
) -> np.ndarray:
    """Return the step in `residual` across each internal vertical seam, per tile row.

    The flat mean over the `STRIP` px before the seam against the `STRIP` px after the
    overlap's ramp. `grid` is the tiles' (x, y) offsets; transpose the planes and swap
    the grid for the horizontal seams.
    """
    xs, ys = grid
    steps = []
    width = residual.shape[1]
    for k in range(1, len(xs)):
        ramp = min(xs[k - 1] + size - xs[k], size // 2)
        x = xs[k] + shift
        if x - STRIP < 0 or x + ramp + STRIP > width:
            continue
        near, far = slice(x - STRIP, x), slice(x + ramp, x + ramp + STRIP)
        for y in ys:
            rows = slice(y + MARGIN, y + size - MARGIN)
            before = masked_mean(residual[rows, near], mask[rows, near])
            after = masked_mean(residual[rows, far], mask[rows, far])
            if not math.isnan(before - after):
                steps.append(abs(after - before))
    return np.array(steps)


def seams(residual: np.ndarray, flat: np.ndarray, size: int, overlap: int) -> dict:
    """Return the seam steps against the mid-tile steps, and the tiles' spread.

    A step of 1.5 levels at every vertical seam shows (9 of the 17 seams here):

    >>> plane = np.zeros((600, 976))
    >>> for x in tile_offsets(976, 256, 16)[1:]:
    ...     plane[:, x + 8 :] += 1.5
    >>> found = seams(plane, np.ones(plane.shape, bool), 256, 16)
    >>> round(found["seam_excess"], 2), round(found["seam_null"], 2)
    (0.79, 0.0)
    """
    height, width = residual.shape
    xs, ys = tile_offsets(width, size, overlap), tile_offsets(height, size, overlap)
    mask = flat.astype(float)
    middle = -((size - overlap) // 2)
    seam, null = [], []
    for shift, found in ((0, seam), (middle, null)):
        found += [
            seam_steps(residual, mask, (xs, ys), size, shift),
            seam_steps(residual.T, mask.T, (ys, xs), size, shift),
        ]
    seam_all, null_all = np.concatenate(seam), np.concatenate(null)
    out: dict[str, float] = {}
    if len(seam_all) and len(null_all):
        out |= {
            "seam": float(seam_all.mean()),
            "seam_null": float(null_all.mean()),
            "seam_excess": float(seam_all.mean() - null_all.mean()),
            "seam_p90": float(np.percentile(seam_all, 90)),
        }
    inner = slice(INSET, size - INSET)
    tiles = [
        masked_mean(
            residual[y : y + size, x : x + size][inner, inner],
            mask[y : y + size, x : x + size][inner, inner],
        )
        for y in ys
        for x in xs
    ]
    if np.isfinite(tiles).sum() > 1:
        out["tile_sd"] = float(np.nanstd(tiles))
    return out


def wavelengths(shape: tuple[int, int]) -> np.ndarray:
    """Return the wavelength in pixels of each coefficient of a plane's real FFT."""
    fy = np.fft.fftfreq(shape[0])[:, None]
    fx = np.fft.rfftfreq(shape[1])[None, :]
    return 1 / np.maximum(np.hypot(fy, fx), 1e-9)


def spectrum(plane: np.ndarray) -> np.ndarray:
    """Return a plane's real FFT, mean removed, in single precision."""
    centred = (plane - plane.mean()).astype(np.float32)
    return np.asarray(scipy.fft.rfft2(centred, workers=-1))


def band(
    spectrum_: np.ndarray, lam: np.ndarray, low: float, high: float, shape
) -> np.ndarray:
    """Return the part of a plane (by its spectrum) between two wavelengths."""
    kept = spectrum_ * ((lam >= low) & (lam < high))
    return np.asarray(scipy.fft.irfft2(kept, s=shape, workers=-1))


def rms(values: np.ndarray) -> float:
    """Return the root mean square of `values`."""
    return float(np.sqrt(np.mean(values.astype(np.float64) ** 2)))


def flat_bands(before: np.ndarray, residual: np.ndarray, eroded: np.ndarray) -> dict:
    """Return each band's residual RMS, gain and correlation over flat pixels; mottle.

    Invented low-frequency texture is mottle, uncorrelated with the input:

    >>> from opai_bench.synthetic import gradient as synthetic
    >>> plane = luma(synthetic(800, 600))
    >>> noise = np.random.default_rng(1).normal(0, 1, plane.shape)
    >>> field = ndimage.gaussian_filter(noise, 12)
    >>> found = flat_bands(plane, field * 2.0 / field.std(), np.ones(plane.shape, bool))
    >>> found["mottle"] > 3, abs(found["band_corr_16_32"]) < 0.2
    (True, True)
    """
    out: dict[str, float] = {}
    shape = before.shape
    lam = wavelengths(shape)
    # The image restricted to its flat pixels before filtering, so edges do not leak.
    smooth = ndimage.gaussian_filter(before, 8)
    flat_input = spectrum(np.where(eroded, before, smooth))
    flat_residual = spectrum(np.where(eroded, residual, 0.0))
    for low, high in itertools.pairwise(BAND_EDGES):
        a = band(flat_input, lam, low, high, shape)[eroded]
        r = band(flat_residual, lam, low, high, shape)[eroded]
        name = band_name(low, high)
        out[f"band_rms_{name}"] = rms(r)
        if rms(a) > 0:
            out[f"band_gain_{name}"] = rms(a + r) / rms(a)
        if a.std() > 0 and r.std() > 0:
            out[f"band_corr_{name}"] = float(np.corrcoef(a, r)[0, 1])
    low_passed = np.where(eroded, residual, 0.0)
    low_passed = np.asarray(ndimage.gaussian_filter(low_passed, 3))[eroded]
    spread = np.percentile(low_passed, 99) - np.percentile(low_passed, 1)
    out["mottle"] = float(spread)
    return out


def edge_gains(before: np.ndarray, after: np.ndarray) -> dict:
    """Return each band's gain, output over input, over the input's strong edges."""
    edges = gradient(before) > EDGE_GRADIENT
    out = {"edge_share": float(edges.mean())}
    if edges.sum() <= 100:
        return out
    shape = before.shape
    lam = wavelengths(shape)
    whole_in, whole_out = spectrum(before), spectrum(after)
    for low, high in EDGE_BANDS:
        a = band(whole_in, lam, low, high, shape)[edges]
        if rms(a) > 0:
            b = band(whole_out, lam, low, high, shape)[edges]
            out[f"edge_gain_{band_name(low, high)}"] = rms(b) / rms(a)
    return out


def box_twice(plane: np.ndarray, radius: int) -> np.ndarray:
    """Return a box mean of `radius` applied twice (a tent), normalised at the edges."""
    size = 2 * radius + 1
    counts = ndimage.uniform_filter(np.ones_like(plane), size, mode="constant")
    for _ in range(2):
        plane = ndimage.uniform_filter(plane, size, mode="constant") / counts
    return plane


def invention(before: np.ndarray, after: np.ndarray) -> float:
    """Return the invention ratio of one tile (RGB in levels).

    The RMS of the change finer than `box_twice` at `INVENTION_RADIUS`, over the
    input's own such detail plus `INVENTION_FLOOR`.

    >>> rng = np.random.default_rng(0)
    >>> tile = np.full((64, 64, 3), 128.0)
    >>> invention(tile, tile)
    0.0
    >>> invention(tile, tile + rng.normal(0, 10, tile.shape)) > 5  # speckle on flat
    True
    """
    added = own = 0.0
    for channel in range(3):
        a, b = before[..., channel] / 255, after[..., channel] / 255
        change = b - a
        added += float(((change - box_twice(change, INVENTION_RADIUS)) ** 2).sum())
        own += float(((a - box_twice(a, INVENTION_RADIUS)) ** 2).sum())
    count = before.size
    return math.sqrt(added / count) / (math.sqrt(own / count) + INVENTION_FLOOR)


def inventions(before: np.ndarray, after: np.ndarray, size: int, overlap: int) -> dict:
    """Return the invention ratio over every tile: largest, 90th percentile, median."""
    height, width = before.shape[:2]
    ratios = [
        invention(before[y : y + size, x : x + size], after[y : y + size, x : x + size])
        for y in tile_offsets(height, size, overlap)
        for x in tile_offsets(width, size, overlap)
    ]
    return {
        "inv_max": float(np.max(ratios)),
        "inv_p90": float(np.percentile(ratios, 90)),
        "inv_median": float(np.median(ratios)),
    }


def measures(
    before: np.ndarray, after: np.ndarray, size: int = 256, overlap: int = 16
) -> dict[str, float]:
    """Return every measure of the tier for `after` against `before` (8-bit RGB).

    >>> from opai_bench.synthetic import gradient as synthetic
    >>> image = synthetic(600, 400)
    >>> same = measures(image, image)
    >>> same["mottle"], same["seam_excess"], same["inv_max"]
    (0.0, 0.0, 0.0)
    """
    a, b = luma(before), luma(after)
    residual = b - a
    flat = flat_mask(a)
    out = {"flat_share": float(flat.mean())}
    if flat.any():
        out["dc"] = masked_mean(residual, flat.astype(float))
    out |= seams(residual, flat, size, overlap)
    eroded = flat
    if flat.mean() < 0.99:
        eroded = np.asarray(ndimage.binary_erosion(flat, iterations=BAND_EROSION))
    if eroded.sum() > 1000:
        out |= flat_bands(a, residual, eroded)
    out |= edge_gains(a, b)
    out |= inventions(
        before.astype(np.float64), after.astype(np.float64), size, overlap
    )
    return {key: value for key, value in out.items() if not math.isnan(value)}
