"""Score the lens blur: its matte against AIM-500's, its depth against Spring's.

The subject's matte is scored against a hand-made one, the depth map against stereo
disparity.
"""

__author__ = "Joseph Reagle"
__copyright__ = "Copyright (C) 2026 Joseph Reagle"
__license__ = "MIT"

from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

BOUNDARY_RATIOS = (1.05, 1.10, 1.15, 1.20, 1.25)
# How much blur, in pixels of radius, the background beside the subject must get for
# that stretch of its outline to be scored: where it blurs less, the subject's edge is
# not the edge of what is sharp (a floor at the seat's distance, a kite against the
# sky beyond the hyperfocal distance), and the ground truth's object outline says
# nothing of the blur.
SEPARATED_PX = 2.0
# And how little the subject beside it must blur: a part of the subject nearer or
# farther than the focus (a sofa's arm seen from above) rightly blurs, and its outline
# in the ground truth is not the edge of what is sharp either.
SHARP_PX = 1.0
BILINEAR = Image.Resampling.BILINEAR
# Pillow's default for `resize`, which the scene's depth was always read with.
BICUBIC = Image.Resampling.BICUBIC


def dilate(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Return a binary dilation of `mask`."""
    return np.asarray(ndimage.binary_dilation(mask, iterations=iterations))


def erode(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Return a binary erosion of `mask`."""
    return np.asarray(ndimage.binary_erosion(mask, iterations=iterations))


def resize(plane: np.ndarray, shape: tuple[int, ...], resample=BILINEAR) -> np.ndarray:
    """Return a float plane resized to `shape` (height, width)."""
    size = (shape[1], shape[0])
    return np.asarray(Image.fromarray(plane).resize(size, resample))


def focus_inside(truth: np.ndarray) -> tuple[float, float]:
    """Return the subject's point farthest from its edge, as fractions of the frame.

    >>> truth = np.zeros((10, 20)); truth[2:8, 10:18] = 1
    >>> focus_inside(truth)
    (0.6, 0.4)
    """
    inside = np.asarray(ndimage.distance_transform_edt(truth > 0.95))
    y, x = np.unravel_index(int(np.argmax(inside)), inside.shape)
    return round(int(x) / truth.shape[1], 3), round(int(y) / truth.shape[0], 3)


def separated(
    error: np.ndarray, truth: np.ndarray, radius: np.ndarray, band, ring: int
) -> dict:
    """Return the `sep_` errors and the scene's depth, along the edge of what is sharp.

    Only the stretches of the outline where what lies just outside blurs by at least
    `SEPARATED_PX` and what lies just inside by under `SHARP_PX` (`radius`, the blur's
    radius at each pixel of the depth map) are scored: there the subject's edge is the
    edge of what is sharp, and the matte's edge quality shows. `sep_share` is the
    share of the outline that qualifies.
    """
    blurred = np.abs(resize(radius, truth.shape))
    subject = truth > 0.5
    outside = dilate(subject, iterations=2 * ring)
    outside &= ~dilate(subject, iterations=ring)
    inside = erode(subject, iterations=ring)
    inside &= ~erode(subject, iterations=2 * ring)
    apart = outside & (blurred >= SEPARATED_PX)
    sharp = inside & (blurred < SHARP_PX)
    scored = band & dilate(apart, iterations=2 * ring)
    scored &= dilate(sharp, iterations=2 * ring)
    scores = {
        "sep_share": float(scored.sum() / max(band.sum(), 1)),
        "sep_sad": float(np.abs(error[scored]).sum() / 1000),
        "sep_mse": float((error[scored] ** 2).mean()) if scored.any() else 0.0,
    }
    return scores | scene_depth(resize(radius, truth.shape, BICUBIC), truth, ring)


def matte_gradient(plane: np.ndarray) -> np.ndarray:
    """Return a matte's gradient magnitude at sigma 1.4, as matting benchmarks do."""
    # The stubs say `order` is an int; scipy takes one per axis too.
    across = ndimage.gaussian_filter(plane, 1.4, order=(0, 1))  # pyright: ignore[reportArgumentType]
    down = ndimage.gaussian_filter(plane, 1.4, order=(1, 0))  # pyright: ignore[reportArgumentType]
    return np.hypot(across, down)


def specks(matte: np.ndarray, truth: np.ndarray, ring: int) -> float:
    """Return the number of small islands where matte and truth disagree."""
    disagree = (matte > 0.5) ^ (truth > 0.5)
    labels, count = ndimage.label(disagree)  # pyright: ignore[reportGeneralTypeIssues]
    if not count:
        return 0.0
    sizes = np.bincount(np.asarray(labels, np.int64).ravel())[1:]
    speck_area = (ring // 4) ** 2 or 1
    return float(((sizes >= 4) & (sizes < max(speck_area, 16))).sum())


def score_matte(
    matte: np.ndarray | None, truth: np.ndarray, radius: np.ndarray | None
) -> dict:
    """Return the matting errors of `matte` against `truth`.

    The usual ones (SAD and gradient in thousands), the squared error within the
    truth's edge band, specks, and the `sep_` errors (`separated`). A lens keeps sharp
    whatever is at the subject's distance, not the person alone (a truck beside her
    stays sharp), so the whole-frame errors mostly measure that.
    """
    if matte is None:
        return {"matted": False}
    if matte.shape != truth.shape:
        truth = resize(truth, matte.shape)
    matte = np.clip(matte, 0, 1)
    error = matte - truth
    ring = max(1, round(0.007 * max(truth.shape)))
    edge = (truth > 0.01) & (truth < 0.99)
    edge |= dilate(truth > 0.5) ^ erode(truth > 0.5)
    band = dilate(edge, iterations=ring)
    scores = {} if radius is None else separated(error, truth, radius, band, ring)
    gradient_error = (matte_gradient(matte) - matte_gradient(truth)) ** 2
    return scores | {
        "matted": True,
        "sad": float(np.abs(error).sum() / 1000),
        "mse": float((error**2).mean()),
        "grad": float(gradient_error.sum() / 1000),
        "edge_mse": float((error[band] ** 2).mean()) if band.any() else 0.0,
        "specks": specks(matte, truth, ring),
    }


def scene_depth(radius: np.ndarray, truth: np.ndarray, ring: int) -> dict:
    """Return what the photograph's depth gives a lens blur to do.

    From the signed blur `radius` (positive nearer than the focus, negative beyond),
    for choosing test photographs rather than scoring a build:

    - `subject_blur`: the blur the subject's own nearest or farthest tenth would get
      (a face sharp, the feet not).
    - `behind_median`, `behind_far`: the background's blur behind the focus, at its
      middle and its farthest tenth; far apart where the background recedes (ground,
      a floor, a room) rather than standing at one distance.
    - `front_share`: the share of the background nearer than the focus and blurred,
      something in front of the subject.
    """
    subject = erode(truth > 0.5, iterations=ring)
    background = truth < 0.05
    behind = -radius[background & (radius < 0)]
    front = (background & (radius >= SEPARATED_PX)).sum() / max(background.sum(), 1)
    own = np.percentile(np.abs(radius[subject]), 90) if subject.any() else 0.0
    return {
        "subject_blur": float(own),
        "behind_median": float(np.median(behind)) if behind.size else 0.0,
        "behind_far": float(np.percentile(behind, 90)) if behind.size else 0.0,
        "front_share": float(front),
    }


def read_disparity(path: Path) -> np.ndarray:
    """Return a Spring disparity map (HDF5)."""
    try:
        import h5py  # no wheels for Python 3.15 yet; see pyproject.toml
    except ImportError as error:
        raise SystemExit(
            "Spring's disparity needs h5py, which has no wheels for this Python yet: "
            "run with Python 3.14 (`uv run --python 3.14 opai-bench ...`) or --only aim"
        ) from error
    with h5py.File(path, "r") as file:
        return np.asarray(file["disparity"], np.float32)


def boundaries(depth: np.ndarray, ratio: float) -> np.ndarray:
    """Return occlusion boundaries, Depth Pro style.

    A pixel nearer than a 4-neighbour by more than `ratio`.

    >>> depth = np.ones((3, 4)); depth[:, 2:] = 2.0
    >>> boundaries(depth, 1.1).astype(int)[0].tolist()
    [0, 1, 0, 0]
    """
    near = np.zeros(depth.shape, bool)
    for axis in (0, 1):
        a, b = np.moveaxis(depth, axis, 0)[:-1], np.moveaxis(depth, axis, 0)[1:]
        view = np.moveaxis(near, axis, 0)
        view[:-1] |= b / a > ratio
        view[1:] |= a / b > ratio
    return near


def boundary_f1(fitted: np.ndarray, truth: np.ndarray, valid: np.ndarray) -> dict:
    """Return the F1 of the depth's occlusion boundaries against the truth's.

    1 px tolerance, mean over `BOUNDARY_RATIOS`; and how many edges each has.
    """
    truth_depth = np.where(valid, 1.0 / np.where(valid, truth, 1), np.nan)
    f1s, found = [], {}
    for limit in BOUNDARY_RATIOS:
        mine = boundaries(1.0 / fitted, limit)
        theirs = boundaries(np.nan_to_num(truth_depth, nan=1e9), limit) & valid
        if limit == BOUNDARY_RATIOS[1]:
            # So an empty prediction (a flat map) is told from a misplaced one.
            found = {
                "edges_found": float(mine.sum()),
                "edges_true": float(theirs.sum()),
            }
        if not theirs.any():
            continue
        precision = (mine & dilate(theirs)).sum() / max(mine.sum(), 1)
        recall = (theirs & dilate(mine)).sum() / theirs.sum()
        total = precision + recall
        f1s.append(0.0 if total == 0 else 2 * precision * recall / total)
    return found | {"boundary_f1": float(np.mean(f1s)) if f1s else 0.0}


def score_depth(depth: np.ndarray | None, disparity_path: Path) -> dict:
    """Return the depth's ordering and its occlusion boundaries against the disparity.

    Ordering: delta-1 and AbsRel of the inverse depth after a scale-and-shift fit to
    the disparity. Boundaries: `boundary_f1`.
    """
    if depth is None:
        return {"depth": False}
    truth = read_disparity(disparity_path)
    height, width = truth.shape[0] // 2, truth.shape[1] // 2
    truth = truth[::2, ::2][:height, :width]
    predicted = 1.0 / np.maximum(resize(depth, (height, width)), 1e-6)
    valid = np.isfinite(truth) & (truth > 0)
    a = np.stack([predicted[valid], np.ones(valid.sum())], 1)
    scale, shift = np.linalg.lstsq(a, truth[valid], rcond=None)[0]
    fitted = np.maximum(scale * predicted + shift, 1e-6)
    ratio = np.maximum(fitted[valid] / truth[valid], truth[valid] / fitted[valid])
    relative = np.abs(fitted[valid] - truth[valid]) / truth[valid]
    edges = boundary_f1(fitted, truth, valid)
    f1 = edges.pop("boundary_f1")
    return edges | {
        "depth": True,
        "delta1": float((ratio < 1.25).mean()),
        "absrel": float(relative.mean()),
        "boundary_f1": f1,
    }
