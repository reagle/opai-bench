"""Pick the lens blur core tier from reference runs.

Photographs and Spring scenes that span the kinds of difficulty.
`opai-bench select-core REFERENCE_TAG [MORE_TAGS...]` prints each item's features and
the chosen ids, to paste into manifest.toml; AIM-500's rows are pooled over every tag.
"""

__author__ = "Joseph Reagle"
__copyright__ = "Copyright (C) 2026 Joseph Reagle"
__license__ = "MIT"

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage

from opai_bench.datasets import labels, matte_items
from opai_bench.metrics.lensblur import dilate
from opai_bench.report import rows

MIN_SCORED = 0.05
# The share of photographs with a plainer background than any core photograph's.
BUSY_SHARE = 0.75


def photo_features(image: Path, truth_path: Path) -> dict[str, Any]:
    """Return a photograph's hair share, background brightness, size and detail."""
    truth = np.asarray(Image.open(truth_path).convert("L"), np.float32) / 255
    photo = np.asarray(Image.open(image).convert("L"), np.float32) / 255
    subject = truth > 0.5
    ring = max(1, round(0.007 * max(truth.shape)))
    outside = dilate(subject, iterations=3 * ring) & ~dilate(subject, iterations=ring)
    return {
        "hair": float(((truth > 0.05) & (truth < 0.95)).sum() / max(subject.sum(), 1)),
        "background": float(photo[outside].mean()) if outside.any() else 0.5,
        "size": float(subject.mean()),
        "detail": background_detail(photo, truth),
    }


def plain(image: Path, truth_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return the photograph's luminance and its reference matte, from 0 to 1."""
    photo = np.asarray(Image.open(image).convert("L"), np.float32) / 255
    return photo, np.asarray(Image.open(truth_path).convert("L"), np.float32) / 255


def background_detail(photo: np.ndarray, truth: np.ndarray) -> float:
    """Return the fine contrast of the background away from the subject.

    At 1,024 px on the long side: high where it is busy and sharp (a large depth of
    field), the photographs where a blur shows. A plain or already blurred one scores
    near 0.
    """
    scale = 1024 / max(photo.shape)
    photo, truth = (
        np.asarray(ndimage.zoom(photo, scale, order=1)),
        np.asarray(ndimage.zoom(truth, scale, order=1)),
    )
    background = ~dilate(truth > 0.05, iterations=12)
    if background.mean() < 0.1:
        return 0.0
    fine = np.abs(photo - ndimage.gaussian_filter(photo, 1.0))
    return float(fine[background].mean())


def aim_core(tags: list[str]) -> list[str]:
    """Return AIM-500 core ids by their labels.

    Fur, plant gaps, see-through things, hair, an object with holes, a solid control,
    and the worst the reference did; within each, the one the reference did worst on
    (`sep_mse`), so the tier stays hard.
    """
    scored = {
        row["id"]: row
        for tag in tags
        for row in rows(tag, "aim")
        if row.get("matted") and "sep_mse" in row
    }
    kinds = labels("aim")
    pairs = {image.stem: (image, truth) for image, truth in matte_items("aim", "full")}
    features: dict[str, dict[str, Any]] = {
        i: photo_features(*pairs[i]) | scored[i] | kinds[i]
        for i in scored
        if i in pairs
    }
    chosen: list[str] = []

    def pick(why: str, test, key: str = "sep_mse", reverse: bool = True) -> None:
        # An item needs enough of its outline scored to say anything (`sep_share`;
        # see metrics.lensblur.separated).
        usable = (
            i
            for i in features
            if features[i]["sep_share"] >= MIN_SCORED and test(features[i])
        )
        for i in sorted(usable, key=lambda i: features[i][key], reverse=reverse):
            if i not in chosen:
                chosen.append(i)
                f = features[i]
                print(
                    f"  {i} ({why}): {f['category']}/{f['type']} hair {f['hair']:.3f}"
                    f" background {f['background']:.2f}"
                    f" detail {f['detail']:.4f} sep_mse {f['sep_mse']:.3f} sep_share {f['sep_share']:.2f}"
                )
                return
        print(f"  (none for {why})")

    # Only photographs whose background is busy and sharp: on a plain or already soft
    # one (a kite against the sky, a plant on white) a blur changes nothing visible,
    # so the edge between them tests little. The threshold is over every photograph,
    # scored or not, so a run over busy ones only is not cut down again.
    every = sorted(background_detail(*plain(*pair)) for pair in pairs.values())
    busy = every[round(BUSY_SHARE * (len(every) - 1))]
    features = {i: f for i, f in features.items() if f["detail"] >= busy}
    furry = sorted(f["hair"] for f in features.values() if f["category"] == "animal")
    fur = furry[len(furry) // 2] if furry else 1
    print("aim core:")
    pick(
        "fur, bright background",
        lambda f: (
            f["category"] == "animal" and f["hair"] > fur and f["background"] > 0.5
        ),
    )
    pick(
        "fur, dark background",
        lambda f: (
            f["category"] == "animal" and f["hair"] > fur and f["background"] <= 0.5
        ),
    )
    pick("plant", lambda f: f["category"] == "plant")
    pick("see-through", lambda f: f["category"] == "transparent")
    pick("hair", lambda f: f["category"] == "portrait", key="hair")
    pick(
        "holes in an object",
        lambda f: f["category"] in ("furniture", "toy") and f["type"] == "STM",
    )
    pick(
        "solid control",
        lambda f: f["category"] in ("fruit", "toy", "furniture") and f["type"] == "SO",
        reverse=False,
    )
    pick("hair, second", lambda f: f["category"] == "portrait", key="hair")
    pick("worst", lambda f: True)
    pick("worst", lambda f: True)
    return chosen


def select_core(tags: list[str]) -> None:
    """Print each item's features and the chosen ids, to paste into manifest.toml."""
    tag = tags[0]
    if rows(tag, "aim"):
        print(json.dumps({"aim": aim_core(tags)}))
    spring = {row["id"]: row for row in rows(tag, "spring") if row.get("depth")}
    if not spring:
        return
    order = sorted(spring, key=lambda i: spring[i]["boundary_f1"])
    scenes = [order[0], order[len(order) // 2], order[-1]]  # worst, median, best edges
    for i in sorted(
        spring, key=lambda i: spring[i]["edges_true"], reverse=True
    ):  # the most depth edges
        if i not in scenes:
            scenes.append(i)
            break
    for i in sorted(spring, key=lambda i: spring[i]["delta1"]):  # the worst ordering
        if i not in scenes:
            scenes.append(i)
            break
    print("spring core:")
    for i in scenes:
        r = spring[i]
        print(
            f"  {i}: boundary_f1 {r['boundary_f1']:.3f} delta1 {r['delta1']:.3f} edges_true {r['edges_true']:.0f}"
        )
    print(json.dumps({"spring": scenes}))
