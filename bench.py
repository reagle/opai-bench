#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = ["numpy", "pillow", "scipy", "h5py", "pyarrow"]
# ///
"""Score Open Photo AI's lens blur against datasets with ground truth.

Commands:
    bench.py fetch spring              Download one frame a scene of Spring (frame and disparity) into data/spring.
    bench.py fetch aim                 Download AIM-500 (photos, mattes, labels) into data/AIM-500.
    bench.py run TAG [--opai PATH] [--tier core|quick|full] [--only aim,spring] [--ids ID,...]
                                       Run an opai build over the datasets and score it; scores go to runs/TAG/.
    bench.py summary TAG               Print a run's mean scores.
    bench.py compare OLD NEW           Mean scores side by side, and the items that moved most.
    bench.py check OLD NEW             Compare, then fail (exit 1) where NEW is worse than OLD past REGIME.md's limits.

The opai build must have the fork's OPAI_BENCH_DUMP hook (it writes the depth map and the subject's matte).
"""

import argparse
import csv
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

Image.MAX_IMAGE_PIXELS = None
ROOT = Path(__file__).resolve().parent
MANIFEST = tomllib.loads((ROOT / "manifest.toml").read_text())
DEFAULT_OPAI = Path.home() / ".local/bin/opai"

# A matte dataset's photo is rendered at its manifest's `f_number` (AIM-500's are small and carry no EXIF, so the lens
# is assumed f/8 equivalent, and at f/1.4 a sixth still blur under the 3 px a matte needs; those count as not matted,
# by design). Spring: depth only, so no matting runs.
MATTE_SETS = [name for name, spec in MANIFEST.items() if spec.get("kind") == "matte"]
DATASETS = [*MATTE_SETS, "spring"]
SPRING_ARGS = ["--lb", "--lb-f", "2", "--lb-focus", "0.5,0.5", "--lb-edges", "depth"]
BOUNDARY_RATIOS = (1.05, 1.10, 1.15, 1.20, 1.25)
# How much blur, in pixels of radius, the background beside the subject must get for that stretch of its outline to be
# scored: where it blurs less, the subject's edge is not the edge of what is sharp (a floor at the seat's distance, a
# kite against the sky beyond the hyperfocal distance), and the ground truth's object outline says nothing of the blur.
SEPARATED_PX = 2.0
# And how little the subject beside it must blur: a part of the subject nearer or farther than the focus (a sofa's arm
# seen from above) rightly blurs, and its outline in the ground truth is not the edge of what is sharp either.
SHARP_PX = 1.0


# --- Fetching ------------------------------------------------------------------------------------------------------


class RemoteFile(io.RawIOBase):
    """A seekable read-only file over HTTP range requests, so single members can be read out of a large zip."""

    def __init__(self, url: str) -> None:
        self.url, self.pos = url, 0
        with urllib.request.urlopen(urllib.request.Request(url, headers={"Range": "bytes=0-0"})) as response:
            self.size = int(response.headers["Content-Range"].split("/")[1])

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def readinto(self, buffer) -> int:
        count = min(len(buffer), self.size - self.pos)
        if count <= 0:
            return 0
        headers = {"Range": f"bytes={self.pos}-{self.pos + count - 1}"}
        with urllib.request.urlopen(urllib.request.Request(self.url, headers=headers)) as response:
            data = response.read()
        buffer[: len(data)] = data
        self.pos += len(data)
        return len(data)


def darus_url(file_id: int) -> str:
    """The signed storage URL DaRUS redirects a file id to."""

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            return None

    opener = urllib.request.build_opener(NoRedirect)
    try:
        opener.open(f"https://darus.uni-stuttgart.de/api/access/datafile/{file_id}")
    except urllib.error.HTTPError as error:
        if error.code in (301, 302, 303, 307):
            return error.headers["Location"]
        raise
    raise RuntimeError(f"DaRUS did not redirect file {file_id}")


def remote_zip(file_id: int) -> zipfile.ZipFile:
    return zipfile.ZipFile(io.BufferedReader(RemoteFile(darus_url(file_id)), 1 << 20))


def fetch_spring() -> None:
    spec = MANIFEST["spring"]
    out = ROOT / spec["default_path"]
    frames, disparities = remote_zip(spec["frames_file"]), remote_zip(spec["disparity_file"])
    scenes: dict[str, list[int]] = {}
    for name in frames.namelist():
        parts = name.split("/")
        if name.endswith(".png"):
            scenes.setdefault(parts[2], []).append(int(parts[-1].rsplit("_", 1)[1].split(".")[0]))
    for scene, numbers in sorted(scenes.items()):
        frame = sorted(numbers)[len(numbers) // 2]
        target = out / scene
        if (target / "disparity.dsp5").exists():
            continue
        target.mkdir(parents=True, exist_ok=True)
        (target / "frame.png").write_bytes(frames.read(spec["member_frame"].format(scene=scene, frame=frame)))
        disparity = disparities.read(spec["member_disparity"].format(scene=scene, frame=frame))
        (target / "disparity.dsp5").write_bytes(disparity)
        (target / "source.txt").write_text(f"Spring train scene {scene}, frame {frame} (CC BY 4.0)\n")
        print(f"spring {scene} frame {frame}", flush=True)


def fetch_aim() -> None:
    """AIM-500 from its Hugging Face mirror (one parquet), unpacked as the authors' folders plus `labels.csv`."""
    import pyarrow.parquet as pq

    spec = MANIFEST["aim"]
    out = ROOT / spec["default_path"]
    parquet = out / "test.parquet"
    if not parquet.exists():
        out.mkdir(parents=True, exist_ok=True)
        print(f"aim: downloading {spec['mirror']}", flush=True)
        urllib.request.urlretrieve(spec["mirror"], parquet)
    table = pq.read_table(parquet)
    (out / spec["images"]).mkdir(exist_ok=True)
    (out / spec["mattes"]).mkdir(exist_ok=True)
    with (out / "labels.csv").open("w", newline="") as labels:
        writer = csv.writer(labels)
        writer.writerow(["id", "category", "type"])
        for row in table.to_pylist():
            (out / spec["images"] / row["image"]["path"]).write_bytes(row["image"]["bytes"])
            (out / spec["mattes"] / row["mask"]["path"]).write_bytes(row["mask"]["bytes"])
            writer.writerow([row["image_name"], row["category"], row["type"]])
    parquet.unlink()
    print(f"aim: {table.num_rows} photos in {out}", flush=True)


def labels(name: str) -> dict[str, dict[str, str]]:
    """A dataset's per-item labels (`labels.csv` beside its folders), by id; empty where it has none."""
    path = Path(os.environ.get(f"OPAI_BENCH_{name.upper()}") or ROOT / MANIFEST[name]["default_path"]) / "labels.csv"
    return {row["id"]: row for row in csv.DictReader(path.open())} if path.exists() else {}


# --- Running -------------------------------------------------------------------------------------------------------


def load_dump(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    width, height = np.frombuffer(raw[:8], "<u4")
    return np.frombuffer(raw[8:], "<f4").reshape(int(height), int(width))


def run_opai(opai: Path, source: Path, args: list[str], work: Path) -> tuple[dict[str, np.ndarray], str]:
    """Run `opai` on `source`, returning what it dumped (`depth`, `matte`, `radius`: the blur's radius at each depth
    pixel; any it did not write is missing) and its last line."""
    prefix = work / "dump"
    for stale in work.glob("dump-*"):
        stale.unlink()
    env = os.environ | {"OPAI_BENCH_DUMP": str(prefix)}
    output = work / f"out{source.suffix}"
    done = subprocess.run(
        [str(opai), str(source), *args, "-o", str(output)], env=env, capture_output=True, text=True, check=False
    )
    lines = (done.stdout + done.stderr).strip().splitlines()
    output.unlink(missing_ok=True)
    dumps = {
        name: load_dump(Path(f"{prefix}-{name}.f32"))
        for name in ("depth", "matte", "radius")
        if Path(f"{prefix}-{name}.f32").exists()
    }
    return dumps, lines[-1] if lines else f"exit {done.returncode}"


def matte_items(name: str, tier: str) -> list[tuple[Path, Path]]:
    """A matte dataset's (photo, ground-truth matte) pairs for `tier`: `quick` takes every `quick_every`th in name
    order, so each kind the names start with is in it."""
    spec = MANIFEST[name]
    root = Path(os.environ.get(f"OPAI_BENCH_{name.upper()}") or ROOT / spec["default_path"])
    if not (root / spec["images"]).is_dir():
        print(f"{name}: not found at {root} (download it from {spec['source']})", file=sys.stderr)
        return []
    kinds = labels(name)
    skip = set(spec.get("skip_types", []))
    pairs = [
        (image, root / spec["mattes"] / f"{image.stem}{spec['matte_suffix']}")
        for image in sorted((root / spec["images"]).iterdir())
        if image.is_file() and not image.name.startswith(".") and kinds.get(image.stem, {}).get("type") not in skip
    ]
    if tier == "core":
        return [pair for pair in pairs if pair[0].stem in spec["core"]]
    return pairs[:: spec["quick_every"]] if tier == "quick" else pairs


def spring_items(tier: str) -> list[Path]:
    spec = MANIFEST["spring"]
    root = ROOT / spec["default_path"]
    scenes = sorted(path for path in root.glob("*") if (path / "disparity.dsp5").exists()) if root.is_dir() else []
    if not scenes:
        print(f"spring: nothing in {root} (run `bench.py fetch spring`)", file=sys.stderr)
    if tier == "core":
        return [scene for scene in scenes if scene.name in spec["core"]]
    return scenes[: spec["quick"]] if tier == "quick" else scenes


def focus_inside(truth: np.ndarray) -> tuple[float, float]:
    """The point of the subject farthest from its edge, as fractions of the frame."""
    inside = ndimage.distance_transform_edt(truth > 0.95)
    y, x = np.unravel_index(int(np.argmax(inside)), inside.shape)
    return round(x / truth.shape[1], 3), round(y / truth.shape[0], 3)


def run(tag: str, opai: Path, tier: str, only: set[str], ids: set[str] | None = None) -> None:
    """Score `opai` over the `only` datasets' `tier`; with `ids`, over those items of the whole datasets instead."""
    out = ROOT / "runs" / tag
    out.mkdir(parents=True, exist_ok=True)
    version = subprocess.run([str(opai), "--version"], capture_output=True, text=True, check=False).stdout.strip()
    (out / "build.txt").write_text(f"{version}\n{opai}\ntier {tier}\n")
    with tempfile.TemporaryDirectory(prefix="opai-bench-") as scratch:
        work = Path(scratch)
        for name in [name for name in MATTE_SETS if name in only]:
            args = ["--lb", "--lb-f", str(MANIFEST[name]["f_number"])]
            with (out / f"{name}.jsonl").open("w") as scores:
                for image, truth_path in matte_items(name, "full" if ids else tier):
                    if ids and image.stem not in ids:
                        continue
                    truth = np.asarray(Image.open(truth_path).convert("L"), np.float32) / 255
                    fx, fy = focus_inside(truth)
                    dumps, last = run_opai(opai, image, [*args, "--lb-focus", f"{fx},{fy}"], work)
                    row = {"id": image.stem, "focus": [fx, fy], "last": last}
                    row |= score_matte(dumps.get("matte"), truth, dumps.get("radius"))
                    scores.write(json.dumps(row) + "\n")
                    print(f"{name} {image.stem}: {short(row)}", flush=True)
        if "spring" in only:
            with (out / "spring.jsonl").open("w") as scores:
                for scene in spring_items("full" if ids else tier):
                    if ids and scene.name not in ids:
                        continue
                    dumps, last = run_opai(opai, scene / "frame.png", SPRING_ARGS, work)
                    row = {"id": scene.name, "last": last} | score_depth(dumps.get("depth"), scene / "disparity.dsp5")
                    scores.write(json.dumps(row) + "\n")
                    print(f"spring {scene.name}: {short(row)}", flush=True)
    summary(tag)


def short(row: dict) -> str:
    return " ".join(f"{key}={value:.4g}" for key, value in row.items() if isinstance(value, float))


# --- Scoring -------------------------------------------------------------------------------------------------------


def score_matte(matte: np.ndarray | None, truth: np.ndarray, radius: np.ndarray | None) -> dict:
    """The usual matting errors of `matte` against `truth` (SAD and gradient in thousands), the squared error within
    the truth's edge band, and specks: small islands where the two disagree about subject or background.

    A lens keeps sharp whatever is at the subject's distance, not the person alone (a truck beside her stays sharp), so
    the whole-frame errors mostly measure that. The `sep_` scores are taken only along the stretches of the person's
    outline where what lies just outside blurs by at least `SEPARATED_PX` and what lies just inside by under `SHARP_PX`
    (`radius`, the blur's radius at each pixel of the depth map): there the subject's edge is the edge of what is
    sharp, and the matte's edge quality shows.
    `sep_share` is the share of the outline that qualifies."""
    if matte is None:
        return {"matted": False}
    if matte.shape != truth.shape:
        truth = np.asarray(Image.fromarray(truth).resize((matte.shape[1], matte.shape[0]), Image.BILINEAR))
    matte = np.clip(matte, 0, 1)
    error = matte - truth
    ring = max(1, round(0.007 * max(truth.shape)))
    edge = (truth > 0.01) & (truth < 0.99)
    edge |= ndimage.binary_dilation(truth > 0.5) ^ ndimage.binary_erosion(truth > 0.5)
    band = ndimage.binary_dilation(edge, iterations=ring)

    def gradient(plane: np.ndarray) -> np.ndarray:
        across = ndimage.gaussian_filter(plane, 1.4, order=(0, 1))
        down = ndimage.gaussian_filter(plane, 1.4, order=(1, 0))
        return np.hypot(across, down)

    disagree = (matte > 0.5) ^ (truth > 0.5)
    labels, count = ndimage.label(disagree)
    sizes = np.bincount(labels.ravel())[1:] if count else np.array([])
    speck_area = (ring // 4) ** 2 or 1
    separated = {}
    if radius is not None:
        blurred = np.abs(np.asarray(Image.fromarray(radius).resize((truth.shape[1], truth.shape[0]), Image.BILINEAR)))
        subject = truth > 0.5
        outside = ndimage.binary_dilation(subject, iterations=2 * ring)
        outside &= ~ndimage.binary_dilation(subject, iterations=ring)
        inside = ndimage.binary_erosion(subject, iterations=ring) & ~ndimage.binary_erosion(subject, iterations=2 * ring)
        apart = outside & (blurred >= SEPARATED_PX)
        sharp = inside & (blurred < SHARP_PX)
        scored = band & ndimage.binary_dilation(apart, iterations=2 * ring)
        scored &= ndimage.binary_dilation(sharp, iterations=2 * ring)
        separated = {
            "sep_share": float(scored.sum() / max(band.sum(), 1)),
            "sep_sad": float(np.abs(error[scored]).sum() / 1000),
            "sep_mse": float((error[scored] ** 2).mean()) if scored.any() else 0.0,
        } | scene_depth(np.asarray(Image.fromarray(radius).resize((truth.shape[1], truth.shape[0]))), truth, ring)
    return separated | {
        "matted": True,
        "sad": float(np.abs(error).sum() / 1000),
        "mse": float((error**2).mean()),
        "grad": float(((gradient(matte) - gradient(truth)) ** 2).sum() / 1000),
        "edge_mse": float((error[band] ** 2).mean()) if band.any() else 0.0,
        "specks": float(((sizes >= 4) & (sizes < max(speck_area, 16))).sum()),
    }


def scene_depth(radius: np.ndarray, truth: np.ndarray, ring: int) -> dict:
    """What the photograph's depth gives a lens blur to do, from the signed blur `radius` (positive nearer than the
    focus, negative beyond), for choosing test photographs rather than scoring a build:

    `subject_blur`: the blur the subject's own nearest or farthest tenth would get (a face sharp, the feet not).
    `behind_median`, `behind_far`: the background's blur behind the focus, at its middle and its farthest tenth; far
    apart where the background recedes (ground, a floor, a room) rather than standing at one distance.
    `front_share`: the share of the background nearer than the focus and blurred, something in front of the subject."""
    subject = ndimage.binary_erosion(truth > 0.5, iterations=ring)
    background = truth < 0.05
    behind = -radius[background & (radius < 0)]
    return {
        "subject_blur": float(np.percentile(np.abs(radius[subject]), 90)) if subject.any() else 0.0,
        "behind_median": float(np.median(behind)) if behind.size else 0.0,
        "behind_far": float(np.percentile(behind, 90)) if behind.size else 0.0,
        "front_share": float((background & (radius >= SEPARATED_PX)).sum() / max(background.sum(), 1)),
    }


def read_disparity(path: Path) -> np.ndarray:
    import h5py

    with h5py.File(path, "r") as file:
        return np.asarray(file["disparity"], np.float32)


def boundaries(depth: np.ndarray, ratio: float) -> np.ndarray:
    """Occlusion boundaries, Depth Pro style: a pixel nearer than a 4-neighbour by more than `ratio`."""
    near = np.zeros(depth.shape, bool)
    for axis in (0, 1):
        a, b = np.moveaxis(depth, axis, 0)[:-1], np.moveaxis(depth, axis, 0)[1:]
        view = np.moveaxis(near, axis, 0)
        view[:-1] |= b / a > ratio
        view[1:] |= a / b > ratio
    return near


def score_depth(depth: np.ndarray | None, disparity_path: Path) -> dict:
    """The depth's ordering (delta-1 and AbsRel of its inverse after a scale-and-shift fit to the disparity) and its
    occlusion boundaries (F1 against the disparity's, 1 px tolerance, mean over the ratios)."""
    if depth is None:
        return {"depth": False}
    truth = read_disparity(disparity_path)
    height, width = truth.shape[0] // 2, truth.shape[1] // 2
    truth = truth[::2, ::2][:height, :width]
    predicted = 1.0 / np.maximum(np.asarray(Image.fromarray(depth).resize((width, height), Image.BILINEAR)), 1e-6)
    valid = np.isfinite(truth) & (truth > 0)
    a = np.stack([predicted[valid], np.ones(valid.sum())], 1)
    scale, shift = np.linalg.lstsq(a, truth[valid], rcond=None)[0]
    fitted = np.maximum(scale * predicted + shift, 1e-6)
    ratio = np.maximum(fitted[valid] / truth[valid], truth[valid] / fitted[valid])

    truth_depth = np.where(valid, 1.0 / np.where(valid, truth, 1), np.nan)
    f1s, found = [], {}
    for limit in BOUNDARY_RATIOS:
        mine, theirs = boundaries(1.0 / fitted, limit), boundaries(np.nan_to_num(truth_depth, nan=1e9), limit) & valid
        if limit == BOUNDARY_RATIOS[1]:
            # So an empty prediction (a flat map) is told from a misplaced one.
            found = {"edges_found": float(mine.sum()), "edges_true": float(theirs.sum())}
        if not theirs.any():
            continue
        precision = (mine & ndimage.binary_dilation(theirs)).sum() / max(mine.sum(), 1)
        recall = (theirs & ndimage.binary_dilation(mine)).sum() / theirs.sum()
        f1s.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return found | {
        "depth": True,
        "delta1": float((ratio < 1.25).mean()),
        "absrel": float((np.abs(fitted[valid] - truth[valid]) / truth[valid]).mean()),
        "boundary_f1": float(np.mean(f1s)) if f1s else 0.0,
    }


# --- Reporting -----------------------------------------------------------------------------------------------------


def rows(tag: str, dataset: str) -> list[dict]:
    path = ROOT / "runs" / tag / f"{dataset}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def means(items: list[dict]) -> dict[str, float]:
    keys = [key for key, value in items[0].items() if isinstance(value, float)] if items else []
    return {key: float(np.mean([item[key] for item in items if key in item])) for key in keys}


def flag(dataset: str) -> str:
    """The field a row of `dataset` says it was scored by."""
    return "depth" if dataset == "spring" else "matted"


def summary(tag: str) -> None:
    for dataset in DATASETS:
        items = rows(tag, dataset)
        if items:
            done = [item for item in items if item.get(flag(dataset))]
            print(f"{tag} {dataset}: {len(done)}/{len(items)} scored; " + short(means(done)))


def compare(old: str, new: str) -> None:
    for dataset in DATASETS:
        key = "boundary_f1" if dataset == "spring" else "sep_mse"
        before, after = {r["id"]: r for r in rows(old, dataset)}, {r["id"]: r for r in rows(new, dataset)}
        common = [i for i in before if i in after and key in before[i] and key in after[i]]
        if not common:
            continue
        a, b = means([before[i] for i in common]), means([after[i] for i in common])
        print(f"{dataset} ({len(common)} items)")
        for metric in a:
            print(f"  {metric:12} {a[metric]:10.4g} -> {b.get(metric, float('nan')):10.4g}")
        moved = sorted(common, key=lambda i: abs(after[i][key] - before[i][key]), reverse=True)[:5]
        print(f"  most moved by {key}: " + ", ".join(f"{i} {before[i][key]:.4g}->{after[i][key]:.4g}" for i in moved))


# The limits of `check` (REGIME.md): a mean may move by this much in the worse direction before it is a regression.
MATTE_LIMITS = {"sep_mse": ("up", 0.05, "relative"), "sep_sad": ("up", 0.05, "relative")}
LIMITS = {name: MATTE_LIMITS for name in MATTE_SETS} | {
    "spring": {"boundary_f1": ("down", 0.01, "absolute"), "delta1": ("down", 0.01, "absolute")},
}


def check(old: str, new: str) -> int:
    """`compare`, then each limit, and each item NEW no longer scores (no matte or no depth where OLD had one)."""
    compare(old, new)
    failures = []
    for dataset, limits in LIMITS.items():
        scored = flag(dataset)
        before, after = {r["id"]: r for r in rows(old, dataset)}, {r["id"]: r for r in rows(new, dataset)}
        lost = [i for i in before if before[i].get(scored) and i in after and not after[i].get(scored)]
        failures += [f"{dataset} {i}: no longer scored" for i in lost]
        common = [i for i in before if i in after and before[i].get(scored) and after[i].get(scored)]
        if not common:
            continue
        a, b = means([before[i] for i in common]), means([after[i] for i in common])
        for metric, (worse, limit, kind) in limits.items():
            change = b[metric] - a[metric]
            if worse == "down":
                change = -change
            allowed = limit * abs(a[metric]) if kind == "relative" else limit
            if change > allowed:
                failures.append(f"{dataset} {metric}: {a[metric]:.4g} -> {b[metric]:.4g} (limit {limit:g} {kind})")
    for failure in failures:
        print(f"REGRESSION {failure}")
    print("check: " + ("FAIL" if failures else "pass"))
    return 1 if failures else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch")
    fetch.add_argument("dataset", choices=["aim", "spring"])
    runner = commands.add_parser("run")
    runner.add_argument("tag")
    runner.add_argument("--opai", type=Path, default=DEFAULT_OPAI)
    runner.add_argument("--tier", choices=["core", "quick", "full"], default="core")
    runner.add_argument("--only", default=",".join(DATASETS))
    runner.add_argument("--ids", help="comma-separated item ids (any tier); for a closer look at a few")
    commands.add_parser("summary").add_argument("tag")
    for name in ("compare", "check"):
        comparer = commands.add_parser(name)
        comparer.add_argument("old")
        comparer.add_argument("new")
    args = parser.parse_args()

    if args.command == "fetch":
        fetch_aim() if args.dataset == "aim" else fetch_spring()
    elif args.command == "run":
        if not shutil.which(str(args.opai)) and not args.opai.exists():
            sys.exit(f"no opai at {args.opai}")
        run(args.tag, args.opai, args.tier, set(args.only.split(",")), set(args.ids.split(",")) if args.ids else None)
    elif args.command == "summary":
        summary(args.tag)
    elif args.command == "check":
        sys.exit(check(args.old, args.new))
    else:
        compare(args.old, args.new)


if __name__ == "__main__":
    main()
