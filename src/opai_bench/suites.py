"""Run the suites: lens blur, sharpen/denoise, precision, models.

The lens blur suite scores depth and matte against ground truth; the sharpen/denoise
suite the quality tier of every sharpen model and Stockholm; precision and models
compare renderings. Each writes one JSON line an item to `runs/<tag>/`, with the
build's version in `build.txt` and what was run in `run.json`, shows a progress bar
while it runs, and prints the concise table when done.
"""

__author__ = "Joseph Reagle"
__copyright__ = "Copyright (C) 2026 Joseph Reagle"
__license__ = "MIT"

import argparse
import json
import logging
import re
import shlex
import shutil
import tempfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from opai_bench import report, synthetic
from opai_bench.config import MANIFEST, dataset_dir, runs_dir
from opai_bench.datasets import MATTE_SETS, matte_items, spring_items, suite_photos
from opai_bench.metrics import lensblur, pixels, quality
from opai_bench.runner import (
    FAMILIES,
    Build,
    Home,
    Result,
    enhance,
    family_of,
    model_args,
    run_dump,
)

SPRING_ARGS = ["--lb", "--lb-f", "2", "--lb-focus", "0.5,0.5", "--lb-edges", "depth"]
# The quality measures a precision or model comparison keeps per output (the sharpen
# suite keeps them all).
HEADLINE = ("mottle", "seam_excess", "inv_max", "edge_gain_2_8")
QUALITY_FAMILIES = ("sharpen", "denoise")
stderr = Console(stderr=True)


def progress() -> Progress:
    """Return a progress bar on stderr: what runs now, how far, and the time left."""
    return Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TextColumn("left"),
        TimeRemainingColumn(),
        console=stderr,
        transient=True,
    )


def write_build(out: Path, builds: list[Build], meta: dict) -> None:
    """Write a run's `build.txt` and `run.json`."""
    out.mkdir(parents=True, exist_ok=True)
    if meta.get("kind", "lensblur") == "lensblur" and len(builds) == 1:
        # As every earlier run has it: version, path, tier.
        build = builds[0]
        (out / "build.txt").write_text(
            f"{build.version}\n{build.path}\ntier {meta['tier']}\n"
        )
    else:
        lines = "".join(f"{b.label}: {b.version}\n  {b.path}\n" for b in builds)
        (out / "build.txt").write_text(lines)
    listed = [
        {"label": b.label, "path": str(b.path), "version": b.version} for b in builds
    ]
    (out / "run.json").write_text(
        json.dumps(meta | {"builds": listed}, indent=1) + "\n"
    )


# --- Lens blur ---------------------------------------------------------------------


def lensblur_items(tier: str, only: set[str], ids: set[str] | None) -> list[tuple]:
    """Return the lens blur suite's items: (dataset, id, photo, truth or scene)."""
    items = []
    tier = "full" if ids else tier
    for name in [name for name in MATTE_SETS if name in only]:
        for image, truth in matte_items(name, tier):
            items.append((name, image.stem, image, truth))
    if "spring" in only:
        for scene in spring_items(tier):
            items.append(("spring", scene.name, scene / "frame.png", scene))
    return [item for item in items if not ids or item[1] in ids]


def score_item(build: Build, item: tuple, work: Path) -> dict:
    """Render and score one lens blur item."""
    dataset, name, image, truth_path = item
    if dataset == "spring":
        dumps, last = run_dump(build.path, image, SPRING_ARGS, work)
        depth = lensblur.score_depth(dumps.get("depth"), truth_path / "disparity.dsp5")
        return {"id": name, "last": last} | depth
    truth = np.asarray(Image.open(truth_path).convert("L"), np.float32) / 255
    fx, fy = lensblur.focus_inside(truth)
    args = [
        "--lb",
        "--lb-f",
        str(MANIFEST[dataset]["f_number"]),
        "--lb-focus",
        f"{fx},{fy}",
    ]
    dumps, last = run_dump(build.path, image, args, work)
    row = {"id": name, "focus": [fx, fy], "last": last}
    return row | lensblur.score_matte(dumps.get("matte"), truth, dumps.get("radius"))


def run_lensblur(
    tag: str, build: Build, tier: str, only: set[str], ids: set[str] | None = None
) -> None:
    """Score `build` over the `only` datasets' `tier` (or the `ids` of all)."""
    out = runs_dir() / tag
    meta = {
        "kind": "lensblur",
        "tier": tier,
        "only": sorted(only),
        "ids": sorted(ids or []),
    }
    write_build(out, [build], meta)
    items = lensblur_items(tier, only, ids)
    files = {
        dataset: (out / f"{dataset}.jsonl").open("w")
        for dataset in {i[0] for i in items}
    }
    try:
        with (
            tempfile.TemporaryDirectory(prefix="opai-bench-") as scratch,
            progress() as bar,
        ):
            task = bar.add_task("lens blur", total=len(items))
            for item in items:
                bar.update(task, description=f"{item[0]} {item[1]}")
                row = score_item(build, item, Path(scratch))
                files[item[0]].write(json.dumps(row) + "\n")
                files[item[0]].flush()
                logging.info("%s %s: %s", item[0], item[1], report.short(row))
                bar.advance(task)
    finally:
        for scores in files.values():
            scores.close()
    report.summary(tag)


# --- Enhancement runs --------------------------------------------------------------


@dataclass
class Photo:
    """Name a photograph to enhance, with a lens blur's focus where it has one."""

    id: str
    path: Path
    focus: str | None = None  # the subject's middle, from AIM-500's matte


@dataclass
class Options:
    """Say what every enhancement run shares: the photographs, the grid, the calls."""

    tier: str = "core"
    ids: set[str] | None = None
    photos: list[Path] = field(default_factory=list)
    limit: int | None = None
    synthetic: bool = True
    warmup: bool = True
    keep: bool = False
    scale: float = 2.0
    extra: list[str] = field(default_factory=list)
    tile: int = MANIFEST["sharpen"]["tile"]
    overlap: int = MANIFEST["sharpen"]["overlap"]

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "Options":
        """Return the options the command line gave."""
        return cls(
            tier=args.tier,
            ids=set(args.ids.split(",")) if args.ids else None,
            photos=[Path(p).expanduser() for p in args.photos or []],
            limit=args.limit,
            synthetic=args.synthetic,
            warmup=args.warmup,
            keep=args.keep,
            scale=args.scale,
            extra=shlex.split(args.args or ""),
            tile=args.tile,
            overlap=args.overlap,
        )

    def meta(self) -> dict:
        """Return the options as `run.json` records them."""
        return {
            "tier": self.tier,
            "ids": sorted(self.ids or []),
            "photos": [str(p) for p in self.photos],
            "limit": self.limit,
            "synthetic": self.synthetic,
            "scale": self.scale,
            "extra": self.extra,
            "tile": self.tile,
            "overlap": self.overlap,
        }


def matte_focus(photo: Photo) -> None:
    """Focus a photograph's lens blur on its AIM-500 subject's middle, if any."""
    spec = MANIFEST["aim"]
    matte = dataset_dir("aim") / spec["mattes"] / f"{photo.id}{spec['matte_suffix']}"
    if matte.exists():
        truth = np.asarray(Image.open(matte).convert("L"), np.float32) / 255
        photo.focus = ",".join(map(str, lensblur.focus_inside(truth)))


def photo_set(options: Options, families: set[str]) -> list[Photo]:
    """Return the photographs of an enhancement run.

    The user's own (`--photos`), or the sharpen suite's AIM-500 tier (or `--ids` from
    the whole set); then the synthetic gradient, at half size for an upscale.
    """
    spec = MANIFEST["sharpen"]
    if options.photos:
        chosen = [Photo(path.stem, path) for path in options.photos]
    else:
        tier = "full" if options.ids else options.tier
        pool = suite_photos(tier, spec["core"], spec["quick_every"])
        chosen = [
            Photo(p.stem, p) for p in pool if not options.ids or p.stem in options.ids
        ]
        if "lensblur" in families:
            for photo in chosen:
                matte_focus(photo)
    if options.limit:
        chosen = chosen[: options.limit]
    if options.synthetic:
        size = (1500, 1000) if "upscale" in families else synthetic.SIZE
        chosen.append(Photo("synthetic", synthetic.ensure(*size)))
    return chosen


@dataclass(frozen=True)
class Config:
    """Name one way of rendering a photograph: a build, a model and a quality."""

    name: str
    build: Build
    model: str
    quality: str | None

    def args(self, photo: Photo, options: Options) -> list[str]:
        """Return the CLI arguments for this config on `photo`."""
        return model_args(
            self.model, self.quality, options.scale, options.extra, photo.focus
        )


def default_tag(kind: str, subject: str, build: Build) -> str:
    """Return a run's tag where none is given: what was run and the build's version.

    >>> class Fake: version = "Open Photo AI - CLI 26.10.6+reagle.20261030.review"
    >>> default_tag("precision", "sharpen", Fake())
    'precision-sharpen-26.10.6-reagle.20261030.review'
    """
    version = build.version.split()[-1] if build.version else "unknown"
    return re.sub(r"[^A-Za-z0-9._-]+", "-", f"{kind}-{subject}-{version}")


Outputs = dict[str, tuple[Path, Result]]


def render(
    configs: list[Config], photos: list[Photo], options: Options, work: Path, home: Home
) -> Iterator[tuple[Photo, Outputs]]:
    """Yield, for each photograph, every config's output and its timing.

    Warms each config up first (untimed, on the first photograph) so a first use's
    compile or load is not counted; then, photograph by photograph, runs every config
    with a cleared run cache.
    """
    warmups = configs if options.warmup else []
    with progress() as bar:
        task = bar.add_task("", total=len(warmups) + len(photos) * len(configs))
        for config in warmups:
            bar.update(task, description=f"warm-up {config.name}")
            warm = work / "warmup.png"
            first = photos[0]
            result = enhance(
                config.build, home, first.path, config.args(first, options), warm
            )
            logging.info("warm-up %s: %.1f s", config.name, result.seconds)
            warm.unlink(missing_ok=True)
            bar.advance(task)
        for photo in photos:
            outputs: Outputs = {}
            for config in configs:
                bar.update(task, description=f"{photo.id} {config.name}")
                path = work / config.name / f"{photo.id}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                args = config.args(photo, options)
                outputs[config.name] = (
                    path,
                    enhance(config.build, home, photo.path, args, path),
                )
                bar.advance(task)
            bar.update(task, description=f"{photo.id} scoring")
            yield photo, outputs
            if not options.keep:
                for path, _ in outputs.values():
                    path.unlink(missing_ok=True)


def headline(measures: dict[str, float], suffix: str = "") -> dict[str, float]:
    """Return the headline quality measures, with `suffix` on their keys."""
    return {f"{key}{suffix}": measures[key] for key in HEADLINE if key in measures}


Score = Callable[[Photo, Outputs, np.ndarray], list[dict]]


@dataclass
class Plan:
    """Say what an enhancement run is: kind, tag, builds, configs, and how it scores."""

    kind: str
    tag: str
    builds: list[Build]
    configs: list[Config]
    meta: dict
    score: Score


def enhancement_run(plan: Plan, options: Options) -> None:
    """Run what every enhancement run shares: run.json, the photographs, the rendering.

    Then one row an item from `plan.score(photo, outputs, input)`, and the concise
    table.
    """
    out = runs_dir() / plan.tag
    families = {family_of(config.model) for config in plan.configs}
    write_build(out, plan.builds, {"kind": plan.kind, **plan.meta, **options.meta()})
    photos = photo_set(options, families)
    if not photos:
        raise SystemExit("no photographs to run on")
    stderr.print(
        f"{plan.kind} {plan.tag}: {len(plan.configs)} configs x {len(photos)} photographs"
    )
    home = Home()
    work = (
        out / "out" if options.keep else Path(tempfile.mkdtemp(prefix="opai-", dir=out))
    )
    try:
        with (out / f"{plan.kind}.jsonl").open("w") as rows:
            for photo, outputs in render(plan.configs, photos, options, work, home):
                source = pixels.load(photo.path)
                for row in plan.score(photo, outputs, source):
                    rows.write(json.dumps(row) + "\n")
                    rows.flush()
                    logging.info(
                        "%s %s: %s", photo.id, row.get("model"), report.short(row)
                    )
    finally:
        if not options.keep:
            shutil.rmtree(work, ignore_errors=True)
    report.summary(plan.tag)


def run_sharpen(
    tag: str, build: Build, options: Options, quality_level: str | None
) -> None:
    """Run the sharpen/denoise suite: each `[sharpen] models` model at its defaults.

    Its quality tier against the input, and its time.
    """
    configs = [
        Config(m, build, m, quality_level) for _, m in MANIFEST["sharpen"]["models"]
    ]

    def score(photo: Photo, outputs: Outputs, source: np.ndarray) -> list[dict]:
        found = []
        for config in configs:
            path, result = outputs[config.name]
            row = {
                "id": photo.id,
                "model": config.model,
                "family": family_of(config.model),
            }
            row |= {"ok": result.ok, "seconds": result.seconds, "last": result.last}
            if result.ok:
                output = pixels.load(path)
                row |= quality.measures(source, output, options.tile, options.overlap)
            found.append(row)
        return found

    models = [c.model for c in configs]
    meta = {"suite": "sharpen", "quality": quality_level, "models": models}
    enhancement_run(Plan("sharpen", tag, [build], configs, meta, score), options)


def models_of(subject: str) -> list[str]:
    """Return a model's name, or every model of a family.

    >>> models_of("sharpen"), models_of("Petersburg")
    (['moscow', 'petersburg', 'novgorod'], ['petersburg'])
    """
    if subject.lower() in FAMILIES:
        return FAMILIES[subject.lower()][1]
    family_of(subject)  # refuses an unknown model
    return [subject.lower()]


def target(source: np.ndarray, output: np.ndarray) -> np.ndarray:
    """Return the input as an output is compared against: resized to it (an upscale)."""
    return pixels.resized(source, (output.shape[1], output.shape[0]))


def precision_row(model: str, outputs: Outputs, source: np.ndarray, options: Options):
    """Return one photograph's row of a precision run: SD against HD, and the times."""
    (hd_path, hd), (sd_path, sd) = outputs[f"{model}-hd"], outputs[f"{model}-sd"]
    row = {"model": model, "family": family_of(model), "ok": hd.ok and sd.ok}
    row |= {"seconds_hd": hd.seconds, "seconds_sd": sd.seconds}
    if not row["ok"]:
        return row | {"last": hd.last if not hd.ok else sd.last}
    high, low = pixels.load(hd_path), pixels.load(sd_path)
    flat = pixels.flat_mask(pixels.luma(target(source, high)))
    row |= {"size": [high.shape[1], high.shape[0]]} | pixels.compare(high, low, flat)
    if family_of(model) in QUALITY_FAMILIES:
        for suffix, output in (("_hd", high), ("_sd", low)):
            found = quality.measures(source, output, options.tile, options.overlap)
            row |= headline(found, suffix)
    return row


def run_precision(
    subject: str, build: Build, options: Options, tag: str | None
) -> None:
    """Run each model at HD (FP32) and SD (FP16) on the same photographs.

    SD against HD, per photograph, overall and split into flat and detailed areas,
    with each precision's time.
    """
    models = models_of(subject)
    configs = [Config(f"{m}-{q}", build, m, q) for m in models for q in ("hd", "sd")]
    tag = tag or default_tag("precision", subject.lower(), build)

    def score(photo: Photo, outputs: Outputs, source: np.ndarray) -> list[dict]:
        return [
            {"id": photo.id} | precision_row(model, outputs, source, options)
            for model in models
        ]

    meta = {"subject": subject, "models": models}
    enhancement_run(Plan("precision", tag, [build], configs, meta, score), options)


@dataclass
class Comparison:
    """Say which models of a family to compare, on which builds, against what."""

    family: str
    names: list[str]
    builds: list[Build]
    ref: str = "input"
    quality: str | None = None

    def configs(self) -> list[Config]:
        """Return a config per build and model (`model@label` with several builds)."""
        several = len(self.builds) > 1
        return [
            Config(
                f"{name}@{build.label}" if several else name,
                build,
                name.lower(),
                self.quality,
            )
            for build in self.builds
            for name in self.names
        ]


def models_row(
    config: Config,
    rendered: tuple[Path, Result],
    *,
    source: np.ndarray,
    against: np.ndarray | None,
    options: Options,
    ref: str,
) -> dict:
    """Return one config's row of a models run on one photograph."""
    path, result = rendered
    row = {"model": config.name, "build": config.build.label, "ref": ref}
    row |= {
        "family": family_of(config.model),
        "ok": result.ok,
        "seconds": result.seconds,
    }
    if not result.ok:
        return row | {"last": result.last}
    output = pixels.load(path)
    base = target(source, output)
    reference = base if ref == "input" else against
    if reference is not None and config.name != ref:
        row |= pixels.compare(reference, output, pixels.flat_mask(pixels.luma(base)))
    if row["family"] in QUALITY_FAMILIES:
        row |= headline(quality.measures(source, output, options.tile, options.overlap))
    return row


def run_models(comparison: Comparison, options: Options, tag: str | None) -> None:
    """Run models of one family (on one or more builds) on the same photographs.

    Each against the input or a reference config, and its time; with the quality
    tier's headline measures against the input for a sharpen or denoise.
    """
    family, ref = comparison.family, comparison.ref
    for name in comparison.names:
        if family_of(name) != family:
            raise SystemExit(f"{name} is not a {family} model")
    configs = comparison.configs()
    known = [c.name for c in configs]
    if ref != "input" and ref not in known:
        raise SystemExit(f"--ref {ref}: not one of input, {', '.join(known)}")
    subject = f"{family}-{'-'.join(comparison.names)}"
    tag = tag or default_tag("models", subject, comparison.builds[0])

    def score(photo: Photo, outputs: Outputs, source: np.ndarray) -> list[dict]:
        against = None
        if ref != "input" and outputs[ref][1].ok:
            against = pixels.load(outputs[ref][0])
        return [
            {"id": photo.id}
            | models_row(
                config,
                outputs[config.name],
                source=source,
                against=against,
                options=options,
                ref=ref,
            )
            for config in configs
        ]

    meta = {
        "family": family,
        "models": comparison.names,
        "ref": ref,
        "quality": comparison.quality,
        "configs": known,
    }
    plan = Plan("models", tag, comparison.builds, configs, meta, score)
    enhancement_run(plan, options)
