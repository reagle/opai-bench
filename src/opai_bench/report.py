"""Read runs back: `summary`, `compare` and `check` for every kind of run.

A run's kind is in its `run.json` (`lensblur`, `sharpen`, `precision`, `models`); a
run made before there was one is a lens blur run. Lens blur rows are per dataset
(`aim.jsonl`, `spring.jsonl`); the others are one row per photograph and model in
`<kind>.jsonl`, compared model by model.
"""

__author__ = "Joseph Reagle"
__copyright__ = "Copyright (C) 2026 Joseph Reagle"
__license__ = "MIT"

import json
import math
import operator
from dataclasses import dataclass, field

import numpy as np
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from opai_bench.config import runs_dir
from opai_bench.datasets import DATASETS, MATTE_SETS

# Wide enough for a table when the output is not a terminal (a pipe, a log file),
# where rich would otherwise assume 80 columns and truncate.
console = Console(highlight=False, width=None if Console().is_terminal else 160)

# --- Reading -----------------------------------------------------------------------


def kind_of(tag: str) -> str:
    """Return a run's kind, from its `run.json`; without one, a lens blur run."""
    meta = runs_dir() / tag / "run.json"
    if not meta.exists():
        return "lensblur"
    return json.loads(meta.read_text()).get("kind", "lensblur")


def rows(tag: str, dataset: str) -> list[dict]:
    """Return a run's rows for one dataset or kind (`<dataset>.jsonl`)."""
    path = runs_dir() / tag / f"{dataset}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def means(items: list[dict]) -> dict[str, float]:
    """Return the mean of each float field over the items that have it, finite only.

    >>> means([{"a": 1.0, "b": "x"}, {"a": 3.0}])
    {'a': 2.0}
    """
    keys = [
        key for item in items for key, value in item.items() if isinstance(value, float)
    ]
    found = {}
    for key in dict.fromkeys(keys):
        values = [
            item[key]
            for item in items
            if isinstance(item.get(key), float) and math.isfinite(item[key])
        ]
        if values:
            found[key] = float(np.mean(values))
    return found


def short(row: dict) -> str:
    """Return a row's float fields as `key=value`."""
    return " ".join(f"{k}={v:.4g}" for k, v in row.items() if isinstance(v, float))


def by_model(items: list[dict]) -> dict[str, dict[str, dict]]:
    """Return rows grouped by model, then by photograph id."""
    grouped: dict[str, dict[str, dict]] = {}
    for item in items:
        grouped.setdefault(item["model"], {})[item["id"]] = item
    return grouped


def flag(dataset: str) -> str:
    """Return the field a lens blur row of `dataset` says it was scored by."""
    return "depth" if dataset == "spring" else "matted"


# --- Concise tables ----------------------------------------------------------------

OPS = {"<": operator.lt, ">": operator.gt}


@dataclass(frozen=True)
class Flag:
    """Mark a row whose `key` crosses `value`, optionally only for some families."""

    key: str
    op: str
    value: float
    families: tuple[str, ...] = ()
    when_ref: bool = False  # models: only against a reference model, not the input

    def crossed(self, row: dict) -> bool:
        """Return whether `row` crosses this flag.

        >>> low = Flag("psnr", "<", 40)
        >>> low.crossed({"psnr": 21.7}), low.crossed({"psnr": 62.0}), low.crossed({})
        (True, False, False)
        """
        if self.families and row.get("family") not in self.families:
            return False
        if self.when_ref and row.get("ref", "input") == "input":
            return False
        value = row.get(self.key)
        return isinstance(value, float) and OPS[self.op](value, self.value)

    def __str__(self) -> str:
        """Return the flag as a marker names it."""
        return f"{self.key} {self.op} {self.value:g}"


@dataclass(frozen=True)
class View:
    """Say what a kind's concise table shows, what it flags, and how to read it."""

    columns: list[tuple[str, str, str]]  # key, heading, format
    flags: list[Flag] = field(default_factory=list)
    guide: str = ""


QUALITY_COLUMNS = [
    ("mottle", "mottle", ".2f"),
    ("inv_max", "inv max", ".2f"),
    ("edge_gain_2_8", "edge 2-8", ".3f"),
]
QUALITY_FLAGS = [
    Flag("mottle", ">", 2.0),
    Flag("seam_excess", ">", 0.1),
    Flag("inv_max", ">", 5.0),
    Flag("edge_gain_2_8", "<", 1.0, ("sharpen",)),
]
QUALITY_GUIDE = (
    "mottle and seam in grey levels (about 1 is visible on a smooth gradient); "
    "inv max over 5 is a tile inventing texture (the app's guard limit); "
    "edge 2-8 over 1 sharpens."
)
VIEWS = {
    "sharpen": View(
        [
            ("seam_excess", "seam", ".3f"),
            ("tile_sd", "tile sd", ".3f"),
            *QUALITY_COLUMNS,
            ("seconds", "s", ".1f"),
        ],
        QUALITY_FLAGS,
        f"Against each model's input: {QUALITY_GUIDE}",
    ),
    "precision": View(
        [
            ("max_abs", "max|d|", ".0f"),
            ("mean_abs", "mean|d|", ".3f"),
            ("psnr", "PSNR", ".1f"),
            ("ssim", "SSIM", ".4f"),
            ("psnr_flat", "PSNR flat", ".1f"),
            ("psnr_detail", "PSNR detail", ".1f"),
            ("edge_gain_2_8_sd", "edge SD", ".3f"),
            ("seconds_hd", "s HD", ".1f"),
            ("seconds_sd", "s SD", ".1f"),
        ],
        [
            Flag("psnr", "<", 40.0),
            Flag("max_abs", ">", 16.0),
            Flag("inv_max_sd", ">", 5.0),
            Flag("edge_gain_2_8_sd", "<", 1.0, ("sharpen",)),
        ],
        "SD (FP16) against HD (FP32). PSNR over 50 dB and SSIM over 0.999: "
        "indistinguishable at 8 bits; max|d| is in 8-bit levels, and far above mean|d| "
        "it is a local blow-up (rerun with --keep and look).",
    ),
    "models": View(
        [
            ("mean_abs", "mean|d|", ".3f"),
            ("psnr", "PSNR", ".1f"),
            ("ssim", "SSIM", ".4f"),
            ("psnr_flat", "PSNR flat", ".1f"),
            ("psnr_detail", "PSNR detail", ".1f"),
            *QUALITY_COLUMNS,
            ("seconds", "s", ".1f"),
        ],
        [Flag("psnr", "<", 40.0, when_ref=True), *QUALITY_FLAGS],
        f"PSNR and SSIM against the reference (or the input); {QUALITY_GUIDE}",
    ),
    "aim": View(
        [
            ("sep_mse", "sep_mse", ".4f"),
            ("sep_share", "sep share", ".2f"),
            ("mse", "mse", ".3f"),
            ("specks", "specks", ".0f"),
        ],
        [],
        "sep_mse: squared matte error where a sharp subject meets a blurred background "
        "(lower is better); a photograph with no matte blurred too little to need one.",
    ),
    "spring": View(
        [
            ("delta1", "delta1", ".3f"),
            ("boundary_f1", "boundary F1", ".3f"),
            ("edges_found", "edges", ".0f"),
        ],
        [Flag("edges_found", "<", 1.0)],
        "delta1: depth ordering; boundary F1: occlusion edges (higher is better).",
    ),
}


def crossings(view: View, row: dict) -> str:
    """Return the flags a row crosses, as the marker shows them."""
    crossed = [str(flagged) for flagged in view.flags if flagged.crossed(row)]
    if not row.get("ok", True):
        crossed.insert(0, f"FAILED: {row.get('last', '')}"[:60])
    return escape("! " + ", ".join(crossed)) if crossed else ""


def cell(row: dict, key: str, form: str) -> str:
    """Return one cell of a concise table: the value, or a dash."""
    value = row.get(key)
    return format(value, form) if isinstance(value, float) else "-"


def table(title: str, view: View, items: list[dict], mean_rows: list[dict]) -> None:
    """Print a concise table: a row per item, the means, the flags and a guide."""
    shown = [
        c for c in view.columns if any(isinstance(i.get(c[0]), float) for i in items)
    ]
    grid = Table(title=title, title_justify="left", box=None, pad_edge=False)
    grid.add_column("photograph", no_wrap=True)
    grid.add_column("model", no_wrap=True)
    for _, heading, _ in shown:
        grid.add_column(heading, justify="right", no_wrap=True)
    grid.add_column("")
    for item in items:
        cells = [cell(item, key, form) for key, _, form in shown]
        grid.add_row(item["id"], item.get("model", ""), *cells, crossings(view, item))
    if mean_rows:
        grid.add_section()
    for mean in mean_rows:
        cells = [cell(mean, key, form) for key, _, form in shown]
        grid.add_row(
            "[bold]mean[/]", mean.get("model", ""), *cells, mean.get("count", "")
        )
    console.print(grid)
    if view.guide:
        console.print(f"[dim]{view.guide} ! marks a row past a threshold.[/]")


# --- Summaries ---------------------------------------------------------------------


def summary_lensblur(tag: str, everything: bool = False) -> None:
    """Print a lens blur run: a concise table per dataset, or every score."""
    for dataset in DATASETS:
        items = rows(tag, dataset)
        if not items:
            continue
        done = [item for item in items if item.get(flag(dataset))]
        if everything:
            for item in items:
                console.print(f"{dataset} {item['id']}: {short(item)}")
        mean = means(done) | {"count": f"{len(done)}/{len(items)} scored"}
        view = VIEWS["aim" if dataset in MATTE_SETS else dataset]
        table(f"{tag} {dataset}", view, items, [mean])


def summary_enhancement(tag: str, kind: str, everything: bool = False) -> None:
    """Print an enhancement run: a concise table with each model's means."""
    items = rows(tag, kind)
    if everything:
        for item in items:
            console.print(f"{item['id']} {item.get('model', '')}: {short(item)}")
    mean_rows = []
    for model, found in by_model(items).items():
        done = [row for row in found.values() if row.get("ok")]
        family = next((row.get("family") for row in done), None)
        mean_rows.append(
            means(done)
            | {
                "model": model,
                "family": family,
                "count": f"{len(done)}/{len(found)} ok",
            }
        )
    table(tag, VIEWS[kind], items, mean_rows)


def summary(tag: str, everything: bool = False) -> None:
    """Print a run's scores: concise, or with `everything` every field of every row."""
    kind = kind_of(tag)
    if kind == "lensblur":
        summary_lensblur(tag, everything)
    else:
        summary_enhancement(tag, kind, everything)


# --- Limits (REGIME.md) --------------------------------------------------------------

# A mean may move by this much in the worse direction before it is a regression:
# (worse, limit, "relative" or "absolute", floor). A relative limit is never tighter
# than its floor, so a measure near zero does not fail on noise. "either" is any
# change past the limit: for scores against the input, where a change is not better
# or worse, but the output is not the same any more. Provisional, all of them.
Rule = tuple[str, float, str, float]
MATTE_LIMITS: dict[str, Rule] = {
    "sep_mse": ("up", 0.05, "relative", 0.0),
    "sep_sad": ("up", 0.05, "relative", 0.0),
}
LENSBLUR_LIMITS = dict.fromkeys(MATTE_SETS, MATTE_LIMITS) | {
    "spring": {
        "boundary_f1": ("down", 0.01, "absolute", 0.0),
        "delta1": ("down", 0.01, "absolute", 0.0),
    },
}
QUALITY_LIMITS: dict[str, Rule] = {
    "seam_excess": ("up", 0.10, "absolute", 0.0),
    "mottle": ("up", 0.20, "relative", 0.10),
    "inv_max": ("up", 0.20, "relative", 0.20),
    "edge_gain_2_8": ("down", 0.03, "absolute", 0.0),
}
LIMITS: dict[str, dict[str, Rule]] = {
    "sharpen": QUALITY_LIMITS,
    "precision": {
        "psnr": ("down", 1.0, "absolute", 0.0),
        "psnr_detail": ("down", 1.0, "absolute", 0.0),
        "ssim": ("down", 0.002, "absolute", 0.0),
    }
    | {f"{key}_sd": rule for key, rule in QUALITY_LIMITS.items()},
    "models": {
        "psnr": ("either", 1.0, "absolute", 0.0),
        "ssim": ("either", 0.002, "absolute", 0.0),
    }
    | QUALITY_LIMITS,
}
# A sharpener whose 2--8 px edge gain falls under this no longer sharpens (sharpen
# suite, sharpen models only).
MINIMUMS = {"edge_gain_2_8": 1.0}
# Time is reported, not judged: a slower mean past this is a warning.
SLOWER = 0.25
TIMES = ("seconds", "seconds_hd", "seconds_sd")
KEY = {"sharpen": "mottle", "precision": "psnr", "models": "psnr"}


def past(before: float, after: float, rule: Rule) -> bool:
    """Return whether `after` is worse than `before` past a limit.

    >>> past(1.0, 1.04, ("up", 0.05, "relative", 0.0))
    False
    >>> past(1.0, 1.06, ("up", 0.05, "relative", 0.0))
    True
    >>> past(0.0, 0.05, ("up", 0.2, "relative", 0.1))
    False
    >>> past(0.5, 0.4, ("either", 0.05, "absolute", 0.0))
    True
    """
    worse, limit, kind, floor = rule
    change = after - before
    if worse == "down":
        change = -change
    elif worse == "either":
        change = abs(change)
    allowed = max(limit * abs(before), floor) if kind == "relative" else limit
    return change > allowed


def judged(name: str, before: dict, after: dict, limits: dict[str, Rule]) -> list[str]:
    """Return a failure for each mean of `limits` past its rule."""
    return [
        f"{name} {metric}: {before[metric]:.4g} -> {after[metric]:.4g} "
        f"(limit {rule[1]:g} {rule[2]})"
        for metric, rule in limits.items()
        if metric in before
        and metric in after
        and past(before[metric], after[metric], rule)
    ]


def side_by_side(name: str, before: dict, after: dict, count: int) -> None:
    """Print two sets of means side by side."""
    console.print(f"{name} ({count} items)")
    for metric, value in before.items():
        if metric in after:
            console.print(f"  {metric:16} {value:10.4g} -> {after[metric]:10.4g}")


def most_moved(
    old: dict[str, dict], new: dict[str, dict], common: list, key: str
) -> None:
    """Print the five items that moved most by `key`."""
    moved = [i for i in common if key in old[i] and key in new[i]]
    moved.sort(key=lambda i: abs(new[i][key] - old[i][key]), reverse=True)
    listed = ", ".join(f"{i} {old[i][key]:.4g}->{new[i][key]:.4g}" for i in moved[:5])
    console.print(f"  most moved by {key}: {listed}")


def lensblur_pairs(old: str, new: str):
    """Yield each dataset's rows of two lens blur runs and the ids both scored."""
    for dataset in DATASETS:
        scored = flag(dataset)
        before = {r["id"]: r for r in rows(old, dataset)}
        after = {r["id"]: r for r in rows(new, dataset)}
        common = [
            i
            for i in before
            if i in after and before[i].get(scored) and after[i].get(scored)
        ]
        yield dataset, before, after, common


def enhancement_pairs(old: str, new: str, kind: str):
    """Yield each model's rows of two runs, and the ids both have an output for."""
    before, after = by_model(rows(old, kind)), by_model(rows(new, kind))
    for model, items in before.items():
        theirs = after.get(model, {})
        common = [
            i
            for i, row in items.items()
            if row.get("ok") and theirs.get(i, {}).get("ok")
        ]
        yield model, items, theirs, common


def compare(old: str, new: str) -> None:
    """Print two runs' means side by side, and the items that moved most."""
    kind = kind_of(old)
    if kind != kind_of(new):
        raise SystemExit(f"{old} is a {kind} run and {new} a {kind_of(new)} run")
    if kind == "lensblur":
        for dataset, before, after, common in lensblur_pairs(old, new):
            if common:
                a, b = (
                    means([before[i] for i in common]),
                    means([after[i] for i in common]),
                )
                side_by_side(dataset, a, b, len(common))
                key = "boundary_f1" if dataset == "spring" else "sep_mse"
                most_moved(before, after, common, key)
        return
    for model, before, after, common in enhancement_pairs(old, new, kind):
        if common:
            a, b = means([before[i] for i in common]), means([after[i] for i in common])
            side_by_side(model, a, b, len(common))
            most_moved(before, after, common, KEY[kind])


def check_lensblur(old: str, new: str) -> list[str]:
    """Return the lens blur run's regressions: limits, and items no longer scored."""
    failures = []
    for dataset, before, after, common in lensblur_pairs(old, new):
        scored = flag(dataset)
        lost = [
            i
            for i in before
            if before[i].get(scored) and i in after and not after[i].get(scored)
        ]
        failures += [f"{dataset} {i}: no longer scored" for i in lost]
        if common:
            a, b = means([before[i] for i in common]), means([after[i] for i in common])
            failures += judged(dataset, a, b, LENSBLUR_LIMITS[dataset])
    return failures


def check_enhancement(old: str, new: str, kind: str) -> tuple[list[str], list[str]]:
    """Return an enhancement run's regressions and its timing warnings."""
    failures, warnings = [], []
    for model, before, after, common in enhancement_pairs(old, new, kind):
        if not after:
            failures.append(f"{model}: not in {new}")
            continue
        lost = [i for i, row in before.items() if row.get("ok") and i in after]
        failures += [
            f"{model} {i}: no output any more" for i in lost if not after[i].get("ok")
        ]
        if not common:
            continue
        a, b = means([before[i] for i in common]), means([after[i] for i in common])
        failures += judged(model, a, b, LIMITS[kind])
        if kind == "sharpen" and before[common[0]].get("family") == "sharpen":
            failures += [
                f"{model} {metric}: {b[metric]:.4g} is under the minimum {minimum:g}"
                for metric, minimum in MINIMUMS.items()
                if metric in b and b[metric] < minimum
            ]
        warnings += [
            f"{model} {metric}: {a[metric]:.3g} -> {b[metric]:.3g} s"
            for metric in TIMES
            if metric in a and metric in b and b[metric] > (1 + SLOWER) * a[metric]
        ]
    return failures, warnings


def check(old: str, new: str) -> int:
    """Compare, then judge each limit and each item NEW no longer scores; 1 on failure.

    No longer scored: no matte or no depth, or no output, where OLD had one.
    """
    compare(old, new)
    kind = kind_of(old)
    if kind == "lensblur":
        failures, warnings = check_lensblur(old, new), []
    else:
        failures, warnings = check_enhancement(old, new, kind)
    for warning in warnings:
        console.print(f"[yellow]SLOWER[/] {warning}")
    for failure in failures:
        console.print(f"[red]REGRESSION[/] {failure}")
    console.print("check: " + ("[red]FAIL[/]" if failures else "[green]pass[/]"))
    return 1 if failures else 0
