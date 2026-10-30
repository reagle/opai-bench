"""Call the opai CLI: the lens blur with its bench dump, an enhancement timed.

Enhancements run under an isolated HOME (`Home`) whose run cache is cleared before
each timed call; the lens blur runs under the user's own, as it always has.
"""

__author__ = "Joseph Reagle"
__copyright__ = "Copyright (C) 2026 Joseph Reagle"
__license__ = "MIT"

import functools
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from opai_bench.config import HOME, home_dir

APP = "io.vinicius.opai"
# What an isolated home links to in the user's own config folder: ONNX Runtime and
# its libraries. The weights are linked one by one (`Home`).
SHARED = ("runtime", "libs")
MARKER = ".opai-bench-home"

# Each family's flag and its models (the CLI's `--help` lists them; the first is the
# CLI's default).
FAMILIES: dict[str, tuple[str, list[str]]] = {
    "sharpen": ("--sh", ["moscow", "petersburg", "novgorod"]),
    "denoise": ("--dn", ["stockholm", "gothenburg", "malmo"]),
    "upscale": ("--up", ["tokyo", "kyoto", "saitama"]),
    "light": ("--la", ["paris", "lyon"]),
    "colour": ("--cb", ["rio", "saopaulo"]),
    "lensblur": ("--lb", ["lisbon"]),
}


def family_of(model: str) -> str:
    """Return the family a model belongs to.

    >>> family_of("petersburg"), family_of("Kyoto")
    ('sharpen', 'upscale')
    """
    for family, (_, models) in FAMILIES.items():
        if model.lower() in models:
            return family
    known = ", ".join(m for _, ms in FAMILIES.values() for m in ms)
    raise SystemExit(f"unknown model {model!r}; known: {known}")


@dataclass(frozen=True)
class Build:
    """Name an opai CLI to run, by a short label (`installed`, `dev`) for reports."""

    label: str
    path: Path

    @classmethod
    def parse(cls, text: str, index: int = 0) -> "Build":
        """Return the build `LABEL=PATH` names, or a bare path labelled by position.

        >>> Build.parse("dev=/tmp/opai")
        Build(label='dev', path=PosixPath('/tmp/opai'))
        >>> Build.parse("/usr/bin/opai", 1).label
        'b2'
        """
        if "=" in text:
            label, _, path = text.partition("=")
        else:
            label, path = f"b{index + 1}", text
        return cls(label, Path(path).expanduser())

    def check(self) -> None:
        """Exit unless the build exists."""
        if not shutil.which(str(self.path)) and not self.path.exists():
            raise SystemExit(f"no opai at {self.path}")

    @functools.cached_property
    def version(self) -> str:
        """Return the build's `--version`."""
        done = subprocess.run(
            [str(self.path), "--version"], capture_output=True, text=True, check=False
        )
        return done.stdout.strip() or f"(no version: exit {done.returncode})"

    @functools.cached_property
    def help(self) -> str:
        """Return the build's `--help`."""
        command = [str(self.path), "--help"]
        return subprocess.run(
            command, capture_output=True, text=True, check=False
        ).stdout

    def supports(self, flag: str) -> bool:
        """Return whether the build's `--help` names `flag`."""
        return flag in self.help


# --- Lens blur, with the bench dump --------------------------------------------------


def load_dump(path: Path) -> np.ndarray:
    """Return one `OPAI_BENCH_DUMP` plane: width, height (u32), then f32 values."""
    raw = path.read_bytes()
    width, height = np.frombuffer(raw[:8], "<u4")
    return np.frombuffer(raw[8:], "<f4").reshape(int(height), int(width))


def run_dump(
    opai: Path, source: Path, args: list[str], work: Path
) -> tuple[dict[str, np.ndarray], str]:
    """Run `opai` on `source`, returning what it dumped and its last line.

    The dumps are `depth`, `matte` and `radius` (the blur's radius at each depth
    pixel); any it did not write is missing. Under the user's own home and run cache.
    """
    prefix = work / "dump"
    for stale in work.glob("dump-*"):
        stale.unlink()
    env = os.environ | {"OPAI_BENCH_DUMP": str(prefix)}
    output = work / f"out{source.suffix}"
    command = [str(opai), str(source), *args, "-o", str(output)]
    logging.debug("%s", " ".join(command))
    done = subprocess.run(command, env=env, capture_output=True, text=True, check=False)
    lines = (done.stdout + done.stderr).strip().splitlines()
    output.unlink(missing_ok=True)
    dumps = {
        name: load_dump(Path(f"{prefix}-{name}.f32"))
        for name in ("depth", "matte", "radius")
        if Path(f"{prefix}-{name}.f32").exists()
    }
    return dumps, lines[-1] if lines else f"exit {done.returncode}"


# --- Enhancements, timed under an isolated home -------------------------------------


def user_config() -> Path:
    """Return the user's own opai config folder (the platform's config dir)."""
    if sys.platform == "darwin":
        return HOME / "Library/Application Support" / APP
    return Path(os.environ.get("XDG_CONFIG_HOME") or HOME / ".config") / APP


class Home:
    """Give opai a HOME of the bench's own, with the user's weights linked in.

    Its own run cache and settings: the cache is cleared before every timed run, so
    each run computes and times its result, and the GUI's saved settings are not there
    to change what the CLI does. Compiled engines (CoreML) are kept, as in use.
    """

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or home_dir()
        if sys.platform not in ("darwin", "linux"):
            raise SystemExit("an isolated HOME for opai works on macOS and Linux only")
        relative = (
            "Library/Application Support" if sys.platform == "darwin" else ".config"
        )
        self.config = self.root / relative / APP
        self.config.mkdir(parents=True, exist_ok=True)
        (self.root / MARKER).touch()
        own = user_config()
        for name in SHARED:
            link = self.config / name
            if (own / name).exists() and not link.exists() and not link.is_symlink():
                link.symlink_to(own / name)
        self.link_weights(own / "models")

    def link_weights(self, weights: Path) -> None:
        """Link the user's weights one by one into a models folder of the home's own.

        A model the CLI downloads (one not installed at some precision) then lands
        here, never among the user's.
        """
        models = self.config / "models"
        if models.is_symlink() or not weights.is_dir():
            return
        models.mkdir(exist_ok=True)
        for weight in weights.iterdir():
            link = models / weight.name
            if not link.exists() and not link.is_symlink():
                link.symlink_to(weight)

    @property
    def env(self) -> dict[str, str]:
        """Return the environment opai runs in: this HOME, no dump, no XDG override."""
        env = os.environ | {"HOME": str(self.root)}
        env.pop("XDG_CONFIG_HOME", None)
        env.pop("OPAI_BENCH_DUMP", None)
        return env

    def clear_cache(self) -> None:
        """Clear the run cache, only ever under a home this class made (the marker)."""
        cache = self.config / "cache"
        if (self.root / MARKER).exists() and cache.is_dir() and not cache.is_symlink():
            shutil.rmtree(cache)


@dataclass
class Result:
    """Say how one CLI call went: its time, whether it wrote, its last line."""

    seconds: float
    ok: bool
    last: str


def enhance(
    build: Build, home: Home, source: Path, args: list[str], output: Path
) -> Result:
    """Run one enhancement of `source` to `output` with a cleared run cache, timed.

    The time is the whole call's (start-up, model load and inference: what a user
    waits for).
    """
    home.clear_cache()
    output.unlink(missing_ok=True)
    command = [str(build.path), str(source), *args, "-o", str(output)]
    if build.supports("--no-gui-settings"):
        command.append("--no-gui-settings")
    logging.debug("%s", " ".join(command))
    start = time.perf_counter()
    done = subprocess.run(
        command, env=home.env, capture_output=True, text=True, check=False
    )
    seconds = time.perf_counter() - start
    lines = (done.stdout + done.stderr).strip().splitlines()
    last = lines[-1] if lines else f"exit {done.returncode}"
    ok = done.returncode == 0 and output.exists()
    if not ok:
        logging.warning("%s failed: %s", " ".join(command), last)
    return Result(seconds, ok, last)


def model_args(
    model: str, quality: str | None, scale: float, extra: list[str], focus: str | None
) -> list[str]:
    """Return the CLI's arguments for one model at one quality (`hd`, `sd` or default).

    An upscale is at an explicit factor, since the CLI's own choice depends on the
    photograph; a lens blur at f/1.4 on `focus`.

    >>> model_args("petersburg", "sd", 2.0, [], None)
    ['--sh', 'petersburg', '-q', 'sd']
    >>> model_args("kyoto", None, 2.0, [], None)
    ['--up', 'kyoto', '--scale', '2']
    """
    family = family_of(model)
    args = [FAMILIES[family][0], model.lower()]
    if family == "upscale":
        args += ["--scale", f"{scale:g}"]
    if family == "lensblur":
        args += ["--lb-f", "1.4", "--lb-focus", focus or "0.5,0.5"]
    if quality:
        args += ["-q", quality]
    return args + extra
