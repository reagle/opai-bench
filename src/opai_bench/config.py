"""Locate the data, the runs and the isolated opai home; load the manifest.

Each location is an environment variable, set by the command line's `--data`, `--runs`
and `--home`, so every module (and `select-core`) reads the same one. Relative paths
are taken from the current folder, so a clone's `data` and `runs` (often symlinks to a
bigger disk) are found when it is run from the clone.
"""

__author__ = "Joseph Reagle"
__copyright__ = "Copyright (C) 2026 Joseph Reagle"
__license__ = "MIT"

import os
import shutil
import tomllib
from importlib import resources
from pathlib import Path

HOME = Path.home()
MANIFEST: dict = tomllib.loads(
    resources.files("opai_bench").joinpath("manifest.toml").read_text()
)
INSTALLED_OPAI = HOME / ".local/bin/opai"


def default_opai() -> Path:
    """Return the opai used without `--opai`: the installed one, else PATH's."""
    if INSTALLED_OPAI.exists():
        return INSTALLED_OPAI
    found = shutil.which("opai")
    return Path(found) if found else INSTALLED_OPAI


def _place(variable: str, default: str) -> Path:
    return Path(os.environ.get(variable) or default).expanduser().absolute()


def data_dir() -> Path:
    """Return where datasets live: `OPAI_BENCH_DATA`, or `./data`."""
    return _place("OPAI_BENCH_DATA", "data")


def runs_dir() -> Path:
    """Return where runs go, one folder a tag: `OPAI_BENCH_RUNS`, or `./runs`."""
    return _place("OPAI_BENCH_RUNS", "runs")


def home_dir() -> Path:
    """Return the HOME enhancements run opai under: `OPAI_BENCH_HOME`, or `./opai-home`.

    Its run cache is cleared before each timed run without touching the user's own.
    """
    return _place("OPAI_BENCH_HOME", "opai-home")


def dataset_dir(name: str) -> Path:
    """Return one dataset's folder: `OPAI_BENCH_<NAME>`, or its own in `data_dir()`.

    >>> os.environ["OPAI_BENCH_DATA"] = "/tmp/d"
    >>> str(dataset_dir("aim"))
    '/tmp/d/AIM-500'
    >>> del os.environ["OPAI_BENCH_DATA"]
    """
    override = os.environ.get(f"OPAI_BENCH_{name.upper()}")
    if override:
        return Path(override).expanduser().absolute()
    return data_dir() / MANIFEST[name]["folder"]
