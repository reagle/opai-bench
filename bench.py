#!/usr/bin/env python3
"""Run `uv run opai-bench ARGS` in this clone, as the old single-script bench.py ran.

Kept for scripts and habits: this clone's `data`, `runs` and `opai-home` are used
unless OPAI_BENCH_DATA, OPAI_BENCH_RUNS or OPAI_BENCH_HOME say otherwise.
"""

__author__ = "Joseph Reagle"
__copyright__ = "Copyright (C) 2026 Joseph Reagle"
__license__ = "MIT"

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for name, folder in (("DATA", "data"), ("RUNS", "runs"), ("HOME", "opai-home")):
    os.environ.setdefault(f"OPAI_BENCH_{name}", str(ROOT / folder))
os.execvp(
    "uv", ["uv", "run", "--quiet", "--project", str(ROOT), "opai-bench", *sys.argv[1:]]
)
