"""Benchmark Open Photo AI's CLI: lens blur, sharpen/denoise, precision, models.

Commands:
    fetch aim|spring [--accept] [--dry-run]
        Show a dataset's terms, ask for agreement, download it.
    run TAG [--suite lensblur|sharpen] [--opai PATH] [--tier core|quick|full]
        A suite over a tier; scores go to runs/TAG/. lensblur: depth (Spring) and
        matte (AIM-500) against ground truth, with a build that has the
        OPAI_BENCH_DUMP hook. sharpen: seams, mottle, bands, invention and time of
        every sharpen model and Stockholm denoise, against their input.
    precision MODEL|FAMILY [--opai PATH]
        Each model at HD (FP32) and SD (FP16): SD against HD, and the times.
    models FAMILY A B ... [--opai [LABEL=]PATH ...] [--ref input|MODEL[@LABEL]]
        Models (on one or more builds) on the same photographs, each against the
        input or a reference, and the times.
    summary TAG [--all]          A run's concise table, or every score.
    compare OLD NEW              Means side by side, and the items that moved most.
    check OLD NEW                Compare, then exit 1 past the limits (REGIME.md).
    select-core TAG [TAG...]     Suggest the lens blur core tier from reference runs.

Data, runs and the isolated opai home are ./data, ./runs and ./opai-home, or --data,
--runs and --home (or OPAI_BENCH_DATA, OPAI_BENCH_RUNS, OPAI_BENCH_HOME).
"""

__author__ = "Joseph Reagle"
__copyright__ = "Copyright (C) 2026 Joseph Reagle"
__license__ = "MIT"

import argparse  # http://docs.python.org/dev/library/argparse.html
import logging
import os
import sys
from pathlib import Path

from opai_bench import __version__, report, suites
from opai_bench.config import INSTALLED_OPAI, MANIFEST, default_opai
from opai_bench.datasets import DATASETS, FETCHABLE, fetch
from opai_bench.runner import FAMILIES, Build

LOG_FILE = "opai-bench.log"


class Stderr:
    """Write to whatever `sys.stderr` is now, so log lines pass the progress bar."""

    def write(self, text: str) -> int:
        return sys.stderr.write(text)

    def flush(self) -> None:
        sys.stderr.flush()


def setup_logging(verbosity: int, log_to_file: bool) -> None:
    """Configure logging based on verbosity flags."""
    log_level = logging.ERROR - (verbosity * 10)
    handler = (
        logging.FileHandler(LOG_FILE, mode="w")
        if log_to_file
        else logging.StreamHandler(Stderr())
    )
    logging.basicConfig(
        level=log_level,
        format="%(levelname).4s %(funcName).10s:%(lineno)-4d| %(message)s",
        handlers=[handler],
        force=True,
    )


def common_options() -> argparse.ArgumentParser:
    """Return the options every command takes: locations, verbosity, log file."""
    common = argparse.ArgumentParser(add_help=False)
    group = common.add_argument_group("locations")
    group.add_argument("--data", type=Path, help="datasets folder (default ./data)")
    group.add_argument("--runs", type=Path, help="runs folder (default ./runs)")
    group.add_argument(
        "--home", type=Path, help="isolated HOME for opai (default ./opai-home)"
    )
    common.add_argument(
        "-L",
        "--log-to-file",
        action="store_true",
        default=False,
        help=f"log to file {LOG_FILE}",
    )
    common.add_argument(
        "-V",
        "--verbose",
        action="count",
        default=0,
        help="increase verbosity from critical though error, warning, info, and debug",
    )
    return common


def photo_options(parser: argparse.ArgumentParser) -> None:
    """Add the options of the commands that enhance photographs."""
    group = parser.add_argument_group("photographs and opai")
    group.add_argument("--tier", choices=["core", "quick", "full"], default="core")
    group.add_argument("--ids", help="comma-separated AIM-500 ids instead of the tier")
    group.add_argument(
        "--photos", nargs="+", metavar="FILE", help="your own photographs instead"
    )
    group.add_argument("--limit", type=int, help="only the first N photographs")
    group.add_argument(
        "--no-synthetic",
        dest="synthetic",
        action="store_false",
        help="leave out the synthetic gradient",
    )
    group.add_argument(
        "--no-warmup",
        dest="warmup",
        action="store_false",
        help="skip the untimed first run of each model",
    )
    group.add_argument(
        "--keep", action="store_true", help="keep outputs in runs/TAG/out"
    )
    group.add_argument(
        "--scale", type=float, default=2.0, help="upscale factor (default: %(default)s)"
    )
    group.add_argument("--args", help='more opai arguments, quoted ("--sh-str 150")')
    sharpen = MANIFEST["sharpen"]
    group.add_argument(
        "--tile", type=int, default=sharpen["tile"], help="tile (default: %(default)s)"
    )
    group.add_argument(
        "--overlap",
        type=int,
        default=sharpen["overlap"],
        help="tile overlap (default: %(default)s)",
    )


def opai_option(parser: argparse.ArgumentParser, several: bool = False) -> None:
    """Add `--opai`: one build, or (`several`) one or more labelled builds."""
    if several:
        parser.add_argument(
            "--opai",
            action="append",
            metavar="[LABEL=]PATH",
            help=f"a build; repeat to compare builds (default {INSTALLED_OPAI})",
        )
    else:
        parser.add_argument(
            "--opai", type=Path, help=f"the build (default {INSTALLED_OPAI})"
        )


def add_commands(commands, common: argparse.ArgumentParser) -> None:
    """Add every command's parser."""
    fetcher = commands.add_parser("fetch", parents=[common], help="download a dataset")
    fetcher.add_argument("dataset", choices=FETCHABLE)
    fetcher.add_argument(
        "--accept", action="store_true", help="agree to the terms without being asked"
    )
    fetcher.add_argument(
        "--dry-run", action="store_true", help="show what would be done; change nothing"
    )

    runner = commands.add_parser(
        "run", parents=[common], help="run a suite over a tier"
    )
    runner.add_argument("tag")
    runner.add_argument("--suite", choices=["lensblur", "sharpen"], default="lensblur")
    opai_option(runner)
    runner.add_argument(
        "--only", default=",".join(DATASETS), help="lensblur: datasets (aim,spring)"
    )
    runner.add_argument(
        "-q", "--quality", choices=["hd", "sd"], help="sharpen: quality (CLI's default)"
    )
    photo_options(runner)

    precision = commands.add_parser("precision", parents=[common], help="SD against HD")
    precision.add_argument(
        "subject",
        metavar="MODEL|FAMILY",
        help=f"a model or a family: {', '.join(FAMILIES)}",
    )
    opai_option(precision)
    precision.add_argument(
        "--tag", help="the run's tag (default precision-SUBJECT-VERSION)"
    )
    photo_options(precision)

    models = commands.add_parser("models", parents=[common], help="models side by side")
    models.add_argument("family", choices=list(FAMILIES))
    models.add_argument("names", nargs="+", metavar="MODEL")
    opai_option(models, several=True)
    models.add_argument("--ref", default="input", help="input, or MODEL[@LABEL]")
    models.add_argument(
        "-q", "--quality", choices=["hd", "sd"], help="quality (the CLI's default)"
    )
    models.add_argument(
        "--tag", help="the run's tag (default models-FAMILY-...-VERSION)"
    )
    photo_options(models)

    summary = commands.add_parser("summary", parents=[common], help="a run's scores")
    summary.add_argument("tag")
    summary.add_argument("--all", action="store_true", help="every score of every item")
    for name, text in (("compare", "two runs side by side"), ("check", "judge limits")):
        comparer = commands.add_parser(name, parents=[common], help=text)
        comparer.add_argument("old")
        comparer.add_argument("new")
    select = commands.add_parser(
        "select-core", parents=[common], help="suggest the lens blur core tier"
    )
    select.add_argument("tags", nargs="+")


def parser() -> argparse.ArgumentParser:
    """Return the command line's parser."""
    arg_parser = argparse.ArgumentParser(
        prog="opai-bench",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    arg_parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    commands = arg_parser.add_subparsers(dest="command", required=True)
    add_commands(commands, common_options())
    return arg_parser


def announce(builds: list[Build], defaulted: bool) -> None:
    """Print which opai builds a run uses, and warn when none was named."""
    for build in builds:
        build.check()
        report.console.print(f"opai {build.label}: {build.version}  ({build.path})")
    if defaulted:
        report.console.print(
            "[yellow]No --opai given: using the default opai; name a build with --opai.[/]"
        )


def one_build(given: Path | None) -> Build:
    """Return the build `--opai` names, or the default, announced."""
    build = Build("b1", given or default_opai())
    announce([build], given is None)
    return build


def run(args: argparse.Namespace) -> None:
    """Run a suite."""
    build = one_build(args.opai)
    if args.suite == "lensblur":
        ids = set(args.ids.split(",")) if args.ids else None
        suites.run_lensblur(args.tag, build, args.tier, set(args.only.split(",")), ids)
    else:
        options = suites.Options.from_args(args)
        suites.run_sharpen(args.tag, build, options, args.quality)


def models(args: argparse.Namespace) -> None:
    """Compare models, or builds."""
    given = args.opai or [str(default_opai())]
    builds = [Build.parse(text, i) for i, text in enumerate(given)]
    announce(builds, not args.opai)
    comparison = suites.Comparison(
        args.family, args.names, builds, args.ref, args.quality
    )
    suites.run_models(comparison, suites.Options.from_args(args), args.tag)


def select_core(args: argparse.Namespace) -> None:
    """Suggest the lens blur core tier."""
    from opai_bench.select_core import select_core as select  # numpy-heavy, rarely used

    select(args.tags)


COMMANDS = {
    "fetch": lambda args: sys.exit(fetch(args.dataset, args.accept, args.dry_run)),
    "run": run,
    "precision": lambda args: suites.run_precision(
        args.subject, one_build(args.opai), suites.Options.from_args(args), args.tag
    ),
    "models": models,
    "summary": lambda args: report.summary(args.tag, args.all),
    "compare": lambda args: report.compare(args.old, args.new),
    "check": lambda args: sys.exit(report.check(args.old, args.new)),
    "select-core": select_core,
}


def main(argv: list[str] | None = None) -> None:
    """Process arguments and run the command."""
    args = parser().parse_args(argv)
    setup_logging(args.verbose, args.log_to_file)
    for name in ("data", "runs", "home"):
        if getattr(args, name, None):
            place = getattr(args, name).expanduser().absolute()
            os.environ[f"OPAI_BENCH_{name.upper()}"] = str(place)
    COMMANDS[args.command](args)


if __name__ == "__main__":
    main()
