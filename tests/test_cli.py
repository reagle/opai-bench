"""Test what doctests cannot: the command line end to end, over made-up runs.

No opai and no dataset is needed.
"""

__author__ = "Joseph Reagle"
__copyright__ = "Copyright (C) 2026 Joseph Reagle"
__license__ = "MIT"

import json
from pathlib import Path

import pytest

from opai_bench import cli, report


@pytest.fixture
def places(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point data and runs at a fresh folder."""
    monkeypatch.setenv("OPAI_BENCH_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("OPAI_BENCH_RUNS", str(tmp_path / "runs"))
    monkeypatch.delenv("OPAI_BENCH_AIM", raising=False)
    monkeypatch.delenv("OPAI_BENCH_SPRING", raising=False)
    return tmp_path


def run_cli(*argv: str) -> int:
    """Run the command line, returning its exit status."""
    try:
        cli.main(list(argv))
    except SystemExit as done:
        return int(done.code or 0)
    return 0


def write_run(root: Path, tag: str, kind: str, rows: list[dict]) -> None:
    """Write a made-up run of `kind`."""
    folder = root / "runs" / tag
    folder.mkdir(parents=True)
    (folder / "run.json").write_text(json.dumps({"kind": kind, "builds": []}))
    lines = "".join(json.dumps(row) + "\n" for row in rows)
    (folder / f"{kind}.jsonl").write_text(lines)


def sharpen_row(photo: str, mottle: float, gain: float = 1.3, ok: bool = True) -> dict:
    """Return a made-up row of the sharpen suite."""
    return {
        "id": photo,
        "model": "petersburg",
        "family": "sharpen",
        "ok": ok,
        "seconds": 5.0,
        "mottle": mottle,
        "seam_excess": 0.04,
        "inv_max": 1.0,
        "edge_gain_2_8": gain,
    }


def test_fetch_dry_run_with_accept_changes_nothing(places: Path) -> None:
    """Agree and dry-run: nothing is downloaded or recorded."""
    elsewhere = places / "elsewhere"
    status = run_cli("fetch", "aim", "--accept", "--dry-run", "--data", str(elsewhere))
    assert status == 0
    assert not elsewhere.exists()


def test_check_passes_a_run_against_itself_and_fails_more_mottle(places: Path) -> None:
    """Fail a run whose mottle rose past the limit."""
    write_run(places, "old", "sharpen", [sharpen_row("a", 0.7), sharpen_row("b", 0.8)])
    write_run(places, "same", "sharpen", [sharpen_row("a", 0.7), sharpen_row("b", 0.8)])
    write_run(
        places, "worse", "sharpen", [sharpen_row("a", 4.5), sharpen_row("b", 3.1)]
    )
    assert run_cli("check", "old", "same") == 0
    assert run_cli("check", "old", "worse") == 1


def test_check_fails_a_sharpener_that_stops_sharpening(places: Path) -> None:
    """Fail a sharpener whose 2-8 px edge gain fell under 1."""
    write_run(places, "old", "sharpen", [sharpen_row("a", 0.7, 1.02)])
    write_run(places, "flat", "sharpen", [sharpen_row("a", 0.7, 0.99)])
    assert run_cli("check", "old", "flat") == 1


def test_check_fails_an_output_that_is_lost(places: Path) -> None:
    """Fail a run that no longer writes an output the reference wrote."""
    write_run(places, "old", "sharpen", [sharpen_row("a", 0.7), sharpen_row("b", 0.8)])
    lost = [sharpen_row("a", 0.7), sharpen_row("b", 0.8, ok=False)]
    write_run(places, "lost", "sharpen", lost)
    assert run_cli("check", "old", "lost") == 1


def test_summary_flags_a_precision_blow_up(
    places: Path, capsys: pytest.CaptureFixture
) -> None:
    """Mark a photograph whose SD output departs from HD."""
    row = {"id": "o_ddae6d29", "model": "petersburg", "family": "sharpen", "ok": True}
    row |= {"max_abs": 255.0, "psnr": 21.7, "ssim": 0.88}
    write_run(places, "blown", "precision", [row])
    assert run_cli("summary", "blown") == 0
    assert "! psnr < 40, max_abs > 16" in capsys.readouterr().out


def test_a_run_without_run_json_is_a_lens_blur_run(places: Path) -> None:
    """Read runs made before run.json existed as lens blur runs."""
    folder = places / "runs/legacy"
    folder.mkdir(parents=True)
    row = {"id": "o_1", "matted": True, "sep_mse": 0.2, "sep_sad": 1.0}
    (folder / "aim.jsonl").write_text(json.dumps(row) + "\n")
    assert report.kind_of("legacy") == "lensblur"
    assert run_cli("check", "legacy", "legacy") == 0
