"""Fetch the datasets, after the user agrees to their terms, and list their items.

`fetch` shows a dataset's terms and asks for `yes` (or takes `--accept`), records the
agreement in `<dataset>/ACCEPTED`, then downloads: AIM-500 from its Hugging Face mirror,
Spring one frame a scene out of DaRUS's remote zips.
"""

__author__ = "Joseph Reagle"
__copyright__ = "Copyright (C) 2026 Joseph Reagle"
__license__ = "MIT"

import csv
import getpass
import io
import logging
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from opai_bench.config import MANIFEST, dataset_dir

MATTE_SETS = [name for name, spec in MANIFEST.items() if spec.get("kind") == "matte"]
DATASETS = [*MATTE_SETS, "spring"]
FETCHABLE = ["aim", "spring"]


# --- Consent ---------------------------------------------------------------------


def terms(name: str) -> str:
    """Return what a dataset's user agrees to, as `fetch` shows it."""
    spec = MANIFEST[name]
    lines = [spec["name"], f"  source: {spec['source']}", f"  size:   {spec['size']}"]
    return "\n".join([*lines, "", *(f"  - {term}" for term in spec["terms"])])


def accepted(name: str) -> str | None:
    """Return the record of the user's agreement to a dataset's terms, if any."""
    record = dataset_dir(name) / "ACCEPTED"
    return record.read_text() if record.exists() else None


def record_consent(name: str, how: str, now: datetime | None = None) -> Path:
    """Write `<dataset>/ACCEPTED`: when, by whom and how, and the terms themselves.

    >>> import os, tempfile
    >>> os.environ["OPAI_BENCH_DATA"] = tempfile.mkdtemp()
    >>> record = record_consent("spring", "typed yes", datetime(2026, 10, 30, 12, 0))
    >>> record.read_text().startswith("Agreed 2026-10-30T12:00:00 by ")
    True
    >>> "(typed yes)" in accepted("spring")
    True
    >>> del os.environ["OPAI_BENCH_DATA"]
    """
    folder = dataset_dir(name)
    folder.mkdir(parents=True, exist_ok=True)
    stamp = (now or datetime.now()).isoformat(timespec="seconds")
    record = folder / "ACCEPTED"
    user = getpass.getuser()
    record.write_text(f"Agreed {stamp} by {user} ({how}) to:\n\n{terms(name)}\n")
    return record


def consent(
    name: str, accept: bool, dry_run: bool = False, ask: Callable[[str], str] = input
) -> bool:
    """Show a dataset's terms and get the user's agreement.

    An earlier `ACCEPTED`, `--accept`, or `yes` typed (or piped) at the prompt agrees.
    A dry run asks but records nothing.

    >>> import os, tempfile
    >>> os.environ["OPAI_BENCH_DATA"] = tempfile.mkdtemp()
    >>> consent("spring", False, ask=lambda _: "no")  # doctest: +ELLIPSIS
    Spring ...
    Not agreed; nothing downloaded.
    False
    >>> consent("spring", True, dry_run=True)  # doctest: +ELLIPSIS
    Spring ...
    (dry run: would record the agreement in .../spring/ACCEPTED)
    True
    >>> accepted("spring") is None
    True
    >>> consent("spring", False, ask=lambda _: "yes")  # doctest: +ELLIPSIS
    Spring ...
    Agreement recorded in .../spring/ACCEPTED
    True
    >>> consent("spring", False, ask=lambda _: "never asked")  # doctest: +ELLIPSIS
    Spring ...
    Already agreed: Agreed ... (typed yes) to:
    True
    >>> del os.environ["OPAI_BENCH_DATA"]
    """
    print(terms(name), "\n")
    if earlier := accepted(name):
        print(f"Already agreed: {earlier.splitlines()[0]}")
        return True
    how = "--accept"
    if not accept:
        try:
            answer = ask("Type yes to agree to these terms: ")
        except EOFError:
            print("\nNo answer to read: read the terms, then run again with --accept.")
            return False
        if answer.strip().lower() != "yes":
            print("Not agreed; nothing downloaded.")
            return False
        how = "typed yes"
    if dry_run:
        where = dataset_dir(name) / "ACCEPTED"
        print(f"(dry run: would record the agreement in {where})")
    else:
        print(f"Agreement recorded in {record_consent(name, how)}")
    return True


# --- Fetching --------------------------------------------------------------------


class RemoteFile(io.RawIOBase):
    """Read a remote file over HTTP range requests, seekably.

    So single members can be read out of a large remote zip.
    """

    def __init__(self, url: str) -> None:
        self.url, self.pos = url, 0
        request = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(request) as response:
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
        request = urllib.request.Request(self.url, headers=headers)
        with urllib.request.urlopen(request) as response:
            data = response.read()
        buffer[: len(data)] = data
        self.pos += len(data)
        return len(data)


def darus_url(file_id: int) -> str:
    """Return the signed storage URL DaRUS redirects a file id to."""

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs) -> None:
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
    """Open a DaRUS file as a zip, reading only the parts asked for."""
    return zipfile.ZipFile(io.BufferedReader(RemoteFile(darus_url(file_id)), 1 << 20))


def fetch_spring(dry_run: bool = False) -> None:
    """Fetch the middle frame of each Spring training scene, and its disparity."""
    spec = MANIFEST["spring"]
    out = dataset_dir("spring")
    if dry_run:
        ids = f"{spec['frames_file']} and {spec['disparity_file']}"
        print(f"(dry run: would read DaRUS files {ids} into {out})")
        return
    frames = remote_zip(spec["frames_file"])
    disparities = remote_zip(spec["disparity_file"])
    scenes: dict[str, list[int]] = {}
    for name in frames.namelist():
        parts = name.split("/")
        if name.endswith(".png"):
            number = int(parts[-1].rsplit("_", 1)[1].split(".")[0])
            scenes.setdefault(parts[2], []).append(number)
    for scene, numbers in sorted(scenes.items()):
        frame = sorted(numbers)[len(numbers) // 2]
        target = out / scene
        if (target / "disparity.dsp5").exists():
            continue
        target.mkdir(parents=True, exist_ok=True)
        member = spec["member_frame"].format(scene=scene, frame=frame)
        (target / "frame.png").write_bytes(frames.read(member))
        member = spec["member_disparity"].format(scene=scene, frame=frame)
        (target / "disparity.dsp5").write_bytes(disparities.read(member))
        source = f"Spring train scene {scene}, frame {frame} (CC BY 4.0)\n"
        (target / "source.txt").write_text(source)
        print(f"spring {scene} frame {frame}", flush=True)
    print(f"spring: {len(scenes)} scenes in {out}")


def fetch_aim(dry_run: bool = False) -> None:
    """Fetch AIM-500 from its Hugging Face mirror, as the authors' folders + labels."""
    import pyarrow.parquet as pq  # only fetching needs it

    spec = MANIFEST["aim"]
    out = dataset_dir("aim")
    if (out / "labels.csv").exists() and len(photos("aim")) >= 500:
        print(f"aim: already in {out}")
        return
    if dry_run:
        print(f"(dry run: would download {spec['mirror']} and unpack it into {out})")
        return
    parquet = out / "test.parquet"
    if not parquet.exists():
        out.mkdir(parents=True, exist_ok=True)
        print(f"aim: downloading {spec['mirror']}", flush=True)
        urllib.request.urlretrieve(spec["mirror"], parquet)
    table = pq.read_table(parquet)
    (out / spec["images"]).mkdir(exist_ok=True)
    (out / spec["mattes"]).mkdir(exist_ok=True)
    with (out / "labels.csv").open("w", newline="") as labels_file:
        writer = csv.writer(labels_file)
        writer.writerow(["id", "category", "type"])
        for row in table.to_pylist():
            image, mask = row["image"], row["mask"]
            (out / spec["images"] / image["path"]).write_bytes(image["bytes"])
            (out / spec["mattes"] / mask["path"]).write_bytes(mask["bytes"])
            writer.writerow([row["image_name"], row["category"], row["type"]])
    parquet.unlink()
    print(f"aim: {table.num_rows} photos in {out}", flush=True)


def fetch(name: str, accept: bool, dry_run: bool) -> int:
    """Show the terms, get the agreement, download; return 1 where not agreed."""
    if not consent(name, accept, dry_run):
        return 1
    (fetch_aim if name == "aim" else fetch_spring)(dry_run)
    return 0


# --- Items -----------------------------------------------------------------------


def labels(name: str) -> dict[str, dict[str, str]]:
    """Return a dataset's per-item labels (its `labels.csv`) by id; empty if none."""
    path = dataset_dir(name) / "labels.csv"
    if not path.exists():
        return {}
    with path.open() as rows:
        return {row["id"]: row for row in csv.DictReader(rows)}


def photos(name: str) -> list[Path]:
    """Return every photograph of a matte dataset, in name order."""
    folder = dataset_dir(name) / MANIFEST[name]["images"]
    if not folder.is_dir():
        logging.error("%s: not in %s (run `opai-bench fetch %s`)", name, folder, name)
        return []
    return sorted(
        image
        for image in folder.iterdir()
        if image.is_file() and not image.name.startswith(".")
    )


def matte_items(name: str, tier: str) -> list[tuple[Path, Path]]:
    """Return a matte dataset's (photo, ground-truth matte) pairs for `tier`.

    `quick` takes every `quick_every`th in name order, so each kind the names start
    with is in it; non-salient photographs (no single subject) are skipped.
    """
    spec = MANIFEST[name]
    kinds = labels(name)
    skip = set(spec.get("skip_types", []))
    mattes = dataset_dir(name) / spec["mattes"]
    pairs = [
        (image, mattes / f"{image.stem}{spec['matte_suffix']}")
        for image in photos(name)
        if kinds.get(image.stem, {}).get("type") not in skip
    ]
    if tier == "core":
        return [pair for pair in pairs if pair[0].stem in spec["core"]]
    return pairs[:: spec["quick_every"]] if tier == "quick" else pairs


def spring_items(tier: str) -> list[Path]:
    """Return the Spring scenes of `tier`: core by name, quick the first N, or all."""
    spec = MANIFEST["spring"]
    root = dataset_dir("spring")
    found = root.glob("*") if root.is_dir() else []
    scenes = sorted(path for path in found if (path / "disparity.dsp5").exists())
    if not scenes:
        logging.error("spring: nothing in %s (run `opai-bench fetch spring`)", root)
    if tier == "core":
        return [scene for scene in scenes if scene.name in spec["core"]]
    return scenes[: spec["quick"]] if tier == "quick" else scenes


def suite_photos(tier: str, core: list[str], every: int) -> list[Path]:
    """Return AIM-500 photographs for an enhancement suite.

    `core` by id, `quick` the core plus every `every`th photograph, `full` all of them
    (non-salient ones too: an enhancement needs no subject).
    """
    every_photo = photos("aim")
    if tier == "full":
        return every_photo
    chosen = [image for image in every_photo if image.stem in core]
    if tier == "quick":
        chosen += [image for image in every_photo[::every] if image not in chosen]
    return chosen
