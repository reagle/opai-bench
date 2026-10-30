# opai-bench

Benchmarks for [Open Photo AI](https://github.com/vegidio/open-photo-ai), run through its `opai` CLI, so a change to a model, a precision or the code around them is judged on many photographs and in numbers, not on the few it was tuned on. Code and scores only: **no dataset file is in this repository; they are fetched with the `fetch` subcommand** (`opai-bench fetch aim`, `opai-bench fetch spring`), which shows each dataset's terms and asks you to agree first (see [Datasets and their terms](#datasets-and-their-terms)).

## What it measures

| Command | What | Against |
|---|---|---|
| `run TAG` (suite `lensblur`) | the lens blur's **depth map** (ordering: delta-1 and AbsRel of inverse depth after a scale-and-shift fit; occlusion boundaries: Depth Pro-style boundary F1) and its **subject matte** (SAD, MSE, gradient error, MSE in the edge band, specks, and the `sep_` errors along the edge of what is sharp) | ground truth: Spring's stereo disparity, AIM-500's hand-made mattes |
| `run TAG --suite sharpen` | the **sharpen/denoise quality tier** of every sharpen model (Moscow, Petersburg, Novgorod) and Stockholm denoise at their defaults: tile seams, flat-area mottle, change per wavelength band, edge gain, invention ratio, time | each model's own input |
| `precision MODEL\|FAMILY` | **FP16 (SD) against FP32 (HD)** for any enhancement: largest and mean absolute difference, PSNR and SSIM, overall and split into flat and detailed areas; each precision's time; for sharpen and denoise, the headline quality measures at each precision | the HD output |
| `models FAMILY A B ...` | **models against each other**, or **builds against each other** (`--opai` twice): the same pixel scores, the time, and for sharpen and denoise the headline quality measures | the input, or a reference model (`--ref`) |

Every run records the build's `--version`, so two builds (installed and dev, say) are compared by running each under its own tag and then `compare` or `check`; `models` can also run several builds side by side in one run.

### The quality tier, in brief

Measured on the output's luma against the input's (`src/opai_bench/metrics/quality.py` says exactly how; ported from the scripts that found Petersburg's mottle):

- `seam`, `seam_null`, `seam_excess`: the step in the residual (output minus input, over flat pixels) across each internal seam of the model's tile grid, against the same step half a tile away where no seam is. The grid is the app's 256-px tiles with 16-px overlap, a seam every 240 px (`--tile`/`--overlap` for a build that differs). `tile_sd`: the spread of the tiles' mean residual.
- `mottle`: the residual low-passed at sigma 3 px, 1st to 99th percentile spread over flat pixels, in grey levels. About one level is the edge of visibility on a smooth gradient.
- `band_rms_*`, `band_gain_*`, `band_corr_*`: per wavelength band (2--4, 4--8, 8--16, 16--32, 32--64 px and over 64; 2 px is the Nyquist limit) over flat pixels, the residual's RMS, the output's RMS over the input's, and the residual's correlation with the input's own band (amplified detail correlates; invented detail does not). `edge_gain_*`: the gain over strong edges; `edge_gain_2_8` is the sharpening itself.
- `inv_max`, `inv_p90`, `inv_median`: per tile, the RMS of the change finer than a radius-16 box applied twice, over the input's own detail at that scale plus one grey level --- the app's own `filter::invention`, here on the final photograph rather than the raw model output.

Flat areas are where the input's smoothed gradient (Sobel at sigma 2) stays under 8 levels over a 15-px neighbourhood; detailed is the rest.

## Datasets and their terms

**The datasets are not in this repository.** Each one is downloaded separately, from its own source, by whoever runs the benchmark, with the `fetch` command: `opai-bench fetch aim` and `opai-bench fetch spring`. Run them once, before the suites that need them (the synthetic gradient needs no download). `fetch` first shows the dataset's terms and asks you to agree; the files go to `data/` (or `--data`), which git ignores.

| Dataset | Used for | Licence and terms | Get it |
|---|---|---|---|
| [AIM-500](https://github.com/JizhiziLi/AIM) (Li, Zhang and Tao, IJCAI 2021) | the lens blur's matte (467 photographs with a single subject); the photographs of the sharpen suite, `precision` and `models` | MIT under the [AIM-500 release agreement](https://jizhizili.github.io/files/aim_dataset_agreement/AIM-500_Dataset_Release_Agreement.pdf): read and agree to it first, cite the paper; the photographs' copyright stays with their owners; do not redistribute | `fetch aim`, from the Hugging Face mirror `nobg/AIM-500` (~470 MB; adds each photo's category and type); the official download is linked from the [AIM repository](https://github.com/JizhiziLi/AIM), should the mirror go |
| [Spring](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/DARUS-3376) (Mehl et al., CVPR 2023) | the lens blur's depth map | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/): attribute it | `fetch spring` reads one frame a scene (37 frames, ~300 MB) out of the remote zips on DaRUS |
| synthetic gradient | the sharpen suite, `precision`, `models` | part of this repository (MIT) | made on first use: a smooth 3000x2000 colour gradient with fine grain, where invented texture and seams show most; nobody's photograph needed |

`fetch` shows the dataset's terms and asks you to type `yes`; `--accept` agrees without asking (for scripts), and either is recorded with the date in `data/<dataset>/ACCEPTED`. `--dry-run` shows the terms and what would be downloaded, and changes nothing. Each dataset is used under its own terms, downloaded from its source by whoever runs the benchmark; nothing here redistributes it. Scores are facts, not derivatives of the images.

## Install

Needs [uv](https://docs.astral.sh/uv/), Python 3.13 or later, and an `opai` CLI (`--opai PATH`; without it, `~/.local/bin/opai` or the `opai` on PATH, and the run says so). The clone pins Python 3.14 (`.python-version`), where everything runs. On Python 3.15, h5py (Spring's disparity reader) has no wheels yet, so it is left out there: everything else runs, and the lens blur suite needs `--only aim` (or `uv run --python 3.14 ...`). From a clone:

```sh
uv run opai-bench --help
```

(`./bench.py ...` still works in a clone, as before: it runs `uv run opai-bench ...` with the clone's own `data`, `runs` and `opai-home`), or installed as a tool, run from whatever folder should hold `data/` and `runs/`:

```sh
uv tool install git+https://github.com/reagle/opai-bench
opai-bench --help
```

The lens blur suite needs a build with the fork's `OPAI_BENCH_DUMP` hook (bookmark `add-lens-blur-8-matting` and later in `reagle/open-photo-ai`): with `OPAI_BENCH_DUMP=<prefix>` the CLI writes the depth map to `<prefix>-depth.f32`, the blur's radius at each of its pixels to `<prefix>-radius.f32` and the subject's matte to `<prefix>-matte.f32`. The other commands need only a stock CLI with `-q`.

## Quick start

```sh
# 1. Fetch the datasets (once; they are downloaded separately, not part of the repo)
uv run opai-bench fetch aim                  # shows the terms, asks for "yes"
uv run opai-bench fetch spring --accept      # agrees without asking (recorded)

# Lens blur: core tier (20 photographs, 5 Spring scenes), then against a reference
uv run opai-bench run before --opai /path/to/old/opai
uv run opai-bench run after  --opai /path/to/new/opai
uv run opai-bench check before after          # exit 1 past the limits

# Sharpen/denoise suite: 4 models x (6 photographs + the synthetic gradient)
uv run opai-bench run sharpen-base --suite sharpen --opai ~/.local/bin/opai
uv run opai-bench run sharpen-dev  --suite sharpen --opai /path/to/dev/opai
uv run opai-bench check sharpen-base sharpen-dev

# FP16 against FP32: one model, or every model of a family
uv run opai-bench precision petersburg --ids o_f12a8afc,o_093764ca
uv run opai-bench precision upscale --scale 2 --limit 3

# Models against each other, against the input or one of them
uv run opai-bench models sharpen moscow petersburg novgorod
uv run opai-bench models denoise stockholm gothenburg --ref stockholm

# Two builds, one model, pixel for pixel
uv run opai-bench models sharpen petersburg --opai installed=~/.local/bin/opai --opai dev=/path/to/dev/opai \
    --ref petersburg@installed

uv run opai-bench summary sharpen-dev         # the concise table again
uv run opai-bench summary sharpen-dev --all   # every score of every item
uv run opai-bench compare sharpen-base sharpen-dev
```

Families: `sharpen` (moscow, petersburg, novgorod), `denoise` (stockholm, gothenburg, malmo), `upscale` (tokyo, kyoto, saitama; compared at the output size, the input resized with Lanczos, at `--scale`, default 2), `light` (paris, lyon), `colour` (rio, saopaulo), `lensblur` (lisbon; at f/1.4, focused on the AIM-500 subject's middle).

Photographs for the sharpen suite, `precision` and `models`: `--tier core` (the six of `[sharpen] core` in the manifest: large flat skies and backdrops where mottle and seams show, and detailed ones where the edge gain shows), `quick` (those plus every 10th AIM-500 photograph), `full` (all 500); `--ids` for chosen ones, `--photos FILE...` for your own, `--limit N` for the first N, `--no-synthetic` to leave out the gradient. `--args "..."` passes more CLI arguments to every run (`--args "--sh-str 150"`); `-q hd|sd` fixes the quality of the sharpen suite and `models` (default: the CLI's, HD).

## Where things live

- `data/` (`--data`, `OPAI_BENCH_DATA`): the datasets and the synthetic image. `OPAI_BENCH_AIM` and `OPAI_BENCH_SPRING` still point at one dataset each.
- `runs/` (`--runs`, `OPAI_BENCH_RUNS`): one folder a tag, with `build.txt` (the build's version and path), `run.json` (what was run; runs made before it existed are lens blur runs) and one JSON line an item (`aim.jsonl`, `spring.jsonl`, `sharpen.jsonl`, `precision.jsonl`, `models.jsonl`). `--keep` keeps the output images in `runs/<tag>/out/`.
- `opai-home/` (`--home`, `OPAI_BENCH_HOME`): the HOME the sharpen suite, `precision` and `models` run opai under. Your installed runtime is linked into it and your weights are linked one by one, so a model the CLI downloads (one not installed at some precision) lands in `opai-home`, never among yours; its run cache is cleared before every timed run, so each run is computed and timed afresh, and the GUI's saved settings are not there to change what the CLI does. Compiled engines are kept. Each model gets one untimed warm-up run first (`--no-warmup` to skip). The lens blur suite runs under your own HOME and run cache, as it always has.
- `src/opai_bench/manifest.toml`: the datasets, their terms and the tiers; package data, so an installed `opai-bench` reads the same one.

`data`, `runs` and `opai-home` are relative to the current folder; in a clone they are often symlinks to a larger disk.

## Reading results and `check`

While a run works, a progress bar on stderr names the photograph and model in hand, with the time spent and left; `-V` (repeatable) adds log lines, `-L` writes them to `opai-bench.log`. Each run starts by naming the opai build (or builds) it uses, and ends with a concise table: one row per photograph and model with a few key columns, each model's means, a one-line reading guide, and `!` against any row past a threshold (for `precision`, PSNR under 40 dB, `max|d|` over 16 levels, an SD invention ratio over 5, or a sharpener's SD edge gain under 1; for the sharpen suite and `models`, mottle over 2 levels, seam excess over 0.1, invention over 5, or a sharpener's edge gain under 1). Every score is in `runs/<tag>/<kind>.jsonl`, and `summary TAG --all` prints them all.

`summary` prints each photograph's row and each model's means; `compare OLD NEW` prints the means side by side and the five items that moved most (by `sep_mse` or `boundary_f1` for the lens blur, `mottle` for the sharpen suite, `psnr` for `precision` and `models`); `check OLD NEW` does the same and exits 1 where NEW's mean is worse than OLD's past a limit, or where an item OLD scored has no score (or output) any more. Times are reported, and a mean more than 25% slower is printed as `SLOWER`, but never fails a check. The limits, what they guard and when to run which tier: [REGIME.md](REGIME.md). They are provisional.

For `precision`, PSNR above about 50 dB and SSIM above 0.999 mean FP16 is indistinguishable at 8 bits; a low `psnr_flat` with a high `psnr_detail` (or the reverse) says where FP16 departs. A `max_abs` far above `mean_abs` is a local blow-up: look at that photograph with `--keep`.

## Limits

- AIM-500's photographs are small (1,080--2,066 px) and carry no EXIF, so blurs are small and a sixth are not matted at all; there is no ground truth for a fence or a nearer person. Spring is rendered, so it scores depth edges exactly, but on renders. Nothing here scores the final blur against real optics.
- The sharpen tier and `precision` measure what changed, not whether a person prefers it: there is no ground-truth sharp photograph. Mottle, seams and invention are failure measures; edge gain says that a sharpener still sharpens, not that it sharpens well.
- The invention ratio is the app's measure on the final photograph: after the anchor and the blend, so smaller than what the app's guard sees on the raw model output.
- Times are wall-clock per CLI call (start-up, model load and inference) on whatever machine runs it, under whatever else it is doing; compare times only within one machine and one sitting.
- Each photograph is one CLI call, so batch behaviour (memory across a batch, shared sessions) is not measured.

## Developing

```sh
uv run ruff check && uv run pyright && uv run pytest     # no opai or dataset needed
```

Doctests in the modules are the main tests; `tests/` holds the command-line ones that need made-up runs. The code follows the ruff, pyright and pytest settings in `pyproject.toml`: comments and docstrings wrapped at 88 columns, docstrings with an imperative one-line summary (pydocstyle), complexity at most 10.

## Licence

The code is MIT-licensed (see `LICENSE`). The datasets are not part of this repository and keep their own licences (above).

> [!NOTE]
> AI disclosure: written by Claude Code (Claude Opus 5.5, low effort) for Joseph Reagle.
