# Benchmark regime

How Open Photo AI's lens blur and its sharpen and denoise models are checked against opai-bench, so a change is judged on many photographs and not on the few it was tuned on (bonsai, Tokyo, Nora for the lens blur; the synthetic gradient and three AIM-500 photographs for sharpening). Commands are run from the clone (`uv run opai-bench ...`, on the pinned Python 3.14, or `./bench.py ...`) or with an installed `opai-bench`.

## When

| Moment | Suite and tier | Time | Command |
|---|---|---|---|
| Before committing a lens-blur change (depth, trimap, seeds, join, fill, render) | lens blur core: 20 AIM-500 photographs with busy, sharp backgrounds (curated by hand, listed in `manifest.toml`) and 5 Spring scenes chosen by `select-core` | ~8 min | `opai-bench run <change> --opai <dev build>` then `opai-bench check base-<installed> <change>` |
| A lens-blur change that could matter widely (a model, the trimap's band) | lens blur quick: every 5th AIM-500 photograph (94), 10 scenes | ~35 min | `--tier quick` |
| Before merging a lens-blur change into `integration`, and after a depth or matting model swap | lens blur full: 467 photographs, 37 scenes | ~3 h | the same with `--tier full` |
| Before committing a change to sharpen or denoise (a model, its guard or anchor, the tiling, the blend) | sharpen core: Moscow, Petersburg, Novgorod, Stockholm on 6 AIM-500 photographs (flat skies and backdrops, and detailed ones) and the synthetic gradient | ~5 min | `opai-bench run <change> --suite sharpen --opai <dev build>` then `opai-bench check sharpen-base-<installed> <change>` |
| Before merging a sharpen or denoise change, or a model swap | sharpen quick: the core plus every 10th AIM-500 photograph (~55) | ~25 min | `--suite sharpen --tier quick` |
| A change of precision or provider (an FP16 export, a CoreML setting) | `precision` of the family on the sharpen core | ~5 min a model | `opai-bench precision <family> --opai <dev build>`, then `check` against the installed build's |
| After installing a new integration build | lens blur full, sharpen quick | ~1.5 h | `opai-bench run base-<short commit> --tier full`, `opai-bench run sharpen-base-<short commit> --suite sharpen --tier quick`: the new references |

Tags: `base-<short commit>` and `sharpen-base-<short commit>` for an installed build, `<bookmark>-<n>` for a change under test; `precision` and `models` name their own runs (`precision-<model>-<version>`) unless given `--tag`. Runs live in `runs/` (on the bench disk, `/Volumes/opai-bench`, not in the repository). Only one opai runs at a time: the suites run each photograph in turn.

## How much to run

Outputs are deterministic, so more photographs buy coverage of kinds of photograph, not statistical confidence. Escalate only as far as the decision needs:

1. **The problem photographs** behind the change (3--5): does it fix what it was made for?
2. **core** (20) for any change to a pipeline (depth, trimap, matte, render, a filter).
3. **quick** (94) only for a model swap, or when core is close or mixed (a gain of a few per cent, or items moving both ways).
4. **full** (467) only before merging something that could matter widely.

Reuse an existing reference run of the same build rather than re-running it. Clear the run cache only when time or first-run behaviour is what is measured; otherwise the cache makes repeats free and is keyed by inputs, so it cannot serve a stale result. Run one heavy job at a time, so times and memory are not bent by a second one. When reviewing in the GUI, open only the photographs the change concerns.

## What counts as a regression

`check` fails where the new build's means are worse than the reference's by more than the limit, model by model for the enhancement suites. A relative limit never allows less than its floor, so a measure near zero does not fail on noise.

| Suite | Metric | Worse is | Limit |
|---|---|---|---|
| lens blur, AIM-500 | `sep_mse` (squared error along the outline where a sharp subject meets a blurred background) | up | 5% of the reference |
| lens blur, AIM-500 | `sep_sad` | up | 5% of the reference |
| lens blur, Spring | `boundary_f1` (occlusion boundaries of the depth) | down | 0.01 |
| lens blur, Spring | `delta1` (depth ordering) | down | 0.01 |
| sharpen | `seam_excess` (step at the tile seams over the step mid-tile, grey levels) | up | 0.1 level |
| sharpen | `mottle` (low-passed residual, 1st--99th percentile spread over flat areas, levels) | up | 20%, at least 0.1 level |
| sharpen | `inv_max` (largest invention ratio of any tile) | up | 20%, at least 0.2 |
| sharpen | `edge_gain_2_8` (2--8 px detail at strong edges, output over input) | down | 0.03; and under 1.0 for a sharpen model fails outright (it no longer sharpens) |
| precision | `psnr`, `psnr_detail` (SD against HD) | down | 1 dB |
| precision | `ssim` | down | 0.002 |
| precision | `mottle_sd`, `seam_excess_sd`, `inv_max_sd`, `edge_gain_2_8_sd` (sharpen and denoise) | as for the sharpen suite | as for the sharpen suite |
| models | `psnr`, `ssim` against the input or reference | either way | 1 dB, 0.002: the output is not the same any more |
| models | the sharpen suite's four, for sharpen and denoise | as for the sharpen suite | as for the sharpen suite |

and wherever an item the reference scored is no longer scored (no matte, no depth, no output). Times are compared but never fail: a mean more than 25% slower prints `SLOWER`.

The lens blur suite goes through opai's run cache, which keeps each photograph's depth and mattes by their inputs: a repeat of a build scores identically, and a new build reuses them unless the change touches what they are made from. So there is no run-to-run noise to allow for; the limits only keep small trades from failing. The enhancement suites clear their own run cache before every call (they run under an isolated HOME), so their pixel scores are as deterministic as the model on its provider, and their times are fresh. All the limits are provisional: the sharpen ones are set from the 2026-10-30 measurements (Petersburg's mottle 4.5 levels before its anchor was tightened, 0.7 after; seam excess 0.04; the speckled tile's invention ratio 13 against at most 2 elsewhere).

`check` passing is necessary, not sufficient: the strips of the photographs the change was made for are still looked at. A failing `check` is not a veto either: `compare` names the items that moved most, and a fix may trade a small loss on some for a large gain on others; that trade is then said in the commit.

## Known limits of the reference

- Lens blur photographs are rendered at f/1.4: AIM-500's are small and have no EXIF (lens assumed f/8 equivalent), and 16 of the quick tier's 94 still blur under the 3 px a matte needs. They count as not matted, by design (a blur that small is not worth a matte); `check` only fails on items the reference matted.
- The ground truth outlines the whole object; a lens keeps sharp what is at the focus distance. So the `sep_` errors count only the stretches of the outline where the subject is sharp (radius under 1 px) and the background beside it blurs (2 px or more), by the radius the build itself renders.
- Spring 0041: the depth model squashes the whole outdoor frame into 0.5--1 m, so no depth edge is found (`edges_found` 0, boundary F1 0). A real failure of the model, kept in the set.
- The sharpen suite has no ground truth: it measures what each model does beyond its purpose (seams, mottle, invented texture) and that a sharpener still sharpens. The synthetic gradient has no edges, so it carries no edge gain.

## What is not measured

- Whole-frame lens blur errors (`sad`, `mse`, `grad`): mostly the depth of field keeping things at the subject's distance sharp (a floor at a sofa seat's distance), which an object's matte counts as error. Reported, not judged.
- `specks`: noisy (hair, fur). Compared between runs only.
- Fences, a nearer person: no open ground truth. These remain the gate photographs' job (bonsai, Tokyo, Nora).
- The final blur against real optics; whether a sharpened photograph looks better.
- `band_*` and `tile_sd`: reported to explain a failing headline measure, not judged.

## Who runs it

Launching and watching a run is mechanical: an agent on a small model does it and reports `check`'s output. Choosing what to change, and reading a failure, is not.

> [!NOTE]
> AI disclosure: written by Claude Code (Claude Opus 5.5, low effort) for Joseph Reagle.
