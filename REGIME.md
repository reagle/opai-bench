# Benchmark regime

How Open Photo AI's lens blur is checked against opai-bench, so a change is judged on many photographs and not on the three it was tuned on (bonsai, Tokyo, Nora).

## When

| Moment | Tier | Time | Command |
|---|---|---|---|
| Before committing a lens-blur change (depth, trimap, seeds, join, fill, render) | core: 20 AIM-500 photographs with busy, sharp backgrounds (curated by hand, listed in `manifest.toml`) and 5 Spring scenes chosen by `select_core.py` | ~8 min | `./bench.py run <change> --opai <dev build>` then `./bench.py check base-<installed> <change>` |
| A change that could matter widely (a model, the trimap's band) | quick: every 5th AIM-500 photograph (94), 10 scenes | ~35 min | `--tier quick` |
| Before merging into `integration`, and after a model swap | full: 467 photographs, 37 scenes | ~3 h | the same with `--tier full` |
| After installing a new integration build | full | ~1 h | `./bench.py run base-<short commit> --tier full`: the new reference |

Tags: `base-<short commit>` for an installed build, `<bookmark>-<n>` for a change under test. Runs live in `runs/` (on the bench disk, `/Volumes/opai-bench`, not in the repository).

## What counts as a regression

`check` fails where the new build's means are worse than the reference's by more than:

| Dataset | Metric | Worse is | Limit |
|---|---|---|---|
| AIM-500 | `sep_mse` (squared error along the outline where a sharp subject meets a blurred background) | up | 5% of the reference |
| AIM-500 | `sep_sad` | up | 5% of the reference |
| Spring | `boundary_f1` (occlusion boundaries of the depth) | down | 0.01 |
| Spring | `delta1` (depth ordering) | down | 0.01 |

and wherever an item the reference scored is no longer scored (no matte, or no depth). Runs go through opai's run cache, which keeps each photograph's depth and mattes by their inputs: a repeat of a build scores identically, and a new build reuses them unless the change touches what they are made from. So there is no run-to-run noise to allow for; the limits only keep small trades from failing. They are provisional.

`check` passing is necessary, not sufficient: the strips of the photographs the change was made for are still looked at. A failing `check` is not a veto either: `compare` names the items that moved most, and a fix may trade a small loss on some for a large gain on others; that trade is then said in the commit.

## Known limits of the reference

- Photographs are rendered at f/1.4: AIM-500's are small and have no EXIF (lens assumed f/8 equivalent), and 16 of the quick tier's 94 still blur under the 3 px a matte needs. They count as not matted, by design (a blur that small is not worth a matte); `check` only fails on items the reference matted.
- The ground truth outlines the whole object; a lens keeps sharp what is at the focus distance. So the `sep_` errors count only the stretches of the outline where the subject is sharp (radius under 1 px) and the background beside it blurs (2 px or more), by the radius the build itself renders.
- Spring 0041: the depth model squashes the whole outdoor frame into 0.5--1 m, so no depth edge is found (`edges_found` 0, boundary F1 0). A real failure of the model, kept in the set.

## What is not measured

- Whole-frame errors (`sad`, `mse`, `grad`): mostly the depth of field keeping things at the subject's distance sharp (a floor at a sofa seat's distance), which an object's matte counts as error. Reported, not judged.
- `specks`: noisy (hair, fur). Compared between runs only.
- Fences, a nearer person: no open ground truth. These remain the gate photographs' job (bonsai, Tokyo, Nora).
- The final blur against real optics.

## Who runs it

Launching and watching a run is mechanical: an agent on a small model does it and reports `check`'s output. Choosing what to change, and reading a failure, is not.
