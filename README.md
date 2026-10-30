# opai-bench

Scores [Open Photo AI](https://github.com/vegidio/open-photo-ai)'s lens blur against datasets with ground truth, so a change to the depth, the matte or a model is judged on many photographs and not on the few it was tuned on. Scripts and scores only: no dataset file is in this repository.

| Dataset | What it scores | Licence | How to get it |
|---|---|---|---|
| [Spring](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/DARUS-3376) (Mehl et al., CVPR 2023) | the depth map: ordering (delta-1, AbsRel of inverse depth after a scale-and-shift fit) and occlusion boundaries (Depth Pro-style boundary F1) against stereo disparity | CC BY 4.0 | `./bench.py fetch spring` reads one frame a scene (37 frames, ~300 MB) out of the remote zips |
| [AIM-500](https://github.com/JizhiziLi/AIM) (Li, Zhang and Tao, IJCAI 2021) | the subject's matte against hand-made mattes of 467 photographs (animals, portraits, plants, furniture, toys, see-through things; the 33 with no single subject are skipped): SAD, MSE, gradient error, MSE in the edge band, specks, and the `sep_` errors along the edge of what is sharp | MIT ([release agreement](https://jizhizili.github.io/files/aim_dataset_agreement/AIM-500_Dataset_Release_Agreement.pdf): read and agree to it first, cite the paper); the photos' copyright stays with their owners | `./bench.py fetch aim` (from the Hugging Face mirror `nobg/AIM-500`, which adds each photo's category and type); the official download is linked from the [AIM repository](https://github.com/JizhiziLi/AIM), should the mirror go |

Each dataset is used under its own terms, downloaded from its source by whoever runs the benchmark; nothing here redistributes it. Scores are facts, not derivatives of the images.

## Use

```sh
./bench.py fetch spring
./bench.py fetch aim
./bench.py run before --opai /path/to/old/opai                 # core: 20 photographs, 5 Spring scenes
./bench.py run before --opai /path/to/old/opai --tier quick    # every 5th photograph (94), 10 Spring scenes
./bench.py run after  --opai /path/to/new/opai --tier quick
./bench.py compare before after
./bench.py check before after      # exit 1 past the limits in REGIME.md
```

When to run which tier, and what counts as a regression: [REGIME.md](REGIME.md).

`--tier full` runs all 467 photographs and 37 scenes; `--ids a,b` runs just those items. Scores go to `runs/<tag>/` (one JSON line an item) with the build's version in `build.txt`.

The opai build needs the fork's `OPAI_BENCH_DUMP` hook (bookmark `add-lens-blur-8-matting` and later in `reagle/open-photo-ai`): with `OPAI_BENCH_DUMP=<prefix>` the CLI writes the depth map to `<prefix>-depth.f32`, the blur's radius at each of its pixels to `<prefix>-radius.f32` and the subject's matte to `<prefix>-matte.f32`. Photographs are rendered at `--lb-f 1.4` focused on the point of the ground-truth subject farthest from its edge, and the `sep_` errors are taken only where the outline separates a sharp subject (radius under 1 px) from a blurred background (2 px or more): a ground truth's object outline is not the edge of what a lens keeps sharp elsewhere (a floor at the seat's distance, a sofa's nearer arm); Spring frames with `--lb-edges depth`, so only the depth model runs. Runs go through opai's run cache, which grows by the results (a few GB for a full run); a repeat with the same build is fast.

## Limits

- AIM-500's photographs are small (1,080--2,066 px) and carry no EXIF, so blurs are small and a sixth are not matted at all; there is no ground truth for a fence or a nearer person.
- Spring is rendered (the Blender open film *Spring*): it scores depth edges exactly, but on renders, not photographs.
- Nothing here scores the final blur against real optics.

## Licence

The scripts are MIT-licensed (see `LICENSE`). The datasets are not part of this repository and keep their own licences (above).

> [!NOTE]
> AI disclosure: written by Claude Code (Claude Opus 5.5, low effort) for Joseph Reagle.
