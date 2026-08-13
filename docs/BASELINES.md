# Baselines

Every baseline number in the paper was produced by **re-training the official
implementation ourselves**, not by quoting published figures. This document
records the exact upstream commit and our modifications so the comparison can be
reproduced.

The upstream repositories are **not** vendored in this repo (they carry their own
licenses and are large). Clone them at the commits below, apply the patch, and
copy in the driver script.

## Sources

| Baseline | Repository | Commit |
|---|---|---|
| Gaze360 (LSTM) | https://github.com/erkil1452/gaze360 | `cbeb8e4` |
| STAGE | https://github.com/jswati31/stage | `81f9a09` |
| CrossGaze | https://github.com/AndyCatruna/CrossGaze | `73215e3` |
| GazeTR-Hybrid | https://github.com/yihuacheng/GazeTR | `d7cf716` |
| L2CS-Net | https://github.com/Ahmednull/L2CS-Net | `a4d8f7f` |
| MCGaze | https://github.com/zgchen33/MCGaze | `61950c8` |

## Applying our changes

```bash
git clone https://github.com/erkil1452/gaze360 && cd gaze360
git checkout cbeb8e4
git apply /path/to/EyeTAG/baselines/patches/gaze360.patch
cp /path/to/EyeTAG/baselines/scripts/gaze360/* code/
```

`baselines/patches/` contains our diffs against the upstream commit:

| Patch | What it changes |
|---|---|
| `gaze360.patch` | adds a VGGFace2 ResNet-50 face backbone alongside the original ImageNet ResNet-18 (`VGGFace2Backbone`, `GazeLSTM_VGGFace2`, `build_model`) |
| `stage.patch` | Gaze360 datasource, config plumbing, and a VGGFace2 ResNet-50 variant of the STAGE transformer |
| `GazeTR.patch` | Gaze360 train/test config paths |
| `MCGaze.patch` | frame-reorganisation helper for our Gaze360 extraction |

`baselines/scripts/` contains the driver scripts we added
(`run_gaze360.py`, `run_eve.py`, `test_subsets.py`, `eval_with_std.py`,
`inference_gaze360_for_jitter.py`, and the STAGE configs).

## Evaluation protocol

* Same preprocessed Gaze360 test split as EyeTAG (**N = 16,031** frames), and the
  same three yaw subsets, so every row of the main table is directly comparable.
* Reported values are mean ± standard deviation over the test frames.
* Frame-ordered prediction dumps for the temporal-stability table are produced by
  `inference_gaze360_for_jitter.py` (STAGE) or the per-baseline `run_gaze360.py`,
  and are consumed by `analysis/temporal_stability.py` in this repo.

### Notes on individual baselines

**STAGE.** STAGE is designed around per-subject calibration (Gaussian-process
personalisation). Our setting is calibration-free and causal, so STAGE is
evaluated without personalisation, which is why it scores lower on Gaze360 than
its published personalised numbers. Its Front-subset value uses STAGE's own
subset convention (|pitch|, |yaw| ≤ 20°, N = 4,246) rather than the yaw-only
definition used elsewhere in the table.

**MCGaze.** Trained and evaluated with its own official protocol and official test
split (**N = 25,969**), which is a different frame pool from the other rows
(N = 16,031). Our re-training reproduces the published semi-front number
(10.71° vs 10.74° reported). Any table containing MCGaze must carry this footnote.

**CrossGaze** already uses VGGFace2 pretraining upstream
(`InceptionResnetV1(pretrained='vggface2')`), and **STAGE** uses gaze-specific
GazeCLR pretraining — i.e. domain-specific pretraining is not unique to EyeTAG.

### Backbone-controlled comparison

To isolate the encoder from the architecture, we re-trained the two multi-frame
baselines with EyeTAG's exact face encoder (VGGFace2 ResNet-50, same weights),
keeping every other part of their recipe unchanged:

| Method | Face encoder | Mean MAE (deg) |
|---|---|---|
| Gaze360 | ImageNet R-18 (original) | 9.99 |
| Gaze360 | VGGFace2 R-50 | 9.76 |
| STAGE | GazeCLR R-18 (original) | 11.05 |
| STAGE | VGGFace2 R-50 | 10.43 |
| EyeTAG | VGGFace2 R-50 | **8.83** |

The stronger encoder buys the baselines ~0.2–0.6°, so the ~1° gap to EyeTAG is
not explained by the backbone. The same holds for temporal behaviour: with the
encoder matched, Gaze360-LSTM still over-smooths (`B = −3.34`, vs `+0.05` for
EyeTAG), so the stability difference is architectural.
