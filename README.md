# EyeTAG: Eye Trajectory-Aware Gaze Estimation

Official implementation of **EyeTAG**, a causal multi-frame gaze estimator built
around an *explicit* first-order gaze prior: at every step the model
differentiates its own recent predictions and feeds that differential trajectory
back in as a compact kinematic token.

> Most video gaze estimators do model time, but only *implicitly* — motion has to
> be recovered from high-dimensional appearance features that must simultaneously
> encode identity, illumination and head pose. EyeTAG instead carries the gaze
> trajectory as an explicit variable.

---

## Results

### Gaze360 — mean angular error (deg, lower is better)

| Type | Method | All | Semi-Front | Front | Mean |
|---|---|---|---|---|---|
| SF | L2CS-Net | 11.10 | 10.77 | 9.92 | 10.60 |
| SF | GazeTR-Hybrid | 10.56 | 10.30 | 9.63 | 10.16 |
| SF | CrossGaze | 10.47 | 10.29 | 9.23 | 10.00 |
| MF | Gaze360 | 10.46 | 10.18 | 9.32 | 9.99 |
| MF | STAGE | 12.08 | 11.63 | 9.43 | 11.05 |
| MF | **EyeTAG (ours)** | **9.29** | **9.12** | **8.08** | **8.83** |

### EVE — official validation split, `webcam_c` @ 30 Hz

| Method | MAE (deg) |
|---|---|
| CrossGaze | 3.30 |
| Gaze360 | 2.73 |
| STAGE | 2.58 |
| **EyeTAG (ours)** | **2.56** |

### Temporal stability on Gaze360 (deg/frame)

`J` fixation jitter (lower better), `B` signed saccade bias (closer to 0 better),
`M = (J + |B|) / 2`.

| Method | J | B | M |
|---|---|---|---|
| CrossGaze | 5.06 | +0.23 | 2.65 |
| Gaze360 (non-causal) | 1.71 | −4.24 | 2.98 |
| STAGE | 3.81 | −0.89 | 2.35 |
| **EyeTAG (ours)** | 3.68 | **+0.05** | **1.87** |

Removing only the kinematic prior (same encoder, same everything else) restores a
systematic saccade under-shoot: visual-only `B = −0.46`, normal prior
`B = −0.53`. Reproduce with `scripts/temporal_stability.sh`.

Complexity of the released model: **37.91 M parameters**, 121.57 GFLOPs and
19.97 ms per 48-frame window (RTX A6000, FP32, batch size 1). The GFLOPs figure
is a no-caching upper bound — the visual encoders are per-frame, so a streaming
deployment can cache them across the sliding window.

---

## Installation

```bash
git clone https://github.com/peter8366/EyeTAG.git
cd EyeTAG
conda create -n eyetag python=3.11 -y && conda activate eyetag
pip install -r requirements.txt
```

`decord` and `h5py` are only needed for EVE. Gaze360 runs without them.

### Pretrained face backbone

The face encoder is a ResNet-50 pretrained on **VGGFace2**. Download
`resnet50_ft_weight.pkl` from the
[VGGFace2 pytorch release](https://github.com/cydonia999/VGGFace2-pytorch)
and place it at:

```
checkpoints/resnet50_ft_weight.pkl
```

---

## Data preparation

### Gaze360

Use the standard preprocessed Gaze360 layout (per-frame JPEG crops plus `.label`
index files), as produced by [GazeHub](http://phi-ai.buaa.edu.cn/Gazehub/):

```
<GAZE360_ROOT>/
├── train.label  val.label  test.label
├── train/Face/<sid>/<sid>_<idx>.jpg
├── train/Left/...   train/Right/...
└── test/...
```

Each label line is `Face Left Right Origin 3DGaze 2DGaze`, where `3DGaze` is
`gx,gy,gz` (unit vector) and `2DGaze` is `gyaw,gpitch` in radians.

The three evaluation subsets follow the original Gaze360 convention and are
selected with `--gaze-subset`: `full` (|yaw| ≤ 180°), `semi-front` (≤ 90°),
`front` (≤ 20°).

### EVE

Use the official [EVE](https://ait.ethz.ch/eve) release as-is (MP4 + HDF5):

```
<EVE_ROOT>/
├── train01/ ... val01/ ... test01/ ...
│   └── step008_image_MIT-i2263005350/
│       ├── webcam_c.mp4  webcam_c_eyes.mp4  webcam_c.h5
```

The paper uses the frontal `webcam_c` stream only, subsampled to 30 Hz.

> `--split test` in `eve/test.py` selects participants `val01..val05`, i.e. the
> official EVE **validation** split. This is what the paper reports — the official
> test annotations are not publicly available.

---

## Training

```bash
# Gaze360 (main result)
bash scripts/train_gaze360.sh /path/to/gaze360 0

# EVE
bash scripts/train_eve.sh /path/to/eve_dataset 0
```

Both scripts spell out every hyper-parameter used for the paper numbers; see
the docstring at the top of `gaze360/train.py` for the equivalent raw command.

**Scheduled sampling.** The gaze history fed to TAKE is sampled from the ground
truth with probability α, otherwise from a cache of the model's own predictions
that is rebuilt once per epoch (`--cache-mode ar`). α is annealed 1.0 → 0.0 over
epochs 1–5, after which the history is fully autoregressive. Training remains
fully parallel — no sequential rollout is needed, because the cache is built
once per epoch rather than on the fly.

Key flags:

| Flag | Paper setting | Meaning |
|---|---|---|
| `--prev-input` | `delta` | `delta` = kinematic prior ΔĜ; `abs` = normal prior Ĝ |
| `--prev-repr` | `pitchyaw` | also `vector`, `tangent`, `angvel` (geometrically exact alternatives) |
| `--num-frames` | `48` | temporal window T |
| `--fusion-type` | `cross_attn` | also `bidir_cross_attn`, `self_attn_concat` |
| `--prev-gt-end-epoch` | `5` | end of the scheduled-sampling ramp |
| `--prev-mode` | `mlp` | `none` disables the prior entirely (visual-only) |

---

## Evaluation

```bash
# Gaze360: autoregressive inference on all three yaw subsets
bash scripts/eval_gaze360.sh /path/to/gaze360 work_dir/eyetag_t48_delta/best.pth 0

# EVE
bash scripts/eval_eve.sh /path/to/eve_dataset work_dir/eyetag_eve_t48_delta/latest.pth 0
```

`--autoregressive` is the deployment setting reported in the paper: the gaze
history is built from the model's own predictions, rolled out per subject track.
Without it, the model is evaluated under teacher forcing.

### Temporal stability (Table 3)

```bash
bash scripts/temporal_stability.sh /path/to/gaze360 work_dir/eyetag_t48_delta/best.pth 0
```

This dumps frame-ordered predictions to `analysis/data/*.npz` and then computes
J / B / M plus the saccade-level diagnostics (directional error, near-static
prediction rate). Drop additional `.npz` dumps into the same directory to compare
several models in one table.

---

## Repository layout

```
eyetag/                  shared model package
├── geometry.py          pitchyaw <-> vector, angular error, differential gaze
└── models/
    ├── model.py         GazeEstimator: DSVE + TAKE + CCAF + GVP
    ├── face_vggface2.py face encoder (ResNet-50, VGGFace2)     [ours]
    ├── face_resnet.py   face encoder (ResNet-18, ImageNet)     [ablation]
    ├── eye_net.py       eye encoders: resnet18 [ours], efficientnet_b0, tsm, 3dcnn
    ├── prev_encoder.py  TAKE — differential gaze trajectory -> kinematic token
    ├── temporal_fusion.py  CCAF — cross-attention + causal Transformer decoder
    └── head.py          GVP — gaze regression head

gaze360/                 Gaze360 pipeline (train / test / dataset / utils)
eve/                     EVE pipeline (train / test / dataset / utils)
analysis/                temporal-stability analysis (paper Table 3)
scripts/                 one-command reproduction of every paper number
baselines/               patches + scripts used to re-train the baselines
docs/                    reproduction notes and baseline provenance
```

The two dataset pipelines are kept separate on purpose: they differ in media
(JPEG frames vs MP4 + HDF5), label convention and sampling, and merging them
would obscure the exact configuration behind each reported number. Everything
they share — the model and the gaze-space math — lives in `eyetag/`.

---

## Baselines

All baselines in the comparison tables were **re-trained by us** under an
identical protocol rather than quoted. The upstream repositories are not vendored
here; `docs/BASELINES.md` lists each source commit together with the patches in
`baselines/patches/` and the driver scripts in `baselines/scripts/` needed to
reproduce our runs, including the backbone-controlled variants.

---

## Citation

```bibtex
@inproceedings{eyetag2026,
  title     = {EyeTAG: Eye Trajectory-Aware Gaze Estimation},
  author    = {Lee, Jungmin and Ullah, Niamat and Han, Yoseob},
  booktitle = {British Machine Vision Conference (BMVC)},
  year      = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE). Note that the Gaze360 and EVE datasets, and the
VGGFace2 pretrained weights, carry their own licenses and must be obtained from
their original providers.

## Acknowledgements

The face encoder uses VGGFace2 pretrained weights; the eye-encoder ablations use
TSM, C3D and EfficientNet-B0. We thank the authors of Gaze360, EVE, L2CS-Net,
GazeTR, CrossGaze, STAGE and MCGaze for releasing their code.
