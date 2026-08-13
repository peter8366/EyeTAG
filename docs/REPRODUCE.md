# Reproducing the paper

Exact configurations, recovered from the `args` stored inside the released
checkpoints. Every training run used a single GPU, seed 42, 15 epochs.

## Main models

| | Gaze360 | EVE |
|---|---|---|
| face backbone | `vggface2` (ResNet-50) | `vggface2` |
| eye backbone | `resnet18` (shared L/R) | `resnet18` |
| prior | `--prev-mode mlp --prev-input delta` | same |
| prior representation | `pitchyaw` (Δpitch, Δyaw) | same |
| fusion | `cross_attn` | same |
| temporal | `causal`, L=2, h=4, d=256 | same |
| output space | `vector` | `vector` |
| window T | 48 | 48 |
| frame stride / seq stride | 1 / 4 | – / 4 |
| crop size | 128×128 (face and eyes) | 128×128 |
| optimiser | AdamW, lr 3e-4, wd 0.02 | same |
| schedule | cosine, 3-epoch warm-up | same |
| batch size | 32 | 32 |
| grad clip | 1.0 | 1.0 |
| backbone lr scale | 0.1 | 0.1 |
| scheduled sampling | α 1.0 → 0.0 over epochs 1–5 | same |
| cache mode | `ar` | `ar` |
| extra flags | `--autoreg-val --lr-restart-at-prev-zero` | same, plus `--no-pog --cameras webcam_c --target-hz 30` |
| reported checkpoint | `best.pth` | `latest.pth` |

> The EVE number in the paper (2.56°) comes from **`latest.pth`** (final epoch),
> not `best.pth`. Selecting `best.pth` there gives 3.14°.

Run them with:

```bash
bash scripts/train_gaze360.sh /path/to/gaze360 0
bash scripts/eval_gaze360.sh  /path/to/gaze360 work_dir/eyetag_t48_delta/best.pth 0

bash scripts/train_eve.sh /path/to/eve_dataset 0
bash scripts/eval_eve.sh  /path/to/eve_dataset work_dir/eyetag_eve_t48_delta/latest.pth 0
```

## Ablations

All ablations start from the main Gaze360 command and change **one** flag.

| Paper table | Variant | Flag change |
|---|---|---|
| Table 4 (a) | Self-attn + concat | `--fusion-type self_attn_concat` |
| Table 4 (b) | Bidirectional cross-attn | `--fusion-type bidir_cross_attn` |
| Table 5 (a) | Visual only (no prior) | `--prev-mode none` |
| Table 5 (b) | Normal gaze prior Ĝ | `--prev-input abs` |
| Table 6 | Window-length sweep | `--num-frames {4,8,16,24,32,48,64}` |
| Table S1 (a) | Face encoder from scratch | omit `--pretrained-vggface` |
| Table S1 (b) | ImageNet face encoder | `--face-backbone resnet` |
| Table S2 | Eye encoder | `--eye-backbone {efficientnet_b0,tsm,3dcnn}` |
| Table S3 (a) | Teacher forcing only | `--prev-gt-end 1.0` |
| Table S3 (b) | Fully autoregressive | `--prev-gt-start 0.0` |
| Table S4 | Normal-prior window sweep | `--prev-input abs --num-frames ...` |
| Geometric prior variants | tangent / angular velocity | `--prev-repr {tangent,angvel}` |

Two caveats worth knowing before you re-run these:

* **Table S1 (b)** was run with `--face-backbone resnet`, which is an ImageNet
  **ResNet-18**, not a ResNet-50. The supplementary caption states that the face
  architecture is held fixed at ResNet-50 across the three rows; that is not what
  the released configuration does.
* **Table 5 (a)** (visual-only, 9.72 / 9.55 / 8.30) predates the T=48 sweep. A
  re-run of the visual-only variant at T=48 with the current recipe lands near
  8.7 mean, i.e. better than the reported row. Re-derive this row before relying
  on the +0.36° figure attributed to the prior.

The temporal-stability conclusion is unaffected by either caveat: it is measured
within EyeTAG with the encoder held fixed (see below).

## Temporal stability (Table 3)

```bash
# One dump per model you want in the table
python analysis/dump_predictions.py --data-root /path/to/gaze360 \
    --ckpt work_dir/eyetag_t48_delta/best.pth \
    --out analysis/data/eyetag.npz --gpu 0

python analysis/dump_predictions.py --data-root /path/to/gaze360 \
    --ckpt work_dir/eyetag_visual_only/best.pth \
    --out analysis/data/eyetag_visual_only.npz --gpu 0

python analysis/temporal_stability.py --npz-dir analysis/data --out analysis/stability
```

Expected output for the released checkpoint (N_fix = 1,003, N_sac = 3,306):

```
Model                       N_fix  N_sac |      J       B      M |   DirErr  <90deg%  static%
eyetag                       1003   3306 |   3.68    0.05   1.87 |    46.08     84.1      0.6
```

`static%` guards against the "an overly static predictor also gets low jitter"
failure mode: EyeTAG's predictions are near-static on only 0.6% of saccade frame
pairs, and its predicted/GT velocity correlation is 0.824.

## Complexity numbers

Measured with `fvcore` on a single RTX A6000, FP32, batch size 1, averaged over
10 runs with the extremes trimmed: 37.91 M parameters, 121.57 GFLOPs and 19.97 ms
per 48-frame window. Parameters split as face encoder 23.56 M, shared eye encoder
11.19 M, TAKE + CCAF + GVP ≈ 3.2 M.
