# Asymmetry-Aware-Landmark Network: AALNet

AALNet detects 33 landmarks on posteroanterior (PA) cephalograms. It is trained
with a DSNT loss on the landmark coordinates and an *asymmetry loss* that directly
supervises the facial-asymmetry measurements derived from the predicted landmarks
(midline direction, lower-face, midface and dental deviation, occlusal canting).

This repository contains the model, training and evaluation code of the paper
*"Asymmetry-aware landmark Network: clinical-asymmetry-aware deep learning for
automatic landmark detection on posteroanterior cephalograms"* (under review).

> **Code-only release.** The radiographs cannot be shared, and trained weights
> are not included. To use the code you need your own PA cephalograms annotated
> with the 33 landmarks listed [below](#landmarks).

## Model architecture

```
PA cephalogram
   │  skull ROI crop (Otsu) + letterbox to 800×800            aalnet/preprocess/
   ▼
HRNet-W48 backbone (ImageNet-initialised, timm)               models/backbones/hrnet.py
   │  4 feature maps: stride 4 / 8 / 16 / 32
   │                  128 / 256 / 512 / 1024 channels
   ▼
FPN-style fusion decoder                                      models/decoder/fpn_decoder.py
   │  top-down fusion  stride 32 → 4   (256 channels)
   │  learned upsampling stride 4 → 2 → 1  (128 → 64 channels)
   ▼
Heatmap layers: 3×3 conv → BN → ReLU → 1×1 conv               models/heatmap/heatmap_layers.py
   │  33 heatmaps at 800×800
   ▼
Soft-argmax (spatial softmax, expected position)              utils/heatmap_utils.py
   │  33 (x, y) coordinates
   ▼
mapped back to the original image → mm → 9 asymmetry measurements
```

The three parts are assembled in `aalnet/models/aalnet.py`.

| Component | Details | Parameters |
|---|---|---|
| Backbone | HRNet-W48, multi-resolution features at strides 4–32 | 67.1 M |
| Decoder | 1×1 lateral projection + LayerNorm at each level; a residual conv block and 3 ConvNeXt V2 blocks at stride 32, then 2 ConvNeXt V2 blocks at strides 16 and 8, with bilinear ×2 upsampling and addition between levels; two learned upsampling stages to full resolution (3×3 conv, with 2 ConvNeXt V2 blocks at stride 2) | 6.1 M |
| Heatmap layers | one full-resolution heatmap per landmark | 0.04 M |
| Total | | 73.3 M |

**Base loss.** Each heatmap is turned into a probability map by a spatial softmax
and the coordinate is its expected position (DSNT, Nibali et al. 2018). With
coordinates normalised to [−1, 1], the base loss is the mean squared coordinate
error plus the Jensen–Shannon divergence between the probability map and a
Gaussian (σ = 5 px) centred on the ground truth
(`losses/dsnt_loss.py`).

**Asymmetry loss** (`losses/asymmetry_loss.py`). Twelve of the 33 landmarks define
two reference midlines and five deviation measurements:

- MSR: the perpendicular bisector of the right and left latero-orbitale;
- Cg–ANS line: from crista galli to the anterior nasal spine;
- lower-face deviation (menton to MSR), midface asymmetry (right vs. left zygoma
  to MSR), two dental deviations (upper dental midline, and the midpoint of the
  maxillary central-incisor crowns, to MSR), and occlusal canting (difference of
  the right and left maxillary first-molar crowns projected along the MSR).

```math
\mathcal{L}_{\mathrm{asym}} = \sum_{m} \alpha_m \left(1 - \cos\theta_m\right) + \alpha_{\mathrm{dev}} \sum_{k=1}^{5} \left| d_k(\hat{\mathbf{p}}) - d_k(\mathbf{p}^{*}) \right|
```

where *m* runs over the two midlines (MSR and Cg–ANS), θ<sub>m</sub> is the angle
between the predicted and ground-truth direction of midline *m*, and
d<sub>k</sub> are the five deviation measurements. Predicted deviations are
measured against the MSR built from the *predicted* landmarks, and ground-truth
deviations against the ground-truth MSR. We use α<sub>MSR</sub> = 0.6,
α<sub>Cg–ANS</sub> = 0.4 and α<sub>dev</sub> = 1.0.

The training objective is

```math
\mathcal{L} = \mathcal{L}_{\mathrm{base}} + \lambda_{\mathrm{asym}}\, \mathcal{L}_{\mathrm{asym}}
```

with λ<sub>asym</sub> = 0.2 for AALNet. Setting λ<sub>asym</sub> = 0 gives
*AALNet without asymmetry loss*: the same network trained with the base loss
alone.

## Repository layout

```
.
├── aalnet/
│   ├── configs/        # the five configurations reported in the paper
│   ├── datasets/       # dataset class and training augmentation
│   ├── losses/         # base loss (dsnt_loss.py), asymmetry loss (asymmetry_loss.py)
│   ├── models/         # AALNet (aalnet.py): backbones/, decoder/, heatmap/
│   ├── preprocess/     # ROI detection, crop + letterbox, fold splits, pixel spacing
│   ├── scripts/        # train.py, eval_checkpoint.py
│   ├── trainers/       # training and validation loops
│   └── utils/          # MRE/SDR, asymmetry measurements, heatmap utilities, coordinate mapping
└── pyproject.toml
```

## Setup

```bash
# Python 3.10+, PyTorch with CUDA
pip install -e .
```

## Data preparation

To train on your own data, put each radiograph and its 33 annotated landmarks
into the layout below. Everything else (cropping, splitting, mapping predictions
back to the original image) is done by the scripts.

```
dataset/                             # any name; passed as --src_root
├── images/raw/<name>.jpg            # original radiograph, 8-bit (read as grayscale)
└── coords/<name>.csv                # landmark coordinates of the same image
```

`<name>` is `<patientID>_<timepoint>`, e.g. `P0001_preop`. The part before the
last underscore is taken as the patient ID, so that all images of one patient
fall into the same split. Use pseudonymised IDs.

Each coordinate file has three rows: the landmark names, the x coordinates and
the y coordinates, in pixels of the original image (origin at the top-left
corner), with the 33 landmarks in the order of the [table below](#landmarks):

```
,Antegonion right,Antegonion left,Anterior nasal spine,...,Lower dental midline
X,412.0,1523.5,968.0,...,960.5
Y,1488.0,1470.0,905.5,...,1402.0
```

Millimetre evaluation also needs the pixel spacing of every original image, as a
JSON file mapping `<name>` to mm/pixel:

```json
{"P0001_preop": 0.135, "P0001_pod1y": 0.135, "P0002_init": 0.150}
```

(`aalnet/preprocess/build_pixel_spacing_map.py` can build this file when the pixel
spacing is known per image size.) Then run:

```bash
# 1. Skull ROI bounding boxes by Otsu thresholding
python -m aalnet.preprocess.roi_detect \
    --src_root /path/to/dataset --output_dir /path/to/roi_otsu

# 2. Crop + letterbox to 800×800; landmark coordinates are transformed accordingly
python -m aalnet.preprocess.crop_and_letterbox \
    --bbox_csv /path/to/roi_otsu/bboxes.csv \
    --src_root /path/to/dataset \
    --dst_root /path/to/dataset_800 --target_size 800

# 3. Patient-level split: fixed test set (10%) + nine folds on the rest
python -m aalnet.preprocess.create_fold_splits --data_dir /path/to/dataset_800
```

This produces the directory passed as `--data_root` to training and evaluation:

```
dataset_800/
├── images/<name>.jpg              # 800×800 network input
├── coords/<name>.csv              # landmark coordinates in the 800×800 image
├── transform_params/<name>.json   # crop box, scale and padding, used to map predictions back
└── fold_splits.json               # test_files and folds[i].train_files / val_files
```

<a name="landmarks"></a>
<details>
<summary>The 33 landmarks, in the column order expected in the coordinate files</summary>

| # | Landmark | # | Landmark | # | Landmark |
|---|---|---|---|---|---|
| 0 | Antegonion right | 11 | Left maxillary 6 root | 22 | Zygomatico-maxillary suture left |
| 1 | Antegonion left | 12 | Left mandibular 6 crown | 23 | Jugal process left |
| 2 | **Anterior nasal spine** | 13 | Left mandibular 6 root | 24 | Jugal process right |
| 3 | **Menton** | 14 | Left mandibular 1 crown | 25 | **Zygoma right** |
| 4 | **Right maxillary 6 crown** | 15 | Left mandibular 1 root | 26 | **Zygoma left** |
| 5 | Right maxillary 6 root | 16 | Right mandibular 1 crown | 27 | Mastoid right |
| 6 | **Right maxillary 1 crown** | 17 | Right mandibular 1 root | 28 | Mastoid left |
| 7 | Right maxillary 1 root | 18 | Right mandibular 6 crown | 29 | **Latero-orbitale right** |
| 8 | **Left maxillary 1 crown** | 19 | Right mandibular 6 root | 30 | **Latero-orbitale left** |
| 9 | Left maxillary 1 root | 20 | **Crista galli** | 31 | **Upper dental midline** |
| 10 | **Left maxillary 6 crown** | 21 | Zygomatico-maxillary suture right | 32 | Lower dental midline |

The 12 landmarks in bold enter the asymmetry loss and the asymmetry measurements.
The indices are fixed in `aalnet/utils/clinical_metrics.py`, and the
left/right pairs swapped under horizontal flipping in
`aalnet/datasets/landmark_dataset.py`.

</details>

## Training

Training uses an effective batch size of 64 (number of GPUs × `batch_size` ×
`accum_iter`). The configs are set for two GPUs (`batch_size 8`, `accum_iter 4`);
change `--accum_iter` on the command line for another number of GPUs, keeping
`batch_size` at 8 per GPU.

```bash
# two GPUs (config default): 2 × 8 × 4 = 64
torchrun --nproc_per_node=2 -m aalnet.scripts.train \
    --config aalnet/configs/lambda02.yaml \
    --data_root /path/to/dataset_800 \
    --fold 0 --output_dir outputs/lambda02/fold0

# one GPU: 1 × 8 × 8 = 64
python -m aalnet.scripts.train \
    --config aalnet/configs/lambda02.yaml \
    --data_root /path/to/dataset_800 \
    --fold 0 --accum_iter 8 --output_dir outputs/lambda02/fold0
```

Use the output directory pattern `<output_root>/<config name>/fold<N>`; the
evaluation script discovers checkpoints by it. The paper trains folds 0–4 of
each configuration and reports the mean over the five models on the fixed test
set.

| Config | Model | λ<sub>asym</sub> |
|---|---|---|
| `lambda02.yaml` | **AALNet** | 0.2 |
| `withoutAsyLoss.yaml` | AALNet without asymmetry loss | 0 |
| `lambda005.yaml` | loss-weight ablation | 0.05 |
| `lambda05.yaml` | loss-weight ablation | 0.5 |
| `lambda10.yaml` | loss-weight ablation | 1.0 |

The five configurations differ only in the asymmetry-loss weight. Common
settings: AdamW (learning rate 10<sup>−4</sup>, weight decay 0.05), 5 warm-up epochs
followed by cosine decay, effective batch size 64
(GPUs × `batch_size` × `accum_iter`), up to 400 epochs with early stopping on the
validation MRE (patience 60), bfloat16 mixed precision. Augmentation: rotation
(±10°), scaling (0.9–1.1), translation (±10 px), brightness and contrast jitter,
and horizontal flipping with the left and right landmark labels swapped.
Arguments given on the command line override the values in the config file.

Training writes `checkpoint_best.pth` (lowest MRE, in pixels, on the validation
fold), `best_mre_per_landmark.csv`, `config.json` and `log.jsonl` to the output
directory. The ImageNet weights of the backbone are downloaded by timm on first
use.

## Evaluation

```bash
python -m aalnet.scripts.eval_checkpoint \
    --output_root outputs \
    --data_root /path/to/dataset_800 \
    --pixel_spacing_file /path/to/pixel_spacing_per_image.json \
    --split test            # or: --split val
```

Predictions are mapped back to the original image and converted to millimetres.
For every `<config>/fold<N>` the script writes per-image CSV files, and under
`<output_root>/eval/` the results aggregated over folds:

- **MRE** (mean radial error, mm), overall and per landmark, and **SDR** at 2,
  2.5, 3 and 4 mm;
- the nine **asymmetry measurement errors**: angle (°) and horizontal
  displacement (mm) of the MSR and of the Cg–ANS line, lower-face deviation,
  midface asymmetry, the two dental deviations and occlusal canting (mm), computed
  in `aalnet/utils/clinical_metrics.py`.

Overlays of predicted and ground-truth landmarks on the original radiographs are
saved when `--original_images_dir /path/to/dataset/images/raw` is given.

## License

This code is released under the
[Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/)
(CC BY-NC 4.0; see [LICENSE](LICENSE)): it may be used, shared and adapted for
non-commercial purposes with attribution. Commercial use is not permitted.
Third-party projects this code builds on are listed in [NOTICE](NOTICE).

## Citation

The paper is under review; the reference will be added here on publication.
