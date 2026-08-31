# HyperICM 

This directory is the portable release of the HyperICM paper project. It uses
the user-supplied attachment for online encoder adaptation (choice A) and the
A+2 inference path: real entropy-coded compression/decompression with optional
Hyperfree downstream classification.

The source is self-contained, while data and model weights remain external
artifacts. The paper PDF in the parent research workspace is not required at
runtime.

## Contents

- `train.py` — fixed-eight-band online encoder adaptation, validation, and checkpointing.
- `inference.py` — entropy coding at one or more QPs and optional classification.
- `model/` — selected `model_wo_adapter` codec implementation.
- `Hyperfree/` — vendored Hyperfree model and mask-generation code.
- `hypericm/` — metadata, raster, patch, checkpoint, and codec helpers.
- `requirement.yaml` — Conda environment description.
- `checkpoints/README.md` — external checkpoint inventory and SHA-256 hashes.
- `ATTACHMENT_DIFF.md` — requested comparison with `single_image_encoder_patch.py`.
- `examples/train.yaml` — path-editable training configuration.

## Installation

```bash
conda env create -f requirement.yaml
conda activate hypericm
```

The YAML targets Python 3.10, PyTorch 2.1, torchvision 0.16, and CUDA 12.1.
For another CUDA version or CPU-only use, change/remove `pytorch-cuda=12.1`
and install a matching PyTorch/torchvision pair. `opencv-python`,
`pycocotools`, and GDAL are optional Hyperfree-utility dependencies and are
not needed by the two release entry points.

## Checkpoints

Weights are not copied into GitHub because all supplied `.pth` files exceed
100 MB. Follow [`checkpoints/README.md`](checkpoints/README.md) to copy them
from your external artifact directory, use Git LFS, or pass an external path.
Default training expects
`Ep10-dev0.076-0.000870-qprandom.pth` and `HyperFree-b.pth`.

## Data layout

`--patch-dir` must contain TIFF patches read as `bands × height × width`.
Each patch must match the selected dataset and is min-max normalized
independently.

| Dataset | Bands | Base GSD |
| --- | ---: | ---: |
| `WHU-Hi-LongKou` | 270 | 0.463 |
| `WHU-Hi-HanChuan` | 274 | 0.109 |
| `WHU-Hi-HongHu` | 270 | 0.043 |

Exact attachment wavelengths are embedded in `hypericm/datasets.py`. Images
with nodata/masked or non-finite values are rejected; label nodata is mapped
to background class 0.

## Training

Edit `examples/train.yaml`, then run from this repository root:

```bash
python train.py --config examples/train.yaml
```

Equivalent command-line example:

```bash
python train.py \
  --dataset WHU-Hi-LongKou \
  --patch-dir /path/to/patches \
  --base-checkpoint checkpoints/Ep10-dev0.076-0.000870-qprandom.pth \
  --hyperfree-checkpoint checkpoints/HyperFree-b.pth \
  --device cuda:0 --hyperfree-device cuda:0 \
  --qp 21 --epochs 100 --batch-size 1 \
  --loss-weights 0.5 0.06 5 10
```

`--loss-weights` uses `[semantic, MSE, rate, global_scale]`; `--para` is a
compatibility alias. `--qp 0` samples a concrete QP per batch.
`--data-parallel-devices 0 1 2` enables optional CUDA DataParallel (the first
index must match `--device`). Add `--full-image PATH` for periodic codec
evaluation and add `--run-downstream --ground-truth PATH` for classification.

Runs write timestamped directories containing TensorBoard scalars, validation
metrics, and checkpoints. The rate term intentionally keeps the attachment's
latent-element likelihood surrogate and group aggregation for compatibility.

## Inference

Basic entropy-coded inference:

```bash
python inference.py \
  --input /path/to/WHU-Hi-LongKou.tif \
  --checkpoint checkpoints/Ep10-dev0.076-0.000870-qprandom.pth \
  --qps 1 21 63 --device cuda:0 --output-dir results
```

With Hyperfree classification:

```bash
python inference.py \
  --input /path/to/WHU-Hi-LongKou.tif \
  --checkpoint /path/to/online-adapted.pth \
  --qps 21 --dataset WHU-Hi-LongKou \
  --downstream --ground-truth /path/to/WHU-Hi-LongKou_gt.tif \
  --hyperfree-checkpoint checkpoints/HyperFree-b.pth
```

Inference accepts QP 1–63 and uses actual CompressAI streams. For each QP,
`<output-dir>/<input-stem>/` receives a reconstructed `.npy`, `metrics.csv`,
and `run.json`; downstream mode also writes a classification GeoTIFF. `bpsp`
is compressed bits divided by input pixels. MSE/PSNR use the reconstruction
after codec padding is cropped to the original size.

## Reproducibility notes

- Codec: `model_wo_adapter`; `model_single_head` and `model_w_ckpt` are unused.
- Only I-frame `enc` and P-frame `encoder` parameters are optimized.
- Fixed eight-band groups, 16-pixel spatial padding, and an empty-tail guard are used.
- Full-group reconstructions are restored in spectral order; the attachment's
  original `cat(dim=0)`/`view` sequence scrambled bands across groups.
- Training likelihood-rate and inference byte-rate are different measurements.
- If Hyperfree creates no mask covering a few-shot seed, classification fails
  explicitly instead of silently omitting that class.

See [`ATTACHMENT_DIFF.md`](ATTACHMENT_DIFF.md) for the complete comparison.

## Citation and licensing

Please cite the associated HyperICM paper and preserve
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). Copied files retain their
copyright headers; verify upstream licensing before redistributing a fork.
