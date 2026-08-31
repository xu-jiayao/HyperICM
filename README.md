# HyperICM: Hyperspectral Image Compression for Machines with Task-Agnostic Semantics from Foundation Models

**Jiayao Xu\***, **Yujie Chen\***, **Dingquan Li**, **Wenhan Yang†**<br>
Pengcheng Laboratory<br>
\* Equal contribution · † Corresponding author

HyperICM is a machine-oriented hyperspectral image compression framework. Instead of optimizing only pixel fidelity, it uses task-agnostic semantic features from the hyperspectral foundation model [HyperFree](https://openaccess.thecvf.com/content/CVPR2025/html/Li_HyperFree_A_Channel-adaptive_and_Tuning-free_Foundation_Model_for_Hyperspectral_Remote_CVPR_2025_paper.html) to preserve information useful to downstream vision tasks at extremely low bitrates.

## Overview

Hyperspectral images contain hundreds of strongly correlated spectral bands and are expensive to transmit or store. HyperICM addresses this with a two-stage design:

1. **Offline rate-distortion pretraining.** Adjacent spectral bands are divided into groups and encoded sequentially. Each band is conditioned on features from previously reconstructed bands, allowing the codec to exploit inter-band dependencies. A hyperprior entropy model and learnable QP-dependent rate-control vectors provide practical compression and variable-rate operation.
2. **Semantic-guided online adaptation.** For each target image, only the encoder is fine-tuned while the decoder remains fixed. Multi-layer HyperFree feature consistency between the original and reconstructed hyperspectral images provides task-agnostic semantic supervision without downstream labels or extra model-transmission cost.

This repository contains the online encoder-adaptation and entropy-coded inference paths used for the paper experiments. The offline-pretrained codec is provided as an external checkpoint.

## Main results

The paper evaluates HyperICM on three WHU-Hi datasets over approximately 0.005–0.04 bits per spectral pixel (BPSP). Relative to JPEG 2000, semantic-guided online adaptation achieves the following OA-oriented BD-Rate reductions:

| Dataset | Bands | BD-Rate (OA) ↓ | BD-OA ↑ |
| --- | ---: | ---: | ---: |
| WHU-Hi-LongKou | 270 | **−75.207%** | **0.175** |
| WHU-Hi-HongHu | 270 | **−72.090%** | **0.277** |
| WHU-Hi-HanChuan | 274 | **−46.550%** | **0.118** |

The evaluation includes JPEG 2000, HEVC, VVC, and LoR-SGS baselines. Visual quality is measured with PSNR, while machine performance is measured by HyperFree semantic-segmentation overall accuracy (OA).

In the paper protocol, online adaptation uses `256 × 256` patches, a learning rate of `1e-5` for LongKou/HongHu and `5e-5` for HanChuan, and 100/30/30 epochs for LongKou/HongHu/HanChuan, respectively. The dataset- and rate-specific settings are:

| Target BPSP | LongKou `[QP, alpha, beta]` | HongHu `[QP, alpha, beta]` | HanChuan `[QP, alpha, beta]` |
| ---: | --- | --- | --- |
| 0.005 | `[1, 4.8, 50]` | `[1, 0.5, 10]` | `[2, 1, 50]` |
| 0.010 | `[2, 6, 50]` | `[1, 10^3, 10]` | `[2, 4.7, 50]` |
| 0.020 | `[5, 15, 50]` | `[5, 5, 20]` | `[6, 33, 50]` |
| 0.030 | `[9, 29, 50]` | `[9, 10, 20]` | `[15, 33, 50]` |
| 0.040 | `[17, 40, 50]` | `[13, 20, 20]` | `[34, 41, 50]` |

## Repository structure

```text
.
├── train.py              # Per-image semantic-guided encoder adaptation
├── inference.py          # Entropy coding and optional downstream evaluation
├── model/                # HyperICM codec implementation
├── Hyperfree/            # HyperFree model and downstream utilities
├── hypericm/             # Dataset, raster, patch, checkpoint, and codec helpers
├── checkpoints/          # Local checkpoint destination (weights are not in Git)
├── examples/train.yaml   # Example training configuration
└── requirement.yaml      # Conda environment
```

## Installation

```bash
conda env create -f requirement.yaml
conda activate hypericm
```

The provided environment targets Python 3.10, PyTorch 2.1, torchvision 0.16, and CUDA 12.1. For a different CUDA version or CPU-only execution, install a compatible PyTorch/torchvision pair and adjust `requirement.yaml`. The codec and HyperFree paths are GPU-oriented.

## Checkpoints

Download the pretrained weights from **[Google Drive](https://drive.google.com/open?id=1asrxOPqvi2K9d3jzegpYri7XX2mIsseC)** and place them in `checkpoints/`:

| Checkpoint | Purpose | Size | SHA-256 |
| --- | --- | ---: | --- |
| [`Ep10-dev0.076-0.000870-qprandom.pth`](https://drive.google.com/open?id=19bv2YJc3eEbbkrtQOaWclk-JTr8IsMDW) | Offline-pretrained HyperICM codec | 194,261,069 bytes | `3131D51DD10EFDAA75DAF8A78FFE17F582B5D9B89CEA68E54210DD1F97387618` |
| [`Ep40-bpp0.0123-fl0.3819-ml0.0008-fb0.1-mb0.1-qp21.0-featureindex0_1_2_3_4.pth`](https://drive.google.com/open?id=1aRBmYXLW0wVg5ae7TkZc6B9wj-n3zuxk) | Example online-adapted codec | 194,559,642 bytes | `36403A04DFBBA2C86500F898A431EB623B37D0B3BA37518C71C569EEE5774D53` |
| [`HyperFree-b.pth`](https://drive.google.com/open?id=1ocgjYiRkUPoVeEtE_J1_skL2_ftHDoiW) | HyperFree ViT-B backbone | 615,520,591 bytes | `BF207BF0D6A64E5A5E939CFFBB850B0C72E29DDA894EF9EDA8950958688664C3` |

The files exceed GitHub's regular 100 MB limit and are intentionally excluded from Git history.

## Data preparation

`--patch-dir` must contain TIFF patches stored as `bands × height × width`. The default `--patch-size` is 256, so each patch must be exactly `256 × 256` unless that option is changed. Every patch must have the band count of the selected dataset and is min-max normalized independently.

| Dataset | Bands | Base GSD |
| --- | ---: | ---: |
| `WHU-Hi-LongKou` | 270 | 0.463 |
| `WHU-Hi-HanChuan` | 274 | 0.109 |
| `WHU-Hi-HongHu` | 270 | 0.043 |

The exact wavelengths used in the experiments are defined in `hypericm/datasets.py`. Images containing nodata, masked, NaN, or infinite values are rejected; label nodata is mapped to background class 0.

## Online adaptation

Edit `examples/train.yaml`, then run:

```bash
python train.py --config examples/train.yaml
```

Command-line example for the packaged training implementation:

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

`--loss-weights` represents the implementation-specific tuple `[semantic, MSE, rate, global_scale]`; it is not a direct transcription of the paper's symbolic `[alpha, beta]` pair. `--para` is retained as a compatibility alias, and `--qp 0` samples a concrete QP for each batch. Add `--full-image PATH` for periodic codec evaluation and `--run-downstream --ground-truth PATH` for few-shot semantic-segmentation evaluation.

## Inference

Entropy-coded reconstruction:

```bash
python inference.py \
  --input /path/to/WHU-Hi-LongKou.tif \
  --checkpoint checkpoints/Ep10-dev0.076-0.000870-qprandom.pth \
  --qps 1 21 63 --device cuda:0 --output-dir results
```

Reconstruction with HyperFree downstream evaluation:

```bash
python inference.py \
  --input /path/to/WHU-Hi-LongKou.tif \
  --checkpoint /path/to/online-adapted.pth \
  --qps 21 --dataset WHU-Hi-LongKou \
  --downstream --ground-truth /path/to/WHU-Hi-LongKou_gt.tif \
  --hyperfree-checkpoint checkpoints/HyperFree-b.pth
```

For each QP, `<output-dir>/<input-stem>/` contains the reconstructed array, `metrics.csv`, and `run.json`; downstream mode additionally writes a semantic-segmentation GeoTIFF. The saved `.npy` reconstruction remains in min-max-normalized units; the original minimum and maximum are recorded in `run.json` but are not reapplied automatically. Reported BPSP counts the `y`/`z` entropy payload bytes divided by the number of spatial pixels and spectral bands; container and side metadata are not included.

## Implementation notes

- Spectral bands are processed in fixed groups of eight, with a guard for a shorter final group.
- Spatial dimensions are padded to multiples of 16 and cropped back after decoding.
- Online adaptation updates only the I-frame `enc` and P-frame `encoder` modules.
- Inference uses actual CompressAI entropy streams; its entropy-payload BPSP differs from the likelihood-based rate surrogate used during optimization.

## Citation

If you use this repository, please cite HyperICM:

```bibtex
@inproceedings{xu2026hypericm,
  title={Hypericm: Hyperspectral Image Compression for Machines with Task-Agnostic Semantics from Foundation Models},
  author={Xu, Jiayao and Chen, Yujie and Li, Dingquan and Yang, Wenhan},
  booktitle={2026 IEEE International Conference on Image Processing (ICIP)},
  pages={1--6},
  year={2026},
  organization={IEEE}
}
```

HyperICM uses [HyperFree](https://doi.org/10.1109/CVPR52734.2025.02146) as its task-agnostic hyperspectral foundation model. Please also cite:

```bibtex
@InProceedings{Li_2025_CVPR,
  author    = {Li, Jingtao and Liu, Yingyi and Wang, Xinyu and Peng, Yunning and Sun, Chen and Wang, Shaoyu and Sun, Zhendong and Ke, Tian and Jiang, Xiao and Lu, Tangwei and Zhao, Anran and Zhong, Yanfei},
  title     = {HyperFree: A Channel-adaptive and Tuning-free Foundation Model for Hyperspectral Remote Sensing Imagery},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2025},
  pages     = {23048--23058},
  doi       = {10.1109/CVPR52734.2025.02146}
}
```

