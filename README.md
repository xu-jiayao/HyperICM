# HyperICM: Hyperspectral Image Compression for Machines with Task-Agnostic Semantics from Foundation Models

**Jiayao Xu\***, **Yujie Chen\***, **Dingquan Li**, **Wenhan Yang†**<br>
Pengcheng Laboratory<br>
\* Equal contribution · † Corresponding author

HyperICM is a machine-oriented hyperspectral image compression framework. Instead of optimizing only pixel fidelity, it uses task-agnostic semantic features from the hyperspectral foundation model [HyperFree](https://openaccess.thecvf.com/content/CVPR2025/html/Li_HyperFree_A_Channel-adaptive_and_Tuning-free_Foundation_Model_for_Hyperspectral_Remote_CVPR_2025_paper.html) to preserve information useful to downstream vision tasks at extremely low bitrates.

## Overview

Hyperspectral images contain hundreds of strongly correlated spectral bands and are expensive to transmit or store. HyperICM addresses this with a two-stage design:

1. **Offline rate-distortion pretraining.** Adjacent spectral bands are divided into groups and encoded sequentially. Each band is conditioned on features from previously reconstructed bands, allowing the codec to exploit inter-band dependencies. A hyperprior entropy model and learnable QP-dependent rate-control vectors provide practical compression and variable-rate operation.
2. **Semantic-guided online adaptation.** For each target image, only the encoder is fine-tuned while the decoder remains fixed. Multi-layer HyperFree feature consistency between the original and reconstructed hyperspectral images provides task-agnostic semantic supervision without downstream labels or extra model-transmission cost.

## Repository structure

```text
.
├── train.py              # Per-image semantic-guided encoder adaptation
├── inference.py          # Entropy coding and optional downstream evaluation
├── model/                # HyperICM codec implementation
├── Hyperfree/            # HyperFree model and downstream utilities
├── hypericm/             # Dataset, raster, patch, checkpoint, and codec helpers
├── checkpoints/          # Local checkpoint destination (weights are not in Git)
└── requirement.yaml      # Conda environment
```

## Installation

```bash
conda env create -f requirement.yaml
conda activate hypericm
```

## Checkpoints

Download the pretrained weights from **[Google Drive](https://drive.google.com/open?id=1asrxOPqvi2K9d3jzegpYri7XX2mIsseC)** and place them in `checkpoints/`:

| Checkpoint | Purpose | Size | SHA-256 |
| --- | --- | ---: | --- |
| [`Ep10-dev0.076-0.000870-qprandom.pth`](https://drive.google.com/open?id=19bv2YJc3eEbbkrtQOaWclk-JTr8IsMDW) | Offline-pretrained HyperICM codec | 194,261,069 bytes | `3131D51DD10EFDAA75DAF8A78FFE17F582B5D9B89CEA68E54210DD1F97387618` |
| [`Ep40-bpp0.0123-fl0.3819-ml0.0008-fb0.1-mb0.1-qp21.0-featureindex0_1_2_3_4.pth`](https://drive.google.com/open?id=1aRBmYXLW0wVg5ae7TkZc6B9wj-n3zuxk) | Online-adapted HyperICM codec | 194,559,642 bytes | `36403A04DFBBA2C86500F898A431EB623B37D0B3BA37518C71C569EEE5774D53` |
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

Run `python train.py --help` for the complete command-line interface. Configuration files are optional; provide one with `--config PATH`, or pass the required dataset, patch, checkpoint, device, optimization, and evaluation options directly.

## Inference

Reconstruction with HyperFree downstream evaluation:

```bash
python inference.py \
  --input /path/to/WHU-Hi-LongKou.tif \
  --checkpoint /path/to/online-adapted.pth \
  --qps 21 --dataset WHU-Hi-LongKou \
  --downstream --ground-truth /path/to/WHU-Hi-LongKou_gt.tif \
  --hyperfree-checkpoint checkpoints/HyperFree-b.pth
```


## Citation

If you use this repository, please cite HyperICM:

```bibtex
@misc{xu2026hypericm,
  title  = {{HyperICM}: Hyperspectral Image Compression for Machines with Task-Agnostic Semantics from Foundation Models},
  author = {Xu, Jiayao and Chen, Yujie and Li, Dingquan and Yang, Wenhan},
  year   = {2026},
  note   = {Manuscript}
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
``
