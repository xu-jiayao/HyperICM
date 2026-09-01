"""Compatibility evaluation entry point used by :mod:`clean.train`.

The original training script imported ``run_eval`` from a repository-level
experiment helper.  This release-local implementation keeps that small API
while using the packaged codec, raster and checkpoint helpers.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

if __package__:
    from .hypericm.checkpoints import load_codec_checkpoint, unwrap_model
    from .hypericm.codec import entropy_coded_inference
    from .hypericm.raster import normalize_minmax, read_chw
    from .model import DMC, DMCI
else:  # pragma: no cover - direct execution from the release directory
    from hypericm.checkpoints import load_codec_checkpoint, unwrap_model
    from hypericm.codec import entropy_coded_inference
    from hypericm.raster import normalize_minmax, read_chw
    from model import DMC, DMCI


ROOT = Path(__file__).resolve().parent


def _as_device(value: str | torch.device) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for evaluation, but CUDA is unavailable.")
        if device.index is None:
            device = torch.device("cuda:0")
        index = 0 if device.index is None else device.index
        if index >= torch.cuda.device_count():
            raise ValueError(f"Unavailable CUDA device: {device}")
    return device


def run_eval(
    date_str: str | None,
    model_path: str | Path,
    gpus: str | torch.device,
    test_img_path: str | Path,
    metric_save_path: str | Path = "metrics.csv",
    gop: int = 8,
    qps: Iterable[int] = (21,),
    save_root: str | Path | None = None,
    models=None,
):
    """Compress one full image and return ``(run_dir, bpsp, psnr)``.

    Reconstructions intentionally use the legacy ``<stem>_<qp>.npy`` naming
    expected by ``HyperspectralTester``.
    """

    checkpoint_path = Path(model_path)
    image_path = Path(test_img_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing codec checkpoint: {checkpoint_path}")
    if not image_path.is_file():
        raise FileNotFoundError(f"Missing evaluation image: {image_path}")
    if gop < 1:
        raise ValueError("gop must be at least 1.")

    qp_values = [int(qp) for qp in qps]
    if not qp_values:
        raise ValueError("qps must contain at least one QP value.")
    invalid_qps = [qp for qp in qp_values if not 1 <= qp <= 63]
    if invalid_qps:
        raise ValueError(f"Evaluation QP values must be in [1, 63]: {invalid_qps}")

    device = _as_device(gpus)
    image, _, _ = normalize_minmax(read_chw(image_path))
    if models is None:
        i_model = DMCI().to(device)
        p_model = DMC().to(device)
        load_codec_checkpoint(checkpoint_path, i_model, p_model, strict=True)
    else:
        if len(models) != 2:
            raise ValueError("models must be an (i_model, p_model) pair.")
        i_model, p_model = (unwrap_model(model) for model in models)
        model_device = next(i_model.parameters()).device
        if model_device != device:
            raise ValueError(
                f"Provided codec models are on {model_device}, but evaluation requested {device}."
            )
    i_model.eval()
    p_model.eval()

    root = Path(save_root) if save_root is not None else ROOT / "results"
    run_dir = root / str(date_str or "evaluation") / image_path.stem
    run_dir.mkdir(parents=True, exist_ok=True)
    metric_path = Path(metric_save_path)
    if not metric_path.is_absolute():
        metric_path = run_dir / metric_path.name
    metric_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    last_metrics = None
    with torch.no_grad():
        for qp in qp_values:
            reconstruction, metrics = entropy_coded_inference(
                i_model,
                p_model,
                image,
                qp,
                device,
                group_size=gop,
            )
            np.save(run_dir / f"{image_path.stem}_{qp}.npy", reconstruction)
            rows.append(metrics)
            last_metrics = metrics
            print(
                f"Evaluation QP={qp}: bpsp={metrics['bpsp']:.6f}, "
                f"psnr={metrics['psnr']:.4f}"
            )

    with metric_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("qp", "bpsp", "bits", "mse", "psnr"))
        writer.writeheader()
        writer.writerows(rows)

    assert last_metrics is not None
    return str(run_dir), float(last_metrics["bpsp"]), float(last_metrics["psnr"])


__all__ = ["run_eval"]
