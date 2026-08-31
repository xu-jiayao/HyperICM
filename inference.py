"""Entropy-coded HyperICM inference with optional Hyperfree classification."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

if __package__:  # ``python -m clean.inference``
    from .hypericm.checkpoints import load_codec_checkpoint
    from .hypericm.codec import entropy_coded_inference
    from .hypericm.datasets import DATASET_NAMES, get_dataset_spec
    from .hypericm.raster import normalize_minmax, read_chw, read_label, validate_label_reference, write_label
    from .model import DMC, DMCI
else:  # ``python inference.py`` from the release directory
    from hypericm.checkpoints import load_codec_checkpoint
    from hypericm.codec import entropy_coded_inference
    from hypericm.datasets import DATASET_NAMES, get_dataset_spec
    from hypericm.raster import normalize_minmax, read_chw, read_label, validate_label_reference, write_label
    from model import DMC, DMCI


ROOT = Path(__file__).resolve().parent
DEFAULT_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description="HyperICM compression and downstream inference")
    parser.add_argument("--input", type=Path, required=True, help="Input hyperspectral TIFF")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Base or online-adapted HyperICM checkpoint")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--qps", type=int, nargs="+", default=[21])
    parser.add_argument("--device", default=DEFAULT_DEVICE)

    parser.add_argument("--downstream", action="store_true", help="Run Hyperfree few-shot classification")
    parser.add_argument("--dataset", choices=DATASET_NAMES)
    parser.add_argument("--ground-truth", type=Path)
    parser.add_argument(
        "--hyperfree-checkpoint",
        type=Path,
        default=ROOT / "checkpoints" / "HyperFree-b.pth",
    )
    parser.add_argument("--hyperfree-device", default=DEFAULT_DEVICE)
    parser.add_argument("--feature-index", type=int, choices=range(5), default=1)
    parser.add_argument("--pred-iou-thresh", type=float, default=0.2)
    parser.add_argument("--stability-score-thresh", type=float, default=0.3)
    parser.add_argument("--open-set-threshold", type=float, default=0.0)
    args = parser.parse_args()

    invalid_qps = [qp for qp in args.qps if not 1 <= qp <= 63]
    if invalid_qps:
        parser.error(f"Inference QP values must be in [1, 63]: {invalid_qps}")
    if args.downstream and (args.dataset is None or args.ground_truth is None):
        parser.error("--downstream requires --dataset and --ground-truth")
    for name, value in (
        ("--pred-iou-thresh", args.pred_iou_thresh),
        ("--stability-score-thresh", args.stability_score_thresh),
    ):
        if not 0.0 <= value <= 1.0:
            parser.error(f"{name} must be in [0, 1]")
    if not -1.0 <= args.open_set_threshold <= 1.0:
        parser.error("--open-set-threshold must be in [-1, 1]")
    return args


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but CUDA is unavailable.")
    if device.type == "cuda" and device.index is not None and device.index >= torch.cuda.device_count():
        raise ValueError(f"Unavailable CUDA device: {device}")
    if not args.input.is_file():
        raise FileNotFoundError(f"Missing input raster: {args.input}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Missing codec checkpoint: {args.checkpoint}")
    if args.downstream:
        if not args.hyperfree_checkpoint.is_file():
            raise FileNotFoundError(f"Missing Hyperfree checkpoint: {args.hyperfree_checkpoint}")
        validate_label_reference(args.ground_truth, args.input)

    image, input_min, input_max = normalize_minmax(read_chw(args.input))
    if args.dataset:
        spec = get_dataset_spec(args.dataset)
        if image.shape[0] != spec.bands:
            raise ValueError(f"Input has {image.shape[0]} bands; {spec.name} expects {spec.bands}.")
    else:
        spec = None

    i_model = DMCI().to(device)
    p_model = DMC().to(device)
    checkpoint = load_codec_checkpoint(args.checkpoint, i_model, p_model, strict=True)
    i_model.eval()
    p_model.eval()

    hyperfree_model = None
    classifier = None
    label = None
    if args.downstream:
        hyperfree_device = torch.device(args.hyperfree_device)
        if hyperfree_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("A CUDA Hyperfree device was requested, but CUDA is unavailable.")
        if (
            hyperfree_device.type == "cuda"
            and hyperfree_device.index is not None
            and hyperfree_device.index >= torch.cuda.device_count()
        ):
            raise ValueError(f"Unavailable Hyperfree CUDA device: {hyperfree_device}")
        if __package__:
            from .Hyperfree.downstream import build_classifier, classification_metrics, classify
        else:
            from Hyperfree.downstream import build_classifier, classification_metrics, classify

        hyperfree_model, classifier = build_classifier(
            args.hyperfree_checkpoint,
            hyperfree_device,
            pred_iou_thresh=args.pred_iou_thresh,
            stability_score_thresh=args.stability_score_thresh,
        )
        label = read_label(args.ground_truth)

    run_dir = args.output_dir / args.input.stem
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for qp in args.qps:
        print(f"Running QP={qp}")
        reconstruction, metrics = entropy_coded_inference(i_model, p_model, image, qp, device)
        reconstruction_path = run_dir / f"{args.input.stem}_qp{qp}.npy"
        np.save(reconstruction_path, reconstruction)

        if args.downstream:
            prediction = classify(
                classifier,
                np.transpose(np.clip(reconstruction, 0.0, 1.0), (1, 2, 0)),
                spec,
                feature_index=args.feature_index,
                open_set_threshold=args.open_set_threshold,
            )
            metrics.update(classification_metrics(prediction, label))
            write_label(
                run_dir / f"{args.input.stem}_qp{qp}_classification.tif",
                prediction,
                args.ground_truth,
            )

        rows.append(metrics)
        summary = ", ".join(f"{key}={value:.6f}" for key, value in metrics.items() if key != "bits")
        print(summary)

    fieldnames = ["qp", "bpsp", "bits", "mse", "psnr", "oa", "aa", "kappa"]
    with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    metadata = {
        "input": str(args.input.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "normalization": {"minimum": input_min, "maximum": input_max},
        "qps": args.qps,
        "downstream": args.downstream,
        "dataset": args.dataset,
    }
    with (run_dir / "run.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    del hyperfree_model


if __name__ == "__main__":
    main()
