"""Online encoder adaptation for HyperICM.

This release entry point is based on the user-provided attachment. It keeps
the attachment's fixed eight-band grouping, spatial padding, encoder-only
optimization, and one optimizer step per epoch while removing machine-specific
paths and GPU assumptions.
"""

import argparse
import csv
import math
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

if __package__:  # ``python -m clean.train``
    from .Hyperfree import SamAutomaticMaskGenerator
    from .Hyperfree.downstream import classification_metrics, classify
    from .Hyperfree.features import HyperfreeFeatureExtractor
    from .hypericm.checkpoints import checkpoint_state, load_codec_checkpoint, unwrap_model
    from .hypericm.codec import differentiable_reconstruction, entropy_coded_inference
    from .hypericm.datasets import DATASET_NAMES, get_dataset_spec
    from .hypericm.patches import HyperspectralPatchDataset
    from .hypericm.raster import normalize_minmax, read_chw, read_label, validate_label_reference, write_label
    from .model import DMC, DMCI
else:  # ``python train.py`` from the release directory
    from Hyperfree import SamAutomaticMaskGenerator
    from Hyperfree.downstream import classification_metrics, classify
    from Hyperfree.features import HyperfreeFeatureExtractor
    from hypericm.checkpoints import checkpoint_state, load_codec_checkpoint, unwrap_model
    from hypericm.codec import differentiable_reconstruction, entropy_coded_inference
    from hypericm.datasets import DATASET_NAMES, get_dataset_spec
    from hypericm.patches import HyperspectralPatchDataset
    from hypericm.raster import normalize_minmax, read_chw, read_label, validate_label_reference, write_label
    from model import DMC, DMCI


ROOT = Path(__file__).resolve().parent
DEFAULT_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def build_parser(defaults=None, config_dir=None):
    parser = argparse.ArgumentParser(description="HyperICM online encoder adaptation")
    parser.add_argument("--config", type=Path, help="Optional YAML file with argument defaults")
    parser.add_argument("--dataset", choices=DATASET_NAMES)
    parser.add_argument("--patch-dir", type=Path, help="Directory containing TIFF patches")
    parser.add_argument("--patch-size", type=int, default=256, help="Expected patch side length (metadata/check only)")
    parser.add_argument(
        "--base-checkpoint", "--checkpoint",
        type=Path,
        default=ROOT / "checkpoints" / "Ep10-dev0.076-0.000870-qprandom.pth",
    )
    parser.add_argument(
        "--hyperfree-checkpoint",
        type=Path,
        default=ROOT / "checkpoints" / "HyperFree-b.pth",
    )
    parser.add_argument(
        "--output-dir", "--exp-name", dest="output_dir",
        type=Path, default=ROOT / "outputs" / "online_adaptation"
    )
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="Primary compression-model device")
    parser.add_argument("--hyperfree-device", default=DEFAULT_DEVICE)
    parser.add_argument(
        "--data-parallel-devices",
        nargs="*",
        type=int,
        default=None,
        help="Optional CUDA device indices, e.g. 0 1 2",
    )
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=1)
    parser.add_argument("--qp", type=int, default=21, help="1-63; use 0 to sample one QP per batch")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", "--lr", dest="learning_rate", type=float, default=1e-4)
    parser.add_argument(
        "--loss-weights",
        "--para",
        dest="loss_weights",
        type=float,
        nargs=4,
        metavar=("SEMANTIC", "MSE", "RATE", "SCALE"),
        default=(0.5, 0.06, 5.0, 10.0),
        help="Attachment-compatible [semantic, MSE, rate, global scale] weights",
    )
    parser.add_argument("--seed", type=int, default=28)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--full-image", "--full_image", dest="full_image", type=Path, help="Optional full-resolution TIFF for periodic codec evaluation")
    parser.add_argument("--ground-truth", "--ground_truth", dest="ground_truth", type=Path, help="Optional label TIFF for periodic downstream evaluation")
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--run-downstream", action="store_true", default=False)
    parser.add_argument("--feature-index", type=int, choices=range(5), default=1)
    parser.add_argument("--pred-iou-thresh", type=float, default=0.2)
    parser.add_argument("--stability-score-thresh", type=float, default=0.3)
    if defaults:
        # Keep the public YAML spelling compatible with the command-line API.
        aliases = {
            "lr": "learning_rate",
            "para": "loss_weights",
            "exp_name": "output_dir",
            "model_path": "base_checkpoint",
            "checkpoint": "base_checkpoint",
        }
        normalized = {}
        for key, value in defaults.items():
            target = aliases.get(key, key)
            if target is not None:
                normalized[target] = value
        config_dir = Path.cwd() if config_dir is None else Path(config_dir).resolve()
        for key in ("base_checkpoint", "hyperfree_checkpoint", "output_dir", "full_image", "ground_truth", "patch_dir"):
            if key in normalized and normalized[key] is not None:
                value = Path(normalized[key])
                normalized[key] = value if value.is_absolute() else config_dir / value
        for key in ("batch_size", "qp", "epochs", "seed", "save_every", "eval_every", "feature_index", "patch_size"):
            if key in normalized and normalized[key] is not None:
                value = normalized[key]
                if isinstance(value, bool) or (isinstance(value, float) and not value.is_integer()):
                    raise ValueError(f"{key} must be an integer in the YAML config.")
                normalized[key] = int(value)
        for key in ("learning_rate", "pred_iou_thresh", "stability_score_thresh"):
            if key in normalized and normalized[key] is not None:
                normalized[key] = float(normalized[key])
                if not math.isfinite(normalized[key]):
                    raise ValueError(f"{key} must be finite in the YAML config.")
        if "loss_weights" in normalized:
            normalized["loss_weights"] = tuple(float(value) for value in normalized["loss_weights"])
            if not all(math.isfinite(value) for value in normalized["loss_weights"]):
                raise ValueError("loss_weights must contain finite values.")
        if "data_parallel_devices" in normalized and normalized["data_parallel_devices"] is not None:
            devices = normalized["data_parallel_devices"]
            if not isinstance(devices, (list, tuple)):
                raise ValueError("data_parallel_devices must be a list of integer CUDA indices.")
            normalized_devices = []
            for value in devices:
                if isinstance(value, bool) or (isinstance(value, float) and not value.is_integer()):
                    raise ValueError("data_parallel_devices must contain integer CUDA indices.")
                normalized_devices.append(int(value))
            normalized["data_parallel_devices"] = normalized_devices
        if "run_downstream" in normalized:
            value = normalized["run_downstream"]
            if isinstance(value, str):
                text_value = value.strip().lower()
                if text_value not in {"0", "1", "true", "false", "yes", "no", "on", "off"}:
                    raise ValueError("run_downstream must be a boolean in the YAML config.")
                normalized["run_downstream"] = text_value in {"1", "true", "yes", "on"}
            elif isinstance(value, bool):
                normalized["run_downstream"] = value
            else:
                raise ValueError("run_downstream must be a boolean in the YAML config.")
        if "device" in normalized:
            normalized["device"] = str(normalized["device"])
        if "hyperfree_device" in normalized:
            normalized["hyperfree_device"] = str(normalized["hyperfree_device"])
        unknown = sorted(set(normalized) - {action.dest for action in parser._actions})
        if unknown:
            raise ValueError(f"Unknown training config keys: {unknown}")
        parser.set_defaults(**normalized)
    return parser


def parse_args():
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=Path)
    known, _ = bootstrap.parse_known_args()
    defaults = {}
    if known.config:
        with known.config.open("r", encoding="utf-8") as handle:
            defaults = yaml.safe_load(handle) or {}
        if not isinstance(defaults, dict):
            raise TypeError("Training config must contain a YAML mapping.")
    config_dir = known.config.parent if known.config else None
    args = build_parser(defaults, config_dir=config_dir).parse_args()
    if args.dataset is None:
        raise SystemExit("--dataset is required (or set dataset in --config).")
    if args.dataset not in DATASET_NAMES:
        raise SystemExit(f"Unsupported dataset {args.dataset!r}; choose one of {DATASET_NAMES}.")
    if args.patch_dir is None:
        raise SystemExit("--patch-dir is required (or set patch_dir in --config).")
    return args


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def validate_args(args):
    if not 0 <= args.qp <= 63:
        raise ValueError("--qp must be in [0, 63].")
    if args.batch_size < 1 or args.epochs < 1 or args.patch_size < 1:
        raise ValueError("--batch-size, --epochs, and --patch-size must be positive.")
    if args.eval_every < 1 or args.save_every < 1:
        raise ValueError("--eval-every and --save-every must be positive.")
    if len(args.loss_weights) != 4:
        raise ValueError("--loss-weights/para must contain four values.")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("--learning-rate must be finite and positive.")
    if not all(math.isfinite(value) for value in args.loss_weights):
        raise ValueError("--loss-weights must contain finite values.")
    if not 0 <= args.feature_index <= 4:
        raise ValueError("--feature-index must be in [0, 4].")
    for name, value in (
        ("--pred-iou-thresh", args.pred_iou_thresh),
        ("--stability-score-thresh", args.stability_score_thresh),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1].")
    if args.run_downstream and (args.full_image is None or args.ground_truth is None):
        raise ValueError("--run-downstream requires --full-image and --ground-truth.")
    compression_device = torch.device(args.device)
    hyperfree_device = torch.device(args.hyperfree_device)
    if (compression_device.type == "cuda" or hyperfree_device.type == "cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA devices were requested, but CUDA is unavailable.")
    if args.data_parallel_devices:
        if compression_device.type != "cuda":
            raise ValueError("DataParallel requires a CUDA --device.")
        if compression_device.index is not None and compression_device.index != args.data_parallel_devices[0]:
            raise ValueError("The first --data-parallel-devices entry must match --device.")
        if any(index < 0 for index in args.data_parallel_devices):
            raise ValueError("CUDA device indices must be non-negative.")
        if len(set(args.data_parallel_devices)) != len(args.data_parallel_devices):
            raise ValueError("CUDA device indices must be unique.")
        if compression_device.index is None:
            compression_device = torch.device(f"cuda:{args.data_parallel_devices[0]}")
        invalid = [index for index in args.data_parallel_devices if index >= torch.cuda.device_count()]
        if invalid:
            raise ValueError(f"Unavailable CUDA device indices: {invalid}")
    for label, device in (("compression", compression_device), ("Hyperfree", hyperfree_device)):
        if device.type == "cuda" and device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(f"Unavailable {label} CUDA device: {device}")
    return compression_device, hyperfree_device


def set_encoder_only(i_model, p_model):
    for parameter in i_model.parameters():
        parameter.requires_grad_(False)
    for name, parameter in i_model.named_parameters():
        if name == "enc" or name.startswith("enc."):
            parameter.requires_grad_(True)

    for parameter in p_model.parameters():
        parameter.requires_grad_(False)
    for name, parameter in p_model.named_parameters():
        if name == "encoder" or name.startswith("encoder."):
            parameter.requires_grad_(True)


def extracted_features(extractor, images, gsds, wavelengths):
    per_sample = []
    for index in range(images.shape[0]):
        per_sample.append(extractor.extract(images[index], float(gsds[index]), wavelengths))
    return [
        torch.cat([sample[stage] for sample in per_sample], dim=0)
        for stage in range(len(per_sample[0]))
    ]


def run_epoch(args, i_model, p_model, loader, extractor, qp_rng, optimizer=None):
    training = optimizer is not None
    i_model.train(training)
    p_model.train(training)
    if training:
        optimizer.zero_grad(set_to_none=True)

    spec = get_dataset_spec(args.dataset)
    semantic_weight, mse_weight, rate_weight, global_scale = args.loss_weights
    totals = {"loss": 0.0, "bpp": 0.0, "mse": 0.0, "feature_loss": 0.0, "qp": 0.0}
    description = "Training" if training else "Validation"
    progress = tqdm(loader, desc=description, dynamic_ncols=True)

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, target_features, gsds, _ in progress:
            images = images.to(args.compression_device, non_blocking=True)
            current_qp = qp_rng.randint(1, 63) if args.qp == 0 else args.qp
            reconstruction, rate_loss, distortion_loss = differentiable_reconstruction(
                i_model, p_model, images, current_qp
            )
            reconstructed_features = extracted_features(
                extractor, reconstruction, gsds, spec.wavelengths
            )
            target_features = [feature.to(extractor.device) for feature in target_features]
            feature_loss = torch.stack(
                [F.mse_loss(current, target) for current, target in zip(reconstructed_features, target_features)]
            ).mean()
            total_loss = global_scale * (
                rate_weight * rate_loss
                + mse_weight * distortion_loss
                + semantic_weight * feature_loss.to(args.compression_device)
            )
            if training:
                (total_loss / len(loader)).backward()

            totals["loss"] += float(total_loss.detach())
            totals["bpp"] += float(rate_loss.detach())
            totals["mse"] += float(distortion_loss.detach())
            totals["feature_loss"] += float(feature_loss.detach())
            totals["qp"] += current_qp
            count = progress.n + 1
            progress.set_postfix(
                loss=f"{totals['loss'] / count:.4f}",
                bpp=f"{totals['bpp'] / count:.4f}",
                feature=f"{totals['feature_loss'] / count:.4f}",
                qp=current_qp,
            )

    if training:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return {key: value / len(loader) for key, value in totals.items()}


def periodic_full_image_eval(args, i_model, p_model, extractor, downstream_generator, epoch_dir, qp_rng):
    spec = get_dataset_spec(args.dataset)
    image, _, _ = normalize_minmax(read_chw(args.full_image))
    if image.shape[0] != spec.bands:
        raise ValueError(f"Full image has {image.shape[0]} bands; {spec.name} expects {spec.bands}.")
    evaluation_qp = qp_rng.randint(1, 63) if args.qp == 0 else args.qp
    reconstruction, metrics = entropy_coded_inference(
        unwrap_model(i_model), unwrap_model(p_model), image, evaluation_qp, args.compression_device
    )
    epoch_dir.mkdir(parents=True, exist_ok=True)
    np.save(epoch_dir / f"{args.dataset}_{evaluation_qp}.npy", reconstruction)

    if args.run_downstream:
        prediction = classify(
            downstream_generator,
            np.transpose(np.clip(reconstruction, 0.0, 1.0), (1, 2, 0)),
            spec,
            feature_index=args.feature_index,
        )
        label = read_label(args.ground_truth)
        metrics.update(classification_metrics(prediction, label))
        write_label(
            epoch_dir / f"{args.dataset}_{evaluation_qp}_classification.tif",
            prediction,
            args.ground_truth,
        )

    with (epoch_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics))
        writer.writeheader()
        writer.writerow(metrics)
    return metrics


def main():
    args = parse_args()
    compression_device, hyperfree_device = validate_args(args)
    args.compression_device = compression_device
    for path, label in (
        (args.base_checkpoint, "base checkpoint"),
        (args.hyperfree_checkpoint, "Hyperfree checkpoint"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")
    if not args.patch_dir.is_dir():
        raise NotADirectoryError(args.patch_dir)
    if args.full_image is not None and not args.full_image.is_file():
        raise FileNotFoundError(f"Missing full image: {args.full_image}")
    if args.ground_truth is not None and not args.ground_truth.is_file():
        raise FileNotFoundError(f"Missing ground truth: {args.ground_truth}")
    if args.run_downstream and args.full_image is not None and args.ground_truth is not None:
        validate_label_reference(args.ground_truth, args.full_image)
    seed_everything(args.seed)
    train_qp_rng = random.Random(args.seed + 101)
    validation_qp_rng = random.Random(args.seed + 202)
    full_eval_qp_rng = random.Random(args.seed + 303)

    spec = get_dataset_spec(args.dataset)
    extractor = HyperfreeFeatureExtractor(args.hyperfree_checkpoint, hyperfree_device)
    dataset = HyperspectralPatchDataset(args.patch_dir, spec, extractor, expected_size=args.patch_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=compression_device.type == "cuda",
    )

    i_model = DMCI()
    p_model = DMC()
    base_checkpoint = load_codec_checkpoint(args.base_checkpoint, i_model, p_model, strict=True)
    i_model.to(compression_device)
    p_model.to(compression_device)
    set_encoder_only(i_model, p_model)

    if args.data_parallel_devices and len(args.data_parallel_devices) > 1:
        i_model = torch.nn.DataParallel(i_model, device_ids=args.data_parallel_devices)
        p_model = torch.nn.DataParallel(p_model, device_ids=args.data_parallel_devices)

    trainable = [parameter for model in (i_model, p_model) for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.98)

    run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = args.output_dir / f"{args.dataset}-qp{args.qp}-{run_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(output_dir / "tensorboard")

    downstream_generator = None
    if args.run_downstream:
        downstream_generator = SamAutomaticMaskGenerator(
            model=extractor.model,
            pred_iou_thresh=args.pred_iou_thresh,
            stability_score_thresh=args.stability_score_thresh,
            points_per_side=32,
        )

    print(f"Loaded base checkpoint epoch: {base_checkpoint.get('epoch', 'unknown')}")
    print(f"Training patches: {len(dataset)}")
    print(f"Output directory: {output_dir}")

    try:
        for epoch in range(args.epochs):
            train_metrics = run_epoch(args, i_model, p_model, loader, extractor, train_qp_rng, optimizer)
            validation_metrics = run_epoch(args, i_model, p_model, loader, extractor, validation_qp_rng)
            scheduler.step()

            for key, value in train_metrics.items():
                writer.add_scalar(f"train/{key}", value, epoch)
            for key, value in validation_metrics.items():
                writer.add_scalar(f"validation/{key}", value, epoch)

            metrics = {
                "bpp": validation_metrics["bpp"],
                "mse": validation_metrics["mse"],
                "feature_loss": validation_metrics["feature_loss"],
                "qp": validation_metrics["qp"],
            }
            if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs:
                filename = (
                    f"epoch-{epoch + 1:03d}-bpp-{metrics['bpp']:.6f}"
                    f"-mse-{metrics['mse']:.6f}-feature-{metrics['feature_loss']:.6f}.pth"
                )
                torch.save(
                    checkpoint_state(i_model, p_model, epoch + 1, metrics, optimizer, scheduler),
                    output_dir / filename,
                )

            if args.full_image and ((epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs):
                full_metrics = periodic_full_image_eval(
                    args,
                    i_model,
                    p_model,
                    extractor,
                    downstream_generator,
                    output_dir / f"evaluation_epoch_{epoch + 1:03d}",
                    full_eval_qp_rng,
                )
                for key, value in full_metrics.items():
                    writer.add_scalar(f"full_image/{key}", value, epoch)
    finally:
        writer.close()


if __name__ == "__main__":
    main()
