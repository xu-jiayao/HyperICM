"""Hyperfree few-shot classification used for downstream evaluation."""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, cohen_kappa_score, recall_score

if __package__ and __package__.startswith("clean."):
    from ..hypericm.datasets import DatasetSpec
else:  # pragma: no cover - direct release-directory execution
    from hypericm.datasets import DatasetSpec

from . import HyperFree_model_registry, SamAutomaticMaskGenerator


def build_classifier(checkpoint: str | Path, device, pred_iou_thresh=0.2, stability_score_thresh=0.3):
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    model = HyperFree_model_registry["vit_b"](
        checkpoint=str(checkpoint),
        encoder_global_attn_indexes=-1,
        merge_indexs=None,
    ).to(device)
    model.eval()
    generator = SamAutomaticMaskGenerator(
        model=model,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        points_per_side=32,
    )
    return model, generator


def _few_shot_maps(spec: DatasetSpec, height: int, width: int):
    class_count = max(point[0] for point in spec.few_shot_points) + 1
    maps = [np.zeros((height, width), dtype=np.uint8) for _ in range(class_count)]
    for class_index, row, column in spec.few_shot_points:
        if not (0 <= row < height and 0 <= column < width):
            raise ValueError(
                f"Few-shot point ({row}, {column}) for class {class_index + 1} is outside {height}x{width}."
            )
        maps[class_index][row, column] = 1
    return maps


def _cosine(first: np.ndarray, second: np.ndarray) -> float:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator == 0:
        return -1.0
    return float(np.dot(first, second) / denominator)


@torch.no_grad()
def classify(
    generator,
    image_hwc: np.ndarray,
    spec: DatasetSpec,
    feature_index: int = 1,
    open_set_threshold: float = 0.0,
):
    if not -1.0 <= open_set_threshold <= 1.0:
        raise ValueError("open_set_threshold must be in [-1, 1].")
    height, width, channels = image_hwc.shape
    if channels != spec.bands:
        raise ValueError(f"Reconstruction has {channels} bands; {spec.name} expects {spec.bands}.")
    ratio = 1024.0 / max(height, width)
    gsd = torch.tensor([spec.base_gsd / ratio], dtype=torch.float32, device=generator.predictor.device)

    annotations = generator.generate(image_hwc, spec.wavelengths, gsd)
    masks = generator.anns2mask(annotations)
    if masks is None or masks.shape[0] <= 1:
        raise RuntimeError("Hyperfree did not generate usable masks.")

    generator.predictor.set_image(image_hwc, True, spec.wavelengths, gsd)
    stages = generator.predictor.model.image_encoder.multi_stage_features
    if feature_index < 0 or feature_index >= len(stages):
        raise IndexError(f"feature_index={feature_index}, available stages={len(stages)}")
    features = stages[feature_index].detach()
    features = F.interpolate(features, (max(height, width), max(height, width)), mode="nearest")
    features = features[:, :, :height, :width].cpu()

    prototypes = []
    for seed_map in _few_shot_maps(spec, height, width):
        class_prototypes = []
        seed_rows, seed_columns = np.where(seed_map == 1)
        for row, column in zip(seed_rows, seed_columns):
            covering = np.where(masks[:, row, column] == 1)[0]
            if covering.size == 0:
                continue
            selected = masks[covering[0]].astype(bool)
            rows, columns = np.where(selected)
            if rows.size == 0:
                continue
            prototype = features[0, :, rows, columns].mean(dim=1).numpy()
            class_prototypes.append(prototype)
        if not class_prototypes:
            raise RuntimeError(
                "Hyperfree generated no mask covering the few-shot point for "
                f"class {len(prototypes) + 1}."
            )
        prototypes.append(class_prototypes)

    prediction = np.zeros((height, width), dtype=np.uint8)
    for segmentation in reversed(masks[:-1]):
        rows, columns = np.where(segmentation == 1)
        if rows.size == 0:
            continue
        segment_feature = features[0, :, rows, columns].mean(dim=1).numpy()
        best_class = None
        best_score = -1.0
        for class_index, class_prototypes in enumerate(prototypes):
            for prototype in class_prototypes:
                score = _cosine(segment_feature, prototype)
                if score > best_score:
                    best_score = score
                    best_class = class_index
        if best_class is not None and best_score > open_set_threshold:
            prediction[rows, columns] = best_class + 1
    return prediction


def classification_metrics(prediction: np.ndarray, label: np.ndarray):
    if prediction.shape != label.shape:
        raise ValueError(
            f"Prediction and ground truth must have the same shape, got {prediction.shape} and {label.shape}."
        )
    valid = label != 0
    if not np.any(valid):
        raise ValueError("Ground-truth label contains no non-background pixels.")
    y_true = label[valid].reshape(-1)
    y_pred = prediction[valid].reshape(-1)
    labels = np.unique(y_true)
    return {
        "oa": float(accuracy_score(y_true, y_pred)),
        "aa": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
    }
