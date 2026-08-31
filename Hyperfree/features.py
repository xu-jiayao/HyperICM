"""Feature extraction wrapper used by HyperICM online adaptation."""

from pathlib import Path

import torch

from .build_HyperFree import HyperFree_model_registry
from .automatic_mask_generator import SamAutomaticMaskGenerator


class HyperfreeFeatureExtractor:
    def __init__(
        self,
        checkpoint: str | Path,
        device: str | torch.device,
        feature_indices=(0, 1, 2, 3, 4),
    ):
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        self.device = torch.device(device)
        self.feature_indices = tuple(feature_indices)
        if not self.feature_indices or min(self.feature_indices) < 0:
            raise ValueError("feature_indices must contain non-negative stage indices.")
        self.model = HyperFree_model_registry["vit_b"](
            checkpoint=str(checkpoint),
            encoder_global_attn_indexes=-1,
            merge_indexs=None,
        ).to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.model,
            pred_iou_thresh=0.3,
            stability_score_thresh=0.4,
            points_per_side=32,
        )

    def extract(self, image_chw: torch.Tensor, gsd: float, wavelengths: list[float]):
        if image_chw.ndim != 3:
            raise ValueError(f"Expected CHW input, got shape {tuple(image_chw.shape)}")
        if image_chw.shape[0] != len(wavelengths):
            raise ValueError(
                f"Image has {image_chw.shape[0]} bands but {len(wavelengths)} wavelengths were supplied."
            )
        image = image_chw.to(self.device)
        gsd_tensor = torch.tensor([gsd], dtype=torch.float32, device=self.device)
        self.mask_generator.predictor.set_image(
            image.permute(1, 2, 0),
            True,
            wavelengths,
            GSD=gsd_tensor,
        )
        features = self.mask_generator.predictor.model.image_encoder.multi_stage_features
        if max(self.feature_indices) >= len(features):
            raise IndexError(
                f"Requested feature index {max(self.feature_indices)}, but Hyperfree returned {len(features)} stages."
            )
        return [features[index] for index in self.feature_indices]
