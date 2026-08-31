"""Patch dataset for attachment-derived online encoder adaptation."""

from pathlib import Path

import torch
from torch.utils.data import Dataset

from .datasets import DatasetSpec
from .raster import normalize_minmax, read_chw


class HyperspectralPatchDataset(Dataset):
    def __init__(self, root: str | Path, spec: DatasetSpec, feature_extractor, expected_size: int | None = None):
        self.root = Path(root)
        candidates = []
        for pattern in ("*.tif", "*.tiff", "*.TIF", "*.TIFF"):
            candidates.extend(self.root.glob(pattern))
        self.files = sorted(set(candidates))
        if not self.files:
            raise FileNotFoundError(f"No TIFF patches found in {self.root}")
        self.spec = spec
        self.feature_extractor = feature_extractor
        self.expected_size = expected_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        image, _, _ = normalize_minmax(read_chw(path))
        if image.shape[0] != self.spec.bands:
            raise ValueError(
                f"{path} has {image.shape[0]} bands; {self.spec.name} expects {self.spec.bands}."
            )
        if self.expected_size is not None and image.shape[-2:] != (self.expected_size, self.expected_size):
            raise ValueError(
                f"{path} has spatial size {image.shape[-2:]}; expected "
                f"{self.expected_size}x{self.expected_size}."
            )
        tensor = torch.from_numpy(image)
        height, width = tensor.shape[-2:]
        ratio = 1024.0 / max(height, width)
        gsd = self.spec.base_gsd / ratio
        with torch.no_grad():
            features = self.feature_extractor.extract(tensor, gsd, self.spec.wavelengths)
        features = [feature.squeeze(0).detach().cpu() for feature in features]
        return tensor, features, float(gsd), str(path)
