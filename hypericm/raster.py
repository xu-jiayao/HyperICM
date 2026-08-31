"""Raster I/O and normalization helpers."""

from pathlib import Path

import numpy as np
import rasterio


def raster_shape(path: str | Path) -> tuple[int, int, int]:
    """Return ``(bands, height, width)`` without loading raster pixels."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with rasterio.open(path) as dataset:
        return dataset.count, dataset.height, dataset.width


def read_chw(path: str | Path, dtype=np.float32) -> np.ndarray:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with rasterio.open(path) as dataset:
        data = dataset.read(masked=True)
    mask = getattr(data, "mask", np.ma.nomask)
    if mask is not np.ma.nomask and bool(np.asarray(mask).any()):
        raise ValueError(f"Raster contains nodata or masked pixels: {path}")
    data = np.asarray(data)
    if not np.isfinite(data).all():
        raise ValueError(f"Raster contains non-finite values: {path}")
    return data.astype(dtype, copy=False)


def read_label(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with rasterio.open(path) as dataset:
        if dataset.count != 1:
            raise ValueError(f"Expected a single-band label raster, got {dataset.count} bands.")
        data = dataset.read(1, masked=True)
    data = np.asarray(data.filled(0) if np.ma.isMaskedArray(data) else data)
    if not np.isfinite(data).all():
        raise ValueError(f"Label raster contains non-finite values: {path}")
    if np.issubdtype(data.dtype, np.floating) and not np.equal(data, np.floor(data)).all():
        raise ValueError(f"Label raster contains non-integer class values: {path}")
    if np.any(data < 0):
        raise ValueError(f"Label raster contains negative class values: {path}")
    return data.astype(np.int64, copy=False)


def validate_label_reference(label_path: str | Path, image_path: str | Path) -> None:
    """Fail early when a label raster cannot be aligned with an input image."""
    with rasterio.open(label_path) as label_src, rasterio.open(image_path) as image_src:
        if label_src.count != 1:
            raise ValueError(f"Expected one label band, got {label_src.count}.")
        if (label_src.height, label_src.width) != (image_src.height, image_src.width):
            raise ValueError("Label and image rasters must have matching height and width.")
        if label_src.crs != image_src.crs or label_src.transform != image_src.transform:
            raise ValueError("Label and image rasters must have matching CRS and affine transform.")


def normalize_minmax(data: np.ndarray) -> tuple[np.ndarray, float, float]:
    mask = getattr(data, "mask", np.ma.nomask)
    if mask is not np.ma.nomask and bool(np.asarray(mask).any()):
        raise ValueError("Cannot normalize a raster containing masked pixels.")
    data = np.asarray(data)
    if data.size == 0 or not np.isfinite(data).all():
        raise ValueError("Raster values must be finite and non-empty.")
    minimum = float(data.min())
    maximum = float(data.max())
    scale = maximum - minimum
    if scale <= 0:
        raise ValueError("The raster is constant and cannot be min-max normalized.")
    normalized = (data.astype(np.float32) - minimum) / scale
    return normalized, minimum, maximum


def write_label(path: str | Path, label: np.ndarray, reference: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(reference) as source:
        profile = source.profile.copy()
        reference_shape = (source.height, source.width)
    if label.shape != reference_shape:
        raise ValueError(f"Label shape {label.shape} does not match reference raster {reference_shape}.")
    profile.update(count=1, dtype=str(label.dtype), nodata=0)
    with rasterio.open(path, "w", **profile) as destination:
        destination.write(label, 1)
