"""Downstream Hyperfree evaluation used by online adaptation."""

from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np

if __package__:
    from .Hyperfree.downstream import build_classifier, classification_metrics, classify
    from .hypericm.datasets import get_dataset_spec
    from .hypericm.raster import raster_shape, read_label, validate_label_reference, write_label
else:  # pragma: no cover - direct execution from the release directory
    from Hyperfree.downstream import build_classifier, classification_metrics, classify
    from hypericm.datasets import get_dataset_spec
    from hypericm.raster import raster_shape, read_label, validate_label_reference, write_label


class HyperspectralTester:
    """Run few-shot semantic classification on codec reconstructions.

    Args may be a mapping or an argparse namespace. Passing the training
    process's mask generator as classifier avoids loading Hyperfree twice.
    """

    def __init__(self, args):
        self.args = SimpleNamespace(**args) if isinstance(args, dict) else args
        self.spec = get_dataset_spec(self.args.dataset)
        self.classifier = getattr(self.args, "classifier", None)
        if self.classifier is None:
            self._classifier_model, self.classifier = build_classifier(
                self.args.ckpt_pth,
                getattr(self.args, "device", "cuda:0"),
                pred_iou_thresh=getattr(self.args, "pred_iou_thresh", 0.2),
                stability_score_thresh=getattr(self.args, "stability_score_thresh", 0.3),
            )
        else:
            self._classifier_model = None

    def _reconstruction_path(self, qp: int) -> Path:
        root = Path(self.args.recon_path)
        input_stem = Path(self.args.input).stem
        candidates = (
            root / f"{input_stem}_{qp}.npy",
            root / f"{self.spec.name}_{qp}.npy",
            root / f"{input_stem}_qp{qp}.npy",
            root / f"{self.spec.name}_qp{qp}.npy",
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        names = ", ".join(path.name for path in candidates)
        raise FileNotFoundError(f"No reconstruction for QP={qp} in {root}; tried: {names}")

    def run_benchmark(self):
        method = getattr(self.args, "method", "hyper")
        if method != "hyper":
            raise ValueError("The packaged tester currently supports method='hyper' only.")

        validate_label_reference(self.args.gt_path, self.args.input)
        bands, height, width = raster_shape(self.args.input)
        if bands != self.spec.bands:
            raise ValueError(
                f"Input has {bands} bands; {self.spec.name} expects {self.spec.bands}."
            )
        label = read_label(self.args.gt_path)
        if label.shape != (height, width):
            raise ValueError(
                f"Ground truth shape {label.shape} does not match image shape {(height, width)}."
            )

        qps = [int(qp) for qp in getattr(self.args, "qps", [3])]
        if not qps:
            raise ValueError("qps must contain at least one value.")
        invalid_qps = [qp for qp in qps if not 1 <= qp <= 63]
        if invalid_qps:
            raise ValueError(f"QP values must be in [1, 63]: {invalid_qps}")
        output_dir = Path(getattr(self.args, "save_dir", "my_results"))
        output_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        for qp in qps:
            reconstruction_path = self._reconstruction_path(qp)
            reconstruction = np.load(reconstruction_path, allow_pickle=False)
            if reconstruction.ndim == 4 and reconstruction.shape[0] == 1:
                reconstruction = reconstruction[0]
            if reconstruction.ndim != 3:
                raise ValueError(
                    f"Expected CHW reconstruction, got shape {reconstruction.shape} at "
                    f"{reconstruction_path}."
                )
            if not np.isfinite(reconstruction).all():
                raise ValueError(f"Reconstruction contains non-finite values: {reconstruction_path}")

            prediction = classify(
                self.classifier,
                np.transpose(np.clip(reconstruction, 0.0, 1.0), (1, 2, 0)),
                self.spec,
                feature_index=getattr(self.args, "feature_index", 1),
                open_set_threshold=getattr(self.args, "open_set_threshold", 0.0),
            )
            metrics = classification_metrics(prediction, label)
            metrics["qp"] = qp
            rows.append(metrics)
            write_label(
                output_dir / f"{self.spec.name}_qp{qp}_classification.tif",
                prediction,
                self.args.gt_path,
            )
            print(
                f"Downstream QP={qp}: OA={metrics['oa']:.6f}, "
                f"AA={metrics['aa']:.6f}, Kappa={metrics['kappa']:.6f}"
            )

        csv_path = getattr(self.args, "csv_path", None)
        if csv_path:
            csv_path = Path(csv_path)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=("qp", "oa", "aa", "kappa"))
                writer.writeheader()
                writer.writerows(rows)

        final = rows[-1]
        return final["oa"], final["aa"], final["kappa"]


__all__ = ["HyperspectralTester"]
