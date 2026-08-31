"""Checkpoint loading helpers with explicit compatibility checks."""

from collections import OrderedDict
from pathlib import Path
from typing import Mapping

import torch


def strip_module_prefix(state_dict: Mapping[str, torch.Tensor]) -> OrderedDict:
    cleaned = OrderedDict()
    for key, value in state_dict.items():
        cleaned[key[7:] if key.startswith("module.") else key] = value
    return cleaned


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def load_codec_checkpoint(path: str | Path, i_model, p_model, strict: bool = True) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location="cpu")
    required = {"i_state_dict", "p_state_dict"}
    missing = required.difference(checkpoint)
    if missing:
        raise KeyError(f"Checkpoint {path} is missing keys: {sorted(missing)}")
    i_model.load_state_dict(strip_module_prefix(checkpoint["i_state_dict"]), strict=strict)
    p_model.load_state_dict(strip_module_prefix(checkpoint["p_state_dict"]), strict=strict)
    return checkpoint


def checkpoint_state(i_model, p_model, epoch: int, metrics: dict, optimizer, scheduler) -> dict:
    state = {
        "epoch": epoch,
        "i_state_dict": unwrap_model(i_model).state_dict(),
        "p_state_dict": unwrap_model(p_model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": scheduler.state_dict(),
    }
    state.update(metrics)
    return state
