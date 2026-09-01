"""Differentiable training path and entropy-coded HyperICM inference."""

import math
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F


GROUP_SIZE = 8
DISTORTION_WEIGHTS = (0.5, 1.2, 0.5, 0.9, 0.5, 0.9, 0.5, 0.9)


def pad_spatial(tensor: torch.Tensor, multiple: int = 16):
    if tensor.ndim != 4:
        raise ValueError(f"Expected BCHW tensor, got {tuple(tensor.shape)}")
    height, width = tensor.shape[-2:]
    pad_h = (-height) % multiple
    pad_w = (-width) % multiple
    return F.pad(tensor, (0, pad_w, 0, pad_h), mode="constant", value=0), pad_h, pad_w


def unpad_spatial(tensor: torch.Tensor, pad_h: int, pad_w: int):
    height_end = tensor.shape[-2] - pad_h if pad_h else tensor.shape[-2]
    width_end = tensor.shape[-1] - pad_w if pad_w else tensor.shape[-1]
    return tensor[..., :height_end, :width_end]


def likelihood_bpp(latent: torch.Tensor, likelihoods: torch.Tensor) -> torch.Tensor:
    """Attachment-compatible likelihood-rate surrogate."""
    denominator = -math.log(2.0) * latent.numel()
    return torch.log(likelihoods).sum() / denominator


def _forward_group(i_model, p_model, frames: torch.Tensor, qp: int):
    length = frames.shape[1]
    if length < 1:
        raise ValueError("A codec group must contain at least one band.")
    reconstructions = []
    rate = frames.new_zeros(())
    distortion = frames.new_zeros(())
    previous = None
    feature = None

    for index in range(length):
        frame = frames[:, index:index + 1]
        if index == 0:
            output = i_model(frame, qp)
            x_hat, y_hat, y_likelihoods, z_hat, z_likelihoods = output
        elif index == 1:
            output = p_model(frame, previous, None, qp)
            x_hat, y_hat, y_likelihoods, z_hat, z_likelihoods, feature = output
        else:
            output = p_model(frame, None, feature, qp)
            x_hat, y_hat, y_likelihoods, z_hat, z_likelihoods, feature = output
        previous = x_hat
        reconstructions.append(x_hat)
        rate = rate + (likelihood_bpp(y_hat, y_likelihoods) + likelihood_bpp(z_hat, z_likelihoods)) / length
        distortion = distortion + F.mse_loss(x_hat, frame) * DISTORTION_WEIGHTS[index] / length

    return torch.cat(reconstructions, dim=1), rate, distortion


def differentiable_reconstruction(i_model, p_model, images: torch.Tensor, qp: int):
    """Attachment-derived B=8 path with a guard for empty tail groups."""
    padded, pad_h, pad_w = pad_spatial(images, 16)
    batch, channels, height, width = padded.shape
    full_groups, remainder = divmod(channels, GROUP_SIZE)
    recon_parts = []
    rate_parts = []
    distortion_parts = []

    if full_groups:
        front = padded[:, :full_groups * GROUP_SIZE]
        front = front.reshape(batch * full_groups, GROUP_SIZE, height, width)
        recon, rate, distortion = _forward_group(i_model, p_model, front, qp)
        recon_parts.append(recon.reshape(batch, full_groups * GROUP_SIZE, height, width))
        rate_parts.append(rate)
        distortion_parts.append(distortion)

    if remainder:
        tail = padded[:, full_groups * GROUP_SIZE:]
        recon, rate, distortion = _forward_group(i_model, p_model, tail, qp)
        recon_parts.append(recon)
        rate_parts.append(rate)
        distortion_parts.append(distortion)

    if not recon_parts:
        raise ValueError("Input contains no spectral bands.")

    reconstruction = unpad_spatial(torch.cat(recon_parts, dim=1), pad_h, pad_w)
    return reconstruction, sum(rate_parts), sum(distortion_parts)


def _string_bits(*strings: Iterable[bytes]) -> int:
    total = 0
    for value in strings:
        if value is None:
            continue
        items = value if isinstance(value, (list, tuple)) else [value]
        total += sum(len(item) * 8 for item in items)
    return total


@torch.no_grad()
def entropy_coded_inference(
    i_model,
    p_model,
    image_chw: np.ndarray,
    qp: int,
    device: torch.device,
    group_size: int = GROUP_SIZE,
):
    image = torch.as_tensor(image_chw, dtype=torch.float32, device=device)
    if image.ndim != 3 or image.shape[0] < 1:
        raise ValueError(f"Expected non-empty CHW image, got {tuple(image.shape)}")
    if not 0 <= qp <= 63:
        raise ValueError("qp must be in [0, 63] for entropy-coded inference; choose a concrete QP.")
    if group_size < 1:
        raise ValueError("group_size must be at least 1.")
    padded, pad_h, pad_w = pad_spatial(image.unsqueeze(0), 16)
    channels, original_h, original_w = image.shape
    reconstructions = []
    total_bits = 0

    i_model.eval()
    p_model.eval()
    i_model.update()
    p_model.update()

    for start in range(0, channels, group_size):
        stop = min(start + group_size, channels)
        feature = None
        for local_index, channel_index in enumerate(range(start, stop)):
            frame = padded[:, channel_index:channel_index + 1]
            if local_index == 0:
                y_strings, z_strings, z_shape, y_shape = i_model.compress(frame, qp)
                x_hat = i_model.decompress(y_strings, z_strings, z_shape, y_shape, qp)
                feature = p_model.get_i_feature(x_hat)
            else:
                y_strings, z_strings, z_shape, y_shape = p_model.compress(frame, feature, qp)
                x_hat, decoded_feature = p_model.decompress(
                    y_strings, z_strings, feature, z_shape, y_shape, qp
                )
                feature = p_model.get_p_feature(decoded_feature)
            total_bits += _string_bits(y_strings, z_strings)
            reconstructions.append(x_hat.squeeze(0).squeeze(0))

    reconstruction = torch.stack(reconstructions, dim=0).unsqueeze(0)
    reconstruction = unpad_spatial(reconstruction, pad_h, pad_w).squeeze(0)
    reference = image
    mse = float(F.mse_loss(reconstruction, reference).item())
    psnr = float("inf") if mse == 0 else 10.0 * math.log10(1.0 / mse)
    bpsp = total_bits / (channels * original_h * original_w)
    return reconstruction.cpu().numpy(), {"qp": qp, "bpsp": bpsp, "mse": mse, "psnr": psnr, "bits": total_bits}
