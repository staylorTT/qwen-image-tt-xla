# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Image utilities for Qwen-Image pipeline.

Handles aspect ratio calculation, image saving, and latent packing/unpacking.
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image


# Supported aspect ratios for Qwen-Image (multiples of vae_scale_factor * patch_size = 16)
ASPECT_RATIOS = {
    "1:1":   (1024, 1024),
    "1:1_lg": (1328, 1328),
    "16:9":  (1664, 928),
    "9:16":  (928, 1664),
    "4:3":   (1216, 912),
    "3:4":   (912, 1216),
    "3:2":   (1296, 864),
    "2:3":   (864, 1296),
    "21:9":  (1792, 768),
    "9:21":  (768, 1792),
}


def get_closest_aspect_ratio(width: int, height: int) -> Tuple[int, int]:
    """Find the closest supported resolution for the given width and height.

    Args:
        width: Desired image width.
        height: Desired image height.

    Returns:
        (width, height) tuple aligned to supported dimensions.
    """
    target_ratio = width / height
    best_ratio = None
    best_diff = float("inf")

    for name, (w, h) in ASPECT_RATIOS.items():
        ratio = w / h
        diff = abs(ratio - target_ratio)
        if diff < best_diff:
            best_diff = diff
            best_ratio = (w, h)

    return best_ratio


def align_to_patch(value: int, vae_scale_factor: int = 8, patch_size: int = 2) -> int:
    """Align a dimension to be divisible by vae_scale_factor * patch_size.

    Qwen-Image requires dimensions divisible by 16 (8 * 2).
    """
    alignment = vae_scale_factor * patch_size
    return (value // alignment) * alignment


def pack_latents(
    latents: torch.Tensor,
    batch_size: int,
    num_channels_latents: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """Pack spatial latents into 2x2 patches for the MMDiT.

    Converts [B, C, H, W] latents to [B, (H/2)*(W/2), C*4] packed sequence.
    Matches QwenImagePipeline._pack_latents.
    """
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
    return latents


def unpack_latents(
    latents: torch.Tensor,
    height: int,
    width: int,
    vae_scale_factor: int = 8,
) -> torch.Tensor:
    """Unpack 2x2 patches back to spatial latents for VAE decode.

    Converts [B, seq, C*4] packed sequence to [B, C, 1, H, W] spatial format.
    Matches QwenImagePipeline._unpack_latents.
    """
    batch_size, num_patches, channels = latents.shape

    height = 2 * (int(height) // (vae_scale_factor * 2))
    width = 2 * (int(width) // (vae_scale_factor * 2))

    latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    latents = latents.reshape(batch_size, channels // 4, 1, height, width)
    return latents


def save_image(image: torch.Tensor, filepath: str = "output.png"):
    """Save a tensor image to disk.

    Args:
        image: Image tensor of shape [B, C, H, W] or [C, H, W], float values in [-1, 1].
        filepath: Output file path.
    """
    if image.dim() == 4:
        image = image[0]  # take first batch element

    image = (torch.clamp(image / 2 + 0.5, 0.0, 1.0) * 255.0).to(dtype=torch.uint8)
    image_np = image.cpu().numpy()

    if image_np.shape[0] in (1, 3):
        image_np = image_np.transpose(1, 2, 0)  # C, H, W -> H, W, C
    if image_np.shape[-1] == 1:
        image_np = image_np.squeeze(-1)

    Image.fromarray(image_np).save(filepath)
    print(f"Image saved to {filepath}")


def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    """Calculate the timestep shift mu for the flow matching scheduler.

    Matches the pipeline's calculate_shift function.
    """
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu
