# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tensor Parallelism utilities for Qwen-Image MMDiT on tt-xla.

Provides Megatron-style TP sharding specs for the dual-stream MMDiT architecture.
Follows the canonical pattern from tt-xla/examples/pytorch/qwen3_tp.py.

Attribute paths are based on the actual diffusers QwenImageTransformer2DModel structure:
  - block.attn.to_q / to_k / to_v           — image Q/K/V (column-parallel)
  - block.attn.add_q_proj / add_k_proj / add_v_proj — text Q/K/V (column-parallel)
  - block.attn.to_out[0]                    — image attention output (row-parallel)
  - block.attn.to_add_out                   — text attention output (row-parallel)
  - block.img_mlp.net.0.proj / net.2        — image FFN up-gate / down
  - block.txt_mlp.net.0.proj / net.2        — text FFN up-gate / down
"""

import os
from typing import Dict, Tuple

import numpy as np
import torch

import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh


def setup_spmd() -> Tuple[Mesh, "torch.device", int]:
    """Initialize SPMD mode and create the device mesh.

    Returns:
        (mesh, device, num_devices)
    """
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()

    num_devices = xr.global_runtime_device_count()
    device = torch_xla.device()

    mesh_shape = (1, num_devices)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

    return mesh, device, num_devices


def build_mmdit_shard_specs(transformer) -> Dict[torch.Tensor, Tuple]:
    """Build Megatron-style TP sharding specs for QwenImageTransformer2DModel.

    The dual-stream MMDiT has:
    - Joint attention with separate image/text Q/K/V projections
    - Separate image/text FFNs per block
    - AdaLN modulation (replicated — small relative to attention/FFN)

    Sharding strategy (Megatron-style):
    - Column-parallel: Q/K/V projections, FFN up/gate → shard output dim ("model", None)
    - Row-parallel: attention output, FFN down → shard input dim (None, "model")
    - Replicate: norms, biases, AdaLN, embeddings

    Args:
        transformer: QwenImageTransformer2DModel instance (already on device).

    Returns:
        Dict mapping parameter tensors to their sharding specs.
    """
    shard_specs = {}

    for block in transformer.transformer_blocks:
        # --- Joint Attention ---
        # Image-stream Q/K/V: column-parallel (head-parallel)
        shard_specs[block.attn.to_q.weight] = ("model", None)
        shard_specs[block.attn.to_k.weight] = ("model", None)
        shard_specs[block.attn.to_v.weight] = ("model", None)

        # Text-stream Q/K/V: column-parallel (head-parallel)
        shard_specs[block.attn.add_q_proj.weight] = ("model", None)
        shard_specs[block.attn.add_k_proj.weight] = ("model", None)
        shard_specs[block.attn.add_v_proj.weight] = ("model", None)

        # Image attention output: row-parallel
        shard_specs[block.attn.to_out[0].weight] = (None, "model")

        # Text attention output: row-parallel
        shard_specs[block.attn.to_add_out.weight] = (None, "model")

        # --- Image-stream FFN ---
        # up/gate (GEGLU): column-parallel
        shard_specs[block.img_mlp.net[0].proj.weight] = ("model", None)
        # down: row-parallel
        shard_specs[block.img_mlp.net[2].weight] = (None, "model")

        # --- Text-stream FFN ---
        # up/gate (GEGLU): column-parallel
        shard_specs[block.txt_mlp.net[0].proj.weight] = ("model", None)
        # down: row-parallel
        shard_specs[block.txt_mlp.net[2].weight] = (None, "model")

    return shard_specs


def apply_tp_sharding(transformer, mesh: Mesh):
    """Apply TP sharding annotations to the transformer.

    Args:
        transformer: QwenImageTransformer2DModel instance (already on device).
        mesh: SPMD mesh.
    """
    # Validate head divisibility
    num_heads = transformer.config.num_attention_heads
    num_devices = mesh.mesh_shape[1]  # model dimension
    if num_heads % num_devices != 0:
        raise ValueError(
            f"Number of attention heads ({num_heads}) must be divisible by "
            f"number of devices ({num_devices}) for head-parallel sharding. "
            f"{num_heads} heads work with: {[d for d in range(2, num_heads+1) if num_heads % d == 0]}"
        )

    shard_specs = build_mmdit_shard_specs(transformer)

    for tensor, spec in shard_specs.items():
        xs.mark_sharding(tensor, mesh, spec)

    print(f"Applied TP sharding: {len(shard_specs)} tensors across {num_devices} devices")
    print(f"  Heads per device: {num_heads // num_devices}")


def print_sharding_summary(transformer):
    """Print a summary of which parameters are sharded and how."""
    total_params = 0
    sharded_params = 0

    for name, param in transformer.named_parameters():
        total_params += param.numel()
        # Check if tensor has sharding annotation
        is_shardable = any(
            keyword in name
            for keyword in [
                "to_q.weight", "to_k.weight", "to_v.weight",
                "add_q_proj.weight", "add_k_proj.weight", "add_v_proj.weight",
                "to_out.0.weight", "to_add_out.weight",
                "img_mlp.net.0.proj.weight", "img_mlp.net.2.weight",
                "txt_mlp.net.0.proj.weight", "txt_mlp.net.2.weight",
            ]
        )
        if is_shardable:
            sharded_params += param.numel()

    print(f"\nSharding summary:")
    print(f"  Total params: {total_params / 1e9:.2f}B")
    print(f"  Sharded params: {sharded_params / 1e9:.2f}B ({100 * sharded_params / total_params:.1f}%)")
    print(f"  Replicated params: {(total_params - sharded_params) / 1e9:.2f}B ({100 * (total_params - sharded_params) / total_params:.1f}%)")
