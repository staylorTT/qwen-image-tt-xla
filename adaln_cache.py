# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Phase 4.1: AdaLN Pre-computation and Caching.

Qwen-Image's AdaLN modules (img_mod and txt_mod in each block) depend ONLY on the
timestep embedding. Since timesteps are discrete during inference (e.g. 50 steps),
the AdaLN outputs can be pre-computed for all timesteps and cached.

This eliminates the need to load ~7B AdaLN parameters per denoising step, saving:
  - ~14GB DRAM bandwidth per step (7B params * 2 bytes bf16)
  - ~7GB device memory (if weights are unloaded after caching)

The cached values are small: 60 blocks * 2 streams * [B, 6*3072] * 50 timesteps
= ~220MB for batch=1, which fits easily in L1/DRAM.

Usage:
    from adaln_cache import AdaLNCacheManager
    cache_mgr = AdaLNCacheManager(transformer)
    cache = cache_mgr.precompute(timesteps, device)
    # During denoising: use cached values instead of running img_mod/txt_mod
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class AdaLNCacheManager:
    """Pre-computes and caches AdaLN modulation outputs for all timesteps.

    The Qwen-Image MMDiT has two AdaLN modules per block:
      - block.img_mod: SiLU -> Linear(3072, 18432) — image stream modulation
      - block.txt_mod: SiLU -> Linear(3072, 18432) — text stream modulation

    Both take only `temb` (timestep embedding) as input. During inference with
    fixed timesteps, we pre-compute all outputs once and look them up during
    the denoising loop.
    """

    def __init__(self, transformer):
        """Initialize with a QwenImageTransformer2DModel.

        Args:
            transformer: The loaded transformer model (can be on any device).
        """
        self.config = transformer.config
        self.num_blocks = self.config.num_layers
        self.hidden_dim = self.config.num_attention_heads * self.config.attention_head_dim

        # Extract references to the AdaLN modules and timestep embedding
        self._time_text_embed = transformer.time_text_embed
        self._blocks = transformer.transformer_blocks
        self._norm_out = transformer.norm_out

    def precompute(
        self,
        timesteps: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """Pre-compute AdaLN outputs for all timesteps.

        Args:
            timesteps: 1D tensor of timestep values (e.g. from scheduler).
            device: Device to compute on.
            dtype: Computation dtype.

        Returns:
            Dict mapping timestep_index -> {
                "temb": [B, hidden_dim],
                "blocks": List of (img_mod_output, txt_mod_output) per block,
                "norm_out_temb": temb for the output norm,
            }
        """
        cache = {}

        # We need a dummy hidden_states input just for the timestep embedding shape
        dummy_hidden = torch.zeros(1, 1, self.hidden_dim, dtype=dtype, device=device)

        print(f"Pre-computing AdaLN cache for {len(timesteps)} timesteps, {self.num_blocks} blocks...")

        with torch.no_grad():
            for step_idx, t in enumerate(timesteps):
                # Compute timestep embedding (same as transformer.forward)
                t_tensor = t.unsqueeze(0).to(dtype=dtype, device=device) if t.dim() == 0 else t.to(dtype=dtype, device=device)
                # Normalize timestep the same way as the pipeline: timestep / 1000
                t_normalized = t_tensor / 1000
                temb = self._time_text_embed(t_normalized, dummy_hidden)

                # Compute AdaLN modulation outputs for each block
                block_outputs = []
                for block in self._blocks:
                    img_mod_out = block.img_mod(temb)  # [B, 6*dim]
                    txt_mod_out = block.txt_mod(temb)  # [B, 6*dim]
                    block_outputs.append((
                        img_mod_out.detach(),
                        txt_mod_out.detach(),
                    ))

                cache[step_idx] = {
                    "temb": temb.detach(),
                    "blocks": block_outputs,
                }

        # Report cache size
        total_bytes = 0
        for step_data in cache.values():
            total_bytes += step_data["temb"].numel() * 2  # bf16
            for img_mod, txt_mod in step_data["blocks"]:
                total_bytes += (img_mod.numel() + txt_mod.numel()) * 2

        print(f"  Cache size: {total_bytes / 1e6:.1f}MB")
        print(f"  vs AdaLN weight size: ~{self._estimate_adaln_weight_size() / 1e9:.1f}GB")

        return cache

    def _estimate_adaln_weight_size(self) -> int:
        """Estimate total AdaLN parameter size in bytes (bf16)."""
        total = 0
        for block in self._blocks:
            for param in block.img_mod.parameters():
                total += param.numel() * 2
            for param in block.txt_mod.parameters():
                total += param.numel() * 2
        return total


class CachedTransformerBlock(nn.Module):
    """Wrapper around QwenImageTransformerBlock that uses pre-computed AdaLN.

    Instead of running img_mod(temb) and txt_mod(temb), this module looks up
    the pre-computed values from the cache. This avoids loading the ~7B AdaLN
    parameters entirely.
    """

    def __init__(self, original_block, cached_img_mod: torch.Tensor, cached_txt_mod: torch.Tensor):
        """Wrap a block with cached modulation outputs.

        Args:
            original_block: The original QwenImageTransformerBlock.
            cached_img_mod: Pre-computed img_mod output [B, 6*dim].
            cached_txt_mod: Pre-computed txt_mod output [B, 6*dim].
        """
        super().__init__()
        # Keep references to non-AdaLN parts of the block
        self.img_norm1 = original_block.img_norm1
        self.attn = original_block.attn
        self.img_norm2 = original_block.img_norm2
        self.img_mlp = original_block.img_mlp
        self.txt_norm1 = original_block.txt_norm1
        self.txt_norm2 = original_block.txt_norm2
        self.txt_mlp = original_block.txt_mlp

        # Store cached modulation values
        self.register_buffer("cached_img_mod", cached_img_mod)
        self.register_buffer("cached_txt_mod", cached_txt_mod)

    def _modulate(self, x, mod_params):
        """Apply modulation to input tensor (same as original block)."""
        shift, scale, gate = mod_params.chunk(3, dim=-1)
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1), gate.unsqueeze(1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb=None,
        joint_attention_kwargs=None,
    ):
        """Forward pass using cached AdaLN values instead of computing them."""
        # Use cached modulation params instead of block.img_mod(temb) / block.txt_mod(temb)
        img_mod_params = self.cached_img_mod
        txt_mod_params = self.cached_txt_mod

        # Split for norm1 and norm2
        img_mod1, img_mod2 = img_mod_params.chunk(2, dim=-1)
        txt_mod1, txt_mod2 = txt_mod_params.chunk(2, dim=-1)

        # Image stream - norm1 + modulation
        img_normed = self.img_norm1(hidden_states)
        img_modulated, img_gate1 = self._modulate(img_normed, img_mod1)

        # Text stream - norm1 + modulation
        txt_normed = self.txt_norm1(encoder_hidden_states)
        txt_modulated, txt_gate1 = self._modulate(txt_normed, txt_mod1)

        # Joint attention
        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_output = self.attn(
            hidden_states=img_modulated,
            encoder_hidden_states=txt_modulated,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )
        img_attn_output, txt_attn_output = attn_output

        # Residual + gate
        hidden_states = hidden_states + img_gate1 * img_attn_output
        encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn_output

        # Image stream - norm2 + MLP
        img_normed2 = self.img_norm2(hidden_states)
        img_modulated2, img_gate2 = self._modulate(img_normed2, img_mod2)
        img_mlp_output = self.img_mlp(img_modulated2)
        hidden_states = hidden_states + img_gate2 * img_mlp_output

        # Text stream - norm2 + MLP
        txt_normed2 = self.txt_norm2(encoder_hidden_states)
        txt_modulated2, txt_gate2 = self._modulate(txt_normed2, txt_mod2)
        txt_mlp_output = self.txt_mlp(txt_modulated2)
        encoder_hidden_states = encoder_hidden_states + txt_gate2 * txt_mlp_output

        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states
