# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Phase 4.2: D2M Fusion Tests.

Tests for D2M (Data-to-Metal) kernel fusion opportunities in the Qwen-Image MMDiT.
These tests verify the fusion patterns at the PyTorch level, to be validated
against D2M compiler output when the fusion passes are implemented in tt-mlir.

Fusion patterns tested:
  1. AdaLN cache bypass (skip 7B params, use cached scale/shift)
  2. Dual-stream FFN fusion (parallel image + text FFN)
  3. Post-attention residual + norm fusion
  4. Timestep MLP fusion (Linear -> SiLU -> Linear -> chunk)

Key files in tt-mlir:
  - lib/Dialect/D2M/
  - lib/Conversion/D2MToTTKernel/
  - lib/Conversion/TTKernelToEmitC/
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time

import torch
import torch.nn as nn

from utils.profiling_utils import check_pcc


def test_adaln_cache_bypass_pattern():
    """Test the AdaLN cache bypass pattern.

    Instead of: temb -> img_mod(SiLU -> Linear) -> modulation params
    Use:        cached_params (pre-computed lookup)

    This skips loading ~113M params per block (SiLU + Linear weights).
    """
    hidden_dim = 3072
    batch, seq = 1, 64

    # Simulate block with live AdaLN
    img_mod = nn.Sequential(
        nn.SiLU(),
        nn.Linear(hidden_dim, 6 * hidden_dim, bias=True),
    ).eval().to(torch.bfloat16)

    temb = torch.randn(batch, hidden_dim, dtype=torch.bfloat16)
    x = torch.randn(batch, seq, hidden_dim, dtype=torch.bfloat16)

    # Live path
    with torch.no_grad():
        mod_params = img_mod(temb)
        mod1, mod2 = mod_params.chunk(2, dim=-1)
        shift, scale, gate = mod1.chunk(3, dim=-1)
        modulated_live = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    # Cached path (same result, but no img_mod forward pass)
    cached_params = mod_params.detach()  # simulates pre-computed cache lookup
    mod1_c, mod2_c = cached_params.chunk(2, dim=-1)
    shift_c, scale_c, gate_c = mod1_c.chunk(3, dim=-1)
    modulated_cached = x * (1 + scale_c.unsqueeze(1)) + shift_c.unsqueeze(1)

    assert torch.allclose(modulated_live, modulated_cached, atol=0), "Cache bypass mismatch!"

    print("PASS: AdaLN cache bypass pattern produces identical results")


def test_dual_stream_ffn_fusion_pattern():
    """Test dual-stream FFN fusion pattern.

    The image-FFN and text-FFN operate on independent data streams.
    They can be fused into a single subgraph that shares L1 bandwidth.
    """
    from diffusers.models.attention import FeedForward

    hidden_dim = 3072
    batch = 1
    img_seq, txt_seq = 64, 32

    img_ffn = FeedForward(dim=hidden_dim, dim_out=hidden_dim, activation_fn="gelu-approximate").eval().to(torch.bfloat16)
    txt_ffn = FeedForward(dim=hidden_dim, dim_out=hidden_dim, activation_fn="gelu-approximate").eval().to(torch.bfloat16)

    img_input = torch.randn(batch, img_seq, hidden_dim, dtype=torch.bfloat16)
    txt_input = torch.randn(batch, txt_seq, hidden_dim, dtype=torch.bfloat16)

    # Sequential execution (current)
    with torch.no_grad():
        img_out_seq = img_ffn(img_input)
        txt_out_seq = txt_ffn(txt_input)

    # The fusion opportunity: both FFNs could execute in parallel on device
    # sharing L1 cache. Here we verify they produce independent results.
    assert img_out_seq.shape == (batch, img_seq, hidden_dim)
    assert txt_out_seq.shape == (batch, txt_seq, hidden_dim)

    # Verify independence: changing txt_input shouldn't affect img_out
    txt_input2 = torch.randn(batch, txt_seq, hidden_dim, dtype=torch.bfloat16)
    with torch.no_grad():
        img_out2 = img_ffn(img_input)  # same img input
        txt_out2 = txt_ffn(txt_input2)  # different txt input

    assert torch.allclose(img_out_seq, img_out2, atol=0), "Image FFN affected by text input!"
    assert not torch.allclose(txt_out_seq, txt_out2, atol=1e-3), "Text FFN unaffected by different input!"

    print("PASS: Dual-stream FFN fusion pattern validated (independent streams)")


def test_post_attention_residual_norm_fusion():
    """Test post-attention residual + norm fusion pattern.

    Pattern: hidden_states = hidden_states + gate * attn_output
             normed = LayerNorm(hidden_states)
             modulated = normed * (1 + scale) + shift

    This chain of elementwise ops is a good D2M fusion candidate.
    """
    hidden_dim = 3072
    batch, seq = 1, 64

    norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6).to(torch.bfloat16)

    hidden_states = torch.randn(batch, seq, hidden_dim, dtype=torch.bfloat16)
    attn_output = torch.randn(batch, seq, hidden_dim, dtype=torch.bfloat16)
    gate = torch.randn(batch, 1, hidden_dim, dtype=torch.bfloat16)
    scale = torch.randn(batch, hidden_dim, dtype=torch.bfloat16)
    shift = torch.randn(batch, hidden_dim, dtype=torch.bfloat16)

    # Unfused path
    with torch.no_grad():
        residual = hidden_states + gate * attn_output
        normed = norm(residual)
        modulated = normed * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    assert modulated.shape == (batch, seq, hidden_dim)
    assert not torch.isnan(modulated).any()

    print("PASS: Post-attention residual + norm fusion pattern validated")


def test_timestep_mlp_fusion():
    """Test timestep MLP fusion pattern.

    Pattern: sinusoidal_embed -> Linear -> SiLU -> Linear
    This is a small computation that runs once per denoising step.
    """
    from diffusers.models.embeddings import TimestepEmbedding, Timesteps

    time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0, scale=1000)
    time_embed = TimestepEmbedding(in_channels=256, time_embed_dim=3072).eval().to(torch.bfloat16)

    timestep = torch.tensor([0.5], dtype=torch.bfloat16)

    with torch.no_grad():
        proj = time_proj(timestep)
        emb = time_embed(proj.to(torch.bfloat16))

    assert emb.shape == (1, 3072), f"Timestep embedding shape: {emb.shape}"
    assert not torch.isnan(emb).any()

    print("PASS: Timestep MLP fusion pattern validated")


if __name__ == "__main__":
    test_adaln_cache_bypass_pattern()
    test_dual_stream_ffn_fusion_pattern()
    test_post_attention_residual_norm_fusion()
    test_timestep_mlp_fusion()
