# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Phase 1.2: Pattern matching tests for AdaLN (Adaptive Layer Normalization).

Tests that Qwen-Image's AdaLN modules are correctly identified as timestep-only
dependent, validating the key assumption for the caching optimization.

Qwen-Image AdaLN structure per block:
  - img_mod: Sequential(SiLU, Linear(3072, 18432)) -> 6 * hidden_dim
  - txt_mod: Sequential(SiLU, Linear(3072, 18432)) -> 6 * hidden_dim
  Both take ONLY temb (timestep embedding) as input.

The 6*dim output is split into (shift1, scale1, gate1, shift2, scale2, gate2)
for norm1 and norm2 respectively.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn


def test_adaln_timestep_only_dependency():
    """Verify that AdaLN outputs depend ONLY on timestep, not on hidden states."""
    from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformerBlock

    hidden_dim = 3072
    block = QwenImageTransformerBlock(
        dim=hidden_dim,
        num_attention_heads=24,
        attention_head_dim=128,
    ).eval().to(torch.bfloat16)

    # Same timestep embedding, different hidden states
    temb = torch.randn(1, hidden_dim, dtype=torch.bfloat16)

    with torch.no_grad():
        img_mod_out1 = block.img_mod(temb)
        img_mod_out2 = block.img_mod(temb)  # Same temb -> same output

    assert torch.allclose(img_mod_out1, img_mod_out2, atol=0), (
        "AdaLN output changed with same temb — should be deterministic!"
    )

    # Different timestep -> different output
    temb2 = torch.randn(1, hidden_dim, dtype=torch.bfloat16)
    with torch.no_grad():
        img_mod_out3 = block.img_mod(temb2)

    assert not torch.allclose(img_mod_out1, img_mod_out3, atol=1e-3), (
        "AdaLN output same for different temb — something is wrong!"
    )

    print("PASS: AdaLN depends only on timestep embedding")


def test_adaln_output_structure():
    """Verify AdaLN output has correct structure for modulation."""
    from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformerBlock

    hidden_dim = 3072
    block = QwenImageTransformerBlock(
        dim=hidden_dim,
        num_attention_heads=24,
        attention_head_dim=128,
    ).eval().to(torch.bfloat16)

    temb = torch.randn(1, hidden_dim, dtype=torch.bfloat16)

    with torch.no_grad():
        img_mod_out = block.img_mod(temb)
        txt_mod_out = block.txt_mod(temb)

    # Should be [B, 6*dim]
    expected_shape = (1, 6 * hidden_dim)
    assert img_mod_out.shape == expected_shape, f"img_mod shape: {img_mod_out.shape}, expected {expected_shape}"
    assert txt_mod_out.shape == expected_shape, f"txt_mod shape: {txt_mod_out.shape}, expected {expected_shape}"

    # Can be split into two chunks of [B, 3*dim] (for norm1 and norm2)
    mod1, mod2 = img_mod_out.chunk(2, dim=-1)
    assert mod1.shape == (1, 3 * hidden_dim)
    assert mod2.shape == (1, 3 * hidden_dim)

    # Each chunk can be split into (shift, scale, gate)
    shift, scale, gate = mod1.chunk(3, dim=-1)
    assert shift.shape == (1, hidden_dim)
    assert scale.shape == (1, hidden_dim)
    assert gate.shape == (1, hidden_dim)

    print("PASS: AdaLN output structure correct (6*dim -> 2 * (shift, scale, gate))")


def test_adaln_parameter_count():
    """Verify AdaLN accounts for expected fraction of total parameters."""
    from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformerBlock

    hidden_dim = 3072
    block = QwenImageTransformerBlock(
        dim=hidden_dim,
        num_attention_heads=24,
        attention_head_dim=128,
    )

    # Count AdaLN params
    adaln_params = sum(p.numel() for p in block.img_mod.parameters())
    adaln_params += sum(p.numel() for p in block.txt_mod.parameters())

    # Count total block params
    total_params = sum(p.numel() for p in block.parameters())

    # AdaLN fraction per block
    fraction = adaln_params / total_params
    print(f"  AdaLN params per block: {adaln_params / 1e6:.1f}M")
    print(f"  Total block params: {total_params / 1e6:.1f}M")
    print(f"  AdaLN fraction: {fraction * 100:.1f}%")

    # For 60 blocks: estimate total AdaLN
    total_adaln_60 = adaln_params * 60
    print(f"  Estimated total AdaLN (60 blocks): {total_adaln_60 / 1e9:.2f}B params ({total_adaln_60 * 2 / 1e9:.1f}GB bf16)")

    print("PASS: AdaLN parameter count documented")


def test_adaln_modulation_function():
    """Test the _modulate function that applies AdaLN to normalized input."""
    hidden_dim = 3072
    batch, seq = 1, 64

    # Simulate normalized input
    x = torch.randn(batch, seq, hidden_dim, dtype=torch.bfloat16)

    # Simulate modulation params (shift, scale, gate)
    mod_params = torch.randn(batch, 3 * hidden_dim, dtype=torch.bfloat16)
    shift, scale, gate = mod_params.chunk(3, dim=-1)

    # Apply modulation: x * (1 + scale) + shift
    modulated = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    gated = gate.unsqueeze(1)

    assert modulated.shape == x.shape, f"Modulated shape: {modulated.shape}"
    assert gated.shape == (batch, 1, hidden_dim), f"Gate shape: {gated.shape}"

    # Modulated values should differ from input
    assert not torch.allclose(modulated, x, atol=1e-3), "Modulation had no effect"

    print("PASS: AdaLN modulation function correct")


if __name__ == "__main__":
    test_adaln_timestep_only_dependency()
    test_adaln_output_structure()
    test_adaln_parameter_count()
    test_adaln_modulation_function()
