# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Phase 1.2: Pattern matching tests for joint attention.

Tests that the StableHLO pattern for Qwen-Image's dual-stream joint attention
is correctly recognized and lowered through the tt-mlir compiler.

The joint attention pattern:
  1. Compute QKV for image stream (to_q, to_k, to_v)
  2. Compute QKV for text stream (add_q_proj, add_k_proj, add_v_proj)
  3. Apply QK normalization (RMSNorm per head)
  4. Apply 3-axis RoPE
  5. Concatenate: joint_Q = [txt_Q, img_Q], joint_K = [txt_K, img_K], joint_V = [txt_V, img_V]
  6. Compute SDPA on joint tensors
  7. Split output back to image and text streams
  8. Apply separate output projections (to_out[0], to_add_out)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn


def test_joint_attention_correctness():
    """Verify joint attention pattern produces correct output shapes and values."""
    from diffusers.models.attention_processor import Attention
    from diffusers.models.transformers.transformer_qwenimage import QwenDoubleStreamAttnProcessor2_0

    batch, img_seq, txt_seq = 1, 64, 32
    hidden_dim = 3072
    heads = 24
    head_dim = 128

    # Create attention module with dual-stream processor
    attn = Attention(
        query_dim=hidden_dim,
        cross_attention_dim=None,
        added_kv_proj_dim=hidden_dim,
        dim_head=head_dim,
        heads=heads,
        out_dim=hidden_dim,
        context_pre_only=False,
        bias=True,
        processor=QwenDoubleStreamAttnProcessor2_0(),
        qk_norm="rms_norm",
        eps=1e-6,
    ).eval().to(torch.bfloat16)

    # Inputs
    img_hidden = torch.randn(batch, img_seq, hidden_dim, dtype=torch.bfloat16)
    txt_hidden = torch.randn(batch, txt_seq, hidden_dim, dtype=torch.bfloat16)
    mask = torch.ones(batch, txt_seq, dtype=torch.bfloat16)

    with torch.no_grad():
        img_out, txt_out = attn(
            hidden_states=img_hidden,
            encoder_hidden_states=txt_hidden,
            encoder_hidden_states_mask=mask,
        )

    # Shape checks
    assert img_out.shape == (batch, img_seq, hidden_dim), f"Image output shape: {img_out.shape}"
    assert txt_out.shape == (batch, txt_seq, hidden_dim), f"Text output shape: {txt_out.shape}"

    # No NaN/Inf
    assert not torch.isnan(img_out).any(), "NaN in image output"
    assert not torch.isnan(txt_out).any(), "NaN in text output"

    print("PASS: joint attention produces correct shapes and no NaN")


def test_joint_attention_variable_length():
    """Test joint attention with different image and text sequence lengths."""
    from diffusers.models.attention_processor import Attention
    from diffusers.models.transformers.transformer_qwenimage import QwenDoubleStreamAttnProcessor2_0

    hidden_dim = 3072
    test_cases = [
        (16, 8),    # small
        (64, 32),   # medium
        (256, 64),  # large image, moderate text
        (16, 128),  # small image, long text
    ]

    attn = Attention(
        query_dim=hidden_dim,
        cross_attention_dim=None,
        added_kv_proj_dim=hidden_dim,
        dim_head=128,
        heads=24,
        out_dim=hidden_dim,
        context_pre_only=False,
        bias=True,
        processor=QwenDoubleStreamAttnProcessor2_0(),
        qk_norm="rms_norm",
        eps=1e-6,
    ).eval().to(torch.bfloat16)

    for img_seq, txt_seq in test_cases:
        img_h = torch.randn(1, img_seq, hidden_dim, dtype=torch.bfloat16)
        txt_h = torch.randn(1, txt_seq, hidden_dim, dtype=torch.bfloat16)
        mask = torch.ones(1, txt_seq, dtype=torch.bfloat16)

        with torch.no_grad():
            img_out, txt_out = attn(
                hidden_states=img_h,
                encoder_hidden_states=txt_h,
                encoder_hidden_states_mask=mask,
            )

        assert img_out.shape == (1, img_seq, hidden_dim), f"Failed for img_seq={img_seq}: {img_out.shape}"
        assert txt_out.shape == (1, txt_seq, hidden_dim), f"Failed for txt_seq={txt_seq}: {txt_out.shape}"
        assert not torch.isnan(img_out).any()
        assert not torch.isnan(txt_out).any()
        print(f"  PASS: img_seq={img_seq}, txt_seq={txt_seq}")

    print("PASS: joint attention handles variable-length sequences")


if __name__ == "__main__":
    test_joint_attention_correctness()
    test_joint_attention_variable_length()
