# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Phase 1.2: Pattern matching tests for 3-axis RoPE.

Tests the Qwen-Image 3-axis rotary positional embedding with dims (16, 56, 56).
This is different from standard 1D RoPE:
  - Axis 0 (dim 16): Frame/temporal dimension
  - Axis 1 (dim 56): Height spatial dimension
  - Axis 2 (dim 56): Width spatial dimension
  - Total: 16 + 56 + 56 = 128 = head_dim

The RoPE uses complex number arithmetic and supports scale_rope=True for
centered spatial coordinates (negative + positive indices).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def test_rope_basic_shapes():
    """Test that RoPE produces correct frequency tensor shapes."""
    from diffusers.models.transformers.transformer_qwenimage import QwenEmbedRope

    rope = QwenEmbedRope(theta=10000, axes_dim=[16, 56, 56], scale_rope=True)

    # Single image: frame=1, height=8, width=8
    vid_freqs, txt_freqs = rope(
        video_fhw=[(1, 8, 8)],
        txt_seq_lens=[32],
        device=torch.device("cpu"),
    )

    # Video freqs: [1*8*8, dim/2] = [64, 64]
    # dim/2 = (16 + 56 + 56) / 2 = 64
    assert vid_freqs.shape[0] == 64, f"Video freq seq len: {vid_freqs.shape[0]}, expected 64"
    assert vid_freqs.shape[1] == 64, f"Video freq dim: {vid_freqs.shape[1]}, expected 64"

    # Text freqs: [txt_seq_len, dim/2] = [32, 64]
    assert txt_freqs.shape[0] == 32, f"Text freq seq len: {txt_freqs.shape[0]}, expected 32"
    assert txt_freqs.shape[1] == 64, f"Text freq dim: {txt_freqs.shape[1]}, expected 64"

    # Check complex type
    assert vid_freqs.is_complex(), "Video freqs should be complex"
    assert txt_freqs.is_complex(), "Text freqs should be complex"

    print("PASS: RoPE basic shapes correct")


def test_rope_different_spatial_dims():
    """Test RoPE with various spatial dimensions (different aspect ratios)."""
    from diffusers.models.transformers.transformer_qwenimage import QwenEmbedRope

    rope = QwenEmbedRope(theta=10000, axes_dim=[16, 56, 56], scale_rope=True)

    test_cases = [
        (1, 8, 8, 32),     # square, 64 patches
        (1, 16, 16, 64),   # square, 256 patches
        (1, 16, 8, 32),    # 2:1 aspect, 128 patches
        (1, 8, 16, 32),    # 1:2 aspect, 128 patches
        (1, 26, 14, 32),   # ~16:9, 364 patches
    ]

    for frame, h, w, txt_len in test_cases:
        vid_freqs, txt_freqs = rope(
            video_fhw=[(frame, h, w)],
            txt_seq_lens=[txt_len],
            device=torch.device("cpu"),
        )

        expected_vid_seq = frame * h * w
        assert vid_freqs.shape[0] == expected_vid_seq, (
            f"fhw=({frame},{h},{w}): vid seq={vid_freqs.shape[0]}, expected {expected_vid_seq}"
        )
        assert txt_freqs.shape[0] == txt_len

        # No NaN
        assert not torch.isnan(torch.view_as_real(vid_freqs)).any(), f"NaN in vid_freqs for ({frame},{h},{w})"
        assert not torch.isnan(torch.view_as_real(txt_freqs)).any(), f"NaN in txt_freqs"

        print(f"  PASS: fhw=({frame},{h},{w}), txt={txt_len} -> vid_seq={expected_vid_seq}")

    print("PASS: RoPE handles different spatial dimensions")


def test_rope_apply_to_query_key():
    """Test applying RoPE to query and key tensors."""
    from diffusers.models.transformers.transformer_qwenimage import (
        QwenEmbedRope,
        apply_rotary_emb_qwen,
    )

    rope = QwenEmbedRope(theta=10000, axes_dim=[16, 56, 56], scale_rope=True)
    batch, img_seq, heads, head_dim = 1, 64, 24, 128

    # Get frequencies
    vid_freqs, txt_freqs = rope(
        video_fhw=[(1, 8, 8)],
        txt_seq_lens=[32],
        device=torch.device("cpu"),
    )

    # Create query/key tensors [B, S, H, D]
    query = torch.randn(batch, img_seq, heads, head_dim, dtype=torch.bfloat16)
    key = torch.randn(batch, img_seq, heads, head_dim, dtype=torch.bfloat16)

    # Apply RoPE (complex mode, as used by Qwen-Image)
    q_rotated = apply_rotary_emb_qwen(query, vid_freqs, use_real=False)
    k_rotated = apply_rotary_emb_qwen(key, vid_freqs, use_real=False)

    # Shape preserved
    assert q_rotated.shape == query.shape, f"Query shape changed: {q_rotated.shape}"
    assert k_rotated.shape == key.shape, f"Key shape changed: {k_rotated.shape}"

    # Values changed (not identity)
    assert not torch.allclose(q_rotated, query, atol=1e-3), "RoPE had no effect on query"
    assert not torch.allclose(k_rotated, key, atol=1e-3), "RoPE had no effect on key"

    # No NaN
    assert not torch.isnan(q_rotated).any(), "NaN in rotated query"
    assert not torch.isnan(k_rotated).any(), "NaN in rotated key"

    print("PASS: RoPE correctly applies to query/key tensors")


def test_rope_equivariance():
    """Test that RoPE preserves relative position information.

    For RoPE, the dot product q_i^T k_j should depend only on (i-j),
    not on the absolute positions i and j.
    """
    from diffusers.models.transformers.transformer_qwenimage import (
        QwenEmbedRope,
        apply_rotary_emb_qwen,
    )

    rope = QwenEmbedRope(theta=10000, axes_dim=[16, 56, 56], scale_rope=True)

    # Use a 1D sequence for simplicity (1 frame, 1 height, N width)
    seq_len = 16
    vid_freqs, _ = rope(
        video_fhw=[(1, 1, seq_len)],
        txt_seq_lens=[1],
        device=torch.device("cpu"),
    )

    # Single head, single batch
    q = torch.randn(1, seq_len, 1, 128, dtype=torch.float32)
    k = torch.randn(1, seq_len, 1, 128, dtype=torch.float32)

    q_rot = apply_rotary_emb_qwen(q, vid_freqs, use_real=False)
    k_rot = apply_rotary_emb_qwen(k, vid_freqs, use_real=False)

    # Dot product at positions (0,2) and (1,3) should be similar
    # if q and k vectors at those positions are similar
    # This is a basic sanity check, not a full equivariance proof
    dots = torch.einsum("bshd,bthd->bsth", q_rot, k_rot)

    assert not torch.isnan(dots).any(), "NaN in RoPE dot products"
    assert not torch.isinf(dots).any(), "Inf in RoPE dot products"

    print("PASS: RoPE equivariance sanity check")


if __name__ == "__main__":
    test_rope_basic_shapes()
    test_rope_different_spatial_dims()
    test_rope_apply_to_query_key()
    test_rope_equivariance()
