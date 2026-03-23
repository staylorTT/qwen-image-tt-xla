# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Phase 0.5: End-to-End StableHLO Capture / Smoke Compile Test.

Attempts to compile the QwenImageTransformer2DModel (or a single block) through
the tt-xla backend. This validates the StableHLO → TTIR → TTNN pipeline.

Usage:
    python test_smoke_compile.py [--weights-dir ./weights/qwen-image] [--full-model]
"""

import argparse
import os

import torch
import torch_xla
import torch_xla.runtime as xr


def create_dummy_inputs(
    device,
    batch: int = 1,
    img_seq_len: int = 64,
    txt_seq_len: int = 32,
    in_channels: int = 64,
    joint_attention_dim: int = 3584,
    dtype=torch.bfloat16,
):
    """Create minimal dummy inputs for the transformer forward pass.

    Note: Latents are already packed [B, seq, in_channels] at this point.
    The packing happens before the transformer is called.
    """
    # Packed latent hidden states: [B, img_seq, in_channels]
    # in_channels=64 (packed from 16 channels * 2*2 patches)
    hidden_states = torch.randn(batch, img_seq_len, in_channels, dtype=dtype).to(device)

    # Text encoder hidden states: [B, txt_seq, joint_attention_dim]
    encoder_hidden_states = torch.randn(batch, txt_seq_len, joint_attention_dim, dtype=dtype).to(device)

    # Text attention mask: [B, txt_seq]
    encoder_hidden_states_mask = torch.ones(batch, txt_seq_len, dtype=dtype).to(device)

    # Timestep (normalized to [0, 1] — pipeline divides by 1000)
    timestep = torch.tensor([0.5], dtype=dtype).to(device)

    # Image spatial shapes: list of [(frame, height, width)]
    # For img_seq_len=64 with patch packing: sqrt(64) = 8, so 8x8 spatial
    h = w = int(img_seq_len**0.5)
    img_shapes = [[(1, h, w)]] * batch

    # Text sequence lengths
    txt_seq_lens = [txt_seq_len] * batch

    return {
        "hidden_states": hidden_states,
        "encoder_hidden_states": encoder_hidden_states,
        "encoder_hidden_states_mask": encoder_hidden_states_mask,
        "timestep": timestep,
        "img_shapes": img_shapes,
        "txt_seq_lens": txt_seq_lens,
        "return_dict": False,
    }


def test_single_block_compile(weights_dir: str):
    """Test compilation of a single MMDiT block."""
    from diffusers import DiffusionPipeline

    print("=" * 60)
    print("TEST: Single Block StableHLO Compile")
    print("=" * 60)

    # Load model on CPU
    pipe = DiffusionPipeline.from_pretrained(weights_dir, torch_dtype=torch.bfloat16)
    block = pipe.transformer.transformer_blocks[0].eval()

    # Move to XLA device
    device = torch_xla.device()
    block = block.to(device)

    # Create block-level inputs
    batch = 1
    img_seq = 64
    txt_seq = 32
    hidden_dim = pipe.transformer.config.num_attention_heads * pipe.transformer.config.attention_head_dim  # 3072

    hidden_states = torch.randn(batch, img_seq, hidden_dim, dtype=torch.bfloat16).to(device)
    encoder_hidden_states = torch.randn(batch, txt_seq, hidden_dim, dtype=torch.bfloat16).to(device)
    encoder_hidden_states_mask = torch.ones(batch, txt_seq, dtype=torch.bfloat16).to(device)
    temb = torch.randn(batch, hidden_dim, dtype=torch.bfloat16).to(device)

    # Create dummy RoPE frequencies
    pos_embed = pipe.transformer.pos_embed
    img_freqs, txt_freqs = pos_embed(
        video_fhw=[(1, 8, 8)],
        txt_seq_lens=[txt_seq],
        device=device,
    )

    # Compile
    print("Compiling single block...")
    compiled_block = torch.compile(block, backend="tt")

    with torch.no_grad():
        txt_out, img_out = compiled_block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            temb=temb,
            image_rotary_emb=(img_freqs, txt_freqs),
        )

    print(f"  Image output shape: {list(img_out.shape)} (expected [{batch}, {img_seq}, {hidden_dim}])")
    print(f"  Text output shape:  {list(txt_out.shape)} (expected [{batch}, {txt_seq}, {hidden_dim}])")

    # Check for NaN/Inf
    assert not torch.isnan(img_out.cpu()).any(), "NaN in image output!"
    assert not torch.isinf(img_out.cpu()).any(), "Inf in image output!"
    assert not torch.isnan(txt_out.cpu()).any(), "NaN in text output!"
    assert not torch.isinf(txt_out.cpu()).any(), "Inf in text output!"

    print("PASS: Single block compiled and executed successfully")
    return True


def test_full_model_compile(weights_dir: str):
    """Test compilation of the full transformer."""
    from diffusers import DiffusionPipeline

    print("=" * 60)
    print("TEST: Full Transformer StableHLO Compile")
    print("=" * 60)

    # Load model on CPU
    pipe = DiffusionPipeline.from_pretrained(weights_dir, torch_dtype=torch.bfloat16)
    transformer = pipe.transformer.eval()

    # Move to XLA device
    device = torch_xla.device()
    transformer = transformer.to(device)

    # Create dummy inputs (small spatial dims)
    inputs = create_dummy_inputs(device, img_seq_len=64, txt_seq_len=32)

    # Compile
    print("Compiling full transformer (this may take a while)...")
    compiled = torch.compile(transformer, backend="tt")

    with torch.no_grad():
        output = compiled(**inputs)

    result = output[0]
    print(f"  Output shape: {list(result.shape)}")

    # Check for NaN/Inf
    result_cpu = result.cpu()
    assert not torch.isnan(result_cpu).any(), "NaN in output!"
    assert not torch.isinf(result_cpu).any(), "Inf in output!"

    print("PASS: Full transformer compiled and executed successfully")
    return True


def main():
    parser = argparse.ArgumentParser(description="Smoke compile test for Qwen-Image on tt-xla")
    parser.add_argument("--weights-dir", type=str, default="./weights/qwen-image")
    parser.add_argument("--full-model", action="store_true", help="Test full model (slower)")
    args = parser.parse_args()

    xr.set_device_type("TT")

    passed = test_single_block_compile(args.weights_dir)

    if args.full_model and passed:
        test_full_model_compile(args.weights_dir)


if __name__ == "__main__":
    main()
