# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Phase 2.3: Full 60-layer MMDiT with Tensor Parallelism.

Tests the complete QwenImageTransformer2DModel with SPMD tensor parallelism.
Validates that the full model fits in device memory with TP and produces
correct output.

Tests:
  1. Full model loads and shards without OOM
  2. Single forward pass produces correct shape
  3. PCC >= 0.99 vs CPU reference (relaxed for 60-layer accumulation)

Usage:
    python test_full_mmdit_tp.py [--weights-dir ./weights/qwen-image]
"""

import argparse

import torch
import torch_xla

import torch_xla.runtime as xr

from utils.profiling_utils import check_no_nan_inf, check_pcc
from utils.tp_utils import apply_tp_sharding, print_sharding_summary, setup_spmd


def test_full_mmdit_tp(weights_dir: str) -> bool:
    """Test full 60-layer MMDiT with TP."""
    from diffusers import DiffusionPipeline

    print("=" * 60)
    print("TEST: tp_full_mmdit_60_blocks")
    print("=" * 60)

    # --- Setup SPMD ---
    mesh, device, num_devices = setup_spmd()
    print(f"  Devices: {num_devices}")

    # --- Load model ---
    print("  Loading model...")
    pipe = DiffusionPipeline.from_pretrained(weights_dir, torch_dtype=torch.bfloat16)
    transformer = pipe.transformer.eval()

    config = transformer.config
    hidden_dim = config.num_attention_heads * config.attention_head_dim
    print(f"  Config: {config.num_layers} layers, {config.num_attention_heads} heads, hidden={hidden_dim}")

    if config.num_attention_heads % num_devices != 0:
        print(f"  SKIP: {config.num_attention_heads} heads not divisible by {num_devices} devices")
        return True

    # --- CPU Reference (small dims for speed) ---
    img_seq, txt_seq = 64, 32
    print(f"  Running CPU reference (img_seq={img_seq}, txt_seq={txt_seq})...")

    gen = torch.Generator().manual_seed(42)
    hs = torch.randn(1, img_seq, config.in_channels, dtype=torch.bfloat16, generator=gen)
    eh = torch.randn(1, txt_seq, config.joint_attention_dim, dtype=torch.bfloat16, generator=gen)
    em = torch.ones(1, txt_seq, dtype=torch.bfloat16)
    timestep = torch.tensor([0.5], dtype=torch.bfloat16)

    h = w = int(img_seq**0.5)
    img_shapes = [[(1, h, w)]]
    txt_seq_lens = [txt_seq]

    with torch.no_grad():
        ref_output = transformer(
            hidden_states=hs,
            encoder_hidden_states=eh,
            encoder_hidden_states_mask=em,
            timestep=timestep,
            img_shapes=img_shapes,
            txt_seq_lens=txt_seq_lens,
            return_dict=False,
        )[0]

    print(f"  CPU reference shape: {list(ref_output.shape)}")

    # --- Device with TP ---
    print("  Moving model to device and applying TP sharding...")
    transformer = transformer.to(device)
    apply_tp_sharding(transformer, mesh)
    print_sharding_summary(transformer)

    hs_d = hs.to(device)
    eh_d = eh.to(device)
    em_d = em.to(device)
    ts_d = timestep.to(device)

    print("  Compiling full model...")
    compiled = torch.compile(transformer, backend="tt")

    print("  Running forward pass...")
    with torch.no_grad():
        output = compiled(
            hidden_states=hs_d,
            encoder_hidden_states=eh_d,
            encoder_hidden_states_mask=em_d,
            timestep=ts_d,
            img_shapes=img_shapes,
            txt_seq_lens=txt_seq_lens,
            return_dict=False,
        )[0]

    output_cpu = output.cpu()

    # --- Checks ---
    all_pass = True
    all_pass &= check_no_nan_inf(output_cpu, "full_mmdit_tp")

    # Shape check
    expected_shape = ref_output.shape
    shape_ok = output_cpu.shape == expected_shape
    print(f"  Shape check: {list(output_cpu.shape)} == {list(expected_shape)} [{'PASS' if shape_ok else 'FAIL'}]")
    all_pass &= shape_ok

    # PCC (relaxed for 60-layer accumulation)
    all_pass &= check_pcc(output_cpu, ref_output, threshold=0.99, label="full_mmdit_tp_pcc")

    print(f"\n{'PASS' if all_pass else 'FAIL'}: tp_full_mmdit_60_blocks ({num_devices} devices)")
    return all_pass


def main():
    parser = argparse.ArgumentParser(description="Full MMDiT TP correctness test")
    parser.add_argument("--weights-dir", type=str, default="./weights/qwen-image")
    args = parser.parse_args()

    xr.set_device_type("TT")
    test_full_mmdit_tp(args.weights_dir)


if __name__ == "__main__":
    main()
