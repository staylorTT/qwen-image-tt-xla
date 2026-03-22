# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Device smoke test: Compile and run a single MMDiT block on TT hardware.

Uses randomly initialized weights (no model download needed).
Validates StableHLO → TTIR → TTNN → TT-Metal pipeline for the core
Qwen-Image MMDiT operations:
  - Dual-stream joint attention (Q/K/V projection, SDPA, output projection)
  - AdaLN modulation (SiLU + Linear)
  - Dual-stream FeedForward (GEGLU)
  - LayerNorm, residual connections

Usage:
    ./run.sh test_device_smoke.py
"""

import os
import sys
import time

import torch
import torch_xla

import torch_xla.runtime as xr

# Must be set before importing diffusers models
xr.set_device_type("TT")

from diffusers.models.transformers.transformer_qwenimage import (
    QwenEmbedRope,
    QwenImageTransformerBlock,
)

from utils.profiling_utils import check_no_nan_inf, check_pcc


def test_single_block_on_device():
    """Compile and run a single MMDiT block on TT device."""
    print("=" * 60)
    print("TEST: Single MMDiT Block on TT Device")
    print("=" * 60)

    device = torch_xla.device()
    print(f"  Device: {device}")
    print(f"  Devices available: {xr.global_runtime_device_count()}")

    # Create a single block with random weights
    hidden_dim = 3072
    num_heads = 24
    head_dim = 128
    batch = 1
    img_seq = 64  # small for compile speed
    txt_seq = 32

    print(f"  Block config: hidden={hidden_dim}, heads={num_heads}, head_dim={head_dim}")
    print(f"  Input: batch={batch}, img_seq={img_seq}, txt_seq={txt_seq}")

    block = QwenImageTransformerBlock(
        dim=hidden_dim,
        num_attention_heads=num_heads,
        attention_head_dim=head_dim,
    ).eval().to(torch.bfloat16)

    # Create inputs
    gen = torch.Generator().manual_seed(42)
    hs = torch.randn(batch, img_seq, hidden_dim, dtype=torch.bfloat16, generator=gen)
    eh = torch.randn(batch, txt_seq, hidden_dim, dtype=torch.bfloat16, generator=gen)
    em = torch.ones(batch, txt_seq, dtype=torch.bfloat16)
    temb = torch.randn(batch, hidden_dim, dtype=torch.bfloat16, generator=gen)

    # Create RoPE frequencies — these are complex tensors on CPU.
    # TT device doesn't support complex64, so we skip RoPE for this
    # device compile test. RoPE can be pre-applied on CPU or decomposed
    # to real (cos, sin) pairs in the full pipeline.
    # For correctness comparison, we run both CPU and device WITHOUT RoPE.
    image_rotary_emb = None

    # --- CPU Reference ---
    print("\n  Running CPU reference (no RoPE for device compatibility)...")
    with torch.no_grad():
        txt_ref, img_ref = block(
            hidden_states=hs.clone(),
            encoder_hidden_states=eh.clone(),
            encoder_hidden_states_mask=em.clone(),
            temb=temb.clone(),
            image_rotary_emb=image_rotary_emb,
        )
    print(f"  CPU ref shapes: img={list(img_ref.shape)}, txt={list(txt_ref.shape)}")

    # --- TT Device ---
    print("\n  Moving block to TT device...")
    block_dev = block.to(device)
    hs_d = hs.to(device)
    eh_d = eh.to(device)
    em_d = em.to(device)
    temb_d = temb.to(device)

    print("  Compiling with torch.compile(backend='tt')...")
    t0 = time.perf_counter()
    compiled_block = torch.compile(block_dev, backend="tt")

    with torch.no_grad():
        txt_out, img_out = compiled_block(
            hidden_states=hs_d,
            encoder_hidden_states=eh_d,
            encoder_hidden_states_mask=em_d,
            temb=temb_d,
            image_rotary_emb=image_rotary_emb,
        )

    # Force execution
    torch_xla.sync()
    img_out_cpu = img_out.cpu()
    txt_out_cpu = txt_out.cpu()
    compile_time = time.perf_counter() - t0
    print(f"  First run (compile + execute): {compile_time:.1f}s")
    print(f"  Device output shapes: img={list(img_out_cpu.shape)}, txt={list(txt_out_cpu.shape)}")

    # --- Second run (should use cached graph) ---
    print("\n  Running second pass (cached graph)...")
    t1 = time.perf_counter()
    with torch.no_grad():
        txt_out2, img_out2 = compiled_block(
            hidden_states=hs_d,
            encoder_hidden_states=eh_d,
            encoder_hidden_states_mask=em_d,
            temb=temb_d,
            image_rotary_emb=image_rotary_emb,
        )
    torch_xla.sync()
    _ = img_out2.cpu()
    run_time = time.perf_counter() - t1
    print(f"  Second run: {run_time:.3f}s (vs {compile_time:.1f}s first run)")

    # --- Correctness checks ---
    print("\n  Correctness checks:")
    all_pass = True
    all_pass &= check_no_nan_inf(img_out_cpu, "device_img")
    all_pass &= check_no_nan_inf(txt_out_cpu, "device_txt")
    all_pass &= check_pcc(img_out_cpu, img_ref, threshold=0.99, label="img_pcc_vs_cpu")
    all_pass &= check_pcc(txt_out_cpu, txt_ref, threshold=0.99, label="txt_pcc_vs_cpu")

    # Shape checks
    shape_ok = img_out_cpu.shape == img_ref.shape and txt_out_cpu.shape == txt_ref.shape
    print(f"  Shape match: {'PASS' if shape_ok else 'FAIL'}")
    all_pass &= shape_ok

    print(f"\n{'PASS' if all_pass else 'FAIL'}: Single MMDiT Block on TT Device")
    return all_pass


def test_basic_ops_on_device():
    """Quick test of fundamental ops that the MMDiT uses."""
    print("\n" + "=" * 60)
    print("TEST: Basic Ops on TT Device")
    print("=" * 60)

    device = torch_xla.device()

    tests = []

    # 1. Linear (matmul + bias)
    linear = torch.nn.Linear(3072, 3072, bias=True).to(torch.bfloat16).to(device)
    x = torch.randn(1, 64, 3072, dtype=torch.bfloat16).to(device)
    compiled_linear = torch.compile(linear, backend="tt")
    with torch.no_grad():
        y = compiled_linear(x)
    torch_xla.sync()
    y_cpu = y.cpu()
    ok = not torch.isnan(y_cpu).any()
    print(f"  Linear(3072 -> 3072): shape={list(y_cpu.shape)} [{'PASS' if ok else 'FAIL'}]")
    tests.append(ok)

    # 2. LayerNorm
    ln = torch.nn.LayerNorm(3072, elementwise_affine=False).to(torch.bfloat16).to(device)
    x2 = torch.randn(1, 64, 3072, dtype=torch.bfloat16).to(device)
    compiled_ln = torch.compile(ln, backend="tt")
    with torch.no_grad():
        y2 = compiled_ln(x2)
    torch_xla.sync()
    y2_cpu = y2.cpu()
    ok2 = not torch.isnan(y2_cpu).any()
    print(f"  LayerNorm(3072): shape={list(y2_cpu.shape)} [{'PASS' if ok2 else 'FAIL'}]")
    tests.append(ok2)

    # 3. SiLU activation
    x3 = torch.randn(1, 3072, dtype=torch.bfloat16).to(device)
    silu = torch.nn.SiLU().to(device)
    compiled_silu = torch.compile(silu, backend="tt")
    with torch.no_grad():
        y3 = compiled_silu(x3)
    torch_xla.sync()
    y3_cpu = y3.cpu()
    ok3 = not torch.isnan(y3_cpu).any()
    print(f"  SiLU: shape={list(y3_cpu.shape)} [{'PASS' if ok3 else 'FAIL'}]")
    tests.append(ok3)

    all_pass = all(tests)
    print(f"\n{'PASS' if all_pass else 'FAIL'}: Basic Ops on TT Device")
    return all_pass


if __name__ == "__main__":
    ops_pass = test_basic_ops_on_device()
    if ops_pass:
        test_single_block_on_device()
    else:
        print("\nSkipping block test — basic ops failed")
        sys.exit(1)
