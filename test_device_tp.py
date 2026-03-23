# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Device TP test: Single MMDiT block with tensor parallelism on TT hardware.

Tests Megatron-style TP sharding of a QwenImageTransformerBlock across
multiple Blackhole chips using SPMD. Validates head-parallel attention
and column/row-parallel FFN sharding.

Uses randomly initialized weights (no model download needed).

Usage:
    ./run.sh test_device_tp.py
"""

import os
import sys
import time

import numpy as np
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh

# Must be set before importing diffusers models
os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
xr.use_spmd()
xr.set_device_type("TT")

from diffusers.models.transformers.transformer_qwenimage import (
    QwenImageTransformerBlock,
)

from utils.profiling_utils import check_no_nan_inf, check_pcc


def build_block_shard_specs(block):
    """Build TP shard specs for a single MMDiT block."""
    shard_specs = {}

    # Joint attention — column-parallel (head-parallel)
    shard_specs[block.attn.to_q.weight] = ("model", None)
    shard_specs[block.attn.to_k.weight] = ("model", None)
    shard_specs[block.attn.to_v.weight] = ("model", None)
    shard_specs[block.attn.add_q_proj.weight] = ("model", None)
    shard_specs[block.attn.add_k_proj.weight] = ("model", None)
    shard_specs[block.attn.add_v_proj.weight] = ("model", None)

    # Attention output — row-parallel
    shard_specs[block.attn.to_out[0].weight] = (None, "model")
    shard_specs[block.attn.to_add_out.weight] = (None, "model")

    # Image FFN — column-parallel up/gate, row-parallel down
    shard_specs[block.img_mlp.net[0].proj.weight] = ("model", None)
    shard_specs[block.img_mlp.net[2].weight] = (None, "model")

    # Text FFN — column-parallel up/gate, row-parallel down
    shard_specs[block.txt_mlp.net[0].proj.weight] = ("model", None)
    shard_specs[block.txt_mlp.net[2].weight] = (None, "model")

    return shard_specs


def test_single_block_tp():
    """Test a single MMDiT block with TP across TT devices."""
    num_devices = xr.global_runtime_device_count()
    device = torch_xla.device()

    print("=" * 60)
    print(f"TEST: Single MMDiT Block with TP ({num_devices} devices)")
    print("=" * 60)

    # Create mesh
    mesh_shape = (1, num_devices)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

    hidden_dim = 3072
    num_heads = 24
    head_dim = 128
    batch = 1
    img_seq = 64
    txt_seq = 32

    # Check head divisibility
    if num_heads % num_devices != 0:
        print(f"  SKIP: {num_heads} heads not divisible by {num_devices} devices")
        return True

    print(f"  Heads per device: {num_heads // num_devices}")

    # Create block with random weights
    block = QwenImageTransformerBlock(
        dim=hidden_dim,
        num_attention_heads=num_heads,
        attention_head_dim=head_dim,
    ).eval().to(torch.bfloat16)

    # Create deterministic inputs
    gen = torch.Generator().manual_seed(42)
    hs = torch.randn(batch, img_seq, hidden_dim, dtype=torch.bfloat16, generator=gen)
    eh = torch.randn(batch, txt_seq, hidden_dim, dtype=torch.bfloat16, generator=gen)
    em = torch.ones(batch, txt_seq, dtype=torch.bfloat16)
    temb = torch.randn(batch, hidden_dim, dtype=torch.bfloat16, generator=gen)

    # --- CPU Reference (no TP, no RoPE) ---
    print("\n  Running CPU reference...")
    with torch.no_grad():
        txt_ref, img_ref = block(
            hidden_states=hs.clone(),
            encoder_hidden_states=eh.clone(),
            encoder_hidden_states_mask=em.clone(),
            temb=temb.clone(),
            image_rotary_emb=None,
        )
    print(f"  CPU ref shapes: img={list(img_ref.shape)}, txt={list(txt_ref.shape)}")

    # --- Device with TP ---
    print("\n  Moving block to device and applying TP sharding...")
    block_dev = block.to(device)

    # Apply sharding
    shard_specs = build_block_shard_specs(block_dev)
    for tensor, spec in shard_specs.items():
        xs.mark_sharding(tensor, mesh, spec)
    print(f"  Applied TP sharding: {len(shard_specs)} tensors")

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
            image_rotary_emb=None,
        )

    torch_xla.sync()
    img_out_cpu = img_out.cpu()
    txt_out_cpu = txt_out.cpu()
    compile_time = time.perf_counter() - t0
    print(f"  First run (compile + execute): {compile_time:.1f}s")
    print(f"  Device output shapes: img={list(img_out_cpu.shape)}, txt={list(txt_out_cpu.shape)}")

    # Second run
    t1 = time.perf_counter()
    with torch.no_grad():
        txt_out2, img_out2 = compiled_block(
            hidden_states=hs_d,
            encoder_hidden_states=eh_d,
            encoder_hidden_states_mask=em_d,
            temb=temb_d,
            image_rotary_emb=None,
        )
    torch_xla.sync()
    _ = img_out2.cpu()
    run_time = time.perf_counter() - t1
    print(f"  Second run: {run_time:.3f}s")

    # --- Correctness checks ---
    print("\n  Correctness checks:")
    all_pass = True
    all_pass &= check_no_nan_inf(img_out_cpu, "tp_img")
    all_pass &= check_no_nan_inf(txt_out_cpu, "tp_txt")
    all_pass &= check_pcc(img_out_cpu, img_ref, threshold=0.99, label="tp_img_pcc_vs_cpu")
    all_pass &= check_pcc(txt_out_cpu, txt_ref, threshold=0.99, label="tp_txt_pcc_vs_cpu")

    shape_ok = img_out_cpu.shape == img_ref.shape and txt_out_cpu.shape == txt_ref.shape
    print(f"  Shape match: {'PASS' if shape_ok else 'FAIL'}")
    all_pass &= shape_ok

    print(f"\n{'PASS' if all_pass else 'FAIL'}: Single MMDiT Block with TP ({num_devices} devices)")
    return all_pass


if __name__ == "__main__":
    test_single_block_tp()
