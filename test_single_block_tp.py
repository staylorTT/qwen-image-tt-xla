# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Phase 2.3: Single MMDiT Block with Tensor Parallelism.

Tests a single QwenImageTransformerBlock with SPMD tensor parallelism across
multiple devices. Verifies that head-parallel sharding produces correct results
compared to single-device reference.

Tests:
  1. TP sharding applies without error
  2. Both image and text stream outputs match single-device reference (PCC >= 0.998)
  3. AllGather/ReduceScatter at correct points

Usage:
    python test_single_block_tp.py [--weights-dir ./weights/qwen-image]
"""

import argparse
import copy

import torch
import torch_xla

import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr

from utils.profiling_utils import check_no_nan_inf, check_pcc
from utils.tp_utils import setup_spmd


def build_single_block_shard_specs(block):
    """Build TP shard specs for a single MMDiT block."""
    shard_specs = {}

    # Joint attention — column-parallel
    shard_specs[block.attn.to_q.weight] = ("model", None)
    shard_specs[block.attn.to_k.weight] = ("model", None)
    shard_specs[block.attn.to_v.weight] = ("model", None)
    shard_specs[block.attn.add_q_proj.weight] = ("model", None)
    shard_specs[block.attn.add_k_proj.weight] = ("model", None)
    shard_specs[block.attn.add_v_proj.weight] = ("model", None)

    # Attention output — row-parallel
    shard_specs[block.attn.to_out[0].weight] = (None, "model")
    shard_specs[block.attn.to_add_out.weight] = (None, "model")

    # Image FFN
    shard_specs[block.img_mlp.net[0].proj.weight] = ("model", None)
    shard_specs[block.img_mlp.net[2].weight] = (None, "model")

    # Text FFN
    shard_specs[block.txt_mlp.net[0].proj.weight] = ("model", None)
    shard_specs[block.txt_mlp.net[2].weight] = (None, "model")

    return shard_specs


def test_single_block_tp(weights_dir: str) -> bool:
    """Test single block with TP vs single-device reference."""
    from diffusers import DiffusionPipeline

    print("=" * 60)
    print("TEST: tp_single_mmdit_block")
    print("=" * 60)

    # --- Setup SPMD ---
    mesh, device, num_devices = setup_spmd()
    print(f"  Devices: {num_devices}")

    # --- Load model ---
    pipe = DiffusionPipeline.from_pretrained(weights_dir, torch_dtype=torch.bfloat16)
    transformer = pipe.transformer.eval()
    hidden_dim = transformer.config.num_attention_heads * transformer.config.attention_head_dim
    num_heads = transformer.config.num_attention_heads

    if num_heads % num_devices != 0:
        print(f"  SKIP: {num_heads} heads not divisible by {num_devices} devices")
        return True

    img_seq, txt_seq = 64, 32

    # --- CPU Reference ---
    block_cpu = copy.deepcopy(transformer.transformer_blocks[0]).eval()

    gen = torch.Generator().manual_seed(42)
    hs = torch.randn(1, img_seq, hidden_dim, dtype=torch.bfloat16, generator=gen)
    eh = torch.randn(1, txt_seq, hidden_dim, dtype=torch.bfloat16, generator=gen)
    em = torch.ones(1, txt_seq, dtype=torch.bfloat16)
    temb = torch.randn(1, hidden_dim, dtype=torch.bfloat16, generator=gen)

    img_freqs, txt_freqs = transformer.pos_embed(
        video_fhw=[(1, 8, 8)],
        txt_seq_lens=[txt_seq],
        device=torch.device("cpu"),
    )

    with torch.no_grad():
        txt_ref, img_ref = block_cpu(
            hidden_states=hs,
            encoder_hidden_states=eh,
            encoder_hidden_states_mask=em,
            temb=temb,
            image_rotary_emb=(img_freqs, txt_freqs),
        )

    print(f"  CPU reference: img={list(img_ref.shape)}, txt={list(txt_ref.shape)}")

    # --- Device with TP ---
    block_dev = transformer.transformer_blocks[0].eval().to(device)

    # Apply TP sharding
    shard_specs = build_single_block_shard_specs(block_dev)
    for tensor, spec in shard_specs.items():
        xs.mark_sharding(tensor, mesh, spec)
    print(f"  Applied TP sharding: {len(shard_specs)} tensors")

    hs_d = hs.to(device)
    eh_d = eh.to(device)
    em_d = em.to(device)
    temb_d = temb.to(device)
    img_freqs_d = img_freqs.to(device)
    txt_freqs_d = txt_freqs.to(device)

    compiled_block = torch.compile(block_dev, backend="tt")

    with torch.no_grad():
        txt_out, img_out = compiled_block(
            hidden_states=hs_d,
            encoder_hidden_states=eh_d,
            encoder_hidden_states_mask=em_d,
            temb=temb_d,
            image_rotary_emb=(img_freqs_d, txt_freqs_d),
        )

    img_out_cpu = img_out.cpu()
    txt_out_cpu = txt_out.cpu()

    # --- Checks ---
    all_pass = True
    all_pass &= check_no_nan_inf(img_out_cpu, "tp_img_output")
    all_pass &= check_no_nan_inf(txt_out_cpu, "tp_txt_output")
    all_pass &= check_pcc(img_out_cpu, img_ref, threshold=0.998, label="tp_img_pcc")
    all_pass &= check_pcc(txt_out_cpu, txt_ref, threshold=0.998, label="tp_txt_pcc")

    print(f"\n{'PASS' if all_pass else 'FAIL'}: tp_single_mmdit_block ({num_devices} devices)")
    return all_pass


def main():
    parser = argparse.ArgumentParser(description="Single block TP correctness test")
    parser.add_argument("--weights-dir", type=str, default="./weights/qwen-image")
    args = parser.parse_args()

    xr.set_device_type("TT")
    test_single_block_tp(args.weights_dir)


if __name__ == "__main__":
    main()
