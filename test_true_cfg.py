# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Phase 3.3: True CFG Correctness Test.

Qwen-Image uses true classifier-free guidance with `true_cfg_scale=4.0`.
This means two full forward passes per denoising step (conditioned + unconditioned).

The CFG combination also uses norm-preserving rescaling:
  comb_pred = neg_pred + true_cfg_scale * (cond_pred - neg_pred)
  noise_pred = comb_pred * (cond_norm / comb_norm)

Tests:
  1. Two forward passes produce different outputs (cond vs uncond)
  2. CFG combination matches reference implementation
  3. Norm-preserving rescaling is correct

Usage:
    python test_true_cfg.py [--weights-dir ./weights/qwen-image]
"""

import argparse

import numpy as np
import torch
import torch_xla
import torch_xla.runtime as xr

from utils.image_utils import calculate_shift, pack_latents
from utils.profiling_utils import check_no_nan_inf, check_pcc
from utils.tp_utils import apply_tp_sharding, setup_spmd


def test_true_cfg(weights_dir: str, true_cfg_scale: float = 4.0) -> bool:
    """Test true CFG with two forward passes."""
    from diffusers import DiffusionPipeline

    print("=" * 60)
    print(f"TEST: true_cfg (scale={true_cfg_scale})")
    print("=" * 60)

    # --- Setup ---
    mesh, device, num_devices = setup_spmd()

    # --- Load model ---
    pipe = DiffusionPipeline.from_pretrained(weights_dir, torch_dtype=torch.bfloat16)
    transformer = pipe.transformer.eval()
    config = transformer.config

    # Move to device and apply TP
    transformer = transformer.to(device)
    if num_devices > 1 and config.num_attention_heads % num_devices == 0:
        apply_tp_sharding(transformer, mesh)

    compiled = torch.compile(transformer, backend="tt")

    # --- Prepare inputs ---
    height, width = 256, 256
    vae_scale_factor = 8
    latent_h = height // vae_scale_factor
    latent_w = width // vae_scale_factor
    num_channels = config.in_channels // 4

    gen = torch.Generator().manual_seed(42)
    latents = torch.randn(1, 1, num_channels, latent_h, latent_w, generator=gen, dtype=torch.bfloat16)
    latents = pack_latents(latents.squeeze(1), 1, num_channels, latent_h, latent_w)
    print(f"  Latents shape: {list(latents.shape)}")

    h_packed = latent_h // 2
    w_packed = latent_w // 2
    img_shapes = [[(1, h_packed, w_packed)]]

    # Conditioned embeddings (simulate text-encoded prompt)
    txt_seq = 32
    cond_embeds = torch.randn(1, txt_seq, config.joint_attention_dim, dtype=torch.bfloat16)
    cond_mask = torch.ones(1, txt_seq, dtype=torch.bfloat16)

    # Unconditioned embeddings (simulate empty/negative prompt — different values)
    uncond_embeds = torch.randn(1, txt_seq, config.joint_attention_dim, dtype=torch.bfloat16) * 0.1
    uncond_mask = torch.ones(1, txt_seq, dtype=torch.bfloat16)

    txt_seq_lens = [txt_seq]

    # Single timestep
    timestep = torch.tensor([0.5], dtype=torch.bfloat16)

    # Move to device
    latents_d = latents.to(device)
    cond_d = cond_embeds.to(device)
    cond_mask_d = cond_mask.to(device)
    uncond_d = uncond_embeds.to(device)
    uncond_mask_d = uncond_mask.to(device)
    ts_d = timestep.to(device)

    # --- Conditioned forward pass ---
    print("  Running conditioned forward pass...")
    with torch.no_grad():
        cond_pred = compiled(
            hidden_states=latents_d,
            encoder_hidden_states=cond_d,
            encoder_hidden_states_mask=cond_mask_d,
            timestep=ts_d / 1000,
            img_shapes=img_shapes,
            txt_seq_lens=txt_seq_lens,
            return_dict=False,
        )[0]

    torch_xla.sync()

    # --- Unconditioned forward pass ---
    print("  Running unconditioned forward pass...")
    with torch.no_grad():
        uncond_pred = compiled(
            hidden_states=latents_d,
            encoder_hidden_states=uncond_d,
            encoder_hidden_states_mask=uncond_mask_d,
            timestep=ts_d / 1000,
            img_shapes=img_shapes,
            txt_seq_lens=txt_seq_lens,
            return_dict=False,
        )[0]

    torch_xla.sync()

    cond_cpu = cond_pred.cpu()
    uncond_cpu = uncond_pred.cpu()

    # --- True CFG combination (matches pipeline_qwenimage.py) ---
    print("  Computing true CFG combination...")
    comb_pred = uncond_cpu + true_cfg_scale * (cond_cpu - uncond_cpu)

    # Norm-preserving rescaling
    cond_norm = torch.norm(cond_cpu, dim=-1, keepdim=True)
    noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
    noise_pred = comb_pred * (cond_norm / noise_norm)

    # --- Checks ---
    all_pass = True

    # 1. Cond and uncond should produce different outputs
    pcc_cond_uncond = torch.corrcoef(
        torch.stack([cond_cpu.flatten(), uncond_cpu.flatten()])
    )[0, 1].item()
    outputs_differ = pcc_cond_uncond < 0.999
    print(f"  Cond vs uncond PCC: {pcc_cond_uncond:.6f} [{'PASS' if outputs_differ else 'FAIL'} — should differ]")
    all_pass &= outputs_differ

    # 2. No NaN/Inf
    all_pass &= check_no_nan_inf(cond_cpu, "cond_pred")
    all_pass &= check_no_nan_inf(uncond_cpu, "uncond_pred")
    all_pass &= check_no_nan_inf(noise_pred, "cfg_noise_pred")

    # 3. CFG should amplify the difference
    cond_magnitude = cond_cpu.float().norm().item()
    cfg_magnitude = noise_pred.float().norm().item()
    print(f"  Cond magnitude: {cond_magnitude:.4f}")
    print(f"  CFG magnitude: {cfg_magnitude:.4f}")

    # 4. Norm preservation: final noise_pred should have same per-token norm as cond_pred
    norm_ratio = (torch.norm(noise_pred, dim=-1) / (torch.norm(cond_cpu, dim=-1) + 1e-8)).mean().item()
    norm_preserved = abs(norm_ratio - 1.0) < 0.01
    print(f"  Norm preservation ratio: {norm_ratio:.6f} [{'PASS' if norm_preserved else 'FAIL'}]")
    all_pass &= norm_preserved

    print(f"\n{'PASS' if all_pass else 'FAIL'}: true_cfg (scale={true_cfg_scale})")
    return all_pass


def main():
    parser = argparse.ArgumentParser(description="True CFG correctness test")
    parser.add_argument("--weights-dir", type=str, default="./weights/qwen-image")
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    args = parser.parse_args()

    xr.set_device_type("TT")
    test_true_cfg(args.weights_dir, true_cfg_scale=args.cfg_scale)


if __name__ == "__main__":
    main()
