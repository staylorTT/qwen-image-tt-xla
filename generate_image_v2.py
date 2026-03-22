# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end image generation on TT hardware — block-level compilation.

Instead of compiling the full 60-block transformer as one graph, compiles
each block individually and loops in Python. This avoids graph-size
issues and matches the pattern proven to work in single-block tests.

IMPORTANT: Uses decomposed attention (matmul + softmax + matmul) instead of
F.scaled_dot_product_attention, which has a correctness bug in the TT
compiler's SDPA lowering that corrupts the image-stream output while
leaving the text-stream correct. The manual decomposition produces
PCC > 0.996 vs CPU reference.

Usage:
    ./run.sh generate_image_v2.py --prompt "A cat" --width 256 --height 256
"""

import argparse
import gc
import math
import os
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh

os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
xr.use_spmd()
xr.set_device_type("TT")

from diffusers import DiffusionPipeline
from PIL import Image

from utils.image_utils import calculate_shift, pack_latents, unpack_latents


def complex_to_real_rope(complex_freqs):
    """Convert complex RoPE freqs to (cos, sin) for [B, S, H, D] layout."""
    cos = complex_freqs.real.float().repeat_interleave(2, dim=-1).unsqueeze(1)
    sin = complex_freqs.imag.float().repeat_interleave(2, dim=-1).unsqueeze(1)
    return (cos, sin)


def apply_rope_real(x, cos, sin):
    """Apply rotary embedding using real arithmetic. x: [B,S,H,D]."""
    x_r, x_i = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
    x_rot = torch.stack([-x_i, x_r], dim=-1).flatten(3)
    return (x.float() * cos + x_rot.float() * sin).to(x.dtype)


def manual_attention(q, k, v):
    """Decomposed SDPA: matmul + softmax + matmul.  q/k/v: [B, H, S, D].

    Workaround for TT compiler bug in F.scaled_dot_product_attention that
    corrupts image-stream attention output.
    """
    scale = 1.0 / math.sqrt(q.shape[-1])
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn_weights = torch.softmax(attn_weights, dim=-1)
    return torch.matmul(attn_weights, v)


def patched_block_forward(block, hidden_states, encoder_hidden_states,
                          encoder_hidden_states_mask, temb, image_rotary_emb):
    """Forward through one MMDiT block with real-valued RoPE and manual attention."""
    attn = block.attn

    # AdaLN modulation
    img_mod_params = block.img_mod(temb)
    txt_mod_params = block.txt_mod(temb)
    img_mod1, img_mod2 = img_mod_params.chunk(2, dim=-1)
    txt_mod1, txt_mod2 = txt_mod_params.chunk(2, dim=-1)

    def modulate(x, mod):
        shift, scale, gate = mod.chunk(3, dim=-1)
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1), gate.unsqueeze(1)

    img_normed = block.img_norm1(hidden_states)
    img_mod_out, img_gate1 = modulate(img_normed, img_mod1)
    txt_normed = block.txt_norm1(encoder_hidden_states)
    txt_mod_out, txt_gate1 = modulate(txt_normed, txt_mod1)

    # QKV
    img_q = attn.to_q(img_mod_out).unflatten(-1, (attn.heads, -1))
    img_k = attn.to_k(img_mod_out).unflatten(-1, (attn.heads, -1))
    img_v = attn.to_v(img_mod_out).unflatten(-1, (attn.heads, -1))
    txt_q = attn.add_q_proj(txt_mod_out).unflatten(-1, (attn.heads, -1))
    txt_k = attn.add_k_proj(txt_mod_out).unflatten(-1, (attn.heads, -1))
    txt_v = attn.add_v_proj(txt_mod_out).unflatten(-1, (attn.heads, -1))

    # QK norm
    if attn.norm_q is not None:
        img_q = attn.norm_q(img_q)
    if attn.norm_k is not None:
        img_k = attn.norm_k(img_k)
    if attn.norm_added_q is not None:
        txt_q = attn.norm_added_q(txt_q)
    if attn.norm_added_k is not None:
        txt_k = attn.norm_added_k(txt_k)

    # RoPE (real cos/sin)
    if image_rotary_emb is not None:
        (img_cos, img_sin), (txt_cos, txt_sin) = image_rotary_emb
        img_q = apply_rope_real(img_q, img_cos, img_sin)
        img_k = apply_rope_real(img_k, img_cos, img_sin)
        txt_q = apply_rope_real(txt_q, txt_cos, txt_sin)
        txt_k = apply_rope_real(txt_k, txt_cos, txt_sin)

    # Joint attention — use decomposed matmul+softmax (workaround for TT SDPA bug)
    seq_txt = txt_q.shape[1]
    q = torch.cat([txt_q, img_q], dim=1)
    k = torch.cat([txt_k, img_k], dim=1)
    v = torch.cat([txt_v, img_v], dim=1)

    out = manual_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
    ).transpose(1, 2).flatten(2, 3).to(q.dtype)

    txt_attn = attn.to_add_out(out[:, :seq_txt])
    img_attn = attn.to_out[0](out[:, seq_txt:])

    # Residual + gate
    hidden_states = hidden_states + img_gate1 * img_attn
    encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn

    # FFN
    img_n2 = block.img_norm2(hidden_states)
    img_m2, img_gate2 = modulate(img_n2, img_mod2)
    hidden_states = hidden_states + img_gate2 * block.img_mlp(img_m2)

    txt_n2 = block.txt_norm2(encoder_hidden_states)
    txt_m2, txt_gate2 = modulate(txt_n2, txt_mod2)
    encoder_hidden_states = encoder_hidden_states + txt_gate2 * block.txt_mlp(txt_m2)

    return encoder_hidden_states, hidden_states


def generate(
    weights_dir, prompt, negative_prompt="", width=512, height=512,
    num_steps=20, true_cfg_scale=4.0, seed=42, output_path="output.png",
):
    total_start = time.perf_counter()

    num_devices = xr.global_runtime_device_count()
    device = torch_xla.device()
    mesh = Mesh(np.array(range(num_devices)), (1, num_devices), ("batch", "model"))
    print(f"Devices: {num_devices}")

    # Load
    print(f"Loading model...")
    pipe = DiffusionPipeline.from_pretrained(weights_dir, torch_dtype=torch.bfloat16)
    transformer = pipe.transformer.eval()
    config = transformer.config
    vae_sf = 2 ** len(pipe.vae.temperal_downsample) if hasattr(pipe.vae, "temperal_downsample") else 8

    align = vae_sf * config.patch_size
    width = (width // align) * align
    height = (height // align) * align
    print(f"Generating {width}x{height}, {num_steps} steps, cfg={true_cfg_scale}")

    # Text encode (CPU)
    t0 = time.perf_counter()
    prompt_embeds, prompt_mask = pipe.encode_prompt(prompt=prompt, device="cpu")
    do_cfg = true_cfg_scale > 1.0 and negative_prompt is not None
    if do_cfg:
        neg_embeds, neg_mask = pipe.encode_prompt(prompt=negative_prompt or "", device="cpu")
    print(f"Text encode: {time.perf_counter()-t0:.1f}s, embeds: {list(prompt_embeds.shape)}")

    # Latents
    lh, lw = height // vae_sf, width // vae_sf
    nc = config.in_channels // 4
    gen = torch.Generator().manual_seed(seed)
    latents = pack_latents(
        torch.randn(1, nc, lh, lw, generator=gen, dtype=torch.bfloat16),
        1, nc, lh, lw,
    )
    print(f"Latents: {list(latents.shape)}")

    img_shapes = [[(1, lh // 2, lw // 2)]]
    txt_seq_lens = prompt_mask.sum(dim=1).tolist()

    # RoPE (CPU -> real)
    img_fc, txt_fc = transformer.pos_embed(img_shapes, txt_seq_lens, device=torch.device("cpu"))
    img_rope = complex_to_real_rope(img_fc)
    txt_rope = complex_to_real_rope(txt_fc)
    if do_cfg:
        neg_tsl = neg_mask.sum(dim=1).tolist()
        _, neg_txt_fc = transformer.pos_embed(img_shapes, neg_tsl, device=torch.device("cpu"))
        neg_txt_rope = complex_to_real_rope(neg_txt_fc)

    # Move to device — input projections + norms stay as eager
    print("Moving to device...")
    transformer = transformer.to(device)

    # Apply TP to all blocks
    if num_devices > 1 and config.num_attention_heads % num_devices == 0:
        for block in transformer.transformer_blocks:
            specs = {}
            specs[block.attn.to_q.weight] = ("model", None)
            specs[block.attn.to_k.weight] = ("model", None)
            specs[block.attn.to_v.weight] = ("model", None)
            specs[block.attn.add_q_proj.weight] = ("model", None)
            specs[block.attn.add_k_proj.weight] = ("model", None)
            specs[block.attn.add_v_proj.weight] = ("model", None)
            specs[block.attn.to_out[0].weight] = (None, "model")
            specs[block.attn.to_add_out.weight] = (None, "model")
            specs[block.img_mlp.net[0].proj.weight] = ("model", None)
            specs[block.img_mlp.net[2].weight] = (None, "model")
            specs[block.txt_mlp.net[0].proj.weight] = ("model", None)
            specs[block.txt_mlp.net[2].weight] = (None, "model")
            for t, s in specs.items():
                xs.mark_sharding(t, mesh, s)
        print(f"  TP applied: {config.num_attention_heads // num_devices} heads/device")

    # Compile each block individually — torch.compile bakes in module weights,
    # so we need a separate compiled function per block.
    # The graph structure is identical so compilation should be fast after the first.
    compiled_blocks = []
    for idx, block in enumerate(transformer.transformer_blocks):
        def make_fn(b):
            def fn(hs, ehs, em, temb, rope):
                return patched_block_forward(b, hs, ehs, em, temb, rope)
            return fn
        compiled_blocks.append(torch.compile(make_fn(block), backend="tt"))
    print(f"  Compiled {len(compiled_blocks)} blocks")

    # Move RoPE to device
    rope_d = (
        (img_rope[0].to(torch.bfloat16).to(device), img_rope[1].to(torch.bfloat16).to(device)),
        (txt_rope[0].to(torch.bfloat16).to(device), txt_rope[1].to(torch.bfloat16).to(device)),
    )
    if do_cfg:
        neg_rope_d = (
            rope_d[0],
            (neg_txt_rope[0].to(torch.bfloat16).to(device), neg_txt_rope[1].to(torch.bfloat16).to(device)),
        )

    cond_d = prompt_embeds.to(device)
    cond_mask_d = prompt_mask.to(torch.bfloat16).to(device)
    if do_cfg:
        neg_d = neg_embeds.to(device)
        neg_mask_d = neg_mask.to(torch.bfloat16).to(device)
    latents = latents.to(device)

    # Scheduler
    scheduler = pipe.scheduler
    sigmas = np.linspace(1.0, 1 / num_steps, num_steps)
    mu = calculate_shift(latents.shape[1])
    scheduler.set_timesteps(num_steps, sigmas=sigmas, mu=mu)
    scheduler.set_begin_index(0)

    # Denoising
    print(f"Denoising ({num_steps} steps, 60 blocks each)...")
    denoise_start = time.perf_counter()

    for i, t_val in enumerate(scheduler.timesteps):
        ts = t_val.expand(1).to(torch.bfloat16).to(device) / 1000

        with torch.no_grad():
            # --- Cond pass ---
            hs = transformer.img_in(latents)
            ehs = transformer.txt_in(transformer.txt_norm(cond_d))
            temb = transformer.time_text_embed(ts.to(hs.dtype), hs)

            for cb in compiled_blocks:
                ehs, hs = cb(hs, ehs, cond_mask_d, temb, rope_d)

            hs = transformer.norm_out(hs, temb)
            noise_cond = transformer.proj_out(hs)

            if do_cfg:
                # --- Uncond pass ---
                hs2 = transformer.img_in(latents)
                ehs2 = transformer.txt_in(transformer.txt_norm(neg_d))
                temb2 = transformer.time_text_embed(ts.to(hs2.dtype), hs2)

                for cb in compiled_blocks:
                    ehs2, hs2 = cb(hs2, ehs2, neg_mask_d, temb2, neg_rope_d)

                hs2 = transformer.norm_out(hs2, temb2)
                noise_uncond = transformer.proj_out(hs2)

                # CFG combination with norm preservation
                comb = noise_uncond + true_cfg_scale * (noise_cond - noise_uncond)
                cn = torch.norm(noise_cond, dim=-1, keepdim=True)
                nn_ = torch.norm(comb, dim=-1, keepdim=True)
                noise_pred = comb * (cn / nn_)
            else:
                noise_pred = noise_cond

        # Scheduler step (CPU)
        lat_cpu = latents.cpu()
        np_cpu = noise_pred.cpu()
        latents = scheduler.step(np_cpu, t_val, lat_cpu, return_dict=False)[0].to(device)
        torch_xla.sync()

        if (i + 1) % 5 == 0 or i == 0:
            el = time.perf_counter() - denoise_start
            print(f"  Step {i+1}/{num_steps} ({el:.1f}s)")

    dt = time.perf_counter() - denoise_start
    print(f"  Done: {dt:.1f}s ({dt/num_steps:.2f}s/step)")

    # VAE decode
    print("VAE decode (CPU)...")
    lf = latents.cpu()
    lf = unpack_latents(lf, height, width, vae_sf).to(pipe.vae.dtype)
    if hasattr(pipe.vae.config, "latents_mean") and pipe.vae.config.latents_mean is not None:
        lm = torch.tensor(pipe.vae.config.latents_mean).view(1, pipe.vae.config.z_dim, 1, 1, 1).to(lf.device, lf.dtype)
        ls = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(1, pipe.vae.config.z_dim, 1, 1, 1).to(lf.device, lf.dtype)
        lf = lf / ls + lm
    with torch.no_grad():
        image = pipe.vae.decode(lf, return_dict=False)[0]
        if image.dim() == 5:
            image = image[:, :, 0]

    image = (torch.clamp(image / 2 + 0.5, 0.0, 1.0) * 255).to(torch.uint8)
    Image.fromarray(image[0].cpu().numpy().transpose(1, 2, 0)).save(output_path)
    print(f"Saved: {output_path} ({time.perf_counter()-total_start:.1f}s total)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", default='A cat holding a sign that says "Hello Tenstorrent"')
    p.add_argument("--negative_prompt", default="")
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--cfg_scale", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="output.png")
    p.add_argument("--weights-dir", default="./weights/qwen-image")
    a = p.parse_args()
    generate(a.weights_dir, a.prompt, a.negative_prompt, a.width, a.height,
             a.steps, a.cfg_scale, a.seed, a.output)


if __name__ == "__main__":
    main()
