# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end image generation on TT hardware.

Works around the complex64 RoPE limitation by decomposing complex
rotary embeddings to real (cos, sin) pairs before sending to device,
and patching the attention processor to use real-arithmetic RoPE.

Usage:
    ./run.sh generate_image.py --prompt "A cat holding a sign that says hello"
"""

import argparse
import gc
import os
import time
from typing import List, Optional, Tuple

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


def complex_to_real_rope(complex_freqs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert complex RoPE frequencies to real (cos, sin) pair.

    The complex freqs are e^{i*theta} = cos(theta) + i*sin(theta).
    We extract cos and sin and interleave to match head_dim layout.

    Qwen-Image attention uses x shape [B, S, H, D], so cos/sin need shape
    [S, 1, D] for proper broadcasting (1 broadcasts over H=heads).

    Args:
        complex_freqs: Complex tensor [S, D//2] from QwenEmbedRope.

    Returns:
        (cos, sin) tuple, each [S, 1, D] real tensors.
    """
    cos = complex_freqs.real.float().repeat_interleave(2, dim=-1)  # [S, D]
    sin = complex_freqs.imag.float().repeat_interleave(2, dim=-1)  # [S, D]
    # Add head broadcast dim: [S, D] -> [S, 1, D] for [B, S, H, D] layout
    return (cos.unsqueeze(1), sin.unsqueeze(1))


def apply_tp_sharding(transformer, mesh):
    """Apply Megatron-style TP sharding to all blocks."""
    specs = {}
    for block in transformer.transformer_blocks:
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
    for tensor, spec in specs.items():
        xs.mark_sharding(tensor, mesh, spec)
    print(f"  Applied TP: {len(specs)} tensors sharded")


def manual_transformer_forward(
    transformer,
    hidden_states,
    encoder_hidden_states,
    encoder_hidden_states_mask,
    timestep,
    image_rotary_emb_real,
    txt_seq_lens,
):
    """Manual forward pass through the transformer, using real-valued RoPE.

    This replaces transformer.forward() to avoid the internal pos_embed call
    that produces complex tensors. Instead, we pass pre-computed real (cos, sin)
    RoPE tensors that are already on the TT device.
    """
    # Input projections (same as transformer.forward)
    hidden_states = transformer.img_in(hidden_states)
    timestep = timestep.to(hidden_states.dtype)
    encoder_hidden_states = transformer.txt_norm(encoder_hidden_states)
    encoder_hidden_states = transformer.txt_in(encoder_hidden_states)

    # Timestep embedding
    temb = transformer.time_text_embed(timestep, hidden_states)

    # Run all 60 blocks
    for block in transformer.transformer_blocks:
        encoder_hidden_states, hidden_states = block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            temb=temb,
            image_rotary_emb=image_rotary_emb_real,
        )

    # Output projection
    hidden_states = transformer.norm_out(hidden_states, temb)
    output = transformer.proj_out(hidden_states)
    return output


def patch_attention_for_real_rope():
    """Monkey-patch the QwenDoubleStreamAttnProcessor to use real RoPE.

    The original processor calls apply_rotary_emb_qwen with use_real=False
    (complex mode). We patch it to detect when freqs are a tuple of real
    tensors and use use_real=True instead.
    """
    from diffusers.models.transformers.transformer_qwenimage import (
        QwenDoubleStreamAttnProcessor2_0,
    )

    original_call = QwenDoubleStreamAttnProcessor2_0.__call__

    def patched_call(self, attn, hidden_states, encoder_hidden_states=None,
                     encoder_hidden_states_mask=None, attention_mask=None,
                     image_rotary_emb=None):
        if encoder_hidden_states is None:
            raise ValueError("Requires encoder_hidden_states")

        seq_txt = encoder_hidden_states.shape[1]

        # QKV for image stream
        img_query = attn.to_q(hidden_states)
        img_key = attn.to_k(hidden_states)
        img_value = attn.to_v(hidden_states)

        # QKV for text stream
        txt_query = attn.add_q_proj(encoder_hidden_states)
        txt_key = attn.add_k_proj(encoder_hidden_states)
        txt_value = attn.add_v_proj(encoder_hidden_states)

        # Reshape for multi-head
        img_query = img_query.unflatten(-1, (attn.heads, -1))
        img_key = img_key.unflatten(-1, (attn.heads, -1))
        img_value = img_value.unflatten(-1, (attn.heads, -1))
        txt_query = txt_query.unflatten(-1, (attn.heads, -1))
        txt_key = txt_key.unflatten(-1, (attn.heads, -1))
        txt_value = txt_value.unflatten(-1, (attn.heads, -1))

        # QK norm
        if attn.norm_q is not None:
            img_query = attn.norm_q(img_query)
        if attn.norm_k is not None:
            img_key = attn.norm_k(img_key)
        if attn.norm_added_q is not None:
            txt_query = attn.norm_added_q(txt_query)
        if attn.norm_added_k is not None:
            txt_key = attn.norm_added_k(txt_key)

        # Apply RoPE using real cos/sin arithmetic.
        # x shape: [B, S, H, D], cos/sin shape: [S, 1, D] (broadcasts over H)
        if image_rotary_emb is not None:
            img_rope, txt_rope = image_rotary_emb
            img_cos, img_sin = img_rope  # each [S_img, 1, D]
            txt_cos, txt_sin = txt_rope  # each [S_txt, 1, D]

            def apply_rope_real(x, cos, sin):
                # x: [B, S, H, D] — decompose last dim into pairs
                x_r, x_i = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B,S,H,D//2]
                x_rot = torch.stack([-x_i, x_r], dim=-1).flatten(3)    # [B,S,H,D]
                return (x.float() * cos + x_rot.float() * sin).to(x.dtype)

            img_query = apply_rope_real(img_query, img_cos, img_sin)
            img_key = apply_rope_real(img_key, img_cos, img_sin)
            txt_query = apply_rope_real(txt_query, txt_cos, txt_sin)
            txt_key = apply_rope_real(txt_key, txt_cos, txt_sin)

        # Joint attention
        joint_query = torch.cat([txt_query, img_query], dim=1)
        joint_key = torch.cat([txt_key, img_key], dim=1)
        joint_value = torch.cat([txt_value, img_value], dim=1)

        joint_hidden_states = F.scaled_dot_product_attention(
            joint_query.transpose(1, 2),
            joint_key.transpose(1, 2),
            joint_value.transpose(1, 2),
            dropout_p=0.0,
            is_causal=False,
        ).transpose(1, 2)

        joint_hidden_states = joint_hidden_states.flatten(2, 3)
        joint_hidden_states = joint_hidden_states.to(joint_query.dtype)

        # Split back
        txt_attn_output = joint_hidden_states[:, :seq_txt, :]
        img_attn_output = joint_hidden_states[:, seq_txt:, :]

        # Output projections
        img_attn_output = attn.to_out[0](img_attn_output)
        if len(attn.to_out) > 1:
            img_attn_output = attn.to_out[1](img_attn_output)
        txt_attn_output = attn.to_add_out(txt_attn_output)

        return img_attn_output, txt_attn_output

    QwenDoubleStreamAttnProcessor2_0.__call__ = patched_call
    print("  Patched attention processor for real-valued RoPE")


def generate(
    weights_dir: str,
    prompt: str,
    negative_prompt: str = "",
    width: int = 1024,
    height: int = 1024,
    num_steps: int = 50,
    true_cfg_scale: float = 4.0,
    seed: int = 42,
    output_path: str = "output.png",
):
    """Generate an image end-to-end on TT hardware."""
    total_start = time.perf_counter()

    # --- Setup device ---
    num_devices = xr.global_runtime_device_count()
    device = torch_xla.device()
    mesh = Mesh(np.array(range(num_devices)), (1, num_devices), ("batch", "model"))
    print(f"Devices: {num_devices} Blackhole chips")

    # --- Patch attention for real RoPE ---
    patch_attention_for_real_rope()

    # --- Load pipeline ---
    print(f"Loading model from {weights_dir}...")
    pipe = DiffusionPipeline.from_pretrained(weights_dir, torch_dtype=torch.bfloat16)
    transformer = pipe.transformer.eval()
    config = transformer.config
    vae_scale_factor = 2 ** len(pipe.vae.temperal_downsample) if hasattr(pipe.vae, "temperal_downsample") else 8

    # Align dimensions
    alignment = vae_scale_factor * config.patch_size
    width = (width // alignment) * alignment
    height = (height // alignment) * alignment
    print(f"Generating {width}x{height}, {num_steps} steps, cfg={true_cfg_scale}")

    # --- 1. Encode text on CPU ---
    print("Encoding prompt on CPU...")
    t0 = time.perf_counter()

    prompt_embeds, prompt_mask = pipe.encode_prompt(prompt=prompt, device="cpu")
    do_cfg = true_cfg_scale > 1.0 and negative_prompt is not None
    if do_cfg:
        neg_embeds, neg_mask = pipe.encode_prompt(prompt=negative_prompt or "", device="cpu")

    print(f"  Text encoding: {time.perf_counter() - t0:.1f}s")
    print(f"  Prompt embeds: {list(prompt_embeds.shape)}")

    # --- 2. Prepare latents ---
    latent_h = height // vae_scale_factor
    latent_w = width // vae_scale_factor
    num_channels = config.in_channels // 4  # 16

    generator = torch.Generator().manual_seed(seed)
    latents = torch.randn(1, 1, num_channels, latent_h, latent_w, generator=generator, dtype=torch.bfloat16)
    latents = pack_latents(latents.squeeze(1), 1, num_channels, latent_h, latent_w)
    print(f"  Latents: {list(latents.shape)} (packed)")

    img_seq_len = latents.shape[1]
    h_packed = latent_h // 2
    w_packed = latent_w // 2
    img_shapes = [[(1, h_packed, w_packed)]]
    txt_seq_lens = prompt_mask.sum(dim=1).tolist()

    # --- 3. Pre-compute RoPE on CPU, convert to real ---
    print("Pre-computing RoPE (CPU → real cos/sin)...")
    pos_embed = transformer.pos_embed
    img_freqs_complex, txt_freqs_complex = pos_embed(
        img_shapes, txt_seq_lens, device=torch.device("cpu")
    )
    img_rope_real = complex_to_real_rope(img_freqs_complex)
    txt_rope_real = complex_to_real_rope(txt_freqs_complex)

    if do_cfg:
        neg_txt_seq_lens = neg_mask.sum(dim=1).tolist()
        _, neg_txt_freqs_complex = pos_embed(
            img_shapes, neg_txt_seq_lens, device=torch.device("cpu")
        )
        neg_txt_rope_real = complex_to_real_rope(neg_txt_freqs_complex)

    # --- 4. Move transformer to device + TP ---
    print("Moving transformer to device...")
    # Remove pos_embed before moving (not needed on device)
    transformer.pos_embed = torch.nn.Identity()
    transformer = transformer.to(device)

    if num_devices > 1 and config.num_attention_heads % num_devices == 0:
        apply_tp_sharding(transformer, mesh)

    def fwd_fn(hs, ehs, ehm, ts, rope, tsl):
        return manual_transformer_forward(transformer, hs, ehs, ehm, ts, rope, tsl)

    compiled_fwd = torch.compile(fwd_fn, backend="tt")

    # Move tensors to device
    img_rope_dev = (img_rope_real[0].to(torch.bfloat16).to(device),
                    img_rope_real[1].to(torch.bfloat16).to(device))
    txt_rope_dev = (txt_rope_real[0].to(torch.bfloat16).to(device),
                    txt_rope_real[1].to(torch.bfloat16).to(device))
    rope_dev = (img_rope_dev, txt_rope_dev)

    if do_cfg:
        neg_txt_rope_dev = (neg_txt_rope_real[0].to(torch.bfloat16).to(device),
                            neg_txt_rope_real[1].to(torch.bfloat16).to(device))
        neg_rope_dev = (img_rope_dev, neg_txt_rope_dev)

    cond_d = prompt_embeds.to(device)
    cond_mask_d = prompt_mask.to(torch.bfloat16).to(device)
    if do_cfg:
        neg_d = neg_embeds.to(device)
        neg_mask_d = neg_mask.to(torch.bfloat16).to(device)

    latents = latents.to(device)

    # --- 5. Scheduler ---
    scheduler = pipe.scheduler
    sigmas = np.linspace(1.0, 1 / num_steps, num_steps)
    mu = calculate_shift(img_seq_len)
    scheduler.set_timesteps(num_steps, sigmas=sigmas, mu=mu)
    scheduler.set_begin_index(0)

    # --- 6. Denoising loop ---
    print(f"Denoising ({num_steps} steps)...")
    denoise_start = time.perf_counter()

    for i, t in enumerate(scheduler.timesteps):
        timestep = t.expand(1).to(torch.bfloat16).to(device) / 1000

        with torch.no_grad():
            noise_pred_cond = compiled_fwd(
                latents, cond_d, cond_mask_d, timestep,
                rope_dev, txt_seq_lens,
            )

            if do_cfg:
                noise_pred_uncond = compiled_fwd(
                    latents, neg_d, neg_mask_d, timestep,
                    neg_rope_dev, neg_txt_seq_lens,
                )
                comb = noise_pred_uncond + true_cfg_scale * (noise_pred_cond - noise_pred_uncond)
                cond_norm = torch.norm(noise_pred_cond, dim=-1, keepdim=True)
                noise_norm = torch.norm(comb, dim=-1, keepdim=True)
                noise_pred = comb * (cond_norm / noise_norm)
            else:
                noise_pred = noise_pred_cond

        # Scheduler step on CPU
        latents_cpu = latents.cpu()
        noise_cpu = noise_pred.cpu()
        latents_next = scheduler.step(noise_cpu, t, latents_cpu, return_dict=False)[0]
        latents = latents_next.to(device)

        torch_xla.sync()

        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.perf_counter() - denoise_start
            print(f"  Step {i+1}/{num_steps} ({elapsed:.1f}s elapsed)")

    denoise_time = time.perf_counter() - denoise_start
    print(f"  Denoising complete: {denoise_time:.1f}s ({denoise_time/num_steps:.2f}s/step)")

    # --- 7. VAE decode on CPU ---
    print("VAE decoding on CPU...")
    t0 = time.perf_counter()
    latents_final = latents.cpu()
    latents_final = unpack_latents(latents_final, height, width, vae_scale_factor)
    latents_final = latents_final.to(pipe.vae.dtype)

    if hasattr(pipe.vae.config, "latents_mean") and pipe.vae.config.latents_mean is not None:
        latents_mean = (
            torch.tensor(pipe.vae.config.latents_mean)
            .view(1, pipe.vae.config.z_dim, 1, 1, 1)
            .to(latents_final.device, latents_final.dtype)
        )
        latents_std = (
            1.0 / torch.tensor(pipe.vae.config.latents_std)
            .view(1, pipe.vae.config.z_dim, 1, 1, 1)
            .to(latents_final.device, latents_final.dtype)
        )
        latents_final = latents_final / latents_std + latents_mean

    with torch.no_grad():
        image = pipe.vae.decode(latents_final, return_dict=False)[0]
        if image.dim() == 5:
            image = image[:, :, 0]

    print(f"  VAE decode: {time.perf_counter() - t0:.1f}s")

    # --- 8. Save image ---
    image = (torch.clamp(image / 2 + 0.5, 0.0, 1.0) * 255).to(torch.uint8)
    image_np = image[0].cpu().numpy().transpose(1, 2, 0)
    Image.fromarray(image_np).save(output_path)

    total_time = time.perf_counter() - total_start
    print(f"\nImage saved to {output_path}")
    print(f"Total time: {total_time:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Generate image on TT hardware")
    parser.add_argument("--prompt", type=str, default='A cat holding a sign that says "Hello Tenstorrent"')
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="output.png")
    parser.add_argument("--weights-dir", type=str, default="./weights/qwen-image")
    args = parser.parse_args()

    generate(
        weights_dir=args.weights_dir,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        width=args.width,
        height=args.height,
        num_steps=args.steps,
        true_cfg_scale=args.cfg_scale,
        seed=args.seed,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
