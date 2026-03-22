# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Interactive image generation demo on TT hardware.

Optimizations over generate_image_v2.py:
  - Batched CFG: cond+uncond run as batch_size=2 in one pass (halves block calls)
  - On-device scheduler step: no CPU round-trip per step
  - Fixed text padding: all prompts reuse the same compiled graph
  - Manual attention: workaround for TT SDPA compiler bug

Usage:
    ./run.sh demo.py [--width 256] [--height 256] [--steps 20]
"""

import argparse
import math
import os
import time

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
    cos = complex_freqs.real.float().repeat_interleave(2, dim=-1).unsqueeze(1)
    sin = complex_freqs.imag.float().repeat_interleave(2, dim=-1).unsqueeze(1)
    return (cos, sin)


def apply_rope_real(x, cos, sin):
    x_r, x_i = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
    x_rot = torch.stack([-x_i, x_r], dim=-1).flatten(3)
    return (x.float() * cos + x_rot.float() * sin).to(x.dtype)


def manual_attention(q, k, v):
    scale = 1.0 / math.sqrt(q.shape[-1])
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn_weights = torch.softmax(attn_weights, dim=-1)
    return torch.matmul(attn_weights, v)


def patched_block_forward(block, hidden_states, encoder_hidden_states,
                          encoder_hidden_states_mask, temb, image_rotary_emb):
    attn = block.attn

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

    img_q = attn.to_q(img_mod_out).unflatten(-1, (attn.heads, -1))
    img_k = attn.to_k(img_mod_out).unflatten(-1, (attn.heads, -1))
    img_v = attn.to_v(img_mod_out).unflatten(-1, (attn.heads, -1))
    txt_q = attn.add_q_proj(txt_mod_out).unflatten(-1, (attn.heads, -1))
    txt_k = attn.add_k_proj(txt_mod_out).unflatten(-1, (attn.heads, -1))
    txt_v = attn.add_v_proj(txt_mod_out).unflatten(-1, (attn.heads, -1))

    if attn.norm_q is not None:
        img_q = attn.norm_q(img_q)
    if attn.norm_k is not None:
        img_k = attn.norm_k(img_k)
    if attn.norm_added_q is not None:
        txt_q = attn.norm_added_q(txt_q)
    if attn.norm_added_k is not None:
        txt_k = attn.norm_added_k(txt_k)

    if image_rotary_emb is not None:
        (img_cos, img_sin), (txt_cos, txt_sin) = image_rotary_emb
        img_q = apply_rope_real(img_q, img_cos, img_sin)
        img_k = apply_rope_real(img_k, img_cos, img_sin)
        txt_q = apply_rope_real(txt_q, txt_cos, txt_sin)
        txt_k = apply_rope_real(txt_k, txt_cos, txt_sin)

    seq_txt = txt_q.shape[1]
    q = torch.cat([txt_q, img_q], dim=1)
    k = torch.cat([txt_k, img_k], dim=1)
    v = torch.cat([txt_v, img_v], dim=1)

    out = manual_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
    ).transpose(1, 2).flatten(2, 3).to(q.dtype)

    txt_attn = attn.to_add_out(out[:, :seq_txt])
    img_attn = attn.to_out[0](out[:, seq_txt:])

    hidden_states = hidden_states + img_gate1 * img_attn
    encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn

    img_n2 = block.img_norm2(hidden_states)
    img_m2, img_gate2 = modulate(img_n2, img_mod2)
    hidden_states = hidden_states + img_gate2 * block.img_mlp(img_m2)

    txt_n2 = block.txt_norm2(encoder_hidden_states)
    txt_m2, txt_gate2 = modulate(txt_n2, txt_mod2)
    encoder_hidden_states = encoder_hidden_states + txt_gate2 * block.txt_mlp(txt_m2)

    return encoder_hidden_states, hidden_states


class QwenImageDemo:
    def __init__(self, weights_dir, width=256, height=256, num_steps=20, cfg_scale=4.0):
        self.num_steps = num_steps
        self.cfg_scale = cfg_scale
        self.img_count = 0

        self.device = torch_xla.device()
        num_devices = xr.global_runtime_device_count()
        mesh = Mesh(np.array(range(num_devices)), (1, num_devices), ("batch", "model"))
        print(f"Devices: {num_devices}")

        # Load pipeline
        print("Loading model...")
        t0 = time.perf_counter()
        self.pipe = DiffusionPipeline.from_pretrained(weights_dir, torch_dtype=torch.bfloat16)
        self.transformer = self.pipe.transformer.eval()
        self.config = self.transformer.config
        self.vae_sf = 8
        self.hidden_dim = self.config.num_attention_heads * self.config.attention_head_dim
        print(f"  Model loaded: {time.perf_counter()-t0:.1f}s")

        # Align dimensions
        align = self.vae_sf * self.config.patch_size
        self.width = (width // align) * align
        self.height = (height // align) * align
        self.lh = self.height // self.vae_sf
        self.lw = self.width // self.vae_sf
        self.nc = self.config.in_channels // 4
        self.img_shapes = [[(1, self.lh // 2, self.lw // 2)]]
        self.max_txt_len = 128

        # Pre-compute RoPE (fixed for this resolution + padded text length)
        self._precompute_rope()

        # Move transformer to device
        print("Moving transformer to device...")
        self.transformer = self.transformer.to(self.device)

        # Apply TP
        if num_devices > 1 and self.config.num_attention_heads % num_devices == 0:
            for block in self.transformer.transformer_blocks:
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
            print(f"  TP: {self.config.num_attention_heads // num_devices} heads/device")

        # Compile blocks
        print("Compiling blocks...")
        self.compiled_blocks = []
        for block in self.transformer.transformer_blocks:
            def make_fn(b):
                def fn(hs, ehs, em, temb, rope):
                    return patched_block_forward(b, hs, ehs, em, temb, rope)
                return fn
            self.compiled_blocks.append(torch.compile(make_fn(block), backend="tt"))
        print(f"  {len(self.compiled_blocks)} blocks registered")

        # Pre-compute scheduler sigmas
        self._precompute_scheduler()

        # Warmup: compile both batch_size=2 (CFG) and batch_size=1 (no CFG)
        print("Warming up (compiling graphs)...")
        t0 = time.perf_counter()
        self._generate_image("warmup", seed=0)  # CFG path (batch=2)
        t1 = time.perf_counter()
        print(f"  Warmup CFG path: {t1-t0:.1f}s")
        old_cfg = self.cfg_scale
        self.cfg_scale = 1.0
        self._generate_image("warmup", negative_prompt=None, seed=0)  # no-CFG path (batch=1)
        self.cfg_scale = old_cfg
        print(f"  Warmup no-CFG path: {time.perf_counter()-t1:.1f}s")

        print(f"\nReady! Generating {self.width}x{self.height} images, "
              f"{self.num_steps} steps, cfg={self.cfg_scale}")

    def _precompute_rope(self):
        """Pre-compute RoPE for fixed resolution + padded text length."""
        img_fc, txt_fc = self.transformer.pos_embed(
            self.img_shapes, [self.max_txt_len], device=torch.device("cpu"))
        img_rope = complex_to_real_rope(img_fc)
        txt_rope = complex_to_real_rope(txt_fc)
        # For batch=1 (no CFG)
        self.rope_dev = (
            (img_rope[0].to(torch.bfloat16).to(self.device),
             img_rope[1].to(torch.bfloat16).to(self.device)),
            (txt_rope[0].to(torch.bfloat16).to(self.device),
             txt_rope[1].to(torch.bfloat16).to(self.device)),
        )
        # For batch=2 (CFG) — RoPE broadcasts over batch dim, same tensors work

    def _precompute_scheduler(self):
        """Pre-compute sigma pairs for on-device Euler step."""
        scheduler = self.pipe.scheduler
        sigmas = np.linspace(1.0, 1 / self.num_steps, self.num_steps)
        img_seq_len = (self.lh // 2) * (self.lw // 2)
        mu = calculate_shift(img_seq_len)
        scheduler.set_timesteps(self.num_steps, sigmas=sigmas, mu=mu)
        scheduler.set_begin_index(0)

        # Store timesteps and dt values for on-device stepping
        self.timesteps = scheduler.timesteps.clone()
        self.sigmas_sched = scheduler.sigmas.clone()

    def _pad_text(self, embeds, mask):
        """Pad text embeddings and mask to fixed max_txt_len."""
        txt_len = embeds.shape[1]
        if txt_len >= self.max_txt_len:
            return embeds[:, :self.max_txt_len, :], mask[:, :self.max_txt_len]
        pad_len = self.max_txt_len - txt_len
        embeds = F.pad(embeds, (0, 0, 0, pad_len))
        mask = F.pad(mask, (0, pad_len), value=0)
        return embeds, mask

    def _generate_image(self, prompt, negative_prompt="", seed=None):
        """Generate a single image with batched CFG and on-device scheduler."""
        if seed is None:
            seed = int(time.time() * 1000) % (2**31)

        do_cfg = self.cfg_scale > 1.0 and negative_prompt is not None

        # Text encode (CPU)
        t_enc = time.perf_counter()
        prompt_embeds, prompt_mask = self.pipe.encode_prompt(prompt=prompt, device="cpu")
        prompt_embeds, prompt_mask = self._pad_text(prompt_embeds, prompt_mask)

        if do_cfg:
            neg_embeds, neg_mask = self.pipe.encode_prompt(
                prompt=negative_prompt or "", device="cpu")
            neg_embeds, neg_mask = self._pad_text(neg_embeds, neg_mask)
            # Batch: [cond, uncond] along batch dim
            all_embeds = torch.cat([prompt_embeds, neg_embeds], dim=0)  # [2, S, D]
            all_masks = torch.cat([prompt_mask, neg_mask], dim=0)        # [2, S]
        t_enc = time.perf_counter() - t_enc

        # Latents
        gen = torch.Generator().manual_seed(seed)
        latents = pack_latents(
            torch.randn(1, self.nc, self.lh, self.lw, generator=gen, dtype=torch.bfloat16),
            1, self.nc, self.lh, self.lw,
        )

        # Move to device
        latents_d = latents.to(self.device)
        if do_cfg:
            embeds_d = all_embeds.to(self.device)
            masks_d = all_masks.to(torch.bfloat16).to(self.device)
        else:
            embeds_d = prompt_embeds.to(self.device)
            masks_d = prompt_mask.to(torch.bfloat16).to(self.device)

        # Denoising loop — all on device
        t_denoise = time.perf_counter()

        for i in range(len(self.timesteps)):
            t_val = self.timesteps[i]
            ts = t_val.expand(1).to(torch.bfloat16).to(self.device) / 1000

            with torch.no_grad():
                if do_cfg:
                    # Batch: duplicate latents for cond + uncond
                    lat_batch = latents_d.expand(2, -1, -1)  # [2, S, C]
                    ts_batch = ts.expand(2)

                    hs = self.transformer.img_in(lat_batch)
                    ehs = self.transformer.txt_in(self.transformer.txt_norm(embeds_d))
                    temb = self.transformer.time_text_embed(ts_batch.to(hs.dtype), hs)

                    for cb in self.compiled_blocks:
                        ehs, hs = cb(hs, ehs, masks_d, temb, self.rope_dev)

                    hs = self.transformer.norm_out(hs, temb)
                    noise_both = self.transformer.proj_out(hs)

                    # Split and apply CFG
                    noise_cond = noise_both[0:1]
                    noise_uncond = noise_both[1:2]
                    comb = noise_uncond + self.cfg_scale * (noise_cond - noise_uncond)
                    cn = torch.norm(noise_cond, dim=-1, keepdim=True)
                    nn_ = torch.norm(comb, dim=-1, keepdim=True)
                    noise_pred = comb * (cn / nn_)
                else:
                    hs = self.transformer.img_in(latents_d)
                    ehs = self.transformer.txt_in(self.transformer.txt_norm(embeds_d))
                    temb = self.transformer.time_text_embed(ts.to(hs.dtype), hs)

                    for cb in self.compiled_blocks:
                        ehs, hs = cb(hs, ehs, masks_d, temb, self.rope_dev)

                    hs = self.transformer.norm_out(hs, temb)
                    noise_pred = self.transformer.proj_out(hs)

                # On-device Euler step: prev_sample = sample + dt * model_output
                sigma = self.sigmas_sched[i]
                sigma_next = self.sigmas_sched[i + 1]
                dt = (sigma_next - sigma).to(torch.bfloat16).to(self.device)
                latents_d = latents_d + dt * noise_pred

            # Sync every 5 steps instead of every step — lets XLA pipeline work
            if (i + 1) % 5 == 0 or i == len(self.timesteps) - 1:
                torch_xla.sync()

        t_denoise = time.perf_counter() - t_denoise

        # VAE decode (CPU)
        t_vae = time.perf_counter()
        lf = latents_d.cpu()
        lf = unpack_latents(lf, self.height, self.width, self.vae_sf).to(self.pipe.vae.dtype)
        if hasattr(self.pipe.vae.config, "latents_mean") and self.pipe.vae.config.latents_mean is not None:
            lm = torch.tensor(self.pipe.vae.config.latents_mean).view(1, self.pipe.vae.config.z_dim, 1, 1, 1).to(lf.device, lf.dtype)
            ls = 1.0 / torch.tensor(self.pipe.vae.config.latents_std).view(1, self.pipe.vae.config.z_dim, 1, 1, 1).to(lf.device, lf.dtype)
            lf = lf / ls + lm
        with torch.no_grad():
            image = self.pipe.vae.decode(lf, return_dict=False)[0]
            if image.dim() == 5:
                image = image[:, :, 0]
        t_vae = time.perf_counter() - t_vae

        # Save
        self.img_count += 1
        filename = f"demo_{self.img_count:03d}.png"
        image = (torch.clamp(image / 2 + 0.5, 0.0, 1.0) * 255).to(torch.uint8)
        Image.fromarray(image[0].cpu().numpy().transpose(1, 2, 0)).save(filename)

        return filename, seed, t_enc, t_denoise, t_vae

    def run_interactive(self):
        """Main interactive loop."""
        print("\n" + "=" * 60)
        print("  Qwen-Image 20B on Tenstorrent — Interactive Demo")
        print("=" * 60)
        print(f"  Resolution: {self.width}x{self.height}")
        print(f"  Steps: {self.num_steps}, CFG: {self.cfg_scale}")
        print()
        print("Commands:")
        print("  <prompt>          Generate an image")
        print("  seed:<N> <prompt>  Use specific seed")
        print("  cfg:<N> <prompt>   Override CFG scale")
        print("  steps:<N> <prompt> Override step count")
        print("  quit / exit       Exit demo")
        print()

        while True:
            try:
                user_input = input("prompt> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                print("Bye!")
                break

            # Parse inline overrides
            seed = None
            cfg = self.cfg_scale
            steps = self.num_steps
            prompt = user_input

            parts = prompt.split()
            remaining = []
            for part in parts:
                if part.startswith("seed:"):
                    try:
                        seed = int(part.split(":", 1)[1])
                    except ValueError:
                        remaining.append(part)
                elif part.startswith("cfg:"):
                    try:
                        cfg = float(part.split(":", 1)[1])
                    except ValueError:
                        remaining.append(part)
                elif part.startswith("steps:"):
                    try:
                        steps = int(part.split(":", 1)[1])
                    except ValueError:
                        remaining.append(part)
                else:
                    remaining.append(part)
            prompt = " ".join(remaining)

            if not prompt:
                print("  No prompt provided.")
                continue

            # Temporarily override settings
            old_steps, old_cfg = self.num_steps, self.cfg_scale
            self.num_steps, self.cfg_scale = steps, cfg

            # Recompute scheduler if steps changed
            if steps != old_steps:
                self._precompute_scheduler()

            neg = "" if cfg > 1.0 else None
            print(f"  Generating... (seed={seed or 'random'}, steps={steps}, cfg={cfg})")

            t_total = time.perf_counter()
            try:
                filename, used_seed, t_enc, t_denoise, t_vae = self._generate_image(
                    prompt, negative_prompt=neg, seed=seed)
                t_total = time.perf_counter() - t_total

                step_time = t_denoise / steps
                print(f"  Saved: {filename} (seed={used_seed})")
                print(f"  Timing: encode={t_enc:.1f}s, denoise={t_denoise:.1f}s "
                      f"({step_time:.2f}s/step), vae={t_vae:.1f}s, total={t_total:.1f}s")
            except Exception as e:
                print(f"  Error: {e}")
                import traceback
                traceback.print_exc()

            # Restore
            self.num_steps, self.cfg_scale = old_steps, old_cfg
            if steps != old_steps:
                self._precompute_scheduler()


def main():
    p = argparse.ArgumentParser(description="Interactive Qwen-Image demo on TT hardware")
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--steps", type=int, default=15)
    p.add_argument("--cfg_scale", type=float, default=4.0)
    p.add_argument("--weights-dir", default="./weights/qwen-image")
    a = p.parse_args()

    demo = QwenImageDemo(a.weights_dir, a.width, a.height, a.steps, a.cfg_scale)
    demo.run_interactive()


if __name__ == "__main__":
    main()
