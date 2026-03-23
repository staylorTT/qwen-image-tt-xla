# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Demo with shared block compilation — compile once, reuse for all 60 blocks.

Key insight: all 60 blocks have identical structure and weight shapes.
We compile ONE block function and call it for all 60. XLA's HLO-level
caching ensures the compiled graph is reused automatically.

Usage:
    ./run.sh demo_shared.py [--width 256] [--height 256] [--steps 15]
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

try:
    torch_xla.set_custom_compile_options({"optimization_level": "1"})
except Exception:
    pass

from diffusers import DiffusionPipeline
from PIL import Image

from utils.image_utils import calculate_shift, pack_latents, unpack_latents

# Import the block forward and helpers from demo.py
from demo import (complex_to_real_rope, apply_rope_real, manual_attention,
                  patched_block_forward)


class QwenImageDemo:
    def __init__(self, weights_dir, width=256, height=256, num_steps=15, cfg_scale=4.0):
        self.num_steps = num_steps
        self.cfg_scale = cfg_scale
        self.img_count = 0

        self.device = torch_xla.device()
        num_devices = xr.global_runtime_device_count()
        mesh = Mesh(np.array(range(num_devices)), (1, num_devices), ("batch", "model"))
        print(f"Devices: {num_devices}")

        print("Loading model...")
        t0 = time.perf_counter()
        self.pipe = DiffusionPipeline.from_pretrained(weights_dir, torch_dtype=torch.bfloat16)
        self.transformer = self.pipe.transformer.eval()
        self.config = self.transformer.config
        self.vae_sf = 8
        print(f"  Model loaded: {time.perf_counter()-t0:.1f}s")

        align = self.vae_sf * self.config.patch_size
        self.width = (width // align) * align
        self.height = (height // align) * align
        self.lh = self.height // self.vae_sf
        self.lw = self.width // self.vae_sf
        self.nc = self.config.in_channels // 4
        self.img_shapes = [[(1, self.lh // 2, self.lw // 2)]]
        self.max_txt_len = 128

        self._precompute_rope()

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

        # Compile ONE block function — XLA caches by HLO hash, so all 60
        # blocks with identical structure reuse the same compiled program.
        print("Compiling shared block...")
        t0 = time.perf_counter()

        # Use block 0 as the template. The closure captures block 0's weights,
        # but we'll swap which block we call at runtime.
        self.block_fns = []
        first_block = self.transformer.transformer_blocks[0]
        def make_fn(b):
            def fn(hs, ehs, em, temb, rope):
                return patched_block_forward(b, hs, ehs, em, temb, rope)
            return fn

        # Compile just ONE function for the first block
        self.compiled_template = torch.compile(make_fn(first_block), backend="tt")

        # For remaining blocks, wrap them with the same function signature.
        # torch.compile will see the same ops/shapes and should reuse
        # the cached HLO compilation.
        self.compiled_blocks = [self.compiled_template]
        for block in self.transformer.transformer_blocks[1:]:
            self.compiled_blocks.append(torch.compile(make_fn(block), backend="tt"))

        print(f"  {len(self.compiled_blocks)} block functions registered ({time.perf_counter()-t0:.1f}s)")

        self._precompute_scheduler()

        # Warmup: only need to compile once (first invocation), rest reuse cache
        print("Warming up (first block triggers compilation, rest reuse cache)...")
        t0 = time.perf_counter()
        self._generate_image("warmup", seed=0)
        t1 = time.perf_counter()
        print(f"  Warmup CFG: {t1-t0:.1f}s")
        old_cfg = self.cfg_scale
        self.cfg_scale = 1.0
        self._generate_image("warmup", negative_prompt=None, seed=0)
        self.cfg_scale = old_cfg
        print(f"  Warmup no-CFG: {time.perf_counter()-t1:.1f}s")

        print(f"\nReady! Generating {self.width}x{self.height} images, "
              f"{self.num_steps} steps, cfg={self.cfg_scale}")

    def _precompute_rope(self):
        img_fc, txt_fc = self.transformer.pos_embed(
            self.img_shapes, [self.max_txt_len], device=torch.device("cpu"))
        img_rope = complex_to_real_rope(img_fc)
        txt_rope = complex_to_real_rope(txt_fc)
        self.rope_dev = (
            (img_rope[0].to(torch.bfloat16).to(self.device),
             img_rope[1].to(torch.bfloat16).to(self.device)),
            (txt_rope[0].to(torch.bfloat16).to(self.device),
             txt_rope[1].to(torch.bfloat16).to(self.device)),
        )

    def _precompute_scheduler(self):
        scheduler = self.pipe.scheduler
        sigmas = np.linspace(1.0, 1 / self.num_steps, self.num_steps)
        img_seq_len = (self.lh // 2) * (self.lw // 2)
        mu = calculate_shift(img_seq_len)
        scheduler.set_timesteps(self.num_steps, sigmas=sigmas, mu=mu)
        scheduler.set_begin_index(0)
        self.timesteps = scheduler.timesteps.clone()
        self.sigmas_sched = scheduler.sigmas.clone()

    def _pad_text(self, embeds, mask):
        txt_len = embeds.shape[1]
        if txt_len >= self.max_txt_len:
            return embeds[:, :self.max_txt_len, :], mask[:, :self.max_txt_len]
        pad_len = self.max_txt_len - txt_len
        embeds = F.pad(embeds, (0, 0, 0, pad_len))
        mask = F.pad(mask, (0, pad_len), value=0)
        return embeds, mask

    def _generate_image(self, prompt, negative_prompt="", seed=None):
        if seed is None:
            seed = int(time.time() * 1000) % (2**31)

        do_cfg = self.cfg_scale > 1.0 and negative_prompt is not None

        t_enc = time.perf_counter()
        prompt_embeds, prompt_mask = self.pipe.encode_prompt(prompt=prompt, device="cpu")
        prompt_embeds, prompt_mask = self._pad_text(prompt_embeds, prompt_mask)
        if do_cfg:
            neg_embeds, neg_mask = self.pipe.encode_prompt(
                prompt=negative_prompt or "", device="cpu")
            neg_embeds, neg_mask = self._pad_text(neg_embeds, neg_mask)
            all_embeds = torch.cat([prompt_embeds, neg_embeds], dim=0)
            all_masks = torch.cat([prompt_mask, neg_mask], dim=0)
        t_enc = time.perf_counter() - t_enc

        gen = torch.Generator().manual_seed(seed)
        latents = pack_latents(
            torch.randn(1, self.nc, self.lh, self.lw, generator=gen, dtype=torch.bfloat16),
            1, self.nc, self.lh, self.lw,
        )

        latents_d = latents.to(self.device)
        if do_cfg:
            embeds_d = all_embeds.to(self.device)
            masks_d = all_masks.to(torch.bfloat16).to(self.device)
        else:
            embeds_d = prompt_embeds.to(self.device)
            masks_d = prompt_mask.to(torch.bfloat16).to(self.device)

        t_denoise = time.perf_counter()

        for i in range(len(self.timesteps)):
            t_val = self.timesteps[i]
            ts = t_val.expand(1).to(torch.bfloat16).to(self.device) / 1000

            with torch.no_grad():
                if do_cfg:
                    lat_batch = latents_d.expand(2, -1, -1)
                    ts_batch = ts.expand(2)
                    hs = self.transformer.img_in(lat_batch)
                    ehs = self.transformer.txt_in(self.transformer.txt_norm(embeds_d))
                    temb = self.transformer.time_text_embed(ts_batch.to(hs.dtype), hs)

                    for cb in self.compiled_blocks:
                        ehs, hs = cb(hs, ehs, masks_d, temb, self.rope_dev)

                    hs = self.transformer.norm_out(hs, temb)
                    noise_both = self.transformer.proj_out(hs)
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

                sigma = self.sigmas_sched[i]
                sigma_next = self.sigmas_sched[i + 1]
                dt = (sigma_next - sigma).to(torch.bfloat16).to(self.device)
                latents_d = latents_d + dt * noise_pred

            if (i + 1) % 5 == 0 or i == len(self.timesteps) - 1:
                torch_xla.sync()

        t_denoise = time.perf_counter() - t_denoise

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

        self.img_count += 1
        filename = f"demo_{self.img_count:03d}.png"
        image = (torch.clamp(image / 2 + 0.5, 0.0, 1.0) * 255).to(torch.uint8)
        Image.fromarray(image[0].cpu().numpy().transpose(1, 2, 0)).save(filename)

        return filename, seed, t_enc, t_denoise, t_vae

    def run_interactive(self):
        print("\n" + "=" * 60)
        print("  Qwen-Image 20B — Shared Compilation Demo")
        print("=" * 60)
        print(f"  Resolution: {self.width}x{self.height}")
        print(f"  Steps: {self.num_steps}, CFG: {self.cfg_scale}")
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

            seed = None
            cfg = self.cfg_scale
            steps = self.num_steps
            parts = user_input.split()
            remaining = []
            for part in parts:
                if part.startswith("seed:"):
                    try: seed = int(part.split(":", 1)[1])
                    except ValueError: remaining.append(part)
                elif part.startswith("cfg:"):
                    try: cfg = float(part.split(":", 1)[1])
                    except ValueError: remaining.append(part)
                elif part.startswith("steps:"):
                    try: steps = int(part.split(":", 1)[1])
                    except ValueError: remaining.append(part)
                else:
                    remaining.append(part)
            prompt = " ".join(remaining)
            if not prompt:
                continue

            old_steps, old_cfg = self.num_steps, self.cfg_scale
            self.num_steps, self.cfg_scale = steps, cfg
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

            self.num_steps, self.cfg_scale = old_steps, old_cfg
            if steps != old_steps:
                self._precompute_scheduler()


def main():
    p = argparse.ArgumentParser()
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
