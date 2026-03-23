"""Test fix: replace F.scaled_dot_product_attention with manual matmul+softmax.

The TT compiler's SDPA lowering produces wrong results for image tokens.
Test if decomposed attention works correctly.

Run with: ./run.sh debug_sdpa_fix.py
"""

import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch_xla
import torch_xla.runtime as xr

os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
xr.use_spmd()
xr.set_device_type("TT")

from diffusers import DiffusionPipeline
from utils.image_utils import calculate_shift, pack_latents, unpack_latents
from utils.profiling_utils import compute_pcc


def manual_attention(q, k, v):
    """Manual SDPA: matmul + softmax + matmul. q/k/v: [B, H, S, D]."""
    scale = 1.0 / math.sqrt(q.shape[-1])
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn_weights = torch.softmax(attn_weights, dim=-1)
    return torch.matmul(attn_weights, v)


def complex_to_real_rope(complex_freqs):
    cos = complex_freqs.real.float().repeat_interleave(2, dim=-1).unsqueeze(1)
    sin = complex_freqs.imag.float().repeat_interleave(2, dim=-1).unsqueeze(1)
    return (cos, sin)


def apply_rope_real(x, cos, sin):
    x_r, x_i = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
    x_rot = torch.stack([-x_i, x_r], dim=-1).flatten(3)
    return (x.float() * cos + x_rot.float() * sin).to(x.dtype)


def patched_block_forward_manual_attn(block, hidden_states, encoder_hidden_states,
                                       encoder_hidden_states_mask, temb, image_rotary_emb):
    """Block forward with decomposed attention (no F.scaled_dot_product_attention)."""
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

    # === KEY CHANGE: manual attention instead of F.scaled_dot_product_attention ===
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


def main():
    weights_dir = "./weights/qwen-image"
    device = torch_xla.device()
    num_devices = xr.global_runtime_device_count()
    print(f"TT Devices: {num_devices}")

    print("Loading model...")
    pipe = DiffusionPipeline.from_pretrained(weights_dir, torch_dtype=torch.bfloat16)
    transformer = pipe.transformer.eval()
    config = transformer.config

    width, height = 256, 256
    vae_sf = 8
    lh, lw = height // vae_sf, width // vae_sf
    nc = config.in_channels // 4

    prompt = "A cat"
    prompt_embeds, prompt_mask = pipe.encode_prompt(prompt=prompt, device="cpu")
    gen = torch.Generator().manual_seed(42)
    latents = pack_latents(
        torch.randn(1, nc, lh, lw, generator=gen, dtype=torch.bfloat16), 1, nc, lh, lw,
    )
    img_shapes = [[(1, lh // 2, lw // 2)]]
    txt_seq_lens = prompt_mask.sum(dim=1).tolist()
    timestep_div = torch.tensor([999.0], dtype=torch.bfloat16) / 1000

    # RoPE
    img_fc, txt_fc = transformer.pos_embed(img_shapes, txt_seq_lens, device=torch.device("cpu"))
    img_rope = complex_to_real_rope(img_fc)
    txt_rope = complex_to_real_rope(txt_fc)
    rope_cpu = (img_rope, txt_rope)

    # Prepare block inputs on CPU
    with torch.no_grad():
        hs_in = transformer.img_in(latents)
        ehs_in = transformer.txt_in(transformer.txt_norm(prompt_embeds))
        temb = transformer.time_text_embed(timestep_div.to(hs_in.dtype), hs_in)

    block = transformer.transformer_blocks[0]
    attn = block.attn

    # Build Q/K/V for SDPA test
    with torch.no_grad():
        img_mod1 = block.img_mod(temb)[:, :block.dim * 3]
        txt_mod1 = block.txt_mod(temb)[:, :block.dim * 3]

        def mod_fn(x, mod):
            s, sc, g = mod.chunk(3, dim=-1)
            return x * (1 + sc.unsqueeze(1)) + s.unsqueeze(1)

        img_normed = block.img_norm1(hs_in)
        img_modulated = mod_fn(img_normed, img_mod1)
        txt_normed = block.txt_norm1(ehs_in)
        txt_modulated = mod_fn(txt_normed, txt_mod1)

        img_q = attn.to_q(img_modulated).unflatten(-1, (attn.heads, -1))
        img_k = attn.to_k(img_modulated).unflatten(-1, (attn.heads, -1))
        img_v = attn.to_v(img_modulated).unflatten(-1, (attn.heads, -1))
        txt_q = attn.add_q_proj(txt_modulated).unflatten(-1, (attn.heads, -1))
        txt_k = attn.add_k_proj(txt_modulated).unflatten(-1, (attn.heads, -1))
        txt_v = attn.add_v_proj(txt_modulated).unflatten(-1, (attn.heads, -1))

        if attn.norm_q is not None:
            img_q = attn.norm_q(img_q)
            img_k = attn.norm_k(img_k)
            txt_q = attn.norm_added_q(txt_q)
            txt_k = attn.norm_added_k(txt_k)

        q = torch.cat([txt_q, img_q], dim=1)
        k = torch.cat([txt_k, img_k], dim=1)
        v = torch.cat([txt_v, img_v], dim=1)

    txt_seq = ehs_in.shape[1]  # 7
    print(f"Q/K/V shape: {list(q.shape)}")

    # CPU reference for SDPA
    with torch.no_grad():
        sdpa_ref = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            dropout_p=0.0, is_causal=False,
        ).transpose(1, 2)

    # ============================================================
    # TEST 1: Manual attention (matmul+softmax) on CPU — verify equivalence
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 1: Manual attention vs SDPA on CPU")
    print("=" * 60)

    with torch.no_grad():
        manual_cpu = manual_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        ).transpose(1, 2)

    pcc_manual_cpu = compute_pcc(manual_cpu, sdpa_ref)
    print(f"  PCC (manual vs SDPA on CPU): {pcc_manual_cpu:.6f}")

    # ============================================================
    # TEST 2: Manual attention compiled on TT device
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 2: Manual attention on TT device")
    print("=" * 60)

    def manual_attn_fn(q, k, v):
        return manual_attention(q, k, v)

    c_manual = torch.compile(manual_attn_fn, backend="tt")
    with torch.no_grad():
        out_manual_d = c_manual(
            q.transpose(1, 2).to(device),
            k.transpose(1, 2).to(device),
            v.transpose(1, 2).to(device),
        )
    torch_xla.sync()
    out_manual_d_cpu = out_manual_d.cpu().transpose(1, 2)

    pcc_all = compute_pcc(out_manual_d_cpu, sdpa_ref)
    pcc_txt = compute_pcc(out_manual_d_cpu[:, :txt_seq], sdpa_ref[:, :txt_seq])
    pcc_img = compute_pcc(out_manual_d_cpu[:, txt_seq:], sdpa_ref[:, txt_seq:])
    print(f"  Manual attn PCC (all): {pcc_all:.6f}")
    print(f"  Manual attn PCC (txt): {pcc_txt:.6f}")
    print(f"  Manual attn PCC (img): {pcc_img:.6f}")

    # ============================================================
    # TEST 3: F.sdpa compiled on TT (confirm it's broken)
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 3: F.scaled_dot_product_attention on TT (confirm broken)")
    print("=" * 60)

    def sdpa_fn(q, k, v):
        return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)

    c_sdpa = torch.compile(sdpa_fn, backend="tt")
    with torch.no_grad():
        out_sdpa_d = c_sdpa(
            q.transpose(1, 2).to(device),
            k.transpose(1, 2).to(device),
            v.transpose(1, 2).to(device),
        )
    torch_xla.sync()
    out_sdpa_d_cpu = out_sdpa_d.cpu().transpose(1, 2)

    pcc_sdpa_all = compute_pcc(out_sdpa_d_cpu, sdpa_ref)
    pcc_sdpa_txt = compute_pcc(out_sdpa_d_cpu[:, :txt_seq], sdpa_ref[:, :txt_seq])
    pcc_sdpa_img = compute_pcc(out_sdpa_d_cpu[:, txt_seq:], sdpa_ref[:, txt_seq:])
    print(f"  SDPA PCC (all): {pcc_sdpa_all:.6f}")
    print(f"  SDPA PCC (txt): {pcc_sdpa_txt:.6f}")
    print(f"  SDPA PCC (img): {pcc_sdpa_img:.6f}")

    if pcc_img > 0.99:
        # ============================================================
        # TEST 4: Single block with manual attention on device
        # ============================================================
        print("\n" + "=" * 60)
        print("TEST 4: Single block with manual attention on device")
        print("=" * 60)

        # CPU reference
        with torch.no_grad():
            txt_ref, img_ref = patched_block_forward_manual_attn(
                block, hs_in.clone(), ehs_in.clone(),
                prompt_mask.to(torch.bfloat16), temb.clone(), rope_cpu,
            )

        block_dev = block.to(device)
        def make_fn(b):
            def fn(hs, ehs, em, temb, rope):
                return patched_block_forward_manual_attn(b, hs, ehs, em, temb, rope)
            return fn

        compiled_b = torch.compile(make_fn(block_dev), backend="tt")
        rope_dev = (
            (img_rope[0].to(torch.bfloat16).to(device), img_rope[1].to(torch.bfloat16).to(device)),
            (txt_rope[0].to(torch.bfloat16).to(device), txt_rope[1].to(torch.bfloat16).to(device)),
        )

        with torch.no_grad():
            txt_d, img_d = compiled_b(
                hs_in.to(device), ehs_in.to(device),
                prompt_mask.to(torch.bfloat16).to(device), temb.to(device), rope_dev,
            )
        torch_xla.sync()

        pcc_block_img = compute_pcc(img_d.cpu(), img_ref)
        pcc_block_txt = compute_pcc(txt_d.cpu(), txt_ref)
        print(f"  Block img PCC: {pcc_block_img:.6f}")
        print(f"  Block txt PCC: {pcc_block_txt:.6f}")

        block = block.cpu()
        transformer.transformer_blocks[0] = block

        if pcc_block_img > 0.99:
            # ============================================================
            # TEST 5: Full pipeline with manual attention — generate image!
            # ============================================================
            print("\n" + "=" * 60)
            print("TEST 5: Full pipeline — 20 steps with manual attention on device")
            print("=" * 60)

            transformer = transformer.to(device)

            compiled_blocks = []
            for idx, blk in enumerate(transformer.transformer_blocks):
                compiled_blocks.append(torch.compile(make_fn(blk), backend="tt"))
            print(f"  Compiled {len(compiled_blocks)} blocks")

            num_steps = 20
            scheduler = pipe.scheduler
            sigmas = np.linspace(1.0, 1 / num_steps, num_steps)
            gen2 = torch.Generator().manual_seed(42)
            lat = pack_latents(
                torch.randn(1, nc, lh, lw, generator=gen2, dtype=torch.bfloat16), 1, nc, lh, lw,
            )
            mu = calculate_shift(lat.shape[1])
            scheduler.set_timesteps(num_steps, sigmas=sigmas, mu=mu)
            scheduler.set_begin_index(0)

            lat_d = lat.to(device)
            pe_d = prompt_embeds.to(device)
            pm_d = prompt_mask.to(torch.bfloat16).to(device)

            print(f"  Denoising {num_steps} steps...")
            t0 = time.perf_counter()

            for i, t_val in enumerate(scheduler.timesteps):
                ts = t_val.expand(1).to(torch.bfloat16).to(device) / 1000

                with torch.no_grad():
                    hs = transformer.img_in(lat_d)
                    ehs = transformer.txt_in(transformer.txt_norm(pe_d))
                    temb_d = transformer.time_text_embed(ts.to(hs.dtype), hs)

                    for cb in compiled_blocks:
                        ehs, hs = cb(hs, ehs, pm_d, temb_d, rope_dev)

                    hs = transformer.norm_out(hs, temb_d)
                    noise_pred = transformer.proj_out(hs)

                lat_cpu = lat_d.cpu()
                np_cpu = noise_pred.cpu()
                lat_d = scheduler.step(np_cpu, t_val, lat_cpu, return_dict=False)[0].to(device)
                torch_xla.sync()

                if (i + 1) % 5 == 0 or i == 0:
                    el = time.perf_counter() - t0
                    print(f"    Step {i+1}/{num_steps} ({el:.1f}s)")

            print(f"  Denoising: {time.perf_counter()-t0:.1f}s")

            # VAE decode on CPU
            lf = lat_d.cpu()
            lf = unpack_latents(lf, height, width, vae_sf).to(pipe.vae.dtype)
            if hasattr(pipe.vae.config, "latents_mean") and pipe.vae.config.latents_mean is not None:
                lm = torch.tensor(pipe.vae.config.latents_mean).view(1, pipe.vae.config.z_dim, 1, 1, 1).to(lf.device, lf.dtype)
                ls = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(1, pipe.vae.config.z_dim, 1, 1, 1).to(lf.device, lf.dtype)
                lf = lf / ls + lm

            with torch.no_grad():
                image = pipe.vae.decode(lf, return_dict=False)[0]
                if image.dim() == 5:
                    image = image[:, :, 0]

            from PIL import Image as PILImage
            image = (torch.clamp(image / 2 + 0.5, 0.0, 1.0) * 255).to(torch.uint8)
            PILImage.fromarray(image[0].cpu().numpy().transpose(1, 2, 0)).save("output_manual_attn_device.png")
            print("  Saved: output_manual_attn_device.png")

    print("\nDone!")


if __name__ == "__main__":
    main()
