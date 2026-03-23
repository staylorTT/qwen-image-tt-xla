"""Device debug: find where TT compilation diverges from CPU.

Tests each component individually on TT device vs CPU reference:
  1. Input projections (img_in, txt_in, txt_norm)
  2. time_text_embed
  3. Single block WITHOUT RoPE
  4. Single block WITH RoPE
  5. Full transformer forward (1 step)
  6. Multi-step denoising

Run with: ./run.sh debug_device_compare.py
"""

import os
import sys
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


def complex_to_real_rope(complex_freqs):
    cos = complex_freqs.real.float().repeat_interleave(2, dim=-1).unsqueeze(1)
    sin = complex_freqs.imag.float().repeat_interleave(2, dim=-1).unsqueeze(1)
    return (cos, sin)


def apply_rope_real(x, cos, sin):
    x_r, x_i = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
    x_rot = torch.stack([-x_i, x_r], dim=-1).flatten(3)
    return (x.float() * cos + x_rot.float() * sin).to(x.dtype)


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

    out = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        dropout_p=0.0, is_causal=False,
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

    print("Loading model (bf16, CPU)...")
    pipe = DiffusionPipeline.from_pretrained(weights_dir, torch_dtype=torch.bfloat16)
    transformer = pipe.transformer.eval()
    config = transformer.config
    hidden_dim = config.num_attention_heads * config.attention_head_dim

    # Small resolution
    width, height = 256, 256
    vae_sf = 8
    lh, lw = height // vae_sf, width // vae_sf
    nc = config.in_channels // 4

    # Prepare inputs
    prompt = "A cat"
    prompt_embeds, prompt_mask = pipe.encode_prompt(prompt=prompt, device="cpu")
    gen = torch.Generator().manual_seed(42)
    latents = pack_latents(
        torch.randn(1, nc, lh, lw, generator=gen, dtype=torch.bfloat16), 1, nc, lh, lw,
    )
    img_shapes = [[(1, lh // 2, lw // 2)]]
    txt_seq_lens = prompt_mask.sum(dim=1).tolist()
    timestep_div = torch.tensor([999.0], dtype=torch.bfloat16) / 1000

    # Pre-compute RoPE
    img_fc, txt_fc = transformer.pos_embed(img_shapes, txt_seq_lens, device=torch.device("cpu"))
    img_rope_real = complex_to_real_rope(img_fc)
    txt_rope_real = complex_to_real_rope(txt_fc)

    print(f"Latents: {list(latents.shape)}, prompt: {list(prompt_embeds.shape)}")

    # ============================================================
    # TEST 1: Input projections on device
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 1: Input projections (img_in, txt_norm, txt_in) on device")
    print("=" * 60)

    # CPU reference
    with torch.no_grad():
        hs_cpu = transformer.img_in(latents.clone())
        ehs_cpu = transformer.txt_in(transformer.txt_norm(prompt_embeds.clone()))

    # Move just the input projection layers to device
    img_in_dev = transformer.img_in.to(device)
    txt_norm_dev = transformer.txt_norm.to(device)
    txt_in_dev = transformer.txt_in.to(device)

    compiled_img_in = torch.compile(img_in_dev, backend="tt")
    compiled_txt_norm = torch.compile(txt_norm_dev, backend="tt")
    compiled_txt_in = torch.compile(txt_in_dev, backend="tt")

    with torch.no_grad():
        hs_dev = compiled_img_in(latents.to(device))
        torch_xla.sync()
        normed = compiled_txt_norm(prompt_embeds.to(device))
        torch_xla.sync()
        ehs_dev = compiled_txt_in(normed)
        torch_xla.sync()

    pcc_hs = compute_pcc(hs_dev.cpu(), hs_cpu)
    pcc_ehs = compute_pcc(ehs_dev.cpu(), ehs_cpu)
    print(f"  img_in PCC:  {pcc_hs:.6f}")
    print(f"  txt_in PCC:  {pcc_ehs:.6f}")

    # Move back to CPU for later tests
    transformer.img_in = transformer.img_in.cpu()
    transformer.txt_norm = transformer.txt_norm.cpu()
    transformer.txt_in = transformer.txt_in.cpu()

    # ============================================================
    # TEST 2: time_text_embed on device
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 2: time_text_embed on device")
    print("=" * 60)

    with torch.no_grad():
        temb_cpu = transformer.time_text_embed(timestep_div.to(hs_cpu.dtype), hs_cpu)

    tte_dev = transformer.time_text_embed.to(device)
    compiled_tte = torch.compile(tte_dev, backend="tt")

    with torch.no_grad():
        temb_dev = compiled_tte(timestep_div.to(torch.bfloat16).to(device), hs_cpu.to(device))
        torch_xla.sync()

    pcc_temb = compute_pcc(temb_dev.cpu(), temb_cpu)
    print(f"  temb PCC: {pcc_temb:.6f}")
    transformer.time_text_embed = transformer.time_text_embed.cpu()

    # ============================================================
    # TEST 3: Single block WITHOUT RoPE on device
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 3: Single block WITHOUT RoPE on device")
    print("=" * 60)

    with torch.no_grad():
        temb_cpu = transformer.time_text_embed(timestep_div.to(hs_cpu.dtype), hs_cpu)

    block0 = transformer.transformer_blocks[0]

    with torch.no_grad():
        txt_ref, img_ref = block0(
            hidden_states=hs_cpu.clone(), encoder_hidden_states=ehs_cpu.clone(),
            encoder_hidden_states_mask=prompt_mask.to(torch.bfloat16),
            temb=temb_cpu.clone(), image_rotary_emb=None,
        )

    block0_dev = block0.to(device)
    compiled_block = torch.compile(block0_dev, backend="tt")

    t0 = time.perf_counter()
    with torch.no_grad():
        txt_dev, img_dev = compiled_block(
            hidden_states=hs_cpu.to(device), encoder_hidden_states=ehs_cpu.to(device),
            encoder_hidden_states_mask=prompt_mask.to(torch.bfloat16).to(device),
            temb=temb_cpu.to(device), image_rotary_emb=None,
        )
    torch_xla.sync()
    print(f"  Compile+run: {time.perf_counter()-t0:.1f}s")

    pcc_img = compute_pcc(img_dev.cpu(), img_ref)
    pcc_txt = compute_pcc(txt_dev.cpu(), txt_ref)
    print(f"  img PCC (no RoPE): {pcc_img:.6f}")
    print(f"  txt PCC (no RoPE): {pcc_txt:.6f}")

    block0 = block0.cpu()
    transformer.transformer_blocks[0] = block0

    # ============================================================
    # TEST 4: Single block WITH RoPE on device (patched forward)
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 4: Single block WITH RoPE on device (patched forward)")
    print("=" * 60)

    rope_cpu = (img_rope_real, txt_rope_real)
    with torch.no_grad():
        txt_ref_r, img_ref_r = patched_block_forward(
            block0, hs_cpu.clone(), ehs_cpu.clone(),
            prompt_mask.to(torch.bfloat16), temb_cpu.clone(), rope_cpu,
        )

    block0_dev = block0.to(device)

    def make_block_fn(b):
        def fn(hs, ehs, em, temb, rope):
            return patched_block_forward(b, hs, ehs, em, temb, rope)
        return fn

    compiled_block_rope = torch.compile(make_block_fn(block0_dev), backend="tt")

    rope_dev = (
        (img_rope_real[0].to(torch.bfloat16).to(device), img_rope_real[1].to(torch.bfloat16).to(device)),
        (txt_rope_real[0].to(torch.bfloat16).to(device), txt_rope_real[1].to(torch.bfloat16).to(device)),
    )

    t0 = time.perf_counter()
    with torch.no_grad():
        txt_dev_r, img_dev_r = compiled_block_rope(
            hs_cpu.to(device), ehs_cpu.to(device),
            prompt_mask.to(torch.bfloat16).to(device), temb_cpu.to(device),
            rope_dev,
        )
    torch_xla.sync()
    print(f"  Compile+run: {time.perf_counter()-t0:.1f}s")

    pcc_img_r = compute_pcc(img_dev_r.cpu(), img_ref_r)
    pcc_txt_r = compute_pcc(txt_dev_r.cpu(), txt_ref_r)
    print(f"  img PCC (with RoPE): {pcc_img_r:.6f}")
    print(f"  txt PCC (with RoPE): {pcc_txt_r:.6f}")

    if pcc_img_r < 0.99:
        print("  >>> Checking RoPE application in isolation...")
        # Compare RoPE alone
        test_q = torch.randn(1, 256, 24, 128, dtype=torch.bfloat16)
        cos_cpu = img_rope_real[0]
        sin_cpu = img_rope_real[1]
        ref_out = apply_rope_real(test_q, cos_cpu, sin_cpu)

        def rope_fn(x, cos, sin):
            return apply_rope_real(x, cos, sin)
        compiled_rope = torch.compile(rope_fn, backend="tt")

        with torch.no_grad():
            dev_out = compiled_rope(
                test_q.to(device),
                cos_cpu.to(torch.bfloat16).to(device),
                sin_cpu.to(torch.bfloat16).to(device),
            )
        torch_xla.sync()
        pcc_rope = compute_pcc(dev_out.cpu(), ref_out)
        print(f"  RoPE-only PCC: {pcc_rope:.6f}")

    block0 = block0.cpu()
    transformer.transformer_blocks[0] = block0

    # ============================================================
    # TEST 5: Stack 3 blocks with RoPE
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 5: Stack 3 blocks with RoPE on device")
    print("=" * 60)

    n_test = 3
    # CPU ref
    hs_c, ehs_c = hs_cpu.clone(), ehs_cpu.clone()
    with torch.no_grad():
        for i in range(n_test):
            ehs_c, hs_c = patched_block_forward(
                transformer.transformer_blocks[i], hs_c, ehs_c,
                prompt_mask.to(torch.bfloat16), temb_cpu.clone(), rope_cpu,
            )

    # Device
    compiled_blocks = []
    for i in range(n_test):
        b = transformer.transformer_blocks[i].to(device)
        compiled_blocks.append(torch.compile(make_block_fn(b), backend="tt"))

    hs_d = hs_cpu.to(device)
    ehs_d = ehs_cpu.to(device)
    em_d = prompt_mask.to(torch.bfloat16).to(device)
    temb_d = temb_cpu.to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        for cb in compiled_blocks:
            ehs_d, hs_d = cb(hs_d, ehs_d, em_d, temb_d, rope_dev)
    torch_xla.sync()
    print(f"  Compile+run: {time.perf_counter()-t0:.1f}s")

    pcc_3 = compute_pcc(hs_d.cpu(), hs_c)
    print(f"  3-block img PCC: {pcc_3:.6f}")

    for i in range(n_test):
        transformer.transformer_blocks[i] = transformer.transformer_blocks[i].cpu()

    # ============================================================
    # TEST 6: Full denoising — 1 step comparison
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 6: Full transformer — 1 denoising step on device vs CPU")
    print("=" * 60)

    # CPU ref: full manual forward
    with torch.no_grad():
        hs_full = transformer.img_in(latents.clone())
        ehs_full = transformer.txt_in(transformer.txt_norm(prompt_embeds.clone()))
        temb_full = transformer.time_text_embed(timestep_div.to(hs_full.dtype), hs_full)
        for block in transformer.transformer_blocks:
            ehs_full, hs_full = patched_block_forward(
                block, hs_full, ehs_full,
                prompt_mask.to(torch.bfloat16), temb_full, rope_cpu,
            )
        hs_full = transformer.norm_out(hs_full, temb_full)
        out_cpu = transformer.proj_out(hs_full)

    # Device: compile each block, run the full pipeline
    print("  Moving transformer to device...")
    transformer = transformer.to(device)

    # Compile blocks
    dev_compiled = []
    for idx, block in enumerate(transformer.transformer_blocks):
        dev_compiled.append(torch.compile(make_block_fn(block), backend="tt"))
    print(f"  Compiled {len(dev_compiled)} blocks")

    lat_d = latents.to(device)
    pe_d = prompt_embeds.to(device)
    pm_d = prompt_mask.to(torch.bfloat16).to(device)
    ts_d = timestep_div.to(torch.bfloat16).to(device)

    print("  Running full forward on device...")
    t0 = time.perf_counter()
    with torch.no_grad():
        hs_d2 = transformer.img_in(lat_d)
        ehs_d2 = transformer.txt_in(transformer.txt_norm(pe_d))
        temb_d2 = transformer.time_text_embed(ts_d.to(hs_d2.dtype), hs_d2)
        torch_xla.sync()

        for i, cb in enumerate(dev_compiled):
            ehs_d2, hs_d2 = cb(hs_d2, ehs_d2, pm_d, temb_d2, rope_dev)
            if i == 0:
                torch_xla.sync()
                # Check first block intermediate
                first_pcc = compute_pcc(hs_d2.cpu(), None) if False else "skipped"

        torch_xla.sync()
        hs_d2 = transformer.norm_out(hs_d2, temb_d2)
        out_dev = transformer.proj_out(hs_d2)
        torch_xla.sync()

    dt = time.perf_counter() - t0
    print(f"  Device forward: {dt:.1f}s")

    pcc_full = compute_pcc(out_dev.cpu(), out_cpu)
    print(f"  Full forward PCC: {pcc_full:.6f}")
    print(f"  CPU stats:  mean={out_cpu.float().mean():.4f}, std={out_cpu.float().std():.4f}")
    print(f"  Dev stats:  mean={out_dev.cpu().float().mean():.4f}, std={out_dev.cpu().float().std():.4f}")

    has_nan = torch.isnan(out_dev.cpu()).any().item()
    has_inf = torch.isinf(out_dev.cpu()).any().item()
    print(f"  Device output: nan={has_nan}, inf={has_inf}")

    print("\nDone!")


if __name__ == "__main__":
    main()
