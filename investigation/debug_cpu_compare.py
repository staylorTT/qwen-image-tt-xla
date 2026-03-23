"""CPU-only debug: compare HF transformer.forward() vs our manual forward.

This script does NOT use TT hardware. It isolates whether the bug is in:
  (a) our manual forward logic (RoPE conversion, block patching, etc.)
  (b) the TT device compilation

Run with: ./run.sh debug_cpu_compare.py
"""

import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DiffusionPipeline

from utils.image_utils import calculate_shift, pack_latents, unpack_latents
from utils.profiling_utils import compute_pcc


def complex_to_real_rope(complex_freqs):
    """Convert complex RoPE freqs to (cos, sin) — same as generate_image_v2.py."""
    cos = complex_freqs.real.float().repeat_interleave(2, dim=-1).unsqueeze(1)
    sin = complex_freqs.imag.float().repeat_interleave(2, dim=-1).unsqueeze(1)
    return (cos, sin)


def apply_rope_real(x, cos, sin):
    """Apply rotary embedding using real arithmetic. x: [B,S,H,D]."""
    x_r, x_i = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
    x_rot = torch.stack([-x_i, x_r], dim=-1).flatten(3)
    return (x.float() * cos + x_rot.float() * sin).to(x.dtype)


def patched_block_forward(block, hidden_states, encoder_hidden_states,
                          encoder_hidden_states_mask, temb, image_rotary_emb):
    """Forward through one MMDiT block with real-valued RoPE — same as generate_image_v2.py."""
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


def manual_forward(transformer, latents, prompt_embeds, prompt_mask, timestep_val,
                   img_shapes, txt_seq_lens):
    """Our manual forward path — same logic as generate_image_v2.py."""
    config = transformer.config

    # Pre-compute RoPE on CPU, convert to real
    img_fc, txt_fc = transformer.pos_embed(img_shapes, txt_seq_lens, device=torch.device("cpu"))
    img_rope = complex_to_real_rope(img_fc)
    txt_rope = complex_to_real_rope(txt_fc)
    rope = (img_rope, txt_rope)

    # Input projections
    hs = transformer.img_in(latents)
    ehs = transformer.txt_in(transformer.txt_norm(prompt_embeds))

    # Timestep embedding
    ts = timestep_val.to(hs.dtype)
    temb = transformer.time_text_embed(ts, hs)

    # Run all blocks with our patched forward
    for block in transformer.transformer_blocks:
        ehs, hs = patched_block_forward(block, hs, ehs, prompt_mask, temb, rope)

    # Output
    hs = transformer.norm_out(hs, temb)
    output = transformer.proj_out(hs)
    return output


def hf_forward(transformer, latents, prompt_embeds, prompt_mask, timestep_val,
               img_shapes, txt_seq_lens):
    """Gold-standard HF forward — calls transformer.forward() directly."""
    output = transformer(
        hidden_states=latents,
        timestep=timestep_val,
        encoder_hidden_states=prompt_embeds,
        encoder_hidden_states_mask=prompt_mask,
        img_shapes=img_shapes,
        txt_seq_lens=txt_seq_lens,
        return_dict=False,
    )[0]
    return output


def main():
    weights_dir = "./weights/qwen-image"

    print("Loading model (bf16, CPU)...")
    pipe = DiffusionPipeline.from_pretrained(weights_dir, torch_dtype=torch.bfloat16)
    transformer = pipe.transformer.eval()
    config = transformer.config

    # Use small resolution for speed
    width, height = 256, 256
    vae_sf = 8
    align = vae_sf * config.patch_size
    width = (width // align) * align
    height = (height // align) * align

    print(f"Resolution: {width}x{height}")

    # Text encode
    prompt = "A cat"
    prompt_embeds, prompt_mask = pipe.encode_prompt(prompt=prompt, device="cpu")
    print(f"Prompt embeds: {list(prompt_embeds.shape)}, mask: {list(prompt_mask.shape)}")

    # Latents
    lh, lw = height // vae_sf, width // vae_sf
    nc = config.in_channels // 4
    gen = torch.Generator().manual_seed(42)
    latents = pack_latents(
        torch.randn(1, nc, lh, lw, generator=gen, dtype=torch.bfloat16),
        1, nc, lh, lw,
    )
    print(f"Latents: {list(latents.shape)}")

    img_shapes = [[(1, lh // 2, lw // 2)]]
    txt_seq_lens = prompt_mask.sum(dim=1).tolist()

    # Single timestep
    timestep_raw = torch.tensor([999.0], dtype=torch.bfloat16)
    timestep_div = timestep_raw / 1000  # Pipeline divides by 1000

    print(f"\nTimestep raw: {timestep_raw.item()}, divided: {timestep_div.item()}")

    # ============================================================
    # TEST 1: Compare single block with vs without RoPE
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 1: Single block — HF native vs our patched (with RoPE)")
    print("=" * 60)

    block = transformer.transformer_blocks[0]

    # Prepare inputs as transformer.forward would
    hs_in = transformer.img_in(latents.clone())
    ehs_in = transformer.txt_in(transformer.txt_norm(prompt_embeds.clone()))
    temb = transformer.time_text_embed(timestep_div.to(hs_in.dtype), hs_in)

    # HF native RoPE (complex)
    img_fc, txt_fc = transformer.pos_embed(img_shapes, txt_seq_lens, device=torch.device("cpu"))
    hf_rope = (img_fc, txt_fc)

    # Our real-valued RoPE
    our_rope = (complex_to_real_rope(img_fc), complex_to_real_rope(txt_fc))

    with torch.no_grad():
        # HF native block forward (uses complex RoPE internally)
        txt_hf, img_hf = block(
            hidden_states=hs_in.clone(),
            encoder_hidden_states=ehs_in.clone(),
            encoder_hidden_states_mask=prompt_mask.to(torch.bfloat16),
            temb=temb.clone(),
            image_rotary_emb=hf_rope,
        )

        # Our patched block forward (uses real RoPE)
        txt_ours, img_ours = patched_block_forward(
            block, hs_in.clone(), ehs_in.clone(),
            prompt_mask.to(torch.bfloat16), temb.clone(), our_rope,
        )

    img_pcc = compute_pcc(img_ours, img_hf)
    txt_pcc = compute_pcc(txt_ours, txt_hf)
    print(f"  Image stream PCC: {img_pcc:.6f}")
    print(f"  Text stream PCC:  {txt_pcc:.6f}")

    if img_pcc < 0.99 or txt_pcc < 0.99:
        print("  >>> BUG FOUND: Single block diverges with RoPE!")
        # Deeper diagnosis: test without RoPE
        with torch.no_grad():
            txt_hf_nr, img_hf_nr = block(
                hidden_states=hs_in.clone(), encoder_hidden_states=ehs_in.clone(),
                encoder_hidden_states_mask=prompt_mask.to(torch.bfloat16),
                temb=temb.clone(), image_rotary_emb=None,
            )
            txt_ours_nr, img_ours_nr = patched_block_forward(
                block, hs_in.clone(), ehs_in.clone(),
                prompt_mask.to(torch.bfloat16), temb.clone(), None,
            )
        print(f"  Without RoPE - img PCC: {compute_pcc(img_ours_nr, img_hf_nr):.6f}")
        print(f"  Without RoPE - txt PCC: {compute_pcc(txt_ours_nr, txt_hf_nr):.6f}")

        # Test RoPE conversion itself
        from diffusers.models.transformers.transformer_qwenimage import apply_rotary_emb_qwen
        test_q = torch.randn(1, 16, 24, 128, dtype=torch.bfloat16)
        # Complex path
        q_complex = apply_rotary_emb_qwen(test_q.clone(), img_fc[:16], use_real=False)
        # Real path
        cos, sin = complex_to_real_rope(img_fc[:16])
        q_real = apply_rope_real(test_q.clone(), cos, sin)
        rope_pcc = compute_pcc(q_real, q_complex)
        print(f"  RoPE conversion PCC: {rope_pcc:.6f}")

        if rope_pcc < 0.999:
            print("  >>> ROOT CAUSE: RoPE real/complex conversion mismatch!")
            # Print shapes for debugging
            print(f"  img_fc shape: {img_fc.shape}")
            print(f"  cos shape: {cos.shape}, sin shape: {sin.shape}")
            print(f"  test_q shape: {test_q.shape}")
            print(f"  q_complex[:, 0, 0, :4]: {q_complex[0, 0, 0, :4]}")
            print(f"  q_real[:, 0, 0, :4]:    {q_real[0, 0, 0, :4]}")
    else:
        print("  PASS: Single block matches with RoPE")

    # ============================================================
    # TEST 2: Full transformer forward — HF vs manual
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 2: Full transformer forward — HF vs manual")
    print("=" * 60)

    with torch.no_grad():
        out_hf = hf_forward(transformer, latents.clone(), prompt_embeds.clone(),
                            prompt_mask, timestep_div.clone(), img_shapes, txt_seq_lens)
        out_manual = manual_forward(transformer, latents.clone(), prompt_embeds.clone(),
                                     prompt_mask.to(torch.bfloat16), timestep_div.clone(),
                                     img_shapes, txt_seq_lens)

    full_pcc = compute_pcc(out_manual, out_hf)
    print(f"  Full forward PCC: {full_pcc:.6f}")
    print(f"  HF output stats:     mean={out_hf.float().mean():.4f}, std={out_hf.float().std():.4f}")
    print(f"  Manual output stats: mean={out_manual.float().mean():.4f}, std={out_manual.float().std():.4f}")

    if full_pcc < 0.99:
        print("  >>> BUG FOUND: Full transformer diverges!")
        # Bisect: run increasing numbers of blocks
        for n_blocks in [1, 5, 10, 30, 60]:
            n_blocks = min(n_blocks, len(transformer.transformer_blocks))
            img_fc2, txt_fc2 = transformer.pos_embed(img_shapes, txt_seq_lens, device=torch.device("cpu"))
            rope2 = (complex_to_real_rope(img_fc2), complex_to_real_rope(txt_fc2))

            hs_h = transformer.img_in(latents.clone())
            ehs_h = transformer.txt_in(transformer.txt_norm(prompt_embeds.clone()))
            temb_h = transformer.time_text_embed(timestep_div.clone().to(hs_h.dtype), hs_h)

            hs_m = hs_h.clone()
            ehs_m = ehs_h.clone()
            temb_m = temb_h.clone()

            with torch.no_grad():
                for i in range(n_blocks):
                    block = transformer.transformer_blocks[i]
                    ehs_h, hs_h = block(
                        hidden_states=hs_h, encoder_hidden_states=ehs_h,
                        encoder_hidden_states_mask=prompt_mask.to(torch.bfloat16),
                        temb=temb_h, image_rotary_emb=(img_fc2, txt_fc2),
                    )
                    ehs_m, hs_m = patched_block_forward(
                        block, hs_m, ehs_m,
                        prompt_mask.to(torch.bfloat16), temb_m, rope2,
                    )

            pcc_n = compute_pcc(hs_m, hs_h)
            print(f"  After {n_blocks:2d} blocks — img PCC: {pcc_n:.6f}")
    else:
        print("  PASS: Full transformer forward matches!")

    # ============================================================
    # TEST 3: Full pipeline — manual forward, CPU decode
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 3: Generate image using manual forward (CPU only)")
    print("=" * 60)

    num_steps = 20
    scheduler = pipe.scheduler
    sigmas = np.linspace(1.0, 1 / num_steps, num_steps)
    mu = calculate_shift(latents.shape[1])
    scheduler.set_timesteps(num_steps, sigmas=sigmas, mu=mu)
    scheduler.set_begin_index(0)

    # Reset latents
    gen2 = torch.Generator().manual_seed(42)
    lat = pack_latents(
        torch.randn(1, nc, lh, lw, generator=gen2, dtype=torch.bfloat16),
        1, nc, lh, lw,
    )

    print(f"  Denoising {num_steps} steps...")
    t0 = time.perf_counter()

    for i, t_val in enumerate(scheduler.timesteps):
        ts = t_val.expand(1).to(torch.bfloat16) / 1000
        with torch.no_grad():
            noise_pred = manual_forward(
                transformer, lat.clone(), prompt_embeds.clone(),
                prompt_mask.to(torch.bfloat16), ts, img_shapes, txt_seq_lens,
            )
        lat = scheduler.step(noise_pred, t_val, lat, return_dict=False)[0]
        if (i + 1) % 5 == 0:
            print(f"    Step {i+1}/{num_steps} ({time.perf_counter()-t0:.1f}s)")

    print(f"  Denoising: {time.perf_counter()-t0:.1f}s")

    # VAE decode
    lf = unpack_latents(lat, height, width, vae_sf).to(pipe.vae.dtype)
    if hasattr(pipe.vae.config, "latents_mean") and pipe.vae.config.latents_mean is not None:
        lm = torch.tensor(pipe.vae.config.latents_mean).view(1, pipe.vae.config.z_dim, 1, 1, 1).to(lf.device, lf.dtype)
        ls = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(1, pipe.vae.config.z_dim, 1, 1, 1).to(lf.device, lf.dtype)
        lf = lf / ls + lm

    with torch.no_grad():
        image = pipe.vae.decode(lf, return_dict=False)[0]
        if image.dim() == 5:
            image = image[:, :, 0]

    from PIL import Image
    image = (torch.clamp(image / 2 + 0.5, 0.0, 1.0) * 255).to(torch.uint8)
    Image.fromarray(image[0].cpu().numpy().transpose(1, 2, 0)).save("output_debug_manual_cpu.png")
    print("  Saved: output_debug_manual_cpu.png")

    print("\nDone!")


if __name__ == "__main__":
    main()
