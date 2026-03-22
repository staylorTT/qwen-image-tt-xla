"""Bisect the single-block failure: which sub-operation breaks image stream?

Known: img_in, txt_in, temb all correct. Single block img_out is garbage.
We test each sub-operation of the block individually.

Run with: ./run.sh debug_block_bisect.py
"""

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
from utils.profiling_utils import compute_pcc


def main():
    weights_dir = "./weights/qwen-image"
    device = torch_xla.device()
    num_devices = xr.global_runtime_device_count()
    print(f"TT Devices: {num_devices}")

    print("Loading model...")
    pipe = DiffusionPipeline.from_pretrained(weights_dir, torch_dtype=torch.bfloat16)
    transformer = pipe.transformer.eval()
    config = transformer.config
    hidden_dim = config.num_attention_heads * config.attention_head_dim

    # Prepare realistic inputs
    width, height = 256, 256
    vae_sf = 8
    lh, lw = height // vae_sf, width // vae_sf
    nc = config.in_channels // 4

    prompt = "A cat"
    prompt_embeds, prompt_mask = pipe.encode_prompt(prompt=prompt, device="cpu")
    gen = torch.Generator().manual_seed(42)
    from utils.image_utils import pack_latents
    latents = pack_latents(
        torch.randn(1, nc, lh, lw, generator=gen, dtype=torch.bfloat16), 1, nc, lh, lw,
    )

    # Compute CPU inputs for block 0
    timestep_div = torch.tensor([999.0], dtype=torch.bfloat16) / 1000
    with torch.no_grad():
        hs_in = transformer.img_in(latents)         # [1, 256, 3072]
        ehs_in = transformer.txt_in(transformer.txt_norm(prompt_embeds))  # [1, 7, 3072]
        temb = transformer.time_text_embed(timestep_div.to(hs_in.dtype), hs_in)  # [1, 3072]

    block = transformer.transformer_blocks[0]
    attn = block.attn
    img_seq = hs_in.shape[1]   # 256
    txt_seq = ehs_in.shape[1]  # 7

    print(f"img_seq={img_seq}, txt_seq={txt_seq}, hidden_dim={hidden_dim}")
    print(f"hs_in stats: mean={hs_in.float().mean():.4f}, std={hs_in.float().std():.4f}")
    print(f"ehs_in stats: mean={ehs_in.float().mean():.4f}, std={ehs_in.float().std():.4f}")
    print(f"temb stats: mean={temb.float().mean():.4f}, std={temb.float().std():.4f}")

    # ============================================================
    # TEST A: Reproduce failure — native block forward
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST A: Native block forward on device (confirm failure)")
    print("=" * 60)

    # CPU ref
    with torch.no_grad():
        txt_ref, img_ref = block(
            hidden_states=hs_in.clone(), encoder_hidden_states=ehs_in.clone(),
            encoder_hidden_states_mask=prompt_mask.to(torch.bfloat16),
            temb=temb.clone(), image_rotary_emb=None,
        )

    block_dev = block.to(device)
    compiled_block = torch.compile(block_dev, backend="tt")
    with torch.no_grad():
        txt_d, img_d = compiled_block(
            hidden_states=hs_in.to(device), encoder_hidden_states=ehs_in.to(device),
            encoder_hidden_states_mask=prompt_mask.to(torch.bfloat16).to(device),
            temb=temb.to(device), image_rotary_emb=None,
        )
    torch_xla.sync()
    print(f"  img PCC: {compute_pcc(img_d.cpu(), img_ref):.6f}")
    print(f"  txt PCC: {compute_pcc(txt_d.cpu(), txt_ref):.6f}")
    block = block.cpu()
    transformer.transformer_blocks[0] = block

    # ============================================================
    # TEST B: Same block with RANDOM inputs (like test_device_real_weights)
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST B: Same block with random inputs")
    print("=" * 60)

    gen2 = torch.Generator().manual_seed(42)
    hs_rand = torch.randn(1, 64, hidden_dim, dtype=torch.bfloat16, generator=gen2)
    ehs_rand = torch.randn(1, 32, hidden_dim, dtype=torch.bfloat16, generator=gen2)
    em_rand = torch.ones(1, 32, dtype=torch.bfloat16)
    temb_rand = torch.randn(1, hidden_dim, dtype=torch.bfloat16, generator=gen2)

    with torch.no_grad():
        txt_ref_r, img_ref_r = block(
            hidden_states=hs_rand.clone(), encoder_hidden_states=ehs_rand.clone(),
            encoder_hidden_states_mask=em_rand, temb=temb_rand.clone(), image_rotary_emb=None,
        )

    block_dev = block.to(device)
    compiled_block2 = torch.compile(block_dev, backend="tt")
    with torch.no_grad():
        txt_d2, img_d2 = compiled_block2(
            hidden_states=hs_rand.to(device), encoder_hidden_states=ehs_rand.to(device),
            encoder_hidden_states_mask=em_rand.to(device), temb=temb_rand.to(device), image_rotary_emb=None,
        )
    torch_xla.sync()
    print(f"  img PCC (rand 64+32): {compute_pcc(img_d2.cpu(), img_ref_r):.6f}")
    print(f"  txt PCC (rand 64+32): {compute_pcc(txt_d2.cpu(), txt_ref_r):.6f}")
    block = block.cpu()
    transformer.transformer_blocks[0] = block

    # ============================================================
    # TEST C: Real inputs but seq_len=64 (same as passing test)
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST C: Real weights, truncated to img=64, txt=7")
    print("=" * 60)

    hs_short = hs_in[:, :64, :].clone()
    with torch.no_grad():
        txt_ref_s, img_ref_s = block(
            hidden_states=hs_short.clone(), encoder_hidden_states=ehs_in.clone(),
            encoder_hidden_states_mask=prompt_mask.to(torch.bfloat16),
            temb=temb.clone(), image_rotary_emb=None,
        )

    block_dev = block.to(device)
    compiled_block3 = torch.compile(block_dev, backend="tt")
    with torch.no_grad():
        txt_d3, img_d3 = compiled_block3(
            hidden_states=hs_short.to(device), encoder_hidden_states=ehs_in.to(device),
            encoder_hidden_states_mask=prompt_mask.to(torch.bfloat16).to(device),
            temb=temb.to(device), image_rotary_emb=None,
        )
    torch_xla.sync()
    print(f"  img PCC (64+7): {compute_pcc(img_d3.cpu(), img_ref_s):.6f}")
    print(f"  txt PCC (64+7): {compute_pcc(txt_d3.cpu(), txt_ref_s):.6f}")
    block = block.cpu()
    transformer.transformer_blocks[0] = block

    # ============================================================
    # TEST D: Real inputs, img_seq=256, txt_seq=32 (pad text)
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST D: img=256, txt=32 (pad text to 32)")
    print("=" * 60)

    ehs_padded = torch.zeros(1, 32, hidden_dim, dtype=torch.bfloat16)
    ehs_padded[:, :txt_seq, :] = ehs_in
    em_padded = torch.zeros(1, 32, dtype=torch.bfloat16)
    em_padded[:, :txt_seq] = prompt_mask.to(torch.bfloat16)

    with torch.no_grad():
        txt_ref_p, img_ref_p = block(
            hidden_states=hs_in.clone(), encoder_hidden_states=ehs_padded.clone(),
            encoder_hidden_states_mask=em_padded, temb=temb.clone(), image_rotary_emb=None,
        )

    block_dev = block.to(device)
    compiled_block4 = torch.compile(block_dev, backend="tt")
    with torch.no_grad():
        txt_d4, img_d4 = compiled_block4(
            hidden_states=hs_in.to(device), encoder_hidden_states=ehs_padded.to(device),
            encoder_hidden_states_mask=em_padded.to(device), temb=temb.to(device), image_rotary_emb=None,
        )
    torch_xla.sync()
    print(f"  img PCC (256+32): {compute_pcc(img_d4.cpu(), img_ref_p):.6f}")
    print(f"  txt PCC (256+32): {compute_pcc(txt_d4.cpu(), txt_ref_p):.6f}")
    block = block.cpu()
    transformer.transformer_blocks[0] = block

    # ============================================================
    # TEST E: AdaLN alone on device
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST E: AdaLN modulation on device")
    print("=" * 60)

    with torch.no_grad():
        img_mod_cpu = block.img_mod(temb.clone())
        txt_mod_cpu = block.txt_mod(temb.clone())

    img_mod_dev = block.img_mod.to(device)
    txt_mod_dev = block.txt_mod.to(device)
    c_img_mod = torch.compile(img_mod_dev, backend="tt")
    c_txt_mod = torch.compile(txt_mod_dev, backend="tt")

    with torch.no_grad():
        img_mod_d = c_img_mod(temb.to(device))
        txt_mod_d = c_txt_mod(temb.to(device))
    torch_xla.sync()

    print(f"  img_mod PCC: {compute_pcc(img_mod_d.cpu(), img_mod_cpu):.6f}")
    print(f"  txt_mod PCC: {compute_pcc(txt_mod_d.cpu(), txt_mod_cpu):.6f}")
    block.img_mod = block.img_mod.cpu()
    block.txt_mod = block.txt_mod.cpu()

    # ============================================================
    # TEST F: Attention sub-operations individually
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST F: Attention projections on device")
    print("=" * 60)

    # First compute AdaLN modulated inputs on CPU
    with torch.no_grad():
        img_mod_params = block.img_mod(temb.clone())
        txt_mod_params = block.txt_mod(temb.clone())
        img_mod1, img_mod2 = img_mod_params.chunk(2, dim=-1)
        txt_mod1, txt_mod2 = txt_mod_params.chunk(2, dim=-1)
        shift, scale, gate = img_mod1.chunk(3, dim=-1)
        img_normed = block.img_norm1(hs_in.clone())
        img_modulated = img_normed * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    print(f"  img_modulated: shape={list(img_modulated.shape)}, "
          f"mean={img_modulated.float().mean():.4f}, std={img_modulated.float().std():.4f}")

    # Test img Q projection alone
    with torch.no_grad():
        q_cpu = attn.to_q(img_modulated.clone())

    to_q_dev = attn.to_q.to(device)
    c_q = torch.compile(to_q_dev, backend="tt")
    with torch.no_grad():
        q_d = c_q(img_modulated.to(device))
    torch_xla.sync()
    pcc_q = compute_pcc(q_d.cpu(), q_cpu)
    print(f"  to_q PCC (img 256 tokens): {pcc_q:.6f}")
    attn.to_q = attn.to_q.cpu()

    # Test with shorter sequence
    with torch.no_grad():
        q_cpu_short = attn.to_q(img_modulated[:, :64, :].clone())
    to_q_dev = attn.to_q.to(device)
    c_q2 = torch.compile(to_q_dev, backend="tt")
    with torch.no_grad():
        q_d_short = c_q2(img_modulated[:, :64, :].to(device))
    torch_xla.sync()
    pcc_q_short = compute_pcc(q_d_short.cpu(), q_cpu_short)
    print(f"  to_q PCC (img 64 tokens):  {pcc_q_short:.6f}")
    attn.to_q = attn.to_q.cpu()

    # ============================================================
    # TEST G: Scaled dot product attention on device
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST G: SDPA on device (256 img + 7 txt)")
    print("=" * 60)

    # Build Q/K/V on CPU through the block
    with torch.no_grad():
        img_q = attn.to_q(img_modulated).unflatten(-1, (attn.heads, -1))
        img_k = attn.to_k(img_modulated).unflatten(-1, (attn.heads, -1))
        img_v = attn.to_v(img_modulated).unflatten(-1, (attn.heads, -1))

        t_shift, t_scale, t_gate = txt_mod1.chunk(3, dim=-1)
        txt_normed = block.txt_norm1(ehs_in.clone())
        txt_modulated = txt_normed * (1 + t_scale.unsqueeze(1)) + t_shift.unsqueeze(1)

        txt_q = attn.add_q_proj(txt_modulated).unflatten(-1, (attn.heads, -1))
        txt_k = attn.add_k_proj(txt_modulated).unflatten(-1, (attn.heads, -1))
        txt_v = attn.add_v_proj(txt_modulated).unflatten(-1, (attn.heads, -1))

        if attn.norm_q is not None:
            img_q = attn.norm_q(img_q)
            img_k = attn.norm_k(img_k)
            txt_q = attn.norm_added_q(txt_q)
            txt_k = attn.norm_added_k(txt_k)

        q = torch.cat([txt_q, img_q], dim=1)  # [1, 263, 24, 128]
        k = torch.cat([txt_k, img_k], dim=1)
        v = torch.cat([txt_v, img_v], dim=1)

        # CPU SDPA ref
        out_cpu = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            dropout_p=0.0, is_causal=False,
        ).transpose(1, 2)

    print(f"  Q shape: {list(q.shape)}, K: {list(k.shape)}")
    print(f"  SDPA out shape: {list(out_cpu.shape)}")
    print(f"  SDPA txt part stats: mean={out_cpu[:, :txt_seq].float().mean():.4f}")
    print(f"  SDPA img part stats: mean={out_cpu[:, txt_seq:].float().mean():.4f}")

    def sdpa_fn(q, k, v):
        return F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            dropout_p=0.0, is_causal=False,
        ).transpose(1, 2)

    c_sdpa = torch.compile(sdpa_fn, backend="tt")
    with torch.no_grad():
        out_d = c_sdpa(q.to(device), k.to(device), v.to(device))
    torch_xla.sync()

    out_d_cpu = out_d.cpu()
    pcc_sdpa_all = compute_pcc(out_d_cpu, out_cpu)
    pcc_sdpa_txt = compute_pcc(out_d_cpu[:, :txt_seq], out_cpu[:, :txt_seq])
    pcc_sdpa_img = compute_pcc(out_d_cpu[:, txt_seq:], out_cpu[:, txt_seq:])
    print(f"  SDPA PCC (all):  {pcc_sdpa_all:.6f}")
    print(f"  SDPA PCC (txt):  {pcc_sdpa_txt:.6f}")
    print(f"  SDPA PCC (img):  {pcc_sdpa_img:.6f}")

    # ============================================================
    # TEST H: Try different image sequence lengths
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST H: Vary image sequence length")
    print("=" * 60)

    for seq_len in [16, 32, 64, 128, 256]:
        hs_test = hs_in[:, :seq_len, :].clone()
        with torch.no_grad():
            txt_ref_h, img_ref_h = block(
                hidden_states=hs_test, encoder_hidden_states=ehs_in.clone(),
                encoder_hidden_states_mask=prompt_mask.to(torch.bfloat16),
                temb=temb.clone(), image_rotary_emb=None,
            )

        block_h = block.to(device)
        c_h = torch.compile(block_h, backend="tt")
        with torch.no_grad():
            txt_h, img_h = c_h(
                hidden_states=hs_test.to(device), encoder_hidden_states=ehs_in.to(device),
                encoder_hidden_states_mask=prompt_mask.to(torch.bfloat16).to(device),
                temb=temb.to(device), image_rotary_emb=None,
            )
        torch_xla.sync()
        pcc_h = compute_pcc(img_h.cpu(), img_ref_h)
        pcc_h_t = compute_pcc(txt_h.cpu(), txt_ref_h)
        print(f"  seq={seq_len:3d}: img PCC={pcc_h:.6f}, txt PCC={pcc_h_t:.6f}")
        block = block.cpu()
        transformer.transformer_blocks[0] = block

    print("\nDone!")


if __name__ == "__main__":
    main()
