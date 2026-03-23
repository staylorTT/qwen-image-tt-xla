"""Test SDPA with actual Q/K/V from the real model to reproduce the failure.

We know: random Q/K/V → works, real model Q/K/V → fails.
Find what property of real values triggers the bug.
"""

import math
import os

import torch
import torch.nn.functional as F
import torch_xla
import torch_xla.runtime as xr

os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
xr.use_spmd()
xr.set_device_type("TT")

from diffusers import DiffusionPipeline
from utils.image_utils import pack_latents
from utils.profiling_utils import compute_pcc


def main():
    device = torch_xla.device()
    weights_dir = "./weights/qwen-image"

    print("Loading model...")
    pipe = DiffusionPipeline.from_pretrained(weights_dir, torch_dtype=torch.bfloat16)
    transformer = pipe.transformer.eval()
    config = transformer.config

    # Build real inputs
    width, height = 256, 256
    vae_sf = 8
    lh, lw = height // vae_sf, width // vae_sf
    nc = config.in_channels // 4

    prompt_embeds, prompt_mask = pipe.encode_prompt(prompt="A cat", device="cpu")
    gen = torch.Generator().manual_seed(42)
    latents = pack_latents(
        torch.randn(1, nc, lh, lw, generator=gen, dtype=torch.bfloat16), 1, nc, lh, lw,
    )
    timestep_div = torch.tensor([999.0], dtype=torch.bfloat16) / 1000

    block = transformer.transformer_blocks[0]
    attn = block.attn

    # Build Q/K/V through the block's real computation
    with torch.no_grad():
        hs_in = transformer.img_in(latents)
        ehs_in = transformer.txt_in(transformer.txt_norm(prompt_embeds))
        temb = transformer.time_text_embed(timestep_div.to(hs_in.dtype), hs_in)

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

        # Concatenate: [txt, img]
        q = torch.cat([txt_q, img_q], dim=1)  # [1, 263, 24, 128]
        k = torch.cat([txt_k, img_k], dim=1)
        v = torch.cat([txt_v, img_v], dim=1)

        # Transpose for SDPA: [B, S, H, D] → [B, H, S, D]
        q_t = q.transpose(1, 2).contiguous()  # [1, 24, 263, 128]
        k_t = k.transpose(1, 2).contiguous()
        v_t = v.transpose(1, 2).contiguous()

    txt_seq = ehs_in.shape[1]  # 7
    print(f"Q/K/V shape: {list(q_t.shape)}, txt_seq={txt_seq}")
    print(f"Q stats: mean={q_t.float().mean():.4f}, std={q_t.float().std():.4f}, "
          f"min={q_t.float().min():.4f}, max={q_t.float().max():.4f}")
    print(f"K stats: mean={k_t.float().mean():.4f}, std={k_t.float().std():.4f}")

    # CPU reference
    ref = F.scaled_dot_product_attention(q_t.float(), k_t.float(), v_t.float(),
                                          dropout_p=0.0, is_causal=False).to(torch.bfloat16)
    print(f"CPU ref: mean={ref.float().mean():.4f}, std={ref.float().std():.4f}")

    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 1: SDPA with real Q/K/V on device (reproduce failure)")
    print("=" * 60)

    def sdpa_fn(q, k, v):
        return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)

    compiled = torch.compile(sdpa_fn, backend="tt")
    with torch.no_grad():
        out = compiled(q_t.to(device), k_t.to(device), v_t.to(device))
    torch_xla.sync()
    out_cpu = out.cpu()

    pcc_all = compute_pcc(out_cpu, ref)
    pcc_txt = compute_pcc(out_cpu[:, :, :txt_seq], ref[:, :, :txt_seq])
    pcc_img = compute_pcc(out_cpu[:, :, txt_seq:], ref[:, :, txt_seq:])
    print(f"  PCC all={pcc_all:.6f}, txt={pcc_txt:.6f}, img={pcc_img:.6f}")

    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 2: Manual matmul+softmax+matmul with real Q/K/V on device")
    print("=" * 60)

    def manual_sdpa(q, k, v):
        scale = 1.0 / math.sqrt(q.shape[-1])
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, v)

    compiled2 = torch.compile(manual_sdpa, backend="tt")
    with torch.no_grad():
        out2 = compiled2(q_t.to(device), k_t.to(device), v_t.to(device))
    torch_xla.sync()
    out2_cpu = out2.cpu()

    pcc2_all = compute_pcc(out2_cpu, ref)
    pcc2_txt = compute_pcc(out2_cpu[:, :, :txt_seq], ref[:, :, :txt_seq])
    pcc2_img = compute_pcc(out2_cpu[:, :, txt_seq:], ref[:, :, txt_seq:])
    print(f"  PCC all={pcc2_all:.6f}, txt={pcc2_txt:.6f}, img={pcc2_img:.6f}")

    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 3: Pad Q/K/V to tile-aligned 288 then SDPA")
    print("=" * 60)

    # Pad from 263 to 288 (9×32)
    pad_to = 288
    pad_size = pad_to - q_t.shape[2]
    q_pad = F.pad(q_t, (0, 0, 0, pad_size))
    k_pad = F.pad(k_t, (0, 0, 0, pad_size), value=0)
    v_pad = F.pad(v_t, (0, 0, 0, pad_size))

    ref_pad = F.scaled_dot_product_attention(q_pad.float(), k_pad.float(), v_pad.float(),
                                              dropout_p=0.0, is_causal=False).to(torch.bfloat16)

    compiled3 = torch.compile(sdpa_fn, backend="tt")
    with torch.no_grad():
        out3 = compiled3(q_pad.to(device), k_pad.to(device), v_pad.to(device))
    torch_xla.sync()
    out3_cpu = out3.cpu()

    # Compare only the valid (non-padded) positions
    pcc3_valid = compute_pcc(out3_cpu[:, :, :263], ref_pad[:, :, :263])
    pcc3_txt = compute_pcc(out3_cpu[:, :, :txt_seq], ref_pad[:, :, :txt_seq])
    pcc3_img = compute_pcc(out3_cpu[:, :, txt_seq:263], ref_pad[:, :, txt_seq:263])
    print(f"  PCC valid={pcc3_valid:.6f}, txt={pcc3_txt:.6f}, img={pcc3_img:.6f}")

    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 4: Pad K with -inf instead of 0 then SDPA")
    print("=" * 60)

    k_pad_neginf = F.pad(k_t, (0, 0, 0, pad_size), value=0)
    # Create attention mask: -inf for padded K positions
    mask = torch.zeros(1, 1, q_t.shape[2], pad_to, dtype=q_t.dtype)
    mask[:, :, :, 263:] = float('-inf')
    mask_pad = F.pad(mask, (0, 0, 0, pad_size))  # pad Q dim too

    # Can't easily pass mask through F.sdpa compiled, test manual with mask
    def manual_sdpa_masked(q, k, v, mask):
        scale = 1.0 / math.sqrt(q.shape[-1])
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale + mask
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, v)

    ref4 = manual_sdpa_masked(q_pad.float(), k_pad.float(), v_pad.float(),
                               mask_pad.float()).to(torch.bfloat16)

    compiled4 = torch.compile(manual_sdpa_masked, backend="tt")
    with torch.no_grad():
        out4 = compiled4(q_pad.to(device), k_pad.to(device), v_pad.to(device),
                         mask_pad.to(device))
    torch_xla.sync()
    out4_cpu = out4.cpu()

    pcc4_valid = compute_pcc(out4_cpu[:, :, :263], ref4[:, :, :263])
    pcc4_txt = compute_pcc(out4_cpu[:, :, :txt_seq], ref4[:, :, :txt_seq])
    pcc4_img = compute_pcc(out4_cpu[:, :, txt_seq:263], ref4[:, :, txt_seq:263])
    print(f"  PCC valid={pcc4_valid:.6f}, txt={pcc4_txt:.6f}, img={pcc4_img:.6f}")

    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 5: Scale Q/K/V values — are large magnitudes the issue?")
    print("=" * 60)

    # Try scaling real Q/K/V to have same stats as randn
    q_norm = q_t / q_t.float().std()
    k_norm = k_t / k_t.float().std()
    v_norm = v_t / v_t.float().std()

    ref5 = F.scaled_dot_product_attention(q_norm.float(), k_norm.float(), v_norm.float(),
                                           dropout_p=0.0, is_causal=False).to(torch.bfloat16)

    compiled5 = torch.compile(sdpa_fn, backend="tt")
    with torch.no_grad():
        out5 = compiled5(q_norm.to(device), k_norm.to(device), v_norm.to(device))
    torch_xla.sync()
    out5_cpu = out5.cpu()

    pcc5_all = compute_pcc(out5_cpu, ref5)
    pcc5_txt = compute_pcc(out5_cpu[:, :, :txt_seq], ref5[:, :, :txt_seq])
    pcc5_img = compute_pcc(out5_cpu[:, :, txt_seq:], ref5[:, :, txt_seq:])
    print(f"  PCC all={pcc5_all:.6f}, txt={pcc5_txt:.6f}, img={pcc5_img:.6f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
