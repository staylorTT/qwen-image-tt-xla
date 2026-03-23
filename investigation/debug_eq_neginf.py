"""Test if ttnn.eq(scores, -inf) produces wrong results for real model scores.

Hypothesis: the eq comparison or its bf16 output type mishandles certain value ranges,
causing the safe-softmax where() to incorrectly zero out valid attention weights.
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

    # Build real Q/K/V
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

        q = torch.cat([txt_q, img_q], dim=1).transpose(1, 2).contiguous()
        k = torch.cat([txt_k, img_k], dim=1).transpose(1, 2).contiguous()

    print(f"Q shape: {list(q.shape)}")
    print(f"Q stats: mean={q.float().mean():.4f}, std={q.float().std():.4f}, "
          f"min={q.float().min():.4f}, max={q.float().max():.4f}")
    print(f"K stats: mean={k.float().mean():.4f}, std={k.float().std():.4f}, "
          f"min={k.float().min():.4f}, max={k.float().max():.4f}")

    # Compute attention scores exactly as the broken SDPA IR does:
    # Q_scaled = Q * sqrt(scale), K_scaled = K * sqrt(scale)
    sqrt_scale = math.sqrt(1.0 / math.sqrt(128))  # 0.2973
    q_scaled = q.float() * sqrt_scale
    k_scaled = k.float() * sqrt_scale
    scores = torch.matmul(q_scaled, k_scaled.transpose(-2, -1))

    print(f"\nScores stats: mean={scores.mean():.4f}, std={scores.std():.4f}, "
          f"min={scores.min():.4f}, max={scores.max():.4f}")
    print(f"Any -inf in scores? {torch.isinf(scores).any().item()}")
    print(f"Any NaN in scores?  {torch.isnan(scores).any().item()}")

    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 1: eq(real_scores, -inf) on device")
    print("=" * 60)

    neginf = torch.tensor(float('-inf'), dtype=torch.float32)

    def eq_neginf(x):
        return (x == float('-inf'))

    # CPU ref
    eq_cpu = eq_neginf(scores)
    print(f"  CPU: any True? {eq_cpu.any().item()}, count={eq_cpu.sum().item()}")

    compiled_eq = torch.compile(eq_neginf, backend="tt")
    with torch.no_grad():
        eq_dev = compiled_eq(scores.to(device))
    torch_xla.sync()
    eq_dev_cpu = eq_dev.cpu()
    print(f"  Dev: any True? {eq_dev_cpu.any().item()}, count={eq_dev_cpu.float().sum().item()}")

    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 2: Full safe-softmax chain with real scores on device")
    print("=" * 60)

    def safe_softmax_chain(scores):
        """Replicate exact IR from broken SDPA"""
        is_neginf = (scores == float('-inf')).to(torch.bfloat16)
        not_neginf = 1.0 - is_neginf  # logical_not as arithmetic
        count = not_neginf.sum(dim=-1, keepdim=False)
        has_valid = (count != 0).to(torch.bfloat16)
        all_neginf = 1.0 - has_valid
        mask = all_neginf.unsqueeze(-1).expand_as(scores)
        s = torch.softmax(scores, dim=-1)
        z = torch.zeros_like(s)
        return torch.where(mask.bool(), z, s)

    ref_chain = safe_softmax_chain(scores)
    ref_plain = torch.softmax(scores, dim=-1)
    print(f"  CPU: chain == plain softmax? PCC = {compute_pcc(ref_chain, ref_plain):.6f}")

    compiled_chain = torch.compile(safe_softmax_chain, backend="tt")
    with torch.no_grad():
        chain_dev = compiled_chain(scores.to(device))
    torch_xla.sync()
    chain_dev_cpu = chain_dev.cpu()

    pcc_all = compute_pcc(chain_dev_cpu, ref_plain)
    pcc_txt = compute_pcc(chain_dev_cpu[:, :, :7], ref_plain[:, :, :7])
    pcc_img = compute_pcc(chain_dev_cpu[:, :, 7:], ref_plain[:, :, 7:])
    print(f"  Dev: PCC all={pcc_all:.6f}, txt={pcc_txt:.6f}, img={pcc_img:.6f}")

    # Check for zeros introduced by where
    dev_zeros = (chain_dev_cpu == 0).sum().item()
    ref_zeros = (ref_plain == 0).sum().item()
    print(f"  Zeros: dev={dev_zeros}, ref={ref_zeros}")

    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 3: Just softmax of real scores (no eq/where chain)")
    print("=" * 60)

    compiled_sm = torch.compile(lambda x: torch.softmax(x, dim=-1), backend="tt")
    with torch.no_grad():
        sm_dev = compiled_sm(scores.to(device))
    torch_xla.sync()
    sm_pcc = compute_pcc(sm_dev.cpu(), ref_plain)
    print(f"  Plain softmax PCC: {sm_pcc:.6f}")

    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 4: Matmul Q_scaled @ K_scaled^T on device")
    print("=" * 60)

    def compute_scores(q, k):
        sqrt_s = 0.297301769  # sqrt(1/sqrt(128))
        return torch.matmul(q * sqrt_s, (k * sqrt_s).transpose(-2, -1))

    compiled_scores = torch.compile(compute_scores, backend="tt")
    with torch.no_grad():
        scores_dev = compiled_scores(q.float().to(device), k.float().to(device))
    torch_xla.sync()
    scores_pcc = compute_pcc(scores_dev.cpu(), scores)
    print(f"  Scores matmul PCC: {scores_pcc:.6f}")
    scores_dev_cpu = scores_dev.cpu()
    print(f"  Dev scores: mean={scores_dev_cpu.mean():.4f}, std={scores_dev_cpu.std():.4f}")
    print(f"  Any -inf? {torch.isinf(scores_dev_cpu).any().item()}")

    print("\nDone!")


if __name__ == "__main__":
    main()
