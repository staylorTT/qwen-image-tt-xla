"""Test a single MMDiT block using direct ttnn ops + tt-lang kernels.

Loads real Qwen-Image weights, runs PyTorch reference and ttnn forward,
compares outputs. Uses broadcast_row tt-lang kernel for mod param expansion
and ttnn.linear for fused matmul+bias.

Weight keys for block 0 (non-affine LN, biased QKV, GELU FFN):
  attn.to_q.{weight,bias}, attn.to_k.{weight,bias}, attn.to_v.{weight,bias}
  attn.add_q_proj.{weight,bias}, attn.add_k_proj.{weight,bias}, attn.add_v_proj.{weight,bias}
  attn.norm_q.weight, attn.norm_k.weight, attn.norm_added_q.weight, attn.norm_added_k.weight
  attn.to_out.0.{weight,bias}, attn.to_add_out.{weight,bias}
  img_mod.1.{weight,bias}, txt_mod.1.{weight,bias}
  img_mlp.net.0.proj.{weight,bias}, img_mlp.net.2.{weight,bias}
  txt_mlp.net.0.proj.{weight,bias}, txt_mlp.net.2.{weight,bias}
"""
import sys
import os
sys.path.insert(0, "/tmp")
sys.path.insert(0, "/workspace/qwen-image-tt-xla")

import torch
import torch.nn.functional as F
import time
import math
import safetensors.torch
import ttnn

from broadcast_row import broadcast_row_kernel

TILE = 32
N_HEADS = 24
HEAD_DIM = 128
HIDDEN_DIM = N_HEADS * HEAD_DIM  # 3072
SCALE = 1.0 / math.sqrt(HEAD_DIM)


def to_tt(t, device):
    if t.dim() == 1:
        t = t.unsqueeze(0)
    h, w_dim = t.shape[-2], t.shape[-1]
    ph = ((h + TILE - 1) // TILE) * TILE - h
    pw = ((w_dim + TILE - 1) // TILE) * TILE - w_dim
    if ph > 0 or pw > 0:
        t = F.pad(t, (0, pw, 0, ph))
    return ttnn.from_torch(t.contiguous().to(torch.bfloat16),
                           dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def to_tt_1d(t, device):
    """Convert 1D bias to ttnn format suitable for ttnn.linear bias arg."""
    return ttnn.from_torch(t.unsqueeze(0).to(torch.bfloat16),
                           dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def from_tt(t):
    return ttnn.to_torch(t).float()


def load_block_weights(weights_dir, block_idx=0):
    transformer_dir = os.path.join(weights_dir, "transformer")
    st_files = sorted([f for f in os.listdir(transformer_dir) if f.endswith(".safetensors")])
    prefix = f"transformer_blocks.{block_idx}."
    weights = {}
    for st_file in st_files:
        path = os.path.join(transformer_dir, st_file)
        with safetensors.torch.safe_open(path, framework="pt") as f:
            for key in f.keys():
                if key.startswith(prefix):
                    weights[key[len(prefix):]] = f.get_tensor(key).float()
    print(f"  Block {block_idx}: {len(weights)} tensors loaded")
    return weights


def rms_norm_pt(x, weight, eps=1e-6):
    rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x / rms * weight


def pytorch_block_forward(w, img_hs, txt_hs, temb):
    """PyTorch reference. Non-affine LN, biased QKV, GELU FFN, no RoPE."""
    temb_silu = F.silu(temb)
    img_mod = temb_silu @ w["img_mod.1.weight"].T + w["img_mod.1.bias"]
    txt_mod = temb_silu @ w["txt_mod.1.weight"].T + w["txt_mod.1.bias"]

    img_mod1, img_mod2 = img_mod.chunk(2, dim=-1)
    i_sh1, i_sc1, i_g1 = img_mod1.chunk(3, dim=-1)
    i_sh2, i_sc2, i_g2 = img_mod2.chunk(3, dim=-1)
    txt_mod1, txt_mod2 = txt_mod.chunk(2, dim=-1)
    t_sh1, t_sc1, t_g1 = txt_mod1.chunk(3, dim=-1)
    t_sh2, t_sc2, t_g2 = txt_mod2.chunk(3, dim=-1)

    img_n = F.layer_norm(img_hs, [HIDDEN_DIM])
    img_m = img_n * (1 + i_sc1) + i_sh1
    txt_n = F.layer_norm(txt_hs, [HIDDEN_DIM])
    txt_m = txt_n * (1 + t_sc1) + t_sh1

    img_q = (img_m @ w["attn.to_q.weight"].T + w["attn.to_q.bias"]).unflatten(-1, (N_HEADS, HEAD_DIM))
    img_k = (img_m @ w["attn.to_k.weight"].T + w["attn.to_k.bias"]).unflatten(-1, (N_HEADS, HEAD_DIM))
    img_v = (img_m @ w["attn.to_v.weight"].T + w["attn.to_v.bias"]).unflatten(-1, (N_HEADS, HEAD_DIM))
    txt_q = (txt_m @ w["attn.add_q_proj.weight"].T + w["attn.add_q_proj.bias"]).unflatten(-1, (N_HEADS, HEAD_DIM))
    txt_k = (txt_m @ w["attn.add_k_proj.weight"].T + w["attn.add_k_proj.bias"]).unflatten(-1, (N_HEADS, HEAD_DIM))
    txt_v = (txt_m @ w["attn.add_v_proj.weight"].T + w["attn.add_v_proj.bias"]).unflatten(-1, (N_HEADS, HEAD_DIM))

    img_q = rms_norm_pt(img_q, w["attn.norm_q.weight"])
    img_k = rms_norm_pt(img_k, w["attn.norm_k.weight"])
    txt_q = rms_norm_pt(txt_q, w["attn.norm_added_q.weight"])
    txt_k = rms_norm_pt(txt_k, w["attn.norm_added_k.weight"])

    txt_seq = txt_q.shape[1]
    q = torch.cat([txt_q, img_q], dim=1).transpose(1, 2)
    k = torch.cat([txt_k, img_k], dim=1).transpose(1, 2)
    v = torch.cat([txt_v, img_v], dim=1).transpose(1, 2)
    attn_out = F.scaled_dot_product_attention(q, k, v, scale=SCALE)
    attn_out = attn_out.transpose(1, 2).flatten(2, 3)

    txt_a = attn_out[:, :txt_seq] @ w["attn.to_add_out.weight"].T + w["attn.to_add_out.bias"]
    img_a = attn_out[:, txt_seq:] @ w["attn.to_out.0.weight"].T + w["attn.to_out.0.bias"]

    img_hs = img_hs + i_g1 * img_a
    txt_hs = txt_hs + t_g1 * txt_a

    def ffn(hs, sc, sh, ff1_w, ff1_b, ff2_w, ff2_b):
        n = F.layer_norm(hs, [HIDDEN_DIM])
        m = n * (1 + sc) + sh
        ff = m @ ff1_w.T + ff1_b
        ff = F.gelu(ff, approximate="tanh")
        return ff @ ff2_w.T + ff2_b

    img_ff = ffn(img_hs, i_sc2, i_sh2,
                 w["img_mlp.net.0.proj.weight"], w["img_mlp.net.0.proj.bias"],
                 w["img_mlp.net.2.weight"], w["img_mlp.net.2.bias"])
    img_hs = img_hs + i_g2 * img_ff

    txt_ff = ffn(txt_hs, t_sc2, t_sh2,
                 w["txt_mlp.net.0.proj.weight"], w["txt_mlp.net.0.proj.bias"],
                 w["txt_mlp.net.2.weight"], w["txt_mlp.net.2.bias"])
    txt_hs = txt_hs + t_g2 * txt_ff

    return img_hs, txt_hs


def expand_mod_tt(mod_3d, seq_len, device):
    """Expand [1, 32, D] mod param to [1, S, D] using broadcast_row kernel."""
    D_padded = mod_3d.shape[-1]
    mod_clean = ttnn.clone(mod_3d, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    mod_2d = ttnn.reshape(mod_clean, (TILE, D_padded))
    seq_padded = ((seq_len + TILE - 1) // TILE) * TILE
    out_2d = ttnn.from_torch(
        torch.zeros(seq_padded, D_padded, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    broadcast_row_kernel(mod_2d, out_2d)
    out_3d = ttnn.reshape(out_2d, (1, seq_padded, D_padded))
    return ttnn.clone(out_3d, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def ttnn_block_forward(w_tt, img_hs, txt_hs, temb, device):
    """Direct ttnn forward for one MMDiT block with tt-lang broadcast."""
    img_seq = img_hs.shape[-2]
    txt_seq = txt_hs.shape[-2]

    # AdaLN: SiLU -> Linear -> slice into 12 mod params
    temb_silu = ttnn.silu(temb)
    img_mod = ttnn.linear(temb_silu, w_tt["img_mod_w"], bias=w_tt["img_mod_b"])
    txt_mod = ttnn.linear(temb_silu, w_tt["txt_mod_w"], bias=w_tt["txt_mod_b"])

    D = HIDDEN_DIM
    i_sh1 = img_mod[:, :, :D]
    i_sc1 = img_mod[:, :, D:2*D]
    i_g1 = img_mod[:, :, 2*D:3*D]
    i_sh2 = img_mod[:, :, 3*D:4*D]
    i_sc2 = img_mod[:, :, 4*D:5*D]
    i_g2 = img_mod[:, :, 5*D:]

    t_sh1 = txt_mod[:, :, :D]
    t_sc1 = txt_mod[:, :, D:2*D]
    t_g1 = txt_mod[:, :, 2*D:3*D]
    t_sh2 = txt_mod[:, :, 3*D:4*D]
    t_sc2 = txt_mod[:, :, 4*D:5*D]
    t_g2 = txt_mod[:, :, 5*D:]

    # Broadcast mod params from [1,32,D] to [1,S,D]
    i_sh1_e = expand_mod_tt(i_sh1, img_seq, device)
    i_sc1_e = expand_mod_tt(i_sc1, img_seq, device)
    i_g1_e = expand_mod_tt(i_g1, img_seq, device)
    t_sh1_e = expand_mod_tt(t_sh1, txt_seq, device)
    t_sc1_e = expand_mod_tt(t_sc1, txt_seq, device)
    t_g1_e = expand_mod_tt(t_g1, txt_seq, device)

    # Non-affine LN + adaLN modulate
    img_n = ttnn.layer_norm(img_hs)
    sc1p1 = ttnn.add(i_sc1_e, 1.0)
    img_m = ttnn.add(ttnn.multiply(img_n, sc1p1), i_sh1_e)
    print(f"  img_m: {img_m.shape}")

    txt_n = ttnn.layer_norm(txt_hs)
    tsc1p1 = ttnn.add(t_sc1_e, 1.0)
    txt_m = ttnn.add(ttnn.multiply(txt_n, tsc1p1), t_sh1_e)
    print(f"  txt_m: {txt_m.shape}")

    # QKV with bias (use ttnn.linear for fused matmul+bias)
    print("  QKV projections...")
    img_q = ttnn.linear(img_m, w_tt["img_to_q"], bias=w_tt["img_to_q_b"])
    img_k = ttnn.linear(img_m, w_tt["img_to_k"], bias=w_tt["img_to_k_b"])
    img_v = ttnn.linear(img_m, w_tt["img_to_v"], bias=w_tt["img_to_v_b"])
    txt_q = ttnn.linear(txt_m, w_tt["txt_to_q"], bias=w_tt["txt_to_q_b"])
    txt_k = ttnn.linear(txt_m, w_tt["txt_to_k"], bias=w_tt["txt_to_k_b"])
    txt_v = ttnn.linear(txt_m, w_tt["txt_to_v"], bias=w_tt["txt_to_v_b"])
    print(f"    img_q: {img_q.shape}, txt_q: {txt_q.shape}")

    # Reshape [B, S, H*D] -> [B, S, H, D]
    B = 1
    img_q = ttnn.reshape(img_q, (B, img_seq, N_HEADS, HEAD_DIM))
    img_k = ttnn.reshape(img_k, (B, img_seq, N_HEADS, HEAD_DIM))
    img_v = ttnn.reshape(img_v, (B, img_seq, N_HEADS, HEAD_DIM))
    txt_q = ttnn.reshape(txt_q, (B, txt_seq, N_HEADS, HEAD_DIM))
    txt_k = ttnn.reshape(txt_k, (B, txt_seq, N_HEADS, HEAD_DIM))
    txt_v = ttnn.reshape(txt_v, (B, txt_seq, N_HEADS, HEAD_DIM))
    print("  Reshape done")

    # QK RMSNorm
    img_q = ttnn.rms_norm(img_q, weight=w_tt["norm_q"])
    img_k = ttnn.rms_norm(img_k, weight=w_tt["norm_k"])
    txt_q = ttnn.rms_norm(txt_q, weight=w_tt["norm_added_q"])
    txt_k = ttnn.rms_norm(txt_k, weight=w_tt["norm_added_k"])
    print("  RMSNorm done")

    # Joint SDPA
    q = ttnn.concat([txt_q, img_q], dim=1)
    k = ttnn.concat([txt_k, img_k], dim=1)
    v = ttnn.concat([txt_v, img_v], dim=1)

    q = ttnn.transpose(q, 1, 2)
    k = ttnn.transpose(k, 1, 2)
    v = ttnn.transpose(v, 1, 2)

    grid = device.compute_with_storage_grid_size()
    sdpa_cfg = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid,
        q_chunk_size=256, k_chunk_size=256, exp_approx_mode=False,
    )
    sdpa_cc = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False, fp32_dest_acc_en=True,
    )

    print("  SDPA...")
    attn_out = ttnn.transformer.scaled_dot_product_attention(
        q, k, v, is_causal=False, scale=SCALE,
        program_config=sdpa_cfg, compute_kernel_config=sdpa_cc,
    )
    print(f"    attn_out: {attn_out.shape}")

    attn_out = ttnn.transpose(attn_out, 1, 2)
    S_total = txt_seq + img_seq
    attn_out = ttnn.reshape(attn_out, (B, S_total, HIDDEN_DIM))

    txt_a = attn_out[:, :txt_seq, :]
    img_a = attn_out[:, txt_seq:, :]

    # Output projections with bias
    img_a = ttnn.linear(img_a, w_tt["img_to_out"], bias=w_tt["img_to_out_b"])
    txt_a = ttnn.linear(txt_a, w_tt["txt_to_out"], bias=w_tt["txt_to_out_b"])
    print("  Output proj done")

    # Gated residual
    img_hs = ttnn.add(img_hs, ttnn.multiply(i_g1_e, img_a))
    txt_hs = ttnn.add(txt_hs, ttnn.multiply(t_g1_e, txt_a))
    print("  Gated residual 1 done")

    # FFN
    i_sh2_e = expand_mod_tt(i_sh2, img_seq, device)
    i_sc2_e = expand_mod_tt(i_sc2, img_seq, device)
    i_g2_e = expand_mod_tt(i_g2, img_seq, device)
    t_sh2_e = expand_mod_tt(t_sh2, txt_seq, device)
    t_sc2_e = expand_mod_tt(t_sc2, txt_seq, device)
    t_g2_e = expand_mod_tt(t_g2, txt_seq, device)

    def ffn_tt(hs, sc, sh, ff1_w, ff1_b, ff2_w, ff2_b):
        n = ttnn.layer_norm(hs)
        scp1 = ttnn.add(sc, 1.0)
        m = ttnn.add(ttnn.multiply(n, scp1), sh)
        ff = ttnn.linear(m, ff1_w, bias=ff1_b)
        ff = ttnn.gelu(ff, fast_and_approximate_mode=True)
        return ttnn.linear(ff, ff2_w, bias=ff2_b)

    print("  FFN img...")
    img_ff = ffn_tt(img_hs, i_sc2_e, i_sh2_e,
                    w_tt["img_ff1_w"], w_tt["img_ff1_b"],
                    w_tt["img_ff2_w"], w_tt["img_ff2_b"])
    img_hs = ttnn.add(img_hs, ttnn.multiply(i_g2_e, img_ff))

    print("  FFN txt...")
    txt_ff = ffn_tt(txt_hs, t_sc2_e, t_sh2_e,
                    w_tt["txt_ff1_w"], w_tt["txt_ff1_b"],
                    w_tt["txt_ff2_w"], w_tt["txt_ff2_b"])
    txt_hs = ttnn.add(txt_hs, ttnn.multiply(t_g2_e, txt_ff))

    return img_hs, txt_hs


def load_weights_to_device(w, device):
    w_tt = {}
    # AdaLN MLP (weight transposed, bias as 1D for ttnn.linear)
    w_tt["img_mod_w"] = to_tt(w["img_mod.1.weight"].T.contiguous(), device)
    w_tt["img_mod_b"] = to_tt_1d(w["img_mod.1.bias"], device)
    w_tt["txt_mod_w"] = to_tt(w["txt_mod.1.weight"].T.contiguous(), device)
    w_tt["txt_mod_b"] = to_tt_1d(w["txt_mod.1.bias"], device)

    # QKV weights + biases
    w_tt["img_to_q"] = to_tt(w["attn.to_q.weight"].T.contiguous(), device)
    w_tt["img_to_q_b"] = to_tt_1d(w["attn.to_q.bias"], device)
    w_tt["img_to_k"] = to_tt(w["attn.to_k.weight"].T.contiguous(), device)
    w_tt["img_to_k_b"] = to_tt_1d(w["attn.to_k.bias"], device)
    w_tt["img_to_v"] = to_tt(w["attn.to_v.weight"].T.contiguous(), device)
    w_tt["img_to_v_b"] = to_tt_1d(w["attn.to_v.bias"], device)
    w_tt["txt_to_q"] = to_tt(w["attn.add_q_proj.weight"].T.contiguous(), device)
    w_tt["txt_to_q_b"] = to_tt_1d(w["attn.add_q_proj.bias"], device)
    w_tt["txt_to_k"] = to_tt(w["attn.add_k_proj.weight"].T.contiguous(), device)
    w_tt["txt_to_k_b"] = to_tt_1d(w["attn.add_k_proj.bias"], device)
    w_tt["txt_to_v"] = to_tt(w["attn.add_v_proj.weight"].T.contiguous(), device)
    w_tt["txt_to_v_b"] = to_tt_1d(w["attn.add_v_proj.bias"], device)

    # QK norm weights (1D)
    w_tt["norm_q"] = to_tt_1d(w["attn.norm_q.weight"], device)
    w_tt["norm_k"] = to_tt_1d(w["attn.norm_k.weight"], device)
    w_tt["norm_added_q"] = to_tt_1d(w["attn.norm_added_q.weight"], device)
    w_tt["norm_added_k"] = to_tt_1d(w["attn.norm_added_k.weight"], device)

    # Output projections
    w_tt["img_to_out"] = to_tt(w["attn.to_out.0.weight"].T.contiguous(), device)
    w_tt["img_to_out_b"] = to_tt_1d(w["attn.to_out.0.bias"], device)
    w_tt["txt_to_out"] = to_tt(w["attn.to_add_out.weight"].T.contiguous(), device)
    w_tt["txt_to_out_b"] = to_tt_1d(w["attn.to_add_out.bias"], device)

    # FFN weights + biases
    w_tt["img_ff1_w"] = to_tt(w["img_mlp.net.0.proj.weight"].T.contiguous(), device)
    w_tt["img_ff1_b"] = to_tt_1d(w["img_mlp.net.0.proj.bias"], device)
    w_tt["img_ff2_w"] = to_tt(w["img_mlp.net.2.weight"].T.contiguous(), device)
    w_tt["img_ff2_b"] = to_tt_1d(w["img_mlp.net.2.bias"], device)
    w_tt["txt_ff1_w"] = to_tt(w["txt_mlp.net.0.proj.weight"].T.contiguous(), device)
    w_tt["txt_ff1_b"] = to_tt_1d(w["txt_mlp.net.0.proj.bias"], device)
    w_tt["txt_ff2_w"] = to_tt(w["txt_mlp.net.2.weight"].T.contiguous(), device)
    w_tt["txt_ff2_b"] = to_tt_1d(w["txt_mlp.net.2.bias"], device)
    return w_tt


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    torch.manual_seed(42)

    img_seq = 1024  # 512x512: 32x32 patches
    txt_seq = 128

    print("Loading block 0 weights...")
    w = load_block_weights("/workspace/qwen-image-tt-xla/weights/qwen-image", block_idx=0)

    img_hs = torch.randn(1, img_seq, HIDDEN_DIM, dtype=torch.float32) * 0.1
    txt_hs = torch.randn(1, txt_seq, HIDDEN_DIM, dtype=torch.float32) * 0.1
    temb = torch.randn(1, 1, HIDDEN_DIM, dtype=torch.float32) * 0.1

    # Also compute bf16 reference for fairer comparison
    print("\nPyTorch bf16 reference...")
    w_bf16 = {k: v.bfloat16() for k, v in w.items()}
    with torch.no_grad():
        ref_img_bf16, ref_txt_bf16 = pytorch_block_forward(w_bf16, img_hs.bfloat16(), txt_hs.bfloat16(), temb.bfloat16())
    ref_img_bf16 = ref_img_bf16.float()
    ref_txt_bf16 = ref_txt_bf16.float()

    print("\nPyTorch fp32 reference...")
    with torch.no_grad():
        ref_img, ref_txt = pytorch_block_forward(w, img_hs, txt_hs, temb)
    print(f"  img: [{ref_img.min():.4f}, {ref_img.max():.4f}]")
    print(f"  txt: [{ref_txt.min():.4f}, {ref_txt.max():.4f}]")

    print("\nLoading weights to device...")
    w_tt = load_weights_to_device(w, device)

    print("Running TTNN forward...")
    img_tt = to_tt(img_hs.to(torch.bfloat16), device)
    txt_tt = to_tt(txt_hs.to(torch.bfloat16), device)
    temb_tt = to_tt(temb.to(torch.bfloat16), device)

    try:
        # Quick correctness check: just the adaLN + matmul part
        temb_silu = ttnn.silu(temb_tt)
        img_mod_ref = F.silu(temb) @ w["img_mod.1.weight"].T + w["img_mod.1.bias"]
        img_mod_tt = ttnn.linear(temb_silu, w_tt["img_mod_w"], bias=w_tt["img_mod_b"])
        img_mod_check = from_tt(img_mod_tt)[:, :1, :HIDDEN_DIM*6]
        img_mod_err = (img_mod_check - img_mod_ref[:, :1, :].float()).abs()
        print(f"  adaln_mod err: max={img_mod_err.max():.4f} mean={img_mod_err.mean():.6f}")

        tt_img, tt_txt = ttnn_block_forward(w_tt, img_tt, txt_tt, temb_tt, device)
        ttnn.synchronize_device(device)

        r_img = from_tt(tt_img)[:, :img_seq, :HIDDEN_DIM]
        r_txt = from_tt(tt_txt)[:, :txt_seq, :HIDDEN_DIM]

        # Compare vs fp32 ref
        img_err = (r_img - ref_img.float()).abs()
        txt_err = (r_txt - ref_txt.float()).abs()
        print(f"\n  vs fp32: img max_err={img_err.max():.4f} mean={img_err.mean():.6f}")
        print(f"  vs fp32: txt max_err={txt_err.max():.4f} mean={txt_err.mean():.6f}")

        # Compare vs bf16 ref (fairer)
        img_err_bf = (r_img - ref_img_bf16[:, :img_seq, :HIDDEN_DIM]).abs()
        txt_err_bf = (r_txt - ref_txt_bf16[:, :txt_seq, :HIDDEN_DIM]).abs()
        print(f"  vs bf16: img max_err={img_err_bf.max():.4f} mean={img_err_bf.mean():.6f}")
        print(f"  vs bf16: txt max_err={txt_err_bf.max():.4f} mean={txt_err_bf.mean():.6f}")

        # bf16 self-error (how much does just bf16 drift from fp32?)
        bf16_self_img = (ref_img_bf16[:, :img_seq, :HIDDEN_DIM] - ref_img.float()).abs()
        bf16_self_txt = (ref_txt_bf16[:, :txt_seq, :HIDDEN_DIM] - ref_txt.float()).abs()
        print(f"  bf16 drift: img max={bf16_self_img.max():.4f} txt max={bf16_self_txt.max():.4f}")

        img_ok = img_err_bf.max().item() < 50.0
        txt_ok = txt_err_bf.max().item() < 50.0
        print(f"  [{'PASS' if img_ok else 'FAIL'}] img  [{'PASS' if txt_ok else 'FAIL'}] txt")

        # Benchmark
        print("\nBenchmarking...")
        ttnn.synchronize_device(device)
        t0 = time.time()
        N_RUNS = 5
        for _ in range(N_RUNS):
            tt_img, tt_txt = ttnn_block_forward(w_tt, img_tt, txt_tt, temb_tt, device)
        ttnn.synchronize_device(device)
        elapsed = (time.time() - t0) / N_RUNS
        print(f"  Block time: {elapsed*1000:.2f}ms")
        print(f"  60 blocks: {elapsed*60*1000:.0f}ms = {elapsed*60:.2f}s")
        print(f"  50 steps x 60 blocks: {elapsed*60*50:.1f}s")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

    ttnn.close_device(device)
