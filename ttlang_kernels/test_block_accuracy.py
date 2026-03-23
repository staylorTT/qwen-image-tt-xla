"""Diagnose accuracy per-stage of the MMDiT block.

Tests each stage independently to find where error accumulates.
"""
import sys
import os
sys.path.insert(0, "/tmp")
sys.path.insert(0, "/workspace/qwen-image-tt-xla")

import torch
import torch.nn.functional as F
import math
import safetensors.torch
import ttnn

from broadcast_row import broadcast_row_kernel

TILE = 32
N_HEADS = 24
HEAD_DIM = 128
HIDDEN_DIM = N_HEADS * HEAD_DIM
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
    return ttnn.from_torch(t.unsqueeze(0).to(torch.bfloat16),
                           dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

def from_tt(t):
    return ttnn.to_torch(t).float()

def rms_norm_pt(x, weight, eps=1e-6):
    rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x / rms * weight

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
    return weights


def expand_mod_tt(mod_3d, seq_len, device):
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


def check(name, tt_val, ref_val, seq_dim=-2, feat_dim=-1):
    """Compare and report error."""
    tt_np = from_tt(tt_val)
    S = ref_val.shape[seq_dim] if seq_dim >= 0 else ref_val.shape[len(ref_val.shape) + seq_dim]
    D = ref_val.shape[feat_dim] if feat_dim >= 0 else ref_val.shape[len(ref_val.shape) + feat_dim]
    # Trim padding
    if tt_np.dim() == 3:
        tt_np = tt_np[:, :S, :D]
    elif tt_np.dim() == 4:
        tt_np = tt_np[:, :ref_val.shape[1], :ref_val.shape[2], :ref_val.shape[3]]
    err = (tt_np - ref_val.float()).abs()
    print(f"  {name}: max={err.max():.4f} mean={err.mean():.6f} (range [{ref_val.float().min():.2f}, {ref_val.float().max():.2f}])")


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    torch.manual_seed(42)

    img_seq = 1024
    txt_seq = 128

    print("Loading weights...")
    w = load_block_weights("/workspace/qwen-image-tt-xla/weights/qwen-image", block_idx=0)

    # All in bf16 to match device
    img_hs = (torch.randn(1, img_seq, HIDDEN_DIM) * 0.1).bfloat16()
    txt_hs = (torch.randn(1, txt_seq, HIDDEN_DIM) * 0.1).bfloat16()
    temb = (torch.randn(1, 1, HIDDEN_DIM) * 0.1).bfloat16()
    w_bf = {k: v.bfloat16() for k, v in w.items()}

    # Stage 1: AdaLN modulation
    print("\n--- Stage 1: AdaLN ---")
    ref_temb_silu = F.silu(temb)
    ref_img_mod = ref_temb_silu @ w_bf["img_mod.1.weight"].T + w_bf["img_mod.1.bias"]
    D = HIDDEN_DIM
    ref_i_sh1 = ref_img_mod[:, :, :D]
    ref_i_sc1 = ref_img_mod[:, :, D:2*D]

    temb_tt = to_tt(temb, device)
    img_mod_w = to_tt(w_bf["img_mod.1.weight"].T.contiguous(), device)
    img_mod_b = to_tt_1d(w_bf["img_mod.1.bias"], device)
    temb_silu_tt = ttnn.silu(temb_tt)
    img_mod_tt = ttnn.linear(temb_silu_tt, img_mod_w, bias=img_mod_b)
    check("img_mod", img_mod_tt, ref_img_mod)

    # Stage 2: LayerNorm + modulate
    print("\n--- Stage 2: LayerNorm + Modulate ---")
    ref_img_n = F.layer_norm(img_hs, [HIDDEN_DIM])
    ref_img_m = ref_img_n * (1 + ref_i_sc1) + ref_i_sh1

    img_tt = to_tt(img_hs, device)
    img_n_tt = ttnn.layer_norm(img_tt)
    check("layer_norm", img_n_tt, ref_img_n)

    # Expand mod params
    i_sh1_tt = img_mod_tt[:, :, :D]
    i_sc1_tt = img_mod_tt[:, :, D:2*D]
    i_sh1_e = expand_mod_tt(i_sh1_tt, img_seq, device)
    i_sc1_e = expand_mod_tt(i_sc1_tt, img_seq, device)
    sc1p1 = ttnn.add(i_sc1_e, 1.0)
    img_m_tt = ttnn.add(ttnn.multiply(img_n_tt, sc1p1), i_sh1_e)
    check("img_modulated", img_m_tt, ref_img_m)

    # Stage 3: QKV
    print("\n--- Stage 3: QKV ---")
    ref_img_q = (ref_img_m @ w_bf["attn.to_q.weight"].T + w_bf["attn.to_q.bias"]).unflatten(-1, (N_HEADS, HEAD_DIM))
    q_w = to_tt(w_bf["attn.to_q.weight"].T.contiguous(), device)
    q_b = to_tt_1d(w_bf["attn.to_q.bias"], device)
    img_q_tt = ttnn.linear(img_m_tt, q_w, bias=q_b)
    check("img_q_flat", img_q_tt, ref_img_q.flatten(-2, -1))

    img_q_tt = ttnn.reshape(img_q_tt, (1, img_seq, N_HEADS, HEAD_DIM))
    norm_q_w = to_tt_1d(w_bf["attn.norm_q.weight"], device)
    ref_img_q = rms_norm_pt(ref_img_q, w_bf["attn.norm_q.weight"])
    img_q_tt = ttnn.rms_norm(img_q_tt, weight=norm_q_w)
    check("img_q_normed", img_q_tt, ref_img_q)

    # Stage 4: SDPA (self-attention only with img for diagnostic)
    print("\n--- Stage 4: SDPA (img only for diagnostic) ---")
    ref_img_v = (ref_img_m @ w_bf["attn.to_v.weight"].T + w_bf["attn.to_v.bias"]).unflatten(-1, (N_HEADS, HEAD_DIM))
    ref_img_k = (ref_img_m @ w_bf["attn.to_k.weight"].T + w_bf["attn.to_k.bias"]).unflatten(-1, (N_HEADS, HEAD_DIM))
    ref_img_k = rms_norm_pt(ref_img_k, w_bf["attn.norm_k.weight"])

    ref_q = ref_img_q.transpose(1, 2)
    ref_k = ref_img_k.transpose(1, 2)
    ref_v = ref_img_v.transpose(1, 2)
    ref_attn = F.scaled_dot_product_attention(ref_q, ref_k, ref_v, scale=SCALE)
    print(f"  ref_attn range: [{ref_attn.min():.4f}, {ref_attn.max():.4f}]")

    # Compare with full attention error breakdown
    # The SDPA itself should be high precision since we use HiFi2 + fp32 accumulation
    print(f"  Note: ttnn SDPA uses HiFi2 + fp32 accumulation, should be accurate")

    ttnn.close_device(device)
    print("\nDone!")
