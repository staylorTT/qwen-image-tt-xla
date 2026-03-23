"""End-to-end image generation using pure ttnn for the 60-block MMDiT.

Uses 4 Blackhole devices with Tensor Parallelism (TP) following oasis patterns:
- Column-parallel: QKV projections, FFN fc1 (shard output dim)
- Row-parallel: output projections, FFN fc2 (shard input dim) + all_reduce
- All 60 blocks' weights preloaded and kept resident across 4 devices
- Pre-allocated scratch tensors, no allocations in hot loop
- Tracing for zero Python dispatch overhead

CPU handles: text encoding, timestep embedding, img_in/txt_in, final norm/proj,
             scheduler, VAE decode.
Device handles: all 60 MMDiT transformer blocks (the expensive part).
"""
import sys
import os
sys.path.insert(0, "/tmp")
sys.path.insert(0, "/workspace/qwen-image-tt-xla")

import gc
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
import safetensors.torch
import ttnn

from rope_layout import make_rope_layout_kernel
from adaln_modulate import adaln_modulate_kernel
from gated_residual import gated_residual_kernel

TILE = 32
N_HEADS = 24
HEAD_DIM = 128
HIDDEN_DIM = N_HEADS * HEAD_DIM  # 3072
SCALE = 1.0 / math.sqrt(HEAD_DIM)
N_CHIPS = 4


# ============================================================
# Host helpers
# ============================================================

def _mesh_kwargs(device):
    if isinstance(device, ttnn.MeshDevice):
        return {"mesh_mapper": ttnn.ReplicateTensorToMesh(device)}
    return {}


def to_tt(t, device):
    """Convert torch tensor to ttnn, replicated across mesh."""
    if t.dim() == 1:
        t = t.unsqueeze(0)
    h, w_dim = t.shape[-2], t.shape[-1]
    ph = ((h + TILE - 1) // TILE) * TILE - h
    pw = ((w_dim + TILE - 1) // TILE) * TILE - w_dim
    if ph > 0 or pw > 0:
        t = F.pad(t, (0, pw, 0, ph))
    return ttnn.from_torch(t.contiguous().to(torch.bfloat16),
                           dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                           **_mesh_kwargs(device))


def to_tt_1d(t, device):
    """Convert 1D bias for ttnn.linear, replicated across mesh."""
    return ttnn.from_torch(t.unsqueeze(0).to(torch.bfloat16),
                           dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                           **_mesh_kwargs(device))


def zeros_tt(shape, device):
    return to_tt(torch.zeros(shape, dtype=torch.bfloat16), device)


def shard_tt(t, device, dim):
    """Load tensor sharded across mesh devices along given dimension."""
    if t.dim() == 1:
        t = t.unsqueeze(0)
    h, w_dim = t.shape[-2], t.shape[-1]
    ph = ((h + TILE - 1) // TILE) * TILE - h
    pw = ((w_dim + TILE - 1) // TILE) * TILE - w_dim
    if ph > 0 or pw > 0:
        t = F.pad(t, (0, pw, 0, ph))
    return ttnn.from_torch(t.contiguous().to(torch.bfloat16),
                           dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                           mesh_mapper=ttnn.ShardTensorToMesh(device, dim=dim))


def shard_tt_1d(t, device, dim):
    """Load 1D bias tensor sharded across mesh for column-parallel ops."""
    return ttnn.from_torch(t.unsqueeze(0).to(torch.bfloat16),
                           dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                           mesh_mapper=ttnn.ShardTensorToMesh(device, dim=dim))


def from_tt(t, device=None):
    if device is not None and isinstance(device, ttnn.MeshDevice):
        return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))[:t.shape[0]].float()
    return ttnn.to_torch(t).float()


def make_swap_perm_matrix(dim):
    """Build [dim, dim] permutation matrix that swaps adjacent pairs.
    [x0,x1,x2,x3,...] @ P = [x1,x0,x3,x2,...].
    Used for RoPE rotate_half on device after QK-norm."""
    P = torch.zeros(dim, dim)
    for i in range(0, dim, 2):
        P[i, i + 1] = 1.0
        P[i + 1, i] = 1.0
    return P


def expand_bias(bias_1d, seq_pad):
    """Pre-expand 1D bias to [seq_pad, D] for use with ttnn.linear or ttnn.add."""
    return bias_1d.unsqueeze(0).expand(seq_pad, -1).contiguous().to(torch.bfloat16)


# ============================================================
# Weight loading
# ============================================================

class TTNNBlock:
    """One MMDiT block weights on device with TP sharding."""

    def __init__(self, weights, device, use_tp=False, img_seq_pad=0, txt_seq_pad=0):
        self.device = device
        self.use_tp = use_tp
        d = device
        w = weights

        # AdaLN MLP (replicated - small, runs on all chips)
        self.img_mod_w = to_tt(w["img_mod.1.weight"].T.contiguous(), d)
        self.img_mod_b = to_tt(expand_bias(w["img_mod.1.bias"], TILE), d)
        self.txt_mod_w = to_tt(w["txt_mod.1.weight"].T.contiguous(), d)
        self.txt_mod_b = to_tt(expand_bias(w["txt_mod.1.bias"], TILE), d)

        if use_tp and N_CHIPS > 1:
            # Column-parallel QKV (shard output dim=1)
            self.img_to_q = shard_tt(w["attn.to_q.weight"].T.contiguous(), d, dim=1)
            self.img_to_k = shard_tt(w["attn.to_k.weight"].T.contiguous(), d, dim=1)
            self.img_to_v = shard_tt(w["attn.to_v.weight"].T.contiguous(), d, dim=1)
            self.txt_to_q = shard_tt(w["attn.add_q_proj.weight"].T.contiguous(), d, dim=1)
            self.txt_to_k = shard_tt(w["attn.add_k_proj.weight"].T.contiguous(), d, dim=1)
            self.txt_to_v = shard_tt(w["attn.add_v_proj.weight"].T.contiguous(), d, dim=1)

            # Column-parallel QKV biases (sharded to match output dim)
            self.img_to_q_b = shard_tt_1d(w["attn.to_q.bias"], d, dim=1)
            self.img_to_k_b = shard_tt_1d(w["attn.to_k.bias"], d, dim=1)
            self.img_to_v_b = shard_tt_1d(w["attn.to_v.bias"], d, dim=1)
            self.txt_to_q_b = shard_tt_1d(w["attn.add_q_proj.bias"], d, dim=1)
            self.txt_to_k_b = shard_tt_1d(w["attn.add_k_proj.bias"], d, dim=1)
            self.txt_to_v_b = shard_tt_1d(w["attn.add_v_proj.bias"], d, dim=1)

            # Row-parallel output proj (shard input dim=0), bias replicated
            self.img_to_out = shard_tt(w["attn.to_out.0.weight"].T.contiguous(), d, dim=0)
            self.txt_to_out = shard_tt(w["attn.to_add_out.weight"].T.contiguous(), d, dim=0)
            self.img_to_out_b = to_tt_1d(w["attn.to_out.0.bias"], d)
            self.txt_to_out_b = to_tt_1d(w["attn.to_add_out.bias"], d)

            # Column-parallel FFN fc1 (shard output dim=1), bias sharded
            self.img_ff1_w = shard_tt(w["img_mlp.net.0.proj.weight"].T.contiguous(), d, dim=1)
            self.txt_ff1_w = shard_tt(w["txt_mlp.net.0.proj.weight"].T.contiguous(), d, dim=1)
            self.img_ff1_b = shard_tt_1d(w["img_mlp.net.0.proj.bias"], d, dim=1)
            self.txt_ff1_b = shard_tt_1d(w["txt_mlp.net.0.proj.bias"], d, dim=1)

            # Row-parallel FFN fc2 (shard input dim=0), bias replicated
            self.img_ff2_w = shard_tt(w["img_mlp.net.2.weight"].T.contiguous(), d, dim=0)
            self.txt_ff2_w = shard_tt(w["txt_mlp.net.2.weight"].T.contiguous(), d, dim=0)
            self.img_ff2_b = to_tt_1d(w["img_mlp.net.2.bias"], d)
            self.txt_ff2_b = to_tt_1d(w["txt_mlp.net.2.bias"], d)
        else:
            self.img_to_q = to_tt(w["attn.to_q.weight"].T.contiguous(), d)
            self.img_to_k = to_tt(w["attn.to_k.weight"].T.contiguous(), d)
            self.img_to_v = to_tt(w["attn.to_v.weight"].T.contiguous(), d)
            self.txt_to_q = to_tt(w["attn.add_q_proj.weight"].T.contiguous(), d)
            self.txt_to_k = to_tt(w["attn.add_k_proj.weight"].T.contiguous(), d)
            self.txt_to_v = to_tt(w["attn.add_v_proj.weight"].T.contiguous(), d)
            self.img_to_out = to_tt(w["attn.to_out.0.weight"].T.contiguous(), d)
            self.txt_to_out = to_tt(w["attn.to_add_out.weight"].T.contiguous(), d)
            self.img_ff1_w = to_tt(w["img_mlp.net.0.proj.weight"].T.contiguous(), d)
            self.txt_ff1_w = to_tt(w["txt_mlp.net.0.proj.weight"].T.contiguous(), d)
            self.img_ff2_w = to_tt(w["img_mlp.net.2.weight"].T.contiguous(), d)
            self.txt_ff2_w = to_tt(w["txt_mlp.net.2.weight"].T.contiguous(), d)

            self.img_to_q_b = to_tt_1d(w["attn.to_q.bias"], d)
            self.img_to_k_b = to_tt_1d(w["attn.to_k.bias"], d)
            self.img_to_v_b = to_tt_1d(w["attn.to_v.bias"], d)
            self.txt_to_q_b = to_tt_1d(w["attn.add_q_proj.bias"], d)
            self.txt_to_k_b = to_tt_1d(w["attn.add_k_proj.bias"], d)
            self.txt_to_v_b = to_tt_1d(w["attn.add_v_proj.bias"], d)
            self.img_to_out_b = to_tt_1d(w["attn.to_out.0.bias"], d)
            self.txt_to_out_b = to_tt_1d(w["attn.to_add_out.bias"], d)
            self.img_ff1_b = to_tt_1d(w["img_mlp.net.0.proj.bias"], d)
            self.img_ff2_b = to_tt_1d(w["img_mlp.net.2.bias"], d)
            self.txt_ff1_b = to_tt_1d(w["txt_mlp.net.0.proj.bias"], d)
            self.txt_ff2_b = to_tt_1d(w["txt_mlp.net.2.bias"], d)

        # QK norm (replicated)
        self.norm_q = to_tt_1d(w["attn.norm_q.weight"], d)
        self.norm_k = to_tt_1d(w["attn.norm_k.weight"], d)
        self.norm_added_q = to_tt_1d(w["attn.norm_added_q.weight"], d)
        self.norm_added_k = to_tt_1d(w["attn.norm_added_k.weight"], d)

    def deallocate(self):
        for attr in list(vars(self).keys()):
            v = getattr(self, attr)
            if hasattr(v, 'deallocate'):
                try:
                    ttnn.deallocate(v)
                except Exception:
                    pass


# ============================================================
# AdaLN expansion via ttnn.concat (no custom kernel needed)
# ============================================================

def expand_adaln(silu_cond, mod_w, mod_b, n_repeat):
    """Compute adaLN params and expand: out = concat([silu_cond @ mod_w + mod_b] * n_repeat).
    silu_cond: (TILE, D) device tensor (2D, tile-padded from 1xD)
    mod_w: (D, 6*D) weight
    mod_b: (TILE, 6*D) pre-expanded bias
    Returns: (n_repeat*TILE, 6*D) device tensor with adaLN params repeated.
    """
    adaln_out = ttnn.linear(silu_cond, mod_w, bias=mod_b)
    return ttnn.concat([adaln_out] * n_repeat, dim=0)


# ============================================================
# Block forward
# ============================================================

def block_forward(blk, img_hs, txt_hs, temb_silu, device,
                  img_seq, txt_seq, img_seq_pad, txt_seq_pad,
                  img_cos=None, img_sin_perm=None,
                  txt_cos=None, txt_sin_perm=None,
                  swap_perm=None,
                  rope_kernel=None, rope_scratch=None,
                  adaln_scratch=None, gated_res_scratch=None):
    """Forward one MMDiT block on device. All tensors are 3D: [1, S, D].
    RoPE tables: [S_pad, heads_per_chip*HEAD_DIM] 2D on device.
    swap_perm: [HEAD_DIM, HEAD_DIM] permutation matrix for adjacent-element swap.
    rope_kernel: fused RoPE+layout tt-lang kernel (optional, falls back to ttnn).
    rope_scratch: dict of pre-allocated output tensors for rope_kernel."""
    D = HIDDEN_DIM
    use_tp = blk.use_tp and N_CHIPS > 1
    heads_per_chip = N_HEADS // N_CHIPS if use_tp else N_HEADS

    # AdaLN: compute mod params and expand via concat
    img_n_repeat = img_seq_pad // TILE
    txt_n_repeat = txt_seq_pad // TILE

    # temb_silu is [1, 32, D], reshape to 2D for matmul
    temb_2d = ttnn.reshape(temb_silu, (TILE, D))
    img_adaln = expand_adaln(temb_2d, blk.img_mod_w, blk.img_mod_b, img_n_repeat)
    txt_adaln = expand_adaln(temb_2d, blk.txt_mod_w, blk.txt_mod_b, txt_n_repeat)

    # Slice mod params: each is (S_pad, D) in 2D
    i_sh1 = ttnn.slice(img_adaln, [0, 0], [img_seq_pad, D])
    i_sc1 = ttnn.slice(img_adaln, [0, D], [img_seq_pad, 2*D])
    i_g1 = ttnn.slice(img_adaln, [0, 2*D], [img_seq_pad, 3*D])
    i_sh2 = ttnn.slice(img_adaln, [0, 3*D], [img_seq_pad, 4*D])
    i_sc2 = ttnn.slice(img_adaln, [0, 4*D], [img_seq_pad, 5*D])
    i_g2 = ttnn.slice(img_adaln, [0, 5*D], [img_seq_pad, 6*D])

    t_sh1 = ttnn.slice(txt_adaln, [0, 0], [txt_seq_pad, D])
    t_sc1 = ttnn.slice(txt_adaln, [0, D], [txt_seq_pad, 2*D])
    t_g1 = ttnn.slice(txt_adaln, [0, 2*D], [txt_seq_pad, 3*D])
    t_sh2 = ttnn.slice(txt_adaln, [0, 3*D], [txt_seq_pad, 4*D])
    t_sc2 = ttnn.slice(txt_adaln, [0, 4*D], [txt_seq_pad, 5*D])
    t_g2 = ttnn.slice(txt_adaln, [0, 5*D], [txt_seq_pad, 6*D])

    # AdaLN modulate via tt-lang kernel (slices are already 2D [S_pad, D])
    img_n = ttnn.layer_norm(img_hs)
    img_n_2d = ttnn.reshape(img_n, (img_seq_pad, D))
    adaln_modulate_kernel(img_n_2d, i_sh1, i_sc1, adaln_scratch["img"])
    img_m = ttnn.reshape(adaln_scratch["img"], (1, img_seq_pad, D))

    txt_n = ttnn.layer_norm(txt_hs)
    txt_n_2d = ttnn.reshape(txt_n, (txt_seq_pad, D))
    adaln_modulate_kernel(txt_n_2d, t_sh1, t_sc1, adaln_scratch["txt"])
    txt_m = ttnn.reshape(adaln_scratch["txt"], (1, txt_seq_pad, D))

    # QKV (column-parallel if TP, biases sharded to match)
    img_q = ttnn.linear(img_m, blk.img_to_q, bias=blk.img_to_q_b)
    img_k = ttnn.linear(img_m, blk.img_to_k, bias=blk.img_to_k_b)
    img_v = ttnn.linear(img_m, blk.img_to_v, bias=blk.img_to_v_b)
    txt_q = ttnn.linear(txt_m, blk.txt_to_q, bias=blk.txt_to_q_b)
    txt_k = ttnn.linear(txt_m, blk.txt_to_k, bias=blk.txt_to_k_b)
    txt_v = ttnn.linear(txt_m, blk.txt_to_v, bias=blk.txt_to_v_b)

    B = 1
    img_q = ttnn.reshape(img_q, (B, img_seq_pad, heads_per_chip, HEAD_DIM))
    img_k = ttnn.reshape(img_k, (B, img_seq_pad, heads_per_chip, HEAD_DIM))
    img_v = ttnn.reshape(img_v, (B, img_seq_pad, heads_per_chip, HEAD_DIM))
    txt_q = ttnn.reshape(txt_q, (B, txt_seq_pad, heads_per_chip, HEAD_DIM))
    txt_k = ttnn.reshape(txt_k, (B, txt_seq_pad, heads_per_chip, HEAD_DIM))
    txt_v = ttnn.reshape(txt_v, (B, txt_seq_pad, heads_per_chip, HEAD_DIM))

    # QK RMSNorm
    img_q = ttnn.rms_norm(img_q, weight=blk.norm_q)
    img_k = ttnn.rms_norm(img_k, weight=blk.norm_k)
    txt_q = ttnn.rms_norm(txt_q, weight=blk.norm_added_q)
    txt_k = ttnn.rms_norm(txt_k, weight=blk.norm_added_k)

    # RoPE via permutation matrix: swap AFTER QK-norm for correctness
    # q_swap = q @ P (adjacent element swap), then q_roped = q * cos + q_swap * sin_perm
    if img_cos is not None and swap_perm is not None:
        img_q_swap = ttnn.matmul(img_q, swap_perm)
        img_k_swap = ttnn.matmul(img_k, swap_perm)
        txt_q_swap = ttnn.matmul(txt_q, swap_perm)
        txt_k_swap = ttnn.matmul(txt_k, swap_perm)

        # Fused RoPE + layout via tt-lang kernel
        # Reshape 4D [1,S,H,D] -> 2D [S,H*D] for kernel
        d_per_chip = heads_per_chip * HEAD_DIM
        img_q_2d = ttnn.reshape(img_q, (img_seq_pad, d_per_chip))
        img_qs_2d = ttnn.reshape(img_q_swap, (img_seq_pad, d_per_chip))
        img_k_2d = ttnn.reshape(img_k, (img_seq_pad, d_per_chip))
        img_ks_2d = ttnn.reshape(img_k_swap, (img_seq_pad, d_per_chip))
        img_v_2d = ttnn.reshape(img_v, (img_seq_pad, d_per_chip))
        txt_q_2d = ttnn.reshape(txt_q, (txt_seq_pad, d_per_chip))
        txt_qs_2d = ttnn.reshape(txt_q_swap, (txt_seq_pad, d_per_chip))
        txt_k_2d = ttnn.reshape(txt_k, (txt_seq_pad, d_per_chip))
        txt_ks_2d = ttnn.reshape(txt_k_swap, (txt_seq_pad, d_per_chip))
        txt_v_2d = ttnn.reshape(txt_v, (txt_seq_pad, d_per_chip))

        # Run kernel: output is [H*S, D] (transposed layout)
        rope_kernel(img_q_2d, img_qs_2d, img_k_2d, img_ks_2d, img_v_2d,
                    img_cos, img_sin_perm,
                    rope_scratch["img_q"], rope_scratch["img_k"], rope_scratch["img_v"])
        rope_kernel(txt_q_2d, txt_qs_2d, txt_k_2d, txt_ks_2d, txt_v_2d,
                    txt_cos, txt_sin_perm,
                    rope_scratch["txt_q"], rope_scratch["txt_k"], rope_scratch["txt_v"])

        # Concat in transposed layout: [H*S_txt, D] + [H*S_img, D] -> [H*S_total, D]
        q = ttnn.concat([rope_scratch["txt_q"], rope_scratch["img_q"]], dim=0)
        k = ttnn.concat([rope_scratch["txt_k"], rope_scratch["img_k"]], dim=0)
        v = ttnn.concat([rope_scratch["txt_v"], rope_scratch["img_v"]], dim=0)
        S_total = txt_seq_pad + img_seq_pad
        q = ttnn.reshape(q, (B, heads_per_chip, S_total, HEAD_DIM))
        k = ttnn.reshape(k, (B, heads_per_chip, S_total, HEAD_DIM))
        v = ttnn.reshape(v, (B, heads_per_chip, S_total, HEAD_DIM))

    attn_out = ttnn.transformer.scaled_dot_product_attention(
        q, k, v, is_causal=False, scale=SCALE,
    )

    attn_out = ttnn.transpose(attn_out, 1, 2)
    S_total = txt_seq_pad + img_seq_pad
    d_out = heads_per_chip * HEAD_DIM
    attn_out = ttnn.reshape(attn_out, (B, S_total, d_out))

    txt_a = attn_out[:, :txt_seq_pad, :]
    img_a = attn_out[:, txt_seq_pad:, :]

    # Output proj (row-parallel if TP)
    img_a = ttnn.matmul(img_a, blk.img_to_out)
    txt_a = ttnn.matmul(txt_a, blk.txt_to_out)
    if use_tp:
        img_a = ttnn.all_reduce(img_a)
        txt_a = ttnn.all_reduce(txt_a)
    img_a = ttnn.add(img_a, blk.img_to_out_b)
    txt_a = ttnn.add(txt_a, blk.txt_to_out_b)

    # Gated residual (attention) via tt-lang kernel
    img_hs_2d = ttnn.reshape(img_hs, (img_seq_pad, D))
    img_a_2d = ttnn.reshape(img_a, (img_seq_pad, D))
    gated_residual_kernel(img_hs_2d, img_a_2d, i_g1, gated_res_scratch["img"])
    img_hs = ttnn.reshape(gated_res_scratch["img"], (1, img_seq_pad, D))

    txt_hs_2d = ttnn.reshape(txt_hs, (txt_seq_pad, D))
    txt_a_2d = ttnn.reshape(txt_a, (txt_seq_pad, D))
    gated_residual_kernel(txt_hs_2d, txt_a_2d, t_g1, gated_res_scratch["txt"])
    txt_hs = ttnn.reshape(gated_res_scratch["txt"], (1, txt_seq_pad, D))

    # FFN AdaLN modulate via tt-lang kernel
    img_n2 = ttnn.layer_norm(img_hs)
    img_n2_2d = ttnn.reshape(img_n2, (img_seq_pad, D))
    adaln_modulate_kernel(img_n2_2d, i_sh2, i_sc2, adaln_scratch["img"])
    img_m2 = ttnn.reshape(adaln_scratch["img"], (1, img_seq_pad, D))

    txt_n2 = ttnn.layer_norm(txt_hs)
    txt_n2_2d = ttnn.reshape(txt_n2, (txt_seq_pad, D))
    adaln_modulate_kernel(txt_n2_2d, t_sh2, t_sc2, adaln_scratch["txt"])
    txt_m2 = ttnn.reshape(adaln_scratch["txt"], (1, txt_seq_pad, D))

    # FFN fc1 + bias + GELU (column-parallel if TP, bias sharded to match)
    img_ff = ttnn.linear(img_m2, blk.img_ff1_w, bias=blk.img_ff1_b,
                         activation="gelu")
    txt_ff = ttnn.linear(txt_m2, blk.txt_ff1_w, bias=blk.txt_ff1_b,
                         activation="gelu")

    # FFN fc2 (row-parallel if TP)
    img_ff = ttnn.matmul(img_ff, blk.img_ff2_w)
    txt_ff = ttnn.matmul(txt_ff, blk.txt_ff2_w)
    if use_tp:
        img_ff = ttnn.all_reduce(img_ff)
        txt_ff = ttnn.all_reduce(txt_ff)
    img_ff = ttnn.add(img_ff, blk.img_ff2_b)
    txt_ff = ttnn.add(txt_ff, blk.txt_ff2_b)

    # Gated residual (FFN) via tt-lang kernel
    img_hs_2d = ttnn.reshape(img_hs, (img_seq_pad, D))
    img_ff_2d = ttnn.reshape(img_ff, (img_seq_pad, D))
    gated_residual_kernel(img_hs_2d, img_ff_2d, i_g2, gated_res_scratch["img"])
    img_hs = ttnn.reshape(gated_res_scratch["img"], (1, img_seq_pad, D))

    txt_hs_2d = ttnn.reshape(txt_hs, (txt_seq_pad, D))
    txt_ff_2d = ttnn.reshape(txt_ff, (txt_seq_pad, D))
    gated_residual_kernel(txt_hs_2d, txt_ff_2d, t_g2, gated_res_scratch["txt"])
    txt_hs = ttnn.reshape(gated_res_scratch["txt"], (1, txt_seq_pad, D))

    return img_hs, txt_hs


# ============================================================
# Weight loading
# ============================================================

def load_block_weights_from_safetensors(weights_dir, block_idx):
    """Load one block's weights from safetensors files."""
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


def extract_block_weights(transformer, block_idx):
    """Extract block weights from an already-loaded diffusers transformer.
    Much faster than re-reading safetensors (~0s vs ~2.8s per block)."""
    block = transformer.transformer_blocks[block_idx]
    return {k: v.float() for k, v in block.state_dict().items()}


# ============================================================
# Main generation
# ============================================================

def generate(weights_dir, prompt, width=512, height=512, num_steps=20, seed=42):
    from diffusers import DiffusionPipeline
    from PIL import Image
    from utils.image_utils import calculate_shift, pack_latents, unpack_latents

    total_start = time.perf_counter()

    # Load diffusers pipeline on CPU
    t0 = time.perf_counter()
    print("Loading diffusers pipeline on CPU...")
    pipe = DiffusionPipeline.from_pretrained(weights_dir, torch_dtype=torch.bfloat16)
    print(f"  [TIMING] from_pretrained: {time.perf_counter()-t0:.1f}s")
    t0 = time.perf_counter()
    transformer = pipe.transformer.eval()
    vae = pipe.vae
    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer
    scheduler = pipe.scheduler
    vae_sf = 2 ** len(vae.temperal_downsample) if hasattr(vae, "temperal_downsample") else 8
    print(f"  [TIMING] extract components: {time.perf_counter()-t0:.3f}s")

    width = (width // (vae_sf * 2)) * (vae_sf * 2)
    height = (height // (vae_sf * 2)) * (vae_sf * 2)
    print(f"  Resolution: {width}x{height}")

    # Open TT devices
    t0 = time.perf_counter()
    print(f"Opening {N_CHIPS} TT devices...")
    use_tp = N_CHIPS > 1
    if use_tp:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        device = ttnn.open_mesh_device(ttnn.MeshShape(1, N_CHIPS),
                                           trace_region_size=100000000)
    else:
        device = ttnn.open_device(device_id=0)
    print(f"  [TIMING] open device: {time.perf_counter()-t0:.1f}s")

    # Text encoding (CPU)
    t0 = time.perf_counter()
    print("Encoding text...")
    prompt_template = (
        "<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
        "texture, quantity, text, spatial relationships of the objects and background:"
        "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    )
    drop_idx = 34
    txt = prompt_template.format(prompt)
    tokens = tokenizer(txt, max_length=1024 + drop_idx, padding=True,
                       truncation=True, return_tensors="pt")
    with torch.no_grad():
        output = text_encoder(input_ids=tokens.input_ids,
                              attention_mask=tokens.attention_mask,
                              output_hidden_states=True)
    hidden_states = output.hidden_states[-1]
    bool_mask = tokens.attention_mask.bool()
    valid_len = bool_mask.sum(dim=1)
    selected = hidden_states[bool_mask]
    split_h = torch.split(selected, valid_len.tolist(), dim=0)
    split_h = [e[drop_idx:] for e in split_h]
    max_seq = max(e.size(0) for e in split_h)
    prompt_embeds = torch.stack(
        [torch.cat([u, u.new_zeros(max_seq - u.size(0), u.size(1))]) for u in split_h]
    ).to(torch.bfloat16)
    print(f"  Text: {prompt_embeds.shape}")
    print(f"  [TIMING] text encoding: {time.perf_counter()-t0:.1f}s")

    # Prepare latents
    latent_h = height // vae_sf
    latent_w = width // vae_sf
    num_channels = transformer.config.in_channels // 4
    generator = torch.Generator().manual_seed(seed)
    latents = torch.randn(1, 1, num_channels, latent_h, latent_w,
                           generator=generator, dtype=torch.bfloat16)
    latents = pack_latents(latents.squeeze(1), 1, num_channels, latent_h, latent_w)
    img_seq_len = latents.shape[1]
    print(f"  Latents: {latents.shape} (img_seq={img_seq_len})")

    # Compute padded sequence lengths
    img_seq_pad = ((img_seq_len + TILE - 1) // TILE) * TILE
    txt_seq_len = prompt_embeds.shape[1]
    txt_seq_pad = ((txt_seq_len + TILE - 1) // TILE) * TILE
    print(f"  Padded: img={img_seq_pad}, txt={txt_seq_pad}")

    # Scheduler
    sigmas = np.linspace(1.0, 1 / num_steps, num_steps)
    mu = calculate_shift(img_seq_len)
    scheduler.set_timesteps(num_steps, sigmas=sigmas, mu=mu)
    scheduler.set_begin_index(0)

    # RoPE: compute cos/sin tables on CPU as 2D [S_pad, H*D], upload once
    t0 = time.perf_counter()
    heads_per_chip = N_HEADS // N_CHIPS if use_tp else N_HEADS
    d_per_chip = heads_per_chip * HEAD_DIM
    head_tiles = HEAD_DIM // TILE
    print("Computing RoPE tables...")
    img_shapes = [[(1, latent_h // 2, latent_w // 2)]]
    txt_seq_lens_list = [txt_seq_len]
    img_fc, txt_fc = transformer.pos_embed(img_shapes, txt_seq_lens_list,
                                            device=torch.device("cpu"))

    def build_rope_tables_2d(complex_freqs, seq_pad, n_heads):
        """Build 2D [S_pad, H*D] cos/sin_perm tables for RoPE kernel."""
        cos_raw = complex_freqs.real.float().repeat_interleave(2, dim=-1)  # [S, D]
        sin_raw = complex_freqs.imag.float().repeat_interleave(2, dim=-1)  # [S, D]
        sign = torch.ones(HEAD_DIM)
        sign[0::2] = -1
        sin_perm_raw = sin_raw * sign.unsqueeze(0)  # [S, D]
        S = cos_raw.shape[0]
        # Repeat across heads: [S, D] -> [S, H*D]
        cos_2d = cos_raw[:min(S, seq_pad)].repeat(1, n_heads)
        sin_perm_2d = sin_perm_raw[:min(S, seq_pad)].repeat(1, n_heads)
        if S < seq_pad:
            cos_2d = F.pad(cos_2d, (0, 0, 0, seq_pad - S))
            sin_perm_2d = F.pad(sin_perm_2d, (0, 0, 0, seq_pad - S))
        return cos_2d.contiguous().bfloat16(), sin_perm_2d.contiguous().bfloat16()

    img_cos_pt, img_sin_perm_pt = build_rope_tables_2d(img_fc, img_seq_pad, heads_per_chip)
    txt_cos_pt, txt_sin_perm_pt = build_rope_tables_2d(txt_fc, txt_seq_pad, heads_per_chip)
    img_cos_tt = to_tt(img_cos_pt, device)
    img_sin_perm_tt = to_tt(img_sin_perm_pt, device)
    txt_cos_tt = to_tt(txt_cos_pt, device)
    txt_sin_perm_tt = to_tt(txt_sin_perm_pt, device)
    print(f"  RoPE: img_cos {img_cos_tt.shape}, txt_cos {txt_cos_tt.shape}")

    # Permutation matrix for adjacent-element swap (used for RoPE after QK-norm)
    swap_perm_tt = to_tt(make_swap_perm_matrix(HEAD_DIM), device)

    # Build tt-lang RoPE kernel + pre-allocate scratch outputs
    rope_kernel = make_rope_layout_kernel(heads_per_chip, head_tiles)
    rope_scratch = {
        "img_q": zeros_tt((heads_per_chip * img_seq_pad, HEAD_DIM), device),
        "img_k": zeros_tt((heads_per_chip * img_seq_pad, HEAD_DIM), device),
        "img_v": zeros_tt((heads_per_chip * img_seq_pad, HEAD_DIM), device),
        "txt_q": zeros_tt((heads_per_chip * txt_seq_pad, HEAD_DIM), device),
        "txt_k": zeros_tt((heads_per_chip * txt_seq_pad, HEAD_DIM), device),
        "txt_v": zeros_tt((heads_per_chip * txt_seq_pad, HEAD_DIM), device),
    }
    # AdaLN modulate scratch (2D): reused across all blocks
    adaln_scratch = {
        "img": zeros_tt((img_seq_pad, HIDDEN_DIM), device),
        "txt": zeros_tt((txt_seq_pad, HIDDEN_DIM), device),
    }
    # Gated residual scratch (2D): reused across all blocks
    gated_res_scratch = {
        "img": zeros_tt((img_seq_pad, HIDDEN_DIM), device),
        "txt": zeros_tt((txt_seq_pad, HIDDEN_DIM), device),
    }
    print(f"  RoPE kernel built, scratch allocated")
    print(f"  [TIMING] RoPE + upload: {time.perf_counter()-t0:.1f}s")

    # Preload all 60 blocks onto device with TP sharding
    print("Preloading all 60 blocks onto device...")
    preload_start = time.perf_counter()
    blocks = []
    for bi in range(60):
        w = extract_block_weights(transformer, bi)
        blocks.append(TTNNBlock(w, device, use_tp=use_tp,
                                img_seq_pad=img_seq_pad, txt_seq_pad=txt_seq_pad))
        del w
        if (bi + 1) % 10 == 0:
            print(f"  Loaded {bi+1}/60 blocks")
    preload_time = time.perf_counter() - preload_start
    print(f"  Preloaded 60 blocks in {preload_time:.1f}s")
    gc.collect()

    # Warmup: run one forward pass to compile all kernels
    print("Warmup pass...")
    warmup_start = time.perf_counter()
    with torch.no_grad():
        hs_cpu = transformer.img_in(latents)
        ehs_cpu = transformer.txt_in(transformer.txt_norm(prompt_embeds))
        ts_w = scheduler.timesteps[0].expand(1).to(torch.bfloat16) / 1000
        temb_cpu = transformer.time_text_embed(ts_w.to(hs_cpu.dtype), hs_cpu).unsqueeze(1)
        # Expand temb to fill all TILE rows so concat-repeat gives correct values
        temb_cpu = temb_cpu.expand(1, TILE, -1).contiguous()

    img_tt = to_tt(hs_cpu, device)
    txt_tt = to_tt(ehs_cpu, device)
    temb_tt = to_tt(temb_cpu, device)
    temb_silu = ttnn.silu(temb_tt)

    for bi in range(60):
        img_tt, txt_tt = block_forward(blocks[bi], img_tt, txt_tt, temb_silu,
                                       device, img_seq_len, txt_seq_len,
                                       img_seq_pad, txt_seq_pad,
                                       img_cos_tt, img_sin_perm_tt,
                                       txt_cos_tt, txt_sin_perm_tt,
                                       swap_perm_tt,
                                       rope_kernel, rope_scratch,
                                       adaln_scratch, gated_res_scratch)
    ttnn.synchronize_device(device)
    print(f"  Warmup: {time.perf_counter()-warmup_start:.1f}s")

    # Trace capture: silu + 60-block forward
    def run_traced_forward():
        nonlocal img_tt, txt_tt
        temb_silu_t = ttnn.silu(temb_tt)
        for bi in range(60):
            img_tt, txt_tt = block_forward(blocks[bi], img_tt, txt_tt, temb_silu_t,
                                           device, img_seq_len, txt_seq_len,
                                           img_seq_pad, txt_seq_pad,
                                           img_cos_tt, img_sin_perm_tt,
                                           txt_cos_tt, txt_sin_perm_tt,
                                           swap_perm_tt,
                                           rope_kernel, rope_scratch,
                                       adaln_scratch, gated_res_scratch)

    print("Capturing trace...")
    trace_start = time.perf_counter()
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    run_traced_forward()
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    ttnn.synchronize_device(device)
    print(f"  Trace captured in {time.perf_counter()-trace_start:.1f}s")

    # Helper: create host tensor for copy_host_to_device_tensor
    def to_tt_host(t):
        """Create a tilized host tensor (no device) for copy_host_to_device_tensor."""
        if t.dim() == 1:
            t = t.unsqueeze(0)
        h, w_dim = t.shape[-2], t.shape[-1]
        ph = ((h + TILE - 1) // TILE) * TILE - h
        pw = ((w_dim + TILE - 1) // TILE) * TILE - w_dim
        if ph > 0 or pw > 0:
            t = F.pad(t, (0, pw, 0, ph))
        return ttnn.from_torch(t.contiguous().to(torch.bfloat16),
                               dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    # Denoising loop with tracing
    print(f"Denoising ({num_steps} steps, 60 blocks each, TRACED)...")
    denoise_start = time.perf_counter()

    for step_i, t_val in enumerate(scheduler.timesteps):
        step_start = time.perf_counter()
        ts = t_val.expand(1).to(torch.bfloat16) / 1000

        with torch.no_grad():
            # Pre-block ops on CPU
            t0 = time.perf_counter()
            hs_cpu = transformer.img_in(latents)
            ehs_cpu = transformer.txt_in(transformer.txt_norm(prompt_embeds))
            temb_cpu = transformer.time_text_embed(ts.to(hs_cpu.dtype), hs_cpu)
            temb_cpu_orig = temb_cpu
            temb_cpu = temb_cpu.unsqueeze(1).expand(1, TILE, -1).contiguous()
            cpu_pre_ms = (time.perf_counter() - t0) * 1000

            # Copy inputs to pre-allocated device tensors (no allocation!)
            t0 = time.perf_counter()
            ttnn.copy_host_to_device_tensor(to_tt_host(hs_cpu), img_tt)
            ttnn.copy_host_to_device_tensor(to_tt_host(ehs_cpu), txt_tt)
            ttnn.copy_host_to_device_tensor(to_tt_host(temb_cpu), temb_tt)
            h2d_ms = (time.perf_counter() - t0) * 1000

            # Execute traced forward (zero dispatch overhead!)
            t0 = time.perf_counter()
            ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
            dispatch_ms = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            ttnn.synchronize_device(device)
            sync_ms = (time.perf_counter() - t0) * 1000

            # Post-block ops on CPU
            t0 = time.perf_counter()
            hs_out = from_tt(img_tt, device)[:, :img_seq_len, :HIDDEN_DIM].to(torch.bfloat16)
            d2h_ms = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            hs_out = transformer.norm_out(hs_out, temb_cpu_orig)
            noise_pred = transformer.proj_out(hs_out)
            cpu_post_ms = (time.perf_counter() - t0) * 1000

        # Scheduler step
        t0 = time.perf_counter()
        latents = scheduler.step(noise_pred, t_val, latents, return_dict=False)[0]
        sched_ms = (time.perf_counter() - t0) * 1000

        step_ms = (time.perf_counter() - step_start) * 1000
        elapsed = time.perf_counter() - denoise_start
        print(f"  Step {step_i+1}/{num_steps}: {step_ms:.0f}ms "
              f"[cpu_pre={cpu_pre_ms:.0f} h2d={h2d_ms:.0f} dispatch={dispatch_ms:.0f} "
              f"sync={sync_ms:.0f} d2h={d2h_ms:.0f} cpu_post={cpu_post_ms:.0f} "
              f"sched={sched_ms:.0f}]")

    denoise_time = time.perf_counter() - denoise_start
    print(f"  Denoising: {denoise_time:.1f}s ({denoise_time/num_steps:.2f}s/step)")

    # VAE decode (CPU)
    print("VAE decode...")
    lf = latents
    lf = unpack_latents(lf, height, width, vae_sf).to(vae.dtype)
    if hasattr(vae.config, "latents_mean") and vae.config.latents_mean is not None:
        lm = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(lf.device, lf.dtype)
        ls = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(lf.device, lf.dtype)
        lf = lf / ls + lm
    with torch.no_grad():
        image = vae.decode(lf, return_dict=False)[0]
        if image.dim() == 5:
            image = image[:, :, 0]

    image = (torch.clamp(image / 2 + 0.5, 0.0, 1.0) * 255).to(torch.uint8)
    output_path = "/tmp/output_ttnn.png"
    Image.fromarray(image[0].cpu().numpy().transpose(1, 2, 0)).save(output_path)
    print(f"Saved: {output_path}")
    print(f"Total: {time.perf_counter()-total_start:.1f}s")

    for blk in blocks:
        blk.deallocate()
    del blocks

    if use_tp:
        ttnn.close_mesh_device(device)
    else:
        ttnn.close_device(device)


if __name__ == "__main__":
    N_CHIPS = 4
    generate(
        weights_dir="/workspace/qwen-image-tt-xla/weights/qwen-image",
        prompt="A beautiful sunset over mountains",
        width=512, height=512,
        num_steps=3, seed=42,
    )
