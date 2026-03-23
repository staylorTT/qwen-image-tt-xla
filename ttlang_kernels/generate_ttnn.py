"""End-to-end image generation using pure ttnn/tt-lang for the 60-block MMDiT.

Uses 4 Blackhole devices with Tensor Parallelism (TP) following oasis patterns:
- Column-parallel: QKV projections, FFN fc1 (shard output dim)
- Row-parallel: output projections, FFN fc2 (shard input dim) + all_reduce
- All 60 blocks' weights preloaded and kept resident across 4 devices

CPU handles: text encoding, timestep embedding, img_in/txt_in, final norm/proj,
             scheduler, VAE decode.
Device handles: all 60 MMDiT transformer blocks (the expensive part).

Usage (on remote):
    python3 generate_ttnn.py --prompt "A cat" --width 512 --height 512 --steps 20
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

from broadcast_row import broadcast_row_kernel

TILE = 32
N_HEADS = 24
HEAD_DIM = 128
HIDDEN_DIM = N_HEADS * HEAD_DIM  # 3072
SCALE = 1.0 / math.sqrt(HEAD_DIM)
N_CHIPS = 4


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
        **_mesh_kwargs(device),
    )
    broadcast_row_kernel(mod_2d, out_2d)
    out_3d = ttnn.reshape(out_2d, (1, seq_padded, D_padded))
    return ttnn.clone(out_3d, memory_config=ttnn.DRAM_MEMORY_CONFIG)


class TTNNBlock:
    """One MMDiT block weights on device with TP sharding."""

    def __init__(self, weights, device, use_tp=False):
        self.device = device
        self.use_tp = use_tp
        d = device
        w = weights

        # AdaLN MLP (replicated - small, runs on all chips)
        self.img_mod_w = to_tt(w["img_mod.1.weight"].T.contiguous(), d)
        self.img_mod_b = to_tt_1d(w["img_mod.1.bias"], d)
        self.txt_mod_w = to_tt(w["txt_mod.1.weight"].T.contiguous(), d)
        self.txt_mod_b = to_tt_1d(w["txt_mod.1.bias"], d)

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

            # Row-parallel output proj (shard input dim=0), bias replicated (added after all_reduce)
            self.img_to_out = shard_tt(w["attn.to_out.0.weight"].T.contiguous(), d, dim=0)
            self.txt_to_out = shard_tt(w["attn.to_add_out.weight"].T.contiguous(), d, dim=0)
            self.img_to_out_b = to_tt_1d(w["attn.to_out.0.bias"], d)
            self.txt_to_out_b = to_tt_1d(w["attn.to_add_out.bias"], d)

            # Column-parallel FFN fc1 (shard output dim=1), bias sharded
            self.img_ff1_w = shard_tt(w["img_mlp.net.0.proj.weight"].T.contiguous(), d, dim=1)
            self.txt_ff1_w = shard_tt(w["txt_mlp.net.0.proj.weight"].T.contiguous(), d, dim=1)
            self.img_ff1_b = shard_tt_1d(w["img_mlp.net.0.proj.bias"], d, dim=1)
            self.txt_ff1_b = shard_tt_1d(w["txt_mlp.net.0.proj.bias"], d, dim=1)

            # Row-parallel FFN fc2 (shard input dim=0), bias replicated (added after all_reduce)
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

            # Biases (replicated for single-chip)
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
        """Free all device tensors."""
        for attr in list(vars(self).keys()):
            v = getattr(self, attr)
            if hasattr(v, 'deallocate'):
                try:
                    ttnn.deallocate(v)
                except Exception:
                    pass


def block_forward(blk, img_hs, txt_hs, temb_tt, device):
    """Forward one MMDiT block on device."""
    img_seq = img_hs.shape[-2]
    txt_seq = txt_hs.shape[-2]
    D = HIDDEN_DIM
    use_tp = blk.use_tp and N_CHIPS > 1
    heads_per_chip = N_HEADS // N_CHIPS if use_tp else N_HEADS

    # AdaLN
    temb_silu = ttnn.silu(temb_tt)
    img_mod = ttnn.linear(temb_silu, blk.img_mod_w, bias=blk.img_mod_b)
    txt_mod = ttnn.linear(temb_silu, blk.txt_mod_w, bias=blk.txt_mod_b)

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

    i_sh1_e = expand_mod_tt(i_sh1, img_seq, device)
    i_sc1_e = expand_mod_tt(i_sc1, img_seq, device)
    i_g1_e = expand_mod_tt(i_g1, img_seq, device)
    t_sh1_e = expand_mod_tt(t_sh1, txt_seq, device)
    t_sc1_e = expand_mod_tt(t_sc1, txt_seq, device)
    t_g1_e = expand_mod_tt(t_g1, txt_seq, device)

    # LN + modulate
    img_n = ttnn.layer_norm(img_hs)
    img_m = ttnn.add(ttnn.multiply(img_n, ttnn.add(i_sc1_e, 1.0)), i_sh1_e)
    txt_n = ttnn.layer_norm(txt_hs)
    txt_m = ttnn.add(ttnn.multiply(txt_n, ttnn.add(t_sc1_e, 1.0)), t_sh1_e)

    # QKV (column-parallel if TP, biases sharded to match)
    img_q = ttnn.linear(img_m, blk.img_to_q, bias=blk.img_to_q_b)
    img_k = ttnn.linear(img_m, blk.img_to_k, bias=blk.img_to_k_b)
    img_v = ttnn.linear(img_m, blk.img_to_v, bias=blk.img_to_v_b)
    txt_q = ttnn.linear(txt_m, blk.txt_to_q, bias=blk.txt_to_q_b)
    txt_k = ttnn.linear(txt_m, blk.txt_to_k, bias=blk.txt_to_k_b)
    txt_v = ttnn.linear(txt_m, blk.txt_to_v, bias=blk.txt_to_v_b)

    B = 1
    img_q = ttnn.reshape(img_q, (B, img_seq, heads_per_chip, HEAD_DIM))
    img_k = ttnn.reshape(img_k, (B, img_seq, heads_per_chip, HEAD_DIM))
    img_v = ttnn.reshape(img_v, (B, img_seq, heads_per_chip, HEAD_DIM))
    txt_q = ttnn.reshape(txt_q, (B, txt_seq, heads_per_chip, HEAD_DIM))
    txt_k = ttnn.reshape(txt_k, (B, txt_seq, heads_per_chip, HEAD_DIM))
    txt_v = ttnn.reshape(txt_v, (B, txt_seq, heads_per_chip, HEAD_DIM))

    # QK RMSNorm
    img_q = ttnn.rms_norm(img_q, weight=blk.norm_q)
    img_k = ttnn.rms_norm(img_k, weight=blk.norm_k)
    txt_q = ttnn.rms_norm(txt_q, weight=blk.norm_added_q)
    txt_k = ttnn.rms_norm(txt_k, weight=blk.norm_added_k)

    # Joint SDPA
    q = ttnn.concat([txt_q, img_q], dim=1)
    k = ttnn.concat([txt_k, img_k], dim=1)
    v = ttnn.concat([txt_v, img_v], dim=1)
    q = ttnn.transpose(q, 1, 2)
    k = ttnn.transpose(k, 1, 2)
    v = ttnn.transpose(v, 1, 2)

    attn_out = ttnn.transformer.scaled_dot_product_attention(
        q, k, v, is_causal=False, scale=SCALE,
    )

    attn_out = ttnn.transpose(attn_out, 1, 2)
    S_total = txt_seq + img_seq
    d_out = heads_per_chip * HEAD_DIM
    attn_out = ttnn.reshape(attn_out, (B, S_total, d_out))

    txt_a = attn_out[:, :txt_seq, :]
    img_a = attn_out[:, txt_seq:, :]

    # Output proj (row-parallel if TP)
    img_a = ttnn.matmul(img_a, blk.img_to_out)
    txt_a = ttnn.matmul(txt_a, blk.txt_to_out)
    if use_tp:
        img_a = ttnn.all_reduce(img_a)
        txt_a = ttnn.all_reduce(txt_a)
    img_a = ttnn.add(img_a, blk.img_to_out_b)
    txt_a = ttnn.add(txt_a, blk.txt_to_out_b)

    # Gated residual
    img_hs = ttnn.add(img_hs, ttnn.multiply(i_g1_e, img_a))
    txt_hs = ttnn.add(txt_hs, ttnn.multiply(t_g1_e, txt_a))

    # FFN
    i_sh2_e = expand_mod_tt(i_sh2, img_seq, device)
    i_sc2_e = expand_mod_tt(i_sc2, img_seq, device)
    i_g2_e = expand_mod_tt(i_g2, img_seq, device)
    t_sh2_e = expand_mod_tt(t_sh2, txt_seq, device)
    t_sc2_e = expand_mod_tt(t_sc2, txt_seq, device)
    t_g2_e = expand_mod_tt(t_g2, txt_seq, device)

    img_n2 = ttnn.layer_norm(img_hs)
    img_m2 = ttnn.add(ttnn.multiply(img_n2, ttnn.add(i_sc2_e, 1.0)), i_sh2_e)
    txt_n2 = ttnn.layer_norm(txt_hs)
    txt_m2 = ttnn.add(ttnn.multiply(txt_n2, ttnn.add(t_sc2_e, 1.0)), t_sh2_e)

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

    img_hs = ttnn.add(img_hs, ttnn.multiply(i_g2_e, img_ff))
    txt_hs = ttnn.add(txt_hs, ttnn.multiply(t_g2_e, txt_ff))

    return img_hs, txt_hs


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


def generate(weights_dir, prompt, width=512, height=512, num_steps=20, seed=42):
    from diffusers import DiffusionPipeline
    from PIL import Image
    from utils.image_utils import calculate_shift, pack_latents, unpack_latents

    total_start = time.perf_counter()

    # Load diffusers pipeline on CPU
    print("Loading diffusers pipeline on CPU...")
    pipe = DiffusionPipeline.from_pretrained(weights_dir, torch_dtype=torch.bfloat16)
    transformer = pipe.transformer.eval()
    vae = pipe.vae
    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer
    scheduler = pipe.scheduler
    vae_sf = 2 ** len(vae.temperal_downsample) if hasattr(vae, "temperal_downsample") else 8

    width = (width // (vae_sf * 2)) * (vae_sf * 2)
    height = (height // (vae_sf * 2)) * (vae_sf * 2)
    print(f"  Resolution: {width}x{height}")

    # Open TT devices
    print(f"Opening {N_CHIPS} TT devices...")
    use_tp = N_CHIPS > 1
    if use_tp:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        device = ttnn.open_mesh_device(ttnn.MeshShape(1, N_CHIPS),
                                           trace_region_size=100000000)
    else:
        device = ttnn.open_device(device_id=0)

    # SDPA uses defaults (no explicit program_config needed)

    # Text encoding (CPU)
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

    # Scheduler
    sigmas = np.linspace(1.0, 1 / num_steps, num_steps)
    mu = calculate_shift(img_seq_len)
    scheduler.set_timesteps(num_steps, sigmas=sigmas, mu=mu)
    scheduler.set_begin_index(0)

    # Preload all 60 blocks onto device with TP sharding
    print("Preloading all 60 blocks onto device...")
    preload_start = time.perf_counter()
    blocks = []
    for bi in range(60):
        w = load_block_weights_from_safetensors(weights_dir, bi)
        blocks.append(TTNNBlock(w, device, use_tp=use_tp))
        del w
        if (bi + 1) % 10 == 0:
            print(f"  Loaded {bi+1}/60 blocks")
    preload_time = time.perf_counter() - preload_start
    print(f"  Preloaded 60 blocks in {preload_time:.1f}s")
    gc.collect()

    # Denoising loop
    print(f"Denoising ({num_steps} steps, 60 blocks each)...")
    denoise_start = time.perf_counter()

    for step_i, t_val in enumerate(scheduler.timesteps):
        step_start = time.perf_counter()
        ts = t_val.expand(1).to(torch.bfloat16) / 1000

        with torch.no_grad():
            # Pre-block ops on CPU
            hs_cpu = transformer.img_in(latents)
            ehs_cpu = transformer.txt_in(transformer.txt_norm(prompt_embeds))
            temb_cpu = transformer.time_text_embed(ts.to(hs_cpu.dtype), hs_cpu)
            temb_cpu = temb_cpu.unsqueeze(1)

            # Move to device
            img_tt = to_tt(hs_cpu, device)
            txt_tt = to_tt(ehs_cpu, device)
            temb_tt = to_tt(temb_cpu, device)

            # 60 blocks on device (weights already resident)
            for bi in range(60):
                img_tt, txt_tt = block_forward(blocks[bi], img_tt, txt_tt, temb_tt,
                                               device)

            ttnn.synchronize_device(device)

            # Post-block ops on CPU
            hs_out = from_tt(img_tt, device)[:, :img_seq_len, :HIDDEN_DIM].to(torch.bfloat16)
            hs_out = transformer.norm_out(hs_out, temb_cpu.squeeze(1))
            noise_pred = transformer.proj_out(hs_out)

        # Scheduler step
        latents = scheduler.step(noise_pred, t_val, latents, return_dict=False)[0]

        step_ms = (time.perf_counter() - step_start) * 1000
        elapsed = time.perf_counter() - denoise_start
        print(f"  Step {step_i+1}/{num_steps}: {step_ms:.0f}ms (total {elapsed:.1f}s)")

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
        width=256, height=256,
        num_steps=1, seed=42,
    )
