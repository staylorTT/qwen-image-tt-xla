"""Dump the TTNN IR for a single compiled MMDiT block.

Run with: ./run.sh dump_ttnn_ir.py
"""

import math
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch_xla
import torch_xla.runtime as xr

os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
os.environ["TTXLA_LOGGER_LEVEL"] = "DEBUG"
xr.use_spmd()
xr.set_device_type("TT")

from diffusers import DiffusionPipeline
from utils.image_utils import pack_latents

# Set export path to dump IRs
EXPORT_DIR = "./ir_dump"
torch_xla.set_custom_compile_options({
    "export_path": EXPORT_DIR,
})

def complex_to_real_rope(complex_freqs):
    cos = complex_freqs.real.float().repeat_interleave(2, dim=-1).unsqueeze(1)
    sin = complex_freqs.imag.float().repeat_interleave(2, dim=-1).unsqueeze(1)
    return (cos, sin)

def apply_rope_real(x, cos, sin):
    x_r, x_i = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
    x_rot = torch.stack([-x_i, x_r], dim=-1).flatten(3)
    return (x.float() * cos + x_rot.float() * sin).to(x.dtype)

def manual_attention(q, k, v):
    scale = 1.0 / math.sqrt(q.shape[-1])
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn_weights = torch.softmax(attn_weights, dim=-1)
    return torch.matmul(attn_weights, v)

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

    print("Loading model...")
    pipe = DiffusionPipeline.from_pretrained(weights_dir, torch_dtype=torch.bfloat16)
    transformer = pipe.transformer.eval()
    config = transformer.config

    # Prepare inputs (256x256)
    width, height = 256, 256
    vae_sf = 8
    lh, lw = height // vae_sf, width // vae_sf
    nc = config.in_channels // 4

    prompt_embeds, prompt_mask = pipe.encode_prompt(prompt="A cat", device="cpu")
    # Pad to 128
    max_txt = 128
    prompt_embeds = F.pad(prompt_embeds, (0, 0, 0, max_txt - prompt_embeds.shape[1]))
    prompt_mask = F.pad(prompt_mask, (0, max_txt - prompt_mask.shape[1]), value=0)

    gen = torch.Generator().manual_seed(42)
    latents = pack_latents(
        torch.randn(1, nc, lh, lw, generator=gen, dtype=torch.bfloat16), 1, nc, lh, lw,
    )

    img_shapes = [[(1, lh // 2, lw // 2)]]
    txt_seq_lens = [max_txt]

    # Compute inputs
    timestep_div = torch.tensor([999.0], dtype=torch.bfloat16) / 1000
    with torch.no_grad():
        hs_in = transformer.img_in(latents)
        ehs_in = transformer.txt_in(transformer.txt_norm(prompt_embeds))
        temb = transformer.time_text_embed(timestep_div.to(hs_in.dtype), hs_in)

    # RoPE
    img_fc, txt_fc = transformer.pos_embed(img_shapes, txt_seq_lens, device=torch.device("cpu"))
    img_rope = complex_to_real_rope(img_fc)
    txt_rope = complex_to_real_rope(txt_fc)

    # Move block 0 to device and compile
    print(f"Compiling block 0 with IR dump to {EXPORT_DIR}/...")
    block = transformer.transformer_blocks[0].to(device)

    def make_fn(b):
        def fn(hs, ehs, em, temb, rope):
            return patched_block_forward(b, hs, ehs, em, temb, rope)
        return fn

    compiled = torch.compile(make_fn(block), backend="tt")

    rope_dev = (
        (img_rope[0].to(torch.bfloat16).to(device), img_rope[1].to(torch.bfloat16).to(device)),
        (txt_rope[0].to(torch.bfloat16).to(device), txt_rope[1].to(torch.bfloat16).to(device)),
    )

    with torch.no_grad():
        txt_out, img_out = compiled(
            hs_in.to(device), ehs_in.to(device),
            prompt_mask.to(torch.bfloat16).to(device), temb.to(device),
            rope_dev,
        )
    torch_xla.sync()

    print(f"\nDone! Check {EXPORT_DIR}/irs/ for:")
    print("  - shlo_*.mlir  (StableHLO)")
    print("  - ttir_*.mlir  (TTIR)")
    print("  - ttnn_*.mlir  (TTNN)")

    # List what was dumped
    irs_dir = os.path.join(EXPORT_DIR, "irs")
    if os.path.exists(irs_dir):
        files = sorted(os.listdir(irs_dir))
        print(f"\nDumped {len(files)} IR files:")
        for f in files:
            size = os.path.getsize(os.path.join(irs_dir, f))
            print(f"  {f} ({size // 1024}KB)")
    else:
        print(f"\nNo irs/ directory found. Checking {EXPORT_DIR}:")
        if os.path.exists(EXPORT_DIR):
            for f in sorted(os.listdir(EXPORT_DIR)):
                print(f"  {f}")


if __name__ == "__main__":
    main()
