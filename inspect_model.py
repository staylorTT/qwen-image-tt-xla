# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Phase 0.3: Inspect Qwen-Image MMDiT model architecture.

Discovers the dual-stream MMDiT structure, prints every Linear layer path and shape,
documents the attention mechanism, AdaLN structure, and RoPE configuration.

This is the most critical discovery step — the output of this script determines
the correct attribute paths for TP sharding, AdaLN caching, and all downstream code.

Usage:
    python inspect_model.py [--weights-dir ./weights/qwen-image]
"""

import argparse

import torch
import torch.nn as nn


def inspect_transformer(weights_dir: str):
    """Inspect QwenImageTransformer2DModel architecture."""
    from diffusers import DiffusionPipeline

    print("=" * 80)
    print("Loading pipeline (CPU, bf16)...")
    print("=" * 80)

    pipe = DiffusionPipeline.from_pretrained(
        weights_dir,
        torch_dtype=torch.bfloat16,
    )

    # --- Pipeline Components ---
    print("\n=== Pipeline Components ===")
    print(f"Components: {list(pipe.components.keys())}")

    transformer = pipe.transformer
    print(f"\nTransformer class: {type(transformer).__name__}")
    print(f"Transformer config: {transformer.config}")

    # --- Config Validation ---
    print("\n=== Config Validation ===")
    config = transformer.config
    expected = {
        "num_layers": 60,
        "attention_head_dim": 128,
        "num_attention_heads": 24,
        "joint_attention_dim": 3584,
        "in_channels": 64,
        "out_channels": 16,
        "patch_size": 2,
        "axes_dims_rope": [16, 56, 56],
    }
    for key, expected_val in expected.items():
        actual = getattr(config, key, "MISSING")
        status = "OK" if str(actual) == str(expected_val) else "MISMATCH"
        print(f"  {key}: {actual} (expected {expected_val}) [{status}]")

    # --- Parameter Count ---
    print("\n=== Parameter Counts ===")
    total_params = sum(p.numel() for p in transformer.parameters())
    print(f"  Total transformer params: {total_params / 1e9:.2f}B ({total_params * 2 / 1e9:.1f}GB bf16)")

    # Count AdaLN parameters (img_mod + txt_mod in each block)
    adaln_params = 0
    for block in transformer.transformer_blocks:
        for name, param in block.named_parameters():
            if "mod" in name:  # img_mod, txt_mod
                adaln_params += param.numel()
    # Also count norm_out (AdaLayerNormContinuous)
    for param in transformer.norm_out.parameters():
        adaln_params += param.numel()
    print(f"  AdaLN params: {adaln_params / 1e9:.2f}B ({adaln_params * 2 / 1e9:.1f}GB bf16)")
    print(f"  Non-AdaLN params: {(total_params - adaln_params) / 1e9:.2f}B")

    # --- All Linear Layers ---
    print("\n=== All Linear Layers (full paths and shapes) ===")
    linear_count = 0
    for name, module in transformer.named_modules():
        if isinstance(module, nn.Linear):
            bias_str = f", bias={module.bias.shape}" if module.bias is not None else ", no bias"
            print(f"  {name}: weight={list(module.weight.shape)}{bias_str}")
            linear_count += 1
    print(f"  Total Linear layers: {linear_count}")

    # --- Block Structure ---
    print("\n=== Single Block Structure (block 0) ===")
    block = transformer.transformer_blocks[0]
    print(f"  Block class: {type(block).__name__}")
    print(f"  Block attributes:")
    for name, child in block.named_children():
        print(f"    {name}: {type(child).__name__}")
        if hasattr(child, "named_children"):
            for subname, subchild in child.named_children():
                print(f"      {name}.{subname}: {type(subchild).__name__}")

    # --- Attention Structure ---
    print("\n=== Attention Structure (block 0) ===")
    attn = block.attn
    print(f"  Attention class: {type(attn).__name__}")
    print(f"  heads: {attn.heads}")
    print(f"  Image-stream projections:")
    print(f"    to_q:     {list(attn.to_q.weight.shape)}")
    print(f"    to_k:     {list(attn.to_k.weight.shape)}")
    print(f"    to_v:     {list(attn.to_v.weight.shape)}")
    print(f"    to_out[0]:{list(attn.to_out[0].weight.shape)}")
    print(f"  Text-stream projections:")
    print(f"    add_q_proj: {list(attn.add_q_proj.weight.shape)}")
    print(f"    add_k_proj: {list(attn.add_k_proj.weight.shape)}")
    print(f"    add_v_proj: {list(attn.add_v_proj.weight.shape)}")
    print(f"    to_add_out: {list(attn.to_add_out.weight.shape)}")

    # Check QK norm
    print(f"  QK norm (img):  norm_q={type(attn.norm_q).__name__}, norm_k={type(attn.norm_k).__name__}")
    print(f"  QK norm (txt):  norm_added_q={type(attn.norm_added_q).__name__}, norm_added_k={type(attn.norm_added_k).__name__}")

    # --- FFN Structure ---
    print("\n=== FFN Structure (block 0) ===")
    print(f"  Image FFN class: {type(block.img_mlp).__name__}")
    for name, mod in block.img_mlp.named_modules():
        if isinstance(mod, nn.Linear):
            print(f"    img_mlp.{name}: {list(mod.weight.shape)}")
    print(f"  Text FFN class: {type(block.txt_mlp).__name__}")
    for name, mod in block.txt_mlp.named_modules():
        if isinstance(mod, nn.Linear):
            print(f"    txt_mlp.{name}: {list(mod.weight.shape)}")

    # --- AdaLN Structure ---
    print("\n=== AdaLN Structure (block 0) ===")
    print(f"  img_mod: {block.img_mod}")
    print(f"  txt_mod: {block.txt_mod}")
    for name, mod in block.img_mod.named_modules():
        if isinstance(mod, nn.Linear):
            print(f"    img_mod.{name}: weight={list(mod.weight.shape)}, bias={mod.bias is not None}")
    for name, mod in block.txt_mod.named_modules():
        if isinstance(mod, nn.Linear):
            print(f"    txt_mod.{name}: weight={list(mod.weight.shape)}, bias={mod.bias is not None}")
    print(f"  AdaLN depends ONLY on temb (timestep embedding) — cacheable!")

    # --- RoPE ---
    print("\n=== RoPE Structure ===")
    pos_embed = transformer.pos_embed
    print(f"  RoPE class: {type(pos_embed).__name__}")
    print(f"  theta: {pos_embed.theta}")
    print(f"  axes_dim: {pos_embed.axes_dim}")
    print(f"  scale_rope: {pos_embed.scale_rope}")
    print(f"  pos_freqs shape: {pos_embed.pos_freqs.shape}")
    print(f"  neg_freqs shape: {pos_embed.neg_freqs.shape}")

    # --- Timestep Embedding ---
    print("\n=== Timestep Embedding ===")
    tte = transformer.time_text_embed
    print(f"  Class: {type(tte).__name__}")
    for name, mod in tte.named_modules():
        if isinstance(mod, nn.Linear):
            print(f"    {name}: {list(mod.weight.shape)}")

    # --- Input/Output Projections ---
    print("\n=== Input/Output Projections ===")
    print(f"  img_in: {list(transformer.img_in.weight.shape)}")
    print(f"  txt_in: {list(transformer.txt_in.weight.shape)}")
    print(f"  txt_norm: {type(transformer.txt_norm).__name__}")
    print(f"  norm_out: {type(transformer.norm_out).__name__}")
    print(f"  proj_out: {list(transformer.proj_out.weight.shape)}")

    # --- Sharding-relevant Summary ---
    print("\n=== TP Sharding Summary ===")
    print(f"  Attention heads: {config.num_attention_heads}")
    print(f"  Head dim: {config.attention_head_dim}")
    print(f"  Hidden dim: {config.num_attention_heads * config.attention_head_dim}")
    print(f"  Heads divisible by 2: {config.num_attention_heads % 2 == 0}")
    print(f"  Heads divisible by 4: {config.num_attention_heads % 4 == 0}")
    print(f"  Heads divisible by 8: {config.num_attention_heads % 8 == 0}")

    print("\n  Shardable layers per block (Megatron-style TP):")
    print("    Column-parallel (shard output dim):")
    print("      block.attn.to_q.weight        — image Q projection")
    print("      block.attn.to_k.weight        — image K projection")
    print("      block.attn.to_v.weight        — image V projection")
    print("      block.attn.add_q_proj.weight  — text Q projection")
    print("      block.attn.add_k_proj.weight  — text K projection")
    print("      block.attn.add_v_proj.weight  — text V projection")
    print("      block.img_mlp.net.0.proj.weight — image FFN up/gate")
    print("      block.txt_mlp.net.0.proj.weight — text FFN up/gate")
    print("    Row-parallel (shard input dim):")
    print("      block.attn.to_out[0].weight   — image attention output")
    print("      block.attn.to_add_out.weight  — text attention output")
    print("      block.img_mlp.net.2.weight    — image FFN down")
    print("      block.txt_mlp.net.2.weight    — text FFN down")
    print("    Replicate:")
    print("      block.img_mod, block.txt_mod  — AdaLN (small relative to above)")
    print("      block.*norm*                  — layer norms")

    # --- Forward Signature ---
    print("\n=== Transformer Forward Signature ===")
    import inspect
    sig = inspect.signature(transformer.forward)
    print(f"  {sig}")

    print("\n" + "=" * 80)
    print("INSPECTION COMPLETE")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Inspect Qwen-Image model architecture")
    parser.add_argument("--weights-dir", type=str, default="./weights/qwen-image")
    args = parser.parse_args()
    inspect_transformer(args.weights_dir)


if __name__ == "__main__":
    main()
