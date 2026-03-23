# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Phase 0.4: Inspect Qwen2.5-VL-7B text encoder.

Qwen-Image uses Qwen2.5-VL-7B-Instruct as its text encoder — a full 7B multimodal VLM,
not a lightweight CLIP/T5 encoder. This script inspects its structure, parameter count,
output shapes, and memory requirements.

Usage:
    python inspect_text_encoder.py [--weights-dir ./weights/qwen-image]
"""

import argparse

import torch
import torch.nn as nn


def inspect_text_encoder(weights_dir: str):
    """Inspect the Qwen2.5-VL-7B text encoder."""
    from diffusers import DiffusionPipeline

    print("=" * 80)
    print("Loading pipeline (CPU, bf16)...")
    print("=" * 80)

    pipe = DiffusionPipeline.from_pretrained(
        weights_dir,
        torch_dtype=torch.bfloat16,
    )

    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer

    # --- Encoder Info ---
    print("\n=== Text Encoder ===")
    print(f"  Class: {type(text_encoder).__name__}")
    print(f"  Config: {text_encoder.config}")

    total_params = sum(p.numel() for p in text_encoder.parameters())
    print(f"\n  Total params: {total_params / 1e9:.2f}B ({total_params * 2 / 1e9:.1f}GB bf16)")

    # --- Tokenizer Info ---
    print("\n=== Tokenizer ===")
    print(f"  Class: {type(tokenizer).__name__}")
    print(f"  Vocab size: {tokenizer.vocab_size}")

    # --- Test Encoding ---
    print("\n=== Test Encoding ===")
    test_prompts = [
        'A cat sitting on a windowsill',
        'A sign that reads "Hello Tenstorrent" in front of a modern office building',
    ]

    # Use the pipeline's prompt template (matches pipeline_qwenimage.py)
    template = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    drop_idx = 34

    for prompt in test_prompts:
        txt = template.format(prompt)
        tokens = tokenizer(
            txt,
            max_length=1024 + drop_idx,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        print(f"\n  Prompt: '{prompt}'")
        print(f"    Input IDs shape: {list(tokens.input_ids.shape)}")
        print(f"    Attention mask shape: {list(tokens.attention_mask.shape)}")
        print(f"    Token count: {tokens.attention_mask.sum().item()}")

        with torch.no_grad():
            output = text_encoder(
                input_ids=tokens.input_ids,
                attention_mask=tokens.attention_mask,
                output_hidden_states=True,
            )
            hidden = output.hidden_states[-1]
            print(f"    Hidden states shape: {list(hidden.shape)}")
            print(f"    Hidden states dtype: {hidden.dtype}")
            print(f"    Hidden states range: [{hidden.min().item():.4f}, {hidden.max().item():.4f}]")

            # Apply the same masking as the pipeline
            bool_mask = tokens.attention_mask.bool()
            valid_lengths = bool_mask.sum(dim=1)
            selected = hidden[bool_mask]
            split_result = torch.split(selected, valid_lengths.tolist(), dim=0)
            trimmed = split_result[0][drop_idx:]
            print(f"    After mask+trim: {list(trimmed.shape)} (joint_attention_dim={trimmed.shape[-1]})")

    # --- Module Tree (top-level only) ---
    print("\n=== Top-level Module Tree ===")
    for name, child in text_encoder.named_children():
        param_count = sum(p.numel() for p in child.parameters())
        print(f"  {name}: {type(child).__name__} ({param_count / 1e6:.1f}M params)")

    # --- Memory Analysis ---
    print("\n=== Memory Analysis ===")
    print(f"  Text encoder size (bf16): {total_params * 2 / 1e9:.1f}GB")
    print(f"  Can fit alongside 20B MMDiT on device? Probably NOT (~14GB + ~26GB = ~40GB)")
    print(f"  Recommended: Run on CPU, encode once per prompt before denoising loop")
    print(f"  Alternative: Run on device BEFORE loading MMDiT, then free memory")

    print("\n" + "=" * 80)
    print("TEXT ENCODER INSPECTION COMPLETE")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Inspect Qwen-Image text encoder")
    parser.add_argument("--weights-dir", type=str, default="./weights/qwen-image")
    args = parser.parse_args()
    inspect_text_encoder(args.weights_dir)


if __name__ == "__main__":
    main()
