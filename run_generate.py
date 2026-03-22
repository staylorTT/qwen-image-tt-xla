# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Phase 3.4: End-to-end image generation CLI.

Runs the full Qwen-Image pipeline on Tenstorrent silicon via tt-xla.

Usage:
    python run_generate.py \
        --prompt 'A sign that reads "Hello Tenstorrent" in front of a modern office building, Ultra HD, 4K' \
        --width 1664 --height 928 \
        --steps 50 \
        --cfg_scale 4.0 \
        --seed 42 \
        --output output.png
"""

import argparse
import time

import torch_xla.runtime as xr

from pipeline import QwenImageXLAPipeline
from utils.image_utils import save_image


def main():
    parser = argparse.ArgumentParser(description="Qwen-Image text-to-image generation on Tenstorrent")
    parser.add_argument(
        "--prompt",
        type=str,
        default='A cat holding a sign that says "Hello World"',
        help="Text prompt for image generation",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="",
        help="Negative prompt for CFG",
    )
    parser.add_argument("--width", type=int, default=1024, help="Image width")
    parser.add_argument("--height", type=int, default=1024, help="Image height")
    parser.add_argument("--steps", type=int, default=50, help="Number of denoising steps")
    parser.add_argument("--cfg_scale", type=float, default=4.0, help="True CFG scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="output.png", help="Output image path")
    parser.add_argument("--weights-dir", type=str, default="./weights/qwen-image", help="Model weights directory")
    parser.add_argument("--max-seq-len", type=int, default=512, help="Max text token length")
    args = parser.parse_args()

    # Initialize tt-xla
    xr.set_device_type("TT")

    # Create pipeline
    print(f"Initializing pipeline from {args.weights_dir}...")
    pipe = QwenImageXLAPipeline(args.weights_dir)

    # Generate
    print(f"\nPrompt: '{args.prompt}'")
    if args.negative_prompt:
        print(f"Negative: '{args.negative_prompt}'")

    start = time.perf_counter()
    image = pipe.generate(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt or "",
        width=args.width,
        height=args.height,
        num_inference_steps=args.steps,
        true_cfg_scale=args.cfg_scale,
        seed=args.seed,
        max_sequence_length=args.max_seq_len,
    )
    elapsed = time.perf_counter() - start

    # Save image
    save_image(image, args.output)
    print(f"\nTotal wall time: {elapsed:.1f}s")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
