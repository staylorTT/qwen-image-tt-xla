"""Benchmark baseline: run the existing demo at 1024x1024 and measure timing.

This runs on the remote with the actual model weights to get baseline numbers.
Uses the existing XLA compilation path with manual attention.
"""
import sys
import os
sys.path.insert(0, "/workspace/qwen-image-tt-xla")
os.chdir("/workspace/qwen-image-tt-xla")

# Set environment before imports
os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"

import time
import torch
import torch_xla
import torch_xla.runtime as xr
xr.use_spmd()
xr.set_device_type("TT")

try:
    torch_xla.set_custom_compile_options({"optimization_level": "1"})
except Exception:
    pass

from demo import QwenImageDemo

if __name__ == "__main__":
    weights_dir = "/workspace/qwen-image-tt-xla/weights/qwen-image"

    print("=" * 60)
    print("Baseline Benchmark: 1024x1024, 20 steps")
    print("=" * 60)

    demo = QwenImageDemo(
        weights_dir=weights_dir,
        width=1024,
        height=1024,
        num_steps=20,
        cfg_scale=4.0,
    )

    # Generate a few images to get stable timing
    for i in range(3):
        t0 = time.perf_counter()
        result = demo._generate_image(
            f"A cat holding a sign that says hello world",
            negative_prompt="",
            seed=42 + i,
        )
        total = time.perf_counter() - t0
        fname, seed, t_enc, t_denoise, t_vae = result
        print(f"\nRun {i+1}: total={total:.1f}s")
        print(f"  text_encode={t_enc:.2f}s denoising={t_denoise:.2f}s vae={t_vae:.2f}s")
        print(f"  per_step={t_denoise/20:.3f}s")
        print(f"  saved: {fname}")
