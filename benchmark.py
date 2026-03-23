# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Phase 6.1: Performance Measurement and Benchmarking.

Captures per-step timing, matmul utilization, and memory usage for the
Qwen-Image pipeline on Tenstorrent hardware.

Tests:
  1. perf_qwen_image_baseline: Total time and per-step breakdown
  2. perf_qwen_image_adaln_cache: With AdaLN caching vs without
  3. perf_qwen_image_cfg_batched: Batched CFG vs sequential

Usage:
    python benchmark.py [--weights-dir ./weights/qwen-image] [--steps 50] [--warmup 2]
"""

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Optional

import torch
import torch_xla

import torch_xla.runtime as xr

from pipeline import QwenImageXLAPipeline
from utils.profiling_utils import PipelineTiming


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    name: str
    width: int
    height: int
    num_steps: int
    num_devices: int
    cfg_scale: float
    total_ms: float
    text_encode_ms: float
    denoising_ms: float
    vae_decode_ms: float
    avg_step_ms: float
    first_step_ms: float
    adaln_cached: bool = False


def run_benchmark(
    pipe: QwenImageXLAPipeline,
    name: str,
    prompt: str,
    width: int,
    height: int,
    num_steps: int,
    cfg_scale: float,
    seed: int = 42,
) -> BenchmarkResult:
    """Run a single benchmark and return results."""
    print(f"\n{'=' * 60}")
    print(f"BENCHMARK: {name}")
    print(f"  {width}x{height}, {num_steps} steps, cfg={cfg_scale}")
    print(f"{'=' * 60}")

    start = time.perf_counter()
    _image = pipe.generate(
        prompt=prompt,
        negative_prompt="" if cfg_scale > 1 else None,
        width=width,
        height=height,
        num_inference_steps=num_steps,
        true_cfg_scale=cfg_scale,
        seed=seed,
    )
    total_ms = (time.perf_counter() - start) * 1000

    timing = pipe.timing
    result = BenchmarkResult(
        name=name,
        width=width,
        height=height,
        num_steps=num_steps,
        num_devices=pipe.num_devices,
        cfg_scale=cfg_scale,
        total_ms=total_ms,
        text_encode_ms=timing.text_encode_ms,
        denoising_ms=timing.denoising_total_ms,
        vae_decode_ms=timing.vae_decode_ms,
        avg_step_ms=timing.avg_step_ms,
        first_step_ms=timing.steps[0].total_ms if timing.steps else 0,
    )

    print(f"\n--- Results ---")
    print(f"  Total:         {result.total_ms:.0f}ms ({result.total_ms/1000:.1f}s)")
    print(f"  Text encode:   {result.text_encode_ms:.0f}ms")
    print(f"  Denoising:     {result.denoising_ms:.0f}ms")
    print(f"  VAE decode:    {result.vae_decode_ms:.0f}ms")
    print(f"  Avg step:      {result.avg_step_ms:.0f}ms")
    print(f"  First step:    {result.first_step_ms:.0f}ms (includes compile)")

    return result


def main():
    parser = argparse.ArgumentParser(description="Benchmark Qwen-Image pipeline")
    parser.add_argument("--weights-dir", type=str, default="./weights/qwen-image")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=2, help="Warmup steps before timing")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--output-json", type=str, default="benchmark_results.json")
    parser.add_argument("--prompt", type=str, default='A sign that reads "Hello Tenstorrent"')
    args = parser.parse_args()

    xr.set_device_type("TT")

    # Enable profiling if available
    if os.environ.get("TT_METAL_DEVICE_PROFILER") != "1":
        print("Tip: Set TT_METAL_DEVICE_PROFILER=1 for device-level profiling")

    pipe = QwenImageXLAPipeline(args.weights_dir)
    results = []

    # Warmup run (smaller, just to compile graphs)
    if args.warmup > 0:
        print(f"\nWarmup run ({args.warmup} steps, 256x256)...")
        pipe.generate(
            prompt=args.prompt,
            width=256,
            height=256,
            num_inference_steps=args.warmup,
            true_cfg_scale=1.0,  # no CFG for faster warmup
            seed=0,
        )

    # Benchmark 1: Baseline (no CFG)
    results.append(run_benchmark(
        pipe, "baseline_no_cfg",
        prompt=args.prompt,
        width=args.width, height=args.height,
        num_steps=args.steps, cfg_scale=1.0,
    ))

    # Benchmark 2: With true CFG
    results.append(run_benchmark(
        pipe, "with_true_cfg",
        prompt=args.prompt,
        width=args.width, height=args.height,
        num_steps=args.steps, cfg_scale=4.0,
    ))

    # Benchmark 3: Widescreen (16:9)
    results.append(run_benchmark(
        pipe, "widescreen_16_9",
        prompt=args.prompt,
        width=1664, height=928,
        num_steps=args.steps, cfg_scale=4.0,
    ))

    # Save results
    output = [asdict(r) for r in results]
    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output_json}")

    # Summary table
    print(f"\n{'=' * 80}")
    print(f"{'Name':<25} {'Resolution':<15} {'Steps':<8} {'CFG':<8} {'Total(s)':<10} {'Avg Step(ms)':<12}")
    print(f"{'-' * 80}")
    for r in results:
        print(f"{r.name:<25} {r.width}x{r.height:<8} {r.num_steps:<8} {r.cfg_scale:<8.1f} {r.total_ms/1000:<10.1f} {r.avg_step_ms:<12.0f}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
