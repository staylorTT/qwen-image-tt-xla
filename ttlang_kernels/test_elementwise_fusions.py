"""Test fused elementwise kernels at Qwen-Image dimensions.

Qwen-Image 20B: hidden_dim=3072, 24 heads, 128 head_dim.
For 1024x1024 image: latent = 128x128 -> 64x64 patches -> seq_len=4096 (img)
+ ~128 text tokens -> total joint seq ~4224

We test at representative sizes:
  - (seq_len, hidden_dim) = (4096, 3072) for full image stream
  - (128, 3072) for text stream
  - (1, 3072) for timestep embedding broadcast
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import time
import ttnn

from adaln_modulate import adaln_modulate_kernel
from gated_residual import gated_residual_kernel
from silu import silu_kernel

TILE = 32


def to_tt(t, device):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def from_tt(t):
    return ttnn.to_torch(t).float()


def check(name, result, expected, atol=0.5, rtol=0.1):
    diff = (result - expected).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    ok = max_err < atol
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}: max_err={max_err:.4f} mean_err={mean_err:.6f}")
    if not ok:
        print(f"    result range: [{result.min():.4f}, {result.max():.4f}]")
        print(f"    expected range: [{expected.min():.4f}, {expected.max():.4f}]")
    return ok


def test_adaln_modulate(device, seq_len=256, hidden_dim=3072):
    print(f"\n--- AdaLN Modulate ({seq_len}, {hidden_dim}) ---")
    torch.manual_seed(42)

    x_torch = torch.randn(seq_len, hidden_dim, dtype=torch.bfloat16) * 0.5
    shift_torch = torch.randn(seq_len, hidden_dim, dtype=torch.bfloat16) * 0.1
    scale_torch = torch.randn(seq_len, hidden_dim, dtype=torch.bfloat16) * 0.1

    expected = (x_torch.float() * (scale_torch.float() + 1.0) + shift_torch.float())

    x_tt = to_tt(x_torch, device)
    shift_tt = to_tt(shift_torch, device)
    scale_tt = to_tt(scale_torch, device)
    out_tt = to_tt(torch.zeros_like(x_torch), device)

    # Warmup
    adaln_modulate_kernel(x_tt, shift_tt, scale_tt, out_tt)

    # Timed
    ttnn.synchronize_device(device)
    t0 = time.time()
    for _ in range(10):
        adaln_modulate_kernel(x_tt, shift_tt, scale_tt, out_tt)
    ttnn.synchronize_device(device)
    elapsed = (time.time() - t0) / 10
    print(f"  Time: {elapsed*1000:.2f}ms")

    result = from_tt(out_tt)
    return check("adaln_modulate", result, expected)


def test_gated_residual(device, seq_len=256, hidden_dim=3072):
    print(f"\n--- Gated Residual ({seq_len}, {hidden_dim}) ---")
    torch.manual_seed(42)

    residual_torch = torch.randn(seq_len, hidden_dim, dtype=torch.bfloat16) * 0.5
    x_torch = torch.randn(seq_len, hidden_dim, dtype=torch.bfloat16) * 0.5
    gate_torch = torch.randn(seq_len, hidden_dim, dtype=torch.bfloat16) * 0.3

    expected = residual_torch.float() + x_torch.float() * gate_torch.float()

    res_tt = to_tt(residual_torch, device)
    x_tt = to_tt(x_torch, device)
    gate_tt = to_tt(gate_torch, device)
    out_tt = to_tt(torch.zeros_like(residual_torch), device)

    # Warmup
    gated_residual_kernel(res_tt, x_tt, gate_tt, out_tt)

    # Timed
    ttnn.synchronize_device(device)
    t0 = time.time()
    for _ in range(10):
        gated_residual_kernel(res_tt, x_tt, gate_tt, out_tt)
    ttnn.synchronize_device(device)
    elapsed = (time.time() - t0) / 10
    print(f"  Time: {elapsed*1000:.2f}ms")

    result = from_tt(out_tt)
    return check("gated_residual", result, expected)


def test_silu(device, seq_len=32, hidden_dim=3072):
    print(f"\n--- SiLU ({seq_len}, {hidden_dim}) ---")
    torch.manual_seed(42)

    x_torch = torch.randn(seq_len, hidden_dim, dtype=torch.bfloat16) * 0.5
    expected = (x_torch.float() * torch.sigmoid(x_torch.float()))

    x_tt = to_tt(x_torch, device)
    out_tt = to_tt(torch.zeros_like(x_torch), device)

    # Warmup
    silu_kernel(x_tt, out_tt)

    # Timed
    ttnn.synchronize_device(device)
    t0 = time.time()
    for _ in range(10):
        silu_kernel(x_tt, out_tt)
    ttnn.synchronize_device(device)
    elapsed = (time.time() - t0) / 10
    print(f"  Time: {elapsed*1000:.2f}ms")

    result = from_tt(out_tt)
    return check("silu", result, expected)


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)

    all_pass = True
    # Start with smaller sizes for quick validation, then scale up
    all_pass &= test_silu(device, seq_len=32, hidden_dim=3072)
    all_pass &= test_adaln_modulate(device, seq_len=256, hidden_dim=3072)
    all_pass &= test_gated_residual(device, seq_len=256, hidden_dim=3072)

    if all_pass:
        print("\n=== ALL TESTS PASSED ===")
    else:
        print("\n=== SOME TESTS FAILED ===")

    ttnn.close_device(device)
