"""Test fused LayerNorm + adaLN modulate kernel."""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import time
import ttnn

from layernorm_adaln import make_layernorm_adaln_kernel

TILE = 32


def to_tt(t, device, mem=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=mem)


def to_tt_l1(t, device):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.L1_MEMORY_CONFIG)


def from_tt(t):
    return ttnn.to_torch(t).float()


def test_layernorm_adaln(device, seq_len=64, hidden_dim=3072):
    dim_tiles = hidden_dim // TILE
    print(f"\n--- LayerNorm+adaLN ({seq_len}, {hidden_dim}), dim_tiles={dim_tiles} ---")
    torch.manual_seed(42)

    x_torch = torch.randn(seq_len, hidden_dim, dtype=torch.bfloat16) * 0.5
    shift_torch = torch.randn(seq_len, hidden_dim, dtype=torch.bfloat16) * 0.1
    scale_torch = torch.randn(seq_len, hidden_dim, dtype=torch.bfloat16) * 0.1

    # PyTorch reference: LayerNorm then adaLN modulate
    x_f = x_torch.float()
    mean = x_f.mean(dim=-1, keepdim=True)
    var = x_f.var(dim=-1, keepdim=True, unbiased=False)
    normed = (x_f - mean) / torch.sqrt(var + 1e-6)
    expected = normed * (scale_torch.float() + 1.0) + shift_torch.float()

    # Scaler and mean_scale constants
    scaler_torch = torch.ones(TILE, TILE, dtype=torch.bfloat16)
    mean_scale_torch = torch.full((TILE, TILE), 1.0 / hidden_dim, dtype=torch.bfloat16)

    x_tt = to_tt(x_torch, device)
    shift_tt = to_tt(shift_torch, device)
    scale_tt = to_tt(scale_torch, device)
    scaler_tt = to_tt_l1(scaler_torch, device)
    ms_tt = to_tt_l1(mean_scale_torch, device)
    out_tt = to_tt(torch.zeros_like(x_torch), device)

    kernel = make_layernorm_adaln_kernel(dim_tiles)

    # Run
    kernel(x_tt, shift_tt, scale_tt, scaler_tt, ms_tt, out_tt)
    result = from_tt(out_tt)

    diff = (result - expected).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    ok = max_err < 2.0  # LN has more numerical error in bf16
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] max_err={max_err:.4f} mean_err={mean_err:.6f}")
    if not ok:
        print(f"    result[0,:5]: {result[0,:5].tolist()}")
        print(f"    expected[0,:5]: {expected[0,:5].tolist()}")

    # Benchmark
    ttnn.synchronize_device(device)
    t0 = time.time()
    for _ in range(10):
        kernel(x_tt, shift_tt, scale_tt, scaler_tt, ms_tt, out_tt)
    ttnn.synchronize_device(device)
    elapsed = (time.time() - t0) / 10
    print(f"  Time: {elapsed*1000:.2f}ms")

    return ok


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)

    all_pass = True
    # Small test first
    all_pass &= test_layernorm_adaln(device, seq_len=32, hidden_dim=1024)
    # Qwen-Image scale
    all_pass &= test_layernorm_adaln(device, seq_len=64, hidden_dim=3072)

    if all_pass:
        print("\n=== ALL TESTS PASSED ===")
    else:
        print("\n=== SOME TESTS FAILED ===")

    ttnn.close_device(device)
