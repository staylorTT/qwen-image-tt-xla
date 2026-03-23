"""Test if ttnn native SDPA works on this Blackhole device.

The current pipeline uses manual_attention (matmul+softmax+matmul) as a workaround
for a TT compiler SDPA bug. If native SDPA works, it's a free speedup since it
uses chunked/tiled attention internally.
"""
import torch
import torch.nn.functional as F
import time
import ttnn

TILE = 32


def to_tt(t, device, mem=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=mem)


def from_tt(t):
    return ttnn.to_torch(t).float()


def test_sdpa_native(device, batch=1, n_heads=6, seq_len=4224, head_dim=128):
    """Test native ttnn SDPA at Qwen-Image dimensions (per-device with TP=4)."""
    print(f"\n--- TTNN Native SDPA (B={batch}, H={n_heads}, S={seq_len}, D={head_dim}) ---")
    torch.manual_seed(42)

    # Pad seq_len to tile boundary
    seq_pad = ((seq_len + TILE - 1) // TILE) * TILE
    print(f"  seq_len={seq_len}, padded={seq_pad}")

    q_torch = torch.randn(batch, n_heads, seq_pad, head_dim, dtype=torch.bfloat16) * 0.3
    k_torch = torch.randn(batch, n_heads, seq_pad, head_dim, dtype=torch.bfloat16) * 0.3
    v_torch = torch.randn(batch, n_heads, seq_pad, head_dim, dtype=torch.bfloat16) * 0.3

    # PyTorch reference
    scale = 1.0 / (head_dim ** 0.5)
    ref = F.scaled_dot_product_attention(q_torch.float(), k_torch.float(), v_torch.float(),
                                          scale=scale)

    q_tt = to_tt(q_torch, device)
    k_tt = to_tt(k_torch, device)
    v_tt = to_tt(v_torch, device)

    full_grid = device.compute_with_storage_grid_size()
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=full_grid,
        q_chunk_size=256,
        k_chunk_size=256,
        exp_approx_mode=False,
    )

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
    )

    try:
        out_tt = ttnn.transformer.scaled_dot_product_attention(
            q_tt, k_tt, v_tt,
            is_causal=False,
            scale=scale,
            program_config=program_config,
            compute_kernel_config=compute_config,
        )
        result = from_tt(out_tt)

        # Check accuracy on valid (non-padded) region
        diff = (result[:, :, :seq_len, :] - ref[:, :, :seq_len, :]).abs()
        max_err = diff.max().item()
        mean_err = diff.mean().item()
        print(f"  max_err={max_err:.4f} mean_err={mean_err:.6f}")
        if max_err < 5.0:
            print("  [PASS] Native SDPA works!")
        else:
            print("  [FAIL] Accuracy too low")

        # Benchmark
        ttnn.synchronize_device(device)
        t0 = time.time()
        for _ in range(5):
            out_tt = ttnn.transformer.scaled_dot_product_attention(
                q_tt, k_tt, v_tt,
                is_causal=False,
                scale=scale,
                program_config=program_config,
                compute_kernel_config=compute_config,
            )
        ttnn.synchronize_device(device)
        elapsed = (time.time() - t0) / 5
        print(f"  Time: {elapsed*1000:.2f}ms")
        return True

    except Exception as e:
        print(f"  [ERROR] Native SDPA failed: {e}")
        return False


def test_manual_sdpa(device, batch=1, n_heads=6, seq_len=4224, head_dim=128):
    """Benchmark the manual matmul+softmax+matmul approach for comparison."""
    print(f"\n--- Manual SDPA (B={batch}, H={n_heads}, S={seq_len}, D={head_dim}) ---")
    torch.manual_seed(42)

    seq_pad = ((seq_len + TILE - 1) // TILE) * TILE

    q_torch = torch.randn(batch, n_heads, seq_pad, head_dim, dtype=torch.bfloat16) * 0.3
    k_torch = torch.randn(batch, n_heads, seq_pad, head_dim, dtype=torch.bfloat16) * 0.3
    v_torch = torch.randn(batch, n_heads, seq_pad, head_dim, dtype=torch.bfloat16) * 0.3

    q_tt = to_tt(q_torch, device)
    k_tt = to_tt(k_torch, device)
    v_tt = to_tt(v_torch, device)

    scale = 1.0 / (head_dim ** 0.5)

    try:
        # Manual decomposition matching generate_image_v2.py
        k_t = ttnn.transpose(k_tt, -2, -1)
        attn = ttnn.matmul(q_tt, k_t)
        attn = ttnn.mul(attn, scale)
        attn = ttnn.softmax(attn, dim=-1)
        out_tt = ttnn.matmul(attn, v_tt)

        # Benchmark
        ttnn.synchronize_device(device)
        t0 = time.time()
        for _ in range(5):
            k_t = ttnn.transpose(k_tt, -2, -1)
            attn = ttnn.matmul(q_tt, k_t)
            attn = ttnn.mul(attn, scale)
            attn = ttnn.softmax(attn, dim=-1)
            out_tt = ttnn.matmul(attn, v_tt)
        ttnn.synchronize_device(device)
        elapsed = (time.time() - t0) / 5
        print(f"  Time: {elapsed*1000:.2f}ms")
        print("  [OK] Manual SDPA works")
        return True

    except Exception as e:
        print(f"  [ERROR] Manual SDPA failed: {e}")
        return False


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)

    # Test at per-device dimensions (TP=4, so 24/4=6 heads per device)
    # Start small to verify, then scale up
    test_sdpa_native(device, batch=1, n_heads=6, seq_len=256, head_dim=128)
    test_manual_sdpa(device, batch=1, n_heads=6, seq_len=256, head_dim=128)

    # Full 1024x1024 scale
    test_sdpa_native(device, batch=1, n_heads=6, seq_len=4224, head_dim=128)
    test_manual_sdpa(device, batch=1, n_heads=6, seq_len=4224, head_dim=128)

    ttnn.close_device(device)
