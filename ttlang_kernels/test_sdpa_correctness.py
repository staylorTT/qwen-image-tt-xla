"""Test if ttnn SDPA is correct with Qwen-Image-like large attention scores.

Qwen-Image has QK-norm producing scores with mean ~5000, max ~35000.
The softmax patch clamps exp inputs to [-88, 0] to avoid garbage.
If this test fails, the softmax patch needs to be applied.
"""
import torch
import torch.nn.functional as F
import ttnn

TILE = 32


def to_tt(t, device):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def from_tt(t):
    return ttnn.to_torch(t).float()


def test_sdpa_large_scores(device):
    """Test SDPA with large scores similar to Qwen-Image."""
    print("\n--- SDPA with large attention scores ---")
    torch.manual_seed(42)

    B, H, S, D = 1, 6, 256, 128
    scale = 1.0 / (D ** 0.5)

    # Create Q,K that produce large scores (like QK-normed Qwen-Image)
    q = torch.randn(B, H, S, D, dtype=torch.bfloat16) * 10.0
    k = torch.randn(B, H, S, D, dtype=torch.bfloat16) * 10.0
    v = torch.randn(B, H, S, D, dtype=torch.bfloat16) * 0.5

    # Check score magnitudes
    scores = (q.float() @ k.float().transpose(-2, -1)) * scale
    print(f"  Score range: [{scores.min():.0f}, {scores.max():.0f}], mean={scores.mean():.0f}")

    ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), scale=scale)

    q_tt = to_tt(q, device)
    k_tt = to_tt(k, device)
    v_tt = to_tt(v, device)

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
        fp32_dest_acc_en=True,
    )

    out_tt = ttnn.transformer.scaled_dot_product_attention(
        q_tt, k_tt, v_tt, is_causal=False, scale=scale,
        program_config=program_config, compute_kernel_config=compute_config,
    )
    result = from_tt(out_tt)

    diff = (result - ref).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()

    # Check for NaN/Inf (sign of softmax explosion)
    has_nan = torch.isnan(result).any().item()
    has_inf = torch.isinf(result).any().item()
    print(f"  max_err={max_err:.4f} mean_err={mean_err:.6f}")
    print(f"  NaN={has_nan}, Inf={has_inf}")
    print(f"  Result range: [{result.min():.4f}, {result.max():.4f}]")
    print(f"  Ref range: [{ref.min():.4f}, {ref.max():.4f}]")

    ok = max_err < 5.0 and not has_nan and not has_inf
    print(f"  [{'PASS' if ok else 'FAIL'}] SDPA with large scores")
    if not ok:
        print("  WARNING: Softmax patch may not be applied!")
    return ok


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    test_sdpa_large_scores(device)
    ttnn.close_device(device)
