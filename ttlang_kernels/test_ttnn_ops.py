"""Isolate ttnn ops to find shape/broadcast issues."""
import torch
import torch.nn.functional as F
import ttnn

TILE = 32
HIDDEN_DIM = 3072


def to_tt(t, device):
    if t.dim() == 1:
        t = t.unsqueeze(0)
    h, w = t.shape[-2], t.shape[-1]
    ph = ((h + TILE - 1) // TILE) * TILE - h
    pw = ((w + TILE - 1) // TILE) * TILE - w
    if ph > 0 or pw > 0:
        t = F.pad(t, (0, pw, 0, ph))
    return ttnn.from_torch(t.contiguous().to(torch.bfloat16),
                           dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    torch.manual_seed(42)

    # Test broadcasting: [1, 1, D] * [1, S, D]
    S = 128
    scale = torch.randn(1, 1, HIDDEN_DIM, dtype=torch.bfloat16) * 0.1
    x = torch.randn(1, S, HIDDEN_DIM, dtype=torch.bfloat16) * 0.5

    scale_tt = to_tt(scale, device)
    x_tt = to_tt(x, device)
    print(f"scale shape: {scale_tt.shape}")
    print(f"x shape: {x_tt.shape}")

    # Test 1: scalar + tensor
    print("\nTest 1: 1.0 + scale_tt")
    try:
        r = 1.0 + scale_tt
        print(f"  OK: {r.shape}")
    except Exception as e:
        print(f"  FAIL: {e}")

    # Test 2: multiply with broadcast
    print("\nTest 2: x_tt * scale_tt (broadcast)")
    try:
        r = x_tt * scale_tt
        print(f"  OK: {r.shape}")
    except Exception as e:
        print(f"  FAIL: {e}")

    # Test 3: layer_norm
    print("\nTest 3: layer_norm(x_tt)")
    try:
        r = ttnn.layer_norm(x_tt)
        print(f"  OK: {r.shape}")
    except Exception as e:
        print(f"  FAIL: {e}")

    # Test 4: full adaLN pattern
    print("\nTest 4: layer_norm(x) * (1 + scale) + shift")
    shift = torch.randn(1, 1, HIDDEN_DIM, dtype=torch.bfloat16) * 0.1
    shift_tt = to_tt(shift, device)
    try:
        n = ttnn.layer_norm(x_tt)
        scaled = 1.0 + scale_tt
        modulated = n * scaled + shift_tt
        print(f"  OK: {modulated.shape}")
    except Exception as e:
        print(f"  FAIL: {e}")

    # Test 5: try with expanded tensors instead of broadcast
    print("\nTest 5: expand scale to match x, then multiply")
    scale_expanded = scale.expand(1, S, HIDDEN_DIM).contiguous()
    scale_exp_tt = to_tt(scale_expanded, device)
    try:
        r = x_tt * scale_exp_tt
        print(f"  OK: {r.shape}")
    except Exception as e:
        print(f"  FAIL: {e}")

    # Test 6: matmul
    print("\nTest 6: matmul [1,S,D] @ [D,D]")
    w = torch.randn(HIDDEN_DIM, HIDDEN_DIM, dtype=torch.bfloat16) * 0.01
    w_tt = to_tt(w, device)
    try:
        r = ttnn.matmul(x_tt, w_tt)
        print(f"  OK: {r.shape}")
    except Exception as e:
        print(f"  FAIL: {e}")

    # Test 7: silu
    print("\nTest 7: silu")
    temb = torch.randn(1, 1, HIDDEN_DIM, dtype=torch.bfloat16) * 0.1
    temb_tt = to_tt(temb, device)
    try:
        r = ttnn.silu(temb_tt)
        print(f"  OK: {r.shape}")
    except Exception as e:
        print(f"  FAIL: {e}")

    ttnn.close_device(device)
    print("\nDone!")
