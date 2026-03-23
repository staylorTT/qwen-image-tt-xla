"""Test broadcast_row kernel: [32, D] -> [S, D] by replicating row 0."""
import torch
import torch.nn.functional as F
import ttnn

TILE = 32
HIDDEN_DIM = 3072
SEQ_LEN = 1024

from broadcast_row import broadcast_row_kernel


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

    # Simulate tile-padded mod param: [1, 1, D] -> padded to [1, 32, D]
    mod_param = torch.randn(1, 1, HIDDEN_DIM, dtype=torch.bfloat16)
    mod_padded = F.pad(mod_param, (0, 0, 0, 31))  # [1, 32, 3072]

    # Expected: row 0 broadcast to all SEQ_LEN rows
    expected = mod_param.expand(1, SEQ_LEN, HIDDEN_DIM).contiguous()

    # Convert to ttnn (strip batch dim for 2D tile layout)
    inp_tt = to_tt(mod_padded.squeeze(0), device)  # [32, 3072]
    out_torch = torch.zeros(SEQ_LEN, HIDDEN_DIM, dtype=torch.bfloat16)
    out_tt = to_tt(out_torch, device)  # [1024, 3072]

    print(f"inp shape: {inp_tt.shape}")
    print(f"out shape: {out_tt.shape}")

    broadcast_row_kernel(inp_tt, out_tt)
    if hasattr(ttnn, 'synchronize_device'):
        ttnn.synchronize_device(device)

    result = ttnn.to_torch(out_tt).float()
    expected_2d = expected.squeeze(0).float()

    err = (result[:SEQ_LEN, :HIDDEN_DIM] - expected_2d).abs()
    print(f"max error: {err.max():.6f}")
    print(f"mean error: {err.mean():.6f}")

    # Check a few rows
    for r in [0, 1, 32, 100, SEQ_LEN - 1]:
        row_err = (result[r, :HIDDEN_DIM] - expected_2d[r, :HIDDEN_DIM]).abs().max().item()
        print(f"  row {r}: err={row_err:.6f} out={result[r, :4]} exp={expected_2d[r, :4]}")

    ok = err.max().item() < 0.01
    print(f"\n{'PASS' if ok else 'FAIL'}: broadcast_row_kernel")

    ttnn.close_device(device)
