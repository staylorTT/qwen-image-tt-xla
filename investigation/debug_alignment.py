"""Test if tile alignment is the root cause of the SDPA failure.

The broken SDPA fails with seq_len=263 (not tile-aligned) but works with 384 (aligned).
Test the full safe-softmax chain with different sequence lengths.
"""

import math
import os

import torch
import torch.nn.functional as F
import torch_xla
import torch_xla.runtime as xr

os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
xr.use_spmd()
xr.set_device_type("TT")

from utils.profiling_utils import compute_pcc


def main():
    device = torch_xla.device()
    torch.manual_seed(42)

    # Test SDPA with various sequence lengths
    for seq_len in [32, 64, 96, 128, 256, 263, 288, 384]:
        is_aligned = seq_len % 32 == 0

        q = torch.randn(1, 24, seq_len, 128, dtype=torch.bfloat16)
        k = torch.randn(1, 24, seq_len, 128, dtype=torch.bfloat16)
        v = torch.randn(1, 24, seq_len, 128, dtype=torch.bfloat16)

        # CPU reference
        ref = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)

        # Device
        def sdpa_fn(q, k, v):
            return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)

        compiled = torch.compile(sdpa_fn, backend="tt")
        with torch.no_grad():
            dev_out = compiled(q.to(device), k.to(device), v.to(device))
        torch_xla.sync()

        pcc = compute_pcc(dev_out.cpu(), ref)
        tag = "ALIGNED" if is_aligned else "UNALIGNED"
        status = "OK" if pcc > 0.99 else "FAIL"
        print(f"  seq={seq_len:3d} [{tag:>9s}]: PCC={pcc:.6f} [{status}]")

    print("\nDone!")


if __name__ == "__main__":
    main()
