"""Dump TTNN IR for the BROKEN F.scaled_dot_product_attention path.

Compare this against the working manual attention IR to find the fused SDPA op.
"""

import math
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch_xla
import torch_xla.runtime as xr

os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
os.environ["TTXLA_LOGGER_LEVEL"] = "INFO"

EXPORT_DIR = "./ir_dump_broken_sdpa"
os.makedirs(EXPORT_DIR, exist_ok=True)

try:
    torch_xla.set_custom_compile_options({"export_path": EXPORT_DIR})
except Exception:
    pass

xr.use_spmd()
xr.set_device_type("TT")


def main():
    device = torch_xla.device()

    # Build Q/K/V matching our real use case: [1, 24, 384, 128]
    # 384 = 256 img + 128 padded txt (tile-aligned)
    torch.manual_seed(42)
    q = torch.randn(1, 24, 384, 128, dtype=torch.bfloat16)
    k = torch.randn(1, 24, 384, 128, dtype=torch.bfloat16)
    v = torch.randn(1, 24, 384, 128, dtype=torch.bfloat16)

    # Test 1: F.scaled_dot_product_attention (the broken path)
    def sdpa_fn(q, k, v):
        return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)

    print("Compiling F.scaled_dot_product_attention...")
    compiled_sdpa = torch.compile(sdpa_fn, backend="tt")
    with torch.no_grad():
        out = compiled_sdpa(q.to(device), k.to(device), v.to(device))
    torch_xla.sync()
    print(f"  Output shape: {list(out.cpu().shape)}")

    # List dumped files
    irs_dir = os.path.join(EXPORT_DIR, "irs")
    if os.path.exists(irs_dir):
        files = sorted(os.listdir(irs_dir))
        print(f"\nDumped {len(files)} IR files:")
        for f in files:
            size = os.path.getsize(os.path.join(irs_dir, f))
            print(f"  {f} ({size // 1024}KB)")

        # Show the TTNN IR (look for sdpa ops)
        for f in files:
            if f.startswith("ttnn_"):
                path = os.path.join(irs_dir, f)
                print(f"\n{'='*60}")
                print(f"TTNN IR: {f}")
                print(f"{'='*60}")
                with open(path) as fh:
                    content = fh.read()
                    # Print lines with key ops
                    for i, line in enumerate(content.split('\n'), 1):
                        stripped = line.strip()
                        if any(kw in stripped for kw in [
                            'scaled_dot_product', 'sdpa', 'softmax',
                            'matmul', 'func.func @main'
                        ]):
                            print(f"  L{i}: {stripped[:120]}")


if __name__ == "__main__":
    main()
