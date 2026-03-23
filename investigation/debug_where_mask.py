"""Isolate the safe-softmax mask bug in F.scaled_dot_product_attention.

The broken SDPA computes: where(all_neginf_mask, zeros, softmax(scores))
This should be a no-op when no scores are -inf. Test each op in the chain.
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

    # Build realistic attention scores (no -inf values)
    # Shape: [1, 24, 384, 384] matching our use case
    scores = torch.randn(1, 24, 384, 384, dtype=torch.float32) * 0.5
    neginf = float('-inf')

    print("=" * 60)
    print("TEST 1: eq(scores, -inf) on device")
    print("=" * 60)
    # CPU ref
    eq_cpu = (scores == neginf)
    print(f"  CPU: any True? {eq_cpu.any().item()}, sum={eq_cpu.sum().item()}")

    def eq_fn(x):
        return (x == float('-inf')).to(torch.bfloat16)

    c_eq = torch.compile(eq_fn, backend="tt")
    with torch.no_grad():
        eq_dev = c_eq(scores.to(device))
    torch_xla.sync()
    eq_dev_cpu = eq_dev.cpu()
    print(f"  Dev: any True? {(eq_dev_cpu > 0).any().item()}, sum={eq_dev_cpu.sum().item()}")
    pcc = compute_pcc(eq_dev_cpu.float(), eq_cpu.float())
    print(f"  PCC: {pcc:.6f}")

    print("\n" + "=" * 60)
    print("TEST 2: Full safe-softmax mask chain on device")
    print("=" * 60)

    # Replicate the exact chain from the broken SDPA IR
    def safe_softmax_mask(scores):
        is_neginf = (scores == float('-inf'))           # eq with -inf
        not_neginf = ~is_neginf                         # logical_not
        count = not_neginf.sum(dim=-1)                  # sum over key dim
        has_valid = (count != 0)                         # ne with 0
        all_neginf = ~has_valid                          # logical_not
        mask = all_neginf.unsqueeze(-1).expand_as(scores)  # broadcast
        return mask.float()

    mask_cpu = safe_softmax_mask(scores)
    print(f"  CPU mask: any True? {(mask_cpu > 0).any().item()}, sum={mask_cpu.sum().item()}")

    c_mask = torch.compile(safe_softmax_mask, backend="tt")
    with torch.no_grad():
        mask_dev = c_mask(scores.to(device))
    torch_xla.sync()
    mask_dev_cpu = mask_dev.cpu()
    print(f"  Dev mask: any True? {(mask_dev_cpu > 0).any().item()}, sum={mask_dev_cpu.sum().item()}")

    print("\n" + "=" * 60)
    print("TEST 3: where(mask, zeros, softmax) on device")
    print("=" * 60)

    softmax_out = torch.softmax(scores, dim=-1)
    zeros = torch.zeros_like(softmax_out)

    # CPU ref: where should be no-op since mask is all-false
    result_cpu = torch.where(mask_cpu.bool(), zeros, softmax_out)
    pcc_sanity = compute_pcc(result_cpu, softmax_out)
    print(f"  CPU: where is no-op? PCC vs softmax = {pcc_sanity:.6f}")

    # Device: where with device-computed mask
    def full_safe_softmax(scores):
        is_neginf = (scores == float('-inf'))
        not_neginf = ~is_neginf
        count = not_neginf.float().sum(dim=-1)
        has_valid = (count != 0)
        all_neginf = ~has_valid
        mask = all_neginf.unsqueeze(-1).expand_as(scores)

        s_out = torch.softmax(scores, dim=-1)
        z = torch.zeros_like(s_out)
        return torch.where(mask, z, s_out)

    c_full = torch.compile(full_safe_softmax, backend="tt")
    with torch.no_grad():
        result_dev = c_full(scores.to(device))
    torch_xla.sync()
    result_dev_cpu = result_dev.cpu()

    pcc_full = compute_pcc(result_dev_cpu, softmax_out)
    print(f"  Dev: PCC vs CPU softmax = {pcc_full:.6f}")

    # Check first 7 rows vs rest (mimicking txt vs img)
    pcc_first7 = compute_pcc(result_dev_cpu[:, :, :7, :], softmax_out[:, :, :7, :])
    pcc_rest = compute_pcc(result_dev_cpu[:, :, 7:, :], softmax_out[:, :, 7:, :])
    print(f"  Dev rows 0-6 PCC:  {pcc_first7:.6f}")
    print(f"  Dev rows 7+ PCC:   {pcc_rest:.6f}")

    # Check if mask is actually non-zero on device
    print(f"  Dev result stats: mean={result_dev_cpu.mean():.6f}, "
          f"zeros={( result_dev_cpu == 0).sum().item()}/{result_dev_cpu.numel()}")
    print(f"  CPU softmax stats: mean={softmax_out.mean():.6f}, "
          f"zeros={(softmax_out == 0).sum().item()}/{softmax_out.numel()}")

    print("\n" + "=" * 60)
    print("TEST 4: Plain softmax (no where) on device — sanity check")
    print("=" * 60)

    def plain_softmax(scores):
        return torch.softmax(scores, dim=-1)

    c_plain = torch.compile(plain_softmax, backend="tt")
    with torch.no_grad():
        plain_dev = c_plain(scores.to(device))
    torch_xla.sync()
    pcc_plain = compute_pcc(plain_dev.cpu(), softmax_out)
    print(f"  Plain softmax PCC: {pcc_plain:.6f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
