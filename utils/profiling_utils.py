# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Profiling utilities for Qwen-Image pipeline benchmarking.

Provides timing helpers, memory tracking, and PCC (Pearson Correlation Coefficient)
computation for correctness verification against CPU reference.
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch


@dataclass
class StepTiming:
    """Timing for a single denoising step."""
    step_idx: int
    total_ms: float
    transformer_ms: float = 0.0
    scheduler_ms: float = 0.0


@dataclass
class PipelineTiming:
    """Aggregate timing for a full pipeline run."""
    text_encode_ms: float = 0.0
    denoising_total_ms: float = 0.0
    vae_decode_ms: float = 0.0
    total_ms: float = 0.0
    steps: List[StepTiming] = field(default_factory=list)

    @property
    def avg_step_ms(self) -> float:
        if not self.steps:
            return 0.0
        return sum(s.total_ms for s in self.steps) / len(self.steps)

    def summary(self) -> str:
        lines = [
            f"Pipeline Timing Summary:",
            f"  Text encoding:   {self.text_encode_ms:.1f}ms",
            f"  Denoising total: {self.denoising_total_ms:.1f}ms ({len(self.steps)} steps)",
            f"  Avg step:        {self.avg_step_ms:.1f}ms",
            f"  VAE decode:      {self.vae_decode_ms:.1f}ms",
            f"  Total:           {self.total_ms:.1f}ms ({self.total_ms / 1000:.2f}s)",
        ]
        return "\n".join(lines)


class Timer:
    """Simple wall-clock timer."""

    def __init__(self):
        self._start = None

    def start(self):
        self._start = time.perf_counter()

    def stop(self) -> float:
        """Returns elapsed time in milliseconds."""
        elapsed = (time.perf_counter() - self._start) * 1000
        self._start = None
        return elapsed


@contextmanager
def timed(label: str = ""):
    """Context manager that prints elapsed time."""
    start = time.perf_counter()
    yield
    elapsed_ms = (time.perf_counter() - start) * 1000
    if label:
        print(f"[{label}] {elapsed_ms:.1f}ms")


def compute_pcc(
    actual: torch.Tensor,
    reference: torch.Tensor,
) -> float:
    """Compute Pearson Correlation Coefficient between two tensors.

    Used for correctness verification: PCC >= 0.998 for single block,
    PCC >= 0.99 for full 60-block stack.

    Args:
        actual: Output from device/compiled model.
        reference: Output from CPU reference.

    Returns:
        PCC value in [0, 1]. Values >= 0.998 indicate excellent agreement.
    """
    a = actual.float().flatten()
    b = reference.float().flatten()

    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: actual={a.shape}, reference={b.shape}")

    a_mean = a.mean()
    b_mean = b.mean()
    a_centered = a - a_mean
    b_centered = b - b_mean

    numerator = (a_centered * b_centered).sum()
    denominator = torch.sqrt((a_centered**2).sum() * (b_centered**2).sum())

    if denominator < 1e-12:
        # Both tensors are constant — PCC is undefined, treat as perfect
        return 1.0

    return (numerator / denominator).item()


def check_pcc(
    actual: torch.Tensor,
    reference: torch.Tensor,
    threshold: float = 0.998,
    label: str = "",
) -> bool:
    """Check PCC meets threshold and print result.

    Args:
        actual: Output tensor from device.
        reference: Output tensor from CPU reference.
        threshold: Minimum acceptable PCC.
        label: Description for logging.

    Returns:
        True if PCC >= threshold.
    """
    pcc = compute_pcc(actual, reference)
    status = "PASS" if pcc >= threshold else "FAIL"
    prefix = f"[{label}] " if label else ""
    print(f"{prefix}PCC = {pcc:.6f} (threshold = {threshold}) [{status}]")
    return pcc >= threshold


def check_no_nan_inf(tensor: torch.Tensor, label: str = "") -> bool:
    """Check tensor has no NaN or Inf values."""
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    prefix = f"[{label}] " if label else ""

    if has_nan or has_inf:
        print(f"{prefix}FAIL: nan={has_nan}, inf={has_inf}")
        return False
    print(f"{prefix}No NaN/Inf: PASS")
    return True
