# SDPA Investigation: Fused FlashAttention-2 on Tenstorrent

## Summary

We got Qwen-Image 20B generating correct images on TT hardware (4x Blackhole, 4-way TP)
at 1.75s/step (256x256) and 5.4s/step (512x512). Attention uses a manual decomposition
(matmul + softmax + matmul = 3 kernel dispatches per block). Fusing to a single
FlashAttention-2 dispatch would cut kernel launches from 180 → 60 per step and avoid
materializing the N×N attention matrix.

## Bug Found: ttnn.softmax Precision with Large Inputs

**Root cause:** QK-norm learnable weights amplify Q/K magnitudes (~25x std). The f32
attention scores reach mean=5298, max=34487. `ttnn.softmax` breaks because `exp_tile`
produces garbage for extreme negative inputs after `x - max` subtraction.

**Fix:** Added `clamp_tile(-88, 0)` in softmax kernels + set `math_approx_mode=false`.
See `patches/tt-metal-softmax-fix.patch`.

**Evidence:**
- Random inputs (std=1): softmax PCC = 0.999 (passes)
- Real model inputs (std=25): softmax PCC = 0.117 (fails)
- Real model inputs + our fix: softmax PCC = 0.993 (passes)
- This is why existing unit tests never caught it — they use random inputs

## SDPA Fusing: Current Status

### How to enable it
```python
torch_xla.set_custom_compile_options({"optimization_level": "1"})
```

This activates the chain:
```
optimization_level >= 1
  → optimizerPassEnabled = true
    → enableOpConstraints = true
      → SDPAFusing pattern added to fusing pass
```

The fusing pattern in `TTNNFusing.cpp` matches:
```
matmul(softmax(matmul(Q, K^T) * scale), V)
```
and replaces it with `ttnn.scaled_dot_product_attention` (FlashAttention-2).

### Blocker: concatenate_heads dimension mismatch

When the fusing pattern fires, it crashes:
```
'ttnn.concatenate_heads' op input sequence dimension must match output sequence dimension,
got input sequence size = 24, output sequence size = 384
```

**Cause:** Our attention output goes through `[B,H,S,D] → transpose(1,2) → [B,S,H,D] → flatten(2,3) → [B,S,H*D]`. The fusing pattern inserts `concatenate_heads` which expects `[B,H,S,D] → [B,S,H*D]` directly without the transpose. It confuses num_heads (24) with seq_len (384).

**Fix needed in:** `TTNNFusing.cpp` around the `createSDPAOp` function (~line 936). The pattern needs to detect and absorb the post-SDPA transpose+flatten instead of blindly inserting concatenate_heads.

### Alternative: tt.scaled_dot_product_attention custom op

tt-xla provides `torch.ops.tt.scaled_dot_product_attention` which emits a StableHLO
custom call directly lowered to fused SDPA (bypassing pattern matching). However, when
used inside a ~200-op compiled block, the compiler process explodes (7868 threads, 112GB
RSS, never completes). This path needs compiler team investigation.

## Performance Comparison

| Configuration | Per-step (256x256 CFG) | Per-step (512x512 CFG) |
|--------------|----------------------|----------------------|
| Current (manual 3-op attention) | 1.75s | 5.4s |
| F.sdpa (17-op decomposition) | ~2.1s (slower) | ~7s (slower) |
| Fused FlashAttention-2 (target) | ~0.8-1.0s (estimated) | ~2-3s (estimated) |

## Key Files

```
# Our application
demo.py                          — Interactive demo (the shipping code)
generate_image_v2.py             — Standalone generation script
utils/                           — Image, profiling, TP utilities

# Our fixes
patches/tt-metal-softmax-fix.patch — Softmax kernel precision fix

# Compiler (in tt-mlir)
lib/Dialect/TTNN/Transforms/TTNNFusing.cpp     — SDPA fusing pattern
lib/Dialect/TTNN/Pipelines/TTNNPipelines.cpp   — Pipeline setup (opt level gating)
include/ttmlir/Dialect/TTNN/Transforms/Passes.td — enableOpConstraints option

# Runtime (in tt-metal)
ttnn/operations/normalization/softmax/         — Softmax kernels (our fix)
ttnn/operations/transformer/sdpa/              — FlashAttention-2 implementation

# tt-xla
python_package/tt_torch/custom_ops.py          — tt.scaled_dot_product_attention custom op
pjrt_implementation/src/api/module_builder/    — Compiler pipeline invocation
```
