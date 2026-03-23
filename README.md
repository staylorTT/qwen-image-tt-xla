# Qwen-Image 20B on Tenstorrent via tt-xla

End-to-end image generation with the Qwen-Image 20B MMDiT model on Tenstorrent Blackhole
hardware through the tt-xla (PyTorch/XLA → StableHLO → tt-mlir → TTNN) stack.

## Quick Start

```bash
# Interactive demo (256x256, 15 steps, CFG=4.0)
./run.sh demo.py

# Standalone generation
./run.sh generate_image_v2.py --prompt "A cat" --width 512 --height 512

# With custom settings
./run.sh demo.py --width 512 --height 512 --steps 20 --cfg_scale 4.0
```

## Performance (4x Blackhole, 4-way TP, warm)

| Resolution | CFG | Steps | Per-step | Total |
|-----------|-----|-------|----------|-------|
| 256x256 | 4.0 | 15 | 1.75s | 28s |
| 256x256 | 1.0 | 15 | 1.04s | 18s |
| 512x512 | 4.0 | 20 | 5.42s | 112s |

First image includes ~180s compilation (cached for subsequent images).

## Architecture

- **Text encoder** (Qwen2.5-VL-7B): CPU — too large to co-locate with transformer
- **MMDiT transformer** (20B, 60 blocks): TT device with 4-way TP
  - Each block compiled individually via `torch.compile(backend="tt")`
  - Manual attention decomposition (matmul + softmax + matmul) — workaround for SDPA bug
  - Real-valued RoPE (pre-computed on CPU) — workaround for complex64 unsupported on TT
  - Batched CFG (cond + uncond as batch=2 in single forward pass)
  - On-device Euler scheduler step (no CPU round-trip)
- **VAE decoder**: CPU — 3D causal convolutions not supported on TT
- **Text embeddings**: Padded to fixed 128 tokens to avoid recompilation across prompts

## File Structure

| File | Description |
|------|-------------|
| `demo.py` | Interactive demo with all optimizations |
| `generate_image_v2.py` | Standalone generation script |
| `pipeline.py` | Pipeline class (reference) |
| `adaln_cache.py` | AdaLN caching optimization (future) |
| `benchmark.py` | Performance benchmarking |
| `inspect_model.py` | Model architecture inspection |
| `download_weights.py` | Weight downloader |
| `utils/` | Image, profiling, TP utilities |
| `tests/` | Pattern matching tests (RoPE, AdaLN, joint attention, D2M fusions) |
| `test_*.py` | Device correctness tests (single block, TP, denoising loop, CFG) |
| `patches/` | Fixes for tt-metal softmax kernel |
| `investigation/` | Debug scripts and IR dumps from SDPA investigation |
| `docs/` | Investigation writeup and findings |

## Known Issues & Fixes

### 1. ttnn.softmax precision bug (FIXED)

`ttnn.softmax` produces wrong results for inputs > ~1000 (common with QK-norm'd models).
Root cause: hardware `exp_tile` fails on extreme negative values after stable-softmax
`x - max` subtraction. Fix in `patches/tt-metal-softmax-fix.patch`.

### 2. SDPA fusing pattern dimension mismatch (OPEN)

Setting `optimization_level=1` activates the SDPA fusing pass, but `concatenate_heads`
confuses num_heads with seq_len when the post-SDPA output is transposed before flattening.
See `docs/sdpa-investigation.md` for full analysis and proposed fix location.

### 3. Complex64 RoPE not supported on TT (WORKAROUND)

Qwen-Image uses complex-valued RoPE. TT devices don't support complex64. Workaround:
pre-compute RoPE on CPU and convert to real-valued (cos, sin) pairs. Validated at PCC=1.0
vs the complex path on CPU.

## Setup

```bash
# Weights (one-time)
./run.sh download_weights.py --model-id Qwen/Qwen-Image --output-dir ./weights/qwen-image

# Apply softmax fix to tt-metal (if not already applied)
cd /path/to/tt-metal
git apply /path/to/patches/tt-metal-softmax-fix.patch
```
