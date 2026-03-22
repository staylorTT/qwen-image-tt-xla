# Qwen-Image 20B on Tenstorrent via tt-xla

End-to-end text-to-image generation using the Qwen-Image 20B MMDiT model
on Tenstorrent silicon through the tt-xla (PyTorch/XLA → StableHLO → tt-mlir → TTNN) pipeline.

## Quick Start

```bash
# 1. Download weights
python download_weights.py --model-id Qwen/Qwen-Image --output-dir ./weights/qwen-image

# 2. Inspect model architecture (discover attribute paths)
python inspect_model.py --weights-dir ./weights/qwen-image

# 3. Run smoke compile test
python test_smoke_compile.py --weights-dir ./weights/qwen-image

# 4. Generate an image
python run_generate.py \
    --prompt 'A sign that reads "Hello Tenstorrent"' \
    --width 1024 --height 1024 \
    --steps 50 --cfg_scale 4.0 \
    --output output.png
```

## File Structure

| File | Phase | Purpose |
|------|-------|---------|
| `download_weights.py` | 0.2 | Download model weights from HuggingFace |
| `inspect_model.py` | 0.3 | Discover MMDiT attribute paths and structure |
| `inspect_text_encoder.py` | 0.4 | Inspect Qwen2.5-VL-7B text encoder |
| `test_smoke_compile.py` | 0.5 | StableHLO capture smoke test |
| `test_single_block.py` | 1.3 | Single block correctness (PCC >= 0.998) |
| `test_single_block_tp.py` | 2.3 | Single block with TP |
| `test_full_mmdit_tp.py` | 2.3 | Full 60-layer with TP |
| `test_denoising_loop.py` | 3.3 | Multi-step denoising loop |
| `test_true_cfg.py` | 3.3 | True CFG correctness |
| `pipeline.py` | 3.3 | QwenImageXLAPipeline class |
| `run_generate.py` | 3.4 | End-to-end CLI |
| `adaln_cache.py` | 4.1 | AdaLN pre-computation |
| `benchmark.py` | 6.1 | Performance measurement |
| `utils/tp_utils.py` | — | TP sharding utilities |
| `utils/image_utils.py` | — | Image handling utilities |
| `utils/profiling_utils.py` | — | PCC, timing, profiling |
| `tests/` | 1.2, 4.x | Pattern matching and fusion tests |
