# Patches

## tt-metal-softmax-fix.patch

Fixes `ttnn.softmax` correctness for large input values (attention scores > ~1000).

**Root cause:** The hardware `exp_tile` function produces garbage for very large negative
inputs after the stable-softmax `x - max` subtraction. Models with QK-norm (like Qwen-Image)
produce attention scores with mean ~5000 and max ~35000. After max subtraction, values down
to -35000 are passed to exp, which returns non-zero garbage instead of 0.

**Fix (3 files):**

1. `softmax.cpp` + `softmax_large_tensor.cpp`: Add `clamp_tile(dst, -88.0f, 0.0f)` after
   `sub_tiles_bcast_cols` and before `exp_tile`. This is mathematically safe since
   exp(-88) < FLT_MIN in f32.

2. `softmax_device_operation.cpp`: Change `default_approx_mode` from `true` to `false`.
   The approx exp path only clamps at -42; the non-approx path has tighter internal
   clamping at [-88.5, 88.5].

**Impact:** PCC of softmax with real Qwen-Image scores goes from 0.12 → 0.99.

**Apply:**
```bash
cd /path/to/tt-metal
git apply /path/to/tt-metal-softmax-fix.patch
```

**Upstream status:** Novel fix, not yet in upstream tt-metal. Should be proposed as a PR.
The `math_approx_mode=false` change alone may be sufficient (the non-approx path has
internal clamping), but the explicit `clamp_tile` provides defense-in-depth.
