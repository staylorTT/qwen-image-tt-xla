import sys
sys.path.insert(0, "/workspace/qwen-image-tt-xla")
try:
    import torch_xla
    print("torch_xla: available")
except ImportError as e:
    print(f"torch_xla: NOT available ({e})")

try:
    from diffusers import DiffusionPipeline
    print("diffusers: available")
except ImportError as e:
    print(f"diffusers: NOT available ({e})")

import os
weights = "/workspace/qwen-image-tt-xla/weights/qwen-image"
if os.path.exists(weights):
    files = os.listdir(weights)
    print(f"weights dir: {len(files)} items")
    for f in sorted(files)[:10]:
        print(f"  {f}")
else:
    print("weights dir: NOT FOUND")
