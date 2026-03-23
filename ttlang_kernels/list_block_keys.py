"""List all weight keys for block 0."""
import os
import safetensors.torch

weights_dir = "/workspace/qwen-image-tt-xla/weights/qwen-image"
transformer_dir = os.path.join(weights_dir, "transformer")
st_files = sorted([f for f in os.listdir(transformer_dir) if f.endswith(".safetensors")])

prefix = "transformer_blocks.0."
keys = []
for st_file in st_files:
    path = os.path.join(transformer_dir, st_file)
    with safetensors.torch.safe_open(path, framework="pt") as f:
        for key in f.keys():
            if key.startswith(prefix):
                short = key[len(prefix):]
                keys.append(short)

for k in sorted(keys):
    print(k)
