"""Profile memory during 1024x1024 block compilation.

Tracks RSS at each phase to find where the blowup happens.
"""

import os
import sys
import time
import resource
import threading

# Memory tracking
peak_rss_mb = [0]
phase_log = []

def get_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024

def log_phase(name):
    rss = get_rss_mb()
    peak_rss_mb[0] = max(peak_rss_mb[0], rss)
    phase_log.append((name, rss, time.perf_counter()))
    print(f"  [{rss:6d}MB] {name}")
    sys.stdout.flush()

# Memory monitor thread — logs RSS every 10 seconds
stop_monitor = threading.Event()
def memory_monitor():
    while not stop_monitor.is_set():
        rss = get_rss_mb()
        peak_rss_mb[0] = max(peak_rss_mb[0], rss)
        stop_monitor.wait(10)

monitor = threading.Thread(target=memory_monitor, daemon=True)
monitor.start()

log_phase("start")

import torch
import torch.nn.functional as F
import math

log_phase("torch imported")

os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
import torch_xla
import torch_xla.runtime as xr

# Use opt level 0 to avoid optimizer — we want to find the BASE compilation bottleneck
try:
    torch_xla.set_custom_compile_options({"optimization_level": "0"})
except:
    pass

xr.use_spmd()
xr.set_device_type("TT")

log_phase("xla initialized")

from diffusers import DiffusionPipeline
from utils.image_utils import pack_latents

log_phase("diffusers imported")

# Prepare 1024x1024 inputs
device = torch_xla.device()
pipe = DiffusionPipeline.from_pretrained("./weights/qwen-image", torch_dtype=torch.bfloat16)
transformer = pipe.transformer.eval()
config = transformer.config

log_phase("model loaded")

width, height = 1024, 1024
vae_sf = 8
lh, lw = height // vae_sf, width // vae_sf
nc = config.in_channels // 4
max_txt = 128

prompt_embeds, prompt_mask = pipe.encode_prompt(prompt="A cat", device="cpu")
prompt_embeds = F.pad(prompt_embeds, (0, 0, 0, max_txt - prompt_embeds.shape[1]))
prompt_mask = F.pad(prompt_mask, (0, max_txt - prompt_mask.shape[1]))

gen = torch.Generator().manual_seed(42)
latents = pack_latents(
    torch.randn(1, nc, lh, lw, generator=gen, dtype=torch.bfloat16), 1, nc, lh, lw,
)

log_phase("inputs prepared")

# Move block 0 to device
block = transformer.transformer_blocks[0].to(device)

log_phase("block on device")

# Prepare block inputs
from demo import patched_block_forward, complex_to_real_rope

img_shapes = [[(1, lh // 2, lw // 2)]]
img_fc, txt_fc = transformer.pos_embed(img_shapes, [max_txt], device=torch.device("cpu"))
img_rope = complex_to_real_rope(img_fc)
txt_rope = complex_to_real_rope(txt_fc)

hs = transformer.img_in(latents).to(device)
ehs = transformer.txt_in(transformer.txt_norm(prompt_embeds)).to(device)
ts = torch.tensor([0.999], dtype=torch.bfloat16).to(device)
temb = transformer.time_text_embed(ts.to(hs.dtype), hs)
rope_dev = (
    (img_rope[0].to(torch.bfloat16).to(device), img_rope[1].to(torch.bfloat16).to(device)),
    (txt_rope[0].to(torch.bfloat16).to(device), txt_rope[1].to(torch.bfloat16).to(device)),
)
mask_d = prompt_mask.to(torch.bfloat16).to(device)

log_phase("block inputs ready")

print(f"\nInput shapes:")
print(f"  hs:   {list(hs.shape)}")
print(f"  ehs:  {list(ehs.shape)}")
print(f"  temb: {list(temb.shape)}")
print(f"  Total seq: {hs.shape[1] + ehs.shape[1]}")
print(f"\nCompiling block 0 at 1024x1024...")
sys.stdout.flush()

# Compile and execute
def make_fn(b):
    def fn(hs, ehs, em, temb, rope):
        return patched_block_forward(b, hs, ehs, em, temb, rope)
    return fn

compiled = torch.compile(make_fn(block), backend="tt")

log_phase("torch.compile registered")

t0 = time.perf_counter()

# This is where the compilation actually happens (first call triggers it)
try:
    with torch.no_grad():
        txt_out, img_out = compiled(hs, ehs, mask_d, temb, rope_dev)
    torch_xla.sync()
    log_phase("compilation + execution done")
except Exception as e:
    log_phase(f"FAILED: {type(e).__name__}")
    print(f"  Error: {e}")

dt = time.perf_counter() - t0
stop_monitor.set()

print(f"\n{'='*60}")
print(f"MEMORY PROFILE SUMMARY")
print(f"{'='*60}")
print(f"Peak RSS: {peak_rss_mb[0]}MB")
print(f"Compilation time: {dt:.1f}s")
print(f"\nPhase timeline:")
t_start = phase_log[0][2]
for name, rss, t in phase_log:
    print(f"  {t - t_start:7.1f}s  {rss:6d}MB  {name}")
