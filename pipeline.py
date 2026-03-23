# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Phase 3.3: QwenImageXLAPipeline — Full pipeline for Qwen-Image on tt-xla.

Orchestrates text encoding (CPU), MMDiT denoising (XLA devices with TP),
and VAE decoding (CPU) for end-to-end text-to-image generation.

Component placement:
  - Text encoder (Qwen2.5-VL-7B): CPU — 7B VLM too large for device alongside MMDiT
  - MMDiT transformer (20B): XLA devices with TP — runs 50x per image
  - VAE decoder: CPU — runs once, causal 3D conv may lack XLA support
  - Scheduler: CPU — pure arithmetic

Usage:
    from pipeline import QwenImageXLAPipeline
    pipe = QwenImageXLAPipeline("./weights/qwen-image")
    image = pipe.generate("A cat holding a sign that says hello world")
"""

import gc
import os
import time
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch_xla
import torch_xla.runtime as xr
from diffusers import DiffusionPipeline

from utils.image_utils import (
    align_to_patch,
    calculate_shift,
    pack_latents,
    save_image,
    unpack_latents,
)
from utils.profiling_utils import PipelineTiming, StepTiming, Timer, check_no_nan_inf
from utils.tp_utils import apply_tp_sharding, setup_spmd


class QwenImageXLAPipeline:
    """End-to-end Qwen-Image text-to-image pipeline on Tenstorrent via tt-xla.

    Architecture:
        1. Text encoding on CPU (Qwen2.5-VL-7B)
        2. Denoising loop on XLA devices with TP (20B MMDiT, 50 steps)
        3. VAE decoding on CPU
    """

    def __init__(
        self,
        weights_dir: str = "./weights/qwen-image",
        dtype: torch.dtype = torch.bfloat16,
    ):
        """Initialize pipeline.

        Args:
            weights_dir: Path to Qwen-Image model weights.
            dtype: Model dtype (bf16 recommended for TT hardware).
        """
        self.dtype = dtype
        self.timing = PipelineTiming()

        # --- SPMD Setup ---
        self.mesh, self.device, self.num_devices = setup_spmd()
        print(f"QwenImageXLAPipeline: {self.num_devices} device(s)")

        # --- Load pipeline on CPU ---
        print("Loading model components...")
        pipe = DiffusionPipeline.from_pretrained(weights_dir, torch_dtype=dtype)

        # Text encoder stays on CPU (7B VLM)
        self.text_encoder = pipe.text_encoder
        self.tokenizer = pipe.tokenizer

        # Scheduler stays on CPU
        self.scheduler = pipe.scheduler

        # VAE stays on CPU
        self.vae = pipe.vae

        # Pipeline config
        self.vae_scale_factor = 2 ** len(self.vae.temperal_downsample) if hasattr(self.vae, "temperal_downsample") else 8
        self.prompt_template = pipe.prompt_template_encode if hasattr(pipe, "prompt_template_encode") else (
            "<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
            "texture, quantity, text, spatial relationships of the objects and background:"
            "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        )
        self.prompt_template_drop_idx = 34
        self.tokenizer_max_length = 1024

        # --- Move MMDiT to device and apply TP ---
        self.transformer = pipe.transformer.eval().to(self.device)
        config = self.transformer.config

        if self.num_devices > 1 and config.num_attention_heads % self.num_devices == 0:
            apply_tp_sharding(self.transformer, self.mesh)

        self.compiled_transformer = torch.compile(self.transformer, backend="tt")

        # NOTE: RoPE / complex tensor limitation
        # The transformer's pos_embed (QwenImageRotaryEmbedding) returns complex64
        # tensors for rotary embeddings. TT device does not support complex64 yet,
        # so pos_embed must run on CPU. The transformer.forward() calls
        # self.pos_embed() internally with device=hidden_states.device, which would
        # place complex tensors on TT device and fail.
        #
        # We keep a CPU-side reference to pos_embed for pre-computing RoPE via
        # _prepare_rotary_emb(), which callers can use to pass pre-computed
        # embeddings to block-level forward calls.
        #
        # For end-to-end pipeline generation (which goes through
        # transformer.forward()), full support requires either:
        #   a. tt-xla / tt-mlir support for complex64 ops, OR
        #   b. Decomposing RoPE complex multiply to real arithmetic (cos/sin)
        #      as a composite op registered in tt-xla.
        # Move pos_embed back to CPU so its buffers (pos_freqs, neg_freqs) stay
        # off-device.  pos_embed is a nn.Module with registered buffers, so .cpu()
        # moves all its parameters/buffers to CPU in-place.
        self.transformer.pos_embed = self.transformer.pos_embed.cpu()
        self._pos_embed_cpu = self.transformer.pos_embed

        # Free CPU copy of transformer
        del pipe.transformer
        gc.collect()

        print("Pipeline ready.")

    def _extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        """Extract hidden states where mask is 1, matching pipeline_qwenimage.py."""
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_result = torch.split(selected, valid_lengths.tolist(), dim=0)
        return split_result

    def _prepare_rotary_emb(
        self,
        img_shapes: List[List[Tuple[int, int, int]]],
        txt_seq_lens: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pre-compute RoPE embeddings on CPU.

        The transformer's pos_embed produces complex64 tensors, which are not
        supported on TT device.  This helper runs pos_embed on CPU and returns
        the (img_freqs, txt_freqs) tuple that can be threaded into block-level
        forward calls.

        Args:
            img_shapes: Packed image shapes, e.g. [[(1, h_packed, w_packed)]].
            txt_seq_lens: Text sequence lengths per batch element.

        Returns:
            (img_freqs, txt_freqs) — complex64 tensors on CPU.
        """
        with torch.no_grad():
            img_freqs, txt_freqs = self._pos_embed_cpu(
                img_shapes, txt_seq_lens, device=torch.device("cpu")
            )
        return img_freqs, txt_freqs

    def encode_prompt(
        self,
        prompt: str,
        max_sequence_length: int = 512,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a text prompt using the Qwen2.5-VL-7B text encoder on CPU.

        Returns:
            (prompt_embeds, prompt_embeds_mask) — both [1, seq, dim]
        """
        template = self.prompt_template
        drop_idx = self.prompt_template_drop_idx

        txt = template.format(prompt)
        tokens = self.tokenizer(
            txt,
            max_length=self.tokenizer_max_length + drop_idx,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            output = self.text_encoder(
                input_ids=tokens.input_ids,
                attention_mask=tokens.attention_mask,
                output_hidden_states=True,
            )

        hidden_states = output.hidden_states[-1]
        split_hidden = self._extract_masked_hidden(hidden_states, tokens.attention_mask)
        split_hidden = [e[drop_idx:] for e in split_hidden]
        attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden]

        max_seq_len = max(e.size(0) for e in split_hidden)
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden]
        )
        encoder_attention_mask = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
        )

        # Trim to max_sequence_length
        prompt_embeds = prompt_embeds[:, :max_sequence_length].to(dtype=self.dtype)
        encoder_attention_mask = encoder_attention_mask[:, :max_sequence_length]

        return prompt_embeds, encoder_attention_mask

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 50,
        true_cfg_scale: float = 4.0,
        seed: int = 42,
        max_sequence_length: int = 512,
    ) -> torch.Tensor:
        """Generate an image from a text prompt.

        Args:
            prompt: Text prompt describing the image to generate.
            negative_prompt: Negative prompt for CFG. Empty string disables CFG.
            width: Output image width (will be aligned to 16).
            height: Output image height (will be aligned to 16).
            num_inference_steps: Number of denoising steps.
            true_cfg_scale: True CFG guidance scale. Set <= 1.0 to disable CFG.
            seed: Random seed for reproducibility.
            max_sequence_length: Max text token length.

        Returns:
            Image tensor [B, C, H, W] in float32.
        """
        timer = Timer()
        total_timer = Timer()
        total_timer.start()

        # Align dimensions
        width = align_to_patch(width, self.vae_scale_factor)
        height = align_to_patch(height, self.vae_scale_factor)
        print(f"Generating {width}x{height} image, {num_inference_steps} steps, cfg={true_cfg_scale}")

        # --- 1. Encode text (CPU) ---
        timer.start()
        print("Encoding prompt...")
        cond_embeds, cond_mask = self.encode_prompt(prompt, max_sequence_length)

        do_true_cfg = true_cfg_scale > 1.0 and negative_prompt is not None
        if do_true_cfg:
            neg_embeds, neg_mask = self.encode_prompt(negative_prompt, max_sequence_length)
        else:
            neg_embeds, neg_mask = None, None

        self.timing.text_encode_ms = timer.stop()
        print(f"  Text encoding: {self.timing.text_encode_ms:.0f}ms")

        # Move embeddings to device
        cond_embeds_d = cond_embeds.to(self.device)
        cond_mask_d = cond_mask.to(self.dtype).to(self.device)
        cond_txt_seq_lens = cond_mask.sum(dim=1).tolist()

        if do_true_cfg:
            neg_embeds_d = neg_embeds.to(self.device)
            neg_mask_d = neg_mask.to(self.dtype).to(self.device)
            neg_txt_seq_lens = neg_mask.sum(dim=1).tolist()

        # --- 2. Prepare latents ---
        config = self.transformer.config
        latent_h = height // self.vae_scale_factor
        latent_w = width // self.vae_scale_factor
        num_channels = config.in_channels // 4  # 16

        generator = torch.Generator().manual_seed(seed)
        latents = torch.randn(
            1, 1, num_channels, latent_h, latent_w,
            generator=generator,
            dtype=self.dtype,
        )
        latents = pack_latents(latents.squeeze(1), 1, num_channels, latent_h, latent_w)
        latents = latents.to(self.device)

        img_seq_len = latents.shape[1]
        h_packed = latent_h // 2
        w_packed = latent_w // 2
        img_shapes = [[(1, h_packed, w_packed)]]

        # --- 3. Prepare scheduler ---
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        mu = calculate_shift(img_seq_len)
        self.scheduler.set_timesteps(num_inference_steps, sigmas=sigmas, mu=mu)
        timesteps = self.scheduler.timesteps
        self.scheduler.set_begin_index(0)

        # --- 4. Denoising loop ---
        timer.start()
        print(f"Denoising ({num_inference_steps} steps)...")

        for i, t in enumerate(timesteps):
            step_timer = Timer()
            step_timer.start()

            timestep = t.expand(latents.shape[0]).to(self.dtype).to(self.device)

            with torch.no_grad():
                # Conditioned forward pass
                noise_pred_cond = self.compiled_transformer(
                    hidden_states=latents,
                    encoder_hidden_states=cond_embeds_d,
                    encoder_hidden_states_mask=cond_mask_d,
                    timestep=timestep / 1000,
                    img_shapes=img_shapes,
                    txt_seq_lens=cond_txt_seq_lens,
                    return_dict=False,
                )[0]

                if do_true_cfg:
                    # Unconditioned forward pass
                    noise_pred_uncond = self.compiled_transformer(
                        hidden_states=latents,
                        encoder_hidden_states=neg_embeds_d,
                        encoder_hidden_states_mask=neg_mask_d,
                        timestep=timestep / 1000,
                        img_shapes=img_shapes,
                        txt_seq_lens=neg_txt_seq_lens,
                        return_dict=False,
                    )[0]

                    # True CFG combination with norm-preserving rescaling
                    comb_pred = noise_pred_uncond + true_cfg_scale * (noise_pred_cond - noise_pred_uncond)
                    cond_norm = torch.norm(noise_pred_cond, dim=-1, keepdim=True)
                    noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
                    noise_pred = comb_pred * (cond_norm / noise_norm)
                else:
                    noise_pred = noise_pred_cond

            # Scheduler step (CPU)
            latents_cpu = latents.cpu()
            noise_pred_cpu = noise_pred.cpu()
            latents_next = self.scheduler.step(noise_pred_cpu, t, latents_cpu, return_dict=False)[0]
            latents = latents_next.to(self.device)

            torch_xla.sync()

            step_ms = step_timer.stop()
            self.timing.steps.append(StepTiming(step_idx=i, total_ms=step_ms))

            if (i + 1) % 10 == 0 or i == 0:
                print(f"  Step {i+1}/{num_inference_steps}: {step_ms:.0f}ms")

        self.timing.denoising_total_ms = timer.stop()

        # --- 5. VAE decode (CPU) ---
        timer.start()
        print("Decoding with VAE...")
        latents_for_decode = latents.cpu()
        latents_for_decode = unpack_latents(latents_for_decode, height, width, self.vae_scale_factor)
        latents_for_decode = latents_for_decode.to(self.vae.dtype)

        # Apply latent normalization (from pipeline)
        if hasattr(self.vae.config, "latents_mean") and self.vae.config.latents_mean is not None:
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents_for_decode.device, latents_for_decode.dtype)
            )
            latents_std = (
                1.0 / torch.tensor(self.vae.config.latents_std)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents_for_decode.device, latents_for_decode.dtype)
            )
            latents_for_decode = latents_for_decode / latents_std + latents_mean

        with torch.no_grad():
            image = self.vae.decode(latents_for_decode, return_dict=False)[0]
            # Remove temporal dim: [B, C, T, H, W] -> [B, C, H, W]
            if image.dim() == 5:
                image = image[:, :, 0]

        self.timing.vae_decode_ms = timer.stop()
        self.timing.total_ms = (time.perf_counter() - (total_timer._start or time.perf_counter())) * 1000

        print(self.timing.summary())
        return image
