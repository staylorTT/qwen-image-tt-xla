# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Phase 0.2: Download Qwen-Image weights from HuggingFace.

Downloads the Qwen-Image model weights (transformer, text encoder, VAE, scheduler)
to a local directory for offline use.

Usage:
    python download_weights.py [--model-id Qwen/Qwen-Image-2512] [--output-dir ./weights/qwen-image]
"""

import argparse
import os

from huggingface_hub import snapshot_download


KNOWN_MODEL_IDS = [
    "Qwen/Qwen-Image",          # Original (~40GB bf16)
    "Qwen/Qwen-Image-2512",     # Latest weights (Dec 2025 update, same architecture)
]


def download_weights(model_id: str, output_dir: str, token: str = None):
    """Download Qwen-Image model weights from HuggingFace Hub.

    Args:
        model_id: HuggingFace model identifier.
        output_dir: Local directory to save weights.
        token: Optional HuggingFace auth token for gated models.
    """
    print(f"Downloading {model_id} to {output_dir}...")
    snapshot_download(
        model_id,
        local_dir=output_dir,
        allow_patterns=[
            "*.safetensors",
            "*.json",
            "*.txt",
            "*.py",
            "*.model",          # sentencepiece models
            "*.tiktoken",       # tokenizer
            "tokenizer*",
        ],
        token=token,
    )
    print(f"Download complete: {output_dir}")

    # Print summary of downloaded files
    total_size = 0
    file_count = 0
    for root, _dirs, files in os.walk(output_dir):
        for f in files:
            path = os.path.join(root, f)
            size = os.path.getsize(path)
            total_size += size
            file_count += 1
    print(f"  {file_count} files, {total_size / 1e9:.1f} GB total")


def main():
    parser = argparse.ArgumentParser(description="Download Qwen-Image model weights")
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen-Image",
        help=f"HuggingFace model ID. Known: {KNOWN_MODEL_IDS}",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./weights/qwen-image",
        help="Local directory to save weights",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace auth token (or set HF_TOKEN env var)",
    )
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    download_weights(args.model_id, args.output_dir, token=token)


if __name__ == "__main__":
    main()
