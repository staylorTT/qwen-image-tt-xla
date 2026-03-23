#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Launcher script for running Qwen-Image tt-xla scripts on TT hardware.
#
# Uses the tt-xla build environment which includes:
#   - Python 3.11 with torch, torch_xla, diffusers
#   - Built PJRT plugin for TT devices
#   - tt-mlir toolchain
#
# Usage:
#   ./run.sh <script.py> [args...]
#   ./run.sh tests/test_pattern_match_joint_attn.py
#   ./run.sh test_smoke_compile.py --weights-dir ./weights/qwen-image

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TT_XLA_DIR="/home/ttuser/james/tt-xla"

if [ ! -d "$TT_XLA_DIR" ]; then
    echo "ERROR: tt-xla directory not found at $TT_XLA_DIR"
    exit 1
fi

# Activate tt-xla environment
cd "$TT_XLA_DIR"
source venv/activate 2>/dev/null

# Add our project to PYTHONPATH so utils/ imports work
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# Return to script directory for relative paths
cd "$SCRIPT_DIR"

# Run the requested script
exec "$TT_XLA_DIR/venv/bin/python3" "$@"
