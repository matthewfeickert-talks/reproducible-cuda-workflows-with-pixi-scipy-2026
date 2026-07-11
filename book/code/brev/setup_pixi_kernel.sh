#!/usr/bin/env bash

# Register a global "Python (Pixi)" Jupyter kernel for VS Code Remote.
#
# The kernel launches `pixi run python -m ipykernel_launcher` from the
# notebook's directory, so it automatically resolves whichever Pixi project
# the notebook lives in — no per-project setup, no need to open the project
# folder as the VS Code workspace root. The only requirement is that each
# Pixi project has `ipykernel` in its environment (pulled in transitively by
# `jupyterlab` or `notebook`).
#
# Run once per machine (e.g. from your Brev setup script, after installing pixi).
set -euo pipefail

PIXI_BIN="$(command -v pixi || echo "$HOME/.pixi/bin/pixi")"
KERNEL_DIR="$HOME/.local/share/jupyter/kernels/pixi-kernel-python3"

mkdir -p "$KERNEL_DIR"
cat > "$KERNEL_DIR/kernel.json" <<EOF
{
  "argv": [
    "$PIXI_BIN",
    "run",
    "python",
    "-m",
    "ipykernel_launcher",
    "-f",
    "{connection_file}"
  ],
  "display_name": "Python (Pixi)",
  "language": "python",
  "metadata": {
    "debugger": true
  }
}
EOF

echo "Registered kernelspec at $KERNEL_DIR/kernel.json"
