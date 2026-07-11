#!/usr/bin/env bash

# Install Pixi
curl -fsSL https://pixi.sh/install.sh | bash
echo 'eval "$(pixi completion --shell bash)"' >> ${HOME}/.bashrc

# Register a global "Python (Pixi)" Jupyter kernel for VS Code Remote.
curl -sL https://raw.githubusercontent.com/matthewfeickert-talks/reproducible-cuda-workflows-with-pixi-scipy-2026/refs/heads/main/book/code/brev/setup_pixi_kernel.sh | bash

# Clone the exercise repository
git clone https://github.com/matthewfeickert-talks/reproducible-cuda-workflows-with-pixi-scipy-2026.git ${HOME}/reproducible-cuda-workflows-with-pixi-scipy-2026

. ${HOME}/.bashrc
