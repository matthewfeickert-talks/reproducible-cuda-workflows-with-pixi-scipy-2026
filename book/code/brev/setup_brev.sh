#!/usr/bin/env bash

curl -fsSL https://pixi.sh/install.sh | bash
echo 'eval "$(pixi completion --shell bash)"' >> ${HOME}/.bashrc

curl -sL https://raw.githubusercontent.com/matthewfeickert-talks/reproducible-cuda-workflows-with-pixi-scipy-2026/refs/heads/main/book/code/brev/setup_pixi_kernel.sh | bash

. ${HOME}/.bashrc
