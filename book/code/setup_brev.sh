#!/usr/bin/env bash

curl -fsSL https://pixi.sh/install.sh | bash
echo 'eval "$(pixi completion --shell bash)"' >> ${HOME}/.bashrc
. ${HOME}/.bashrc
