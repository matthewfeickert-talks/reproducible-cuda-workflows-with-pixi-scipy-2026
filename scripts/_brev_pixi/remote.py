"""Remote shell script builders for the Brev Pixi runner."""

from __future__ import annotations

import shlex
from pathlib import PurePosixPath

from .models import RemoteProject


def bootstrap_script() -> str:
    """Build the remote script that installs Pixi if needed and checks CUDA."""

    return "\n".join(
        [
            "set -euo pipefail",
            'export PATH="$HOME/.pixi/bin:$PATH"',
            "if ! command -v pixi >/dev/null 2>&1; then",
            "  curl -fsSL https://pixi.sh/install.sh | bash",
            '  export PATH="$HOME/.pixi/bin:$PATH"',
            "fi",
            "pixi --version",
            "nvidia-smi",
        ]
    )


def remote_project_script(remote: RemoteProject, command: str) -> str:
    """Build a remote script that runs inside the uploaded Pixi workspace."""

    return "\n".join(
        [
            "set -euo pipefail",
            'export PATH="$HOME/.pixi/bin:$PATH"',
            f"cd {shlex.quote(str(remote.manifest_dir))}",
            command,
        ]
    )


def create_enter_script(remote: RemoteProject, enter_script: PurePosixPath) -> str:
    """Build the remote script that writes the interactive Pixi shell helper."""

    body = "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            'export PATH="$HOME/.pixi/bin:$PATH"',
            f"cd {shlex.quote(str(remote.manifest_dir))}",
            "eval \"$(pixi shell-hook --shell bash --manifest-path "
            + shlex.quote(remote.manifest_name)
            + ")\"",
            'echo "Activated Pixi environment for $(pwd)."',
        ]
    )
    quoted_body = shlex.quote(body)
    return "\n".join(
        [
            "set -euo pipefail",
            f"mkdir -p {shlex.quote(str(enter_script.parent))}",
            f"printf '%s\\n' {quoted_body} > {shlex.quote(str(enter_script))}",
            f"chmod +x {shlex.quote(str(enter_script))}",
        ]
    )
