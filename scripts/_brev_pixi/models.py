"""Data models and defaults for the Brev Pixi runner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath

DEFAULT_INSTANCE = "pixi-cuda"
DEFAULT_GPU = "g2-standard-4:nvidia-l4:1"
DEFAULT_REMOTE_ROOT = PurePosixPath("/home/ubuntu/brev-pixi-runs")
EXCLUDED_UPLOAD_PATTERNS = (".git", ".jj", ".pixi", "__pycache__", "_build")


@dataclass(frozen=True)
class LocalProject:
    """Local files to upload and the Pixi manifest to use."""

    context: Path
    manifest_relative: Path


@dataclass(frozen=True)
class RemoteProject:
    """Remote paths for one Brev run."""

    run_dir: PurePosixPath
    project_dir: PurePosixPath
    manifest_relative: PurePosixPath

    @property
    def manifest_dir(self) -> PurePosixPath:
        """Directory containing the remote Pixi manifest."""

        return self.project_dir / self.manifest_relative.parent.as_posix()

    @property
    def manifest_name(self) -> str:
        """File name of the remote Pixi manifest."""

        return self.manifest_relative.name


@dataclass(frozen=True)
class Options:
    """Runtime options for one Brev Pixi run."""

    instance: str
    gpu: str
    remote_root: PurePosixPath
    provision: bool
    keep_running: bool
    dry_run: bool
