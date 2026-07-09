"""Local project resolution and upload helpers."""

from __future__ import annotations

import datetime as dt
import os
import shlex
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

from .brev import BrevClient
from .exceptions import BrevPixiError
from .models import EXCLUDED_UPLOAD_PATTERNS, LocalProject, RemoteProject


def resolve_project(pixi_toml: Path, context: Path | None) -> LocalProject:
    """Resolve and validate the Pixi manifest and upload context."""

    resolved_toml = pixi_toml.expanduser().resolve()
    if not resolved_toml.exists():
        raise BrevPixiError(f"pixi.toml does not exist: {pixi_toml}")
    if resolved_toml.name != "pixi.toml":
        raise BrevPixiError(f"expected a pixi.toml file, got: {pixi_toml}")

    resolved_context = context.expanduser().resolve() if context else resolved_toml.parent
    if not resolved_context.is_dir():
        raise BrevPixiError(f"context is not a directory: {resolved_context}")

    try:
        manifest_relative = resolved_toml.relative_to(resolved_context)
    except ValueError as error:
        raise BrevPixiError(
            f"pixi.toml must be inside the upload context. pixi.toml={resolved_toml}, "
            f"context={resolved_context}"
        ) from error
    if should_exclude(manifest_relative):
        raise BrevPixiError(
            f"pixi.toml is under an excluded upload path: {manifest_relative}"
        )

    return LocalProject(context=resolved_context, manifest_relative=manifest_relative)


def make_remote_project(remote_root: PurePosixPath, manifest_relative: Path) -> RemoteProject:
    """Create unique remote paths for one run."""

    run_id = dt.datetime.now(tz=dt.UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = remote_root / run_id
    return RemoteProject(
        run_dir=run_dir,
        project_dir=run_dir / "project",
        manifest_relative=PurePosixPath(manifest_relative.as_posix()),
    )


def upload_project(brev: BrevClient, project: LocalProject, remote: RemoteProject) -> None:
    """Archive the local project and extract it on the Brev instance."""

    with tempfile.TemporaryDirectory(prefix="brev-pixi-") as tmpdir:
        archive = Path(tmpdir) / "project.tar.gz"
        remote_archive = remote.run_dir / archive.name
        create_project_archive(project.context, archive)

        prepare_script = "\n".join(
            [
                "set -euo pipefail",
                f"mkdir -p {shlex.quote(str(remote.run_dir))}",
            ]
        )
        brev.exec(prepare_script, label="prepare remote upload directory")
        brev.copy_to(archive, remote_archive)

        extract_script = "\n".join(
            [
                "set -euo pipefail",
                f"mkdir -p {shlex.quote(str(remote.project_dir))}",
                f"tar -xzf {shlex.quote(str(remote_archive))} "
                f"-C {shlex.quote(str(remote.project_dir))}",
                "test -f "
                + shlex.quote(str(remote.project_dir / remote.manifest_relative.as_posix())),
            ]
        )
        brev.exec(extract_script, label="extract project archive")


def create_project_archive(context: Path, archive: Path) -> None:
    """Create a tarball containing the upload context contents."""

    with tarfile.open(archive, "w:gz") as tar:
        for root, dirnames, filenames in os.walk(context):
            root_path = Path(root)

            kept_dirnames = []
            for dirname in sorted(dirnames):
                path = root_path / dirname
                relative = path.relative_to(context)
                if should_exclude(relative):
                    continue
                kept_dirnames.append(dirname)
                tar.add(path, arcname=relative.as_posix(), recursive=False)
            dirnames[:] = kept_dirnames

            for filename in sorted(filenames):
                path = root_path / filename
                relative = path.relative_to(context)
                if should_exclude(relative):
                    continue
                tar.add(path, arcname=relative.as_posix(), recursive=False)


def should_exclude(relative: Path) -> bool:
    """Return whether a path should be skipped during upload."""

    return any(part in EXCLUDED_UPLOAD_PATTERNS for part in relative.parts)
