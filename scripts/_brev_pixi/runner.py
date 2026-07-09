"""High-level Brev Pixi workflows."""

from __future__ import annotations

import shlex
import subprocess
import sys
from collections.abc import Generator, Sequence
from contextlib import contextmanager

from .brev import BrevClient
from .exceptions import BrevPixiError
from .models import LocalProject, Options, RemoteProject
from .project import make_remote_project, upload_project
from .remote import bootstrap_script, create_enter_script, remote_project_script


def run_on_brev(project: LocalProject, options: Options, *, command: Sequence[str]) -> None:
    """Upload a Pixi workspace to Brev and run a command through Pixi."""

    brev = BrevClient(options.instance, dry_run=options.dry_run)
    remote = make_remote_project(options.remote_root, project.manifest_relative)
    with stop_instance_afterwards(brev, keep_running=options.keep_running):
        prepare_remote_project(brev, project, remote, options)
        remote_command = "pixi run --manifest-path " + shlex.quote(remote.manifest_name)
        remote_command += " -- " + shlex.join(command)
        brev.exec(remote_project_script(remote, remote_command), label="run Pixi command")


def open_shell_on_brev(project: LocalProject, options: Options) -> None:
    """Upload a Pixi workspace to Brev and open an interactive Brev shell."""

    brev = BrevClient(options.instance, dry_run=options.dry_run)
    remote = make_remote_project(options.remote_root, project.manifest_relative)
    with stop_instance_afterwards(brev, keep_running=options.keep_running):
        prepare_remote_project(brev, project, remote, options)
        enter_script = remote.run_dir / "enter.sh"
        brev.exec(create_enter_script(remote, enter_script), label="create interactive helper")
        print("\nInteractive Brev shell is ready.")
        print("Run this command inside the shell to enter the Pixi environment:")
        print(f"\n  source {enter_script}\n")
        brev.shell()


@contextmanager
def stop_instance_afterwards(
    brev: BrevClient, *, keep_running: bool
) -> Generator[None, None, None]:
    """Stop the Brev instance after use without masking run failures."""

    caught_error: BaseException | None = None
    try:
        yield
    except BaseException as error:
        caught_error = error
        raise
    finally:
        if keep_running:
            print(f"\nLeaving Brev instance running: {brev.instance}")
        else:
            try:
                brev.stop()
            except (subprocess.CalledProcessError, BrevPixiError, OSError):
                if caught_error is None:
                    raise
                print(
                    f"warning: failed to stop Brev instance {brev.instance!r}; "
                    "the original error will still be reported",
                    file=sys.stderr,
                )
            else:
                if brev.dry_run:
                    print(f"\nDry run: did not stop Brev instance: {brev.instance}")
                else:
                    print(f"\nStopped Brev instance: {brev.instance}")


def prepare_remote_project(
    brev: BrevClient, project: LocalProject, remote: RemoteProject, options: Options
) -> None:
    """Provision, upload, bootstrap, and install a Pixi workspace on Brev."""

    brev.ensure_available()
    if options.provision:
        brev.provision(options.gpu)
    upload_project(brev, project, remote)
    brev.exec(bootstrap_script(), label="bootstrap Pixi")
    brev.exec(
        remote_project_script(
            remote,
            "pixi install --manifest-path " + shlex.quote(remote.manifest_name),
        ),
        label="install Pixi environment",
    )
