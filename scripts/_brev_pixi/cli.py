"""Command-line interface for the Brev Pixi runner."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path, PurePosixPath

from .exceptions import BrevPixiError
from .models import DEFAULT_GPU, DEFAULT_INSTANCE, DEFAULT_REMOTE_ROOT, Options
from .project import resolve_project
from .runner import open_shell_on_brev, run_on_brev


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Run a Pixi workspace on Brev.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="Run a command with `pixi run` on Brev.",
        epilog="Command arguments go after `--`, for example: run pixi.toml -- python app.py",
    )
    add_common_args(run_parser)

    shell_parser = subparsers.add_parser(
        "shell",
        help="Prepare the workspace, open `brev shell`, then stop the instance on exit.",
    )
    add_common_args(shell_parser)

    parsed_argv = list(argv) if argv is not None else sys.argv[1:]
    command: tuple[str, ...] = ()
    if parsed_argv and parsed_argv[0] == "run":
        parsed_argv, command = split_run_command(parsed_argv)

    args = parser.parse_args(parsed_argv)
    if args.mode == "run":
        args.command = command
    return args


def split_run_command(argv: Sequence[str]) -> tuple[list[str], tuple[str, ...]]:
    """Split `run` parser arguments from the command after `--`."""

    try:
        separator = argv.index("--", 1)
    except ValueError:
        return list(argv), ()
    return list(argv[:separator]), tuple(argv[separator + 1 :])


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared by run and shell modes."""

    parser.add_argument("pixi_toml", type=Path, help="Path to the pixi.toml to run on Brev.")
    parser.add_argument(
        "--context",
        type=Path,
        help="Directory to upload. Defaults to the directory containing pixi.toml.",
    )
    parser.add_argument(
        "--instance",
        default=os.environ.get("BREV_INSTANCE", DEFAULT_INSTANCE),
        help=f"Brev instance name. Default: {DEFAULT_INSTANCE}",
    )
    parser.add_argument(
        "--gpu",
        "--gpu-type",
        dest="gpu",
        default=os.environ.get("BREV_GPU", DEFAULT_GPU),
        help=f"Brev GPU type used when creating a new instance. Default: {DEFAULT_GPU}",
    )
    parser.add_argument(
        "--remote-root",
        type=PurePosixPath,
        default=PurePosixPath(os.environ.get("BREV_REMOTE_ROOT", str(DEFAULT_REMOTE_ROOT))),
        help=f"Remote directory for uploaded runs. Default: {DEFAULT_REMOTE_ROOT}",
    )
    parser.add_argument(
        "--no-provision",
        action="store_true",
        help="Do not start/provision the Brev instance before running.",
    )
    parser.add_argument(
        "--keep-running",
        action="store_true",
        help="Do not stop the Brev instance after the command or shell exits.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print Brev commands without running them.",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Brev Pixi command-line interface."""

    args = parse_args(argv)
    try:
        options = Options(
            instance=args.instance,
            gpu=args.gpu,
            remote_root=args.remote_root,
            provision=not args.no_provision,
            keep_running=args.keep_running,
            dry_run=args.dry_run,
        )
        project = resolve_project(args.pixi_toml, args.context)
        if args.mode == "run":
            command = normalize_command(args.command)
            run_on_brev(project, options, command=command)
        elif args.mode == "shell":
            open_shell_on_brev(project, options)
    except BrevPixiError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except subprocess.CalledProcessError as error:
        print(f"error: command failed with exit code {error.returncode}", file=sys.stderr)
        return error.returncode if error.returncode > 0 else 1
    return 0


def normalize_command(raw_command: Sequence[str]) -> tuple[str, ...]:
    """Normalize the command parsed after the `--` separator."""

    command = tuple(raw_command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise BrevPixiError("run mode requires a command after `--`, for example: -- python app.py")
    return command
