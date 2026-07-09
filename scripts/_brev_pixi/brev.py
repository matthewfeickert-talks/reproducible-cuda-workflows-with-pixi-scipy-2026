"""Thin wrapper around the Brev CLI."""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from .exceptions import BrevPixiError


class BrevClient:
    """Typed wrapper for the small subset of the Brev CLI we use."""

    def __init__(self, instance: str, *, dry_run: bool = False) -> None:
        self.instance = instance
        self.dry_run = dry_run

    def ensure_available(self) -> None:
        """Verify that the Brev CLI is available unless this is a dry run."""

        if self.dry_run:
            return
        if shutil.which("brev") is None:
            raise BrevPixiError(
                "Brev CLI not found. Install it with `pixi global install brev` "
                "or run this through the `brev` Pixi feature."
            )

    def provision(self, gpu_type: str) -> None:
        """Start an existing instance or create one with the requested GPU type."""

        if self.dry_run:
            self._run(["brev", "ls", "--json"], label="list Brev instances")
            self._run(
                ["brev", "create", self.instance, "--type", gpu_type],
                label="create Brev instance if missing",
            )
            self._run(
                ["brev", "start", self.instance],
                label="start Brev instance if stopped",
            )
            return

        status = self.instance_status()
        if status is None:
            self._run(
                ["brev", "create", self.instance, "--type", gpu_type],
                label="create Brev instance",
            )
            return

        if status == "STOPPING":
            self.wait_for_status({"STOPPED", "MISSING"}, label="wait for instance to stop")
            status = self.instance_status()

        if status in {None, "MISSING"}:
            self._run(
                ["brev", "create", self.instance, "--type", gpu_type],
                label="create Brev instance",
            )
            return

        if status == "STOPPED":
            self._run(["brev", "start", self.instance], label="start Brev instance")
            return

        print(
            f"\n==> use existing Brev instance ({self.instance}: {status})",
            flush=True,
        )

    def stop(self, *, attempts: int = 12, delay_seconds: int = 15) -> None:
        """Stop the Brev instance, retrying while new instances settle."""

        if self.dry_run:
            self._run(["brev", "stop", self.instance], label="stop Brev instance")
            return

        status = self.instance_status()
        if status in {None, "MISSING", "STOPPED"}:
            print(f"\n==> Brev instance already stopped: {self.instance}", flush=True)
            return
        if status == "STOPPING":
            self.wait_for_status({"STOPPED", "MISSING"}, label="wait for instance to stop")
            return

        for attempt in range(1, attempts + 1):
            try:
                self._run(["brev", "stop", self.instance], label="stop Brev instance")
                return
            except subprocess.CalledProcessError:
                status = self.instance_status()
                if status in {None, "MISSING", "STOPPED"}:
                    print(
                        f"\n==> Brev instance already stopped: {self.instance}",
                        flush=True,
                    )
                    return
                if attempt == attempts:
                    raise
                print(
                    "Stop failed; waiting before retry "
                    f"{attempt + 1}/{attempts}...",
                    flush=True,
                )
                time.sleep(delay_seconds)

    def copy_to(self, local_path: Path, remote_path: PurePosixPath) -> None:
        """Copy a local file or directory to the Brev instance."""

        local_arg = f"{local_path}/" if local_path.is_dir() else str(local_path)
        self._run(
            ["brev", "copy", local_arg, f"{self.instance}:{remote_path}"],
            label="copy project to Brev",
        )

    def exec(self, script: str, *, label: str) -> None:
        """Execute a bash script on the Brev instance."""

        command = "bash -lc " + shlex.quote(script)
        self._run(["brev", "exec", self.instance, command], label=label)

    def shell(self) -> None:
        """Open an interactive Brev shell."""

        args = ["brev", "shell", self.instance]
        print("\n==> open Brev shell", flush=True)
        print(f"$ {shlex.join(args)}", flush=True)
        if self.dry_run:
            return
        result = subprocess.run(args, check=False)
        if result.returncode:
            print(
                f"Brev shell exited with status {result.returncode}.",
                file=sys.stderr,
                flush=True,
            )

    def _run(self, args: Sequence[str], *, label: str) -> None:
        print(f"\n==> {label}", flush=True)
        print(f"$ {shlex.join(args)}", flush=True)
        if self.dry_run:
            return
        subprocess.run(args, check=True)

    def instance_status(self) -> str | None:
        """Return the current Brev status for the configured instance."""

        instances = self._capture(["brev", "ls", "--json"], label="list Brev instances")
        return parse_instance_status(instances, self.instance)

    def wait_for_status(
        self,
        statuses: set[str],
        *,
        label: str,
        attempts: int = 30,
        delay_seconds: int = 10,
    ) -> None:
        """Wait until the instance status is one of the requested statuses."""

        print(f"\n==> {label}", flush=True)
        for attempt in range(1, attempts + 1):
            status = self.instance_status() or "MISSING"
            if status in statuses:
                print(f"Instance {self.instance} reached status {status}.", flush=True)
                return
            if attempt == attempts:
                raise BrevPixiError(
                    f"Timed out waiting for {self.instance} to reach one of "
                    f"{sorted(statuses)}; last status was {status}."
                )
            print(
                f"Instance {self.instance} is {status}; waiting "
                f"{delay_seconds}s ({attempt}/{attempts})...",
                flush=True,
            )
            time.sleep(delay_seconds)

    def _capture(self, args: Sequence[str], *, label: str) -> str:
        print(f"\n==> {label}", flush=True)
        print(f"$ {shlex.join(args)}", flush=True)
        if self.dry_run:
            return ""
        result = subprocess.run(args, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout, end="", flush=True)
        if result.stderr:
            print(result.stderr, end="", flush=True)
        return result.stdout


def parse_instance_status(brev_ls_output: str, instance: str) -> str | None:
    """Parse the status for an instance from `brev ls --json` output."""

    if not brev_ls_output.strip():
        return None

    try:
        payload = json.loads(brev_ls_output)
    except json.JSONDecodeError as error:
        raise BrevPixiError("Failed to parse `brev ls --json` output.") from error

    for item in iter_instance_records(payload):
        if item.get("name") == instance:
            status = item.get("status")
            return str(status).upper() if status is not None else None
    return None


def iter_instance_records(payload: Any) -> Sequence[Mapping[str, Any]]:
    """Return instance records from known Brev JSON list shapes."""

    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, Mapping)]
    if isinstance(payload, Mapping):
        records: list[Mapping[str, Any]] = []
        for key in ("workspaces", "instances", "nodes"):
            value = payload.get(key)
            if isinstance(value, list):
                records.extend(item for item in value if isinstance(item, Mapping))
        return records
    return []
