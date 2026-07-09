#!/usr/bin/env python3
"""CLI entrypoint for running local Pixi workspaces on NVIDIA Brev."""

from __future__ import annotations

from _brev_pixi.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
