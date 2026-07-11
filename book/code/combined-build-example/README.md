# combined-build-example 🌀⚡🐍

One Pixi workspace that builds **two** source packages — the same Mandelbrot
fractal through two different [Pixi Build](https://pixi.prefix.dev/latest/build/)
backends:

- [`cuda-brot`](src/cuda-brot) — CUDA C++ via `pixi-build-cmake` (`nvcc` + CMake)
- [`cutile-brot`](src/cutile-brot) — Python/cuTile via `pixi-build-python` (hatchling)

Both packages live under `src/` and are added to the top-level workspace as
source dependencies, so `pixi run` (re)builds whichever one a task needs.

## Layout

```
combined-build-example/
├── pixi.toml                    # the workspace: dependencies + tasks
└── src/
    ├── cuda-brot/               # pixi-build-cmake package
    │   ├── pixi.toml            #   package manifest
    │   ├── CMakeLists.txt
    │   └── src/main.cu
    └── cutile-brot/             # pixi-build-python package
        ├── pixi.toml            #   package manifest (build backend)
        ├── pyproject.toml       #   name / version / deps (hatchling)
        └── src/cutile_brot/
            └── mandelbrot.py
```

## Run the examples

```console
pixi run cuda-brot        # cuda-brot:   classic full-set view (CUDA C++)
pixi run cuda-seahorse    # cuda-brot:   deep zoom into seahorse valley
pixi run cutile-brot      # cutile-brot: Mandelbrot rendered in the terminal
pixi run cutile-seahorse  # cutile-brot: deep zoom into seahorse valley
```

The first invocation of each builds the relevant package from source (Pixi
downloads the toolchain, compiles, and caches). Subsequent runs are instant
unless sources change.

## Requirements

Linux (x86-64) with an NVIDIA GPU and a CUDA 13 driver — the workspace targets
`linux-64` with `cuda = "13"`. No system CUDA toolkit needed: `nvcc`, the CUDA
runtime, the C++ compiler, CMake and the Python stack all come from
conda-forge, managed by Pixi.

To resolve/build on a machine without an NVIDIA driver (e.g. CI):

```console
CONDA_OVERRIDE_CUDA=13 pixi install
```
