# cuda-brot 🌀⚡

A Mandelbrot renderer for your terminal, computed on your GPU — and a complete
example of building **and packaging** CUDA C++ software with
[Pixi](https://pixi.sh) and the `pixi-build-cmake` backend.

This is one of the two packages in the parent [`combined-build-example`](../../)
workspace — the CUDA C++ counterpart to the Python/cuTile
[`cutile-brot`](../cutile-brot) package, which builds the same fractal with the
`pixi-build-python` backend. One fractal, two backends: `nvcc` + CMake here,
cuTile + hatchling there.

Every pixel is one CUDA thread. The image is drawn with Unicode half-blocks
(`▀`) and 24-bit ANSI colors, so each terminal cell holds two pixels.

## Layout

```
cuda-brot/
├── pixi.toml            # package (pixi-build-cmake) definition
├── CMakeLists.txt       # CXX + CUDA project, installs the binary into bin/
└── src/
    └── main.cu          # the kernel + terminal renderer
```

## Run it

From the workspace root (`../../`):

```console
pixi run render     # the classic full-set view
pixi run seahorse   # deep zoom into seahorse valley
```

The first invocation builds the package from source (Pixi downloads the
toolchain, compiles, and caches the result). Subsequent runs are instant
unless sources change.

## Package it

```console
pixi publish
```

This produces a relocatable `cuda-brot-0.1.0-<build>.conda` package you can
upload to any conda channel, e.g. your own channel on
[prefix.dev](https://prefix.dev).

## Requirements

- Linux (x86-64) with an NVIDIA GPU and a CUDA 13 driver
- [Pixi](https://pixi.sh/latest/#installation)

No system CUDA toolkit installation — `nvcc`, the CUDA runtime, the C++
compiler, and CMake all come from conda-forge, managed by Pixi.
