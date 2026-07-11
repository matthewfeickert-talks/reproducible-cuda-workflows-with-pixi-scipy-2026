# cutile-brot

A minimal [cuTile](https://github.com/NVIDIA/cutile-python) example, packaged as
a distributable Python package and built from source with
[Pixi Build](https://pixi.prefix.dev/latest/build/python/).

This is one of the two packages in the parent
[`combined-build-example`](../../) workspace — the Python/cuTile counterpart to
the CUDA C++ [`cuda-brot`](../cuda-brot) package. One fractal, two backends.

## Layout

```
cutile-brot/
├── pixi.toml                     # package (build) definition
├── pyproject.toml                # Python package metadata (hatchling)
└── src/
    └── cutile_brot/
        ├── __init__.py
        └── mandelbrot.py         # GPU Mandelbrot rendered in your terminal
```

## Run it

From the workspace root (`../../`):

```bash
pixi run mandelbrot   # renders a Mandelbrot set in the terminal
```

`cutile-brot` accepts `--center-re`, `--center-im`, `--span`, and `--max-iter`
to explore the fractal, e.g.:

```bash
pixi run mandelbrot --center-re -0.743 --center-im 0.131 --span 0.02 --max-iter 512
```

## How the Mandelbrot maps onto cuTile

The classic CUDA kernel runs a per-thread `while` loop until each pixel
escapes. The tile model works on a whole tile of pixels at once, so the loop
becomes a **fixed-trip `for` loop over `max_iter`** and a boolean mask
(`ct.where`) freezes each pixel the moment it escapes — keeping the
first-escape magnitude for band-free smooth coloring. The complex-plane
coordinates are built on the host with NumPy and uploaded, so the kernel simply
`ct.load`s two input tiles. Palette and ANSI half-block rendering happen on the
host.

## Use it as a library

```python
from cutile_brot import mandelbrot_compute

rgb, kernel_ms = mandelbrot_compute(200, 112, -0.6, 0.0, 3.2, 256)
```

## Requirements

An NVIDIA GPU with a CUDA 13 driver and a 24-bit-color terminal.
