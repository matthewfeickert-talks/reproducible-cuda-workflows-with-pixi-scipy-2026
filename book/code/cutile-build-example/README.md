# cutile-example

Minimal [cuTile](https://github.com/NVIDIA/cutile-python) examples, packaged as
a distributable Python package and built from source with
[Pixi Build](https://pixi.prefix.dev/latest/build/python/).

## Layout

```
cutile-build-example/
├── pixi.toml                     # Pixi workspace + package (build) definition
├── pyproject.toml                # Python package metadata (hatchling)
└── src/
    └── cutile_example/
        ├── __init__.py
        ├── vector_add.py         # elementwise add: cuTile kernel + run() + main()
        └── mandelbrot.py         # GPU Mandelbrot rendered in your terminal
```

## Run the examples

The package builds from source (Pixi Build is a preview feature) and installs
two console scripts:

```bash
pixi run start        # vector add, prints "✓ vector_add passed!"
pixi run mandelbrot   # renders a Mandelbrot set in the terminal
```

`cutile-mandelbrot` accepts `--center-re`, `--center-im`, `--span`, and
`--max-iter` to explore the fractal, e.g.:

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
`ct.load`s two input tiles just like the vector-add sample. Palette and ANSI
half-block rendering happen on the host.

## Use it as a library

```python
from cutile_example import run, mandelbrot_compute

run(vector_size=2**12, tile_size=2**4)
rgb, kernel_ms = mandelbrot_compute(200, 112, -0.6, 0.0, 3.2, 256)
```

## Requirements

An NVIDIA GPU with a CUDA 13 driver and a 24-bit-color terminal — the workspace
targets the `linux-64-cuda` platform.
