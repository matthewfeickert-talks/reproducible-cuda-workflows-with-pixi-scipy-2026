# Pixi Build: exercises

In these exercises you will build the [`combined-build-example`](https://github.com/matthewfeickert-talks/reproducible-cuda-workflows-with-pixi-scipy-2026/tree/main/book/code/combined-build-example) from scratch: one Pixi workspace that builds **two** conda packages — the same Mandelbrot fractal through two [Pixi Build](https://pixi.prefix.dev/latest/build/backends/) backends.
See the [Pixi Build introduction chapter](pixi-build-cuda.md) for the concepts, and the [Pixi Build docs](https://pixi.prefix.dev/latest/build/getting_started/) for the full reference.

You will write the **manifests** yourself.
The application source (the cuTile kernel and the CUDA C++ kernel) is *given* — the point here is packaging, not writing GPU code.

::: {warning} These exercises need a GPU
Building the CUDA C++ package needs the CUDA toolchain (Pixi installs it for you from conda-forge), and *running* either package needs an NVIDIA GPU.

Work through this chapter on your [Brev instance](setup.md#prepare-brev-instance).
:::

:::::{tip} Exercise 1: Create the build workspace
1. Create a new Pixi workspace named `combined-build` in your repository.
2. Enable the Pixi Build [preview feature](https://pixi.prefix.dev/latest/reference/pixi_manifest/#preview-features). (edit `pixi.toml` manually)
3. Give it a single [rich platform](https://pixi.prefix.dev/latest/workspace/multi_platform_configuration/#declaring-virtual-packages-per-platform): `linux-64` that declares a CUDA 13 driver.
4. Point its channel at `https://prefix.dev/conda-forge`.

::::{hint} Solution
:class: dropdown
Use [`pixi init`](https://pixi.prefix.dev/latest/reference/cli/pixi/init/) to create the workspace:
```bash
# 1
pixi init combined-build
cd combined-build
```
Then edit `pixi.toml` so the `[workspace]` table enables the preview feature and declares the CUDA-rich platform:
:::{code} toml
:filename: pixi.toml
:linenos:
:emphasize-lines: 3,5,6
[workspace]
authors = ["Jane Doe <jane.doe@example.com>"]
channels = ["https://prefix.dev/conda-forge"]
name = "combined-build"
platforms = [{ platform = "linux-64", cuda = "13" }]
preview = ["pixi-build"]
version = "0.1.0"

[tasks]

[dependencies]
:::
::::
:::::

:::::{tip} Exercise 2: Package the cuTile app (`pixi-build-python`)
Build a Python/cuTile package `cutile-brot` and add it to the workspace as a source dependency.

1. Create the package source tree and drop in the given source files (`__init__.py` and `mandelbrot.py`, below).
2. Write a `pyproject.toml` for the package, with a `cutile-brot` entry-point script and the runtime dependencies `numpy`, `cupy`, and `cuda-tile`.
3. Make sure you add the `cutile-brot` entry-point script to the `[project.scripts]` table in `pyproject.toml`.
4. Write the package manifest (`pixi.toml`) that uses the [`pixi-build-python`](https://pixi.prefix.dev/latest/build/backends/pixi-build-python/) backend and maps the `pyproject.toml` dependencies onto conda packages.
5. Add `cutile-brot` to the workspace as a source dependency.
6. Add a `cutile-brot` task that runs the `cutile-brot` command.
7. Run it and watch the fractal render on the GPU.

::::{important} Given source: `src/cutile-brot/src/cutile_brot/__init__.py`
:class: dropdown
:::{code} python
:filename: src/cutile-brot/src/cutile_brot/__init__.py
"""A minimal, distributable cuTile Mandelbrot example."""

from cutile_brot.mandelbrot import compute as mandelbrot_compute

__version__ = "0.1.0"

__all__ = ["mandelbrot_compute", "__version__"]
:::
::::

::::{important} Given source: `src/cutile-brot/src/cutile_brot/mandelbrot.py`
:class: dropdown
:::{code} python
:filename: src/cutile-brot/src/cutile_brot/mandelbrot.py
# SPDX-License-Identifier: Apache-2.0

"""cutile-brot, the cuTile edition.

A Mandelbrot renderer for your terminal, computed on the GPU with cuTile.
"""

import argparse
import shutil

import cupy as cp
import numpy as np
import cuda.tile as ct

TILE_M = 16  # tile height (pixel rows per block)
TILE_N = 16  # tile width  (pixel cols per block)


@ct.kernel
def mandelbrot(c_re, c_im, iters, mag2, max_iter: ct.Constant[int]):
    """Escape-time Mandelbrot over one 2D tile of the complex plane."""
    bi = ct.bid(0)  # row block
    bj = ct.bid(1)  # col block

    cr = ct.load(c_re, index=(bi, bj), shape=(TILE_M, TILE_N))
    ci = ct.load(c_im, index=(bi, bj), shape=(TILE_M, TILE_N))

    re = ct.full((TILE_M, TILE_N), 0.0, dtype=np.float64)
    im = ct.full((TILE_M, TILE_N), 0.0, dtype=np.float64)
    it = ct.full((TILE_M, TILE_N), 0.0, dtype=np.float64)

    # Fixed trip count; the `inside` mask does the early-out per pixel.
    for _ in range(max_iter):
        inside = (re * re + im * im) <= 4.0
        re_next = re * re - im * im + cr
        im_next = 2.0 * re * im + ci
        re = ct.where(inside, re_next, re)
        im = ct.where(inside, im_next, im)
        it = it + ct.where(inside, 1.0, 0.0)

    ct.store(iters, index=(bi, bj), tile=it)
    ct.store(mag2, index=(bi, bj), tile=re * re + im * im)


def compute(width, height, center_re, center_im, span, max_iter):
    """Run the kernel and return an ``(height, width, 3)`` uint8 RGB image."""
    step = span / width

    # Pad the grid up to whole tiles so no block reads/writes out of bounds.
    hp = -(-height // TILE_M) * TILE_M
    wp = -(-width // TILE_N) * TILE_N

    xs = center_re + (np.arange(wp) - width / 2.0) * step
    ys = center_im - (np.arange(hp) - height / 2.0) * step
    grid_re, grid_im = np.meshgrid(xs, ys)

    c_re = cp.asarray(grid_re)
    c_im = cp.asarray(grid_im)
    iters = cp.zeros((hp, wp), dtype=cp.float64)
    mag2 = cp.zeros((hp, wp), dtype=cp.float64)

    grid = (ct.cdiv(hp, TILE_M), ct.cdiv(wp, TILE_N), 1)

    start, stop = cp.cuda.Event(), cp.cuda.Event()
    start.record()
    ct.launch(
        cp.cuda.get_current_stream(),
        grid,
        mandelbrot,
        (c_re, c_im, iters, mag2, max_iter),
    )
    stop.record()
    stop.synchronize()
    ms = cp.cuda.get_elapsed_time(start, stop)

    it = cp.asnumpy(iters)[:height, :width]
    m2 = cp.asnumpy(mag2)[:height, :width]
    return _colorize(it, m2, max_iter), ms


def _colorize(it, mag2, max_iter):
    """Smooth escape count -> Inigo Quilez cosine palette -> uint8 RGB."""
    escaped = it < max_iter
    with np.errstate(divide="ignore", invalid="ignore"):
        log_zn = 0.5 * np.log(mag2)
        nu = it + 1.0 - np.log2(np.maximum(log_zn, 1e-12))
    t = np.where(escaped, nu / max_iter, 0.0)

    r = 0.5 + 0.5 * np.cos(3.0 + 12.0 * t)
    g = 0.5 + 0.5 * np.cos(3.6 + 12.0 * t)
    b = 0.5 + 0.5 * np.cos(4.2 + 12.0 * t)
    rgb = (255.0 * np.stack([r, g, b], axis=-1)).astype(np.uint8)
    rgb[~escaped] = 0  # points in the set are black
    return rgb


def render(rgb):
    """Draw an RGB image with ▀ half-blocks: top pixel fg, bottom pixel bg."""
    height, width, _ = rgb.shape
    lines = []
    for y in range(0, height - 1, 2):
        row = []
        for x in range(width):
            tr, tg, tb = rgb[y, x]
            br, bg, bb = rgb[y + 1, x]
            row.append(f"\x1b[38;2;{tr};{tg};{tb}m\x1b[48;2;{br};{bg};{bb}m▀")
        row.append("\x1b[0m")
        lines.append("".join(row))
    return "\n".join(lines)


def _terminal_size():
    cols, rows = shutil.get_terminal_size(fallback=(100, 30))
    width = cols
    height = 2 * max(rows - 3, 1)  # two pixels per row, leave a status line
    return width, height


def main():
    parser = argparse.ArgumentParser(description="GPU Mandelbrot in your terminal (cuTile).")
    parser.add_argument("--center-re", type=float, default=-0.6)
    parser.add_argument("--center-im", type=float, default=0.0)
    parser.add_argument("--span", type=float, default=3.2)
    parser.add_argument("--max-iter", type=int, default=256)
    args = parser.parse_args()

    width, height = _terminal_size()
    rgb, ms = compute(width, height, args.center_re, args.center_im, args.span, args.max_iter)

    name = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["name"].decode()
    print(render(rgb))
    print(f"{name} | {width}x{height} px | {args.max_iter} iterations | kernel: {ms:.2f} ms")


if __name__ == "__main__":
    main()
:::
::::

::::{hint} Solution
:class: dropdown
```bash
# 1: create the source tree, then paste the given source files into place
mkdir -p src/cutile-brot/src/cutile_brot
# (create src/cutile-brot/src/cutile_brot/__init__.py and mandelbrot.py
#  from the "Given source" dropdowns above)
```
2. Write the package's `pyproject.toml`:
:::{code} toml
:filename: src/cutile-brot/pyproject.toml
:linenos:
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "cutile-brot"
version = "0.1.0"
description = "A minimal cuTile Mandelbrot example, packaged for distribution."
requires-python = ">=3.11"
dependencies = [
    "numpy>=2.5.1,<3",
    "cupy>=14.1.1,<15",
    "cuda-tile>=1.4.0,<2",
]

[project.scripts]
cutile-brot = "cutile_brot.mandelbrot:main"

[tool.hatch.build.targets.wheel]
packages = ["src/cutile_brot"]
:::
3. Write the package manifest. It has **no** `[workspace]` table — only `[package.build]`:
:::{code} toml
:filename: src/cutile-brot/pixi.toml
:linenos:
[package.build]
backend = { name = "pixi-build-python", version = "0.*" }
# Map the pyproject.toml dependencies onto conda-forge packages.
config.ignore-pypi-mapping = false
:::
4. Add the source dependency by editing the workspace `pixi.toml` directly — source `path` dependencies are written into the manifest by hand:
:::{code} toml
:filename: pixi.toml
[dependencies]
cutile-brot = { path = "src/cutile-brot" }
:::
Add the task with [`pixi task add`](https://pixi.prefix.dev/latest/reference/cli/pixi/task/add/) and run it with [`pixi run`](https://pixi.prefix.dev/latest/reference/cli/pixi/run/):
```bash
# 5
pixi task add cutile-brot cutile-brot --description "Render a GPU Mandelbrot set in the terminal (cuTile)"
# 6 (first run builds the package, then renders)
pixi run cutile-brot
```
Resulting workspace `pixi.toml`:
:::{code} toml
:filename: pixi.toml
:linenos:
:emphasize-lines: 10,13
[workspace]
authors = ["Jane Doe <jane.doe@example.com>"]
channels = ["https://prefix.dev/conda-forge"]
name = "combined-build"
platforms = [{ platform = "linux-64", cuda = "13" }]
preview = ["pixi-build"]
version = "0.1.0"

[tasks]
cutile-brot = { cmd = "cutile-brot", description = "Render a GPU Mandelbrot set in the terminal (cuTile)" }

[dependencies]
cutile-brot = { path = "src/cutile-brot" }
:::
::::
:::::

:::::{tip} Exercise 3: Package the CUDA C++ app (`pixi-build-cmake`)
Now build the CUDA C++ counterpart `cuda-brot` with the CMake backend, in the *same* workspace.

1. Create the package source tree and drop in the given `main.cu` and `CMakeLists.txt` (below).
2. Write the package manifest using the [`pixi-build-cmake`](https://pixi.prefix.dev/latest/build/backends/pixi-build-cmake/) backend.
3. Request a `cxx` **and** a `cuda` compiler, so that the right compilers for your device are used.
4. Add `*.cu` files as [`extra-input-globs`](https://pixi.prefix.dev/latest/build/backends/pixi-build-cmake/#extra-input-globs), so that the package gets rebuild when the CUDA sources change
5. Add `cuda-cudart-dev` as a `package.host-dependency`.
6. Pin the CUDA compiler by adding a [`[workspace.build-variants]`](https://pixi.prefix.dev/latest/build/variants/) table to the workspace manifest (`combined-build/pixi.toml`).
7. Add `cuda-brot` to the workspace as a source dependency.
8. Add a `cuda-brot` task and run it.

::::{important} Given source: `src/cuda-brot/CMakeLists.txt`
:class: dropdown
:::{code} cmake
:filename: src/cuda-brot/CMakeLists.txt
cmake_minimum_required(VERSION 3.24...4.3)

# Build for all major GPU architectures by default so the package runs on any
# reasonably modern NVIDIA GPU.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES all-major)
endif()

project(cuda-brot LANGUAGES CXX CUDA)

add_executable(cuda-brot src/main.cu)
target_compile_features(cuda-brot PRIVATE cxx_std_17 cuda_std_17)

# This install rule tells the backend which artifact to package.
install(TARGETS cuda-brot RUNTIME DESTINATION bin)
:::
::::

::::{important} Given source: `src/cuda-brot/src/main.cu`
:class: dropdown
:::{code} cpp
:filename: src/cuda-brot/src/main.cu
// cuda-brot — a Mandelbrot renderer for your terminal, computed on the GPU.
// Every pixel is one CUDA thread, drawn with Unicode half-blocks (▀).

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <sys/ioctl.h>
#include <unistd.h>

#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err_ = (call);                                        \
        if (err_ != cudaSuccess) {                                        \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,   \
                         __LINE__, cudaGetErrorString(err_));             \
            std::exit(1);                                                 \
        }                                                                 \
    } while (0)

// A smooth cosine palette (Inigo Quilez school of shading).
__device__ uchar3 palette(float t)
{
    float r = 0.5f + 0.5f * cosf(3.0f + 12.0f * t);
    float g = 0.5f + 0.5f * cosf(3.6f + 12.0f * t);
    float b = 0.5f + 0.5f * cosf(4.2f + 12.0f * t);
    return make_uchar3(static_cast<unsigned char>(255.0f * r),
                       static_cast<unsigned char>(255.0f * g),
                       static_cast<unsigned char>(255.0f * b));
}

__global__ void mandelbrot(uchar3 *out, int width, int height, double center_re,
                           double center_im, double step, int max_iter)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    double c_re = center_re + (x - width / 2.0) * step;
    double c_im = center_im - (y - height / 2.0) * step;

    double re = 0.0, im = 0.0;
    int it = 0;
    while (re * re + im * im <= 4.0 && it < max_iter) {
        double tmp = re * re - im * im + c_re;
        im = 2.0 * re * im + c_im;
        re = tmp;
        ++it;
    }

    uchar3 color = make_uchar3(0, 0, 0);
    if (it < max_iter) {
        float log_zn = logf(static_cast<float>(re * re + im * im)) * 0.5f;
        float nu = it + 1.0f - log2f(log_zn);
        color = palette(nu / max_iter);
    }
    out[y * width + x] = color;
}

static void terminal_size(int *width, int *height)
{
    winsize ws{};
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0 && ws.ws_row > 2) {
        *width = ws.ws_col;
        *height = 2 * (ws.ws_row - 3);  // half-blocks: two pixels per row
    } else {
        *width = 100;
        *height = 56;
    }
}

int main(int argc, char **argv)
{
    // Usage: cuda-brot [center_re center_im span [max_iter]]
    double center_re = -0.6, center_im = 0.0, span = 3.2;
    int max_iter = 256;
    if (argc >= 4) {
        center_re = std::atof(argv[1]);
        center_im = std::atof(argv[2]);
        span = std::atof(argv[3]);
    }
    if (argc >= 5) {
        max_iter = std::atoi(argv[4]);
    }

    int width = 0, height = 0;
    terminal_size(&width, &height);
    double step = span / width;

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    uchar3 *d_pixels = nullptr;
    CUDA_CHECK(cudaMalloc(&d_pixels, sizeof(uchar3) * width * height));

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    mandelbrot<<<grid, block>>>(d_pixels, width, height, center_re, center_im, step, max_iter);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    std::vector<uchar3> pixels(static_cast<size_t>(width) * height);
    CUDA_CHECK(cudaMemcpy(pixels.data(), d_pixels, sizeof(uchar3) * pixels.size(),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_pixels));

    std::string frame;
    frame.reserve(static_cast<size_t>(width) * height * 20);
    char buf[64];
    for (int y = 0; y + 1 < height; y += 2) {
        for (int x = 0; x < width; ++x) {
            uchar3 top = pixels[static_cast<size_t>(y) * width + x];
            uchar3 bottom = pixels[static_cast<size_t>(y + 1) * width + x];
            std::snprintf(buf, sizeof(buf), "\x1b[38;2;%d;%d;%dm\x1b[48;2;%d;%d;%dm",
                          top.x, top.y, top.z, bottom.x, bottom.y, bottom.z);
            frame += buf;
            frame += "▀";
        }
        frame += "\x1b[0m\n";
    }
    std::fputs(frame.c_str(), stdout);

    std::printf("%s | %dx%d px | %d iterations | kernel: %.2f ms\n", prop.name, width,
                height, max_iter, ms);
    return 0;
}
:::
::::

::::{hint} Solution
:class: dropdown
```bash
# 1: create the source tree, then paste the given source files into place
mkdir -p src/cuda-brot/src
# (create src/cuda-brot/CMakeLists.txt and src/cuda-brot/src/main.cu
#  from the "Given source" dropdowns above)
```
2. Write the package manifest. Each CUDA-specific line is annotated:
:::{code} toml
:filename: src/cuda-brot/pixi.toml
:linenos:
[package]
name = "cuda-brot"
version = "0.1.0"

[package.build]
backend = { name = "pixi-build-cmake", version = "0.*" }
# Request a C++ and a CUDA compiler from the backend.
config.compilers = ["cxx", "cuda"]
# `.cu` files aren't a default build input, so declare them.
config.extra-input-globs = ["*.cu"]

[package.host-dependencies]
# Headers + libcudart to compile/link against. Its run-export makes the built
# package depend on `cuda-cudart` at runtime automatically.
cuda-cudart-dev = "*"
:::
3. Pin the CUDA compiler in the **workspace** manifest. On conda-forge the CUDA compiler ships in `cuda-nvcc`:
:::{code} toml
:filename: pixi.toml
[workspace.build-variants]
cuda_compiler = ["cuda-nvcc"]
cuda_compiler_version = ["13.1"]
:::
4. Add the source dependency by editing the workspace `pixi.toml` directly:
:::{code} toml
:filename: pixi.toml
[dependencies]
cuda-brot = { path = "src/cuda-brot" }
:::
```bash
# 5 (first run compiles with nvcc — this can take a little while)
pixi task add cuda-brot cuda-brot --description "Render the classic Mandelbrot set (CUDA C++)"
pixi run cuda-brot
```
Resulting workspace `pixi.toml`:
:::{code} toml
:filename: pixi.toml
:linenos:
:emphasize-lines: 11,15,17-19
[workspace]
authors = ["Jane Doe <jane.doe@example.com>"]
channels = ["https://prefix.dev/conda-forge"]
name = "combined-build"
platforms = [{ platform = "linux-64", cuda = "13" }]
preview = ["pixi-build"]
version = "0.1.0"

[tasks]
cutile-brot = { cmd = "cutile-brot", description = "Render a GPU Mandelbrot set in the terminal (cuTile)" }
cuda-brot = { cmd = "cuda-brot", description = "Render the classic Mandelbrot set (CUDA C++)" }

[dependencies]
cutile-brot = { path = "src/cutile-brot" }
cuda-brot = { path = "src/cuda-brot" }

[workspace.build-variants]
cuda_compiler = ["cuda-nvcc"]
cuda_compiler_version = ["13.1"]
:::
::::
:::::

:::::{tip} Exercise 4: Build and install your packages
So far the packages only exist inside the workspace environment. Now turn them into real conda packages and install them globally, mimicking what a user gets after you publish.

1. Build each package into a `.conda` artifact with [`pixi publish`](https://pixi.prefix.dev/latest/reference/cli/pixi/publish/).
2. Install each package globally straight from its source directory with [`pixi global install`](https://pixi.prefix.dev/latest/reference/cli/pixi/global/install/).
3. Verify: from *outside* the workspace (e.g. your home directory), run `cutile-brot` and `cuda-brot` directly — no `pixi run`.
4. **Bonus:** add "seahorse valley" deep-zoom tasks that pass coordinates to each renderer. With these parameters (`-0.743643887 0.131825904 0.00008 1500`), the Mandelbrot set renders a seahorse-shaped valley in the fractal.

::::{hint} Solution
:class: dropdown
```bash
# 1: produce relocatable .conda artifacts
pixi publish --path src/cutile-brot
pixi publish --path src/cuda-brot

# 2: install globally from the local package directories
pixi global install --path src/cutile-brot
pixi global install --path src/cuda-brot

# 3: now the commands work from anywhere
cd ~
cutile-brot
cuda-brot
cd combined-build

# 4: bonus zoom tasks
pixi task add cutile-seahorse "cutile-brot --center-re -0.743643887 --center-im 0.131825904 --span 0.00008 --max-iter 1500"
pixi task add cuda-seahorse "cuda-brot -0.743643887 0.131825904 0.00008 1500"
pixi run cutile-seahorse
pixi run cuda-seahorse
```
::::
:::::

:::::{tip} Bonus Exercise: Push your work to GitHub
1. Commit your `combined-build` workspace and both packages.
2. Push to your remote GitHub repository.

::::{hint} Solution
:class: dropdown
```bash
cd ~/reproducible-cuda-scipy-2026
git add combined-build
git commit -m "Add combined Pixi Build CUDA example"
git push origin main
```
::::
:::::

:::{seealso} Reference for the commands used here

- CLI: [`pixi init`](https://pixi.prefix.dev/latest/reference/cli/pixi/init/) · [`pixi run`](https://pixi.prefix.dev/latest/reference/cli/pixi/run/) · [`pixi install`](https://pixi.prefix.dev/latest/reference/cli/pixi/install/) · [`pixi task add`](https://pixi.prefix.dev/latest/reference/cli/pixi/task/add/) · [`pixi global install`](https://pixi.prefix.dev/latest/reference/cli/pixi/global/install/)
- Build: [getting started](https://pixi.prefix.dev/latest/build/getting_started/) · [backends](https://pixi.prefix.dev/latest/build/backends/) · [dependency types](https://pixi.prefix.dev/latest/build/dependency_types/) · [build variants](https://pixi.prefix.dev/latest/build/variants/)
- Manifest: [reference](https://pixi.prefix.dev/latest/reference/pixi_manifest/) · [preview features](https://pixi.prefix.dev/latest/reference/pixi_manifest/#preview-features) · [virtual packages per platform](https://pixi.prefix.dev/latest/workspace/multi_platform_configuration/#declaring-virtual-packages-per-platform)
:::
