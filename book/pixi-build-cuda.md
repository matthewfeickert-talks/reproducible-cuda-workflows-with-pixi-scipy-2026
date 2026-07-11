# Pixi Build: CUDA packages from source

Earlier, in the [Pixi introduction](pixi.md#preview-pixi-build), we saw a teaser: besides managing environments and tasks, Pixi can also *build* conda packages from source.
This chapter is where we return to that.

We will build **two** conda packages, both computing the same Mandelbrot fractal on the GPU, each through a different [Pixi Build](https://pixi.prefix.dev/latest/build/backends/) backend:

- `cutile-brot` — a Python/[cuTile](https://github.com/NVIDIA/cutile-python) package, built with the [`pixi-build-python`](https://pixi.prefix.dev/latest/build/backends/pixi-build-python/) backend.
- `cuda-brot` — a CUDA C++ package, built with the [`pixi-build-cmake`](https://pixi.prefix.dev/latest/build/backends/pixi-build-cmake/) backend (`nvcc` + CMake).

See the [Pixi Build getting started guide](https://pixi.prefix.dev/latest/build/getting_started/) for the full picture; this chapter walks the CUDA-flavored version of it.

Both live in **one workspace**, added as *source dependencies*, so `pixi run` (re)builds whichever package a task needs.

:::{note} Preview feature
Pixi Build is currently a [preview feature](https://pixi.prefix.dev/latest/reference/pixi_manifest/#preview-features).
Its manifest surface is still evolving, so pin your backends (as we do below) and expect small changes between Pixi versions.
:::

# Workspaces and packages: two roles for one manifest

Until now every `pixi.toml` we wrote had a `[workspace]` table.
Pixi Build introduces a second role for the manifest, and it is important to keep the two straight.
Both are described in full in the [Pixi manifest reference](https://pixi.prefix.dev/latest/reference/pixi_manifest/).

| Manifest role | Marked by | Describes | Example in this chapter |
|---|---|---|---|
| **Workspace** | [`[workspace]`](https://pixi.prefix.dev/latest/reference/pixi_manifest/#the-workspace-table) | *Your* development environment: dependencies, tasks, platforms, channels | the top-level `pixi.toml` |
| **Package** | [`[package]`](https://pixi.prefix.dev/latest/reference/pixi_manifest/#the-package-section) (no `[workspace]`) | How to build *one* distributable conda package | `src/cutile-brot/pixi.toml`, `src/cuda-brot/pixi.toml` |

A workspace is what you `pixi run` and `pixi install`.
A package is what Pixi turns into a `.conda` file.
A directory can be one or the other, and a workspace can *depend on* packages that live inside it.

The layout we are building towards:

```
combined-build/
├── pixi.toml                    # the workspace: dependencies + tasks
└── src/
    ├── cutile-brot/             # a pixi-build-python package
    │   ├── pixi.toml            #   package manifest (build backend)
    │   ├── pyproject.toml       #   name / version / deps (hatchling)
    │   └── src/cutile_brot/
    │       ├── __init__.py
    │       └── mandelbrot.py
    └── cuda-brot/               # a pixi-build-cmake package
        ├── pixi.toml            #   package manifest
        ├── CMakeLists.txt
        └── src/main.cu
```

# Enabling Pixi Build

Because it is a preview feature, source-built dependencies require `preview = ["pixi-build"]` in the `[workspace]` table.
Because both packages target the GPU, the workspace also declares a [rich CUDA platform](cuda-conda.md#cuda-use-with-pixi) (see [multi-platform configuration](https://pixi.prefix.dev/latest/workspace/multi_platform_configuration/#declaring-virtual-packages-per-platform)) so the solver picks GPU-enabled builds:

```{code} toml
:filename: pixi.toml
:linenos:
[workspace]
name = "combined-build"
channels = ["https://prefix.dev/conda-forge"]
platforms = [{ platform = "linux-64", cuda = "13" }]
preview = ["pixi-build"]
```

The `cuda = "13"` entry declares the `__cuda` virtual package on this platform, exactly as we did in the [CUDA conda packages chapter](cuda-conda.md#adding-a-cuda-rich-platform).

# Source dependencies

A workspace depends on a local package the same way it depends on anything else, except the specifier is a `path` pointing at the package's directory:

```{code} toml
:filename: pixi.toml
:linenos:
[dependencies]
cutile-brot = { path = "src/cutile-brot" }
cuda-brot = { path = "src/cuda-brot" }
```

When a dependency points at a directory containing a `[package]` manifest, Pixi *builds* that package with its backend and installs the result into the environment.
[`pixi install`](https://pixi.prefix.dev/latest/reference/cli/pixi/install/) and [`pixi run`](https://pixi.prefix.dev/latest/reference/cli/pixi/run/) rebuild a source dependency automatically whenever its inputs change, and cache the result otherwise, so subsequent runs are instant. There are multiple dependencies that can be part of a package, those you need for building and those you need for running the package. The different types are described in the [Pixi Build dependency types documentation](https://pixi.prefix.dev/latest/build/dependency_types/).

# Build backends

The backend is the piece that actually knows how to compile and package your source.
It is declared in the **package** manifest under `[package.build]`.
Pixi ships [several backends](https://pixi.prefix.dev/latest/build/backends/); the two we use here are the most common for scientific work:

| Backend | Builds | Reads metadata from |
|---|---|---|
| [`pixi-build-python`](https://pixi.prefix.dev/latest/build/backends/pixi-build-python/) | Python packages | `pyproject.toml` (e.g. hatchling) |
| [`pixi-build-cmake`](https://pixi.prefix.dev/latest/build/backends/pixi-build-cmake/) | C / C++ / CUDA / Fortran via CMake | `CMakeLists.txt` |
| [`pixi-build-rust`](https://pixi.prefix.dev/latest/build/backends/pixi-build-rust/) | Rust crates via Cargo | `Cargo.toml` |

Let's build one package with each of the first two.

# A Python package: `pixi-build-python`

The [`pixi-build-python`](https://pixi.prefix.dev/latest/build/backends/pixi-build-python/) backend turns `cutile-brot` into a conda package.
It is a normal Python package: a `pyproject.toml` with metadata and dependencies, an entry-point script, and the source under `src/`.

```{code} toml
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
```

The `[project.scripts]` entry is what gives us a `cutile-brot` command once the package is installed.

The **package** manifest sitting next to it is tiny — the Python backend reads the name, version and dependencies from `pyproject.toml`:

```{code} toml
:filename: src/cutile-brot/pixi.toml
:linenos:
[package.build]
backend = { name = "pixi-build-python", version = "0.*" }
# Map the pyproject.toml dependencies onto conda-forge packages instead of
# ignoring them. The backend warns if it can't find a mapping for one.
config.ignore-pypi-mapping = false
```

By default the Python backend ignores `pyproject.toml`'s `[project.dependencies]` (it assumes you declare runtime dependencies as conda packages yourself).
Setting `config.ignore-pypi-mapping = false` tells it to instead **map** those PyPI dependencies onto their conda-forge equivalents, so `numpy`, `cupy` and `cuda-tile` become conda dependencies of the built package automatically. This will probably change in the future when we feel the automatic mapping is reliable enough to be the default.

# A CUDA C++ package: `pixi-build-cmake`

The [`pixi-build-cmake`](https://pixi.prefix.dev/latest/build/backends/pixi-build-cmake/) backend compiles `cuda-brot`'s `.cu` file with `nvcc` through CMake.
The CMake project is ordinary, with one line that matters for packaging — the `install(TARGETS ...)` rule tells the backend which artifact to put into the package:

```{code} cmake
:filename: src/cuda-brot/CMakeLists.txt
:linenos:
:emphasize-lines: 5
cmake_minimum_required(VERSION 3.24...4.3)
project(cuda-brot LANGUAGES CXX CUDA)

add_executable(cuda-brot src/main.cu)
install(TARGETS cuda-brot RUNTIME DESTINATION bin)
```

The package manifest is where the CUDA-specific configuration lives.
There are three non-obvious pieces; each is annotated below:

```{code} toml
:filename: src/cuda-brot/pixi.toml
:linenos:
[package]
name = "cuda-brot"
version = "0.1.0"

[package.build]
backend = { name = "pixi-build-cmake", version = "0.*" }
# Ask the backend for a C++ *and* a CUDA compiler. On conda-forge the CUDA
# compiler ships in the `cuda-nvcc` package (see build-variants below).
config.compilers = ["cxx", "cuda"]
# `.cu` files aren't a default build input, so declare them as one.
# This allows Pixi to automatically rebuild the package if any `.cu` file changes.
config.extra-input-globs = ["*.cu"]

[package.host-dependencies]
# Headers + libcudart to compile and link against. Its run-export makes the
# built package depend on `cuda-cudart` at runtime automatically.
cuda-cudart-dev = "*"
```

Two things worth calling out:

- **`host-dependencies` vs `run-dependencies`.** Host dependencies are present *while building* (headers, import libraries). `cuda-cudart-dev` carries a *run-export*, so Pixi automatically adds the matching runtime dependency (`cuda-cudart`) to the finished package, you don't declare the runtime dependency yourself.
- **Pinning the CUDA compiler.** `config.compilers = ["cuda"]` says "give me a CUDA compiler"; a [**build variant**](https://pixi.prefix.dev/latest/build/variants/) on the *workspace* says *which* one. Add this to the top-level workspace manifest:

```{code} toml
:filename: pixi.toml
:linenos:
[workspace.build-variants]
cuda_compiler = ["cuda-nvcc"]
cuda_compiler_version = ["13.1"]
```

The headline result: **no system CUDA toolkit is required.** `nvcc`, the CUDA runtime, the C++ compiler, and CMake all come from conda-forge, managed by Pixi.

# Running, building, and installing

## Run from the workspace

[Tasks](https://pixi.prefix.dev/latest/workspace/advanced_tasks/) in the workspace point at the packages' entry points.
Because each package is a source dependency, the first [`pixi run`](https://pixi.prefix.dev/latest/reference/cli/pixi/run/) builds it (download toolchain, compile, cache) and later runs are instant:

```{code} toml
:filename: pixi.toml
:linenos:
[tasks.cutile-brot]
cmd = "cutile-brot"
description = "Render a GPU Mandelbrot set in the terminal (cuTile)"

[tasks.cuda-brot]
cmd = "cuda-brot"
description = "Render the classic Mandelbrot set (CUDA C++)"
```

```console
pixi run cutile-brot
pixi run cuda-brot
```

## Solve or build without a GPU

Building compiles code but does not need an NVIDIA driver; the *rich platform* just needs to believe a CUDA driver exists.
On a machine without one (e.g. CI), override the virtual package:

```console
CONDA_OVERRIDE_CUDA=13 pixi install
```

## Produce a `.conda` artifact

To turn a package into a distributable, relocatable conda package, use [`pixi publish`](https://pixi.prefix.dev/latest/build/getting_started/):

```console
pixi publish --path src/cutile-brot
pixi publish --path src/cuda-brot
```

Each produces a `<name>-0.1.0-<build>.conda` file.

:::{note}
We use `pixi publish` as we deprecated the existing `pixi build` command. The new command is more explicit about the fact that it produces a distributable package, and it will eventually replace `pixi build`.

`pixi build` might be reused in the future for building a workspace environment, but that is not yet implemented.
:::

## Install it globally

You can install a built package into your [global environment](https://pixi.prefix.dev/latest/global_tools/introduction/) straight from its source directory with [`pixi global install`](https://pixi.prefix.dev/latest/reference/cli/pixi/global/install/), which exposes its command line applications machine-wide.
This mimics what a user would get *after* you published the package to a channel, without needing a channel at all:

```console
pixi global install --path src/cutile-brot
pixi global install --path src/cuda-brot
```

Now `cutile-brot` and `cuda-brot` run from any directory.

:::{note} Publishing to a channel
When you *do* want to share a package with others, [`pixi publish`](https://pixi.prefix.dev/latest/reference/cli/pixi/publish/) uploads the built `.conda` to a conda channel (for example your own channel on [prefix.dev](https://prefix.dev)).
That's an optional next step beyond this tutorial.
:::

# Recap

| Concept | What it is |
|---|---|
| Workspace manifest (`[workspace]`) | Your dev environment; what you `pixi run` |
| Package manifest (`[package]`) | How to build one conda package |
| `preview = ["pixi-build"]` | Enables source-built dependencies |
| Source dependency (`{ path = ... }`) | A dependency Pixi builds from a local package |
| Build backend (`[package.build]`) | Knows how to compile/package (`pixi-build-python`, `pixi-build-cmake`, ...) |
| `host-dependencies` + run-export | Build-time deps that auto-add the matching runtime dep |
| `[workspace.build-variants]` | Pins *which* compiler (e.g. `cuda-nvcc` `13.1`) |
| `pixi publish` | Produces a `.conda` artifact |
| `pixi global install --path` | Installs a local package globally, as if published |

:::{seealso} Pixi Build documentation

- [Building packages getting started](https://pixi.prefix.dev/latest/build/getting_started/)
- [Build backends overview](https://pixi.prefix.dev/latest/build/backends/) · [`pixi-build-python`](https://pixi.prefix.dev/latest/build/backends/pixi-build-python/) · [`pixi-build-cmake`](https://pixi.prefix.dev/latest/build/backends/pixi-build-cmake/) · [`pixi-build-rust`](https://pixi.prefix.dev/latest/build/backends/pixi-build-rust/)
- [Dependency types](https://pixi.prefix.dev/latest/build/dependency_types/) · [Build variants](https://pixi.prefix.dev/latest/build/variants/) · [Workspace vs. package](https://pixi.prefix.dev/latest/build/workspace/)
- [Manifest reference](https://pixi.prefix.dev/latest/reference/pixi_manifest/) · [Multi-platform (virtual packages)](https://pixi.prefix.dev/latest/workspace/multi_platform_configuration/#declaring-virtual-packages-per-platform)
:::

Now put it into practice in the [Pixi Build exercises](pixi-build-cuda-exercise.md).
