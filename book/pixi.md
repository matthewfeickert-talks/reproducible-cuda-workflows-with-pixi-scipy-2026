# Pixi: Introduction


![Pixi banner](https://github.com/user-attachments/assets/fa2e98c2-0913-4098-9579-8f2efff7f814)


Pixi is a cross-platform workspace manager for reproducible software environments and development workflows.
For scientific Python developers, a useful mental model is that Pixi combines parts of `conda`/`mamba`, `conda-lock`, `uv`, and a task runner, while also being able to manage non-Python dependencies such as CUDA, C/C++, Fortran, R, and Rust packages.

Most participants will already have pieces of this workflow: `pip` or `uv` for Python packages, `conda` or `mamba` for binary scientific packages, and shell scripts or Makefiles for repeatable commands.
Pixi's value is not that these tools cannot install packages, but that one workspace can describe the full environment and workflow reproducibly.

For `pip` and `uv` users, Pixi adds first-class support for conda packages, compiled scientific libraries, CUDA libraries, compilers, and multi-platform lock files.
For `conda` and `mamba` users, Pixi keeps the conda package ecosystem while adding project-local lock files, tasks, named environments, and rich platform declarations.

There are two main workflows:
- Installing standalone tools globally (`pixi global`)
- Managing reproducible project workspaces

# What does Pixi solve?
Pixi helps make the answer to *"what did you run, and where?"* explicit and reproducible.
For this tutorial, the important pieces are:

1. **Reproducibility by default**: Pixi writes a `pixi.lock` file with the exact packages for each environment and platform.
2. **Conda and PyPI together**: Pixi can install conda packages and PyPI packages into the same environment, using `uv` for PyPI dependencies.
3. **Project-local environments**: Environments live with the workspace, so collaborators do not need to recreate your local conda environment naming scheme.
4. **{term}`Cross-Platform` and {term}`Cross-Language` workflows**: Pixi can describe Linux, macOS, Windows, CUDA, Python, C/C++, Fortran, Rust, R, and more in one project model.
5. **Tasks**: Pixi provides a cross-platform task runner so commands like testing, training, building, and linting are part of the shared workspace.
6. **Package building**: Pixi can build conda packages from source, including CUDA-enabled packages, and publish them to a channel for others to use.

For common workflow steps, Pixi can replace a few commands you might otherwise reach for from different tools:

| Task | You might use | Pixi |
|---|---|---|
| Add a conda package | `conda install scipy` / `mamba install scipy` | `pixi add scipy` |
| Add a PyPI package | `pip install pydantic` / `uv add pydantic` | `pixi add --pypi pydantic` |
| Run a command in the environment | `conda run ...` / `uv run ...` | `pixi run ...` |
| Start an activated shell | `conda activate my-env` | `pixi shell` |
| Lock an environment | `conda-lock` / `uv lock` | `pixi lock` |
| Run project commands | `make test` / shell scripts | `pixi run test` |
| Install a global CLI tool | `pipx install ruff` / `uv tool install ruff` | `pixi global install ruff` |

# The project workflow
Pixi is designed to be used in a project-based workflow.
Tools like `uv`, `npm`, `deno`, `cargo`, `maven` and `pixi` are all designed to be used in a project-based workflow.
This means that you can create a project and then use Pixi to manage the dependencies and tasks for that project.
You can think of a project as a self-contained directory that contains all the files and configurations needed to build and run your application.
Often the project will keep the environment it installs close to the project folder itself, so it will not clutter the system.
This is a great way to keep your projects organized and to avoid conflicts between different projects.

### Project-based vs Environment-based vs System-based
To give a little background why Pixi is designed this way, let's take a look at the different ways to manage packages and environments.

::::{tab-set}
:::{tab-item}Project-based workflow
**Supporting tools:** `pixi`, `uv`, `npm`, `deno`, `cargo`, `maven`

**Pros:**
- Isolated environments per project (no conflicts)
- Easy to reproduce and share with others (declarative)
- Keeps dependencies close to the project

**Cons:**
- Potentially use more disk space (multiple environments)
- Managing environments over multiple projects is less straightforward.
:::
:::{tab-item}Environment-based workflow
**Supporting tools:** `conda`, `mamba`, `micromamba`, `pip`, `pipenv`, `uv pip`

**Pros:**
- Easy to share environments across multiple projects
- Isolated environments (no conflicts)

**Cons:**
- You keep a mental model of what is installed in which environment
- Harder to reproduce exact environments later

:::
:::{tab-item}System-based workflow
**Supporting tools:** `apt`, `rpm`, `brew`, `choco`, `winget`, `scoop`, `flatpak`

**Pros:**
- Simple, everything installed globally
- Every OS has a default package manager
- Guaranteed to work with system libraries

**Cons:**
- Very High risk of dependency/version conflicts with other projects (especially Python)
- Hard to reproduce or share setups
- Limited flexibility when it comes to versions
:::
::::


# Creating a project
As Pixi uses the project-based workflow, it uses a manifest file to keep track of the dependencies and tasks for the project.
This is also known as **declarative** configuration, where you describe what you want, and Pixi will take care of the rest.
The manifest file is called `pixi.toml`, or you can use `pyproject.toml`, and it is located in the root of the project.


To create a new project, you can use the `pixi init` command.
::::{tab-set}
:::{tab-item}pixi.toml
```bash
pixi init my_project
```
This will create a new directory called `my_project` and initialize a new `pixi.toml` file in it.
```bash
my_project
├── .gitattributes
├── .gitignore
└── pixi.toml
```

The `pixi.toml` file is a {term}`TOML` file that contains the configuration for the project.

```{code} toml
:filename: pixi.toml
:linenos:

[workspace]
authors = ["Jane Doe <jane.doe@example.com>"]
channels = ["https://prefix.dev/conda-forge"]
name = "my_project"
platforms = ["osx-arm64"]
version = "0.1.0"

[tasks]

[dependencies]
```

The `pixi.toml` doesn't have the basic Python package structure like the `pyproject.toml` file, because it is not a Python package by default.

As `pixi.toml` has a JSON schema, it is possible to use IDE's like VSCode to edit the field with autocompletion.
Install the [Even Better TOML](https://marketplace.visualstudio.com/items?itemName=tamasfe.even-better-toml) VSCode extension to get the best experience. Or use the integrated schema support in PyCharm.

:::

:::{tab-item}pyproject.toml
```bash
pixi init my_pyproject --format pyproject
```
This will create a new directory called `my_pyproject` and initialize a new `pyproject.toml` file in it.
It will also create a `src` directory with a `my_pyproject` subdirectory and an `__init__.py` file in it, to make it a simple Python package.
```bash
my_project
├── .gitattributes
├── .gitignore
├── pyproject.toml
└── src
    └── my_pyproject
        └── __init__.py
```

The `pyproject.toml` file is a {term}`TOML` file that contains the configuration for the project.

```{code} toml
:filename: pyproject.toml
:linenos:
[project]
authors = [{name = "Jane Doe", email = "jane.doe@example.com"}]
dependencies = []
name = "my_pyproject"
requires-python = ">= 3.11"
version = "0.1.0"

[build-system]
build-backend = "hatchling.build"
requires = ["hatchling"]

[tool.pixi.workspace]
channels = ["https://prefix.dev/conda-forge"]
platforms = ["osx-arm64"]

[tool.pixi.pypi-dependencies]
my_pyproject = { path = ".", editable = true }

[tool.pixi.tasks]
```
Note that the `pyproject.toml` file is a little different from the `pixi.toml` file.
The `pyproject.toml` file is a standard file used by many Python tools, thus it makes it easier to share the project with others.
The only difference between the two files is that the `pyproject.toml` file has a `[tool.pixi` prefix to the Pixi specific sections.
:::
::::

For the rest of this tutorial, we will use the `pixi.toml` file as the main file.

# Managing dependencies
After creating the project, you can start adding dependencies to the project.
Pixi uses the `pixi add` command to add dependencies to the project.
By default, this command adds the conda dependency to the `pixi.toml` or `pyproject.toml` file, solves the dependencies, writes the lockfile, and installs the package in the environment. For example, let's add `numpy` and `pytest` to the project.
```bash
pixi add numpy pytest
```
This will result in the following manifest file:

```{code} toml
:filename: pixi.toml
:linenos:
:emphasize-lines: 11-12
[workspace]
authors = ["Jane Doe <jane.doe@example.com>"]
channels = ["https://prefix.dev/conda-forge"]
name = "my_project"
platforms = ["osx-arm64"]
version = "0.1.0"

[tasks]

[dependencies]
numpy = ">=2.5.1,<3"
pytest = ">=9.1.1,<10"
```

If you want a specific version or range, provide it when adding the package.
The most common forms are:

```bash
pixi add "numpy==2.2.6"
pixi add "numpy>=2.2,<3"
pixi add "python=3.12.*"
pixi add conda-forge::numpy
```

Pixi supports conda MatchSpecs, so you can be more specific when needed, but most projects only need package names, version ranges, and occasionally a channel-qualified dependency.

## PyPI dependencies
Pixi can also install packages from PyPI through its integration with `uv`.
In the Rust code, Pixi depends on the `uv` package manager to install the packages from PyPI.
This means that you can use the `pixi add --pypi` command to install packages from PyPI.

```bash
pixi add --pypi pydantic
```

Which results in it being added to the manifest file as:
::::{tab-set}
:::{tab-item}pixi.toml
In the `pixi.toml` file, it will be added to the `[pypi-dependencies]` section.
```{code} toml
:filename: pixi.toml
:linenos:
[pypi-dependencies]
pydantic = ">=2.13.4, <3"
```
:::
:::{tab-item}pyproject.toml
In the `pyproject.toml` file, it will be added to the `[project]` section in the normal case.
```{code} toml
:filename: pyproject.toml
:linenos:
[project]
# ...
dependencies = ["pydantic>=2.13.4,<3"]
```
Things like `path` and `editable` are added through the Pixi specific section.
For example:
```{code} toml
:filename: pyproject.toml
:linenos:
[tool.pixi.pypi-dependencies]
my_pyproject = { path = ".", editable = true }
```
:::
::::


What pixi does differently from managing PyPI packages through other package managers, is that it will install the packages in the same environment as the conda packages, but will not overwrite the conda packages.
We've got a mapping between the conda packages and the PyPI packages, so that we can let `uv` know which packages to install and which packages to ignore because they are already installed.

::: {note} Pixi doesn't install `uv`!
While Pixi uses `uv` to install the PyPI packages, it doesn't install `uv` itself.
So you cannot use `uv` directly in the project, without installing it first.
:::

### Special types of dependencies
Pixi has a few special types of dependencies that you can use in the project.

| Type | Description | Example |
|---|---|---|
| `git` | Install a package from a git repository | `git = "https://github.com/user/repo.git"`|
| `branch` | Install a specific branch from a git repository (requires `git`) | `branch = "main"` |
| `tag` | Install a specific tag from a git repository (requires `git`)| `tag = "v1.0.0"` |
| `rev` | Install a specific commit from a git repository (requires `git`) | `rev = "abc123"` |
| `path` | Install a package from a local directory | `path = "./local-python-package"` |
| `editable` (`pypi` only) | Install a package in editable mode | `editable = true` |
| `url` | Install a package from a URL | `url = "https://example.com/package.whl"` |


## Lockfile
The lockfile is a file that contains the exact versions of the packages that were installed in the environment.
This file is used to ensure that the same versions of the packages are installed in the environment when the project is shared with others.
What should you know about the lockfile?
- The lockfile is called `pixi.lock` and is located in the root of the project, next to the `pixi.toml` or `pyproject.toml` file.
- The lockfile is a YAML file that contains the exact versions of the packages that were installed in the environment.
- The lockfile is automatically (re-)generated when you `add`, `remove`, `update` a package in the project, or when you run `pixi install/run/shell/lock` and it's not existing yet.
- The lockfile is meant to be shared with others, so that they can reproduce the same environment.
- The lockfile is not meant to be edited manually, as it is automatically generated by Pixi.

```{code} yaml
:filename: pixi.lock
:caption: Example lockfile, highly simplified for readability
:linenos:
version: 7
platforms:
- name: linux-64
  virtual-packages:
  - __unix=0=0
  - __linux=4.18
  - __glibc=2.28
  - __archspec=0=x86_64
environments:
  default:
    channels:
    - url: https://prefix.dev/conda-forge/
    indexes:
    - https://pypi.org/simple
    packages:
      linux-64:
      - conda: https://prefix.dev/conda-forge/linux-64/bzip2-1.0.8-h99b78c6_7.conda
      - pypi: ...
packages:
- conda: https://prefix.dev/conda-forge/linux-64/bzip2-1.0.8-h99b78c6_7.conda
  sha256: adfa71f158cbd872a36394c56c3568e6034aa55c623634b37a4836bd036e6b91
  md5: fc6948412dbbbe9a4c9ddbbcfe0a79ab
  depends:
  - __unix
  license: bzip2-1.0.6
  license_family: BSD
  size: 122909
  timestamp: 1720974522888
- pypi: ...
```

# Managing tasks
Pixi has a built-in {term}cross-platform task runner that allows you to define tasks in the manifest.
This is a great way to share tasks with others and to ensure that the same tasks are run in the same environment.
The tasks are defined in the `[tasks]` section.

## Basic tasks

You can use the `pixi task` command to modify the tasks in the project.
```bash
pixi task add hello "echo Hello World"
```
This will add a new task called `hello` to the project, which will print `Hello World` to the console.
```{code} toml
:filename: pixi.toml
:linenos:
[tasks]
hello = "echo Hello World"
```
You can also use the `pixi run` command to run the tasks in the project.
```bash
pixi run hello
```
This will run the `hello` task and print `Hello World` to the console.

# Environments
Now you know the basics of dealing with the Pixi manifest basics.
Next step is actually use the environments it can create for you.

## Activating environments
Because Pixi creates virtual environments for you, it is important to activate the environment before running any commands.
You can do this by using the `pixi shell` or the `pixi run` command, these commands will automatically activate the environment for you.

```bash
pixi run python -VV
# or:
pixi shell
python -VV
exit
```

Activating an environment is not much more than running a script that sets the environment variables for you.
To investigate this, you can use `pixi shell-hook` to view what the shell script looks like.
```bash
pixi shell-hook
```
This will print the shell script that is used to activate the environment.

## Platforms
Pixi solves environments for the platforms listed in the workspace manifest.
A typical scientific Python project might support macOS laptops, Windows laptops, and Linux servers from one file:

```{code} toml
:filename: pixi.toml
:linenos:
[workspace]
channels = ["conda-forge"]
platforms = ["osx-arm64", "linux-64", "win-64"]
```

Pixi can also use richer platform declarations.
For example, later in this tutorial we will distinguish a regular Linux target from a CUDA-capable Linux target:

```{code} toml
:filename: pixi.toml
:linenos:
[workspace]
channels = ["conda-forge"]
platforms = [
  "osx-arm64",
  "win-64",
  "linux-64",
  { name = "linux-64-cuda", platform = "linux-64", cuda = "12" },
]
```

This lets one workspace describe both local CPU development and remote GPU execution with the platform requirements for each made explicit.

## Features and environments
Pixi separates reusable configuration from the environments you actually run.

A **feature** is a named block of configuration.
It can define dependencies, tasks, platforms, channels, or activation settings.

An **environment** is a named combination of features.
It is what Pixi solves, installs, and runs.

The top-level `[dependencies]` and `[tasks]` tables are the **default feature**.
The default feature is included automatically unless `no-default-feature = true` is set.

```{code} toml
:filename: pixi.toml
:linenos:
[workspace]
channels = ["conda-forge"]
platforms = ["linux-64", "osx-arm64", "win-64"]

# Default feature: shared runtime dependencies.
[dependencies]
python = ">=3.14.6,<3.15"
numpy = ">=2.5.1,<3"

[tasks]
run-example = "python -c 'import numpy; print(numpy.__version__)'"

[feature.notebook.dependencies]
jupyterlab = ">=4.6.1,<5"

[feature.notebook.tasks]
notebook = "jupyter lab"

[feature.docs.dependencies]
mystmd = ">=1.10.1,<2"

[feature.docs.tasks]
docs = "myst build --html"

[environments]
default = { features = [] }
notebook = { features = ["notebook"] }

# Documentation can be independent of the runtime dependencies.
docs = { features = ["docs"], no-default-feature = true }
```

This gives you three environments:

| Environment you run | Features combined into it | Example command |
|---|---|---|
| `default` | `default` | `pixi run run-example` |
| `notebook` | `default` + `notebook` | `pixi run notebook` |
| `docs` | `docs` only | `pixi run docs` |

Pixi can run `notebook` and `docs` without `-e` because each task is available in only one environment.
Use `-e` or `--environment` when you want to select an environment explicitly.

Use a **feature** to define a reusable workflow layer.
Use an **environment** to choose which layers Pixi should install and run.
Use `no-default-feature = true` when an environment should not include the shared runtime dependencies.

(preview-pixi-build)=
# Building packages
So far, we have used Pixi to manage environments and tasks.
Pixi can also build conda packages from source through build backends.
In practice, anything that can be packaged as a conda package can be built this way with a matching backend, including Python packages, C/C++ or Fortran libraries, and internal libraries that are not already packaged.

Pixi Build is currently a preview feature, enabled in the workspace manifest:

```{code} toml
:filename: pixi.toml
:linenos:
:emphasize-lines: 2
[workspace]
preview = ["pixi-build"]
channels = ["conda-forge"]
platforms = ["linux-64"]
```

A workspace can then depend on a local source package:

```{code} toml
:filename: pixi.toml
:linenos:
[dependencies]
my-library = { path = "src/my-library" }
```

If `my-library` contains Pixi package metadata, Pixi can build it as a conda package and install the built package into the environment.
We will return to this in a later chapter.
