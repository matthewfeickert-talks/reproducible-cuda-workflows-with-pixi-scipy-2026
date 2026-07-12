# Introduction to cuTile Python

There are many applications of CUDA in scientific computing and in recent years a large growth of applications in the scientific Python ecosystem with the advent of [CUDA Python](https://github.com/nvidia/cuda-python).
In this section of the tutorial, we will familiarize ourselves with a [recent addition to the CUDA Python ecosystem](https://youtu.be/uZTtViomW6w?si=MTbRATK9d4hfXINd): [cuTile Python](https://docs.nvidia.com/cuda/cutile-python/).
cuTile is a parallel programming model for NVIDIA GPUs and a Python-based {term}`DSL`.

For the purposes of this tutorial, we are not going to fully work through all of the interesting foundations of cuTile and [CUDA Tile](https://github.com/nvidia/cuda-tile).
Instead, we are going to take a high-level approach to learning the cuTile Python API through working through a cuTile demonstration in a [Jupyter notebook in the tutorial source repository](https://github.com/matthewfeickert-talks/reproducible-cuda-workflows-with-pixi-scipy-2026/blob/main/book/code/cutile-python-intro/cutile-python-intro.ipynb).

## Move work over to Brev

Use of CUDA requires an NVIDIA GPU, so the following sections of the tutorial will require access to a machine with an NVIDIA GPU and CUDA driver.
We will use the NVIDIA Brev platform to achieve this.
Part of the tutorial setup [covered Brev setup, login, and instance provisioning](#prepare-brev-instance).
Return to that section now and provision a Brev instance as described.
When you have finished with that return to this section.

## Reviewing Pixi workspace for cuTile

Once you have provisioned and logged into your Brev instance you will find that the instance has already cloned the tutorial source repository under your home directory.
Navigate to the top level of the repository

```
cd ${HOME}/reproducible-cuda-workflows-with-pixi-scipy-2026
```

and then to `book/code/cutile-python-intro`

```
cd book/code/cutile-python-intro
```

We have already provided a Pixi workspace for cuTile, but let's quickly inspect it to make sure that you could create it yourself.

```{literalinclude} code/cutile-python-intro/pixi.toml
```
