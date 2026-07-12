# Reproducible CUDA Accelerated Workflows for Scientists with Pixi

Taught at [SciPy 2026](https://www.scipy2026.scipy.org/) as a [tutorial](https://pretalx.com/scipy-2026/talk/9FQMMN/) on Monday July 13th, 2026

## Abstract

Scientific researchers need reproducible software environments for complex applications that can run across heterogeneous computing platforms.
Modern open source tools, like [Pixi](https://pixi.prefix.dev/), provide automatic reproducibility solutions for all dependencies while providing a high level interface well suited for researchers.

This tutorial will provide a practical introduction to using Pixi to easily create scientific and AI/ML environments that benefit from hardware acceleration, across multiple machines and platforms.
The focus will be on CUDA applications, such as machine learning frameworks and use of CUDA Tile, as well as using pixi-build to construct bespoke CUDA enabled conda packages.

## SciPy Logistical Information

* Tutorial name: [Reproducible CUDA Accelerated Workflows for Scientists with Pixi](https://pretalx.com/scipy-2026/talk/9FQMMN/)
* Date: 2026-07-13 (Monday)
* Time: 13:30&ndash;17:30 Central
* Location: University Hall, University of Minnesota

## Rough Outline

**00:00 &ndash; 00:05 (5 min):**

* Personal Introductions

**00:05 &ndash; 00:15 (10 min):**

* [Setup instructions](setup.md), setup your machine for the tutorial.

**00:15 &ndash; 00:30 (15 min):**

* [Introduction to Philosophy](introduction.md), an overview of the philosophy behind this tutorial.

**00:30 &ndash; 01:00 (30 min):**

* [Pixi introduction](pixi.md), an overview of Pixi's features and capabilities.

**01:00 &ndash; 01:40 (40 min):**

* [Pixi exercises](pixi-exercise.md), play around with Pixi and create a reproducible Python environment

**01:40 &ndash; 01:55 (15 min):**

* Break, grab a snack and stretch your legs.

**01:55 &ndash; 02:35 (40 min):**

* [Introduction to CUDA and CUDA conda packages](cuda-conda.md), the history and overview of how to use CUDA with Pixi and conda packages.

**02:35 &ndash; 02:45 (10 min):**

* Break, grab a snack and stretch your legs.

**02:45 &ndash; 03:15 (30 min):**

* [Building CUDA packages from source with Pixi Build](pixi-build-cuda.md), Introduction to building CUDA packages from source with Pixi Build.

**03:15 &ndash; 03:55 (40 min):**

* [Building Package Exercises](pixi-build-cuda-exercise.md), package your own CUDA C++ and Python/cuTile code as conda packages.

**03:55 &ndash; 04:00 (5 min):**

* Discussion over solutions to exercises.
* Time for participants to start exploring their own projects with Pixi and CUDA.
* General question and answer time with instructor team.

This tutorial was supported by the [University of Wisconsin&ndash;Madison Data Science Institute](https://dsi.wisc.edu/), [prefix.dev GmbH](https://prefix.dev/), and [NVIDIA](https://www.nvidia.com/).

## Acknowledgements

Special thanks to [Tim de Jager](https://github.com/tdejager) and [Wolf Vollprecht](https://github.com/wolfv) of prefix.dev GmBH and [Daniel Ching](https://github.com/carterbox) of NVIDIA for their review, feedback, and contributions to this tutorial.
Special thanks to [Nick Hodgskin](https://github.com/veckothegecko) for volunteering to serve as a teaching assistant for this tutorial.
