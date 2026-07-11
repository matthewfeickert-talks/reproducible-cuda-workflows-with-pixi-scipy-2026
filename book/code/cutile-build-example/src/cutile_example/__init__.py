"""A minimal, distributable cuTile vector-addition example."""

from cutile_example.mandelbrot import compute as mandelbrot_compute
from cutile_example.vector_add import run, vector_add

__version__ = "0.1.0"

__all__ = ["run", "vector_add", "mandelbrot_compute", "__version__"]
