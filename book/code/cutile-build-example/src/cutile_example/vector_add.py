# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Simple elementwise vector addition using a cuTile kernel.

Adapted from NVIDIA's cuTile ``VectorAdd_quickstart.py`` sample and packaged
as a reusable, distributable module.
"""

import cupy as cp
import numpy as np
import cuda.tile as ct


@ct.kernel
def vector_add(a, b, c, tile_size: ct.Constant[int]):
    """Add vectors ``a`` and ``b`` elementwise into ``c``, one tile per block."""
    # Get the 1D program id.
    pid = ct.bid(0)

    # Load input tiles.
    a_tile = ct.load(a, index=(pid,), shape=(tile_size,))
    b_tile = ct.load(b, index=(pid,), shape=(tile_size,))

    # Perform elementwise addition.
    result = a_tile + b_tile

    # Store result.
    ct.store(c, index=(pid,), tile=result)


def run(vector_size: int = 2**12, tile_size: int = 2**4) -> cp.ndarray:
    """Launch the ``vector_add`` kernel on random input and return the result.

    Args:
        vector_size: Number of elements in each input vector.
        tile_size: Number of elements processed per block.

    Returns:
        The device array holding the elementwise sum.
    """
    grid = (ct.cdiv(vector_size, tile_size), 1, 1)

    rng = cp.random.default_rng()
    a = rng.random(vector_size)
    b = rng.random(vector_size)
    c = cp.zeros_like(a)

    ct.launch(
        cp.cuda.get_current_stream(),
        grid,  # 1D grid of processors
        vector_add,
        (a, b, c, tile_size),
    )

    # Verify against a NumPy reference on the host.
    expected = cp.asnumpy(a) + cp.asnumpy(b)
    np.testing.assert_array_almost_equal(cp.asnumpy(c), expected)

    return c


def main() -> None:
    """Console entry point: run the example and report success."""
    run()
    print("✓ vector_add passed!")


if __name__ == "__main__":
    main()
