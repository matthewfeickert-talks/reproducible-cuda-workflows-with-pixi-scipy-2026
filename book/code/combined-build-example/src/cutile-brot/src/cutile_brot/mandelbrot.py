# SPDX-License-Identifier: Apache-2.0

"""cutile-brot, the cuTile edition.

A Mandelbrot renderer for your terminal, computed on the GPU with cuTile.
Each pixel's escape-time iteration is computed on the device; the result is
drawn with Unicode half-blocks (``▀``) and 24-bit ANSI colors, so each
terminal cell holds two vertically stacked pixels.

This is a cuTile port of the classic one-thread-per-pixel CUDA C++ kernel.
The tile model has no per-thread ``while`` loop: a whole tile of pixels is
iterated together for a fixed ``max_iter`` steps, and a mask freezes each
pixel as soon as it escapes.
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
        # Freeze pixels that have already escaped so `mag2` keeps the
        # first-escape magnitude for band-free smooth coloring.
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

    # Build the complex plane on the host (cheap) and upload it.
    xs = center_re + (np.arange(wp) - width / 2.0) * step
    ys = center_im - (np.arange(hp) - height / 2.0) * step
    grid_re, grid_im = np.meshgrid(xs, ys)  # both (hp, wp)

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
