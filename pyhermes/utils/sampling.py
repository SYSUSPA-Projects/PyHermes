"""Sampling helpers for synthetic point catalogs."""

import numpy as np


def random_box_positions(count, box_size, ndim=3, rng=None, seed=None):
    """Draw uniformly distributed random points inside a periodic box."""
    if rng is None:
        rng = np.random.default_rng(seed=seed)
    return rng.uniform(0.0, box_size, size=(count, ndim))


def regular_grid_positions(n_grid, box_size, offset=0.5, ndim=3):
    """Return positions on a regular periodic grid and the grid spacing."""
    n_grid = int(n_grid)
    ndim = int(ndim)
    if n_grid <= 0:
        raise ValueError("n_grid must be positive.")
    if ndim <= 0:
        raise ValueError("ndim must be positive.")

    box_arr = np.asarray(box_size, dtype=np.float32)
    if box_arr.ndim == 0:
        box_arr = np.full(ndim, float(box_arr), dtype=np.float32)
    elif box_arr.shape != (ndim,):
        raise ValueError("box_size must be a scalar or a 1D array with length ndim.")

    offset_arr = np.asarray(offset, dtype=np.float32)
    if offset_arr.ndim == 0:
        offset_arr = np.full(ndim, float(offset_arr), dtype=np.float32)
    elif offset_arr.shape != (ndim,):
        raise ValueError("offset must be a scalar or a 1D array with length ndim.")

    dx = box_arr / n_grid
    axes = [
        ((np.arange(n_grid, dtype=np.float32) + offset_arr[i]) * dx[i]) % box_arr[i]
        for i in range(ndim)
    ]
    mesh = np.meshgrid(*axes, indexing="ij")
    pos = np.column_stack([coord.ravel() for coord in mesh]).astype(np.float32, copy=False)
    if np.ndim(box_size) == 0:
        return pos, float(dx[0])
    return pos, dx
