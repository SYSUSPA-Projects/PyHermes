"""Sampling helpers for synthetic point catalogs."""

import numpy as np


def random_points_box(count=None, box_size=None, ndim=3, rng=None, seed=None, N=None):
    """Draw uniformly distributed random points inside a periodic box."""
    if count is None:
        count = N
    if rng is None:
        rng = np.random.default_rng(seed=seed)
    return rng.uniform(0.0, box_size, size=(count, ndim))
