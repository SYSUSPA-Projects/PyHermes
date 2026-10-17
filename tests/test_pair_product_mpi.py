"""A root-only low-level pairing must not enter world collectives."""

import numpy as np
import pytest

from pyhermes.io import SFCField, WindowFunc
from pyhermes.utils.mpi_util import MPI

comm = MPI.COMM_WORLD
pytestmark = pytest.mark.skipif(comm.Get_size() < 2, reason='Requires real MPI ranks')


def test_root_only_pair_batch_while_other_ranks_wait():
    value = None
    if comm.Get_rank() == 0:
        first = SFCField(threads=1)
        first.sfc_info = dict(J=3, L=8, V=64.**3, box_size=64., scale_factor=.125,
                               phi_resolution=64, wavelet_mode='db2', wavelet_level=6,
                               field_kind='derived_field', weight_normalization='catalog',
                               field_integral=1.)
        first.epsilon = np.random.default_rng(5).normal(size=(8, 8, 8))
        first.format_sfc_params()
        def windows():
            return [WindowFunc(dict(type='shell', len_args=dict(R=r)), first.sfc_info, threads=1)
                    for r in (0., 2., 4.)]
        expected = np.asarray([((first @ w)*first).as_array().mean() for w in windows()])
        with first.pair_with() as pair:
            value = pair.evaluate(windows())
        np.testing.assert_allclose(value, expected, rtol=1e-10, atol=1e-12)
    value = comm.bcast(value, root=0)
    assert value.shape == (3,) and np.all(np.isfinite(value))
