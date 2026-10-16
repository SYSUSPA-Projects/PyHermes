"""Run with mpiexec -n 2 python -m pytest tests/test_corr2pcf_mpi.py."""

import numpy as np
import pytest

from pyhermes.io import SFCField
from pyhermes.theory.corr2pcf import Corr_2PCF
from pyhermes.utils.mpi_util import MPI

COMM = MPI.COMM_WORLD
pytestmark = pytest.mark.skipif(COMM.Get_size() < 2, reason='requires a real multi-rank MPI run')


def field(seed):
    if COMM.Get_rank() != 0:
        return None
    result = SFCField(threads=1)
    result.sfc_info = dict(J=3, L=8, V=512, box_size=64., scale_factor=.125,
                           phi_resolution=64, wavelet_mode='db2', wavelet_level=6,
                           field_kind='catalog_field', weight_normalization='catalog',
                           field_integral=1., raw_field_weighted_sum=100.,
                           catalog_weight_sum=100., catalog_weight_sq_sum=100., particle_count=100)
    values = np.random.default_rng(seed).uniform(.1, 1., (8, 8, 8))
    result.epsilon = values/values.sum()
    result.format_sfc_params()
    return result


@pytest.mark.parametrize('nonuniform', [False, True])
def test_multi_rank_products_and_global_fft_counts(nonuniform):
    first = field(701)
    second = field(702) if nonuniform else first
    random1, random2 = (field(703), field(704)) if nonuniform else ('uniform', 'uniform')
    results = []
    for method, reuse, window_mode in [('convolution', False, 'materialized'), ('convolution', True, 'materialized'), ('spectral', False, 'materialized'), ('spectral', True, 'materialized'), ('spectral', True, 'fused'), ('spectral', True, 'shell_edges')]:
        params = dict(threads=1, random='uniform', sampling={'s': [4., 8., 12.]},
                      binning_window=dict(type='thick_shell', len_args={'R': None, 'delta_R': 4.},
                                          mapping='s_to_R', kernel_mode='octant'),
                      pair_reduction=method, reuse_input_fields=reuse, spectral_window=window_mode,
                      products=['dd', 'dr', 'rd', 'delta_dd', 'rr', 'xi'])
        current = Corr_2PCF({'Corr_2PCF': params})
        current.prepare_input_fields(sfc_field1=first, sfc_field2=second, random1=random1, random2=random2)
        output = current.run(save_result=False)
        error = None
        if COMM.Get_rank() == 0:
            try:
                values = {name: getattr(output, name).copy() for name in params['products']}
                if results:
                    for name, value in values.items():
                        np.testing.assert_allclose(value, results[0][name], rtol=1e-8, atol=1e-11)
                results.append(values)
                metrics = current.runtime_metrics['per_rank']
                if method == 'spectral':
                    assert sum(row['legacy_inverse_fft_count'] for row in metrics) == 0
                    if not nonuniform and reuse:
                        # dd and delta_dd are two different fields, each reused across bins.
                        assert sum(row['forward_fft_count'] for row in metrics) == 2*COMM.Get_size()
            except (AssertionError, ValueError) as exc:
                error = repr(exc)
        error = COMM.bcast(error, root=0)
        assert error is None, error


@pytest.mark.parametrize('path', ['fused', 'shell_edges'])
def test_window_reduction_more_ranks_than_bins(path):
    from test_corr2pcf_window_reduction import explicit_edges
    first, second = field(750), field(751)
    saved = None
    for mode in ['materialized', path]:
        params = dict(threads=1, random='uniform', sampling={'s': [0.]},
            binning_window=dict(type='binned_shell', len_args=['R_in', 'R_out'], mapping=explicit_edges, kernel_mode='octant'),
            pair_reduction='spectral', spectral_window=mode, reuse_input_fields=True,
            products=['dd', 'delta_dd', 'rr', 'xi'])
        current=Corr_2PCF({'Corr_2PCF': params})
        current.prepare_input_fields(sfc_field1=first, sfc_field2=second)
        output=current.run(save_result=False)
        error=None
        if COMM.Get_rank()==0:
            try:
                if saved is None:
                    saved=output.xi.copy()
                else:
                    np.testing.assert_allclose(saved, output.xi, rtol=1e-8, atol=1e-11)
                    rows=current.runtime_metrics['per_rank']
                    if path=='shell_edges':
                        assert sum(row['shell_edge_count'] for row in rows)==4  # two edges x two powers
                        assert sum(row['shell_result_count'] for row in rows)==2
            except (AssertionError, ValueError) as exc:
                error=repr(exc)
        assert COMM.bcast(error, root=0) is None
