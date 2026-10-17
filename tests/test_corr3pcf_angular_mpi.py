"""Launch with mpiexec -n4; fixed layout and seeds on both implementations."""

import numpy as np
import pytest
from test_corr3pcf_angular_optimization import ConvolutionPairs, angular_task

from pyhermes.theory.corr3pcf import Corr_3PCF
from pyhermes.utils.mpi_util import MPI

pytestmark = pytest.mark.skipif(MPI.COMM_WORLD.Get_size() < 2, reason="requires real MPI ranks")


@pytest.mark.parametrize("center", ["box_random", "particle"])
@pytest.mark.parametrize("nonuniform", [False, True])
def test_root_only_pair_postprocessing_and_same_sampled_products(center, nonuniform):
    reference = angular_task(ConvolutionPairs, center, nonuniform, True).run(save_result=False)
    task = angular_task(Corr_3PCF, center, nonuniform, True)
    optimized = task.run(save_result=False)
    if MPI.COMM_WORLD.Get_rank() == 0:
        for name in ["ddd", "rrr", "delta_ddd", "xi12", "xi13", "xi23", "zeta", "zeta_H", "Q"]:
            np.testing.assert_allclose(getattr(optimized, name), getattr(reference, name), rtol=3e-10, atol=1e-22)
    assert task._box_delta_fields is None and task._particle_delta_fields is None
