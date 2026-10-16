"""The original Figure13 volume-normalised finite spherical shell."""
import numpy as np
from numba import njit
from pyhermes.utils.window_functions import window_function_sphere_numba

EDGES = np.linspace(5., 150., 32)
CENTRES = (EDGES[:-1] + EDGES[1:]) / 2


@njit(cache=True)
def binned_shell(ki, kj, kk, R_in, R_out):
    inner, outer = R_in**3, R_out**3
    return (outer * window_function_sphere_numba(ki, kj, kk, R_out)
            - inner * window_function_sphere_numba(ki, kj, kk, R_in)) / (outer - inner)


def mapping(sample, window):
    index = int(np.argmin(np.abs(CENTRES - float(sample['s']))))
    result = dict(window)
    result['len_args'] = dict(R_in=float(EDGES[index]), R_out=float(EDGES[index + 1]))
    return result


WINDOW = dict(type='binned_shell', func=binned_shell,
              len_args=['R_in', 'R_out'], mapping=mapping, kernel_mode='octant')
