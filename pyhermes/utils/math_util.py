"""Compatibility facade for PyHermes mathematical utilities.

New code should import from the narrower modules in ``pyhermes.utils``:

- ``wavelet_grid`` for wavelet samples, particle projection, and spectra.
- ``convolution`` for FFT convolutions and generic window arrays.
- ``special_functions`` for Numba-compatible special functions.
- ``legendre_windows`` for multipole window kernels.
- ``corr3pcf_kernels`` for real-space 3PCF estimator kernels.
- ``corr3pcf_multipoles`` for multipole convolution and CUDA summation.
"""

import warnings

from numba.core.errors import NumbaExperimentalFeatureWarning

warnings.filterwarnings("ignore", category=NumbaExperimentalFeatureWarning)
warnings.warn(
    "pyhermes.utils.math_util is deprecated and will be removed in a future release. "
    "Import from pyhermes.utils.runtime, wavelet_grid, convolution, special_functions, "
    "legendre_windows, corr3pcf_kernels, or corr3pcf_multipoles instead.",
    FutureWarning,
    stacklevel=2,
)


from pyhermes.utils.convolution import (  # noqa: E402
    calculate_w_numba,
    calculate_real_window_octant_array_numba,
    call_calculate_window_array,
    specialized_convolution_3d,
    specialized_convolution_3d_complex,
)
from pyhermes.utils.corr3pcf_kernels import (  # noqa: E402
    estimate_triplet_contrast_particle_centers_legacy,
    estimate_triplet_product_box_random_centers,
    estimate_triplet_product_particle_centers,
    generate_triangle_offsets,
    third_side,
    third_side_from_mu,
)
from pyhermes.utils.corr3pcf_multipoles import (  # noqa: E402
    REDUCE_THREADS,
    _cache_file_path,
    _prepare_legendre_convolution_context,
    _prepare_multipole_gpu_context,
    _stream_convolution_fields,
    cal_gamma,
    calc_DDD_multipole,
    combine_multipole_m_terms,
    compute_3d_result_gpu,
    compute_multipole_m_summand,
    reduce_complex_sum_kernel,
)
from pyhermes.utils.legendre_windows import (  # noqa: E402
    calculate_legendre_window_array,
    calculate_legendre_window_array_numba,
    window_function_legendre,
    window_function_legendre_numba,
)
from pyhermes.utils.runtime import configure  # noqa: E402
from pyhermes.utils.special_functions import (  # noqa: E402
    _angles_from_k,
    _factorial_small,
    _k_norm,
    _phase_from_kR,
    assoc_legendre_numba,
    build_mixing_matrix,
    j0_numba,
    j1_numba,
    jn_numba,
    legendre_triple_coeff,
    solve_multipoles_from_ratio,
    spherical_harmonic_numba,
    spherical_jn_numba,
)
from pyhermes.utils.wavelet_grid import (  # noqa: E402
    bit,
    do_wavelet,
    int_data,
    n_at_pos_numba,
    partition_data_single,
    phi_at_pos_numba,
    power_spectrum,
    random_points_box,
    scaling_function_numba,
    scaling_function_numba_part,
    spectrum_vectorized,
)


__all__ = [
    "configure",
    "do_wavelet",
    "random_points_box",
    "calculate_real_window_octant_array_numba",
    "call_calculate_window_array",
    "calculate_w_numba",
    "scaling_function_numba",
    "int_data",
    "bit",
    "scaling_function_numba_part",
    "partition_data_single",
    "specialized_convolution_3d",
    "specialized_convolution_3d_complex",
    "power_spectrum",
    "spectrum_vectorized",
    "phi_at_pos_numba",
    "n_at_pos_numba",
    "third_side",
    "third_side_from_mu",
    "generate_triangle_offsets",
    "estimate_triplet_product_particle_centers",
    "estimate_triplet_product_box_random_centers",
    "estimate_triplet_contrast_particle_centers_legacy",
    "_k_norm",
    "_phase_from_kR",
    "_angles_from_k",
    "_factorial_small",
    "spherical_jn_numba",
    "j0_numba",
    "j1_numba",
    "jn_numba",
    "assoc_legendre_numba",
    "spherical_harmonic_numba",
    "window_function_legendre_numba",
    "window_function_legendre",
    "calculate_legendre_window_array_numba",
    "calculate_legendre_window_array",
    "cal_gamma",
    "compute_3d_result_gpu",
    "REDUCE_THREADS",
    "reduce_complex_sum_kernel",
    "combine_multipole_m_terms",
    "_cache_file_path",
    "_prepare_legendre_convolution_context",
    "_stream_convolution_fields",
    "_prepare_multipole_gpu_context",
    "compute_multipole_m_summand",
    "calc_DDD_multipole",
    "legendre_triple_coeff",
    "build_mixing_matrix",
    "solve_multipoles_from_ratio",
]
