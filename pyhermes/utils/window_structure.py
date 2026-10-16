"""Declared mathematical structure, independent of the 2PCF execution backend.

An explicit custom callable remains authoritative. A custom author may promise
radial symmetry with ``structure='radial'``; no property is inferred from a name.
"""

from dataclasses import dataclass

import numpy as np

from pyhermes.utils.window_functions import (
    axial_cos_factor,
    axial_sinc_factor,
    disk_transverse_factor,
    ring_transverse_factor,
)


@dataclass(frozen=True)
class WindowStructure:
    kind: str = "general"
    transverse: object = None
    longitudinal: object = None
    ball_bounds: object = None
    zero_dc: bool = False


def _explicit_bounds(window):
    return tuple(window.rescale_len_args[key] for key in ("R_in", "R_out"))


def _thick_bounds(window):
    radius, width = (window.rescale_len_args[key] for key in ("R", "delta_R"))
    return max(0.0, radius - width / 2), radius + width / 2


RADIAL = WindowStructure("radial")
GENERAL = WindowStructure()
BUILTIN_STRUCTURES = {
    name: RADIAL
    for name in (
        "sphere",
        "gaussian",
        "shell",
        "gaussian_shell",
        "cw",
        "cws",
        "gdw",
        "laplacian",
    )
}
BUILTIN_STRUCTURES.update(
    {
        "binned_shell": WindowStructure("radial", ball_bounds=_explicit_bounds),
        "thick_shell": WindowStructure("radial", ball_bounds=_thick_bounds),
        "inverse_laplacian": WindowStructure("radial", zero_dc=True),
        "ring": WindowStructure(
            "separable_axial", ring_transverse_factor, axial_cos_factor
        ),
        "disk": WindowStructure(
            "separable_axial", disk_transverse_factor, axial_cos_factor
        ),
        "cylshell": WindowStructure(
            "separable_axial", ring_transverse_factor, axial_sinc_factor
        ),
        "cylinder": WindowStructure(
            "separable_axial", disk_transverse_factor, axial_sinc_factor
        ),
    }
)


def describe_window(window):
    if getattr(window, "has_custom_func", False):
        declaration = window.window_params.get("structure")
        if declaration is None or declaration == "general":
            return GENERAL
        if declaration == "radial":
            return RADIAL
        raise ValueError("Custom window structure must be 'general' or 'radial'.")
    return BUILTIN_STRUCTURES.get(window.type, GENERAL)


def linear_ball_bounds(window):
    provider = describe_window(window).ball_bounds
    if provider is None:
        return None
    inner, outer = provider(window)
    if not (np.isfinite(inner) and np.isfinite(outer) and 0 <= inner < outer):
        return None
    return float(inner), float(outer)


def exact_grid_axis(window):
    """Axis-aligned fast paths need exact zeros, not a tolerance-based rotation."""
    los = np.asarray(
        [
            window.los_args.get(key, default)
            for key, default in [("nx", 0.0), ("ny", 0.0), ("nz", 1.0)]
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(los)) or np.count_nonzero(los) != 1:
        return None
    norm = np.sqrt(np.sum(los * los))
    if not np.isfinite(norm) or norm == 0:
        return None
    axis = int(np.flatnonzero(los)[0])
    return axis if abs(los[axis] / norm) == 1.0 else None
