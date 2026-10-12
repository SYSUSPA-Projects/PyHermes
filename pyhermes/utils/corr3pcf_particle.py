"""Particle-centred contractions of reconstructed 3PCF harmonic fields.

This evaluates A_j(x_i) B_j(x_i), not P_j(A_j B_j)(x_i). It therefore
shares the continuum target with the Gamma contraction, but generally has
a different finite-resolution discretization. No centre window is applied.
"""

import numba
import numpy as np


def prepare_particle_catalog(field):
    """Cache the readout stencil and *projection* weights of a catalog field.

    Positions are physical coordinates. The stencil uses the same sampled
    scaling function, floor convention and periodic images as SFC painting.
    Catalogue normalization alone is insufficient for marked/weighted fields;
    get_particle_data supplies the effective weight including the field value.
    """
    if getattr(field, "field_kind", None) != "catalog_field":
        raise ValueError("Particle multipoles require a retrievable catalog field at vertex 1.")
    if not getattr(field, "particle_data_retrievable", True):
        raise ValueError("Vertex 1 no longer has a retrievable particle catalogue.")
    particles = field.get_particle_data()
    positions = np.asarray(particles["pos"], dtype=np.float64)
    weights = np.asarray(particles["projection_weight"], dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("Particle positions must have shape (N, 3).")
    if weights.shape != (len(positions),):
        raise ValueError("Particle projection weights must have shape (N,).")
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(weights)):
        raise ValueError("Particle positions and projection weights must be finite.")
    if field.phi_array.size != field.phi_support * field.phi_resolution:
        raise ValueError("phi_array must contain support*resolution samples.")
    scaled = np.remainder(positions * field.scale_factor, field.L)
    cells = np.floor(scaled).astype(np.int64)
    fraction = ((scaled - cells) * field.phi_resolution).astype(np.int64)
    offsets = np.arange(field.phi_support) * field.phi_resolution
    stencil = field.phi_array[fraction[:, :, None] + offsets]
    return {
        "cells": np.ascontiguousarray(cells),
        "stencil": np.ascontiguousarray(stencil, dtype=np.float64),
        "weights": np.ascontiguousarray(weights),
        "grid_size": int(field.L),
    }


@numba.njit(parallel=True, cache=True)
def particle_product_sum(first, second, cells, stencil, weights):
    """Sum w_i A(x_i)B(x_i); reconstruct the two complex arms separately."""
    length = first.shape[0]
    support = stencil.shape[2]
    real_sum = 0.0
    imag_sum = 0.0
    for i in numba.prange(weights.size):
        a = 0.0j
        b = 0.0j
        for x in range(support):
            ix = (cells[i, 0] - x) % length
            px = stencil[i, 0, x]
            for y in range(support):
                iy = (cells[i, 1] - y) % length
                pxy = px * stencil[i, 1, y]
                for z in range(support):
                    iz = (cells[i, 2] - z) % length
                    phi = pxy * stencil[i, 2, z]
                    a += phi * first[ix, iy, iz]
                    b += phi * second[ix, iy, iz]
        value = weights[i] * a * b
        real_sum += value.real
        imag_sum += value.imag
    return complex(real_sum, imag_sum)


@numba.njit(parallel=True, cache=True)
def bilinear_product_sum(first, second):
    """Integral of two orthonormal scaling expansions in grid coordinates.

    There is deliberately no complex conjugation: the caller already supplies
    harmonic m and -m fields, matching the Gamma path's phase convention.
    """
    a = first.ravel()
    b = second.ravel()
    real_sum = 0.0
    imag_sum = 0.0
    for i in numba.prange(a.size):
        value = a[i] * b[i]
        real_sum += value.real
        imag_sum += value.imag
    return complex(real_sum, imag_sum)


def compute_particle_summand(first, second, spec):
    """Contract particle measures plus an optional signed uniform density.

    For delta_ddd the centre is data minus random, including a negative
    uniform term when random='uniform'. The result here is unnormalized;
    4*pi / L**3 and the m-combination stay in the existing multipole helpers.
    """
    length = spec["grid_size"]
    a = np.ascontiguousarray(first, dtype=np.complex128)
    b = np.ascontiguousarray(second, dtype=np.complex128)
    if a.shape != (length,) * 3 or b.shape != a.shape:
        raise ValueError("Particle summation requires two matching cubic coefficient grids.")
    result = 0.0j
    for component, factor in spec["components"]:
        result += factor * particle_product_sum(
            a, b, component["cells"], component["stencil"], component["weights"]
        )
    density = spec["uniform_density"]
    if density != 0.0:
        result += density * bilinear_product_sum(a, b)
    return result
