"""Wavelet scaling-function grids, particle projection, and spectra."""

import math

import numpy as np
import pywt
from numba import njit, prange


def sample_scaling_function(wavelet_name="db2", level=10):
    """Return sampled scaling-function values for a PyWavelets wavelet."""
    wavelet = pywt.Wavelet(wavelet_name)
    phi_values, _, _ = wavelet.wavefun(level=level)
    return phi_values[:-1]


def random_points_box(count=None, box_size=None, ndim=3, rng=None, seed=None, N=None):
    """Draw uniformly distributed random points inside a periodic box."""
    if count is None:
        count = N
    if rng is None:
        rng = np.random.default_rng(seed=seed)
    return rng.uniform(0.0, box_size, size=(count, ndim))


@njit
def _project_scaling_grid_impl(
        positions,
        weights,
        phi_array,
        output_x,
        base_x,
        x_padding,
        phi_resolution,
        grid_power,
        box_size):
    grid_size = 1 << grid_power
    phi_support = phi_array.shape[0] // phi_resolution
    scale_factor = grid_size / box_size
    phi_offsets = np.arange(phi_support, dtype=np.int32) * phi_resolution
    scaled_pos = positions[:, :3] * scale_factor
    coarse_pos = np.floor(scaled_pos).astype(np.int32)
    frac_index = ((scaled_pos - coarse_pos) * phi_resolution).astype(np.int32)
    particle_count = positions.shape[0]
    scaling_grid = np.zeros((output_x, grid_size, grid_size), dtype=np.float64)
    weights = weights.astype(np.float64)
    for particle_idx in range(particle_count):
        weight = weights[particle_idx]
        x_cell = coarse_pos[particle_idx, 0]
        y_cell = coarse_pos[particle_idx, 1]
        z_cell = coarse_pos[particle_idx, 2]
        x_frac = frac_index[particle_idx, 0]
        y_frac = frac_index[particle_idx, 1]
        z_frac = frac_index[particle_idx, 2]
        for x_stencil in range(phi_support):
            phi_x = weight * phi_array[x_frac + phi_offsets[x_stencil]]
            local_x = (x_cell - x_stencil) - base_x + x_padding
            for y_stencil in range(phi_support):
                phi_xy = phi_x * phi_array[y_frac + phi_offsets[y_stencil]]
                for z_stencil in range(phi_support):
                    scaling_grid[local_x, y_cell - y_stencil, z_cell - z_stencil] += (
                        phi_xy * phi_array[z_frac + phi_offsets[z_stencil]]
                    )
    return scaling_grid


@njit
def project_scaling_grid_numba(
        positions,
        weights,
        phi_array,
        phi_resolution=1024,
        J=8,
        box_size=1000.0):
    """Project weighted particles onto the full 3D scaling-function grid."""
    grid_size = 1 << J
    return _project_scaling_grid_impl(
        positions, weights, phi_array, grid_size, 0, 0, phi_resolution, J, box_size
    )


@njit
def project_scaling_slab_numba(
        slab_index,
        positions,
        weights,
        phi_array,
        core_width,
        phi_resolution=1024,
        J=8,
        box_size=1000.0):
    """Project weighted particles onto one x-slab plus scaling-function padding."""
    phi_support = phi_array.shape[0] // phi_resolution
    pad_width = phi_support - 1
    slab_width = core_width + 2 * pad_width
    base_x = slab_index * core_width
    return _project_scaling_grid_impl(
        positions, weights, phi_array, slab_width, base_x, pad_width,
        phi_resolution, J, box_size
    )


def fourier_power_spectrum(values, k_min, k_max, k_count, phi_resolution):
    """Return the squared magnitude of the sampled 1D Fourier spectrum."""
    spectrum = fourier_spectrum_vectorized(values, k_min, k_max, k_count, phi_resolution)
    power = np.zeros(k_count + 1, dtype=np.double)
    for k_index in range(k_count + 1):
        power[k_index] = spectrum[2 * k_index] ** 2 + spectrum[2 * k_index + 1] ** 2
    return power


def fourier_spectrum_vectorized(values, k_min, k_max, k_count, phi_resolution):
    """Compute interleaved real/imaginary Fourier samples of a 1D vector."""
    sample_count = values.shape[0]
    x_min = 0
    x_max = values.shape[0] / phi_resolution
    dx = (x_max - x_min) / sample_count
    dk = (k_max - k_min) / k_count
    x_samples = np.arange(sample_count) * dx
    k_samples = np.arange(k_count + 1) * dk
    x_grid, k_grid = np.meshgrid(x_samples, k_samples)
    real_part = np.sum(values * dx * np.cos(-2 * np.pi * k_grid * x_grid), axis=1)
    imag_part = np.sum(values * dx * np.sin(-2 * np.pi * k_grid * x_grid), axis=1)
    spectrum = np.empty((k_count + 1) * 2, dtype=np.double)
    spectrum[::2] = real_part
    spectrum[1::2] = imag_part
    return spectrum


@njit(parallel=True)
def scaling_stencil_at_pos_numba(positions, phi_array, scale_factor, phi_resolution, phi_support):
    """Evaluate local scaling-function stencil values around each position."""
    phi_offsets = np.arange(phi_support) * phi_resolution
    scaled_pos = positions * scale_factor
    coarse_pos = np.floor(scaled_pos).astype(np.int32)
    frac_index = ((scaled_pos - coarse_pos) * phi_resolution).astype(np.int32)
    point_count = scaled_pos.shape[0]
    local_phi = np.zeros((point_count, phi_support, phi_support, phi_support), dtype=np.float64)
    for point_idx in prange(point_count):
        x_frac, y_frac, z_frac = frac_index[point_idx]
        for x_stencil in range(phi_support):
            phi_x = phi_array[x_frac + phi_offsets[x_stencil]]
            for y_stencil in range(phi_support):
                phi_xy = phi_x * phi_array[y_frac + phi_offsets[y_stencil]]
                for z_stencil in range(phi_support):
                    local_phi[point_idx, -x_stencil, -y_stencil, -z_stencil] = (
                        phi_xy * phi_array[z_frac + phi_offsets[z_stencil]]
                    )
    return coarse_pos, local_phi


@njit(parallel=True)
def interpolate_grid_at_pos_numba(
        output_values,
        scaled_positions,
        grid_values,
        phi_array,
        grid_size,
        phi_resolution,
        phi_support,
        dx=0.0,
        dy=0.0,
        dz=0.0):
    """Evaluate normalized ``n(x)`` on scaled grid positions, with optional offsets."""
    grid_mask = grid_size - 1
    point_count = scaled_positions.shape[0]

    for point_idx in prange(point_count):
        scaled_x = scaled_positions[point_idx, 0] + dx
        scaled_y = scaled_positions[point_idx, 1] + dy
        scaled_z = scaled_positions[point_idx, 2] + dz

        x_cell = int(math.floor(scaled_x))
        y_cell = int(math.floor(scaled_y))
        z_cell = int(math.floor(scaled_z))

        x_frac = int((scaled_x - x_cell) * phi_resolution)
        y_frac = int((scaled_y - y_cell) * phi_resolution)
        z_frac = int((scaled_z - z_cell) * phi_resolution)

        interpolated = 0.0
        for x_stencil in range(phi_support):
            x_grid = (x_cell - x_stencil) & grid_mask
            phi_x = phi_array[x_frac + x_stencil * phi_resolution]
            for y_stencil in range(phi_support):
                y_grid = (y_cell - y_stencil) & grid_mask
                phi_y = phi_array[y_frac + y_stencil * phi_resolution]
                for z_stencil in range(phi_support):
                    z_grid = (z_cell - z_stencil) & grid_mask
                    phi_z = phi_array[z_frac + z_stencil * phi_resolution]
                    interpolated += (
                        grid_values[x_grid, y_grid, z_grid] * phi_x * phi_y * phi_z
                    )

        output_values[point_idx] = interpolated
