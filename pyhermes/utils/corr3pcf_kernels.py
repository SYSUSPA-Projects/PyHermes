"""Real-space 3PCF geometry and Monte Carlo estimator kernels."""

import math

import numpy as np
from numba import njit

from pyhermes.utils.wavelet_grid import interpolate_grid_at_pos_numba


def third_side(r1, r2, theta):
    """Return r23 from (r1, r2, theta) using the law of cosines."""
    return np.sqrt(r1**2 + r2**2 - 2.0 * r1 * r2 * np.cos(theta))


def third_side_from_mu(r1, r2, mu):
    """Return r23 from (r1, r2, mu) with mu = cos(theta)."""
    return np.sqrt(np.clip(r1**2 + r2**2 - 2.0 * r1 * r2 * mu, 0.0, None))


@njit
def generate_triangle_offsets(R1, R2, mu, phi, costheta1, alpha):
    """Generate two grid offsets forming a triangle with side lengths R1 and R2."""
    sintheta1 = math.sqrt(1.0 - costheta1 * costheta1)
    dx1 = sintheta1 * math.cos(phi)
    dy1 = sintheta1 * math.sin(phi)
    dz1 = costheta1

    x2 = R1 * dx1
    y2 = R1 * dy1
    z2 = R1 * dz1

    if abs(dx1) > 1e-10 or abs(dy1) > 1e-10:
        ox1, oy1, oz1 = -dy1, dx1, 0.0
    else:
        ox1, oy1, oz1 = 0.0, -dz1, dy1

    norm = math.sqrt(ox1 * ox1 + oy1 * oy1 + oz1 * oz1)
    ox1 /= norm
    oy1 /= norm
    oz1 /= norm

    ox2 = dy1 * oz1 - dz1 * oy1
    oy2 = dz1 * ox1 - dx1 * oz1
    oz2 = dx1 * oy1 - dy1 * ox1

    cos_alpha = math.cos(alpha)
    sin_alpha = math.sin(alpha)
    cos_theta = mu
    sin_theta_sq = 1.0 - mu * mu
    if sin_theta_sq < 0.0:
        sin_theta_sq = 0.0
    sin_theta = math.sqrt(sin_theta_sq)

    dx2 = cos_theta * dx1 + sin_theta * (cos_alpha * ox1 + sin_alpha * ox2)
    dy2 = cos_theta * dy1 + sin_theta * (cos_alpha * oy1 + sin_alpha * oy2)
    dz2 = cos_theta * dz1 + sin_theta * (cos_alpha * oz1 + sin_alpha * oz2)

    x3 = R2 * dx2
    y3 = R2 * dy2
    z3 = R2 * dz2

    return x2, y2, z2, x3, y3, z3


@njit
def estimate_triplet_product_particle_centers(
    R1_scaled, R2_scaled, mu,
    center_scaled, center_weight, center_weight_sum, n_rot,
    rho1, epsilon2, epsilon3,
    phi_array, L, phi_resolution, phi_support,
    seed_base_rot=-1, theta_index=-1
):
    """Estimate DDD using object positions as triangle centers."""
    npos = center_scaled.shape[0]
    two_pi = 2.0 * math.pi
    n2 = np.empty(npos, dtype=np.float64)
    n3 = np.empty(npos, dtype=np.float64)
    total_sum = 0.0
    for irot in range(n_rot):
        if seed_base_rot >= 0:
            seed_rot = seed_base_rot + irot
            if theta_index >= 0:
                seed_rot += theta_index * 1000003
            np.random.seed(seed_rot)
        a, b, c = np.random.rand(3)
        phi = two_pi * a
        costheta1 = 2.0 * b - 1.0
        alpha = two_pi * c
        x2, y2, z2, x3, y3, z3 = generate_triangle_offsets(R1_scaled, R2_scaled, mu, phi, costheta1, alpha)
        interpolate_grid_at_pos_numba(n2, center_scaled, epsilon2, phi_array, L, phi_resolution, phi_support, x2, y2, z2)
        interpolate_grid_at_pos_numba(n3, center_scaled, epsilon3, phi_array, L, phi_resolution, phi_support, x3, y3, z3)
        s = 0.0
        for i in range(npos):
            s += center_weight[i] * n2[i] * n3[i]
        total_sum += s
    return rho1 * total_sum / (n_rot * center_weight_sum)


@njit
def estimate_triplet_product_box_random_centers(
    R1_scaled, R2_scaled, mu,
    center_scaled, n_rot,
    epsilon1, epsilon2, epsilon3,
    phi_array, L, phi_resolution, phi_support,
    seed_base_rot=-1, theta_index=-1
):
    """Estimate DDD by averaging over uniform random centers in the periodic box."""
    npos = center_scaled.shape[0]
    two_pi = 2.0 * math.pi

    n1 = np.empty(npos, dtype=np.float64)
    n2 = np.empty(npos, dtype=np.float64)
    n3 = np.empty(npos, dtype=np.float64)

    interpolate_grid_at_pos_numba(n1, center_scaled, epsilon1, phi_array, L, phi_resolution, phi_support)
    total_sum = 0.0

    for irot in range(n_rot):
        if seed_base_rot >= 0:
            seed_rot = seed_base_rot + irot
            if theta_index >= 0:
                seed_rot += theta_index * 1000003
            np.random.seed(seed_rot)
        a, b, c = np.random.rand(3)
        phi = two_pi * a
        costheta1 = 2.0 * b - 1.0
        alpha = two_pi * c

        x2, y2, z2, x3, y3, z3 = generate_triangle_offsets(R1_scaled, R2_scaled, mu, phi, costheta1, alpha)

        interpolate_grid_at_pos_numba(n2, center_scaled, epsilon2, phi_array, L, phi_resolution, phi_support, x2, y2, z2)
        interpolate_grid_at_pos_numba(n3, center_scaled, epsilon3, phi_array, L, phi_resolution, phi_support, x3, y3, z3)

        s = 0.0
        for i in range(npos):
            s += n1[i] * n2[i] * n3[i]
        total_sum += s

    return total_sum / (n_rot * npos)


@njit
def estimate_triplet_contrast_particle_centers_legacy(
    R1_scaled, R2_scaled, mu,
    pos_scaled, rand_scaled, n_rot,
    rho1, epsilon2, epsilon3,
    phi_array, L, phi_resolution, phi_support,
    seed=-1
):
    """Estimate the legacy d_delta_dd control-variate quantity."""
    if seed >= 0:
        np.random.seed(seed)
    npos = pos_scaled.shape[0]
    two_pi = 2.0 * math.pi
    n2 = np.empty(npos, dtype=np.float64)
    n3 = np.empty(npos, dtype=np.float64)
    n20 = np.empty(npos, dtype=np.float64)
    n30 = np.empty(npos, dtype=np.float64)
    total_sum = 0.0
    for _ in range(n_rot):
        a, b, c = np.random.rand(3)
        phi = two_pi * a
        costheta1 = 2.0 * b - 1.0
        alpha = two_pi * c
        x2, y2, z2, x3, y3, z3 = generate_triangle_offsets(R1_scaled, R2_scaled, mu, phi, costheta1, alpha)
        interpolate_grid_at_pos_numba(n2, pos_scaled, epsilon2, phi_array, L, phi_resolution, phi_support, x2, y2, z2)
        interpolate_grid_at_pos_numba(n3, pos_scaled, epsilon3, phi_array, L, phi_resolution, phi_support, x3, y3, z3)
        interpolate_grid_at_pos_numba(n20, rand_scaled, epsilon2, phi_array, L, phi_resolution, phi_support, x2, y2, z2)
        interpolate_grid_at_pos_numba(n30, rand_scaled, epsilon3, phi_array, L, phi_resolution, phi_support, x3, y3, z3)
        s = 0.0
        for i in range(npos):
            s += n2[i] * n3[i] - n20[i] * n30[i]
        total_sum += s
    return rho1 * total_sum / (n_rot * npos)
