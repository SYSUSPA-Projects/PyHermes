"""Numba-compatible special functions used by PyHermes kernels."""

import math

import numpy as np
from numba import njit
from scipy.special import gammaln


@njit
def _k_norm(ki, kj, kk):
    """Return the Euclidean norm of a 3D k-vector."""
    return math.sqrt(ki * ki + kj * kj + kk * kk)


@njit
def _phase_from_kR(ki, kj, kk, R):
    """Return the Fourier phase 2*pi*|k|*R."""
    return 2.0 * math.pi * _k_norm(ki, kj, kk) * R


@njit
def _angles_from_k(ki, kj, kk):
    """Return |k|, polar angle theta, and azimuth phi for a k-vector."""
    k = _k_norm(ki, kj, kk)
    if k == 0.0:
        return 0.0, 0.0, 0.0

    mu = kk / k
    if mu > 1.0:
        mu = 1.0
    elif mu < -1.0:
        mu = -1.0

    theta = math.acos(mu)
    phi = math.atan2(kj, ki)
    return k, theta, phi


@njit
def _factorial_small(n):
    """Compute n! as float for small nonnegative integer n."""
    out = 1.0
    for i in range(2, n + 1):
        out *= i
    return out


@njit
def spherical_jn_numba(l, x):
    """Compute spherical Bessel j_l(x) with stable small-x and high-l handling."""
    if l < 0:
        return np.nan

    if x == 0.0:
        if l == 0:
            return 1.0
        return 0.0

    ax = abs(x)
    if ax <= 5.0e-2:
        if l == 0:
            x2 = x * x
            return 1.0 - x2 / 6.0 + x2 * x2 / 120.0
        denom = 1.0
        for n in range(1, l + 1):
            denom *= 2.0 * n + 1.0
        x2 = x * x
        return (x ** l / denom) * (
            1.0
            - x2 / (2.0 * (2.0 * l + 3.0))
            + x2 * x2 / (8.0 * (2.0 * l + 3.0) * (2.0 * l + 5.0))
        )

    sx = math.sin(x)
    cx = math.cos(x)
    j0 = sx / x
    if l == 0:
        return j0

    j1 = sx / (x * x) - cx / x
    if l == 1:
        return j1

    if ax < l + 1.0:
        n_stop = l + 40
        jp1 = 0.0
        jcur = 1.0
        j_target = 0.0
        j_down_0 = 0.0
        j_down_1 = 0.0
        # Upward recurrence is unstable here; Miller downward recurrence is
        # normalized against stable closed forms for j0 or j1 after the sweep.
        for n in range(n_stop, 0, -1):
            jm1 = ((2.0 * n + 1.0) / x) * jcur - jp1
            if n - 1 == l:
                j_target = jm1
            if n - 1 == 1:
                j_down_1 = jm1
            if n - 1 == 0:
                j_down_0 = jm1
            jp1 = jcur
            jcur = jm1
            if abs(jcur) > 1.0e200:
                jp1 *= 1.0e-200
                jcur *= 1.0e-200
                j_target *= 1.0e-200
                j_down_0 *= 1.0e-200
                j_down_1 *= 1.0e-200
            elif abs(jcur) < 1.0e-200 and jcur != 0.0:
                jp1 *= 1.0e200
                jcur *= 1.0e200
                j_target *= 1.0e200
                j_down_0 *= 1.0e200
                j_down_1 *= 1.0e200

        if abs(j_down_0) >= abs(j_down_1) or j_down_1 == 0.0:
            return j_target * (j0 / j_down_0)
        return j_target * (j1 / j_down_1)

    jm1 = j0
    jcur = j1
    for ell in range(1, l):
        jnext = ((2.0 * ell + 1.0) / x) * jcur - jm1
        jm1 = jcur
        jcur = jnext

    return jcur


@njit
def j0_numba(x):
    """Compute cylindrical Bessel J_0(x) using Cephes-style approximations."""
    ax = abs(x)
    if ax < 8.0:
        y = x * x
        ans1 = (
            57568490574.0
            + y * (
                -13362590354.0
                + y * (651619640.7 + y * (-11214424.18 + y * (77392.33017 + y * -184.9052456)))
            )
        )
        ans2 = (
            57568490411.0
            + y * (
                1029532985.0
                + y * (9494680.718 + y * (59272.64853 + y * (267.8532712 + y)))
            )
        )
        return ans1 / ans2

    z = 8.0 / ax
    y = z * z
    xx = ax - 0.785398164
    ans1 = (
        1.0
        + y * (
            -0.1098628627e-2
            + y * (0.2734510407e-4 + y * (-0.2073370639e-5 + y * 0.2093887211e-6))
        )
    )
    ans2 = (
        -0.1562499995e-1
        + y * (
            0.1430488765e-3
            + y * (-0.6911147651e-5 + y * (0.7621095161e-6 - y * 0.934945152e-7))
        )
    )
    return math.sqrt(0.636619772 / ax) * (math.cos(xx) * ans1 - z * math.sin(xx) * ans2)


@njit
def j1_numba(x):
    """Compute cylindrical Bessel J_1(x) using Cephes-style approximations."""
    ax = abs(x)
    if ax < 8.0:
        y = x * x
        ans1 = x * (
            72362614232.0
            + y * (
                -7895059235.0
                + y * (242396853.1 + y * (-2972611.439 + y * (15704.48260 + y * -30.16036606)))
            )
        )
        ans2 = (
            144725228442.0
            + y * (
                2300535178.0
                + y * (18583304.74 + y * (99447.43394 + y * (376.9991397 + y)))
            )
        )
        return ans1 / ans2

    z = 8.0 / ax
    y = z * z
    xx = ax - 2.356194491
    ans1 = (
        1.0
        + y * (
            0.183105e-2
            + y * (-0.3516396496e-4 + y * (0.2457520174e-5 + y * -0.240337019e-6))
        )
    )
    ans2 = (
        0.04687499995
        + y * (
            -0.2002690873e-3
            + y * (0.8449199096e-5 + y * (-0.88228987e-6 + y * 0.105787412e-6))
        )
    )
    ans = math.sqrt(0.636619772 / ax) * (math.cos(xx) * ans1 - z * math.sin(xx) * ans2)
    if x < 0.0:
        ans = -ans
    return ans


@njit
def jn_numba(n, x):
    """Compute cylindrical Bessel J_n(x) for integer order n."""
    order = n
    sign = 1.0
    if order < 0:
        order = -order
        if order % 2 == 1:
            sign = -sign
    if x < 0.0:
        x = -x
        if order % 2 == 1:
            sign = -sign

    if order == 0:
        return sign * j0_numba(x)
    if order == 1:
        return sign * j1_numba(x)
    if x == 0.0:
        return 0.0

    tox = 2.0 / x
    if x > order:
        bjm = j0_numba(x)
        bj = j1_numba(x)
        for j in range(1, order):
            bjp = j * tox * bj - bjm
            bjm = bj
            bj = bjp
        return sign * bj

    m = 2 * ((order + int(math.sqrt(40.0 * order))) // 2)
    jsum = False
    bjp = 0.0
    ans = 0.0
    total = 0.0
    bj = 1.0
    for j in range(m, 0, -1):
        bjm = j * tox * bj - bjp
        bjp = bj
        bj = bjm
        if abs(bj) > 1.0e10:
            bj *= 1.0e-10
            bjp *= 1.0e-10
            ans *= 1.0e-10
            total *= 1.0e-10
        if jsum:
            total += bj
        jsum = not jsum
        if j == order:
            ans = bjp
    total = 2.0 * total - bj
    return sign * ans / total


@njit
def assoc_legendre_numba(l, m, x):
    """Compute the associated Legendre function P_l^m(x) for nonnegative m."""
    if l < 0:
        return np.nan
    if m < 0 or m > l:
        return np.nan

    pmm = 1.0
    if m > 0:
        somx2 = math.sqrt(max(0.0, 1.0 - x * x))
        fact = 1.0
        for _ in range(m):
            pmm *= -fact * somx2
            fact += 2.0

    if l == m:
        return pmm

    pmmp1 = x * (2.0 * m + 1.0) * pmm
    if l == m + 1:
        return pmmp1

    pll_minus2 = pmm
    pll_minus1 = pmmp1
    pll = 0.0
    for ell in range(m + 2, l + 1):
        pll = ((2.0 * ell - 1.0) * x * pll_minus1 - (ell + m - 1.0) * pll_minus2) / (ell - m)
        pll_minus2 = pll_minus1
        pll_minus1 = pll

    return pll


@njit
def spherical_harmonic_numba(l, m, ki, kj, kk):
    """Evaluate complex spherical harmonic Y_l^m in the direction of k."""
    if l < 0:
        return np.nan + 0.0j
    if abs(m) > l:
        return 0.0 + 0.0j

    k, theta, phi = _angles_from_k(ki, kj, kk)
    if k == 0.0:
        if l == 0 and m == 0:
            return 1.0 / math.sqrt(4.0 * math.pi) + 0.0j
        return 0.0 + 0.0j

    x = math.cos(theta)
    if m >= 0:
        P = assoc_legendre_numba(l, m, x)
        norm = math.sqrt(
            (2.0 * l + 1.0) / (4.0 * math.pi) *
            _factorial_small(l - m) / _factorial_small(l + m)
        )
        phase = complex(math.cos(m * phi), math.sin(m * phi))
        return norm * P * phase

    mp = -m
    ypos = spherical_harmonic_numba(l, mp, ki, kj, kk)
    sign = -1.0 if (mp % 2 == 1) else 1.0
    return sign * np.conj(ypos)


def legendre_triple_coeff(l: int, lp: int, k: int) -> float:
    """
    Return the Legendre triple-product coupling coefficient.

    The coefficient is
    ``G_{lp,l,k} = (2k+1)/2 * int_-1^1 P_lp(mu) P_l(mu) P_k(mu) dmu``.
    It is nonzero only when the triangle and parity selection rules hold.
    """
    s = l + lp + k

    if (k > l + lp) or (k < abs(l - lp)) or (s % 2 != 0):
        return 0.0

    g = s // 2
    log_val = (
        gammaln(s - 2 * l + 1)
        + gammaln(s - 2 * lp + 1)
        + gammaln(s - 2 * k + 1)
        - gammaln(s + 2)
        + 2 * (
            gammaln(g + 1)
            - gammaln(g - l + 1)
            - gammaln(g - lp + 1)
            - gammaln(g - k + 1)
        )
    )
    return float((2 * k + 1) * np.exp(log_val))


def build_mixing_matrix(B: np.ndarray, lmax: int) -> np.ndarray:
    """Return the multipole mixing matrix for ``A(mu) = B(mu) * C(mu)``."""
    B = np.asarray(B, dtype=np.float64)
    B0 = B[0]

    if B0 == 0:
        raise ValueError("B[0] is zero.")

    f = B / B0
    M = np.zeros((lmax + 1, lmax + 1), dtype=np.float64)

    for l in range(lmax + 1):
        for lp in range(1, lmax + 1):
            if l == lp:
                M[0, l] += f[lp] / (2 * l + 1)

    for k in range(1, lmax + 1):
        for l in range(lmax + 1):
            for lp in range(1, lmax + 1):
                M[k, l] += legendre_triple_coeff(l, lp, k) * f[lp]

    M += np.eye(lmax + 1)
    M *= B0
    return M


def solve_multipoles_from_ratio(A: np.ndarray, B: np.ndarray, lmax: int, rcond_warning: float = 1e12):
    """Return multipoles ``C_l`` for the ratio problem ``A(mu) = B(mu) * C(mu)``."""
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)

    if A.shape[0] < lmax + 1 or B.shape[0] < lmax + 1:
        raise ValueError("A and B must have length at least lmax+1.")

    A_cut = A[:lmax + 1]
    B_cut = B[:lmax + 1]

    M = build_mixing_matrix(B_cut, lmax)
    condM = np.linalg.cond(M)
    if condM > rcond_warning:
        print(f"[warning] mixing matrix is ill-conditioned: cond(M) = {condM:.3e}")

    C = np.linalg.solve(M, A_cut)
    return C, M, condM
