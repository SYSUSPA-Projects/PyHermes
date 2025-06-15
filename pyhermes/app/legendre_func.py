import numpy as np
from numba import njit


@njit
def window_function_legendre_l0_m0_numba(ki, kj, kk, R):
    k = np.sqrt(ki**2 + kj**2 + kk**2)
    if k == 0:
        return 1
    # Use np.where to handle the k == 0 case
    Phase = 2 * np.pi * k * R
    result = np.sin(Phase) / Phase / np.sqrt(4 * np.pi)
    return result


@njit
def window_function_legendre_l1_m_minus1_numba(ki, kj, kk, R):
    k = np.sqrt(ki**2 + kj**2 + kk**2)
    if k == 0:
        return 0
    # Spherical harmonics Y_1^-1 proportional to sin(theta) * exp(-i*phi)
    sin_theta = np.sqrt(ki**2 + kj**2) / k
    phi = np.arctan2(kj, ki)  # Azimuthal angle
    Phase = 2 * np.pi * k * R
    j1 = np.sin(Phase) / Phase**2 - np.cos(Phase) / Phase
    result = j1 * sin_theta * np.exp(-1j * phi) * np.sqrt(3 / (8 * np.pi))
    return result


@njit
def window_function_legendre_l1_m0_numba(ki, kj, kk, R):
    k = np.sqrt(ki**2 + kj**2 + kk**2)
    if k == 0:
        return 0
    # Spherical harmonics Y_1^0 proportional to cos(theta)
    cos_theta = kk / k  # Direction cosine along z-axis
    Phase = 2 * np.pi * k * R
    j1 = np.sin(Phase) / Phase**2 - np.cos(Phase) / Phase
    result = j1 * cos_theta * np.sqrt(3 / (4 * np.pi))
    return result


@njit
def window_function_legendre_l1_m1_numba(ki, kj, kk, R):
    k = np.sqrt(ki**2 + kj**2 + kk**2)
    if k == 0:
        return 0
    # Spherical harmonics Y_1^1 proportional to sin(theta) * exp(i*phi)
    sin_theta = np.sqrt(ki**2 + kj**2) / k
    phi = np.arctan2(kj, ki)  # Azimuthal angle
    Phase = 2 * np.pi * k * R
    j1 = np.sin(Phase) / Phase**2 - np.cos(Phase) / Phase
    result = j1 * sin_theta * np.exp(1j * phi) * (-np.sqrt(3 / (8 * np.pi)))
    return result


@njit
def window_function_legendre_l2_m_minus2_numba(ki, kj, kk, R):
    k = np.sqrt(ki**2 + kj**2 + kk**2)
    if k == 0:
        return 0
    sin_theta_squared = (ki**2 + kj**2) / k**2
    phi = np.arctan2(kj, ki)  # Azimuthal angle
    Phase = 2 * np.pi * k * R
    j2 = (3 / Phase**2 - 1) * np.sin(Phase) / Phase - 3 * np.cos(Phase) / Phase**2
    result = j2 * sin_theta_squared * np.exp(-2j * phi) * np.sqrt(15 / (32 * np.pi))
    return result


@njit
def window_function_legendre_l2_m_minus1_numba(ki, kj, kk, R):
    k = np.sqrt(ki**2 + kj**2 + kk**2)
    if k == 0:
        return 0
    sin_theta = np.sqrt((ki**2 + kj**2)) / k
    cos_theta = kk / k
    phi = np.arctan2(kj, ki)  # Azimuthal angle
    Phase = 2 * np.pi * k * R
    j2 = (3 / Phase**2 - 1) * np.sin(Phase) / Phase - 3 * np.cos(Phase) / Phase**2
    result = j2 * sin_theta * cos_theta * np.exp(-1j * phi) * np.sqrt(15 / (8 * np.pi))
    return result


@njit
def window_function_legendre_l2_m0_numba(ki, kj, kk, R):
    k = np.sqrt(ki**2 + kj**2 + kk**2)
    if k == 0:
        return 0
    cos_theta = kk / k
    Phase = 2 * np.pi * k * R
    j2 = (3 / Phase**2 - 1) * np.sin(Phase) / Phase - 3 * np.cos(Phase) / Phase**2
    result = j2 * (3 * cos_theta**2 - 1) * np.sqrt(5 / (16 * np.pi))
    return result


@njit
def window_function_legendre_l2_m1_numba(ki, kj, kk, R):
    k = np.sqrt(ki**2 + kj**2 + kk**2)
    if k == 0:
        return 0
    sin_theta = np.sqrt((ki**2 + kj**2)) / k
    cos_theta = kk / k
    phi = np.arctan2(kj, ki)  # Azimuthal angle
    Phase = 2 * np.pi * k * R
    j2 = (3 / Phase**2 - 1) * np.sin(Phase) / Phase - 3 * np.cos(Phase) / Phase**2
    result = j2 * sin_theta * cos_theta * np.exp(1j * phi) * (-np.sqrt(15 / (8 * np.pi)))
    return result


@njit
def window_function_legendre_l2_m2_numba(ki, kj, kk, R):
    k = np.sqrt(ki**2 + kj**2 + kk**2)
    if k == 0:
        return 0
    sin_theta_squared = (ki**2 + kj**2) / k**2
    phi = np.arctan2(kj, ki)  # Azimuthal angle
    Phase = 2 * np.pi * k * R
    j2 = (3 / Phase**2 - 1) * np.sin(Phase) / Phase - 3 * np.cos(Phase) / Phase**2
    result = j2 * sin_theta_squared * np.exp(2j * phi) * np.sqrt(15 / (32 * np.pi))
    return result


@njit
def window_function_legendre_l3_m_minus3_numba(ki, kj, kk, R):
    k = np.sqrt(ki**2 + kj**2 + kk**2)
    if k == 0:
        return 0
    sin_theta_cubed = (np.sqrt(ki**2 + kj**2) / k) ** 3
    phi = np.arctan2(kj, ki)
    Phase = 2 * np.pi * k * R
    j3 = (15 / Phase**3 - 6 / Phase) * np.sin(Phase) / Phase - (15 / Phase**2 - 1) * np.cos(Phase) / Phase
    result = j3 * sin_theta_cubed * np.exp(-3j * phi) * np.sqrt(35 / (64 * np.pi))
    return result


@njit
def window_function_legendre_l3_m_minus2_numba(ki, kj, kk, R):
    k = np.sqrt(ki**2 + kj**2 + kk**2)
    if k == 0:
        return 0
    sin_theta_squared = (ki**2 + kj**2) / k**2
    cos_theta = kk / k
    phi = np.arctan2(kj, ki)
    Phase = 2 * np.pi * k * R
    j3 = (15 / Phase**3 - 6 / Phase) * np.sin(Phase) / Phase - (15 / Phase**2 - 1) * np.cos(Phase) / Phase
    result = j3 * sin_theta_squared * cos_theta * np.exp(-2j * phi) * np.sqrt(105 / (32 * np.pi))
    return result


@njit
def window_function_legendre_l3_m_minus1_numba(ki, kj, kk, R):
    k = np.sqrt(ki**2 + kj**2 + kk**2)
    if k == 0:
        return 0
    sin_theta = np.sqrt(ki**2 + kj**2) / k
    cos_theta_squared = (kk / k) ** 2
    phi = np.arctan2(kj, ki)
    Phase = 2 * np.pi * k * R
    j3 = (15 / Phase**3 - 6 / Phase) * np.sin(Phase) / Phase - (15 / Phase**2 - 1) * np.cos(Phase) / Phase
    result = j3 * sin_theta * (5 * cos_theta_squared - 1) * np.exp(-1j * phi) * np.sqrt(21 / (64 * np.pi))
    return result


@njit
def window_function_legendre_l3_m0_numba(ki, kj, kk, R):
    k = np.sqrt(ki**2 + kj**2 + kk**2)
    if k == 0:
        return 0
    cos_theta = kk / k
    Phase = 2 * np.pi * k * R
    j3 = (15 / Phase**3 - 6 / Phase) * np.sin(Phase) / Phase - (15 / Phase**2 - 1) * np.cos(Phase) / Phase
    result = j3 * (5 * cos_theta**3 - 3 * cos_theta) * np.sqrt(7 / (16 * np.pi))
    return result


@njit
def window_function_legendre_l3_m1_numba(ki, kj, kk, R):
    k = np.sqrt(ki**2 + kj**2 + kk**2)
    if k == 0:
        return 0
    sin_theta = np.sqrt(ki**2 + kj**2) / k
    cos_theta_squared = (kk / k) ** 2
    phi = np.arctan2(kj, ki)
    Phase = 2 * np.pi * k * R
    j3 = (15 / Phase**3 - 6 / Phase) * np.sin(Phase) / Phase - (15 / Phase**2 - 1) * np.cos(Phase) / Phase
    result = j3 * sin_theta * (5 * cos_theta_squared - 1) * np.exp(1j * phi) * (-np.sqrt(21 / (64 * np.pi)))
    return result


@njit
def window_function_legendre_l3_m2_numba(ki, kj, kk, R):
    k = np.sqrt(ki**2 + kj**2 + kk**2)
    if k == 0:
        return 0
    sin_theta_squared = (ki**2 + kj**2) / k**2
    cos_theta = kk / k
    phi = np.arctan2(kj, ki)
    Phase = 2 * np.pi * k * R
    j3 = (15 / Phase**3 - 6 / Phase) * np.sin(Phase) / Phase - (15 / Phase**2 - 1) * np.cos(Phase) / Phase
    result = j3 * sin_theta_squared * cos_theta * np.exp(2j * phi) * np.sqrt(105 / (32 * np.pi))
    return result


@njit
def window_function_legendre_l3_m3_numba(ki, kj, kk, R):
    k = np.sqrt(ki**2 + kj**2 + kk**2)
    if k == 0:
        return 0
    sin_theta_cubed = (np.sqrt(ki**2 + kj**2) / k) ** 3
    phi = np.arctan2(kj, ki)
    Phase = 2 * np.pi * k * R
    j3 = (15 / Phase**3 - 6 / Phase) * np.sin(Phase) / Phase - (15 / Phase**2 - 1) * np.cos(Phase) / Phase
    result = j3 * sin_theta_cubed * np.exp(3j * phi) * (-np.sqrt(35 / (64 * np.pi)))
    return result


@njit
def window_function_legendre_l4_m_minus4_numba(ki, kj, kk, R):
    k = np.sqrt(ki**2 + kj**2 + kk**2)
    if k == 0:
        return 0
    sin_theta_fourth = (np.sqrt(ki**2 + kj**2) / k) ** 4
    phi = np.arctan2(kj, ki)
    Phase = 2 * np.pi * k * R
    j4 = (
        5 * Phase * (-21 + 2 * Phase**2) * np.cos(Phase) + (105 - 45 * Phase**2 + Phase**4) * np.sin(Phase)
    ) / Phase**5
    result = j4 * sin_theta_fourth * np.exp(-4j * phi) * np.sqrt(315 / (512 * np.pi))
    return result


@njit
def window_function_legendre_l4_m_minus3_numba(ki, kj, kk, R):
    k = np.sqrt(ki**2 + kj**2 + kk**2)
    if k == 0:
        return 0
    sin_theta_cubed = (np.sqrt(ki**2 + kj**2) / k) ** 3
    cos_theta = kk / k
    phi = np.arctan2(kj, ki)
    Phase = 2 * np.pi * k * R
    j4 = (
        5 * Phase * (-21 + 2 * Phase**2) * np.cos(Phase) + (105 - 45 * Phase**2 + Phase**4) * np.sin(Phase)
    ) / Phase**5
    result = j4 * sin_theta_cubed * cos_theta * np.exp(-3j * phi) * np.sqrt(315 / (64 * np.pi))
    return result


@njit
def window_function_legendre_l4_m_minus2_numba(ki, kj, kk, R):
    k = np.sqrt(ki**2 + kj**2 + kk**2)
    if k == 0:
        return 0
    sin_theta_squared = (ki**2 + kj**2) / k**2
    cos_theta_squared = (kk / k) ** 2
    phi = np.arctan2(kj, ki)
    Phase = 2 * np.pi * k * R
    j4 = (
        5 * Phase * (-21 + 2 * Phase**2) * np.cos(Phase) + (105 - 45 * Phase**2 + Phase**4) * np.sin(Phase)
    ) / Phase**5
    result = j4 * sin_theta_squared * (7 * cos_theta_squared - 1) * np.exp(-2j * phi) * np.sqrt(45 / (128 * np.pi))
    return result


@njit
def window_function_legendre_l4_m_minus1_numba(ki, kj, kk, R):
    k = np.sqrt(ki**2 + kj**2 + kk**2)
    if k == 0:
        return 0
    sin_theta = np.sqrt(ki**2 + kj**2) / k
    cos_theta = kk / k
    cos_theta_squared = cos_theta**2
    phi = np.arctan2(kj, ki)
    Phase = 2 * np.pi * k * R
    j4 = (
        5 * Phase * (-21 + 2 * Phase**2) * np.cos(Phase) + (105 - 45 * Phase**2 + Phase**4) * np.sin(Phase)
    ) / Phase**5
    result = j4 * sin_theta * cos_theta * (7 * cos_theta_squared - 3) * np.exp(-1j * phi) * np.sqrt(45 / (64 * np.pi))
    return result


@njit
def window_function_legendre_l4_m0_numba(ki, kj, kk, R):
    k = np.sqrt(ki**2 + kj**2 + kk**2)
    if k == 0:
        return 0
    cos_theta = kk / k
    Phase = 2 * np.pi * k * R
    j4 = (
        5 * Phase * (-21 + 2 * Phase**2) * np.cos(Phase) + (105 - 45 * Phase**2 + Phase**4) * np.sin(Phase)
    ) / Phase**5
    result = j4 * (35 * cos_theta**4 - 30 * cos_theta**2 + 3) * np.sqrt(9 / (256 * np.pi))
    return result


@njit
def window_function_legendre_l4_m1_numba(ki, kj, kk, R):
    k = np.sqrt(ki**2 + kj**2 + kk**2)
    if k == 0:
        return 0
    sin_theta = np.sqrt(ki**2 + kj**2) / k
    cos_theta = kk / k
    cos_theta_squared = cos_theta**2
    phi = np.arctan2(kj, ki)
    Phase = 2 * np.pi * k * R
    j4 = (
        5 * Phase * (-21 + 2 * Phase**2) * np.cos(Phase) + (105 - 45 * Phase**2 + Phase**4) * np.sin(Phase)
    ) / Phase**5
    result = j4 * sin_theta * cos_theta * (7 * cos_theta_squared - 3) * np.exp(1j * phi) * (-np.sqrt(45 / (64 * np.pi)))
    return result


@njit
def window_function_legendre_l4_m2_numba(ki, kj, kk, R):
    k = np.sqrt(ki**2 + kj**2 + kk**2)
    if k == 0:
        return 0
    sin_theta_squared = (ki**2 + kj**2) / k**2
    cos_theta_squared = (kk / k) ** 2
    phi = np.arctan2(kj, ki)
    Phase = 2 * np.pi * k * R
    j4 = (
        5 * Phase * (-21 + 2 * Phase**2) * np.cos(Phase) + (105 - 45 * Phase**2 + Phase**4) * np.sin(Phase)
    ) / Phase**5
    result = j4 * sin_theta_squared * (7 * cos_theta_squared - 1) * np.exp(2j * phi) * np.sqrt(45 / (128 * np.pi))
    return result


@njit
def window_function_legendre_l4_m3_numba(ki, kj, kk, R):
    k = np.sqrt(ki**2 + kj**2 + kk**2)
    if k == 0:
        return 0
    sin_theta_cubed = (np.sqrt(ki**2 + kj**2) / k) ** 3
    cos_theta = kk / k
    phi = np.arctan2(kj, ki)
    Phase = 2 * np.pi * k * R
    j4 = (
        5 * Phase * (-21 + 2 * Phase**2) * np.cos(Phase) + (105 - 45 * Phase**2 + Phase**4) * np.sin(Phase)
    ) / Phase**5
    result = j4 * sin_theta_cubed * cos_theta * np.exp(3j * phi) * (-np.sqrt(315 / (64 * np.pi)))
    return result


@njit
def window_function_legendre_l4_m4_numba(ki, kj, kk, R):
    k = np.sqrt(ki**2 + kj**2 + kk**2)
    if k == 0:
        return 0
    sin_theta_fourth = (np.sqrt(ki**2 + kj**2) / k) ** 4
    phi = np.arctan2(kj, ki)
    Phase = 2 * np.pi * k * R
    j4 = (
        5 * Phase * (-21 + 2 * Phase**2) * np.cos(Phase) + (105 - 45 * Phase**2 + Phase**4) * np.sin(Phase)
    ) / Phase**5
    result = j4 * sin_theta_fourth * np.exp(4j * phi) * np.sqrt(315 / (512 * np.pi))
    return result


@njit
def window_function_legendre_l5_m_minus5_numba(ki, kj, kk, R):
    k = np.sqrt(ki**2 + kj**2 + kk**2)
    if k == 0:
        return 0
    sin_theta_fifth = (np.sqrt(ki**2 + kj**2) / k) ** 5
    phi = np.arctan2(kj, ki)
    Phase = 2 * np.pi * k * R
    j5 = (
        -Phase * (945 - 105 * Phase**2 + Phase**4) * np.cos(Phase)
        + 15 * (63 - 28 * Phase**2 + Phase**4) * np.sin(Phase)
    ) / Phase**6
    result = j5 * sin_theta_fifth * np.exp(-5j * phi) * np.sqrt(693 / (1024 * np.pi))
    return result


@njit
def window_function_legendre_l5_m_minus4_numba(ki, kj, kk, R):
    k = np.sqrt(ki**2 + kj**2 + kk**2)
    if k == 0:
        return 0
    sin_theta_fourth = (np.sqrt(ki**2 + kj**2) / k) ** 4
    cos_theta = kk / k
    phi = np.arctan2(kj, ki)
    Phase = 2 * np.pi * k * R
    j5 = (
        -Phase * (945 - 105 * Phase**2 + Phase**4) * np.cos(Phase)
        + 15 * (63 - 28 * Phase**2 + Phase**4) * np.sin(Phase)
    ) / Phase**6
    result = j5 * sin_theta_fourth * cos_theta * np.exp(-4j * phi) * np.sqrt(3465 / (512 * np.pi))
    return result


@njit
def window_function_legendre_l5_m_minus3_numba(ki, kj, kk, R):
    k = np.sqrt(ki**2 + kj**2 + kk**2)
    if k == 0:
        return 0
    sin_theta_cubed = (np.sqrt(ki**2 + kj**2) / k) ** 3
    cos_theta_squared = (kk / k) ** 2
    phi = np.arctan2(kj, ki)
    Phase = 2 * np.pi * k * R
    j5 = (
        -Phase * (945 - 105 * Phase**2 + Phase**4) * np.cos(Phase)
        + 15 * (63 - 28 * Phase**2 + Phase**4) * np.sin(Phase)
    ) / Phase**6
    result = j5 * sin_theta_cubed * (9 * cos_theta_squared - 1) * np.exp(-3j * phi) * np.sqrt(385 / (1024 * np.pi))
    return result


@njit
def window_function_legendre_l5_m_minus2_numba(ki, kj, kk, R):
    k = np.sqrt(ki**2 + kj**2 + kk**2)
    if k == 0:
        return 0
    sin_theta_squared = (ki**2 + kj**2) / k**2
    cos_theta = kk / k
    cos_theta_cubed = cos_theta**3
    phi = np.arctan2(kj, ki)
    Phase = 2 * np.pi * k * R
    j5 = (
        -Phase * (945 - 105 * Phase**2 + Phase**4) * np.cos(Phase)
        + 15 * (63 - 28 * Phase**2 + Phase**4) * np.sin(Phase)
    ) / Phase**6
    result = (
        j5 * sin_theta_squared * (3 * cos_theta_cubed - cos_theta) * np.exp(-2j * phi) * np.sqrt(1155 / (128 * np.pi))
    )
    return result


@njit
def window_function_legendre_l5_m_minus1_numba(ki, kj, kk, R):
    k = np.sqrt(ki**2 + kj**2 + kk**2)
    if k == 0:
        return 0
    sin_theta = np.sqrt(ki**2 + kj**2) / k
    cos_theta = kk / k
    cos_theta_squared = cos_theta**2
    cos_theta_fourth = cos_theta**4
    phi = np.arctan2(kj, ki)
    Phase = 2 * np.pi * k * R
    j5 = (
        -Phase * (945 - 105 * Phase**2 + Phase**4) * np.cos(Phase)
        + 15 * (63 - 28 * Phase**2 + Phase**4) * np.sin(Phase)
    ) / Phase**6
    result = (
        j5
        * sin_theta
        * (21 * cos_theta_fourth - 14 * cos_theta_squared + 1)
        * np.exp(-1j * phi)
        * np.sqrt(165 / (512 * np.pi))
    )
    return result


@njit
def window_function_legendre_l5_m0_numba(ki, kj, kk, R):
    k = np.sqrt(ki**2 + kj**2 + kk**2)
    if k == 0:
        return 0
    cos_theta = kk / k
    cos_theta_squared = cos_theta**2
    cos_theta_fourth = cos_theta_squared**2
    Phase = 2 * np.pi * k * R
    j5 = (
        -Phase * (945 - 105 * Phase**2 + Phase**4) * np.cos(Phase)
        + 15 * (63 - 28 * Phase**2 + Phase**4) * np.sin(Phase)
    ) / Phase**6
    result = j5 * cos_theta * (63 * cos_theta_fourth - 70 * cos_theta_squared + 15) * np.sqrt(11 / (256 * np.pi))
    return result


@njit
def window_function_legendre_l5_m1_numba(ki, kj, kk, R):
    k = np.sqrt(ki**2 + kj**2 + kk**2)
    if k == 0:
        return 0
    sin_theta = np.sqrt(ki**2 + kj**2) / k
    cos_theta = kk / k
    cos_theta_squared = cos_theta**2
    cos_theta_fourth = cos_theta**4
    phi = np.arctan2(kj, ki)
    Phase = 2 * np.pi * k * R
    j5 = (
        -Phase * (945 - 105 * Phase**2 + Phase**4) * np.cos(Phase)
        + 15 * (63 - 28 * Phase**2 + Phase**4) * np.sin(Phase)
    ) / Phase**6
    result = (
        j5
        * sin_theta
        * (21 * cos_theta_fourth - 14 * cos_theta_squared + 1)
        * np.exp(1j * phi)
        * (-np.sqrt(165 / (512 * np.pi)))
    )
    return result


@njit
def window_function_legendre_l5_m2_numba(ki, kj, kk, R):
    k = np.sqrt(ki**2 + kj**2 + kk**2)
    if k == 0:
        return 0
    sin_theta_squared = (ki**2 + kj**2) / k**2
    cos_theta = kk / k
    cos_theta_cubed = cos_theta**3
    phi = np.arctan2(kj, ki)
    Phase = 2 * np.pi * k * R
    j5 = (
        -Phase * (945 - 105 * Phase**2 + Phase**4) * np.cos(Phase)
        + 15 * (63 - 28 * Phase**2 + Phase**4) * np.sin(Phase)
    ) / Phase**6
    result = (
        j5 * sin_theta_squared * (3 * cos_theta_cubed - cos_theta) * np.exp(2j * phi) * np.sqrt(1155 / (128 * np.pi))
    )
    return result


@njit
def window_function_legendre_l5_m3_numba(ki, kj, kk, R):
    k = np.sqrt(ki**2 + kj**2 + kk**2)
    if k == 0:
        return 0
    sin_theta_cubed = (np.sqrt(ki**2 + kj**2) / k) ** 3
    cos_theta_squared = (kk / k) ** 2
    phi = np.arctan2(kj, ki)
    Phase = 2 * np.pi * k * R
    j5 = (
        -Phase * (945 - 105 * Phase**2 + Phase**4) * np.cos(Phase)
        + 15 * (63 - 28 * Phase**2 + Phase**4) * np.sin(Phase)
    ) / Phase**6
    result = j5 * sin_theta_cubed * (9 * cos_theta_squared - 1) * np.exp(3j * phi) * (-np.sqrt(385 / (1024 * np.pi)))
    return result


@njit
def window_function_legendre_l5_m4_numba(ki, kj, kk, R):
    k = np.sqrt(ki**2 + kj**2 + kk**2)
    if k == 0:
        return 0
    sin_theta_fourth = (np.sqrt(ki**2 + kj**2) / k) ** 4
    cos_theta = kk / k
    phi = np.arctan2(kj, ki)
    Phase = 2 * np.pi * k * R
    j5 = (
        -Phase * (945 - 105 * Phase**2 + Phase**4) * np.cos(Phase)
        + 15 * (63 - 28 * Phase**2 + Phase**4) * np.sin(Phase)
    ) / Phase**6
    result = j5 * sin_theta_fourth * cos_theta * np.exp(4j * phi) * np.sqrt(3465 / (512 * np.pi))
    return result


@njit
def window_function_legendre_l5_m5_numba(ki, kj, kk, R):
    k = np.sqrt(ki**2 + kj**2 + kk**2)
    if k == 0:
        return 0
    sin_theta_fifth = (np.sqrt(ki**2 + kj**2) / k) ** 5
    phi = np.arctan2(kj, ki)
    Phase = 2 * np.pi * k * R
    j5 = (
        -Phase * (945 - 105 * Phase**2 + Phase**4) * np.cos(Phase)
        + 15 * (63 - 28 * Phase**2 + Phase**4) * np.sin(Phase)
    ) / Phase**6
    result = j5 * sin_theta_fifth * np.exp(5j * phi) * (-np.sqrt(693 / (1024 * np.pi)))
    return result


@njit
def window_function_legendre_l6_m_minus6_numba(ki, kj, kk, R):
    k = np.sqrt(ki**2 + kj**2 + kk**2)
    if k == 0:
        return 0
    sin_theta_sixth = (np.sqrt(ki**2 + kj**2) / k) ** 6
    phi = np.arctan2(kj, ki)
    Phase = 2 * np.pi * k * R
    numerator_cos = 21 * Phase * (495 - 60 * Phase**2 + Phase**4) * np.cos(Phase)
    numerator_sin = (-10395 + 4725 * Phase**2 - 210 * Phase**4 + Phase**6) * np.sin(Phase)
    numerator = -(numerator_cos + numerator_sin)
    denominator = Phase**7
    j6 = numerator / denominator
    result = j6 * sin_theta_sixth * np.exp(-6j * phi) * np.sqrt(3003 / (64**2 * np.pi))
    return result


@njit
def window_function_legendre_l6_m_minus5_numba(ki, kj, kk, R):
    k = np.sqrt(ki**2 + kj**2 + kk**2)
    if k == 0:
        return 0
    sin_theta_fifth = (np.sqrt(ki**2 + kj**2) / k) ** 5
    cos_theta = kk / k
    phi = np.arctan2(kj, ki)
    Phase = 2 * np.pi * k * R
    numerator_cos = 21 * Phase * (495 - 60 * Phase**2 + Phase**4) * np.cos(Phase)
    numerator_sin = (-10395 + 4725 * Phase**2 - 210 * Phase**4 + Phase**6) * np.sin(Phase)
    numerator = -(numerator_cos + numerator_sin)
    denominator = Phase**7
    j6 = numerator / denominator
    result = j6 * sin_theta_fifth * cos_theta * np.exp(-5j * phi) * np.sqrt(9009 / (1024 * np.pi))
    return result


@njit
def window_function_legendre_l6_m_minus4_numba(ki, kj, kk, R):
    k = np.sqrt(ki**2 + kj**2 + kk**2)
    if k == 0:
        return 0
    sin_theta_fourth = (np.sqrt(ki**2 + kj**2) / k) ** 4
    cos_theta_squared = (kk / k) ** 2
    phi = np.arctan2(kj, ki)
    Phase = 2 * np.pi * k * R
    numerator_cos = 21 * Phase * (495 - 60 * Phase**2 + Phase**4) * np.cos(Phase)
    numerator_sin = (-10395 + 4725 * Phase**2 - 210 * Phase**4 + Phase**6) * np.sin(Phase)
    numerator = -(numerator_cos + numerator_sin)
    denominator = Phase**7
    j6 = numerator / denominator
    result = j6 * sin_theta_fourth * (11 * cos_theta_squared - 1) * np.exp(-4j * phi) * np.sqrt(819 / (2048 * np.pi))
    return result


@njit
def window_function_legendre_l6_m_minus3_numba(ki, kj, kk, R):
    k = np.sqrt(ki**2 + kj**2 + kk**2)
    if k == 0:
        return 0
    sin_theta_cubed = (np.sqrt(ki**2 + kj**2) / k) ** 3
    cos_theta = kk / k
    cos_theta_cubed = cos_theta**3
    phi = np.arctan2(kj, ki)
    Phase = 2 * np.pi * k * R
    numerator_cos = 21 * Phase * (495 - 60 * Phase**2 + Phase**4) * np.cos(Phase)
    numerator_sin = (-10395 + 4725 * Phase**2 - 210 * Phase**4 + Phase**6) * np.sin(Phase)
    numerator = -(numerator_cos + numerator_sin)
    denominator = Phase**7
    j6 = numerator / denominator
    result = (
        j6
        * sin_theta_cubed
        * (11 * cos_theta_cubed - 3 * cos_theta)
        * np.exp(-3j * phi)
        * np.sqrt(1365 / (1024 * np.pi))
    )
    return result


@njit
def window_function_legendre_l6_m_minus2_numba(ki, kj, kk, R):
    k = np.sqrt(ki**2 + kj**2 + kk**2)
    if k == 0:
        return 0
    sin_theta_squared = (ki**2 + kj**2) / k**2
    cos_theta_squared = (kk / k) ** 2
    cos_theta_fourth = cos_theta_squared**2
    phi = np.arctan2(kj, ki)
    Phase = 2 * np.pi * k * R
    numerator_cos = 21 * Phase * (495 - 60 * Phase**2 + Phase**4) * np.cos(Phase)
    numerator_sin = (-10395 + 4725 * Phase**2 - 210 * Phase**4 + Phase**6) * np.sin(Phase)
    numerator = -(numerator_cos + numerator_sin)
    denominator = Phase**7
    j6 = numerator / denominator
    result = (
        j6
        * sin_theta_squared
        * (33 * cos_theta_fourth - 18 * cos_theta_squared + 1)
        * np.exp(-2j * phi)
        * np.sqrt(1365 / (64**2 * np.pi))
    )
    return result


@njit
def window_function_legendre_l6_m_minus1_numba(ki, kj, kk, R):
    k = np.sqrt(ki**2 + kj**2 + kk**2)
    if k == 0:
        return 0
    sin_theta = np.sqrt(ki**2 + kj**2) / k
    cos_theta = kk / k
    cos_theta_cubed = cos_theta**3
    cos_theta_fifth = cos_theta**5
    phi = np.arctan2(kj, ki)
    Phase = 2 * np.pi * k * R
    numerator_cos = 21 * Phase * (495 - 60 * Phase**2 + Phase**4) * np.cos(Phase)
    numerator_sin = (-10395 + 4725 * Phase**2 - 210 * Phase**4 + Phase**6) * np.sin(Phase)
    numerator = -(numerator_cos + numerator_sin)
    denominator = Phase**7
    j6 = numerator / denominator
    result = (
        j6
        * sin_theta
        * (33 * cos_theta_fifth - 30 * cos_theta_cubed + 5 * cos_theta)
        * np.exp(-1j * phi)
        * np.sqrt(273 / (512 * np.pi))
    )
    return result


@njit
def window_function_legendre_l6_m0_numba(ki, kj, kk, R):
    k = np.sqrt(ki**2 + kj**2 + kk**2)
    if k == 0:
        return 0
    cos_theta = kk / k
    cos_theta_squared = cos_theta**2
    cos_theta_fourth = cos_theta_squared**2
    cos_theta_sixth = cos_theta_fourth * cos_theta_squared
    Phase = 2 * np.pi * k * R
    numerator_cos = 21 * Phase * (495 - 60 * Phase**2 + Phase**4) * np.cos(Phase)
    numerator_sin = (-10395 + 4725 * Phase**2 - 210 * Phase**4 + Phase**6) * np.sin(Phase)
    numerator = -(numerator_cos + numerator_sin)
    denominator = Phase**7
    j6 = numerator / denominator
    result = (
        j6
        * (231 * cos_theta_sixth - 315 * cos_theta_fourth + 105 * cos_theta_squared - 5)
        * np.sqrt(13 / (1024 * np.pi))
    )
    return result


@njit
def window_function_legendre_l6_m1_numba(ki, kj, kk, R):
    k = np.sqrt(ki**2 + kj**2 + kk**2)
    if k == 0:
        return 0
    sin_theta = np.sqrt(ki**2 + kj**2) / k
    cos_theta = kk / k
    cos_theta_cubed = cos_theta**3
    cos_theta_fifth = cos_theta**5
    phi = np.arctan2(kj, ki)
    Phase = 2 * np.pi * k * R
    numerator_cos = 21 * Phase * (495 - 60 * Phase**2 + Phase**4) * np.cos(Phase)
    numerator_sin = (-10395 + 4725 * Phase**2 - 210 * Phase**4 + Phase**6) * np.sin(Phase)
    numerator = -(numerator_cos + numerator_sin)
    denominator = Phase**7
    j6 = numerator / denominator
    result = (
        j6
        * sin_theta
        * (33 * cos_theta_fifth - 30 * cos_theta_cubed + 5 * cos_theta)
        * np.exp(1j * phi)
        * (-np.sqrt(273 / (512 * np.pi)))
    )
    return result


@njit
def window_function_legendre_l6_m2_numba(ki, kj, kk, R):
    k = np.sqrt(ki**2 + kj**2 + kk**2)
    if k == 0:
        return 0
    sin_theta_squared = (ki**2 + kj**2) / k**2
    cos_theta_squared = (kk / k) ** 2
    cos_theta_fourth = cos_theta_squared**2
    phi = np.arctan2(kj, ki)
    Phase = 2 * np.pi * k * R
    numerator_cos = 21 * Phase * (495 - 60 * Phase**2 + Phase**4) * np.cos(Phase)
    numerator_sin = (-10395 + 4725 * Phase**2 - 210 * Phase**4 + Phase**6) * np.sin(Phase)
    numerator = -(numerator_cos + numerator_sin)
    denominator = Phase**7
    j6 = numerator / denominator
    result = (
        j6
        * sin_theta_squared
        * (33 * cos_theta_fourth - 18 * cos_theta_squared + 1)
        * np.exp(2j * phi)
        * np.sqrt(1365 / (64**2 * np.pi))
    )
    return result


@njit
def window_function_legendre_l6_m3_numba(ki, kj, kk, R):
    k = np.sqrt(ki**2 + kj**2 + kk**2)
    if k == 0:
        return 0
    sin_theta_cubed = (np.sqrt(ki**2 + kj**2) / k) ** 3
    cos_theta = kk / k
    cos_theta_cubed = cos_theta**3
    phi = np.arctan2(kj, ki)
    Phase = 2 * np.pi * k * R
    numerator_cos = 21 * Phase * (495 - 60 * Phase**2 + Phase**4) * np.cos(Phase)
    numerator_sin = (-10395 + 4725 * Phase**2 - 210 * Phase**4 + Phase**6) * np.sin(Phase)
    numerator = -(numerator_cos + numerator_sin)
    denominator = Phase**7
    j6 = numerator / denominator
    result = (
        j6
        * sin_theta_cubed
        * (11 * cos_theta_cubed - 3 * cos_theta)
        * np.exp(3j * phi)
        * (-np.sqrt(1365 / (1024 * np.pi)))
    )
    return result


@njit
def window_function_legendre_l6_m4_numba(ki, kj, kk, R):
    k = np.sqrt(ki**2 + kj**2 + kk**2)
    if k == 0:
        return 0
    sin_theta_fourth = (np.sqrt(ki**2 + kj**2) / k) ** 4
    cos_theta_squared = (kk / k) ** 2
    phi = np.arctan2(kj, ki)
    Phase = 2 * np.pi * k * R
    numerator_cos = 21 * Phase * (495 - 60 * Phase**2 + Phase**4) * np.cos(Phase)
    numerator_sin = (-10395 + 4725 * Phase**2 - 210 * Phase**4 + Phase**6) * np.sin(Phase)
    numerator = -(numerator_cos + numerator_sin)
    denominator = Phase**7
    j6 = numerator / denominator
    result = j6 * sin_theta_fourth * (11 * cos_theta_squared - 1) * np.exp(4j * phi) * np.sqrt(819 / (2048 * np.pi))
    return result


@njit
def window_function_legendre_l6_m5_numba(ki, kj, kk, R):
    k = np.sqrt(ki**2 + kj**2 + kk**2)
    if k == 0:
        return 0
    sin_theta_fifth = (np.sqrt(ki**2 + kj**2) / k) ** 5
    cos_theta = kk / k
    phi = np.arctan2(kj, ki)
    Phase = 2 * np.pi * k * R
    numerator_cos = 21 * Phase * (495 - 60 * Phase**2 + Phase**4) * np.cos(Phase)
    numerator_sin = (-10395 + 4725 * Phase**2 - 210 * Phase**4 + Phase**6) * np.sin(Phase)
    numerator = -(numerator_cos + numerator_sin)
    denominator = Phase**7
    j6 = numerator / denominator
    result = j6 * sin_theta_fifth * cos_theta * np.exp(5j * phi) * (-np.sqrt(9009 / (1024 * np.pi)))
    return result


@njit
def window_function_legendre_l6_m6_numba(ki, kj, kk, R):
    k = np.sqrt(ki**2 + kj**2 + kk**2)
    if k == 0:
        return 0
    sin_theta_sixth = (np.sqrt(ki**2 + kj**2) / k) ** 6
    phi = np.arctan2(kj, ki)
    Phase = 2 * np.pi * k * R
    numerator_cos = 21 * Phase * (495 - 60 * Phase**2 + Phase**4) * np.cos(Phase)
    numerator_sin = (-10395 + 4725 * Phase**2 - 210 * Phase**4 + Phase**6) * np.sin(Phase)
    numerator = -(numerator_cos + numerator_sin)
    denominator = Phase**7
    j6 = numerator / denominator
    result = j6 * sin_theta_sixth * np.exp(6j * phi) * (np.sqrt(3003 / (64**2 * np.pi)))
    return result


@njit
def window_function_legendre_l7_m_minus7_numba(ki, kj, kk, R):
    k = np.sqrt(ki**2 + kj**2 + kk**2)
    if k == 0:
        return 0
    sin_theta_seventh = (np.sqrt(ki**2 + kj**2) / k) ** 7
    phi = np.arctan2(kj, ki)
    Phase = 2 * np.pi * k * R
    numerator_cos = Phase * (-135135 + 17325 * Phase**2 - 378 * Phase**4 + Phase**6) * np.cos(Phase)
    numerator_sin = -7 * (-19305 + 8910 * Phase**2 - 450 * Phase**4 + 4 * Phase**6) * np.sin(Phase)
    numerator = numerator_cos + numerator_sin
    denominator = Phase**8
    j7 = numerator / denominator
    result = j7 * sin_theta_seventh * np.exp(-7j * phi) * 3 / 64 * np.sqrt(715 / (2 * np.pi))
    return result


@njit
def window_function_legendre_l7_m_minus6_numba(ki, kj, kk, R):
    k = np.sqrt(ki**2 + kj**2 + kk**2)
    if k == 0:
        return 0
    sin_theta_sixth = (np.sqrt(ki**2 + kj**2) / k) ** 6
    cos_theta = kk / k
    phi = np.arctan2(kj, ki)
    Phase = 2 * np.pi * k * R
    numerator_cos = Phase * (-135135 + 17325 * Phase**2 - 378 * Phase**4 + Phase**6) * np.cos(Phase)
    numerator_sin = -7 * (-19305 + 8910 * Phase**2 - 450 * Phase**4 + 4 * Phase**6) * np.sin(Phase)
    numerator = numerator_cos + numerator_sin
    denominator = Phase**8
    j7 = numerator / denominator
    result = j7 * sin_theta_sixth * cos_theta * np.exp(-6j * phi) * 3 * np.sqrt(5005 / (64**2 * np.pi))
    return result


@njit
def window_function_legendre_l7_m_minus5_numba(ki, kj, kk, R):
    k = np.sqrt(ki**2 + kj**2 + kk**2)
    if k == 0:
        return 0
    sin_theta_fifth = (np.sqrt(ki**2 + kj**2) / k) ** 5
    cos_theta_squared = (kk / k) ** 2
    phi = np.arctan2(kj, ki)
    Phase = 2 * np.pi * k * R
    numerator_cos = Phase * (-135135 + 17325 * Phase**2 - 378 * Phase**4 + Phase**6) * np.cos(Phase)
    numerator_sin = -7 * (-19305 + 8910 * Phase**2 - 450 * Phase**4 + 4 * Phase**6) * np.sin(Phase)
    numerator = numerator_cos + numerator_sin
    denominator = Phase**8
    j7 = numerator / denominator
    result = (
        j7 * sin_theta_fifth * (13 * cos_theta_squared - 1) * np.exp(-5j * phi) * 3 * np.sqrt(385 / (64**2 * 2 * np.pi))
    )
    return result


@njit
def window_function_legendre_l7_m_minus4_numba(ki, kj, kk, R):
    k = np.sqrt(ki**2 + kj**2 + kk**2)
    if k == 0:
        return 0
    sin_theta_fourth = (np.sqrt(ki**2 + kj**2) / k) ** 4
    cos_theta = kk / k
    cos_theta_cubed = cos_theta**3
    phi = np.arctan2(kj, ki)
    Phase = 2 * np.pi * k * R
    numerator_cos = Phase * (-135135 + 17325 * Phase**2 - 378 * Phase**4 + Phase**6) * np.cos(Phase)
    numerator_sin = -7 * (-19305 + 8910 * Phase**2 - 450 * Phase**4 + 4 * Phase**6) * np.sin(Phase)
    numerator = numerator_cos + numerator_sin
    denominator = Phase**8
    j7 = numerator / denominator
    result = (
        j7
        * sin_theta_fourth
        * (13 * cos_theta_cubed - 3 * cos_theta)
        * np.exp(-4j * phi)
        * 3
        * np.sqrt(385 / (32**2 * 2 * np.pi))
    )
    return result


@njit
def window_function_legendre_l7_m_minus3_numba(ki, kj, kk, R):
    k = np.sqrt(ki**2 + kj**2 + kk**2)
    if k == 0:
        return 0
    sin_theta_cubed = (np.sqrt(ki**2 + kj**2) / k) ** 3
    cos_theta_squared = (kk / k) ** 2
    cos_theta_fourth = cos_theta_squared**2
    phi = np.arctan2(kj, ki)
    Phase = 2 * np.pi * k * R
    numerator_cos = Phase * (-135135 + 17325 * Phase**2 - 378 * Phase**4 + Phase**6) * np.cos(Phase)
    numerator_sin = -7 * (-19305 + 8910 * Phase**2 - 450 * Phase**4 + 4 * Phase**6) * np.sin(Phase)
    numerator = numerator_cos + numerator_sin
    denominator = Phase**8
    j7 = numerator / denominator
    result = (
        j7
        * sin_theta_cubed
        * (143 * cos_theta_fourth - 66 * cos_theta_squared + 3)
        * np.exp(-3j * phi)
        * 3
        * np.sqrt(35 / (64**2 * 2 * np.pi))
    )
    return result


@njit
def window_function_legendre_l7_m_minus2_numba(ki, kj, kk, R):
    k = np.sqrt(ki**2 + kj**2 + kk**2)
    if k == 0:
        return 0
    sin_theta_squared = (ki**2 + kj**2) / k**2
    cos_theta = kk / k
    cos_theta_cubed = cos_theta**3
    cos_theta_fifth = cos_theta_cubed * (kk / k) ** 2
    phi = np.arctan2(kj, ki)
    Phase = 2 * np.pi * k * R
    numerator_cos = Phase * (-135135 + 17325 * Phase**2 - 378 * Phase**4 + Phase**6) * np.cos(Phase)
    numerator_sin = -7 * (-19305 + 8910 * Phase**2 - 450 * Phase**4 + 4 * Phase**6) * np.sin(Phase)
    numerator = numerator_cos + numerator_sin
    denominator = Phase**8
    j7 = numerator / denominator
    result = (
        j7
        * sin_theta_squared
        * (143 * cos_theta_fifth - 110 * cos_theta_cubed + 15 * cos_theta)
        * np.exp(-2j * phi)
        * 3
        * np.sqrt(35 / (64**2 * np.pi))
    )
    return result


@njit
def window_function_legendre_l7_m_minus1_numba(ki, kj, kk, R):
    k = np.sqrt(ki**2 + kj**2 + kk**2)
    if k == 0:
        return 0
    sin_theta = np.sqrt(ki**2 + kj**2) / k
    cos_theta_squared = (kk / k) ** 2
    cos_theta_fourth = cos_theta_squared**2
    cos_theta_sixth = cos_theta_squared**3
    phi = np.arctan2(kj, ki)
    Phase = 2 * np.pi * k * R
    numerator_cos = Phase * (-135135 + 17325 * Phase**2 - 378 * Phase**4 + Phase**6) * np.cos(Phase)
    numerator_sin = -7 * (-19305 + 8910 * Phase**2 - 450 * Phase**4 + 4 * Phase**6) * np.sin(Phase)
    numerator = numerator_cos + numerator_sin
    denominator = Phase**8
    j7 = numerator / denominator
    result = (
        j7
        * sin_theta
        * (429 * cos_theta_sixth - 495 * cos_theta_fourth + 135 * cos_theta_squared - 5)
        * np.exp(-1j * phi)
        * np.sqrt(105 / (64**2 * 2 * np.pi))
    )
    return result


@njit
def window_function_legendre_l7_m0_numba(ki, kj, kk, R):
    k = np.sqrt(ki**2 + kj**2 + kk**2)
    if k == 0:
        return 0
    cos_theta = kk / k
    cos_theta_squared = cos_theta**2
    cos_theta_fourth = cos_theta_squared**2
    cos_theta_sixth = cos_theta_fourth * cos_theta_squared
    Phase = 2 * np.pi * k * R
    numerator_cos = Phase * (-135135 + 17325 * Phase**2 - 378 * Phase**4 + Phase**6) * np.cos(Phase)
    numerator_sin = -7 * (-19305 + 8910 * Phase**2 - 450 * Phase**4 + 4 * Phase**6) * np.sin(Phase)
    numerator = numerator_cos + numerator_sin
    denominator = Phase**8
    j7 = numerator / denominator
    result = (
        j7
        * cos_theta
        * (429 * cos_theta_sixth - 693 * cos_theta_fourth + 315 * cos_theta_squared - 35)
        * np.sqrt(15 / (1024 * np.pi))
    )
    return result


@njit
def window_function_legendre_l7_m1_numba(ki, kj, kk, R):
    k = np.sqrt(ki**2 + kj**2 + kk**2)
    if k == 0:
        return 0
    sin_theta = np.sqrt(ki**2 + kj**2) / k
    cos_theta_squared = (kk / k) ** 2
    cos_theta_fourth = cos_theta_squared**2
    cos_theta_sixth = cos_theta_squared**3
    phi = np.arctan2(kj, ki)
    Phase = 2 * np.pi * k * R
    numerator_cos = Phase * (-135135 + 17325 * Phase**2 - 378 * Phase**4 + Phase**6) * np.cos(Phase)
    numerator_sin = -7 * (-19305 + 8910 * Phase**2 - 450 * Phase**4 + 4 * Phase**6) * np.sin(Phase)
    numerator = numerator_cos + numerator_sin
    denominator = Phase**8
    j7 = numerator / denominator
    result = (
        j7
        * sin_theta
        * (429 * cos_theta_sixth - 495 * cos_theta_fourth + 135 * cos_theta_squared - 5)
        * np.exp(1j * phi)
        * (-np.sqrt(105 / (64**2 * 2 * np.pi)))
    )
    return result


@njit
def window_function_legendre_l7_m2_numba(ki, kj, kk, R):
    k = np.sqrt(ki**2 + kj**2 + kk**2)
    if k == 0:
        return 0
    sin_theta_squared = (ki**2 + kj**2) / k**2
    cos_theta = kk / k
    cos_theta_cubed = cos_theta**3
    cos_theta_fifth = cos_theta_cubed * (kk / k) ** 2
    phi = np.arctan2(kj, ki)
    Phase = 2 * np.pi * k * R
    numerator_cos = Phase * (-135135 + 17325 * Phase**2 - 378 * Phase**4 + Phase**6) * np.cos(Phase)
    numerator_sin = -7 * (-19305 + 8910 * Phase**2 - 450 * Phase**4 + 4 * Phase**6) * np.sin(Phase)
    numerator = numerator_cos + numerator_sin
    denominator = Phase**8
    j7 = numerator / denominator
    result = (
        j7
        * sin_theta_squared
        * (143 * cos_theta_fifth - 110 * cos_theta_cubed + 15 * cos_theta)
        * np.exp(2j * phi)
        * 3
        * np.sqrt(35 / (64**2 * np.pi))
    )
    return result


@njit
def window_function_legendre_l7_m3_numba(ki, kj, kk, R):
    k = np.sqrt(ki**2 + kj**2 + kk**2)
    if k == 0:
        return 0
    sin_theta_cubed = (np.sqrt(ki**2 + kj**2) / k) ** 3
    cos_theta_squared = (kk / k) ** 2
    cos_theta_fourth = cos_theta_squared**2
    phi = np.arctan2(kj, ki)
    Phase = 2 * np.pi * k * R
    numerator_cos = Phase * (-135135 + 17325 * Phase**2 - 378 * Phase**4 + Phase**6) * np.cos(Phase)
    numerator_sin = -7 * (-19305 + 8910 * Phase**2 - 450 * Phase**4 + 4 * Phase**6) * np.sin(Phase)
    numerator = numerator_cos + numerator_sin
    denominator = Phase**8
    j7 = numerator / denominator
    result = (
        j7
        * sin_theta_cubed
        * (143 * cos_theta_fourth - 66 * cos_theta_squared + 3)
        * np.exp(3j * phi)
        * 3
        * (-np.sqrt(35 / (64**2 * 2 * np.pi)))
    )
    return result


@njit
def window_function_legendre_l7_m4_numba(ki, kj, kk, R):
    k = np.sqrt(ki**2 + kj**2 + kk**2)
    if k == 0:
        return 0
    sin_theta_fourth = (np.sqrt(ki**2 + kj**2) / k) ** 4
    cos_theta = kk / k
    cos_theta_cubed = cos_theta**3
    phi = np.arctan2(kj, ki)
    Phase = 2 * np.pi * k * R
    numerator_cos = Phase * (-135135 + 17325 * Phase**2 - 378 * Phase**4 + Phase**6) * np.cos(Phase)
    numerator_sin = -7 * (-19305 + 8910 * Phase**2 - 450 * Phase**4 + 4 * Phase**6) * np.sin(Phase)
    numerator = numerator_cos + numerator_sin
    denominator = Phase**8
    j7 = numerator / denominator
    result = (
        j7
        * sin_theta_fourth
        * (13 * cos_theta_cubed - 3 * cos_theta)
        * np.exp(4j * phi)
        * 3
        * np.sqrt(385 / (32**2 * 2 * np.pi))
    )
    return result


@njit
def window_function_legendre_l7_m5_numba(ki, kj, kk, R):
    k = np.sqrt(ki**2 + kj**2 + kk**2)
    if k == 0:
        return 0
    sin_theta_fifth = (np.sqrt(ki**2 + kj**2) / k) ** 5
    cos_theta_squared = (kk / k) ** 2
    phi = np.arctan2(kj, ki)
    Phase = 2 * np.pi * k * R
    numerator_cos = Phase * (-135135 + 17325 * Phase**2 - 378 * Phase**4 + Phase**6) * np.cos(Phase)
    numerator_sin = -7 * (-19305 + 8910 * Phase**2 - 450 * Phase**4 + 4 * Phase**6) * np.sin(Phase)
    numerator = numerator_cos + numerator_sin
    denominator = Phase**8
    j7 = numerator / denominator
    result = (
        j7
        * sin_theta_fifth
        * (13 * cos_theta_squared - 1)
        * np.exp(5j * phi)
        * (-3)
        * np.sqrt(385 / (64**2 * 2 * np.pi))
    )
    return result


@njit
def window_function_legendre_l7_m6_numba(ki, kj, kk, R):
    k = np.sqrt(ki**2 + kj**2 + kk**2)
    if k == 0:
        return 0
    sin_theta_sixth = (np.sqrt(ki**2 + kj**2) / k) ** 6
    cos_theta = kk / k
    phi = np.arctan2(kj, ki)
    Phase = 2 * np.pi * k * R
    numerator_cos = Phase * (-135135 + 17325 * Phase**2 - 378 * Phase**4 + Phase**6) * np.cos(Phase)
    numerator_sin = -7 * (-19305 + 8910 * Phase**2 - 450 * Phase**4 + 4 * Phase**6) * np.sin(Phase)
    numerator = numerator_cos + numerator_sin
    denominator = Phase**8
    j7 = numerator / denominator
    result = j7 * sin_theta_sixth * cos_theta * np.exp(6j * phi) * 3 * np.sqrt(5005 / (64**2 * np.pi))
    return result


@njit
def window_function_legendre_l7_m7_numba(ki, kj, kk, R):
    k = np.sqrt(ki**2 + kj**2 + kk**2)
    if k == 0:
        return 0
    sin_theta_seventh = (np.sqrt(ki**2 + kj**2) / k) ** 7
    phi = np.arctan2(kj, ki)
    Phase = 2 * np.pi * k * R
    numerator_cos = Phase * (-135135 + 17325 * Phase**2 - 378 * Phase**4 + Phase**6) * np.cos(Phase)
    numerator_sin = -7 * (-19305 + 8910 * Phase**2 - 450 * Phase**4 + 4 * Phase**6) * np.sin(Phase)
    numerator = numerator_cos + numerator_sin
    denominator = Phase**8
    j7 = numerator / denominator
    result = j7 * sin_theta_seventh * np.exp(7j * phi) * 3 * (-np.sqrt(715 / (64**2 * 2 * np.pi)))
    return result
