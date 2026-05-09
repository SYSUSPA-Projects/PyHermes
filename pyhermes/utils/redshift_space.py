"""Utilities for redshift-space coordinate transforms."""

import warnings

import numpy as np


_LOS_AXIS = {
    "x": np.array([1.0, 0.0, 0.0], dtype=np.float64),
    "y": np.array([0.0, 1.0, 0.0], dtype=np.float64),
    "z": np.array([0.0, 0.0, 1.0], dtype=np.float64),
}


def _fallback_hubble_at_redshift(redshift: float, omega_m: float, omega_l: float) -> float:
    """Return the flat-LambdaCDM H(z) approximation in km/s/(Mpc/h)."""
    return float(100.0 * np.sqrt(omega_m * (1.0 + redshift) ** 3 + omega_l))


def hubble_at_redshift(redshift: float, omega_m: float=0.3175, omega_l: float=0.6825) -> float:
    """Return H(z) in km/s/(Mpc/h), preferring CAMB when available."""
    try:
        import camb
    except ImportError:
        warnings.warn(
            "CAMB is not installed; falling back to the simple flat-LambdaCDM H(z) approximation.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _fallback_hubble_at_redshift(redshift, omega_m, omega_l)

    try:
        params = camb.CAMBparams()
        omega_b = min(max(float(omega_m) * 0.155, 1.0e-6), float(omega_m))
        params.set_cosmology(
            H0=100.0,
            ombh2=omega_b,
            omch2=float(omega_m) - omega_b,
            omk=float(1.0 - omega_m - omega_l),
        )
        params.InitPower.set_params(As=2.0e-9, ns=0.965)
        results = camb.get_background(params)
        return float(results.hubble_parameter(float(redshift)))
    except Exception as exc:
        warnings.warn(
            f"CAMB failed while computing H(z): {exc}; falling back to the simple flat-LambdaCDM approximation.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _fallback_hubble_at_redshift(redshift, omega_m, omega_l)


def _normalize_los(los, n_positions: int) -> np.ndarray:
    """Return one or many unit line-of-sight vectors."""
    if isinstance(los, str):
        key = los.strip().lower()
        if key not in _LOS_AXIS:
            raise ValueError(f"los must be 'x', 'y', 'z', or a vector, got {los!r}.")
        return _LOS_AXIS[key]
    if isinstance(los, (int, np.integer)):
        axis_to_name = {0: "x", 1: "y", 2: "z"}
        if int(los) not in axis_to_name:
            raise ValueError(f"los axis must be 0, 1, or 2, got {los}.")
        return _LOS_AXIS[axis_to_name[int(los)]]

    los_arr = np.asarray(los, dtype=np.float64)
    if los_arr.shape == (3,):
        norm = np.linalg.norm(los_arr)
        if norm == 0:
            raise ValueError("los vector must be non-zero.")
        return los_arr / norm
    if los_arr.shape == (n_positions, 3):
        norm = np.linalg.norm(los_arr, axis=1)
        if np.any(norm == 0):
            raise ValueError("los vector array must not contain zero-length rows.")
        return los_arr / norm[:, None]
    raise ValueError(f"los must be 'x', 'y', 'z', a 3-vector, or an (N, 3) array, got shape {los_arr.shape}.")


def redshift_space_positions(
    pos: np.ndarray,
    vel: np.ndarray,
    box_size: float,
    hubble: float,
    redshift: float,
    los="z",
    return_shift=False
) -> tuple[np.ndarray, np.ndarray]:
    """Shift positions along a line of sight using periodic boundary conditions."""
    pos = np.asarray(pos)
    vel = np.asarray(vel)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"pos must have shape (N, 3), got {pos.shape}.")
    if vel.shape != pos.shape:
        raise ValueError(f"vel must have shape {pos.shape}, got {vel.shape}.")
    if hubble == 0:
        raise ValueError("hubble must be non-zero.")

    los_hat = _normalize_los(los, pos.shape[0])
    pos_rsd = np.array(pos, dtype=np.float32, copy=True)
    vel_los = np.einsum("ij,ij->i", vel.astype(np.float64), np.broadcast_to(los_hat, pos.shape))
    shift = vel_los * (1.0 + redshift) / hubble
    displacement = shift[:, None] * np.broadcast_to(los_hat, pos.shape)
    pos_rsd[:, :] = np.mod(pos_rsd.astype(np.float64) + displacement, box_size)
    if return_shift:
        return pos_rsd, shift.astype(np.float32)
    else:
        return pos_rsd