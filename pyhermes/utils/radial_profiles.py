"""Radial-profile definitions and Hankel tabulation for multipole windows."""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import math

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import spherical_jn


BUILTIN_RADIAL_PROFILE_TYPES = {
    "shell",
    "sphere",
    "thick_shell",
    "gaussian",
    "gaussian_shell",
}
CUSTOM_RADIAL_PROFILE_TYPES = {"custom_real", "custom_kspace"}
SUPPORTED_RADIAL_PROFILE_TYPES = BUILTIN_RADIAL_PROFILE_TYPES | CUSTOM_RADIAL_PROFILE_TYPES

_GAUSS_LEGENDRE_CACHE = {}
_RADIAL_TABLE_CACHE = OrderedDict()
_RADIAL_TABLE_CACHE_LIMIT = 256

# The table is locally cubically interpolated while a Fourier kernel is built.
# These defaults keep its interpolation error below the projection error for
# the built-in profiles, while callers can override them per profile.
_DEFAULT_QUADRATURE_POINTS = 1024
_DEFAULT_QUADRATURE_POINTS_PER_CYCLE = 32
_DEFAULT_TABLE_POINTS = 4097
_DEFAULT_TABLE_POINTS_PER_CYCLE = 64


def real_space_profile(func, r_max, *, r_min=0.0, normalization="unit_integral", allow_signed=False,
                       quadrature_points=None, table_points=None, args=None):
    """Describe a user-defined real-space radial profile for 3PCF multipoles.

    ``func`` receives a NumPy array of radii and the sampled ``len_args`` as
    keyword arguments, and returns ``w(r)``.  The default normalization is
    ``4*pi*int r^2 w(r) dr = 1`` over the supplied support.
    """
    if not callable(func):
        raise TypeError("real_space_profile requires a callable func(r, **len_args).")
    return {
        "kind": "real_callable",
        "func": func,
        "r_min": float(r_min),
        "r_max": float(r_max),
        "normalization": str(normalization),
        "allow_signed": bool(allow_signed),
        "quadrature_points": quadrature_points,
        "table_points": table_points,
        "args": {} if args is None else dict(args),
    }


def tabulated_real_space_profile(r, w, *, normalization="unit_integral", allow_signed=False,
                                 quadrature_points=None, table_points=None):
    """Describe a tabulated user-defined real-space radial profile."""
    r = np.asarray(r, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    if r.ndim != 1 or w.ndim != 1 or r.size != w.size or r.size < 2:
        raise ValueError("tabulated_real_space_profile requires matching one-dimensional r and w arrays.")
    return {
        "kind": "real_tabulated",
        "r": r,
        "w": w,
        "normalization": str(normalization),
        "allow_signed": bool(allow_signed),
        "quadrature_points": quadrature_points,
        "table_points": table_points,
    }


def kspace_profile(func, r_max, inverse_k_max, *, r_min=0.0, normalization="unit_integral",
                   allow_signed=True, inverse_points=None, quadrature_points=None, table_points=None,
                   args=None):
    """Describe a user-defined isotropic k-space profile for inverse-Hankel use."""
    if not callable(func):
        raise TypeError("kspace_profile requires a callable func(k, **len_args).")
    return {
        "kind": "kspace_callable",
        "func": func,
        "r_min": float(r_min),
        "r_max": float(r_max),
        "inverse_k_max": float(inverse_k_max),
        "normalization": str(normalization),
        "allow_signed": bool(allow_signed),
        "inverse_points": inverse_points,
        "quadrature_points": quadrature_points,
        "table_points": table_points,
        "args": {} if args is None else dict(args),
    }


def _require_length(len_args, name, radial_type):
    if name not in len_args:
        raise ValueError(f"radial profile '{radial_type}' requires len_args['{name}'].")
    value = float(len_args[name])
    if not np.isfinite(value):
        raise ValueError(f"radial profile '{radial_type}' requires finite len_args['{name}'].")
    return value


def validate_radial_profile_request(radial_type, len_args, profile_config=None):
    """Validate profile-specific parameters without constructing a kernel."""
    radial_type = str(radial_type).strip().lower()
    if radial_type not in SUPPORTED_RADIAL_PROFILE_TYPES:
        supported = ", ".join(sorted(SUPPORTED_RADIAL_PROFILE_TYPES))
        raise ValueError(f"Unsupported radial profile '{radial_type}'. Supported profiles: {supported}.")

    if radial_type in {"shell", "sphere", "gaussian", "thick_shell"}:
        radius = _require_length(len_args, "R", radial_type)
        if radius < 0.0:
            raise ValueError(f"radial profile '{radial_type}' requires a non-negative R.")
    if radial_type == "thick_shell":
        width = _require_length(len_args, "delta_R", radial_type)
        if width <= 0.0:
            raise ValueError("radial profile 'thick_shell' requires delta_R > 0.")
    if radial_type == "gaussian_shell":
        radius = _require_length(len_args, "R_shell", radial_type)
        smooth = _require_length(len_args, "R_smooth", radial_type)
        if radius < 0.0 or smooth < 0.0:
            raise ValueError("radial profile 'gaussian_shell' requires non-negative R_shell and R_smooth.")
    if radial_type in CUSTOM_RADIAL_PROFILE_TYPES:
        profile = _resolve_profile_config(profile_config)
        expected_kind = "real" if radial_type == "custom_real" else "kspace"
        if not str(profile.get("kind", "")).startswith(expected_kind):
            raise ValueError(f"radial profile '{radial_type}' requires a {expected_kind}-space profile specification.")
        _validate_profile_support(profile)


def _resolve_profile_config(profile_config):
    if profile_config is None:
        raise ValueError("Custom radial profiles require other_args['profile'].")
    if not isinstance(profile_config, dict):
        raise TypeError("Custom radial profile configuration must be a dictionary.")
    profile = profile_config.get("profile", profile_config)
    if not isinstance(profile, dict):
        raise TypeError("other_args['profile'] must be a dictionary created by real_space_profile or kspace_profile.")
    merged = dict(profile)
    for key in (
        "r_min", "r_max", "inverse_k_max", "normalization", "allow_signed",
        "quadrature_points", "inverse_points", "table_points",
    ):
        if key in profile_config:
            merged[key] = profile_config[key]
    return merged


def _validate_profile_support(profile):
    kind = str(profile.get("kind", "")).strip().lower()
    if kind in {"real_callable", "kspace_callable"} and not callable(profile.get("func")):
        raise TypeError(f"Profile kind '{kind}' requires a callable func.")
    if kind == "real_tabulated":
        r = np.asarray(profile.get("r"), dtype=np.float64)
        w = np.asarray(profile.get("w"), dtype=np.float64)
        if r.ndim != 1 or w.ndim != 1 or r.size != w.size or r.size < 2:
            raise ValueError("real_tabulated profiles require matching one-dimensional r and w arrays.")
        if not np.all(np.isfinite(r)) or not np.all(np.isfinite(w)) or np.any(np.diff(r) <= 0.0):
            raise ValueError("real_tabulated profile radii must be finite and strictly increasing.")
        return
    if kind not in {"real_callable", "kspace_callable"}:
        raise ValueError(f"Unsupported custom radial profile kind '{kind}'.")
    if "r_min" not in profile or "r_max" not in profile:
        raise ValueError(f"Profile kind '{kind}' requires finite r_min and r_max.")
    r_min = float(profile["r_min"])
    r_max = float(profile["r_max"])
    if not np.isfinite(r_min) or not np.isfinite(r_max) or r_min < 0.0 or r_max <= r_min:
        raise ValueError("Custom radial profile support must satisfy 0 <= r_min < r_max.")
    if kind == "kspace_callable":
        k_max = float(profile.get("inverse_k_max", np.nan))
        if not np.isfinite(k_max) or k_max <= 0.0:
            raise ValueError("kspace_callable profiles require a finite positive inverse_k_max.")


def _builtin_real_profile(radial_type, len_args):
    radial_type = str(radial_type).strip().lower()
    if radial_type == "sphere":
        radius = _require_length(len_args, "R", radial_type)
        if radius <= 0.0:
            raise ValueError("sphere radial multipoles require R > 0.")

        def profile(r):
            return np.where(r <= radius, 3.0 / (4.0 * np.pi * radius ** 3), 0.0)

        return profile, 0.0, radius, {"normalization": "provided", "allow_signed": False}

    if radial_type == "thick_shell":
        radius = _require_length(len_args, "R", radial_type)
        width = _require_length(len_args, "delta_R", radial_type)
        if width <= 0.0:
            raise ValueError("thick_shell radial multipoles require delta_R > 0.")
        r_min = max(0.0, radius - 0.5 * width)
        r_max = radius + 0.5 * width
        volume = 4.0 * np.pi * (r_max ** 3 - r_min ** 3) / 3.0

        def profile(r):
            return np.where((r >= r_min) & (r <= r_max), 1.0 / volume, 0.0)

        return profile, r_min, r_max, {"normalization": "provided", "allow_signed": False}

    if radial_type == "gaussian":
        radius = _require_length(len_args, "R", radial_type)
        if radius <= 0.0:
            raise ValueError("gaussian radial multipoles require R > 0.")
        norm = (2.0 * np.pi * radius * radius) ** (-1.5)

        def profile(r):
            return norm * np.exp(-0.5 * (r / radius) ** 2)

        return profile, 0.0, 8.0 * radius, {"normalization": "unit_integral", "allow_signed": False}

    raise ValueError(f"No analytic real-space profile is available for '{radial_type}'.")


def _standard_monopole_transfer(radial_type, k, len_args):
    radial_type = str(radial_type).strip().lower()
    k = np.asarray(k, dtype=np.float64)
    if radial_type == "shell":
        radius = _require_length(len_args, "R", radial_type)
        return spherical_jn(0, 2.0 * np.pi * k * radius)
    if radial_type == "sphere":
        radius = _require_length(len_args, "R", radial_type)
        q = 2.0 * np.pi * k * radius
        result = np.ones_like(q)
        nonzero = q != 0.0
        result[nonzero] = 3.0 * (np.sin(q[nonzero]) - q[nonzero] * np.cos(q[nonzero])) / q[nonzero] ** 3
        return result
    if radial_type == "thick_shell":
        radius = _require_length(len_args, "R", radial_type)
        width = _require_length(len_args, "delta_R", radial_type)
        r_min = max(0.0, radius - 0.5 * width)
        r_max = radius + 0.5 * width
        denominator = r_max ** 3 - r_min ** 3
        if denominator == 0.0:
            return np.ones_like(k)
        return (
            r_max ** 3 * _standard_monopole_transfer("sphere", k, {"R": r_max})
            - r_min ** 3 * _standard_monopole_transfer("sphere", k, {"R": r_min})
        ) / denominator
    if radial_type == "gaussian":
        radius = _require_length(len_args, "R", radial_type)
        q = 2.0 * np.pi * k * radius
        return np.exp(-0.5 * q * q)
    if radial_type == "gaussian_shell":
        radius = _require_length(len_args, "R_shell", radial_type)
        smooth = _require_length(len_args, "R_smooth", radial_type)
        q_shell = 2.0 * np.pi * k * radius
        q_smooth = 2.0 * np.pi * k * smooth
        denominator = radius * radius + smooth * smooth
        if denominator == 0.0:
            return np.ones_like(k)
        shell_part = np.ones_like(q_shell)
        nonzero = q_shell != 0.0
        shell_part[nonzero] = np.sin(q_shell[nonzero]) / q_shell[nonzero]
        return (
            (smooth * smooth * np.cos(q_shell) + radius * radius * shell_part) / denominator
            * np.exp(-0.5 * q_smooth * q_smooth)
        )
    raise ValueError(f"No built-in k-space monopole is available for '{radial_type}'.")


def _gauss_legendre_interval(r_min, r_max, points):
    points = int(points)
    if points < 8:
        raise ValueError("Radial Hankel quadrature requires at least 8 points.")
    nodes_weights = _GAUSS_LEGENDRE_CACHE.get(points)
    if nodes_weights is None:
        nodes_weights = leggauss(points)
        _GAUSS_LEGENDRE_CACHE[points] = nodes_weights
    nodes, weights = nodes_weights
    half_width = 0.5 * (r_max - r_min)
    return half_width * (nodes + 1.0) + r_min, half_width * weights


def _adaptive_points(scale, minimum, points_per_cycle, configured, label):
    if configured is not None:
        points = int(configured)
        if points < 8:
            raise ValueError(f"{label} must be at least 8.")
        return points
    return max(int(minimum), int(math.ceil(points_per_cycle * max(float(scale), 1.0))) + 1)


def _evaluate_profile_function(func, radii, len_args, extra_args):
    kwargs = dict(extra_args)
    kwargs.update(len_args)
    try:
        values = np.asarray(func(radii, **kwargs), dtype=np.float64)
    except Exception as exc:
        raise ValueError("Custom radial profile callable failed for the requested sampled parameters.") from exc
    if values.ndim == 0:
        values = np.full(radii.shape, float(values), dtype=np.float64)
    if values.shape != radii.shape or not np.all(np.isfinite(values)):
        raise ValueError("Custom radial profile must return finite values with the same shape as its input.")
    return values


def _numerical_profile_options(options, profile_config):
    """Apply optional quadrature controls to a built-in profile."""
    options = dict(options)
    if profile_config is None:
        return options
    if not isinstance(profile_config, dict):
        raise TypeError("Radial profile configuration must be a dictionary.")
    for key in ("quadrature_points", "inverse_points", "table_points"):
        if key in profile_config:
            options[key] = profile_config[key]
    return options


def _real_quadrature(radial_type, len_args, profile_config, k_max):
    if radial_type in {"sphere", "thick_shell", "gaussian"}:
        func, r_min, r_max, options = _builtin_real_profile(radial_type, len_args)
        options = _numerical_profile_options(options, profile_config)
        points = _adaptive_points(
            k_max * (r_max - r_min),
            _DEFAULT_QUADRATURE_POINTS,
            _DEFAULT_QUADRATURE_POINTS_PER_CYCLE,
            options.get("quadrature_points"),
            "quadrature_points",
        )
        radii, weights = _gauss_legendre_interval(r_min, r_max, points)
        return radii, weights, func(radii), options

    profile = _resolve_profile_config(profile_config)
    _validate_profile_support(profile)
    kind = str(profile["kind"]).strip().lower()
    if kind == "real_tabulated":
        tab_r = np.asarray(profile["r"], dtype=np.float64)
        tab_w = np.asarray(profile["w"], dtype=np.float64)
        r_min, r_max = float(tab_r[0]), float(tab_r[-1])
        points = _adaptive_points(
            k_max * (r_max - r_min), max(_DEFAULT_QUADRATURE_POINTS, tab_r.size),
            _DEFAULT_QUADRATURE_POINTS_PER_CYCLE,
            profile.get("quadrature_points"), "quadrature_points"
        )
        radii, weights = _gauss_legendre_interval(r_min, r_max, points)
        values = np.interp(radii, tab_r, tab_w)
    elif kind == "real_callable":
        r_min, r_max = float(profile["r_min"]), float(profile["r_max"])
        points = _adaptive_points(
            k_max * (r_max - r_min),
            _DEFAULT_QUADRATURE_POINTS,
            _DEFAULT_QUADRATURE_POINTS_PER_CYCLE,
            profile.get("quadrature_points"),
            "quadrature_points",
        )
        radii, weights = _gauss_legendre_interval(r_min, r_max, points)
        values = _evaluate_profile_function(profile["func"], radii, len_args, profile.get("args", {}))
    else:
        raise ValueError(f"Expected a real-space profile, got kind '{kind}'.")
    return radii, weights, values, profile


def _inverse_hankel_quadrature(radial_type, len_args, profile_config, k_max):
    if radial_type == "gaussian_shell":
        radius = _require_length(len_args, "R_shell", radial_type)
        smooth = _require_length(len_args, "R_smooth", radial_type)
        if smooth == 0.0:
            raise ValueError(
                "gaussian_shell with R_smooth=0 must use the exact shell path."
            )
        profile = {
            "kind": "builtin_inverse",
            "r_min": 0.0,
            "r_max": radius + 8.0 * smooth,
            "inverse_k_max": max(k_max, 8.0 / (2.0 * np.pi * smooth)),
            "normalization": "unit_integral",
            "allow_signed": True,
        }
        profile = _numerical_profile_options(profile, profile_config)

        def transfer(k):
            return _standard_monopole_transfer("gaussian_shell", k, len_args)

    else:
        profile = _resolve_profile_config(profile_config)
        _validate_profile_support(profile)
        if str(profile["kind"]).strip().lower() != "kspace_callable":
            raise ValueError("custom_kspace requires a profile created by kspace_profile.")

        def transfer(k):
            return _evaluate_profile_function(profile["func"], k, len_args, profile.get("args", {}))

    r_min = float(profile["r_min"])
    r_max = float(profile["r_max"])
    inverse_k_max = float(profile["inverse_k_max"])
    r_points = _adaptive_points(
        k_max * (r_max - r_min),
        _DEFAULT_QUADRATURE_POINTS,
        _DEFAULT_QUADRATURE_POINTS_PER_CYCLE,
        profile.get("quadrature_points"),
        "quadrature_points",
    )
    radii, r_weights = _gauss_legendre_interval(r_min, r_max, r_points)
    k_points = _adaptive_points(
        inverse_k_max * r_max,
        _DEFAULT_QUADRATURE_POINTS,
        _DEFAULT_QUADRATURE_POINTS_PER_CYCLE,
        profile.get("inverse_points"),
        "inverse_points",
    )
    k_nodes, k_weights = _gauss_legendre_interval(0.0, inverse_k_max, k_points)
    transfer_values = transfer(k_nodes)
    prefactor = 4.0 * np.pi * k_weights * k_nodes ** 2 * transfer_values
    values = np.empty_like(radii)
    batch_size = 64
    for start in range(0, radii.size, batch_size):
        stop = min(start + batch_size, radii.size)
        values[start:stop] = spherical_jn(0, 2.0 * np.pi * np.outer(radii[start:stop], k_nodes)) @ prefactor
    return radii, r_weights, values, profile


def _normalize_profile_values(radii, weights, values, options):
    values = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("Radial profile evaluation produced non-finite values.")
    allow_signed = bool(options.get("allow_signed", False))
    if not allow_signed and np.any(values < 0.0):
        raise ValueError("Real-space radial profile has negative values; set allow_signed=True for compensated filters.")
    normalization = str(options.get("normalization", "unit_integral")).strip().lower()
    if normalization == "provided":
        return values
    if normalization != "unit_integral":
        raise ValueError("Radial profile normalization must be 'unit_integral' or 'provided'.")
    norm = 4.0 * np.pi * np.sum(weights * radii ** 2 * values)
    if not np.isfinite(norm) or np.isclose(norm, 0.0):
        raise ValueError("Radial profile has a non-finite or zero volume integral.")
    return values / norm


def _forward_hankel(radii, weights, values, l, k_values):
    prefactor = 4.0 * np.pi * weights * radii ** 2 * values
    output = np.empty(k_values.size, dtype=np.float64)
    batch_size = 64
    for start in range(0, k_values.size, batch_size):
        stop = min(start + batch_size, k_values.size)
        output[start:stop] = spherical_jn(l, 2.0 * np.pi * np.outer(k_values[start:stop], radii)) @ prefactor
    return output


def _freeze_for_cache(value):
    if callable(value):
        return ("callable", id(value))
    if isinstance(value, np.ndarray):
        digest = hashlib.sha1(np.ascontiguousarray(value).view(np.uint8)).hexdigest()
        return ("array", value.shape, str(value.dtype), digest)
    if isinstance(value, dict):
        return tuple(sorted((str(key), _freeze_for_cache(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_for_cache(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _table_cache_key(radial_type, len_args, profile_config, l, k_max):
    return (
        str(radial_type).strip().lower(),
        _freeze_for_cache(dict(len_args)),
        _freeze_for_cache(profile_config or {}),
        int(l),
        float(k_max),
    )


def build_radial_multipole_table(radial_type, len_args, l, k_max, profile_config=None):
    """Return a uniform physical-k table for ``U_l(k)``.

    The table is built from analytic real-space profiles, inverse-Hankel
    recovery of an isotropic k-space profile, or a user-supplied real profile.
    Thin shells are intentionally excluded here because their exact kernel is
    evaluated directly by :mod:`radial_multipole_windows`.
    """
    radial_type = str(radial_type).strip().lower()
    validate_radial_profile_request(radial_type, len_args, profile_config)
    if radial_type == "shell":
        raise ValueError("shell uses the analytic multipole kernel and does not require a radial table.")
    if not np.isfinite(k_max) or k_max <= 0.0:
        raise ValueError("k_max must be finite and positive when tabulating radial multipole windows.")

    cache_key = _table_cache_key(radial_type, len_args, profile_config, l, k_max)
    cached = _RADIAL_TABLE_CACHE.get(cache_key)
    if cached is not None:
        _RADIAL_TABLE_CACHE.move_to_end(cache_key)
        return cached

    if radial_type == "gaussian_shell" and float(len_args["R_smooth"]) == 0.0:
        radius = _require_length(len_args, "R_shell", radial_type)
        table_points = _adaptive_points(
            k_max * radius,
            _DEFAULT_TABLE_POINTS,
            _DEFAULT_TABLE_POINTS_PER_CYCLE,
            None,
            "table_points",
        )
        k_values = np.linspace(0.0, float(k_max), table_points, dtype=np.float64)
        result = (float(k_max), np.ascontiguousarray(spherical_jn(int(l), 2.0 * np.pi * k_values * radius)))
        _RADIAL_TABLE_CACHE[cache_key] = result
        _RADIAL_TABLE_CACHE.move_to_end(cache_key)
        while len(_RADIAL_TABLE_CACHE) > _RADIAL_TABLE_CACHE_LIMIT:
            _RADIAL_TABLE_CACHE.popitem(last=False)
        return result

    if radial_type in {"sphere", "thick_shell", "gaussian", "custom_real"}:
        radii, weights, values, options = _real_quadrature(radial_type, len_args, profile_config, k_max)
    elif radial_type in {"gaussian_shell", "custom_kspace"}:
        radii, weights, values, options = _inverse_hankel_quadrature(radial_type, len_args, profile_config, k_max)
    else:
        raise ValueError(f"Unsupported tabulated radial profile '{radial_type}'.")

    values = _normalize_profile_values(radii, weights, values, options)
    r_max = float(np.max(radii))
    configured_points = options.get("table_points")
    table_points = _adaptive_points(
        k_max * r_max,
        _DEFAULT_TABLE_POINTS,
        _DEFAULT_TABLE_POINTS_PER_CYCLE,
        configured_points,
        "table_points",
    )
    k_values = np.linspace(0.0, float(k_max), table_points, dtype=np.float64)
    table = _forward_hankel(radii, weights, values, int(l), k_values)
    table = np.ascontiguousarray(table, dtype=np.float64)
    result = (float(k_max), table)
    _RADIAL_TABLE_CACHE[cache_key] = result
    _RADIAL_TABLE_CACHE.move_to_end(cache_key)
    while len(_RADIAL_TABLE_CACHE) > _RADIAL_TABLE_CACHE_LIMIT:
        _RADIAL_TABLE_CACHE.popitem(last=False)
    return result
