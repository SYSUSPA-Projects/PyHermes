import copy

import numpy as np


ANISOTROPIC_AUTO_WINDOW_TYPES = {"ring", "disk", "cylinder", "cylshell"}
COMPLEX_RFFT_WINDOW_TYPES = {"directional_derivative"}
COMPLEX_FULL_FFT_WINDOW_TYPES = {"legendre_multipole", "radial_multipole"}
VALID_KERNEL_MODES = {"auto", "octant", "full_rfft", "complex_rfft", "complex_full_fft"}
LOS_ARG_KEYS = ("nx", "ny", "nz")
DEFAULT_LOS_ARGS = {"nx": 0.0, "ny": 0.0, "nz": 1.0}
LOS_AWARE_WINDOW_TYPES = ANISOTROPIC_AUTO_WINDOW_TYPES | COMPLEX_RFFT_WINDOW_TYPES
BUILTIN_BINNING_WINDOW_TYPES = {"shell"} | ANISOTROPIC_AUTO_WINDOW_TYPES


def default_binning_window():
    return {
        "type": "shell",
        "len_args": {"R": None},
        "los_args": {},
        "other_args": {},
        "mapping": "s_to_R",
        "kernel_mode": "octant",
    }


def normalize_len_args(len_args):
    if len_args is None:
        return {}
    if isinstance(len_args, dict):
        return copy.deepcopy(len_args)
    if isinstance(len_args, str):
        return {len_args: None}
    if isinstance(len_args, (list, tuple)):
        normalized = {}
        for item in len_args:
            if not isinstance(item, str):
                raise TypeError("len_args entries must be strings.")
            normalized[item] = None
        return normalized
    raise TypeError("len_args must be a dict, string, list, tuple, or None.")


def merge_len_arg_defaults(len_args, names):
    normalized = normalize_len_args(len_args)
    for name in names:
        normalized.setdefault(name, None)
    return normalized


def normalize_los_args(los_args, window_type=None):
    if los_args is None:
        los_args = {}
    if isinstance(los_args, (list, tuple, np.ndarray)):
        arr = np.asarray(los_args, dtype=np.float64)
        if arr.shape != (3,):
            raise ValueError("los_args array must contain exactly three values: [nx, ny, nz].")
        los_args = {key: float(value) for key, value in zip(LOS_ARG_KEYS, arr)}
    elif isinstance(los_args, dict):
        los_args = copy.deepcopy(los_args)
    else:
        raise TypeError("los_args must be a dict, length-3 array, or None.")

    if not los_args and window_type in LOS_AWARE_WINDOW_TYPES:
        los_args = copy.deepcopy(DEFAULT_LOS_ARGS)
    if los_args:
        if not all(key in los_args for key in LOS_ARG_KEYS):
            raise ValueError("los_args must define nx, ny, and nz together.")
        los = np.array([los_args[key] for key in LOS_ARG_KEYS], dtype=np.float64)
        if np.linalg.norm(los) == 0.0:
            raise ValueError("los_args vector must be non-zero.")
        los_args = {key: float(los_args[key]) for key in LOS_ARG_KEYS}
    return los_args


def default_kernel_mode(window_type, has_custom_func=False):
    if window_type in COMPLEX_FULL_FFT_WINDOW_TYPES:
        return "complex_full_fft"
    if window_type in COMPLEX_RFFT_WINDOW_TYPES:
        return "complex_rfft"
    if window_type in ANISOTROPIC_AUTO_WINDOW_TYPES:
        return "auto"
    if has_custom_func:
        return "full_rfft"
    return "octant"


def normalize_kernel_mode(kernel_mode):
    kernel_mode = str(kernel_mode).strip().lower()
    if kernel_mode not in VALID_KERNEL_MODES:
        raise ValueError(
            f"Unsupported kernel_mode '{kernel_mode}'. "
            f"Supported values are {sorted(VALID_KERNEL_MODES)}."
        )
    return kernel_mode


def callable_metadata(value):
    target = getattr(value, "py_func", value)
    module = getattr(target, "__module__", getattr(value, "__module__", type(value).__module__))
    name = getattr(target, "__qualname__", getattr(target, "__name__", type(value).__name__))
    return {"kind": "callable", "name": f"{module}.{name}"}


def serialize_window_params(value):
    if isinstance(value, dict):
        serialized = {}
        for key, item in value.items():
            if key == "func":
                serialized["custom_func"] = callable_metadata(item)
            else:
                serialized[key] = serialize_window_params(item)
        return serialized
    if callable(value):
        return callable_metadata(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [serialize_window_params(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return copy.deepcopy(value)


def apply_builtin_binning_window_defaults(binning_window):
    params = copy.deepcopy(binning_window)
    window_type = params.get("type")
    if not window_type:
        return params
    window_type = str(window_type).strip().lower()
    params["type"] = window_type
    if window_type == "shell":
        params["len_args"] = merge_len_arg_defaults(params.get("len_args", {}), ("R",))
        params.setdefault("los_args", {})
        params.setdefault("other_args", {})
        params.setdefault("mapping", "s_to_R")
    elif window_type in LOS_AWARE_WINDOW_TYPES:
        params["len_args"] = merge_len_arg_defaults(params.get("len_args", {}), ("R", "H"))
        params.setdefault("los_args", copy.deepcopy(DEFAULT_LOS_ARGS))
        params.setdefault("other_args", {})
        params.setdefault("mapping", "smu_to_RH")
    return params


def binning_window_from_string(binning_window):
    window_type = binning_window.strip().lower()
    if window_type not in BUILTIN_BINNING_WINDOW_TYPES:
        raise ValueError(
            f"Unsupported binning_window string '{binning_window}'. "
            "Supported built-in strings are 'shell', 'ring', 'disk', "
            "'cylinder', and 'cylshell'."
        )
    return apply_builtin_binning_window_defaults({"type": window_type})


def normalize_binning_window_template(binning_window):
    if binning_window is None:
        binning_window = default_binning_window()
    elif isinstance(binning_window, str):
        binning_window = binning_window_from_string(binning_window)
    elif not isinstance(binning_window, dict):
        raise TypeError(
            f"Unsupported binning_window input: expected dict, string, or None, got {type(binning_window)}."
        )

    normalized = apply_builtin_binning_window_defaults(binning_window)
    if not normalized.get("type"):
        normalized["type"] = "custom" if normalized.get("func") is not None else "shell"
    normalized["len_args"] = normalize_len_args(normalized.get("len_args", {}))
    normalized["los_args"] = normalize_los_args(normalized.get("los_args", {}), normalized.get("type"))
    normalized.setdefault("other_args", {})
    kernel_mode = normalized.get("kernel_mode")
    if not kernel_mode:
        kernel_mode = default_kernel_mode(
            normalized.get("type"),
            has_custom_func=normalized.get("func") is not None,
        )
    normalized["kernel_mode"] = normalize_kernel_mode(kernel_mode)
    return normalized
