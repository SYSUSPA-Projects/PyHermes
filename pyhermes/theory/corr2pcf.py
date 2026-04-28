import time
import pickle
import copy
import hashlib
import json
import os

import numpy as np

from pyhermes.io import WindowFunc
from pyhermes.io import ConvolsData
from pyhermes.io import Corr2PCFData
from pyhermes.utils import func_util
from pyhermes.utils.convolution import specialized_convolution_3d
from pyhermes.pipeline import TaskBase


# Product and runtime configuration.
PRODUCT_RULES = {
    "allowed": {"dd", "dr", "rd", "delta_dd", "rr", "xi"},
    "deps": {
        "xi": ["delta_dd", "rr"],
    },
}

PRODUCT_INPUT_FLAGS = {
    "dd": (True, False),
    "dr": (True, True),
    "rd": (True, True),
    "delta_dd": (True, True),
    "rr": (False, True),
    "xi": (True, True),
}

ANISOTROPIC_AUTO_WINDOW_TYPES = {"ring", "disk", "cylinder"}
VALID_KERNEL_MODES = {"auto", "octant", "full_rfft"}


def parse_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def los_to_vector(los):
    if isinstance(los, str):
        axis = los.strip().lower()
        if axis == "x":
            return (1.0, 0.0, 0.0)
        if axis == "y":
            return (0.0, 1.0, 0.0)
        if axis == "z":
            return (0.0, 0.0, 1.0)
        raise ValueError("Corr_2PCF 'los' must be one of 'x', 'y', 'z', or a length-3 array.")
    arr = np.asarray(los, dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError("Corr_2PCF 'los' must be one of 'x', 'y', 'z', or a length-3 array.")
    if np.linalg.norm(arr) == 0.0:
        raise ValueError("Corr_2PCF 'los' vector must be non-zero.")
    return tuple(float(v) for v in arr)


def default_pair_window(mode, los_vector):
    if mode == "smu":
        nx, ny, nz = los_vector
        return {
            "type": "ring",
            "len_args": {"R": None, "H": None},
            "other_args": {"nx": nx, "ny": ny, "nz": nz},
            "mapping": "smu_to_RH",
            "kernel_mode": "auto",
        }
    return {"type": "shell", "len_args": {"R": None}, "other_args": {}, "mapping": "s_to_R", "kernel_mode": "octant"}


def default_kernel_mode(window_type, has_custom_func=False):
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


def normalize_products(products):
    if isinstance(products, str):
        products = [products]
    elif products is None:
        products = ["xi"]
    elif not isinstance(products, (list, tuple, set)):
        raise TypeError(
            f"Unsupported products input: expected string or array of strings, got {type(products)}."
        )

    allowed = PRODUCT_RULES["allowed"]
    normalized = []
    for item in products:
        if not isinstance(item, str):
            raise TypeError("Each product name must be a string.")
        name = item.strip().lower()
        if name not in allowed:
            raise ValueError(f"Unsupported product '{item}'. Allowed values are {sorted(allowed)}.")
        if name not in normalized:
            normalized.append(name)
    return normalized


def expand_products(products):
    expanded = list(products)
    idx = 0
    while idx < len(expanded):
        for dep in PRODUCT_RULES["deps"].get(expanded[idx], []):
            if dep not in expanded:
                expanded.append(dep)
        idx += 1
    return expanded


def normalize_sampling_array(values, name, positive=True):
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise TypeError(f"'{name}' must be a 1D array-like input.")
    if arr.size == 0:
        raise ValueError(f"'{name}' must contain at least one sampling point.")
    if positive and np.any(arr <= 0.0):
        raise ValueError(f"'{name}' values must be strictly positive.")
    diffs = np.diff(arr)
    if np.any(diffs < 0.0) and np.any(diffs > 0.0):
        raise ValueError(f"'{name}' values must be monotonic.")
    return np.ascontiguousarray(arr, dtype=np.float64)


# Logging and serialization helpers.
def describe_sampling(mode, n_s, s_min, s_max, n_mu, mu_min, mu_max, s_spec, mu_spec):
    base = f"mode={mode}, n_s={n_s}, s_min={s_min}, s_max={s_max}"
    if mode == "smu":
        base += f", n_mu={n_mu}, mu_min={mu_min}, mu_max={mu_max}"
    if isinstance(s_spec, dict) and (mode == "s" or isinstance(mu_spec, dict)):
        return base
    return f"{base}, source=explicit sampling array"


def format_product_list(products):
    return "[" + ", ".join(products) + "]"


def describe_products(products, expanded_products):
    computed = [product for product in expanded_products if product != "xi"]
    derived = ["xi"] if "xi" in expanded_products else []
    text = (
        f"Products: requested={format_product_list(products)} | "
        f"computed={format_product_list(computed)}"
    )
    if derived:
        text += f" | derived={format_product_list(derived)}"
    return text


def describe_task_distribution(total_tasks, n_ranks):
    min_tasks = total_tasks // n_ranks
    max_tasks = min_tasks + (1 if total_tasks % n_ranks else 0)
    return f"Sampling tasks: total={total_tasks}, ranks={n_ranks}, per_rank={min_tasks}-{max_tasks}"


def effective_pair_los(pair_window, mode, los_vector):
    if mode != "smu" or not isinstance(pair_window, dict):
        return None
    other_args = pair_window.get("other_args", {})
    if not all(axis in other_args for axis in ("nx", "ny", "nz")):
        return None
    los_values = tuple(other_args.get(axis) for axis in ("nx", "ny", "nz"))
    if all(value is not None for value in los_values):
        return los_values
    return los_vector


def describe_pair_window(pair_window, mode, los_vector):
    if isinstance(pair_window, dict):
        mapping = pair_window.get("mapping", "custom")
        mapping_name = mapping if isinstance(mapping, str) else getattr(mapping, "__name__", "custom callable")
        parts = [f"type={pair_window.get('type', 'custom')}", f"mapping={mapping_name}"]
        if pair_window.get("kernel_mode") is not None:
            parts.append(f"kernel_mode={pair_window.get('kernel_mode')}")
        # len_args = pair_window.get("len_args", {})
        # other_args = pair_window.get("other_args", {})
        # if len_args:
        #     parts.append(f"len_args={len_args}")
        # if other_args:
        #     parts.append(f"other_args={other_args}")
        pair_los = effective_pair_los(pair_window, mode, los_vector)
        if pair_los is not None:
            parts.append(f"los={pair_los}")
        return "Pair window: " + " | ".join(parts)
    return "Pair window: default shell mapping"


def compact_window_desc(win):
    if win is None:
        return "window=none"
    if isinstance(win, dict):
        args = win.get("len_args", {})
        if args:
            return f"window={win.get('type', 'custom')} {args}"
        return f"window={win.get('type', 'custom')}"
    if isinstance(win, WindowFunc):
        args = getattr(win, "len_args", {})
        if args:
            return f"window={getattr(win, 'type', 'custom')} {args}"
        return f"window={getattr(win, 'type', 'custom')}"
    return "window=custom"


def serialize_convols_input(value):
    if isinstance(value, str):
        return value
    if value == "uniform":
        return "uniform"
    if isinstance(value, (float, int, np.floating)):
        return float(value)
    if isinstance(value, ConvolsData):
        return {
            "kind": "ConvolsData",
            "L": value.L,
            "box_size": value.box_size,
            "wavelet_mode": value.wavelet_mode,
            "wavelet_level": value.wavelet_level,
        }
    return value


def serialize_window_input(value):
    if isinstance(value, dict):
        return copy.deepcopy(value)
    if isinstance(value, WindowFunc):
        return {
            "kind": "WindowFunc",
            "type": getattr(value, "type", "custom"),
            "len_args": copy.deepcopy(getattr(value, "len_args", {})),
            "other_args": copy.deepcopy(getattr(value, "other_args", {})),
        }
    return value


# Sampling and result helpers.
def make_sampling_tasks(mode, s_arr, mu_arr):
    if mode == "smu":
        tasks = np.array(
            [(i, j, s_arr[i], mu_arr[j]) for i in range(len(s_arr)) for j in range(len(mu_arr))],
            dtype=np.float64,
        )
        return tasks, (len(s_arr), len(mu_arr))
    tasks = np.array([(i, -1, s_arr[i], np.nan) for i in range(len(s_arr))], dtype=np.float64)
    return tasks, (len(s_arr),)


def build_result_from_gathered(gathered_tasks, gathered_values, result_shape, mode):
    result = np.empty(result_shape, dtype=np.float64)
    flat_values = [item for sublist in gathered_values for item in sublist]
    flat_tasks = [item for sublist in gathered_tasks for item in sublist]
    for (s_idx, mu_idx), value in zip(flat_tasks, flat_values):
        if mode == "smu":
            result[s_idx, mu_idx] = value
        else:
            result[s_idx] = value
    return result


# Pair-window mappings.
def mapping_s_to_R(s, mu, pair_window):
    params = copy.deepcopy(pair_window)
    params.setdefault("len_args", {})
    if params["len_args"].get("R") is None:
        params["len_args"]["R"] = s
    return params


def mapping_smu_to_RH(s, mu, pair_window, los_vector=None):
    if mu is None:
        raise ValueError("Corr_2PCF mapping='smu_to_RH' requires a mu value.")
    params = copy.deepcopy(pair_window)
    params.setdefault("len_args", {})
    params.setdefault("other_args", {})
    if params["len_args"].get("R") is None:
        params["len_args"]["R"] = s * np.sqrt(max(0.0, 1.0 - mu * mu))
    if params["len_args"].get("H") is None:
        params["len_args"]["H"] = s * mu
    if los_vector is not None:
        for key, value in zip(("nx", "ny", "nz"), los_vector):
            if key in params["other_args"] and params["other_args"][key] is None:
                params["other_args"][key] = value
    return params


PAIR_WINDOW_MAPPING_MODES = {
    "s_to_R": "s",
    "smu_to_RH": "smu",
}

PAIR_WINDOW_MAPPINGS = {
    "s_to_R": mapping_s_to_R,
    "smu_to_RH": mapping_smu_to_RH,
}


# Pair-product kernels.
def field_density(field):
    if isinstance(field, (float, int, np.floating)):
        return float(field)
    if isinstance(field, ConvolsData):
        return 1.0 / field.V
    raise TypeError(f"Unsupported field type for density extraction: {type(field)}")


def pair_product_with_window(field1, field2, pair_window_obj, threads):
    if field2 is None:
        field2 = field1
    if isinstance(field1, (float, int, np.floating)) or isinstance(field2, (float, int, np.floating)):
        return field_density(field1) * field_density(field2)
    conv = specialized_convolution_3d(field1.epsilon, pair_window_obj.as_array(), threads=threads)
    return float(np.einsum("ijk,ijk->", conv, field2.epsilon, optimize=True) / conv.size)


def compute_pair_product_at_smu(s, mu, convols_data1, convols_data2=None, pair_window=None):
    if convols_data2 is None:
        convols_data2 = convols_data1
    if isinstance(convols_data1, (float, int, np.floating)) or isinstance(convols_data2, (float, int, np.floating)):
        return field_density(convols_data1) * field_density(convols_data2)
    if pair_window is None:
        pair_window = {"type": "shell", "len_args": {"R": None}, "other_args": {}, "mapping": "s_to_R"}
    mapping = pair_window.get("mapping")
    if mapping is None:
        mapping = "smu_to_RH" if pair_window.get("type") in ("ring", "disk", "cylinder") and mu is not None else "s_to_R"
    if isinstance(mapping, str):
        if mapping not in PAIR_WINDOW_MAPPINGS:
            raise ValueError(
                f"Unsupported pair_window mapping '{mapping}'. "
                f"Supported built-in mappings: {sorted(PAIR_WINDOW_MAPPINGS)}."
            )
        mapper = PAIR_WINDOW_MAPPINGS[mapping]
    elif callable(mapping):
        mapper = mapping
    else:
        raise TypeError("pair_window mapping must be a string or callable.")
    pair_window_params = mapper(s, mu, pair_window)
    if not isinstance(pair_window_params, dict):
        raise TypeError("pair_window mapping must return a pair_window dictionary.")
    pair_window_params = copy.deepcopy(pair_window_params)
    pair_window_params.pop("mapping", None)
    pair_window_params.setdefault("len_args", {})
    pair_window_params.setdefault("other_args", {})
    pair_window_obj = WindowFunc(pair_window_params, convols_data1.convols_info, threads=convols_data1.threads)
    return pair_product_with_window(convols_data1, convols_data2, pair_window_obj, convols_data1.threads)


class Corr_2PCF(TaskBase):

    def __init__(self, param_task=None):
        if param_task is None:
            param_task = {"Corr_2PCF": {}}
        self.task_name = str(self.__class__.__name__)
        super().__init__(param_task=param_task)
        self.format_params()
        self.pair_window = None
        self._fields_prepared = False

    # Parameter formatting and validation.
    def _sync_runtime_options(self, log_runtime=True):
        self.mode = str(self.mode).strip().lower()
        if self.mode not in ("s", "smu"):
            raise ValueError("Corr_2PCF mode must be either 's' or 'smu'.")
        self.los_vector = los_to_vector(self.los)
        self._sync_sampling_attribute_overrides()
        if self._pair_window_from_default and self.pair_window is None:
            self.pair_window_params = default_pair_window(self.mode, self.los_vector)
        self.threads = max(1, int(self.threads))
        self.memory_strategy = str(self.memory_strategy).strip().lower()
        if self.memory_strategy not in ("speed", "memory"):
            raise ValueError("Corr_2PCF memory_strategy must be either 'speed' or 'memory'.")
        self.pair_window_cache = parse_bool(self.pair_window_cache)
        self.task_params['threads'] = self.threads
        self.task_params['products'] = copy.deepcopy(self.products)
        self.task_params['mode'] = self.mode
        self.task_params['los'] = copy.deepcopy(self.los)
        self.task_params['s'] = copy.deepcopy(self.s)
        self.task_params['mu'] = copy.deepcopy(self.mu)
        self.task_params['memory_strategy'] = self.memory_strategy
        self.task_params['pair_window_cache'] = self.pair_window_cache
        self.task_params['pair_window_cache_dir'] = self.pair_window_cache_dir
        if log_runtime:
            self.sync_runtime_options(context="Corr_2PCF runtime configuration")

    def format_params(self):
        self.convols_data = self.task_params.get('convols_data', '')
        self.convols_data1 = self.task_params.get('convols_data1', '') or self.convols_data
        self.convols_data2 = self.task_params.get('convols_data2', '') or self.convols_data
        self.random = self.task_params.get('random', None)
        self.random1 = self.task_params.get('random1', None)
        self.random2 = self.task_params.get('random2', None)
        if self.random1 in (None, ""):
            self.random1 = self.random
        if self.random2 in (None, ""):
            self.random2 = self.random

        window = self.task_params.get('window', None)
        self.window = window if (window and window.get('type')) else None
        for i in range(1, 3):
            window_i = self.task_params.get(f'window{i}', None)
            window_i = window_i if (window_i and window_i.get('type')) else None
            if (not window_i) and self.window:
                window_i = dict(self.window)
            setattr(self, f'window{i}', window_i)
        self.mode = str(self.task_params.get("mode", "s")).strip().lower()
        if self.mode not in ("s", "smu"):
            raise ValueError("Corr_2PCF mode must be either 's' or 'smu'.")
        self.los = self.task_params.get("los", "z")
        self.los_vector = los_to_vector(self.los)
        pair_window_params = self.task_params.get('pair_window', None)
        if pair_window_params and (pair_window_params.get('type') or pair_window_params.get('func') is not None):
            self.pair_window_params = copy.deepcopy(pair_window_params)
            self._pair_window_from_default = False
        else:
            self.pair_window_params = default_pair_window(self.mode, self.los_vector)
            self._pair_window_from_default = True

        self.s = copy.deepcopy(self.task_params['s'])
        self.mu = copy.deepcopy(self.task_params.get('mu', {"mu_min": 0.0, "mu_max": 1.0, "n_mu": 20}))
        self.s_arr = None
        self.mu_arr = None
        self.s_min = None
        self.s_max = None
        self.n_s = None
        self.mu_min = None
        self.mu_max = None
        self.n_mu = None
        self.threads = int(self.task_params['threads'])
        self.products = self._normalize_products(self.task_params.get('products', 'xi'))
        self.memory_strategy = str(self.task_params.get("memory_strategy", "speed")).strip().lower()
        self.pair_window_cache = parse_bool(self.task_params.get("pair_window_cache", False))
        self.pair_window_cache_dir = self.task_params.get("pair_window_cache_dir", "")
        self.fout_path = self.task_params['fout_path']

    # Sampling, products, and pair-window normalization.
    def _sync_sampling_attribute_overrides(self):
        if isinstance(self.s, dict):
            for attr in ("s_min", "s_max", "n_s"):
                value = getattr(self, attr, None)
                if value is not None:
                    self.s[attr] = value
        if isinstance(self.mu, dict):
            for attr in ("mu_min", "mu_max", "n_mu"):
                value = getattr(self, attr, None)
                if value is not None:
                    self.mu[attr] = value

    # Product and logging helpers.
    def _normalize_products(self, products):
        return normalize_products(products)

    def _expanded_products(self):
        return expand_products(self.products)

    def _describe_sampling(self):
        return describe_sampling(
            self.mode,
            self.n_s,
            self.s_min,
            self.s_max,
            self.n_mu,
            self.mu_min,
            self.mu_max,
            self.s,
            self.mu,
        )

    def _describe_products(self, expanded_products):
        return describe_products(self.products, expanded_products)

    def _describe_pair_window(self, pair_window):
        return describe_pair_window(pair_window, self.mode, self.los_vector)

    def _describe_task_distribution(self, total_tasks, n_ranks):
        return describe_task_distribution(total_tasks, n_ranks)

    def _normalize_pair_window(self, pair_window):
        if pair_window is None:
            return copy.deepcopy(self.pair_window_params)
        if not isinstance(pair_window, dict):
            raise TypeError(
                f"Unsupported pair_window input: expected dict or None, got {type(pair_window)}."
            )
        normalized = copy.deepcopy(pair_window)
        if not normalized.get("type"):
            normalized["type"] = "custom" if normalized.get("func") is not None else "shell"
        normalized.setdefault("len_args", {})
        normalized.setdefault("other_args", {})
        kernel_mode = normalized.get("kernel_mode")
        if not kernel_mode:
            kernel_mode = default_kernel_mode(
                normalized.get("type"),
                has_custom_func=normalized.get("func") is not None,
            )
        normalized["kernel_mode"] = normalize_kernel_mode(kernel_mode)
        if not normalized.get("mapping"):
            normalized["mapping"] = "smu_to_RH" if self.mode == "smu" else "s_to_R"
        mapping = normalized.get("mapping")
        if isinstance(mapping, str):
            expected_mode = PAIR_WINDOW_MAPPING_MODES.get(mapping)
            if expected_mode is None:
                raise ValueError(
                    f"Unsupported pair_window mapping '{mapping}'. "
                    f"Supported built-in mappings: {sorted(PAIR_WINDOW_MAPPING_MODES)}."
                )
            if expected_mode != self.mode:
                raise ValueError(
                    f"pair_window mapping '{mapping}' is only valid for mode='{expected_mode}', "
                    f"got mode='{self.mode}'."
                )
        elif not callable(mapping):
            raise TypeError("pair_window mapping must be a string or callable.")
        if self.mode == "smu" and normalized.get("type") in ("ring", "disk", "cylinder"):
            nx, ny, nz = self.los_vector
            normalized["other_args"].setdefault("nx", nx)
            normalized["other_args"].setdefault("ny", ny)
            normalized["other_args"].setdefault("nz", nz)
        return normalized

    def _resolve_sampling(self):
        if self.s is None:
            raise ValueError("Corr_2PCF requires 's' to be provided as a dict or array-like input.")
        if isinstance(self.s, dict):
            s_min = float(self.s["s_min"])
            s_max = float(self.s["s_max"])
            n_s = int(self.s["n_s"])
            s_arr = np.linspace(s_min, s_max, n_s, dtype=np.float64)
        else:
            s_arr = normalize_sampling_array(self.s, "s")
        self.s_arr = np.ascontiguousarray(s_arr, dtype=np.float64)
        self.n_s = int(self.s_arr.size)
        self.s_min = float(np.min(self.s_arr))
        self.s_max = float(np.max(self.s_arr))

        if self.mode == "smu":
            if self.mu is None:
                raise ValueError("Corr_2PCF mode='smu' requires 'mu' sampling.")
            if isinstance(self.mu, dict):
                mu_min = float(self.mu["mu_min"])
                mu_max = float(self.mu["mu_max"])
                n_mu = int(self.mu["n_mu"])
                mu_arr = np.linspace(mu_min, mu_max, n_mu, dtype=np.float64)
            else:
                mu_arr = normalize_sampling_array(self.mu, "mu", positive=False)
            self.mu_arr = np.ascontiguousarray(mu_arr, dtype=np.float64)
            self.n_mu = int(self.mu_arr.size)
            self.mu_min = float(np.min(self.mu_arr))
            self.mu_max = float(np.max(self.mu_arr))
        else:
            self.mu_arr = None
            self.n_mu = None
            self.mu_min = None
            self.mu_max = None

    def _current_task_params_snapshot(self):
        params = {}
        params['convols_data'] = serialize_convols_input(self.convols_data)
        params['convols_data1'] = serialize_convols_input(self.convols_data1)
        params['convols_data2'] = serialize_convols_input(self.convols_data2)
        params['random'] = serialize_convols_input(self.random)
        params['random1'] = serialize_convols_input(self.random1)
        params['random2'] = serialize_convols_input(self.random2)
        params['window'] = serialize_window_input(self.window)
        params['window1'] = serialize_window_input(self.window1)
        params['window2'] = serialize_window_input(self.window2)
        params['pair_window'] = copy.deepcopy(
            self.pair_window if self.pair_window is not None else self.pair_window_params
        )
        params['mode'] = self.mode
        params['los'] = copy.deepcopy(self.los)
        params['los_vector'] = list(self.los_vector)
        params['s_spec'] = copy.deepcopy(self.s) if isinstance(self.s, dict) else np.asarray(self.s, dtype=np.float64).tolist()
        params['s_min'] = self.s_min
        params['s_max'] = self.s_max
        params['n_s'] = self.n_s
        if self.mode == "smu":
            params['mu_spec'] = copy.deepcopy(self.mu) if isinstance(self.mu, dict) else np.asarray(self.mu, dtype=np.float64).tolist()
            params['mu_min'] = self.mu_min
            params['mu_max'] = self.mu_max
            params['n_mu'] = self.n_mu
        params['threads'] = self.threads
        params['products'] = copy.deepcopy(self.products)
        params['memory_strategy'] = self.memory_strategy
        params['pair_window_cache'] = self.pair_window_cache
        params['pair_window_cache_dir'] = self.pair_window_cache_dir
        params['fout_path'] = self.fout_path
        return params

    def _memory_leg_desc(self, source_desc, leg_idx):
        return f"{source_desc}, {compact_window_desc(getattr(self, f'window{leg_idx}'))}"

    # Input field resolution.
    def _resolve_base_convols(self, leg_idx, provided_convols, base_convols_cache):
        if provided_convols is not None:
            if isinstance(provided_convols, str):
                base_path = provided_convols
                if base_path not in base_convols_cache:
                    base_convols_cache[base_path] = ConvolsData(data_path=base_path, threads=self.threads)
                return base_convols_cache[base_path], f"path={base_path}"
            if not isinstance(provided_convols, ConvolsData):
                self.logger.error(
                    f"Unexpected input: 'convols_data{leg_idx}' must be a string path or a ConvolsData instance."
                )
                func_util.safe_exit(1)
            return provided_convols, f"provided convols_data{leg_idx}"

        base_input = getattr(self, f"convols_data{leg_idx}")
        if isinstance(base_input, str) and base_input:
            if base_input not in base_convols_cache:
                base_convols_cache[base_input] = ConvolsData(data_path=base_input, threads=self.threads)
            return base_convols_cache[base_input], f"path={base_input}"
        if isinstance(base_input, ConvolsData):
            return base_input, f"provided convols_data{leg_idx}"
        if base_input not in (None, ""):
            self.logger.error(
                f"Unexpected input: 'convols_data{leg_idx}' must be a string path or a ConvolsData instance."
            )
            func_util.safe_exit(1)
        if not self.convols_data and not self.convols_data1 and not self.convols_data2:
            self.logger.error(
                f"Missing input for field leg {leg_idx}. Please pass convols_data{leg_idx} or set "
                f"'convols_data{leg_idx}' / 'convols_data'."
            )
            func_util.safe_exit(1)
        self.logger.error(
            f"Missing usable input for field leg {leg_idx}. Expected a string path or ConvolsData instance in "
            f"'convols_data{leg_idx}' or shared 'convols_data'."
        )
        func_util.safe_exit(1)

    def _resolve_random_base(self, leg_idx, provided_random, base_convols_cache):
        if provided_random is None or provided_random == "":
            return None, "no random input"
        if provided_random == "uniform":
            return "uniform", "uniform random density"
        if isinstance(provided_random, str):
            base_path = provided_random
            if base_path not in base_convols_cache:
                base_convols_cache[base_path] = ConvolsData(data_path=base_path, threads=self.threads)
            return base_convols_cache[base_path], f"path={base_path}"
        if isinstance(provided_random, ConvolsData):
            return provided_random, f"provided random{leg_idx}"
        self.logger.error(
            f"Unexpected input: 'random{leg_idx}' must be 'uniform', a string path, a ConvolsData instance, or None."
        )
        func_util.safe_exit(1)

    def _field_density(self, field):
        return field_density(field)

    def _resolve_window(self, leg_idx, base_convols, provided_window):
        if provided_window is None:
            return None, "no additional window convolution"
        else:
            if isinstance(provided_window, WindowFunc):
                return provided_window, "provided WindowFunc instance"
            elif isinstance(provided_window, dict):
                return WindowFunc(provided_window, base_convols.convols_info, threads=self.threads), (
                    f"provided window dict | {func_util.describe_window_action(provided_window)}"
                )
            else:
                self.logger.error(
                    f"Unsupported window input for leg {leg_idx}. Expected dict, WindowFunc, or None, "
                    f"got {type(provided_window)}."
                )
                func_util.safe_exit(1)

    def _required_input_flags(self):
        needs_data = False
        needs_random = False
        for product in self._expanded_products():
            product_needs_data, product_needs_random = PRODUCT_INPUT_FLAGS[product]
            needs_data = needs_data or product_needs_data
            needs_random = needs_random or product_needs_random
        return needs_data, needs_random

    # Pair-product computation.
    def calc_pair_product(self, s, field1, field2=None, mu=None, pair_window=None):
        if field2 is None:
            field2 = field1
        if pair_window is None:
            pair_window = self.pair_window
        pair_window = self._normalize_pair_window(pair_window)
        if isinstance(field1, (float, int, np.floating)) or isinstance(field2, (float, int, np.floating)):
            return self._field_density(field1) * self._field_density(field2)
        return compute_pair_product_at_smu(s, mu, field1, field2, pair_window=pair_window)

    def _build_pair_window_for_sample(self, s, mu, reference_field):
        pair_window_params = self._pair_window_params_for_sample(s, mu, self.pair_window)
        pair_window_obj = WindowFunc(pair_window_params, reference_field.convols_info, threads=self.threads)
        if self.pair_window_cache:
            cache_path = self._pair_window_cache_path(pair_window_params, reference_field)
            if os.path.exists(cache_path):
                pair_window_obj.w_kernel = np.load(cache_path)
            else:
                pair_window_obj.as_array()
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                np.save(cache_path, pair_window_obj.w_kernel)
        return pair_window_obj

    def _pair_window_cache_path(self, pair_window_params, reference_field):
        cache_dir = self.pair_window_cache_dir
        if not cache_dir:
            if self.fout_path:
                cache_dir = os.path.join(os.path.dirname(self.fout_path) or ".", "pair_window_cache")
            else:
                cache_dir = ".pyhermes_pair_window_cache"
        cache_key = {
            "pair_window": pair_window_params,
            "convols_info": {
                key: reference_field.convols_info.get(key)
                for key in ConvolsData._REQUIRED_ARGV
            },
            "bandwidth": getattr(reference_field, "bandwidth", 1),
        }
        digest = hashlib.sha1(json.dumps(cache_key, sort_keys=True, default=str).encode("utf-8")).hexdigest()
        return os.path.join(cache_dir, f"{digest}.npy")

    def _pair_window_params_for_sample(self, s, mu, pair_window):
        if pair_window is None:
            pair_window = self._normalize_pair_window(None)
        elif not isinstance(pair_window, dict):
            raise TypeError(
                f"Unsupported pair_window input: expected dict, got {type(pair_window)}."
            )
        mapping = pair_window.get("mapping", "smu_to_RH" if self.mode == "smu" else "s_to_R")
        if isinstance(mapping, str):
            if mapping == "smu_to_RH":
                params = mapping_smu_to_RH(s, mu, pair_window, self.los_vector)
            else:
                params = PAIR_WINDOW_MAPPINGS[mapping](s, mu, pair_window)
        else:
            params = mapping(s, mu, pair_window)
        if not isinstance(params, dict):
            raise TypeError("pair_window mapping must return a pair_window dictionary.")
        params = copy.deepcopy(params)
        params.pop("mapping", None)
        params.setdefault("len_args", {})
        params.setdefault("other_args", {})
        return params

    def _reference_pair_field(self, *fields):
        for field in fields:
            if isinstance(field, ConvolsData):
                return field
        return None

    def _delta_field(self, data_field, random_field):
        if isinstance(random_field, (float, int, np.floating)):
            return data_field - self._field_density(random_field)
        return data_field - random_field

    def _compute_products_for_sample(
        self,
        s,
        mu,
        expanded_products,
        field1,
        field2,
        random1,
        random2,
        delta1,
        delta2,
    ):
        reference_field = self._reference_pair_field(field1, field2, random1, random2, delta1, delta2)
        pair_window_obj = None

        def product(a, b):
            nonlocal pair_window_obj
            if isinstance(a, (float, int, np.floating)) or isinstance(b, (float, int, np.floating)):
                return self._field_density(a) * self._field_density(b)
            if pair_window_obj is None:
                pair_window_obj = self._build_pair_window_for_sample(s, mu, reference_field)
            return pair_product_with_window(a, b, pair_window_obj, self.threads)

        values = {
            "dd": None,
            "dr": None,
            "rd": None,
            "delta_dd": None,
            "rr": None,
            "xi": None,
        }
        if "dd" in expanded_products:
            values["dd"] = product(field1, field2)
        if "dr" in expanded_products:
            values["dr"] = product(field1, random2)
        if "rd" in expanded_products:
            values["rd"] = product(random1, field2)
        if "delta_dd" in expanded_products:
            values["delta_dd"] = product(delta1, delta2)
        if "rr" in expanded_products:
            values["rr"] = product(random1, random2)
        if "xi" in expanded_products:
            values["xi"] = values["delta_dd"] / values["rr"]
        return values

    # Memory-strategy execution helpers.
    def _prepare_memory_leg_field(self, kind, leg_idx, base_convols_cache):
        if kind == "data":
            base_field, source_desc = self._resolve_base_convols(leg_idx, None, base_convols_cache)
        elif kind == "random":
            base_field, source_desc = self._resolve_random_base(
                leg_idx, getattr(self, f"random{leg_idx}"), base_convols_cache
            )
            if base_field is None:
                self.logger.error(
                    f"Missing input for random leg {leg_idx}. Products {self.products} require "
                    f"'random{leg_idx}' or shared 'random'."
                )
                func_util.safe_exit(1)
            if base_field == "uniform":
                signal_ref, _ = self._resolve_base_convols(leg_idx, None, base_convols_cache)
                rho = 1.0 / signal_ref.V
                self._record_memory_convols_info(leg_idx, signal_ref)
                return rho, f"{source_desc}, rho={rho:.5e}"
        else:
            raise ValueError(f"Unsupported memory leg kind: {kind}")

        window_obj, window_desc = self._resolve_window(leg_idx, base_field, getattr(self, f"window{leg_idx}"))
        if window_obj is not None:
            final_field = base_field @ window_obj
        else:
            final_field = base_field.copy()
            final_field.format_convols_params()
        self._record_memory_convols_info(leg_idx, final_field)
        return final_field, self._memory_leg_desc(source_desc, leg_idx)

    def _record_memory_convols_info(self, leg_idx, field):
        if self.rank == 0 and isinstance(field, ConvolsData):
            setattr(self.corr2pcf_data, f"convols_info{leg_idx}", copy.deepcopy(field.convols_info))

    def _prepare_memory_product_fields(self, product):
        base_convols_cache = {}
        if product == "dd":
            field1, desc1 = self._prepare_memory_leg_field("data", 1, base_convols_cache)
            field2, desc2 = self._prepare_memory_leg_field("data", 2, base_convols_cache)
        elif product == "dr":
            field1, desc1 = self._prepare_memory_leg_field("data", 1, base_convols_cache)
            field2, desc2 = self._prepare_memory_leg_field("random", 2, base_convols_cache)
        elif product == "rd":
            field1, desc1 = self._prepare_memory_leg_field("random", 1, base_convols_cache)
            field2, desc2 = self._prepare_memory_leg_field("data", 2, base_convols_cache)
        elif product == "rr":
            field1, desc1 = self._prepare_memory_leg_field("random", 1, base_convols_cache)
            field2, desc2 = self._prepare_memory_leg_field("random", 2, base_convols_cache)
        elif product == "delta_dd":
            data1, data_desc1 = self._prepare_memory_leg_field("data", 1, base_convols_cache)
            random1, random_desc1 = self._prepare_memory_leg_field("random", 1, base_convols_cache)
            data2, data_desc2 = self._prepare_memory_leg_field("data", 2, base_convols_cache)
            random2, random_desc2 = self._prepare_memory_leg_field("random", 2, base_convols_cache)
            field1 = self._delta_field(data1, random1)
            field2 = self._delta_field(data2, random2)
            desc1 = f"delta1=({data_desc1}) - ({random_desc1})"
            desc2 = f"delta2=({data_desc2}) - ({random_desc2})"
            del data1, random1, data2, random2
        else:
            raise ValueError(f"Unsupported memory product: {product}")

        compat_fields = [field for field in (field1, field2) if isinstance(field, ConvolsData)]
        if compat_fields:
            func_util.validate_convols_compatibility(
                compat_fields,
                ConvolsData._REQUIRED_ARGV,
                logger=self.logger,
                label=f"Corr_2PCF memory product '{product}' input fields",
            )
        self.logger.info(
            f"Memory product {product} fields ready:\n"
            f"  leg1: {desc1}\n"
            f"  leg2: {desc2}"
        )
        return field1, field2

    def _compute_single_product_for_sample(self, s, mu, field1, field2):
        if isinstance(field1, (float, int, np.floating)) or isinstance(field2, (float, int, np.floating)):
            return self._field_density(field1) * self._field_density(field2)
        reference_field = self._reference_pair_field(field1, field2)
        pair_window_obj = self._build_pair_window_for_sample(s, mu, reference_field)
        return pair_product_with_window(field1, field2, pair_window_obj, self.threads)

    def _run_memory_product(self, product, tasks, result_shape):
        comm = self.comm
        rank = self.rank
        size = comm.Get_size()
        if rank == 0:
            time_product_start = time.perf_counter()
            root_field1, root_field2 = self._prepare_memory_product_fields(product)
            task_sub_arrs = np.array_split(tasks, size)
        else:
            root_field1 = None
            root_field2 = None
            task_sub_arrs = None

        field1 = self._broadcast_field(root_field1)
        field2 = self._broadcast_field(root_field2)
        task_sub_arr = comm.scatter(task_sub_arrs, root=0)
        local_values = []
        local_tasks = []
        total_tasks = len(tasks)
        max_local_tasks = max(comm.allgather(len(task_sub_arr)))
        local_completed = 0
        step_report_interval = max(1, max_local_tasks // 10)
        if rank == 0:
            report_interval = max(1, total_tasks // 10)
            next_report_threshold = report_interval
            self.logger.info(f"Memory product {product} progress:   0.00%")
        for local_idx in range(max_local_tasks):
            if local_idx < len(task_sub_arr):
                task = task_sub_arr[local_idx]
                s_idx = int(task[0])
                mu_idx = int(task[1])
                s_value = float(task[2])
                mu_value = None if mu_idx < 0 else float(task[3])
                local_values.append(self._compute_single_product_for_sample(s_value, mu_value, field1, field2))
                local_tasks.append((s_idx, mu_idx))
                local_completed += 1
            if (
                (local_idx + 1) % step_report_interval == 0
                or local_idx + 1 == max_local_tasks
            ):
                gathered_completed = comm.gather(local_completed, root=0)
                if rank == 0:
                    global_completed = int(np.sum(gathered_completed))
                    if (
                        global_completed >= next_report_threshold
                        or global_completed == total_tasks
                    ):
                        progress = (global_completed / total_tasks) * 100.0
                        self.logger.info(f"Memory product {product} progress: {progress:6.2f}%")
                        next_report_threshold += report_interval

        gathered_values = comm.gather(local_values, root=0)
        gathered_tasks = comm.gather(local_tasks, root=0)
        if rank == 0:
            result = build_result_from_gathered(gathered_tasks, gathered_values, result_shape, self.mode)
            if total_tasks == 0:
                self.logger.info(f"Memory product {product} progress: 100.00%")
            self.logger.info(f"Memory product {product} time: {time.perf_counter() - time_product_start:.4f} sec")
            return result
        return None

    def _run_memory(self, save_result=True, overwrite=False):
        try:
            comm = self.comm
            rank = self.rank
            if rank == 0:
                time_run_1 = time.perf_counter()
            self.corr2pcf_data = Corr2PCFData(threads=self.threads)
            self._sync_runtime_options()
            self._resolve_sampling()
            self.products = self._normalize_products(self.products)
            self.pair_window = self._normalize_pair_window(self.pair_window)
            expanded_products = self._expanded_products()
            products_to_compute = [product for product in expanded_products if product != "xi"]

            if rank == 0:
                tasks, result_shape = make_sampling_tasks(self.mode, self.s_arr, self.mu_arr)
                self.logger.info("Start to calculate 2PCF in memory strategy ...")
                self.logger.info(self._describe_sampling())
                self.logger.info(self._describe_products(expanded_products))
                self.logger.info(self._describe_pair_window(self.pair_window))
                self.logger.info(f"Pair-window cache: enabled={self.pair_window_cache}")
                self.logger.info(self._describe_task_distribution(len(tasks), comm.Get_size()))
            else:
                tasks = None
                result_shape = None

            tasks = comm.bcast(tasks, root=0)
            result_shape = comm.bcast(result_shape, root=0)
            results = {}
            for product in products_to_compute:
                results[product] = self._run_memory_product(product, tasks, result_shape)
                comm.Barrier()

            if rank == 0:
                self.corr2pcf_data.mode = self.mode
                self.corr2pcf_data.s = self.s_arr.copy()
                self.corr2pcf_data.mu = None if self.mode == "s" else self.mu_arr.copy()
                self.corr2pcf_data.dd = results.get("dd") if "dd" in expanded_products else None
                self.corr2pcf_data.dr = results.get("dr") if "dr" in expanded_products else None
                self.corr2pcf_data.rd = results.get("rd") if "rd" in expanded_products else None
                self.corr2pcf_data.delta_dd = results.get("delta_dd") if "delta_dd" in expanded_products else None
                self.corr2pcf_data.rr = results.get("rr") if "rr" in expanded_products else None
                if "xi" in expanded_products:
                    self.corr2pcf_data.xi = self.corr2pcf_data.delta_dd / self.corr2pcf_data.rr
                else:
                    self.corr2pcf_data.xi = None

                self.corr2pcf_data.corr2pcf_info = self._current_task_params_snapshot()
                self.corr2pcf_data.task_params = self._current_task_params_snapshot()
                if save_result:
                    if self.fout_path:
                        self.corr2pcf_data.saveflag = True
                        self.corr2pcf_data.save_corr2pcf(self.fout_path, overwrite=overwrite)
                    else:
                        self.logger.warning("No output path provided. The 2PCF result will not be saved.")
                time_run_2 = time.perf_counter()
                print("")
                self.logger.info(f"The time for task: {time_run_2 - time_run_1:.4f} sec")
        except Exception as e:
            self.logger.error(f"Error in process {self.rank}: {str(e)}")
            func_util.safe_exit(1)
        return self.corr2pcf_data

    # Speed-strategy input preparation and execution.
    def prepare_input_fields(
        self,
        convols_data1=None,
        convols_data2=None,
        random1=None,
        random2=None,
        window1=None,
        window2=None,
        pair_window=None,
    ):
        self.corr2pcf_data = Corr2PCFData(threads=self.threads)
        self._sync_runtime_options()
        self._resolve_sampling()
        self.products = self._normalize_products(self.products)
        expanded_products = self._expanded_products()
        if convols_data1 is None:
            convols_data1 = self.convols_data1
        if convols_data2 is None:
            convols_data2 = self.convols_data2
        if random1 is None:
            random1 = self.random1
        if random2 is None:
            random2 = self.random2
        if window1 is None:
            window1 = self.window1
        if window2 is None:
            window2 = self.window2
        if pair_window is None:
            pair_window = self.pair_window
        self.pair_window = self._normalize_pair_window(pair_window)
        needs_data, needs_random = self._required_input_flags()
        if self.rank == 0:
            self.logger.info("Preparing Corr_2PCF input fields ...")
            self.logger.info(f"{self._describe_sampling()}, threads={self.threads}")
            self.logger.info(self._describe_products(expanded_products))
            self.logger.info(self._describe_pair_window(self.pair_window))
            base_convols_cache = {}
            resolved_data_legs = []
            if needs_data:
                for i, cdata, win in zip([1, 2], [convols_data1, convols_data2], [window1, window2]):
                    base_convols, source_desc = self._resolve_base_convols(i, cdata, base_convols_cache)
                    resolved_data_legs.append((i, base_convols, source_desc, win))

            resolved_random_legs = []
            if needs_random:
                for i, rdata, win in zip([1, 2], [random1, random2], [window1, window2]):
                    base_random, source_desc = self._resolve_random_base(i, rdata, base_convols_cache)
                    if base_random is None:
                        self.logger.error(
                            f"Missing input for random leg {i}. Products {self.products} require "
                            f"'random{i}' or shared 'random'. Corr_2PCF now defaults random to null, "
                            f"so please set it explicitly when RR/DR/RD/delta_DD/xi-related products are requested."
                        )
                        func_util.safe_exit(1)
                    resolved_random_legs.append((i, base_random, source_desc, win))

            compat_fields = [item[1] for item in resolved_data_legs if isinstance(item[1], ConvolsData)]
            compat_fields.extend(item[1] for item in resolved_random_legs if isinstance(item[1], ConvolsData))
            if compat_fields:
                shared_required = func_util.validate_convols_compatibility(
                    compat_fields,
                    ConvolsData._REQUIRED_ARGV,
                    logger=self.logger,
                    label="Corr_2PCF input fields",
                )
                shared_required_text = ", ".join([f"{k}={v}" for k, v in shared_required.items()])
                self.logger.info("Corr_2PCF input compatibility check passed.")
                self.logger.info(f"Shared required parameters | {shared_required_text}")

            for i, base_convols, source_desc, win in resolved_data_legs:
                window_obj, window_desc = self._resolve_window(i, base_convols, win)
                if window_obj is not None:
                    final_convols = base_convols @ window_obj
                else:
                    final_convols = base_convols.copy()
                    final_convols.format_convols_params()
                setattr(self, f"convols_data{i}", final_convols)
                setattr(self.corr2pcf_data, f"convols_info{i}", final_convols.convols_info)
                self.logger.info(
                    f"Field leg {i} ready | source={source_desc} | window={window_desc}"
                )

            for i, base_random, source_desc, win in resolved_random_legs:
                if base_random == "uniform":
                    signal_ref = getattr(self, f"convols_data{i}", None)
                    if not isinstance(signal_ref, ConvolsData):
                        signal_ref, _ = self._resolve_base_convols(i, None, base_convols_cache)
                    rho = 1.0 / signal_ref.V
                    setattr(self, f"random{i}", rho)
                    self.logger.info(
                        f"Random leg {i} ready | source={source_desc} | window=uniform shortcut | rho={rho:.5e}"
                    )
                else:
                    window_obj, window_desc = self._resolve_window(i, base_random, win)
                    if window_obj is not None:
                        final_random = base_random @ window_obj
                    else:
                        final_random = base_random.copy()
                        final_random.format_convols_params()
                    setattr(self, f"random{i}", final_random)
                    self.logger.info(
                        f"Random leg {i} ready | source={source_desc} | window={window_desc}"
                    )

            self.window1 = window1
            self.window2 = window2
            self.corr2pcf_data.corr2pcf_info = self._current_task_params_snapshot()
        self._fields_prepared = True

    def _broadcast_field(self, value):
        comm = self.comm
        rank = self.rank
        serialized = None
        is_density = None
        density_value = None

        if rank == 0:
            is_density = isinstance(value, (float, int, np.floating))
            if is_density:
                density_value = float(value)
            else:
                serialized = pickle.dumps(value.convols_info)
                value.epsilon = np.ascontiguousarray(value.epsilon, dtype=np.float64)
                local_value = value
        else:
            local_value = None

        is_density = comm.bcast(is_density, root=0)
        if is_density:
            return float(comm.bcast(density_value, root=0))

        serialized = comm.bcast(serialized, root=0)
        if rank != 0:
            local_value = ConvolsData(threads=self.threads)
            local_value.convols_info = pickle.loads(serialized)
            local_value.format_convols_params()
            local_value.epsilon = np.empty((local_value.L, local_value.L, local_value.L), dtype=np.float64)

        comm.Bcast(local_value.epsilon, root=0)
        comm.Barrier()
        return local_value

    def run(self, save_result=True, overwrite=False):
        self._sync_runtime_options(log_runtime=False)
        if self.memory_strategy == "memory":
            if self._fields_prepared and self.rank == 0:
                self.logger.warning(
                    "memory_strategy='memory' is most effective when run() prepares fields itself. "
                    "Calling prepare_input_fields() beforehand may keep extra fields resident."
                )
            return self._run_memory(save_result=save_result, overwrite=overwrite)
        try:
            comm = self.comm
            rank = self.rank
            size = comm.Get_size()
            if rank == 0:
                time_run_1 = time.perf_counter()
            if not self._fields_prepared:
                self.prepare_input_fields()
            expanded_products = self._expanded_products()
            needs_data, needs_random = self._required_input_flags()
            _local_convols1 = self._broadcast_field(self.convols_data1) if needs_data else None
            _local_convols2 = self._broadcast_field(self.convols_data2) if needs_data else None
            _local_random1 = self._broadcast_field(self.random1) if needs_random else None
            _local_random2 = self._broadcast_field(self.random2) if needs_random else None
            _local_delta1 = None
            _local_delta2 = None
            if "delta_dd" in expanded_products:
                _local_delta1 = self._delta_field(_local_convols1, _local_random1)
                _local_delta2 = self._delta_field(_local_convols2, _local_random2)
            self.corr2pcf_data.corr2pcf_info = self._current_task_params_snapshot()
            self.corr2pcf_data.task_params = self._current_task_params_snapshot()
            if rank == 0:
                self.logger.info("Start to calculate 2PCF ...")
                time_start = time.perf_counter()
                self.logger.info(f"Pre-2PCF setup time: {time_start - time_run_1:.4f} sec")
            # Generate sampling tasks at rank0.
            if rank == 0:
                if self.mode == "smu":
                    tasks = np.array(
                        [(i, j, self.s_arr[i], self.mu_arr[j]) for i in range(self.n_s) for j in range(self.n_mu)],
                        dtype=np.float64,
                    )
                    result_shape = (self.n_s, self.n_mu)
                else:
                    tasks = np.array([(i, -1, self.s_arr[i], np.nan) for i in range(self.n_s)], dtype=np.float64)
                    result_shape = (self.n_s,)
                task_sub_arrs = np.array_split(tasks, size)
                # Global process status
                arr_complete = np.zeros(size, dtype=int)
                total_tasks = len(tasks)
                report_interval = max(1, total_tasks // 10)
                next_report_threshold = report_interval
                requests = [None] + [comm.irecv(source=r, tag=r) for r in range(1, size)]
                count_all = False
                self.logger.info(self._describe_task_distribution(total_tasks, size))
                self.logger.info("Progress:   0.00%")
            else:
                task_sub_arrs = None
                result_shape = None
            # Scatter to all ranks
            task_sub_arr = comm.scatter(task_sub_arrs, root=0)
            # Local process status
            local_completed = 0
            local_report_interval = max(1, len(task_sub_arr) // 10)
            # Init local 2pcf results
            local_xi = []
            local_dd = []
            local_dr = []
            local_rd = []
            local_delta_dd = []
            local_rr = []
            local_tasks = []
            for task in task_sub_arr:
                s_idx = int(task[0])
                mu_idx = int(task[1])
                s_value = float(task[2])
                mu_value = None if mu_idx < 0 else float(task[3])
                values = self._compute_products_for_sample(
                    s_value,
                    mu_value,
                    expanded_products,
                    _local_convols1,
                    _local_convols2,
                    _local_random1,
                    _local_random2,
                    _local_delta1,
                    _local_delta2,
                )
                local_dd.append(values["dd"])
                local_dr.append(values["dr"])
                local_rd.append(values["rd"])
                local_delta_dd.append(values["delta_dd"])
                local_rr.append(values["rr"])
                local_xi.append(values["xi"])
                local_tasks.append((s_idx, mu_idx))
                local_completed += 1
                if local_completed % local_report_interval == 0:
                    if rank == 0:
                        arr_complete[0] = local_completed
                    else:
                        comm.isend(local_completed, dest=0, tag=rank)
                if rank == 0:
                    for r, req in enumerate(requests[1:], start=1):
                        status = req.test()
                        # Check whether the data is received
                        if status[0]: 
                            # Renew the completed task num
                            arr_complete[r] = status[1] 
                            # Reset the request flag
                            requests[r] = comm.irecv(source=r, tag=r) 
                    global_completed = np.sum(arr_complete)
                    # Show status
                    if global_completed >= next_report_threshold:
                        progress = (global_completed / total_tasks) * 100
                        self.logger.info(f"Progress: {progress:6.2f}%")
                        # Renew next report checkpoint
                        next_report_threshold += report_interval
                        if global_completed == total_tasks:
                            count_all = True
            comm.Barrier()
            if rank == 0:
                arr_complete[0] = local_completed
                for r, req in enumerate(requests[1:], start=1):
                    status = req.test()
                    if status[0]:
                        arr_complete[r] = status[1]
                    else:
                        if hasattr(req, "cancel"):
                            req.cancel()
                        elif hasattr(req, "Cancel"):
                            req.Cancel()
                        if hasattr(req, "wait"):
                            req.wait()
                        elif hasattr(req, "Wait"):
                            req.Wait()
                if hasattr(comm, "Iprobe"):
                    for source in range(1, size):
                        while comm.Iprobe(source=source, tag=source):
                            arr_complete[source] = comm.recv(source=source, tag=source)
            # Gathering to rank0
            gathered_xi = comm.gather(local_xi, root=0)
            gathered_dd = comm.gather(local_dd, root=0)
            gathered_dr = comm.gather(local_dr, root=0)
            gathered_rd = comm.gather(local_rd, root=0)
            gathered_delta_dd = comm.gather(local_delta_dd, root=0)
            gathered_rr = comm.gather(local_rr, root=0)
            gathered_tasks = comm.gather(local_tasks, root=0)
            if rank == 0:
                flat_tasks = [item for sublist in gathered_tasks for item in sublist]

                def build_result(gathered_values):
                    result = np.empty(result_shape, dtype=np.float64)
                    for (s_idx, mu_idx), value in zip(flat_tasks, [item for sublist in gathered_values for item in sublist]):
                        if self.mode == "smu":
                            result[s_idx, mu_idx] = value
                        else:
                            result[s_idx] = value
                    return result

                self.corr2pcf_data.mode = self.mode
                self.corr2pcf_data.s = self.s_arr.copy()
                self.corr2pcf_data.mu = None if self.mode == "s" else self.mu_arr.copy()
                self.corr2pcf_data.dd = None if 'dd' not in expanded_products else build_result(gathered_dd)
                self.corr2pcf_data.dr = None if 'dr' not in expanded_products else build_result(gathered_dr)
                self.corr2pcf_data.rd = None if 'rd' not in expanded_products else build_result(gathered_rd)
                self.corr2pcf_data.delta_dd = None if 'delta_dd' not in expanded_products else build_result(gathered_delta_dd)
                self.corr2pcf_data.rr = None if 'rr' not in expanded_products else build_result(gathered_rr)
                self.corr2pcf_data.xi = None if 'xi' not in expanded_products else build_result(gathered_xi)
                if not count_all:
                    progress = 100.
                    self.logger.info(f"Progress: {progress:6.2f}%")
                time_end = time.perf_counter()
                self.logger.info(f"2PCF main loop time: {time_end - time_start:.4f} sec")
                self.logger.info("Main 2PCF loop finished, gathering results on rank 0 ...")
                # Output the 2pcf
                if save_result:
                    if self.fout_path:
                        self.corr2pcf_data.saveflag = True
                        self.corr2pcf_data.save_corr2pcf(self.fout_path, overwrite=overwrite) 
                    else:
                        self.logger.warning("No output path provided. The 2PCF result will not be saved.")
        except Exception as e:
            self.logger.error(f"Error in process {self.rank}: {str(e)}")
            func_util.safe_exit(1)
        if self.rank == 0:
            time_run_2 = time.perf_counter()
            print("")
            self.logger.info(f"The time for task: {time_run_2 - time_run_1:.4f} sec")
        return self.corr2pcf_data
