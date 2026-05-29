import time
import pickle
import copy
import hashlib
import json
import os

import numpy as np

from pyhermes.io import WindowFunc
from pyhermes.io import ConvolsData, normalize_weight_normalization
from pyhermes.io import Corr2PCFData
from pyhermes.utils import func_util
from pyhermes.utils.convolution import specialized_convolution_3d
from pyhermes.utils.window_params import (
    LOS_ARG_KEYS,
    default_pair_window,
    normalize_pair_window_template,
    serialize_window_params,
)
from pyhermes.pipeline import TaskBase


# Product and runtime configuration.
PRODUCT_NAMES = ("dd", "dr", "rd", "delta_dd", "rr", "xi")
PRODUCT_RULES = {
    "allowed": set(PRODUCT_NAMES),
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

NONNEGATIVE_SAMPLING_ARGS = {"s", "rp", "pi"}


def parse_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


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


def normalize_sampling_array(values, name, nonnegative=True):
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise TypeError(f"'{name}' must be a 1D array-like input.")
    if arr.size == 0:
        raise ValueError(f"'{name}' must contain at least one sampling point.")
    if nonnegative and np.any(arr < 0.0):
        raise ValueError(f"'{name}' values must be non-negative.")
    diffs = np.diff(arr)
    if np.any(diffs < 0.0) and np.any(diffs > 0.0):
        raise ValueError(f"'{name}' values must be monotonic.")
    return np.ascontiguousarray(arr, dtype=np.float64)


# Logging and serialization helpers.
def describe_sampling(sampling_names, sampling_arrays, sampling_specs):
    parts = []
    for name in sampling_names:
        arr = sampling_arrays[name]
        text = f"{name}: n={arr.size}, min={float(np.min(arr))}, max={float(np.max(arr))}"
        if not isinstance(sampling_specs.get(name), dict):
            text += ", source=explicit array"
        parts.append(text)
    return "Sampling: " + " | ".join(parts)


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


def effective_pair_los(pair_window):
    if not isinstance(pair_window, dict):
        return None
    los_args = pair_window.get("los_args", {})
    if not all(axis in los_args for axis in LOS_ARG_KEYS):
        return None
    los_values = tuple(los_args.get(axis) for axis in LOS_ARG_KEYS)
    if all(value is not None for value in los_values):
        return los_values
    return None


def describe_pair_window(pair_window):
    if isinstance(pair_window, dict):
        mapping = pair_window.get("mapping", "custom")
        mapping_name = mapping if isinstance(mapping, str) else getattr(mapping, "__name__", "custom callable")
        parts = [f"type={pair_window.get('type', 'custom')}", f"mapping={mapping_name}"]
        if pair_window.get("kernel_mode") is not None:
            parts.append(f"kernel_mode={pair_window.get('kernel_mode')}")
        pair_los = effective_pair_los(pair_window)
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
        return serialize_window_params(value)
    if isinstance(value, WindowFunc):
        return {
            "kind": "WindowFunc",
            "type": getattr(value, "type", "custom"),
            "len_args": serialize_window_params(getattr(value, "len_args", {})),
            "los_args": serialize_window_params(getattr(value, "los_args", {})),
            "other_args": serialize_window_params(getattr(value, "other_args", {})),
        }
    return value


# Sampling and result helpers.
def normalize_sampling_spec(name, spec):
    nonnegative = name in NONNEGATIVE_SAMPLING_ARGS
    if isinstance(spec, dict):
        sampling_min = float(spec["min"])
        sampling_max = float(spec["max"])
        sampling_n = int(spec["n"])
        if sampling_n <= 0:
            raise ValueError(f"'{name}' sampling n must be positive.")
        arr = np.linspace(sampling_min, sampling_max, sampling_n, dtype=np.float64)
    else:
        arr = normalize_sampling_array(spec, name, nonnegative=nonnegative)
    if nonnegative and np.any(arr < 0.0):
        raise ValueError(f"'{name}' sampling values must be non-negative.")
    return np.ascontiguousarray(arr, dtype=np.float64)


def make_sampling_tasks(sampling_names, sampling_arrays):
    result_shape = tuple(int(sampling_arrays[name].size) for name in sampling_names)
    rows = []
    for index_tuple in np.ndindex(result_shape):
        value_tuple = tuple(float(sampling_arrays[name][idx]) for name, idx in zip(sampling_names, index_tuple))
        rows.append(tuple(index_tuple) + value_tuple)
    return np.asarray(rows, dtype=np.float64), result_shape


def task_to_sample(task, sampling_names):
    n_dim = len(sampling_names)
    indices = tuple(int(task[i]) for i in range(n_dim))
    sample = {name: float(task[n_dim + i]) for i, name in enumerate(sampling_names)}
    return indices, sample


def build_result_from_gathered(gathered_tasks, gathered_values, result_shape):
    result = np.empty(result_shape, dtype=np.float64)
    flat_values = [item for sublist in gathered_values for item in sublist]
    flat_tasks = [item for sublist in gathered_tasks for item in sublist]
    for index_tuple, value in zip(flat_tasks, flat_values):
        result[tuple(index_tuple)] = value
    return result


def populate_corr2pcf_data(corr2pcf_data, sampling_names, sampling_arrays, expanded_products, product_results):
    corr2pcf_data.set_sampling(tuple(sampling_names), {
        name: sampling_arrays[name].copy()
        for name in sampling_names
    })
    for product in PRODUCT_NAMES:
        setattr(
            corr2pcf_data,
            product,
            product_results.get(product) if product in expanded_products else None,
        )
    if "xi" in expanded_products and corr2pcf_data.xi is None:
        corr2pcf_data.xi = corr2pcf_data.delta_dd / corr2pcf_data.rr


# Pair-window mappings.
def mapping_s_to_R(sample, pair_window):
    params = copy.deepcopy(pair_window)
    params.setdefault("len_args", {})
    if params["len_args"].get("R") is None:
        params["len_args"]["R"] = sample["s"]
    return params


def mapping_smu_to_RH(sample, pair_window):
    params = copy.deepcopy(pair_window)
    params.setdefault("len_args", {})
    params.setdefault("los_args", {})
    params.setdefault("other_args", {})
    if params["len_args"].get("R") is None:
        params["len_args"]["R"] = sample["s"] * np.sqrt(max(0.0, 1.0 - sample["mu"] * sample["mu"]))
    if params["len_args"].get("H") is None:
        params["len_args"]["H"] = sample["s"] * sample["mu"]
    return params


def mapping_rppi_to_RH(sample, pair_window):
    params = copy.deepcopy(pair_window)
    params.setdefault("len_args", {})
    params.setdefault("los_args", {})
    params.setdefault("other_args", {})
    if params["len_args"].get("R") is None:
        params["len_args"]["R"] = sample["rp"]
    if params["len_args"].get("H") is None:
        params["len_args"]["H"] = sample["pi"]
    return params


PAIR_WINDOW_MAPPING_SPECS = {
    "s_to_R": {
        "sampling_args": ("s",),
        "len_args": ("R",),
        "func": mapping_s_to_R,
    },
    "smu_to_RH": {
        "sampling_args": ("s", "mu"),
        "len_args": ("R", "H"),
        "func": mapping_smu_to_RH,
    },
    "rppi_to_RH": {
        "sampling_args": ("rp", "pi"),
        "len_args": ("R", "H"),
        "func": mapping_rppi_to_RH,
    },
}


def normalize_pair_window_params(pair_window, require_mapping=True):
    normalized = normalize_pair_window_template(pair_window)
    mapping = normalized.get("mapping")
    if not mapping:
        if require_mapping:
            raise ValueError("pair_window requires a 'mapping' field.")
        return normalized
    if isinstance(mapping, str):
        if mapping not in PAIR_WINDOW_MAPPING_SPECS:
            raise ValueError(
                f"Unsupported pair_window mapping '{mapping}'. "
                f"Supported built-in mappings: {sorted(PAIR_WINDOW_MAPPING_SPECS)}."
            )
    elif not callable(mapping):
        raise TypeError("pair_window mapping must be a string or callable.")
    return normalized


def build_pair_window_params_for_sample(sample, pair_window):
    pair_window = normalize_pair_window_params(pair_window)
    mapping = pair_window.get("mapping")
    if isinstance(mapping, str):
        mapper = PAIR_WINDOW_MAPPING_SPECS[mapping]["func"]
    else:
        mapper = mapping
    params = mapper(sample, pair_window)
    if not isinstance(params, dict):
        raise TypeError("pair_window mapping must return a pair_window dictionary.")
    params = copy.deepcopy(params)
    params.pop("mapping", None)
    params.setdefault("len_args", {})
    params.setdefault("los_args", {})
    params.setdefault("other_args", {})
    return params


# Pair-product kernels.
def field_mean_density(field, value_unit="grid"):
    if isinstance(field, (float, int, np.floating)):
        return float(field)
    if isinstance(field, ConvolsData):
        value = field.field_mean_density(value_unit=value_unit)
        if value is None:
            raise ValueError("Cannot extract a uniform density from a field without field_integral metadata.")
        return value
    raise TypeError(f"Unsupported field type for density extraction: {type(field)}")


def pair_product_with_window(field1, field2, pair_window_obj, threads):
    if field2 is None:
        field2 = field1
    if isinstance(field1, (float, int, np.floating)) or isinstance(field2, (float, int, np.floating)):
        return field_mean_density(field1) * field_mean_density(field2)
    conv = specialized_convolution_3d(field1.epsilon, pair_window_obj.as_array(), threads=threads)
    return float(np.einsum("ijk,ijk->", conv, field2.epsilon, optimize=True) / conv.size)


def compute_pair_product_at_sample(sample, convols_data1, convols_data2=None, pair_window=None):
    if convols_data2 is None:
        convols_data2 = convols_data1
    if isinstance(convols_data1, (float, int, np.floating)) or isinstance(convols_data2, (float, int, np.floating)):
        return field_mean_density(convols_data1) * field_mean_density(convols_data2)
    pair_window_params = build_pair_window_params_for_sample(sample, pair_window)
    pair_window_obj = WindowFunc(pair_window_params, convols_data1.convols_info, threads=convols_data1.threads)
    return pair_product_with_window(convols_data1, convols_data2, pair_window_obj, convols_data1.threads)


class Corr_2PCF(TaskBase):

    def __init__(self, param_task=None):
        if param_task is None:
            param_task = {"Corr_2PCF": {}}
        self.task_name = str(self.__class__.__name__)
        super().__init__(param_task=param_task)
        self.format_params()
        self._fields_prepared = False

    # Parameter formatting and validation.
    def _sync_runtime_options(self, log_runtime=True):
        self.threads = max(1, int(self.threads))
        self.memory_strategy = str(self.memory_strategy).strip().lower()
        if self.memory_strategy not in ("speed", "memory"):
            raise ValueError("Corr_2PCF memory_strategy must be either 'speed' or 'memory'.")
        self.pair_window_cache = parse_bool(self.pair_window_cache)
        self.task_params['threads'] = self.threads
        self.task_params['products'] = copy.deepcopy(self.products)
        self.task_params['sampling'] = copy.deepcopy(self.sampling)
        self.task_params['pair_window'] = copy.deepcopy(self.pair_window)
        self.task_params['memory_strategy'] = self.memory_strategy
        self.task_params['pair_window_cache'] = self.pair_window_cache
        self.task_params['pair_window_cache_dir'] = self.pair_window_cache_dir
        self.task_params['weight_normalization'] = self.weight_normalization
        if log_runtime:
            self.sync_runtime_options(context="Corr_2PCF runtime configuration")
            if self.rank == 0:
                self.logger.info(
                    "Field weight normalization rule | "
                    f"task weight_normalization={self.weight_normalization} | "
                    "catalog fields are converted to the task rule; derived fields are used as-is."
                )

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
        self.weight_normalization = normalize_weight_normalization(self.task_params.get("weight_normalization", "catalog"))

        window = self.task_params.get('window', None)
        self.window = window if (window and window.get('type')) else None
        for i in range(1, 3):
            window_i = self.task_params.get(f'window{i}', None)
            window_i = window_i if (window_i and window_i.get('type')) else None
            if (not window_i) and self.window:
                window_i = dict(self.window)
            setattr(self, f'window{i}', window_i)
        self.pair_window = normalize_pair_window_params(self.task_params.get('pair_window', default_pair_window()))

        self.sampling = copy.deepcopy(self.task_params['sampling'])
        self.sampling_names = ()
        self.sampling_arrays = {}
        self.sampling_specs = {}
        self.threads = int(self.task_params['threads'])
        self.products = normalize_products(self.task_params.get('products', 'xi'))
        self.memory_strategy = str(self.task_params.get("memory_strategy", "speed")).strip().lower()
        self.pair_window_cache = parse_bool(self.task_params.get("pair_window_cache", False))
        self.pair_window_cache_dir = self.task_params.get("pair_window_cache_dir", "")
        self.fout_path = self.task_params['fout_path']

    def _resolve_sampling(self):
        if not isinstance(self.sampling, dict) or not self.sampling:
            raise ValueError("Corr_2PCF requires a non-empty 'sampling' dictionary.")
        mapping = self.pair_window.get("mapping")
        if isinstance(mapping, str):
            required_names = PAIR_WINDOW_MAPPING_SPECS[mapping]["sampling_args"]
        else:
            required_names = tuple(self.sampling.keys())
        missing = [name for name in required_names if name not in self.sampling]
        extra = [name for name in self.sampling if name not in required_names]
        if missing or extra:
            raise ValueError(
                f"sampling keys must match mapping '{mapping}': "
                f"expected {list(required_names)}, missing={missing}, extra={extra}."
            )
        self.sampling_names = tuple(required_names)
        self.sampling_specs = copy.deepcopy(self.sampling)
        self.sampling_arrays = {
            name: normalize_sampling_spec(name, self.sampling[name])
            for name in self.sampling_names
        }

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
        params['pair_window'] = serialize_window_input(self.pair_window)
        params['sampling_spec'] = copy.deepcopy(self.sampling_specs)
        params['sampling_names'] = list(self.sampling_names)
        params['sampling'] = {
            name: self.sampling_arrays[name].tolist()
            for name in self.sampling_names
        }
        params['threads'] = self.threads
        params['products'] = copy.deepcopy(self.products)
        params['memory_strategy'] = self.memory_strategy
        params['pair_window_cache'] = self.pair_window_cache
        params['pair_window_cache_dir'] = self.pair_window_cache_dir
        params['weight_normalization'] = self.weight_normalization
        params['fout_path'] = self.fout_path
        return params

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
        for product in expand_products(self.products):
            product_needs_data, product_needs_random = PRODUCT_INPUT_FLAGS[product]
            needs_data = needs_data or product_needs_data
            needs_random = needs_random or product_needs_random
        return needs_data, needs_random

    def _validate_uniform_random_signal(self, field):
        if not isinstance(field, ConvolsData) or field.field_mean_density(value_unit="grid") is None:
            raise ValueError(
                "random='uniform' requires a ConvolsData field with a defined field_integral."
            )

    def _remember_field_scale(self, field):
        if isinstance(field, ConvolsData):
            self._density_physical_scale = float(field.scale_factor) ** 3

    def _pair_product_physical_scale(self):
        return float(getattr(self, "_density_physical_scale", 1.0)) ** 2

    def _scale_pair_product_results(self, product_results):
        factor = self._pair_product_physical_scale()
        if np.isclose(factor, 1.0):
            return product_results
        for product in ("dd", "dr", "rd", "delta_dd", "rr"):
            if product in product_results and product_results[product] is not None:
                product_results[product] = product_results[product] * factor
        return product_results

    def _field_in_task_normalization(self, field):
        self._remember_field_scale(field)
        input_kind = getattr(field, "field_kind", None)
        input_norm = getattr(field, "weight_normalization", None)
        if getattr(field, "field_kind", None) != "catalog_field":
            self.logger.info(
                "Field weight normalization | "
                f"field_kind={input_kind} | input={input_norm} | "
                "effective=as-is; task weight_normalization does not apply to derived fields."
            )
            return field.copy()
        self.logger.info(
            "Field weight normalization | "
            f"field_kind={input_kind} | input={input_norm} | "
            f"task={self.weight_normalization} | effective={self.weight_normalization}"
        )
        return field.switch_weight_normalization(self.weight_normalization)

    # Pair-product computation.
    def calc_pair_product(self, sample, field1, field2=None, pair_window=None):
        if field2 is None:
            field2 = field1
        if pair_window is None:
            pair_window = self.pair_window
        pair_window = normalize_pair_window_params(pair_window)
        if isinstance(field1, (float, int, np.floating)) or isinstance(field2, (float, int, np.floating)):
            return field_mean_density(field1) * field_mean_density(field2)
        return compute_pair_product_at_sample(
            sample,
            field1,
            field2,
            pair_window=pair_window,
        )

    def _build_pair_window_for_sample(self, sample, reference_field):
        pair_window_params = build_pair_window_params_for_sample(sample, self.pair_window)
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

    def _reference_pair_field(self, *fields):
        for field in fields:
            if isinstance(field, ConvolsData):
                return field
        return None

    def _delta_field(self, data_field, random_field):
        if isinstance(random_field, (float, int, np.floating)):
            return data_field - field_mean_density(random_field)
        return data_field - random_field

    def _compute_products_for_sample(
        self,
        sample,
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
                return field_mean_density(a) * field_mean_density(b)
            if pair_window_obj is None:
                pair_window_obj = self._build_pair_window_for_sample(sample, reference_field)
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
                signal_ref = self._field_in_task_normalization(signal_ref)
                self._validate_uniform_random_signal(signal_ref)
                rho = field_mean_density(signal_ref)
                if self.rank == 0:
                    setattr(self.corr2pcf_data, f"convols_info{leg_idx}", copy.deepcopy(signal_ref.convols_info))
                return rho, f"{source_desc}, rho={rho:.5e}"
        else:
            raise ValueError(f"Unsupported memory leg kind: {kind}")

        base_field = self._field_in_task_normalization(base_field)

        window_obj, window_desc = self._resolve_window(leg_idx, base_field, getattr(self, f"window{leg_idx}"))
        if window_obj is not None:
            final_field = base_field @ window_obj
        else:
            final_field = base_field.copy()
            final_field.format_convols_params()
        if self.rank == 0:
            setattr(self.corr2pcf_data, f"convols_info{leg_idx}", copy.deepcopy(final_field.convols_info))
        window_desc = compact_window_desc(getattr(self, f"window{leg_idx}"))
        return final_field, f"{source_desc}, {window_desc}"

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

    def _compute_single_product_for_sample(self, sample, field1, field2):
        if isinstance(field1, (float, int, np.floating)) or isinstance(field2, (float, int, np.floating)):
            return field_mean_density(field1) * field_mean_density(field2)
        reference_field = self._reference_pair_field(field1, field2)
        pair_window_obj = self._build_pair_window_for_sample(sample, reference_field)
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
                index_tuple, sample = task_to_sample(task, self.sampling_names)
                local_values.append(self._compute_single_product_for_sample(sample, field1, field2))
                local_tasks.append(index_tuple)
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
            result = build_result_from_gathered(gathered_tasks, gathered_values, result_shape)
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
            self.products = normalize_products(self.products)
            self.pair_window = normalize_pair_window_params(self.pair_window)
            self._resolve_sampling()
            expanded_products = expand_products(self.products)
            products_to_compute = [product for product in expanded_products if product != "xi"]

            if rank == 0:
                tasks, result_shape = make_sampling_tasks(self.sampling_names, self.sampling_arrays)
                self.logger.info("Start to calculate 2PCF in memory strategy ...")
                self.logger.info(describe_sampling(self.sampling_names, self.sampling_arrays, self.sampling_specs))
                self.logger.info(describe_products(self.products, expanded_products))
                self.logger.info(describe_pair_window(self.pair_window))
                self.logger.info(f"Pair-window cache: enabled={self.pair_window_cache}")
                self.logger.info(describe_task_distribution(len(tasks), comm.Get_size()))
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
                self._scale_pair_product_results(results)
                populate_corr2pcf_data(
                    self.corr2pcf_data,
                    self.sampling_names,
                    self.sampling_arrays,
                    expanded_products,
                    results,
                )
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
        sync_runtime=True,
    ):
        self.corr2pcf_data = Corr2PCFData(threads=self.threads)
        if sync_runtime:
            self._sync_runtime_options()
        self.products = normalize_products(self.products)
        expanded_products = expand_products(self.products)
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
        self.pair_window = normalize_pair_window_params(pair_window)
        self._resolve_sampling()
        needs_data, needs_random = self._required_input_flags()
        if self.rank == 0:
            self.logger.info("Preparing Corr_2PCF input fields ...")
            self.logger.info(f"{describe_sampling(self.sampling_names, self.sampling_arrays, self.sampling_specs)}, threads={self.threads}")
            self.logger.info(describe_products(self.products, expanded_products))
            self.logger.info(describe_pair_window(self.pair_window))
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
                base_convols = self._field_in_task_normalization(base_convols)
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
                        signal_ref = self._field_in_task_normalization(signal_ref)
                    self._validate_uniform_random_signal(signal_ref)
                    rho = field_mean_density(signal_ref)
                    setattr(self, f"random{i}", rho)
                    self.logger.info(
                        f"Random leg {i} ready | source={source_desc} | window=uniform shortcut | rho={rho:.5e}"
                    )
                else:
                    base_random = self._field_in_task_normalization(base_random)
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
        self._sync_runtime_options()
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
                self.prepare_input_fields(sync_runtime=False)
            expanded_products = expand_products(self.products)
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
                tasks, result_shape = make_sampling_tasks(self.sampling_names, self.sampling_arrays)
                task_sub_arrs = np.array_split(tasks, size)
                # Global process status
                arr_complete = np.zeros(size, dtype=int)
                total_tasks = len(tasks)
                report_interval = max(1, total_tasks // 10)
                next_report_threshold = report_interval
                requests = [None] + [comm.irecv(source=r, tag=r) for r in range(1, size)]
                count_all = False
                self.logger.info(describe_task_distribution(total_tasks, size))
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
                index_tuple, sample = task_to_sample(task, self.sampling_names)
                values = self._compute_products_for_sample(
                    sample,
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
                local_tasks.append(index_tuple)
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
                gathered_by_product = {
                    "dd": gathered_dd,
                    "dr": gathered_dr,
                    "rd": gathered_rd,
                    "delta_dd": gathered_delta_dd,
                    "rr": gathered_rr,
                    "xi": gathered_xi,
                }
                product_results = {
                    product: build_result_from_gathered(
                        gathered_tasks,
                        gathered_values,
                        result_shape,
                    )
                    for product, gathered_values in gathered_by_product.items()
                    if product in expanded_products
                }
                self._scale_pair_product_results(product_results)
                populate_corr2pcf_data(
                    self.corr2pcf_data,
                    self.sampling_names,
                    self.sampling_arrays,
                    expanded_products,
                    product_results,
                )
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
