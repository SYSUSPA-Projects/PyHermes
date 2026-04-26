import time
import pickle
import copy

import numpy as np

from pyhermes.io import WindowFunc
from pyhermes.io import ConvolsData
from pyhermes.io import Corr2PCFData
from pyhermes.utils import func_util
from pyhermes.utils.convolution import specialized_convolution_3d
from pyhermes.pipeline import TaskBase


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


def _los_to_vector(los):
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


def _pair_window_params_for_sample(s, mu, pair_window):
    if pair_window is None:
        params = {"type": "shell", "len_args": {"R": s}, "other_args": {}}
    else:
        if not isinstance(pair_window, dict):
            raise TypeError(
                f"Unsupported pair_window input: expected dict, got {type(pair_window)}."
            )
        params = copy.deepcopy(pair_window)
        params.setdefault("len_args", {})
        if params.get("type") == "ring":
            if mu is None:
                raise ValueError("Corr_2PCF mode='smu' ring pair_window requires a mu value.")
            params["len_args"]["R"] = s * np.sqrt(max(0.0, 1.0 - mu * mu))
            params["len_args"]["H"] = s * mu
        else:
            params["len_args"]["R"] = s
    return params


def _pair_product_with_window(field1, field2, pair_window_obj, threads):
    if field2 is None:
        field2 = field1
    if isinstance(field1, (float, int, np.floating)) or isinstance(field2, (float, int, np.floating)):
        return _field_density(field1) * _field_density(field2)
    conv = specialized_convolution_3d(field1.epsilon, pair_window_obj.as_array(), threads=threads)
    return float(np.einsum("ijk,ijk->", conv, field2.epsilon, optimize=True) / conv.size)


def _field_density(field):
    if isinstance(field, (float, int, np.floating)):
        return float(field)
    if isinstance(field, ConvolsData):
        return 1.0 / field.V
    raise TypeError(f"Unsupported field type for density extraction: {type(field)}")


def compute_pair_product_at_smu(s, mu, convols_data1, convols_data2=None, pair_window=None):
    if convols_data2 is None:
        convols_data2 = convols_data1
    if isinstance(convols_data1, (float, int, np.floating)) or isinstance(convols_data2, (float, int, np.floating)):
        return _field_density(convols_data1) * _field_density(convols_data2)
    pair_window_params = _pair_window_params_for_sample(s, mu, pair_window)
    pair_window_obj = WindowFunc(pair_window_params, convols_data1.convols_info, threads=convols_data1.threads)
    return _pair_product_with_window(convols_data1, convols_data2, pair_window_obj, convols_data1.threads)


class Corr_2PCF(TaskBase):

    def __init__(self, param_task=None):
        if param_task is None:
            param_task = {"Corr_2PCF": {}}
        self.task_name = str(self.__class__.__name__)
        super().__init__(param_task=param_task)
        self.format_params()
        self.pair_window = None
        self._fields_prepared = False

    def _sync_runtime_options(self):
        self.mode = str(self.mode).strip().lower()
        if self.mode not in ("s", "smu"):
            raise ValueError("Corr_2PCF mode must be either 's' or 'smu'.")
        self.los_vector = _los_to_vector(self.los)
        self._sync_sampling_attribute_overrides()
        if self._pair_window_from_default and self.pair_window is None:
            self.pair_window_params = self._default_pair_window()
        self.threads = max(1, int(self.threads))
        self.task_params['threads'] = self.threads
        self.task_params['products'] = copy.deepcopy(self.products)
        self.task_params['mode'] = self.mode
        self.task_params['los'] = copy.deepcopy(self.los)
        self.task_params['s'] = copy.deepcopy(self.s)
        self.task_params['mu'] = copy.deepcopy(self.mu)
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
        self.los_vector = _los_to_vector(self.los)
        pair_window_params = self.task_params.get('pair_window', None)
        if pair_window_params and pair_window_params.get('type'):
            self.pair_window_params = copy.deepcopy(pair_window_params)
            self._pair_window_from_default = False
        else:
            self.pair_window_params = self._default_pair_window()
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
        self.fout_path = self.task_params['fout_path']

    def _default_pair_window(self):
        if self.mode == "smu":
            nx, ny, nz = self.los_vector
            return {
                "type": "ring",
                "len_args": {"R": None, "H": None},
                "other_args": {"nx": nx, "ny": ny, "nz": nz},
            }
        return {"type": "shell", "len_args": {"R": None}, "other_args": {}}

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

    def _normalize_products(self, products):
        if isinstance(products, str):
            products = [products]
        elif products is None:
            products = ['xi']
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

    def _expanded_products(self):
        expanded = list(self.products)
        idx = 0
        while idx < len(expanded):
            for dep in PRODUCT_RULES["deps"].get(expanded[idx], []):
                if dep not in expanded:
                    expanded.append(dep)
            idx += 1
        return expanded

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
        if self.mode == "smu" and normalized.get("type") == "ring":
            nx, ny, nz = self.los_vector
            normalized["other_args"].setdefault("nx", nx)
            normalized["other_args"].setdefault("ny", ny)
            normalized["other_args"].setdefault("nz", nz)
        return normalized

    def _normalize_sampling_array(self, values, name, positive=True):
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

    def _describe_sampling(self):
        base = f"mode={self.mode}, n_s={self.n_s}, s_min={self.s_min}, s_max={self.s_max}"
        if self.mode == "smu":
            base += f", n_mu={self.n_mu}, mu_min={self.mu_min}, mu_max={self.mu_max}, los={self.los_vector}"
        if isinstance(self.s, dict) and (self.mode == "s" or isinstance(self.mu, dict)):
            return base
        return f"{base}, source=explicit sampling array"

    def _resolve_sampling(self):
        if self.s is None:
            raise ValueError("Corr_2PCF requires 's' to be provided as a dict or array-like input.")
        if isinstance(self.s, dict):
            s_min = float(self.s["s_min"])
            s_max = float(self.s["s_max"])
            n_s = int(self.s["n_s"])
            s_arr = np.linspace(s_min, s_max, n_s, dtype=np.float64)
        else:
            s_arr = self._normalize_sampling_array(self.s, "s")
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
                mu_arr = self._normalize_sampling_array(self.mu, "mu", positive=False)
            self.mu_arr = np.ascontiguousarray(mu_arr, dtype=np.float64)
            self.n_mu = int(self.mu_arr.size)
            self.mu_min = float(np.min(self.mu_arr))
            self.mu_max = float(np.max(self.mu_arr))
        else:
            self.mu_arr = None
            self.n_mu = None
            self.mu_min = None
            self.mu_max = None

    def _serialize_convols_input(self, value):
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

    def _serialize_window_input(self, value):
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

    def _current_task_params_snapshot(self):
        params = {}
        params['convols_data'] = self._serialize_convols_input(self.convols_data)
        params['convols_data1'] = self._serialize_convols_input(self.convols_data1)
        params['convols_data2'] = self._serialize_convols_input(self.convols_data2)
        params['random'] = self._serialize_convols_input(self.random)
        params['random1'] = self._serialize_convols_input(self.random1)
        params['random2'] = self._serialize_convols_input(self.random2)
        params['window'] = self._serialize_window_input(self.window)
        params['window1'] = self._serialize_window_input(self.window1)
        params['window2'] = self._serialize_window_input(self.window2)
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
        params['fout_path'] = self.fout_path
        return params

    def _describe_pair_window(self, pair_window):
        if isinstance(pair_window, dict):
            return f"pair_window dict | {func_util.describe_window_action(pair_window)} | runtime separation follows current s"
        return "pair_window dict | default window with runtime separation s"

    def _describe_random_input(self, value):
        if value == "uniform":
            return "uniform random density"
        if isinstance(value, str):
            return f"path={value}"
        if isinstance(value, ConvolsData):
            return "provided random ConvolsData"
        if isinstance(value, (float, int, np.floating)):
            return f"density={float(value):.5e}"
        return "unset"

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
        return _field_density(field)

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
        pair_window_params = _pair_window_params_for_sample(s, mu, self.pair_window)
        return WindowFunc(pair_window_params, reference_field.convols_info, threads=self.threads)

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
            return _pair_product_with_window(a, b, pair_window_obj, self.threads)

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
            self.logger.info(f"requested_products={self.products}, expanded_products={expanded_products}")
            self.logger.info(f"Pair-correlation window: {self._describe_pair_window(self.pair_window)}")
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
                self.logger.info(self._describe_sampling())
                self.logger.info(
                    f"requested_products={self.products}, expanded_products={expanded_products}"
                )
                time_start = time.perf_counter()
                self.logger.info(f"Pre-2PCF setup time: {time_start - time_run_1:.4f} sec")
                self.logger.info(f"Main 2PCF loop products: {expanded_products}")
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
                next_report_threshold = 0
                requests = [None] + [comm.irecv(source=r, tag=r) for r in range(1, size)]
                count_all = False
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
                        self.logger.info(f" Progress: {progress:6.2f}%")
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
                    self.logger.info(f" Progress: {progress:6.2f}%")
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
