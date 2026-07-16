import copy
import itertools
import os
import pickle
import time

import numpy as np
from mpi4py import MPI

from pyhermes.io import WindowFunc, SFCField, Corr3PCFMultipoleData, normalize_task_weight_normalization
from pyhermes.pipeline import TaskBase
from pyhermes.utils import corr3pcf_multipoles as multipole_util
from pyhermes.utils import func_util
from pyhermes.utils.radial_multipole_windows import validate_radial_multipole_profile
from pyhermes.utils.radial_profiles import diagnose_radial_multipole_table
from pyhermes.utils.special_functions import solve_multipoles_from_ratio
from pyhermes.utils.window_params import serialize_window_params


PRODUCT_RULES = {
    "allowed": {"ddd_l", "rrr_l", "delta_ddd_l", "zeta_l"},
    "deps": {
        "zeta_l": ["delta_ddd_l", "rrr_l"],
    },
}

PRODUCT_INPUT_FLAGS = {
    "ddd_l": (True, False),
    "rrr_l": (False, True),
    "delta_ddd_l": (True, True),
    "zeta_l": (True, True),
}


class Corr_3PCF_Multipole(TaskBase):
    """Compute Legendre multipoles of 3PCF-like triplet products."""

    def __init__(self, param_task=None):
        if param_task is None:
            param_task = {"Corr_3PCF_Multipole": {}}
        self.task_name = str(self.__class__.__name__)
        super().__init__(param_task=param_task)
        self.format_params()
        self._fields_prepared = False

    def format_params(self):
        self.sfc_field = self.task_params.get("sfc_field", "")
        self.sfc_field1 = self.task_params.get("sfc_field1", "") or self.sfc_field
        self.sfc_field2 = self.task_params.get("sfc_field2", "") or self.sfc_field
        self.sfc_field3 = self.task_params.get("sfc_field3", "") or self.sfc_field
        self.random = self.task_params.get("random", None)
        self.random1 = self.task_params.get("random1", None)
        self.random2 = self.task_params.get("random2", None)
        self.random3 = self.task_params.get("random3", None)
        self.random1 = self._fallback_random(self.random1)
        self.random2 = self._fallback_random(self.random2)
        self.random3 = self._fallback_random(self.random3)
        self.weight_normalization = normalize_task_weight_normalization(self.task_params.get("weight_normalization", "catalog"))
        legacy_radius_keys = [key for key in ("r12", "r13", "r1", "r2") if key in self.task_params]
        if legacy_radius_keys:
            raise ValueError(
                "Corr_3PCF_Multipole no longer accepts top-level radius keys "
                f"{legacy_radius_keys}. Define radii through sampling and map them into binning_window12/13."
            )

        window = self.task_params.get("window", None)
        self.window = window if (isinstance(window, dict) and window.get("type")) else None
        for i in range(1, 4):
            window_i = self.task_params.get(f"window{i}", None)
            window_i = window_i if (isinstance(window_i, dict) and window_i.get("type")) else None
            if (not window_i) and self.window:
                window_i = dict(self.window)
            setattr(self, f"window{i}", window_i)

        self.binning_window12_template = self._normalize_edge_binning_window(
            self.task_params.get("binning_window12"), "binning_window12"
        )
        self.binning_window13_template = self._normalize_edge_binning_window(
            self.task_params.get("binning_window13"), "binning_window13"
        )
        self.sampling = self._normalize_sampling(self.task_params.get("sampling"))
        self.samples = self._expand_sampling(self.sampling)
        self.sample_params = self._sample_params_from_samples(self.samples)
        self.binning_window12 = [
            self._build_sample_binning_window(self.binning_window12_template, sample, "binning_window12")
            for sample in self.samples
        ]
        self.binning_window13 = [
            self._build_sample_binning_window(self.binning_window13_template, sample, "binning_window13")
            for sample in self.samples
        ]
        self.r12 = self.sample_params.get("r12")
        self.r13 = self.sample_params.get("r13")
        self.l_min = int(self.task_params["l_min"])
        self.l_max = int(self.task_params["l_max"])
        if self.l_min < 0 or self.l_max < 0 or self.l_min > self.l_max:
            raise ValueError("Corr_3PCF_Multipole requires 0 <= l_min <= l_max.")
        self._validate_binning_window_requests()
        self.gpu_device_id = int(self.task_params["gpu_device_id"])
        self.gpu_threads_per_block = multipole_util.normalize_gpu_threads_per_block(
            self.task_params.get("gpu_threads_per_block", (8, 8, 8))
        )
        self.summation_backend = multipole_util.normalize_summation_backend(
            self.task_params.get("summation_backend", "gpu")
        )
        self.execution_mode = str(self.task_params["execution_mode"]).strip().lower()
        if self.execution_mode not in {"serial", "pair_mpi", "sample_mpi"}:
            raise ValueError("Corr_3PCF_Multipole execution_mode must be 'serial', 'pair_mpi', or 'sample_mpi'.")
        self.sample_mpi = self._normalize_sample_mpi(self.task_params.get("sample_mpi", {}))
        self.cache_multipole_fields = bool(self.task_params["cache_multipole_fields"])
        self.cache_dir = self.task_params["cache_dir"]
        self.verbose_m_progress = bool(self.task_params["verbose_m_progress"])
        self.verbose_profile = bool(self.task_params.get("verbose_profile", False))
        self.zeta_condition_warning = float(self.task_params.get("zeta_condition_warning", 1.0e12))
        if not np.isfinite(self.zeta_condition_warning) or self.zeta_condition_warning <= 0.0:
            raise ValueError("zeta_condition_warning must be a finite positive number.")
        self.radial_profile_diagnostics_enabled = bool(
            self.task_params.get("radial_profile_diagnostics", True)
        )
        self.radial_profile_diagnostic_tolerance = float(
            self.task_params.get("radial_profile_diagnostic_tolerance", 1.0e-5)
        )
        if (
            not np.isfinite(self.radial_profile_diagnostic_tolerance)
            or self.radial_profile_diagnostic_tolerance <= 0.0
        ):
            raise ValueError("radial_profile_diagnostic_tolerance must be finite and positive.")
        self.radial_profile_diagnostic_probes = int(
            self.task_params.get("radial_profile_diagnostic_probes", 33)
        )
        if self.radial_profile_diagnostic_probes < 3:
            raise ValueError("radial_profile_diagnostic_probes must be at least 3.")
        self.threads = int(self.task_params["threads"])
        self.products = self._normalize_products(self.task_params.get("products", "zeta_l"))
        self.rho = None
        self.rho_legs = [None, None, None]
        self.reference_sfc = None
        self.resolution_diagnostics = []
        self.radial_profile_diagnostics = []
        self._last_product_profile = None
        self._role_layout_logged = False
        self.fout_path = self.task_params["fout_path"]

    def _fallback_random(self, value):
        return self.random if value is None or value == "" else value

    def _fallback_sfc(self, value):
        return self.sfc_field if value is None or value == "" else value

    def _fallback_window(self, value):
        if value is None:
            return self.window
        if isinstance(value, dict) and not value.get("type") and self.window:
            return self.window
        return value

    def _normalize_sample_mpi(self, value):
        if value is None:
            value = {}
        if not isinstance(value, dict):
            raise TypeError("Corr_3PCF_Multipole.sample_mpi must be a dictionary.")
        params = copy.deepcopy(value)
        ranks_per_sample = int(params.get("ranks_per_sample", 1))
        if ranks_per_sample != 1:
            raise ValueError("sample_mpi currently supports only ranks_per_sample=1.")
        gpu_device_ids = params.get("gpu_device_ids", [])
        if gpu_device_ids in (None, ""):
            gpu_device_ids = []
        elif isinstance(gpu_device_ids, (int, np.integer)):
            gpu_device_ids = [int(gpu_device_ids)]
        elif isinstance(gpu_device_ids, (list, tuple)):
            gpu_device_ids = [int(item) for item in gpu_device_ids]
        else:
            raise TypeError("sample_mpi.gpu_device_ids must be an integer or a list of integers.")
        return {
            "ranks_per_sample": ranks_per_sample,
            "gpu_device_ids": gpu_device_ids,
        }

    def _local_rank(self, rank):
        for env_name in (
            "OMPI_COMM_WORLD_LOCAL_RANK",
            "MV2_COMM_WORLD_LOCAL_RANK",
            "SLURM_LOCALID",
            "PMI_LOCAL_RANK",
        ):
            value = os.environ.get(env_name)
            if value is not None:
                try:
                    return int(value)
                except ValueError:
                    pass
        return int(rank)

    def _rank_gpu_device_id(self, rank):
        gpu_device_ids = self.sample_mpi.get("gpu_device_ids", [])
        if not gpu_device_ids:
            return self.gpu_device_id
        return int(gpu_device_ids[self._local_rank(rank) % len(gpu_device_ids)])

    def _normalize_edge_binning_window(self, value, name):
        if not isinstance(value, dict):
            raise TypeError(f"{name} must be a dictionary.")
        if not value.get("type"):
            raise ValueError(f"{name} must define a non-empty 'type'.")
        params = copy.deepcopy(value)
        params["type"] = str(params["type"]).strip().lower()
        params.setdefault("len_args", {})
        params.setdefault("other_args", {})
        params.setdefault("mapping", {})
        if params["len_args"] is None:
            params["len_args"] = {}
        if params["other_args"] is None:
            params["other_args"] = {}
        if params["mapping"] is None:
            params["mapping"] = {}
        if not isinstance(params["len_args"], dict):
            raise TypeError(f"{name}.len_args must be a dictionary.")
        if not isinstance(params["other_args"], dict):
            raise TypeError(f"{name}.other_args must be a dictionary.")
        if not isinstance(params["mapping"], dict):
            raise TypeError(f"{name}.mapping must be a dictionary.")
        return params

    def _normalize_sampling(self, sampling):
        if not isinstance(sampling, dict):
            raise TypeError("Corr_3PCF_Multipole requires a 'sampling' dictionary.")
        normalized = copy.deepcopy(sampling)
        mode = str(normalized.pop("mode", "grid")).strip().lower()
        if mode not in {"grid", "paired"}:
            raise ValueError("sampling.mode must be 'grid' or 'paired'.")
        values = {}
        for name, spec in normalized.items():
            values[name] = self._sampling_values(name, spec)
        if not values:
            raise ValueError("sampling must contain at least one sampled parameter.")
        return {"mode": mode, "values": values}

    def _sampling_values(self, name, spec):
        if isinstance(spec, dict):
            if "values" in spec:
                values = np.asarray(spec["values"], dtype=np.float64)
            elif {"min", "max", "n"}.issubset(spec):
                values = np.linspace(float(spec["min"]), float(spec["max"]), int(spec["n"]), dtype=np.float64)
            elif {"start", "stop", "step"}.issubset(spec):
                start = float(spec["start"])
                stop = float(spec["stop"])
                step = float(spec["step"])
                if step == 0.0:
                    raise ValueError(f"sampling.{name}.step must be non-zero.")
                values = np.arange(start, stop + 0.5 * step, step, dtype=np.float64)
            else:
                raise ValueError(
                    f"sampling.{name} must use 'values', 'min/max/n', 'start/stop/step', or a scalar."
                )
        elif isinstance(spec, (list, tuple, np.ndarray)):
            values = np.asarray(spec, dtype=np.float64)
        else:
            values = np.asarray([spec], dtype=np.float64)
        if values.ndim != 1 or values.size == 0:
            raise ValueError(f"sampling.{name} must expand to a non-empty 1D array.")
        return values

    def _expand_sampling(self, sampling):
        values = sampling["values"]
        names = list(values)
        if sampling["mode"] == "grid":
            samples = []
            for combo in itertools.product(*(values[name] for name in names)):
                samples.append({name: float(value) for name, value in zip(names, combo)})
            return samples

        lengths = [arr.size for arr in values.values()]
        n_samples = max(lengths)
        if any(length not in (1, n_samples) for length in lengths):
            raise ValueError("sampling.mode='paired' requires each parameter to have length 1 or the shared length.")
        samples = []
        for idx in range(n_samples):
            sample = {}
            for name, arr in values.items():
                sample[name] = float(arr[0] if arr.size == 1 else arr[idx])
            samples.append(sample)
        return samples

    def _sample_params_from_samples(self, samples):
        names = list(samples[0])
        return {
            name: np.asarray([sample[name] for sample in samples], dtype=np.float64)
            for name in names
        }

    def _resolve_sample_mapping_value(self, sample, source):
        if isinstance(source, str):
            if source not in sample:
                raise KeyError(f"Sampling parameter '{source}' is not defined.")
            return float(sample[source])
        return float(source)

    def _assign_mapped_binning_value(self, params, target, value):
        if "." not in target:
            params.setdefault("len_args", {})[target] = value
            return
        section, key = target.split(".", 1)
        if section not in {"len_args", "other_args"}:
            raise ValueError(f"Unsupported binning-window mapping target '{target}'.")
        params.setdefault(section, {})[key] = value

    def _build_sample_binning_window(self, template, sample, name):
        params = copy.deepcopy(template)
        mapping = params.pop("mapping")
        params["len_args"] = copy.deepcopy(params.get("len_args", {}))
        params["other_args"] = copy.deepcopy(params.get("other_args", {}))
        for target, source in mapping.items():
            value = self._resolve_sample_mapping_value(sample, source)
            self._assign_mapped_binning_value(params, target, value)
        if not params["len_args"]:
            raise ValueError(f"{name} produced an empty len_args dictionary.")
        return params

    def _validate_binning_window_requests(self):
        for name, windows in (
            ("binning_window12", self.binning_window12),
            ("binning_window13", self.binning_window13),
        ):
            for sample_idx, params in enumerate(windows):
                try:
                    validate_radial_multipole_profile(
                        params["type"], params["len_args"], self.l_max, params.get("other_args", {})
                    )
                except (TypeError, ValueError, KeyError) as exc:
                    raise ValueError(f"Invalid {name} for sample {sample_idx}: {exc}") from exc

    def _sync_runtime_options(self):
        self.threads = max(1, int(self.threads))
        self.products = self._normalize_products(self.products)
        self.task_params["threads"] = self.threads
        self.task_params["products"] = copy.deepcopy(self.products)
        self.task_params["weight_normalization"] = self.weight_normalization
        self.task_params["gpu_threads_per_block"] = list(self.gpu_threads_per_block)
        self.task_params["summation_backend"] = self.summation_backend
        self.task_params["zeta_condition_warning"] = self.zeta_condition_warning
        self.task_params["radial_profile_diagnostics"] = self.radial_profile_diagnostics_enabled
        self.task_params["radial_profile_diagnostic_tolerance"] = self.radial_profile_diagnostic_tolerance
        self.task_params["radial_profile_diagnostic_probes"] = self.radial_profile_diagnostic_probes
        self.sync_runtime_options(context="Corr_3PCF multipole runtime configuration", blank_line=True)
        if self.rank == 0:
            self.logger.info(
                "Field weight normalization rule | "
                f"task weight_normalization={self.weight_normalization} | "
                "catalog fields are converted to the task rule; derived fields are used as-is except for task='unit'."
            )

    def _normalize_products(self, products):
        if isinstance(products, str):
            products = [products]
        elif products is None:
            products = ["zeta_l"]
        elif not isinstance(products, (list, tuple, set)):
            raise TypeError(
                f"Unsupported products input: expected string or array of strings, got {type(products)}."
            )

        normalized = []
        for item in products:
            if not isinstance(item, str):
                raise TypeError("Each product name must be a string.")
            name = item.strip().lower()
            if name not in PRODUCT_RULES["allowed"]:
                raise ValueError(f"Unsupported product '{item}'. Allowed values are {sorted(PRODUCT_RULES['allowed'])}.")
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

    def _required_input_flags(self):
        needs_data = False
        needs_random = False
        for product in self._expanded_products():
            product_needs_data, product_needs_random = PRODUCT_INPUT_FLAGS[product]
            needs_data = needs_data or product_needs_data
            needs_random = needs_random or product_needs_random
        return needs_data, needs_random

    def _serialize_sfc_input(self, value):
        if isinstance(value, str) or value is None:
            return value
        if isinstance(value, SFCField):
            return {
                "kind": "SFCField",
                "L": getattr(value, "L", value.epsilon.shape[0] if value.epsilon is not None else None),
                "box_size": getattr(value, "box_size", None),
                "wavelet_mode": getattr(value, "wavelet_mode", None),
                "wavelet_level": getattr(value, "wavelet_level", None),
            }
        if np.isscalar(value):
            return float(value)
        return str(type(value))

    def _serialize_window_input(self, value):
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

    def _current_task_params_snapshot(self):
        return {
            "sfc_field": self._serialize_sfc_input(self.sfc_field),
            "sfc_field1": self._serialize_sfc_input(self.sfc_field1),
            "sfc_field2": self._serialize_sfc_input(self.sfc_field2),
            "sfc_field3": self._serialize_sfc_input(self.sfc_field3),
            "random": self._serialize_sfc_input(self.random),
            "random1": self._serialize_sfc_input(self.random1),
            "random2": self._serialize_sfc_input(self.random2),
            "random3": self._serialize_sfc_input(self.random3),
            "weight_normalization": self.weight_normalization,
            "window": self._serialize_window_input(self.window),
            "window1": self._serialize_window_input(self.window1),
            "window2": self._serialize_window_input(self.window2),
            "window3": self._serialize_window_input(self.window3),
            "binning_window12_template": self._serialize_window_input(self.binning_window12_template),
            "binning_window13_template": self._serialize_window_input(self.binning_window13_template),
            "sampling": serialize_window_params(self.sampling),
            "n_samples": len(self.samples),
            "l_min": self.l_min,
            "l_max": self.l_max,
            "gpu_device_id": self.gpu_device_id,
            "gpu_threads_per_block": list(self.gpu_threads_per_block),
            "summation_backend": self.summation_backend,
            "execution_mode": self.execution_mode,
            "sample_mpi": serialize_window_params(self.sample_mpi),
            "cache_multipole_fields": self.cache_multipole_fields,
            "cache_dir": self.cache_dir,
            "verbose_m_progress": self.verbose_m_progress,
            "verbose_profile": self.verbose_profile,
            "zeta_condition_warning": self.zeta_condition_warning,
            "radial_profile_diagnostics": self.radial_profile_diagnostics_enabled,
            "radial_profile_diagnostic_tolerance": self.radial_profile_diagnostic_tolerance,
            "radial_profile_diagnostic_probes": self.radial_profile_diagnostic_probes,
            "threads": self.threads,
            "products": copy.deepcopy(self.products),
            "expanded_products": self._expanded_products(),
            "rho_legs": copy.deepcopy(self.rho_legs),
            "resolution_diagnostics": copy.deepcopy(self.resolution_diagnostics),
            "radial_profile_diagnostics_result": copy.deepcopy(self.radial_profile_diagnostics),
            "fout_path": self.fout_path,
        }

    def _load_sfc_from_path(self, path):
        if self.execution_mode == "sample_mpi":
            if self.rank == 0:
                self.logger.info(
                    f"Sample-MPI input loading | rank0 reads SFCField from {path} and broadcasts it to all ranks."
                )
                field = SFCField(data_path=path, threads=self.threads)
            else:
                field = None
            return self._broadcast_sfc(self.rank, self.comm, field)
        return SFCField(data_path=path, threads=self.threads)

    def _resolve_base_sfc(self, leg_idx, provided_sfc, base_sfc_cache):
        if isinstance(provided_sfc, str) and provided_sfc:
            if provided_sfc not in base_sfc_cache:
                base_sfc_cache[provided_sfc] = self._load_sfc_from_path(provided_sfc)
            return base_sfc_cache[provided_sfc], f"path={provided_sfc}"
        if isinstance(provided_sfc, SFCField):
            return provided_sfc, f"provided sfc_field{leg_idx}"
        if provided_sfc in (None, ""):
            self.logger.error(
                f"Missing input for field leg {leg_idx}. Products {self._expanded_products()} require "
                f"'sfc_field{leg_idx}' or shared 'sfc_field'."
            )
            func_util.safe_exit(1)
        self.logger.error(
            f"Unexpected input: 'sfc_field{leg_idx}' must be a string path or a SFCField instance."
        )
        func_util.safe_exit(1)

    def _resolve_random_base(self, leg_idx, provided_random, random_cache):
        if provided_random is None or provided_random == "":
            self.logger.error(
                f"Missing input for random leg {leg_idx}. Products {self._expanded_products()} require "
                f"'random{leg_idx}' or shared 'random'. Use 'uniform' for the analytic uniform random field."
            )
            func_util.safe_exit(1)
        if isinstance(provided_random, str):
            if provided_random == "uniform":
                return "uniform", "uniform random density"
            if provided_random not in random_cache:
                random_cache[provided_random] = self._load_sfc_from_path(provided_random)
            return random_cache[provided_random], f"path={provided_random}"
        if isinstance(provided_random, SFCField):
            return provided_random, f"provided random{leg_idx}"
        self.logger.error(
            f"Unexpected input: 'random{leg_idx}' must be 'uniform', a string path, or a SFCField instance."
        )
        func_util.safe_exit(1)

    def _resolve_window(self, leg_idx, base_sfc, provided_window):
        if isinstance(provided_window, WindowFunc):
            return provided_window, "provided WindowFunc instance"
        if isinstance(provided_window, dict):
            return WindowFunc(provided_window, base_sfc.sfc_info, threads=self.threads), (
                f"provided window dict | {func_util.describe_window_action(provided_window)}"
            )
        if provided_window is not None:
            self.logger.error(
                f"Unsupported window input for leg {leg_idx}. Expected dict, WindowFunc, or None, got {type(provided_window)}."
            )
            func_util.safe_exit(1)
        return None, "no additional window convolution"

    def _field_mean_density(self, field):
        if isinstance(field, (float, int, np.floating)):
            return float(field)
        if isinstance(field, SFCField):
            value = field.field_mean_density(value_unit="grid")
            if value is None:
                raise ValueError("Cannot extract a uniform density from a field without field_integral metadata.")
            return value
        raise TypeError(f"Unsupported field type for density extraction: {type(field)}")

    def _remember_field_scale(self, field):
        if isinstance(field, SFCField):
            self._density_physical_scale = float(field.scale_factor) ** 3

    def _triplet_product_physical_scale(self):
        return float(getattr(self, "_density_physical_scale", 1.0)) ** 3

    def _scale_triplet_product_value(self, value):
        if value is None:
            return None
        factor = self._triplet_product_physical_scale()
        if np.isclose(factor, 1.0):
            return value
        return value * factor

    def _field_in_task_normalization(self, field):
        self._remember_field_scale(field)
        input_kind = getattr(field, "field_kind", None)
        input_norm = getattr(field, "weight_normalization", None)
        if self.weight_normalization == "unit":
            self.logger.info(
                "Field weight normalization | "
                f"field_kind={input_kind} | input={input_norm} | "
                "task=unit | effective=unit"
            )
            return field.with_normalization("unit")
        if input_kind != "catalog_field":
            self.logger.info(
                "Field weight normalization | "
                f"field_kind={input_kind} | input={input_norm} | "
                "effective=as-is; task weight_normalization applies only to catalog fields unless task='unit'."
            )
            return field.copy()
        self.logger.info(
            "Field weight normalization | "
            f"field_kind={input_kind} | input={input_norm} | "
            f"task={self.weight_normalization} | effective={self.weight_normalization}"
        )
        return field.switch_weight_normalization(self.weight_normalization)

    def _uniform_density(self):
        if self.rho is None:
            raise ValueError("Shared density is not initialized.")
        return float(self.rho)

    def _materialize_uniform_random(self, reference_field, rho, leg_idx):
        field = reference_field._spawn_like()
        field.epsilon = np.full((reference_field.L,) * 3, rho, dtype=np.float64)
        field.sfc_info.update({
            "catalog_weight_sum": None,
            "catalog_weight_sq_sum": None,
            "raw_field_weighted_sum": None,
            "field_integral": float(rho) * float(reference_field.V),
            "weight_normalization": None,
            "particle_count": None,
            "particle_data_retrievable": False,
            "particle_data_path": "",
            "particle_data_format": "",
            "field_kind": "derived_field",
            "uniform_random_materialized": True,
            "uniform_random_leg": int(leg_idx),
        })
        field.format_sfc_params()
        return field

    def _uniform_random_after_window(self, reference_field, rho, window_obj, leg_idx):
        """Return a uniform random leg after the same field-level window."""
        if window_obj is None:
            return float(rho), "uniform shortcut"

        zero_mode = complex(np.asarray(window_obj.as_array())[(0, 0, 0)])
        if abs(zero_mode.imag) <= 1.0e-12 * max(1.0, abs(zero_mode.real)) and np.isclose(
            zero_mode.real, 1.0, rtol=1.0e-10, atol=1.0e-12
        ):
            return float(rho), "uniform shortcut with W(0)=1"

        uniform_field = self._materialize_uniform_random(reference_field, rho, leg_idx)
        return uniform_field @ window_obj, f"uniform field convolved with W(0)={zero_mode:.6g}"

    def _record_resolution_diagnostics(self):
        self.resolution_diagnostics = []
        if self.l_max == 0:
            return

        scale_keys = {
            "shell": "R",
            "thick_shell": "R",
            "sphere": "R",
            "gaussian": "R",
            "gaussian_shell": "R_shell",
        }
        for name, windows in (
            ("binning_window12", self.binning_window12),
            ("binning_window13", self.binning_window13),
        ):
            radial_type = str(windows[0]["type"]).strip().lower()
            if radial_type not in scale_keys:
                self.logger.info(
                    f"Multipole resolution diagnostic | {name}: custom radial profile; "
                    "use its declared r_max to assess Hankel-table convergence."
                )
                continue
            scale_key = scale_keys[radial_type]
            radii = np.asarray([float(params["len_args"][scale_key]) for params in windows], dtype=np.float64)
            positive_radii = radii[radii > 0.0]
            if positive_radii.size == 0:
                continue
            q_grid_min = float(np.pi * positive_radii.min() * self.reference_sfc.L / self.reference_sfc.box_size)
            ratio = float(self.l_max / q_grid_min)
            diagnostic = {
                "edge": name,
                "radial_type": radial_type,
                "min_radius": float(positive_radii.min()),
                "max_radius": float(positive_radii.max()),
                "q_grid_min": q_grid_min,
                "l_max_over_q_grid_min": ratio,
            }
            self.resolution_diagnostics.append(diagnostic)
            if ratio >= 0.7:
                self.logger.warning(
                    "Multipole resolution diagnostic | "
                    f"{name}: l_max={self.l_max}, min radial scale={positive_radii.min():.6g}, "
                    f"pi R / dx={q_grid_min:.3f}, l_max/(pi R/dx)={ratio:.3f}. "
                    "The highest multipoles are close to the grid high-k range; check J convergence."
                )

    def _record_radial_profile_diagnostics(self, sample_indices=None):
        """Validate each tabulated radial multipole before convolution work."""
        self.radial_profile_diagnostics = []
        if not self.radial_profile_diagnostics_enabled:
            return

        if sample_indices is None:
            sample_indices = range(len(self.samples))
        else:
            sample_indices = [int(sample_idx) for sample_idx in sample_indices]

        k_max = float(np.sqrt(3.0) * self.reference_sfc.L / self.reference_sfc.box_size)
        for edge_name, windows in (
            ("binning_window12", self.binning_window12),
            ("binning_window13", self.binning_window13),
        ):
            for sample_idx in sample_indices:
                params = windows[sample_idx]
                radial_type = str(params["type"]).strip().lower()
                if radial_type == "shell":
                    continue
                if radial_type == "gaussian_shell" and float(params["len_args"]["R_smooth"]) == 0.0:
                    continue

                first_tabulated_l = self.l_min if radial_type.startswith("custom_") else max(1, self.l_min)
                for l in range(first_tabulated_l, self.l_max + 1):
                    diagnostic = diagnose_radial_multipole_table(
                        radial_type,
                        params["len_args"],
                        l,
                        k_max,
                        profile_config=params.get("other_args", {}),
                        tolerance=self.radial_profile_diagnostic_tolerance,
                        probe_count=self.radial_profile_diagnostic_probes,
                    )
                    diagnostic.update({"edge": edge_name, "sample_idx": int(sample_idx)})
                    self.radial_profile_diagnostics.append(diagnostic)
                    self.logger.info(
                        "Radial profile diagnostic | "
                        f"{edge_name}[{sample_idx}], type={radial_type}, l={l}, "
                        f"zero_error={diagnostic['zero_mode_error']:.3e}, "
                        f"table_error={diagnostic['table_convergence_error']:.3e}, "
                        f"inverse_error={diagnostic['inverse_roundtrip_error']}"
                    )
                    if not diagnostic["passed"]:
                        self.logger.warning(
                            "Radial profile diagnostic exceeded tolerance | "
                            f"{edge_name}[{sample_idx}], type={radial_type}, l={l}, "
                            f"tolerance={self.radial_profile_diagnostic_tolerance:.3e}. "
                            "Increase profile quadrature/table points or verify the profile support."
                        )

    def _broadcast_sfc(self, rank, comm, sfc_field):
        serialized = pickle.dumps(sfc_field.sfc_info) if rank == 0 else None
        serialized = comm.bcast(serialized, root=0)
        if rank == 0:
            local = sfc_field
            local.epsilon = np.ascontiguousarray(local.epsilon, dtype=np.float64)
        else:
            local = SFCField(threads=self.threads)
            local.sfc_info = pickle.loads(serialized)
            local.format_sfc_params()
            local.epsilon = np.empty((local.L, local.L, local.L), dtype=np.float64)
        comm.Bcast(local.epsilon, root=0)
        return local

    def _broadcast_sfc_to_ranks(self, comm, sfc_field, target_ranks):
        """
        Broadcast one SFCField only to ranks that need it.

        Rank 0 is always included as the sender. If rank 0 is not in
        target_ranks, it participates in the broadcast but returns None so the
        pair-MPI compute path does not keep an extra local reference.
        """
        rank = comm.Get_rank()
        members = set(target_ranks)
        members.add(0)
        subcomm = comm.Split(0 if rank in members else MPI.UNDEFINED, rank)
        if subcomm == MPI.COMM_NULL:
            return None

        serialized = pickle.dumps(sfc_field.sfc_info) if rank == 0 else None
        serialized = subcomm.bcast(serialized, root=0)
        if rank == 0:
            send_field = sfc_field
            send_field.epsilon = np.ascontiguousarray(send_field.epsilon, dtype=np.float64)
            subcomm.Bcast(send_field.epsilon, root=0)
            local = send_field if rank in target_ranks else None
        else:
            local = SFCField(threads=self.threads)
            local.sfc_info = pickle.loads(serialized)
            local.format_sfc_params()
            local.epsilon = np.empty((local.L, local.L, local.L), dtype=np.float64)
            subcomm.Bcast(local.epsilon, root=0)
        subcomm.Free()
        return local

    def _store_sampling_metadata(self):
        self.corr3pcf_multipole_data.sample_params = copy.deepcopy(self.sample_params)
        self.corr3pcf_multipole_data.binning_window12 = copy.deepcopy(self.binning_window12)
        self.corr3pcf_multipole_data.binning_window13 = copy.deepcopy(self.binning_window13)
        self.corr3pcf_multipole_data.r12 = copy.deepcopy(self.r12)
        self.corr3pcf_multipole_data.r13 = copy.deepcopy(self.r13)

    def prepare_input_fields(
        self,
        sfc_field1=None,
        sfc_field2=None,
        sfc_field3=None,
        random1=None,
        random2=None,
        random3=None,
        window1=None,
        window2=None,
        window3=None,
    ):
        self.corr3pcf_multipole_data = Corr3PCFMultipoleData()
        self._store_sampling_metadata()
        self._sync_runtime_options()

        if "zeta_l" in self._expanded_products() and self.l_min > 0:
            if self.rank == 0:
                self.logger.error("zeta_l requires l_min=0 because the ratio solve needs multipoles from l=0.")
            func_util.safe_exit(1)

        needs_data, needs_random = self._required_input_flags()
        data_inputs = [
            sfc_field1 if sfc_field1 is not None else self._fallback_sfc(self.sfc_field1),
            sfc_field2 if sfc_field2 is not None else self._fallback_sfc(self.sfc_field2),
            sfc_field3 if sfc_field3 is not None else self._fallback_sfc(self.sfc_field3),
        ]
        random_inputs = [
            random1 if random1 is not None else self._fallback_random(self.random1),
            random2 if random2 is not None else self._fallback_random(self.random2),
            random3 if random3 is not None else self._fallback_random(self.random3),
        ]
        window_inputs = [
            window1 if window1 is not None else self._fallback_window(self.window1),
            window2 if window2 is not None else self._fallback_window(self.window2),
            window3 if window3 is not None else self._fallback_window(self.window3),
        ]

        prepare_on_this_rank = self.rank == 0 or self.execution_mode == "sample_mpi"
        if prepare_on_this_rank:
            self.logger.info("Preparing Corr_3PCF multipole input fields ...")
            self.logger.info(
                f"execution_mode={self.execution_mode}, l_min={self.l_min}, l_max={self.l_max}, "
                f"n_samples={len(self.samples)}, threads={self.threads}, "
                f"summation_backend={self.summation_backend}, gpu_device_id={self.gpu_device_id}, "
                f"gpu_threads_per_block={self.gpu_threads_per_block}, "
                f"cache_multipole_fields={self.cache_multipole_fields}, "
                f"verbose_m_progress={self.verbose_m_progress}, verbose_profile={self.verbose_profile}"
            )
            self.logger.info(
                f"requested_products={self.products}, expanded_products={self._expanded_products()}"
            )

            base_sfc_cache = {}
            random_cache = {}
            data_legs = []
            random_legs = []
            compatibility_fields = []

            if needs_data:
                for i, cdata in enumerate(data_inputs, start=1):
                    base_sfc, source_desc = self._resolve_base_sfc(i, cdata, base_sfc_cache)
                    data_legs.append((i, base_sfc, source_desc, window_inputs[i - 1]))
                    compatibility_fields.append(base_sfc)

            if needs_random:
                for i, random_input in enumerate(random_inputs, start=1):
                    base_random, source_desc = self._resolve_random_base(i, random_input, random_cache)
                    random_legs.append((i, base_random, source_desc, window_inputs[i - 1]))
                    if isinstance(base_random, SFCField):
                        compatibility_fields.append(base_random)

            if not compatibility_fields:
                for i, cdata in enumerate(data_inputs, start=1):
                    if cdata is None or cdata == "":
                        continue
                    base_sfc, source_desc = self._resolve_base_sfc(i, cdata, base_sfc_cache)
                    compatibility_fields.append(base_sfc)
                    self.logger.info(
                        f"Geometry reference loaded from field leg {i} | source={source_desc}"
                    )
                    break

            if not compatibility_fields:
                self.logger.error(
                    "At least one SFCField input is required to define the grid geometry and shared density."
                )
                func_util.safe_exit(1)

            shared_required = func_util.validate_sfc_compatibility(
                compatibility_fields,
                SFCField._REQUIRED_ARGV,
                logger=self.logger,
                label="Corr_3PCF multipole input fields",
            )
            shared_required_text = ", ".join([f"{k}={v}" for k, v in shared_required.items()])
            self.reference_sfc = compatibility_fields[0]._spawn_like()
            self.reference_sfc.format_sfc_params()
            self.rho = self._field_mean_density(self._field_in_task_normalization(compatibility_fields[0]))
            self.rho_legs = [self.rho, self.rho, self.rho]
            self._record_resolution_diagnostics()
            if self.execution_mode == "sample_mpi":
                self._record_radial_profile_diagnostics(
                    self._sample_indices_for_rank(self.rank, self.comm.Get_size())
                )
            elif self.rank == 0:
                self._record_radial_profile_diagnostics()
            self.logger.info("Corr_3PCF multipole input compatibility check passed.")
            self.logger.info(f"Shared required parameters | {shared_required_text}")
            self.logger.info(f"Shared density | rho={self.rho:.6g}")

            normalized_data_fields = {}
            if needs_data:
                for i, base_sfc, source_desc, win in data_legs:
                    base_sfc = self._field_in_task_normalization(base_sfc)
                    normalized_data_fields[i] = base_sfc
                    self.rho_legs[i - 1] = self._field_mean_density(base_sfc)
                    window_obj, window_desc = self._resolve_window(i, base_sfc, win)
                    if window_obj is not None:
                        final_sfc = base_sfc @ window_obj
                    else:
                        final_sfc = base_sfc.copy()
                        final_sfc.format_sfc_params()

                    setattr(self, f"sfc_field{i}", final_sfc)
                    setattr(self.corr3pcf_multipole_data, f"sfc_info{i}", final_sfc.sfc_info)
                    self.logger.info(f"Field leg {i} ready | source={source_desc} | window={window_desc}")
            else:
                for i in range(1, 4):
                    setattr(self.corr3pcf_multipole_data, f"sfc_info{i}", self.reference_sfc.sfc_info)

            if needs_random:
                for i, base_random, source_desc, win in random_legs:
                    if isinstance(base_random, str) and base_random == "uniform":
                        reference_field = normalized_data_fields.get(i, self.reference_sfc)
                        leg_rho = float(self.rho_legs[i - 1])
                        window_obj, _ = self._resolve_window(i, reference_field, win)
                        uniform_random, uniform_desc = self._uniform_random_after_window(
                            reference_field, leg_rho, window_obj, i
                        )
                        setattr(self, f"random{i}", uniform_random)
                        self.logger.info(
                            f"Random leg {i} ready | source={source_desc} | window={uniform_desc} | rho={leg_rho:.6g}"
                        )
                        continue
                    base_random = self._field_in_task_normalization(base_random)
                    window_obj, window_desc = self._resolve_window(i, base_random, win)
                    if window_obj is not None:
                        final_random = base_random @ window_obj
                    else:
                        final_random = base_random.copy()
                        final_random.format_sfc_params()
                    setattr(self, f"random{i}", final_random)
                    self.logger.info(f"Random leg {i} ready | source={source_desc} | window={window_desc}")

            snapshot = self._current_task_params_snapshot()
            self.corr3pcf_multipole_data.corr3pcf_multipole_info = snapshot
            self.corr3pcf_multipole_data.task_params = snapshot

        if self.execution_mode == "sample_mpi":
            gathered_diagnostics = self.comm.gather(self.radial_profile_diagnostics, root=0)
            if self.rank == 0:
                self.radial_profile_diagnostics = [
                    diagnostic
                    for rank_diagnostics in gathered_diagnostics
                    for diagnostic in rank_diagnostics
                ]
                snapshot = self._current_task_params_snapshot()
                self.corr3pcf_multipole_data.corr3pcf_multipole_info = snapshot
                self.corr3pcf_multipole_data.task_params = snapshot
        self._fields_prepared = True

    def _store_product(self, product_name, l_arr, values):
        self._store_sampling_metadata()
        self.corr3pcf_multipole_data.l = np.asarray(l_arr, dtype=np.int32)
        values = np.asarray(values, dtype=np.float64)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if product_name in {"ddd_l", "rrr_l", "delta_ddd_l"}:
            values = self._scale_triplet_product_value(values)
        setattr(self.corr3pcf_multipole_data, product_name, values)

    def _log_helpers(self, product_name):
        def _format_complex(value):
            real = float(value.real)
            imag = float(value.imag)
            if abs(imag) < 1e-12 * max(1.0, abs(real)):
                return f"{real:.5e}"
            return f"({real:.5e}, {imag:.5e})"

        def _log_l_progress(
            l,
            l_max,
            ddd_l,
            zeta_l,
            elapsed_sec,
            conv_elapsed_sec,
            sum_elapsed_sec,
            completed_m_tasks,
            total_m_tasks,
        ):
            progress = (completed_m_tasks / total_m_tasks) * 100.0
            msg = f" l={l:2d}/{l_max:2d} done | progress={progress:6.2f}%"
            if self.verbose_profile:
                msg += (
                    f" | elapsed={elapsed_sec:.2f} sec | conv={conv_elapsed_sec:.2f} sec | "
                    f"sum={sum_elapsed_sec:.2f} sec"
                )
            self.logger.info(msg)

        def _log_m_progress(l, l_max, m, m_max, value, elapsed_sec, completed_m_tasks, total_m_tasks):
            progress = (completed_m_tasks / total_m_tasks) * 100.0
            msg = (
                f"   m={m:2d}/{m_max:2d} in l={l:2d}/{l_max:2d} | "
                f"value={_format_complex(value)} | elapsed={elapsed_sec:.2f} sec | "
                f"progress={progress:6.2f}% ({completed_m_tasks}/{total_m_tasks})"
            )
            print(msg, flush=True)

        return _log_l_progress, _log_m_progress

    def _run_serial_mode(self, rank, fields, product_name, binning_window12, binning_window13, sample_idx):
        if rank != 0:
            return None, None

        self.logger.info(
            f"Start to calculate 3PCF multipole product '{product_name}' "
            f"for sample {sample_idx + 1}/{len(self.samples)} ..."
        )
        self.logger.info(
            f"execution_mode={self.execution_mode}, l_min={self.l_min}, l_max={self.l_max}, "
            f"threads={self.threads}, summation_backend={self.summation_backend}, gpu_device_id={self.gpu_device_id}, "
            f"gpu_threads_per_block={self.gpu_threads_per_block}, "
            f"cache_multipole_fields={self.cache_multipole_fields}, "
            f"verbose_m_progress={self.verbose_m_progress}, verbose_profile={self.verbose_profile}"
        )

        log_l_progress, log_m_progress = self._log_helpers(product_name)
        l_arr, multipole_l, timing_info = multipole_util.calc_DDD_multipole(
            fields[0], fields[1], fields[2],
            binning_window12, binning_window13, self.l_min, self.l_max,
            summation_backend=self.summation_backend,
            gpu_device_id=self.gpu_device_id,
            gpu_threads_per_block=self.gpu_threads_per_block,
            cache_multipole_fields=self.cache_multipole_fields,
            cache_dir=self.cache_dir,
            cache_namespace=f"{product_name}_sample{sample_idx:04d}",
            threads=self.threads,
            progress_callback=log_l_progress,
            m_progress_callback=log_m_progress if self.verbose_m_progress else None,
        )
        self._last_product_profile = {
            "conv": timing_info["conv_elapsed_sec"],
            "sum": timing_info["sum_elapsed_sec"],
            "gpu_sum": timing_info["sum_elapsed_sec"],
            "h2d": timing_info["sum_h2d_elapsed_sec"],
            "kernel": timing_info["sum_kernel_elapsed_sec"],
            "d2h": timing_info["sum_d2h_elapsed_sec"],
            "reduce": timing_info["sum_reduce_elapsed_sec"],
        }
        if self.verbose_profile:
            self.logger.info(
                f"3PCF multipole timing [{product_name}] | convolution={timing_info['conv_elapsed_sec']:.2f} sec | "
                f"summation={timing_info['sum_elapsed_sec']:.2f} sec"
            )
            self.logger.info(
                f"3PCF multipole summation breakdown [{product_name}, backend={self.summation_backend}] | "
                f"h2d={timing_info['sum_h2d_elapsed_sec']:.2f} sec | "
                f"kernel={timing_info['sum_kernel_elapsed_sec']:.2f} sec | "
                f"d2h={timing_info['sum_d2h_elapsed_sec']:.2f} sec | "
                f"reduce={timing_info['sum_reduce_elapsed_sec']:.2f} sec | "
                f"callback={timing_info['sum_callback_elapsed_sec']:.2f} sec"
            )
        return l_arr, multipole_l

    def _run_pair_mpi_mode(self, comm, rank, local_fields, product_name, binning_window12, binning_window13, sample_idx):
        size = comm.Get_size()
        if size == 1:
            if rank == 0:
                self.logger.warning(
                    "execution_mode='pair_mpi' requested with a single MPI rank. Falling back to serial execution."
                )
                self.execution_mode = "serial"
            return self._run_serial_mode(rank, local_fields, product_name, binning_window12, binning_window13, sample_idx)
        if size < 2 or size % 2 != 0:
            self.logger.error("execution_mode='pair_mpi' requires an even number of MPI ranks.")
            func_util.safe_exit(1)
        n_pairs = size // 2

        if rank == 0:
            self.logger.info(
                f"Start to calculate 3PCF multipole product '{product_name}' "
                f"for sample {sample_idx + 1}/{len(self.samples)} ..."
            )
            self.logger.info(
                f"execution_mode={self.execution_mode}, l_min={self.l_min}, l_max={self.l_max}, "
                f"threads={self.threads}, ranks={size}, pairs={n_pairs}, summation_backend={self.summation_backend}, "
                f"gpu_device_id={self.gpu_device_id}, gpu_threads_per_block={self.gpu_threads_per_block}, "
                f"cache_multipole_fields={self.cache_multipole_fields}, "
                f"verbose_m_progress={self.verbose_m_progress}, verbose_profile={self.verbose_profile}"
            )
            if self.summation_backend == "cpu":
                self.logger.info(
                    "Pair-MPI CPU summation layout | each rank pair evaluates its m-term locally; "
                    "rank 0 gathers scalar results only."
                )

        field1, field2, field3 = local_fields
        pair_idx = rank if rank < n_pairs else rank - n_pairs
        is_r1_rank = rank < n_pairs

        conv_context_r1 = multipole_util._prepare_legendre_convolution_context(field2) if is_r1_rank else None
        conv_context_r2 = multipole_util._prepare_legendre_convolution_context(field3) if not is_r1_rank else None
        sum_context = (
            multipole_util._prepare_multipole_sum_context(
                field1,
                summation_backend=self.summation_backend,
                gpu_device_id=self.gpu_device_id,
                gpu_threads_per_block=self.gpu_threads_per_block,
            )
            if rank == 0 or (self.summation_backend == "cpu" and is_r1_rank)
            else None
        )

        task_list = []
        l_arr = np.arange(self.l_min, self.l_max + 1, dtype=np.int32)
        for l_idx, l in enumerate(range(self.l_min, self.l_max + 1)):
            for m in range(0, l + 1):
                task_list.append((l_idx, l, m))
        multipole_l = np.empty(l_arr.size, dtype=np.float64) if rank == 0 else None
        m_storage = {int(l): np.empty(int(l) + 1, dtype=np.complex128) for l in l_arr} if rank == 0 else None
        done_per_l = {int(l): 0 for l in l_arr} if rank == 0 else None
        total_conv_elapsed = 0.0
        total_sum_elapsed = 0.0
        total_comm_elapsed = 0.0
        total_h2d = total_kernel = total_reduce = total_d2h = 0.0
        total_m_tasks = len(task_list)
        completed_m_tasks = 0
        _, log_m_progress = self._log_helpers(product_name)
        l_wall_starts = {int(l): None for l in l_arr} if rank == 0 else None
        l_conv_accum = ({int(l): 0.0 for l in l_arr} if (rank == 0 and self.verbose_profile) else None)
        l_comm_accum = ({int(l): 0.0 for l in l_arr} if (rank == 0 and self.verbose_profile) else None)
        l_sum_accum = ({int(l): 0.0 for l in l_arr} if (rank == 0 and self.verbose_profile) else None)

        n_rounds = (len(task_list) + n_pairs - 1) // n_pairs
        for round_idx in range(n_rounds):
            active_tasks = task_list[round_idx * n_pairs : (round_idx + 1) * n_pairs]
            active_count = len(active_tasks)
            round_meta = np.full((n_pairs, 3), -1, dtype=np.int32)
            for idx, (l_idx, l, m) in enumerate(active_tasks):
                round_meta[idx] = (l_idx, l, m)
            comm.Bcast(round_meta, root=0)

            t_conv = time.perf_counter()
            local_field = None
            local_meta = tuple(round_meta[pair_idx])
            if pair_idx < active_count:
                _, l, m = local_meta
                if is_r1_rank:
                    local_field = multipole_util._stream_convolution_fields(
                        field2, binning_window12, int(l), threads=self.threads, m_values=[int(m)],
                        conv_context=conv_context_r1,
                        cache_multipole_fields=self.cache_multipole_fields,
                        cache_dir=self.cache_dir,
                        cache_namespace=f"{product_name}_sample{sample_idx:04d}_side12",
                    )[0]
                else:
                    local_field = multipole_util._stream_convolution_fields(
                        field3, binning_window13, int(l), threads=self.threads, m_values=[-int(m)],
                        conv_context=conv_context_r2,
                        cache_multipole_fields=self.cache_multipole_fields,
                        cache_dir=self.cache_dir,
                        cache_namespace=f"{product_name}_sample{sample_idx:04d}_side13",
                    )[0]
            conv_elapsed = time.perf_counter() - t_conv
            total_conv_elapsed += conv_elapsed

            round_summands = {} if rank == 0 else None
            if self.summation_backend == "cpu":
                local_result = None
                comm_elapsed = 0.0
                if pair_idx < active_count:
                    l_idx, l, m = map(int, local_meta)
                    tag = 200000 + round_idx * 100 + pair_idx
                    if is_r1_rank:
                        t_recv = time.perf_counter()
                        field_r2_m = np.empty(local_field.shape, dtype=np.complex128)
                        comm.Recv([field_r2_m, MPI.COMPLEX16], source=n_pairs + pair_idx, tag=tag)
                        comm_elapsed += time.perf_counter() - t_recv

                        t_sum = time.perf_counter()
                        value, timing = multipole_util.compute_multipole_m_summand(
                            local_field, field_r2_m, sum_context
                        )
                        sum_elapsed = time.perf_counter() - t_sum
                        total_sum_elapsed += sum_elapsed
                        local_result = (l_idx, l, m, value, timing, sum_elapsed)
                        del local_field, field_r2_m
                    else:
                        t_send = time.perf_counter()
                        comm.Send(
                            [np.ascontiguousarray(local_field, dtype=np.complex128), MPI.COMPLEX16],
                            dest=pair_idx,
                            tag=tag,
                        )
                        comm_elapsed += time.perf_counter() - t_send
                        del local_field

                t_gather = time.perf_counter()
                gathered_results = comm.gather(local_result, root=0)
                comm_elapsed += time.perf_counter() - t_gather
                if rank == 0:
                    for result in gathered_results:
                        if result is None:
                            continue
                        recv_l_idx, recv_l, recv_m, value, timing, sum_elapsed = result
                        round_summands[(recv_l_idx, recv_l, recv_m)] = (value, timing, sum_elapsed)
                    if len(round_summands) != active_count:
                        raise RuntimeError(
                            "pair_mpi CPU summation returned "
                            f"{len(round_summands)}/{active_count} active m-task results."
                        )
                total_comm_elapsed += comm_elapsed
            else:
                t_comm = time.perf_counter()
                if pair_idx < active_count:
                    l_idx, l, m = local_meta
                    tag_base = 200000 + round_idx * 100 + pair_idx
                    if rank == 0:
                        root_comm_elapsed = 0.0
                        for idx in range(active_count):
                            recv_l_idx, recv_l, recv_m = map(int, round_meta[idx])
                            key = (recv_l_idx, recv_l, recv_m)
                            if idx == 0:
                                field_r1_m = local_field
                                t_recv = time.perf_counter()
                                recv_r2 = np.empty(local_field.shape, dtype=np.complex128)
                                comm.Recv([recv_r2, MPI.COMPLEX16], source=n_pairs, tag=tag_base + 50)
                                root_comm_elapsed += time.perf_counter() - t_recv
                                field_r2_m = recv_r2
                            else:
                                t_recv = time.perf_counter()
                                recv_r1 = np.empty(local_field.shape, dtype=np.complex128)
                                recv_r2 = np.empty(local_field.shape, dtype=np.complex128)
                                comm.Recv([recv_r1, MPI.COMPLEX16], source=idx, tag=200000 + round_idx * 100 + idx)
                                comm.Recv([recv_r2, MPI.COMPLEX16], source=n_pairs + idx, tag=200000 + round_idx * 100 + idx + 50)
                                root_comm_elapsed += time.perf_counter() - t_recv
                                field_r1_m = recv_r1
                                field_r2_m = recv_r2
                            t_sum = time.perf_counter()
                            value, timing = multipole_util.compute_multipole_m_summand(
                                field_r1_m, field_r2_m, sum_context
                            )
                            sum_elapsed = time.perf_counter() - t_sum
                            round_summands[key] = (value, timing, sum_elapsed)
                            del field_r1_m, field_r2_m
                            if idx == 0:
                                del recv_r2
                            else:
                                del recv_r1, recv_r2
                        local_field = None
                        comm_elapsed = root_comm_elapsed
                    elif is_r1_rank:
                        if rank != 0:
                            comm.Send(
                                [np.ascontiguousarray(local_field, dtype=np.complex128), MPI.COMPLEX16],
                                dest=0,
                                tag=tag_base,
                            )
                            del local_field
                    else:
                        comm.Send(
                            [np.ascontiguousarray(local_field, dtype=np.complex128), MPI.COMPLEX16],
                            dest=0,
                            tag=tag_base + 50,
                        )
                        del local_field
                if rank != 0:
                    comm_elapsed = time.perf_counter() - t_comm
                total_comm_elapsed += comm_elapsed

            round_timings = None
            if self.verbose_m_progress or self.verbose_profile:
                local_timing = np.array(
                    [
                        int(pair_idx if pair_idx < active_count else -1),
                        int(0 if is_r1_rank else 1),
                        float(conv_elapsed),
                        float(comm_elapsed),
                    ],
                    dtype=np.float64,
                )
                round_timings = comm.gather(local_timing, root=0)

            if rank == 0:
                timing_by_task = {}
                if self.verbose_m_progress or self.verbose_profile:
                    for item in round_timings:
                        idx_float, side_float, conv_val, comm_val = item
                        idx_task = int(idx_float)
                        if idx_task < 0 or idx_task >= active_count:
                            continue
                        side = int(side_float)
                        timing_by_task.setdefault(idx_task, {})[side] = (float(conv_val), float(comm_val))
                for idx in range(active_count):
                    l_idx, l, m = map(int, round_meta[idx])
                    key = (l_idx, l, m)
                    if l_wall_starts[l] is None:
                        l_wall_starts[l] = time.perf_counter()
                    task_timing = timing_by_task.get(idx, {})
                    conv_r1 = task_timing.get(0, (0.0, 0.0))[0]
                    conv_r2 = task_timing.get(1, (0.0, 0.0))[0]
                    comm_r1 = task_timing.get(0, (0.0, 0.0))[1]
                    comm_r2 = task_timing.get(1, (0.0, 0.0))[1]
                    value, timing, sum_elapsed = round_summands[key]
                    if self.summation_backend != "cpu":
                        total_sum_elapsed += sum_elapsed
                    total_h2d += timing["h2d_elapsed_sec"]
                    total_kernel += timing["kernel_elapsed_sec"]
                    total_reduce += timing["reduce_elapsed_sec"]
                    total_d2h += timing["d2h_elapsed_sec"]
                    m_storage[l][m] = value
                    done_per_l[l] += 1
                    completed_m_tasks += 1
                    if self.verbose_profile:
                        l_conv_accum[l] += max(conv_r1, conv_r2)
                        l_comm_accum[l] += max(comm_r1, comm_r2)
                        l_sum_accum[l] += sum_elapsed
                    if self.verbose_m_progress:
                        log_m_progress(
                            l=l, l_max=self.l_max, m=m, m_max=l, value=value,
                            elapsed_sec=max(conv_r1, conv_r2) + max(comm_r1, comm_r2) + sum_elapsed,
                            completed_m_tasks=completed_m_tasks, total_m_tasks=total_m_tasks,
                        )
                    if done_per_l[l] == l + 1:
                        multipole_l[l_idx] = multipole_util.combine_multipole_m_terms(m_storage[l], l)
                        progress = (completed_m_tasks / total_m_tasks) * 100.0
                        if self.verbose_profile:
                            self.logger.info(
                                f" l={l:2d}/{self.l_max:2d} done | "
                                f"elapsed={time.perf_counter() - l_wall_starts[l]:.2f} sec | "
                                f"conv={l_conv_accum[l]:.2f} sec | comm={l_comm_accum[l]:.2f} sec | "
                                f"sum={l_sum_accum[l]:.2f} sec | "
                                f"progress={progress:6.2f}% ({completed_m_tasks}/{total_m_tasks} m-tasks)"
                            )
                        else:
                            self.logger.info(
                                f" l={l:2d}/{self.l_max:2d} done | "
                                f"progress={progress:6.2f}% ({completed_m_tasks}/{total_m_tasks} m-tasks)"
                            )

        conv_max_rank = comm.reduce(total_conv_elapsed, op=MPI.MAX, root=0)
        comm_max_rank = comm.reduce(total_comm_elapsed, op=MPI.MAX, root=0)
        sum_max_rank = comm.reduce(total_sum_elapsed, op=MPI.MAX, root=0)
        sum_all_ranks = comm.reduce(total_sum_elapsed, op=MPI.SUM, root=0)
        conv_sum_all = comm.reduce(total_conv_elapsed, op=MPI.SUM, root=0) if self.verbose_profile else None
        comm_sum_all = comm.reduce(total_comm_elapsed, op=MPI.SUM, root=0) if self.verbose_profile else None
        if rank == 0:
            self._last_product_profile = {
                "conv_rank0": total_conv_elapsed,
                "conv_max": conv_max_rank,
                "comm_max": comm_max_rank,
                "sum": sum_max_rank,
                "sum_all_ranks": sum_all_ranks,
                "gpu_sum": sum_max_rank,
                "h2d": total_h2d,
                "kernel": total_kernel,
                "d2h": total_d2h,
                "reduce": total_reduce,
            }
        if rank == 0 and self.verbose_profile:
            self.logger.info(
                f"Pair-MPI timing [{product_name}] | conv_rank0={total_conv_elapsed:.2f} sec | "
                f"conv_sum_all={conv_sum_all:.2f} sec | conv_max_rank={conv_max_rank:.2f} sec | "
                f"comm_sum_all={comm_sum_all:.2f} sec | comm_max_rank={comm_max_rank:.2f} sec | "
                f"sum_max_rank={sum_max_rank:.2f} sec | sum_all_ranks={sum_all_ranks:.2f} sec"
            )
            self.logger.info(
                f"Pair-MPI summation breakdown [{product_name}, backend={self.summation_backend}] | h2d={total_h2d:.2f} sec | "
                f"kernel={total_kernel:.2f} sec | d2h={total_d2h:.2f} sec | reduce={total_reduce:.2f} sec"
            )
        return l_arr if rank == 0 else None, multipole_l

    def _is_uniform_random(self, field):
        return isinstance(field, (float, int, np.floating))

    def _delta_field(self, data_field, random_field):
        if self._is_uniform_random(random_field):
            return data_field - self._field_mean_density(random_field)
        return data_field - random_field

    def _prepare_product_fields(self, product_name):
        if product_name == "ddd_l":
            return [self.sfc_field1, self.sfc_field2, self.sfc_field3]
        if product_name == "delta_ddd_l":
            return [
                self._delta_field(self.sfc_field1, self.random1),
                self._delta_field(self.sfc_field2, self.random2),
                self._delta_field(self.sfc_field3, self.random3),
            ]
        if product_name == "rrr_l":
            reference = self.reference_sfc
            fields = []
            for i, random_field in enumerate([self.random1, self.random2, self.random3], start=1):
                if self._is_uniform_random(random_field):
                    self.logger.info(
                        f"Random leg {i} for rrr_l materialized from uniform density for the generic multipole kernel."
                    )
                    fields.append(self._materialize_uniform_random(reference, float(random_field), i))
                else:
                    fields.append(random_field)
            return fields
        raise ValueError(f"Product '{product_name}' does not map to direct multipole fields.")

    def _all_random_uniform(self):
        return all(self._is_uniform_random(getattr(self, f"random{i}", None)) for i in range(1, 4))

    def _analytic_uniform_rrr_l(self):
        l_arr = np.arange(self.l_min, self.l_max + 1, dtype=np.int32)
        rrr_l = np.zeros(l_arr.size, dtype=np.float64)
        zero_idx = np.where(l_arr == 0)[0]
        if zero_idx.size:
            rrr_l[zero_idx[0]] = float(self.random1) * float(self.random2) * float(self.random3)
        return l_arr, rrr_l

    def _sample_indices_for_rank(self, rank, size):
        return list(range(int(rank), len(self.samples), int(size)))

    def _run_sample_mpi_local_sample(self, rank, fields, product_name, binning_window12, binning_window13, sample_idx):
        gpu_device_id = self._rank_gpu_device_id(rank)
        backend_location = f"gpu_device_id={gpu_device_id}" if self.summation_backend == "gpu" else "CPU"
        self.logger.info(
            f"Rank {rank} starts 3PCF multipole product '{product_name}' "
            f"for sample {sample_idx + 1}/{len(self.samples)} with "
            f"summation_backend={self.summation_backend} on {backend_location}."
        )
        log_l_progress, log_m_progress = self._log_helpers(product_name)
        progress_callback = log_l_progress if (self.verbose_profile or self.verbose_m_progress) else None
        l_arr, multipole_l, timing_info = multipole_util.calc_DDD_multipole(
            fields[0], fields[1], fields[2],
            binning_window12, binning_window13, self.l_min, self.l_max,
            summation_backend=self.summation_backend,
            gpu_device_id=gpu_device_id,
            gpu_threads_per_block=self.gpu_threads_per_block,
            cache_multipole_fields=self.cache_multipole_fields,
            cache_dir=self.cache_dir,
            cache_namespace=f"{product_name}_sample{sample_idx:04d}",
            threads=self.threads,
            progress_callback=progress_callback,
            m_progress_callback=log_m_progress if self.verbose_m_progress else None,
        )
        return l_arr, multipole_l, timing_info

    def _compute_product_sample_mpi(self, product_name):
        comm = self.comm
        rank = self.rank
        size = comm.Get_size()
        local_sample_indices = self._sample_indices_for_rank(rank, size)
        if rank == 0:
            sample_counts = [len(self._sample_indices_for_rank(i, size)) for i in range(size)]
            self.logger.info(
                f"execution_mode=sample_mpi | ranks={size}, n_samples={len(self.samples)}, "
                "assignment=static_round_robin | "
                f"samples_per_rank=[{min(sample_counts)}, {max(sample_counts)}]"
            )
            if self.verbose_profile:
                assignment = "; ".join(
                    f"rank{i}:{self._sample_indices_for_rank(i, size)}" for i in range(size)
                )
                self.logger.info(f"sample_mpi assignment | {assignment}")

        local_l_arr = None
        local_rows = []
        local_timing = {
            "conv": 0.0,
            "sum": 0.0,
            "gpu_sum": 0.0,
            "h2d": 0.0,
            "kernel": 0.0,
            "d2h": 0.0,
            "reduce": 0.0,
        }
        fields = self._prepare_product_fields(product_name)
        for sample_idx in local_sample_indices:
            l_arr, product_l, timing_info = self._run_sample_mpi_local_sample(
                rank,
                fields,
                product_name,
                self.binning_window12[sample_idx],
                self.binning_window13[sample_idx],
                sample_idx,
            )
            local_l_arr = l_arr
            local_rows.append((sample_idx, product_l))
            local_timing["conv"] += timing_info["conv_elapsed_sec"]
            local_timing["sum"] += timing_info["sum_elapsed_sec"]
            local_timing["gpu_sum"] += timing_info["sum_elapsed_sec"]
            local_timing["h2d"] += timing_info["sum_h2d_elapsed_sec"]
            local_timing["kernel"] += timing_info["sum_kernel_elapsed_sec"]
            local_timing["d2h"] += timing_info["sum_d2h_elapsed_sec"]
            local_timing["reduce"] += timing_info["sum_reduce_elapsed_sec"]
        del fields

        gathered = comm.gather((local_l_arr, local_rows, local_timing), root=0)
        if rank != 0:
            return None, None

        l_arr = next((item[0] for item in gathered if item[0] is not None), None)
        if l_arr is None:
            raise ValueError("sample_mpi did not produce any local multipole result.")
        product_values = np.empty((len(self.samples), len(l_arr)), dtype=np.float64)
        seen = np.zeros(len(self.samples), dtype=bool)
        timing_keys = ("conv", "sum", "gpu_sum", "h2d", "kernel", "d2h", "reduce")
        timing_max = {key: max(float(item[2][key]) for item in gathered) for key in timing_keys}
        timing_sum = {key: sum(float(item[2][key]) for item in gathered) for key in timing_keys}
        for _, rows, _ in gathered:
            for sample_idx, product_l in rows:
                product_values[int(sample_idx)] = np.asarray(product_l, dtype=np.float64)
                seen[int(sample_idx)] = True
        if not np.all(seen):
            missing = np.where(~seen)[0].tolist()
            raise ValueError(f"sample_mpi missing results for sample indices {missing}.")
        self._last_product_profile = {
            "conv": timing_max["conv"],
            "sum": timing_max["sum"],
            "gpu_sum": timing_max["gpu_sum"],
            "h2d": timing_max["h2d"],
            "kernel": timing_max["kernel"],
            "d2h": timing_max["d2h"],
            "reduce": timing_max["reduce"],
            "sample_mpi_conv_sum": timing_sum["conv"],
            "sample_mpi_sum": timing_sum["sum"],
            "sample_mpi_gpu_sum": timing_sum["gpu_sum"],
        }
        self.logger.info(
            f"sample_mpi timing [{product_name}] | conv_max_rank={timing_max['conv']:.2f} sec | "
            f"sum_max_rank={timing_max['sum']:.2f} sec | conv_sum_all={timing_sum['conv']:.2f} sec | "
            f"sum_all={timing_sum['sum']:.2f} sec"
        )
        return l_arr, product_values

    def _compute_product_multipole(self, product_name, fields, binning_window12, binning_window13, sample_idx):
        comm = self.comm
        rank = self.rank
        if self.execution_mode == "pair_mpi":
            size = comm.Get_size()
            if size == 1:
                return self._run_pair_mpi_mode(
                    comm, rank, fields if rank == 0 else None, product_name,
                    binning_window12, binning_window13, sample_idx,
                )
            if size < 2 or size % 2 != 0:
                self.logger.error("execution_mode='pair_mpi' requires an even number of MPI ranks.")
                func_util.safe_exit(1)
            n_pairs = size // 2
            field1_ranks = set(range(n_pairs)) if self.summation_backend == "cpu" else {0}
            if rank == 0:
                self.logger.info(
                    f"Initializing multipole input for '{product_name}': role-aware broadcast to {size} MPI ranks ..."
                )
                if not self._role_layout_logged:
                    self.logger.info(
                        f"Role-aware layout | field1 -> ranks {sorted(field1_ranks)} | "
                        f"field2 -> ranks 0-{n_pairs - 1} | field3 -> ranks {n_pairs}-{size - 1}"
                    )
                    self._role_layout_logged = True
            local_fields = [
                self._broadcast_sfc_to_ranks(comm, fields[0] if rank == 0 else None, field1_ranks),
                self._broadcast_sfc_to_ranks(comm, fields[1] if rank == 0 else None, set(range(n_pairs))),
                self._broadcast_sfc_to_ranks(comm, fields[2] if rank == 0 else None, set(range(n_pairs, size))),
            ]
            if rank == 0:
                fields[:] = [None, None, None]
            if rank == 0:
                self.logger.info(f"Initializing multipole input for '{product_name}': role-aware broadcast complete.")
            return self._run_pair_mpi_mode(
                comm, rank, local_fields, product_name, binning_window12, binning_window13, sample_idx
            )
        return self._run_serial_mode(
            rank, fields if rank == 0 else None, product_name, binning_window12, binning_window13, sample_idx
        )

    def _compute_zeta_l(self):
        delta_ddd_l = self.corr3pcf_multipole_data.delta_ddd_l
        rrr_l = self.corr3pcf_multipole_data.rrr_l
        if delta_ddd_l is None or rrr_l is None:
            self.logger.error("zeta_l requires both delta_ddd_l and rrr_l.")
            func_util.safe_exit(1)
        delta_ddd_l = np.asarray(delta_ddd_l, dtype=np.float64)
        rrr_l = np.asarray(rrr_l, dtype=np.float64)
        if delta_ddd_l.ndim == 1:
            delta_ddd_l = delta_ddd_l.reshape(1, -1)
        if rrr_l.ndim == 1:
            rrr_l = rrr_l.reshape(1, -1)
        zeta_rows = []
        cond_values = []
        for sample_idx in range(delta_ddd_l.shape[0]):
            zeta_row, _, cond_m = solve_multipoles_from_ratio(
                delta_ddd_l[sample_idx],
                rrr_l[sample_idx],
                self.l_max,
                rcond_warning=np.inf,
            )
            zeta_rows.append(zeta_row)
            cond_values.append(cond_m)
        zeta_l = np.asarray(zeta_rows, dtype=np.float64)
        self.corr3pcf_multipole_data.zeta_l = zeta_l
        self.corr3pcf_multipole_data.zeta_condition = np.asarray(cond_values, dtype=np.float64)
        bad_conditions = np.where(self.corr3pcf_multipole_data.zeta_condition > self.zeta_condition_warning)[0]
        if bad_conditions.size:
            self.logger.warning(
                "zeta_l mixing matrices exceed zeta_condition_warning="
                f"{self.zeta_condition_warning:.3e} for samples {bad_conditions.tolist()}."
            )
        self.logger.info(
            "zeta_l solved from multipole ratio | "
            f"mixing matrix cond range=[{np.min(cond_values):.3e}, {np.max(cond_values):.3e}]"
        )

    def _log_product_timing(self, product_name, elapsed_sec):
        profile = self._last_product_profile or {}
        msg = f"Product timing | {product_name}={elapsed_sec:.4f} sec"
        if profile:
            if "conv_max" in profile:
                msg += (
                    f" | conv_max={profile['conv_max']:.2f} sec"
                    f" | comm_max={profile['comm_max']:.2f} sec"
                    f" | sum={profile.get('sum', profile.get('gpu_sum', 0.0)):.2f} sec"
                )
            elif "conv" in profile:
                msg += f" | conv={profile['conv']:.2f} sec | sum={profile.get('sum', profile.get('gpu_sum', 0.0)):.2f} sec"
            if self.verbose_profile and "kernel" in profile:
                msg += (
                    f" | h2d={profile['h2d']:.2f} sec"
                    f" | kernel={profile['kernel']:.2f} sec"
                    f" | d2h={profile['d2h']:.2f} sec"
                    f" | reduce={profile['reduce']:.2f} sec"
                )
        self.logger.info(msg)

    def run(self, save_result=True, overwrite=False):
        try:
            comm = self.comm
            rank = self.rank
            if rank == 0:
                t0 = time.perf_counter()

            if not self._fields_prepared:
                self.prepare_input_fields()

            self.rho = comm.bcast(self.rho if rank == 0 else None, root=0)
            expanded_products = self._expanded_products()
            if rank == 0:
                snapshot = self._current_task_params_snapshot()
                self.corr3pcf_multipole_data.corr3pcf_multipole_info = snapshot
                self.corr3pcf_multipole_data.task_params = snapshot
                t_start = time.perf_counter()
                self.logger.info(f"Pre-3PCF multipole setup time: {t_start - t0:.4f} sec")
                self.logger.info(f"Main 3PCF multipole products: {[p for p in expanded_products if p != 'zeta_l']}")

            for product_name in expanded_products:
                if product_name == "zeta_l":
                    continue
                if rank == 0:
                    product_t0 = time.perf_counter()
                    self._last_product_profile = None
                    self.logger.info(f"Computing product '{product_name}' ...")
                all_random_uniform = comm.bcast(
                    self._all_random_uniform() if rank == 0 else None,
                    root=0,
                )
                if product_name == "rrr_l" and all_random_uniform:
                    if rank == 0:
                        l_arr, product_l = self._analytic_uniform_rrr_l()
                        product_values = np.repeat(product_l.reshape(1, -1), len(self.samples), axis=0)
                        self._store_product(product_name, l_arr, product_values)
                        self.logger.info(
                            f"Product 'rrr_l' used all-uniform analytic shortcut | rho^3={self._uniform_density() ** 3:.6e}"
                        )
                        self._log_product_timing(product_name, time.perf_counter() - product_t0)
                    continue

                if self.execution_mode == "sample_mpi":
                    l_arr, product_values = self._compute_product_sample_mpi(product_name)
                    if rank == 0:
                        self._store_product(product_name, l_arr, product_values)
                        self._log_product_timing(product_name, time.perf_counter() - product_t0)
                    continue

                product_rows = []
                l_arr = None
                reusable_fields = self._prepare_product_fields(product_name) if (
                    rank == 0 and self.execution_mode == "serial"
                ) else None
                for sample_idx, (binning_window12, binning_window13) in enumerate(
                    zip(self.binning_window12, self.binning_window13)
                ):
                    fields = reusable_fields if self.execution_mode == "serial" else (
                        self._prepare_product_fields(product_name) if rank == 0 else None
                    )
                    l_arr, product_l = self._compute_product_multipole(
                        product_name,
                        fields,
                        binning_window12,
                        binning_window13,
                        sample_idx,
                    )
                    if rank == 0 and self.execution_mode != "serial":
                        product_rows.append(product_l)
                        del fields
                    elif rank == 0:
                        product_rows.append(product_l)
                if rank == 0 and reusable_fields is not None:
                    del reusable_fields
                if rank == 0:
                    product_values = np.asarray(product_rows, dtype=np.float64)
                    self._store_product(product_name, l_arr, product_values)
                    self._log_product_timing(product_name, time.perf_counter() - product_t0)

            if rank == 0:
                if "zeta_l" in expanded_products:
                    product_t0 = time.perf_counter()
                    self._last_product_profile = None
                    self.logger.info("Computing product 'zeta_l' from delta_ddd_l and rrr_l ...")
                    self._compute_zeta_l()
                    self._log_product_timing("zeta_l", time.perf_counter() - product_t0)

                t_end = time.perf_counter()
                self.logger.info(f"The time for 3PCF multipole: {t_end - t_start:.4f} sec")

                if save_result and self.fout_path:
                    self.corr3pcf_multipole_data.saveflag = True
                    self.corr3pcf_multipole_data.save_corr3pcf_multipole(self.fout_path, overwrite=overwrite)

            comm.Barrier()
        except Exception as e:
            self.logger.error(f"Error in process {self.rank}: {str(e)}")
            func_util.safe_exit(1)

        if rank == 0:
            t1 = time.perf_counter()
            print("")
            self.logger.info(f"The time for task: {t1 - t0:.4f} sec")
        return self.corr3pcf_multipole_data
