import time
import pickle
import copy
import numpy as np

from pyhermes.io import WindowFunc, ConvolsData, Corr3PCFData, normalize_field_normalization
from pyhermes.utils import func_util
from pyhermes.utils.corr3pcf_kernels import (
    estimate_triplet_product_box_random_centers,
    estimate_triplet_product_particle_centers,
    third_side_from_mu,
)
from pyhermes.utils.mpi_util import MPI
from pyhermes.utils.sampling import random_box_positions
from pyhermes.utils.window_params import serialize_window_params
from pyhermes.pipeline import TaskBase

from .corr2pcf import compute_pair_product_at_sample


### Product dependency metadata ###

# Product expansion rules are kept in one place so the runtime logic only needs
# to consume an expanded execution plan instead of duplicating dependency checks.
PRODUCT_RULES = {
    "box_random": {
        "allowed": {"ddd", "rrr", "delta_ddd", "xi12", "xi13", "xi23", "zeta", "zeta_H", "Q"},
        "deps": {
            "Q": ["zeta", "zeta_H"],
            "zeta_H": ["xi12", "xi13", "xi23"],
            "zeta": ["delta_ddd", "rrr"],
        },
    },
    "particle": {
        "allowed": {"ddd", "rrr", "d_delta_dd", "r_delta_dd", "delta_ddd", "xi12", "xi13", "xi23", "zeta", "zeta_H", "Q"},
        "deps": {
            "Q": ["zeta", "zeta_H"],
            "zeta_H": ["xi12", "xi13", "xi23"],
            "zeta": ["delta_ddd", "rrr"],
            "delta_ddd": ["d_delta_dd", "r_delta_dd"],
        },
    },
}

# Each product declares whether it needs data legs, random legs, or both.
PRODUCT_INPUT_FLAGS = {
    "ddd": (True, False),
    "rrr": (False, True),
    "d_delta_dd": (True, True),
    "r_delta_dd": (True, True),
    "delta_ddd": (True, True),
    "xi12": (True, True),
    "xi13": (True, True),
    "xi23": (True, True),
    "zeta": (True, True),
    "zeta_H": (True, True),
    "Q": (True, True),
}


### Shared triplet estimator dispatch ###

def estimate_triplet_product_with_sampled_centers(
    r12_scaled, r13_scaled, mu, center_scaled, n_rot,
    convols_meta, convols_data2, convols_data3,
    center="box_random",
    center_weight=None,
    center_weight_sum=None,
    seed_base_rot=-1,
    mu_index=-1,
    eps1=None,
    rho1=None,
):
    """Estimate a triplet product using either box-random or particle centers."""
    kwargs_common = {
        "phi_array": convols_meta.phi_array,
        "L": convols_meta.L,
        "phi_resolution": convols_meta.phi_resolution,
        "phi_support": convols_meta.phi_support,
        "seed_base_rot": seed_base_rot,
        "mu_index": mu_index,
    }

    eps2 = convols_data2.epsilon
    eps3 = convols_data3.epsilon

    if center == "box_random":
        if eps1 is None:
            raise ValueError("eps1 must be provided when center='box_random'.")
        return estimate_triplet_product_box_random_centers(
            r12_scaled, r13_scaled, mu,
            center_scaled, n_rot,
            eps1, eps2, eps3,
            **kwargs_common,
        )

    if center == "particle":
        if rho1 is None:
            rho1 = 1 / convols_meta.L ** 3
        if center_scaled.shape[0] == 0:
            return 0.0
        if center_weight is None:
            center_weight = np.ones(center_scaled.shape[0], dtype=np.float64)
        if center_weight.shape[0] != center_scaled.shape[0]:
            raise ValueError("center_weight must have the same length as center_scaled.")
        if center_weight_sum is None:
            center_weight_sum = float(np.sum(center_weight))
        if center_weight_sum <= 0.0:
            raise ValueError("Particle-center weights must have a positive sum.")
        return estimate_triplet_product_particle_centers(
            r12_scaled, r13_scaled, mu,
            center_scaled, center_weight, center_weight_sum, n_rot,
            rho1, eps2, eps3,
            **kwargs_common,
        )

    raise ValueError(f"Unknown center='{center}'. Use 'box_random' or 'particle'.")


class Corr_3PCF(TaskBase):
    """Three-point correlation estimator with box-random and particle-center modes."""

    ### Construction and user parameter normalization ###

    def __init__(self, param_task=None):
        if param_task is None:
            param_task = {"Corr_3PCF": {}}
        self.task_name = str(self.__class__.__name__)
        super().__init__(param_task=param_task)
        self.format_params()
        self.rho = None
        self._fields_prepared = False

    def _sync_runtime_options(self):
        self.threads = max(1, int(self.threads))
        self.task_params["threads"] = self.threads
        self.task_params["products"] = copy.deepcopy(self.products)
        self.task_params["particle_weight1"] = self.particle_weight1
        self.task_params["random_weight1"] = self.random_weight1
        self.task_params["field_normalization"] = self.field_normalization
        self.sync_runtime_options(context="Corr_3PCF runtime configuration", blank_line=True)

    def format_params(self):
        """Read user-facing task parameters into runtime attributes."""
        if "normalization" in self.task_params:
            raise TypeError(
                "Corr_3PCF.normalization has been removed. "
                "Use field_normalization='none' or field_normalization='mean'."
            )
        self.convols_data = self.task_params.get("convols_data", "")
        self.convols_data1 = self.task_params.get("convols_data1", "") or self.convols_data
        self.convols_data2 = self.task_params.get("convols_data2", "") or self.convols_data
        self.convols_data3 = self.task_params.get("convols_data3", "") or self.convols_data
        self.particle_pos1 = self.task_params.get("particle_pos1", None)
        self.particle_weight1 = self.task_params.get("particle_weight1", None)
        self.random = self.task_params.get("random", None)
        self.random1 = self.task_params.get("random1", None)
        self.random2 = self.task_params.get("random2", None)
        self.random3 = self.task_params.get("random3", None)
        if self.random1 in (None, ""):
            self.random1 = self.random
        if self.random2 in (None, ""):
            self.random2 = self.random
        if self.random3 in (None, ""):
            self.random3 = self.random
        self.random_pos1 = self.task_params.get("random_pos1", None)
        self.random_weight1 = self.task_params.get("random_weight1", None)
        self.field_normalization = normalize_field_normalization(self.task_params.get("field_normalization", "none"))

        window = self.task_params.get("window", None)
        self.window = window if (window and (window.get("type") or window.get("func"))) else None
        for i in range(1, 4):
            window_i = self.task_params.get(f"window{i}", None)
            window_i = window_i if (window_i and (window_i.get("type") or window_i.get("func"))) else None
            if (not window_i) and self.window:
                window_i = dict(self.window)
            setattr(self, f"window{i}", window_i)

        self.r12 = float(self.task_params["r12"])
        self.r13 = float(self.task_params["r13"])
        self.angle_param = self.task_params["angle_param"]
        self.theta = copy.deepcopy(self.task_params["theta"])
        self.mu = copy.deepcopy(self.task_params["mu"])
        self.theta_arr = None
        self.mu_arr = None
        self.theta_min = None
        self.theta_max = None
        self.mu_min = None
        self.mu_max = None
        self.n_theta = None
        self.n_mu = None
        self.n_rot = int(self.task_params["n_rot"])
        self.center = self.task_params["center"]
        self.n_box_centers = int(self.task_params["n_box_centers"])
        self.base_seed = int(self.task_params["base_seed"])
        self.threads = int(self.task_params["threads"])
        self.products = self._normalize_products(self.task_params.get("products", "Q"))
        self.fout_path = self.task_params["fout_path"]

    ### Product planning and angle sampling ###

    def _normalize_products(self, products):
        """Normalize requested products and validate them against the current center mode."""
        if isinstance(products, str):
            products = [products]
        elif products is None:
            products = ["Q"]
        elif not isinstance(products, (list, tuple, set)):
            raise TypeError(
                f"Unsupported products input: expected string or array of strings, got {type(products)}."
            )

        allowed = PRODUCT_RULES[self.center]["allowed"]

        normalized = []
        for item in products:
            if not isinstance(item, str):
                raise TypeError("Each product name must be a string.")
            raw_name = item.strip()
            if raw_name.upper() == "Q":
                name = "Q"
            elif raw_name.lower() == "zeta_h":
                name = "zeta_H"
            else:
                name = raw_name.lower()
            if name not in allowed:
                raise ValueError(f"Unsupported product '{item}' for center='{self.center}'. Allowed values are {sorted(allowed)}.")
            if name not in normalized:
                normalized.append(name)
        return normalized

    def _expanded_products(self):
        """Expand requested products to the full dependency-closed execution plan."""
        expanded = list(self.products)
        rules = PRODUCT_RULES[self.center]["deps"]
        idx = 0
        while idx < len(expanded):
            for dep in rules.get(expanded[idx], []):
                if dep not in expanded:
                    expanded.append(dep)
            idx += 1
        return expanded

    def _required_input_flags(self):
        """Determine whether the current execution plan requires data and/or random inputs."""
        needs_data = False
        needs_random = False
        for product in self._expanded_products():
            product_needs_data, product_needs_random = PRODUCT_INPUT_FLAGS[product]
            needs_data = needs_data or product_needs_data
            needs_random = needs_random or product_needs_random
        return needs_data, needs_random

    def _normalize_random(self, value):
        if value in (None, ""):
            return None
        if value == "uniform":
            return "uniform"
        return value

    def _serialize_angle_spec(self, value):
        if isinstance(value, dict):
            return copy.deepcopy(value)
        if value is None:
            return None
        arr = np.asarray(value, dtype=np.float64)
        return arr.tolist()

    def _normalize_sampling_array(self, values, name, vmin, vmax):
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim != 1:
            raise TypeError(f"'{name}' must be a 1D array-like input.")
        if arr.size == 0:
            raise ValueError(f"'{name}' must contain at least one sampling point.")
        if np.any(arr < vmin) or np.any(arr > vmax):
            raise ValueError(f"'{name}' values must lie in [{vmin}, {vmax}].")
        diffs = np.diff(arr)
        if np.any(diffs < 0.0) and np.any(diffs > 0.0):
            raise ValueError(f"'{name}' values must be monotonic.")
        return np.ascontiguousarray(arr, dtype=np.float64)

    def _resolve_angle_sampling(self):
        angle_param = str(self.angle_param).strip().lower()
        if angle_param not in {"theta", "mu"}:
            raise ValueError(f"Unsupported angle_param='{self.angle_param}'. Use 'theta' or 'mu'.")

        if angle_param == "theta":
            theta_spec = self.theta
            if theta_spec is None:
                raise ValueError("angle_param='theta' requires 'theta' to be provided as a dict or array-like input.")
            if isinstance(theta_spec, dict):
                theta_min = float(theta_spec.get("theta_min", 0.0))
                theta_max = float(theta_spec.get("theta_max", float(np.pi)))
                n_theta = int(theta_spec.get("n_theta", 20))
                theta_arr = np.linspace(theta_min, theta_max, n_theta, dtype=np.float64)
            else:
                theta_arr = self._normalize_sampling_array(theta_spec, "theta", 0.0, float(np.pi))
            mu_arr = np.cos(theta_arr)
        else:
            mu_spec = self.mu
            if mu_spec is None:
                raise ValueError("angle_param='mu' requires 'mu' to be provided as a dict or array-like input.")
            if isinstance(mu_spec, dict):
                mu_min = float(mu_spec.get("mu_min", -1.0))
                mu_max = float(mu_spec.get("mu_max", 1.0))
                n_mu = int(mu_spec.get("n_mu", 20))
                mu_arr = np.linspace(mu_min, mu_max, n_mu, dtype=np.float64)
            else:
                mu_arr = self._normalize_sampling_array(mu_spec, "mu", -1.0, 1.0)
            theta_arr = np.arccos(np.clip(mu_arr, -1.0, 1.0))

        self.angle_param = angle_param
        self.theta_arr = np.ascontiguousarray(theta_arr, dtype=np.float64)
        self.mu_arr = np.ascontiguousarray(mu_arr, dtype=np.float64)
        self.n_theta = int(self.theta_arr.size)
        self.n_mu = int(self.mu_arr.size)
        self.theta_min = float(np.min(self.theta_arr))
        self.theta_max = float(np.max(self.theta_arr))
        self.mu_min = float(np.min(self.mu_arr))
        self.mu_max = float(np.max(self.mu_arr))

    ### Task snapshot serialization ###

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
        """Record a serializable snapshot of the effective task configuration."""
        params = {}
        params["convols_data"] = self._serialize_convols_input(self.convols_data)
        params["convols_data1"] = self._serialize_convols_input(self.convols_data1)
        params["convols_data2"] = self._serialize_convols_input(self.convols_data2)
        params["convols_data3"] = self._serialize_convols_input(self.convols_data3)
        if self.particle_pos1 is None:
            params["particle_pos1"] = None
        else:
            arr = np.asarray(self.particle_pos1)
            params["particle_pos1"] = {"kind": "particle_pos1", "shape": tuple(arr.shape)}
        if self.particle_weight1 is None:
            params["particle_weight1"] = None
        else:
            arr = np.asarray(self.particle_weight1)
            params["particle_weight1"] = {"kind": "particle_weight1", "shape": tuple(arr.shape)}
        params["random"] = self._serialize_convols_input(self.random)
        params["random1"] = self._serialize_convols_input(self.random1)
        params["random2"] = self._serialize_convols_input(self.random2)
        params["random3"] = self._serialize_convols_input(self.random3)
        if self.random_pos1 is None:
            params["random_pos1"] = None
        else:
            arr = np.asarray(self.random_pos1)
            params["random_pos1"] = {"kind": "random_pos1", "shape": tuple(arr.shape)}
        if self.random_weight1 is None:
            params["random_weight1"] = None
        else:
            arr = np.asarray(self.random_weight1)
            params["random_weight1"] = {"kind": "random_weight1", "shape": tuple(arr.shape)}
        params["field_normalization"] = self.field_normalization
        params["window"] = self._serialize_window_input(self.window)
        params["window1"] = self._serialize_window_input(self.window1)
        params["window2"] = self._serialize_window_input(self.window2)
        params["window3"] = self._serialize_window_input(self.window3)
        params["r12"] = self.r12
        params["r13"] = self.r13
        params["angle_param"] = self.angle_param
        params["theta_spec"] = self._serialize_angle_spec(self.theta)
        params["mu_spec"] = self._serialize_angle_spec(self.mu)
        params["n_theta"] = self.n_theta
        params["n_mu"] = self.n_mu
        params["theta_min"] = self.theta_min
        params["theta_max"] = self.theta_max
        params["mu_min"] = self.mu_min
        params["mu_max"] = self.mu_max
        params["n_rot"] = self.n_rot
        params["center"] = self.center
        params["n_box_centers"] = self.n_box_centers
        params["base_seed"] = self.base_seed
        params["threads"] = self.threads
        params["products"] = copy.deepcopy(self.products)
        params["fout_path"] = self.fout_path
        return params

    ### Input resolution and field preparation helpers ###

    def _resolve_base_convols(self, leg_idx, provided_convols, cache):
        """Resolve one signal leg from a path, shared fallback, or ConvolsData instance."""
        if provided_convols is not None:
            if isinstance(provided_convols, str):
                if provided_convols not in cache:
                    cache[provided_convols] = ConvolsData(data_path=provided_convols, threads=self.threads)
                return cache[provided_convols], f"path={provided_convols}"
            if not isinstance(provided_convols, ConvolsData):
                self.logger.error(
                    f"Unexpected input: 'convols_data{leg_idx}' must be a string path or a ConvolsData instance."
                )
                func_util.safe_exit(1)
            return provided_convols, f"provided convols_data{leg_idx}"

        base_input = getattr(self, f"convols_data{leg_idx}")
        if isinstance(base_input, str) and base_input:
            if base_input not in cache:
                cache[base_input] = ConvolsData(data_path=base_input, threads=self.threads)
            return cache[base_input], f"path={base_input}"
        if isinstance(base_input, ConvolsData):
            return base_input, f"provided convols_data{leg_idx}"
        self.logger.error(
            f"Missing usable input for field leg {leg_idx}. Expected a string path or ConvolsData instance in "
            f"'convols_data{leg_idx}' or shared 'convols_data'."
        )
        func_util.safe_exit(1)

    def _resolve_random_base(self, leg_idx, provided_random, cache):
        """Resolve one random leg, allowing the special uniform shortcut."""
        provided_random = self._normalize_random(provided_random)
        if provided_random is None:
            return None, "no random input"
        if provided_random == "uniform":
            return "uniform", "uniform random density"
        if isinstance(provided_random, str):
            if provided_random not in cache:
                cache[provided_random] = ConvolsData(data_path=provided_random, threads=self.threads)
            return cache[provided_random], f"path={provided_random}"
        if isinstance(provided_random, ConvolsData):
            return provided_random, f"provided random{leg_idx}"
        self.logger.error(
            f"Unexpected input: 'random{leg_idx}' must be 'uniform', a string path, a ConvolsData instance, or None."
        )
        func_util.safe_exit(1)

    def _resolve_window(self, leg_idx, base_convols, provided_window):
        """Resolve a smoothing window for one leg from dict/WindowFunc/None."""
        if isinstance(provided_window, WindowFunc):
            return provided_window, "provided WindowFunc instance"
        if isinstance(provided_window, dict):
            return WindowFunc(provided_window, base_convols.convols_info, threads=self.threads), (
                f"provided window dict | {func_util.describe_window_action(provided_window)}"
            )
        if provided_window is not None:
            self.logger.error(
                f"Unsupported window input for leg {leg_idx}. Expected dict, WindowFunc, or None, got {type(provided_window)}."
            )
            func_util.safe_exit(1)
        return None, "no additional window convolution"

    def _field_density(self, field):
        """Extract the uniform-density representation used by shortcut branches."""
        if isinstance(field, (float, int, np.floating)):
            return float(field)
        if isinstance(field, ConvolsData):
            return 1.0 / field.V
        raise TypeError(f"Unsupported field type for density extraction: {type(field)}")

    def _compute_delta_field(self, field, random_field):
        """Build one contrast field D-R, using a scalar-density shortcut when possible."""
        if isinstance(random_field, (float, int, np.floating)):
            return field - self._field_density(random_field)
        return field - random_field

    def _shared_density(self):
        """Return the common density implied by the validated compatible fields."""
        if self.rho is None:
            raise RuntimeError("Shared density is not initialized.")
        return self.rho

    def _find_geometry_reference(self, *candidates):
        """Pick the first ConvolsData object that can define geometry/scale metadata."""
        for candidate in candidates:
            if isinstance(candidate, ConvolsData):
                return candidate
        return None

    ### Center coordinate and weight handling ###

    def _normalize_particle_data(self, value):
        """Normalize explicit center positions to a contiguous (N, 3) float64 array."""
        if value is None:
            return None
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(
                f"particle_pos1 must be an array-like with shape (N, 3), got shape {arr.shape}."
            )
        return np.ascontiguousarray(arr, dtype=np.float64)

    def _normalize_particle_weight(self, value, npos, label="particle_weight1"):
        """Normalize center weights to a contiguous positive-sum float64 array."""
        if value is None:
            arr = np.ones(npos, dtype=np.float64)
        else:
            arr = np.asarray(value, dtype=np.float64)
            if arr.ndim != 1 or arr.shape[0] != npos:
                raise ValueError(f"{label} must be a 1D array with length {npos}, got shape {arr.shape}.")
            arr = np.ascontiguousarray(arr, dtype=np.float64)
        if float(np.sum(arr)) <= 0.0:
            raise ValueError(f"{label} must have a positive sum.")
        return arr

    def _center_projection_weight(self, field, particle_data):
        weight = np.asarray(particle_data["projection_weight"], dtype=np.float64)
        if self.field_normalization == "mean":
            normalizer = getattr(field, "field_normalization_value", None)
            if normalizer is None:
                normalizer = getattr(field, "field_integral", None)
            if normalizer is None or not np.isfinite(normalizer) or np.isclose(normalizer, 0.0):
                raise ValueError("Cannot normalize particle-center field without a non-zero field_integral.")
            weight = weight / normalizer
        return weight

    def _resolve_pos1_array(self, provided_pos, provided_weight, fallback_field, label, explicit_name):
        """Resolve leg-1 center coordinates from explicit arrays or from a ConvolsData source."""
        pos_arr = self._normalize_particle_data(provided_pos)
        if pos_arr is not None:
            weight_arr = self._normalize_particle_weight(provided_weight, pos_arr.shape[0])
            return pos_arr, weight_arr, f"provided {explicit_name}"
        if fallback_field is None:
            return None, None, None
        try:
            particle_data = fallback_field.get_particle_data()
            pos_arr = self._normalize_particle_data(particle_data["pos"])
            if provided_weight is not None:
                raw_weight = provided_weight
            else:
                raw_weight = self._center_projection_weight(fallback_field, particle_data)
            weight_arr = self._normalize_particle_weight(raw_weight, pos_arr.shape[0], label=f"{label} particle weight")
            return pos_arr, weight_arr, f"from {label}.get_particle_data()"
        except Exception:
            self.logger.error(
                f"For center='particle', {label} could not provide usable particle coordinates. "
                f"Please provide {explicit_name} explicitly."
            )
            func_util.safe_exit(1)

    ### Runtime input fallbacks and MPI data movement ###

    def _resolve_runtime_inputs(
        self,
        convols_data1, convols_data2, convols_data3,
        particle_pos1, particle_weight1, random1, random2, random3, random_pos1, random_weight1,
        window1, window2, window3,
    ):
        """Apply shared-input fallbacks right before preparation/run-time use."""
        if convols_data1 is None:
            convols_data1 = self.convols_data1 if self.convols_data1 not in (None, "") else self.convols_data
        if convols_data2 is None:
            convols_data2 = self.convols_data2 if self.convols_data2 not in (None, "") else self.convols_data
        if convols_data3 is None:
            convols_data3 = self.convols_data3 if self.convols_data3 not in (None, "") else self.convols_data
        if particle_pos1 is None:
            particle_pos1 = self.particle_pos1
        if particle_weight1 is None:
            particle_weight1 = self.particle_weight1
        if random1 is None:
            random1 = self.random1 if self.random1 not in (None, "") else self.random
        if random2 is None:
            random2 = self.random2 if self.random2 not in (None, "") else self.random
        if random3 is None:
            random3 = self.random3 if self.random3 not in (None, "") else self.random
        if random_pos1 is None:
            random_pos1 = self.random_pos1
        if random_weight1 is None:
            random_weight1 = self.random_weight1
        if window1 is None:
            window1 = self.window1 if self.window1 is not None else self.window
        if window2 is None:
            window2 = self.window2 if self.window2 is not None else self.window
        if window3 is None:
            window3 = self.window3 if self.window3 is not None else self.window
        return (
            convols_data1, convols_data2, convols_data3, particle_pos1, particle_weight1, 
            random1, random2, random3, random_pos1, random_weight1,
            window1, window2, window3,
        )

    def _broadcast_field(self, value):
        """Broadcast a ConvolsData field or scalar density to all MPI ranks."""
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

    def _scatter_positions(self, pos_all):
        """Scatter a rank-0 position array into rank-local position chunks."""
        comm = self.comm
        rank = self.rank
        size = comm.Get_size()
        if rank == 0:
            n_all = pos_all.shape[0]
            counts = np.full(size, n_all // size, dtype=np.int64)
            counts[: (n_all % size)] += 1
            displs = np.zeros(size, dtype=np.int64)
            displs[1:] = np.cumsum(counts[:-1])
            sendbuf = np.ascontiguousarray(pos_all, dtype=np.float64).ravel()
            counts3 = counts * 3
            displs3 = displs * 3
        else:
            counts = sendbuf = counts3 = displs3 = None
        n_local = int(comm.scatter(counts, root=0))
        recvbuf = np.empty(n_local * 3, dtype=np.float64)
        comm.Scatterv([sendbuf, counts3, displs3, MPI.DOUBLE], recvbuf, root=0)
        return recvbuf.reshape(n_local, 3)

    def _scatter_weights(self, weight_all):
        """Scatter a rank-0 1D weight array into rank-local chunks."""
        comm = self.comm
        rank = self.rank
        size = comm.Get_size()
        if rank == 0:
            n_all = weight_all.shape[0]
            counts = np.full(size, n_all // size, dtype=np.int64)
            counts[: (n_all % size)] += 1
            displs = np.zeros(size, dtype=np.int64)
            displs[1:] = np.cumsum(counts[:-1])
            sendbuf = np.ascontiguousarray(weight_all, dtype=np.float64)
        else:
            counts = sendbuf = displs = None
        n_local = int(comm.scatter(counts, root=0))
        recvbuf = np.empty(n_local, dtype=np.float64)
        comm.Scatterv([sendbuf, counts, displs, MPI.DOUBLE], recvbuf, root=0)
        return recvbuf

    ### Low-order pair and random-product shortcuts ###

    def calc_pair_product(self, radius, field1, field2):
        """Compute a pair product, using density shortcuts whenever one leg is uniform."""
        if isinstance(field1, (float, int, np.floating)) or isinstance(field2, (float, int, np.floating)):
            return self._field_density(field1) * self._field_density(field2)
        return compute_pair_product_at_sample({"s": radius}, field1, field2)

    def _compute_rrr_value(
        self, mu, r23_value, center, pos_local, seed_base_rot, mu_index,
        random1, random2, random3, rr23_cache=None, center_weight=None, center_weight_sum=None
    ):
        """Compute <R1 R2 R3>, reducing to lower-order shortcuts whenever possible."""
        n_uniform = sum(isinstance(x, (float, int, np.floating)) for x in [random1, random2, random3])
        if n_uniform >= 2:
            return self._field_density(random1) * self._field_density(random2) * self._field_density(random3)
        if isinstance(random1, (float, int, np.floating)):
            if rr23_cache is None:
                rr23_cache = self.calc_pair_product(r23_value, random2, random3)
            return self._field_density(random1) * rr23_cache
        if isinstance(random2, (float, int, np.floating)):
            rr13 = self.calc_pair_product(self.r13, random1, random3)
            return self._field_density(random2) * rr13
        if isinstance(random3, (float, int, np.floating)):
            rr12 = self.calc_pair_product(self.r12, random1, random2)
            return self._field_density(random3) * rr12
        return estimate_triplet_product_with_sampled_centers(
            self.r12_scaled,
            self.r13_scaled,
            mu,
            pos_local,
            self.n_rot,
            self.meta_convols,
            random2,
            random3,
            center=center,
            center_weight=center_weight,
            center_weight_sum=center_weight_sum,
            seed_base_rot=seed_base_rot,
            mu_index=mu_index,
            eps1=random1.epsilon,
            rho1=self._field_density(random1),
        )

    def _configure_particle_center_leg1(
        self, expanded_products, particle_pos1_arr, particle_weight1_arr,
        random1, random2, random3, random_pos1_arr, random_weight1_arr,
        data_legs, random_legs, window1
    ):
        """Resolve particle-center leg-1 state for data centers, random centers, and warnings."""
        leg1_base = next((base for i, base, _, _ in data_legs if i == 1), None)
        random1_base = next((base for i, base, _, _ in random_legs if i == 1), None)

        if particle_pos1_arr is not None and (expanded_products & {"xi12", "xi13", "zeta_H", "Q"}) and leg1_base is None:
            self.logger.error(
                "particle_pos1 can replace convols_data1 only for particle-center products that do not require "
                "xi12/xi13/zeta_H/Q. Please provide convols_data1 as well if those products are requested."
            )
            func_util.safe_exit(1)
        if particle_pos1_arr is not None and leg1_base is not None:
            self.logger.warning(
                "particle_pos1 is provided for center='particle'; it will override leg-1 data centers only, "
                "while convols_data1 will still be used as the leg-1 signal field."
            )
        if random_pos1_arr is not None and random1_base is not None and "r_delta_dd" in expanded_products:
            self.logger.warning(
                "random_pos1 is provided for center='particle'; it will override leg-1 random centers only, "
                "while random1 will still be used as the leg-1 random field."
            )
        if random_pos1_arr is not None and random1_base is None:
            for idx, (i, base, src, win) in enumerate(random_legs):
                if i == 1:
                    random_legs[idx] = (i, "uniform", "uniform random density", win)
                    random1_base = "uniform"
                    break

        particle_pos1, particle_weight1, particle_pos1_source = (
            self._resolve_pos1_array(
                particle_pos1_arr, particle_weight1_arr, leg1_base, "convols_data1", "particle_pos1"
            ) if (expanded_products & {"ddd", "d_delta_dd"}) else (None, None, None)
        )
        random_pos1, random_weight1, random_pos1_source = (
            self._resolve_pos1_array(
                random_pos1_arr,
                random_weight1_arr,
                random1_base if isinstance(random1_base, ConvolsData) else None,
                "random1",
                "random_pos1",
            ) if "r_delta_dd" in expanded_products else (None, None, None)
        )

        if window1 is not None:
            self.logger.warning("window1 has no effect for center='particle'; leg 1 uses particle centers directly.")
            data_legs = [(i, base, src, None if i == 1 else win) for i, base, src, win in data_legs]
            random_legs = [(i, base, src, None if i == 1 else win) for i, base, src, win in random_legs]

        return {
            "data_legs": data_legs,
            "random_legs": random_legs,
            "particle_pos1": particle_pos1,
            "particle_weight1": particle_weight1,
            "particle_pos1_source": particle_pos1_source,
            "random_pos1": random_pos1,
            "random_weight1": random_weight1,
            "random_pos1_source": random_pos1_source,
        }

    ### High-level preparation pipeline ###

    def prepare_input_fields(
        self,
        convols_data1=None,
        convols_data2=None,
        convols_data3=None,
        particle_pos1=None,
        particle_weight1=None,
        random1=None,
        random2=None,
        random3=None,
        random_pos1=None,
        random_weight1=None,
        window1=None,
        window2=None,
        window3=None,
    ):
        """Prepare all fields, random inputs, and center-position side inputs before run()."""
        self.corr3pcf_data = Corr3PCFData()
        self._sync_runtime_options()
        self._resolve_angle_sampling()
        (
            convols_data1, convols_data2, convols_data3,
            particle_pos1, particle_weight1, random1, random2, random3, random_pos1, random_weight1,
            window1, window2, window3,
        ) = self._resolve_runtime_inputs(
            convols_data1, convols_data2, convols_data3,
            particle_pos1, particle_weight1, random1, random2, random3, random_pos1, random_weight1,
            window1, window2, window3,
        )

        needs_data, needs_random = self._required_input_flags()
        expanded_products = set(self._expanded_products())
        particle_pos1_arr = self._normalize_particle_data(particle_pos1)
        particle_weight1_arr = None if particle_weight1 is None else np.asarray(particle_weight1, dtype=np.float64)
        random_pos1_arr = self._normalize_particle_data(random_pos1)
        random_weight1_arr = None if random_weight1 is None else np.asarray(random_weight1, dtype=np.float64)
        use_particle_pos1 = self.center == "particle" and particle_pos1_arr is not None
        use_random_pos1 = self.center == "particle" and random_pos1_arr is not None
        missing_random_inputs = (
            needs_random
            and all(x in (None, "") for x in [random1, random2, random3])
            and not use_random_pos1
        )
        missing_random_inputs = self.comm.bcast(
            missing_random_inputs if self.rank == 0 else None,
            root=0,
        )
        if missing_random_inputs:
            self.logger.error(
                "Random-related products were requested, but no random inputs were provided. "
                "Please set shared 'random', leg-specific 'random1/2/3', or 'random_pos1' when appropriate."
            )
            func_util.safe_exit(1)
        requires_signal_leg1 = needs_data and (
            not use_particle_pos1 or bool(expanded_products & {"xi12", "xi13", "zeta_H", "Q"})
        )

        if self.rank == 0:
            self.logger.info("Preparing Corr_3PCF input fields ...")
            self.logger.info(
                f"center={self.center}, r12={self.r12}, r13={self.r13}, n_rot={self.n_rot}, base_seed={self.base_seed}, threads={self.threads}"
            )
            if self.angle_param == "theta":
                self.logger.info(f"angle_param=theta, theta_min={self.theta_min}, theta_max={self.theta_max}, n_theta={self.n_theta}")
            else:
                self.logger.info(f"angle_param=mu, mu_min={self.mu_min}, mu_max={self.mu_max}, n_mu={self.n_mu}")
            self.logger.info(f"requested_products={self.products}, expanded_products={self._expanded_products()}")
            cache = {}
            data_legs = []
            if needs_data:
                for i, cdata, win in zip([1, 2, 3], [convols_data1, convols_data2, convols_data3], [window1, window2, window3]):
                    if i == 1 and not requires_signal_leg1:
                        continue
                    base_convols, source_desc = self._resolve_base_convols(i, cdata, cache)
                    data_legs.append((i, base_convols, source_desc, win))

            random_legs = []
            if needs_random:
                for i, rdata, win in zip([1, 2, 3], [random1, random2, random3], [window1, window2, window3]):
                    base_random, source_desc = self._resolve_random_base(i, rdata, cache)
                    random_legs.append((i, base_random, source_desc, win))

            if self.center == "particle":
                leg1_ctx = self._configure_particle_center_leg1(
                    expanded_products, particle_pos1_arr, particle_weight1_arr,
                    random1, random2, random3, random_pos1_arr, random_weight1_arr,
                    data_legs, random_legs, window1
                )
                data_legs = leg1_ctx["data_legs"]
                random_legs = leg1_ctx["random_legs"]
                self.particle_pos1 = leg1_ctx["particle_pos1"]
                self.particle_weight1 = leg1_ctx["particle_weight1"]
                self.random_pos1 = leg1_ctx["random_pos1"]
                self.random_weight1 = leg1_ctx["random_weight1"]
                particle_pos1_source = leg1_ctx["particle_pos1_source"]
                random_pos1_source = leg1_ctx["random_pos1_source"]
            else:
                particle_pos1_source = None
                random_pos1_source = None
                self.random_weight1 = None

            # All ConvolsData inputs, including random legs, must agree on the
            # same geometry and wavelet metadata before any mixed statistics are valid.
            compat_fields = [item[1] for item in data_legs if isinstance(item[1], ConvolsData)]
            compat_fields.extend(item[1] for item in random_legs if isinstance(item[1], ConvolsData))
            if compat_fields:
                shared_required = func_util.validate_convols_compatibility(
                    compat_fields,
                    ConvolsData._REQUIRED_ARGV,
                    logger=self.logger,
                    label="Corr_3PCF input fields",
                )
                self.rho = 1.0 / compat_fields[0].V
                shared_required_text = ", ".join([f"{k}={v}" for k, v in shared_required.items()])
                self.logger.info("Corr_3PCF input compatibility check passed.")
                self.logger.info(f"Shared required parameters | {shared_required_text}")
                self.logger.info(f"Shared density | rho={self.rho:.5e}")
            else:
                self.rho = None

            if self.particle_pos1 is not None:
                weight_sum = float(np.sum(self.particle_weight1))
                self.logger.info(
                    f"Particle leg 1 ready | source={particle_pos1_source} | "
                    f"particle_count={self.particle_pos1.shape[0]} | weight_sum={weight_sum:.6e}"
                )
            elif self.center != "particle":
                self.particle_pos1 = None

            self._prepare_signal_legs(data_legs)

            if not requires_signal_leg1:
                self.convols_data1 = None
                self.corr3pcf_data.convols_info1 = None

            if self.random_pos1 is not None:
                weight_sum = float(np.sum(self.random_weight1))
                self.logger.info(
                    f"Random leg 1 centers ready | source={random_pos1_source} | "
                    f"particle_count={self.random_pos1.shape[0]} | weight_sum={weight_sum:.6e}"
                )
            elif self.center != "particle":
                self.random_pos1 = None
                self.random_weight1 = None

            self._prepare_random_legs(data_legs, random_legs)

            self.window1 = window1
            self.window2 = window2
            self.window3 = window3
            snapshot = self._current_task_params_snapshot()
            self.corr3pcf_data.corr3pcf_info = snapshot
            self.corr3pcf_data.task_params = snapshot
        self._fields_prepared = True

    ### 2PCF-derived caches and leg finalization ###

    def _compute_pair_stats(self, field_a, field_b, random_a, random_b, radius):
        """Compute RR, delta_DD, and xi for one pair radius."""
        rr = self.calc_pair_product(radius, random_a, random_b)
        delta_a = self._compute_delta_field(field_a, random_a)
        delta_b = self._compute_delta_field(field_b, random_b)
        delta_dd = self.calc_pair_product(radius, delta_a, delta_b)
        xi = delta_dd / rr
        return {"rr": rr, "delta_dd": delta_dd, "xi": xi}

    def _compute_pair_stats_series(self, field_a, field_b, random_a, random_b, radii):
        """Vectorized convenience wrapper over _compute_pair_stats for radius arrays."""
        radii_arr = np.atleast_1d(np.asarray(radii, dtype=np.float64))
        rr = np.empty_like(radii_arr)
        delta_dd = np.empty_like(radii_arr)
        xi = np.empty_like(radii_arr)
        delta_a = self._compute_delta_field(field_a, random_a)
        delta_b = self._compute_delta_field(field_b, random_b)
        for idx, radius in enumerate(radii_arr):
            rr_val = self.calc_pair_product(float(radius), random_a, random_b)
            delta_val = self.calc_pair_product(float(radius), delta_a, delta_b)
            rr[idx] = rr_val
            delta_dd[idx] = delta_val
            xi[idx] = delta_val / rr_val
        if np.ndim(radii) == 0:
            return {"rr": float(rr[0]), "delta_dd": float(delta_dd[0]), "xi": float(xi[0])}
        return {"rr": rr, "delta_dd": delta_dd, "xi": xi}

    def _prepare_signal_legs(self, data_legs):
        """Apply optional smoothing windows and finalize the signal-leg fields."""
        for i, base_convols, source_desc, win in data_legs:
            window_obj, window_desc = self._resolve_window(i, base_convols, win)
            if window_obj is not None:
                final_convols = base_convols @ window_obj
            else:
                final_convols = base_convols.copy()
                final_convols.format_convols_params()
            final_convols = final_convols._normalize_for_estimator_inplace(self.field_normalization)
            setattr(self, f"convols_data{i}", final_convols)
            setattr(self.corr3pcf_data, f"convols_info{i}", final_convols.convols_info)
            self.logger.info(f"Field leg {i} ready | source={source_desc} | window={window_desc}")

    def _prepare_random_legs(self, data_legs, random_legs):
        """Apply random-leg smoothing and inject uniform-density shortcuts when requested."""
        for i, base_random, source_desc, win in random_legs:
            if (
                self.center == "particle"
                and i == 1
                and self.random_pos1 is not None
                and isinstance(getattr(self, "random1", None), (float, int, np.floating))
            ):
                signal_ref = self._find_geometry_reference(
                    getattr(self, f"convols_data{i}", None),
                    *[item[1] for item in data_legs if isinstance(item[1], ConvolsData)],
                    *[item[1] for item in random_legs if isinstance(item[1], ConvolsData)],
                )
                if signal_ref is None:
                    self.logger.error(
                        "Cannot resolve the geometry for particle-center random leg 1. "
                        "Please provide at least one ConvolsData input field."
                    )
                    func_util.safe_exit(1)
                rho = self._shared_density() if self.rho is not None else (1.0 / signal_ref.V)
                setattr(self, "random1", rho)
                continue
            if base_random == "uniform":
                signal_ref = self._find_geometry_reference(
                    getattr(self, f"convols_data{i}", None),
                    *[item[1] for item in data_legs if isinstance(item[1], ConvolsData)],
                    *[item[1] for item in random_legs if isinstance(item[1], ConvolsData)],
                )
                if signal_ref is None:
                    self.logger.error(
                        f"Cannot resolve the geometry for uniform random leg {i}. "
                        f"Please provide at least one ConvolsData input field or an explicit random field."
                    )
                    func_util.safe_exit(1)
                if self.field_normalization == "none":
                    integral = getattr(signal_ref, "field_integral", None)
                    if integral is None or not np.isfinite(integral) or not np.isclose(integral, 1.0):
                        raise ValueError(
                            "random='uniform' requires a unit-integral signal field. "
                            "Ordinary count fields already satisfy this; for a positive marked field "
                            "set field_normalization='mean'."
                        )
                rho = self._shared_density() if self.rho is not None else (1.0 / signal_ref.V)
                setattr(self, f"random{i}", rho)
                self.logger.info(f"Random leg {i} ready | source={source_desc} | window=uniform shortcut | rho={rho:.5e}")
            else:
                window_obj, window_desc = self._resolve_window(i, base_random, win)
                if window_obj is not None:
                    final_random = base_random @ window_obj
                else:
                    final_random = base_random.copy()
                    final_random.format_convols_params()
                final_random = final_random._normalize_for_estimator_inplace(self.field_normalization)
                setattr(self, f"random{i}", final_random)
                self.logger.info(f"Random leg {i} ready | source={source_desc} | window={window_desc}")

    ### Theta-local triplet kernels ###

    def _compute_random_center_mu(
        self, mu, r23_value, pos_local, seed_base_rot, mu_index,
        local_results, _local_convols1, _local_convols2, _local_convols3,
        _local_random1, _local_random2, _local_random3
    ):
        """Compute mu-local products for the box-random center mode."""
        if "ddd" in local_results:
            local_results["ddd"][mu_index] = estimate_triplet_product_with_sampled_centers(
                self.r12_scaled, self.r13_scaled, mu, pos_local, self.n_rot,
                _local_convols1, _local_convols2, _local_convols3,
                center="box_random", seed_base_rot=seed_base_rot, mu_index=mu_index,
                eps1=_local_convols1.epsilon,
            )
        if "delta_ddd" in local_results:
            field1 = _local_convols1 - self._field_density(_local_random1) if isinstance(_local_random1, (float, int, np.floating)) else _local_convols1 - _local_random1
            field2 = _local_convols2 - self._field_density(_local_random2) if isinstance(_local_random2, (float, int, np.floating)) else _local_convols2 - _local_random2
            field3 = _local_convols3 - self._field_density(_local_random3) if isinstance(_local_random3, (float, int, np.floating)) else _local_convols3 - _local_random3
            local_results["delta_ddd"][mu_index] = estimate_triplet_product_with_sampled_centers(
                self.r12_scaled, self.r13_scaled, mu, pos_local, self.n_rot,
                _local_convols1, field2, field3,
                center="box_random", seed_base_rot=seed_base_rot, mu_index=mu_index,
                eps1=field1.epsilon,
            )
        if "rrr" in local_results:
            local_results["rrr"][mu_index] = self._compute_rrr_value(
                mu, r23_value, "box_random", pos_local, seed_base_rot, mu_index,
                _local_random1, _local_random2, _local_random3, rr23_cache=None
            )

    def _compute_particle_center_mu(
        self, mu, r23_value, pos_local_data, weight_local_data, weight_sum_local_data,
        pos_local_random1, weight_local_random1, weight_sum_local_random1, seed_base_rot, mu_index,
        local_results, _local_convols2, _local_convols3,
        _local_random1, _local_random2, _local_random3
    ):
        """Compute mu-local products for the particle-center mode."""
        rho = self._shared_density()
        if "ddd" in local_results:
            local_results["ddd"][mu_index] = estimate_triplet_product_with_sampled_centers(
                self.r12_scaled, self.r13_scaled, mu, pos_local_data, self.n_rot,
                self.meta_convols, _local_convols2, _local_convols3,
                center="particle", center_weight=weight_local_data, center_weight_sum=weight_sum_local_data,
                seed_base_rot=seed_base_rot, mu_index=mu_index,
                rho1=rho,
            )
        if "rrr" in local_results:
            local_results["rrr"][mu_index] = self._compute_rrr_value(
                mu, r23_value, "particle", pos_local_random1, seed_base_rot, mu_index,
                _local_random1, _local_random2, _local_random3, rr23_cache=None,
                center_weight=weight_local_random1, center_weight_sum=weight_sum_local_random1,
            )
        if "d_delta_dd" in local_results:
            field2, field3 = self._particle_delta_fields
            local_results["d_delta_dd"][mu_index] = estimate_triplet_product_with_sampled_centers(
                self.r12_scaled, self.r13_scaled, mu, pos_local_data, self.n_rot,
                self.meta_convols, field2, field3,
                center="particle", center_weight=weight_local_data, center_weight_sum=weight_sum_local_data,
                seed_base_rot=seed_base_rot, mu_index=mu_index,
                rho1=rho,
            )
        if "r_delta_dd" in local_results:
            if pos_local_random1 is None and isinstance(_local_random1, (float, int, np.floating)):
                # The uniform-random leg-1 shortcut is cheaper to assemble from
                # rrr * xi23 after the loop than to sample explicitly here.
                pass
            else:
                field2, field3 = self._particle_delta_fields
                local_results["r_delta_dd"][mu_index] = estimate_triplet_product_with_sampled_centers(
                    self.r12_scaled, self.r13_scaled, mu, pos_local_random1, self.n_rot,
                    self.meta_convols, field2, field3,
                    center="particle", center_weight=weight_local_random1, center_weight_sum=weight_sum_local_random1,
                    seed_base_rot=seed_base_rot, mu_index=mu_index,
                    rho1=rho,
                )

    ### Post-loop pair cache computation ###

    def _compute_pair_cache(
        self, expanded_products, mu_arr,
        _local_convols1, _local_convols2, _local_convols3,
        _local_random1, _local_random2, _local_random3
    ):
        """Compute and cache all requested 2PCF-derived quantities after the main loop."""
        pair_cache = {}
        rr23_cache = None
        timing = {"xi12": 0.0, "xi13": 0.0, "xi23": 0.0}
        if "xi12" in expanded_products:
            t_pair = time.perf_counter()
            pair_cache["xi12"] = self._compute_pair_stats(
                _local_convols1, _local_convols2, _local_random1, _local_random2, self.r12
            )
            timing["xi12"] = time.perf_counter() - t_pair
        if "xi13" in expanded_products:
            t_pair = time.perf_counter()
            pair_cache["xi13"] = self._compute_pair_stats(
                _local_convols1, _local_convols3, _local_random1, _local_random3, self.r13
            )
            timing["xi13"] = time.perf_counter() - t_pair
        if "xi23" in expanded_products or "rrr" in expanded_products:
            t_pair = time.perf_counter()
            pair_cache["xi23"] = self._compute_pair_stats_series(
                _local_convols2,
                _local_convols3,
                _local_random2,
                _local_random3,
                third_side_from_mu(self.r12, self.r13, mu_arr),
            )
            rr23_cache = pair_cache["xi23"]["rr"]
            timing["xi23"] = time.perf_counter() - t_pair
        return pair_cache, rr23_cache, timing

    def _root_geometry_reference(self):
        """Find a prepared rank-0 field that can define geometry and scaling."""
        return self._find_geometry_reference(
            getattr(self, "convols_data1", None),
            getattr(self, "convols_data2", None),
            getattr(self, "convols_data3", None),
            getattr(self, "random1", None),
            getattr(self, "random2", None),
            getattr(self, "random3", None),
        )

    def _main_loop_products(self, expanded_products, defer_rrr_to_rr23):
        """Return products that need explicit theta-loop evaluation."""
        loop_products = [
            key for key in ["ddd", "delta_ddd", "d_delta_dd", "r_delta_dd", "rrr"]
            if key in expanded_products
        ]
        if defer_rrr_to_rr23 and "rrr" in loop_products:
            loop_products.remove("rrr")
        if self.center == "particle" and "delta_ddd" in loop_products:
            loop_products.remove("delta_ddd")
        if (
            self.center == "particle"
            and self.random_pos1 is None
            and isinstance(getattr(self, "random1", None), (float, int, np.floating))
            and "r_delta_dd" in loop_products
        ):
            loop_products.remove("r_delta_dd")
        return loop_products

    def _product_field_requirements(self, product):
        """Return data/random legs needed by one explicit theta-loop product."""
        if self.center == "box_random":
            if product == "ddd":
                return {1, 2, 3}, set()
            if product == "delta_ddd":
                return {1, 2, 3}, {1, 2, 3}
            if product == "rrr":
                return set(), {1, 2, 3}
        else:
            if product == "ddd":
                return {2, 3}, set()
            if product == "d_delta_dd":
                return {2, 3}, {2, 3}
            if product == "r_delta_dd":
                return {2, 3}, {2, 3}
            if product == "rrr":
                return set(), {1, 2, 3}
        raise ValueError(f"Unsupported explicit 3PCF product '{product}' for center='{self.center}'.")

    def _broadcast_product_runtime(self, product):
        """Broadcast only the fields needed by one product."""
        data_legs, random_legs = self._product_field_requirements(product)
        local_data = {}
        local_random = {}
        for idx in sorted(data_legs):
            local_data[idx] = self._broadcast_field(getattr(self, f"convols_data{idx}"))
        for idx in sorted(random_legs):
            local_random[idx] = self._broadcast_field(getattr(self, f"random{idx}"))
        return local_data, local_random

    def _release_product_runtime(self):
        """Drop product-local cached fields held on the task instance."""
        self.meta_convols = None
        self._particle_delta_fields = None

    def _product_center_summary(
        self,
        product,
        npos_total,
        npos_local,
        npos_total_random1,
        npos_local_random1,
        weight_sum_total,
        weight_sum_total_random1,
    ):
        """Describe the center sample used by one explicit theta-loop product."""
        if self.center == "box_random":
            return {
                "source": "box_random",
                "total": npos_total,
                "local": npos_local,
                "weight_sum": None,
            }
        if product in {"rrr", "r_delta_dd"}:
            return {
                "source": "random1",
                "total": npos_total_random1,
                "local": npos_local_random1,
                "weight_sum": weight_sum_total_random1,
            }
        return {
            "source": "particle1",
            "total": npos_total,
            "local": npos_local,
            "weight_sum": weight_sum_total,
        }

    def _product_field_summary(self, product):
        """Describe the second and third fields sampled by one product."""
        if self.center == "box_random":
            if product == "delta_ddd":
                return "delta2=D2-R2", "delta3=D3-R3"
            if product == "rrr":
                return "random2=R2", "random3=R3"
            return "data2=D2", "data3=D3"
        if product in {"d_delta_dd", "r_delta_dd"}:
            return "delta2=D2-R2", "delta3=D3-R3"
        if product == "rrr":
            return "random2=R2", "random3=R3"
        return "data2=D2", "data3=D3"

    ### Full estimator execution ###

    def run(self, save_result=True, overwrite=False):
        """Execute the full 3PCF workflow, including center generation and post-processing."""
        try:
            comm = self.comm
            rank = self.rank
            size = comm.Get_size()

            if rank == 0:
                t0 = time.perf_counter()

            if not self._fields_prepared:
                self.prepare_input_fields()

            self.rho = comm.bcast(self.rho, root=0)

            expanded_products = self._expanded_products()
            if rank == 0:
                defer_rrr_to_rr23 = (
                    "rrr" in expanded_products
                    and "xi23" in expanded_products
                    and isinstance(self.random1, (float, int, np.floating))
                    and self.random_pos1 is None
                )
                loop_products = self._main_loop_products(expanded_products, defer_rrr_to_rr23)
                has_random_pos1 = self.random_pos1 is not None
            else:
                defer_rrr_to_rr23 = None
                loop_products = None
                has_random_pos1 = None
            defer_rrr_to_rr23 = comm.bcast(defer_rrr_to_rr23, root=0)
            loop_products = comm.bcast(loop_products, root=0)
            has_random_pos1 = comm.bcast(has_random_pos1, root=0)

            snapshot = self._current_task_params_snapshot()
            self.corr3pcf_data.corr3pcf_info = snapshot
            self.corr3pcf_data.task_params = snapshot

            if rank == 0:
                geometry_ref = self._root_geometry_reference()
                if geometry_ref is None:
                    self.logger.error("At least one ConvolsData input is required to define geometry for Corr_3PCF.")
                    func_util.safe_exit(1)
                if self.center == "particle":
                    if self.particle_pos1 is not None:
                        pos_all = self.particle_pos1 * geometry_ref.scale_factor
                        weight_all = self.particle_weight1
                    else:
                        if self.convols_data1 is None:
                            self.logger.error("particle centers require particle_pos1 or a usable convols_data1 source.")
                            func_util.safe_exit(1)
                        particle_data = self.convols_data1.get_particle_data()
                        pos_all = particle_data["pos"] * self.convols_data1.scale_factor
                        weight_all = self._center_projection_weight(self.convols_data1, particle_data)
                    if has_random_pos1:
                        pos_all_random1 = self.random_pos1 * geometry_ref.scale_factor
                        weight_all_random1 = self.random_weight1
                    else:
                        pos_all_random1 = None
                        weight_all_random1 = None
                else:
                    pos_all = None
                    weight_all = None
                    pos_all_random1 = None
                    weight_all_random1 = None
            else:
                pos_all = None
                weight_all = None
                pos_all_random1 = None
                weight_all_random1 = None
            mu_arr = self.mu_arr
            theta_arr = self.theta_arr
            geometry_L = comm.bcast(geometry_ref.L if rank == 0 else None, root=0)
            geometry_scale_factor = comm.bcast(geometry_ref.scale_factor if rank == 0 else None, root=0)
            self.r12_scaled = self.r12 * geometry_scale_factor
            self.r13_scaled = self.r13 * geometry_scale_factor

            if self.center == "box_random":
                if rank == 0:
                    counts = np.full(size, self.n_box_centers // size, dtype=np.int64)
                    counts[: (self.n_box_centers % size)] += 1
                else:
                    counts = None
                n_local = int(comm.scatter(counts, root=0))
                seed_center_rank = self.base_seed + 1000003 * (rank + 1)
                pos_local = random_box_positions(count=n_local, box_size=geometry_L, seed=seed_center_rank)
            else:
                pos_local = self._scatter_positions(pos_all)
                weight_local = self._scatter_weights(weight_all)
                if has_random_pos1:
                    pos_local_random1 = self._scatter_positions(pos_all_random1)
                    weight_local_random1 = self._scatter_weights(weight_all_random1)
                else:
                    pos_local_random1 = None
                    weight_local_random1 = None

            npos_local = pos_local.shape[0]
            npos_total = comm.allreduce(npos_local, op=MPI.SUM)
            if self.center == "particle":
                weight_sum_local = float(np.sum(weight_local))
                weight_sum_total = comm.allreduce(weight_sum_local, op=MPI.SUM)
                if weight_sum_total <= 0.0:
                    self.logger.error("Particle-center weights must have a positive global sum.")
                    func_util.safe_exit(1)
                if weight_local_random1 is not None:
                    weight_sum_local_random1 = float(np.sum(weight_local_random1))
                    weight_sum_total_random1 = comm.allreduce(weight_sum_local_random1, op=MPI.SUM)
                    if weight_sum_total_random1 <= 0.0:
                        self.logger.error("Random particle-center weights must have a positive global sum.")
                        func_util.safe_exit(1)
                    npos_local_random1 = pos_local_random1.shape[0]
                    npos_total_random1 = comm.allreduce(npos_local_random1, op=MPI.SUM)
                else:
                    weight_sum_local_random1 = weight_sum_local
                    weight_sum_total_random1 = weight_sum_total
                    npos_local_random1 = npos_local
                    npos_total_random1 = npos_total
            else:
                weight_local = None
                weight_sum_local = None
                weight_sum_total = None
                weight_local_random1 = None
                weight_sum_local_random1 = None
                weight_sum_total_random1 = None
                npos_local_random1 = None
                npos_total_random1 = None
            seed_base_rot = self.base_seed + 1

            if rank == 0:
                self.logger.info("Start to calculate 3PCF (pos-parallel) ...")
                t_start = time.perf_counter()
                self.logger.info(f"Pre-3PCF setup time: {t_start - t0:.4f} sec")
                self.logger.info(f"Main 3PCF loop products: {loop_products}")

            global_results = {}
            for product in loop_products:
                if rank == 0:
                    t_product_start = time.perf_counter()
                    center_summary = self._product_center_summary(
                        product,
                        npos_total,
                        npos_local,
                        npos_total_random1,
                        npos_local_random1,
                        weight_sum_total,
                        weight_sum_total_random1,
                    )
                    weight_text = ""
                    if center_summary["weight_sum"] is not None:
                        weight_text = f" | weight_sum={center_summary['weight_sum']:.6e}"
                    field2_text, field3_text = self._product_field_summary(product)
                    self.logger.info(
                        f"Computing product '{product}' | centers={center_summary['source']} | "
                        f"field2={field2_text} | field3={field3_text} | "
                        f"total_centers={center_summary['total']} | local_centers_rank0={center_summary['local']}"
                        f"{weight_text}"
                    )

                local_data, local_random = self._broadcast_product_runtime(product)
                self.meta_convols = self._find_geometry_reference(
                    *local_data.values(),
                    *local_random.values(),
                )
                if self.center == "particle" and product in {"d_delta_dd", "r_delta_dd"}:
                    delta_field2 = self._compute_delta_field(local_data[2], local_random[2])
                    delta_field3 = self._compute_delta_field(local_data[3], local_random[3])
                    self._particle_delta_fields = (delta_field2, delta_field3)
                    self.meta_convols = self._find_geometry_reference(delta_field2, delta_field3, self.meta_convols)
                elif self.center == "box_random" and product == "delta_ddd":
                    self.meta_convols = self._find_geometry_reference(local_data.get(1), local_data.get(2), local_data.get(3))
                elif self.meta_convols is None:
                    self.meta_convols = self._find_geometry_reference(
                        local_data.get(1), local_data.get(2), local_data.get(3),
                        local_random.get(1), local_random.get(2), local_random.get(3),
                    )

                if rank == 0:
                    self.logger.info(f"Product '{product}' setup time: {time.perf_counter() - t_product_start:.4f} sec")

                local_results = {product: np.zeros(mu_arr.shape[0], dtype=np.float64)}
                for it, mu in enumerate(mu_arr):
                    t_mu_start = time.perf_counter() if rank == 0 else None
                    r23_value = third_side_from_mu(self.r12, self.r13, mu)
                    if self.center == "box_random":
                        self._compute_random_center_mu(
                            mu, r23_value, pos_local, seed_base_rot, it,
                            local_results,
                            local_data.get(1), local_data.get(2), local_data.get(3),
                            local_random.get(1), local_random.get(2), local_random.get(3),
                        )
                    else:
                        self._compute_particle_center_mu(
                            mu, r23_value,
                            pos_local, weight_local, weight_sum_local,
                            pos_local_random1 if pos_local_random1 is not None else pos_local,
                            weight_local_random1 if weight_local_random1 is not None else weight_local,
                            weight_sum_local_random1,
                            seed_base_rot, it,
                            local_results,
                            local_data.get(2), local_data.get(3),
                            local_random.get(1), local_random.get(2), local_random.get(3),
                        )

                    if rank == 0:
                        elapsed_mu = time.perf_counter() - t_mu_start
                        self.logger.info(
                            f" product={product} | mu[{it + 1:02d}/{mu_arr.shape[0]}] done | "
                            f"mu={mu:.5f} | theta={theta_arr[it]:.5f} rad | elapsed={elapsed_mu:.2f} sec"
                        )

                arr = local_results[product]
                if self.center == "particle" and product in {"ddd", "d_delta_dd"}:
                    local_weighted = arr * weight_sum_local
                    normalizer = weight_sum_total
                elif self.center == "particle" and product in {"rrr", "r_delta_dd"}:
                    local_weighted = arr * weight_sum_local_random1
                    normalizer = weight_sum_total_random1
                else:
                    local_weighted = arr * npos_local
                    normalizer = npos_total
                global_weighted = np.empty_like(arr)
                comm.Allreduce(local_weighted, global_weighted, op=MPI.SUM)
                global_results[product] = global_weighted / normalizer
                if rank == 0:
                    self.logger.info(f"Product '{product}' finished in {time.perf_counter() - t_product_start:.4f} sec")
                del local_results, local_data, local_random
                self._release_product_runtime()

            if rank == 0:
                t_loop_end = time.perf_counter()
                self.logger.info(f"3PCF main loop time: {t_loop_end - t_start:.4f} sec")
                self.logger.info("Main 3PCF loop finished, post-processing on rank 0 ...")

                pair_cache, rr23_cache, pair_timing = self._compute_pair_cache(
                    expanded_products, mu_arr,
                    self.convols_data1, self.convols_data2, self.convols_data3,
                    self.random1, self.random2, self.random3
                )

                pair_timing_parts = []
                for key in ["xi12", "xi13", "xi23"]:
                    if key in expanded_products:
                        pair_timing_parts.append(f"{key}={pair_timing[key]:.2f} sec")
                if pair_timing_parts:
                    self.logger.info(f"2PCF timing | {' | '.join(pair_timing_parts)}")

                self.corr3pcf_data.mu = mu_arr
                self.corr3pcf_data.theta = theta_arr
                self.corr3pcf_data.r23 = third_side_from_mu(self.r12, self.r13, mu_arr)
                self.corr3pcf_data.ddd = global_results.get("ddd")
                self.corr3pcf_data.rrr = global_results.get("rrr")
                self.corr3pcf_data.d_delta_dd = global_results.get("d_delta_dd")
                self.corr3pcf_data.r_delta_dd = global_results.get("r_delta_dd")
                self.corr3pcf_data.delta_ddd = global_results.get("delta_ddd")

                self.corr3pcf_data.xi12 = pair_cache.get("xi12", {}).get("xi")
                self.corr3pcf_data.xi13 = pair_cache.get("xi13", {}).get("xi")
                self.corr3pcf_data.xi23 = pair_cache.get("xi23", {}).get("xi")

                if "rrr" in expanded_products and self.corr3pcf_data.rrr is None:
                    rho1 = self._field_density(self.random1)
                    if rr23_cache is not None and isinstance(self.random1, (float, int, np.floating)):
                        self.corr3pcf_data.rrr = rho1 * rr23_cache
                        self.logger.info("Computed rrr from pair cache and random1 density.")
                    else:
                        self.logger.error("rrr was requested but no loop result or pair-cache shortcut is available.")
                        func_util.safe_exit(1)

                if self.center == "particle" and "r_delta_dd" in expanded_products and self.corr3pcf_data.r_delta_dd is None:
                    if not has_random_pos1 and isinstance(self.random1, (float, int, np.floating)):
                        self.corr3pcf_data.r_delta_dd = self.corr3pcf_data.rrr * self.corr3pcf_data.xi23
                    self.logger.info("Computed r_delta_dd from xi23 and rrr for particle center with uniform random leg 1.")

                if self.center == "particle" and "delta_ddd" in expanded_products:
                    self.corr3pcf_data.delta_ddd = self.corr3pcf_data.d_delta_dd - self.corr3pcf_data.r_delta_dd
                    self.logger.info("Computed delta_ddd from d_delta_dd and r_delta_dd for particle center.")

                if "zeta" in expanded_products:
                    self.corr3pcf_data.zeta = self.corr3pcf_data.delta_ddd / self.corr3pcf_data.rrr
                    self.logger.info("Computed zeta from delta_ddd and rrr.")
                if "zeta_H" in expanded_products:
                    xi12 = self.corr3pcf_data.xi12
                    xi13 = self.corr3pcf_data.xi13
                    xi23 = self.corr3pcf_data.xi23
                    self.corr3pcf_data.zeta_H = xi12 * xi13 + xi12 * xi23 + xi13 * xi23
                    self.logger.info("Computed zeta_H from xi12/xi13/xi23.")
                if "Q" in expanded_products:
                    self.corr3pcf_data.Q = self.corr3pcf_data.zeta / self.corr3pcf_data.zeta_H
                    self.logger.info("Computed Q from zeta and zeta_H.")

                t_end = time.perf_counter()
                self.logger.info(f"The time for 3PCF: {t_end - t_start:.4f} sec")
                if save_result and self.fout_path:
                    self.logger.info("Saving 3PCF result to output file ...")
                    self.corr3pcf_data.saveflag = True
                    self.corr3pcf_data.save_corr3pcf(self.fout_path, overwrite=overwrite)

            comm.Barrier()
        except Exception as e:
            self.logger.error(f"Error in process {self.rank}: {str(e)}")
            func_util.safe_exit(1)

        if self.rank == 0:
            t1 = time.perf_counter()
            print("")
            self.logger.info(f"The time for task: {t1 - t0:.4f} sec")
        return self.corr3pcf_data
