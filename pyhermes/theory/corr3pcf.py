import time
import pickle
import copy
import numpy as np

from pyhermes.io import WindowFunc, ConvolsData, Corr3PCFData
from pyhermes.utils import func_util, math_util
from pyhermes.utils.mpi_util import MPI
from pyhermes.pipeline import TaskBase

from .corr2pcf import calc_DD_mean_r


def calc_DDD_mean_mc(
    r12_scaled, r13_scaled, theta, pos_scaled, n_rot,
    convols_meta, convols_data2, convols_data3,
    center="random",
    seed_base_rot=-1,
    theta_index=-1,
    eps1=None,
    rho1=None,
):
    kwargs_common = {
        "phi_data": convols_meta.phi_data,
        "L": convols_meta.L,
        "SampRate": convols_meta.SampRate,
        "PhiSupport": convols_meta.PhiSupport,
        "seed_base_rot": seed_base_rot,
        "theta_index": theta_index,
    }

    eps2 = convols_data2.epsilon
    eps3 = convols_data3.epsilon

    if center == "random":
        if eps1 is None:
            raise ValueError("eps1 must be provided when center='random'.")
        return math_util.calc_DDD_mc_random_center(
            r12_scaled, r13_scaled, theta,
            pos_scaled, n_rot,
            eps1, eps2, eps3,
            **kwargs_common,
        )

    if center == "particle":
        if rho1 is None:
            raise ValueError("rho1 must be provided when center='particle'.")
        return math_util.calc_DDD_mc_pos_center_fast(
            r12_scaled, r13_scaled, theta,
            pos_scaled, n_rot,
            rho1, eps2, eps3,
            **kwargs_common,
        )

    raise ValueError(f"Unknown center='{center}'. Use 'random' or 'particle'.")


class Corr_3PCF(TaskBase):

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
        self.sync_runtime_options(context="Corr_3PCF runtime configuration", blank_line=True)

    def format_params(self):
        self.convols_data = self.task_params.get("convols_data", "")
        self.convols_data1 = self.task_params.get("convols_data1", "") or self.convols_data
        self.convols_data2 = self.task_params.get("convols_data2", "") or self.convols_data
        self.convols_data3 = self.task_params.get("convols_data3", "") or self.convols_data
        self.particle_data1 = self.task_params.get("particle_data1", None)
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

        self.fout_path = self.task_params["fout_path"]
        self.threads = int(self.task_params["threads"])

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
        self.theta_min = 0.0
        self.theta_max = np.pi
        self.n_theta = int(self.task_params["n_theta"])
        self.n_rot = int(self.task_params["n_rot"])
        self.center = self.task_params["center"]
        self.n_rand = int(self.task_params["n_rand"])
        self.base_seed = int(self.task_params["base_seed"])
        self.products = self._normalize_products(self.task_params.get("products", "Q"))

    def _normalize_products(self, products):
        if isinstance(products, str):
            products = [products]
        elif products is None:
            products = ["Q"]
        elif not isinstance(products, (list, tuple, set)):
            raise TypeError(
                f"Unsupported products input: expected string or array of strings, got {type(products)}."
            )

        allowed_random = {"ddd", "rrr", "delta_ddd", "xi12", "xi13", "xi23", "zeta", "Q"}
        allowed_particle = {"ddd", "rrr", "d_delta_dd", "delta_ddd", "xi12", "xi13", "xi23", "zeta", "Q"}
        allowed = allowed_random if self.center == "random" else allowed_particle

        normalized = []
        for item in products:
            if not isinstance(item, str):
                raise TypeError("Each product name must be a string.")
            raw_name = item.strip()
            name = "Q" if raw_name.upper() == "Q" else raw_name.lower()
            if name not in allowed:
                raise ValueError(f"Unsupported product '{item}' for center='{self.center}'. Allowed values are {sorted(allowed)}.")
            if name not in normalized:
                normalized.append(name)
        return normalized

    def _expanded_products(self):
        expanded = list(self.products)
        if self.center == "random":
            if "Q" in expanded:
                for dep in ["zeta", "xi12", "xi13", "xi23"]:
                    if dep not in expanded:
                        expanded.append(dep)
            if "zeta" in expanded:
                for dep in ["delta_ddd", "rrr"]:
                    if dep not in expanded:
                        expanded.append(dep)
        else:
            if "Q" in expanded:
                for dep in ["zeta", "xi12", "xi13", "xi23"]:
                    if dep not in expanded:
                        expanded.append(dep)
            if "zeta" in expanded:
                for dep in ["delta_ddd", "rrr"]:
                    if dep not in expanded:
                        expanded.append(dep)
            if "delta_ddd" in expanded:
                for dep in ["d_delta_dd", "xi23", "rrr"]:
                    if dep not in expanded:
                        expanded.append(dep)
        return expanded

    def _required_input_flags(self):
        expanded = set(self._expanded_products())
        needs_data = bool(expanded & {"ddd", "d_delta_dd", "delta_ddd", "xi12", "xi13", "xi23", "zeta", "Q"})
        needs_random = bool(expanded & {"rrr", "d_delta_dd", "delta_ddd", "xi12", "xi13", "xi23", "zeta", "Q"})
        return needs_data, needs_random

    def _normalize_random(self, value):
        if value in (None, ""):
            return None
        if value == "uniform":
            return "uniform"
        return value

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
                "SimBoxL": value.SimBoxL,
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
        params["convols_data"] = self._serialize_convols_input(self.convols_data)
        params["convols_data1"] = self._serialize_convols_input(self.convols_data1)
        params["convols_data2"] = self._serialize_convols_input(self.convols_data2)
        params["convols_data3"] = self._serialize_convols_input(self.convols_data3)
        if self.particle_data1 is None:
            params["particle_data1"] = None
        else:
            arr = np.asarray(self.particle_data1)
            params["particle_data1"] = {"kind": "particle_data1", "shape": tuple(arr.shape)}
        params["random"] = self._serialize_convols_input(self.random)
        params["random1"] = self._serialize_convols_input(self.random1)
        params["random2"] = self._serialize_convols_input(self.random2)
        params["random3"] = self._serialize_convols_input(self.random3)
        params["window"] = self._serialize_window_input(self.window)
        params["window1"] = self._serialize_window_input(self.window1)
        params["window2"] = self._serialize_window_input(self.window2)
        params["window3"] = self._serialize_window_input(self.window3)
        params["fout_path"] = self.fout_path
        params["threads"] = self.threads
        params["r12"] = self.r12
        params["r13"] = self.r13
        params["n_theta"] = self.n_theta
        params["n_rot"] = self.n_rot
        params["center"] = self.center
        params["n_rand"] = self.n_rand
        params["base_seed"] = self.base_seed
        params["products"] = copy.deepcopy(self.products)
        return params

    def _resolve_base_convols(self, leg_idx, provided_convols, cache):
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
        if isinstance(field, (float, int, np.floating)):
            return float(field)
        if isinstance(field, ConvolsData):
            return 1.0 / field.V
        raise TypeError(f"Unsupported field type for density extraction: {type(field)}")

    def _find_geometry_reference(self, *candidates):
        for candidate in candidates:
            if isinstance(candidate, ConvolsData):
                return candidate
        return None

    def _normalize_particle_data(self, value):
        if value is None:
            return None
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(
                f"particle_data1 must be an array-like with shape (N, 3), got shape {arr.shape}."
            )
        return np.ascontiguousarray(arr, dtype=np.float64)

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

    def calc_pair_product(self, radius, field1, field2):
        if isinstance(field1, (float, int, np.floating)) or isinstance(field2, (float, int, np.floating)):
            return self._field_density(field1) * self._field_density(field2)
        return calc_DD_mean_r(radius, field1, field2)

    def _compute_rrr_value(self, theta, r23_value, center, pos_local, seed_base_rot, theta_index, random1, random2, random3, rr23_cache=None):
        n_uniform = sum(isinstance(x, (float, int, np.floating)) for x in [random1, random2, random3])
        if n_uniform == 3:
            return self._field_density(random1) * self._field_density(random2) * self._field_density(random3)
        if n_uniform == 2:
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
        return calc_DDD_mean_mc(
            self.r12_scaled,
            self.r13_scaled,
            theta,
            pos_local,
            self.n_rot,
            self.meta_convols,
            random2,
            random3,
            center=center,
            seed_base_rot=seed_base_rot,
            theta_index=theta_index,
            eps1=random1.epsilon,
            rho1=self._field_density(random1),
        )

    def prepare_input_fields(
        self,
        convols_data1=None,
        convols_data2=None,
        convols_data3=None,
        particle_data1=None,
        random1=None,
        random2=None,
        random3=None,
        window1=None,
        window2=None,
        window3=None,
    ):
        self.corr3pcf_data = Corr3PCFData()
        self._sync_runtime_options()
        if convols_data1 is None:
            convols_data1 = self.convols_data1
        if convols_data2 is None:
            convols_data2 = self.convols_data2
        if convols_data3 is None:
            convols_data3 = self.convols_data3
        if particle_data1 is None:
            particle_data1 = self.particle_data1
        if random1 is None:
            random1 = self.random1
        if random2 is None:
            random2 = self.random2
        if random3 is None:
            random3 = self.random3
        if window1 is None:
            window1 = self.window1
        if window2 is None:
            window2 = self.window2
        if window3 is None:
            window3 = self.window3

        needs_data, needs_random = self._required_input_flags()
        expanded_products = set(self._expanded_products())
        particle_data1_arr = self._normalize_particle_data(particle_data1)
        use_particle_data1 = self.center == "particle" and particle_data1_arr is not None
        requires_signal_leg1 = needs_data and (
            not use_particle_data1 or bool(expanded_products & {"xi12", "xi13", "Q"})
        )

        if self.rank == 0:
            self.logger.info("Preparing Corr_3PCF input fields ...")
            self.logger.info(
                f"center={self.center}, n_theta={self.n_theta}, n_rot={self.n_rot}, r12={self.r12}, r13={self.r13}, threads={self.threads}"
            )
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
                if needs_random and random_legs and random_legs[0][1] != "uniform":
                    self.logger.error("For center='particle', random1 must be 'uniform'.")
                    func_util.safe_exit(1)
                if use_particle_data1 and (expanded_products & {"xi12", "xi13", "Q"}):
                    self.logger.error(
                        "particle_data1 can replace convols_data1 only for particle-center products that do not require "
                        "xi12/xi13/Q. Please provide convols_data1 as well if those products are requested."
                    )
                    func_util.safe_exit(1)
                if not use_particle_data1 and requires_signal_leg1:
                    leg1_base = next((base for i, base, _, _ in data_legs if i == 1), None)
                    try:
                        self.particle_data1 = self._normalize_particle_data(leg1_base.get_particle_data())
                    except Exception:
                        self.logger.error(
                            "For center='particle', convols_data1 could not provide usable particle coordinates. "
                            "Please provide particle_data1 explicitly."
                        )
                        func_util.safe_exit(1)
                if window1 is not None:
                    self.logger.warning("window1 has no effect for center='particle'; leg 1 uses particle centers directly.")
                    data_legs = [(i, base, src, None if i == 1 else win) for i, base, src, win in data_legs]
                    random_legs = [(i, base, src, None if i == 1 else win) for i, base, src, win in random_legs]

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

            for i, base_convols, source_desc, win in data_legs:
                window_obj, window_desc = self._resolve_window(i, base_convols, win)
                if window_obj is not None:
                    final_convols = base_convols @ window_obj
                else:
                    final_convols = base_convols.copy()
                    final_convols.format_convols_params()
                setattr(self, f"convols_data{i}", final_convols)
                setattr(self.corr3pcf_data, f"convols_info{i}", final_convols.convols_info)
                self.logger.info(f"Field leg {i} ready | source={source_desc} | window={window_desc}")

            if not requires_signal_leg1:
                self.convols_data1 = None
                self.corr3pcf_data.convols_info1 = None

            if use_particle_data1:
                self.particle_data1 = particle_data1_arr
                self.logger.info(f"Particle leg 1 ready | source=provided particle_data1 | N_particles={self.particle_data1.shape[0]}")
            elif self.center != "particle":
                self.particle_data1 = None

            for i, base_random, source_desc, win in random_legs:
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
                    rho = self.rho if self.rho is not None else (1.0 / signal_ref.V)
                    setattr(self, f"random{i}", rho)
                    self.logger.info(f"Random leg {i} ready | source={source_desc} | window=uniform shortcut | rho={rho:.5e}")
                else:
                    window_obj, window_desc = self._resolve_window(i, base_random, win)
                    if window_obj is not None:
                        final_random = base_random @ window_obj
                    else:
                        final_random = base_random.copy()
                        final_random.format_convols_params()
                    setattr(self, f"random{i}", final_random)
                    self.logger.info(f"Random leg {i} ready | source={source_desc} | window={window_desc}")

            self.window1 = window1
            self.window2 = window2
            self.window3 = window3
            self.corr3pcf_data.corr3pcf_info = self._current_task_params_snapshot()
            self.corr3pcf_data.task_params = self._current_task_params_snapshot()
        self._fields_prepared = True

    def _compute_pair_stats(self, field_a, field_b, random_a, random_b, radius):
        rr = self.calc_pair_product(radius, random_a, random_b)
        delta_a = field_a - self._field_density(random_a) if isinstance(random_a, (float, int, np.floating)) else field_a - random_a
        delta_b = field_b - self._field_density(random_b) if isinstance(random_b, (float, int, np.floating)) else field_b - random_b
        delta_dd = self.calc_pair_product(radius, delta_a, delta_b)
        xi = delta_dd / rr
        return {"rr": rr, "delta_dd": delta_dd, "xi": xi}

    def _compute_pair_stats_series(self, field_a, field_b, random_a, random_b, radii):
        radii_arr = np.atleast_1d(np.asarray(radii, dtype=np.float64))
        rr = np.empty_like(radii_arr)
        delta_dd = np.empty_like(radii_arr)
        xi = np.empty_like(radii_arr)
        for idx, radius in enumerate(radii_arr):
            stats = self._compute_pair_stats(field_a, field_b, random_a, random_b, float(radius))
            rr[idx] = stats["rr"]
            delta_dd[idx] = stats["delta_dd"]
            xi[idx] = stats["xi"]
        if np.ndim(radii) == 0:
            return {"rr": float(rr[0]), "delta_dd": float(delta_dd[0]), "xi": float(xi[0])}
        return {"rr": rr, "delta_dd": delta_dd, "xi": xi}

    def run(self, save_result=True, overwrite=False):
        try:
            comm = self.comm
            rank = self.rank
            size = comm.Get_size()

            if rank == 0:
                t0 = time.perf_counter()

            if not self._fields_prepared:
                self.prepare_input_fields()

            expanded_products = self._expanded_products()
            needs_data, needs_random = self._required_input_flags()
            needs_signal_leg1 = not (self.center == "particle" and self.convols_data1 is None)

            _local_convols1 = self._broadcast_field(self.convols_data1) if (needs_data and needs_signal_leg1) else None
            _local_convols2 = self._broadcast_field(self.convols_data2) if needs_data else None
            _local_convols3 = self._broadcast_field(self.convols_data3) if needs_data else None
            _local_random1 = self._broadcast_field(self.random1) if needs_random else None
            _local_random2 = self._broadcast_field(self.random2) if needs_random else None
            _local_random3 = self._broadcast_field(self.random3) if needs_random else None
            defer_rrr_to_rr23 = (
                "rrr" in expanded_products
                and "xi23" in expanded_products
                and isinstance(_local_random1, (float, int, np.floating))
            )

            self.corr3pcf_data.corr3pcf_info = self._current_task_params_snapshot()
            self.corr3pcf_data.task_params = self._current_task_params_snapshot()

            if rank == 0:
                if self.center == "particle":
                    geometry_ref = self._find_geometry_reference(_local_convols1, _local_convols2, _local_convols3)
                    if geometry_ref is None:
                        self.logger.error("At least one ConvolsData input is required to define geometry for center='particle'.")
                        func_util.safe_exit(1)
                    if self.particle_data1 is not None:
                        pos_all = self.particle_data1 * geometry_ref.ScaleFactor
                    else:
                        pos_all = _local_convols1.get_particle_data() * _local_convols1.ScaleFactor
                    Nall = pos_all.shape[0]
                theta_arr = np.linspace(self.theta_min, self.theta_max, self.n_theta)
            else:
                theta_arr = None
                pos_all = None
                Nall = None
            theta_arr = comm.bcast(theta_arr, root=0)

            if self.center == "random":
                if rank == 0:
                    counts = np.full(size, self.n_rand // size, dtype=np.int64)
                    counts[: (self.n_rand % size)] += 1
                else:
                    counts = None
                n_local = int(comm.scatter(counts, root=0))
                seed_center_rank = self.base_seed + 1000003 * (rank + 1)
                pos_local = math_util.random_points_box(N=n_local, SimBoxL=_local_convols1.L, seed=seed_center_rank)
            else:
                if rank == 0:
                    counts = np.full(size, Nall // size, dtype=np.int64)
                    counts[: (Nall % size)] += 1
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
                pos_local = recvbuf.reshape(n_local, 3)

            npos_local = pos_local.shape[0]
            npos_total = comm.allreduce(npos_local, op=MPI.SUM)
            geometry_ref = self._find_geometry_reference(_local_convols1, _local_convols2, _local_convols3)
            if geometry_ref is None:
                self.logger.error("At least one ConvolsData input is required to define geometry for Corr_3PCF.")
                func_util.safe_exit(1)
            self.meta_convols = geometry_ref
            self.r12_scaled = self.r12 * geometry_ref.ScaleFactor
            self.r13_scaled = self.r13 * geometry_ref.ScaleFactor
            seed_base_rot = self.base_seed + 1

            if rank == 0:
                self.logger.info("Start to calculate 3PCF (pos-parallel) ...")
                self.logger.info(f"total_centers={npos_total}, each rank has n_local={npos_local} centers")
                t_start = time.perf_counter()
                self.logger.info(f"Pre-3PCF setup time: {t_start - t0:.4f} sec")
                loop_products = [key for key in ["ddd", "delta_ddd", "d_delta_dd", "rrr"] if key in expanded_products]
                if defer_rrr_to_rr23 and "rrr" in loop_products:
                    loop_products.remove("rrr")
                if self.center == "particle" and "delta_ddd" in loop_products:
                    loop_products.remove("delta_ddd")
                self.logger.info(f"Main DDD loop products: {loop_products}")

            local_results = {
                key: np.zeros(theta_arr.shape[0], dtype=np.float64)
                for key in ["ddd", "delta_ddd", "d_delta_dd", "rrr"]
                if key in expanded_products
                and not (key == "rrr" and defer_rrr_to_rr23)
                and not (self.center == "particle" and key == "delta_ddd")
            }

            for it, th in enumerate(theta_arr):
                t_theta_start = time.perf_counter() if rank == 0 else None
                r23_value = math_util.third_side(self.r12, self.r13, th)
                if self.center == "random":
                    if "ddd" in expanded_products:
                        local_results["ddd"][it] = calc_DDD_mean_mc(
                            self.r12_scaled, self.r13_scaled, th, pos_local, self.n_rot,
                            _local_convols1, _local_convols2, _local_convols3,
                            center="random", seed_base_rot=seed_base_rot, theta_index=it,
                            eps1=_local_convols1.epsilon,
                        )
                    if "delta_ddd" in expanded_products:
                        field1 = _local_convols1 - self._field_density(_local_random1) if isinstance(_local_random1, (float, int, np.floating)) else _local_convols1 - _local_random1
                        field2 = _local_convols2 - self._field_density(_local_random2) if isinstance(_local_random2, (float, int, np.floating)) else _local_convols2 - _local_random2
                        field3 = _local_convols3 - self._field_density(_local_random3) if isinstance(_local_random3, (float, int, np.floating)) else _local_convols3 - _local_random3
                        local_results["delta_ddd"][it] = calc_DDD_mean_mc(
                            self.r12_scaled, self.r13_scaled, th, pos_local, self.n_rot,
                            _local_convols1, field2, field3,
                            center="random", seed_base_rot=seed_base_rot, theta_index=it,
                            eps1=field1.epsilon,
                        )
                    if "rrr" in local_results:
                        local_results["rrr"][it] = self._compute_rrr_value(
                            th, r23_value, "random", pos_local, seed_base_rot, it,
                            _local_random1, _local_random2, _local_random3, rr23_cache=None
                        )
                else:
                    rho = self.rho if self.rho is not None else (
                        self._field_density(_local_convols1) if _local_convols1 is not None else (1.0 / self.meta_convols.V)
                    )
                    if "ddd" in expanded_products:
                        local_results["ddd"][it] = calc_DDD_mean_mc(
                            self.r12_scaled, self.r13_scaled, th, pos_local, self.n_rot,
                            self.meta_convols, _local_convols2, _local_convols3,
                            center="particle", seed_base_rot=seed_base_rot, theta_index=it,
                            rho1=rho,
                        )
                    if "rrr" in local_results:
                        local_results["rrr"][it] = self._compute_rrr_value(
                            th, r23_value, "particle", pos_local, seed_base_rot, it,
                            _local_random1, _local_random2, _local_random3, rr23_cache=None
                        )
                    if "d_delta_dd" in expanded_products:
                        field2 = _local_convols2 - self._field_density(_local_random2) if isinstance(_local_random2, (float, int, np.floating)) else _local_convols2 - _local_random2
                        field3 = _local_convols3 - self._field_density(_local_random3) if isinstance(_local_random3, (float, int, np.floating)) else _local_convols3 - _local_random3
                        local_results["d_delta_dd"][it] = calc_DDD_mean_mc(
                            self.r12_scaled, self.r13_scaled, th, pos_local, self.n_rot,
                            self.meta_convols, field2, field3,
                            center="particle", seed_base_rot=seed_base_rot, theta_index=it,
                            rho1=rho,
                        )

                if rank == 0:
                    elapsed_theta = time.perf_counter() - t_theta_start
                    self.logger.info(
                        f" theta[{it + 1:02d}/{self.n_theta}] done | theta={th:.5f} rad | "
                        f"elapsed={elapsed_theta:.2f} sec"
                    )

            global_results = {}
            for key, arr in local_results.items():
                local_weighted = arr * npos_local
                global_weighted = np.empty_like(arr)
                comm.Allreduce(local_weighted, global_weighted, op=MPI.SUM)
                global_results[key] = global_weighted / npos_total

            if rank == 0:
                t_loop_end = time.perf_counter()
                self.logger.info(f"DDD main loop time: {t_loop_end - t_start:.4f} sec")
                self.logger.info("Main DDD loop finished, computing xi12/xi13/xi23 on rank 0 ...")

                pair_cache = {}
                rr23_cache = None
                xi12_time = 0.0
                xi13_time = 0.0
                xi23_time = 0.0
                if "xi12" in expanded_products:
                    t_pair = time.perf_counter()
                    pair_cache["xi12"] = self._compute_pair_stats(
                        _local_convols1, _local_convols2, _local_random1, _local_random2, self.r12
                    )
                    xi12_time = time.perf_counter() - t_pair
                if "xi13" in expanded_products:
                    t_pair = time.perf_counter()
                    pair_cache["xi13"] = self._compute_pair_stats(
                        _local_convols1, _local_convols3, _local_random1, _local_random3, self.r13
                    )
                    xi13_time = time.perf_counter() - t_pair
                if "xi23" in expanded_products or "rrr" in expanded_products:
                    t_pair = time.perf_counter()
                    pair_cache["xi23"] = self._compute_pair_stats_series(
                        _local_convols2,
                        _local_convols3,
                        _local_random2,
                        _local_random3,
                        math_util.third_side(self.r12, self.r13, theta_arr),
                    )
                    rr23_cache = pair_cache["xi23"]["rr"]
                    xi23_time = time.perf_counter() - t_pair

                self.corr3pcf_data.theta = theta_arr
                self.corr3pcf_data.r23 = math_util.third_side(self.r12, self.r13, theta_arr)
                self.corr3pcf_data.ddd = global_results.get("ddd")
                self.corr3pcf_data.rrr = global_results.get("rrr")
                self.corr3pcf_data.d_delta_dd = global_results.get("d_delta_dd")
                self.corr3pcf_data.delta_ddd = global_results.get("delta_ddd")

                self.corr3pcf_data.xi12 = pair_cache.get("xi12", {}).get("xi")
                self.corr3pcf_data.xi13 = pair_cache.get("xi13", {}).get("xi")
                self.corr3pcf_data.xi23 = pair_cache.get("xi23", {}).get("xi")

                if "rrr" in expanded_products and self.corr3pcf_data.rrr is None:
                    rho1 = self._field_density(_local_random1)
                    if rr23_cache is not None and isinstance(_local_random1, (float, int, np.floating)):
                        self.corr3pcf_data.rrr = rho1 * rr23_cache
                    else:
                        self.corr3pcf_data.rrr = np.array([
                            self._compute_rrr_value(th, r23, self.center, pos_local, seed_base_rot, i, _local_random1, _local_random2, _local_random3, rr23_cache=None)
                            for i, (th, r23) in enumerate(zip(theta_arr, self.corr3pcf_data.r23))
                        ], dtype=np.float64)

                if self.center == "particle" and "delta_ddd" in expanded_products:
                    self.corr3pcf_data.delta_ddd = self.corr3pcf_data.d_delta_dd - self.corr3pcf_data.xi23 * self.corr3pcf_data.rrr

                if "zeta" in expanded_products:
                    self.corr3pcf_data.zeta = self.corr3pcf_data.delta_ddd / self.corr3pcf_data.rrr
                if "Q" in expanded_products:
                    xi12 = self.corr3pcf_data.xi12
                    xi13 = self.corr3pcf_data.xi13
                    xi23 = self.corr3pcf_data.xi23
                    self.corr3pcf_data.Q = self.corr3pcf_data.zeta / (xi12 * xi13 + xi12 * xi23 + xi13 * xi23)

                t_end = time.perf_counter()
                self.logger.info(
                    f"Post-processing timing | xi12={xi12_time:.2f} sec | "
                    f"xi13={xi13_time:.2f} sec | xi23={xi23_time:.2f} sec"
                )
                self.logger.info(f"The time for 3PCF (pos-parallel): {t_end - t_start:.4f} sec")
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
