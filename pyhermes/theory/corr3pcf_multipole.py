import copy
import pickle
import time

import numpy as np
from mpi4py import MPI

from pyhermes.io import WindowFunc, ConvolsData, Corr3PCFMultipoleData
from pyhermes.pipeline import TaskBase
from pyhermes.utils import func_util, math_util


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
        self.convols_data = self.task_params.get("convols_data", "")
        self.convols_data1 = self.task_params.get("convols_data1", "") or self.convols_data
        self.convols_data2 = self.task_params.get("convols_data2", "") or self.convols_data
        self.convols_data3 = self.task_params.get("convols_data3", "") or self.convols_data
        self.random = self.task_params.get("random", None)
        self.random1 = self.task_params.get("random1", None)
        self.random2 = self.task_params.get("random2", None)
        self.random3 = self.task_params.get("random3", None)
        self.random1 = self._fallback_random(self.random1)
        self.random2 = self._fallback_random(self.random2)
        self.random3 = self._fallback_random(self.random3)
        self.fout_path = self.task_params["fout_path"]

        window = self.task_params.get("window", None)
        self.window = window if (isinstance(window, dict) and window.get("type")) else None
        for i in range(1, 4):
            window_i = self.task_params.get(f"window{i}", None)
            window_i = window_i if (isinstance(window_i, dict) and window_i.get("type")) else None
            if (not window_i) and self.window:
                window_i = dict(self.window)
            setattr(self, f"window{i}", window_i)

        self.r12 = float(self.task_params.get("r12", self.task_params.get("r1")))
        self.r13 = float(self.task_params.get("r13", self.task_params.get("r2")))
        self.l_min = int(self.task_params["l_min"])
        self.l_max = int(self.task_params["l_max"])
        self.gpu_device_id = int(self.task_params["gpu_device_id"])
        self.execution_mode = self.task_params["execution_mode"]
        self.cache_multipole_fields = bool(self.task_params["cache_multipole_fields"])
        self.cache_dir = self.task_params["cache_dir"]
        self.verbose_m_progress = bool(self.task_params["verbose_m_progress"])
        self.threads = int(self.task_params["threads"])
        self.products = self._normalize_products(self.task_params.get("products", "zeta_l"))
        self.rho = None
        self.reference_convols = None

    def _fallback_random(self, value):
        return self.random if value is None or value == "" else value

    def _fallback_convols(self, value):
        return self.convols_data if value is None or value == "" else value

    def _fallback_window(self, value):
        if value is None:
            return self.window
        if isinstance(value, dict) and not value.get("type") and self.window:
            return self.window
        return value

    def _sync_runtime_options(self):
        self.threads = max(1, int(self.threads))
        self.products = self._normalize_products(self.products)
        self.task_params["threads"] = self.threads
        self.task_params["products"] = copy.deepcopy(self.products)
        self.sync_runtime_options(context="Corr_3PCF multipole runtime configuration", blank_line=True)

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

    def _serialize_convols_input(self, value):
        if isinstance(value, str) or value is None:
            return value
        if isinstance(value, ConvolsData):
            return {
                "kind": "ConvolsData",
                "L": getattr(value, "L", value.epsilon.shape[0] if value.epsilon is not None else None),
                "SimBoxL": getattr(value, "SimBoxL", None),
                "wavelet_mode": getattr(value, "wavelet_mode", None),
                "wavelet_level": getattr(value, "wavelet_level", None),
            }
        if np.isscalar(value):
            return float(value)
        return str(type(value))

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
        return {
            "convols_data": self._serialize_convols_input(self.convols_data),
            "convols_data1": self._serialize_convols_input(self.convols_data1),
            "convols_data2": self._serialize_convols_input(self.convols_data2),
            "convols_data3": self._serialize_convols_input(self.convols_data3),
            "random": self._serialize_convols_input(self.random),
            "random1": self._serialize_convols_input(self.random1),
            "random2": self._serialize_convols_input(self.random2),
            "random3": self._serialize_convols_input(self.random3),
            "window": self._serialize_window_input(self.window),
            "window1": self._serialize_window_input(self.window1),
            "window2": self._serialize_window_input(self.window2),
            "window3": self._serialize_window_input(self.window3),
            "fout_path": self.fout_path,
            "threads": self.threads,
            "r12": self.r12,
            "r13": self.r13,
            "l_min": self.l_min,
            "l_max": self.l_max,
            "gpu_device_id": self.gpu_device_id,
            "products": copy.deepcopy(self.products),
            "expanded_products": self._expanded_products(),
            "execution_mode": self.execution_mode,
            "cache_multipole_fields": self.cache_multipole_fields,
            "cache_dir": self.cache_dir,
            "verbose_m_progress": self.verbose_m_progress,
        }

    def _resolve_base_convols(self, leg_idx, provided_convols, base_convols_cache):
        if isinstance(provided_convols, str) and provided_convols:
            if provided_convols not in base_convols_cache:
                base_convols_cache[provided_convols] = ConvolsData(data_path=provided_convols, threads=self.threads)
            return base_convols_cache[provided_convols], f"path={provided_convols}"
        if isinstance(provided_convols, ConvolsData):
            return provided_convols, f"provided convols_data{leg_idx}"
        if provided_convols in (None, ""):
            self.logger.error(
                f"Missing input for field leg {leg_idx}. Products {self._expanded_products()} require "
                f"'convols_data{leg_idx}' or shared 'convols_data'."
            )
            func_util.safe_exit(1)
        self.logger.error(
            f"Unexpected input: 'convols_data{leg_idx}' must be a string path or a ConvolsData instance."
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
                random_cache[provided_random] = ConvolsData(data_path=provided_random, threads=self.threads)
            return random_cache[provided_random], f"path={provided_random}"
        if isinstance(provided_random, ConvolsData):
            return provided_random, f"provided random{leg_idx}"
        self.logger.error(
            f"Unexpected input: 'random{leg_idx}' must be 'uniform', a string path, or a ConvolsData instance."
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

    def _uniform_density(self):
        if self.rho is None:
            raise ValueError("Shared density is not initialized.")
        return float(self.rho)

    def _materialize_uniform_random(self, reference_field, rho, leg_idx):
        field = reference_field._spawn_like()
        field.epsilon = np.full_like(reference_field.epsilon, rho, dtype=np.float64)
        field.convols_info.update({
            "uniform_random_materialized": True,
            "uniform_random_leg": int(leg_idx),
        })
        field.format_convols_params()
        return field

    def _broadcast_convols(self, rank, comm, convols_data):
        serialized = pickle.dumps(convols_data.convols_info) if rank == 0 else None
        serialized = comm.bcast(serialized, root=0)
        if rank == 0:
            local = convols_data
            local.epsilon = np.ascontiguousarray(local.epsilon, dtype=np.float64)
        else:
            local = ConvolsData(threads=self.threads)
            local.convols_info = pickle.loads(serialized)
            local.format_convols_params()
            local.epsilon = np.empty((local.L, local.L, local.L), dtype=np.float64)
        comm.Bcast(local.epsilon, root=0)
        return local

    def _broadcast_convols_to_ranks(self, comm, convols_data, target_ranks):
        """
        Broadcast one ConvolsData only to ranks that need it.

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

        serialized = pickle.dumps(convols_data.convols_info) if rank == 0 else None
        serialized = subcomm.bcast(serialized, root=0)
        if rank == 0:
            send_field = convols_data
            send_field.epsilon = np.ascontiguousarray(send_field.epsilon, dtype=np.float64)
            subcomm.Bcast(send_field.epsilon, root=0)
            local = send_field if rank in target_ranks else None
        else:
            local = ConvolsData(threads=self.threads)
            local.convols_info = pickle.loads(serialized)
            local.format_convols_params()
            local.epsilon = np.empty((local.L, local.L, local.L), dtype=np.float64)
            subcomm.Bcast(local.epsilon, root=0)
        subcomm.Free()
        return local

    def prepare_input_fields(
        self,
        convols_data1=None,
        convols_data2=None,
        convols_data3=None,
        random1=None,
        random2=None,
        random3=None,
        window1=None,
        window2=None,
        window3=None,
    ):
        self.corr3pcf_multipole_data = Corr3PCFMultipoleData()
        self._sync_runtime_options()

        if "zeta_l" in self._expanded_products() and self.l_min > 0:
            if self.rank == 0:
                self.logger.error("zeta_l requires l_min=0 because the ratio solve needs multipoles from l=0.")
            func_util.safe_exit(1)

        needs_data, needs_random = self._required_input_flags()
        data_inputs = [
            convols_data1 if convols_data1 is not None else self._fallback_convols(self.convols_data1),
            convols_data2 if convols_data2 is not None else self._fallback_convols(self.convols_data2),
            convols_data3 if convols_data3 is not None else self._fallback_convols(self.convols_data3),
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

        if self.rank == 0:
            self.logger.info("Preparing Corr_3PCF multipole input fields ...")
            self.logger.info(
                f"execution_mode={self.execution_mode}, l_min={self.l_min}, l_max={self.l_max}, "
                f"r12={self.r12}, r13={self.r13}, threads={self.threads}, "
                f"cache_multipole_fields={self.cache_multipole_fields}, "
                f"verbose_m_progress={self.verbose_m_progress}"
            )
            self.logger.info(
                f"requested_products={self.products}, expanded_products={self._expanded_products()}"
            )

            base_convols_cache = {}
            random_cache = {}
            data_legs = []
            random_legs = []
            compatibility_fields = []

            if needs_data:
                for i, cdata in enumerate(data_inputs, start=1):
                    base_convols, source_desc = self._resolve_base_convols(i, cdata, base_convols_cache)
                    data_legs.append((i, base_convols, source_desc, window_inputs[i - 1]))
                    compatibility_fields.append(base_convols)

            if needs_random:
                for i, random_input in enumerate(random_inputs, start=1):
                    base_random, source_desc = self._resolve_random_base(i, random_input, random_cache)
                    random_legs.append((i, base_random, source_desc, window_inputs[i - 1]))
                    if isinstance(base_random, ConvolsData):
                        compatibility_fields.append(base_random)

            if not compatibility_fields:
                for i, cdata in enumerate(data_inputs, start=1):
                    if cdata is None or cdata == "":
                        continue
                    base_convols, source_desc = self._resolve_base_convols(i, cdata, base_convols_cache)
                    compatibility_fields.append(base_convols)
                    self.logger.info(
                        f"Geometry reference loaded from field leg {i} | source={source_desc}"
                    )
                    break

            if not compatibility_fields:
                self.logger.error(
                    "At least one ConvolsData input is required to define the grid geometry and shared density."
                )
                func_util.safe_exit(1)

            shared_required = func_util.validate_convols_compatibility(
                compatibility_fields,
                ConvolsData._REQUIRED_ARGV,
                logger=self.logger,
                label="Corr_3PCF multipole input fields",
            )
            shared_required_text = ", ".join([f"{k}={v}" for k, v in shared_required.items()])
            self.reference_convols = compatibility_fields[0]
            self.rho = 1.0 / self.reference_convols.V
            self.logger.info("Corr_3PCF multipole input compatibility check passed.")
            self.logger.info(f"Shared required parameters | {shared_required_text}")
            self.logger.info(f"Shared density | rho={self.rho:.6g}")

            if needs_data:
                for i, base_convols, source_desc, win in data_legs:
                    window_obj, window_desc = self._resolve_window(i, base_convols, win)
                    if window_obj is not None:
                        final_convols = base_convols @ window_obj
                    else:
                        final_convols = base_convols.copy()
                        final_convols.format_convols_params()

                    setattr(self, f"convols_data{i}", final_convols)
                    setattr(self.corr3pcf_multipole_data, f"convols_info{i}", final_convols.convols_info)
                    self.logger.info(f"Field leg {i} ready | source={source_desc} | window={window_desc}")
            else:
                for i in range(1, 4):
                    setattr(self.corr3pcf_multipole_data, f"convols_info{i}", self.reference_convols.convols_info)

            if needs_random:
                for i, base_random, source_desc, win in random_legs:
                    if isinstance(base_random, str) and base_random == "uniform":
                        setattr(self, f"random{i}", self.rho)
                        self.logger.info(
                            f"Random leg {i} ready | source={source_desc} | window=uniform shortcut | rho={self.rho:.6g}"
                        )
                        continue
                    window_obj, window_desc = self._resolve_window(i, base_random, win)
                    if window_obj is not None:
                        final_random = base_random @ window_obj
                    else:
                        final_random = base_random.copy()
                        final_random.format_convols_params()
                    setattr(self, f"random{i}", final_random)
                    self.logger.info(f"Random leg {i} ready | source={source_desc} | window={window_desc}")

            snapshot = self._current_task_params_snapshot()
            self.corr3pcf_multipole_data.corr3pcf_multipole_info = snapshot
            self.corr3pcf_multipole_data.task_params = snapshot
        self._fields_prepared = True

    def _store_product(self, product_name, l_arr, values):
        self.corr3pcf_multipole_data.r12 = self.r12
        self.corr3pcf_multipole_data.r13 = self.r13
        self.corr3pcf_multipole_data.l = np.asarray(l_arr, dtype=np.int32)
        setattr(self.corr3pcf_multipole_data, product_name, np.asarray(values, dtype=np.float64))

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
            self.logger.info(
                f" l={l:2d}/{l_max:2d} done | {product_name}={ddd_l:.5e} | "
                f"elapsed={elapsed_sec:.2f} sec | conv={conv_elapsed_sec:.2f} sec | "
                f"sum={sum_elapsed_sec:.2f} sec | progress={progress:6.2f}% "
                f"({completed_m_tasks}/{total_m_tasks} m-tasks)"
            )

        def _log_m_progress(l, l_max, m, m_max, value, elapsed_sec, completed_m_tasks, total_m_tasks):
            progress = (completed_m_tasks / total_m_tasks) * 100.0
            msg = (
                f"   m={m:2d}/{m_max:2d} in l={l:2d}/{l_max:2d} | "
                f"value={_format_complex(value)} | elapsed={elapsed_sec:.2f} sec | "
                f"progress={progress:6.2f}% ({completed_m_tasks}/{total_m_tasks})"
            )
            print(msg, flush=True)

        return _log_l_progress, _log_m_progress

    def _run_serial_mode(self, rank, fields, product_name):
        if rank != 0:
            return None, None

        self.logger.info(f"Start to calculate 3PCF multipole product '{product_name}' ...")
        self.logger.info(
            f"execution_mode={self.execution_mode}, l_min={self.l_min}, l_max={self.l_max}, "
            f"threads={self.threads}, cache_multipole_fields={self.cache_multipole_fields}, "
            f"verbose_m_progress={self.verbose_m_progress}"
        )

        log_l_progress, log_m_progress = self._log_helpers(product_name)
        l_arr, multipole_l, timing_info = math_util.calc_DDD_multipole(
            fields[0], fields[1], fields[2],
            self.r12, self.r13, self.l_min, self.l_max,
            gpu_device_id=self.gpu_device_id,
            cache_multipole_fields=self.cache_multipole_fields,
            cache_dir=self.cache_dir,
            threads=self.threads,
            progress_callback=log_l_progress if self.verbose_m_progress else None,
            m_progress_callback=log_m_progress if self.verbose_m_progress else None,
        )
        if self.verbose_m_progress:
            self.logger.info(
                f"3PCF multipole timing [{product_name}] | convolution={timing_info['conv_elapsed_sec']:.2f} sec | "
                f"summation={timing_info['sum_elapsed_sec']:.2f} sec"
            )
            self.logger.info(
                f"3PCF multipole summation breakdown [{product_name}] | "
                f"h2d={timing_info['sum_h2d_elapsed_sec']:.2f} sec | "
                f"kernel={timing_info['sum_kernel_elapsed_sec']:.2f} sec | "
                f"d2h={timing_info['sum_d2h_elapsed_sec']:.2f} sec | "
                f"reduce={timing_info['sum_reduce_elapsed_sec']:.2f} sec | "
                f"callback={timing_info['sum_callback_elapsed_sec']:.2f} sec"
            )
        return l_arr, multipole_l

    def _run_pair_mpi_mode(self, comm, rank, local_fields, product_name):
        size = comm.Get_size()
        if size == 1:
            if rank == 0:
                self.logger.warning(
                    "execution_mode='pair_mpi' requested with a single MPI rank. Falling back to serial execution."
                )
                self.execution_mode = "serial"
            return self._run_serial_mode(rank, local_fields, product_name)
        if size < 2 or size % 2 != 0:
            self.logger.error("execution_mode='pair_mpi' requires an even number of MPI ranks.")
            func_util.safe_exit(1)
        n_pairs = size // 2

        if rank == 0:
            self.logger.info(f"Start to calculate 3PCF multipole product '{product_name}' ...")
            self.logger.info(
                f"execution_mode={self.execution_mode}, l_min={self.l_min}, l_max={self.l_max}, "
                f"threads={self.threads}, ranks={size}, pairs={n_pairs}, "
                f"cache_multipole_fields={self.cache_multipole_fields}, "
                f"verbose_m_progress={self.verbose_m_progress}"
            )

        field1, field2, field3 = local_fields
        pair_idx = rank if rank < n_pairs else rank - n_pairs
        is_r1_rank = rank < n_pairs

        conv_context_r1 = math_util._prepare_legendre_convolution_context(field2) if is_r1_rank else None
        conv_context_r2 = math_util._prepare_legendre_convolution_context(field3) if not is_r1_rank else None
        gpu_context = math_util._prepare_multipole_gpu_context(field1, gpu_device_id=self.gpu_device_id) if rank == 0 else None

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
        l_wall_starts = ({int(l): None for l in l_arr} if (rank == 0 and self.verbose_m_progress) else None)
        l_conv_accum = ({int(l): 0.0 for l in l_arr} if (rank == 0 and self.verbose_m_progress) else None)
        l_comm_accum = ({int(l): 0.0 for l in l_arr} if (rank == 0 and self.verbose_m_progress) else None)
        l_sum_accum = ({int(l): 0.0 for l in l_arr} if (rank == 0 and self.verbose_m_progress) else None)

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
                    local_field = math_util._stream_convolution_fields(
                        field2, self.r12, int(l), threads=self.threads, m_values=[int(m)], conv_context=conv_context_r1
                    )[0]
                else:
                    local_field = math_util._stream_convolution_fields(
                        field3, self.r13, int(l), threads=self.threads, m_values=[-int(m)], conv_context=conv_context_r2
                    )[0]
            conv_elapsed = time.perf_counter() - t_conv
            total_conv_elapsed += conv_elapsed

            t_comm = time.perf_counter()
            round_summands = {} if rank == 0 else None
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
                        value, timing = math_util.compute_multipole_m_summand(field_r1_m, field_r2_m, gpu_context)
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
                        comm.Send([np.ascontiguousarray(local_field, dtype=np.complex128), MPI.COMPLEX16], dest=0, tag=tag_base)
                        del local_field
                else:
                    comm.Send([np.ascontiguousarray(local_field, dtype=np.complex128), MPI.COMPLEX16], dest=0, tag=tag_base + 50)
                    del local_field
            if rank != 0:
                comm_elapsed = time.perf_counter() - t_comm
            total_comm_elapsed += comm_elapsed

            round_timings = None
            if self.verbose_m_progress:
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
                if self.verbose_m_progress:
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
                    if self.verbose_m_progress and l_wall_starts[l] is None:
                        l_wall_starts[l] = time.perf_counter()
                    task_timing = timing_by_task.get(idx, {})
                    conv_r1 = task_timing.get(0, (0.0, 0.0))[0]
                    conv_r2 = task_timing.get(1, (0.0, 0.0))[0]
                    comm_r1 = task_timing.get(0, (0.0, 0.0))[1]
                    comm_r2 = task_timing.get(1, (0.0, 0.0))[1]
                    value, timing, sum_elapsed = round_summands[key]
                    total_sum_elapsed += sum_elapsed
                    total_h2d += timing["h2d_elapsed_sec"]
                    total_kernel += timing["kernel_elapsed_sec"]
                    total_reduce += timing["reduce_elapsed_sec"]
                    total_d2h += timing["d2h_elapsed_sec"]
                    m_storage[l][m] = value
                    done_per_l[l] += 1
                    completed_m_tasks += 1
                    if self.verbose_m_progress:
                        l_conv_accum[l] += max(conv_r1, conv_r2)
                        l_comm_accum[l] += max(comm_r1, comm_r2)
                        l_sum_accum[l] += sum_elapsed
                        log_m_progress(
                            l=l, l_max=self.l_max, m=m, m_max=l, value=value,
                            elapsed_sec=max(conv_r1, conv_r2) + max(comm_r1, comm_r2) + sum_elapsed,
                            completed_m_tasks=completed_m_tasks, total_m_tasks=total_m_tasks,
                        )
                    if done_per_l[l] == l + 1:
                        multipole_l[l_idx] = math_util.combine_multipole_m_terms(m_storage[l], l)
                        progress = (completed_m_tasks / total_m_tasks) * 100.0
                        stat_str = f"{product_name}={multipole_l[l_idx]:.5e}"
                        if self.verbose_m_progress:
                            self.logger.info(
                                f" l={l:2d}/{self.l_max:2d} done | {stat_str} | "
                                f"elapsed={time.perf_counter() - l_wall_starts[l]:.2f} sec | "
                                f"conv={l_conv_accum[l]:.2f} sec | comm={l_comm_accum[l]:.2f} sec | "
                                f"sum={l_sum_accum[l]:.2f} sec | "
                                f"progress={progress:6.2f}% ({completed_m_tasks}/{total_m_tasks} m-tasks)"
                            )
                        else:
                            self.logger.info(
                                f" l={l:2d}/{self.l_max:2d} done | {stat_str} | "
                                f"progress={progress:6.2f}% ({completed_m_tasks}/{total_m_tasks} m-tasks)"
                            )

        conv_sum_all = comm.reduce(total_conv_elapsed, op=MPI.SUM, root=0) if self.verbose_m_progress else None
        conv_max_rank = comm.reduce(total_conv_elapsed, op=MPI.MAX, root=0) if self.verbose_m_progress else None
        comm_sum_all = comm.reduce(total_comm_elapsed, op=MPI.SUM, root=0) if self.verbose_m_progress else None
        comm_max_rank = comm.reduce(total_comm_elapsed, op=MPI.MAX, root=0) if self.verbose_m_progress else None
        if rank == 0 and self.verbose_m_progress:
            self.logger.info(
                f"Pair-MPI timing [{product_name}] | conv_rank0={total_conv_elapsed:.2f} sec | "
                f"conv_sum_all={conv_sum_all:.2f} sec | conv_max_rank={conv_max_rank:.2f} sec | "
                f"comm_sum_all={comm_sum_all:.2f} sec | comm_max_rank={comm_max_rank:.2f} sec | "
                f"summation={total_sum_elapsed:.2f} sec"
            )
            self.logger.info(
                f"Pair-MPI summation breakdown [{product_name}] | h2d={total_h2d:.2f} sec | "
                f"kernel={total_kernel:.2f} sec | d2h={total_d2h:.2f} sec | reduce={total_reduce:.2f} sec"
            )
        return l_arr if rank == 0 else None, multipole_l

    def _is_uniform_random(self, field):
        return isinstance(field, (float, int, np.floating))

    def _delta_field(self, data_field, random_field):
        if self._is_uniform_random(random_field):
            return data_field - self._field_density(random_field)
        return data_field - random_field

    def _prepare_product_fields(self, product_name):
        if product_name == "ddd_l":
            return [self.convols_data1, self.convols_data2, self.convols_data3]
        if product_name == "delta_ddd_l":
            return [
                self._delta_field(self.convols_data1, self.random1),
                self._delta_field(self.convols_data2, self.random2),
                self._delta_field(self.convols_data3, self.random3),
            ]
        if product_name == "rrr_l":
            rho = self._uniform_density()
            reference = self.reference_convols
            fields = []
            for i, random_field in enumerate([self.random1, self.random2, self.random3], start=1):
                if self._is_uniform_random(random_field):
                    self.logger.info(
                        f"Random leg {i} for rrr_l materialized from uniform density for the generic multipole kernel."
                    )
                    fields.append(self._materialize_uniform_random(reference, rho, i))
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
            rrr_l[zero_idx[0]] = self._uniform_density() ** 3
        return l_arr, rrr_l

    def _compute_product_multipole(self, product_name, fields):
        comm = self.comm
        rank = self.rank
        if self.execution_mode == "pair_mpi":
            size = comm.Get_size()
            if size == 1:
                return self._run_pair_mpi_mode(comm, rank, fields if rank == 0 else None, product_name)
            if size < 2 or size % 2 != 0:
                self.logger.error("execution_mode='pair_mpi' requires an even number of MPI ranks.")
                func_util.safe_exit(1)
            n_pairs = size // 2
            if rank == 0:
                self.logger.info(
                    f"Initializing multipole input for '{product_name}': role-aware broadcast to {size} MPI ranks ..."
                )
                self.logger.info(
                    "Role-aware layout | field1 -> rank0 only | "
                    f"field2 -> ranks 0-{n_pairs - 1} | field3 -> ranks {n_pairs}-{size - 1}"
                )
            local_fields = [
                self._broadcast_convols_to_ranks(comm, fields[0] if rank == 0 else None, {0}),
                self._broadcast_convols_to_ranks(comm, fields[1] if rank == 0 else None, set(range(n_pairs))),
                self._broadcast_convols_to_ranks(comm, fields[2] if rank == 0 else None, set(range(n_pairs, size))),
            ]
            if rank == 0:
                fields[:] = [None, None, None]
            if rank == 0:
                self.logger.info(f"Initializing multipole input for '{product_name}': role-aware broadcast complete.")
            return self._run_pair_mpi_mode(comm, rank, local_fields, product_name)
        return self._run_serial_mode(rank, fields if rank == 0 else None, product_name)

    def _compute_zeta_l(self):
        delta_ddd_l = self.corr3pcf_multipole_data.delta_ddd_l
        rrr_l = self.corr3pcf_multipole_data.rrr_l
        if delta_ddd_l is None or rrr_l is None:
            self.logger.error("zeta_l requires both delta_ddd_l and rrr_l.")
            func_util.safe_exit(1)
        zeta_l, _, cond_m = math_util.solve_multipoles_from_ratio(delta_ddd_l, rrr_l, self.l_max)
        self.corr3pcf_multipole_data.zeta_l = zeta_l
        self.logger.info(f"zeta_l solved from multipole ratio | mixing matrix cond={cond_m:.3e}")

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
                    self.logger.info(f"Computing product '{product_name}' ...")
                all_random_uniform = comm.bcast(
                    self._all_random_uniform() if rank == 0 else None,
                    root=0,
                )
                if product_name == "rrr_l" and all_random_uniform:
                    if rank == 0:
                        l_arr, product_l = self._analytic_uniform_rrr_l()
                        self._store_product(product_name, l_arr, product_l)
                        self.logger.info(
                            f"Product 'rrr_l' used all-uniform analytic shortcut | rho^3={self._uniform_density() ** 3:.6e}"
                        )
                        self.logger.info(
                            f"Product timing | {product_name}={time.perf_counter() - product_t0:.4f} sec"
                        )
                    continue

                fields = self._prepare_product_fields(product_name) if rank == 0 else None
                l_arr, product_l = self._compute_product_multipole(product_name, fields)
                if rank == 0:
                    self._store_product(product_name, l_arr, product_l)
                    del fields
                    self.logger.info(
                        f"Product timing | {product_name}={time.perf_counter() - product_t0:.4f} sec"
                    )

            if rank == 0:
                if "zeta_l" in expanded_products:
                    self.logger.info("Computing product 'zeta_l' from delta_ddd_l and rrr_l ...")
                    self._compute_zeta_l()

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
