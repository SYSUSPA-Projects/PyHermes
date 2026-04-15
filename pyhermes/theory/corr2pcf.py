import time
import pickle
import copy

import numpy as np

from pyhermes.io import WindowFunc
from pyhermes.io import ConvolsData
from pyhermes.io import Corr2PCFData
from pyhermes.utils import func_util
from pyhermes.pipeline import TaskBase


def calc_DD_mean_r(radius, convols_data1, convols_data2=None, pair_window=None):
    if pair_window is None:
        pair_window_params = {"type": "shell", "len_args": {"R": radius}, "other_args": {}}
    else:
        if not isinstance(pair_window, dict):
            raise TypeError(
                f"Unsupported pair_window input: expected dict, got {type(pair_window)}."
            )
        pair_window_params = copy.deepcopy(pair_window)
        pair_window_params.setdefault("len_args", {})
        pair_window_params["len_args"]["R"] = radius
    pair_window_obj = WindowFunc(pair_window_params, convols_data1.convols_info, threads=convols_data1.threads)
    if convols_data2:
        res = convols_data1 @ pair_window_obj * convols_data2
    else:
        res = convols_data1 @ pair_window_obj * convols_data1
    return res.as_array().mean()


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
        self.threads = max(1, int(self.threads))
        self.task_params['threads'] = self.threads
        self.task_params['products'] = copy.deepcopy(self.products)
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
        self.fout_path = self.task_params['fout_path']
        self.threads = int(self.task_params['threads'])
        window = self.task_params.get('window', None)
        self.window = window if (window and window.get('type')) else None
        for i in range(1, 3):
            window_i = self.task_params.get(f'window{i}', None)
            window_i = window_i if (window_i and window_i.get('type')) else None
            if (not window_i) and self.window:
                window_i = dict(self.window)
            setattr(self, f'window{i}', window_i)
        pair_window_params = self.task_params.get('pair_window', None)
        if pair_window_params and pair_window_params.get('type'):
            self.pair_window_params = copy.deepcopy(pair_window_params)
        else:
            self.pair_window_params = {"type": "shell", "len_args": {"R": None}, "other_args": {}}
        self.r_min = self.task_params['r_min']
        self.r_max = self.task_params['r_max']
        self.n_r = int(self.task_params['n_r'])
        self.products = self._normalize_products(self.task_params.get('products', 'xi'))

    def _normalize_products(self, products):
        if isinstance(products, str):
            products = [products]
        elif products is None:
            products = ['xi']
        elif not isinstance(products, (list, tuple, set)):
            raise TypeError(
                f"Unsupported products input: expected string or array of strings, got {type(products)}."
            )

        allowed = ['dd', 'dr', 'rd', 'delta_dd', 'rr', 'xi']
        normalized = []
        for item in products:
            if not isinstance(item, str):
                raise TypeError("Each product name must be a string.")
            name = item.strip().lower()
            if name not in allowed:
                raise ValueError(f"Unsupported product '{item}'. Allowed values are {allowed}.")
            if name not in normalized:
                normalized.append(name)
        if 'xi' in normalized:
            for dep in ['delta_dd', 'rr']:
                if dep not in normalized:
                    normalized.append(dep)
        return normalized

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
        return normalized

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
        params['convols_data'] = self._serialize_convols_input(self.convols_data)
        params['convols_data1'] = self._serialize_convols_input(self.convols_data1)
        params['convols_data2'] = self._serialize_convols_input(self.convols_data2)
        params['random'] = self._serialize_convols_input(self.random)
        params['random1'] = self._serialize_convols_input(self.random1)
        params['random2'] = self._serialize_convols_input(self.random2)
        params['window'] = self._serialize_window_input(self.window)
        params['window1'] = self._serialize_window_input(self.window1)
        params['window2'] = self._serialize_window_input(self.window2)
        params['fout_path'] = self.fout_path
        params['threads'] = self.threads
        params['products'] = copy.deepcopy(self.products)
        params['pair_window'] = copy.deepcopy(
            self.pair_window if self.pair_window is not None else self.pair_window_params
        )
        params['r_min'] = self.r_min
        params['r_max'] = self.r_max
        params['n_r'] = self.n_r
        return params

    def _describe_pair_window(self, pair_window):
        if isinstance(pair_window, dict):
            return f"pair_window dict | {func_util.describe_window_action(pair_window)} | runtime R follows current radius"
        return "pair_window dict | shell window with runtime R=radius"

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
        if isinstance(field, (float, int, np.floating)):
            return float(field)
        if isinstance(field, ConvolsData):
            return 1.0 / field.V
        raise TypeError(f"Unsupported field type for density extraction: {type(field)}")

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
        products = set(self.products)
        needs_data = bool(products & {'dd', 'dr', 'rd', 'delta_dd', 'xi'})
        needs_random = bool(products & {'dr', 'rd', 'rr', 'delta_dd', 'xi'})
        return needs_data, needs_random

    def calc_pair_product(self, radius, field1, field2=None, pair_window=None):
        if field2 is None:
            field2 = field1
        if pair_window is None:
            pair_window = self.pair_window
        pair_window = self._normalize_pair_window(pair_window)
        if isinstance(field1, (float, int, np.floating)) or isinstance(field2, (float, int, np.floating)):
            return self._field_density(field1) * self._field_density(field2)
        return calc_DD_mean_r(radius, field1, field2, pair_window=pair_window)

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
        self.products = self._normalize_products(self.products)
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
            self.logger.info(
                f"products={self.products}, n_r={self.n_r}, "
                f"r_min={self.r_min}, r_max={self.r_max}, threads={self.threads}"
            )
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
                            f"Missing input for random leg {i}. Products {self.products} require 'random{i}' or shared 'random'."
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
            needs_data, needs_random = self._required_input_flags()
            _local_convols1 = self._broadcast_field(self.convols_data1) if needs_data else None
            _local_convols2 = self._broadcast_field(self.convols_data2) if needs_data else None
            _local_random1 = self._broadcast_field(self.random1) if needs_random else None
            _local_random2 = self._broadcast_field(self.random2) if needs_random else None
            self.corr2pcf_data.corr2pcf_info = self._current_task_params_snapshot()
            self.corr2pcf_data.task_params = self._current_task_params_snapshot()
            if rank == 0:
                self.logger.info("Start to calculate 2PCF ...")
                self.logger.info(
                    f"products={self.products}, n_r={self.n_r}, "
                    f"r_min={self.r_min}, r_max={self.r_max}"
                )
                time_start = time.perf_counter()
                self.logger.info(f"Pre-2PCF setup time: {time_start - time_run_1:.4f} sec")
            # Generate r_arr at rank0
            if rank == 0:
                r_arr = np.linspace(self.r_min, self.r_max, self.n_r)
                r_sub_arrs = np.array_split(r_arr, size)
                # Global process status
                arr_complete = np.zeros(size, dtype=int)
                total_tasks = len(r_arr)
                report_interval = total_tasks // 10
                next_report_threshold = 0
                requests = [comm.irecv(source=r, tag=r) for r in range(size)]
                count_all = False
            else:
                r_sub_arrs = None
            # Scatter to all ranks
            r_sub_arr = comm.scatter(r_sub_arrs, root=0)
            # Local process status
            local_completed = 0
            local_report_interval = max(1, len(r_sub_arr) // 10)
            # Init local 2pcf results
            local_xi = []
            local_dd = []
            local_dr = []
            local_rd = []
            local_delta_dd = []
            local_rr = []
            local_r = []
            for i, radius in enumerate(r_sub_arr):
                dd_value = None
                dr_value = None
                rd_value = None
                delta_dd_value = None
                rr_value = None
                xi_value = None
                if 'dd' in self.products:
                    dd_value = self.calc_pair_product(radius, _local_convols1, _local_convols2, pair_window=self.pair_window)
                if 'dr' in self.products:
                    dr_value = self.calc_pair_product(radius, _local_convols1, _local_random2, pair_window=self.pair_window)
                if 'rd' in self.products:
                    rd_value = self.calc_pair_product(radius, _local_random1, _local_convols2, pair_window=self.pair_window)
                if 'delta_dd' in self.products:
                    if isinstance(_local_random1, (float, int, np.floating)):
                        field1 = _local_convols1 - self._field_density(_local_random1)
                    else:
                        field1 = _local_convols1 - _local_random1
                    if isinstance(_local_random2, (float, int, np.floating)):
                        field2 = _local_convols2 - self._field_density(_local_random2)
                    else:
                        field2 = _local_convols2 - _local_random2
                    delta_dd_value = self.calc_pair_product(radius, field1, field2, pair_window=self.pair_window)
                if 'rr' in self.products:
                    rr_value = self.calc_pair_product(radius, _local_random1, _local_random2, pair_window=self.pair_window)
                if 'xi' in self.products:
                    xi_value = delta_dd_value / rr_value
                local_dd.append(dd_value)
                local_dr.append(dr_value)
                local_rd.append(rd_value)
                local_delta_dd.append(delta_dd_value)
                local_rr.append(rr_value)
                local_xi.append(xi_value)
                local_r.append(radius)
                local_completed += 1
                if local_completed % local_report_interval == 0:
                    comm.isend(local_completed, dest=0, tag=rank)
                if rank == 0:
                    for r, req in enumerate(requests):
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
            # Gathering to rank0
            gathered_xi = comm.gather(local_xi, root=0)
            gathered_dd = comm.gather(local_dd, root=0)
            gathered_dr = comm.gather(local_dr, root=0)
            gathered_rd = comm.gather(local_rd, root=0)
            gathered_delta_dd = comm.gather(local_delta_dd, root=0)
            gathered_rr = comm.gather(local_rr, root=0)
            gathered_r = comm.gather(local_r, root=0)
            if rank == 0:
                xi_arr = np.array([item for sublist in gathered_xi for item in sublist], dtype=object)
                dd_arr = np.array([item for sublist in gathered_dd for item in sublist], dtype=object)
                dr_arr = np.array([item for sublist in gathered_dr for item in sublist], dtype=object)
                rd_arr = np.array([item for sublist in gathered_rd for item in sublist], dtype=object)
                delta_dd_arr = np.array([item for sublist in gathered_delta_dd for item in sublist], dtype=object)
                rr_arr = np.array([item for sublist in gathered_rr for item in sublist], dtype=object)
                self.corr2pcf_data.r = np.array([item for sublist in gathered_r for item in sublist])
                self.corr2pcf_data.dd = None if 'dd' not in self.products else np.asarray(dd_arr, dtype=np.float64)
                self.corr2pcf_data.dr = None if 'dr' not in self.products else np.asarray(dr_arr, dtype=np.float64)
                self.corr2pcf_data.rd = None if 'rd' not in self.products else np.asarray(rd_arr, dtype=np.float64)
                self.corr2pcf_data.delta_dd = None if 'delta_dd' not in self.products else np.asarray(delta_dd_arr, dtype=np.float64)
                self.corr2pcf_data.rr = None if 'rr' not in self.products else np.asarray(rr_arr, dtype=np.float64)
                self.corr2pcf_data.xi = None if 'xi' not in self.products else np.asarray(xi_arr, dtype=np.float64)
                if not count_all:
                    progress = 100.
                    self.logger.info(f" Progress: {progress:6.2f}%")
                time_end = time.perf_counter()
                self.logger.info(f"The time for 2PCF: {time_end - time_start:.4f} sec")
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
