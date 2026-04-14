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

    def __init__(self, param_task):
        self.task_name = str(self.__class__.__name__)
        super().__init__(param_task=param_task)
        self.format_params()
        self.convols_data1 = None
        self.convols_data2 = None
        self._fields_prepared = False

    def _sync_runtime_options(self):
        self.threads = max(1, int(self.threads))
        self.task_params['threads'] = self.threads
        self.sync_runtime_options(context="Corr_2PCF runtime configuration")

    def format_params(self):
        self.convols_data_path = self.task_params['convols_data_path']
        self.convols_data1_path = self.task_params.get('convols_data1_path', '') or self.convols_data_path
        self.convols_data2_path = self.task_params.get('convols_data2_path', '') or self.convols_data_path
        self.fout_path = self.task_params['fout_path']
        self.field_mode = self.task_params['field_mode']
        self.threads = int(self.task_params['threads'])
        win_params = self.task_params.get('window', None)
        win_params = win_params if win_params['type'] else None
        for i in range(1, 3):
            win_params_i = self.task_params.get(f'window{i}', None)
            win_params_i = win_params_i if win_params_i['type'] else None
            if (not win_params_i) and win_params:
                win_params_i = dict(win_params)
            setattr(self, f'win_params{i}', win_params_i)
        pair_window_params = self.task_params.get('pair_window', None)
        if pair_window_params and pair_window_params.get('type'):
            self.pair_window_params = copy.deepcopy(pair_window_params)
        else:
            self.pair_window_params = {"type": "shell", "len_args": {"R": None}, "other_args": {}}
        self.r_min = self.task_params['r_min']
        self.r_max = self.task_params['r_max']
        self.n_r = int(self.task_params['n_r'])

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

    def _current_task_params_snapshot(self):
        params = copy.deepcopy(self.task_params)
        params['convols_data_path'] = self.convols_data_path
        params['convols_data1_path'] = self.convols_data1_path
        params['convols_data2_path'] = self.convols_data2_path
        params['fout_path'] = self.fout_path
        params['field_mode'] = self.field_mode
        params['threads'] = self.threads
        params['pair_window'] = copy.deepcopy(self.pair_window_params)
        params['r_min'] = self.r_min
        params['r_max'] = self.r_max
        params['n_r'] = self.n_r
        return params

    def _describe_pair_window(self, pair_window):
        if isinstance(pair_window, dict):
            return f"pair_window dict | {func_util.describe_window_action(pair_window)} | runtime R follows current radius"
        return "pair_window dict | shell window with runtime R=radius"

    def _resolve_base_convols(self, leg_idx, provided_convols, base_convols_cache):
        if provided_convols is not None:
            if not isinstance(provided_convols, ConvolsData):
                self.logger.error(f"Unexpected input: 'convols_data{leg_idx}' is not an instance of 'ConvolsData'.")
                func_util.safe_exit(1)
            return provided_convols, f"argument convols_data{leg_idx}"

        base_path = getattr(self, f"convols_data{leg_idx}_path")
        if not base_path:
            self.logger.error(
                f"Missing input for field leg {leg_idx}. Please pass convols_data{leg_idx} or set "
                f"'convols_data{leg_idx}_path' / 'convols_data_path'."
            )
            func_util.safe_exit(1)
        if base_path not in base_convols_cache:
            base_convols_cache[base_path] = ConvolsData(data_path=base_path, threads=self.threads)
        return base_convols_cache[base_path], f"path={base_path}"

    def _resolve_window(self, leg_idx, base_convols, provided_window):
        if isinstance(provided_window, WindowFunc):
            return provided_window, "provided WindowFunc instance"
        if isinstance(provided_window, dict):
            return WindowFunc(provided_window, base_convols.convols_info, threads=self.threads), (
                f"provided window dict | {func_util.describe_window_action(provided_window)}"
            )
        if provided_window is not None:
            self.logger.error(
                f"Unsupported window input for leg {leg_idx}. Expected dict, WindowFunc, or None, "
                f"got {type(provided_window)}."
            )
            func_util.safe_exit(1)

        config_window = getattr(self, f"win_params{leg_idx}", None)
        if config_window:
            return WindowFunc(config_window, base_convols.convols_info, threads=self.threads), (
                f"config window{leg_idx} | {func_util.describe_window_action(config_window)}"
            )
        return None, "no additional window convolution"

    def prepare_input_fields(self, convols_data1=None, convols_data2=None, window1=None, window2=None, pair_window=None):
        self.corr2pcf_data = Corr2PCFData()
        self._sync_runtime_options()
        self.pair_window = self._normalize_pair_window(pair_window)
        if self.rank == 0:
            self.logger.info("Preparing Corr_2PCF input fields ...")
            self.logger.info(
                f"field_mode={self.field_mode}, n_r={self.n_r}, "
                f"r_min={self.r_min}, r_max={self.r_max}"
            )
            self.logger.info(f"Pair-correlation window: {self._describe_pair_window(self.pair_window)}")
            base_convols_cache = {}
            resolved_legs = []
            for i, cdata, win in zip([1, 2], [convols_data1, convols_data2], [window1, window2]):
                base_convols, source_desc = self._resolve_base_convols(i, cdata, base_convols_cache)
                resolved_legs.append((i, base_convols, source_desc, win))

            shared_required = func_util.validate_convols_compatibility(
                [item[1] for item in resolved_legs],
                ConvolsData._REQUIRED_ARGV,
                logger=self.logger,
                label="Corr_2PCF input fields",
            )
            shared_required_text = ", ".join([f"{k}={v}" for k, v in shared_required.items()])
            self.logger.info("Corr_2PCF input compatibility check passed.")
            self.logger.info(f"Shared required parameters | {shared_required_text}")

            for i, base_convols, source_desc, win in resolved_legs:
                window_obj, window_desc = self._resolve_window(i, base_convols, win)

                if window_obj is not None:
                    final_convols = base_convols @ window_obj
                else:
                    final_convols = base_convols._spawn_like()
                    final_convols.epsilon = base_convols.epsilon
                    final_convols.format_convols_params()

                setattr(self, f"convols_data{i}", final_convols)
                setattr(self.corr2pcf_data, f"convols_info{i}", final_convols.convols_info)
                self.logger.info(
                    f"Field leg {i} ready | source={source_desc} | window={window_desc}"
                )

            self.corr2pcf_data.corr2pcf_info = self._current_task_params_snapshot()
        self._fields_prepared = True

    def _broadcast_input_fields(self):
        comm = self.comm
        rank = self.rank
        convols_info1_serialized = None
        convols_info2_serialized = None

        if rank == 0:
            convols_info1_serialized = pickle.dumps(self.convols_data1.convols_info)
            convols_info2_serialized = pickle.dumps(self.convols_data2.convols_info)
            self.convols_data1.epsilon = np.ascontiguousarray(self.convols_data1.epsilon, dtype=np.float64)
            self.convols_data2.epsilon = np.ascontiguousarray(self.convols_data2.epsilon, dtype=np.float64)
            local_convols1 = self.convols_data1
            local_convols2 = self.convols_data2
        else:
            local_convols1 = ConvolsData(threads=self.threads)
            local_convols2 = ConvolsData(threads=self.threads)

        convols_info1_serialized = comm.bcast(convols_info1_serialized, root=0)
        convols_info2_serialized = comm.bcast(convols_info2_serialized, root=0)

        if rank != 0:
            local_convols1.convols_info = pickle.loads(convols_info1_serialized)
            local_convols1.format_convols_params()
            local_convols1.epsilon = np.empty((local_convols1.L, local_convols1.L, local_convols1.L), dtype=np.float64)

            local_convols2.convols_info = pickle.loads(convols_info2_serialized)
            local_convols2.format_convols_params()
            local_convols2.epsilon = np.empty((local_convols2.L, local_convols2.L, local_convols2.L), dtype=np.float64)

        comm.Bcast(local_convols1.epsilon, root=0)
        comm.Bcast(local_convols2.epsilon, root=0)
        comm.Barrier()

        self.corr2pcf_data.corr2pcf_info = self._current_task_params_snapshot()
        self.corr2pcf_data.task_params = self._current_task_params_snapshot()
        return local_convols1, local_convols2

    def run(self, save_result=True, overwrite=False):
        try:
            comm = self.comm
            rank = self.rank
            size = comm.Get_size()
            if rank == 0:
                time_run_1 = time.perf_counter()
            if not self._fields_prepared:
                self.prepare_input_fields()
            _local_convols1, _local_convols2 = self._broadcast_input_fields()
            if rank == 0:
                self.logger.info("Start to calculate 2PCF ...")
                self.logger.info(
                    f"field_mode={self.field_mode}, n_r={self.n_r}, "
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
            local_r = []
            rho1 = 1.0 / _local_convols1.V
            rho2 = 1.0 / _local_convols2.V
            RR = rho1 * rho2
            if self.field_mode == "delta":
                field1 = _local_convols1 - rho1
                field2 = _local_convols2 - rho2
            elif self.field_mode == "raw":
                field1 = _local_convols1
                field2 = _local_convols2
            else:
                raise ValueError(f"Unknown field_mode='{self.field_mode}'. Use 'raw' or 'delta'.")
            for i, radius in enumerate(r_sub_arr):
                dd_mean = calc_DD_mean_r(radius, field1, field2, pair_window=self.pair_window)
                if self.field_mode == "delta":
                    _xi = dd_mean / RR
                else:
                    _xi = dd_mean / RR - 1.0
                local_dd.append(dd_mean)
                local_xi.append(_xi)
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
            gathered_r = comm.gather(local_r, root=0)
            if rank == 0:
                self.corr2pcf_data.xi = np.array([item for sublist in gathered_xi for item in sublist])
                dd_arr = np.array([item for sublist in gathered_dd for item in sublist])
                self.corr2pcf_data.r = np.array([item for sublist in gathered_r for item in sublist])
                if self.field_mode == "delta":
                    self.corr2pcf_data.dd = None
                    self.corr2pcf_data.delta_dd = dd_arr
                else:
                    self.corr2pcf_data.dd = dd_arr
                    self.corr2pcf_data.delta_dd = None
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
