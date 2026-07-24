import pickle
import time

import numpy as np

from pyhermes.io import (
    CountingData,
    SFCField,
    WindowFunc,
    normalize_task_weight_normalization,
)
from pyhermes.pipeline import TaskBase
from pyhermes.utils import func_util
from pyhermes.utils.sampling import random_box_positions
from pyhermes.utils.window_params import serialize_window_params


class Counting(TaskBase):

    def __init__(self, param_task=None):
        if param_task is None:
            param_task = {"Counting": {}}
        self.task_name = str(self.__class__.__name__)
        super().__init__(param_task=param_task)
        self.format_params()
        self._fields_prepared = False

    def format_params(self):
        # Parameters from json or input
        self.sfc_field     = self.task_params.get('sfc_field', '')
        self.random_count     = int(self.task_params['random_count'])
        self.seed             = int(self.task_params['seed'])
        self.weight_normalization = normalize_task_weight_normalization(self.task_params.get("weight_normalization", "catalog"))
        window = self.task_params.get('window', None)
        self.window = window if (window and window.get('type')) else None
        self.threads          = int(self.task_params['threads'])
        self.fout_path      = self.task_params['fout_path']

    def _sync_runtime_options(self):
        self.threads = max(1, int(self.threads))
        self.task_params = {
            'sfc_field': self.sfc_field,
            'random_count': self.random_count,
            'seed': self.seed,
            'weight_normalization': self.weight_normalization,
            'window': self._serialize_window_input(self.window),
            'threads': self.threads,
            'fout_path': self.fout_path,
        }
        self.sync_runtime_options(context="Counting runtime configuration", blank_line=True)

    def _field_in_task_normalization(self, field):
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

    def _serialize_sfc_input(self, value):
        if isinstance(value, str):
            return value
        if isinstance(value, SFCField):
            return {
                "kind": "SFCField",
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
        params = {}
        params['sfc_field'] = self._serialize_sfc_input(self.sfc_field)
        params['random_count'] = self.random_count
        params['seed'] = self.seed
        params['weight_normalization'] = self.weight_normalization
        params['window'] = self._serialize_window_input(self.window)
        params['threads'] = self.threads
        params['fout_path'] = self.fout_path
        return params

    def prepare_input_fields(self, sfc_field=None, window=None):
        self.counting_data = CountingData()
        self._sync_runtime_options()
        if sfc_field is None:
            sfc_field = self.sfc_field
        if window is None:
            window = self.window

        if self.rank == 0:
            self.logger.info("Preparing Counting input field ...")
            self.logger.info(
                f"random_count={self.random_count}, seed={self.seed}, threads={self.threads}"
            )
            if sfc_field is not None:
                if isinstance(sfc_field, str):
                    base_sfc = SFCField(data_path=sfc_field, threads=self.threads)
                    source_desc = f"path={sfc_field}"
                elif isinstance(sfc_field, SFCField):
                    base_sfc = sfc_field
                    source_desc = "provided sfc_field"
                else:
                    self.logger.error("Unexpected input: 'sfc_field' must be a string path or a SFCField instance.")
                    func_util.safe_exit(1)
            else:
                if not self.sfc_field:
                    self.logger.error(
                        "No input 'sfc_field' provided and shared 'sfc_field' is not set. "
                        "Please either pass a SFCField instance to prepare_input_fields(sfc_field=...) "
                        "or specify 'sfc_field' in task_params."
                    )
                    func_util.safe_exit(1)
                if isinstance(self.sfc_field, str):
                    base_sfc = SFCField(data_path=self.sfc_field, threads=self.threads)
                    source_desc = f"path={self.sfc_field}"
                elif isinstance(self.sfc_field, SFCField):
                    base_sfc = self.sfc_field
                    source_desc = "provided sfc_field"
                else:
                    self.logger.error("Unexpected task attribute 'sfc_field': expected string path or SFCField.")
                    func_util.safe_exit(1)

            if window is not None:
                if isinstance(window, WindowFunc):
                    window_obj = window
                    window_desc = "provided WindowFunc instance"
                elif isinstance(window, dict):
                    window_obj = WindowFunc(window, base_sfc.sfc_info, threads=self.threads)
                    window_desc = f"provided window dict | {func_util.describe_window_action(window)}"
                else:
                    self.logger.error(
                        f"Unsupported window input. Expected dict, WindowFunc, or None, got {type(window)}."
                    )
                    func_util.safe_exit(1)
            else:
                window_obj = None
                window_desc = "no additional window convolution"

            base_sfc = self._field_in_task_normalization(base_sfc)

            if window_obj is not None:
                self.sfc_field = base_sfc @ window_obj
            else:
                self.sfc_field = base_sfc
                self.sfc_field.format_sfc_params()
            self.window = window_obj

            self.logger.info(f"Counting field ready | source={source_desc} | window={window_desc}")
            self.counting_data.counting_info = self._current_task_params_snapshot()
            self.counting_data.sfc_info = self.sfc_field.sfc_info

        self._fields_prepared = True

    def run(self, save_result=True, overwrite=False):
        try:
            comm = self.comm
            rank = self.rank
            size = comm.Get_size()
            if rank == 0:
                time_run_1 = time.perf_counter()
            if not self._fields_prepared:
                self.prepare_input_fields()
            # Naïve optimization scheme for average-MPI
            base_tasks = self.random_count // size
            extra_tasks = self.random_count % size
            if extra_tasks > 0:
                rank == 0 and self.logger.info("Naïve optimization scheme for average-MPI is adopted")
                total_padded_tasks = (base_tasks + 1) * size 
            else:
                total_padded_tasks = self.random_count
            _local_n_tasks = total_padded_tasks // size
            # The SFCField now only loaded to rank0
            sfc_info_serialized = None
            if rank == 0:
                sfc_info_serialized = pickle.dumps(self.sfc_field.sfc_info)
            # Broadcast parameters (read + from SFCField) to all ranks
            sfc_info_serialized = comm.bcast(sfc_info_serialized, root=0)
            if rank == 0:
                self.sfc_field.epsilon = np.ascontiguousarray(self.sfc_field.epsilon, dtype=np.float64)
                _local_sfc = self.sfc_field
            else:
                _local_sfc = SFCField(threads=self.threads)
                _local_sfc.sfc_info = pickle.loads(sfc_info_serialized)
                _local_sfc.format_sfc_params()
                _local_sfc.epsilon = np.empty((_local_sfc.L, _local_sfc.L, _local_sfc.L), dtype=np.float64)
            self.counting_data.counting_info = self._current_task_params_snapshot()
            self.counting_data.task_params = self._current_task_params_snapshot()
            # Broadcast epsilon to all rank
            comm.Bcast(_local_sfc.epsilon, root=0)
            comm.Barrier()
            rank == 0 and self.logger.info("Start Counting ... ")
            end_time1 = time.perf_counter()
            
            # --- generate random positions on each rank ---
            # assume positions are uniform in the simulation box [0, box_size)
            pos = random_box_positions(_local_n_tasks, _local_sfc.box_size, seed=self.seed + rank)
            # --- evaluate number density at positions ---
            _data_local = _local_sfc.field_density_at_pos(pos, value_unit="physical").astype(np.float64, copy=False)

            if rank == 0:
                _data_all = np.empty(total_padded_tasks, dtype=np.float64)
            else:
                _data_all = None

            rank == 0 and self.logger.info("Gathering data from all ranks ... ")
            comm.Gather(_data_local, _data_all, root=0)

            if rank == 0:
                if total_padded_tasks != self.random_count:
                    self.logger.info(f"Padding tasks: computed {total_padded_tasks} samples, keeping {self.random_count}.")
                self.counting_data.nx = _data_all[:self.random_count]

            end_time2 = time.perf_counter()
            rank == 0 and self.logger.info(f"The time for counting is: {end_time2 - end_time1:.4f} sec")            
            # Output the couting
            if rank == 0:
                if save_result and self.fout_path:
                    self.counting_data.save_counting(self.fout_path, overwrite=overwrite)
        except Exception as e:
            self.logger.error(f"Error in process {self.rank}: {str(e)}")
            func_util.safe_exit(1)
        if self.rank == 0:
            time_run_2 = time.perf_counter()
            print("")
            self.logger.info(f"The time for task: {time_run_2 - time_run_1:.4f} sec")
        # The data(s) below ⬇ are only valid on rank 0
        return self.counting_data
                
