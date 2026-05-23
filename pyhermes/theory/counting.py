import os
import time
import pickle
import copy

import numpy as np

from pyhermes.io import CountingData, ConvolsData, WindowFunc
from pyhermes.utils import func_util
from pyhermes.utils.sampling import random_box_positions
from pyhermes.utils.window_params import serialize_window_params
from pyhermes.pipeline import TaskBase



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
        self.convols_data     = self.task_params.get('convols_data', '')
        self.random_count     = int(self.task_params['random_count'])
        self.seed             = int(self.task_params['seed'])
        window = self.task_params.get('window', None)
        self.window = window if (window and window.get('type')) else None
        self.threads          = int(self.task_params['threads'])
        self.fout_path      = self.task_params['fout_path']

    def _sync_runtime_options(self):
        self.threads = max(1, int(self.threads))
        self.task_params = {
            'convols_data': self.convols_data,
            'random_count': self.random_count,
            'seed': self.seed,
            'window': self._serialize_window_input(self.window),
            'threads': self.threads,
            'fout_path': self.fout_path,
        }
        self.sync_runtime_options(context="Counting runtime configuration", blank_line=True)

    def _serialize_convols_input(self, value):
        if isinstance(value, str):
            return value
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
        params = {}
        params['convols_data'] = self._serialize_convols_input(self.convols_data)
        params['random_count'] = self.random_count
        params['seed'] = self.seed
        params['window'] = self._serialize_window_input(self.window)
        params['threads'] = self.threads
        params['fout_path'] = self.fout_path
        return params

    def prepare_input_fields(self, convols_data=None, window=None):
        self.counting_data = CountingData()
        self._sync_runtime_options()
        if convols_data is None:
            convols_data = self.convols_data
        if window is None:
            window = self.window

        if self.rank == 0:
            self.logger.info("Preparing Counting input field ...")
            self.logger.info(
                f"random_count={self.random_count}, seed={self.seed}, threads={self.threads}"
            )
            if convols_data is not None:
                if isinstance(convols_data, str):
                    base_convols = ConvolsData(data_path=convols_data, threads=self.threads)
                    source_desc = f"path={convols_data}"
                elif isinstance(convols_data, ConvolsData):
                    base_convols = convols_data
                    source_desc = "provided convols_data"
                else:
                    self.logger.error("Unexpected input: 'convols_data' must be a string path or a ConvolsData instance.")
                    func_util.safe_exit(1)
            else:
                if not self.convols_data:
                    self.logger.error(
                        "No input 'convols_data' provided and shared 'convols_data' is not set. "
                        "Please either pass a ConvolsData instance to prepare_input_fields(convols_data=...) "
                        "or specify 'convols_data' in task_params."
                    )
                    func_util.safe_exit(1)
                if isinstance(self.convols_data, str):
                    base_convols = ConvolsData(data_path=self.convols_data, threads=self.threads)
                    source_desc = f"path={self.convols_data}"
                elif isinstance(self.convols_data, ConvolsData):
                    base_convols = self.convols_data
                    source_desc = "provided convols_data"
                else:
                    self.logger.error("Unexpected task attribute 'convols_data': expected string path or ConvolsData.")
                    func_util.safe_exit(1)

            if window is not None:
                if isinstance(window, WindowFunc):
                    window_obj = window
                    window_desc = "provided WindowFunc instance"
                elif isinstance(window, dict):
                    window_obj = WindowFunc(window, base_convols.convols_info, threads=self.threads)
                    window_desc = f"provided window dict | {func_util.describe_window_action(window)}"
                else:
                    self.logger.error(
                        f"Unsupported window input. Expected dict, WindowFunc, or None, got {type(window)}."
                    )
                    func_util.safe_exit(1)
            else:
                window_obj = None
                window_desc = "no additional window convolution"

            if window_obj is not None:
                self.convols_data = base_convols @ window_obj
            else:
                self.convols_data = base_convols.copy()
                self.convols_data.format_convols_params()
            self.window = window_obj

            self.logger.info(f"Counting field ready | source={source_desc} | window={window_desc}")
            self.counting_data.counting_info = self._current_task_params_snapshot()
            self.counting_data.convols_info = self.convols_data.convols_info

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
            # The convols data now only loaded to rank0
            convols_info_serialized = None
            if rank == 0:
                convols_info_serialized = pickle.dumps(self.convols_data.convols_info)
            # Broadcast parameters (read + from convols data) to all ranks
            convols_info_serialized = comm.bcast(convols_info_serialized, root=0)
            if rank == 0:
                self.convols_data.epsilon = np.ascontiguousarray(self.convols_data.epsilon, dtype=np.float64)
                _local_convols = self.convols_data
            else:
                _local_convols = ConvolsData(threads=self.threads)
                _local_convols.convols_info = pickle.loads(convols_info_serialized)
                _local_convols.format_convols_params()
                _local_convols.epsilon = np.empty((_local_convols.L, _local_convols.L, _local_convols.L), dtype=np.float64)
            self.counting_data.counting_info = self._current_task_params_snapshot()
            self.counting_data.task_params = self._current_task_params_snapshot()
            # Broadcast epsilon to all rank
            comm.Bcast(_local_convols.epsilon, root=0)
            comm.Barrier()
            rank == 0 and self.logger.info("Start Counting ... ")
            end_time1 = time.perf_counter()
            
            # --- generate random positions on each rank ---
            # assume positions are uniform in the simulation box [0, box_size)
            pos = random_box_positions(_local_n_tasks, _local_convols.box_size, seed=self.seed + rank)
            # --- evaluate number density at positions ---
            _data_local = _local_convols.n_at_pos(pos).astype(np.float64, copy=False)

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
                
