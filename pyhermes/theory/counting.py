import os
import time
import pickle
import copy

import numpy as np

from pyhermes.io import CountingData, ConvolsData, WindowFunc
from pyhermes.utils import func_util, math_util
from pyhermes.pipeline import TaskBase



class Counting(TaskBase):

    def __init__(self, param_task=None):
        if param_task is None:
            param_task = {"Counting": {}}
        self.task_name = str(self.__class__.__name__)
        super().__init__(param_task=param_task)
        self.format_params()
        self.convols_data = None
        self.window = None
        self._fields_prepared = False

    def format_params(self):
        # Parameters from json or input
        self.N_randoms        = int(self.task_params['N_randoms'])
        self.seed             = int(self.task_params['seed'])
        self.convols_data_path             = self.task_params['convols_data_path']
        self.threads          = int(self.task_params['threads'])
        win_params = self.task_params.get('window', None)
        self.win_params = win_params if win_params['type'] else None
        self.fout_path      = self.task_params['fout_path']

    def _sync_runtime_options(self):
        self.threads = max(1, int(self.threads))
        self.task_params['threads'] = self.threads
        self.task_params['N_randoms'] = self.N_randoms
        self.task_params['seed'] = self.seed
        self.task_params['convols_data_path'] = self.convols_data_path
        self.task_params['fout_path'] = self.fout_path
        self.sync_runtime_options(context="Counting runtime configuration", blank_line=True)

    def _current_task_params_snapshot(self):
        params = copy.deepcopy(self.task_params)
        params['N_randoms'] = self.N_randoms
        params['seed'] = self.seed
        params['convols_data_path'] = self.convols_data_path
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
                f"N_randoms={self.N_randoms}, seed={self.seed}, threads={self.threads}"
            )
            if convols_data is not None:
                if not isinstance(convols_data, ConvolsData):
                    self.logger.error("Unexpected input: 'convols_data' is not an instance of 'ConvolsData'.")
                    func_util.safe_exit(1)
                base_convols = convols_data
                source_desc = "provided convols_data"
            else:
                if not self.convols_data_path:
                    self.logger.error(
                        "No input 'convols_data' provided and 'convols_data_path' is not set. "
                        "Please either pass a ConvolsData instance to prepare_input_fields(convols_data=...) "
                        "or specify 'convols_data_path' in task_params."
                    )
                    func_util.safe_exit(1)
                base_convols = ConvolsData(data_path=self.convols_data_path, threads=self.threads)
                source_desc = f"path={self.convols_data_path}"

            if isinstance(window, WindowFunc):
                window_obj = window
                window_desc = "provided WindowFunc instance"
            elif isinstance(window, dict):
                window_obj = WindowFunc(window, base_convols.convols_info, threads=self.threads)
                window_desc = f"provided window dict | {func_util.describe_window_action(window)}"
            elif window is not None:
                self.logger.error(
                    f"Unsupported window input. Expected dict, WindowFunc, or None, got {type(window)}."
                )
                func_util.safe_exit(1)
            elif self.win_params:
                window_obj = WindowFunc(self.win_params, base_convols.convols_info, threads=self.threads)
                window_desc = f"config window | {func_util.describe_window_action(self.win_params)}"
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
            base_tasks = self.N_randoms // size
            extra_tasks = self.N_randoms % size
            if extra_tasks > 0:
                rank == 0 and self.logger.info("Naïve optimization scheme for average-MPI is adopted")
                total_padded_tasks = (base_tasks + 1) * size 
            else:
                total_padded_tasks = self.N_randoms
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
            # assume positions are uniform in the simulation box [0, SimBoxL)
            pos = math_util.random_points_box(_local_n_tasks, _local_convols.SimBoxL, seed=self.seed + rank)
            # --- evaluate number density at positions ---
            _data_local = _local_convols.n_at_pos(pos).astype(np.float64, copy=False)

            if rank == 0:
                _data_all = np.empty(total_padded_tasks, dtype=np.float64)
            else:
                _data_all = None

            rank == 0 and self.logger.info("Gathering data from all ranks ... ")
            comm.Gather(_data_local, _data_all, root=0)

            if rank == 0:
                if total_padded_tasks != self.N_randoms:
                    self.logger.info(f"Padding tasks: computed {total_padded_tasks} samples, keeping {self.N_randoms}.")
                self.counting_data.nx = _data_all[:self.N_randoms]

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
                
