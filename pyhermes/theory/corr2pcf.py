import time
import pickle

import numpy as np

from pyhermes.io import WindowFunc
from pyhermes.io import ConvolsData
from pyhermes.io import Corr2PCFData
from pyhermes.utils import func_util
from pyhermes.utils import math_util
from pyhermes.pipeline import TaskBase



class Corr_2PCF(TaskBase):

    def __init__(self, param_task):
        self.task_name = str(self.__class__.__name__)
        super().__init__(param_task=param_task)

    def format_params(self):
        # Parameters from json or input
        self.convols_data_path = self.task_params['convols_data_path']
        self.fout_path      = self.task_params['fout_path']
        # self.threads        = int(self.task_params['threads'])
        win_params = self.task_params.get('window', None)
        self.win_params = win_params if win_params['type'] else None
        self.R1             = self.task_params['R1']
        self.R2             = self.task_params['R2']
        self.xi_num         = int(self.task_params['xi_num'])

    def run(self, convols_data=None, overwrite=False):
        try:
            comm = self.comm
            rank = self.rank
            size = comm.Get_size()
            if rank == 0:
                time_run_1 = time.perf_counter()
            self.format_params()
            self.corr2pcf_data = Corr2PCFData()

            # The convols data now only loaded to rank0
            convols_info_serialized = None
            if rank == 0:
                if not convols_data:
                    if self.convols_data_path:
                        self.convols_data = ConvolsData(data_path=self.convols_data_path)
                    else:
                        self.logger.error(
                            "No input 'convols_data' provided and 'convols_data_path' is not set. "
                            "Please either pass a ConvolsData instance to run(convols_data=...) "
                            "or specify 'convols_data_path' in task_params."
                        )
                        func_util.safe_exit(1)
                else:
                    self.logger.info("Loading convols data from argument 'convols_data'")
                    if isinstance(convols_data, ConvolsData):
                        self.convols_data = convols_data
                        self.convols_data.convols_info['convols_data_paht'] = 'load from argument'
                        self.task_params['convols_data_path'] = 'load from argument'
                    else:
                        self.logger.error("Unexpected input: 'convols_data' is not an instance of 'ConvolsData'. This should not have happened, program stopped!")
                        func_util.safe_exit(1)
                if self.win_params:
                    self.window = WindowFunc(self.win_params, self.convols_data.convols_info)
                    self.convols_data = self.convols_data @ self.window
                _corr2pcf_info = {
                    **self.task_params,
                    # "convols_info": self.convols_data.convols_info,
                }
                self.corr2pcf_data.corr2pcf_info = dict(_corr2pcf_info)
                self.corr2pcf_data.convols_info = self.convols_data.convols_info
                convols_info_serialized = pickle.dumps(self.convols_data.convols_info)
            # Broadcast parameters (read + from convols data) to all ranks
            convols_info_serialized = comm.bcast(convols_info_serialized, root=0)
            if rank == 0:
                self.convols_data.epsilon = np.ascontiguousarray(self.convols_data.epsilon, dtype=np.float64)
                _local_convols = self.convols_data
            else:
                _local_convols = ConvolsData()
                _local_convols.convols_info = pickle.loads(convols_info_serialized)
                _local_convols.format_convols_params()
                _local_convols.epsilon = np.empty((_local_convols.L, _local_convols.L, _local_convols.L), dtype=np.float64)
            self.corr2pcf_data.task_params = self.task_params
            # Broadcast epsilon to all rank
            comm.Bcast(_local_convols.epsilon, root=0)
            comm.Barrier()
            if rank == 0:
                self.logger.info("Start to calculate 2PCF ...")
                time_start = time.perf_counter()
            # Generate r_arr at rank0
            if rank == 0:
                r_arr = np.linspace(self.R1, self.R2, self.xi_num)
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
            local_r = []
            R = 1 / _local_convols.V
            RR = R ** 2
            DsubR = _local_convols - R
            for i, radius in enumerate(r_sub_arr):
                win_params = {"type": "shell", "len_args": {"R": radius}}
                win_shell = WindowFunc(win_params, _local_convols.convols_info)
                DsubR_square = DsubR @ win_shell * DsubR
                _xi = DsubR_square.as_array().mean() / RR
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
            gathered_r = comm.gather(local_r, root=0)
            if rank == 0:
                self.corr2pcf_data.xi = np.array([item for sublist in gathered_xi for item in sublist])
                self.corr2pcf_data.r = np.array([item for sublist in gathered_r for item in sublist])
                if not count_all:
                    progress = 100.
                    self.logger.info(f" Progress: {progress:6.2f}%")
                time_end = time.perf_counter()
                self.logger.info(f"The time for 2PCF: {time_end - time_start:.4f} sec")
                # Output the 2pcf
                self.corr2pcf_data.saveflag = True
                self.corr2pcf_data.save_corr2pcf(self.fout_path, overwrite=overwrite) 
        except Exception as e:
            self.logger.error(f"Error in process {self.rank}: {str(e)}")
            func_util.safe_exit(1)
        if self.rank == 0:
            time_run_2 = time.perf_counter()
            print("")
            self.logger.info(f"The time for task: {time_run_2 - time_run_1:.4f} sec")
        # The data(s) below ⬇ are only valid on rank 0
        return self.corr2pcf_data