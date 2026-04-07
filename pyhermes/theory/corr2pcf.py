import time
import pickle

import numpy as np

from pyhermes.io import WindowFunc
from pyhermes.io import ConvolsData
from pyhermes.io import Corr2PCFData
from pyhermes.utils import func_util
from pyhermes.pipeline import TaskBase


def calc_DD_mean_r(radius, convols_data1, convols_data2=None):
    win_params = {"type": "shell", "len_args": {"R": radius}}
    win_shell = WindowFunc(win_params, convols_data1.convols_info)
    if convols_data2:
        res = convols_data1 @ win_shell * convols_data2
    else:
        res = convols_data1 @ win_shell * convols_data1
    return res.as_array().mean()


class Corr_2PCF(TaskBase):

    def __init__(self, param_task):
        self.task_name = str(self.__class__.__name__)
        super().__init__(param_task=param_task)

    def format_params(self):
        # Parameters from json or input
        self.convols_data_path = self.task_params['convols_data_path']
        self.fout_path      = self.task_params['fout_path']
        self.field_mode = self.task_params['field_mode']
        # self.threads        = int(self.task_params['threads'])
        win_params = self.task_params.get('window', None)
        win_params = win_params if win_params['type'] else None
        for i in range(1, 3):
            win_params_i = self.task_params.get(f'window{i}', None)
            win_params_i = win_params_i if win_params_i['type'] else None
            if (not win_params_i) and win_params:
                win_params_i = dict(win_params)
            setattr(self, f'win_params{i}', win_params_i)
        self.r_min             = self.task_params['r_min']
        self.r_max             = self.task_params['r_max']
        self.n_r         = int(self.task_params['n_r'])

    def run(self, convols_data1=None, convols_data2=None, overwrite=False):
        try:
            comm = self.comm
            rank = self.rank
            size = comm.Get_size()
            if rank == 0:
                time_run_1 = time.perf_counter()
            self.format_params()
            self.corr2pcf_data = Corr2PCFData()

            # The convols data now only loaded to rank0
            convols_info1_serialized = None
            convols_info2_serialized = None
            if rank == 0:
                if self.convols_data_path:
                    self.convols_data = ConvolsData(data_path=self.convols_data_path)
                else:
                    if not (convols_data1 and convols_data2):
                        self.logger.error(
                            "No input 'convols_data' provided and 'convols_data_path' is not set. "
                            "Please either pass a ConvolsData instance to run(convols_data=...) "
                            "or specify 'convols_data_path' in task_params."
                        )
                        func_util.safe_exit(1)
                for i, cdata in zip([1, 2], [convols_data1, convols_data2]):
                    if cdata:
                        self.logger.info(f"Loading convols data from argument 'convols_data{i}'")
                        if isinstance(cdata, ConvolsData):
                            setattr(self, f"convols_data{i}", cdata)
                        else:
                            self.logger.error(f"Unexpected input: 'convols_data{i}' is not an instance of 'ConvolsData'. This should not have happened, program stopped!")
                            func_util.safe_exit(1)
                    else:
                        if _win_params := getattr(self, f"win_params{i}", None):
                            _window = WindowFunc(_win_params, self.convols_data.convols_info)
                            _convols_data = self.convols_data @ _window
                            setattr(self, f"convols_data{i}", _convols_data)
                        else:
                            _convols_data = self.convols_data._spawn_like()
                            _convols_data.epsilon = self.convols_data.epsilon
                            _convols_data.format_convols_params()
                            setattr(self, f"convols_data{i}", self.convols_data)
                    setattr(self.corr2pcf_data, f"convols_info{i}", _convols_data.convols_info)
                _corr2pcf_info = {
                    **self.task_params,
                }
                self.corr2pcf_data.corr2pcf_info = dict(_corr2pcf_info)
                self.corr2pcf_data.convols_info1 = self.convols_data1.convols_info
                convols_info1_serialized = pickle.dumps(self.convols_data1.convols_info)
                self.corr2pcf_data.convols_info2 = self.convols_data2.convols_info
                convols_info2_serialized = pickle.dumps(self.convols_data2.convols_info)
            # Broadcast parameters (read + from convols data) to all ranks
            convols_info1_serialized = comm.bcast(convols_info1_serialized, root=0)
            convols_info2_serialized = comm.bcast(convols_info2_serialized, root=0)
            if rank == 0:
                self.convols_data1.epsilon = np.ascontiguousarray(self.convols_data1.epsilon, dtype=np.float64)
                _local_convols1 = self.convols_data1
                self.convols_data2.epsilon = np.ascontiguousarray(self.convols_data2.epsilon, dtype=np.float64)
                _local_convols2 = self.convols_data2
            else:
                _local_convols1 = ConvolsData()
                _local_convols1.convols_info = pickle.loads(convols_info1_serialized)
                _local_convols1.format_convols_params()
                _local_convols1.epsilon = np.empty((_local_convols1.L, _local_convols1.L, _local_convols1.L), dtype=np.float64)
                _local_convols2 = ConvolsData()
                _local_convols2.convols_info = pickle.loads(convols_info2_serialized)
                _local_convols2.format_convols_params()
                _local_convols2.epsilon = np.empty((_local_convols2.L, _local_convols2.L, _local_convols2.L), dtype=np.float64)
            self.corr2pcf_data.task_params = self.task_params
            # Broadcast epsilon to all rank
            comm.Bcast(_local_convols1.epsilon, root=0)
            comm.Bcast(_local_convols2.epsilon, root=0)
            comm.Barrier()
            if rank == 0:
                self.logger.info("Start to calculate 2PCF ...")
                self.logger.info(
                    f"field_mode={self.field_mode}, n_r={self.n_r}, "
                    f"r_min={self.r_min}, r_max={self.r_max}"
                )
                time_start = time.perf_counter()
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
            R = 1 / _local_convols1.V
            RR = R ** 2
            if self.field_mode == "delta":
                field1 = _local_convols1 - R
                field2 = _local_convols2 - R
            elif self.field_mode == "raw":
                field1 = _local_convols1
                field2 = _local_convols2
            else:
                raise ValueError(f"Unknown field_mode='{self.field_mode}'. Use 'raw' or 'delta'.")
            for i, radius in enumerate(r_sub_arr):
                dd_mean = calc_DD_mean_r(radius, field1, field2)
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
                if self.fout_path:
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
