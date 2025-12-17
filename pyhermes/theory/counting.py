import os
import time
import pickle

import numpy as np

from pyhermes.io import CountingData
from pyhermes.io import ConvolsData
from pyhermes.utils import func_util
from pyhermes.utils import math_util
from pyhermes.pipeline import TaskBase



class Counting(TaskBase):

    def __init__(self, param_task):
        self.task_name = str(self.__class__.__name__)
        super().__init__(param_task=param_task)

    def format_params_input(self):
        # Parameters from json or input
        self.n_tasks        = int(self.task_params['n_tasks'])
        self.deltac_in_path = self.task_params['deltac_in_path']
        self.fout_path      = self.task_params['fout_path']

    def format_params_deltac(self):
        # Parameters inherited from DeltaC
        self.J             = self.task_params['J']
        self.SampRate      = int(self.task_params['SampRate'])
        self.SimBoxL       = self.task_params['SimBoxL']
        self.wavelet_mode  = self.task_params['wavelet_mode']
        self.wavelet_level = self.task_params['wavelet_level']
        self.window_type   = self.task_params['window']['type']
        self.window_args   = {key : float(value) for key, value in self.task_params['window'].items() if key != 'type'}
        self.L             = 1 << self.J

    def run(self, deltac=None):
        try:
            comm = self.comm
            rank = self.rank
            size = comm.Get_size()
            if rank == 0:
                time_run_1 = time.perf_counter()
            self.format_params_input()
            self.counting = CountingData()
            # Naïve optimization scheme for average-MPI
            base_tasks = self.n_tasks // size
            extra_tasks = self.n_tasks % size
            if extra_tasks > 0:
                rank == 0 and self.logger.info("Naïve optimization scheme for average-MPI is adopted")
                total_padded_tasks = (base_tasks + 1) * size 
            else:
                total_padded_tasks = self.n_tasks
            # _local_n_tasks = self.n_tasks // size
            _local_n_tasks = total_padded_tasks // size
            # The deltac now only loaded to rank0
            params_serialized = None
            if rank == 0:
                if not deltac:
                    self.counting.load_deltac(f_in=self.deltac_in_path, single=True)
                else:
                    rank == 0 and self.logger.info("Loading DeltaC from argument 'deltac'")
                    if isinstance(deltac, ConvolsData):
                        self.counting.deltac = deltac.data
                        self.counting.dict_inht_vonDeltac = deltac.dict_inht_vonDeltac
                        self.task_params['deltac_in_path'] = 'load from argument'
                    else:
                        rank == 0 and self.logger.error("Unexpected input: 'deltac' is not an instance of 'ConvolsData'. This should not have happened, program stopped!")
                        func_util.safe_exit(1)
                self.task_params.update(self.counting.dict_inht_vonDeltac)
                params_serialized = pickle.dumps(self.task_params)
            # Broadcast parameters (read + from DeltaC) to all ranks
            params_serialized = comm.bcast(params_serialized, root=0)
            self.task_params = pickle.loads(params_serialized)
            self.format_params_deltac()
            self.counting.task_params = self.task_params
            self.phi_data = math_util.do_wavelet(self.wavelet_mode, self.wavelet_level)
            if rank != 0 :
                self.counting.deltac = np.empty((self.L, self.L, self.L), dtype=np.float64)
            # Broadcast deltac to all rank
            comm.Bcast(self.counting.deltac, root=0)
            comm.Barrier()
            rank == 0 and self.logger.info("Start Counting ... ")
            end_time1 = time.perf_counter()
            self.counting.data = math_util.result_interpret2(self.counting.deltac, _local_n_tasks, self.phi_data, self.J, self.SampRate)
            if rank == 0:
                # _data_all = np.empty(self.n_tasks, dtype=np.float64)
                _data_all = np.empty(total_padded_tasks, dtype=np.float64)
            else:
                _data_all = None
            rank == 0 and self.logger.info("Gathering data from all ranks ... ")
            comm.Gather(self.counting.data, _data_all, root=0)
            if rank == 0:
                self.counting.data = _data_all[:self.n_tasks]
            end_time2 = time.perf_counter()
            rank == 0 and self.logger.info(f"The time for counting is: {end_time2 - end_time1:.4f} sec")
            # Output the couting
            self.counting.save(self.fout_path)
        except Exception as e:
            self.logger.error(f"Error in process {self.rank}: {str(e)}")
            func_util.safe_exit(1)
        if self.rank == 0:
            time_run_2 = time.perf_counter()
            print("")
            self.logger.info(f"The time for task: {time_run_2 - time_run_1:.4f} sec")
        # The data(s) below ⬇ are only valid on rank 0
        return self.counting
                