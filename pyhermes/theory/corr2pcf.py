import time
import pickle

import numpy as np

from pyhermes.io import Corr2PCFData
from pyhermes.io import ConvolsData
from pyhermes.base import WindowFunc
from pyhermes.utils import func_util
from pyhermes.utils import math_util
from pyhermes.pipeline import TaskBase



class Corr_2PCF(TaskBase):

    def __init__(self, param_task):
        self.task_name = str(self.__class__.__name__)
        super().__init__(param_task=param_task)

    def format_params_input(self):
        # Parameters from json or input
        self.deltac_in_path = self.task_params['deltac_in_path']
        self.fout_path      = self.task_params['fout_path']
        self.threads        = int(self.task_params['threads'])
        self.R1             = self.task_params['R1']
        self.R2             = self.task_params['R2']
        self.xi_num         = int(self.task_params['xi_num'])

    def format_params_deltac(self):
        # Parameters inherited from DeltaC
        self.J             = self.task_params['J']
        self.SampRate      = int(self.task_params['SampRate'])
        self.SimBoxL       = self.task_params['SimBoxL']
        self.wavelet_mode  = self.task_params['wavelet_mode']
        self.wavelet_level = self.task_params['wavelet_level']
        # self.window_type   = self.task_params['window']['type']
        # self.window_args   = {key : float(value) for key, value in self.task_params['window'].items() if key != 'type'}
        self.bandwidth     = self.task_params['bandwidth']
        self.orgDsize      = self.task_params['orgDsize']
        self.L             = 1 << self.J
        self.DeltaXi       = 1. / self.L

    def run(self, deltac=None):
        try:
            comm = self.comm
            rank = self.rank
            size = comm.Get_size()
            if rank == 0:
                time_run_1 = time.perf_counter()
            self.format_params_input()
            self.corr_2pcf = Corr2PCFData()
            # The deltac now only loaded to rank0
            params_serialized = None
            if rank == 0:
                if not deltac:
                    self.corr_2pcf.load_deltac(f_in=self.deltac_in_path, single=True)
                else:
                    self.logger.info("Loading DeltaC from argument 'deltac'")
                    if isinstance(deltac, ConvolsData):
                        self.corr_2pcf.deltac = deltac.data
                        self.corr_2pcf.dict_inht_vonDeltac = deltac.dict_inht_vonDeltac
                        self.task_params['deltac_in_path'] = 'load from argument'
                    else:
                        rank == 0 and self.logger.error("Unexpected input: 'deltac' is not an instance of 'ConvolsData'. This should not have happened, program stopped!")
                        func_util.safe_exit(1)
                self.task_params.update(self.corr_2pcf.dict_inht_vonDeltac)
                params_serialized = pickle.dumps(self.task_params)
            # Broadcast parameters (read + from DeltaC) to all ranks
            params_serialized = comm.bcast(params_serialized, root=0)
            self.task_params = pickle.loads(params_serialized)
            comm.Barrier()
            self.format_params_deltac()
            self.corr_2pcf.task_params = self.task_params
            self.phi_data = math_util.do_wavelet(self.wavelet_mode, self.wavelet_level)
            self.PowerPhi = math_util.power_spectrum(self.phi_data, 0, self.bandwidth, self.L * self.bandwidth, self.SampRate)
            if rank != 0 :
                self.corr_2pcf.deltac = np.empty((self.L, self.L, self.L), dtype=np.float64)
            # Broadcast deltac to all rank
            comm.Bcast(self.corr_2pcf.deltac, root=0)
            comm.Barrier()
            # Init Global 2pcf results
            self.corr_2pcf.xi = []
            self.corr_2pcf.r = []
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
            for i, radius in enumerate(r_sub_arr):
                rescaleR = radius * self.L / self.SimBoxL
                win_params = dict(L=self.L,
                                    bandwidth=self.bandwidth,
                                    DeltaXi=self.DeltaXi,
                                    PowerPhi=self.PowerPhi,
                                    type='shell', R=rescaleR)
                window_func = WindowFunc(win_params=win_params, threads=self.threads)
                # _w_func = math_util.set_window_function('shell', verbose=False)
                # window_array_shell = math_util.call_calculate_window_array(
                #     L                     = self.L,
                #     bandwidth             = self.bandwidth,
                #     DeltaXi               = self.DeltaXi,
                #     PowerPhi              = self.PowerPhi,
                #     window_function_numba = _w_func,
                #     R                     = rescaleR
                #     )
                # w_shell = math_util.calculate_w_numba(window_array_shell)
                # s_sphere_shell = self.specialized_convolution_3d(self.corr_2pcf.deltac, w_shell, self.threads)
                # s_sphere_shell = window_func @ self.corr_2pcf.deltac
                deltac = ConvolsData()
                deltac.deltac = self.corr_2pcf.deltac
                s_sphere_shell = deltac @ window_func
                inner_sum = np.sum(s_sphere_shell * self.corr_2pcf.deltac) * self.L**3 / self.orgDsize **2 - 1
                local_xi.append(inner_sum)
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
                self.corr_2pcf.xi = [item for sublist in gathered_xi for item in sublist]
                self.corr_2pcf.r = [item for sublist in gathered_r for item in sublist]
                if not count_all:
                    progress = 100.
                    self.logger.info(f" Progress: {progress:6.2f}%")
                time_end = time.perf_counter()
                self.logger.info(f"The time for 2PCF: {time_end - time_start:.4f} sec")
                # Output the 2pcf
                self.corr_2pcf.saveflag = True
                self.corr_2pcf.save(self.fout_path) 
        except Exception as e:
            self.logger.error(f"Error in process {self.rank}: {str(e)}")
            func_util.safe_exit(1)
        print('Caonima')
        if self.rank == 0:
            time_run_2 = time.perf_counter()
            print("")
            self.logger.info(f"The time for task: {time_run_2 - time_run_1:.4f} sec")
        # The data(s) below ⬇ are only valid on rank 0
        return self.corr_2pcf