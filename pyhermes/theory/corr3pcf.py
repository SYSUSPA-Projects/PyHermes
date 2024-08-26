import os
import time
import pickle

import numpy as np

from pyhermes.io import Corr3PCFData
from pyhermes.io import ColvolsData
from pyhermes.io import read_particle_data
from pyhermes.utils import func_util
from pyhermes.utils import math_util
from pyhermes.utils.mpi_util import MPI
from pyhermes.pipeline import pipeline as pipeline



class Corr_3PCF(pipeline.TaskBase):

    def __init__(self, param_task):
        self.task_name = str(self.__class__.__name__)
        super().__init__(param_task=param_task)

    def format_params_input(self):
        # Parameters from json or input
        self.fout_dir       = self.task_params['fout_dir']
        self.deltac_in_path = self.task_params['deltac_in_path']
        self.fin_path       = self.task_params['fin']['path']
        self.fin_format     = self.task_params['fin']['format']
        self.NStheta        = int(self.task_params['NStheta'])
        self.R1             = self.task_params['R1']
        self.R2             = self.task_params['R2']
        self.rot_num        = int(self.task_params['rot_num'])

    def format_params_deltac(self):
        # Parameters inherited from DeltaC
        self.J             = self.task_params['J']
        self.SampRate      = int(self.task_params['SampRate'])
        self.SimBoxL       = self.task_params['SimBoxL']
        self.wavelet_mode  = self.task_params['wavelet_mode']
        self.wavelet_level = self.task_params['wavelet_level']
        self.Radius        = self.task_params['Radius']
        self.bandwidth     = self.task_params['bandwidth']
        self.window_type   = self.task_params['window_type']
        self.orgDsize      = self.task_params['orgDsize']
        self.L             = 1 << self.J
        self.ScaleFactor   = self.L / self.SimBoxL

    def run(self, deltac=None, p_dm=None):
        try:
            comm = self.comm
            rank = self.rank
            size = comm.Get_size()
            if rank == 0:
                time_run_1 = time.perf_counter()
            self.format_params_input()
            self.corr_3pcf = Corr3PCFData()
            # The deltac and particle only loaded to rank0
            params_serialized = None
            if rank == 0:
                if not deltac:
                    self.corr_3pcf.load_deltac(f_in=self.deltac_in_path, single=True)
                else:
                    rank == 0 and self.logger.info("Loading DeltaC from argument 'deltac'")
                    if isinstance(deltac, ColvolsData):
                        self.corr_3pcf.deltac = deltac.data
                        self.corr_3pcf.dict_inht_vonDeltac = deltac.dict_inht_vonDeltac
                    else:
                        rank == 0 and self.logger.error("Unexpected input: 'deltac' is not an instance of 'ConvolsData'. This should not have happened, program stopped!")
                        func_util.safe_exit(1)
                self.task_params.update(self.corr_3pcf.dict_inht_vonDeltac)
                params_serialized = pickle.dumps(self.task_params)
            # Broadcast parameters (read + from DeltaC) to all ranks
            params_serialized = self.comm.bcast(params_serialized, root=0)
            self.task_params = pickle.loads(params_serialized)
            self.format_params_deltac()
            self.corr_3pcf.task_params = self.task_params
            self.phi_data = math_util.do_wavelet(self.wavelet_mode, self.wavelet_level)
            _PhiStart = 0
            _PhiEnd = self.phi_data.shape[0] // self.SampRate
            self.PhiSupport = _PhiEnd - _PhiStart 
            self.step = np.arange(self.PhiSupport) * self.SampRate
            if rank == 0:
                if p_dm is None:
                    if self.fin_path == self.corr_3pcf.dict_inht_vonDeltac['fin_path']:
                        pass
                        if self.fin_format == self.corr_3pcf.dict_inht_vonDeltac['fin_format']:
                            pass
                        else:
                            self.logger.warning(f"Input particle format '{self.fin_format}' mismatch the Convols DeltaC format '{self.corr_3pcf.dict_inht_vonDeltac['fin_format']}'")
                            self.logger.warning("Is this discrepancy expected?")
                    else:
                        self.logger.warning(f"Input particle file path '{self.fin_path}' mismatch the Convols DeltaC particle file path '{self.corr_3pcf.dict_inht_vonDeltac['fin_path']}'")
                        self.logger.warning("Is this discrepancy expected?")
                    p_dm, _ = read_particle_data(self.fin_path, self.fin_format)
                else:
                    self.logger.info("Loading Particle data from argument 'p_dm'")
                    self.task_params['fin']['path'] = 'load from argument'
                    self.task_params['fin']['format'] = 'load from argument'
                self.task_params['orgDsize_3pcf'] = p_dm.shape[0]
                if p_dm.shape[1] != 3:
                    self.logger.error("Wrong shape of input particle catalog data! The shape should be (*,3)")
                    func_util.safe_exit(1)
                elif p_dm.shape[0] != self.orgDsize:
                    self.logger.error(f"Input particle size {p_dm.shape[0]} mismatch detected in Convols DeltaC calculation size {self.orgDsize}")
                    self.logger.error("Do you use the same particle data?")
                    self.logger.error("This should not have happend, program stopped!")
                    func_util.safe_exit(1)
                random_data = np.random.rand(self.orgDsize, 3) * self.SimBoxL
                rows_per_rank = self.orgDsize // size
                remainder = self.orgDsize % size
            else:
                self.corr_3pcf.deltac = np.empty((self.L, self.L, self.L), dtype=np.float64)
                p_dm = None
                random_data = None
                rows_per_rank = None
                remainder = None
            # Broadcast n_rows, rows_per_rank, remainder
            rows_per_rank = comm.bcast(rows_per_rank, root=0)
            remainder = comm.bcast(remainder, root=0)
            if rank == 0:
                for i in range(1, size):
                    if i < remainder:
                        start_idx = i * (rows_per_rank + 1)
                        end_idx = start_idx + rows_per_rank + 1
                    else:
                        start_idx = i * rows_per_rank + remainder
                        end_idx = start_idx + rows_per_rank
                    comm.Send(p_dm[start_idx:end_idx], dest=i)
                    comm.Send(random_data[start_idx:end_idx], dest=i)
                start_idx = 0
                end_idx = rows_per_rank + 1 if remainder > 0 else rows_per_rank
                p_dm_local = p_dm[start_idx:end_idx]
                ran_local = random_data[start_idx:end_idx]
            else:
                local_n_rows = (self.orgDsize // size) + 1 if rank < remainder else (self.orgDsize // size)
                p_dm_local = np.empty((local_n_rows, 3), dtype=np.float32)
                ran_local = np.empty((local_n_rows, 3), dtype=np.float64)
                comm.Recv(p_dm_local, source=0)
                comm.Recv(ran_local, source=0)
            # Broadcast deltac to all rank
            comm.Bcast(self.corr_3pcf.deltac, root=0)
            # Set random seed for all ranks, respectively
            np.random.seed(rank)
            comm.Barrier()
            if rank == 0:
                self.corr_3pcf.Q = []
                self.corr_3pcf.theta = []
                self.logger.info("Start to calculate 3PCF ...")
                start_time_ini = time.perf_counter()
            R1_scale = self.R1 * self.ScaleFactor
            R2_scale = self.R2 * self.ScaleFactor
            theta_values = np.linspace(0, np.pi, self.NStheta + 1, endpoint=True)
            if rank == 0:
                arr_complete = np.zeros(size, dtype=int)
                total_tasks = (self.NStheta + 1) * size
                report_interval = total_tasks // 10
                next_report_threshold = 0
                requests = [comm.irecv(source=r, tag=r) for r in range(size)]
                count_all = False
            local_completed = 0
            local_report_interval = max(1, (self.NStheta + 1) // 10)
            for theta in theta_values[:-1]:
                result = math_util.result_3pcf_cpu_location(self.corr_3pcf.deltac, self.phi_data, self.step, p_dm_local, ran_local, self.rot_num, R1_scale, R2_scale, theta)
                result_sum = np.sum(result)
                result_sum = comm.reduce(result_sum, op=MPI.SUM, root=0)
                local_completed += 1
                if local_completed % local_report_interval == 0:
                    comm.isend(local_completed, dest=0, tag=rank)
                if rank == 0:
                    rho = self.orgDsize / self.L**3
                    _Q = result_sum / self.orgDsize / self.rot_num / rho**2
                    # self.logger.info(f" theta = {theta:8.4f}, Q = {_Q:8.4f}")
                    self.corr_3pcf.Q.append(_Q)
                    self.corr_3pcf.theta.append(theta)
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
            if rank == 0:
                if not count_all:
                    progress = 100.
                    self.logger.info(f" Progress: {progress:6.2f}%")
                end_time = time.perf_counter()
                self.logger.info('Finished in {:.1f} sec'.format((end_time - start_time_ini)))
                self.logger.info(f"The time for 3PCF: {end_time - start_time_ini:.4f} sec")
                if self.fout_dir is not None and self.fout_dir != "":
                    self.corr_3pcf.saveflag = True
                    _fout_path = os.path.join(self.fout_dir, f"corr3pcf_r{str(self.Radius)}_R1.{str(self.R1)}_R2.{str(self.R2)}_rotN{str(self.rot_num)}.txt")
                    self.corr_3pcf.save(_fout_path)
        except Exception as e:
            self.logger.error(f"Error in process {self.rank}: {str(e)}")
            func_util.safe_exit(1)
        if rank == 0:
            time_run_2 = time.perf_counter()
            print("")
            self.logger.info(f"The time for task: {time_run_2 - time_run_1:.4f} sec")
        return self.corr_3pcf