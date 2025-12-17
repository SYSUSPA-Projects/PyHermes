import os
import time
import concurrent.futures

import numpy as np

from pyhermes.io import ConvolsData
from pyhermes.io import read_particle_data
from pyhermes.utils import func_util
from pyhermes.utils import math_util
from pyhermes.pipeline import TaskBase



class Convols(TaskBase):

    def __init__(self, param_task):
        self.task_name = str(self.__class__.__name__)
        super().__init__(param_task=param_task)

    def format_params(self):
        self.J             = self.task_params['J']
        self.fin_path      = self.task_params['fin']['path']
        self.fin_format    = self.task_params['fin']['format']
        self.fout_path     = self.task_params['fout_path']
        self.SampRate      = int(self.task_params['SampRate'])
        self.SimBoxL       = self.task_params['SimBoxL']
        self.wavelet_mode  = self.task_params['wavelet_mode']
        self.wavelet_level = self.task_params['wavelet_level']
        # self.window_type   = self.task_params['window']['type']
        # self.window_args   = {key : float(value) for key, value in self.task_params['window'].items() if key != 'type'}
        self.bandwidth     = self.task_params['bandwidth']
        self.threads       = int(self.task_params['threads'])
        self.L             = 1 << self.J

    def run(self, return_pData=False): 
        try:
            comm = self.comm
            rank = self.rank
            _system_cpu = 2 ** int(np.log2(os.cpu_count()))
            size = comm.Get_size() if _system_cpu >= comm.Get_size() else _system_cpu
            p_dm = None
            if rank == 0:
                time_run_1 = time.perf_counter()
            # !NOTICE: the MPI-rank num to calculate scaling coefficient
            # !          should be a power of two
                if self.size != 1 and (self.size & (self.size - 1)) != 0:
                    self.logger.error(f"MPI rank number {self.size} is not a power of two. Please adjust your configuration.")
                    func_util.safe_exit(1)
                elif self.size > _system_cpu:
                    self.logger.warning(f"MPI rank number {self.size} is larger than the system CPU number(2^) {_system_cpu}. Using the system CPU number of {_system_cpu} instead.")
            self.size = size
            # Retrive parameters to locals
            self.format_params()
            # Init deltac instance
            self.deltac = ConvolsData(threads=self.threads)
            # Do wavelet transform
            self.phi_data = math_util.do_wavelet(self.wavelet_mode, self.wavelet_level)
            _PhiStart = 0
            _PhiEnd = self.phi_data.shape[0] // self.SampRate
            self.PhiSupport = _PhiEnd - _PhiStart 
            self.core_width = self.L // self.size
            if rank == 0 :
                # Here we expose the p_dm data(in) interface for better illustion in paper, Figure 8.
                if self.task_params['particle_pos'] != None:
                    p_dm = self.task_params['particle_pos']
                    if isinstance(p_dm, np.ndarray) and p_dm.ndim == 2 and p_dm.shape[1] == 3:
                        _orgDsize = p_dm.shape[0]
                    else:
                        self.logger.error("Wrong input of particle data! 'particle_pos' must be a 2D array of shape (N, 3), but got", type(p_dm), "with shape", getattr(p_dm, 'shape', None))
                        func_util.safe_exit(1)
                else:
                    # Read particle data, the origin data size only store in rank 0
                    p_dm, _orgDsize = read_particle_data(self.fin_path, self.fin_format)
                _ScaleFactor = self.L / self.SimBoxL
                if p_dm.shape[1] != 3:
                    self.logger.error("Wrong shape of input particle catalog data! The shape should be (*,3)")
                    func_util.safe_exit(1)
                # self.deltac.orgData = p_dm
                if size == 1:
                    self.logger.info("Single process mode")
                    time_start = time.perf_counter()
                    _deltas = math_util.scaling_function_numba(
                        p          = p_dm,
                        phi_data   = self.phi_data,
                        SampRate   = self.SampRate,
                        J          = self.J,
                        SimBoxL    = self.SimBoxL
                        )
                else:
                    self.logger.info("Multi-process mode")
                    self.logger.info("Start partition ... ")
                    p_dm_in = math_util.int_data(p_dm, _ScaleFactor)
                    _shrink_p_dm_in = math_util.bit(p_dm_in, self.J, int(np.log2(self.size)))
                    time_start = time.perf_counter()
                    with concurrent.futures.ThreadPoolExecutor(max_workers=self.size) as executor: 
                        shrink_list = list(executor.map(lambda num: math_util.partition_data_single(p_dm, _shrink_p_dm_in, num), range(self.size)))
                    time_end = time.perf_counter()
                    self.logger.info(f"The time for partition data: {time_end - time_start:.4f} sec")
                    data_sub_part = shrink_list[0]
                    self.all_s  = np.zeros((self.size, self.L // self.size + 2 * (self.PhiSupport - 1), self.L, self.L))
                    for i in range(1, self.size):
                        comm.send(shrink_list[i].shape, dest=i)
                        comm.Send(shrink_list[i], dest=i)
            elif rank > 0:
                self.orgDsize = 0
                self.all_s = None
                shrink_list = None
                shape = comm.recv(source=0)
                data_sub_part = np.empty(shape, dtype=np.float32)
                comm.Recv(data_sub_part, source=0)
            if size > 1:
                comm.Barrier()
                rank == 0 and self.logger.info("Start to calculate scaling coefficient... ")
                time_start = time.perf_counter()
                _s_part = math_util.scaling_function_numba_part(
                    part       = rank,
                    p          = data_sub_part,
                    phi_data   = self.phi_data,
                    size       = self.size,
                    core_width = self.core_width,
                    SampRate   = self.SampRate,
                    J          = self.J,
                    SimBoxL    = self.SimBoxL
                    )
                comm.Gather(_s_part, self.all_s, root=0)
                if rank == 0:
                    _deltas = self.sew_up(self.all_s, self.size, self.L,self.PhiSupport)
            if rank == 0:
                _dict_inht_vonDeltac = {
                    "fin_path"     : self.fin_path,
                    "fin_format"   : self.fin_format,
                    "orgDsize"     : _orgDsize,
                    "J"            : self.J,
                    "SampRate"     : self.SampRate,
                    # "window"       : self.task_params['window'],
                    "SimBoxL"      : self.SimBoxL,
                    "bandwidth"    : self.bandwidth,
                    "wavelet_mode" : self.wavelet_mode,
                    "wavelet_level": self.wavelet_level,
                }
                self.deltac.dict_inht_vonDeltac.update(_dict_inht_vonDeltac)
                time_end = time.perf_counter()
                self.logger.info(f"The time for scaling function: {time_end - time_start:.4f} sec")
                # Here we dont conv any window, just keep the orig deltac
                self.deltac.deltac = _deltas
                # Output the deltac
                self.deltac.save(self.fout_path)
        except Exception as e:
            self.logger.error(f"Error in process {self.rank}: {str(e)}")
            func_util.safe_exit(1)
        comm.Barrier()
        if self.rank == 0:
            time_run_2 = time.perf_counter()
            print("")
            self.logger.info(f"The time for task: {time_run_2 - time_run_1:.4f} sec")
        # The data(s) below ⬇ are only valid on rank 0
        if return_pData:
            return self.deltac, p_dm
        else:
            return self.deltac

    def sew_up(self, all_s, size, L, PhiSupport):
        sew_s = np.zeros((L, L, L))
        sew_width = PhiSupport - 1
        for part in range(1, size - 1):
            sew_s[-sew_width+part*self.core_width:(part+1)*self.core_width+sew_width] += all_s[part]
        sew_s[-sew_width:] += all_s[0][:sew_width]
        sew_s[:self.core_width+sew_width] += all_s[0][sew_width:self.core_width+sew_width+sew_width]
        sew_s[-(self.core_width+sew_width):] += all_s[-1][:self.core_width+sew_width]
        sew_s[:sew_width] += all_s[-1][-sew_width:]
        return sew_s
