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
        self.fin_wei_key   = self.task_params['fin']['weight_key']
        self.fout_path     = self.task_params['fout_path']
        self.SampRate      = int(self.task_params['SampRate'])
        self.SimBoxL       = self.task_params['SimBoxL']
        self.wavelet_mode  = self.task_params['wavelet_mode']
        self.wavelet_level = self.task_params['wavelet_level']
        self.bandwidth     = self.task_params['bandwidth']
        self.threads       = int(self.task_params['threads'])
        self.L             = 1 << self.J

    def run(self, return_pData=False): 
        try:
            comm = self.comm
            rank = self.rank
            p_pos = None
            p_wei = None
            if rank == 0:
                time_run_1 = time.perf_counter()
                # !NOTICE: the MPI-rank num to calculate scaling coefficient
                # !          should be a power of two
                if self.size != 1 and (self.size & (self.size - 1)) != 0:
                    self.logger.error(f"MPI rank number {self.size} is not a power of two. Please adjust your configuration.")
                    func_util.safe_exit(1)
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
                # Here we expose the p_pos, wei data(in) interface for better illustion in paper, Figure 8.
                if self.task_params['particle_pos'] != None and self.task_params['particle_weight'] != None:
                    p_pos = self.task_params['particle_pos']
                    p_wei = self.task_params['particle_weight']
                    self.fin_wei_key = 'weight comes from custom input array'
                    # pos check
                    if not (isinstance(p_pos, np.ndarray) and p_pos.ndim == 2 and p_pos.shape[1] == 3):
                        self.logger.error(f"Wrong input of particle data! 'particle_pos' must be a 2D array of shape (N, 3), but got type={type(p_pos)} with shape={getattr(p_pos, 'shape', None)}.")
                        func_util.safe_exit(1)
                    _orgDsize = p_pos.shape[0]
                    # weight check
                    if np.isscalar(p_wei):
                        self.logger.info(f"Input weight is scalar; broadcasting to a uniform per-particle weight array of length {_orgDsize}.")
                        p_wei = np.full(_orgDsize, p_wei, dtype=np.float32)
                    if not isinstance(p_wei, np.ndarray):
                        self.logger.error(f"Wrong input of particle data! 'particle_weight' must be a numpy array, but got type={type(p_wei)}.")
                        func_util.safe_exit(1)
                    if not (p_wei.ndim == 1 and p_wei.shape[0] == _orgDsize):
                        self.logger.error(f"Wrong input of particle data! 'particle_weight' must have shape (N,), but got shape={getattr(p_wei, 'shape', None)} while N={_orgDsize}.")
                        func_util.safe_exit(1)
                else:
                    p_dict_all = read_particle_data(self.fin_path, self.fin_format)
                    p_pos, _orgDsize = p_dict_all['pos'], p_dict_all['size']
                    if not (isinstance(p_pos, np.ndarray) and p_pos.ndim == 2 and p_pos.shape[1] == 3):
                        self.logger.error(f"Wrong input of particle file data! 'pos' must be a 2D array of shape (N, 3), but got type={type(p_pos)} with shape={getattr(p_pos, 'shape', None)}.")
                        func_util.safe_exit(1)
                    if self.fin_format == 'generic_pos':
                        p_wei = np.ones(_orgDsize, dtype=np.float32)
                        self.fin_wei_key = 'no_weight'
                    elif self.fin_format == 'generic_pos_weight':
                        p_wei = p_dict_all['weight']
                        self.fin_wei_key = 'weight'
                    else:
                        _key = self.fin_wei_key
                        if _key is None or _key == 'no_weight':
                            p_wei = np.ones(_orgDsize, dtype=np.float32)
                            self.fin_wei_key = 'no_weight'
                        elif _key in p_dict_all:
                            p_wei = p_dict_all[_key]
                            self.fin_wei_key = [_key]
                        else:
                            self.logger.warning(f"Weight key '{_key}' not found in particle data. Calculating without weight. Available keys: {list(p_dict_all.keys())}. Use weight_key='no_weight' if no weighting is desired.")
                            p_wei = np.ones(_orgDsize, dtype=np.float32)
                            self.fin_wei_key = 'no_weight'
                    # weight check
                    if np.isscalar(p_wei):
                        self.logger.info(f"Input weight is scalar; broadcasting to a uniform per-particle weight array of length {_orgDsize}.")
                        p_wei = np.full(_orgDsize, p_wei, dtype=np.float32)
                    if not isinstance(p_wei, np.ndarray):
                        self.logger.error(f"Wrong input of particle data! 'weight' must be a numpy array, but got type={type(p_wei)}.")
                        func_util.safe_exit(1)
                    if not (p_wei.ndim == 1 and p_wei.shape[0] == _orgDsize):
                        self.logger.error(f"Wrong input of particle weight! 'weight' must be a 1D array of shape (N,), but got shape={getattr(p_wei, 'shape', None)} while N={_orgDsize}.")
                        func_util.safe_exit(1)
                _ScaleFactor = self.L / self.SimBoxL
                if self.size == 1:
                    self.logger.info("Single process mode")
                    time_start = time.perf_counter()
                    _deltas = math_util.scaling_function_numba(
                        p          = p_pos,
                        w          = p_wei,
                        phi_data   = self.phi_data,
                        SampRate   = self.SampRate,
                        J          = self.J,
                        SimBoxL    = self.SimBoxL
                        )
                else:
                    self.logger.info("Multi-process mode")
                    self.logger.info("Start partition ... ")
                    p_pos_in = math_util.int_data(p_pos, _ScaleFactor)
                    _shrink_p_pos_in = math_util.bit(p_pos_in, self.J, int(np.log2(self.size)))
                    time_start = time.perf_counter()
                    with concurrent.futures.ThreadPoolExecutor(max_workers=self.size_local) as executor:
                        shrink_list = list(executor.map(lambda num: (np.ascontiguousarray(math_util.partition_data_single(p_pos, _shrink_p_pos_in, num)), np.ascontiguousarray(p_wei[_shrink_p_pos_in == num], dtype=np.float32)), range(self.size)))
                    time_end = time.perf_counter()
                    self.logger.info(f"The time for partition data: {time_end - time_start:.4f} sec")
                    p_pos_sub, p_wei_sub = shrink_list[0]
                    self.all_s = np.zeros((self.size, self.L // self.size + 2 * (self.PhiSupport - 1), self.L, self.L))
                    for i in range(1, self.size):
                        comm.send((shrink_list[i][0].shape, shrink_list[i][1].shape[0]), dest=i)
                        comm.Send(shrink_list[i][0], dest=i)
                        comm.Send(shrink_list[i][1], dest=i)
            elif rank > 0:
                self.orgDsize = 0
                self.all_s = None
                shrink_list = None
                shape_pos, n_wei = comm.recv(source=0)
                p_pos_sub = np.empty(shape_pos, dtype=np.float32)
                p_wei_sub = np.empty(n_wei, dtype=np.float32)
                comm.Recv(p_pos_sub, source=0)
                comm.Recv(p_wei_sub, source=0)
            if self.size > 1:
                comm.Barrier()
                rank == 0 and self.logger.info("Start to calculate scaling coefficient... ")
                time_start = time.perf_counter()
                _s_part = math_util.scaling_function_numba_part(
                    part       = rank,
                    p          = p_pos_sub,
                    w          = p_wei_sub,
                    phi_data   = self.phi_data,
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
                    "fin_path"      : self.fin_path,
                    "fin_format"    : self.fin_format,
                    "fin_weight_key": self.fin_wei_key,
                    "orgDsize"      : _orgDsize,
                    "J"             : self.J,
                    "SampRate"      : self.SampRate,
                    "SimBoxL"       : self.SimBoxL,
                    "bandwidth"     : self.bandwidth,
                    "wavelet_mode"  : self.wavelet_mode,
                    "wavelet_level" : self.wavelet_level,
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
            return self.deltac, p_pos
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
