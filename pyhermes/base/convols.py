import os
import time
import concurrent.futures

import numpy as np
from scipy.fft import rfftn, irfftn

from pyhermes.io import ColvolsData
from pyhermes.io import read_particle_data
from pyhermes.utils import func_util
from pyhermes.utils import math_util
from pyhermes.pipeline import pipeline as pipeline



class Convols(pipeline.TaskBase):

    def __init__(self, param_task):
        self.task_name = str(self.__class__.__name__)
        super().__init__(param_task=param_task)

    def format_params(self):
        self.J             = self.task_params['J']
        self.fin_path      = self.task_params['fin']['path']
        self.fin_format    = self.task_params['fin']['format']
        self.fout_dir      = self.task_params['fout_dir']
        self.window_type   = self.task_params['window']['type']
        self.SampRate      = int(self.task_params['SampRate'])
        self.SimBoxL       = self.task_params['SimBoxL']
        self.wavelet_mode  = self.task_params['wavelet_mode']
        self.wavelet_level = self.task_params['wavelet_level']
        self.window_args   = {key : float(value) for key, value in self.task_params['window'].items() if key != 'type'}
        self.bandwidth     = self.task_params['bandwidth']
        self.threads       = int(self.task_params['threads'])
        self.L             = 1 << self.J

    def run(self, return_pData_rank0=False): 
        self.deltac = ColvolsData()
        try:
            if self.rank == 0:
                time_run_1 = time.perf_counter()
            # Retrive parameters to locals
            self.format_params()
            # Do wavelet transform
            self.phi_data = math_util.do_wavelet(self.wavelet_mode, self.wavelet_level)
            _PhiStart = 0
            _PhiEnd = self.phi_data.shape[0] // self.SampRate
            self.PhiSupport = _PhiEnd - _PhiStart 
            self.core_width = self.L // self.size
            if self.rank == 0 :
                # Read particle data, the origin data size only store in rank 0
                # _data_in, _orgDsize = read_tristan(self.fin_path)
                p_dm, _orgDsize = read_particle_data(self.fin_path, self.fin_format)
                if p_dm.shape[1] != 3:
                    self.logger.error("Wrong shape of input particle catalog data! The shape should be (*,3)")
                    func_util.safe_exit(1)
                # self.deltac.orgData = p_dm
                self.logger.info("Start partition ... ")
                _ScaleFactor = self.L / self.SimBoxL
                p_dm_in = math_util.int_data(p_dm, _ScaleFactor)
                _shrink_p_dm_in = math_util.bit(p_dm_in, self.J, int(np.log2(self.size)))
                time_start = time.perf_counter()
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.size) as executor: 
                    shrink_list = list(executor.map(lambda num: math_util.partition_data_single(p_dm, _shrink_p_dm_in, num), range(self.size)))
                time_end = time.perf_counter()
                self.logger.info(f"The time for partition data: {time_end - time_start:.4f} sec")
                data_sub_part = shrink_list[0]
                self.all_s  = np.zeros((self.size, self.L // self.size + 2 * (self.PhiSupport - 1), self.L, self.L))
            else:
                self.orgDsize = 0
                self.all_s = None
                shrink_list = None
            if self.rank == 0:
                for i in range(1, self.size):
                    self.comm.send(shrink_list[i].shape, dest=i)
                    self.comm.Send(shrink_list[i], dest=i)
            else:
                shape = self.comm.recv(source=0)
                data_sub_part = np.empty(shape, dtype=np.float32)
                self.comm.Recv(data_sub_part, source=0)
            self.comm.Barrier()
            self.rank == 0 and self.logger.info("Start to calculate scaling coefficient... ")
            time_start = time.perf_counter()
            _s_part = math_util.scaling_function_numba_part(
                part       = self.rank,
                p          = data_sub_part,
                phi_data   = self.phi_data,
                size       = self.size,
                core_width = self.core_width,
                SampRate   = self.SampRate,
                J          = self.J,
                SimBoxL    = self.SimBoxL
                )
            # !NOTICE: the MPI-rank num to calculate scaling coefficient should be
            # !          a power of two
            self.comm.Gather(_s_part, self.all_s, root=0)
            if self.rank == 0:
                _dict_inht_vonDeltac = {
                    "fin_path"     : self.fin_path,
                    "fin_format"   : self.fin_format,
                    "orgDsize"     : _orgDsize,
                    "window_type"  : self.window_type,
                    "J"            : self.J,
                    "SampRate"     : self.SampRate,
                    "window_args"  : self.window_args,
                    "SimBoxL"      : self.SimBoxL,
                    "bandwidth"    : self.bandwidth,
                    "wavelet_mode" : self.wavelet_mode,
                    "wavelet_level": self.wavelet_level,
                }
                self.deltac.dict_inht_vonDeltac.update(_dict_inht_vonDeltac)
                if self.size > 1:
                    _deltas = self.sew_up(self.all_s, self.size, self.L)
                else:
                    _deltas = self.all_s[0, 2:-2, :, :]
                time_end = time.perf_counter()
                self.logger.info(f"The time for scaling function: {time_end - time_start:.4f} sec")
                # Handle window function
                _DeltaXi = 1./(self.L)
                _rescale_win_args = {key : value * self.L / self.SimBoxL for key, value in self.window_args.items()}
                # self.Radius = self.window_args["R"]
                _result_string = "_".join(f"{key}_{value}" for key, value in self.window_args.items())
                # _rescale_win_args = self.Radius * self.L / self.SimBoxL
                _PowerPhi = self.power_spectrum(self.phi_data, 0, self.bandwidth, self.L * self.bandwidth)
                _w_func = math_util.set_window_function(self.window_type)
                _window_array = math_util.call_calculate_window_array(
                    L                     = self.L,
                    bandwidth             = self.bandwidth,
                    DeltaXi               = _DeltaXi,
                    PowerPhi              = _PowerPhi,
                    window_function_numba = _w_func,
                    **_rescale_win_args
                    )
                _w = math_util.calculate_w_numba(_window_array)
                # Do FFT to get convol result
                self.logger.info('Start to calculte FFT')
                time_start = time.perf_counter()
                self.deltac.data = self.specialized_convolution_3d(_deltas, _w, threads=self.threads)
                time_end = time.perf_counter()
                self.logger.info(f"The time for FFT: {time_end - time_start:.4f} sec")
                if self.fout_dir is not None and self.fout_dir != "":
                    # Output the deltac
                    _fout_path = os.path.join(self.fout_dir, f"convols_L{str(self.L)}_{_result_string}_pywt.npy")
                    self.deltac.save(_fout_path)
        except Exception as e:
            self.logger.error(f"Error in process {self.rank}: {str(e)}")
            func_util.safe_exit(1)
        self.comm.Barrier()
        if self.rank == 0:
            time_run_2 = time.perf_counter()
            print("")
            self.logger.info(f"The time for task: {time_run_2 - time_run_1:.4f} sec")
        if return_pData_rank0:
            if self.rank ==0:
                return self.deltac, p_dm
            else:
                return self.deltac, None
        else:
            return self.deltac

    def spectrum_vectorized(self, v, k0, k1, N_k):
        N_x = v.shape[0]
        x0 = 0
        x1 = v.shape[0]/self.SampRate
        Delta_x = (x1 - x0) / N_x
        Delta_k = (k1 - k0) / N_k
        x = np.arange(N_x) * Delta_x
        k = np.arange(N_k + 1) * Delta_k
        # Create 2D grids for x and k
        x_grid, k_grid = np.meshgrid(x, k)
        # Calculate the real and imaginary parts of the spectrum
        s_real = np.sum(v * Delta_x * np.cos(-2 * np.pi * k_grid * x_grid), axis=1)
        s_imag = np.sum(v * Delta_x * np.sin(-2 * np.pi * k_grid * x_grid), axis=1)
        # Interleave the real and imaginary parts
        s = np.empty((N_k + 1) * 2, dtype=np.double)
        s[::2] = s_real
        s[1::2] = s_imag
        return s

    def power_spectrum(self, v, k0, k1, N_k):
        s = self.spectrum_vectorized(v, k0, k1, N_k)
        p = np.zeros(N_k + 1, dtype=np.double)
        for i in range(N_k + 1):
            p[i] = s[2*i] ** 2 + s[2*i+1] ** 2
        return p

    def specialized_convolution_3d(self, s, w, threads):
        # Run FFt in multi-thread manner
        sc = rfftn(s, workers= threads)
        sc *= w
        result_convol3d = irfftn(sc, workers = threads)
        return result_convol3d
    
    def sew_up(self, all_s, size, L):
        sew_s = np.zeros((L, L, L))
        for part in range(1, size - 1):
            # print(sew_s[-2+part*self.core_width:(part+1)*self.core_width+2].shape)
            # print(all_s[part].shape)
            sew_s[-2+part*self.core_width:(part+1)*self.core_width+2] += all_s[part]
        sew_s[-2:] += all_s[0][:2]
        sew_s[:self.core_width+2] += all_s[0][2:self.core_width+2+2]
        sew_s[-(self.core_width+2):] += all_s[-1][:self.core_width+2]
        sew_s[:2] += all_s[-1][-2:]
        return sew_s
