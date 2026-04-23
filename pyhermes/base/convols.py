import time
import concurrent.futures
import copy

import numpy as np

from pyhermes.io import ConvolsData
from pyhermes.io import read_particle_data
from pyhermes.io.funcs import dl_rich_pbar
from pyhermes.utils import func_util
from pyhermes.utils.wavelet_grid import (
    bit,
    do_wavelet,
    int_data,
    partition_data_single,
    scaling_function_numba,
    scaling_function_numba_part,
)
from pyhermes.pipeline import TaskBase



class Convols(TaskBase):

    def __init__(self, param_task=None):
        if param_task is None:
            param_task = {"Convols": {}}
        self.task_name = str(self.__class__.__name__)
        super().__init__(param_task=param_task)
        self.format_params()
        self.convols_data = None
        self._fields_prepared = False
        self._prepared_particle_pos = None
        self._prepared_particle_weight = None

    def format_params(self):
        self.J             = self.task_params['J']
        self.fin           = copy.deepcopy(self.task_params['fin'])
        self.fin_path      = self.fin['path']
        self.fin_url       = self.fin.get('url', '')
        self.fin_format    = self.fin['format']
        self.fin_wei_key   = self.fin['weight_key']
        self.fout_path     = self.task_params['fout_path']
        self.SampRate      = int(self.task_params['SampRate'])
        self.SimBoxL       = self.task_params['SimBoxL']
        self.wavelet_mode  = self.task_params['wavelet_mode']
        self.wavelet_level = self.task_params['wavelet_level']
        self.bandwidth     = self.task_params['bandwidth']
        self.threads       = int(self.task_params['threads'])
        self.particle_pos = self.task_params['particle_pos']
        self.particle_weight = self.task_params['particle_weight']
        self.L             = 1 << self.J

    def _sync_runtime_options(self):
        self.threads = max(1, int(self.threads))
        self.task_params['threads'] = self.threads
        self.task_params['J'] = self.J
        self.task_params['fout_path'] = self.fout_path
        self.task_params['SampRate'] = self.SampRate
        self.task_params['SimBoxL'] = self.SimBoxL
        self.task_params['wavelet_mode'] = self.wavelet_mode
        self.task_params['wavelet_level'] = self.wavelet_level
        self.task_params['bandwidth'] = self.bandwidth
        self.task_params['particle_pos'] = self.particle_pos
        self.task_params['particle_weight'] = self.particle_weight
        base_fin = copy.deepcopy(self.task_params.get('fin', {}))
        if self.fin is None:
            self.fin = base_fin
        else:
            merged_fin = base_fin
            merged_fin.update(self.fin)
            self.fin = merged_fin
        self.task_params['fin'] = copy.deepcopy(self.fin)
        self.fin_path = self.fin['path']
        self.fin_url = self.fin.get('url', '')
        self.fin_format = self.fin['format']
        self.fin_wei_key = self.fin['weight_key']
        self.L = 1 << self.J
        self.sync_runtime_options(context="Convols runtime configuration")

    def _resolve_fin_source(self):
        if self.fin_url:
            if not self.fin_path:
                self.logger.error("When 'fin.url' is provided, 'fin.path' must also be provided as the local download target.")
                func_util.safe_exit(1)
            downloaded_path = dl_rich_pbar(self.fin_url, output_path=self.fin_path)
            self.fin_path = downloaded_path
            self.fin['path'] = downloaded_path
            return f"url={self.fin_url} -> path={downloaded_path}"
        return f"path={self.fin_path}"

    def _load_particle_input(self):
        fin_source_desc = self._resolve_fin_source()
        if self.particle_pos is not None:
            p_pos = self.particle_pos
            self.N_particles = p_pos.shape[0]
            if self.particle_weight is None:
                self.logger.info(
                    f"No particle_weight provided; using unit weights for {self.N_particles} particles."
                )
                p_wei = np.ones(self.N_particles, dtype=np.float32)
                self.fin_wei_key = 'no_weight'
            else:
                p_wei = self.particle_weight
                self.fin_wei_key = 'weight comes from custom input array'
            source_desc = "custom particle_pos array"
        else:
            p_dict_all = read_particle_data(self.fin_path, self.fin_format)
            p_pos, self.N_particles = p_dict_all['pos'], p_dict_all['size']
            if self.fin_format == 'generic_pos':
                p_wei = np.ones(self.N_particles, dtype=np.float32)
                self.fin_wei_key = 'no_weight'
            elif self.fin_format == 'generic_pos_weight':
                p_wei = p_dict_all['weight']
                self.fin_wei_key = 'weight'
            else:
                _key = self.fin_wei_key
                if _key is None or _key == 'no_weight':
                    p_wei = np.ones(self.N_particles, dtype=np.float32)
                    self.fin_wei_key = 'no_weight'
                elif _key in p_dict_all:
                    p_wei = p_dict_all[_key]
                    self.fin_wei_key = [_key]
                else:
                    self.logger.warning(
                        f"Weight key '{_key}' not found in particle data. Calculating without weight. "
                        f"Available keys: {list(p_dict_all.keys())}. Use weight_key='no_weight' if no weighting is desired."
                    )
                    p_wei = np.ones(self.N_particles, dtype=np.float32)
                    self.fin_wei_key = 'no_weight'
            source_desc = f"file={fin_source_desc} format={self.fin_format}"

        if not (isinstance(p_pos, np.ndarray) and p_pos.ndim == 2 and p_pos.shape[1] == 3):
            self.logger.error(
                f"Wrong input of particle data! 'particle_pos' must be a 2D array of shape (N, 3), "
                f"but got type={type(p_pos)} with shape={getattr(p_pos, 'shape', None)}."
            )
            func_util.safe_exit(1)
        if np.isscalar(p_wei):
            self.logger.info(f"Input weight is scalar; broadcasting to a uniform per-particle weight array of length {self.N_particles}.")
            p_wei = np.full(self.N_particles, p_wei, dtype=np.float32)
        if not isinstance(p_wei, np.ndarray):
            self.logger.error(f"Wrong input of particle weight! 'particle_weight' must be a numpy array, but got type={type(p_wei)}.")
            func_util.safe_exit(1)
        if not (p_wei.ndim == 1 and p_wei.shape[0] == self.N_particles):
            self.logger.error(
                f"Wrong input of particle weight! 'particle_weight' must have shape (N,), "
                f"but got shape={getattr(p_wei, 'shape', None)} while N={self.N_particles}."
            )
            func_util.safe_exit(1)
        return p_pos, p_wei.astype(np.float32, copy=False), source_desc

    def prepare_input_fields(self, particle_pos=None, particle_weight=None, fin=None):
        if fin is not None:
            merged_fin = copy.deepcopy(self.fin)
            merged_fin.update(fin)
            self.fin = merged_fin
        if particle_pos is not None:
            self.particle_pos = particle_pos
        if particle_weight is not None:
            self.particle_weight = particle_weight
        self._sync_runtime_options()
        self.convols_data = ConvolsData(threads=self.threads)
        self.phi_data = do_wavelet(self.wavelet_mode, self.wavelet_level)
        _PhiStart = 0
        _PhiEnd = self.phi_data.shape[0] // self.SampRate
        self.PhiSupport = _PhiEnd - _PhiStart
        self.core_width = self.L // self.size
        self.ScaleFactor = self.L / self.SimBoxL
        if self.rank == 0:
            self.logger.info("Preparing Convols input fields ...")
            self.logger.info(
                f"J={self.J}, L={self.L}, SimBoxL={self.SimBoxL}, SampRate={self.SampRate}, "
                f"wavelet_mode={self.wavelet_mode}, wavelet_level={self.wavelet_level}, bandwidth={self.bandwidth}"
            )
            if self.size != 1 and (self.size & (self.size - 1)) != 0:
                self.logger.error(f"MPI rank number {self.size} is not a power of two. Please adjust your configuration.")
                func_util.safe_exit(1)
            p_pos, p_wei, source_desc = self._load_particle_input()
            self.NormFactor = 1 / self.N_particles
            self.logger.info(
                f"Input particles ready | source={source_desc} | N_particles={self.N_particles} | weight_key={self.fin_wei_key}"
            )
            self._prepared_particle_pos = p_pos
            self._prepared_particle_weight = p_wei
        else:
            self._prepared_particle_pos = None
            self._prepared_particle_weight = None
        self._fields_prepared = True

    def run(self, save_result=True, return_pData=False, overwrite=False):
        try:
            comm = self.comm
            rank = self.rank
            p_pos = None
            p_wei = None
            if rank == 0:
                time_run_1 = time.perf_counter()
            if not self._fields_prepared:
                self.prepare_input_fields()
            p_pos = self._prepared_particle_pos
            p_wei = self._prepared_particle_weight
            if rank == 0 and self.size == 1:
                self.logger.info("Single process mode")
                time_start = time.perf_counter()
                _epsilon = scaling_function_numba(
                    p=p_pos,
                    w=p_wei,
                    phi_data=self.phi_data,
                    SampRate=self.SampRate,
                    J=self.J,
                    SimBoxL=self.SimBoxL
                )
            elif rank == 0:
                self.logger.info("Multi-process mode")
                self.logger.info("Start partition ... ")
                p_pos_in = int_data(p_pos, self.ScaleFactor)
                _shrink_p_pos_in = bit(p_pos_in, self.J, int(np.log2(self.size)))
                time_start = time.perf_counter()
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.size_local) as executor:
                    shrink_list = list(
                        executor.map(
                            lambda num: (
                                np.ascontiguousarray(partition_data_single(p_pos, _shrink_p_pos_in, num)),
                                np.ascontiguousarray(p_wei[_shrink_p_pos_in == num], dtype=np.float32)
                            ),
                            range(self.size)
                        )
                    )
                time_end = time.perf_counter()
                self.logger.info(f"The time for partition data: {time_end - time_start:.4f} sec")
                p_pos_sub, p_wei_sub = shrink_list[0]
                self.all_s = np.zeros((self.size, self.L // self.size + 2 * (self.PhiSupport - 1), self.L, self.L))
                for i in range(1, self.size):
                    comm.send((shrink_list[i][0].shape, shrink_list[i][1].shape[0]), dest=i)
                    comm.Send(shrink_list[i][0], dest=i)
                    comm.Send(shrink_list[i][1], dest=i)
            else:
                self.N_particles = 0
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
                _s_part = scaling_function_numba_part(
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
                    _epsilon = self.sew_up(self.all_s, self.size, self.L,self.PhiSupport)
            if rank == 0:
                _convols_info = {
                    "fin_path"      : self.fin_path,
                    "fin_format"    : self.fin_format,
                    "fin_weight_key": self.fin_wei_key,
                    "N_particles"   : self.N_particles,
                    "J"             : self.J,
                    "SampRate"      : self.SampRate,
                    "SimBoxL"       : self.SimBoxL,
                    "bandwidth"     : self.bandwidth,
                    "wavelet_mode"  : self.wavelet_mode,
                    "wavelet_level" : self.wavelet_level,
                    "L"             : self.L,
                    "V"             : self.L ** 3,
                    "ScaleFactor"   : self.ScaleFactor,
                    "NormFactor"    : self.NormFactor,
                    "PhiSupport"    : self.PhiSupport,
                    "phi_data"      : self.phi_data
                }
                self.convols_data.convols_info = dict(_convols_info)
                self.convols_data.format_convols_params()
                time_end = time.perf_counter()
                self.logger.info(f"The time for scaling function: {time_end - time_start:.4f} sec")
                self.convols_data.epsilon = _epsilon * self.NormFactor
                if save_result and self.fout_path:
                    self.convols_data.save_convols(self.fout_path, overwrite=overwrite)
        except Exception as e:
            self.logger.error(f"Error in process {self.rank}: {str(e)}")
            func_util.safe_exit(1)
        comm.Barrier()
        if self.rank == 0:
            time_run_2 = time.perf_counter()
            print("")
            self.logger.info(f"The time for task: {time_run_2 - time_run_1:.4f} sec")
        if return_pData:
            return self.convols_data, p_pos
        else:
            return self.convols_data

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
