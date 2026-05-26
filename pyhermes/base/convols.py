import time
import copy
import os

import numpy as np

from pyhermes.io import ConvolsData
from pyhermes.io.readers import read_particle_data, resolve_particle_value
from pyhermes.utils import func_util
from pyhermes.utils.wavelet_grid import (
    project_scaling_grid_numba,
    project_scaling_slab_numba,
    sample_scaling_function,
)
from pyhermes.pipeline import TaskBase


def _partition_particles_by_x_slab(p_pos, p_wei, scale_factor, J, size):
    size_bit = int(np.log2(size))
    slab_labels = np.floor(p_pos[:, 0] * scale_factor).astype(np.int64) >> (J - size_bit)
    valid = (slab_labels >= 0) & (slab_labels < size)

    slab_labels = slab_labels[valid].astype(np.intp, copy=False)
    p_pos = p_pos[valid]
    p_wei = p_wei[valid]

    order = np.argsort(slab_labels, kind="stable")
    slab_labels = slab_labels[order]
    p_pos = p_pos[order]
    p_wei = p_wei[order]

    counts = np.bincount(slab_labels, minlength=size)[:size]
    split_points = np.cumsum(counts)[:-1]
    pos_parts = np.split(p_pos, split_points)
    wei_parts = np.split(p_wei, split_points)
    return [
        (
            np.ascontiguousarray(pos_part),
            np.ascontiguousarray(wei_part, dtype=np.float32),
        )
        for pos_part, wei_part in zip(pos_parts, wei_parts)
    ]


def _add_slab_to_epsilon(epsilon, slab, part, size, core_width, phi_support):
    sew_width = phi_support - 1
    if sew_width == 0:
        start = part * core_width
        epsilon[start:start + core_width] += slab[:core_width]
        return
    if part == 0:
        epsilon[-sew_width:] += slab[:sew_width]
        epsilon[:core_width + sew_width] += slab[sew_width:core_width + 2 * sew_width]
    elif part == size - 1:
        epsilon[-(core_width + sew_width):] += slab[:core_width + sew_width]
        epsilon[:sew_width] += slab[-sew_width:]
    else:
        start = -sew_width + part * core_width
        stop = (part + 1) * core_width + sew_width
        epsilon[start:stop] += slab



class Convols(TaskBase):

    def __init__(self, param_task=None):
        if param_task is None:
            param_task = {"Convols": {}}
        self.task_name = str(self.__class__.__name__)
        super().__init__(param_task=param_task)
        self.format_params()
        self.convols_data = None
        self._fields_prepared = False

    def format_params(self):
        self.fin = copy.deepcopy(self.task_params['fin'])
        self.particle_pos = self.task_params['particle_pos']
        self.catalog_weight = self.task_params['catalog_weight']
        self.field_value = self.task_params['field_value']
        self.box_size = self.task_params['box_size']
        self.J = self.task_params['J']
        self.L = 1 << self.J
        self.wavelet_mode = self.task_params['wavelet_mode']
        self.wavelet_level = self.task_params['wavelet_level']
        self.phi_resolution = int(self.task_params['phi_resolution'])
        self.save_particle_data = bool(self.task_params['save_particle_data'])
        self.particle_data_path = self.task_params['particle_data_path']
        self.threads = int(self.task_params['threads'])
        self.fout_path = self.task_params['fout_path']

    def _sync_runtime_options(self):
        self.threads = max(1, int(self.threads))
        base_fin = copy.deepcopy(self.task_params.get('fin', {}))
        if self.fin is None:
            self.fin = base_fin
        else:
            merged_fin = base_fin
            merged_fin.update(self.fin)
            self.fin = merged_fin
        if self.fin.get("url"):
            self.logger.error("Convols.fin.url is no longer supported. Download the data first and set Convols.fin.path.")
            func_util.safe_exit(1)
        self.fin.setdefault("reader_params", {})
        if (
            self.fin.get("weight_key") is not None
            or self.task_params.get("particle_weight") is not None
            or getattr(self, "particle_weight", None) is not None
        ):
            raise ValueError(
                "The ambiguous Convols weight interface has been removed. Use "
                "catalog_weight/catalog_weight_key for observational weights and "
                "field_value/field_value_key for the measured physical quantity."
            )
        self.fin.pop("weight_key", None)
        self.fin.setdefault("catalog_weight_key", None)
        self.fin.setdefault("field_value_key", None)
        self.task_params = {
            'fin': copy.deepcopy(self.fin),
            'particle_pos': self.particle_pos,
            'catalog_weight': self.catalog_weight,
            'field_value': self.field_value,
            'box_size': self.box_size,
            'J': self.J,
            'wavelet_mode': self.wavelet_mode,
            'wavelet_level': self.wavelet_level,
            'phi_resolution': self.phi_resolution,
            'save_particle_data': self.save_particle_data,
            'particle_data_path': self.particle_data_path,
            'threads': self.threads,
            'fout_path': self.fout_path,
        }
        self.L = 1 << self.J
        self.sync_runtime_options(context="Convols runtime configuration")

    def _load_particle_input(self):
        if self.particle_pos is not None:
            p_pos = self.particle_pos
            self.particle_count = p_pos.shape[0]
            if self.catalog_weight is None:
                self.logger.info(
                    f"No catalog_weight provided; using unit catalogue weights for {self.particle_count} particles."
                )
                catalog_weight = np.ones(self.particle_count, dtype=np.float32)
                self.fin["catalog_weight_key"] = None
            else:
                catalog_weight = self.catalog_weight
                self.fin["catalog_weight_key"] = "custom"
            if self.field_value is None:
                field_value = np.ones(self.particle_count, dtype=np.float32)
                self.fin["field_value_key"] = None
            else:
                field_value = self.field_value
                self.fin["field_value_key"] = "custom"
            source_desc = "custom particle_pos array"
        else:
            input_format = self.fin.get("format", None)
            p_dict_all = read_particle_data(
                self.fin["path"],
                input_format,
                **self.fin.get("reader_params", {}),
            )
            p_pos, self.particle_count = p_dict_all['pos'], p_dict_all['size']
            catalog_weight, resolved_catalog_key = resolve_particle_value(
                p_dict_all,
                self.fin.get("catalog_weight_key", None),
                label="Catalogue weight",
                logger=self.logger,
            )
            field_value, resolved_value_key = resolve_particle_value(
                p_dict_all,
                self.fin.get("field_value_key", None),
                label="Field value",
                logger=self.logger,
            )
            self.fin["catalog_weight_key"] = resolved_catalog_key
            self.fin["field_value_key"] = resolved_value_key
            source_desc = f"file=path={self.fin['path']} format={input_format or 'auto'}"

        if not (isinstance(p_pos, np.ndarray) and p_pos.ndim == 2 and p_pos.shape[1] == 3):
            self.logger.error(
                f"Wrong input of particle data! 'particle_pos' must be a 2D array of shape (N, 3), "
                f"but got type={type(p_pos)} with shape={getattr(p_pos, 'shape', None)}."
            )
            func_util.safe_exit(1)
        factors = {}
        for name, value in (("catalog_weight", catalog_weight), ("field_value", field_value)):
            if np.isscalar(value):
                self.logger.info(
                    f"Input {name} is scalar; broadcasting to a per-particle array of length {self.particle_count}."
                )
                value = np.full(self.particle_count, value, dtype=np.float32)
            if not isinstance(value, np.ndarray) or value.ndim != 1 or value.shape[0] != self.particle_count:
                self.logger.error(
                    f"Wrong input of {name}! Expected a numpy array with shape (N,), "
                    f"but got type={type(value)} shape={getattr(value, 'shape', None)} while N={self.particle_count}."
                )
                func_util.safe_exit(1)
            if not np.all(np.isfinite(value)):
                self.logger.error(f"Wrong input of {name}! Values must all be finite.")
                func_util.safe_exit(1)
            factors[name] = value.astype(np.float32, copy=False)
        return p_pos, factors["catalog_weight"], factors["field_value"], source_desc

    def _resolve_particle_data_path(self):
        if self.particle_data_path:
            return self.particle_data_path
        if self.fout_path:
            base, _ext = os.path.splitext(self.fout_path)
            return f"{base}_particles.npz"
        self.logger.error(
            "save_particle_data=True requires either 'particle_data_path' or 'fout_path' "
            "so PyHermes can choose where to save the particle dataset."
        )
        func_util.safe_exit(1)

    def _save_particle_dataset(self):
        particle_data_path = self._resolve_particle_data_path()
        output_dir = os.path.dirname(particle_data_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        np.savez(
            particle_data_path,
            pos=np.ascontiguousarray(self.particle_pos, dtype=np.float32),
            catalog_weight=np.ascontiguousarray(self.catalog_weight, dtype=np.float32),
            field_value=np.ascontiguousarray(self.field_value, dtype=np.float32),
        )
        self.particle_data_path = particle_data_path
        self.logger.info(f"Saved particle positions, catalogue weights, and field values to {particle_data_path}")

    def prepare_input_fields(self, particle_pos=None, catalog_weight=None, field_value=None, fin=None):
        if fin is not None:
            merged_fin = copy.deepcopy(self.fin)
            merged_fin.update(fin)
            self.fin = merged_fin
        if particle_pos is not None:
            self.particle_pos = particle_pos
        if catalog_weight is not None:
            self.catalog_weight = catalog_weight
        if field_value is not None:
            self.field_value = field_value
        self._sync_runtime_options()
        self.convols_data = ConvolsData(threads=self.threads)
        self.phi_array = sample_scaling_function(self.wavelet_mode, self.wavelet_level)
        self.phi_support = self.phi_array.shape[0] // self.phi_resolution
        self.core_width = self.L // self.size
        self.scale_factor = self.L / self.box_size
        if self.rank == 0:
            self.logger.info("Preparing Convols input fields ...")
            self.logger.info(
                f"J={self.J}, L={self.L}, box_size={self.box_size}, phi_resolution={self.phi_resolution}, "
                f"wavelet_mode={self.wavelet_mode}, wavelet_level={self.wavelet_level}"
            )
            if self.size != 1 and (self.size & (self.size - 1)) != 0:
                self.logger.error(f"MPI rank number {self.size} is not a power of two. Please adjust your configuration.")
                func_util.safe_exit(1)
            p_pos, catalog_weight, field_value, source_desc = self._load_particle_input()
            self.particle_pos = p_pos
            self.catalog_weight = catalog_weight
            self.field_value = field_value
            self.projection_weight = self.catalog_weight * self.field_value
            self.catalog_weight_sum = float(np.sum(self.catalog_weight, dtype=np.float64))
            self.field_weighted_sum = float(np.sum(self.projection_weight, dtype=np.float64))
            if not np.isfinite(self.catalog_weight_sum) or np.isclose(self.catalog_weight_sum, 0.0):
                self.logger.error(
                    f"Catalogue weights must have a finite, non-zero sum, got {self.catalog_weight_sum!r}."
                )
                func_util.safe_exit(1)
            if not np.isfinite(self.field_weighted_sum):
                self.logger.error(
                    "The product catalog_weight * field_value produced a non-finite summed field value."
                )
                func_util.safe_exit(1)
            self.field_value_mean = self.field_weighted_sum / self.catalog_weight_sum
            self.logger.info(
                f"Input particles ready | source={source_desc} | particle_count={self.particle_count} "
                f"| catalog_weight_key={self.fin['catalog_weight_key']} | field_value_key={self.fin['field_value_key']} "
                f"| catalog_weight_sum={self.catalog_weight_sum:.6e} | field_weighted_sum={self.field_weighted_sum:.6e}"
            )
            if self.save_particle_data:
                self._save_particle_dataset()
        else:
            self.particle_pos = None
            self.catalog_weight = None
            self.field_value = None
            self.projection_weight = None
            self.catalog_weight_sum = None
            self.field_weighted_sum = None
            self.field_value_mean = None
        self._fields_prepared = True

    def run(self, save_result=True, overwrite=False):
        try:
            comm = self.comm
            rank = self.rank
            p_pos = None
            p_wei = None
            if rank == 0:
                time_run_1 = time.perf_counter()
            if not self._fields_prepared:
                self.prepare_input_fields()
            p_pos = self.particle_pos
            p_wei = self.projection_weight
            if rank == 0 and self.size == 1:
                self.logger.info("Single process mode")
                time_start = time.perf_counter()
                _epsilon = project_scaling_grid_numba(
                    positions=p_pos,
                    weights=p_wei,
                    phi_array=self.phi_array,
                    phi_resolution=self.phi_resolution,
                    J=self.J,
                    box_size=self.box_size
                )
            elif rank == 0:
                self.logger.info("Multi-process mode")
                self.logger.info("Start partition ... ")
                time_start = time.perf_counter()
                shrink_list = _partition_particles_by_x_slab(
                    p_pos=p_pos,
                    p_wei=p_wei,
                    scale_factor=self.scale_factor,
                    J=self.J,
                    size=self.size,
                )
                time_end = time.perf_counter()
                self.logger.info(f"The time for partition data: {time_end - time_start:.4f} sec")
                p_pos_sub, p_wei_sub = shrink_list[0]
                for i in range(1, self.size):
                    comm.send((shrink_list[i][0].shape, shrink_list[i][1].shape[0]), dest=i)
                    comm.Send(shrink_list[i][0], dest=i)
                    comm.Send(shrink_list[i][1], dest=i)
                    shrink_list[i] = None
            else:
                self.particle_count = 0
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
                _s_part = project_scaling_slab_numba(
                    slab_index = rank,
                    positions  = p_pos_sub,
                    weights    = p_wei_sub,
                    phi_array   = self.phi_array,
                    core_width = self.core_width,
                    phi_resolution   = self.phi_resolution,
                    J          = self.J,
                    box_size    = self.box_size
                    )
                if rank == 0:
                    _epsilon = np.zeros((self.L, self.L, self.L), dtype=np.float64)
                    _add_slab_to_epsilon(
                        _epsilon, _s_part, rank, self.size, self.core_width, self.phi_support
                    )
                    recv_shape = _s_part.shape
                    for source in range(1, self.size):
                        recv_slab = np.empty(recv_shape, dtype=np.float64)
                        comm.Recv(recv_slab, source=source)
                        _add_slab_to_epsilon(
                            _epsilon, recv_slab, source, self.size, self.core_width, self.phi_support
                        )
                else:
                    comm.Send(_s_part, dest=0)
            if rank == 0:
                _convols_info = {
                    "fin"           : copy.deepcopy(self.fin),
                    "particle_count"   : self.particle_count,
                    "catalog_weight_sum": self.catalog_weight_sum,
                    "field_weighted_sum": self.field_weighted_sum,
                    "field_value_mean": self.field_value_mean,
                    "box_size"       : self.box_size,
                    "J"             : self.J,
                    "L"             : self.L,
                    "V"             : self.L ** 3,
                    "scale_factor"   : self.scale_factor,
                    "coefficient_convention": "raw_catalog_field_value",
                    "field_kind"     : "raw_weighted_field",
                    "normalization": "none",
                    "normalization_sum": None,
                    "wavelet_mode"  : self.wavelet_mode,
                    "wavelet_level" : self.wavelet_level,
                    "phi_resolution"      : self.phi_resolution,
                    "particle_data_path": self.particle_data_path if self.save_particle_data else "",
                    "particle_data_format": "npz" if self.save_particle_data else "",
                    "phi_support"    : self.phi_support,
                    "phi_array"      : self.phi_array
                }
                self.convols_data.convols_info = dict(_convols_info)
                self.convols_data.format_convols_params()
                time_end = time.perf_counter()
                self.logger.info(f"The time for scaling function: {time_end - time_start:.4f} sec")
                self.convols_data.epsilon = _epsilon
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
        return self.convols_data
