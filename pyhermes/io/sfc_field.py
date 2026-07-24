import copy
import os

import numpy as np

from pyhermes.utils import func_util
from pyhermes.utils.convolution import (
    specialized_convolution_3d,
    specialized_convolution_3d_complex,
)
from pyhermes.utils.wavelet_grid import (
    interpolate_grid_at_pos_numba,
    scaling_stencil_at_pos_numba,
)

from .base import HermesData
from .pickle_compat import read_numpy_pickle, write_numpy_pickle
from .readers import read_particle_data, resolve_particle_value


def normalize_weight_normalization(weight_normalization):
    if weight_normalization is None:
        raise ValueError("weight_normalization must be one of 'raw', 'catalog', 'field', or 'unit'.")
    mode = str(weight_normalization).strip().lower()
    if mode == "unit":
        return "field"
    if mode not in {"raw", "catalog", "field"}:
        raise ValueError("weight_normalization must be one of 'raw', 'catalog', 'field', or 'unit'.")
    return mode


def normalize_task_weight_normalization(weight_normalization):
    if weight_normalization is None:
        raise ValueError("task weight_normalization must be one of 'raw', 'catalog', 'field', or 'unit'.")
    mode = str(weight_normalization).strip().lower()
    if mode not in {"raw", "catalog", "field", "unit"}:
        raise ValueError("task weight_normalization must be one of 'raw', 'catalog', 'field', or 'unit'.")
    return mode


def normalize_field_value_normalization(normalization):
    if normalization is None:
        return None
    mode = str(normalization).strip().lower()
    if mode not in {"raw", "catalog", "field", "unit"}:
        raise ValueError("normalization must be None, 'raw', 'catalog', 'field', or 'unit'.")
    return mode


class SFCField(HermesData):
    _REQUIRED_ARGV = ("J", "box_size", "phi_resolution", "wavelet_mode", "wavelet_level")

    def __init__(self, *args, threads=None, **kwargs):
        data_path = kwargs.pop("data_path", None)
        self.sfc_info = {}
        self.epsilon = None
        super().__init__(*args, threads=threads, **kwargs)
        if data_path:
            self.sfc_info['sfc_field_path'] = data_path
            self.load_sfc_field(data_path)

    def _spawn_like(self):
        """
        Create a new empty object of the same class,
        inheriting metadata but NOT copying heavy data arrays.
        """
        new = self.__class__(threads=self.threads)

        # --- MPI / logging ---
        new.comm   = self.comm
        new.rank   = self.rank
        new.logger = self.logger

        # --- copy metadata (shallow copy is enough) ---
        new.sfc_info = copy.deepcopy(self.sfc_info)
        new.task_params  = copy.deepcopy(self.task_params)

        # --- clear data containers ---
        new.epsilon = None
        new.saveflag = False

        return new

    def copy(self):
        """
        Create a deep copy of the object, including data arrays.
        """
        new = self._spawn_like()
        new.epsilon = copy.deepcopy(self.epsilon)
        new.format_sfc_params()
        return new

    def _conv(self, other):
        if not hasattr(other, "as_array"):
            if self.rank == 0:
                self.logger.error(f"Convolution requires `other` to implement as_array(); got {type(other)}.")
            func_util.safe_exit(1)
        a = self.as_array()
        b = other.as_array()
        if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray):
            if self.rank == 0:
                self.logger.error(
                    "Convolution requires both inputs to be numpy arrays. "
                    f"Got a={type(a)}, b={type(b)}."
                )
            func_util.safe_exit(1)
        if a.ndim != 3 or b.ndim != 3:
            if self.rank == 0:
                self.logger.error(
                    f"Convolution expects 3D arrays; got a.ndim={a.ndim}, b.ndim={b.ndim}."
                )
            func_util.safe_exit(1)
        nx, ny, nz = a.shape
        expected_rfft = (nx, ny, nz // 2 + 1)
        expected_full_fft = (nx, ny, nz)
        if b.shape == expected_rfft:
            conv = specialized_convolution_3d(
                a, b, threads=self.threads
            )
        elif b.shape == expected_full_fft and np.iscomplexobj(b):
            conv = specialized_convolution_3d_complex(
                a, b, threads=self.threads
            )
        else:
            if self.rank == 0:
                self.logger.error(
                    f"Convolution requires same shape; got a.shape={a.shape}, b.shape={b.shape}. "
                    f"Expected an rFFT kernel with shape {expected_rfft} or a complex full-FFT kernel "
                    f"with shape {expected_full_fft}. "
                    "In this API, the convolution window is expected on the right-hand side "
                    "(signal @ window). "
                    "If you swapped the operands (window @ signal), this shape mismatch can occur.")
            func_util.safe_exit(1)

        # --- spawn new SFCField ---
        new = self._spawn_like()

        # --- set result ---
        new.epsilon = conv

        # --- optional: record provenance ---
        windows = self.sfc_info.get("convolution_of", [])
        # windows.append(other.window_params)
        new.sfc_info.update({
            "convolution_of": windows + [other.window_params]
        })
        self._mark_derived(new)
        new.format_sfc_params()

        return new

    def __matmul__(self, other):
        if isinstance(other, SFCField):
            return self._conv(other)
        return NotImplemented

    def __rmatmul__(self, other):
        if isinstance(other, SFCField):
            return other._conv(self)  
        return NotImplemented

    def __add__(self, other):
        if isinstance(other, SFCField):
            return self._add_field(other)

        if np.isscalar(other):
            return self._add_scalar(other)

        return NotImplemented

    def __radd__(self, other):
        if np.isscalar(other):
            return self._add_scalar(other)

        if isinstance(other, SFCField):
            return other._add_field(self)

        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, SFCField):
            return self._mul_field(other)

        if np.isscalar(other):
            return self._mul_scalar(other)

        return NotImplemented

    def __rmul__(self, other):
        if np.isscalar(other):
            return self._mul_scalar(other)

        if isinstance(other, SFCField):
            return other._mul_field(self)

        return NotImplemented

    def __truediv__(self, other):
        if np.isscalar(other) and not isinstance(other, (str, bytes)) and np.isrealobj(other):
            return self._div_scalar(float(other))

        return NotImplemented

    # ---------- field + field ----------
    def _add_field(self, other):
        self._validate_field_array_operation(other, "addition")
        new = self._spawn_like()
        new.epsilon = self.epsilon + other.epsilon
        self._mark_derived(new)
        new.format_sfc_params()
        return new

    # ---------- field + scalar ----------
    def _add_scalar(self, scalar):
        if self.epsilon is None:
            self.logger.error("Cannot add scalar: epsilon is None.")
            func_util.safe_exit(1)

        new = self._spawn_like()
        new.epsilon = self.epsilon + scalar
        self._mark_derived(new)
        new.format_sfc_params()
        return new

    # ---------- field × field ----------
    def _mul_field(self, other):
        self._validate_field_array_operation(other, "multiplication")
        new = self._spawn_like()
        new.epsilon = self.epsilon * other.epsilon
        self._mark_derived(new)
        new.format_sfc_params()
        return new

    # ---------- field × scalar ----------
    def _mul_scalar(self, scalar):
        if self.epsilon is None:
            self.logger.error("Cannot multiply by scalar: epsilon is None.")
            func_util.safe_exit(1)

        new = self._spawn_like()
        new.epsilon = self.epsilon * scalar
        self._mark_derived(new)
        new.format_sfc_params()
        return new

    # ---------- field / scalar ----------
    def _div_scalar(self, scalar):
        if self.epsilon is None:
            self.logger.error("Cannot divide by scalar: epsilon is None.")
            func_util.safe_exit(1)
        if scalar == 0.0:
            self.logger.error("Cannot divide SFCField by zero.")
            func_util.safe_exit(1)

        new = self._spawn_like()
        new.epsilon = self.epsilon / scalar
        self._mark_derived(new)
        new.format_sfc_params()
        return new

    def __sub__(self, other):
        # Case 1: field - field
        if isinstance(other, SFCField):
            return self._sub_field(other)

        # Case 2: field - scalar
        if np.isscalar(other):
            return self._sub_scalar(other)

        return NotImplemented

    def __rsub__(self, other):
        # Case: scalar - field
        if np.isscalar(other):
            return -1 * self._sub_scalar(other)

        # Case: field - field (only reached if left side failed)
        if isinstance(other, SFCField):
            return other._sub_field(self)

        return NotImplemented

    def _sub_field(self, other):
        self._validate_field_array_operation(other, "subtraction")
        new = self._spawn_like()
        new.epsilon = self.epsilon - other.epsilon
        self._mark_derived(new)
        new.format_sfc_params()
        return new

    def _sub_scalar(self, scalar):
        if self.epsilon is None:
            self.logger.error("Cannot subtract scalar: epsilon is None.")
            func_util.safe_exit(1)

        new = self._spawn_like()
        new.epsilon = self.epsilon - scalar
        self._mark_derived(new)
        new.format_sfc_params()
        return new

    def _validate_field_array_operation(self, other, operation):
        if self.epsilon is None or other.epsilon is None:
            self.logger.error(f"Cannot perform {operation}: epsilon is None.")
            func_util.safe_exit(1)
        if self.epsilon.shape != other.epsilon.shape:
            self.logger.error(
                f"Shape mismatch in {operation}: {self.epsilon.shape} vs {other.epsilon.shape}"
            )
            func_util.safe_exit(1)

    def _derived_field_integral(self, field):
        if getattr(field, "epsilon", None) is None:
            return None
        total = np.sum(field.epsilon, dtype=np.complex128 if np.iscomplexobj(field.epsilon) else np.float64)
        if np.iscomplexobj(total):
            if not np.isfinite(total.real) or not np.isfinite(total.imag):
                return None
            if abs(total.imag) <= 1e-12 * max(1.0, abs(total.real)):
                return float(total.real)
            return complex(total)
        if not np.isfinite(total):
            return None
        return float(total)

    def _mark_derived(self, field):
        field.sfc_info.update({
            "catalog_weight_sum": None,
            "catalog_weight_sq_sum": None,
            "raw_field_weighted_sum": None,
            "field_integral": self._derived_field_integral(field),
            "weight_normalization": None,
            "particle_count": None,
            "particle_data_retrievable": False,
            "particle_data_path": "",
            "particle_data_format": "",
            "field_kind": "derived_field",
        })

    def _normalization_denominator(self, weight_normalization):
        mode = normalize_weight_normalization(weight_normalization)
        if mode == "raw":
            return 1.0
        if mode == "catalog":
            value = getattr(self, "catalog_weight_sum", None)
            label = "catalog_weight_sum"
        else:
            value = getattr(self, "raw_field_weighted_sum", None)
            label = "raw_field_weighted_sum"
        if value is None or not np.isfinite(value) or np.isclose(value, 0.0):
            raise ValueError(f"Cannot use weight_normalization='{mode}' without a finite, non-zero {label}.")
        return float(value)

    def _field_integral_for_normalization(self, weight_normalization):
        raw_sum = getattr(self, "raw_field_weighted_sum", None)
        if raw_sum is None or not np.isfinite(raw_sum):
            return None
        return float(raw_sum / self._normalization_denominator(weight_normalization))

    def _switch_weight_normalization_inplace(self, weight_normalization):
        mode = normalize_weight_normalization(weight_normalization)
        if getattr(self, "field_kind", None) != "catalog_field":
            raise ValueError("weight normalization can only be switched for catalog fields.")
        current = normalize_weight_normalization(getattr(self, "weight_normalization", "raw"))
        if current == mode:
            return self
        old_den = self._normalization_denominator(current)
        new_den = self._normalization_denominator(mode)
        self.epsilon = np.ascontiguousarray(self.epsilon, dtype=np.float64)
        self.epsilon *= old_den / new_den
        self.sfc_info["weight_normalization"] = mode
        self.sfc_info["field_integral"] = self._field_integral_for_normalization(mode)
        self.format_sfc_params()
        return self

    def switch_weight_normalization(self, weight_normalization):
        new = self.copy()
        return new._switch_weight_normalization_inplace(weight_normalization)

    def switch_normalization(self, weight_normalization):
        return self.with_normalization(weight_normalization)

    def to_unit_weight(self):
        if getattr(self, "field_kind", None) == "catalog_field":
            return self.switch_weight_normalization("field")
        total = getattr(self, "field_integral", None)
        if total is None:
            total = self._derived_field_integral(self)
        if not np.isfinite(total) or np.isclose(total, 0.0):
            raise ValueError("Cannot rescale a derived field with zero or non-finite epsilon sum.")
        new = self.copy()
        new.epsilon = new.epsilon / total
        self._mark_derived(new)
        new.format_sfc_params()
        return new

    def with_normalization(self, normalization=None):
        mode = normalize_field_value_normalization(normalization)
        if mode is None:
            return self
        if mode == "unit":
            return self.to_unit_weight()
        if getattr(self, "field_kind", None) != "catalog_field":
            raise ValueError("normalization='raw', 'catalog', or 'field' can only be applied to catalog fields.")
        return self.switch_weight_normalization(mode)

    def _resolve_value_unit(self, value_unit, sample_field=None):
        field = self if sample_field is None else sample_field
        value_unit = str(value_unit).strip().lower()
        if value_unit not in {"auto", "grid", "physical"}:
            raise ValueError("value_unit must be 'auto', 'grid', or 'physical'.")
        if value_unit == "auto":
            value_unit = "physical" if getattr(field, "field_kind", None) == "catalog_field" else "grid"
        return value_unit

    def field_mean_density(self, normalization=None, value_unit="auto"):
        field = self if normalization is None else self.with_normalization(normalization)
        value_unit = self._resolve_value_unit(value_unit, sample_field=field)
        value = getattr(field, "field_integral", None)
        if value is None:
            return None
        volume = float(field.V) if value_unit == "grid" else float(field.box_size) ** 3
        return value / volume
    
    def get_particle_data(self):
        if getattr(self, "field_kind", None) != "catalog_field":
            raise ValueError(
                "Only catalog fields can provide particle data. Derived fields do not retain a unique particle catalogue."
            )
        if not bool(getattr(self, "particle_data_retrievable", True)):
            raise ValueError(
                "This field no longer maps to one retrievable particle catalogue. "
                "Provide particle_pos1 and particle_weight1 explicitly for particle-centred statistics."
            )
        denominator = self._normalization_denominator(getattr(self, "weight_normalization", "raw"))
        particle_data_path = getattr(self, "particle_data_path", "")
        if particle_data_path:
            with np.load(particle_data_path) as particle_data:
                required = {"pos", "catalog_weight", "field_value"}
                if not required.issubset(particle_data.files):
                    self.logger.error(
                        f"Particle dataset '{particle_data_path}' must contain {sorted(required)} arrays."
                    )
                    func_util.safe_exit(1)
                catalog_weight = np.asarray(particle_data["catalog_weight"], dtype=np.float64)
                field_value = np.asarray(particle_data["field_value"], dtype=np.float64)
                return {
                    "pos": particle_data["pos"],
                    "catalog_weight": catalog_weight,
                    "normalized_catalog_weight": catalog_weight / float(self.catalog_weight_sum),
                    "field_value": field_value,
                    "projection_weight": catalog_weight * field_value / denominator,
                }

        fin = getattr(self, "fin", {})
        if not fin.get("path", ""):
            self.logger.error("Input particle path is not specified.")
            func_util.safe_exit(1)
        particle_data = read_particle_data(
            fin["path"],
            fin.get("format", None),
            download=fin.get("download", {}),
            **fin.get("reader_params", {}),
        )
        try:
            catalog_weight, _ = resolve_particle_value(
                particle_data, fin.get("catalog_weight_key", None), label="Catalogue weight", logger=self.logger
            )
            field_value, _ = resolve_particle_value(
                particle_data, fin.get("field_value_key", None), label="Field value", logger=self.logger
            )
        except Exception:
            func_util.safe_exit(1)
        catalog_weight = np.asarray(catalog_weight, dtype=np.float64)
        field_value = np.asarray(field_value, dtype=np.float64)
        return {
            "pos": particle_data["pos"],
            "catalog_weight": catalog_weight,
            "normalized_catalog_weight": catalog_weight / float(self.catalog_weight_sum),
            "field_value": field_value,
            "projection_weight": catalog_weight * field_value / denominator,
        }
    
    def phi_at_pos(self, pos):
        return scaling_stencil_at_pos_numba(pos, self.phi_array, self.scale_factor, self.phi_resolution, self.phi_support)
    
    def field_density_at_pos(self, pos, epsilon=None, filter=None, normalization=None, value_unit="auto"):
        """
        Evaluate the represented field density at positions.

        normalization:
            None      -> use the current SFCField normalization.
            "raw"     -> use raw catalog weights; catalog fields only.
            "catalog" -> divide by catalog_weight_sum; catalog fields only.
            "field"   -> divide by raw_field_weighted_sum; catalog fields only.
            "unit"    -> divide by the field integral; available for all fields.

        value_unit:
            "auto"     -> physical for catalog fields, grid for derived fields.
            "grid"     -> return values in grid-coordinate units.
            "physical" -> convert density-like grid values to box-coordinate units.
        """
        if epsilon is not None and normalization is not None:
            raise ValueError("normalization cannot be applied when an explicit epsilon array is provided.")

        sample_field = self.with_normalization(normalization)
        value_unit = self._resolve_value_unit(value_unit, sample_field=sample_field)

        if epsilon is None:
            if filter is not None:
                epsilon = sample_field._conv(filter).epsilon
            else:
                epsilon = sample_field.epsilon

        npos = pos.shape[0]
        nx = np.empty(npos, dtype=np.float64)
        pos_scaled = pos * sample_field.scale_factor
        interpolate_grid_at_pos_numba(
            nx,
            pos_scaled,
            epsilon,
            sample_field.phi_array,
            sample_field.L,
            sample_field.phi_resolution,
            sample_field.phi_support,
        )

        if value_unit == "physical":
            return nx * (sample_field.scale_factor ** 3)
        return nx

    def as_array(self):
        return self.epsilon

    def format_sfc_params(self):
        missing = [k for k in self._REQUIRED_ARGV if k not in self.sfc_info]
        if missing:
            self.logger.error(f"SFCField missing required keys: {missing}")
            func_util.safe_exit(1)
        for key, value in self.sfc_info.items():
            setattr(self, key, value)

    def load_sfc_field(self, f_in, single=True):
        self.load(f_in, read_sfc_field=True, single=single)

    def save_sfc_field(self, f_out, single=True, overwrite=False):
        self.save(f_out, save_sfc_field=True, single=single, overwrite=overwrite)

    def _load_sfc_field(self, f_in):
        with open(f_in, 'rb') as f:
            # Read the entire .npy file as bytes
            dataset = read_numpy_pickle(f)
            # Check if the 'data' key is present in the dataset
            if 'epsilon' not in dataset:
                self.logger.error("Failed to load the dataset. The file is missing the 'epsilon' key.")
                func_util.safe_exit(1)
            self.epsilon = dataset['epsilon']
            # Assign the dictionary from the file to self.sfc_info
            # _sfc_info = {key: value for key, value in dataset.items() if key != 'epsilon'}
            _sfc_info = dataset.get('sfc_info')
            if _sfc_info:
                self.sfc_info.update(_sfc_info)
            self.format_sfc_params()

    def _save_sfc_field(self, f_out):
        # Check and create directory if it doesn't exist
        _dir = os.path.dirname(f_out)
        if not os.path.exists(_dir):
            os.makedirs(_dir)
        # Check if the sfc_info is empty
        if not self.sfc_info:
            self.logger.error('The dictionary "sfc_info" is empty.')
            self.logger.error('Please ensure that the required data has been loaded or calculated before attempting to save the dataset.')
            self.logger.error(f"Failed to save the data to the file: '{f_out}'")
            func_util.safe_exit(1)
        # If all required variables are present, create the dataset
        dataset = {
            'sfc_info': self.sfc_info,
            'epsilon': self.epsilon  # Include the actual data
        }
        # Save the dataset to the specified file
        #  ↓ Use Pickle with protocol 4 or higher to handle saving files larger than 4 GiB
        with open(f_out, 'wb') as f:
            write_numpy_pickle(f, dataset)

    
