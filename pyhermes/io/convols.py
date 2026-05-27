import os
import pickle
import copy
import uuid

import numpy as np

from .base import HermesData
from .readers import read_particle_data, resolve_particle_value
from pyhermes.utils import func_util
from pyhermes.utils.convolution import specialized_convolution_3d, specialized_convolution_3d_complex
from pyhermes.utils.wavelet_grid import interpolate_grid_at_pos_numba, scaling_stencil_at_pos_numba


def normalize_field_normalization(field_normalization):
    mode = str(field_normalization if field_normalization is not None else "none").strip().lower()
    if mode not in {"none", "mean"}:
        raise ValueError("field_normalization must be 'none' or 'mean'.")
    return mode


class ConvolsData(HermesData):
    _REQUIRED_ARGV = ("J", "box_size", "phi_resolution", "wavelet_mode", "wavelet_level")

    def __init__(self, *args, threads=None, **kwargs):
        data_path = kwargs.pop("data_path", None)
        self.convols_info = {}
        self.epsilon = None
        super().__init__(*args, threads=threads, **kwargs)
        if data_path:
            self.convols_info['convols_data_path'] = data_path
            self.load_convols(data_path)

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
        new.convols_info = copy.deepcopy(self.convols_info)
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
        new.format_convols_params()
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

        # --- spawn new ConvolsData ---
        new = self._spawn_like()

        # --- set result ---
        new.epsilon = conv

        # --- optional: record provenance ---
        windows = self.convols_info.get("convolution_of", [])
        # windows.append(other.window_params)
        new.convols_info.update({
            "convolution_of": windows + [other.window_params]
        })
        new.format_convols_params()

        return new

    def __matmul__(self, other):
        if isinstance(other, ConvolsData):
            return self._conv(other)
        return NotImplemented

    def __rmatmul__(self, other):
        if isinstance(other, ConvolsData):
            return other._conv(self)  
        return NotImplemented

    def __add__(self, other):
        if isinstance(other, ConvolsData):
            return self._add_field(other)

        if np.isscalar(other):
            return self._add_scalar(other)

        return NotImplemented

    def __radd__(self, other):
        if np.isscalar(other):
            return self._add_scalar(other)

        if isinstance(other, ConvolsData):
            return other._add_field(self)

        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, ConvolsData):
            return self._mul_field(other)

        if np.isscalar(other):
            return self._mul_scalar(other)

        return NotImplemented

    def __rmul__(self, other):
        if np.isscalar(other):
            return self._mul_scalar(other)

        if isinstance(other, ConvolsData):
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
        if self._same_catalog_measure(other):
            self._propagate_catalog_linear(new, other, sign=1.0)
        else:
            self._mark_derived(new)
        new.format_convols_params()
        return new

    # ---------- field + scalar ----------
    def _add_scalar(self, scalar):
        if self.epsilon is None:
            self.logger.error("Cannot add scalar: epsilon is None.")
            func_util.safe_exit(1)

        new = self._spawn_like()
        new.epsilon = self.epsilon + scalar
        integral = self._shifted_integral(scalar)
        self._mark_derived(new, field_integral=integral)
        new.format_convols_params()
        return new

    # ---------- field × field ----------
    def _mul_field(self, other):
        self._validate_field_array_operation(other, "multiplication")
        new = self._spawn_like()
        new.epsilon = self.epsilon * other.epsilon
        self._mark_derived(new, field_kind="derived_field")
        new.format_convols_params()
        return new

    # ---------- field × scalar ----------
    def _mul_scalar(self, scalar):
        if self.epsilon is None:
            self.logger.error("Cannot multiply by scalar: epsilon is None.")
            func_util.safe_exit(1)

        new = self._spawn_like()
        new.epsilon = self.epsilon * scalar
        self._scale_linear_metadata(new, scalar)
        new.format_convols_params()
        return new

    # ---------- field / scalar ----------
    def _div_scalar(self, scalar):
        if self.epsilon is None:
            self.logger.error("Cannot divide by scalar: epsilon is None.")
            func_util.safe_exit(1)
        if scalar == 0.0:
            self.logger.error("Cannot divide ConvolsData by zero.")
            func_util.safe_exit(1)

        new = self._spawn_like()
        new.epsilon = self.epsilon / scalar
        self._scale_linear_metadata(new, 1.0 / scalar)
        new.format_convols_params()
        return new

    def __sub__(self, other):
        # Case 1: field - field
        if isinstance(other, ConvolsData):
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
        if isinstance(other, ConvolsData):
            return other._sub_field(self)

        return NotImplemented

    def _sub_field(self, other):
        self._validate_field_array_operation(other, "subtraction")
        new = self._spawn_like()
        new.epsilon = self.epsilon - other.epsilon
        if self._same_catalog_measure(other):
            self._propagate_catalog_linear(new, other, sign=-1.0)
        elif getattr(self, "field_kind", None) == "catalog_field" and getattr(other, "field_kind", None) == "catalog_field":
            integral = self._difference_integral(other)
            self._mark_contrast(new, integral, (self.catalog_weight_sum, other.catalog_weight_sum))
        else:
            self._mark_derived(new)
        new.format_convols_params()
        return new

    def _sub_scalar(self, scalar):
        if self.epsilon is None:
            self.logger.error("Cannot subtract scalar: epsilon is None.")
            func_util.safe_exit(1)

        new = self._spawn_like()
        new.epsilon = self.epsilon - scalar
        integral = self._shifted_integral(-scalar)
        if getattr(self, "field_kind", None) == "catalog_field":
            self._mark_contrast(new, integral, self.catalog_weight_sum)
        else:
            self._mark_derived(new, field_integral=integral)
        new.format_convols_params()
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

    def _same_catalog_measure(self, other):
        return (
            getattr(self, "field_kind", None) == "catalog_field"
            and getattr(other, "field_kind", None) == "catalog_field"
            and bool(getattr(self, "catalog_recombinable", False))
            and bool(getattr(other, "catalog_recombinable", False))
            and getattr(self, "catalog_id", None) is not None
            and self.catalog_id == getattr(other, "catalog_id", None)
            and self.convols_info.get("convolution_of", []) == other.convols_info.get("convolution_of", [])
        )

    def _propagate_catalog_linear(self, field, other, sign):
        raw_self = getattr(self, "raw_field_weighted_sum", None)
        raw_other = getattr(other, "raw_field_weighted_sum", None)
        integral_self = getattr(self, "field_integral", None)
        integral_other = getattr(other, "field_integral", None)
        scale_self = getattr(self, "particle_projection_scale", None)
        scale_other = getattr(other, "particle_projection_scale", None)
        field.convols_info.update({
            "raw_field_weighted_sum": None if raw_self is None or raw_other is None else raw_self + sign * raw_other,
            "field_integral": None if integral_self is None or integral_other is None else integral_self + sign * integral_other,
            "particle_projection_scale": None if scale_self is None or scale_other is None else scale_self + sign * scale_other,
            "field_kind": "catalog_field",
        })

    def _shifted_integral(self, scalar):
        value = getattr(self, "field_integral", None)
        if value is None or not hasattr(self, "V"):
            return None
        return float(value + scalar * self.V)

    def _difference_integral(self, other):
        left = getattr(self, "field_integral", None)
        right = getattr(other, "field_integral", None)
        return None if left is None or right is None else float(left - right)

    def _mark_derived(self, field, field_kind="derived_field", field_integral=None):
        field.convols_info.update({
            "catalog_weight_sum": None,
            "catalog_weight_sq_sum": None,
            "raw_field_weighted_sum": None,
            "field_integral": field_integral,
            "catalog_id": None,
            "field_normalization": "none",
            "field_normalization_value": None,
            "catalog_recombinable": False,
            "particle_data_retrievable": False,
            "particle_projection_scale": None,
            "field_kind": field_kind,
        })

    def _mark_contrast(self, field, field_integral, source_weight_sum):
        self._mark_derived(field, field_kind="contrast_field", field_integral=field_integral)
        field.convols_info["source_catalog_weight_sum"] = source_weight_sum

    def _scale_linear_metadata(self, field, scalar):
        if getattr(self, "field_kind", None) == "catalog_field":
            weighted_sum = getattr(self, "raw_field_weighted_sum", None)
            integral = getattr(self, "field_integral", None)
            projection_scale = getattr(self, "particle_projection_scale", None)
            field.convols_info["raw_field_weighted_sum"] = (
                None if weighted_sum is None else weighted_sum * scalar
            )
            field.convols_info["field_integral"] = None if integral is None else integral * scalar
            field.convols_info["particle_projection_scale"] = (
                None if projection_scale is None else projection_scale * scalar
            )
            return
        if getattr(self, "field_kind", None) == "contrast_field":
            integral = getattr(self, "field_integral", None)
            field.convols_info["field_integral"] = None if integral is None else integral * scalar
            return
        self._mark_derived(field, field_integral=None)

    def _field_normalization_value(self, field_normalization):
        field_normalization = normalize_field_normalization(field_normalization)
        if field_normalization == "none":
            return field_normalization, None
        value = getattr(self, "field_integral", None)
        if value is None or not np.isfinite(value) or np.isclose(value, 0.0):
            raise ValueError(
                "Cannot apply field_normalization='mean' without a finite, non-zero field_integral."
            )
        return field_normalization, float(value)

    def normalized(self, field_normalization="mean"):
        """
        Return a new field optionally divided by its catalogue-weighted mean.

        Catalogue weights have already been normalized when the field is
        constructed. ``mean`` divides a physical-value field by
        ``field_integral``; ``none`` copies it without additional scaling.
        """
        new = self.copy()
        return new._normalize_for_estimator_inplace(field_normalization)

    def to_unit_weight(self):
        """
        Return a new field rescaled to unit catalogue-weighted field mean.

        This public convenience interface is equivalent to ``normalized("mean")``.
        For an ordinary count field the operation is numerically a no-op.
        """
        return self.normalized("mean")

    def _normalize_for_estimator_inplace(self, field_normalization="none"):
        """
        Normalize an estimator-owned temporary field in place.

        Callers must ensure that this object is not a user-visible raw input field.
        This avoids allocating another full epsilon array after a leg has already
        been copied or convolved for an estimator.
        """
        field_normalization, normalizer = self._field_normalization_value(field_normalization)
        if field_normalization == "none":
            return self
        if getattr(self, "field_kind", None) == "contrast_field":
            raise ValueError("field_normalization cannot be applied after constructing a contrast field.")
        current = getattr(self, "field_normalization", "none")
        if current == field_normalization:
            return self
        if current != "none":
            raise ValueError(
                f"Cannot apply field_normalization='{field_normalization}' to a field already normalized as '{current}'."
            )
        self.epsilon /= normalizer
        self.convols_info["raw_field_weighted_sum"] = None
        self.convols_info["field_integral"] = 1.0
        self.convols_info["field_normalization"] = field_normalization
        self.convols_info["field_normalization_value"] = normalizer
        self.convols_info["catalog_recombinable"] = False
        self.format_convols_params()
        return self

    def as_estimator_field(self, field_normalization="none"):
        """
        Return a new field in the normalization expected by correlation estimators.

        Estimator implementations use an internal in-place conversion on their
        own temporary leg fields to avoid an unnecessary full-array allocation.
        """
        return self.normalized(field_normalization)

    def _validate_catalog_operation(self, other, operation):
        if not isinstance(other, ConvolsData):
            raise TypeError(f"{operation} requires another ConvolsData object.")
        self._validate_field_array_operation(other, operation)
        for field, label in ((self, "left"), (other, "right")):
            if getattr(field, "field_kind", None) != "catalog_field" or not bool(
                getattr(field, "catalog_recombinable", False)
            ):
                raise ValueError(
                    f"{operation} requires catalogue fields before contrast/product construction "
                    f"or field-mean normalization; the {label} field is not recombinable."
                )
        required = self._REQUIRED_ARGV + ("L",)
        mismatched = [
            key for key in required
            if getattr(self, key, None) != getattr(other, key, None)
        ]
        if mismatched:
            raise ValueError(f"{operation} requires matching grid metadata; mismatched keys: {mismatched}.")
        if self.convols_info.get("convolution_of", []) != other.convols_info.get("convolution_of", []):
            raise ValueError(f"{operation} requires matching linear window histories.")
        left_value = getattr(self, "fin", {}).get("field_value_key")
        right_value = getattr(other, "fin", {}).get("field_value_key")
        if left_value not in (None, "custom") and right_value not in (None, "custom") and left_value != right_value:
            raise ValueError(f"{operation} requires fields carrying the same physical quantity.")

    def _catalog_recombine(self, other, sign, operation):
        self._validate_catalog_operation(other, operation)
        left_sum = float(self.catalog_weight_sum)
        right_sum = float(other.catalog_weight_sum)
        total_sum = left_sum + sign * right_sum
        if not np.isfinite(total_sum) or total_sum <= 0.0:
            raise ValueError(f"{operation} requires a positive remaining catalogue_weight_sum.")
        new = self._spawn_like()
        new.epsilon = (left_sum * self.epsilon + sign * right_sum * other.epsilon) / total_sum
        left_raw = getattr(self, "raw_field_weighted_sum", None)
        right_raw = getattr(other, "raw_field_weighted_sum", None)
        raw_total = None if left_raw is None or right_raw is None else left_raw + sign * right_raw
        left_sq = getattr(self, "catalog_weight_sq_sum", None)
        right_sq = getattr(other, "catalog_weight_sq_sum", None)
        sq_total = None if left_sq is None or right_sq is None else left_sq + sign * right_sq
        left_count = getattr(self, "particle_count", None)
        right_count = getattr(other, "particle_count", None)
        particle_count = (
            None if left_count is None or right_count is None
            else int(left_count + int(sign) * right_count)
        )
        new.convols_info.update({
            "particle_count": particle_count,
            "catalog_weight_sum": total_sum,
            "catalog_weight_sq_sum": sq_total,
            "raw_field_weighted_sum": raw_total,
            "field_integral": None if raw_total is None else raw_total / total_sum,
            "catalog_id": uuid.uuid4().hex,
            "field_kind": "catalog_field",
            "field_normalization": "none",
            "field_normalization_value": None,
            "catalog_recombinable": True,
            "particle_data_retrievable": False,
            "particle_projection_scale": None,
            "catalog_operation": operation,
        })
        new.format_convols_params()
        return new

    def combine_catalog(self, other):
        """Return the catalogue-normalized union of two disjoint catalogue fields."""
        return self._catalog_recombine(other, sign=1.0, operation="combine_catalog")

    def exclude_catalog(self, removed):
        """Return the catalogue-normalized field after removing a projected subset."""
        return self._catalog_recombine(removed, sign=-1.0, operation="exclude_catalog")
    
    def get_particle_data(self):
        if not bool(getattr(self, "particle_data_retrievable", True)):
            raise ValueError(
                "This field no longer maps to one retrievable particle catalogue. "
                "Provide particle_pos1 and particle_weight1 explicitly for particle-centred statistics."
            )
        projection_scale = getattr(self, "particle_projection_scale", 1.0)
        particle_data_path = getattr(self, "particle_data_path", "")
        if particle_data_path:
            with np.load(particle_data_path) as particle_data:
                required = {"pos", "catalog_weight", "field_value"}
                if not required.issubset(particle_data.files):
                    self.logger.error(
                        f"Particle dataset '{particle_data_path}' must contain {sorted(required)} arrays."
                    )
                    func_util.safe_exit(1)
                catalog_weight = particle_data["catalog_weight"]
                field_value = particle_data["field_value"] * projection_scale
                catalog_weight_sum = float(self.catalog_weight_sum)
                return {
                    "pos": particle_data["pos"],
                    "catalog_weight": catalog_weight,
                    "normalized_catalog_weight": catalog_weight / catalog_weight_sum,
                    "field_value": field_value,
                    "projection_weight": catalog_weight * field_value / catalog_weight_sum,
                }

        fin = getattr(self, "fin", {})
        if fin.get("url"):
            self.logger.error("fin.url is no longer supported. Download the data first and set fin.path.")
            func_util.safe_exit(1)
        if not fin.get("path", ""):
            self.logger.error("Input particle path is not specified.")
            func_util.safe_exit(1)
        particle_data = read_particle_data(
            fin["path"],
            fin.get("format", None),
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
        field_value = field_value * projection_scale
        return {
            "pos": particle_data["pos"],
            "catalog_weight": catalog_weight,
            "normalized_catalog_weight": catalog_weight / float(self.catalog_weight_sum),
            "field_value": field_value,
            "projection_weight": catalog_weight * field_value / float(self.catalog_weight_sum),
        }
    
    def phi_at_pos(self, pos):
        return scaling_stencil_at_pos_numba(pos, self.phi_array, self.scale_factor, self.phi_resolution, self.phi_support)
    
    def n_at_pos(self, pos, epsilon=None, filter=None, physical=True):
        """
        Evaluate the represented field at positions.

        physical:
            True  -> convert a catalogue-normalized field from grid to physical coordinates.
            False -> return the grid-coordinate value. Use this for derived fields
                     such as a dimensionless density contrast.
        """
        if epsilon is None:
            if filter is not None:
                epsilon = self._conv(filter).epsilon
            else:
                epsilon = self.epsilon

        npos = pos.shape[0]
        nx = np.empty(npos, dtype=np.float64)
        pos_scaled = pos * self.scale_factor
        interpolate_grid_at_pos_numba(
            nx, pos_scaled, epsilon, self.phi_array, self.L, self.phi_resolution, self.phi_support
        )

        if physical:
            return nx * (self.scale_factor ** 3)
        return nx

    def as_array(self):
        return self.epsilon

    def format_convols_params(self):
        missing = [k for k in self._REQUIRED_ARGV if k not in self.convols_info]
        if missing:
            self.logger.error(f"ConvolsData missing required keys: {missing}")
            func_util.safe_exit(1)
        for key, value in self.convols_info.items():
            setattr(self, key, value)

    def load_convols(self, f_in, single=True):
        self.load(f_in, read_convols=True, single=single)

    def save_convols(self, f_out, single=True, overwrite=False):
        self.save(f_out, save_convols=True, single=single, overwrite=overwrite)

    def _load_convols(self, f_in):
        with open(f_in, 'rb') as f:
            # Read the entire .npy file as bytes
            serialized_data = np.lib.format.read_array(f, allow_pickle=True)
            # Convert the bytes back into the original dataset using pickle
            dataset = pickle.loads(serialized_data.tobytes())
            # Check if the 'data' key is present in the dataset
            if 'epsilon' not in dataset:
                self.logger.error(f"Failed to load the dataset. The file is missing the 'epsilon' key.")
                func_util.safe_exit(1)
            self.epsilon = dataset['epsilon']
            # Assign the dictionary from the file to self.convols_info
            # _convols_info = {key: value for key, value in dataset.items() if key != 'epsilon'}
            _convols_info = dataset.get('convols_info')
            if _convols_info:
                self.convols_info.update(_convols_info)
            if self.convols_info.get("coefficient_convention") != "normalized_catalog_weight_field_value":
                raise ValueError(
                    "This ConvolsData file does not use the catalogue-normalized field convention. "
                    "Regenerate it with the current Convols task before loading it."
                )
            self.format_convols_params()

    def _save_convols(self, f_out):
        # Check and create directory if it doesn't exist
        _dir = os.path.dirname(f_out)
        if not os.path.exists(_dir):
            os.makedirs(_dir)
        # Check if the convols_info is empty
        if not self.convols_info:
            self.logger.error('The dictionary "convols_info" is empty.')
            self.logger.error('Please ensure that the required data has been loaded or calculated before attempting to save the dataset.')
            self.logger.error(f"Failed to save the data to the file: '{f_out}'")
            func_util.safe_exit(1)
        # If all required variables are present, create the dataset
        dataset = {
            'convols_info': self.convols_info,
            'epsilon': self.epsilon  # Include the actual data
        }
        # Save the dataset to the specified file
        #  ↓ Use Pickle with protocol 4 or higher to handle saving files larger than 4 GiB
        _serialized_data = pickle.dumps(dataset, protocol=4)
        with open(f_out, 'wb') as f:
            np.lib.format.write_array(f, np.frombuffer(_serialized_data, dtype=np.uint8))

    
