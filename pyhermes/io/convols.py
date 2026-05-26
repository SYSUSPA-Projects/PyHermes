import os
import pickle
import copy

import numpy as np

from .base import HermesData
from .readers import read_particle_data, resolve_particle_weight
from pyhermes.utils import func_util
from pyhermes.utils.convolution import specialized_convolution_3d, specialized_convolution_3d_complex
from pyhermes.utils.wavelet_grid import interpolate_grid_at_pos_numba, scaling_stencil_at_pos_numba


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
        a = self.epsilon
        b = other.epsilon

        if a is None or b is None:
            self.logger.error("Cannot add: epsilon is None.")
            func_util.safe_exit(1)

        if a.shape != b.shape:
            self.logger.error(
                f"Shape mismatch in addition: {a.shape} vs {b.shape}"
            )
            func_util.safe_exit(1)

        new = self._spawn_like()
        new.epsilon = a + b
        if getattr(self, "weight_sum", None) is not None and getattr(other, "weight_sum", None) is not None:
            new.convols_info["weight_sum"] = self.weight_sum + other.weight_sum
        else:
            new.convols_info["weight_sum"] = None
        new.format_convols_params()
        return new

    # ---------- field + scalar ----------
    def _add_scalar(self, scalar):
        if self.epsilon is None:
            self.logger.error("Cannot add scalar: epsilon is None.")
            func_util.safe_exit(1)

        new = self._spawn_like()
        new.epsilon = self.epsilon + scalar
        if getattr(self, "weight_sum", None) is not None:
            new.convols_info["weight_sum"] = self.weight_sum + scalar * self.V
        else:
            new.convols_info["weight_sum"] = None
        new.format_convols_params()
        return new

    # ---------- field × field ----------
    def _mul_field(self, other):
        a = self.epsilon
        b = other.epsilon

        if a is None or b is None:
            self.logger.error("Cannot multiply: epsilon is None.")
            func_util.safe_exit(1)

        if a.shape != b.shape:
            self.logger.error(
                f"Shape mismatch in multiplication: {a.shape} vs {b.shape}"
            )
            func_util.safe_exit(1)

        new = self._spawn_like()
        new.epsilon = a * b
        new.convols_info["weight_sum"] = None
        new.convols_info["field_kind"] = "product"
        new.format_convols_params()
        return new

    # ---------- field × scalar ----------
    def _mul_scalar(self, scalar):
        if self.epsilon is None:
            self.logger.error("Cannot multiply by scalar: epsilon is None.")
            func_util.safe_exit(1)

        new = self._spawn_like()
        new.epsilon = self.epsilon * scalar
        if getattr(self, "weight_sum", None) is not None:
            new.convols_info["weight_sum"] = self.weight_sum * scalar
        else:
            new.convols_info["weight_sum"] = None
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
        if getattr(self, "weight_sum", None) is not None:
            new.convols_info["weight_sum"] = self.weight_sum / scalar
        else:
            new.convols_info["weight_sum"] = None
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
        a = self.epsilon
        b = other.epsilon

        if a is None or b is None:
            self.logger.error("Cannot subtract: epsilon is None.")
            func_util.safe_exit(1)

        if a.shape != b.shape:
            self.logger.error(
                f"Shape mismatch in subtraction: {a.shape} vs {b.shape}"
            )
            func_util.safe_exit(1)

        new = self._spawn_like()
        new.epsilon = a - b
        if getattr(self, "weight_sum", None) is not None and getattr(other, "weight_sum", None) is not None:
            new.convols_info["weight_sum"] = self.weight_sum - other.weight_sum
        else:
            new.convols_info["weight_sum"] = None
        if (
            getattr(self, "field_kind", None) == "unit_weight_density"
            and getattr(other, "field_kind", None) == "unit_weight_density"
            and new.convols_info["weight_sum"] is not None
            and np.isclose(new.convols_info["weight_sum"], 0.0)
        ):
            new.convols_info["field_kind"] = "contrast"
        new.format_convols_params()
        return new

    def _sub_scalar(self, scalar):
        if self.epsilon is None:
            self.logger.error("Cannot subtract scalar: epsilon is None.")
            func_util.safe_exit(1)

        new = self._spawn_like()
        new.epsilon = self.epsilon - scalar
        if getattr(self, "weight_sum", None) is not None:
            new.convols_info["weight_sum"] = self.weight_sum - scalar * self.V
        else:
            new.convols_info["weight_sum"] = None
        if (
            getattr(self, "field_kind", None) == "unit_weight_density"
            and new.convols_info["weight_sum"] is not None
            and np.isclose(new.convols_info["weight_sum"], 0.0)
        ):
            new.convols_info["field_kind"] = "contrast"
        new.format_convols_params()
        return new

    def to_unit_weight(self):
        """
        Return a temporary field rescaled to unit total weight.

        Raw weighted coefficients are the public field representation. Correlation
        estimators use this conversion internally so their normalized statistics
        retain the conventional unit-selection-field interpretation.
        """
        weight_sum = getattr(self, "weight_sum", None)
        if weight_sum is None or not np.isfinite(weight_sum) or np.isclose(weight_sum, 0.0):
            raise ValueError("Cannot rescale a field without a finite, non-zero weight_sum.")
        new = self / weight_sum
        new.convols_info["field_kind"] = "unit_weight_density"
        new.format_convols_params()
        return new

    def as_estimator_field(self):
        """
        Return this field in the normalization expected by correlation estimators.

        Density-like fields are converted to unit total weight. A pre-constructed
        contrast field already carries the desired estimator normalization and is
        therefore passed through unchanged.
        """
        if getattr(self, "field_kind", None) == "contrast":
            return self.copy()
        return self.to_unit_weight()
    
    def get_particle_data(self):
        particle_data_path = getattr(self, "particle_data_path", "")
        if particle_data_path:
            with np.load(particle_data_path) as particle_data:
                if "pos" not in particle_data or "weight" not in particle_data:
                    self.logger.error(
                        f"Particle dataset '{particle_data_path}' must contain 'pos' and 'weight' arrays."
                    )
                    func_util.safe_exit(1)
                return {
                    "pos": particle_data["pos"],
                    "weight": particle_data["weight"],
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
            weight, _ = resolve_particle_weight(particle_data, fin.get("weight_key", None), logger=self.logger)
        except Exception:
            func_util.safe_exit(1)
        return {
            "pos": particle_data["pos"],
            "weight": weight,
        }
    
    def phi_at_pos(self, pos):
        return scaling_stencil_at_pos_numba(pos, self.phi_array, self.scale_factor, self.phi_resolution, self.phi_support)
    
    def n_at_pos(self, pos, epsilon=None, filter=None, physical=True):
        """
        Evaluate the represented field at positions.

        physical:
            True  -> convert a raw weighted density field from grid to physical coordinates.
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
            if self.convols_info.get("coefficient_convention") != "raw_weighted":
                raise ValueError(
                    "This ConvolsData file does not use the raw-weighted coefficient convention. "
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

    
