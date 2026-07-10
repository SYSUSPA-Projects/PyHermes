import os
import pickle
import copy

import numpy as np

from .sfc_field import SFCField
from pyhermes.utils import func_util
from pyhermes.utils.convolution import (
    build_complex_window_rfft_kernel,
    build_real_window_octant_array,
    build_real_window_rfft_kernel,
    fold_octant_window_to_rfft_kernel,
)
from pyhermes.utils.wavelet_grid import fourier_power_spectrum, sample_scaling_function
from pyhermes.utils.window_functions import set_window_function
from pyhermes.utils.window_params import (
    ANISOTROPIC_AUTO_WINDOW_TYPES,
    COMPLEX_FULL_FFT_WINDOW_TYPES,
    COMPLEX_RFFT_WINDOW_TYPES,
    LOS_ARG_KEYS,
    default_kernel_mode,
    normalize_los_args,
    normalize_kernel_mode,
    serialize_window_params,
)


ZERO_MODE_ZERO_WINDOW_TYPES = {"inverse_laplacian"}


class WindowFunc(SFCField):
    def __init__(
        self,
        win_params,
        sfc_params,
        bandwidth=1,
        threads=1,
        phi_array=None,
        phi_fourier_power=None,
    ):
        # Initial MPI, logger mess
        super().__init__(threads=threads)
        missing = [k for k in self._REQUIRED_ARGV if k not in sfc_params]
        if missing:
            if self.rank == 0:
                self.logger.error(f"WindowFunc missing required keys: {missing}")
            func_util.safe_exit(1)
        try:
            self.L = 1 << int(sfc_params['J'])
            self.bandwidth = int(bandwidth)
            self.box_size = sfc_params["box_size"]
            self.phi_resolution = int(sfc_params["phi_resolution"])
            self.wavelet_mode = sfc_params["wavelet_mode"]
            self.wavelet_level = sfc_params["wavelet_level"]
        except Exception as e:
            if self.rank == 0:
                self.logger.error(f"WindowFunc core parameter error: {e}")
            func_util.safe_exit(1)
        self.input_params = {
            **dict(win_params),
            "J": int(sfc_params["J"]),
            "box_size": self.box_size,
            "phi_resolution": self.phi_resolution,
            "wavelet_mode": self.wavelet_mode,
            "wavelet_level": self.wavelet_level,
            "bandwidth": self.bandwidth,
        }
        if phi_array is None:
            phi_array = sample_scaling_function(self.wavelet_mode, self.wavelet_level)
        self.phi_array = phi_array
        if phi_fourier_power is None:
            phi_fourier_power = fourier_power_spectrum(
                phi_array, 0, self.bandwidth, self.L * self.bandwidth, self.phi_resolution
            )
        self.phi_fourier_power = phi_fourier_power
        if not isinstance(self.phi_fourier_power, np.ndarray):
            if self.rank == 0:
                self.logger.error("WindowFunc requires phi_fourier_power to be a numpy.ndarray.")
            func_util.safe_exit(1)
        if self.L <= 0 or self.bandwidth <= 0:
            if self.rank == 0:
                self.logger.error(f"Invalid L/bandwidth: L={self.L}, bandwidth={self.bandwidth}. Must be > 0.")
            func_util.safe_exit(1)
        # There is NO DEFAULT window!!!
        # Missing `type` or `func` will raise an error in set_window_function.
        self.window_params = dict(win_params)
        has_custom_func = "func" in win_params
        self.has_custom_func = has_custom_func
        if has_custom_func:
            self.type = win_params.get('type', None) or "custom"
            self.func = win_params["func"]
        elif win_params.get("type") in COMPLEX_FULL_FFT_WINDOW_TYPES:
            self.type = win_params["type"]
            self.func = None
        else:
            assert "type" in win_params
            self.type = win_params['type']
            self.func = set_window_function(self.type, verbose=False)
        self.kernel_mode = self._resolve_kernel_mode(win_params, has_custom_func)
        self.input_params["kernel_mode"] = self.kernel_mode
        self.window_params["kernel_mode"] = self.kernel_mode
        self.len_args = copy.deepcopy(win_params.get('len_args', {}))
        if self.len_args is None:
            self.len_args = {}
        self.input_params["len_args"] = copy.deepcopy(self.len_args)
        self.window_params["len_args"] = copy.deepcopy(self.len_args)
        self.rescale_len_args = {k: v * self.L / self.box_size for k, v in self.len_args.items()}
        self.los_args = normalize_los_args(win_params.get('los_args', {}), self.type)
        self.other_args = win_params.get('other_args', {})
        self.rescale_other_args = self._rescale_other_args(self.other_args)
        self.input_params["los_args"] = self.los_args
        self.window_params["los_args"] = self.los_args
        self.window_args = dict(self.rescale_len_args)
        self.window_args.update(self.los_args)
        self.window_args.update(self.rescale_other_args)
        self._validate_builtin_arguments()
        self.w_kernel = None
        self.is_composite_window = False

    def _rescale_other_args(self, other_args):
        if other_args is None:
            return {}
        return copy.deepcopy(other_args)

    def _validate_builtin_arguments(self):
        if self.type != "thick_shell":
            return
        try:
            radius = float(self.rescale_len_args["R"])
            width = float(self.rescale_len_args["delta_R"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("thick_shell requires finite len_args {'R': ..., 'delta_R': ...}.") from exc
        if not np.isfinite(radius) or radius < 0.0:
            raise ValueError("thick_shell requires a finite non-negative R.")
        if not np.isfinite(width) or width <= 0.0:
            raise ValueError("thick_shell requires a finite positive delta_R.")

    def _resolve_kernel_mode(self, win_params, has_custom_func):
        kernel_mode = win_params.get("kernel_mode", None)
        if not kernel_mode:
            kernel_mode = default_kernel_mode(self.type, has_custom_func=has_custom_func)
        return normalize_kernel_mode(kernel_mode)

    def _los_is_axis_aligned(self):
        los_args = dict(self.los_args)
        if self.type not in ANISOTROPIC_AUTO_WINDOW_TYPES and not all(
            key in los_args for key in LOS_ARG_KEYS
        ):
            return not self.has_custom_func
        nx = float(los_args.get("nx", 0.0))
        ny = float(los_args.get("ny", 0.0))
        nz = float(los_args.get("nz", 1.0))
        los = np.array([nx, ny, nz], dtype=np.float64)
        norm = np.linalg.norm(los)
        if norm == 0.0:
            return True
        los = np.abs(los / norm)
        return (
            np.isclose(los[0], 1.0) and np.isclose(los[1], 0.0) and np.isclose(los[2], 0.0)
            or np.isclose(los[0], 0.0) and np.isclose(los[1], 1.0) and np.isclose(los[2], 0.0)
            or np.isclose(los[0], 0.0) and np.isclose(los[1], 0.0) and np.isclose(los[2], 1.0)
        )

    def _requires_full_rfft_kernel(self):
        if self.kernel_mode == "full_rfft":
            return True
        if self.kernel_mode == "octant":
            return False
        return not self._los_is_axis_aligned()

    def _requires_complex_rfft_kernel(self):
        if self.kernel_mode == "complex_rfft":
            return True
        return self.kernel_mode in ("auto", "full_rfft") and self.type in COMPLEX_RFFT_WINDOW_TYPES

    def _requires_complex_full_fft_kernel(self):
        if self.kernel_mode == "complex_full_fft":
            return True
        return self.kernel_mode == "auto" and self.type in COMPLEX_FULL_FFT_WINDOW_TYPES

    def _build_complex_full_fft_kernel(self):
        if self.type not in {"legendre_multipole", "radial_multipole"}:
            if self.rank == 0:
                self.logger.error(f"Unsupported complex full-FFT window type: {self.type}")
            func_util.safe_exit(1)
        if self.bandwidth != 1:
            if self.rank == 0:
                self.logger.error(f"{self.type} currently supports only bandwidth=1.")
            func_util.safe_exit(1)
        if self.type == "radial_multipole":
            try:
                radial_type = str(self.other_args["radial_type"])
                l = int(self.other_args["l"])
                m = int(self.other_args["m"])
            except KeyError as exc:
                if self.rank == 0:
                    self.logger.error(f"radial_multipole missing required other_args key: {exc}")
                func_util.safe_exit(1)
            from pyhermes.utils.radial_multipole_windows import calculate_radial_multipole_window_array

            try:
                self.w_kernel = calculate_radial_multipole_window_array(
                    self.L,
                    self.phi_fourier_power,
                    self.len_args,
                    radial_type,
                    l,
                    m,
                    box_size=self.box_size,
                    profile_config=self.other_args.get("profile_config", {}),
                )
            except Exception as exc:
                if self.rank == 0:
                    self.logger.error(f"Failed to build radial_multipole window: {exc}")
                func_util.safe_exit(1)
            return
        if "R" not in self.rescale_len_args:
            if self.rank == 0:
                self.logger.error("legendre_multipole requires len_args={'R': ...}.")
            func_util.safe_exit(1)
        try:
            radius = float(self.rescale_len_args["R"])
            l = int(self.other_args["l"])
            m = int(self.other_args["m"])
            use_fast = bool(self.other_args.get("use_fast", True))
        except KeyError as exc:
            if self.rank == 0:
                self.logger.error(f"legendre_multipole missing required other_args key: {exc}")
            func_util.safe_exit(1)
        from pyhermes.utils.legendre_windows import calculate_legendre_window_array

        self.w_kernel = calculate_legendre_window_array(
            self.L,
            self.phi_fourier_power,
            radius,
            l,
            m,
            use_fast=use_fast,
        )

    def _apply_zero_mode_convention(self):
        if self.type in ZERO_MODE_ZERO_WINDOW_TYPES and self.w_kernel is not None:
            self.w_kernel[0, 0, 0] = 0.0

    def _build_kernel(self):
        if getattr(self, "is_composite_window", False):
            if self.rank == 0:
                self.logger.error("Composite WindowFunc already stores a materialized w_kernel and cannot rebuild it.")
            func_util.safe_exit(1)
        if self.type in COMPLEX_RFFT_WINDOW_TYPES and self.kernel_mode == "octant":
            if self.rank == 0:
                self.logger.error(
                    f"Window type '{self.type}' is complex and directional; "
                    "use kernel_mode='complex_rfft' or kernel_mode='auto'."
                )
            func_util.safe_exit(1)
        if self.type in COMPLEX_FULL_FFT_WINDOW_TYPES and self.kernel_mode not in ("auto", "complex_full_fft"):
            if self.rank == 0:
                self.logger.error(
                    f"Window type '{self.type}' requires kernel_mode='complex_full_fft' or kernel_mode='auto'."
                )
            func_util.safe_exit(1)
        if self._requires_complex_full_fft_kernel():
            self._build_complex_full_fft_kernel()
            self._apply_zero_mode_convention()
            return
        if self._requires_complex_rfft_kernel():
            self.w_kernel = build_complex_window_rfft_kernel(
                L=self.L,
                bandwidth=self.bandwidth,
                phi_fourier_power=self.phi_fourier_power,
                window_function_numba=self.func,
                **self.window_args,
            )
            self._apply_zero_mode_convention()
            return
        if self._requires_full_rfft_kernel():
            self.w_kernel = build_real_window_rfft_kernel(
                L=self.L,
                bandwidth=self.bandwidth,
                phi_fourier_power=self.phi_fourier_power,
                window_function_numba=self.func,
                **self.window_args,
            )
            self._apply_zero_mode_convention()
            return
        _window_array = build_real_window_octant_array(
            L=self.L,
            bandwidth=self.bandwidth,
            phi_fourier_power=self.phi_fourier_power,
            window_function_numba=self.func,
            **self.window_args,
        )
        self.w_kernel = fold_octant_window_to_rfft_kernel(_window_array)
        self._apply_zero_mode_convention()

    def as_array(self):
        if self.w_kernel is None:
            if getattr(self, "is_composite_window", False):
                if self.rank == 0:
                    self.logger.error(
                        "Composite WindowFunc has no w_kernel. Copy composite windows with copy_kernel=True."
                    )
                func_util.safe_exit(1)
            self._build_kernel()
        return self.w_kernel

    def _window_operation_descriptor(self):
        return serialize_window_params(getattr(self, "window_params", {}))

    def _check_window_compatibility(self, other):
        if not isinstance(other, WindowFunc):
            return
        for key in ("L", "bandwidth", "box_size", "phi_resolution", "wavelet_mode", "wavelet_level"):
            if getattr(self, key) != getattr(other, key):
                if self.rank == 0:
                    self.logger.error(
                        f"Cannot combine WindowFunc objects with different {key}: "
                        f"{getattr(self, key)} vs {getattr(other, key)}."
                    )
                func_util.safe_exit(1)

    @staticmethod
    def _is_real_scalar(value):
        return np.isscalar(value) and not isinstance(value, (str, bytes)) and np.isrealobj(value)

    def _spawn_composite(self, w_kernel, operation):
        new = self.__class__.__new__(self.__class__)

        new.comm = self.comm
        new.rank = self.rank
        new.logger = self.logger
        new.saveflag = False
        new.task_params = copy.deepcopy(getattr(self, "task_params", None))
        new.threads = self.threads
        new.sfc_info = copy.deepcopy(getattr(self, "sfc_info", {}))
        new.epsilon = None

        for key in (
            "L",
            "bandwidth",
            "box_size",
            "phi_resolution",
            "wavelet_mode",
            "wavelet_level",
        ):
            setattr(new, key, copy.deepcopy(getattr(self, key)))
        new.phi_array = self.phi_array
        new.phi_fourier_power = self.phi_fourier_power

        new.type = "composite"
        new.func = None
        new.has_custom_func = False
        new.kernel_mode = "materialized"
        new.len_args = {}
        new.rescale_len_args = {}
        new.los_args = {}
        new.other_args = {"operation": copy.deepcopy(operation)}
        new.window_args = {}
        new.window_params = {
            "type": new.type,
            "len_args": new.len_args,
            "los_args": new.los_args,
            "other_args": new.other_args,
            "kernel_mode": new.kernel_mode,
        }
        new.input_params = {
            **copy.deepcopy(new.window_params),
            "J": int(np.log2(self.L)),
            "box_size": self.box_size,
            "phi_resolution": self.phi_resolution,
            "wavelet_mode": self.wavelet_mode,
            "wavelet_level": self.wavelet_level,
            "bandwidth": self.bandwidth,
        }
        new.w_kernel = w_kernel
        new.is_composite_window = True
        return new

    def _binary_window_op(self, other, op_name, op_func):
        if not isinstance(other, WindowFunc):
            return NotImplemented
        self._check_window_compatibility(other)
        left = self.as_array()
        right = other.as_array()
        if left.shape != right.shape:
            if self.rank == 0:
                self.logger.error(
                    f"Cannot combine WindowFunc kernels with different shapes: {left.shape} vs {right.shape}."
                )
            func_util.safe_exit(1)
        operation = {
            "op": op_name,
            "left": self._window_operation_descriptor(),
            "right": other._window_operation_descriptor(),
        }
        return self._spawn_composite(op_func(left, right), operation)

    def _scalar_window_op(self, scalar, op_name, op_func):
        if not self._is_real_scalar(scalar):
            return NotImplemented
        scalar = float(scalar)
        operation = {
            "op": op_name,
            "window": self._window_operation_descriptor(),
            "scalar": scalar,
        }
        return self._spawn_composite(op_func(self.as_array(), scalar), operation)

    def __add__(self, other):
        return self._binary_window_op(other, "add", lambda left, right: left + right)

    def __radd__(self, other):
        if isinstance(other, WindowFunc):
            return other.__add__(self)
        return NotImplemented

    def __sub__(self, other):
        return self._binary_window_op(other, "sub", lambda left, right: left - right)

    def __rsub__(self, other):
        if isinstance(other, WindowFunc):
            return other.__sub__(self)
        return NotImplemented

    def __neg__(self):
        return self._scalar_window_op(-1.0, "mul", lambda window, scalar: window * scalar)

    def __mul__(self, other):
        if isinstance(other, WindowFunc):
            return self._binary_window_op(
                other,
                "projected_kernel_product",
                lambda left, right: left * right,
            )
        return self._scalar_window_op(other, "mul", lambda window, scalar: window * scalar)

    def __rmul__(self, other):
        if isinstance(other, WindowFunc):
            return other.__mul__(self)
        return self.__mul__(other)

    def __truediv__(self, other):
        if self._is_real_scalar(other) and float(other) == 0.0:
            if self.rank == 0:
                self.logger.error("Cannot divide WindowFunc by zero.")
            func_util.safe_exit(1)
        return self._scalar_window_op(other, "div", lambda window, scalar: window / scalar)

    def copy(self, copy_kernel=True):
        """
        Copy this window object without rebuilding the Fourier kernel.

        Parameters
        ----------
        copy_kernel : bool, default=True
            If True, copy the cached ``w_kernel`` array when it has already
            been built. If False, the returned window keeps the same window
            parameters but will rebuild ``w_kernel`` on the next ``as_array()``
            call.
        """
        if getattr(self, "is_composite_window", False) and not copy_kernel:
            if self.rank == 0:
                self.logger.error("Composite WindowFunc cannot be copied with copy_kernel=False.")
            func_util.safe_exit(1)
        new = self.__class__.__new__(self.__class__)
        for key, value in self.__dict__.items():
            if key in ("comm", "logger", "func"):
                setattr(new, key, value)
            elif key == "w_kernel":
                if copy_kernel and value is not None:
                    setattr(new, key, value.copy())
                else:
                    setattr(new, key, None)
            elif isinstance(value, np.ndarray):
                setattr(new, key, value.copy())
            else:
                setattr(new, key, copy.deepcopy(value))
        return new

    def _load_single(self, f_in):
        with open(f_in, 'rb') as f:
            # Read the entire .npy file as bytes
            serialized_data = np.lib.format.read_array(f, allow_pickle=True)
            # Convert the bytes back into the original dataset using pickle
            dataset = pickle.loads(serialized_data.tobytes())
            # Check if the 'data' key is present in the dataset
            if 'window_kernal' not in dataset:
                self.logger.error(f"Failed to load the dataset. The file is missing the 'window_kernal' key.")
                func_util.safe_exit(1)
            # Assign the dictionary from the file to self.sfc_info
            self.input_params = {key: value for key, value in dataset.items() if key != 'window_kernal'}
            self.w_kernel = dataset['window_kernal']

    def _save_single(self, f_out):
        # Check and create directory if it doesn't exist
        _dir = os.path.dirname(f_out)
        if not os.path.exists(_dir):
            os.makedirs(_dir)
        # Check if the input_params is empty
        if not self.input_params:
            self.logger.error('The dict_argv "win_params" is empty.')
            self.logger.error(f"Failed to save the data to the file: '{f_out}'")
            func_util.safe_exit(1)
        # If all required variables are present, create the dataset
        dataset = {
            **self.input_params,  # Add all required variables to the dataset
            'window_kernal': self.w_kernel  # Include the actual data
        }
        # Save the dataset to the specified file
        #  ↓ Use Pickle with protocol 4 or higher to handle saving files larger than 4 GiB
        _serialized_data = pickle.dumps(dataset, protocol=4)
        with open(f_out, 'wb') as f:
            np.lib.format.write_array(f, np.frombuffer(_serialized_data, dtype=np.uint8))
