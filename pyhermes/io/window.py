import os
import pickle

import numpy as np

from .convols import ConvolsData
from pyhermes.utils import func_util
from pyhermes.utils.convolution import call_calculate_window_array, calculate_w_numba
from pyhermes.utils.wavelet_grid import do_wavelet, power_spectrum
from pyhermes.utils.window_functions import set_window_function


class WindowFunc(ConvolsData):
    def __init__(self, win_params, convols_params, bandwidth=1, threads=1):
        # Initial MPI, logger mess
        super().__init__(threads=threads)
        missing = [k for k in self._REQUIRED_ARGV if k not in convols_params]
        if missing:
            if self.rank == 0:
                self.logger.error(f"WindowFunc missing required keys: {missing}")
            func_util.safe_exit(1)
        try:
            self.L = 1 << int(convols_params['J'])
            self.bandwidth = int(bandwidth)
            self.box_size = convols_params["box_size"]
            self.phi_resolution = int(convols_params["phi_resolution"])
            self.wavelet_mode = convols_params["wavelet_mode"]
            self.wavelet_level = convols_params["wavelet_level"]
        except Exception as e:
            if self.rank == 0:
                self.logger.error(f"WindowFunc core parameter error: {e}")
            func_util.safe_exit(1)
        self.input_params = {
            **dict(win_params),
            "J": int(convols_params["J"]),
            "box_size": self.box_size,
            "phi_resolution": self.phi_resolution,
            "wavelet_mode": self.wavelet_mode,
            "wavelet_level": self.wavelet_level,
            "bandwidth": self.bandwidth,
        }
        phi_array = do_wavelet(self.wavelet_mode, self.wavelet_level)
        self.phi_fourier_power = power_spectrum(phi_array, 0, self.bandwidth, self.L * self.bandwidth, self.phi_resolution)
        if not isinstance(self.phi_fourier_power, np.ndarray):
            if self.rank == 0:
                self.logger.error("WindowFunc requires phi_fourier_power to be a numpy.ndarray.")
            func_util.safe_exit(1)
        if self.L <= 0 or self.bandwidth <= 0:
            if self.rank == 0:
                self.logger.error(f"Invalid L/bandwidth: L={self.L}, bandwidth={self.bandwidth}. Must be > 0.")
            func_util.safe_exit(1)
        # There is NO DEFAULT window!!!
        # Missing `type` will raise an error in set_window_function.
        self.window_params = dict(win_params)
        if "func" in win_params:
            self.type = win_params.get('type', None) or "custom"
            self.func = win_params["func"]
        else:
            assert "type" in win_params
            self.type = win_params['type']
            self.func = set_window_function(self.type, verbose=False)
        self.len_args = win_params['len_args']
        self.rescale_len_args = {k: v * self.L / self.box_size for k, v in self.len_args.items()}
        self.other_args = win_params.get('other_args', {})
        self.window_args = dict(self.rescale_len_args)
        self.window_args.update(self.other_args)
        self._window_array = None
        self.w_kernel = None
    
    def _build_window_array(self):
        self._window_array = call_calculate_window_array(
            L=self.L,
            bandwidth=self.bandwidth,
            phi_fourier_power=self.phi_fourier_power,
            window_function_numba=self.func,
            **self.window_args,
        )

    def _build_kernel(self):
        self._build_window_array()
        self.w_kernel = calculate_w_numba(self._window_array)

    def as_array(self):
        if self.w_kernel is None:
            self._build_kernel()
        return self.w_kernel

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
            # Assign the dictionary from the file to self.convols_info
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
