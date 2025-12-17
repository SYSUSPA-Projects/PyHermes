import os
import pickle

import numpy as np

from .base import HermesData
from pyhermes.utils import func_util
from pyhermes.utils import math_util



class ConvolsData(HermesData):

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
                    f"math_util.specialized_convolution_3d expects 3D arrays; got a.ndim={a.ndim}, b.ndim={b.ndim}."
                )
            func_util.safe_exit(1)
        # Window is calculated through rfft, so here to match half dimension
        nx, ny, nz = a.shape
        expected = (nx, ny, nz // 2 + 1)
        if b.shape != expected:
            if self.rank == 0:
                self.logger.error(f"Convolution requires same shape; got a.shape={a.shape}, b.shape={b.shape}.")
            func_util.safe_exit(1)
        return math_util.specialized_convolution_3d(a, b, threads=self.threads)

    def _conv_numpy(self, other):
        a = self.as_array()
        b = np.array(other)
        if a.ndim != 3 or b.ndim != 3:
            if self.rank == 0:
                self.logger.error(
                    f"math_util.specialized_convolution_3d expects 3D arrays; got a.ndim={a.ndim}, b.ndim={b.ndim}."
                )
            func_util.safe_exit(1)
        nx, ny, nz = a.shape
        expected = (nx, ny, nz // 2 + 1)
        if b.shape != expected:
            if self.rank == 0:
                self.logger.error(f"Convolution requires same shape; got a.shape={a.shape}, b.shape={b.shape}.")
            func_util.safe_exit(1)
        return math_util.specialized_convolution_3d(a, b, threads=self.threads)

    def __matmul__(self, other):
        if isinstance(other, ConvolsData):
            return self._conv(other)
        if isinstance(other, np.ndarray):
            return self._conv_numpy(other)
        return NotImplemented

    def __rmatmul__(self, other):
        if isinstance(other, ConvolsData):
            return other._conv(self)  
        if isinstance(other, np.ndarray):
            return self._conv_numpy(other)
        return NotImplemented

    def as_array(self):
        return self.deltac

    def _load_single(self, f_in):
        with open(f_in, 'rb') as f:
            # Read the entire .npy file as bytes
            serialized_data = np.lib.format.read_array(f, allow_pickle=True)
            # Convert the bytes back into the original dataset using pickle
            dataset = pickle.loads(serialized_data.tobytes())
            # Check if the 'data' key is present in the dataset
            if 'deltac' not in dataset:
                self.logger.error(f"Failed to load the dataset. The file is missing the 'data' key.")
                func_util.safe_exit(1)
            # Assign the dictionary from the file to self.dict_inht_vonDeltac
            self.dict_inht_vonDeltac = {key: value for key, value in dataset.items() if key != 'deltac'}
            self.deltac = dataset['deltac']

    def _save_single(self, f_out):
        # Check and create directory if it doesn't exist
        _dir = os.path.dirname(f_out)
        if not os.path.exists(_dir):
            os.makedirs(_dir)
        # Check if the dict_inht_vonDeltac is empty
        if not self.dict_inht_vonDeltac:
            self.logger.error('The dictionary "dict_inht_vonDeltac" is empty.')
            self.logger.error('Please ensure that the required data has been loaded or calculated before attempting to save the dataset.')
            self.logger.error(f"Failed to save the data to the file: '{f_out}'")
            func_util.safe_exit(1)
        # If all required variables are present, create the dataset
        dataset = {
            **self.dict_inht_vonDeltac,  # Add all required variables to the dataset
            'deltac': self.deltac  # Include the actual data
        }
        # Save the dataset to the specified file
        #  ↓ Use Pickle with protocol 4 or higher to handle saving files larger than 4 GiB
        _serialized_data = pickle.dumps(dataset, protocol=4)
        with open(f_out, 'wb') as f:
            np.lib.format.write_array(f, np.frombuffer(_serialized_data, dtype=np.uint8))