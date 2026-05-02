import os
import pickle
import numpy as np

from .base import HermesData
from pyhermes.utils import func_util


class Corr2PCFData(HermesData):
    def __init__(self, *args, threads=None, **kwargs):
        data_path = kwargs.pop("data_path", None)
        self.corr2pcf_info = {}
        self.sampling_names = ()
        self.sampling = {}
        self.s = None
        self.mu = None
        self.dd = None
        self.dr = None
        self.rd = None
        self.delta_dd = None
        self.rr = None
        self.xi = None
        super().__init__(*args, threads=threads, **kwargs)
        if data_path:
            self.corr2pcf_info['corr2pcf_data_path'] = data_path
            self.load_corr2pcf(data_path)

    def format_corr2pcf_params(self):
        for key, value in self.corr2pcf_info.items():
            if key in ('sampling', 'sampling_names', 's', 'mu'):
                continue
            setattr(self, key, value)

    def _ensure_1d_array(self, values, name):
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim != 1:
            raise ValueError(f"'{name}' must be stored as a 1D array, got shape {arr.shape}.")
        return np.ascontiguousarray(arr, dtype=np.float64)

    def _ensure_result_array(self, values, name):
        arr = np.asarray(values, dtype=np.float64)
        expected_ndim = len(self.sampling_names)
        if arr.ndim != expected_ndim:
            raise ValueError(f"'{name}' must be stored as a {expected_ndim}D array for sampling={self.sampling_names}, got shape {arr.shape}.")
        return np.ascontiguousarray(arr, dtype=np.float64)

    def _sync_sampling_attrs(self):
        self.s = self.sampling.get("s")
        self.mu = self.sampling.get("mu")

    def load_corr2pcf(self, f_in, single=True):
        self.load(f_in, read_2pcf=True, single=single)

    def save_corr2pcf(self, f_out, single=True, overwrite=False):
        self.save(f_out, save_2pcf=True, single=single, overwrite=overwrite)

    def _load_corr2pcf(self, f_in):
        with open(f_in, 'rb') as f:
            # Read the entire .npy file as bytes
            serialized_data = np.lib.format.read_array(f, allow_pickle=True)
            # Convert the bytes back into the original dataset using pickle
            dataset = pickle.loads(serialized_data.tobytes())
            if 'sampling' not in dataset:
                self.logger.error("Failed to load the dataset. The file is missing the 'sampling' key.")
                func_util.safe_exit(1)
            self.sampling_names = tuple(dataset.get('sampling_names', dataset['sampling'].keys()))
            self.sampling = {
                name: self._ensure_1d_array(dataset['sampling'][name], name)
                for name in self.sampling_names
            }
            self._sync_sampling_attrs()
            self.dd = dataset.get('dd')
            self.dr = dataset.get('dr')
            self.rd = dataset.get('rd')
            self.delta_dd = dataset.get('delta_dd')
            self.rr = dataset.get('rr')
            self.xi = dataset.get('xi')
            # Assign the dictionary from the file to self.corr2pcf_info
            for i in range(1, 3):
                _convols_info = dataset.get(f'convols_info{i}')
                if _convols_info:
                    setattr(self, f"convols_info{i}", _convols_info)
            _corr2pcf_info = dataset.get('corr2pcf_info')
            if _corr2pcf_info:
                self.corr2pcf_info.update(_corr2pcf_info)
            self.format_corr2pcf_params()

    def _save_corr2pcf(self, f_out):
        # Check and create directory if it doesn't exist
        _dir = os.path.dirname(f_out)
        if not os.path.exists(_dir):
            os.makedirs(_dir)
        # Check if the corr2pcf_info is empty
        if not self.corr2pcf_info:
            self.logger.error('The dictionary "corr2pcf_info" is empty.')
            self.logger.error('Please ensure that the required data has been loaded or calculated before attempting to save the dataset.')
            self.logger.error(f"Failed to save the data to the file: '{f_out}'")
            func_util.safe_exit(1)
        # If all required variables are present, create the dataset
        dataset = {
            'convols_info1': self.convols_info1,
            'convols_info2': self.convols_info2,
            'corr2pcf_info': self.corr2pcf_info,
            'sampling_names': tuple(self.sampling_names),
            'sampling': {
                name: self._ensure_1d_array(self.sampling[name], name)
                for name in self.sampling_names
            },
            'dd': None if self.dd is None else self._ensure_result_array(self.dd, 'dd'),
            'dr': None if self.dr is None else self._ensure_result_array(self.dr, 'dr'),
            'rd': None if self.rd is None else self._ensure_result_array(self.rd, 'rd'),
            'delta_dd': None if self.delta_dd is None else self._ensure_result_array(self.delta_dd, 'delta_dd'),
            'rr': None if self.rr is None else self._ensure_result_array(self.rr, 'rr'),
            'xi': None if self.xi is None else self._ensure_result_array(self.xi, 'xi')
        }
        # Save the dataset to the specified file
        #  ↓ Use Pickle with protocol 4 or higher to handle saving files larger than 4 GiB
        _serialized_data = pickle.dumps(dataset, protocol=4)
        with open(f_out, 'wb') as f:
            np.lib.format.write_array(f, np.frombuffer(_serialized_data, dtype=np.uint8))
