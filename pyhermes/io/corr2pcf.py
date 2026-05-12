import os
import pickle
import numpy as np

from .base import HermesData
from pyhermes.utils import func_util


COORDINATE_NAMES = ("s", "mu", "rp", "pi")


class Corr2PCFData(HermesData):
    def __init__(self, *args, threads=None, **kwargs):
        data_path = kwargs.pop("data_path", None)
        self.corr2pcf_info = {}
        self.sampling_names = ()
        self.sampling = {}
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
            if key in ('sampling', 'sampling_names') or key in COORDINATE_NAMES:
                continue
            setattr(self, key, value)

    def _clear_sampling_attrs(self):
        for name in set(COORDINATE_NAMES) | set(getattr(self, "sampling_names", ())):
            if hasattr(self, name):
                delattr(self, name)

    def set_sampling(self, sampling_names, sampling):
        self._clear_sampling_attrs()
        self.sampling_names = tuple(sampling_names)
        self.sampling = {
            name: self._ensure_1d_array(sampling[name], name)
            for name in self.sampling_names
        }
        for name, values in self.sampling.items():
            setattr(self, name, values)

    def _sync_sampling_from_attrs(self):
        if self.sampling_names:
            names = tuple(self.sampling_names)
        elif self.sampling:
            names = tuple(self.sampling.keys())
        else:
            names = tuple(name for name in COORDINATE_NAMES if hasattr(self, name))
        if not names:
            raise ValueError("Corr2PCFData has no sampling coordinates to save.")

        sampling = {}
        for name in names:
            if hasattr(self, name):
                sampling[name] = getattr(self, name)
            elif name in self.sampling:
                sampling[name] = self.sampling[name]
            else:
                raise ValueError(f"Missing sampling coordinate '{name}'.")
        self.set_sampling(names, sampling)
        return self.sampling

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
            if 'sampling_names' in dataset:
                sampling_names = tuple(dataset['sampling_names'])
            elif 'sampling' in dataset:
                sampling_names = tuple(dataset['sampling'].keys())
            else:
                sampling_names = tuple(name for name in COORDINATE_NAMES if name in dataset)
            if not sampling_names:
                self.logger.error("Failed to load the dataset. No Corr2PCF sampling coordinates were found.")
                func_util.safe_exit(1)

            if all(name in dataset for name in sampling_names):
                sampling = {name: dataset[name] for name in sampling_names}
            elif 'sampling' in dataset:
                sampling = {name: dataset['sampling'][name] for name in sampling_names}
            else:
                self.logger.error("Failed to load the dataset. Sampling coordinate arrays are incomplete.")
                func_util.safe_exit(1)
            self.set_sampling(sampling_names, sampling)

            for name in ('dd', 'dr', 'rd', 'delta_dd', 'rr', 'xi'):
                value = dataset.get(name)
                setattr(
                    self,
                    name,
                    None if value is None else self._ensure_result_array(value, name),
                )
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
        sampling = self._sync_sampling_from_attrs()
        # If all required variables are present, create the dataset
        dataset = {
            'convols_info1': self.convols_info1,
            'convols_info2': self.convols_info2,
            'corr2pcf_info': self.corr2pcf_info,
            'sampling_names': tuple(self.sampling_names),
            'dd': None if self.dd is None else self._ensure_result_array(self.dd, 'dd'),
            'dr': None if self.dr is None else self._ensure_result_array(self.dr, 'dr'),
            'rd': None if self.rd is None else self._ensure_result_array(self.rd, 'rd'),
            'delta_dd': None if self.delta_dd is None else self._ensure_result_array(self.delta_dd, 'delta_dd'),
            'rr': None if self.rr is None else self._ensure_result_array(self.rr, 'rr'),
            'xi': None if self.xi is None else self._ensure_result_array(self.xi, 'xi')
        }
        for name, values in sampling.items():
            dataset[name] = values
        # Save the dataset to the specified file
        #  ↓ Use Pickle with protocol 4 or higher to handle saving files larger than 4 GiB
        _serialized_data = pickle.dumps(dataset, protocol=4)
        with open(f_out, 'wb') as f:
            np.lib.format.write_array(f, np.frombuffer(_serialized_data, dtype=np.uint8))
