import os
import pickle

import numpy as np

from .base import HermesData
from pyhermes.utils import func_util


class CountingData(HermesData):
    def __init__(self, *args, threads=1, **kwargs):
        data_path = kwargs.pop("data_path", None)
        self.counting_info = {}
        self.nx = None
        super().__init__(*args, threads=threads, **kwargs)
        if data_path:
            self.counting_info['counting_data_path'] = data_path
            self.load_counting(data_path)

    def format_counting_params(self):
        for key, value in self.convols_info.items():
            setattr(self, key, value)
        for key, value in self.counting_info.items():
            setattr(self, key, value)

    def load_counting(self, f_in, single=True):
        self.load(f_in, read_counting=True, single=single)

    def save_counting(self, f_out, single=True, overwrite=False):
        self.save(f_out, save_counting=True, single=single, overwrite=overwrite)

    def _load_counting(self, f_in):
        with open(f_in, 'rb') as f:
            # Read the entire .npy file as bytes
            serialized_data = np.lib.format.read_array(f, allow_pickle=True)
            # Convert the bytes back into the original dataset using pickle
            dataset = pickle.loads(serialized_data.tobytes())
            # Check if the 'nx' key is present in the dataset
            if 'nx' not in dataset:
                self.logger.error(f"Failed to load the dataset. The file is missing the 'nx' key.")
                func_util.safe_exit(1)
            self.nx = dataset['nx']
            # Assign the dictionary from the file to self.counting_info
            # self.counting_info = {key: value for key, value in dataset.items() if key != 'nx'}
            _convols_info = dataset.get('convols_info')
            if _convols_info:
                self.convols_info = _convols_info
            _counting_info = dataset.get('counting_info')
            if _counting_info:
                self.counting_info.update(_counting_info)
            self.format_counting_params()

    def _save_counting(self, f_out):
        # Check and create directory if it doesn't exist
        _dir = os.path.dirname(f_out)
        if not os.path.exists(_dir):
            os.makedirs(_dir)
        # Check if the counting_info is empty
        if not self.counting_info:
            self.logger.error('The dictionary "counting_info" is empty.')
            self.logger.error('Please ensure that the required data has been loaded or calculated before attempting to save the dataset.')
            self.logger.error(f"Failed to save the data to the file: '{f_out}'")
            func_util.safe_exit(1)
        # If all required variables are present, create the dataset
        dataset = {
            'convols_info': self.convols_info,
            'counting_info': self.counting_info,
            # **self.counting_info,  # Add all required variables to the dataset
            'nx': self.nx  # Include the actual data
        }
        # Save the dataset to the specified file
        #  ↓ Use Pickle with protocol 4 or higher to handle saving files larger than 4 GiB
        _serialized_data = pickle.dumps(dataset, protocol=4)
        with open(f_out, 'wb') as f:
            np.lib.format.write_array(f, np.frombuffer(_serialized_data, dtype=np.uint8))

    # def __init__(self, *args, threads=1, **kwargs):
    #     self.counting_info = {}
    #     super().__init__(*args, threads=threads, **kwargs)

    # def _load_single(self, f_in):
    #     with open(f_in, 'rb') as f:
    #         # by xutp
    #         # self.data_all = np.load(f)
    #         self.data = np.load(f)
    
    # def _save_single(self, f_out):
    #     _dir = os.path.dirname(f_out)
    #     if not os.path.exists(_dir):
    #         os.makedirs(_dir)
    #     with open(f_out, 'wb') as f:
    #         np.save(f, self.data)