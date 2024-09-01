import os
import pickle

import numpy as np

from .base import HermesData
from pyhermes.utils import func_util



class ConvolsData(HermesData):

    def load_deltac(self, f_in, single=True):
        self.load(f_in, read_deltac=True, single=single)

    def _load_deltac(self, f_in):
        with open(f_in, 'rb') as f:
            # Read the entire .npy file as bytes
            serialized_data = np.lib.format.read_array(f, allow_pickle=True)
            # Convert the bytes back into the original dataset using pickle
            dataset = pickle.loads(serialized_data.tobytes())
            # Check if the 'data' key is present in the dataset
            if 'data' not in dataset:
                self.logger.error(f"Failed to load the dataset. The file is missing the 'data' key.")
                func_util.safe_exit(1)
            # Assign the dictionary from the file to self.dict_inht_vonDeltac
            self.dict_inht_vonDeltac = {key: value for key, value in dataset.items() if key != 'data'}
            self.deltac = dataset['data']
    
    def _load_single(self, f_in):
        with open(f_in, 'rb') as f:
            # Read the entire .npy file as bytes
            serialized_data = np.lib.format.read_array(f, allow_pickle=True)
            # Convert the bytes back into the original dataset using pickle
            dataset = pickle.loads(serialized_data.tobytes())
            # Check if the 'data' key is present in the dataset
            if 'data' not in dataset:
                self.logger.error(f"Failed to load the dataset. The file is missing the 'data' key.")
                func_util.safe_exit(1)
            # Assign the dictionary from the file to self.dict_inht_vonDeltac
            self.dict_inht_vonDeltac = {key: value for key, value in dataset.items() if key != 'data'}
            self.data = dataset['data']

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
            'data': self.data  # Include the actual data
        }
        # Save the dataset to the specified file
        #  ↓ Use Pickle with protocol 4 or higher to handle saving files larger than 4 GiB
        _serialized_data = pickle.dumps(dataset, protocol=4)
        with open(f_out, 'wb') as f:
            np.lib.format.write_array(f, np.frombuffer(_serialized_data, dtype=np.uint8))