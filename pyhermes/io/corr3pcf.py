import os
import pickle
import numpy as np

from .base import HermesData
from pyhermes.utils import func_util


class Corr3PCFData(HermesData):
    def __init__(self, *args, threads=None, **kwargs):
        data_path = kwargs.pop("data_path", None)
        self.corr3pcf_info = {}
        self.theta               = None
        self.r23                 = None
        self.ddd                 = None
        self.delta_ddd           = None
        self.pdelta_ddd          = None
        self.xi12                = None
        self.xi13                = None
        self.xi23                = None
        self.zeta                = None
        self.Q                   = None
        super().__init__(*args, threads=threads, **kwargs)
        if data_path:
            self.corr3pcf_info['corr3pcf_data_path'] = data_path
            self.load_corr3pcf(data_path)

    def format_corr3pcf_params(self):
        for key, value in self.corr3pcf_info.items():
            setattr(self, key, value)

    def load_corr3pcf(self, f_in, single=True):
        self.load(f_in, read_3pcf=True, single=single)

    def save_corr3pcf(self, f_out, single=True, overwrite=False):
        self.save(f_out, save_3pcf=True, single=single, overwrite=overwrite)

    def _load_corr3pcf(self, f_in):
        with open(f_in, 'rb') as f:
            # Read the entire .npy file as bytes
            serialized_data = np.lib.format.read_array(f, allow_pickle=True)
            # Convert the bytes back into the original dataset using pickle
            dataset = pickle.loads(serialized_data.tobytes())
            # Check if the 'r' and 'xi' keys are present in the dataset
            for key in ['theta', 'r23', 'xi12', 'xi13', 'xi23', 'zeta', 'Q']:
                if key not in dataset:
                    self.logger.error(f"Failed to load the dataset. The file is missing the '{key}' key.")
                    func_util.safe_exit(1)
                setattr(self, key, dataset[key])
            self.ddd = dataset.get('ddd')
            self.delta_ddd = dataset.get('delta_ddd')
            self.pdelta_ddd = dataset.get('pdelta_ddd')
            # Assign the dictionary from the file to self.corr3pcf_info
            for i in range(1, 4):
                _convols_info = dataset.get(f'convols_info{i}')
                if _convols_info:
                    setattr(self, f"convols_info{i}", _convols_info)
            _corr3pcf_info = dataset.get('corr3pcf_info')
            if _corr3pcf_info:
                self.corr3pcf_info.update(_corr3pcf_info)
            self.format_corr3pcf_params()

    def _save_corr3pcf(self, f_out):
        # Check and create directory if it doesn't exist
        _dir = os.path.dirname(f_out)
        if not os.path.exists(_dir):
            os.makedirs(_dir)
        # Check if the corr3pcf_info is empty
        if not self.corr3pcf_info:
            self.logger.error('The dictionary "corr3pcf_info" is empty.')
            self.logger.error('Please ensure that the required data has been loaded or calculated before attempting to save the dataset.')
            self.logger.error(f"Failed to save the data to the file: '{f_out}'")
            func_util.safe_exit(1)
        # If all required variables are present, create the dataset
        dataset = {
            'convols_info1': self.convols_info1,
            'convols_info2': self.convols_info2,
            'convols_info3': self.convols_info3,
            'corr3pcf_info': self.corr3pcf_info,
            'theta': self.theta,
            'r23': self.r23,
            'ddd': self.ddd,
            'delta_ddd': self.delta_ddd,
            'pdelta_ddd': self.pdelta_ddd,
            'xi12': self.xi12,
            'xi13': self.xi13,
            'xi23': self.xi23,
            'zeta': self.zeta,
            'Q': self.Q  # Include the actual data
        }
        # Save the dataset to the specified file
        #  ↓ Use Pickle with protocol 4 or higher to handle saving files larger than 4 GiB
        _serialized_data = pickle.dumps(dataset, protocol=4)
        with open(f_out, 'wb') as f:
            np.lib.format.write_array(f, np.frombuffer(_serialized_data, dtype=np.uint8))
