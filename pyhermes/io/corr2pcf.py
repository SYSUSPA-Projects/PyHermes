import os
import pickle
# from datetime import datetime

import numpy as np

# import pyhermes
from .base import HermesData
from pyhermes.utils import func_util


class Corr2PCFData(HermesData):
    def __init__(self, *args, threads=1, **kwargs):
        data_path = kwargs.pop("data_path", None)
        self.corr2pcf_info = {}
        self.r = np.empty(0)
        self.xi = np.empty(0)
        super().__init__(*args, threads=threads, **kwargs)
        if data_path:
            self.corr2pcf_info['corr2pcf_data_path'] = data_path
            self.load_corr2pcf(data_path)

    def format_corr2pcf_params(self):
        for key, value in self.convols_info.items():
            setattr(self, key, value)
        for key, value in self.corr2pcf_info.items():
            setattr(self, key, value)

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
            # Check if the 'r' and 'xi' keys are present in the dataset
            for key in ['r', 'xi']:
                if key not in dataset:
                    self.logger.error(f"Failed to load the dataset. The file is missing the '{key}' key.")
                    func_util.safe_exit(1)
                setattr(self, key, dataset[key])
            # Assign the dictionary from the file to self.corr2pcf_info
            _convols_info = dataset.get('convols_info')
            if _convols_info:
                self.convols_info = _convols_info
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
            'convols_info': self.convols_info,
            'corr2pcf_info': self.corr2pcf_info,
            'r': self.r,
            'xi': self.xi  # Include the actual data
        }
        # Save the dataset to the specified file
        #  ↓ Use Pickle with protocol 4 or higher to handle saving files larger than 4 GiB
        _serialized_data = pickle.dumps(dataset, protocol=4)
        with open(f_out, 'wb') as f:
            np.lib.format.write_array(f, np.frombuffer(_serialized_data, dtype=np.uint8))


    # def _load_corr2pcf(self, f_in):
    #     with open(f_in, 'r') as f:
    #         lines = f.readlines()
    #     data_start = None
    #     for i, line in enumerate(lines):
    #         if line.strip() == "" or line.startswith("#"):
    #             continue
    #         if line.startswith('---'):  
    #             data_start = i + 1
    #             break
    #     _data = np.loadtxt(f_in, delimiter=",", skiprows=data_start+1)
    #     self.r = _data[:, 0]
    #     self.xi = _data[:, 1]

    # def _save_corr2pcf(self, f_out):
    #     _dir = os.path.dirname(f_out)
    #     if not os.path.exists(_dir):
    #         os.makedirs(_dir)
    #     self.r = np.asarray(self.r, dtype=np.float64)
    #     self.xi = np.asarray(self.xi, dtype=np.float64)
    #     version = pyhermes.__version__
    #     current_time = datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
    #     header = (
    #         f"# Corr_2PCF output from PyHermes v{version}, TIME: {current_time}\n"
    #         "# Parameters from input :\n"
    #         f"#  R1             = {self.corr2pcf_info['R1']}\n"
    #         f"#  R2             = {self.corr2pcf_info['R2']}\n"
    #         f"#  xi_num         = {int(self.corr2pcf_info['xi_num'])}\n"
    #         f"#  threads        = {int(self.corr2pcf_info['threads'])}\n"
    #         f"#  fout_path      = {self.corr2pcf_info['fout_path']}\n"
    #         f"#  Window_Info    = {self.corr2pcf_info['window']}\n"
    #         f"#  convols_data_path = {self.corr2pcf_info['convols_data_path']}\n"
    #         "# Parameters from convols data:\n"
    #         f"#  J              = {self.convols_info['J']}\n"
    #         f"#  SimBoxL        = {self.convols_info['SimBoxL']}\n"
    #         f"#  SampRate       = {int(self.convols_info['SampRate'])}\n"
    #         f"#  bandwidth      = {self.convols_info['bandwidth']}\n"
    #         f"#  fin_size       = {self.convols_info['N_particles']}\n"
    #         f"#  fin_path       = {self.convols_info['fin_path']}\n"
    #         f"#  fin_weight_key = {self.convols_info['fin_weight_key']}\n"
    #         f"#  fin_format     = {self.convols_info['fin_format']}\n"
    #         f"#  wavelet_mode   = {self.convols_info['wavelet_mode']}\n"
    #         f"#  wavelet_level  = {self.convols_info['wavelet_level']}\n"
    #         "\n"
    #         "---------------------------\n"
    #         "r[h-1 Mpc]  , xi"
    #     )
    #     data_to_save = np.column_stack((self.r, self.xi))
    #     fmt = '%.6e, %.6e'
    #     delimiter = ",   " 
    #     np.savetxt(f_out, data_to_save, delimiter=delimiter, header=header, comments='', fmt=fmt)



    # def _load_single(self, f_in):
    #     with open(f_in, 'r') as f:
    #         lines = f.readlines()
    #     data_start = None
    #     for i, line in enumerate(lines):
    #         if line.strip() == "" or line.startswith("#"):
    #             continue
    #         if line.startswith('---'):  
    #             data_start = i + 1
    #             break
    #     _data = np.loadtxt(f_in, delimiter=",", skiprows=data_start+1)
    #     self.r = _data[:, 0]
    #     self.xi = _data[:, 1]

    # def _save_single(self, f_out):
    #     _dir = os.path.dirname(f_out)
    #     if not os.path.exists(_dir):
    #         os.makedirs(_dir)
    #     self.r = np.asarray(self.r, dtype=np.float64)
    #     self.xi = np.asarray(self.xi, dtype=np.float64)
    #     version = pyhermes.__version__
    #     current_time = datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
    #     header = (
    #         f"# Corr_2PCF output from PyHermes v{version}, TIME: {current_time}\n"
    #         "# Parameters from input :\n"
    #         f"#  R1             = {self.task_params['R1']}\n"
    #         f"#  R2             = {self.task_params['R2']}\n"
    #         f"#  xi_num         = {int(self.task_params['xi_num'])}\n"
    #         f"#  threads        = {int(self.task_params['threads'])}\n"
    #         f"#  fout_path      = {self.task_params['fout_path']}\n"
    #         f"#  Window_Info    = {self.task_params['window']}\n"
    #         f"#  convols_data_path = {self.task_params['convols_data_path']}\n"
    #         "# Parameters from convols data:\n"
    #         f"#  J              = {self.task_params['J']}\n"
    #         f"#  SimBoxL        = {self.task_params['SimBoxL']}\n"
    #         f"#  SampRate       = {int(self.task_params['SampRate'])}\n"
    #         f"#  bandwidth      = {self.task_params['bandwidth']}\n"
    #         f"#  fin_size       = {self.task_params['N_particles']}\n"
    #         f"#  fin_path       = {self.task_params['fin_path']}\n"
    #         f"#  fin_weight_key = {self.task_params['fin_weight_key']}\n"
    #         f"#  fin_format     = {self.task_params['fin_format']}\n"
    #         f"#  wavelet_mode   = {self.task_params['wavelet_mode']}\n"
    #         f"#  wavelet_level  = {self.task_params['wavelet_level']}\n"
    #         "\n"
    #         "---------------------------\n"
    #         "r[h-1 Mpc]  , xi"
    #     )
    #     data_to_save = np.column_stack((self.r, self.xi))
    #     fmt = '%.6e, %.6e'
    #     delimiter = ",   " 
    #     np.savetxt(f_out, data_to_save, delimiter=delimiter, header=header, comments='', fmt=fmt)

