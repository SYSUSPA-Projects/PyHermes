import os
from datetime import datetime

import numpy as np

import pyhermes
from .corr2pcf import Corr2PCFData



class Corr3PCFData(Corr2PCFData):

    def _load_single(self, f_in):
        with open(f_in, 'r') as f:
            lines = f.readlines()
        data_start = None
        for i, line in enumerate(lines):
            if line.strip() == "" or line.startswith("#"):
                continue
            if line.startswith('---'):  
                data_start = i + 1
                break
        _data = np.loadtxt(f_in, delimiter=",", skiprows=data_start+1)
        self.theta = _data[:, 0]
        self.Q = _data[:, 1]

    def _save_single(self, f_out):
        _dir = os.path.dirname(f_out)
        if not os.path.exists(_dir):
            os.makedirs(_dir)
        self.theta = np.asarray(self.theta, dtype=np.float64)
        self.Q = np.asarray(self.Q, dtype=np.float64)
        version = pyhermes.__version__
        current_time = datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
        header = (
            f"# Corr_3PCF output from PyHermes v{version}, TIME: {current_time}\n"
            "# Parameters from input :\n"
            f"#  R1                = {self.task_params['R1']}\n"
            f"#  R2                = {self.task_params['R2']}\n"
            f"#  rot_num           = {int(self.task_params['rot_num'])}\n"
            f"#  NStheta           = {int(self.task_params['NStheta'])}\n"
            f"#  fin_size          = {self.task_params['orgDsize_3pcf']}\n"
            f"#  fin_path          = {self.task_params['fin']['path']}\n"
            f"#  fin_format        = {self.task_params['fin']['format']}\n"
            f"#  fout_path         = {self.task_params['fout_path']}\n"
            f"#  deltac_in_path    = {self.task_params['deltac_in_path']}\n"
            f"#  corr2pcf_in_path  = {self.task_params['corr2pcf_in_path']}\n"
            "# Parameters from DeltaC:\n"
            f"#  J                 = {self.task_params['J']}\n"
            f"#  SimBoxL           = {self.task_params['SimBoxL']}\n"
            f"#  SampRate          = {int(self.task_params['SampRate'])}\n"
            f"#  bandwidth         = {self.task_params['bandwidth']}\n"
            f"#  fin_size          = {self.task_params['orgDsize']}\n"
            f"#  fin_path          = {self.task_params['fin_path']}\n"
            f"#  fin_format        = {self.task_params['fin_format']}\n"
            f"#  wavelet_mode      = {self.task_params['wavelet_mode']}\n"
            f"#  wavelet_level     = {self.task_params['wavelet_level']}\n"
            f"#  Window_Info       = {self.task_params['window']}\n"
            "\n"
            "---------------------------\n"
            "theta[rad]  , Q"
        )
        data_to_save = np.column_stack((self.theta, self.Q))
        fmt = '%.6e, %.6e'
        delimiter = ",   " 
        np.savetxt(f_out, data_to_save, delimiter=delimiter, header=header, comments='', fmt=fmt)